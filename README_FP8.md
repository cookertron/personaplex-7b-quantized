# PersonaPlex 8-bit Quantization Setup

## Overview
This setup facilitates running the **PersonaPlex-7B-v1** model (approximately 16.7GB in FP16) on NVIDIA GPUs with **16GB VRAM** using 8-bit quantization. By reducing the model footprint to ~14.7GB, it fits comfortably within the 16GB envelope with room for inference overhead.

## Requirements
- **Hardware**: NVIDIA GPU with at least 16GB VRAM.
- **Software Environment**: `micromamba` environment `personaplex` (provided in repo).
- **Key Dependencies**:
  - `bitsandbytes`: For 8-bit quantization kernels.
  - `accelerate`: For handling model loading and offloading.
  - `moshi-personaplex`: This codebase.

### Tested Configuration
- **Hardware**: NVIDIA GPU with 16GB VRAM (Verified working).
- **OS**: Linux (Ubuntu).
- **Python**: 3.11.14
- **PyTorch**: 2.4.1+cu121
- **CUDA**: 12.1
- **Bitsandbytes**: 0.49.1
- **Accelerate**: 1.12.0

## Usage

### Running the Server
The easiest way to run the 8-bit quantized server is using the provided helper script:

```bash
./run_server.sh
```

This runs the server on **http://localhost:8998**.
**Important**: You must access it via `localhost` (not IP address) for the browser to allow microphone access and audio processing.

This script is pre-configured to run:
```bash
micromamba run -n personaplex python -m moshi.server --quantize 8bit
```

### Command Line Arguments
- `--quantize 4bit`: **(Default)** Enables 4-bit quantization (NF4). Recommended for best stability and lowest VRAM usage (~8GB).
- `--quantize 8bit`: Enables 8-bit quantization.

## Technical Details & Modifications

To support quantization on this specific architecture, several targeted modifications were made to the original Moshi codebase:

### 1. Sharded Checkpoint Loading (`moshi/models/loaders.py`)
Standard loading places the full model in RAM/VRAM before quantization, which causes OOM. We implemented a **sharded loading** mechanism:
- The model skeleton is initialized on the `meta` device (zero memory).
- Weights are loaded from disk and immediately saved to a temporary directory in small **1GB sharded checkpoints**.
- `accelerate.utils.load_and_quantize_model` reads these shards sequentially, quantizing layers one-by-one and moving them to the GPU. This ensures peak memory usage never exceeds the quantized model size.

### 2. `ActivationGating` Patch (`moshi/models/loaders.py`)
The `ActivationGating` module originally used `torch.nn.functional.linear` with raw weights.
- **Issue**: `bitsandbytes` replaces `nn.Linear` with `Linear8bitLt` modules. Passing these quantized module weights directly to `F.linear` causes a type mismatch/crash.
- **Fix**: We patch `ActivationGating.forward` at runtime to use `self.linear_in(x)` and `self.linear_out(x)` instead of `F.linear`. This correctly invokes the `bitsandbytes` forward pass.

### 3. `depformer` Exclusion (`moshi/models/loaders.py`)
The `depformer` sub-modules use a custom `multi_linear` function for streaming, which is incompatible with quantized layers.
- **Fix**:
    1.  `depformer` modules (`depformer`, `depformer_in`, `depformer_emb`, `depformer_text_emb`) are **excluded** from quantization.
    2.  They are temporarily replaced with `nn.Identity` during the `bitsandbytes` quantization pass.
    3.  After quantization, the original modules are restored, and their original Float16 weights are loaded onto the GPU.

### 4. Direct Buffer/Parameter Access
We added logic to manually handle specific parameters (like `depformer_text_emb` and `emb` layers) to ensure they are correctly copied and initialized, preventing "weight is on meta device" errors.

### 5. CUDA Graphs Modified (`moshi/models/lm.py`)
- **Issue**: `bitsandbytes` kernels do not currently support CUDA Graph capture, which Moshi uses for optimization.
- **Fix**: The `LMGen` class now detects if `bitsandbytes` modules are present in the model. If detected, it automatically disables CUDA Graph capture (`disable=True`), trading a small amount of speed for compatibility.

## Performance
- **VRAM Usage**: ~14.7 GB (Static allocation via `RingKVCache`).
- **Inference Speed**: 8-bit inference is generally comparable to FP16, with a slight overhead from quantization kernels, but remains fast enough for real-time interaction.

## Troubleshooting

### "ValueError: weight is on the meta device"
This indicates a parameter was missed during the sharded loading process.
- **Check**: Ensure any new sub-modules added to the model are also added to the exclusion/manual loading list in `loaders.py`.

### OOM Errors
- **Check**: Verify no other heavy processes are using the GPU.
- **Check**: Ensure `run_server.sh` is actually using `--quantize 8bit`.

### "RuntimeError: CUDA error: operation failed due to a previous error during capture"
- **Check**: This means CUDA Graphs are trying to capture an incompatible operation. Ensure `LMGen` in `lm.py` is correctly detecting the `bitsandbytes` module and disabling graphs.
