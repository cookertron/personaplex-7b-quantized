# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Retrieves the pretrained models for Moshi and Mimi."""
from pathlib import Path

from safetensors.torch import load_model, load_file
import torch
import torch.nn as nn
import bitsandbytes as bnb
from accelerate import init_empty_weights
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

from .compression import MimiModel
from .lm import LMModel
from ..modules import SEANetEncoder, SEANetDecoder, transformer
from ..quantization import SplitResidualVectorQuantizer

SAMPLE_RATE = 24000
FRAME_RATE = 12.5

TEXT_TOKENIZER_NAME = 'tokenizer_spm_32k_3.model'
MOSHI_NAME = 'model.safetensors'
MIMI_NAME = 'tokenizer-e351c8d8-checkpoint125.safetensors'
DEFAULT_REPO = 'nvidia/personaplex-7b-v1'


_seanet_kwargs = {
    "channels": 1,
    "dimension": 512,
    "causal": True,
    "n_filters": 64,
    "n_residual_layers": 1,
    "activation": "ELU",
    "compress": 2,
    "dilation_base": 2,
    "disable_norm_outer_blocks": 0,
    "kernel_size": 7,
    "residual_kernel_size": 3,
    "last_kernel_size": 3,
    # We train using weight_norm but then the weights are pre-processed for inference so
    # that we can use a normal convolution.
    "norm": "none",
    "pad_mode": "constant",
    "ratios": [8, 6, 5, 4],
    "true_skip": True,
}
_quantizer_kwargs = {
    "dimension": 256,
    "n_q": 32,
    "bins": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimension": _seanet_kwargs["dimension"],
}
_transformer_kwargs = {
    "d_model": _seanet_kwargs["dimension"],
    "num_heads": 8,
    "num_layers": 8,
    "causal": True,
    "layer_scale": 0.01,
    "context": 250,
    "conv_layout": True,
    "max_period": 10000,
    "gating": "none",
    "norm": "layer_norm",
    "positional_embedding": "rope",
    "dim_feedforward": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimensions": [_seanet_kwargs["dimension"]],
}

_lm_kwargs = {
    "dim": 4096,
    "text_card": 32000,
    "existing_text_padding_id": 3,
    "n_q": 16,
    "dep_q": 8,
    "card": _quantizer_kwargs["bins"],
    "num_heads": 32,
    "num_layers": 32,
    "hidden_scale": 4.125,
    "causal": True,
    "layer_scale": None,
    "context": 3000,
    "max_period": 10000,
    "gating": "silu",
    "norm": "rms_norm_f32",
    "positional_embedding": "rope",
    "depformer_dim": 1024,
    "depformer_dim_feedforward": int(4.125 * 1024),
    "depformer_num_heads": 16,
    "depformer_num_layers": 6,
    "depformer_causal": True,
    "depformer_layer_scale": None,
    "depformer_multi_linear": True,
    "depformer_context": 8,
    "depformer_max_period": 10000,
    "depformer_gating": "silu",
    "depformer_pos_emb": "none",
    "depformer_weights_per_step": True,
    "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
}


def _is_safetensors(path: Path | str) -> bool:
    return Path(path).suffix in (".safetensors", ".sft", ".sfts")


def get_mimi(filename: str | Path,
             device: torch.device | str = 'cpu') -> MimiModel:
    """Return a pretrained Mimi model."""
    encoder = SEANetEncoder(**_seanet_kwargs)
    decoder = SEANetDecoder(**_seanet_kwargs)
    encoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    decoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    quantizer = SplitResidualVectorQuantizer(
        **_quantizer_kwargs,
    )
    model = MimiModel(
        encoder,
        decoder,
        quantizer,
        channels=1,
        sample_rate=SAMPLE_RATE,
        frame_rate=FRAME_RATE,
        encoder_frame_rate=SAMPLE_RATE / encoder.hop_length,
        causal=True,
        resample_method="conv",
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
    ).to(device=device)
    model.eval()
    if _is_safetensors(filename):
        load_model(model, filename)
    else:
        pkg = torch.load(filename, "cpu")
        model.load_state_dict(pkg["model"])
    model.set_num_codebooks(8)
    return model


def get_moshi_lm(
    filename: str | Path | None,
    copy_missing_weights: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    delays=None,
    quantize: str | None = None,
) -> LMModel:
    model_class = LMModel
    lm_kwargs = dict(_lm_kwargs)
    lm_kwargs["dep_q"] = 16
    if delays is not None:
        lm_kwargs["delays"] = delays
    
    if quantize:
        # Patch ActivationGating to use module forward instead of functional F.linear
        # This is required because quantization replaces Linear layers with bnb layers,
        # and passing bnb weights (Int8/NF4) to F.linear fails.
        from ..modules.gating import ActivationGating
        
        def new_forward(self, x: torch.Tensor):
            x = self.linear_in(x)
            B, T, _ = x.shape
            x = x.view(B, T, 2, -1)
            x = self.activation(x[..., 0, :]) * x[..., 1, :]
            x = self.linear_out(x)
            return x
            
        print("Patching ActivationGating.forward for quantization compatibility...")
        ActivationGating.forward = new_forward

        print(f"Initializing empty weights for quantization: {quantize}")
        with init_empty_weights():
            model = LMModel(device=torch.device("meta"), dtype=dtype, **lm_kwargs)
            
    else:
        model = LMModel(device=device, dtype=dtype, **lm_kwargs).to(device=device, dtype=dtype)

    if filename is None:
        model.eval()
        return model

    filename = str(filename)

    # Load state_dict
    if filename.endswith(".safetensors"):
        # Load to CPU to avoid OOM before quantization
        state_dict = load_file(filename, device="cpu")
    else:
        # torch checkpoint
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

    # Patch 1: expand depformer self_attn weights if needed
    model_sd = model.state_dict()
    for name, tensor in list(state_dict.items()):
        if "depformer" in name and "self_attn" in name and name in model_sd:
             # handle meta device mismatch during shape check
             target_shape = model_sd[name].shape
             if tensor.shape != target_shape:
                print("Expanding %s", name)
                missing = (
                    tensor
                    if copy_missing_weights
                    else model_sd[name][tensor.shape[0] :]
                )
                state_dict[name] = torch.concat([tensor, missing], dim=0)

    # Patch 2: fill missing keys by copying 0..7 -> 8..15 for certain groups
    if copy_missing_weights:
        to_replace = ["gating", "linears", "depformer_in", "depformer_emb", "emb"]
        for name in model_sd.keys():
            if name in state_dict:
                continue
            replaced = False
            for old, new in zip(range(8), range(8, 16)):
                for rep in to_replace:
                    needle = f"{rep}.{new}."
                    if needle in name:
                        src = name.replace(needle, f"{rep}.{old}.")
                        if src in state_dict:
                            # print("Replacing %s <- %s", name, src)
                            state_dict[name] = state_dict[src]
                            replaced = True
                        break
                if replaced:
                    break
            if not replaced:
                pass
                # print("Missing %s", name)

    if quantize:
        from safetensors.torch import save_file
        import tempfile
        import os
        import json
        
        # Separate depformer weights (to be loaded unquantized) from main weights
        print("Separating depformer weights...")
        dep_keys = [k for k in state_dict.keys() if "depformer" in k]
        dep_sd = {k: state_dict[k] for k in dep_keys}
        for k in dep_keys:
            del state_dict[k]
            
        # temporarily replace depformer modules to skip quantization
        real_depformer = model.depformer
        real_depformer_in = model.depformer_in
        real_depformer_emb = model.depformer_emb
        real_depformer_text_emb = getattr(model, "depformer_text_emb", None)
        
        model.depformer = torch.nn.Identity()
        model.depformer_in = torch.nn.Identity()
        model.depformer_emb = torch.nn.Identity()
        if real_depformer_text_emb is not None:
            model.depformer_text_emb = torch.nn.Identity()
        
        # Save patched state_dict to a TEMPORARY DIRECTORY as SHARDED checkpoints
        # This allows accelerate to load one shard at a time, avoiding OOM on GPU.
        print("Saving patched state dict to temporary directory (sharded) for quantization...")
        
        temp_dir = tempfile.mkdtemp(suffix="_quant_shards")
        
        try:
            # Sharding logic
            shards = {}
            current_shard = {}
            current_size = 0
            shard_id = 0
            MAX_SHARD_SIZE = 1000 * 1024 * 1024 # 1GB
            
            # Helper to estimate tensor size
            def get_tensor_size(t):
                return t.nelement() * t.element_size()
            
            keys = list(state_dict.keys())
            for k in keys:
                t = state_dict[k]
                size = get_tensor_size(t)
                current_shard[k] = t
                current_size += size
                shards[k] = f"model-{shard_id:05d}.safetensors"
                
                # Check if we should flush
                if current_size >= MAX_SHARD_SIZE:
                    shard_name = f"model-{shard_id:05d}.safetensors"
                    save_file(current_shard, os.path.join(temp_dir, shard_name))
                    current_shard = {}
                    current_size = 0
                    shard_id += 1
            
            # Flush remaining
            if current_shard:
                shard_name = f"model-{shard_id:05d}.safetensors"
                for k in current_shard:
                    shards[k] = shard_name
                save_file(current_shard, os.path.join(temp_dir, shard_name))
            
            # Save index.json
            index = {"weight_map": shards}
            with open(os.path.join(temp_dir, "model.safetensors.index.json"), "w") as f:
                json.dump(index, f)
            
            # Clear state_dict from RAM
            del state_dict
            if "model_sd" in locals():
                del model_sd
            import gc
            gc.collect()

            print(f"Quantizing model to {quantize} using accelerate (sharded)...")
            
            bnb_config = BnbQuantizationConfig(
                load_in_4bit=(quantize == "4bit"),
                load_in_8bit=(quantize == "8bit"),
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                llm_int8_threshold=6.0,
            )
            
            device_map = {"": str(device)}
            
            load_and_quantize_model(
                model,
                bnb_quantization_config=bnb_config,
                weights_location=temp_dir,
                device_map=device_map, 
                offload_folder="/tmp/offload",
                offload_state_dict=True
            )
            
            print("Model loaded and quantized successfully.")
            
            # Restore depformer modules and load their weights
            print("Restoring depformer and loading unquantized weights...")
            
            model.depformer = real_depformer
            model.depformer_in = real_depformer_in
            model.depformer_emb = real_depformer_emb
            if real_depformer_text_emb is not None:
                model.depformer_text_emb = real_depformer_text_emb
            
            # Load dep_sd. We need to handle sub-modules manually or put them back in a dict
            # dep_sd contains keys like "depformer.xxx", "depformer_in.xxx", "depformer_emb.xxx"
            # Since we attach them to 'model', 'model.load_state_dict(dep_sd)' should work partially if strict=False
            # But 'model' is now quantized and on GPU/CPU.
            # We want to load specifically to these submodules.
            
            # The easiest way is to call model.load_state_dict(dep_sd, strict=False)
            # This will find keys for depformer*, load them, and ignore missing keys (which are already loaded).
            # assign=True is needed because they are meta.
            
            model.load_state_dict(dep_sd, assign=True, strict=False)
            
            # Move them to device and cast to float16 (since main model output is float16 due to quantization)
            model.depformer.to(device, dtype=torch.float16)
            model.depformer_in.to(device, dtype=torch.float16)
            model.depformer_emb.to(device, dtype=torch.float16)
            if real_depformer_text_emb is not None:
                model.depformer_text_emb.to(device, dtype=torch.float16)
            
            # Clear dep_sd
            del dep_sd
            
        finally:
            import shutil
            if os.path.exists(temp_dir):
                print("Removing temporary shard directory...")
                shutil.rmtree(temp_dir)
    
    else:
        model.load_state_dict(state_dict, strict=False, assign=True)
        model.to(device)
        model.eval()
        
    return model
