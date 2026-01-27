#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""
PersonaPlex Universal Installer

This script simplifies the installation process by:
1. Checking system requirements (Python, CUDA, VRAM, disk space)
2. Setting up dependencies automatically
3. Managing HuggingFace authentication
4. Downloading required model files
5. Verifying the installation
"""

import os
import sys
import subprocess
import shutil
import platform
import argparse
from pathlib import Path

# Minimum requirements
MIN_PYTHON_VERSION = (3, 10)
MIN_DISK_SPACE_GB = 15
RECOMMENDED_VRAM_GB = 8


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header():
    """Print the installer header."""
    print(f"""
{Colors.CYAN}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════════╗
║                  PersonaPlex Installer                        ║
║     Voice and Role Control for Full Duplex Conversation       ║
╚═══════════════════════════════════════════════════════════════╝
{Colors.END}""")


def print_step(step_num: int, total: int, message: str):
    """Print a step indicator."""
    print(f"\n{Colors.BLUE}[{step_num}/{total}]{Colors.END} {Colors.BOLD}{message}{Colors.END}")


def print_success(message: str):
    """Print a success message."""
    print(f"  {Colors.GREEN}✓{Colors.END} {message}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"  {Colors.YELLOW}⚠{Colors.END} {message}")


def print_error(message: str):
    """Print an error message."""
    print(f"  {Colors.RED}✗{Colors.END} {message}")


def print_info(message: str):
    """Print an info message."""
    print(f"  {Colors.CYAN}ℹ{Colors.END} {message}")


def check_python_version() -> bool:
    """Check if Python version meets requirements."""
    current = sys.version_info[:2]
    if current >= MIN_PYTHON_VERSION:
        print_success(f"Python {current[0]}.{current[1]} (required: {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+)")
        return True
    else:
        print_error(f"Python {current[0]}.{current[1]} is too old. Required: {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+")
        return False


def check_cuda_availability() -> tuple[bool, str]:
    """Check CUDA availability and version."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version,name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    driver, name, memory = parts[0], parts[1], parts[2]
                    print_success(f"GPU: {name}")
                    print_success(f"VRAM: {memory}")
                    print_success(f"Driver: {driver}")

                    # Parse VRAM
                    vram_mb = int(''.join(filter(str.isdigit, memory)))
                    vram_gb = vram_mb / 1024
                    if vram_gb < RECOMMENDED_VRAM_GB:
                        print_warning(f"VRAM ({vram_gb:.1f}GB) is below recommended {RECOMMENDED_VRAM_GB}GB")
                        print_info("4-bit quantization will be used to reduce memory usage")
                    return True, "cuda"
        return False, "cpu"
    except FileNotFoundError:
        print_warning("nvidia-smi not found - CUDA may not be available")
        return False, "cpu"
    except Exception as e:
        print_warning(f"Could not check GPU: {e}")
        return False, "cpu"


def check_disk_space() -> bool:
    """Check available disk space."""
    try:
        total, used, free = shutil.disk_usage(Path.cwd())
        free_gb = free / (1024 ** 3)
        if free_gb >= MIN_DISK_SPACE_GB:
            print_success(f"Disk space: {free_gb:.1f}GB available (required: {MIN_DISK_SPACE_GB}GB)")
            return True
        else:
            print_error(f"Insufficient disk space: {free_gb:.1f}GB (required: {MIN_DISK_SPACE_GB}GB)")
            return False
    except Exception as e:
        print_warning(f"Could not check disk space: {e}")
        return True


def check_existing_installation() -> bool:
    """Check if moshi package is already installed."""
    try:
        import moshi
        print_info("Existing moshi installation detected")
        return True
    except ImportError:
        return False


def get_hf_token() -> str | None:
    """Get HuggingFace token from environment or user input."""
    # Check environment variable
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if token:
        print_success("HuggingFace token found in environment")
        return token

    # Check huggingface-cli login
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print_success("HuggingFace token found from previous login")
            return token
    except ImportError:
        pass
    except Exception:
        pass

    # Prompt user
    print_info("HuggingFace token required for model download")
    print_info("Get your token at: https://huggingface.co/settings/tokens")
    print_info("You must also accept the license at: https://huggingface.co/nvidia/personaplex-7b-v1")
    print()

    token = input(f"  {Colors.CYAN}Enter your HuggingFace token (or press Enter to skip): {Colors.END}").strip()

    if token:
        return token
    return None


def save_hf_token(token: str):
    """Save HuggingFace token to config."""
    try:
        from huggingface_hub import HfFolder
        HfFolder.save_token(token)
        print_success("Token saved for future use")
    except Exception as e:
        print_warning(f"Could not save token: {e}")


def install_dependencies(use_cuda: bool = True):
    """Install Python dependencies."""
    print_info("Installing dependencies...")

    # Determine which requirements to use
    script_dir = Path(__file__).parent

    # Install torch first with appropriate CUDA version
    if use_cuda:
        print_info("Installing PyTorch with CUDA support...")
        torch_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch==2.4.1+cu121',
            '--index-url', 'https://download.pytorch.org/whl/cu121',
            '-q'
        ]
    else:
        print_info("Installing PyTorch (CPU only)...")
        torch_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch==2.4.1',
            '-q'
        ]

    result = subprocess.run(torch_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print_error(f"Failed to install PyTorch: {result.stderr}")
        return False
    print_success("PyTorch installed")

    # Install other requirements
    req_file = script_dir / 'requirements.txt'
    if req_file.exists():
        print_info("Installing remaining dependencies...")
        # Filter out torch from requirements since we already installed it
        with open(req_file) as f:
            reqs = [line.strip() for line in f if line.strip() and not line.startswith('torch') and not line.startswith('#')]

        for req in reqs:
            if req == '-e .':
                # Install local package
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', '-e', str(script_dir), '-q'],
                    capture_output=True, text=True
                )
            else:
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', req, '-q'],
                    capture_output=True, text=True
                )

            if result.returncode != 0:
                print_warning(f"Issue installing {req}: {result.stderr[:100]}")

    # Install GUI dependencies
    print_info("Installing GUI dependencies...")
    gui_deps = ['gradio>=4.0.0', 'pyyaml>=6.0']
    for dep in gui_deps:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', dep, '-q'],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print_warning(f"Issue installing {dep}")

    print_success("Dependencies installed")
    return True


def download_models(token: str | None):
    """Download model files from HuggingFace."""
    print_info("Downloading model files (this may take a while)...")

    if token:
        os.environ['HF_TOKEN'] = token

    try:
        from huggingface_hub import hf_hub_download, snapshot_download

        repo_id = "nvidia/personaplex-7b-v1"

        # Download essential files
        files_to_download = [
            "config.json",
            "tokenizer.model",
            "mimi.safetensors",
            "voices.tgz",
        ]

        for filename in files_to_download:
            try:
                print_info(f"Downloading {filename}...")
                hf_hub_download(repo_id, filename)
                print_success(f"Downloaded {filename}")
            except Exception as e:
                if "401" in str(e) or "403" in str(e):
                    print_error(f"Access denied for {filename}. Please ensure you've accepted the license.")
                    print_info("Visit: https://huggingface.co/nvidia/personaplex-7b-v1")
                    return False
                print_warning(f"Could not download {filename}: {e}")

        # Download model shards
        print_info("Downloading model weights (multiple shards)...")
        try:
            # The model is sharded, download all safetensors files
            snapshot_download(
                repo_id,
                allow_patterns=["*.safetensors", "*.json", "*.model"],
                ignore_patterns=[".*"],
            )
            print_success("Model weights downloaded")
        except Exception as e:
            print_warning(f"Some model files may not have downloaded: {e}")

        return True

    except ImportError:
        print_error("huggingface_hub not installed. Run: pip install huggingface-hub")
        return False
    except Exception as e:
        print_error(f"Error downloading models: {e}")
        return False


def create_config_file(device: str):
    """Create default configuration file."""
    config_path = Path(__file__).parent / 'config.yaml'

    if config_path.exists():
        print_info("Configuration file already exists")
        return

    config_content = f"""# PersonaPlex Configuration
# Generated by install.py

server:
  host: localhost
  port: 8998
  quantize: 4bit
  device: {device}

defaults:
  voice_prompt: NATF2.pt
  text_prompt: "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."

# Model settings (usually don't need to change)
model:
  hf_repo: nvidia/personaplex-7b-v1

# GUI settings
gui:
  theme: default
  auto_start_server: true
"""

    with open(config_path, 'w') as f:
        f.write(config_content)

    print_success(f"Configuration saved to {config_path}")


def verify_installation() -> bool:
    """Verify that the installation works."""
    print_info("Verifying installation...")

    try:
        import torch
        print_success(f"PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            print_success(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print_info("CUDA not available, will use CPU")

        import moshi
        print_success("moshi package imported successfully")

        import gradio
        print_success(f"Gradio {gradio.__version__}")

        return True

    except ImportError as e:
        print_error(f"Import error: {e}")
        return False
    except Exception as e:
        print_error(f"Verification failed: {e}")
        return False


def create_launcher_scripts():
    """Create platform-specific launcher scripts."""
    script_dir = Path(__file__).parent

    # Create shell script for Linux/Mac
    sh_script = script_dir / 'start_personaplex.sh'
    sh_content = '''#!/bin/bash
# PersonaPlex Launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Launch GUI
python -m gui.app "$@"
'''
    with open(sh_script, 'w') as f:
        f.write(sh_content)
    os.chmod(sh_script, 0o755)
    print_success(f"Created {sh_script}")

    # Create batch script for Windows
    bat_script = script_dir / 'start_personaplex.bat'
    bat_content = '''@echo off
REM PersonaPlex Launcher

cd /d "%~dp0"

REM Check for virtual environment
if exist "venv\\Scripts\\activate.bat" (
    call venv\\Scripts\\activate.bat
) else if exist ".venv\\Scripts\\activate.bat" (
    call .venv\\Scripts\\activate.bat
)

REM Launch GUI
python -m gui.app %*
'''
    with open(bat_script, 'w') as f:
        f.write(bat_content)
    print_success(f"Created {bat_script}")


def main():
    parser = argparse.ArgumentParser(description='PersonaPlex Installer')
    parser.add_argument('--skip-models', action='store_true', help='Skip model download')
    parser.add_argument('--cpu-only', action='store_true', help='Install for CPU only (no CUDA)')
    parser.add_argument('--no-gui', action='store_true', help='Skip GUI dependencies')
    parser.add_argument('--token', type=str, help='HuggingFace token')
    args = parser.parse_args()

    print_header()

    total_steps = 7
    current_step = 0
    errors = []

    # Step 1: Check Python version
    current_step += 1
    print_step(current_step, total_steps, "Checking Python version")
    if not check_python_version():
        print_error("Please upgrade Python to 3.10 or higher")
        sys.exit(1)

    # Step 2: Check CUDA/GPU
    current_step += 1
    print_step(current_step, total_steps, "Checking GPU and CUDA")
    if args.cpu_only:
        has_cuda = False
        device = "cpu"
        print_info("CPU-only mode selected")
    else:
        has_cuda, device = check_cuda_availability()
        if not has_cuda:
            print_warning("No CUDA GPU detected. Model will run on CPU (very slow)")
            response = input(f"  {Colors.CYAN}Continue anyway? [y/N]: {Colors.END}").strip().lower()
            if response != 'y':
                print_info("Installation cancelled")
                sys.exit(0)

    # Step 3: Check disk space
    current_step += 1
    print_step(current_step, total_steps, "Checking disk space")
    if not check_disk_space():
        errors.append("Insufficient disk space")

    # Step 4: Get HuggingFace token
    current_step += 1
    print_step(current_step, total_steps, "Setting up HuggingFace authentication")
    token = args.token or get_hf_token()
    if token:
        save_hf_token(token)
    else:
        print_warning("No token provided. You'll need to set HF_TOKEN before running.")

    # Step 5: Install dependencies
    current_step += 1
    print_step(current_step, total_steps, "Installing dependencies")
    if not install_dependencies(use_cuda=has_cuda):
        errors.append("Dependency installation failed")

    # Step 6: Download models
    current_step += 1
    print_step(current_step, total_steps, "Downloading models")
    if args.skip_models:
        print_info("Skipping model download (--skip-models)")
    elif not download_models(token):
        errors.append("Model download incomplete")

    # Step 7: Create configuration and launchers
    current_step += 1
    print_step(current_step, total_steps, "Creating configuration files")
    create_config_file(device)
    create_launcher_scripts()

    # Verify installation
    print_step(current_step, total_steps, "Verifying installation")
    if not verify_installation():
        errors.append("Verification failed")

    # Summary
    print(f"\n{Colors.CYAN}{'═' * 60}{Colors.END}")

    if errors:
        print(f"\n{Colors.YELLOW}Installation completed with warnings:{Colors.END}")
        for error in errors:
            print(f"  {Colors.YELLOW}•{Colors.END} {error}")
    else:
        print(f"\n{Colors.GREEN}{Colors.BOLD}Installation completed successfully!{Colors.END}")

    print(f"""
{Colors.CYAN}To start PersonaPlex:{Colors.END}

  Option 1 - GUI (recommended):
    {Colors.BOLD}python -m gui.app{Colors.END}

  Option 2 - Server only:
    {Colors.BOLD}./run_server.sh{Colors.END}
    or
    {Colors.BOLD}python -m moshi.server --quantize 4bit{Colors.END}

  Option 3 - Use launcher scripts:
    Linux/Mac: {Colors.BOLD}./start_personaplex.sh{Colors.END}
    Windows:   {Colors.BOLD}start_personaplex.bat{Colors.END}

{Colors.CYAN}Web UI will be available at:{Colors.END} http://localhost:8998

{Colors.YELLOW}Note:{Colors.END} You must use 'localhost' (not IP) for microphone access.
""")


if __name__ == '__main__':
    main()
