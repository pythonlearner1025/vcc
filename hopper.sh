#!/bin/bash

set -euxo pipefail

# ----[ 0. Environment sanity checks ]----
echo "Using Python: $(which python)"
python --version
nvcc --version
nvidia-smi

# ----[ 1. Install build dependencies ]----
# ----[ 3. Optional: Set env flags for Hopper ]----
# This ensures proper architecture flags for H100 (SM 9.0)
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=$(nproc)
pip install --upgrade pip setuptools wheel ninja cmake packaging
pip install --upgrade flash-attn --no-build-isolation --no-binary flash-attn

# ----[ 2. Optional: Clean up torch cache if retrying ]----
# rm -rf ~/.cache/torch_extensions

# ----[ 4. Done ]----
python -c "import flash_attn; print('✅ FlashAttention build successful.')"
