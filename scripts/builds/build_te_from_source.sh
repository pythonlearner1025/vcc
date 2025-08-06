#!/bin/bash
# Build Transformer-Engine from source for FP8 support on H100

set -e

echo "=== Building Transformer-Engine from source for FP8 support ==="

# Install build dependencies
echo "Installing build dependencies..."
apt-get update -y && \
apt-get install -y --no-install-recommends \
    git \
    ninja-build \
    libcudnn9-dev-cuda-12 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables
export CUDA_HOME=/usr/local/cuda
export CUDNN_PATH=/usr
export NVTE_FRAMEWORK=pytorch
export MAX_JOBS=4
export NVTE_BUILD_THREADS_PER_JOB=1

# Use the correct CUDA architecture for H100 (sm_90)
export TORCH_CUDA_ARCH_LIST="9.0"
export NVTE_CUDA_ARCH="90"

# Clone Transformer-Engine repository
echo "Cloning Transformer-Engine repository..."
cd /tmp
rm -rf TransformerEngine
git clone --recursive https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine

# Checkout stable version
git checkout v1.11.0  # Latest stable version with good H100 support
git submodule update --init --recursive

# Build and install
echo "Building Transformer-Engine (this will take 5-10 minutes)..."
pip install --no-build-isolation --no-deps .

# Install the dependencies separately
pip install pydantic packaging einops

echo "=== Build complete ==="

# Test FP8
cd /workspace/vcc
python -c "
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch import fp8_autocast

print('Testing FP8 support...')
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    
# Test FP8 autocast
model = te.Linear(512, 512).cuda()
x = torch.randn(2, 32, 512).cuda()

try:
    with fp8_autocast(enabled=True):
        y = model(x)
        print('✅ FP8 forward pass successful!')
        print('Output shape:', y.shape)
except Exception as e:
    print(f'❌ FP8 failed: {e}')
    print('Trying with BF16 instead...')
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        y = model(x)
        print('✅ BF16 forward pass successful')
"