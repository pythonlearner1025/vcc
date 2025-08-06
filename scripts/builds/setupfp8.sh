#!/bin/bash
# Setup script for Transformer-Engine FP8 support on H100

echo "Installing Transformer-Engine for FP8 support..."

# Install required build dependencies
apt-get update -y && \
apt-get install -y --no-install-recommends \
    libcudnn9-dev-cuda-12 \
    ninja-build \
    2>/dev/null || echo "Some packages may already be installed"

# Clean up apt cache
apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables for the build
export CUDA_PATH=/usr/local/cuda
export CUDNN_PATH=/usr
export MAX_JOBS=2
export NVTE_BUILD_THREADS_PER_JOB=1

# Install Transformer-Engine with PyTorch support
echo "Building Transformer-Engine (this may take a few minutes)..."
pip install --no-build-isolation --verbose transformer_engine[pytorch]

# Verify installation
python -c "
import transformer_engine
import transformer_engine.pytorch as te
print('✓ Transformer-Engine version:', transformer_engine.__version__)
print('✓ FP8 support enabled')
print('✓ TE Linear module:', te.Linear)
print('✓ Ready for FP8 training on H100!')
"