#!/bin/bash
# Quick install script for Transformer-Engine FP8 support

echo "Installing Transformer-Engine for FP8 support on H100..."

# Check if wheel exists
WHEEL_PATH="/workspace/vcc/transformer_engine-1.9.0+e79d915a-cp311-cp311-linux_x86_64.whl"

if [ -f "$WHEEL_PATH" ]; then
    echo "✓ Found pre-built wheel, installing..."
    pip install --force-reinstall "$WHEEL_PATH"
else
    echo "❌ Wheel not found at $WHEEL_PATH"
    echo "Run build_te_from_source.sh first to build the wheel"
    exit 1
fi

# Test installation
python -c "
import transformer_engine
import transformer_engine.pytorch as te
print('✓ Transformer-Engine version:', transformer_engine.__version__)
print('✓ FP8 support installed successfully')
print('')
print('IMPORTANT NOTES FOR FP8:')
print('------------------------')
print('1. FP8 requires batch sizes divisible by 8')
print('2. Hidden dimensions must be divisible by 16')
print('3. Your model dim=256 is compatible (256 % 16 = 0)')
print('4. Use batch sizes like 8, 16, 24, 32, etc.')
print('')
print('To enable FP8 in training:')
print('- Set use_fp8=True in ConditionalModelConfig')
print('- The training script will automatically use FP8 autocast')
"