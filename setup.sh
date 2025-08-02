#!/bin/bash

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH for current session
export PATH="$HOME/.local/bin:$PATH"

# Check if uv was installed successfully
if [ ! -f "$HOME/.local/bin/uv" ]; then
    echo "Error: uv installation failed!"
    exit 1
fi

# Add uv to PATH permanently by updating shell profile (only if not already present)
if [ -f "$HOME/.bashrc" ] && ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc"; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
elif [ -f "$HOME/.zshrc" ] && ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.zshrc"; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc"
elif [ -f "$HOME/.profile" ] && ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.profile"; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.profile"
fi

# Verify uv is accessible
if command -v uv > /dev/null 2>&1; then
    echo "uv installed and added to PATH successfully!"
    echo "uv version: $(uv --version)"
else
    echo "Error: uv is not accessible in PATH!"
    exit 1
fi

# Set UV_LINK_MODE to copy for cross-platform compatibility
# export UV_LINK_MODE=copy

# go into the workspace directory
cd workspace/vcc || { echo "Failed to change directory to workspace/vcc"; exit 1; }

# Install all packages using uv (from pyproject.toml)
echo "Installing packages with uv..."

uv sync

echo "Setup completed successfully!"
