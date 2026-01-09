#!/bin/bash
# FlashVSR-Pro container entrypoint script
# Automatically activate the Conda environment and set required paths

# Exit immediately if a command exits with a non-zero status
set -e

# Check if Conda is available
if [ ! -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    echo "[ERROR] Conda not found at /opt/conda/etc/profile.d/conda.sh"
    exit 1
fi

# Source Conda initialization
source /opt/conda/etc/profile.d/conda.sh

# Try to activate the flashvsr environment
if ! conda activate flashvsr; then
    echo "[ERROR] Failed to activate 'flashvsr' conda environment"
    echo "Available environments:"
    conda env list
    exit 1
fi

# Activate the required project environment
conda activate flashvsr

# Set runtime library path for PyTorch C++ libs to avoid missing libc10.so issues
export LD_LIBRARY_PATH="/opt/conda/envs/flashvsr/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"

# Optional: change to project directory
cd /workspace/FlashVSR-Pro

# Print confirmation messages (helpful for debugging)
echo "[INFO] Conda environment 'flashvsr' activated."
echo "[INFO] Python path: $(which python)"

# Execute provided command (exec ensures proper signal forwarding)
exec "$@"