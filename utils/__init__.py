# utils/__init__.py
"""
FlashVSR utility modules
"""

# Core utilities
from .core.utils import (
    RMS_norm,
    CausalConv3d,
    PixelShuffle3d,
    Buffer_LQ4x_Proj,
    Causal_LQ4x_Proj,
)

from .core.TCDecoder import build_tcdecoder

# VAE system
from .vae import vae_system

# Audio and tiling utilities will be imported dynamically when used
# Avoid forcing dependencies
