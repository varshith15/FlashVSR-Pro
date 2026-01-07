# utils/__init__.py
"""
FlashVSR utility modules
"""

from .utils import (
    RMS_norm,
    CausalConv3d,
    PixelShuffle3d,
    Buffer_LQ4x_Proj,
    Causal_LQ4x_Proj,
)

from .TCDecoder import build_tcdecoder

# 音频和分块工具将在使用时动态导入
# 避免强制依赖