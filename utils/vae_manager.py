# utils/vae_manager.py
"""
VAE Manager for FlashVSR-Pro
Unified interface for multiple VAE variants with quality/VRAM trade-offs.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Union, Any
import warnings
import os

from .TCDecoder import build_tcdecoder, TAEW2_1DiffusersWrapper, TAEHV
from .tile_utils import vae_decode_tiled


class VAEManager:
    """
    Unified VAE manager supporting multiple decoder variants.
    Provides consistent interface with varying quality/VRAM trade-offs.
    """
    
    VAE_REGISTRY = {
        "wan2.1": {
            "class": "WanVAE",
            "default_path": "models/VAEs/Wan2.1_VAE.pth",
            "is_tcdecoder": False,
            "description": "Original Wan VAE 2.1 - High quality, moderate VRAM"
        },
        "wan2.2": {
            "class": "WanVAE",
            "default_path": "models/VAEs/Wan2.2_VAE.pth",
            "is_tcdecoder": False,
            "description": "Wan VAE 2.2 - Best quality, highest VRAM"
        },
        "light": {
            "class": "WanVAE",
            "default_path": "models/VAEs/lightvaew2_1.pth",
            "is_tcdecoder": False,
            "description": "Lightweight Wan VAE - Reduced VRAM for high-res"
        },
        "tcd": {
            "class": "TAEW2_1DiffusersWrapper",
            "default_path": None,
            "channels": [512, 256, 128, 128],
            "is_tcdecoder": True,
            "description": "Tiny Conditional Decoder - Fastest with moderate quality"
        },
        "tae-hv": {
            "class": "TAEW2_1DiffusersWrapper",
            "default_path": "models/VAEs/lighttaehy1_5.pth",
            "channels": [256, 128, 64, 64],
            "is_tcdecoder": True,
            "description": "Light TAE-HV - Good quality/VRAM balance for video"
        },
        "tae-w2.2": {
            "class": "TAEW2_1DiffusersWrapper",
            "default_path": "models/VAEs/taew2_2.safetensors",
            "channels": [256, 128, 64, 64],
            "is_tcdecoder": True,
            "description": "TAE W2.2 - Improved TAE-HV variant"
        }
    }
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        """Initialize VAE Manager."""
        self.device = device
        self.dtype = dtype
        self.current_vae = None
        self.current_type = None
        
    def load_vae(self,
                 vae_type: str = "wan2.1",
                 custom_path: Optional[str] = None,
                 mode: str = "full",
                 tile_vae: bool = False,
                 tile_size: int = 512,
                 overlap: int = 32,
                 model_dir: str = "./models/FlashVSR") -> nn.Module:
        """Load specified VAE model."""
        vae_type = vae_type.lower()
        if vae_type not in self.VAE_REGISTRY:
            raise ValueError(
                f"Unsupported VAE type: '{vae_type}'. "
                f"Available: {list(self.VAE_REGISTRY.keys())}"
            )
        
        config = self.VAE_REGISTRY[vae_type]
        self.current_type = vae_type
        
        # Determine weight path
        if custom_path:
            weight_path = custom_path
        elif config["default_path"]:
            weight_path = config["default_path"]
        else:
            weight_path = os.path.join(model_dir, "TCDecoder.ckpt")
        
        if weight_path and not os.path.exists(weight_path):
            warnings.warn(
                f"VAE weights not found: {weight_path}. "
                f"Model '{vae_type}' may not load correctly."
            )
        
        print(f"Loading VAE: {vae_type}")
        
        # Load model based on type
        if config["is_tcdecoder"]:
            vae_model = self._load_tcdecoder_vae(vae_type, config, weight_path, mode)
        else:
            vae_model = self._load_wan_vae(vae_type, config, weight_path, mode)
        
        self.current_vae = vae_model
        
        # Enable tiling if requested
        if tile_vae and hasattr(vae_model, 'decode'):
            self._wrap_decode_for_tiling(tile_size, overlap)
        
        print(f"VAE ready: {config['description']}")
        
        return vae_model
    
    def _load_tcdecoder_vae(self, 
                           vae_type: str, 
                           config: Dict, 
                           weight_path: str,
                           mode: str) -> nn.Module:
        """Load TCDecoder type VAE using specialized loaders."""
        from .vae_loaders import VAELoaderFactory

        channels = config.get("channels", [512, 256, 128, 128])

        # Use specialized loader
        vae_model = VAELoaderFactory.load_vae(vae_type, weight_path, channels=channels)

        # Move to device and dtype
        vae_model = vae_model.to(device=self.device, dtype=self.dtype)

        return vae_model
    
    def _load_wan_vae(self,
                    vae_type: str,
                    config: Dict,
                    weight_path: str,
                    mode: str) -> nn.Module:
        """Load Wan VAE type for full mode using specialized loader."""
        from .vae_loaders import VAELoaderFactory

        # Use specialized loader
        vae_model = VAELoaderFactory.load_vae(vae_type, weight_path)

        # Move to device and dtype in single call, then remove encoder
        vae_model = vae_model.to(device=self.device, dtype=self.dtype)

        # Remove encoder to save memory (if present)
        if hasattr(vae_model, 'encoder'):
            vae_model.encoder = None
        if hasattr(vae_model, 'conv1'):
            vae_model.conv1 = None

        return vae_model
    
    def _wrap_decode_for_tiling(self, tile_size: int, overlap: int):
        """Enable tiled VAE decoding."""
        if not self.current_vae or not hasattr(self.current_vae, 'decode'):
            return
        
        original_decode = self.current_vae.decode
        
        def tiled_decode(latents):
            return vae_decode_tiled(
                self.current_vae, 
                latents,
                tile_size=tile_size,
                overlap=overlap
            )
        
        self.current_vae.decode = tiled_decode
        print(f"VAE tiled decoding enabled (tile_size={tile_size}, overlap={overlap})")
    
    def get_current_vae_info(self) -> Dict:
        """Get information about currently loaded VAE."""
        if not self.current_type:
            return {}
        
        config = self.VAE_REGISTRY.get(self.current_type, {})
        return {
            'type': self.current_type,
            'description': config.get('description', ''),
            'is_tcdecoder': config.get('is_tcdecoder', False),
            'weight_path': config.get('default_path', '')
        }
    
    def clean_memory(self):
        """Clean up memory and reset state."""
        if self.current_vae:
            if hasattr(self.current_vae, 'clean_mem'):
                self.current_vae.clean_mem()
            elif hasattr(self.current_vae, 'tcd_model') and hasattr(self.current_vae.tcd_model, 'clean_mem'):
                self.current_vae.tcd_model.clean_mem()
        
        self.current_vae = None
        self.current_type = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()