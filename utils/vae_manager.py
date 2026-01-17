# utils/vae_manager.py
"""
VAE Manager for FlashVSR-Pro
Provides unified interface for multiple VAE options with different quality/VRAM trade-offs.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Union, Any
import warnings
import os

# Import existing TCDecoder utilities
from .TCDecoder import build_tcdecoder, TAEW2_1DiffusersWrapper, TAEHV
from .tile_utils import vae_decode_tiled


class VAEManager:
    """
    Unified VAE manager supporting multiple VAE variants.
    Provides consistent interface for different VAE types with varying quality/VRAM trade-offs.
    """
    
    # Registry mapping VAE types to their configurations
    VAE_REGISTRY = {
        # Original Wan VAE variants (full mode)
        "wan2.1": {
            "class": "WanVAE",  # Will be loaded via ModelManager
            "default_path": "models/VAEs/Wan2.1_VAE.pth",
            "is_tcdecoder": False,
            "description": "Original Wan VAE 2.1, best quality but highest VRAM usage."
        },
        "wan2.2": {
            "class": "WanVAE",  # Same class as wan2.1
            "default_path": "models/VAEs/Wan2.2_VAE.pth",
            "is_tcdecoder": False,
            "description": "Wan VAE 2.2, improved version with better quality/speed balance."
        },
        "light": {
            "class": "WanVAE",  # LightVAE uses same interface
            "default_path": "models/VAEs/lightvaew2_1.pth",
            "is_tcdecoder": False,
            "description": "Lightweight Wan VAE, reduced VRAM usage for high-resolution processing."
        },
        # TCDecoder variants (tiny mode)
        "tcd": {
            "class": "TAEW2_1DiffusersWrapper",
            "default_path": None,  # Will use TCDecoder.ckpt from model directory
            "channels": [512, 256, 128, 128],
            "is_tcdecoder": True,
            "description": "Tiny Conditional Decoder for 'tiny' mode, fastest with moderate quality."
        },
        "tae-hv": {
            "class": "TAEW2_1DiffusersWrapper",  # Similar wrapper for TAE-HV
            "default_path": "models/VAEs/lighttaehy1_5.pth",
            "channels": [256, 128, 64, 64],
            "is_tcdecoder": True,
            "description": "Light TAE-HV decoder, optimized for video with good quality/VRAM balance."
        },
        "tae-w2.2": {
            "class": "TAEW2_1DiffusersWrapper",
            "default_path": "models/VAEs/taew2_2.safetensors",
            "channels": [256, 128, 64, 64],
            "is_tcdecoder": True,
            "description": "TAE W2.2 decoder, improved version of TAE-HV."
        }
    }
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        """
        Initialize VAE Manager.
        
        Args:
            device: Target device (cuda/cpu)
            dtype: Data type for model parameters
        """
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
        """
        Load specified VAE model.
        
        Args:
            vae_type: VAE type key from registry
            custom_path: Custom weight file path (overrides default)
            mode: Inference mode ('full', 'tiny', 'tiny-long')
            tile_vae: Enable tiled decoding for VAE
            tile_size: Tile size for tiled decoding
            overlap: Overlap between tiles
            model_dir: Base model directory for TCDecoder weights
            
        Returns:
            Configured VAE model (nn.Module)
        """
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
            # For TCDecoder, use the model directory's TCDecoder.ckpt
            weight_path = os.path.join(model_dir, "TCDecoder.ckpt")
        
        # Check if weight file exists
        if weight_path and not os.path.exists(weight_path):
            warnings.warn(
                f"Weight file not found: {weight_path}. "
                f"VAE '{vae_type}' may not load correctly."
            )
        
        print(f"[VAE Manager] Loading VAE: {vae_type}")
        print(f"[VAE Manager] Weight path: {weight_path}")
        
        # Load model based on type
        if config["is_tcdecoder"]:
            vae_model = self._load_tcdecoder_vae(vae_type, config, weight_path, mode)
        else:
            vae_model = self._load_wan_vae(vae_type, config, weight_path, mode)
        
        self.current_vae = vae_model
        
        # Wrap decode function for tiling if requested
        if tile_vae and hasattr(vae_model, 'decode'):
            self._wrap_decode_for_tiling(tile_size, overlap)
        
        print(f"[VAE Manager] Successfully loaded: {vae_type}")
        print(f"[VAE Manager] Description: {config['description']}")
        
        return vae_model
    
    def _load_tcdecoder_vae(self, 
                           vae_type: str, 
                           config: Dict, 
                           weight_path: str,
                           mode: str) -> nn.Module:
        """Load TCDecoder type VAE (for tiny modes)."""
        # Build TCDecoder with specified channels
        channels = config.get("channels", [512, 256, 128, 128])
        
        if vae_type == "tcd":
            # Original TCDecoder with 16+768 latent channels
            tcdecoder = build_tcdecoder(
                new_channels=channels,
                new_latent_channels=16 + 768
            ).to(self.device, dtype=self.dtype)
        else:
            # Other TAE variants with 16 latent channels
            tcdecoder = build_tcdecoder(
                new_channels=channels,
                new_latent_channels=16
            ).to(self.device, dtype=self.dtype)
        
        # Load weights if available
        if weight_path and os.path.exists(weight_path):
            try:
                # Handle different file formats
                if weight_path.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    state_dict = load_file(weight_path, device='cpu')
                else:
                    state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
                
                # Load state dict
                missing_keys, unexpected_keys = tcdecoder.load_state_dict(
                    state_dict, strict=False
                )
                if missing_keys:
                    print(f"[VAE Manager] Missing keys when loading {vae_type}: {missing_keys}")
                if unexpected_keys:
                    print(f"[VAE Manager] Unexpected keys when loading {vae_type}: {unexpected_keys}")
                    
            except Exception as e:
                warnings.warn(f"Failed to load weights for {vae_type}: {e}")
        
        # Wrap in DiffusersWrapper for consistent interface
        class TCDecoderWrapper(nn.Module):
            """Wrapper to make TCDecoder compatible with pipeline interface."""
            def __init__(self, tcd_model, device, dtype):
                super().__init__()
                self.tcd_model = tcd_model
                self.device = device
                self.dtype = dtype
                self.config = type('Config', (), {
                    'scaling_factor': 1.0,
                    'latents_mean': torch.zeros(16),
                    'latents_std': torch.ones(16)
                })()
            
            def decode(self, latents):
                """Decode latents to video frames."""
                # latents: (B, C, T, H, W)
                n, c, t, h, w = latents.shape
                
                # TCDecoder expects (N, T, C, H, W)
                latents_ntc = latents.transpose(1, 2)  # (B, T, C, H, W)
                
                # Decode
                decoded = self.tcd_model.decode_video(
                    latents_ntc,
                    parallel=False,
                    show_progress_bar=False
                )
                
                # Convert back to (B, C, T, H, W) and scale
                decoded = decoded.transpose(1, 2)  # (B, C, T, H, W)
                decoded = decoded * 2.0 - 1.0  # Scale to [-1, 1]
                
                return decoded
            
            def to(self, device=None, dtype=None):
                """Move model to device/dtype."""
                if device:
                    self.device = device
                    self.tcd_model = self.tcd_model.to(device)
                if dtype:
                    self.dtype = dtype
                    self.tcd_model = self.tcd_model.to(dtype)
                return self
        
        return TCDecoderWrapper(tcdecoder, self.device, self.dtype)
    
    def _load_wan_vae(self,
                     vae_type: str,
                     config: Dict,
                     weight_path: str,
                     mode: str) -> nn.Module:
        """Load Wan VAE type (for full mode)."""
        # Import here to avoid circular imports
        from diffsynth import ModelManager
        
        # Initialize ModelManager
        mm = ModelManager(torch_dtype=self.dtype, device='cpu')
        
        # Load the VAE model
        if not weight_path or not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"VAE weight file not found: {weight_path}. "
                f"Please download the {vae_type} weights."
            )
        
        try:
            mm.load_models([weight_path])
        except Exception as e:
            raise RuntimeError(f"Failed to load VAE model from {weight_path}: {e}")
        
        # Get the VAE model from ModelManager
        # Note: This depends on diffsynth's internal structure
        vae_model = mm.vae  # Adjust based on actual diffsynth API
        
        # Move to target device and dtype
        vae_model = vae_model.to(self.device).to(self.dtype)
        
        # Remove encoder to save memory (as in original code)
        if hasattr(vae_model, 'encoder'):
            vae_model.encoder = None
        if hasattr(vae_model, 'conv1'):
            vae_model.conv1 = None
        
        return vae_model
    
    def _wrap_decode_for_tiling(self, tile_size: int, overlap: int):
        """Wrap VAE decode function to support tiled processing."""
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
        
        # Replace decode method
        self.current_vae.decode = tiled_decode
        print(f"[VAE Manager] Enabled tiled decoding (tile_size={tile_size}, overlap={overlap})")
    
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
            # Call clean_mem if available (for TCDecoder)
            if hasattr(self.current_vae, 'clean_mem'):
                self.current_vae.clean_mem()
            elif hasattr(self.current_vae, 'tcd_model') and hasattr(self.current_vae.tcd_model, 'clean_mem'):
                self.current_vae.tcd_model.clean_mem()
        
        self.current_vae = None
        self.current_type = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None