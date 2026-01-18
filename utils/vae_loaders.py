# utils/vae_loaders.py
"""
Specialized VAE Loaders for different VAE architectures
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
import warnings
import os

from .TCDecoder import TAEW2_1DiffusersWrapper, build_tcdecoder


class WanVAELoader:
    """Specialized loader for Wan VAE models (wan2.1, wan2.2, lightvae)"""

    @staticmethod
    def load_vae(weight_path: str, vae_type: str) -> nn.Module:
        """Load Wan VAE with proper architecture detection"""
        if not weight_path or not os.path.exists(weight_path):
            raise FileNotFoundError(f"VAE weights not found: {weight_path}")

        # Load checkpoint
        if weight_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(weight_path, device='cpu')
        else:
            checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                raise ValueError("Cannot extract state_dict from checkpoint")

        # Detect architecture from state_dict keys
        keys = list(state_dict.keys())

        # Wan VAEs have specific key patterns
        has_encoder = any(k.startswith('encoder') for k in keys)
        has_decoder = any(k.startswith('decoder') for k in keys)

        if not has_decoder:
            raise ValueError(f"No decoder found in {vae_type} weights")

        # Create WanVideoVAE instance
        from diffsynth.models.wan_video_vae import WanVideoVAE

        # Different VAE types may have different dimensions
        if vae_type == 'light':
            # LightVAE has smaller dimensions
            vae_model = WanVideoVAE(z_dim=16, dim=24)
        elif vae_type == 'wan2.2':
            # Wan2.2 has larger dimensions for better quality
            vae_model = WanVideoVAE(z_dim=96, dim=640)
        else:
            # Standard Wan VAE
            vae_model = WanVideoVAE(z_dim=16, dim=96)

        # Add 'model.' prefix to match WanVideoVAE state_dict keys
        state_dict = {f"model.{k}": v for k, v in state_dict.items()}

        # Load state dict
        missing, unexpected = vae_model.load_state_dict(state_dict, strict=False)

        if missing:
            warnings.warn(f"Missing keys in {vae_type}: {missing[:3]}...")
        if unexpected:
            warnings.warn(f"Unexpected keys in {vae_type}: {unexpected[:3]}...")

        return vae_model


class TAEVAELoader:
    """Specialized loader for TAE VAE models (tae-hv, tae-w2.2)"""

    @staticmethod
    def load_vae(weight_path: str, vae_type: str, channels: list = None) -> nn.Module:
        """Load TAE VAE with proper decoder structure"""
        if not weight_path or not os.path.exists(weight_path):
            raise FileNotFoundError(f"VAE weights not found: {weight_path}")

        if channels is None:
            channels = [256, 128, 64, 64]  # Default for tae-hv/tae-w2.2

        # Load checkpoint
        if weight_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            full_state_dict = load_file(weight_path, device='cpu')
        else:
            checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
            if 'state_dict' in checkpoint:
                full_state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict):
                full_state_dict = checkpoint
            else:
                raise ValueError("Cannot extract state_dict from checkpoint")

        # Extract decoder weights only (TAE models contain encoder + decoder)
        decoder_state_dict = {}
        for key, value in full_state_dict.items():
            if key.startswith('decoder.'):
                decoder_state_dict[key] = value

        if not decoder_state_dict:
            raise ValueError(f"No decoder weights found in {vae_type}")

        # Determine output channels and latent channels based on vae_type
        if vae_type == "tae-hv":
            output_channels = 12
            latent_channels = 32
        elif vae_type == "tae-w2.2":
            output_channels = 12
            latent_channels = 48
        else:
            output_channels = 3
            latent_channels = 32

        # Create TAEW2_1DiffusersWrapper with correct structure
        vae_model = TAEW2_1DiffusersWrapper(pretrained_path=None, channels=channels, output_channels=output_channels)

        # Set correct latent channels for TAE models
        vae_model.taehv.latent_channels = latent_channels

        # Rebuild decoder to match checkpoint structure (no additional deepening)
        vae_model.taehv._rebuild_decoder(channels, latent_channels, vae_type)

        # Load decoder weights
        missing, unexpected = vae_model.taehv.load_state_dict(decoder_state_dict, strict=True)

        if missing or unexpected:
            raise RuntimeError(f"TAE weight loading failed - missing: {len(missing)}, unexpected: {len(unexpected)}")

        return vae_model


class TCDVAELoader:
    """Specialized loader for TCD VAE models"""

    @staticmethod
    def load_vae(weight_path: str, vae_type: str, channels: list = None) -> nn.Module:
        """Load TCD VAE with proper TCDecoder structure"""
        if channels is None:
            channels = [512, 256, 128, 128]  # Default for tcd

        # Build TCDecoder with appropriate channels
        latent_channels = 16 + 768  # TCD specific
        tcdecoder = build_tcdecoder(
            new_channels=channels,
            new_latent_channels=latent_channels
        )

        # Load weights if available
        if weight_path and os.path.exists(weight_path):
            try:
                if weight_path.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    state_dict = load_file(weight_path, device='cpu')
                else:
                    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif isinstance(checkpoint, dict):
                        state_dict = checkpoint
                    else:
                        raise ValueError("Cannot extract state_dict from checkpoint")

                missing, unexpected = tcdecoder.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    warnings.warn(f"TCD weight loading: {len(missing)} missing, {len(unexpected)} unexpected keys")

            except Exception as e:
                warnings.warn(f"Failed to load weights for {vae_type}: {e}")

        # Wrap for pipeline compatibility
        class TCDecoderWrapper(nn.Module):
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
                n, c, t, h, w = latents.shape
                latents_ntc = latents.transpose(1, 2)

                decoded = self.tcd_model.decode_video(
                    latents_ntc, parallel=False, show_progress_bar=False
                )

                decoded = decoded.transpose(1, 2)
                decoded = decoded * 2.0 - 1.0

                return decoded

            def clear_cache(self):
                if hasattr(self.tcd_model, 'clean_mem'):
                    self.tcd_model.clean_mem()

            def clean_mem(self):
                if hasattr(self.tcd_model, 'clean_mem'):
                    self.tcd_model.clean_mem()

            def to(self, device=None, dtype=None):
                if device:
                    self.device = device
                    self.tcd_model = self.tcd_model.to(device)
                if dtype:
                    self.dtype = dtype
                    self.tcd_model = self.tcd_model.to(dtype)
                return self

        return TCDecoderWrapper(tcdecoder, "cuda", torch.bfloat16)


class VAELoaderFactory:
    """Factory for creating appropriate VAE loaders"""

    LOADERS = {
        # Wan series
        "wan2.1": WanVAELoader,
        "wan2.2": WanVAELoader,
        "light": WanVAELoader,

        # TAE series
        "tae-hv": TAEVAELoader,
        "tae-w2.2": TAEVAELoader,

        # TCD series
        "tcd": TCDVAELoader,
    }

    @staticmethod
    def get_loader(vae_type: str):
        """Get the appropriate loader for a VAE type"""
        if vae_type not in VAELoaderFactory.LOADERS:
            raise ValueError(f"No loader available for VAE type: {vae_type}")

        return VAELoaderFactory.LOADERS[vae_type]

    @staticmethod
    def load_vae(vae_type: str, weight_path: str, **kwargs) -> nn.Module:
        """Load VAE using the appropriate specialized loader"""
        loader_class = VAELoaderFactory.get_loader(vae_type)
        return loader_class.load_vae(weight_path, vae_type, **kwargs)