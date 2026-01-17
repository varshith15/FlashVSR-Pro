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
            "class": "WanVAE",
            "default_path": "models/VAEs/Wan2.1_VAE.pth",
            "is_tcdecoder": False,
            "description": "Original Wan VAE 2.1, best quality but highest VRAM usage."
        },
        "wan2.2": {
            "class": "WanVAE",
            "default_path": "models/VAEs/Wan2.2_VAE.pth",
            "is_tcdecoder": False,
            "description": "Wan VAE 2.2, improved version with better quality/speed balance."
        },
        "light": {
            "class": "WanVAE",
            "default_path": "models/VAEs/lightvaew2_1.pth",
            "is_tcdecoder": False,
            "description": "Lightweight Wan VAE, reduced VRAM usage for high-resolution processing."
        },
        # TCDecoder variants (tiny mode)
        "tcd": {
            "class": "TAEW2_1DiffusersWrapper",
            "default_path": None,
            "channels": [512, 256, 128, 128],
            "is_tcdecoder": True,
            "description": "Tiny Conditional Decoder for 'tiny' mode, fastest with moderate quality."
        },
        "tae-hv": {
            "class": "TAEW2_1DiffusersWrapper",
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
            
            def clear_cache(self):
                """Clear cache - required by FlashVSRFullPipeline."""
                if hasattr(self.tcd_model, 'clean_mem'):
                    self.tcd_model.clean_mem()
            
            def clean_mem(self):
                """Clean memory state - required by FlashVSR pipeline."""
                if hasattr(self.tcd_model, 'clean_mem'):
                    self.tcd_model.clean_mem()
            
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
        if not weight_path or not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"VAE weight file not found: {weight_path}. "
                f"Please download the {vae_type} weights."
            )
        
        print(f"   [VAE Loader] Loading VAE from: {weight_path}")
        
        # 首先尝试通过 ModelManager 加载
        from diffsynth import ModelManager
        mm = ModelManager(torch_dtype=self.dtype, device='cpu')
        
        try:
            mm.load_models([weight_path])
            
            # 检查是否成功加载
            if hasattr(mm, 'model') and isinstance(mm.model, list) and len(mm.model) > 0:
                # 成功通过 ModelManager 加载
                vae_model = self._extract_vae_from_model_manager(mm)
                print(f"   [VAE Loader] Successfully loaded via ModelManager")
            else:
                # ModelManager 无法识别，尝试直接加载
                print(f"   [VAE Loader] ModelManager cannot detect model type, trying direct load...")
                vae_model = self._load_vae_directly(weight_path, vae_type)
            
            # Move to target device and dtype
            vae_model = vae_model.to(self.device).to(self.dtype)
            
            # Remove encoder to save memory
            if hasattr(vae_model, 'encoder'):
                vae_model.encoder = None
                print(f"   [VAE Loader] Removed encoder to save memory")
            if hasattr(vae_model, 'conv1'):
                vae_model.conv1 = None
            
            print(f"   [VAE Loader] Successfully loaded VAE model")
            return vae_model
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load VAE model from {weight_path}: {e}"
            )
    
    def _load_vae_directly(self, weight_path: str, vae_type: str) -> nn.Module:
        """直接加载VAE权重（当ModelManager无法识别时）"""
        print(f"   [VAE Loader] Loading state_dict from: {weight_path}")
        
        # 加载权重文件
        if weight_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            checkpoint = load_file(weight_path, device='cpu')
        else:
            checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
        
        # 处理不同的权重文件格式
        if isinstance(checkpoint, nn.Module):
            # 直接是模型对象
            print(f"   [VAE Loader] Loaded complete model object")
            return checkpoint
        
        # 提取 state_dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"   [VAE Loader] Found state_dict in checkpoint")
        elif isinstance(checkpoint, dict):
            # 可能整个文件就是 state_dict
            state_dict = checkpoint
            print(f"   [VAE Loader] Using checkpoint as state_dict")
        else:
            raise ValueError(f"Cannot extract state_dict from checkpoint")
        
        # 创建 VAE 架构
        print(f"   [VAE Loader] Creating WanVideoVAE architecture using ModelManager...")
        
        # 使用 ModelManager 加载一个参考模型（如 wan2.1）来获取架构
        # 然后用当前的 state_dict 覆盖权重
        from diffsynth import ModelManager
        
        # 尝试从 wan2.1 获取架构作为模板
        reference_vae_path = "models/VAEs/Wan2.1_VAE.pth"
        
        if os.path.exists(reference_vae_path):
            print(f"   [VAE Loader] Using {reference_vae_path} as architecture template")
            mm_temp = ModelManager(torch_dtype=self.dtype, device='cpu')
            mm_temp.load_models([reference_vae_path])
            
            # 提取 VAE 架构
            template_vae = self._extract_vae_from_model_manager(mm_temp)
            print(f"   [VAE Loader] Got template VAE: {type(template_vae)}")
            
            # 使用当前的 state_dict 替换权重
            print(f"   [VAE Loader] Loading weights into template architecture...")
            missing_keys, unexpected_keys = template_vae.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"   [VAE Loader] Warning: Missing keys: {len(missing_keys)} keys")
                if len(missing_keys) < 10:
                    print(f"   [VAE Loader] Missing: {missing_keys}")
            if unexpected_keys:
                print(f"   [VAE Loader] Warning: Unexpected keys: {len(unexpected_keys)} keys")
                if len(unexpected_keys) < 10:
                    print(f"   [VAE Loader] Unexpected: {unexpected_keys}")
            
            print(f"   [VAE Loader] Successfully loaded weights into WanVideoVAE")
            return template_vae
        else:
            # 如果没有参考模型，尝试直接创建
            print(f"   [VAE Loader] No reference VAE found, trying direct initialization...")
            from diffsynth.models.wan_video_vae import WanVideoVAE
            
            # 尝试创建空的 VAE 实例
            try:
                vae_model = WanVideoVAE()
                print(f"   [VAE Loader] Created WanVideoVAE with default parameters")
            except Exception as e:
                print(f"   [VAE Loader] Failed to create VAE: {e}")
                
                # 显示构造函数签名以便调试
                import inspect
                try:
                    sig = inspect.signature(WanVideoVAE.__init__)
                    print(f"   [VAE Loader] WanVideoVAE.__init__ signature: {sig}")
                except:
                    pass
                
                raise RuntimeError(
                    f"Cannot initialize WanVideoVAE without reference model. "
                    f"Please ensure Wan2.1_VAE.pth exists as a template, or check WanVideoVAE class definition."
                )
            
            # 加载权重
            missing_keys, unexpected_keys = vae_model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"   [VAE Loader] Warning: Missing keys: {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"   [VAE Loader] Warning: Unexpected keys: {len(unexpected_keys)} keys")
            
            print(f"   [VAE Loader] Successfully created and loaded WanVideoVAE")
            return vae_model
    

    
    def _extract_vae_from_model_manager(self, mm):
        """从ModelManager中提取VAE模型"""
        # 尝试多种方式获取VAE
        vae_model = None
        
        # 方法1: 尝试常见的属性名
        possible_attrs = ['model', 'vae', 'video_vae', 'vae_model']
        for attr in possible_attrs:
            if hasattr(mm, attr):
                candidate = getattr(mm, attr)
                print(f"   [VAE Loader] Checking attribute '{attr}': type={type(candidate)}")
                
                # 如果是列表，尝试提取VAE
                if isinstance(candidate, list):
                    print(f"   [VAE Loader] List length: {len(candidate)}")
                    if len(candidate) == 0:
                        print(f"   [VAE Loader] Empty list, skipping...")
                        continue
                    
                    # 遍历列表查找VAE
                    for idx, item in enumerate(candidate):
                        item_type = str(type(item)).lower()
                        print(f"   [VAE Loader] List item {idx}: {type(item)}")
                        if 'vae' in item_type or 'autoencoder' in item_type:
                            vae_model = item
                            print(f"   [VAE Loader] ✓ Selected VAE from list: {type(item)}")
                            break
                    
                    # 如果没找到VAE但列表不为空，使用第一个
                    if vae_model is None and len(candidate) > 0:
                        vae_model = candidate[0]
                        print(f"   [VAE Loader] Using first item in list: {type(vae_model)}")
                else:
                    # 不是列表，检查是否是VAE类型
                    candidate_type = str(type(candidate)).lower()
                    if 'vae' in candidate_type or 'autoencoder' in candidate_type:
                        vae_model = candidate
                        print(f"   [VAE Loader] ✓ Using attribute directly: {type(vae_model)}")
                
                if vae_model is not None:
                    break
        
        # 方法2: 尝试从 models 字典中查找
        if vae_model is None and hasattr(mm, 'models'):
            print(f"   [VAE Loader] Searching in models dictionary...")
            if isinstance(mm.models, dict):
                print(f"   [VAE Loader] Models dict keys: {list(mm.models.keys())}")
                for model_name, model in mm.models.items():
                    model_type = str(type(model)).lower()
                    print(f"   [VAE Loader] Model '{model_name}': {type(model)}")
                    if 'vae' in model_type or 'autoencoder' in model_type:
                        vae_model = model
                        print(f"   [VAE Loader] ✓ Found VAE in models dict: {model_name}")
                        break
        
        if vae_model is None:
            # 提供更详细的调试信息
            print(f"   [VAE Loader] Failed to find VAE. Debugging info:")
            print(f"   [VAE Loader] - MM attributes: {[a for a in dir(mm) if not a.startswith('_')]}")
            if hasattr(mm, 'model'):
                print(f"   [VAE Loader] - mm.model type: {type(mm.model)}")
                if isinstance(mm.model, list):
                    print(f"   [VAE Loader] - mm.model length: {len(mm.model)}")
            if hasattr(mm, 'models'):
                print(f"   [VAE Loader] - mm.models type: {type(mm.models)}")
                if isinstance(mm.models, dict):
                    print(f"   [VAE Loader] - mm.models keys: {list(mm.models.keys())}")
            
            raise AttributeError(
                f"Cannot find VAE model in ModelManager after loading."
            )
        
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