import os
import torch
import numpy as np
from PIL import Image
from torchvision.io import read_image

from scope.core.config import get_model_file_path
from scope.core.pipelines.interface import Pipeline
from diffsynth import ModelManager, FlashVSRTinyPipeline
from utils.core.utils import Causal_LQ4x_Proj
from utils.vae import vae_system

from .schema import FlashVSRConfig

class FlashVSRPipeline(Pipeline):
    @classmethod
    def get_config_class(cls) -> type["FlashVSRConfig"]:
        return FlashVSRConfig

    def __init__(
        self,
        scale: float = 2.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype
        self.scale = scale

        # Load models
        # We use environment variable or default local path
        self.model_dir = os.getenv("FLASHVSR_MODEL_PATH", "./models/FlashVSR-v1.1")
        
        print(f"Loading FlashVSR models from {self.model_dir}...")
        
        # 1. Load VAE
        self.vae_manager = vae_system.VAESystem(device=str(self.device), dtype=self.dtype)
        # Defaulting to 'tcd' as per simple_infer_tiny.py
        self.vae_model = self.vae_manager.load_vae(vae_type="tcd", model_dir=self.model_dir)
        
        # 2. Load DiT
        self.mm = ModelManager(torch_dtype=self.dtype, device="cpu")
        dit_path = os.path.join(self.model_dir, "diffusion_pytorch_model_streaming_dmd.safetensors")
        if not os.path.exists(dit_path):
             # Fallback or error
             print(f"Warning: DiT model not found at {dit_path}")
        
        self.mm.load_models([dit_path])
        self.pipe = FlashVSRTinyPipeline.from_model_manager(self.mm, device=self.device)
        self.pipe.TCDecoder = self.vae_model
        
        # 3. Load LQ Projection
        lq_proj = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1)
        lq_path = os.path.join(self.model_dir, "LQ_proj_in.ckpt")
        if os.path.exists(lq_path):
            lq_proj.load_state_dict(torch.load(lq_path, map_location="cpu"), strict=True)
        else:
            print(f"Warning: LQ projection not found at {lq_path}")
            
        self.pipe.denoising_model().LQ_proj_in = lq_proj.to(self.device, dtype=self.dtype)
        
        self.pipe.to(self.device)
        self.pipe.init_cross_kv()
        
        # State for streaming
        self.is_initialized = False

    def __call__(self, **kwargs) -> torch.Tensor:
        """Generate a frame (upscale).
        
        Args:
            images: List of image paths. Takes the first one as input.
            
        Returns:
            torch.Tensor: Frame in THWC format (1, H, W, 3) in [0, 1] range
        """
        init_cache = kwargs.get("init_cache", False)
        images = kwargs.get("images")
        
        if init_cache:
            self.pipe.init_cross_kv()
            self.is_initialized = False

        if not images or len(images) == 0:
            # Return black frame or noise if no input
            return torch.zeros((1, 512, 512, 3), device=self.device, dtype=torch.float32)

        # Read input image
        # Scope passes image paths
        image_path = images[0]
        try:
            # Read image using PIL to match simple_infer_tiny logic
            img = Image.open(image_path).convert('RGB')
            w0, h0 = img.size
            
            # Calculate dimensions
            # simple_infer_tiny logic:
            sW = int(round(w0 * self.scale))
            sH = int(round(h0 * self.scale))
            multiple = 128
            tW = (sW // multiple) * multiple
            tH = (sH // multiple) * multiple
            
            # Preprocess
            # upscale then center crop
            up = img.resize((sW, sH), Image.BICUBIC)
            l, t = (sW - tW) // 2, (sH - tH) // 2
            img_out = up.crop((l, t, l + tW, t + tH))
            
            # To tensor
            # (H, W, C) -> (C, H, W)
            t_img = torch.from_numpy(np.asarray(img_out, np.uint8).copy()).to(device=self.device, dtype=self.dtype)
            t_img = t_img.permute(2, 0, 1) / 255.0 * 2.0 - 1.0 # [-1, 1]
            
            # Add batch and time dims: (1, C, 1, H, W)
            # FlashVSRTinyPipeline.stream expects (1, C, T, H, W) where T is chunk size.
            # Here we process 1 frame at a time.
            input_tensor = t_img.unsqueeze(0).unsqueeze(2)
            
            # Run inference
            # stream(x, height, width, seed)
            # x is (B, C, T, H, W)
            out_chunk = self.pipe.stream(input_tensor, height=tH, width=tW, seed=0)
            
            # Output is (B, C, T, H, W)
            # We want (1, H, W, 3) for Scope
            # out_chunk is [-1, 1]
            
            frame = out_chunk[0, :, 0, :, :].permute(1, 2, 0) # (H, W, C)
            frame = (frame + 1) * 0.5 # [0, 1]
            
            return frame.unsqueeze(0).float() # (1, H, W, 3)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return torch.zeros((1, 512, 512, 3), device=self.device, dtype=torch.float32)
