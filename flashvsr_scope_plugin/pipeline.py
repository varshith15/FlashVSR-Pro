import os
import torch
import logging

from scope.core.config import get_model_file_path
from scope.core.pipelines.process import preprocess_chunk, postprocess_chunk
from scope.core.pipelines.interface import Pipeline, Requirements
from diffsynth import ModelManager, FlashVSRTinyPipeline
from flashvsr_utils.core.utils import Causal_LQ4x_Proj
from flashvsr_utils.vae import vae_system

from .schema import FlashVSRConfig

logger = logging.getLogger(__name__)

class FlashVSRPipeline(Pipeline):
    @classmethod
    def get_config_class(cls) -> type["FlashVSRConfig"]:
        return FlashVSRConfig

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        height: int = 512,
        width: int = 512,
        **kwargs,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype
        self.height = height
        self.width = width
        self.multiple = 128
        self.warmed_up_resolution = None

        model_path = str(get_model_file_path("FlashVSR-v1.1"))
        posi_prompt_path = os.path.join(str(get_model_file_path("posi_prompt")), "posi_prompt.pth")

        self.vae_manager = vae_system.VAESystem(device=str(self.device), dtype=self.dtype)
        self.vae_model = self.vae_manager.load_vae(vae_type="tcd", model_dir=model_path)

        self.mm = ModelManager(torch_dtype=self.dtype, device="cpu")
        dit_path = os.path.join(model_path, "diffusion_pytorch_model_streaming_dmd.safetensors")

        self.mm.load_models([dit_path])
        self.pipe = FlashVSRTinyPipeline.from_model_manager(self.mm, device=self.device)
        self.pipe.TCDecoder = self.vae_model

        lq_proj = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1)
        lq_path = os.path.join(model_path, "LQ_proj_in.ckpt")
        lq_proj.load_state_dict(torch.load(lq_path, map_location="cpu"), strict=True)

        self.pipe.denoising_model().LQ_proj_in = lq_proj.to(self.device, dtype=self.dtype)

        self.pipe.to(self.device)
        self.pipe.init_cross_kv(prompt_path=posi_prompt_path)
        self._warmup(self.height, self.width)

    def _warmup(self, height: int, width: int) -> None:
        dummy_frames = torch.randn(1, 3, 25, height, width, device=self.device, dtype=self.dtype).clamp(-1, 1)
        warmup_output = self.pipe.stream(dummy_frames, height=height, width=width, seed=0)
        self.warmed_up_resolution = (height, width)
        logger.info(f"FlashVSR warmup completed successfully at {height}x{width}, output shape: {warmup_output.shape}")


    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=8)

    def __call__(self, **kwargs) -> torch.Tensor:
        video = kwargs.get("video")

        if video is None:
            raise ValueError("Input video cannot be None for FlashVSR pipeline")

        _, h, w, _ = video[0].shape
        target_h = (h // self.multiple) * self.multiple
        target_w = (w // self.multiple) * self.multiple

        input_tensor = preprocess_chunk(video, self.device, self.dtype, height=target_h, width=target_w)
        _, _, T, H, W = input_tensor.shape
        
        if self.warmed_up_resolution != (H, W):
            self._warmup(H, W)
        
        out_chunk = self.pipe.stream(input_tensor, height=H, width=W, seed=0)
        out_chunk = out_chunk.permute(0, 2, 1, 3, 4)
        result = postprocess_chunk(out_chunk)
        return result
