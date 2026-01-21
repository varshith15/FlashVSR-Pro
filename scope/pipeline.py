import os
import torch

from scope.core.config import get_model_file_path
from scope.core.pipelines.process import preprocess_chunk, postprocess_chunk
from scope.core.pipelines.interface import Pipeline, Requirements
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
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype

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
        self._warmup()

    def _warmup(self) -> None:
        dummy_frames = torch.randn(1, 3, 25, 512, 512, device=self.device, dtype=self.dtype).clamp(-1, 1)
        _ = self.pipe.stream(dummy_frames, height=512, width=512, seed=0)

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=8)

    def __call__(self, kwargs) -> torch.Tensor:
        video = kwargs["video"]
        
        input_tensor = preprocess_chunk(video, self.device, self.dtype)
        _, _, _, H, W = input_tensor.shape
        out_chunk = self.pipe.stream(input_tensor, height=H, width=W, seed=0)
        return postprocess_chunk(out_chunk)
