"""Configuration schema for FlashVSR pipeline."""

from scope.core.pipelines.artifacts import HuggingfaceRepoArtifact
from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults, UsageType


class FlashVSRConfig(BasePipelineConfig):
    """Configuration for FlashVSR pipeline."""

    pipeline_id = "flashvsr"
    pipeline_name = "FlashVSR"
    pipeline_description = "Real-time Video Super Resolution using FlashVSR."
    docs_url = "https://github.com/varshith15/FlashVSR-Pro.git"

    usage = [UsageType.POSTPROCESSOR]

    supports_prompts = False

    modes = {
        "video": ModeDefaults(default=True)
    }

    artifacts = [
        HuggingfaceRepoArtifact(
            repo_id="JunhaoZhuang/FlashVSR-v1.1",
            files=["LQ_proj_in.ckpt", "diffusion_pytorch_model_streaming_dmd.safetensors", "TCDecoder.ckpt", "Wan2.1_VAE.pth"],
        ),
        HuggingfaceRepoArtifact(
            repo_id="varb15/posi_prompt",
            files=["posi_prompt.pth"],
        ),
    ]
