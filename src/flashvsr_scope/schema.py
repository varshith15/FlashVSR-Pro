"""Configuration schema for FlashVSR pipeline."""

from typing import ClassVar

from pydantic import Field

from scope.core.pipelines.artifacts import Artifact
from scope.core.pipelines.base_schema import BasePipelineConfig, CtrlInput, ModeDefaults


class FlashVSRConfig(BasePipelineConfig):
    """Configuration for FlashVSR pipeline."""

    pipeline_id = "flashvsr"
    pipeline_name = "FlashVSR Pro"
    pipeline_description = "Real-time Video Super Resolution using FlashVSR."
    docs_url = "https://github.com/FlashVSR/FlashVSR-Pro"

    supports_prompts = False
    default_temporal_interpolation_method = None
    default_spatial_interpolation_method = None

    supports_cache_management = True

    modes = {"video": ModeDefaults(default=True)}

    # We assume models are locally available for now, or managed via env vars
    artifacts: ClassVar[list[Artifact]] = []

    # Controller input support - presence of this field enables controller input capture
    ctrl_input: CtrlInput | None = None

    # Reference images for conditioning (presence enables ImageManager UI)
    # For VSR, this is the input frame(s)
    images: list[str] | None = Field(
        default=None,
        description="Input frames for super resolution",
    )
    
    scale: float = Field(
        default=2.0,
        description="Upscale factor",
        ge=1.0,
        le=4.0,
    )
