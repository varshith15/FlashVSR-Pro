
"""FlashVSR plugin for Daydream Scope."""

import scope.core

from .pipeline import FlashVSRPipeline


@scope.core.hookimpl
def register_pipelines(register):
    register(FlashVSRPipeline)


__all__ = ["FlashVSRPipeline"]