"""Standalone CRIB sidecar pipeline (identity + hard-negative mining).

This package is intentionally isolated from the training trainer / dataset code.
It only reads manifests + images and writes sidecar artifacts.
"""

from .config import CribBuildConfig
from .pipeline import build_crib_sidecar, run_pipeline_from_arrays

__all__ = [
    "CribBuildConfig",
    "build_crib_sidecar",
    "run_pipeline_from_arrays",
]
