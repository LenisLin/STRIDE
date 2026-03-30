"""Backend-specific adapter surfaces used by higher-level STRIDE layers."""
from __future__ import annotations

from .ot_sinkhorn import (
    ERR_UOT_EMPTY_MASS_SOURCE,
    ERR_UOT_EMPTY_MASS_TARGET,
    ERR_UOT_EMPTY_SUPPORT,
    ERR_UOT_NUMERICAL,
    ObservationMatchConfig,
    STATUS_OK,
    batched_uot_solve,
    build_observation_kernels,
    calibrate_match_penalty,
    compute_active_state_support,
    weighted_quantile,
)

__all__ = [
    "ERR_UOT_EMPTY_MASS_SOURCE",
    "ERR_UOT_EMPTY_MASS_TARGET",
    "ERR_UOT_EMPTY_SUPPORT",
    "ERR_UOT_NUMERICAL",
    "ObservationMatchConfig",
    "STATUS_OK",
    "batched_uot_solve",
    "build_observation_kernels",
    "calibrate_match_penalty",
    "compute_active_state_support",
    "weighted_quantile",
]
