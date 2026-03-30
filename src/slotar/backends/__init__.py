"""Transition wrapper for backend implementations used by SLOTAR compatibility surfaces."""
from __future__ import annotations

from .ot_sinkhorn import (
    ObservationMatchConfig,
    batched_uot_solve,
    build_observation_kernels,
    calibrate_match_penalty,
    compute_active_state_support,
)

__all__ = [
    "ObservationMatchConfig",
    "batched_uot_solve",
    "build_observation_kernels",
    "calibrate_match_penalty",
    "compute_active_state_support",
]
