"""Transition wrapper for canonical STRIDE observation discrepancy helpers."""
from __future__ import annotations

from stride.observation.discrepancy import (
    build_observation_kernels,
    calibrate_match_penalty,
    compute_active_state_support,
    compute_observation_discrepancy,
    match_observation_clouds,
)

__all__ = [
    "build_observation_kernels",
    "calibrate_match_penalty",
    "compute_active_state_support",
    "compute_observation_discrepancy",
    "match_observation_clouds",
]
