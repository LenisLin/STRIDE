"""Transition wrapper for canonical STRIDE observation-layer matching."""
from __future__ import annotations

from stride.observation import (
    ObservationDiscrepancyConfig,
    ObservationDiscrepancyResult,
    build_observation_kernels,
    calibrate_match_penalty,
    compute_active_state_support,
    compute_observation_discrepancy,
    match_observation_clouds,
)

ObservationMatchConfig = ObservationDiscrepancyConfig
ObservationMatchResult = ObservationDiscrepancyResult

__all__ = [
    "ObservationMatchConfig",
    "ObservationMatchResult",
    "build_observation_kernels",
    "calibrate_match_penalty",
    "compute_active_state_support",
    "compute_observation_discrepancy",
    "match_observation_clouds",
]
