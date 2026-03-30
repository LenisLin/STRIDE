"""Observation-layer contracts and discrepancy helpers for STRIDE.

OT-backed numerical routines are exposed here only as supporting observation
surfaces; they are not intended to define the center of the public package.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

from .contracts import (
    DomainStratifiedMeasure,
    FovObservation,
    ObservationDiscrepancy,
    ObservationDiscrepancyConfig,
    ObservationDiscrepancyResult,
    validate_fov_observation,
)
from .measures import (
    build_domain_stratified_measure,
    build_fov_observations,
    compute_fov_burden,
    compute_fov_composition,
    stack_observation_measures,
)
from .validation import validate_observation_match_inputs

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "build_observation_kernels": ("stride.adapters.ot_sinkhorn", "build_observation_kernels"),
    "calibrate_match_penalty": ("stride.adapters.ot_sinkhorn", "calibrate_match_penalty"),
    "compute_active_state_support": ("stride.adapters.ot_sinkhorn", "compute_active_state_support"),
    "compute_observation_discrepancy": ("stride.observation.discrepancy", "compute_observation_discrepancy"),
    "match_observation_clouds": ("stride.observation.discrepancy", "match_observation_clouds"),
}


def __getattr__(name: str) -> Any:
    """Lazily resolve adapter-backed helpers to avoid import cycles."""
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "DomainStratifiedMeasure",
    "FovObservation",
    "ObservationDiscrepancy",
    "ObservationDiscrepancyConfig",
    "ObservationDiscrepancyResult",
    "build_domain_stratified_measure",
    "build_fov_observations",
    "build_observation_kernels",
    "calibrate_match_penalty",
    "compute_active_state_support",
    "compute_fov_burden",
    "compute_fov_composition",
    "compute_observation_discrepancy",
    "match_observation_clouds",
    "stack_observation_measures",
    "validate_fov_observation",
    "validate_observation_match_inputs",
]
