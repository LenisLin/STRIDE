"""Transitional SLOTAR compatibility exports backed by STRIDE.

The `slotar` package remains a transition and compatibility namespace while
the future task-insensitive core is assembled under `stride`.
"""
from __future__ import annotations

from .bridge import BridgeConfig, build_patient_relation
from .errors import ContractError
from .io import validate_longitudinal_adata, write_r_handover
from .observation import (
    ObservationMatchConfig,
    ObservationMatchResult,
    build_observation_kernels,
    calibrate_match_penalty,
    compute_active_state_support,
    match_observation_clouds,
)
from .patient import PatientRelation, PatientRelationAudit, PatientRelationSummary
from .recurrence import (
    RecurrenceConfig,
    RecurrenceFamily,
    RecurrenceParameters,
    RecurrenceResult,
    estimate_recurrence,
)
from .state_space import StateAxis, StateBasis, StateGeometry, build_local_state_features, learn_shared_state_axis

__all__ = [
    "__version__",
    "BridgeConfig",
    "ContractError",
    "ObservationMatchConfig",
    "ObservationMatchResult",
    "PatientRelation",
    "PatientRelationAudit",
    "PatientRelationSummary",
    "RecurrenceConfig",
    "RecurrenceFamily",
    "RecurrenceParameters",
    "RecurrenceResult",
    "StateAxis",
    "StateBasis",
    "StateGeometry",
    "build_local_state_features",
    "build_observation_kernels",
    "build_patient_relation",
    "calibrate_match_penalty",
    "compute_active_state_support",
    "estimate_recurrence",
    "learn_shared_state_axis",
    "match_observation_clouds",
    "validate_longitudinal_adata",
    "write_r_handover",
]

__version__ = "0.1.0"
