"""STRIDE shared-state basis construction and projection helpers."""
from __future__ import annotations

from .aggregation import (
    CommunityAggregationConfig,
    aggregate_local_features,
    build_local_state_features,
    learn_shared_state_axis,
)
from .contracts import StateBasis, load_state_basis, validate_state_basis
from .state_projection import aggregate_to_state_basis, assign_state_ids

__all__ = [
    "CommunityAggregationConfig",
    "StateBasis",
    "aggregate_local_features",
    "aggregate_to_state_basis",
    "assign_state_ids",
    "build_local_state_features",
    "learn_shared_state_axis",
    "load_state_basis",
    "validate_state_basis",
]
