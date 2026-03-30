"""Transition wrapper for canonical STRIDE basis aggregation helpers."""
from __future__ import annotations

from stride.basis.aggregation import (
    CommunityAggregationConfig,
    aggregate_local_features,
    build_local_state_features,
    learn_shared_state_axis,
)
from stride.basis.state_projection import aggregate_to_state_basis

__all__ = [
    "CommunityAggregationConfig",
    "aggregate_local_features",
    "aggregate_to_state_basis",
    "build_local_state_features",
    "learn_shared_state_axis",
]


__all__ = [
    "CommunityAggregationConfig",
    "aggregate_local_features",
    "aggregate_to_state_basis",
    "build_local_state_features",
    "learn_shared_state_axis",
]
