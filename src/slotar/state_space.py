"""Transition wrapper for canonical STRIDE shared-state-axis facades."""
from __future__ import annotations

from stride.basis import (
    CommunityAggregationConfig,
    StateBasis,
    aggregate_local_features,
    aggregate_to_state_basis,
    build_local_state_features,
    learn_shared_state_axis,
    load_state_basis,
    validate_state_basis,
)
from stride.geometry import StateGeometry, build_similarity_graph, build_state_geometry

StateAxis = StateBasis

__all__ = [
    "CommunityAggregationConfig",
    "StateAxis",
    "StateBasis",
    "StateGeometry",
    "aggregate_local_features",
    "aggregate_to_state_basis",
    "build_local_state_features",
    "build_similarity_graph",
    "build_state_geometry",
    "learn_shared_state_axis",
    "load_state_basis",
    "validate_state_basis",
]
