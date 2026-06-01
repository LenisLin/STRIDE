"""Contracts and constructors for the shared STRIDE state basis."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy.spatial.distance import cdist

from ..errors import ContractError


@dataclass(frozen=True)
class StateBasis:
    """Canonical shared-state basis used across STRIDE layers.

    The basis defines the shared ``K``-state axis through centroid coordinates,
    a state-to-state cost matrix, and the cost scale used by geometry and
    observation-layer discrepancy routines.
    """

    centroids: np.ndarray
    cost_matrix: np.ndarray
    cost_scale: float
    feature_key: str = "local_state_features"
    state_key: str = "state_id"
    state_ids: tuple[int, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        """Return the shared state-axis size."""
        return int(np.asarray(self.centroids).shape[0])

    @property
    def resolved_state_ids(self) -> tuple[int, ...]:
        """Return explicit state IDs, defaulting to `range(K)`."""
        return self.state_ids or tuple(range(self.n_states))


def validate_state_basis(state_basis: StateBasis) -> None:
    """Validate the core invariants of a shared-state basis payload."""
    centroids = np.asarray(state_basis.centroids, dtype=float)
    cost_matrix = np.asarray(state_basis.cost_matrix, dtype=float)

    if centroids.ndim != 2:
        raise ContractError("StateBasis.centroids must be 2D")
    if cost_matrix.ndim != 2 or cost_matrix.shape[0] != cost_matrix.shape[1]:
        raise ContractError("StateBasis.cost_matrix must be square")
    if cost_matrix.shape[0] != centroids.shape[0]:
        raise ContractError(
            "StateBasis.centroids and StateBasis.cost_matrix disagree on the shared K-state axis"
        )
    if not np.isfinite(centroids).all():
        raise ContractError("StateBasis.centroids contains NaN/Inf")
    if not np.isfinite(cost_matrix).all():
        raise ContractError("StateBasis.cost_matrix contains NaN/Inf")
    if float(state_basis.cost_scale) <= 0.0 or not np.isfinite(float(state_basis.cost_scale)):
        raise ContractError("StateBasis.cost_scale must be finite and strictly positive")
    if state_basis.state_ids is not None and len(state_basis.state_ids) != centroids.shape[0]:
        raise ContractError("StateBasis.state_ids must align to the shared K-state axis")


def load_state_basis(
    *,
    centroids: np.ndarray,
    cost_matrix: np.ndarray | None = None,
    cost_scale: float | None = None,
    feature_key: str = "local_state_features",
    state_key: str = "state_id",
    state_ids: tuple[int, ...] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> StateBasis:
    """Build a validated ``StateBasis`` from explicit shared-axis arrays.

    If the caller omits ``cost_matrix`` or ``cost_scale``, the loader derives
    conservative defaults from the centroid geometry rather than introducing a
    broader basis-learning policy.
    """
    centroids_arr = np.asarray(centroids, dtype=float)
    if centroids_arr.ndim != 2:
        raise ContractError("centroids must be a 2D array")

    if cost_matrix is None:
        cost_matrix_arr = cdist(centroids_arr, centroids_arr, metric="euclidean").astype(
            float,
            copy=False,
        )
    else:
        cost_matrix_arr = np.asarray(cost_matrix, dtype=float)

    if cost_scale is None:
        positive_costs = cost_matrix_arr[cost_matrix_arr > 0.0]
        resolved_cost_scale = float(np.median(positive_costs)) if positive_costs.size > 0 else 1.0
    else:
        resolved_cost_scale = float(cost_scale)

    basis = StateBasis(
        centroids=centroids_arr,
        cost_matrix=cost_matrix_arr,
        cost_scale=resolved_cost_scale,
        feature_key=str(feature_key),
        state_key=str(state_key),
        state_ids=state_ids,
        metadata=dict(metadata or {}),
    )
    validate_state_basis(basis)
    return basis


__all__ = ["StateBasis", "load_state_basis", "validate_state_basis"]
