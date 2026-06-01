"""Geometry-layer helpers derived from a fixed shared STRIDE state basis."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from ..errors import ContractError


@dataclass(frozen=True)
class StateGeometry:
    """Canonical geometry derived on a fixed shared-state basis.

    The geometry layer carries pairwise community-state distances together with
    a sparse neighborhood graph and a scaled similarity graph. It is support
    structure for downstream priors and matching, not a hard transition
    ontology.
    """

    cost_matrix: np.ndarray
    cost_scale: float
    adjacency_matrix: np.ndarray
    similarity_graph: np.ndarray
    state_ids: tuple[int, ...]

    @property
    def distance_matrix(self) -> np.ndarray:
        """Compatibility alias for the pairwise state-distance matrix."""
        return self.cost_matrix


def _validate_geometry_cost_matrix(cost_matrix: np.ndarray) -> np.ndarray:
    """Return a validated geometry cost matrix aligned to the shared state axis."""
    C = np.asarray(cost_matrix, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ContractError("cost_matrix must be square")
    if C.shape[0] == 0:
        raise ContractError("cost_matrix must contain at least one state")
    if not np.isfinite(C).all():
        raise ContractError("cost_matrix must contain only finite values")
    if np.any(C < 0.0):
        raise ContractError("cost_matrix must be nonnegative")
    if not np.allclose(C, C.T, rtol=0.0, atol=1e-12):
        raise ContractError("cost_matrix must be symmetric")
    if not np.allclose(np.diag(C), 0.0, rtol=0.0, atol=1e-12):
        raise ContractError("cost_matrix diagonal must be zero")
    return C


def _positive_off_diagonal_costs(cost_matrix: np.ndarray) -> np.ndarray:
    off_diagonal_mask = ~np.eye(cost_matrix.shape[0], dtype=bool)
    return cost_matrix[off_diagonal_mask & (cost_matrix > 0.0)]


def build_similarity_graph(cost_matrix: np.ndarray, *, n_neighbors: int = 5) -> np.ndarray:
    """Construct a symmetric nearest-neighbor graph on the shared state axis."""
    C = _validate_geometry_cost_matrix(cost_matrix)
    if n_neighbors <= 0:
        raise ContractError("n_neighbors must be strictly positive")

    n_states = C.shape[0]
    adjacency = np.eye(n_states, dtype=float)
    k = min(int(n_neighbors), max(n_states - 1, 0))
    if k == 0:
        return adjacency

    for row_idx in range(n_states):
        order = np.argsort(C[row_idx], kind="mergesort")
        neighbors = order[1 : k + 1]
        adjacency[row_idx, neighbors] = 1.0

    return np.maximum(adjacency, adjacency.T)


def build_state_geometry(
    *,
    centroids: np.ndarray | None = None,
    cost_matrix: np.ndarray | None = None,
    cost_scale: float | None = None,
    state_ids: tuple[int, ...] | None = None,
    n_neighbors: int = 5,
) -> StateGeometry:
    """Build geometry-layer objects from basis centroids or a provided cost matrix."""
    if centroids is None and cost_matrix is None:
        raise ContractError("Either centroids or cost_matrix must be provided")

    if cost_matrix is None:
        centroids_arr = np.asarray(centroids, dtype=float)
        if centroids_arr.ndim != 2:
            raise ContractError("centroids must be a 2D array")
        if not np.isfinite(centroids_arr).all():
            raise ContractError("centroids must contain only finite values")
        C = cdist(centroids_arr, centroids_arr, metric="euclidean").astype(float, copy=False)
    else:
        C = np.asarray(cost_matrix, dtype=float)

    C = _validate_geometry_cost_matrix(C)

    positive_costs = _positive_off_diagonal_costs(C)
    if positive_costs.size == 0:
        raise ContractError("cost_matrix must contain a positive off-diagonal cost")

    scale = float(cost_scale) if cost_scale is not None else float(np.median(positive_costs))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ContractError("cost_scale must be finite and strictly positive")

    adjacency = build_similarity_graph(C, n_neighbors=n_neighbors)
    similarity = np.exp(-C / scale, dtype=float) * adjacency
    resolved_state_ids = state_ids or tuple(range(C.shape[0]))
    if len(resolved_state_ids) != C.shape[0]:
        raise ContractError("state_ids must align to the shared K-state axis")
    return StateGeometry(
        cost_matrix=C,
        cost_scale=scale,
        adjacency_matrix=adjacency,
        similarity_graph=similarity,
        state_ids=resolved_state_ids,
    )


__all__ = ["StateGeometry", "build_similarity_graph", "build_state_geometry"]
