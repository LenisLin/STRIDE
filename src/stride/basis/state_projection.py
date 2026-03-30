"""Projection helpers for assigning and aggregating STRIDE community states."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from ..errors import ContractError
from .contracts import StateBasis


def assign_state_ids(
    feature_matrix: np.ndarray,
    state_basis: StateBasis,
) -> np.ndarray:
    """Assign each feature row to its nearest shared community-state prototype."""
    features = np.asarray(feature_matrix, dtype=float)
    centroids = np.asarray(state_basis.centroids, dtype=float)

    if features.ndim != 2:
        raise ContractError("feature_matrix must be a 2D array")
    if centroids.ndim != 2:
        raise ContractError("state_basis.centroids must be a 2D array")
    if features.shape[1] != centroids.shape[1]:
        raise ContractError(
            "feature_matrix width must match the shared community-state centroid width"
        )
    if not np.isfinite(features).all():
        raise ContractError("feature_matrix contains NaN/Inf")

    distances = cdist(features, centroids, metric="euclidean")
    return np.argmin(distances, axis=1).astype(int, copy=False)


def aggregate_to_state_basis(
    state_ids: np.ndarray,
    observation_ids: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    n_states: int | None = None,
) -> dict[str, np.ndarray]:
    """Aggregate cell-level state assignments into observation-level state payloads."""
    state_arr = np.asarray(state_ids, dtype=int).reshape(-1)
    obs_arr = np.asarray(observation_ids).reshape(-1)
    if state_arr.shape[0] != obs_arr.shape[0]:
        raise ContractError("state_ids and observation_ids must have the same length")
    if state_arr.size == 0:
        raise ContractError("state_ids and observation_ids must be non-empty")

    if weights is None:
        weight_arr = np.ones(state_arr.shape[0], dtype=float)
    else:
        weight_arr = np.asarray(weights, dtype=float).reshape(-1)
        if weight_arr.shape != state_arr.shape:
            raise ContractError("weights must align to state_ids and observation_ids")
        if not np.isfinite(weight_arr).all() or (weight_arr < 0.0).any():
            raise ContractError("weights must be finite and non-negative")

    K = int(n_states) if n_states is not None else int(np.max(state_arr)) + 1
    if K <= 0:
        raise ContractError("n_states must be strictly positive")
    if (state_arr < 0).any() or (state_arr >= K).any():
        raise ContractError("state_ids contains values outside the declared shared state basis")

    unique_obs = tuple(np.unique(obs_arr.astype(str)).tolist())
    burden = np.zeros((len(unique_obs), K), dtype=float)
    obs_str = obs_arr.astype(str)
    for idx, obs_id in enumerate(unique_obs):
        mask = obs_str == obs_id
        burden[idx] = np.bincount(
            state_arr[mask],
            weights=weight_arr[mask],
            minlength=K,
        ).astype(float, copy=False)
    row_sums = np.sum(burden, axis=1, dtype=float)
    composition = np.divide(
        burden,
        row_sums[:, None],
        out=np.zeros_like(burden),
        where=row_sums[:, None] > 0.0,
    )
    return {
        "observation_ids": np.asarray(unique_obs, dtype=object),
        "state_burden": burden,
        "state_composition": composition,
    }


def project_state_basis(*args: object, **kwargs: object) -> dict[str, np.ndarray]:
    """Compatibility alias for ``aggregate_to_state_basis``."""
    return aggregate_to_state_basis(*args, **kwargs)


__all__ = ["aggregate_to_state_basis", "assign_state_ids", "project_state_basis"]
