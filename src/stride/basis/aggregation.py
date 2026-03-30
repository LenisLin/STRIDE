"""Shared community-state feature construction and basis learning for STRIDE."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from ..data.longitudinal import (
    CANONICAL_FEATURE_KEY,
    CANONICAL_STATE_FEATURE_METADATA_KEY,
    resolve_cell_subtype_key,
    resolve_fov_key,
)
from ..errors import ContractError
from .contracts import StateBasis, load_state_basis

try:
    from anndata import AnnData
except ModuleNotFoundError:  # pragma: no cover - optional dependency for non-AnnData paths
    AnnData = object  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class CommunityAggregationConfig:
    """Configuration for deterministic community-state basis learning."""

    k_neighbors: int = 20
    random_state: int = 42
    feature_key: str = CANONICAL_FEATURE_KEY
    state_key: str = "state_id"


def _build_identity_scaler_payload(feature_names: list[str]) -> dict[str, np.ndarray | list[str]]:
    """Compatibility payload for legacy shims that still expect scaler metadata."""
    n_features = len(feature_names)
    return {
        "feature_names": feature_names,
        "center": np.zeros(n_features, dtype=np.float32),
        "scale": np.ones(n_features, dtype=np.float32),
    }


def build_local_state_features(
    adata: AnnData,
    k: int = 20,
    *,
    write_compat_aliases: bool = False,
) -> np.ndarray:
    """Build per-cell neighborhood subtype proportions within each ROI/FOV.

    This is the first canonical upstream STRIDE path:
    1. compute cell-level spatial neighborhoods inside each FOV/ROI;
    2. convert each neighborhood into subtype proportion features;
    3. persist those features for deterministic shared community-state learning.
    """
    if k <= 0:
        raise ContractError("k must be strictly positive")

    subtype_key = resolve_cell_subtype_key(adata)
    fov_key = resolve_fov_key(adata)
    spatial = np.asarray(adata.obsm["spatial"], dtype=float)
    if spatial.ndim != 2 or spatial.shape[1] != 2:
        raise ContractError("adata.obsm['spatial'] must have shape [n_cells, 2]")
    if not np.isfinite(spatial).all():
        raise ContractError("adata.obsm['spatial'] contains NaN/Inf")

    cell_subtypes = adata.obs[subtype_key].astype(str).to_numpy()
    fov_ids = adata.obs[fov_key].astype(str).to_numpy()
    subtype_names = np.unique(cell_subtypes)
    subtype_to_idx = {name: idx for idx, name in enumerate(subtype_names.tolist())}
    subtype_index = np.asarray([subtype_to_idx[name] for name in cell_subtypes], dtype=int)

    features = np.zeros((spatial.shape[0], len(subtype_names)), dtype=np.float32)
    for fov_id in np.unique(fov_ids):
        fov_mask = fov_ids == fov_id
        fov_indices = np.flatnonzero(fov_mask)
        fov_coords = spatial[fov_indices]
        if fov_indices.size < 2:
            raise ContractError(
                f"FOV/ROI {fov_id!r} must contain at least 2 cells for neighborhood features"
            )

        effective_k = min(int(k), int(fov_indices.size) - 1)
        model = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
        model.fit(fov_coords)
        neighbors = model.kneighbors(fov_coords, return_distance=False)[:, 1:]

        neighbor_types = subtype_index[fov_indices][neighbors]
        composition = np.zeros((fov_indices.size, len(subtype_names)), dtype=np.float32)
        row_index = np.repeat(np.arange(fov_indices.size), effective_k)
        np.add.at(composition, (row_index, neighbor_types.reshape(-1)), 1.0)
        composition /= float(effective_k)
        features[fov_indices] = composition

    feature_names = [f"cell_subtype_prop::{name}" for name in subtype_names.tolist()]
    feature_metadata = {
        "feature_names": feature_names,
        "subtype_labels": subtype_names.tolist(),
        "k_neighbors": int(k),
    }

    adata.obsm[CANONICAL_FEATURE_KEY] = features
    adata.uns[CANONICAL_STATE_FEATURE_METADATA_KEY] = feature_metadata
    if write_compat_aliases:
        adata.obsm["community_features"] = features
        adata.uns["scaler_params"] = _build_identity_scaler_payload(feature_names)
    return features


def aggregate_local_features(
    adata: AnnData,
    *,
    group_key: str | None = None,
    feature_key: str = CANONICAL_FEATURE_KEY,
) -> dict[str, np.ndarray]:
    """Aggregate cell-level neighborhood features into observation-level means."""
    active_group_key = group_key or resolve_fov_key(adata)
    if active_group_key not in adata.obs.columns:
        raise ContractError(f"adata.obs is missing grouping key {active_group_key!r}")
    if feature_key not in adata.obsm:
        raise ContractError(f"adata.obsm is missing feature key {feature_key!r}")

    features = np.asarray(adata.obsm[feature_key], dtype=float)
    groups = adata.obs[active_group_key].astype(str).to_numpy()
    unique_groups = tuple(np.unique(groups).tolist())
    aggregated = np.vstack([np.mean(features[groups == group], axis=0) for group in unique_groups]).astype(
        float,
        copy=False,
    )
    return {
        "group_ids": np.asarray(unique_groups, dtype=object),
        "feature_matrix": aggregated,
    }


def learn_shared_state_axis(
    adata: AnnData,
    n_bal: int | None = None,
    K: int | None = None,
    random_state: int = 42,
    feature_key: str | None = None,
    *,
    write_compat_aliases: bool = False,
) -> StateBasis:
    """Learn and persist deterministic shared community-state prototypes.

    ``n_bal`` is retained only as a compatibility parameter. The canonical
    first-pass STRIDE route fits k-means on all valid per-cell neighborhood
    subtype features.
    """
    del n_bal

    if K is None:
        raise ContractError("K must be provided")
    if K <= 0:
        raise ContractError("K must be strictly positive")

    active_feature_key = feature_key
    if active_feature_key is None:
        if CANONICAL_FEATURE_KEY in adata.obsm:
            active_feature_key = CANONICAL_FEATURE_KEY
        elif "community_features" in adata.obsm:
            active_feature_key = "community_features"
        else:
            raise ContractError(
                "adata.obsm must contain 'local_state_features' or 'community_features' before state-axis learning"
            )

    features = np.asarray(adata.obsm[active_feature_key], dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != adata.obs.shape[0]:
        raise ContractError(f"adata.obsm[{active_feature_key!r}] must have shape [n_cells, d]")
    if not np.isfinite(features).all():
        raise ContractError(f"adata.obsm[{active_feature_key!r}] contains NaN/Inf")
    if features.shape[0] < K:
        raise ContractError(
            f"All-cell feature matrix must contain at least K={K} rows, got {features.shape[0]}"
        )

    model = KMeans(n_clusters=K, random_state=random_state, n_init=20)
    model.fit(features)
    state_id = model.predict(features).astype(int, copy=False)

    metadata = {}
    if CANONICAL_STATE_FEATURE_METADATA_KEY in adata.uns:
        metadata[CANONICAL_STATE_FEATURE_METADATA_KEY] = adata.uns[CANONICAL_STATE_FEATURE_METADATA_KEY]
    basis = load_state_basis(
        centroids=np.asarray(model.cluster_centers_, dtype=np.float32),
        feature_key=active_feature_key,
        state_key="state_id",
        metadata=metadata,
    )

    adata.obs["state_id"] = state_id
    adata.uns["state_centroids"] = basis.centroids
    adata.uns["cost_matrix"] = basis.cost_matrix
    adata.uns["cost_scale"] = basis.cost_scale
    if write_compat_aliases:
        adata.obs["proto_id"] = state_id
        adata.uns["prototype_centroids"] = basis.centroids
        adata.uns["s_C"] = basis.cost_scale
        adata.uns["global_cost_scale"] = basis.cost_scale

    return basis


__all__ = [
    "CommunityAggregationConfig",
    "StateBasis",
    "aggregate_local_features",
    "build_local_state_features",
    "learn_shared_state_axis",
]
