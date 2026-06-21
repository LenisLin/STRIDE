"""Shared community-state basis construction for STRIDE .pp."""
from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
from anndata import AnnData
from sklearn.cluster import KMeans

from stride._schema import (
    OBS_STATE_ID_KEY,
    OBSM_LOCAL_STATE_FEATURES_KEY,
    STRIDE_CONFIG_KEY,
    STRIDE_UNS_KEY,
    UNS_STATE_CENTROIDS_KEY,
)
from stride.errors import ContractError


def build_state_basis(adata: AnnData) -> AnnData:
    """Build or validate the shared K-state basis from local state features.

    Uses the declared `n_states` from `adata.uns["stride"]["config"]`.
    Writes `adata.obs["state_id"]` and `adata.uns["state_centroids"]`.
    Reuses an existing valid basis with a warning.
    """
    n_states = _read_n_states(adata)

    has_state_id = OBS_STATE_ID_KEY in adata.obs
    has_centroids = UNS_STATE_CENTROIDS_KEY in adata.uns
    if has_state_id != has_centroids:
        raise ContractError(
            "shared state basis is partial; both adata.obs['state_id'] and "
            "adata.uns['state_centroids'] are required"
        )

    features = _local_features(adata)
    if features.shape[0] < n_states:
        raise ContractError(
            "adata.obsm['local_state_features'] has fewer than config['n_states'] rows"
        )

    if has_state_id and has_centroids:
        _validate_existing_basis(
            adata,
            n_states=n_states,
            n_features=features.shape[1],
        )
        warnings.warn(
            "adata.obs['state_id'] and adata.uns['state_centroids'] already exist; "
            "reusing existing values",
            UserWarning,
            stacklevel=2,
        )
        return adata

    # local_state_features is produced by build_local_features.
    # KMeans parameters are fixed for deterministic shared-state learning.
    model = KMeans(n_clusters=n_states, random_state=42, n_init=20)
    model.fit(features)
    adata.obs[OBS_STATE_ID_KEY] = model.predict(features).astype(int, copy=False)
    adata.uns[UNS_STATE_CENTROIDS_KEY] = np.asarray(model.cluster_centers_, dtype=float)
    # build_state_geometry owns cost_matrix and cost_scale.
    return adata


def _read_n_states(adata: AnnData) -> int:
    """Read the declared shared-state count from STRIDE config."""
    stride_uns = adata.uns.get(STRIDE_UNS_KEY)
    if not isinstance(stride_uns, Mapping):
        raise ContractError("adata.uns['stride'] must be a mapping")
    config = stride_uns.get(STRIDE_CONFIG_KEY)
    if not isinstance(config, Mapping):
        raise ContractError("adata.uns['stride']['config'] must be a mapping")

    value = config.get("n_states")
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ContractError("config['n_states'] must be a positive integer")
    if int(value) <= 0:
        raise ContractError("config['n_states'] must be a positive integer")
    return int(value)


def _local_features(adata: AnnData) -> np.ndarray:
    """Return the local feature matrix used for shared-state learning."""
    if OBSM_LOCAL_STATE_FEATURES_KEY not in adata.obsm:
        raise ContractError("adata.obsm['local_state_features'] is required")
    matrix = np.asarray(adata.obsm[OBSM_LOCAL_STATE_FEATURES_KEY], dtype=float)
    if matrix.ndim != 2:
        raise ContractError("adata.obsm['local_state_features'] must be a 2D matrix")
    if matrix.shape[0] != adata.n_obs:
        raise ContractError(
            "adata.obsm['local_state_features'] row count must align to adata.n_obs"
        )
    if not np.isfinite(matrix).all():
        raise ContractError("adata.obsm['local_state_features'] contains NaN/Inf")
    return matrix


def _validate_existing_basis(
    adata: AnnData,
    *,
    n_states: int,
    n_features: int,
) -> None:
    """Validate existing state assignments and centroids."""
    state_id = np.asarray(adata.obs[OBS_STATE_ID_KEY])
    if state_id.shape[0] != adata.n_obs:
        raise ContractError("adata.obs['state_id'] length must align to adata.n_obs")
    try:
        state_id_float = np.asarray(state_id, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ContractError("adata.obs['state_id'] must be integer-compatible") from exc
    if not np.isfinite(state_id_float).all():
        raise ContractError("adata.obs['state_id'] contains NaN/Inf")
    if not np.equal(state_id_float, np.floor(state_id_float)).all():
        raise ContractError("adata.obs['state_id'] must be integer-compatible")
    state_id_int = state_id_float.astype(int, copy=False)
    if ((state_id_int < 0) | (state_id_int >= n_states)).any():
        raise ContractError("adata.obs['state_id'] values must be in [0, n_states - 1]")

    centroids = np.asarray(adata.uns[UNS_STATE_CENTROIDS_KEY], dtype=float)
    if centroids.ndim != 2:
        raise ContractError("adata.uns['state_centroids'] must be a 2D matrix")
    if not np.isfinite(centroids).all():
        raise ContractError("adata.uns['state_centroids'] contains NaN/Inf")
    if centroids.shape != (n_states, n_features):
        raise ContractError(
            "adata.uns['state_centroids'] shape must be "
            "[n_states, local_state_features.shape[1]]"
        )
