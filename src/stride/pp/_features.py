"""Cell-level local subtype-neighborhood feature construction for STRIDE .pp."""
from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors

from stride._schema import (
    OBS_CELL_TYPE_KEY,
    OBS_FOV_KEY,
    OBS_PATIENT_KEY,
    OBS_TIMEPOINT_KEY,
    OBSM_LOCAL_STATE_FEATURES_KEY,
    OBSM_SPATIAL_KEY,
    STRIDE_CONFIG_KEY,
    STRIDE_UNS_KEY,
    UNS_STATE_FEATURE_METADATA_KEY,
)
from stride.errors import ContractError
from stride.io._validation import validate_raw_adata


def build_local_features(adata: AnnData) -> AnnData:
    """Build or validate cell-level local state features on an .io-origin AnnData.

    Uses the declared `k_neighbors` from `adata.uns["stride"]["config"]`.
    Writes `adata.obsm["local_state_features"]` when the slot is absent.
    Reuses an existing valid slot with a warning.
    """
    validate_raw_adata(adata)
    k_neighbors = _read_k_neighbors(adata)

    if OBSM_LOCAL_STATE_FEATURES_KEY in adata.obsm:
        _validate_existing_features(adata, k_neighbors=k_neighbors)
        warnings.warn(
            "adata.obsm['local_state_features'] already exists; reusing existing values",
            UserWarning,
            stacklevel=2,
        )
        return adata

    cell_subtypes = adata.obs[OBS_CELL_TYPE_KEY].astype(str).to_numpy()
    spatial = np.asarray(adata.obsm[OBSM_SPATIAL_KEY], dtype=float)
    subtype_labels, subtype_codes = np.unique(cell_subtypes, return_inverse=True)

    features = np.zeros((adata.n_obs, subtype_labels.shape[0]), dtype=float)
    group_keys = [OBS_PATIENT_KEY, OBS_TIMEPOINT_KEY, OBS_FOV_KEY]
    # Full FOV identity prevents repeated fov_id values from mixing across patients or timepoints.
    groups = adata.obs.groupby(group_keys, sort=False, observed=False).indices
    for group_key, group_index in groups.items():
        indices = np.asarray(group_index, dtype=int)
        if indices.shape[0] < k_neighbors + 1:
            raise ContractError(
                "FOV group "
                f"{tuple(group_key)!r} has {indices.shape[0]} cells, fewer than "
                "k_neighbors + 1 cells required for local feature construction"
            )

        model = NearestNeighbors(n_neighbors=k_neighbors)
        model.fit(spatial[indices])
        # X=None uses sklearn's training-query path, which removes each row by identity.
        neighbor_positions = model.kneighbors(return_distance=False)
        neighbor_codes = subtype_codes[indices][neighbor_positions]

        group_features = np.zeros((indices.shape[0], subtype_labels.shape[0]), dtype=float)
        row_index = np.repeat(np.arange(indices.shape[0]), k_neighbors)
        # Accumulate subtype counts for each cell-neighborhood row.
        np.add.at(group_features, (row_index, neighbor_codes.ravel()), 1.0)
        features[indices] = group_features / float(k_neighbors)

    feature_names = [f"subtype_fraction:{label}" for label in subtype_labels.tolist()]
    adata.obsm[OBSM_LOCAL_STATE_FEATURES_KEY] = features
    adata.uns[UNS_STATE_FEATURE_METADATA_KEY] = {
        "feature_names": feature_names,
        "subtype_labels": subtype_labels.tolist(),
        "k_neighbors": k_neighbors,
    }
    return adata


def _read_k_neighbors(adata: AnnData) -> int:
    """Read the declared local-neighborhood size from STRIDE config."""
    stride_uns = adata.uns.get(STRIDE_UNS_KEY)
    if not isinstance(stride_uns, Mapping):
        raise ContractError("adata.uns['stride'] must be a mapping")
    config = stride_uns.get(STRIDE_CONFIG_KEY)
    if not isinstance(config, Mapping):
        raise ContractError("adata.uns['stride']['config'] must be a mapping")

    value = config.get("k_neighbors")
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ContractError("config['k_neighbors'] must be a positive integer")
    if int(value) <= 0:
        raise ContractError("config['k_neighbors'] must be a positive integer")
    return int(value)


def _validate_existing_features(adata: AnnData, *, k_neighbors: int) -> None:
    """Validate an existing local feature matrix and its feature metadata."""
    matrix = np.asarray(adata.obsm[OBSM_LOCAL_STATE_FEATURES_KEY], dtype=float)
    if matrix.ndim != 2:
        raise ContractError("adata.obsm['local_state_features'] must be a 2D matrix")
    if matrix.shape[0] != adata.n_obs:
        raise ContractError(
            "adata.obsm['local_state_features'] row count must align to adata.n_obs"
        )
    if not np.isfinite(matrix).all():
        raise ContractError("adata.obsm['local_state_features'] contains NaN/Inf")
    if (matrix < 0).any():
        raise ContractError("adata.obsm['local_state_features'] must be nonnegative")

    metadata = adata.uns.get(UNS_STATE_FEATURE_METADATA_KEY)
    if not isinstance(metadata, Mapping):
        raise ContractError("adata.uns['state_feature_metadata'] must be a mapping")
    metadata_k = metadata.get("k_neighbors")
    if metadata_k != k_neighbors:
        raise ContractError(
            "adata.uns['state_feature_metadata']['k_neighbors'] must match config"
        )

    feature_names = _metadata_sequence(metadata, "feature_names")
    subtype_labels = _metadata_sequence(metadata, "subtype_labels")
    if len(feature_names) != matrix.shape[1]:
        raise ContractError(
            "adata.uns['state_feature_metadata']['feature_names'] length must match "
            "local_state_features width"
        )
    if len(subtype_labels) != matrix.shape[1]:
        raise ContractError(
            "adata.uns['state_feature_metadata']['subtype_labels'] length must match "
            "local_state_features width"
        )


def _metadata_sequence(metadata: Mapping[str, Any], key: str) -> list[Any]:
    value = metadata.get(key)
    if isinstance(value, pd.Index):
        return value.tolist()
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ContractError(f"adata.uns['state_feature_metadata'][{key!r}] must be 1D")
        return value.tolist()
    if isinstance(value, list | tuple):
        return list(value)
    raise ContractError(f"adata.uns['state_feature_metadata'][{key!r}] must be a sequence")
