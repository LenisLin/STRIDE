"""Fit-readiness validation for STRIDE .pp outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import pandas as pd
from anndata import AnnData

from stride._schema import (
    OBS_DOMAIN_KEY,
    OBS_FOV_KEY,
    OBS_PATIENT_KEY,
    OBS_STATE_ID_KEY,
    OBS_TIMEPOINT_KEY,
    STRIDE_CONFIG_KEY,
    STRIDE_FOV_OBSERVATIONS_KEY,
    STRIDE_RELATION_IDS_KEY,
    STRIDE_RELATIONS_KEY,
    STRIDE_UNS_KEY,
    UNS_COST_MATRIX_KEY,
    UNS_COST_SCALE_KEY,
    UNS_STATE_CENTROIDS_KEY,
)
from stride.errors import ContractError

_GEOMETRY_ATOL = 1e-12


def validate_ready(adata: AnnData) -> None:
    """Validate that AnnData contains the preprocessing outputs needed by `stride.tl`.

    Checks the completed representation, geometry, and FOV observation layer.
    """
    config = _read_ready_config(adata)
    n_states = cast(int, config["n_states"])
    _check_state_handoff(adata, n_states)
    _check_geometry_handoff(adata, n_states)
    metadata = _check_fov_observation_handoff(adata, n_states)
    _check_relation_support(metadata, config)


def _read_ready_config(adata: AnnData) -> dict[str, object]:
    """Read `.tl` handoff config values."""
    stride_uns = adata.uns.get(STRIDE_UNS_KEY)
    if not isinstance(stride_uns, Mapping):
        raise ContractError("adata.uns['stride'] must be a mapping")
    config = stride_uns.get(STRIDE_CONFIG_KEY)
    if not isinstance(config, Mapping):
        raise ContractError("adata.uns['stride']['config'] must be a mapping")

    if config.get("community_mode") != "fraction":
        raise ContractError("config['community_mode'] must be 'fraction'")

    n_states_value = config.get("n_states")
    if (
        not isinstance(n_states_value, (int, np.integer))
        or isinstance(n_states_value, (bool, np.bool_))
        or int(n_states_value) <= 0
    ):
        raise ContractError("config['n_states'] must be a positive integer")

    if "source" not in config:
        raise ContractError("config['source'] is required")
    if "target" not in config:
        raise ContractError("config['target'] is required")

    if STRIDE_RELATIONS_KEY not in config:
        raise ContractError("config['relations'] is required")
    relations = np.asarray(config[STRIDE_RELATIONS_KEY], dtype=object)
    if relations.ndim != 2 or relations.shape[1] != 2:
        raise ContractError("config['relations'] shape must be [n_relations, 2]")

    if STRIDE_RELATION_IDS_KEY not in config:
        raise ContractError("config['relation_ids'] is required")
    raw_relation_ids = config[STRIDE_RELATION_IDS_KEY]
    if isinstance(raw_relation_ids, str):
        raise ContractError("config['relation_ids'] must be a sequence, not a string")
    try:
        relation_ids = list(raw_relation_ids)
    except TypeError as exc:
        raise ContractError("config['relation_ids'] must be a sequence") from exc
    if len(relation_ids) != relations.shape[0]:
        raise ContractError(
            "config['relation_ids'] length must match config['relations'] row count"
        )

    return {
        "source": config["source"],
        "target": config["target"],
        "n_states": int(n_states_value),
        STRIDE_RELATIONS_KEY: relations,
        STRIDE_RELATION_IDS_KEY: list(relation_ids),
    }


def _check_state_handoff(adata: AnnData, n_states: int) -> None:
    """Check shared-state slots used by `.tl`."""
    if OBS_STATE_ID_KEY not in adata.obs:
        raise ContractError("adata.obs['state_id'] is required")
    if UNS_STATE_CENTROIDS_KEY not in adata.uns:
        raise ContractError("adata.uns['state_centroids'] is required")

    centroids = np.asarray(adata.uns[UNS_STATE_CENTROIDS_KEY])
    if centroids.ndim < 1 or centroids.shape[0] != n_states:
        raise ContractError("adata.uns['state_centroids'] row count must match config['n_states']")


def _check_geometry_handoff(adata: AnnData, n_states: int) -> None:
    """Check geometry slots used by `.tl`."""
    if UNS_COST_MATRIX_KEY not in adata.uns:
        raise ContractError("adata.uns['cost_matrix'] is required")
    if UNS_COST_SCALE_KEY not in adata.uns:
        raise ContractError("adata.uns['cost_scale'] is required")

    try:
        cost_matrix = np.asarray(adata.uns[UNS_COST_MATRIX_KEY], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ContractError("adata.uns['cost_matrix'] contains NaN/Inf") from exc
    if cost_matrix.ndim != 2 or cost_matrix.shape != (n_states, n_states):
        raise ContractError("adata.uns['cost_matrix'] shape must be [n_states, n_states]")
    if not np.isfinite(cost_matrix).all():
        raise ContractError("adata.uns['cost_matrix'] contains NaN/Inf")
    offdiag_mask = ~np.eye(n_states, dtype=bool)
    if (cost_matrix[offdiag_mask] < 0.0).any():
        raise ContractError("adata.uns['cost_matrix'] off-diagonal entries must be nonnegative")
    if not np.allclose(cost_matrix, cost_matrix.T, rtol=0.0, atol=_GEOMETRY_ATOL):
        raise ContractError("adata.uns['cost_matrix'] must be symmetric")
    if not np.allclose(np.diag(cost_matrix), 0.0, rtol=0.0, atol=_GEOMETRY_ATOL):
        raise ContractError("adata.uns['cost_matrix'] diagonal must be zero")

    try:
        cost_scale = np.asarray(adata.uns[UNS_COST_SCALE_KEY], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ContractError("adata.uns['cost_scale'] must be finite and strictly positive") from exc
    if cost_scale.shape != ():
        raise ContractError("adata.uns['cost_scale'] must be finite and strictly positive")
    scale = float(cost_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ContractError("adata.uns['cost_scale'] must be finite and strictly positive")


def _check_fov_observation_handoff(
    adata: AnnData,
    n_states: int,
) -> pd.DataFrame:
    """Check FOV observation payload shape and row metadata."""
    stride_uns = adata.uns.get(STRIDE_UNS_KEY)
    if not isinstance(stride_uns, Mapping):
        raise ContractError("adata.uns['stride'] must be a mapping")

    if STRIDE_FOV_OBSERVATIONS_KEY not in stride_uns:
        raise ContractError("adata.uns['stride']['fov_observations'] is required")
    fov_observations = stride_uns[STRIDE_FOV_OBSERVATIONS_KEY]
    if not isinstance(fov_observations, Mapping):
        raise ContractError("adata.uns['stride']['fov_observations'] must be a mapping")

    if "community_composition" not in fov_observations:
        raise ContractError(
            "adata.uns['stride']['fov_observations']['community_composition'] is required"
        )
    if "metadata" not in fov_observations:
        raise ContractError("adata.uns['stride']['fov_observations']['metadata'] is required")

    community_composition = np.asarray(fov_observations["community_composition"])
    if community_composition.ndim != 2:
        raise ContractError("fov_observations['community_composition'] must be a 2D matrix")
    if community_composition.shape[1] != n_states:
        raise ContractError(
            "fov_observations['community_composition'] column count must match config['n_states']"
        )

    metadata = fov_observations["metadata"]
    if not isinstance(metadata, pd.DataFrame):
        raise ContractError("fov_observations['metadata'] must be a pandas DataFrame")

    required_columns = [
        OBS_PATIENT_KEY,
        OBS_TIMEPOINT_KEY,
        OBS_FOV_KEY,
        OBS_DOMAIN_KEY,
    ]
    missing_columns = [column for column in required_columns if column not in metadata]
    if missing_columns:
        raise ContractError(
            "fov_observations['metadata'] is missing required columns: "
            + ", ".join(missing_columns)
        )
    if metadata.shape[0] != community_composition.shape[0]:
        raise ContractError(
            "fov_observations['metadata'] row count must match "
            "fov_observations['community_composition'] row count"
        )
    return metadata


def _check_relation_support(
    metadata: pd.DataFrame,
    config: Mapping[str, object],
) -> None:
    """Check declared relation support in FOV observation metadata."""
    source = config["source"]
    target = config["target"]
    relations = np.asarray(config[STRIDE_RELATIONS_KEY], dtype=object)

    for index, (source_domain, target_domain) in enumerate(relations.tolist()):
        source_supported = (
            (metadata[OBS_TIMEPOINT_KEY] == source) & (metadata[OBS_DOMAIN_KEY] == source_domain)
        ).any()
        if not bool(source_supported):
            raise ContractError(
                f"relations[{index}] source domain_label {source_domain!r} "
                f"must have FOV observation support at source timepoint {source!r}"
            )

        target_supported = (
            (metadata[OBS_TIMEPOINT_KEY] == target) & (metadata[OBS_DOMAIN_KEY] == target_domain)
        ).any()
        if not bool(target_supported):
            raise ContractError(
                f"relations[{index}] target domain_label {target_domain!r} "
                f"must have FOV observation support at target timepoint {target!r}"
            )
