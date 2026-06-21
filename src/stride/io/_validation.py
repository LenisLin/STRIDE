"""Small conversion helpers for STRIDE AnnData I/O assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse

from stride._schema import (
    ALLOWED_COMMUNITY_MODES,
    OBS_CELL_TYPE_KEY,
    OBS_DOMAIN_KEY,
    OBS_FOV_KEY,
    OBS_PATIENT_KEY,
    OBS_TIMEPOINT_KEY,
    OBSM_SPATIAL_KEY,
    STRIDE_CONFIG_KEY,
    STRIDE_FOV_METADATA_KEY,
    STRIDE_RELATION_IDS_KEY,
    STRIDE_RELATIONS_KEY,
    STRIDE_UNS_KEY,
)
from stride.errors import ContractError

CANONICAL_OBS_COLUMNS: tuple[str, ...] = (
    OBS_PATIENT_KEY,
    OBS_TIMEPOINT_KEY,
    OBS_FOV_KEY,
    OBS_DOMAIN_KEY,
    OBS_CELL_TYPE_KEY,
)

REQUIRED_CONFIG_KEYS: tuple[str, ...] = (
    "source",
    "target",
    "time_order",
    "community_mode",
    "n_states",
    "k_neighbors",
    STRIDE_RELATIONS_KEY,
    STRIDE_RELATION_IDS_KEY,
)


def _as_matrix(X: object, *, n_obs: int | None = None, name: str = "X") -> np.ndarray:
    """Return a finite dense numeric matrix, optionally checking row count."""
    if sparse.issparse(X):  # noqa: SIM108 - keep sparse densification explicit.
        matrix_input = cast(Any, X).toarray()
    else:
        matrix_input = X
    try:
        matrix = np.asarray(matrix_input, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be coercible to a numeric matrix") from exc

    if matrix.ndim != 2:
        raise ContractError(f"{name} must have shape [n_obs, n_vars]")
    if n_obs is not None and matrix.shape[0] != n_obs:
        raise ContractError(f"{name} row count {matrix.shape[0]} does not match cell rows {n_obs}")
    if not np.isfinite(matrix).all():
        raise ContractError(f"{name} contains NaN/Inf")
    return np.asarray(matrix, dtype=float, order="C")


def _as_var_names(var: Sequence[object], *, n_vars: int) -> pd.DataFrame:
    """Build var metadata from a caller-provided feature-name sequence."""
    if isinstance(var, pd.DataFrame):
        raise ContractError("var must be a sequence of feature names, not a DataFrame")
    if isinstance(var, str):
        raise ContractError("var must be a sequence of feature names, not a string")

    try:
        feature_names = [_as_identifier(name, name="var feature name") for name in var]
    except TypeError as exc:
        raise ContractError("var must be a sequence of feature names") from exc
    if len(feature_names) != n_vars:
        raise ContractError(
            f"var name count {len(feature_names)} does not match X columns {n_vars}"
        )
    return pd.DataFrame(index=pd.Index(feature_names, dtype="object", name="feature"))


def _as_identifier(value: object, *, name: str) -> str:
    """Normalize one scalar declaration or column-name parameter to an ID."""
    if value is None:
        raise ContractError(f"{name} must not be None")
    if isinstance(value, (dict, list, set, tuple)):
        raise ContractError(f"{name} must be a scalar identifier")
    try:
        missing = pd.isna(cast(Any, value))
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        raise ContractError(f"{name} contains a missing value")

    normalized = str(value).strip()
    if normalized == "":
        raise ContractError(f"{name} must be non-empty")
    return normalized


def _as_identifier_column(values: pd.Series, *, where: str, column: str) -> pd.Series:
    """Normalize one cell/fov/obs metadata column to string identifiers."""
    if values.isna().any():
        raise ContractError(f"{where}: column {column!r} contains missing values")
    normalized = values.astype(str).str.strip()
    if (normalized == "").any():
        raise ContractError(f"{where}: column {column!r} contains empty string values")
    return normalized


def _as_spatial(cell_table: pd.DataFrame, *, x: str, y: str, where: str = "cell") -> np.ndarray:
    """Return finite [n_obs, 2] spatial coordinates from x/y columns."""
    try:
        spatial = cell_table.loc[:, [x, y]].to_numpy(dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"{where}: x/y columns must be present and numeric") from exc
    if spatial.ndim != 2 or spatial.shape[1] != 2:
        raise ContractError(f"{where}: spatial coordinates must have shape [n_obs, 2]")
    if not np.isfinite(spatial).all():
        raise ContractError(f"{where}: spatial coordinates contain NaN/Inf")
    return spatial


def validate_raw_adata(adata: Any) -> None:
    """Validate only the raw .io AnnData slots and required STRIDE keys."""
    if getattr(adata, "obs", None) is None:
        raise ContractError("AnnData object must define obs")
    if getattr(adata, "var", None) is None:
        raise ContractError("AnnData object must define var")
    if getattr(adata, "obsm", None) is None:
        raise ContractError("AnnData object must define obsm")
    if getattr(adata, "uns", None) is None:
        raise ContractError("AnnData object must define uns")

    if int(getattr(adata, "n_obs", 0)) <= 0:
        raise ContractError("AnnData object must contain at least one observation row")
    if int(getattr(adata, "n_vars", 0)) <= 0:
        raise ContractError("AnnData object must contain at least one feature column")

    missing_obs = [column for column in CANONICAL_OBS_COLUMNS if column not in adata.obs.columns]
    if missing_obs:
        raise ContractError(f"adata.obs: missing required columns: {missing_obs}")
    for column in CANONICAL_OBS_COLUMNS:
        normalized = _as_identifier_column(adata.obs[column], where="adata.obs", column=column)
        if normalized.shape[0] != adata.n_obs:
            raise ContractError(f"adata.obs[{column!r}] failed identifier normalization")

    if OBSM_SPATIAL_KEY not in adata.obsm:
        raise ContractError(f"adata.obsm: missing required key {OBSM_SPATIAL_KEY!r}")
    spatial = np.asarray(adata.obsm[OBSM_SPATIAL_KEY], dtype=float)
    if spatial.ndim != 2 or spatial.shape[1] != 2:
        raise ContractError("adata.obsm['spatial'] must have shape [n_obs, 2]")
    if spatial.shape[0] != adata.n_obs:
        raise ContractError("adata.obsm['spatial'] row count must align to adata.n_obs")
    if not np.isfinite(spatial).all():
        raise ContractError("adata.obsm['spatial'] contains NaN/Inf")

    if STRIDE_UNS_KEY not in adata.uns or not isinstance(adata.uns[STRIDE_UNS_KEY], Mapping):
        raise ContractError(f"adata.uns must contain mapping key {STRIDE_UNS_KEY!r}")
    stride_uns = adata.uns[STRIDE_UNS_KEY]
    for key in (STRIDE_CONFIG_KEY, STRIDE_FOV_METADATA_KEY):
        if key not in stride_uns:
            raise ContractError(f"adata.uns['stride']: missing required key {key!r}")

    config = stride_uns[STRIDE_CONFIG_KEY]
    if not isinstance(config, Mapping):
        raise ContractError("adata.uns['stride']['config'] must be a mapping")
    missing_config = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing_config:
        raise ContractError(f"config: missing required keys: {missing_config}")

    normalized_source = _as_identifier(config["source"], name="config['source']")
    normalized_target = _as_identifier(config["target"], name="config['target']")
    if normalized_source == normalized_target:
        raise ContractError("config['source'] and config['target'] must be distinct")

    raw_time_order = config["time_order"]
    if isinstance(raw_time_order, str):
        raise ContractError("config['time_order'] must be a sequence, not a string")
    try:
        normalized_time_order = tuple(
            _as_identifier(value, name="config['time_order'] value") for value in raw_time_order
        )
    except TypeError as exc:
        raise ContractError("config['time_order'] must be a sequence") from exc
    if not normalized_time_order:
        raise ContractError("config['time_order'] must contain at least one timepoint")
    if len(set(normalized_time_order)) != len(normalized_time_order):
        raise ContractError("config['time_order'] contains duplicate values")
    if normalized_source not in normalized_time_order:
        raise ContractError("config['source'] must be present in config['time_order']")
    if normalized_target not in normalized_time_order:
        raise ContractError("config['target'] must be present in config['time_order']")

    community_mode = _as_identifier(config["community_mode"], name="config['community_mode']")
    if community_mode not in ALLOWED_COMMUNITY_MODES:
        raise ContractError(
            f"config['community_mode'] must be one of {list(ALLOWED_COMMUNITY_MODES)}"
        )

    for key in ("n_states", "k_neighbors"):
        value = config[key]
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
            raise ContractError(f"config[{key!r}] must be a positive integer")
        if int(value) <= 0:
            raise ContractError(f"config[{key!r}] must be a positive integer")

    _validate_config_relations(config)


def _validate_config_relations(config: Mapping[str, Any]) -> None:
    """Validate declared relation arrays stored by stride.io."""
    relations = np.asarray(config[STRIDE_RELATIONS_KEY], dtype=object)
    if relations.ndim != 2 or relations.shape[1] != 2:
        raise ContractError("config['relations'] must have shape [n_relations, 2]")
    if relations.shape[0] == 0:
        raise ContractError("config['relations'] must contain at least one relation")

    raw_relation_ids = config[STRIDE_RELATION_IDS_KEY]
    if isinstance(raw_relation_ids, str):
        raise ContractError("config['relation_ids'] must be a sequence, not a string")
    try:
        relation_ids = [
            _as_identifier(value, name="config['relation_ids'] value") for value in raw_relation_ids
        ]
    except TypeError as exc:
        raise ContractError("config['relation_ids'] must be a sequence") from exc
    if len(relation_ids) != int(relations.shape[0]):
        raise ContractError("config['relation_ids'] length must match config['relations']")
    if len(set(relation_ids)) != len(relation_ids):
        raise ContractError("config['relation_ids'] contains duplicate values")

    seen_pairs: set[tuple[str, str]] = set()
    for index, (source_domain, target_domain) in enumerate(relations.tolist()):
        pair = (
            _as_identifier(source_domain, name=f"config['relations'][{index}] source_domain_label"),
            _as_identifier(target_domain, name=f"config['relations'][{index}] target_domain_label"),
        )
        if pair in seen_pairs:
            raise ContractError(f"config['relations'] contains duplicate domain pair: {pair}")
        seen_pairs.add(pair)
