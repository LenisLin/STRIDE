"""Table-to-AnnData assembly and validation helpers for STRIDE longitudinal inputs."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..basis.contracts import StateBasis, load_state_basis
from ..errors import ContractError

try:
    from anndata import AnnData
except ModuleNotFoundError:  # pragma: no cover - optional dependency for non-AnnData paths
    class AnnData:  # type: ignore[override]
        """Fallback so non-AnnData workflows can still import contract validators."""

        pass


CANONICAL_TIMEPOINT_KEY = "timepoint"
TIMEPOINT_KEY_ALIASES: tuple[str, ...] = (CANONICAL_TIMEPOINT_KEY, "timepoint_id")
CANONICAL_FOV_KEY = "fov_id"
FOV_KEY_ALIASES: tuple[str, ...] = (CANONICAL_FOV_KEY, "roi_id")
CANONICAL_DOMAIN_KEY = "domain_label"
DOMAIN_KEY_ALIASES: tuple[str, ...] = (CANONICAL_DOMAIN_KEY, "compartment")
CANONICAL_CELL_SUBTYPE_KEY = "cell_subtype_label"
CELL_SUBTYPE_KEY_ALIASES: tuple[str, ...] = (CANONICAL_CELL_SUBTYPE_KEY, "cell_type")

LONGITUDINAL_GROUP_KEYS: tuple[str, ...] = (
    "patient_id",
    CANONICAL_TIMEPOINT_KEY,
    CANONICAL_FOV_KEY,
)
REQUIRED_CELL_TABLE_COLS: tuple[str, ...] = (
    "patient_id",
    CANONICAL_TIMEPOINT_KEY,
    CANONICAL_FOV_KEY,
    "x",
    "y",
    CANONICAL_CELL_SUBTYPE_KEY,
)
REQUIRED_FOV_TABLE_COLS: tuple[str, ...] = (
    "patient_id",
    CANONICAL_TIMEPOINT_KEY,
    CANONICAL_FOV_KEY,
    CANONICAL_DOMAIN_KEY,
)
REQUIRED_OBS_COLS: tuple[str, ...] = (
    "patient_id",
    CANONICAL_TIMEPOINT_KEY,
    CANONICAL_FOV_KEY,
    CANONICAL_DOMAIN_KEY,
    CANONICAL_CELL_SUBTYPE_KEY,
)
OPTIONAL_OBS_COLS: tuple[str, ...] = ("state_id", "block_id")

REQUIRED_OBSM_KEYS: tuple[str, ...] = ("spatial",)
CANONICAL_FEATURE_KEY = "local_state_features"
FEATURE_KEY_ALIASES: tuple[str, ...] = (CANONICAL_FEATURE_KEY, "community_features")
OPTIONAL_OBSM_KEYS: tuple[str, ...] = (CANONICAL_FEATURE_KEY,)
CANONICAL_STATE_ID_KEY = "state_id"
STATE_ID_ALIASES: tuple[str, ...] = (CANONICAL_STATE_ID_KEY, "proto_id")
CANONICAL_STATE_CENTROIDS_KEY = "state_centroids"
STATE_CENTROIDS_ALIASES: tuple[str, ...] = (CANONICAL_STATE_CENTROIDS_KEY, "prototype_centroids")
CANONICAL_COST_SCALE_KEY = "cost_scale"
COST_SCALE_ALIASES: tuple[str, ...] = (CANONICAL_COST_SCALE_KEY, "s_C", "global_cost_scale")
CANONICAL_STATE_FEATURE_METADATA_KEY = "state_feature_metadata"
OPTIONAL_UNS_KEYS: tuple[str, ...] = (
    CANONICAL_STATE_FEATURE_METADATA_KEY,
    "roi_areas",
    CANONICAL_COST_SCALE_KEY,
)
REQUIRED_UNS_KEYS: tuple[str, ...] = ()


def _require_named_columns(columns: Sequence[str], required: Sequence[str], *, where: str) -> None:
    missing = [name for name in required if name not in columns]
    if missing:
        raise ContractError(f"{where}: missing required columns: {missing}")


def _require_any_named_column(columns: Sequence[str], aliases: Sequence[str], *, where: str) -> str:
    for key in aliases:
        if key in columns:
            return key
    raise ContractError(f"{where}: missing required alias set {list(aliases)}")


def _require_mapping_keys(mapping: Mapping[str, Any], keys: Sequence[str], *, where: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ContractError(f"{where}: missing required keys: {missing}")


def _coerce_dataframe(table: Any, *, where: str) -> pd.DataFrame:
    if not isinstance(table, pd.DataFrame):
        raise ContractError(f"{where} must be a pandas.DataFrame")
    return table.copy()


def _series_equal(left: pd.Series, right: pd.Series) -> bool:
    left_norm = left.reset_index(drop=True)
    right_norm = right.reset_index(drop=True)
    return left_norm.equals(right_norm)


def _canonicalize_alias_column(
    frame: pd.DataFrame,
    *,
    canonical: str,
    aliases: Sequence[str],
    required: bool,
    where: str,
) -> pd.DataFrame:
    present = [alias for alias in aliases if alias in frame.columns]
    if not present:
        if required:
            raise ContractError(f"{where}: missing required alias set {list(aliases)}")
        return frame

    active = canonical if canonical in present else present[0]
    active_series = frame[active]
    for alias in present:
        if alias == active:
            continue
        if not _series_equal(active_series, frame[alias]):
            raise ContractError(
                f"{where}: conflicting values across alias columns {active!r} and {alias!r}"
            )

    renamed = frame.drop(columns=[alias for alias in present if alias != active])
    if active != canonical:
        renamed = renamed.rename(columns={active: canonical})
    return renamed


def _normalize_identifier_column(frame: pd.DataFrame, column: str, *, where: str) -> pd.Series:
    series = frame[column]
    if series.isna().any():
        raise ContractError(f"{where}: column {column!r} contains missing values")
    normalized = series.astype(str).str.strip()
    if (normalized == "").any():
        raise ContractError(f"{where}: column {column!r} contains empty string values")
    return normalized


def _normalize_cell_table(cell_table: pd.DataFrame, *, where: str) -> pd.DataFrame:
    frame = _coerce_dataframe(cell_table, where=where)
    if frame.shape[0] == 0:
        raise ContractError(f"{where} must contain at least one cell row")

    frame = _canonicalize_alias_column(
        frame,
        canonical=CANONICAL_TIMEPOINT_KEY,
        aliases=TIMEPOINT_KEY_ALIASES,
        required=True,
        where=where,
    )
    frame = _canonicalize_alias_column(
        frame,
        canonical=CANONICAL_FOV_KEY,
        aliases=FOV_KEY_ALIASES,
        required=True,
        where=where,
    )
    frame = _canonicalize_alias_column(
        frame,
        canonical=CANONICAL_CELL_SUBTYPE_KEY,
        aliases=CELL_SUBTYPE_KEY_ALIASES,
        required=True,
        where=where,
    )
    frame = _canonicalize_alias_column(
        frame,
        canonical=CANONICAL_DOMAIN_KEY,
        aliases=DOMAIN_KEY_ALIASES,
        required=False,
        where=where,
    )
    _require_named_columns(frame.columns, ("patient_id", "x", "y"), where=where)

    for column in ("patient_id", CANONICAL_TIMEPOINT_KEY, CANONICAL_FOV_KEY, CANONICAL_CELL_SUBTYPE_KEY):
        frame[column] = _normalize_identifier_column(frame, column, where=where)
    if CANONICAL_DOMAIN_KEY in frame.columns:
        frame[CANONICAL_DOMAIN_KEY] = _normalize_identifier_column(frame, CANONICAL_DOMAIN_KEY, where=where)

    coords = frame.loc[:, ["x", "y"]].to_numpy(dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ContractError(f"{where}: coordinates must have shape [n_cells, 2]")
    if not np.isfinite(coords).all():
        raise ContractError(f"{where}: coordinates contain NaN/Inf")
    frame["x"] = coords[:, 0]
    frame["y"] = coords[:, 1]

    return frame


def _normalize_fov_table(fov_table: pd.DataFrame, *, where: str) -> pd.DataFrame:
    frame = _coerce_dataframe(fov_table, where=where)
    if frame.shape[0] == 0:
        raise ContractError(f"{where} must contain at least one FOV metadata row")

    frame = _canonicalize_alias_column(
        frame,
        canonical=CANONICAL_TIMEPOINT_KEY,
        aliases=TIMEPOINT_KEY_ALIASES,
        required=True,
        where=where,
    )
    frame = _canonicalize_alias_column(
        frame,
        canonical=CANONICAL_FOV_KEY,
        aliases=FOV_KEY_ALIASES,
        required=True,
        where=where,
    )
    frame = _canonicalize_alias_column(
        frame,
        canonical=CANONICAL_DOMAIN_KEY,
        aliases=DOMAIN_KEY_ALIASES,
        required=True,
        where=where,
    )
    _require_named_columns(frame.columns, ("patient_id",), where=where)

    for column in (*LONGITUDINAL_GROUP_KEYS, CANONICAL_DOMAIN_KEY):
        frame[column] = _normalize_identifier_column(frame, column, where=where)

    if frame.duplicated(subset=LONGITUDINAL_GROUP_KEYS).any():
        dupes = frame.loc[frame.duplicated(subset=LONGITUDINAL_GROUP_KEYS), list(LONGITUDINAL_GROUP_KEYS)]
        raise ContractError(
            f"{where}: duplicate FOV metadata rows for keys {dupes.to_dict(orient='records')}"
        )

    return frame.loc[:, [*LONGITUDINAL_GROUP_KEYS, CANONICAL_DOMAIN_KEY]].copy()


def _derive_fov_metadata_from_cells(cell_frame: pd.DataFrame, *, where: str) -> pd.DataFrame:
    if CANONICAL_DOMAIN_KEY not in cell_frame.columns:
        raise ContractError(
            f"{where}: missing required alias set {list(DOMAIN_KEY_ALIASES)} "
            "or provide fov_table with domain metadata"
        )

    metadata = cell_frame.loc[:, [*LONGITUDINAL_GROUP_KEYS, CANONICAL_DOMAIN_KEY]].drop_duplicates()
    if metadata.duplicated(subset=LONGITUDINAL_GROUP_KEYS).any():
        raise ContractError(
            f"{where}: each patient/timepoint/fov group must map to exactly one domain_label"
        )
    return metadata.reset_index(drop=True)


def _merge_fov_metadata(cell_frame: pd.DataFrame, fov_frame: pd.DataFrame, *, where: str) -> pd.DataFrame:
    cell_keys = cell_frame.loc[:, LONGITUDINAL_GROUP_KEYS].drop_duplicates()
    metadata_keys = fov_frame.loc[:, LONGITUDINAL_GROUP_KEYS].drop_duplicates()

    extra_keys = metadata_keys.merge(cell_keys, on=list(LONGITUDINAL_GROUP_KEYS), how="left", indicator=True)
    extra_keys = extra_keys.loc[extra_keys["_merge"] == "left_only", list(LONGITUDINAL_GROUP_KEYS)]
    if not extra_keys.empty:
        raise ContractError(
            f"{where}: fov_table contains metadata-only empty FOVs: {extra_keys.to_dict(orient='records')}"
        )

    missing_keys = cell_keys.merge(metadata_keys, on=list(LONGITUDINAL_GROUP_KEYS), how="left", indicator=True)
    missing_keys = missing_keys.loc[missing_keys["_merge"] == "left_only", list(LONGITUDINAL_GROUP_KEYS)]
    if not missing_keys.empty:
        raise ContractError(
            f"{where}: fov_table is missing metadata for cell groups: {missing_keys.to_dict(orient='records')}"
        )

    merged = cell_frame.merge(
        fov_frame,
        on=list(LONGITUDINAL_GROUP_KEYS),
        how="left",
        suffixes=("", "_fov"),
        validate="many_to_one",
    )

    if CANONICAL_DOMAIN_KEY in cell_frame.columns:
        fov_domain_key = f"{CANONICAL_DOMAIN_KEY}_fov"
        if fov_domain_key not in merged.columns:
            raise ContractError(f"{where}: failed to merge normalized FOV metadata")
        cell_domain = merged[CANONICAL_DOMAIN_KEY]
        fov_domain = merged[fov_domain_key]
        if not _series_equal(cell_domain, fov_domain):
            raise ContractError(
                f"{where}: cell-level and FOV-level domain metadata disagree for at least one group"
            )
        merged = merged.drop(columns=[fov_domain_key])
    else:
        merged = merged.rename(columns={CANONICAL_DOMAIN_KEY: CANONICAL_DOMAIN_KEY})

    return merged


def assemble_longitudinal_adata(
    cell_table: pd.DataFrame,
    *,
    fov_table: pd.DataFrame | None = None,
) -> AnnData:
    """Assemble canonical STRIDE AnnData from cell and optional FOV metadata tables."""
    cell_frame = _normalize_cell_table(cell_table, where="cell_table")
    if fov_table is None:
        merged = cell_frame
        fov_metadata = _derive_fov_metadata_from_cells(cell_frame, where="cell_table")
    else:
        fov_metadata = _normalize_fov_table(fov_table, where="fov_table")
        merged = _merge_fov_metadata(cell_frame, fov_metadata, where="assemble_longitudinal_adata")

    if CANONICAL_DOMAIN_KEY not in merged.columns:
        raise ContractError("assemble_longitudinal_adata: failed to resolve canonical domain_label")

    obs = merged.loc[:, [*REQUIRED_OBS_COLS]].copy()
    spatial = merged.loc[:, ["x", "y"]].to_numpy(dtype=float)
    if spatial.shape[0] != obs.shape[0]:
        raise ContractError("assemble_longitudinal_adata: spatial rows must align to obs rows")

    adata = AnnData(
        X=np.zeros((obs.shape[0], 0), dtype=float),
        obs=obs,
    )
    adata.obsm["spatial"] = spatial
    adata.uns["fov_metadata"] = fov_metadata.to_dict(orient="records")
    validate_longitudinal_adata(adata)
    return adata


def resolve_timepoint_key(adata: AnnData) -> str:
    """Return the active timepoint column name on an AnnData object."""
    return _require_any_named_column(adata.obs.columns, TIMEPOINT_KEY_ALIASES, where="adata.obs")


def resolve_fov_key(adata: AnnData) -> str:
    """Return the active observation-unit column (`fov_id` or accepted `roi_id`)."""
    return _require_any_named_column(adata.obs.columns, FOV_KEY_ALIASES, where="adata.obs")


def resolve_domain_key(adata: AnnData) -> str:
    """Return the active domain-label column on an AnnData object."""
    return _require_any_named_column(adata.obs.columns, DOMAIN_KEY_ALIASES, where="adata.obs")


def resolve_cell_subtype_key(adata: AnnData) -> str:
    """Return the active cell-subtype label column on an AnnData object."""
    return _require_any_named_column(adata.obs.columns, CELL_SUBTYPE_KEY_ALIASES, where="adata.obs")


def resolve_feature_key(adata: AnnData) -> str:
    """Return the resolved local feature matrix key on an AnnData object."""
    for key in FEATURE_KEY_ALIASES:
        if key in adata.obsm:
            return key
    raise ContractError(f"adata.obsm: missing required alias set {list(FEATURE_KEY_ALIASES)}")


def resolve_state_id_key(adata: AnnData) -> str:
    """Return the active shared-state identifier column on an AnnData object."""
    return _require_any_named_column(adata.obs.columns, STATE_ID_ALIASES, where="adata.obs")


def _validate_group_consistency(adata: AnnData) -> None:
    obs = adata.obs.copy()
    timepoint_key = resolve_timepoint_key(adata)
    fov_key = resolve_fov_key(adata)
    domain_key = resolve_domain_key(adata)

    obs = obs.loc[:, ["patient_id", timepoint_key, fov_key, domain_key]].copy()
    obs.columns = ["patient_id", CANONICAL_TIMEPOINT_KEY, CANONICAL_FOV_KEY, CANONICAL_DOMAIN_KEY]
    grouped = (
        obs.groupby(list(LONGITUDINAL_GROUP_KEYS), sort=False, observed=True)[CANONICAL_DOMAIN_KEY]
        .nunique(dropna=False)
    )
    if (grouped > 1).any():
        bad_keys = grouped[grouped > 1].index.tolist()
        raise ContractError(
            "AnnData obs contains inconsistent domain_label values within a patient/timepoint/fov group: "
            f"{bad_keys}"
        )


def validate_longitudinal_adata(
    adata: AnnData,
    *,
    require_cell_type: bool = False,
    require_representation: bool = False,
    require_state_axis: bool = False,
    require_cost_scale: bool = False,
    require_cost_matrix: bool = False,
) -> None:
    """Validate one AnnData object against the canonical STRIDE longitudinal route."""
    if getattr(adata, "obs", None) is None:
        raise ContractError("AnnData object must define obs")
    if getattr(adata, "obsm", None) is None:
        raise ContractError("AnnData object must define obsm")
    if getattr(adata, "uns", None) is None:
        raise ContractError("AnnData object must define uns")
    if int(getattr(adata, "n_obs", 0)) <= 0:
        raise ContractError("AnnData object must contain at least one observation row")

    _require_named_columns(adata.obs.columns, ("patient_id",), where="adata.obs")
    resolve_timepoint_key(adata)
    resolve_fov_key(adata)
    resolve_domain_key(adata)

    subtype_key = resolve_cell_subtype_key(adata)
    if require_cell_type and subtype_key not in adata.obs.columns:
        raise ContractError(
            f"adata.obs: missing required alias set {list(CELL_SUBTYPE_KEY_ALIASES)}"
        )

    for column in ("patient_id", resolve_timepoint_key(adata), resolve_fov_key(adata), resolve_domain_key(adata), subtype_key):
        normalized = _normalize_identifier_column(adata.obs.copy(), column, where="adata.obs")
        if normalized.shape[0] != adata.n_obs:
            raise ContractError(f"adata.obs[{column!r}] failed normalization")

    _require_mapping_keys(adata.obsm, REQUIRED_OBSM_KEYS, where="adata.obsm")
    spatial = np.asarray(adata.obsm["spatial"], dtype=float)
    if spatial.ndim != 2 or spatial.shape[1] != 2:
        raise ContractError("adata.obsm['spatial'] must have shape [n, 2]")
    if spatial.shape[0] != adata.n_obs:
        raise ContractError("adata.obsm['spatial'] row count must align to adata.n_obs")
    if not np.isfinite(spatial).all():
        raise ContractError("adata.obsm['spatial'] contains NaN/Inf (must fail-fast)")

    _validate_group_consistency(adata)

    if require_representation:
        feature_key = resolve_feature_key(adata)
        feats = np.asarray(adata.obsm[feature_key], dtype=float)
        if feats.ndim != 2 or feats.shape[0] != spatial.shape[0]:
            raise ContractError(f"adata.obsm[{feature_key!r}] must have shape [n, d]")
        if not np.isfinite(feats).all():
            raise ContractError(f"adata.obsm[{feature_key!r}] contains NaN/Inf")

    if require_state_axis:
        resolve_state_id_key(adata)

    _require_mapping_keys(adata.uns, REQUIRED_UNS_KEYS, where="adata.uns")
    if require_cost_scale and not any(key in adata.uns for key in COST_SCALE_ALIASES):
        raise ContractError(
            f"adata.uns: missing cost scale key (expected one of {list(COST_SCALE_ALIASES)})"
        )

    if require_cost_matrix:
        if "cost_matrix" not in adata.uns:
            raise ContractError("adata.uns: missing required key 'cost_matrix'")
        try:
            cost_matrix = np.asarray(adata.uns["cost_matrix"], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ContractError("adata.uns['cost_matrix'] must be numeric / array-like") from exc

        if cost_matrix.ndim != 2:
            raise ContractError("adata.uns['cost_matrix'] must be 2D")
        if cost_matrix.shape[0] != cost_matrix.shape[1]:
            raise ContractError("adata.uns['cost_matrix'] must be square")
        if not np.isfinite(cost_matrix).all():
            raise ContractError("adata.uns['cost_matrix'] contains NaN/Inf")


def load_state_basis_from_adata(adata: AnnData) -> StateBasis:
    """Load a stored shared state basis from dataset-attached AnnData artifacts."""
    feature_key = resolve_feature_key(adata)
    state_key = resolve_state_id_key(adata)
    if "cost_matrix" not in adata.uns:
        raise ContractError("adata.uns: missing required key 'cost_matrix'")
    if CANONICAL_STATE_CENTROIDS_KEY in adata.uns:
        centroids_key = CANONICAL_STATE_CENTROIDS_KEY
    else:
        centroids_key = _require_any_named_column(
            tuple(adata.uns.keys()),
            STATE_CENTROIDS_ALIASES,
            where="adata.uns",
        )
    if not any(key in adata.uns for key in COST_SCALE_ALIASES):
        raise ContractError(
            f"adata.uns: missing cost scale key (expected one of {list(COST_SCALE_ALIASES)})"
        )

    metadata: dict[str, Any] = {}
    if CANONICAL_STATE_FEATURE_METADATA_KEY in adata.uns:
        metadata[CANONICAL_STATE_FEATURE_METADATA_KEY] = adata.uns[CANONICAL_STATE_FEATURE_METADATA_KEY]

    centroids = np.asarray(adata.uns[centroids_key], dtype=float)
    cost_matrix = np.asarray(adata.uns["cost_matrix"], dtype=float)
    resolved_cost_scale = next(float(adata.uns[key]) for key in COST_SCALE_ALIASES if key in adata.uns)
    state_ids = tuple(sorted(pd_unique_int(adata.obs[state_key])))
    return load_state_basis(
        centroids=centroids,
        cost_matrix=cost_matrix,
        cost_scale=resolved_cost_scale,
        feature_key=feature_key,
        state_key=state_key,
        state_ids=state_ids,
        metadata=metadata,
    )


def pd_unique_int(values: Any) -> tuple[int, ...]:
    """Return sorted unique integer identifiers from a pandas-like column."""
    arr = np.asarray(values, dtype=int).reshape(-1)
    return tuple(sorted(np.unique(arr).tolist()))


__all__ = [
    "AnnData",
    "CANONICAL_CELL_SUBTYPE_KEY",
    "CANONICAL_COST_SCALE_KEY",
    "CANONICAL_DOMAIN_KEY",
    "CANONICAL_FEATURE_KEY",
    "CANONICAL_FOV_KEY",
    "CANONICAL_STATE_CENTROIDS_KEY",
    "CANONICAL_STATE_FEATURE_METADATA_KEY",
    "CANONICAL_STATE_ID_KEY",
    "CANONICAL_TIMEPOINT_KEY",
    "CELL_SUBTYPE_KEY_ALIASES",
    "COST_SCALE_ALIASES",
    "DOMAIN_KEY_ALIASES",
    "FEATURE_KEY_ALIASES",
    "FOV_KEY_ALIASES",
    "LONGITUDINAL_GROUP_KEYS",
    "OPTIONAL_OBS_COLS",
    "OPTIONAL_OBSM_KEYS",
    "OPTIONAL_UNS_KEYS",
    "REQUIRED_CELL_TABLE_COLS",
    "REQUIRED_FOV_TABLE_COLS",
    "REQUIRED_OBS_COLS",
    "REQUIRED_OBSM_KEYS",
    "REQUIRED_UNS_KEYS",
    "STATE_CENTROIDS_ALIASES",
    "STATE_ID_ALIASES",
    "TIMEPOINT_KEY_ALIASES",
    "assemble_longitudinal_adata",
    "load_state_basis_from_adata",
    "resolve_cell_subtype_key",
    "resolve_domain_key",
    "resolve_feature_key",
    "resolve_fov_key",
    "resolve_state_id_key",
    "resolve_timepoint_key",
    "validate_longitudinal_adata",
]
