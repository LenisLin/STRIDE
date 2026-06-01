"""Public AnnData assembly wrapper for STRIDE .io."""
from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from stride._schema import (
    ALLOWED_INPUT_MASS_MODES,
    OBS_CELL_TYPE_KEY,
    OBS_DOMAIN_KEY,
    OBS_FOV_KEY,
    OBS_PATIENT_KEY,
    OBS_TIMEPOINT_KEY,
    OBSM_SPATIAL_KEY,
    STRIDE_CONFIG_KEY,
    STRIDE_FOV_METADATA_KEY,
    STRIDE_UNS_KEY,
)
from stride.errors import ContractError

from ._validation import (
    _as_identifier,
    _as_identifier_column,
    _as_matrix,
    _as_spatial,
    _as_var_names,
    validate_raw_adata,
)


def build_adata(
    *,
    # 1. Matrix input: user-loaded expression/feature matrix and feature metadata.
    X: object,
    var: Sequence[object],

    # 2. Metadata input: user-loaded cell-level and optional FOV-level tables.
    cell: pd.DataFrame,
    fov: pd.DataFrame | None = None,

    # 3. Column mapping input: column names in user-provided metadata tables.
    cell_id: str | None = None,
    patient: str,
    time: str,
    fov_id: str,
    domain: str,
    cell_type: str,
    x: str,
    y: str,
    area: str | None = None,

    # 4. STRIDE config input: caller-resolved, task-agnostic declarations.
    source: object,
    target: object,
    time_order: Sequence[object],
    mass_mode: Literal["fraction", "density"],
    n_states: int,
    k_neighbors: int,
) -> AnnData:
    """Assemble a raw STRIDE AnnData object from caller-loaded tables.

    Parameters:
        X: Numeric cell-by-feature expression or feature matrix. Sparse inputs
            are densified at this .io boundary.
        var: Sequence of feature names aligned to ``X`` columns.
        cell: Cell-level metadata table aligned row-for-row to ``X``.
        fov: Optional FOV-level metadata table. Required for
            ``mass_mode="density"`` and allowed to contain unused FOV rows.
        cell_id: Column in ``cell`` used for ``adata.obs_names``. When omitted,
            the first cell table column is used.
        patient, time, fov_id, domain, cell_type: Column names in ``cell`` and,
            where applicable, ``fov`` that map to canonical STRIDE metadata.
        x, y: Column names in ``cell`` used to populate
            ``adata.obsm["spatial"]``.
        area: FOV-level area column in ``fov``; required for density inputs.
        source, target, time_order: Caller-resolved ordered relation
            declarations stored without task-specific inference.
        mass_mode: Input mass declaration, limited to ``"fraction"`` or
            ``"density"`` at the ``stride.io`` boundary.
        n_states: Declared shared-state count for downstream preparation.
        k_neighbors: Declared neighborhood size for downstream preparation.

    Returns:
        AnnData with dense ``X``, canonical ``obs``, ``var``,
        ``obsm["spatial"]``, and flat STRIDE declarations under
        ``uns["stride"]``.
    """
    # ---- Expression and feature metadata ----
    dense_X = _as_matrix(X)
    var_frame = _as_var_names(var, n_vars=dense_X.shape[1])

    # ---- Cell metadata ----
    if not isinstance(cell, pd.DataFrame):
        raise ContractError("cell must be a pandas.DataFrame")
    if cell.shape[0] == 0:
        raise ContractError("cell must contain at least one row")
    if cell.shape[1] == 0:
        raise ContractError("cell must contain at least one column")
    if cell.columns.has_duplicates:
        raise ContractError("cell must not contain duplicate column names")
    cell_table = cell.copy()
    if dense_X.shape[0] != cell_table.shape[0]:
        raise ContractError(
            f"X row count {dense_X.shape[0]} does not match cell rows {cell_table.shape[0]}"
        )

    patient = _as_identifier(patient, name="patient column")
    time = _as_identifier(time, name="time column")
    fov_id = _as_identifier(fov_id, name="fov_id column")
    domain = _as_identifier(domain, name="domain column")
    cell_type = _as_identifier(cell_type, name="cell_type column")
    x = _as_identifier(x, name="x column")
    y = _as_identifier(y, name="y column")
    cell_id_column = _as_identifier(
        cell_id if cell_id is not None else cell_table.columns[0],
        name="cell_id column",
    )

    required_cell_columns = (cell_id_column, patient, time, fov_id, domain, cell_type, x, y)
    missing_cell_columns = [column for column in required_cell_columns if column not in cell_table.columns]
    if missing_cell_columns:
        raise ContractError(f"cell: missing required columns: {missing_cell_columns}")

    # Use caller-declared cell identifiers as obs_names before AnnData construction.
    obs_names = pd.Index(
        _as_identifier_column(
            cell_table[cell_id_column], where="cell", column=cell_id_column
        ).to_numpy(),
        dtype="object",
        name=None,
    )
    if obs_names.has_duplicates:
        duplicates = sorted(obs_names[obs_names.duplicated()].unique().tolist())
        raise ContractError(f"cell: cell_id column contains duplicate values: {duplicates}")
    cell_table.index = obs_names
    cell_table.index.name = None

    for column in (patient, time, fov_id, domain, cell_type):
        cell_table[column] = _as_identifier_column(
            cell_table[column], where="cell", column=column
        )

    spatial_coordinates = _as_spatial(cell_table, x=x, y=y, where="cell")

    # ---- FOV metadata ----
    if mass_mode not in ALLOWED_INPUT_MASS_MODES:
        raise ContractError(
            f"mass_mode must be one of {list(ALLOWED_INPUT_MASS_MODES)}, got {mass_mode!r}"
        )
    area_column = _as_identifier(area, name="area column") if area is not None else None
    if mass_mode == "density" and area_column is None:
        raise ContractError("area is required when mass_mode == 'density'")

    if fov is None:
        if mass_mode == "density":
            raise ContractError("fov is required when mass_mode == 'density'")
        fov_rows = _derive_fov_rows_from_cell(cell_table, keys=(patient, time, fov_id, domain))
    else:
        if not isinstance(fov, pd.DataFrame):
            raise ContractError("fov must be a pandas.DataFrame")
        if fov.shape[0] == 0:
            raise ContractError("fov must contain at least one row")
        if fov.columns.has_duplicates:
            raise ContractError("fov must not contain duplicate column names")
        fov_table = fov.copy()

        fov_columns = [patient, time, fov_id, domain]
        if area_column is not None:
            fov_columns.append(area_column)
        missing_fov_columns = [column for column in fov_columns if column not in fov_table.columns]
        if missing_fov_columns:
            raise ContractError(f"fov: missing required columns: {missing_fov_columns}")

        for column in (patient, time, fov_id, domain):
            fov_table[column] = _as_identifier_column(
                fov_table[column], where="fov", column=column
            )

        # The fov table may contain unused rows; only rows linked to cell keys are materialized.
        fov_rows = _select_fov_rows(
            cell_table,
            fov_table,
            keys=(patient, time, fov_id, domain),
            area=area_column if mass_mode == "density" else None,
        )

    canonical_fov_metadata = _build_fov_metadata(
        cell_table,
        fov_rows,
        keys=(patient, time, fov_id, domain),
        area=area_column if mass_mode == "density" else None,
    )

    # ---- STRIDE config declarations ----
    for name, value in (("n_states", n_states), ("k_neighbors", k_neighbors)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ContractError(f"{name} must be a positive integer")

    if isinstance(time_order, str):
        raise ContractError("time_order must be a sequence of timepoint identifiers, not a string")
    try:
        normalized_time_order = tuple(
            _as_identifier(value, name="time_order value") for value in time_order
        )
    except TypeError as exc:
        raise ContractError("time_order must be a sequence of timepoint identifiers") from exc
    if not normalized_time_order:
        raise ContractError("time_order must contain at least one timepoint")
    if len(set(normalized_time_order)) != len(normalized_time_order):
        raise ContractError("time_order contains duplicate values")

    normalized_source = _as_identifier(source, name="source")
    normalized_target = _as_identifier(target, name="target")
    if normalized_source == normalized_target:
        raise ContractError("source and target must be distinct")
    if normalized_source not in normalized_time_order:
        raise ContractError("source must be present in time_order")
    if normalized_target not in normalized_time_order:
        raise ContractError("target must be present in time_order")

    unselected_times = sorted(
        set(cell_table[time].unique()) - {normalized_source, normalized_target}
    )
    if unselected_times:
        warnings.warn(
            "cell: time labels not selected by source/target pair are retained in AnnData: "
            f"{unselected_times}",
            UserWarning,
            stacklevel=2,
        )

    # ---- AnnData assembly ----
    canonical_obs = _build_obs(
        cell_table,
        mapped_columns=(patient, time, fov_id, domain, cell_type, x, y),
    )

    adata = AnnData(X=dense_X, obs=canonical_obs, var=var_frame)
    # AnnData provides the supported uniqueness policy for duplicate var names.
    adata.var_names_make_unique()
    adata.obsm[OBSM_SPATIAL_KEY] = spatial_coordinates
    adata.uns[STRIDE_UNS_KEY] = {
        STRIDE_CONFIG_KEY: {
            "source": normalized_source,
            "target": normalized_target,
            "time_order": list(normalized_time_order),
            "mass_mode": mass_mode,
            "n_states": n_states,
            "k_neighbors": k_neighbors,
        },
        # Keep FOV metadata tabular so h5ad round-trips mixed string/numeric columns.
        STRIDE_FOV_METADATA_KEY: canonical_fov_metadata,
    }

    validate_raw_adata(adata)
    return adata


def _derive_fov_rows_from_cell(
    cell_table: pd.DataFrame,
    *,
    keys: tuple[str, str, str, str],
) -> pd.DataFrame:
    """Derive used FOV rows from normalized cell metadata and check domains."""
    patient, time, fov_id, domain = keys
    fov_metadata = cell_table.loc[:, [patient, time, fov_id, domain]].drop_duplicates()
    duplicates = fov_metadata.duplicated(subset=[patient, time, fov_id], keep=False)
    if duplicates.any():
        bad_keys = fov_metadata.loc[duplicates, [patient, time, fov_id]].to_dict(orient="records")
        raise ContractError(
            "cell: each patient/time/FOV group must map to exactly one domain value; "
            f"conflicting groups: {bad_keys}"
        )
    return fov_metadata.reset_index(drop=True)


def _select_fov_rows(
    cell_table: pd.DataFrame,
    fov_table: pd.DataFrame,
    *,
    keys: tuple[str, str, str, str],
    area: str | None,
) -> pd.DataFrame:
    """Select used FOV rows from an input fov table and check row-level fields."""
    patient, time, fov_id, domain = keys
    key_columns = [patient, time, fov_id]
    cell_keys = cell_table.loc[:, key_columns].drop_duplicates()

    missing_keys = cell_keys.merge(
        fov_table.loc[:, key_columns].drop_duplicates(),
        on=key_columns,
        how="left",
        indicator=True,
    )
    missing_keys = missing_keys.loc[missing_keys["_merge"] == "left_only", key_columns]
    if not missing_keys.empty:
        raise ContractError(
            f"fov: missing metadata for cell groups: {missing_keys.to_dict(orient='records')}"
        )

    # Extra FOV rows are allowed; only rows linked to observed cells enter STRIDE metadata.
    used_fov = fov_table.merge(cell_keys, on=key_columns, how="inner")
    duplicated = used_fov.duplicated(subset=key_columns, keep=False)
    if duplicated.any():
        bad_keys = used_fov.loc[duplicated, key_columns].to_dict(orient="records")
        raise ContractError(f"fov: duplicate metadata rows for patient/time/FOV keys: {bad_keys}")

    if area is not None:
        try:
            numeric_area = pd.to_numeric(used_fov[area], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ContractError("fov: area column must be numeric when mass_mode == 'density'") from exc
        if not np.isfinite(numeric_area).all() or (numeric_area <= 0).any():
            raise ContractError("fov: area values must be finite positive numbers")
        used_fov = used_fov.copy()
        used_fov[area] = numeric_area

    return used_fov.reset_index(drop=True)


def _build_fov_metadata(
    cell_table: pd.DataFrame,
    fov_rows: pd.DataFrame,
    *,
    keys: tuple[str, str, str, str],
    area: str | None,
) -> pd.DataFrame:
    """Build canonical FOV metadata after checking cell/fov domain alignment."""
    patient, time, fov_id, domain = keys
    key_columns = [patient, time, fov_id]

    domain_metadata = fov_rows.loc[:, [*key_columns, domain]].copy()
    merged = cell_table.merge(
        domain_metadata,
        on=key_columns,
        how="left",
        suffixes=("", "_fov"),
        validate="many_to_one",
    )
    fov_domain = f"{domain}_fov"
    if fov_domain not in merged.columns:
        raise ContractError("build_adata: failed to merge FOV domain metadata")
    if not merged[domain].reset_index(drop=True).equals(merged[fov_domain].reset_index(drop=True)):
        raise ContractError("cell and fov domain metadata disagree for at least one cell group")

    required_columns = [patient, time, fov_id, domain]
    if area is not None:
        required_columns.append(area)

    canonical_columns = {
        OBS_PATIENT_KEY,
        OBS_TIMEPOINT_KEY,
        OBS_FOV_KEY,
        OBS_DOMAIN_KEY,
    }
    if area is not None:
        canonical_columns.add("area")

    extra_fov = fov_rows.drop(columns=required_columns)
    conflicts = [column for column in extra_fov.columns if column in canonical_columns]
    if conflicts:
        raise ContractError(
            f"fov: extra metadata columns conflict with canonical fov metadata columns: {conflicts}"
        )

    canonical_fov_metadata = pd.DataFrame(
        {
            OBS_PATIENT_KEY: fov_rows[patient].to_numpy(),
            OBS_TIMEPOINT_KEY: fov_rows[time].to_numpy(),
            OBS_FOV_KEY: fov_rows[fov_id].to_numpy(),
            OBS_DOMAIN_KEY: fov_rows[domain].to_numpy(),
        }
    )
    if area is not None:
        canonical_fov_metadata["area"] = fov_rows[area].to_numpy(dtype=float)
    if not extra_fov.empty:
        canonical_fov_metadata = pd.concat(
            [canonical_fov_metadata, extra_fov.reset_index(drop=True)],
            axis=1,
        )
    return canonical_fov_metadata


def _build_obs(
    cell_table: pd.DataFrame,
    *,
    mapped_columns: tuple[str, str, str, str, str, str, str],
) -> pd.DataFrame:
    """Build canonical obs fields while retaining unmapped cell metadata."""
    patient, time, fov_id, domain, cell_type, x, y = mapped_columns
    canonical_columns = {
        OBS_PATIENT_KEY,
        OBS_TIMEPOINT_KEY,
        OBS_FOV_KEY,
        OBS_DOMAIN_KEY,
        OBS_CELL_TYPE_KEY,
    }
    drop_columns = {patient, time, fov_id, domain, cell_type, x, y}
    extra_obs = cell_table.drop(columns=[column for column in drop_columns if column in cell_table.columns])
    conflicts = [column for column in extra_obs.columns if column in canonical_columns]
    if conflicts:
        raise ContractError(
            f"cell: extra metadata columns conflict with canonical obs columns: {conflicts}"
        )

    canonical_obs = pd.DataFrame(
        {
            OBS_PATIENT_KEY: cell_table[patient].to_numpy(),
            OBS_TIMEPOINT_KEY: cell_table[time].to_numpy(),
            OBS_FOV_KEY: cell_table[fov_id].to_numpy(),
            OBS_DOMAIN_KEY: cell_table[domain].to_numpy(),
            OBS_CELL_TYPE_KEY: cell_table[cell_type].to_numpy(),
        },
        index=pd.Index(cell_table.index.astype(str), dtype="object", name=None),
    )
    if not extra_obs.empty:
        canonical_obs = pd.concat([canonical_obs, extra_obs], axis=1)
    return canonical_obs
