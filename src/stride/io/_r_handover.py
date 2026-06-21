"""CSV handover helpers for downstream R visualization workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

from stride._schema import (
    OBS_CELL_TYPE_KEY,
    OBS_DOMAIN_KEY,
    OBS_FOV_KEY,
    OBS_PATIENT_KEY,
    OBS_STATE_ID_KEY,
    OBS_TIMEPOINT_KEY,
    STRIDE_CONFIG_KEY,
    STRIDE_FOV_OBSERVATIONS_KEY,
    STRIDE_UNS_KEY,
)
from stride.errors import ContractError
from stride.tl import CohortResult, FitResult, RelationResult


def write_r_handover(
    table: pd.DataFrame,
    output_dir: str | Path,
    filename: str,
    primary_key: str | Sequence[str],
) -> Path:
    """Write one explicit CSV handover table for downstream R plotting.

    Purpose:
        Provide the formal Python-to-R bridge required by the repository
        constraint contract without making plotting functions write files as a
        side effect.

    Interface:
        `table` is the complete plotting table to write. `output_dir`,
        `filename`, and `primary_key` are caller supplied; this helper does not
        discover config files, YAML manifests, or output conventions.

    Boundary:
        This function writes one CSV. It does not compute statistics, generate
        audit matrices, infer color palettes, mutate scientific result objects,
        or call R.
    """
    if not isinstance(table, pd.DataFrame):
        raise ContractError("table must be a pandas DataFrame")
    keys = _normalize_primary_key(primary_key)
    missing = [column for column in keys if column not in table.columns]
    if missing:
        raise ContractError("primary_key columns are missing from table: " + ", ".join(missing))
    if table.loc[:, list(keys)].isna().any(axis=None):
        raise ContractError("primary_key columns must not contain missing values")
    path = Path(output_dir) / str(filename)
    if path.suffix.lower() != ".csv":
        raise ContractError("R handover filename must end with .csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def write_descriptive_tables(
    adata: AnnData,
    output_dir: str | Path,
    *,
    state_key: str = OBS_STATE_ID_KEY,
    cell_type_key: str = OBS_CELL_TYPE_KEY,
    patient_key: str = OBS_PATIENT_KEY,
    fov_key: str = OBS_FOV_KEY,
    domain_key: str = OBS_DOMAIN_KEY,
    time_key: str = OBS_TIMEPOINT_KEY,
    state_labels: Sequence[str] | None = None,
    cell_type_order: Sequence[str] | None = None,
    patient_groups: Mapping[str, str] | pd.Series | pd.DataFrame | None = None,
    group_key: str = "group",
) -> tuple[Path, Path]:
    """Write descriptive community-annotation and FOV-composition tables.

    Outputs:
        `community_annotation.csv` contains one row per
        community-by-cell-subtype fraction with dominant domain/timepoint and
        prevalence context. `fov_composition.csv` contains one row per
        FOV-by-community fraction with aligned metadata.

    Boundary:
        These tables are plotting handover payloads. The function does not
        cluster communities, test abundance, or save figure-specific legends.
    """
    if not isinstance(adata, AnnData):
        raise ContractError("adata must be an AnnData object")
    _require_obs_columns(adata, [state_key, cell_type_key, patient_key, fov_key, domain_key])
    obs = adata.obs.copy()
    state_ids = _state_ids(obs[state_key], n_obs=adata.n_obs)
    n_states = _resolve_n_states(adata, state_ids=state_ids)
    state_ids = _state_ids(obs[state_key], n_states=n_states, n_obs=adata.n_obs)
    labels = _state_labels(n_states, state_labels)
    obs["_stride_handover_state_id"] = state_ids

    if cell_type_order is None:
        cell_types = tuple(sorted(obs[cell_type_key].astype(str).unique()))
    else:
        cell_types = tuple(str(value) for value in cell_type_order)
    observed_types = set(obs[cell_type_key].astype(str))
    missing_types = sorted(observed_types.difference(cell_types))
    if missing_types:
        raise ContractError(
            "cell_type_order is missing observed cell subtype labels: " + ", ".join(missing_types)
        )

    cell_counts = pd.crosstab(obs["_stride_handover_state_id"], obs[cell_type_key].astype(str))
    cell_counts = cell_counts.reindex(index=range(n_states), columns=cell_types, fill_value=0)
    cell_fraction = _row_fraction(cell_counts.to_numpy(dtype=float))
    domain_fraction = _state_category_fraction(
        obs, n_states, "_stride_handover_state_id", domain_key
    )
    time_fraction = _state_category_fraction(obs, n_states, "_stride_handover_state_id", time_key)
    patient_prevalence = _state_prevalence(
        obs,
        n_states,
        "_stride_handover_state_id",
        [patient_key],
    )
    fov_identity = [patient_key, fov_key]
    if time_key in obs:
        fov_identity = [patient_key, time_key, fov_key]
    fov_prevalence = _state_prevalence(obs, n_states, "_stride_handover_state_id", fov_identity)

    annotation_rows = []
    for state_id in range(n_states):
        for cell_type_index, cell_type in enumerate(cell_types):
            annotation_rows.append(
                {
                    "community_id": state_id,
                    "community_label": labels[state_id],
                    "cell_subtype": str(cell_type),
                    "cell_fraction": float(cell_fraction[state_id, cell_type_index]),
                    "dominant_domain": str(domain_fraction.iloc[state_id].idxmax()),
                    "dominant_timepoint": str(time_fraction.iloc[state_id].idxmax()),
                    "patient_prevalence": float(patient_prevalence[state_id]),
                    "fov_prevalence": float(fov_prevalence[state_id]),
                }
            )
    community_annotation = pd.DataFrame(annotation_rows)

    matrix, metadata = _read_fov_observations(adata)
    fov_groups = _normalize_patient_groups(
        patient_groups,
        patient_ids=metadata[OBS_PATIENT_KEY].astype(str).tolist(),
        group_key=group_key,
    )
    fov_rows = []
    for fov_index, meta_row in metadata.reset_index(drop=True).iterrows():
        patient_id = str(meta_row[OBS_PATIENT_KEY])
        for state_id in range(matrix.shape[1]):
            fov_rows.append(
                {
                    "fov_index": int(fov_index),
                    "fov_id": str(meta_row[OBS_FOV_KEY]),
                    "patient_id": patient_id,
                    "timepoint": str(meta_row[OBS_TIMEPOINT_KEY]),
                    "domain_label": str(meta_row[OBS_DOMAIN_KEY]),
                    "group": _lookup_group(fov_groups, patient_id),
                    "community_id": state_id,
                    "community_label": labels[state_id],
                    "fraction": float(matrix[fov_index, state_id]),
                }
            )
    fov_composition = pd.DataFrame(fov_rows)

    first = write_r_handover(
        community_annotation,
        output_dir,
        "community_annotation.csv",
        primary_key=("community_id", "cell_subtype"),
    )
    second = write_r_handover(
        fov_composition,
        output_dir,
        "fov_composition.csv",
        primary_key=("fov_index", "community_id"),
    )
    return first, second


def write_fraction_table(
    table: pd.DataFrame,
    output_dir: str | Path,
    *,
    filename: str = "community_fraction_comparison.csv",
) -> Path:
    """Write one all-community descriptive fraction comparison table."""
    required = {
        "scale",
        "relation_id",
        "group",
        "patient_id",
        "side",
        "community_id",
        "community_label",
        "fraction",
        "p_value",
        "q_value",
        "test_name",
        "correction_method",
    }
    data = _require_table_columns(table, required, "community fraction handover table")
    data = data[
        [
            "scale",
            "relation_id",
            "group",
            "patient_id",
            "side",
            "community_id",
            "community_label",
            "fraction",
            "p_value",
            "q_value",
            "test_name",
            "correction_method",
        ]
    ].copy()
    return write_r_handover(
        data,
        output_dir,
        filename,
        primary_key=("scale", "relation_id", "group", "patient_id", "side", "community_id"),
    )


def write_cohort_table(
    result: CohortResult | RelationResult | FitResult,
    output_dir: str | Path,
    *,
    state_labels: Sequence[str] | None = None,
    filename: str = "cohort_relation.csv",
) -> Path:
    """Write one all-relation cohort template table for R visualization."""
    cohorts = _resolve_cohorts(result)
    n_states = _common_cohort_n_states(cohorts)
    labels = _state_labels(n_states, state_labels)
    rows = []
    for cohort in cohorts:
        A = np.asarray(cohort.template_A, dtype=float)
        d = np.asarray(cohort.template_d, dtype=float)
        e = np.asarray(cohort.template_e, dtype=float)
        for row_id in range(n_states + 1):
            for col_id in range(n_states + 1):
                masked = row_id == n_states and col_id == n_states
                if row_id < n_states and col_id < n_states:
                    value = float(A[row_id, col_id])
                    entry_type = "A"
                elif row_id < n_states and col_id == n_states:
                    value = float(d[row_id])
                    entry_type = "d"
                elif row_id == n_states and col_id < n_states:
                    value = float(e[col_id])
                    entry_type = "e"
                else:
                    value = np.nan
                    entry_type = "masked"
                rows.append(
                    {
                        "relation_id": str(cohort.relation_id),
                        "row_id": row_id,
                        "row_label": labels[row_id] if row_id < n_states else "target_open_e",
                        "col_id": col_id,
                        "col_label": labels[col_id] if col_id < n_states else "source_open_d",
                        "entry_type": entry_type,
                        "value": value,
                        "masked": bool(masked),
                        "support_n_patients": int(cohort.support_n_patients),
                        "dispersion": float(cohort.dispersion),
                    }
                )
    table = pd.DataFrame(rows)
    return write_r_handover(
        table,
        output_dir,
        filename,
        primary_key=("relation_id", "row_id", "col_id"),
    )


def write_program_score_table(
    patient_program_scores: pd.DataFrame,
    output_dir: str | Path,
    *,
    program_stats: pd.DataFrame | None = None,
    filename: str = "relation_program_scores.csv",
) -> Path:
    """Write patient program scores with supplied association metadata."""
    required_scores = {
        "relation_id",
        "program_id",
        "patient_id",
        "group_id",
        "program_component_score",
    }
    scores = _require_table_columns(
        patient_program_scores, required_scores, "patient program scores"
    )
    stat_columns = [
        "comparison_id",
        "comparison_type",
        "test_name",
        "effect_size",
        "effect_size_type",
        "p_value",
        "q_value",
    ]
    if program_stats is None:
        table = scores.copy()
        for column in stat_columns:
            table[column] = pd.NA
    else:
        stats = _require_table_columns(
            program_stats,
            {"relation_id", "program_id", *stat_columns},
            "program association stats",
        )
        table = scores.merge(
            stats[["relation_id", "program_id", *stat_columns]],
            on=["relation_id", "program_id"],
            how="left",
        )
    table = table[
        [
            "relation_id",
            "program_id",
            "patient_id",
            "group_id",
            "program_component_score",
            *stat_columns,
        ]
    ].copy()
    return write_r_handover(
        table,
        output_dir,
        filename,
        primary_key=("relation_id", "program_id", "patient_id", "group_id"),
    )


def _normalize_primary_key(primary_key: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(primary_key, str):
        keys: tuple[str, ...] = (primary_key,)
    else:
        keys = tuple(str(value) for value in primary_key)
    if not keys:
        raise ContractError("primary_key must contain at least one column")
    return keys


def _require_table_columns(table: pd.DataFrame, columns: set[str], name: str) -> pd.DataFrame:
    if not isinstance(table, pd.DataFrame):
        raise ContractError(f"{name} must be a pandas DataFrame")
    missing = sorted(columns.difference(table.columns))
    if missing:
        raise ContractError(f"{name} is missing required columns: " + ", ".join(missing))
    return table.copy()


def _require_obs_columns(adata: AnnData, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in adata.obs]
    if missing:
        raise ContractError("adata.obs is missing required columns: " + ", ".join(missing))


def _state_ids(
    values: object, *, n_states: int | None = None, n_obs: int | None = None
) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ContractError("state ids must be numeric") from exc
    if arr.ndim != 1:
        raise ContractError("state ids must be a 1D array")
    if n_obs is not None and arr.shape[0] != n_obs:
        raise ContractError("state ids length must align to adata.n_obs")
    if not np.all(np.isfinite(arr)):
        raise ContractError("state ids must be finite")
    if not np.all(np.equal(arr, np.floor(arr))):
        raise ContractError("state ids must be integer-compatible")
    ids = arr.astype(int)
    if np.any(ids < 0):
        raise ContractError("state ids must be nonnegative")
    if n_states is not None and np.any(ids >= n_states):
        raise ContractError("state ids values must be in [0, n_states - 1]")
    return ids


def _resolve_n_states(adata: AnnData, *, state_ids: np.ndarray) -> int:
    stride_uns = adata.uns.get(STRIDE_UNS_KEY)
    if isinstance(stride_uns, Mapping):
        config = stride_uns.get(STRIDE_CONFIG_KEY)
        if isinstance(config, Mapping) and "n_states" in config:
            value = config["n_states"]
            if (
                not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_))
                or int(value) <= 0
            ):
                raise ContractError("config['n_states'] must be a positive integer")
            return int(value)
    if state_ids.size == 0:
        raise ContractError("cannot infer n_states from empty state ids")
    return int(state_ids.max()) + 1


def _state_labels(n_states: int, state_labels: Sequence[str] | None) -> tuple[str, ...]:
    if state_labels is None:
        return tuple(f"C{index}" for index in range(n_states))
    labels = tuple(str(label) for label in state_labels)
    if len(labels) != n_states:
        raise ContractError("state_labels length must match n_states")
    return labels


def _row_fraction(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)


def _state_category_fraction(
    obs: pd.DataFrame,
    n_states: int,
    state_column: str,
    category_column: str,
) -> pd.DataFrame:
    if category_column not in obs:
        return pd.DataFrame({"unavailable": np.ones(n_states, dtype=float)})
    table = pd.crosstab(obs[state_column], obs[category_column].astype(str))
    table = table.reindex(index=range(n_states), fill_value=0)
    fractions = _row_fraction(table.to_numpy(dtype=float))
    return pd.DataFrame(fractions, index=table.index, columns=table.columns)


def _state_prevalence(
    obs: pd.DataFrame,
    n_states: int,
    state_column: str,
    identity_columns: Sequence[str],
) -> np.ndarray:
    identity = obs.loc[:, list(identity_columns)].astype(str).agg("|".join, axis=1)
    total = max(int(identity.nunique()), 1)
    values = np.zeros(n_states, dtype=float)
    for state_id in range(n_states):
        values[state_id] = float(identity.loc[obs[state_column] == state_id].nunique()) / float(
            total
        )
    return values


def _read_fov_observations(adata: AnnData) -> tuple[np.ndarray, pd.DataFrame]:
    stride_uns = adata.uns.get(STRIDE_UNS_KEY)
    if not isinstance(stride_uns, Mapping):
        raise ContractError("adata.uns['stride'] must be a mapping")
    fov_slot = stride_uns.get(STRIDE_FOV_OBSERVATIONS_KEY)
    if not isinstance(fov_slot, Mapping):
        raise ContractError("adata.uns['stride']['fov_observations'] must be a mapping")
    matrix = np.asarray(fov_slot.get("community_composition"), dtype=float)
    metadata = fov_slot.get("metadata")
    if not isinstance(metadata, pd.DataFrame):
        raise ContractError("FOV observation metadata must be a pandas DataFrame")
    required = [OBS_PATIENT_KEY, OBS_TIMEPOINT_KEY, OBS_FOV_KEY, OBS_DOMAIN_KEY]
    missing = [column for column in required if column not in metadata]
    if missing:
        raise ContractError("FOV metadata is missing required columns: " + ", ".join(missing))
    if matrix.ndim != 2:
        raise ContractError("FOV community composition must have shape [n_fov, K]")
    if matrix.shape[0] != metadata.shape[0]:
        raise ContractError("FOV community composition row count must match metadata")
    if not np.isfinite(matrix).all() or (matrix < 0.0).any():
        raise ContractError("FOV community composition must be finite and nonnegative")
    return matrix.copy(), metadata.copy()


def _normalize_patient_groups(
    groups: Mapping[str, str] | pd.Series | pd.DataFrame | None,
    *,
    patient_ids: Sequence[str],
    group_key: str,
) -> pd.Series | None:
    if groups is None:
        return None
    if isinstance(groups, pd.DataFrame):
        if OBS_PATIENT_KEY not in groups or group_key not in groups:
            raise ContractError("patient group DataFrame must contain patient_id and group columns")
        series = groups.set_index(OBS_PATIENT_KEY)[group_key]
    elif isinstance(groups, pd.Series):
        series = groups
    else:
        series = pd.Series(dict(groups), dtype=object)
    series.index = series.index.astype(str)
    missing = sorted(set(str(value) for value in patient_ids).difference(series.index.astype(str)))
    if missing:
        raise ContractError("patient_groups is missing patient ids: " + ", ".join(missing))
    return series.astype(str)


def _lookup_group(groups: pd.Series | None, patient_id: str) -> str:
    if groups is None:
        return "unassigned"
    return str(groups.loc[str(patient_id)])


def _resolve_cohorts(result: CohortResult | RelationResult | FitResult) -> tuple[CohortResult, ...]:
    if isinstance(result, CohortResult):
        return (result,)
    if isinstance(result, RelationResult):
        if result.cohort is None:
            raise ContractError("RelationResult.cohort is required")
        return (result.cohort,)
    if isinstance(result, FitResult):
        cohorts = []
        for relation_id in result.relation_ids:
            relation = result.relations[str(relation_id)]
            if relation.cohort is None:
                raise ContractError("RelationResult.cohort is required")
            cohorts.append(relation.cohort)
        return tuple(cohorts)
    raise ContractError("result must be a CohortResult, RelationResult, or FitResult")


def _common_cohort_n_states(cohorts: Sequence[CohortResult]) -> int:
    if not cohorts:
        raise ContractError("cohort handover table has no relations")
    n_states = int(np.asarray(cohorts[0].template_A).shape[0])
    for cohort in cohorts:
        A = np.asarray(cohort.template_A)
        d = np.asarray(cohort.template_d)
        e = np.asarray(cohort.template_e)
        if A.shape != (n_states, n_states):
            raise ContractError("all cohort template_A arrays must have shape [K, K]")
        if d.shape != (n_states,) or e.shape != (n_states,):
            raise ContractError("all cohort template_d/template_e arrays must have shape [K]")
    return n_states
