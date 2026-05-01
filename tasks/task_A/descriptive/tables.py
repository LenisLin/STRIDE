"""Table builders for the Task A descriptive atlas."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .contracts import DescriptiveAtlasContractError


def build_cell_frame(
    *,
    adata: Any,
    field_mapping: dict[str, str],
    patient_ids: tuple[str, ...] | None,
) -> pd.DataFrame:
    if "spatial" not in adata.obsm:
        raise DescriptiveAtlasContractError("Task A descriptive atlas requires adata.obsm['spatial'] for overlays")
    spatial = np.asarray(adata.obsm["spatial"], dtype=float)
    if spatial.ndim != 2 or spatial.shape[0] != adata.n_obs or spatial.shape[1] < 2:
        raise DescriptiveAtlasContractError(
            "Task A descriptive atlas requires spatial coordinates with shape [n_cells, >=2]"
        )

    obs_columns = [
        field_mapping["patient_id_key"],
        field_mapping["domain_key"],
        field_mapping["fov_key"],
        field_mapping["cell_subtype_key"],
        field_mapping["state_id_key"],
    ]
    missing_columns = [column for column in obs_columns if column not in adata.obs.columns]
    if missing_columns:
        raise DescriptiveAtlasContractError(
            f"Task A descriptive atlas is missing required Stage 0 obs columns: {missing_columns}"
        )

    patient_series = adata.obs[field_mapping["patient_id_key"]].astype(str)
    mask = np.ones(adata.n_obs, dtype=bool)
    if patient_ids is not None:
        requested_patient_ids = tuple(str(patient_id) for patient_id in patient_ids)
        observed_patient_ids = set(patient_series.unique().tolist())
        missing_patient_ids = [
            patient_id
            for patient_id in dict.fromkeys(requested_patient_ids)
            if patient_id not in observed_patient_ids
        ]
        if missing_patient_ids:
            raise DescriptiveAtlasContractError(
                "Task A descriptive atlas is missing requested patient_ids in the Stage0 h5ad: "
                f"{missing_patient_ids}"
            )
        mask = patient_series.isin(set(requested_patient_ids)).to_numpy(dtype=bool)
        if not bool(mask.any()):
            raise DescriptiveAtlasContractError(
                "Task A descriptive atlas patient_ids did not match any cells in the Stage0 h5ad"
            )

    obs = adata.obs.loc[mask, obs_columns].copy()
    frame = pd.DataFrame(
        {
            "patient_id": obs[field_mapping["patient_id_key"]].astype(str).to_numpy(),
            "domain_label": obs[field_mapping["domain_key"]].astype(str).to_numpy(),
            "fov_id": obs[field_mapping["fov_key"]].astype(str).to_numpy(),
            "cell_subtype_label": obs[field_mapping["cell_subtype_key"]].astype(str).to_numpy(),
            "community_id": obs[field_mapping["state_id_key"]].astype(int).to_numpy(),
            "x": spatial[mask, 0],
            "y": spatial[mask, 1],
        }
    )
    if frame.empty:
        raise DescriptiveAtlasContractError("Task A descriptive atlas cannot run on an empty selected cohort")
    return frame


def resolve_domain_order(frame: pd.DataFrame, config_bundle: Any) -> list[str]:
    configured = [str(domain) for domain in config_bundle.ordered_proxy.domains]
    observed = sorted(frame["domain_label"].astype(str).unique().tolist())
    extras = [domain for domain in observed if domain not in configured]
    ordered = [domain for domain in configured if domain in observed]
    ordered.extend(extras)
    return ordered


def resolve_community_order(frame: pd.DataFrame, configured_community_ids: list[int]) -> list[int]:
    observed = {int(community_id) for community_id in frame["community_id"].astype(int).unique().tolist()}
    configured_ids = {int(community_id) for community_id in configured_community_ids}
    unconfigured = sorted(community_id for community_id in observed if community_id not in configured_ids)
    if unconfigured:
        raise DescriptiveAtlasContractError(
            "Task A descriptive atlas observed unconfigured community ids in the Stage0 h5ad: "
            f"{unconfigured}"
        )
    configured = [
        int(community_id)
        for community_id in configured_community_ids
        if int(community_id) in observed
    ]
    if configured:
        return configured
    return sorted(observed)


def prepare_matrix_frame(
    frame: pd.DataFrame,
    *,
    row_name: str,
    column_name: str,
    row_order: list[Any],
    column_order: list[Any],
) -> pd.DataFrame:
    matrix = pd.crosstab(frame[row_name], frame[column_name], dropna=False)
    row_index = pd.Index(row_order, name=row_name)
    column_index = pd.Index(column_order, name=column_name)
    matrix = matrix.reindex(index=row_index, columns=column_index, fill_value=0)
    matrix.index.name = row_name
    matrix.columns.name = column_name
    return matrix


def build_domain_distribution_table(
    frame: pd.DataFrame,
    *,
    community_order: list[int],
    domain_order: list[str],
) -> pd.DataFrame:
    grouped = (
        frame.groupby(["community_id", "domain_label"], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    index = pd.MultiIndex.from_product(
        [community_order, domain_order],
        names=["community_id", "domain_label"],
    )
    table = grouped.set_index(["community_id", "domain_label"]).reindex(index, fill_value=0).reset_index()
    community_totals = table.groupby("community_id", observed=False)["n_cells"].transform("sum")
    domain_totals = table.groupby("domain_label", observed=False)["n_cells"].transform("sum")
    table["community_total_cells"] = community_totals.astype(int)
    table["domain_total_cells"] = domain_totals.astype(int)
    table["fraction_within_community"] = np.where(community_totals > 0, table["n_cells"] / community_totals, 0.0)
    table["fraction_within_domain"] = np.where(domain_totals > 0, table["n_cells"] / domain_totals, 0.0)
    return table


def build_domain_roi_prevalence_table(
    frame: pd.DataFrame,
    *,
    community_order: list[int],
    domain_order: list[str],
) -> pd.DataFrame:
    roi_frame = frame[["patient_id", "domain_label", "fov_id"]].drop_duplicates()
    total_rois = roi_frame.groupby("domain_label", observed=False).size().rename("total_rois").to_dict()
    positive_rois = (
        frame.groupby(["community_id", "domain_label", "fov_id"], observed=False)
        .size()
        .reset_index(name="community_cells")
        .groupby(["community_id", "domain_label"], observed=False)["fov_id"]
        .nunique()
        .rename("positive_rois")
        .reset_index()
    )
    index = pd.MultiIndex.from_product(
        [community_order, domain_order],
        names=["community_id", "domain_label"],
    )
    table = positive_rois.set_index(["community_id", "domain_label"]).reindex(index, fill_value=0).reset_index()
    table["total_rois"] = table["domain_label"].map(total_rois).fillna(0).astype(int)
    table["roi_prevalence"] = np.where(table["total_rois"] > 0, table["positive_rois"] / table["total_rois"], 0.0)
    return table


def build_patient_occurrence_tables(
    frame: pd.DataFrame,
    *,
    community_order: list[int],
    patient_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_presence = (
        frame.groupby(["community_id", "patient_id"], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    matrix_index = pd.Index(community_order, name="community_id")
    matrix_columns = pd.Index(patient_order, name="patient_id")
    matrix = pd.crosstab(patient_presence["community_id"], patient_presence["patient_id"], dropna=False)
    matrix = matrix.reindex(index=matrix_index, columns=matrix_columns, fill_value=0)
    matrix = (matrix > 0).astype(int)

    roi_presence = (
        frame.groupby(["community_id", "patient_id", "fov_id"], observed=False)
        .size()
        .rename("community_cells")
        .reset_index()
    )
    total_rois = int(frame[["patient_id", "fov_id"]].drop_duplicates().shape[0])
    patient_summary = (
        patient_presence.groupby("community_id", observed=False)
        .agg(
            n_patients_present=("patient_id", "nunique"),
            total_cells=("n_cells", "sum"),
            median_cells_per_positive_patient=("n_cells", "median"),
            max_cells_in_single_patient=("n_cells", "max"),
        )
        .reindex(matrix_index, fill_value=0)
        .reset_index()
    )
    positive_rois = (
        roi_presence.groupby("community_id", observed=False)["fov_id"]
        .nunique()
        .reindex(matrix_index, fill_value=0)
        .to_numpy(dtype=int)
    )
    patient_summary["n_patients_total"] = int(len(patient_order))
    patient_summary["patient_prevalence"] = np.where(
        patient_summary["n_patients_total"] > 0,
        patient_summary["n_patients_present"] / patient_summary["n_patients_total"],
        0.0,
    )
    patient_summary["n_positive_rois"] = positive_rois
    patient_summary["n_total_rois"] = total_rois
    patient_summary["roi_prevalence"] = np.where(
        patient_summary["n_total_rois"] > 0,
        patient_summary["n_positive_rois"] / patient_summary["n_total_rois"],
        0.0,
    )
    return patient_summary, matrix


def select_representative_overlays(
    frame: pd.DataFrame,
    *,
    max_overlay_communities: int,
) -> pd.DataFrame:
    community_totals = (
        frame.groupby("community_id", observed=False)
        .size()
        .rename("community_total_cells")
        .sort_values(ascending=False)
    )
    selected_communities = community_totals.head(max_overlay_communities).index.tolist()
    roi_counts = (
        frame.groupby(["community_id", "patient_id", "domain_label", "fov_id"], observed=False)
        .size()
        .rename("community_cells")
        .reset_index()
    )
    roi_totals = (
        frame.groupby(["patient_id", "domain_label", "fov_id"], observed=False)
        .size()
        .rename("roi_total_cells")
        .reset_index()
    )
    selection = roi_counts.merge(
        roi_totals,
        on=["patient_id", "domain_label", "fov_id"],
        how="left",
        validate="many_to_one",
    )
    selection["community_fraction_in_roi"] = np.where(
        selection["roi_total_cells"] > 0,
        selection["community_cells"] / selection["roi_total_cells"],
        0.0,
    )
    selection["community_total_cells"] = selection["community_id"].map(community_totals.to_dict()).astype(int)

    records: list[pd.Series] = []
    for community_id in selected_communities:
        community_selection = selection.loc[selection["community_id"] == community_id].copy()
        community_selection = community_selection.sort_values(
            by=["community_fraction_in_roi", "community_cells", "patient_id", "domain_label", "fov_id"],
            ascending=[False, False, True, True, True],
            kind="mergesort",
        )
        if not community_selection.empty:
            records.append(community_selection.iloc[0])
    if records:
        return pd.DataFrame(records).reset_index(drop=True)
    return pd.DataFrame(
        columns=[
            "community_id",
            "community_total_cells",
            "patient_id",
            "domain_label",
            "fov_id",
            "community_cells",
            "roi_total_cells",
            "community_fraction_in_roi",
        ]
    )


__all__ = [
    "build_cell_frame",
    "build_domain_distribution_table",
    "build_domain_roi_prevalence_table",
    "build_patient_occurrence_tables",
    "prepare_matrix_frame",
    "resolve_community_order",
    "resolve_domain_order",
    "select_representative_overlays",
]
