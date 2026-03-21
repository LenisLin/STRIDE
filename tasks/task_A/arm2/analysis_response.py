"""
Module: tasks.task_A.arm2.analysis_response

Postprocessing-only Arm-II response/output correction layer.

This module rebuilds the Arm-II public output surface from the persisted
focused-package summaries when those summaries already contain the required
family-specific, comparator, and patient-level recurrence information.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd

from .analysis_baseline import build_prototype_meaning_table
from .analysis_contract import (
    CONFIRMATORY_FAMILIES,
    FocusedOutputPackage,
    PROTOTYPE_ANNOTATION_COLUMNS,
    PROTOTYPE_ANNOTATION_VALUE_COLUMNS,
)

BASELINE_AND_TISSUE_FILENAMES: tuple[str, ...] = (
    "01_prototype_biological_meaning_table.csv",
    "02_baseline_pair_audit.csv",
    "03_baseline_prototype_confirmatory_summary.csv",
    "04_baseline_patient_family_confirmatory_summary.csv",
    "05_global_transport_summary.csv",
)
LEGACY_COMPARISON_FILENAME = "06_key_prototype_comparison.csv"
LEGACY_RECURRENCE_FILENAME = "07_key_prototype_patient_recurrence.csv"
LEGACY_APPENDIX_FILENAME = "08_minimal_appendix_audit.csv"

CORRECTED_OUTPUT_FILENAMES: tuple[str, ...] = (
    "00_arm2_focused_results_memo.md",
    *BASELINE_AND_TISSUE_FILENAMES,
    "06_uot_shared_transport_anchors.csv",
    "07_balanced_ot_forced_transport_prototypes.csv",
    "08_uot_unmatched_contributors.csv",
    "09_prototype_overlap_conflict_audit.csv",
    "10_prototype_family_specific_summary.csv",
    "11_prototype_patient_recurrence_summary.csv",
    "12_auxiliary_legacy_prototype_comparison.csv",
    "13_auxiliary_legacy_prototype_patient_recurrence.csv",
    "14_minimal_appendix_audit.csv",
)
AUXILIARY_COMPARISON_FILENAME = "12_auxiliary_legacy_prototype_comparison.csv"
AUXILIARY_RECURRENCE_FILENAME = "13_auxiliary_legacy_prototype_patient_recurrence.csv"
AUXILIARY_APPENDIX_FILENAME = "14_minimal_appendix_audit.csv"

TOP_SET_SIZE = 10


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _first_existing_filename(directory: Path, candidates: tuple[str, ...]) -> str | None:
    for filename in candidates:
        if (directory / filename).exists():
            return filename
    return None


def can_rebuild_from_existing_focused_dir(directory: Path) -> bool:
    """Return whether the current focused directory contains enough persisted inputs."""

    if not directory.exists():
        return False
    if any(not (directory / filename).exists() for filename in BASELINE_AND_TISSUE_FILENAMES):
        return False
    comparison_present = _first_existing_filename(
        directory,
        (AUXILIARY_COMPARISON_FILENAME, LEGACY_COMPARISON_FILENAME),
    )
    recurrence_present = _first_existing_filename(
        directory,
        (AUXILIARY_RECURRENCE_FILENAME, LEGACY_RECURRENCE_FILENAME),
    )
    appendix_present = _first_existing_filename(
        directory,
        (AUXILIARY_APPENDIX_FILENAME, LEGACY_APPENDIX_FILENAME),
    )
    return comparison_present is not None and recurrence_present is not None and appendix_present is not None


def _load_persisted_input_tables(directory: Path) -> dict[str, pd.DataFrame]:
    if not can_rebuild_from_existing_focused_dir(directory):
        raise FileNotFoundError(f"Focused directory does not contain a rebuildable Arm-II package: {directory}")

    tables = {
        filename: _safe_read_csv(directory / filename)
        for filename in BASELINE_AND_TISSUE_FILENAMES
    }
    comparison_filename = _first_existing_filename(
        directory,
        (AUXILIARY_COMPARISON_FILENAME, LEGACY_COMPARISON_FILENAME),
    )
    recurrence_filename = _first_existing_filename(
        directory,
        (AUXILIARY_RECURRENCE_FILENAME, LEGACY_RECURRENCE_FILENAME),
    )
    appendix_filename = _first_existing_filename(
        directory,
        (AUXILIARY_APPENDIX_FILENAME, LEGACY_APPENDIX_FILENAME),
    )
    assert comparison_filename is not None
    assert recurrence_filename is not None
    assert appendix_filename is not None
    tables[LEGACY_COMPARISON_FILENAME] = _safe_read_csv(directory / comparison_filename)
    tables[LEGACY_RECURRENCE_FILENAME] = _safe_read_csv(directory / recurrence_filename)
    tables[LEGACY_APPENDIX_FILENAME] = _safe_read_csv(directory / appendix_filename)
    return tables


def _family_suffix(pair_family: str) -> str:
    return str(pair_family).lower().replace("-", "_")


def _decode_h5_strings(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind in {"S", "O"}:
        return np.asarray(
            [
                value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
                for value in array.tolist()
            ],
            dtype=object,
        )
    return array.astype(str)


def _metadata_columns(frame: pd.DataFrame) -> list[str]:
    missing = sorted(set(PROTOTYPE_ANNOTATION_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(
            "Prototype metadata columns are missing from the persisted Arm-II table: "
            f"{missing}"
        )
    return list(PROTOTYPE_ANNOTATION_COLUMNS)


def build_prototype_meaning_from_stage0(
    *,
    stage0_h5ad: Path,
    task_config: Path,
) -> pd.DataFrame:
    """Rebuild the canonical prototype-meaning table from the frozen Stage-0 source."""

    del task_config

    with h5py.File(stage0_h5ad, "r") as handle:
        proto_ids = np.asarray(handle["obs/proto_id"], dtype=int)
        cell_type_codes = np.asarray(handle["obs/cell_type/codes"], dtype=int)
        cell_type_categories = _decode_h5_strings(handle["obs/cell_type/categories"][()])

    valid_proto = proto_ids >= 0
    if not valid_proto.any():
        raise ValueError(f"Stage-0 artifact does not contain any valid proto_id values: {stage0_h5ad}")

    expected_k = int(np.max(proto_ids[valid_proto])) + 1
    prototype_cell_type_counts = np.zeros((expected_k, cell_type_categories.shape[0]), dtype=float)
    valid_cell_type = (
        valid_proto
        & (cell_type_codes >= 0)
        & (cell_type_codes < cell_type_categories.shape[0])
    )
    np.add.at(
        prototype_cell_type_counts,
        (proto_ids[valid_cell_type], cell_type_codes[valid_cell_type]),
        1.0,
    )
    prototype_totals = np.sum(prototype_cell_type_counts, axis=1, dtype=float)
    active_prototype_ids = np.flatnonzero(prototype_totals > 0.0).astype(int)
    if active_prototype_ids.size == 0:
        raise ValueError(f"Stage-0 artifact does not contain any active prototypes: {stage0_h5ad}")

    prototype_fractions = np.zeros_like(prototype_cell_type_counts, dtype=float)
    prototype_fractions[active_prototype_ids] = np.divide(
        prototype_cell_type_counts[active_prototype_ids],
        prototype_totals[active_prototype_ids, None],
        out=np.zeros_like(prototype_cell_type_counts[active_prototype_ids], dtype=float),
        where=prototype_totals[active_prototype_ids, None] > 0.0,
    )
    prototype_records: list[dict[str, object]] = []
    for proto_id in active_prototype_ids.tolist():
        for cell_type_idx, cell_type in enumerate(cell_type_categories.tolist()):
            prototype_records.append(
                {
                    "proto_id": int(proto_id),
                    "cell_type": str(cell_type),
                    "cell_count": float(prototype_cell_type_counts[proto_id, cell_type_idx]),
                    "cell_type_fraction": float(prototype_fractions[proto_id, cell_type_idx]),
                    "prototype_total_cells": float(prototype_totals[proto_id]),
                }
            )
    stage0_bundle = SimpleNamespace(
        prototype_cell_type_table=pd.DataFrame.from_records(prototype_records),
        active_prototype_ids=active_prototype_ids,
    )
    return build_prototype_meaning_table(stage0_bundle)


def _overlay_prototype_meaning(
    frame: pd.DataFrame,
    prototype_meaning: pd.DataFrame,
) -> pd.DataFrame:
    """Replace stale prototype annotation columns with the current Stage-0 meaning table."""

    if "proto_id" not in frame.columns:
        return frame.copy()

    refreshed = frame.drop(
        columns=[column for column in PROTOTYPE_ANNOTATION_VALUE_COLUMNS if column in frame.columns],
        errors="ignore",
    ).merge(
        prototype_meaning,
        on="proto_id",
        how="left",
        validate="many_to_one",
    )
    missing_proto_ids = (
        refreshed.loc[refreshed["dominant_cell_type"].isna(), "proto_id"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if missing_proto_ids:
        raise ValueError(
            "Current Stage-0 prototype annotations do not cover persisted Arm-II proto_id values: "
            f"{missing_proto_ids}"
        )
    return refreshed


def _refresh_annotation_tables(
    tables: dict[str, pd.DataFrame],
    prototype_meaning: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Overlay current Stage-0 prototype annotations onto persisted focused tables."""

    refreshed = {filename: table.copy() for filename, table in tables.items()}
    refreshed["01_prototype_biological_meaning_table.csv"] = prototype_meaning.copy()
    for filename in (
        "03_baseline_prototype_confirmatory_summary.csv",
        LEGACY_COMPARISON_FILENAME,
        LEGACY_RECURRENCE_FILENAME,
    ):
        refreshed[filename] = _overlay_prototype_meaning(
            refreshed[filename],
            prototype_meaning,
        )
    return refreshed


def _count_true(series: pd.Series) -> int:
    values = series.dropna()
    if values.empty:
        return 0
    return int(pd.Series(values).astype(bool).sum())


def _prop_true(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return np.nan
    return float(pd.Series(values).astype(bool).mean())


def _top_set_flags(frame: pd.DataFrame, *, rank_column: str) -> pd.DataFrame:
    output = frame.copy()
    output[f"in_top_{TOP_SET_SIZE}"] = pd.to_numeric(output[rank_column], errors="coerce").astype(float) <= float(TOP_SET_SIZE)
    return output


def build_family_specific_summary(
    legacy_comparison: pd.DataFrame,
    legacy_recurrence: pd.DataFrame,
    baseline_prototype_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per `(prototype, confirmatory_family)` with explicit family identity."""

    annotation_columns = list(PROTOTYPE_ANNOTATION_VALUE_COLUMNS)
    metadata = baseline_prototype_summary.loc[
        :,
        [
            "baseline_priority_rank",
            "proto_id",
            *annotation_columns,
            "paired_patient_count",
            "confirmatory_abs_share_anchor",
            "confirmatory_abs_nonzero_share_anchor",
        ],
    ].copy()

    records: list[dict[str, object]] = []
    for pair_family in CONFIRMATORY_FAMILIES:
        suffix = _family_suffix(pair_family)
        grouped = (
            legacy_recurrence.groupby(_metadata_columns(legacy_recurrence), sort=True, observed=False)
            .agg(
                patient_count=("patient_id", "nunique"),
                paired_confirmatory_patient_count=("has_both_confirmatory_families", lambda s: int(pd.Series(s).astype(bool).sum())),
                uot_transport_positive_patient_count=(f"uot_transport_source_share_{suffix}", lambda s: int((pd.to_numeric(s, errors='coerce').fillna(0.0) > 0.0).sum())),
                uot_transport_positive_patient_prop=(f"uot_transport_source_share_{suffix}", lambda s: float((pd.to_numeric(s, errors='coerce').fillna(0.0) > 0.0).mean())),
                balanced_transport_positive_patient_count=(f"balanced_transport_source_share_{suffix}", lambda s: int((pd.to_numeric(s, errors='coerce').fillna(0.0) > 0.0).sum())),
                balanced_transport_positive_patient_prop=(f"balanced_transport_source_share_{suffix}", lambda s: float((pd.to_numeric(s, errors='coerce').fillna(0.0) > 0.0).mean())),
                balanced_minus_uot_positive_patient_count=(f"balanced_minus_uot_delta_transport_source_share_{suffix}", lambda s: int((pd.to_numeric(s, errors='coerce').fillna(0.0) > 0.0).sum())),
                balanced_minus_uot_positive_patient_prop=(f"balanced_minus_uot_delta_transport_source_share_{suffix}", lambda s: float((pd.to_numeric(s, errors='coerce').fillna(0.0) > 0.0).mean())),
                uot_unmatched_positive_patient_count=(f"confirmatory_uot_unmatched_positive_flag_{suffix}", _count_true),
                uot_unmatched_positive_patient_prop=(f"confirmatory_uot_unmatched_positive_flag_{suffix}", _prop_true),
            )
            .reset_index()
        )
        comparison_subset = legacy_comparison.loc[
            :,
            [
                "proto_id",
                f"{suffix}_median_abs_delta_share",
                f"uot_transport_share_{suffix}",
                f"balanced_transport_share_{suffix}",
                f"balanced_minus_uot_{suffix}",
                f"uot_unmatched_share_{suffix}",
            ],
        ].rename(
            columns={
                f"{suffix}_median_abs_delta_share": "baseline_median_abs_delta_share",
                f"uot_transport_share_{suffix}": "uot_transport_share_median",
                f"balanced_transport_share_{suffix}": "balanced_transport_share_median",
                f"balanced_minus_uot_{suffix}": "balanced_minus_uot_transport_share_median",
                f"uot_unmatched_share_{suffix}": "uot_unmatched_share_median",
            }
        )

        family_frame = metadata.merge(
            grouped,
            on=_metadata_columns(grouped),
            how="left",
            validate="one_to_one",
        ).merge(
            comparison_subset,
            on="proto_id",
            how="left",
            validate="one_to_one",
        )
        family_frame["pair_family"] = pair_family
        records.extend(family_frame.to_dict(orient="records"))

    summary = pd.DataFrame.from_records(records)
    output_columns = [
        "baseline_priority_rank",
        "pair_family",
        "proto_id",
        *annotation_columns,
        "paired_patient_count",
        "patient_count",
        "paired_confirmatory_patient_count",
        "confirmatory_abs_share_anchor",
        "confirmatory_abs_nonzero_share_anchor",
        "baseline_median_abs_delta_share",
        "uot_transport_share_median",
        "uot_transport_positive_patient_count",
        "uot_transport_positive_patient_prop",
        "balanced_transport_share_median",
        "balanced_transport_positive_patient_count",
        "balanced_transport_positive_patient_prop",
        "balanced_minus_uot_transport_share_median",
        "balanced_minus_uot_positive_patient_count",
        "balanced_minus_uot_positive_patient_prop",
        "uot_unmatched_share_median",
        "uot_unmatched_positive_patient_count",
        "uot_unmatched_positive_patient_prop",
    ]
    return summary.loc[:, output_columns].sort_values(
        ["pair_family", "baseline_priority_rank", "proto_id"]
    ).reset_index(drop=True)


def build_uot_shared_transport_anchors(
    legacy_comparison: pd.DataFrame,
    legacy_recurrence: pd.DataFrame,
) -> pd.DataFrame:
    """Build the primary UOT shared-transport anchor table."""

    annotation_columns = list(PROTOTYPE_ANNOTATION_VALUE_COLUMNS)
    recurrence_summary = (
        legacy_recurrence.groupby(_metadata_columns(legacy_recurrence), sort=True, observed=False)
        .agg(
            shared_transport_positive_patient_count=(
                "patient_id",
                lambda s: int(
                    (
                        (pd.to_numeric(legacy_recurrence.loc[s.index, "uot_transport_source_share_tc_im"], errors="coerce").fillna(0.0) > 0.0)
                        & (pd.to_numeric(legacy_recurrence.loc[s.index, "uot_transport_source_share_tc_pt"], errors="coerce").fillna(0.0) > 0.0)
                    ).sum()
                ),
            ),
            shared_transport_and_unmatched_any_patient_count=(
                "patient_id",
                lambda s: int(
                    (
                        (pd.to_numeric(legacy_recurrence.loc[s.index, "uot_transport_source_share_tc_im"], errors="coerce").fillna(0.0) > 0.0)
                        & (pd.to_numeric(legacy_recurrence.loc[s.index, "uot_transport_source_share_tc_pt"], errors="coerce").fillna(0.0) > 0.0)
                        & (
                            legacy_recurrence.loc[s.index, "confirmatory_uot_unmatched_positive_flag_tc_im"].fillna(False).astype(bool)
                            | legacy_recurrence.loc[s.index, "confirmatory_uot_unmatched_positive_flag_tc_pt"].fillna(False).astype(bool)
                        )
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )

    anchors = legacy_comparison.merge(
        recurrence_summary,
        on=_metadata_columns(legacy_comparison),
        how="left",
        validate="one_to_one",
    )
    anchors["shared_transport_anchor_score"] = anchors[
        ["uot_transport_share_tc_im", "uot_transport_share_tc_pt"]
    ].apply(pd.to_numeric, errors="coerce").min(axis=1)
    anchors["shared_transport_recurrence_min"] = anchors[
        ["uot_transport_recurrence_tc_im", "uot_transport_recurrence_tc_pt"]
    ].apply(pd.to_numeric, errors="coerce").min(axis=1)
    anchors["shared_transport_family_gap_abs"] = (
        pd.to_numeric(anchors["uot_transport_share_tc_pt"], errors="coerce").astype(float)
        - pd.to_numeric(anchors["uot_transport_share_tc_im"], errors="coerce").astype(float)
    ).abs()
    anchors["shared_transport_positive_patient_prop"] = (
        pd.to_numeric(anchors["shared_transport_positive_patient_count"], errors="coerce").astype(float)
        / pd.to_numeric(anchors["patient_count"], errors="coerce").astype(float)
    )
    anchors = anchors.sort_values(
        [
            "shared_transport_anchor_score",
            "shared_transport_recurrence_min",
            "shared_transport_positive_patient_prop",
            "proto_id",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    anchors["shared_transport_anchor_rank"] = np.arange(1, anchors.shape[0] + 1, dtype=int)
    anchors = _top_set_flags(anchors, rank_column="shared_transport_anchor_rank")
    output_columns = [
        "shared_transport_anchor_rank",
        "in_top_10",
        "proto_id",
        *annotation_columns,
        "shared_transport_anchor_score",
        "shared_transport_recurrence_min",
        "shared_transport_positive_patient_count",
        "shared_transport_positive_patient_prop",
        "shared_transport_and_unmatched_any_patient_count",
        "uot_transport_share_tc_im",
        "uot_transport_share_tc_pt",
        "uot_transport_recurrence_tc_im",
        "uot_transport_recurrence_tc_pt",
        "shared_transport_family_gap_abs",
        "uot_unmatched_share_tc_im",
        "uot_unmatched_share_tc_pt",
        "patient_count",
        "paired_confirmatory_patient_count",
    ]
    return anchors.loc[:, output_columns]


def build_balanced_ot_forced_transport_prototypes(
    legacy_comparison: pd.DataFrame,
    legacy_recurrence: pd.DataFrame,
) -> pd.DataFrame:
    """Build the primary Balanced-OT-forced transport table."""

    annotation_columns = list(PROTOTYPE_ANNOTATION_VALUE_COLUMNS)
    recurrence_summary = (
        legacy_recurrence.groupby(_metadata_columns(legacy_recurrence), sort=True, observed=False)
        .agg(
            forced_transport_positive_patient_count_tc_im=(
                "balanced_minus_uot_delta_transport_source_share_tc_im",
                lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0.0).sum()),
            ),
            forced_transport_positive_patient_count_tc_pt=(
                "balanced_minus_uot_delta_transport_source_share_tc_pt",
                lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0.0).sum()),
            ),
            forced_transport_positive_patient_count_any_confirmatory=(
                "patient_id",
                lambda s: int(
                    (
                        (pd.to_numeric(legacy_recurrence.loc[s.index, "balanced_minus_uot_delta_transport_source_share_tc_im"], errors="coerce").fillna(0.0) > 0.0)
                        | (pd.to_numeric(legacy_recurrence.loc[s.index, "balanced_minus_uot_delta_transport_source_share_tc_pt"], errors="coerce").fillna(0.0) > 0.0)
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )

    forced = legacy_comparison.merge(
        recurrence_summary,
        on=_metadata_columns(legacy_comparison),
        how="left",
        validate="one_to_one",
    )
    forced["balanced_ot_forced_transport_score"] = forced[
        ["balanced_minus_uot_tc_im", "balanced_minus_uot_tc_pt"]
    ].apply(pd.to_numeric, errors="coerce").clip(lower=0.0).sum(axis=1)
    forced["forced_transport_positive_patient_prop_any_confirmatory"] = (
        pd.to_numeric(forced["forced_transport_positive_patient_count_any_confirmatory"], errors="coerce").astype(float)
        / pd.to_numeric(forced["patient_count"], errors="coerce").astype(float)
    )
    forced["forced_transport_dominant_family"] = np.where(
        pd.to_numeric(forced["balanced_minus_uot_tc_pt"], errors="coerce").fillna(-np.inf)
        > pd.to_numeric(forced["balanced_minus_uot_tc_im"], errors="coerce").fillna(-np.inf),
        "TC-PT",
        "TC-IM",
    )
    forced = forced.sort_values(
        [
            "balanced_ot_forced_transport_score",
            "forced_transport_positive_patient_prop_any_confirmatory",
            "proto_id",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    forced["balanced_ot_forced_transport_rank"] = np.arange(1, forced.shape[0] + 1, dtype=int)
    forced = _top_set_flags(forced, rank_column="balanced_ot_forced_transport_rank")
    output_columns = [
        "balanced_ot_forced_transport_rank",
        "in_top_10",
        "proto_id",
        *annotation_columns,
        "balanced_ot_forced_transport_score",
        "forced_transport_dominant_family",
        "balanced_minus_uot_tc_im",
        "balanced_minus_uot_tc_pt",
        "balanced_minus_uot_recurrence_tc_im",
        "balanced_minus_uot_recurrence_tc_pt",
        "forced_transport_positive_patient_count_tc_im",
        "forced_transport_positive_patient_count_tc_pt",
        "forced_transport_positive_patient_count_any_confirmatory",
        "forced_transport_positive_patient_prop_any_confirmatory",
        "patient_count",
    ]
    return forced.loc[:, output_columns]


def build_uot_unmatched_contributors(
    legacy_comparison: pd.DataFrame,
    legacy_recurrence: pd.DataFrame,
) -> pd.DataFrame:
    """Build the primary UOT unmatched-contributor table."""

    annotation_columns = list(PROTOTYPE_ANNOTATION_VALUE_COLUMNS)
    recurrence_summary = (
        legacy_recurrence.groupby(_metadata_columns(legacy_recurrence), sort=True, observed=False)
        .agg(
            unmatched_positive_patient_count_tc_im=("confirmatory_uot_unmatched_positive_flag_tc_im", _count_true),
            unmatched_positive_patient_count_tc_pt=("confirmatory_uot_unmatched_positive_flag_tc_pt", _count_true),
            unmatched_positive_patient_count_any_confirmatory=(
                "patient_id",
                lambda s: int(
                    (
                        legacy_recurrence.loc[s.index, "confirmatory_uot_unmatched_positive_flag_tc_im"].fillna(False).astype(bool)
                        | legacy_recurrence.loc[s.index, "confirmatory_uot_unmatched_positive_flag_tc_pt"].fillna(False).astype(bool)
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )

    unmatched = legacy_comparison.merge(
        recurrence_summary,
        on=_metadata_columns(legacy_comparison),
        how="left",
        validate="one_to_one",
    )
    unmatched["uot_unmatched_contributor_score"] = unmatched[
        ["uot_unmatched_share_tc_im", "uot_unmatched_share_tc_pt"]
    ].apply(pd.to_numeric, errors="coerce").max(axis=1)
    unmatched["unmatched_positive_patient_prop_any_confirmatory"] = (
        pd.to_numeric(unmatched["unmatched_positive_patient_count_any_confirmatory"], errors="coerce").astype(float)
        / pd.to_numeric(unmatched["patient_count"], errors="coerce").astype(float)
    )
    unmatched["uot_unmatched_dominant_family"] = np.where(
        pd.to_numeric(unmatched["uot_unmatched_share_tc_pt"], errors="coerce").fillna(-np.inf)
        > pd.to_numeric(unmatched["uot_unmatched_share_tc_im"], errors="coerce").fillna(-np.inf),
        "TC-PT",
        "TC-IM",
    )
    unmatched = unmatched.sort_values(
        [
            "uot_unmatched_contributor_score",
            "unmatched_positive_patient_prop_any_confirmatory",
            "proto_id",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    unmatched["uot_unmatched_contributor_rank"] = np.arange(1, unmatched.shape[0] + 1, dtype=int)
    unmatched = _top_set_flags(unmatched, rank_column="uot_unmatched_contributor_rank")
    output_columns = [
        "uot_unmatched_contributor_rank",
        "in_top_10",
        "proto_id",
        *annotation_columns,
        "uot_unmatched_contributor_score",
        "uot_unmatched_dominant_family",
        "uot_unmatched_share_tc_im",
        "uot_unmatched_share_tc_pt",
        "uot_unmatched_recurrence_tc_im",
        "uot_unmatched_recurrence_tc_pt",
        "unmatched_positive_patient_count_tc_im",
        "unmatched_positive_patient_count_tc_pt",
        "unmatched_positive_patient_count_any_confirmatory",
        "unmatched_positive_patient_prop_any_confirmatory",
        "patient_count",
    ]
    return unmatched.loc[:, output_columns]


def build_prototype_overlap_conflict_audit(
    anchors: pd.DataFrame,
    forced: pd.DataFrame,
    unmatched: pd.DataFrame,
) -> pd.DataFrame:
    """Quantify and detail prototype-set overlap explicitly."""

    annotation_columns = list(PROTOTYPE_ANNOTATION_VALUE_COLUMNS)
    top_anchor = set(anchors.loc[anchors["in_top_10"].astype(bool), "proto_id"].astype(int).tolist())
    top_forced = set(forced.loc[forced["in_top_10"].astype(bool), "proto_id"].astype(int).tolist())
    top_unmatched = set(unmatched.loc[unmatched["in_top_10"].astype(bool), "proto_id"].astype(int).tolist())

    summary_records: list[dict[str, object]] = []
    for label, left_set, right_set in (
        ("anchor_vs_forced_top10", top_anchor, top_forced),
        ("anchor_vs_unmatched_top10", top_anchor, top_unmatched),
        ("forced_vs_unmatched_top10", top_forced, top_unmatched),
    ):
        summary_records.append(
            {
                "row_type": "summary",
                "overlap_label": label,
                "top_n": TOP_SET_SIZE,
                "left_set_size": len(left_set),
                "right_set_size": len(right_set),
                "intersection_size": len(left_set & right_set),
                "left_only_size": len(left_set - right_set),
                "right_only_size": len(right_set - left_set),
                "proto_id": np.nan,
                **{column: np.nan for column in annotation_columns},
                "shared_transport_anchor_rank": np.nan,
                "balanced_ot_forced_transport_rank": np.nan,
                "uot_unmatched_contributor_rank": np.nan,
                "in_anchor_top_10": np.nan,
                "in_forced_top_10": np.nan,
                "in_unmatched_top_10": np.nan,
                "shared_transport_anchor_score": np.nan,
                "balanced_ot_forced_transport_score": np.nan,
                "uot_unmatched_contributor_score": np.nan,
            }
        )

    detail = anchors.loc[
        :,
        [
            "proto_id",
            *annotation_columns,
            "shared_transport_anchor_rank",
            "in_top_10",
            "shared_transport_anchor_score",
        ],
    ].rename(columns={"in_top_10": "in_anchor_top_10"})
    detail = detail.merge(
        forced.loc[
            :,
            [
                "proto_id",
                "balanced_ot_forced_transport_rank",
                "in_top_10",
                "balanced_ot_forced_transport_score",
            ],
        ].rename(columns={"in_top_10": "in_forced_top_10"}),
        on="proto_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        unmatched.loc[
            :,
            [
                "proto_id",
                "uot_unmatched_contributor_rank",
                "in_top_10",
                "uot_unmatched_contributor_score",
            ],
        ].rename(columns={"in_top_10": "in_unmatched_top_10"}),
        on="proto_id",
        how="inner",
        validate="one_to_one",
    )
    detail["row_type"] = "prototype"
    detail["overlap_label"] = "top10_membership_detail"
    detail["top_n"] = TOP_SET_SIZE
    detail["left_set_size"] = np.nan
    detail["right_set_size"] = np.nan
    detail["intersection_size"] = np.nan
    detail["left_only_size"] = np.nan
    detail["right_only_size"] = np.nan

    combined = pd.concat(
        [pd.DataFrame.from_records(summary_records), detail],
        ignore_index=True,
        sort=False,
    )
    combined["row_type_order"] = np.where(combined["row_type"].astype(str).eq("summary"), 0, 1)
    output_columns = [
        "row_type",
        "overlap_label",
        "top_n",
        "left_set_size",
        "right_set_size",
        "intersection_size",
        "left_only_size",
        "right_only_size",
        "proto_id",
        *annotation_columns,
        "shared_transport_anchor_rank",
        "balanced_ot_forced_transport_rank",
        "uot_unmatched_contributor_rank",
        "in_anchor_top_10",
        "in_forced_top_10",
        "in_unmatched_top_10",
        "shared_transport_anchor_score",
        "balanced_ot_forced_transport_score",
        "uot_unmatched_contributor_score",
    ]
    combined = combined.sort_values(
        ["row_type_order", "proto_id"],
        ascending=[True, True],
        na_position="first",
    ).reset_index(drop=True)
    return combined.loc[:, output_columns]


def build_prototype_patient_recurrence_summary(
    legacy_recurrence: pd.DataFrame,
    anchors: pd.DataFrame,
    forced: pd.DataFrame,
    unmatched: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate prototype signals to patient-level recurrence summaries."""

    annotation_columns = list(PROTOTYPE_ANNOTATION_VALUE_COLUMNS)
    grouped = (
        legacy_recurrence.groupby(_metadata_columns(legacy_recurrence), sort=True, observed=False)
        .agg(
            patient_count=("patient_id", "nunique"),
            paired_confirmatory_patient_count=("has_both_confirmatory_families", lambda s: int(pd.Series(s).astype(bool).sum())),
            shared_transport_positive_patient_count=(
                "patient_id",
                lambda s: int(
                    (
                        (pd.to_numeric(legacy_recurrence.loc[s.index, "uot_transport_source_share_tc_im"], errors="coerce").fillna(0.0) > 0.0)
                        & (pd.to_numeric(legacy_recurrence.loc[s.index, "uot_transport_source_share_tc_pt"], errors="coerce").fillna(0.0) > 0.0)
                    ).sum()
                ),
            ),
            forced_transport_positive_patient_count_tc_im=(
                "balanced_minus_uot_delta_transport_source_share_tc_im",
                lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0.0).sum()),
            ),
            forced_transport_positive_patient_count_tc_pt=(
                "balanced_minus_uot_delta_transport_source_share_tc_pt",
                lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0.0).sum()),
            ),
            forced_transport_positive_patient_count_any_confirmatory=(
                "patient_id",
                lambda s: int(
                    (
                        (pd.to_numeric(legacy_recurrence.loc[s.index, "balanced_minus_uot_delta_transport_source_share_tc_im"], errors="coerce").fillna(0.0) > 0.0)
                        | (pd.to_numeric(legacy_recurrence.loc[s.index, "balanced_minus_uot_delta_transport_source_share_tc_pt"], errors="coerce").fillna(0.0) > 0.0)
                    ).sum()
                ),
            ),
            unmatched_positive_patient_count_tc_im=("confirmatory_uot_unmatched_positive_flag_tc_im", _count_true),
            unmatched_positive_patient_count_tc_pt=("confirmatory_uot_unmatched_positive_flag_tc_pt", _count_true),
            unmatched_positive_patient_count_any_confirmatory=(
                "patient_id",
                lambda s: int(
                    (
                        legacy_recurrence.loc[s.index, "confirmatory_uot_unmatched_positive_flag_tc_im"].fillna(False).astype(bool)
                        | legacy_recurrence.loc[s.index, "confirmatory_uot_unmatched_positive_flag_tc_pt"].fillna(False).astype(bool)
                    ).sum()
                ),
            ),
            baseline_tc_pt_gt_tc_im_patient_count=("confirmatory_baseline_tc_pt_gt_tc_im_median_abs_delta_share_flag", _count_true),
            shared_transport_and_unmatched_any_patient_count=(
                "patient_id",
                lambda s: int(
                    (
                        (pd.to_numeric(legacy_recurrence.loc[s.index, "uot_transport_source_share_tc_im"], errors="coerce").fillna(0.0) > 0.0)
                        & (pd.to_numeric(legacy_recurrence.loc[s.index, "uot_transport_source_share_tc_pt"], errors="coerce").fillna(0.0) > 0.0)
                        & (
                            legacy_recurrence.loc[s.index, "confirmatory_uot_unmatched_positive_flag_tc_im"].fillna(False).astype(bool)
                            | legacy_recurrence.loc[s.index, "confirmatory_uot_unmatched_positive_flag_tc_pt"].fillna(False).astype(bool)
                        )
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )

    summary = grouped.merge(
        anchors.loc[:, ["proto_id", "shared_transport_anchor_score"]],
        on="proto_id",
        how="left",
        validate="one_to_one",
    ).merge(
        forced.loc[:, ["proto_id", "balanced_ot_forced_transport_score"]],
        on="proto_id",
        how="left",
        validate="one_to_one",
    ).merge(
        unmatched.loc[:, ["proto_id", "uot_unmatched_contributor_score"]],
        on="proto_id",
        how="left",
        validate="one_to_one",
    )
    for count_column in (
        "shared_transport_positive_patient_count",
        "forced_transport_positive_patient_count_tc_im",
        "forced_transport_positive_patient_count_tc_pt",
        "forced_transport_positive_patient_count_any_confirmatory",
        "unmatched_positive_patient_count_tc_im",
        "unmatched_positive_patient_count_tc_pt",
        "unmatched_positive_patient_count_any_confirmatory",
        "baseline_tc_pt_gt_tc_im_patient_count",
        "shared_transport_and_unmatched_any_patient_count",
    ):
        summary[f"{count_column}_prop"] = (
            pd.to_numeric(summary[count_column], errors="coerce").astype(float)
            / pd.to_numeric(summary["patient_count"], errors="coerce").astype(float)
        )
    output_columns = [
        "proto_id",
        *annotation_columns,
        "patient_count",
        "paired_confirmatory_patient_count",
        "shared_transport_anchor_score",
        "shared_transport_positive_patient_count",
        "shared_transport_positive_patient_count_prop",
        "balanced_ot_forced_transport_score",
        "forced_transport_positive_patient_count_tc_im",
        "forced_transport_positive_patient_count_tc_im_prop",
        "forced_transport_positive_patient_count_tc_pt",
        "forced_transport_positive_patient_count_tc_pt_prop",
        "forced_transport_positive_patient_count_any_confirmatory",
        "forced_transport_positive_patient_count_any_confirmatory_prop",
        "uot_unmatched_contributor_score",
        "unmatched_positive_patient_count_tc_im",
        "unmatched_positive_patient_count_tc_im_prop",
        "unmatched_positive_patient_count_tc_pt",
        "unmatched_positive_patient_count_tc_pt_prop",
        "unmatched_positive_patient_count_any_confirmatory",
        "unmatched_positive_patient_count_any_confirmatory_prop",
        "baseline_tc_pt_gt_tc_im_patient_count",
        "baseline_tc_pt_gt_tc_im_patient_count_prop",
        "shared_transport_and_unmatched_any_patient_count",
        "shared_transport_and_unmatched_any_patient_count_prop",
    ]
    return summary.loc[:, output_columns].sort_values(
        ["shared_transport_anchor_score", "proto_id"],
        ascending=[False, True],
    ).reset_index(drop=True)


def build_corrected_results_memo(
    *,
    shared_transport_anchors: pd.DataFrame,
    forced_transport: pd.DataFrame,
    unmatched_contributors: pd.DataFrame,
) -> str:
    """Build the corrected compact Arm-II integrated memo."""

    anchor_overlap = (
        shared_transport_anchors.loc[shared_transport_anchors["in_top_10"].astype(bool), "proto_id"]
        .astype(int)
        .tolist()
    )
    unmatched_overlap = (
        unmatched_contributors.loc[unmatched_contributors["in_top_10"].astype(bool), "proto_id"]
        .astype(int)
        .tolist()
    )
    overlap_size = len(set(anchor_overlap) & set(unmatched_overlap))
    lines = [
        "# Arm-II Corrected Results Memo",
        "",
        "## Scope",
        "",
        "- This package was rebuilt at the analysis/output layer from the persisted Arm-II real-data focused package.",
        "- Confirmatory tissue-level outputs remain `04_baseline_patient_family_confirmatory_summary.csv` and `05_global_transport_summary.csv`.",
        "- The prior aggregate prototype comparison files are retained only as auxiliary outputs `12` and `13`.",
        "",
        "## Prototype-Level Primary Surface",
        "",
        "- `06_uot_shared_transport_anchors.csv`: score=`min(uot_transport_share_tc_im, uot_transport_share_tc_pt)`.",
        "- `07_balanced_ot_forced_transport_prototypes.csv`: score=`max(balanced_minus_uot_tc_im, 0) + max(balanced_minus_uot_tc_pt, 0)`.",
        "- `08_uot_unmatched_contributors.csv`: score=`max(uot_unmatched_share_tc_im, uot_unmatched_share_tc_pt)`.",
        "- `09_prototype_overlap_conflict_audit.csv` quantifies top-set overlap explicitly rather than implying disjoint evidence lines.",
        "- `10_prototype_family_specific_summary.csv` preserves separate `TC-IM` and `TC-PT` rows for each prototype.",
        "- `11_prototype_patient_recurrence_summary.csv` reports per-prototype patient recurrence counts and proportions.",
        "",
        "## Compact Audit Fact",
        "",
        f"- Top-{TOP_SET_SIZE} overlap between UOT shared transport anchors and UOT unmatched contributors remains explicit at `{overlap_size}` prototypes.",
        "",
    ]
    return "\n".join(lines)


def validate_corrected_output_package(package: FocusedOutputPackage) -> pd.DataFrame:
    """Validate the corrected Arm-II package contract."""

    observed_csv = tuple(sorted(package.tables_by_filename))
    expected_csv = tuple(sorted(filename for filename in CORRECTED_OUTPUT_FILENAMES if filename.endswith(".csv")))
    memo_tokens = (
        "## Scope",
        "## Prototype-Level Primary Surface",
        "## Compact Audit Fact",
    )
    validation = pd.DataFrame.from_records(
        [
            {
                "check": "exact_output_csv_filenames",
                "passed": observed_csv == expected_csv,
                "detail": f"expected={list(expected_csv)}, observed={list(observed_csv)}",
            },
            {
                "check": "memo_present",
                "passed": bool(package.memo_text.strip()),
                "detail": f"memo_length={len(package.memo_text)}",
            },
            {
                "check": "memo_tokens_present",
                "passed": all(token in package.memo_text for token in memo_tokens),
                "detail": f"required_tokens={list(memo_tokens)}",
            },
            {
                "check": "all_outputs_are_dataframes",
                "passed": bool(all(isinstance(value, pd.DataFrame) for value in package.tables_by_filename.values())),
                "detail": f"types={[type(value).__name__ for value in package.tables_by_filename.values()]}",
            },
        ]
    )
    failed = validation.loc[~validation["passed"].astype(bool)].copy()
    if not failed.empty:
        details = "; ".join(f"{row['check']}={row['detail']}" for _, row in failed.iterrows())
        raise ValueError(f"Corrected Arm-II package validation failed: {details}")
    return validation


def build_corrected_output_package_from_tables(
    tables: dict[str, pd.DataFrame],
) -> FocusedOutputPackage:
    """Build the corrected write-ready package from persisted legacy tables."""

    baseline_prototype_summary = tables["03_baseline_prototype_confirmatory_summary.csv"].copy()
    legacy_comparison = tables[LEGACY_COMPARISON_FILENAME].copy()
    legacy_recurrence = tables[LEGACY_RECURRENCE_FILENAME].copy()

    family_specific_summary = build_family_specific_summary(
        legacy_comparison=legacy_comparison,
        legacy_recurrence=legacy_recurrence,
        baseline_prototype_summary=baseline_prototype_summary,
    )
    shared_transport_anchors = build_uot_shared_transport_anchors(
        legacy_comparison=legacy_comparison,
        legacy_recurrence=legacy_recurrence,
    )
    forced_transport = build_balanced_ot_forced_transport_prototypes(
        legacy_comparison=legacy_comparison,
        legacy_recurrence=legacy_recurrence,
    )
    unmatched_contributors = build_uot_unmatched_contributors(
        legacy_comparison=legacy_comparison,
        legacy_recurrence=legacy_recurrence,
    )
    overlap_audit = build_prototype_overlap_conflict_audit(
        anchors=shared_transport_anchors,
        forced=forced_transport,
        unmatched=unmatched_contributors,
    )
    recurrence_summary = build_prototype_patient_recurrence_summary(
        legacy_recurrence=legacy_recurrence,
        anchors=shared_transport_anchors,
        forced=forced_transport,
        unmatched=unmatched_contributors,
    )
    memo_text = build_corrected_results_memo(
        shared_transport_anchors=shared_transport_anchors,
        forced_transport=forced_transport,
        unmatched_contributors=unmatched_contributors,
    )

    tables_by_filename = {
        "01_prototype_biological_meaning_table.csv": tables["01_prototype_biological_meaning_table.csv"].copy(),
        "02_baseline_pair_audit.csv": tables["02_baseline_pair_audit.csv"].copy(),
        "03_baseline_prototype_confirmatory_summary.csv": baseline_prototype_summary.copy(),
        "04_baseline_patient_family_confirmatory_summary.csv": tables["04_baseline_patient_family_confirmatory_summary.csv"].copy(),
        "05_global_transport_summary.csv": tables["05_global_transport_summary.csv"].copy(),
        "06_uot_shared_transport_anchors.csv": shared_transport_anchors,
        "07_balanced_ot_forced_transport_prototypes.csv": forced_transport,
        "08_uot_unmatched_contributors.csv": unmatched_contributors,
        "09_prototype_overlap_conflict_audit.csv": overlap_audit,
        "10_prototype_family_specific_summary.csv": family_specific_summary,
        "11_prototype_patient_recurrence_summary.csv": recurrence_summary,
        AUXILIARY_COMPARISON_FILENAME: legacy_comparison.copy(),
        AUXILIARY_RECURRENCE_FILENAME: legacy_recurrence.copy(),
        AUXILIARY_APPENDIX_FILENAME: tables[LEGACY_APPENDIX_FILENAME].copy(),
    }
    package = FocusedOutputPackage(
        memo_text=memo_text,
        tables_by_filename=tables_by_filename,
        package_validation=pd.DataFrame(),
    )
    package.package_validation = validate_corrected_output_package(package)
    return package


def build_corrected_output_package_from_existing_dir(
    directory: Path,
    *,
    stage0_h5ad: Path | None = None,
    task_config: Path | None = None,
) -> FocusedOutputPackage:
    """Load persisted legacy tables from disk and rebuild the corrected package."""

    tables = _load_persisted_input_tables(directory)
    if stage0_h5ad is not None and task_config is not None:
        prototype_meaning = build_prototype_meaning_from_stage0(
            stage0_h5ad=stage0_h5ad,
            task_config=task_config,
        )
        tables = _refresh_annotation_tables(tables, prototype_meaning)
    return build_corrected_output_package_from_tables(tables)


def build_corrected_output_package_from_legacy_package(
    legacy_package: FocusedOutputPackage,
) -> FocusedOutputPackage:
    """Rebuild the corrected package from an in-memory legacy focused package."""

    tables = {
        filename: legacy_package.tables_by_filename[filename].copy()
        for filename in BASELINE_AND_TISSUE_FILENAMES
    }
    tables[LEGACY_COMPARISON_FILENAME] = legacy_package.tables_by_filename[LEGACY_COMPARISON_FILENAME].copy()
    tables[LEGACY_RECURRENCE_FILENAME] = legacy_package.tables_by_filename[LEGACY_RECURRENCE_FILENAME].copy()
    tables[LEGACY_APPENDIX_FILENAME] = legacy_package.tables_by_filename[LEGACY_APPENDIX_FILENAME].copy()
    return build_corrected_output_package_from_tables(tables)


def write_corrected_output_package(package: FocusedOutputPackage, output_dir: Path) -> None:
    """Write the corrected Arm-II package to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    validate_corrected_output_package(package)
    (output_dir / "00_arm2_focused_results_memo.md").write_text(package.memo_text, encoding="utf-8")
    for filename, table in package.tables_by_filename.items():
        table.to_csv(output_dir / filename, index=False)
