"""
Module: tasks.task_A.arm2.analysis_bioinformed

Build a biologically informed Arm-II extraction package on top of the existing
focused-analysis compute surfaces.

This module keeps the current `analysis/focused/` package untouched. It writes
an independent `analysis/bioinformed/` package for panel-aware prototype
comparisons and ordered-direction unmatched decomposition.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .analysis_contract import (
    Arm2FocusedPaths,
    CONFIRMATORY_FAMILIES,
    PAIR_TYPE_ORDER,
    PROTOTYPE_ANNOTATION_COLUMNS,
    PROTOTYPE_ANNOTATION_VALUE_COLUMNS,
)
from .analysis_response import (
    build_corrected_output_package_from_existing_dir,
    can_rebuild_from_existing_focused_dir,
)

TC_DOMINANT_TOP1_MIN = 0.40
TC_DOMINANT_TOP12_MIN = 0.60
IMMUNE_STROMAL_TOP1_MIN = 0.35
PRIMARY_ANCHOR_PAIR_TYPES: tuple[str, ...] = ("TC->IM", "TC->PT")

PANEL_TC_DOMINANT = "tc_dominant"
PANEL_MIXED_INTERFACE = "mixed_interface"
PANEL_IMMUNE_STROMAL = "immune_stromal_enriched"
PANEL_ORDER: dict[str, int] = {
    PANEL_TC_DOMINANT: 0,
    PANEL_MIXED_INTERFACE: 1,
    PANEL_IMMUNE_STROMAL: 2,
}

BIOINFORMED_OUTPUT_FILENAMES: tuple[str, ...] = (
    "20_tc_dominant_family_summary.csv",
    "21_mixed_interface_family_summary.csv",
    "22_bio_annotated_overlap_conflict_table.csv",
    "23_ot_vs_uot_prototype_contrast.csv",
    "24_bd_unmatched_directionality.csv",
    "25_arm2_biointegrated_memo_table.csv",
)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=float),
        where=denominator > 0.0,
    )


def _expected_proto_ids(prototype_meaning: pd.DataFrame) -> np.ndarray:
    proto_ids = np.sort(prototype_meaning["proto_id"].astype(int).to_numpy())
    if proto_ids.size == 0:
        raise ValueError("Prototype meaning table is empty")
    if not np.array_equal(proto_ids, np.unique(proto_ids)):
        raise ValueError("Prototype meaning table contains duplicate proto_id values")
    return proto_ids


def build_panel_assignment_table(prototype_meaning: pd.DataFrame) -> pd.DataFrame:
    """Assign each prototype to the plan-approved biological panel."""

    required = {
        "proto_id",
        "top1_cell_type",
        "top1_fraction",
        "top12_fraction_sum",
    }
    missing = sorted(required - set(prototype_meaning.columns))
    if missing:
        raise ValueError(f"Prototype meaning table is missing panel columns: {missing}")

    panel = prototype_meaning.loc[:, list(PROTOTYPE_ANNOTATION_COLUMNS)].copy()
    top1_is_tc = panel["top1_cell_type"].astype(str).str.startswith("TC_")
    top1_fraction = pd.to_numeric(panel["top1_fraction"], errors="coerce").astype(float)
    top12_fraction_sum = pd.to_numeric(panel["top12_fraction_sum"], errors="coerce").astype(float)

    tc_dominant_mask = (
        top1_is_tc
        & top1_fraction.ge(TC_DOMINANT_TOP1_MIN)
        & top12_fraction_sum.ge(TC_DOMINANT_TOP12_MIN)
    )
    immune_stromal_mask = (~top1_is_tc) & top1_fraction.ge(IMMUNE_STROMAL_TOP1_MIN)

    panel["panel_name"] = np.select(
        [tc_dominant_mask, immune_stromal_mask],
        [PANEL_TC_DOMINANT, PANEL_IMMUNE_STROMAL],
        default=PANEL_MIXED_INTERFACE,
    )
    panel["panel_rule"] = np.select(
        [tc_dominant_mask, immune_stromal_mask, top1_is_tc],
        [
            (
                "top1_cell_type startswith TC_ and "
                f"top1_fraction>={TC_DOMINANT_TOP1_MIN:.2f} and "
                f"top12_fraction_sum>={TC_DOMINANT_TOP12_MIN:.2f}"
            ),
            f"top1_cell_type non-TC and top1_fraction>={IMMUNE_STROMAL_TOP1_MIN:.2f}",
            "top1_cell_type startswith TC_ but misses tc_dominant threshold",
        ],
        default="non-TC fallback into mixed_interface",
    )
    panel["panel_sort_key"] = panel["panel_name"].map(PANEL_ORDER).fillna(99).astype(int)
    panel["is_borderline_tc_like"] = top1_is_tc & (~tc_dominant_mask)
    return panel.sort_values(["panel_sort_key", "proto_id"]).reset_index(drop=True)


def build_directional_unmatched_patient_proto_table(
    pair_level_transport: pd.DataFrame,
    *,
    destroy_abs_surface: np.ndarray,
    birth_abs_surface: np.ndarray,
    prototype_meaning: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate UOT unmatched quantities to patient-by-direction-by-prototype scope.

    Shares are normalized within each `(patient, pair_type)` directional group.
    """

    proto_ids = _expected_proto_ids(prototype_meaning)
    expected_shape = (pair_level_transport.shape[0], proto_ids.size)
    observed = (
        np.asarray(destroy_abs_surface, dtype=float).shape,
        np.asarray(birth_abs_surface, dtype=float).shape,
    )
    if any(shape != expected_shape for shape in observed):
        raise ValueError(
            "Directional unmatched surfaces do not match the expected [pair, proto] shape: "
            f"expected={expected_shape}, observed={observed}"
        )

    if "ordered_direction" not in pair_level_transport.columns or "pair_family" not in pair_level_transport.columns:
        raise ValueError("pair_level_transport must contain ordered_direction and pair_family")

    records: list[dict[str, object]] = []
    grouped = pair_level_transport.groupby(
        ["patient_id", "ordered_direction", "pair_family"],
        sort=True,
        observed=True,
    ).indices
    for (patient_id, ordered_direction, pair_family), idx in grouped.items():
        row_idx = np.asarray(idx, dtype=int)
        destroy_abs = np.nansum(destroy_abs_surface[row_idx], axis=0)
        birth_abs = np.nansum(birth_abs_surface[row_idx], axis=0)
        destroy_total = float(np.sum(destroy_abs, dtype=float))
        birth_total = float(np.sum(birth_abs, dtype=float))
        destroy_share = _safe_divide(destroy_abs, destroy_total)
        birth_share = _safe_divide(birth_abs, birth_total)

        for offset, proto_id in enumerate(proto_ids.tolist()):
            records.append(
                {
                    "patient_id": str(patient_id),
                    "pair_type": str(ordered_direction),
                    "pair_family": str(pair_family),
                    "direction_role": (
                        "primary_anchor"
                        if str(ordered_direction) in PRIMARY_ANCHOR_PAIR_TYPES
                        else "audit_only"
                    ),
                    "proto_id": int(proto_id),
                    "destroy_abs": float(destroy_abs[offset]),
                    "birth_abs": float(birth_abs[offset]),
                    "destroy_share": float(destroy_share[offset]),
                    "birth_share": float(birth_share[offset]),
                    "destroy_minus_birth_share": float(destroy_share[offset] - birth_share[offset]),
                }
            )

    table = pd.DataFrame.from_records(records)
    if table.empty:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "pair_type",
                "pair_family",
                "direction_role",
                "proto_id",
                "destroy_abs",
                "birth_abs",
                "destroy_share",
                "birth_share",
                "destroy_minus_birth_share",
                *PROTOTYPE_ANNOTATION_VALUE_COLUMNS,
            ]
        )

    table = table.merge(
        prototype_meaning.loc[:, list(PROTOTYPE_ANNOTATION_COLUMNS)],
        on="proto_id",
        how="left",
        validate="many_to_one",
    )
    table["pair_type"] = pd.Categorical(
        table["pair_type"],
        categories=PAIR_TYPE_ORDER,
        ordered=True,
    )
    return table.sort_values(["pair_type", "patient_id", "proto_id"]).reset_index(drop=True)


def summarize_directional_unmatched_by_proto(
    directional_patient_proto: pd.DataFrame,
    panel_assignment: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse the patient-by-direction table to the public proto-by-direction summary."""

    grouped = (
        directional_patient_proto.groupby(
            ["pair_type", "pair_family", "direction_role", "proto_id"],
            sort=True,
            observed=True,
        )
        .agg(
            patient_count=("patient_id", "nunique"),
            destroy_share=("destroy_share", "median"),
            birth_share=("birth_share", "median"),
            destroy_minus_birth_share=("destroy_minus_birth_share", "median"),
            destroy_abs=("destroy_abs", "median"),
            birth_abs=("birth_abs", "median"),
            destroy_gt_birth_patient_count=(
                "destroy_minus_birth_share",
                lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0.0).sum()),
            ),
            birth_gt_destroy_patient_count=(
                "destroy_minus_birth_share",
                lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) < 0.0).sum()),
            ),
        )
        .reset_index()
    )
    grouped["destroy_gt_birth_patient_prop"] = (
        pd.to_numeric(grouped["destroy_gt_birth_patient_count"], errors="coerce").astype(float)
        / pd.to_numeric(grouped["patient_count"], errors="coerce").astype(float)
    )
    grouped["birth_gt_destroy_patient_prop"] = (
        pd.to_numeric(grouped["birth_gt_destroy_patient_count"], errors="coerce").astype(float)
        / pd.to_numeric(grouped["patient_count"], errors="coerce").astype(float)
    )
    grouped = grouped.merge(
        panel_assignment.loc[
            :,
            [
                "proto_id",
                *PROTOTYPE_ANNOTATION_VALUE_COLUMNS,
                "panel_name",
                "panel_rule",
                "panel_sort_key",
                "is_borderline_tc_like",
            ],
        ],
        on="proto_id",
        how="left",
        validate="many_to_one",
    )
    grouped["pair_type"] = pd.Categorical(
        grouped["pair_type"],
        categories=PAIR_TYPE_ORDER,
        ordered=True,
    )
    output_columns = [
        "pair_type",
        "pair_family",
        "direction_role",
        "proto_id",
        *PROTOTYPE_ANNOTATION_VALUE_COLUMNS,
        "panel_name",
        "panel_rule",
        "is_borderline_tc_like",
        "patient_count",
        "destroy_share",
        "birth_share",
        "destroy_minus_birth_share",
        "destroy_abs",
        "birth_abs",
        "destroy_gt_birth_patient_count",
        "destroy_gt_birth_patient_prop",
        "birth_gt_destroy_patient_count",
        "birth_gt_destroy_patient_prop",
    ]
    grouped = grouped.sort_values(["pair_type", "panel_sort_key", "proto_id"]).reset_index(drop=True)
    return grouped.loc[:, output_columns]


def _merge_panel_columns(frame: pd.DataFrame, panel_assignment: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(
        panel_assignment.loc[
            :,
            ["proto_id", "panel_name", "panel_rule", "panel_sort_key", "is_borderline_tc_like"],
        ],
        on="proto_id",
        how="left",
        validate="many_to_one",
    )
    return merged


def build_tc_dominant_family_summary(
    family_specific_summary: pd.DataFrame,
    panel_assignment: pd.DataFrame,
    prototype_recurrence_summary: pd.DataFrame,
) -> pd.DataFrame:
    summary = _merge_panel_columns(family_specific_summary, panel_assignment)
    summary = summary.merge(
        prototype_recurrence_summary.loc[
            :,
            [
                "proto_id",
                "shared_transport_anchor_score",
                "uot_unmatched_contributor_score",
                "shared_transport_and_unmatched_any_patient_count",
                "shared_transport_and_unmatched_any_patient_count_prop",
            ],
        ],
        on="proto_id",
        how="left",
        validate="many_to_one",
    )
    summary = summary.loc[
        summary["panel_name"].astype(str).eq(PANEL_TC_DOMINANT)
        & summary["pair_family"].astype(str).isin(CONFIRMATORY_FAMILIES)
    ].copy()
    output_columns = [
        "panel_name",
        "panel_rule",
        "pair_family",
        "proto_id",
        *PROTOTYPE_ANNOTATION_VALUE_COLUMNS,
        "is_borderline_tc_like",
        "baseline_median_abs_delta_share",
        "balanced_transport_share_median",
        "uot_transport_share_median",
        "balanced_minus_uot_transport_share_median",
        "uot_unmatched_share_median",
        "patient_count",
        "paired_confirmatory_patient_count",
        "shared_transport_anchor_score",
        "uot_unmatched_contributor_score",
        "shared_transport_and_unmatched_any_patient_count",
        "shared_transport_and_unmatched_any_patient_count_prop",
    ]
    return summary.loc[:, output_columns].sort_values(
        ["pair_family", "proto_id"]
    ).reset_index(drop=True)


def build_mixed_interface_family_summary(
    family_specific_summary: pd.DataFrame,
    panel_assignment: pd.DataFrame,
    prototype_recurrence_summary: pd.DataFrame,
) -> pd.DataFrame:
    summary = _merge_panel_columns(family_specific_summary, panel_assignment)
    summary = summary.merge(
        prototype_recurrence_summary.loc[
            :,
            [
                "proto_id",
                "shared_transport_anchor_score",
                "uot_unmatched_contributor_score",
                "shared_transport_and_unmatched_any_patient_count",
                "shared_transport_and_unmatched_any_patient_count_prop",
            ],
        ],
        on="proto_id",
        how="left",
        validate="many_to_one",
    )
    summary = summary.loc[
        summary["panel_name"].astype(str).isin((PANEL_MIXED_INTERFACE, PANEL_IMMUNE_STROMAL))
        & summary["pair_family"].astype(str).isin(CONFIRMATORY_FAMILIES)
    ].copy()
    output_columns = [
        "panel_name",
        "panel_rule",
        "pair_family",
        "proto_id",
        *PROTOTYPE_ANNOTATION_VALUE_COLUMNS,
        "is_borderline_tc_like",
        "baseline_median_abs_delta_share",
        "balanced_transport_share_median",
        "uot_transport_share_median",
        "balanced_minus_uot_transport_share_median",
        "uot_unmatched_share_median",
        "patient_count",
        "paired_confirmatory_patient_count",
        "shared_transport_anchor_score",
        "uot_unmatched_contributor_score",
        "shared_transport_and_unmatched_any_patient_count",
        "shared_transport_and_unmatched_any_patient_count_prop",
    ]
    return summary.loc[:, output_columns].sort_values(
        ["panel_name", "pair_family", "proto_id"]
    ).reset_index(drop=True)


def _membership_pattern(frame: pd.DataFrame) -> pd.Series:
    membership = frame.loc[
        :,
        ["in_anchor_top_10", "in_forced_top_10", "in_unmatched_top_10"],
    ].copy()
    for column in membership.columns:
        membership[column] = membership[column].astype("boolean").fillna(False).astype(bool)
    labels = []
    for row in membership.itertuples(index=False):
        terms: list[str] = []
        if row.in_anchor_top_10:
            terms.append("anchor")
        if row.in_forced_top_10:
            terms.append("forced")
        if row.in_unmatched_top_10:
            terms.append("unmatched")
        labels.append("_and_".join(terms) if terms else "none")
    return pd.Series(labels, index=frame.index, dtype="object")


def build_bio_annotated_overlap_conflict_table(
    overlap_conflict: pd.DataFrame,
    panel_assignment: pd.DataFrame,
    prototype_recurrence_summary: pd.DataFrame,
) -> pd.DataFrame:
    detail = overlap_conflict.loc[overlap_conflict["row_type"].astype(str).eq("prototype")].copy()
    detail = _merge_panel_columns(detail, panel_assignment)
    detail = detail.merge(
        prototype_recurrence_summary.loc[
            :,
            [
                "proto_id",
                "shared_transport_and_unmatched_any_patient_count",
                "shared_transport_and_unmatched_any_patient_count_prop",
            ],
        ],
        on="proto_id",
        how="left",
        validate="many_to_one",
    )
    detail["membership_pattern"] = _membership_pattern(detail)
    output_columns = [
        "proto_id",
        *PROTOTYPE_ANNOTATION_VALUE_COLUMNS,
        "panel_name",
        "panel_rule",
        "is_borderline_tc_like",
        "membership_pattern",
        "shared_transport_anchor_rank",
        "balanced_ot_forced_transport_rank",
        "uot_unmatched_contributor_rank",
        "in_anchor_top_10",
        "in_forced_top_10",
        "in_unmatched_top_10",
        "shared_transport_anchor_score",
        "balanced_ot_forced_transport_score",
        "uot_unmatched_contributor_score",
        "shared_transport_and_unmatched_any_patient_count",
        "shared_transport_and_unmatched_any_patient_count_prop",
    ]
    return detail.loc[:, output_columns].sort_values(
        ["panel_name", "proto_id"]
    ).reset_index(drop=True)


def build_ot_vs_uot_prototype_contrast(
    comparison_view: pd.DataFrame,
    panel_assignment: pd.DataFrame,
    anchors: pd.DataFrame,
    forced: pd.DataFrame,
    unmatched: pd.DataFrame,
    prototype_recurrence_summary: pd.DataFrame,
) -> pd.DataFrame:
    contrast = _merge_panel_columns(comparison_view, panel_assignment)
    contrast = contrast.merge(
        anchors.loc[:, ["proto_id", "shared_transport_anchor_score"]],
        on="proto_id",
        how="left",
        validate="many_to_one",
    ).merge(
        forced.loc[:, ["proto_id", "balanced_ot_forced_transport_score"]],
        on="proto_id",
        how="left",
        validate="many_to_one",
    ).merge(
        unmatched.loc[:, ["proto_id", "uot_unmatched_contributor_score"]],
        on="proto_id",
        how="left",
        validate="many_to_one",
    ).merge(
        prototype_recurrence_summary.loc[
            :,
            [
                "proto_id",
                "shared_transport_and_unmatched_any_patient_count",
                "shared_transport_and_unmatched_any_patient_count_prop",
            ],
        ],
        on="proto_id",
        how="left",
        validate="many_to_one",
    )
    output_columns = [
        "proto_id",
        *PROTOTYPE_ANNOTATION_VALUE_COLUMNS,
        "panel_name",
        "panel_rule",
        "is_borderline_tc_like",
        "balanced_transport_share_tc_im",
        "uot_transport_share_tc_im",
        "uot_unmatched_share_tc_im",
        "balanced_minus_uot_tc_im",
        "balanced_transport_share_tc_pt",
        "uot_transport_share_tc_pt",
        "uot_unmatched_share_tc_pt",
        "balanced_minus_uot_tc_pt",
        "shared_transport_anchor_score",
        "balanced_ot_forced_transport_score",
        "uot_unmatched_contributor_score",
        "shared_transport_and_unmatched_any_patient_count",
        "shared_transport_and_unmatched_any_patient_count_prop",
    ]
    return contrast.loc[:, output_columns].sort_values(
        ["panel_name", "proto_id"]
    ).reset_index(drop=True)


def build_biointegrated_memo_table() -> pd.DataFrame:
    records = [
        {
            "panel_name": PANEL_TC_DOMINANT,
            "pair_family": "TC-IM",
            "primary_contrast": "balanced_transport_share vs uot_transport_share",
            "primary_source_file": "20_tc_dominant_family_summary.csv",
            "availability_status": "generated_postprocessing",
            "requires_reextraction": False,
            "nonclaim_flag": False,
        },
        {
            "panel_name": PANEL_TC_DOMINANT,
            "pair_family": "TC-PT",
            "primary_contrast": "balanced_transport_share vs uot_transport_share",
            "primary_source_file": "20_tc_dominant_family_summary.csv",
            "availability_status": "generated_postprocessing",
            "requires_reextraction": False,
            "nonclaim_flag": False,
        },
        {
            "panel_name": PANEL_TC_DOMINANT,
            "pair_family": "TC-IM",
            "primary_contrast": "uot_transport_share vs uot_unmatched_share",
            "primary_source_file": "20_tc_dominant_family_summary.csv",
            "availability_status": "generated_postprocessing",
            "requires_reextraction": False,
            "nonclaim_flag": False,
        },
        {
            "panel_name": PANEL_TC_DOMINANT,
            "pair_family": "TC-PT",
            "primary_contrast": "uot_transport_share vs uot_unmatched_share",
            "primary_source_file": "20_tc_dominant_family_summary.csv",
            "availability_status": "generated_postprocessing",
            "requires_reextraction": False,
            "nonclaim_flag": False,
        },
        {
            "panel_name": PANEL_MIXED_INTERFACE,
            "pair_family": "TC-IM",
            "primary_contrast": "balanced_minus_uot_transport_share vs uot_unmatched_share",
            "primary_source_file": "21_mixed_interface_family_summary.csv",
            "availability_status": "generated_postprocessing",
            "requires_reextraction": False,
            "nonclaim_flag": False,
        },
        {
            "panel_name": PANEL_MIXED_INTERFACE,
            "pair_family": "TC-PT",
            "primary_contrast": "balanced_minus_uot_transport_share vs uot_unmatched_share",
            "primary_source_file": "21_mixed_interface_family_summary.csv",
            "availability_status": "generated_postprocessing",
            "requires_reextraction": False,
            "nonclaim_flag": False,
        },
        {
            "panel_name": PANEL_IMMUNE_STROMAL,
            "pair_family": "TC-IM",
            "primary_contrast": "context-panel transport vs unmatched split",
            "primary_source_file": "21_mixed_interface_family_summary.csv",
            "availability_status": "generated_postprocessing",
            "requires_reextraction": False,
            "nonclaim_flag": False,
        },
        {
            "panel_name": PANEL_IMMUNE_STROMAL,
            "pair_family": "TC-PT",
            "primary_contrast": "context-panel transport vs unmatched split",
            "primary_source_file": "21_mixed_interface_family_summary.csv",
            "availability_status": "generated_postprocessing",
            "requires_reextraction": False,
            "nonclaim_flag": False,
        },
        {
            "panel_name": "all_panels",
            "pair_family": "TC->IM",
            "primary_contrast": "destroy_share vs birth_share",
            "primary_source_file": "24_bd_unmatched_directionality.csv",
            "availability_status": "generated_arm2_only_reextraction",
            "requires_reextraction": True,
            "nonclaim_flag": False,
        },
        {
            "panel_name": "all_panels",
            "pair_family": "TC->PT",
            "primary_contrast": "destroy_share vs birth_share",
            "primary_source_file": "24_bd_unmatched_directionality.csv",
            "availability_status": "generated_arm2_only_reextraction",
            "requires_reextraction": True,
            "nonclaim_flag": False,
        },
        {
            "panel_name": "all_panels",
            "pair_family": "IM-PT",
            "primary_contrast": "exploratory only; excluded from confirmatory claims",
            "primary_source_file": "24_bd_unmatched_directionality.csv",
            "availability_status": "audit_only",
            "requires_reextraction": True,
            "nonclaim_flag": True,
        },
    ]
    return pd.DataFrame.from_records(records)


def validate_bioinformed_output_package(tables_by_filename: dict[str, pd.DataFrame]) -> None:
    observed = tuple(sorted(tables_by_filename))
    expected = tuple(sorted(BIOINFORMED_OUTPUT_FILENAMES))
    if observed != expected:
        raise ValueError(f"Bioinformed output filenames do not match contract: observed={observed}, expected={expected}")

    tc_summary = tables_by_filename["20_tc_dominant_family_summary.csv"]
    if not tc_summary["panel_name"].astype(str).eq(PANEL_TC_DOMINANT).all():
        raise ValueError("TC-dominant summary contains non-tc_dominant rows")
    if not tc_summary["pair_family"].astype(str).isin(CONFIRMATORY_FAMILIES).all():
        raise ValueError("TC-dominant summary contains non-confirmatory families")

    mixed_summary = tables_by_filename["21_mixed_interface_family_summary.csv"]
    if mixed_summary.empty:
        raise ValueError("Mixed/interface summary is empty")
    if not mixed_summary["panel_name"].astype(str).isin(
        (PANEL_MIXED_INTERFACE, PANEL_IMMUNE_STROMAL)
    ).all():
        raise ValueError("Mixed/interface summary contains unexpected panel labels")

    overlap = tables_by_filename["22_bio_annotated_overlap_conflict_table.csv"]
    if overlap.empty:
        raise ValueError("Bio-annotated overlap/conflict table is empty")

    contrast = tables_by_filename["23_ot_vs_uot_prototype_contrast.csv"]
    contrast_proto_ids = sorted(contrast["proto_id"].astype(int).unique().tolist())
    summary_proto_ids = sorted(
        set(tc_summary["proto_id"].astype(int).tolist())
        | set(mixed_summary["proto_id"].astype(int).tolist())
    )
    if contrast_proto_ids != summary_proto_ids:
        raise ValueError("OT-vs-UOT contrast table does not span the same prototype set as the family summaries")

    bd = tables_by_filename["24_bd_unmatched_directionality.csv"]
    if not set(PRIMARY_ANCHOR_PAIR_TYPES).issubset(set(bd["pair_type"].astype(str).tolist())):
        raise ValueError("B/D directionality table is missing the primary anchor directions")

    memo = tables_by_filename["25_arm2_biointegrated_memo_table.csv"]
    if memo.empty:
        raise ValueError("Biointegrated memo table is empty")


def run_bioinformed_analysis(paths: Arm2FocusedPaths) -> dict[str, pd.DataFrame]:
    """Run the full Arm-2 bioinformed extraction workflow."""

    from .analysis_baseline import build_prototype_meaning_table
    from .analysis_compute import (
        build_uot_pair_prototype_unmatched_surface,
        rerun_uot,
    )
    from .analysis_io import load_inputs

    inputs = load_inputs(paths)
    prototype_meaning = build_prototype_meaning_table(inputs.stage0)
    panel_assignment = build_panel_assignment_table(prototype_meaning)

    focused_dir = paths.arm2_metrics_parquet.parent / "analysis" / "focused"
    if not can_rebuild_from_existing_focused_dir(focused_dir):
        raise FileNotFoundError(
            "Current bioinformed extractor requires the persisted Arm-II focused package at "
            f"{focused_dir}"
        )
    corrected_package = build_corrected_output_package_from_existing_dir(
        focused_dir,
        stage0_h5ad=paths.stage0_h5ad,
        task_config=paths.task_config,
    )

    family_specific_summary = corrected_package.tables_by_filename["10_prototype_family_specific_summary.csv"].copy()
    anchors = corrected_package.tables_by_filename["06_uot_shared_transport_anchors.csv"].copy()
    forced = corrected_package.tables_by_filename["07_balanced_ot_forced_transport_prototypes.csv"].copy()
    unmatched = corrected_package.tables_by_filename["08_uot_unmatched_contributors.csv"].copy()
    overlap_conflict = corrected_package.tables_by_filename["09_prototype_overlap_conflict_audit.csv"].copy()
    prototype_recurrence_summary = corrected_package.tables_by_filename["11_prototype_patient_recurrence_summary.csv"].copy()
    comparison_view = corrected_package.tables_by_filename["12_auxiliary_legacy_prototype_comparison.csv"].copy()

    uot_plan = rerun_uot(inputs)
    uot_unmatched_surface = build_uot_pair_prototype_unmatched_surface(inputs, uot_plan)
    directional_patient_proto = build_directional_unmatched_patient_proto_table(
        inputs.pair_tensors.pair_metadata,
        destroy_abs_surface=np.asarray(uot_unmatched_surface.destroy_abs, dtype=float),
        birth_abs_surface=np.asarray(uot_unmatched_surface.birth_abs, dtype=float),
        prototype_meaning=prototype_meaning,
    )
    bd_directionality = summarize_directional_unmatched_by_proto(
        directional_patient_proto=directional_patient_proto,
        panel_assignment=panel_assignment,
    )

    tables_by_filename = {
        "20_tc_dominant_family_summary.csv": build_tc_dominant_family_summary(
            family_specific_summary=family_specific_summary,
            panel_assignment=panel_assignment,
            prototype_recurrence_summary=prototype_recurrence_summary,
        ),
        "21_mixed_interface_family_summary.csv": build_mixed_interface_family_summary(
            family_specific_summary=family_specific_summary,
            panel_assignment=panel_assignment,
            prototype_recurrence_summary=prototype_recurrence_summary,
        ),
        "22_bio_annotated_overlap_conflict_table.csv": build_bio_annotated_overlap_conflict_table(
            overlap_conflict=overlap_conflict,
            panel_assignment=panel_assignment,
            prototype_recurrence_summary=prototype_recurrence_summary,
        ),
        "23_ot_vs_uot_prototype_contrast.csv": build_ot_vs_uot_prototype_contrast(
            comparison_view=comparison_view,
            panel_assignment=panel_assignment,
            anchors=anchors,
            forced=forced,
            unmatched=unmatched,
            prototype_recurrence_summary=prototype_recurrence_summary,
        ),
        "24_bd_unmatched_directionality.csv": bd_directionality,
        "25_arm2_biointegrated_memo_table.csv": build_biointegrated_memo_table(),
    }
    validate_bioinformed_output_package(tables_by_filename)
    return tables_by_filename


def write_bioinformed_output_package(
    tables_by_filename: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Write the bioinformed package to a dedicated output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in tables_by_filename.items():
        table.to_csv(output_dir / filename, index=False)
