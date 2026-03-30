"""
Module: tasks.task_A.arm2.analysis_views

Downstream extracted-view layer for the post-hoc Arm-II focused rewrite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis_contract import (
    BaselineAnalysisTables,
    FocusedPrototypeViews,
    FocusedOutputPackage,
    PROTOTYPE_ANNOTATION_VALUE_COLUMNS,
    RecurrenceAnalysisTables,
    TransportAnalysisTables,
)


def _resolve_selected_proto_ids(
    selected_proto_ids: tuple[int, ...] | None,
    baseline_tables: BaselineAnalysisTables,
    recurrence_tables: RecurrenceAnalysisTables,
) -> tuple[int, ...]:
    """Resolve the downstream prototype subset without altering upstream tables."""

    if not baseline_tables.baseline_prototype_confirmatory.empty:
        ordered_available = (
            baseline_tables.baseline_prototype_confirmatory.sort_values(
                ["baseline_priority_rank", "proto_id"]
            )["proto_id"].astype(int).tolist()
        )
    else:
        ordered_available = (
            recurrence_tables.all_prototype_patient_recurrence["proto_id"]
            .astype(int)
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
    available_set = set(ordered_available)
    if selected_proto_ids is None:
        return tuple(ordered_available)

    requested = tuple(int(proto_id) for proto_id in selected_proto_ids)
    missing = sorted(set(requested) - available_set)
    if missing:
        raise ValueError(f"Selected prototype IDs are absent from the all-prototype tables: {missing}")
    # Preserve user-specified order while still deduplicating.
    seen: set[int] = set()
    resolved: list[int] = []
    for proto_id in requested:
        if proto_id not in seen:
            seen.add(proto_id)
            resolved.append(proto_id)
    return tuple(resolved)


def _prop_positive(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if values.empty:
        return np.nan
    return float((values > 0.0).mean())


def _prop_true(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return np.nan
    return float(pd.Series(values).astype(bool).mean())


def extract_prototype_comparison_view(
    selected_proto_ids: tuple[int, ...] | None,
    baseline_tables: BaselineAnalysisTables,
    transport_tables: TransportAnalysisTables,
    recurrence_tables: RecurrenceAnalysisTables,
) -> pd.DataFrame:
    """
    Build the auxiliary prototype-comparator view for public output `12`.

    This is a pure downstream projection over the all-prototype tables.
    """

    annotation_columns = list(PROTOTYPE_ANNOTATION_VALUE_COLUMNS)
    resolved_proto_ids = _resolve_selected_proto_ids(
        selected_proto_ids,
        baseline_tables,
        recurrence_tables,
    )
    recurrence = recurrence_tables.all_prototype_patient_recurrence.loc[
        recurrence_tables.all_prototype_patient_recurrence["proto_id"].astype(int).isin(resolved_proto_ids)
    ].copy()
    baseline_summary = baseline_tables.baseline_prototype_confirmatory.loc[
        baseline_tables.baseline_prototype_confirmatory["proto_id"].astype(int).isin(resolved_proto_ids)
    ].copy()
    baseline_summary["selected_rank"] = baseline_summary["proto_id"].astype(int).map(
        {proto_id: rank for rank, proto_id in enumerate(resolved_proto_ids, start=1)}
    )

    grouped = (
        recurrence.groupby(
            ["proto_id", *annotation_columns],
            sort=True,
            observed=False,
        )
        .agg(
            patient_count=("patient_id", "nunique"),
            paired_confirmatory_patient_count=("has_both_confirmatory_families", lambda s: int(pd.Series(s).astype(bool).sum())),
            recurrence_patient_level_prop_tc_pt_gt_tc_im_abs_delta_share=(
                "confirmatory_baseline_tc_pt_gt_tc_im_median_abs_delta_share_flag",
                _prop_true,
            ),
            closed_comparator_share_tc_im=("balanced_transport_source_share_tc_im", "median"),
            closed_comparator_share_tc_pt=("balanced_transport_source_share_tc_pt", "median"),
            closed_comparator_recurrence_tc_im=("balanced_transport_source_share_tc_im", _prop_positive),
            closed_comparator_recurrence_tc_pt=("balanced_transport_source_share_tc_pt", _prop_positive),
            continuity_backbone_share_tc_im=("uot_transport_source_share_tc_im", "median"),
            continuity_backbone_share_tc_pt=("uot_transport_source_share_tc_pt", "median"),
            continuity_backbone_recurrence_tc_im=("uot_transport_source_share_tc_im", _prop_positive),
            continuity_backbone_recurrence_tc_pt=("uot_transport_source_share_tc_pt", _prop_positive),
            forced_closure_excess_tc_im=("balanced_minus_uot_delta_transport_source_share_tc_im", "median"),
            forced_closure_excess_tc_pt=("balanced_minus_uot_delta_transport_source_share_tc_pt", "median"),
            forced_closure_recurrence_tc_im=("confirmatory_balanced_minus_uot_positive_flag_tc_im", _prop_true),
            forced_closure_recurrence_tc_pt=("confirmatory_balanced_minus_uot_positive_flag_tc_pt", _prop_true),
            bounded_residual_share_tc_im=("uot_unmatched_share_tc_im", "median"),
            bounded_residual_share_tc_pt=("uot_unmatched_share_tc_pt", "median"),
            bounded_residual_recurrence_tc_im=("confirmatory_uot_unmatched_positive_flag_tc_im", _prop_true),
            bounded_residual_recurrence_tc_pt=("confirmatory_uot_unmatched_positive_flag_tc_pt", _prop_true),
        )
        .reset_index()
    )

    comparison = baseline_summary.merge(
        grouped,
        on=["proto_id", *annotation_columns],
        how="left",
        validate="one_to_one",
    )
    comparison["selected_rank"] = comparison["selected_rank"].astype(int)
    output_columns = [
        "selected_rank",
        "proto_id",
        *annotation_columns,
        "paired_patient_count",
        "confirmatory_abs_share_anchor",
        "confirmatory_abs_nonzero_share_anchor",
        "tc_im_median_abs_delta_share",
        "tc_pt_median_abs_delta_share",
        "patient_median_tc_pt_minus_tc_im_abs_delta_share",
        "prop_paired_patients_tc_pt_gt_tc_im_abs_delta_share",
        "patient_count",
        "paired_confirmatory_patient_count",
        "recurrence_patient_level_prop_tc_pt_gt_tc_im_abs_delta_share",
        "closed_comparator_share_tc_im",
        "closed_comparator_share_tc_pt",
        "closed_comparator_recurrence_tc_im",
        "closed_comparator_recurrence_tc_pt",
        "continuity_backbone_share_tc_im",
        "continuity_backbone_share_tc_pt",
        "continuity_backbone_recurrence_tc_im",
        "continuity_backbone_recurrence_tc_pt",
        "forced_closure_excess_tc_im",
        "forced_closure_excess_tc_pt",
        "forced_closure_recurrence_tc_im",
        "forced_closure_recurrence_tc_pt",
        "bounded_residual_share_tc_im",
        "bounded_residual_share_tc_pt",
        "bounded_residual_recurrence_tc_im",
        "bounded_residual_recurrence_tc_pt",
    ]
    for column in output_columns:
        if column not in comparison.columns:
            comparison[column] = np.nan
    return comparison.loc[:, output_columns].sort_values(["selected_rank", "proto_id"]).reset_index(drop=True)


def extract_prototype_recurrence_view(
    selected_proto_ids: tuple[int, ...] | None,
    recurrence_tables: RecurrenceAnalysisTables,
) -> pd.DataFrame:
    """
    Build the auxiliary prototype-anchor view for public output `13`.

    This is a pure downstream subset over the all-prototype recurrence table.
    """

    annotation_columns = list(PROTOTYPE_ANNOTATION_VALUE_COLUMNS)
    available = (
        recurrence_tables.all_prototype_patient_recurrence["proto_id"]
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if selected_proto_ids is None:
        resolved_proto_ids = tuple(available)
    else:
        missing = sorted(set(int(proto_id) for proto_id in selected_proto_ids) - set(available))
        if missing:
            raise ValueError(f"Selected prototype IDs are absent from the recurrence table: {missing}")
        resolved_proto_ids = tuple(dict.fromkeys(int(proto_id) for proto_id in selected_proto_ids))

    view = recurrence_tables.all_prototype_patient_recurrence.loc[
        recurrence_tables.all_prototype_patient_recurrence["proto_id"].astype(int).isin(resolved_proto_ids)
    ].copy()
    view["selected_rank"] = view["proto_id"].astype(int).map(
        {proto_id: rank for rank, proto_id in enumerate(resolved_proto_ids, start=1)}
    )
    output_columns = [
        "selected_rank",
        "patient_id",
        "proto_id",
        *annotation_columns,
        "has_both_confirmatory_families",
        "baseline_median_abs_delta_share_tc_im",
        "baseline_median_abs_delta_share_tc_pt",
        "confirmatory_baseline_tc_pt_minus_tc_im_median_abs_delta_share",
        "confirmatory_baseline_tc_pt_gt_tc_im_median_abs_delta_share_flag",
        "closed_comparator_source_share_tc_im",
        "closed_comparator_source_share_tc_pt",
        "continuity_backbone_source_share_tc_im",
        "continuity_backbone_source_share_tc_pt",
        "forced_closure_excess_source_share_tc_im",
        "forced_closure_excess_source_share_tc_pt",
        "bounded_residual_share_tc_im",
        "bounded_residual_share_tc_pt",
        "confirmatory_bounded_residual_positive_flag_tc_im",
        "confirmatory_bounded_residual_positive_flag_tc_pt",
    ]
    view = view.rename(
        columns={
            "balanced_transport_source_share_tc_im": "closed_comparator_source_share_tc_im",
            "balanced_transport_source_share_tc_pt": "closed_comparator_source_share_tc_pt",
            "uot_transport_source_share_tc_im": "continuity_backbone_source_share_tc_im",
            "uot_transport_source_share_tc_pt": "continuity_backbone_source_share_tc_pt",
            "balanced_minus_uot_delta_transport_source_share_tc_im": "forced_closure_excess_source_share_tc_im",
            "balanced_minus_uot_delta_transport_source_share_tc_pt": "forced_closure_excess_source_share_tc_pt",
            "confirmatory_uot_unmatched_positive_flag_tc_im": "confirmatory_bounded_residual_positive_flag_tc_im",
            "confirmatory_uot_unmatched_positive_flag_tc_pt": "confirmatory_bounded_residual_positive_flag_tc_pt",
        }
    )
    for column in output_columns:
        if column not in view.columns:
            view[column] = np.nan
    return view.loc[:, output_columns].sort_values(
        ["selected_rank", "proto_id", "patient_id"]
    ).reset_index(drop=True)


def validate_extracted_views(
    selected_proto_ids: tuple[int, ...] | None,
    prototype_comparison_view: pd.DataFrame,
    prototype_recurrence_view: pd.DataFrame,
    recurrence_tables: RecurrenceAnalysisTables,
) -> pd.DataFrame:
    """
    Validate the downstream extracted views used for outputs `06` and `07`.
    """

    all_proto_ids = sorted(
        recurrence_tables.all_prototype_patient_recurrence["proto_id"].astype(int).unique().tolist()
    )
    if selected_proto_ids is None:
        resolved_proto_ids = all_proto_ids
    else:
        resolved_proto_ids = list(dict.fromkeys(int(proto_id) for proto_id in selected_proto_ids))

    comparison_proto_ids = sorted(prototype_comparison_view["proto_id"].astype(int).unique().tolist())
    recurrence_proto_ids = sorted(prototype_recurrence_view["proto_id"].astype(int).unique().tolist())
    recurrence_patient_count = int(
        recurrence_tables.all_prototype_patient_recurrence["patient_id"].astype(str).nunique()
    )

    records = pd.DataFrame.from_records(
        [
            {
                "check": "comparison_view_selected_proto_subset",
                "passed": comparison_proto_ids == sorted(resolved_proto_ids),
                "detail": (
                    f"observed={comparison_proto_ids}, expected={sorted(resolved_proto_ids)}"
                ),
            },
            {
                "check": "recurrence_view_selected_proto_subset",
                "passed": recurrence_proto_ids == sorted(resolved_proto_ids),
                "detail": (
                    f"observed={recurrence_proto_ids}, expected={sorted(resolved_proto_ids)}"
                ),
            },
            {
                "check": "views_do_not_escape_all_prototype_axis",
                "passed": bool(
                    set(comparison_proto_ids).issubset(all_proto_ids)
                    and set(recurrence_proto_ids).issubset(all_proto_ids)
                ),
                "detail": f"all_proto_ids={all_proto_ids}",
            },
            {
                "check": "comparison_view_one_row_per_proto",
                "passed": bool(
                    prototype_comparison_view["proto_id"].astype(int).is_unique
                    if not prototype_comparison_view.empty
                    else True
                ),
                "detail": f"rows={prototype_comparison_view.shape[0]}",
            },
            {
                "check": "recurrence_view_patient_proto_shape",
                "passed": bool(
                    prototype_recurrence_view.shape[0]
                    == recurrence_patient_count * len(resolved_proto_ids)
                ),
                "detail": (
                    f"rows={prototype_recurrence_view.shape[0]}, "
                    f"expected={recurrence_patient_count * len(resolved_proto_ids)}"
                ),
            },
        ]
    )
    failed = records.loc[~records["passed"].astype(bool)].copy()
    if not failed.empty:
        details = "; ".join(f"{row['check']}={row['detail']}" for _, row in failed.iterrows())
        raise ValueError(f"Extracted-view validation failed: {details}")
    return records


def build_focused_prototype_views(
    selected_proto_ids: tuple[int, ...] | None,
    baseline_tables: BaselineAnalysisTables,
    transport_tables: TransportAnalysisTables,
    recurrence_tables: RecurrenceAnalysisTables,
) -> FocusedPrototypeViews:
    """Assemble the downstream extracted-view bundle for outputs `06` and `07`."""

    prototype_comparison_view = extract_prototype_comparison_view(
        selected_proto_ids=selected_proto_ids,
        baseline_tables=baseline_tables,
        transport_tables=transport_tables,
        recurrence_tables=recurrence_tables,
    )
    prototype_recurrence_view = extract_prototype_recurrence_view(
        selected_proto_ids=selected_proto_ids,
        recurrence_tables=recurrence_tables,
    )
    view_validation = validate_extracted_views(
        selected_proto_ids=selected_proto_ids,
        prototype_comparison_view=prototype_comparison_view,
        prototype_recurrence_view=prototype_recurrence_view,
        recurrence_tables=recurrence_tables,
    )
    return FocusedPrototypeViews(
        prototype_comparison_view=prototype_comparison_view,
        prototype_recurrence_view=prototype_recurrence_view,
        view_validation=view_validation,
    )
