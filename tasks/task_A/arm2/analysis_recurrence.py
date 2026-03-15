"""
Module: tasks.task_A.arm2.analysis_recurrence

All-prototype recurrence layer for the post-hoc Arm-II focused rewrite.

This module must build recurrence at all-prototype scope before any focused
prototype extraction occurs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis_contract import (
    AUDIT_ONLY_FAMILIES,
    BaselineAnalysisTables,
    CONFIRMATORY_FAMILIES,
    PAIR_FAMILY_ORDER,
    RecurrenceAnalysisTables,
    TransportAnalysisTables,
)


def _family_suffix(pair_family: str) -> str:
    return str(pair_family).lower().replace("-", "_")


def _pivot_pair_family_values(
    frame: pd.DataFrame,
    *,
    index_columns: list[str],
    value_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Pivot family-specific values into stable wide columns on the patient-prototype axis."""

    expected_columns = [
        f"{prefix}_{value_column}_{_family_suffix(pair_family)}"
        for value_column in value_columns
        for pair_family in PAIR_FAMILY_ORDER
    ]
    if frame.empty:
        return pd.DataFrame(columns=[*index_columns, *expected_columns])

    pivot = frame.pivot(index=index_columns, columns="pair_family", values=value_columns)
    pivot.columns = [
        f"{prefix}_{value_column}_{_family_suffix(pair_family)}"
        for value_column, pair_family in pivot.columns
    ]
    pivot = pivot.reset_index()
    for column in expected_columns:
        if column not in pivot.columns:
            pivot[column] = np.nan
    return pivot.loc[:, [*index_columns, *expected_columns]]


def _proto_reference_table(
    baseline_tables: BaselineAnalysisTables,
) -> pd.DataFrame:
    """Return one row per active prototype with the biological meaning columns."""

    columns = [
        "proto_id",
        "dominant_cell_type",
        "dominant_cell_type_fraction",
        "top_cell_type_mix",
        "total_cells",
    ]
    return (
        baseline_tables.prototype_meaning.loc[:, columns]
        .drop_duplicates(subset=["proto_id"])
        .sort_values("proto_id")
        .reset_index(drop=True)
    )


def _patient_reference_table(
    baseline_tables: BaselineAnalysisTables,
) -> pd.DataFrame:
    """Return one row per patient in the all-prototype baseline table."""

    patients = (
        baseline_tables.all_prototype_baseline_patient_family["patient_id"]
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    return pd.DataFrame({"patient_id": patients})


def _confirmatory_family_presence(
    baseline_tables: BaselineAnalysisTables,
) -> pd.DataFrame:
    """Return per-patient confirmatory family availability flags."""

    family_counts = (
        baseline_tables.all_prototype_baseline_patient_family.loc[
            baseline_tables.all_prototype_baseline_patient_family["pair_family"].astype(str).isin(CONFIRMATORY_FAMILIES)
        ]
        .groupby("patient_id", sort=True, observed=False)["pair_family"]
        .nunique()
        .rename("confirmatory_family_count")
        .reset_index()
    )
    family_counts["has_both_confirmatory_families"] = (
        pd.to_numeric(family_counts["confirmatory_family_count"], errors="coerce")
        .fillna(0)
        .astype(int)
        .eq(len(CONFIRMATORY_FAMILIES))
    )
    return family_counts


def build_all_prototype_patient_recurrence_table(
    baseline_tables: BaselineAnalysisTables,
    transport_tables: TransportAnalysisTables,
) -> pd.DataFrame:
    """
    Build the internal all-prototype recurrence table.

    The recurrence row unit remains patient-by-prototype, with explicit family
    columns retained for confirmatory and audit-only scope.
    """

    proto_reference = _proto_reference_table(baseline_tables)
    patient_reference = _patient_reference_table(baseline_tables)
    patient_proto = patient_reference.assign(_merge_key=1).merge(
        proto_reference.assign(_merge_key=1),
        on="_merge_key",
        how="inner",
        validate="many_to_many",
    ).drop(columns="_merge_key")

    baseline_wide = _pivot_pair_family_values(
        baseline_tables.all_prototype_baseline_patient_family.loc[
            :,
            [
                "patient_id",
                "proto_id",
                "pair_family",
                "ordered_pair_count",
                "median_abs_delta_share",
                "mean_abs_delta_share",
                "median_abs_nonzero_delta_share",
                "prop_nonzero_share",
                "median_abs_delta_count",
                "mean_abs_delta_count",
                "median_abs_nonzero_delta_count",
                "prop_nonzero_count",
                "median_delta_share_context",
                "median_delta_count_context",
            ],
        ],
        index_columns=["patient_id", "proto_id"],
        value_columns=[
            "ordered_pair_count",
            "median_abs_delta_share",
            "mean_abs_delta_share",
            "median_abs_nonzero_delta_share",
            "prop_nonzero_share",
            "median_abs_delta_count",
            "mean_abs_delta_count",
            "median_abs_nonzero_delta_count",
            "prop_nonzero_count",
            "median_delta_share_context",
            "median_delta_count_context",
        ],
        prefix="baseline",
    )

    uot_transport_wide = _pivot_pair_family_values(
        transport_tables.all_prototype_uot_transport_patient_family.loc[
            :,
            [
                "patient_id",
                "proto_id",
                "pair_family",
                "ordered_pair_count",
                "transport_source_abs",
                "transport_source_share",
                "transport_target_abs",
                "transport_target_share",
            ],
        ],
        index_columns=["patient_id", "proto_id"],
        value_columns=[
            "ordered_pair_count",
            "transport_source_abs",
            "transport_source_share",
            "transport_target_abs",
            "transport_target_share",
        ],
        prefix="uot",
    )

    uot_unmatched_frame = transport_tables.all_prototype_uot_unmatched_patient_family.copy()
    uot_unmatched_frame["unmatched_abs"] = (
        pd.to_numeric(uot_unmatched_frame["destroy_abs"], errors="coerce").astype(float)
        + pd.to_numeric(uot_unmatched_frame["birth_abs"], errors="coerce").astype(float)
    )
    uot_unmatched_frame["unmatched_share"] = uot_unmatched_frame[
        ["destroy_share", "birth_share"]
    ].apply(pd.to_numeric, errors="coerce").max(axis=1)
    uot_unmatched_wide = _pivot_pair_family_values(
        uot_unmatched_frame.loc[
            :,
            [
                "patient_id",
                "proto_id",
                "pair_family",
                "destroy_abs",
                "destroy_share",
                "birth_abs",
                "birth_share",
                "unmatched_abs",
                "unmatched_share",
            ],
        ],
        index_columns=["patient_id", "proto_id"],
        value_columns=[
            "destroy_abs",
            "destroy_share",
            "birth_abs",
            "birth_share",
            "unmatched_abs",
            "unmatched_share",
        ],
        prefix="uot",
    )

    balanced_transport_wide = _pivot_pair_family_values(
        transport_tables.all_prototype_balanced_transport_patient_family.loc[
            :,
            [
                "patient_id",
                "proto_id",
                "pair_family",
                "ordered_pair_count",
                "transport_source_abs",
                "transport_source_share",
                "transport_target_abs",
                "transport_target_share",
            ],
        ],
        index_columns=["patient_id", "proto_id"],
        value_columns=[
            "ordered_pair_count",
            "transport_source_abs",
            "transport_source_share",
            "transport_target_abs",
            "transport_target_share",
        ],
        prefix="balanced",
    )

    delta_wide = _pivot_pair_family_values(
        transport_tables.all_prototype_ot_vs_uot_patient_family_delta.loc[
            :,
            [
                "patient_id",
                "proto_id",
                "pair_family",
                "delta_transport_source_abs",
                "delta_transport_source_share",
                "delta_transport_target_abs",
                "delta_transport_target_share",
            ],
        ],
        index_columns=["patient_id", "proto_id"],
        value_columns=[
            "delta_transport_source_abs",
            "delta_transport_source_share",
            "delta_transport_target_abs",
            "delta_transport_target_share",
        ],
        prefix="balanced_minus_uot",
    )

    recurrence = patient_proto.merge(
        _confirmatory_family_presence(baseline_tables),
        on="patient_id",
        how="left",
        validate="many_to_one",
    )
    recurrence["confirmatory_family_count"] = (
        pd.to_numeric(recurrence["confirmatory_family_count"], errors="coerce").fillna(0).astype(int)
    )
    recurrence["has_both_confirmatory_families"] = recurrence["has_both_confirmatory_families"].fillna(False).astype(bool)

    for wide in (
        baseline_wide,
        uot_transport_wide,
        uot_unmatched_wide,
        balanced_transport_wide,
        delta_wide,
    ):
        recurrence = recurrence.merge(
            wide,
            on=["patient_id", "proto_id"],
            how="left",
            validate="one_to_one",
        )

    recurrence["confirmatory_baseline_tc_pt_minus_tc_im_median_abs_delta_share"] = (
        pd.to_numeric(recurrence["baseline_median_abs_delta_share_tc_pt"], errors="coerce").astype(float)
        - pd.to_numeric(recurrence["baseline_median_abs_delta_share_tc_im"], errors="coerce").astype(float)
    )
    recurrence["confirmatory_baseline_tc_pt_minus_tc_im_median_abs_delta_count"] = (
        pd.to_numeric(recurrence["baseline_median_abs_delta_count_tc_pt"], errors="coerce").astype(float)
        - pd.to_numeric(recurrence["baseline_median_abs_delta_count_tc_im"], errors="coerce").astype(float)
    )
    recurrence["confirmatory_baseline_tc_pt_gt_tc_im_median_abs_delta_share_flag"] = (
        pd.to_numeric(
            recurrence["confirmatory_baseline_tc_pt_minus_tc_im_median_abs_delta_share"],
            errors="coerce",
        ).astype(float)
        > 0.0
    )
    recurrence["confirmatory_baseline_tc_pt_gt_tc_im_median_abs_delta_count_flag"] = (
        pd.to_numeric(
            recurrence["confirmatory_baseline_tc_pt_minus_tc_im_median_abs_delta_count"],
            errors="coerce",
        ).astype(float)
        > 0.0
    )
    for pair_family in CONFIRMATORY_FAMILIES:
        family_suffix = _family_suffix(pair_family)
        recurrence[f"confirmatory_uot_unmatched_positive_flag_{family_suffix}"] = (
            pd.to_numeric(recurrence[f"uot_unmatched_share_{family_suffix}"], errors="coerce").astype(float)
            > 0.0
        )
        recurrence[f"confirmatory_balanced_minus_uot_positive_flag_{family_suffix}"] = (
            pd.to_numeric(
                recurrence[f"balanced_minus_uot_delta_transport_source_share_{family_suffix}"],
                errors="coerce",
            ).astype(float)
            > 0.0
        )

    recurrence["audit_family_label"] = AUDIT_ONLY_FAMILIES[0]
    return recurrence.sort_values(["patient_id", "proto_id"]).reset_index(drop=True)


def validate_recurrence_table(
    all_prototype_patient_recurrence: pd.DataFrame,
    baseline_tables: BaselineAnalysisTables,
    transport_tables: TransportAnalysisTables,
) -> pd.DataFrame:
    """
    Validate the all-prototype recurrence table.

    This validates all-prototype coverage, uniqueness, confirmatory-only
    recurrence fields, and prototype-axis alignment with baseline/transport.
    """

    records: list[dict[str, object]] = []
    expected_patients = sorted(
        baseline_tables.all_prototype_baseline_patient_family["patient_id"].astype(str).unique().tolist()
    )
    expected_proto_ids = sorted(
        baseline_tables.prototype_meaning["proto_id"].astype(int).unique().tolist()
    )
    expected_rows = len(expected_patients) * len(expected_proto_ids)

    observed_rows = int(all_prototype_patient_recurrence.shape[0])
    observed_patients = sorted(all_prototype_patient_recurrence["patient_id"].astype(str).unique().tolist())
    observed_proto_ids = sorted(all_prototype_patient_recurrence["proto_id"].astype(int).unique().tolist())
    records.append(
        {
            "check": "all_prototype_coverage",
            "passed": bool(
                observed_rows == expected_rows
                and observed_patients == expected_patients
                and observed_proto_ids == expected_proto_ids
            ),
            "detail": (
                f"rows={observed_rows}, expected_rows={expected_rows}, "
                f"observed_patients={observed_patients}, expected_patients={expected_patients}, "
                f"observed_proto_ids={observed_proto_ids}, expected_proto_ids={expected_proto_ids}"
            ),
        }
    )

    unique_keys = (
        all_prototype_patient_recurrence.loc[:, ["patient_id", "proto_id"]]
        .drop_duplicates()
        .shape[0]
    )
    records.append(
        {
            "check": "one_row_per_patient_proto",
            "passed": bool(unique_keys == observed_rows),
            "detail": f"unique_keys={unique_keys}, rows={observed_rows}",
        }
    )

    confirmatory_columns = [
        column
        for column in all_prototype_patient_recurrence.columns.astype(str)
        if column.startswith("confirmatory_")
    ]
    confirmatory_columns_ok = not any("im_pt" in column for column in confirmatory_columns)
    records.append(
        {
            "check": "confirmatory_fields_exclude_im_pt",
            "passed": bool(confirmatory_columns_ok),
            "detail": f"confirmatory_columns={confirmatory_columns}",
        }
    )

    transport_proto_ids = sorted(
        transport_tables.all_prototype_uot_transport_patient_family["proto_id"].astype(int).unique().tolist()
    )
    transport_unmatched_proto_ids = sorted(
        transport_tables.all_prototype_uot_unmatched_patient_family["proto_id"].astype(int).unique().tolist()
    )
    balanced_proto_ids = sorted(
        transport_tables.all_prototype_balanced_transport_patient_family["proto_id"].astype(int).unique().tolist()
    )
    delta_proto_ids = sorted(
        transport_tables.all_prototype_ot_vs_uot_patient_family_delta["proto_id"].astype(int).unique().tolist()
    )
    axis_alignment_ok = bool(
        observed_proto_ids == expected_proto_ids
        and transport_proto_ids == expected_proto_ids
        and transport_unmatched_proto_ids == expected_proto_ids
        and balanced_proto_ids == expected_proto_ids
        and delta_proto_ids == expected_proto_ids
    )
    records.append(
        {
            "check": "prototype_axis_alignment",
            "passed": axis_alignment_ok,
            "detail": (
                f"recurrence={observed_proto_ids}, baseline={expected_proto_ids}, "
                f"uot_transport={transport_proto_ids}, "
                f"uot_unmatched={transport_unmatched_proto_ids}, "
                f"balanced={balanced_proto_ids}, delta={delta_proto_ids}"
            ),
        }
    )

    validation = pd.DataFrame.from_records(records)
    failed = validation.loc[~validation["passed"].astype(bool)].copy()
    if not failed.empty:
        details = "; ".join(f"{row['check']}={row['detail']}" for _, row in failed.iterrows())
        raise ValueError(f"Recurrence-table validation failed: {details}")
    return validation


def build_recurrence_tables(
    baseline_tables: BaselineAnalysisTables,
    transport_tables: TransportAnalysisTables,
) -> RecurrenceAnalysisTables:
    """Assemble the recurrence-side all-prototype analysis bundle."""

    all_prototype_patient_recurrence = build_all_prototype_patient_recurrence_table(
        baseline_tables=baseline_tables,
        transport_tables=transport_tables,
    )
    recurrence_validation = validate_recurrence_table(
        all_prototype_patient_recurrence=all_prototype_patient_recurrence,
        baseline_tables=baseline_tables,
        transport_tables=transport_tables,
    )
    return RecurrenceAnalysisTables(
        all_prototype_patient_recurrence=all_prototype_patient_recurrence,
        recurrence_validation=recurrence_validation,
    )
