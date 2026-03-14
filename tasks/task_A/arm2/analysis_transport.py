"""
Module: tasks.task_A.arm2.analysis_transport

All-prototype transport and unmatched summary layer for the post-hoc Arm-II
focused rewrite.

This module must summarize transport behavior from the compute-layer surfaces
without rerunning solver internals.

Scope rules for implementation:
- Confirmatory contrasts are restricted to TC-IM and TC-PT.
- IM-PT remains audit-only / exploratory.
- Upstream prototype filtering is forbidden. All active prototypes must remain
  present in the internal transport and unmatched tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis_contract import (
    CONFIRMATORY_FAMILIES,
    ComputedArm2Surfaces,
    LoadedArm2Inputs,
    PAIR_FAMILY_ORDER,
    PAIR_FAMILY_ROLE,
    PairPrototypeTransportSurface,
    PairPrototypeUnmatchedSurface,
    TransportAnalysisTables,
)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide with NaN on zero denominators to preserve scale semantics."""

    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=float),
        where=denominator > 0.0,
    )


def _expected_proto_ids(prototype_meaning: pd.DataFrame) -> np.ndarray:
    """Return the sorted prototype IDs represented by the biological meaning table."""

    proto_ids = np.sort(prototype_meaning["proto_id"].astype(int).to_numpy())
    if proto_ids.size == 0:
        raise ValueError("Prototype meaning table is empty")
    if not np.array_equal(proto_ids, np.unique(proto_ids)):
        raise ValueError("Prototype meaning table contains duplicate proto_id values")
    return proto_ids


def _confirmatory_paired_patient_ids(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return patients with both confirmatory families present."""

    confirmatory = frame.loc[
        frame["pair_family"].astype(str).isin(CONFIRMATORY_FAMILIES)
    ].copy()
    if confirmatory.empty:
        return ()
    family_counts = (
        confirmatory.groupby("patient_id", sort=True, observed=True)["pair_family"]
        .nunique()
        .astype(int)
    )
    return tuple(
        sorted(
            family_counts.loc[family_counts.eq(len(CONFIRMATORY_FAMILIES))].index.astype(str).tolist()
        )
    )


def _pivot_confirmatory_family_values(
    frame: pd.DataFrame,
    *,
    index_columns: list[str],
    value_columns: list[str],
) -> pd.DataFrame:
    """Pivot confirmatory family values into stable `tc_im_*` / `tc_pt_*` columns."""

    expected_columns = [
        f"{str(pair_family).lower().replace('-', '_')}_{value_column}"
        for pair_family in CONFIRMATORY_FAMILIES
        for value_column in value_columns
    ]
    if frame.empty:
        return pd.DataFrame(columns=[*index_columns, *expected_columns])

    confirmatory = frame.loc[
        frame["pair_family"].astype(str).isin(CONFIRMATORY_FAMILIES)
    ].copy()
    if confirmatory.empty:
        return pd.DataFrame(columns=[*index_columns, *expected_columns])

    pivot = confirmatory.pivot(
        index=index_columns,
        columns="pair_family",
        values=value_columns,
    )
    pivot.columns = [
        f"{str(pair_family).lower().replace('-', '_')}_{value_column}"
        for value_column, pair_family in pivot.columns
    ]
    pivot = pivot.reset_index()
    for column in expected_columns:
        if column not in pivot.columns:
            pivot[column] = np.nan
    return pivot.loc[:, [*index_columns, *expected_columns]]


def _aggregate_transport_surface(
    pair_level_transport: pd.DataFrame,
    surface: PairPrototypeTransportSurface,
    prototype_meaning: pd.DataFrame,
    *,
    transport_method: str,
) -> pd.DataFrame:
    """Aggregate a pair-by-prototype transport surface to patient-family-prototype scope."""

    proto_ids = _expected_proto_ids(prototype_meaning)
    if not np.array_equal(np.asarray(surface.proto_ids, dtype=int), proto_ids):
        raise ValueError("Transport surface proto_ids do not match the prototype meaning table")

    records: list[dict[str, object]] = []
    grouped = pair_level_transport.groupby(["patient_id", "pair_family"], sort=True, observed=True).indices
    for (patient_id, pair_family), idx in grouped.items():
        row_idx = np.asarray(idx, dtype=int)
        source_abs = np.nansum(surface.source_abs[row_idx], axis=0)
        target_abs = np.nansum(surface.target_abs[row_idx], axis=0)
        source_total = float(np.sum(source_abs, dtype=float))
        target_total = float(np.sum(target_abs, dtype=float))
        source_share = _safe_divide(source_abs, source_total)
        target_share = _safe_divide(target_abs, target_total)

        for offset, proto_id in enumerate(proto_ids.tolist()):
            records.append(
                {
                    "patient_id": str(patient_id),
                    "pair_family": str(pair_family),
                    "pair_family_role": PAIR_FAMILY_ROLE[str(pair_family)],
                    "proto_id": int(proto_id),
                    "ordered_pair_count": int(row_idx.size),
                    "transport_method": transport_method,
                    "transport_source_total_abs": source_total,
                    "transport_target_total_abs": target_total,
                    "transport_source_abs": float(source_abs[offset]),
                    "transport_source_share": float(source_share[offset]),
                    "transport_target_abs": float(target_abs[offset]),
                    "transport_target_share": float(target_share[offset]),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "pair_family",
                "pair_family_role",
                "proto_id",
                "ordered_pair_count",
                "transport_method",
                "transport_source_total_abs",
                "transport_target_total_abs",
                "transport_source_abs",
                "transport_source_share",
                "transport_target_abs",
                "transport_target_share",
                *prototype_meaning.columns.tolist(),
            ]
        )

    table = pd.DataFrame.from_records(records).merge(
        prototype_meaning,
        on="proto_id",
        how="left",
        validate="many_to_one",
    )
    table["pair_family"] = pd.Categorical(
        table["pair_family"],
        categories=PAIR_FAMILY_ORDER,
        ordered=True,
    )
    return table.sort_values(["patient_id", "pair_family", "proto_id"]).reset_index(drop=True)


def _aggregate_unmatched_surface(
    pair_level_transport: pd.DataFrame,
    surface: PairPrototypeUnmatchedSurface,
    prototype_meaning: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate the UOT unmatched surface to patient-family-prototype scope."""

    proto_ids = _expected_proto_ids(prototype_meaning)
    if not np.array_equal(np.asarray(surface.proto_ids, dtype=int), proto_ids):
        raise ValueError("Unmatched surface proto_ids do not match the prototype meaning table")

    records: list[dict[str, object]] = []
    grouped = pair_level_transport.groupby(["patient_id", "pair_family"], sort=True, observed=True).indices
    for (patient_id, pair_family), idx in grouped.items():
        row_idx = np.asarray(idx, dtype=int)
        destroy_abs = np.nansum(surface.destroy_abs[row_idx], axis=0)
        birth_abs = np.nansum(surface.birth_abs[row_idx], axis=0)
        destroy_total = float(np.sum(destroy_abs, dtype=float))
        birth_total = float(np.sum(birth_abs, dtype=float))
        destroy_share = _safe_divide(destroy_abs, destroy_total)
        birth_share = _safe_divide(birth_abs, birth_total)

        for offset, proto_id in enumerate(proto_ids.tolist()):
            records.append(
                {
                    "patient_id": str(patient_id),
                    "pair_family": str(pair_family),
                    "pair_family_role": PAIR_FAMILY_ROLE[str(pair_family)],
                    "proto_id": int(proto_id),
                    "ordered_pair_count": int(row_idx.size),
                    "destroy_total_abs": destroy_total,
                    "birth_total_abs": birth_total,
                    "destroy_abs": float(destroy_abs[offset]),
                    "destroy_share": float(destroy_share[offset]),
                    "birth_abs": float(birth_abs[offset]),
                    "birth_share": float(birth_share[offset]),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "pair_family",
                "pair_family_role",
                "proto_id",
                "ordered_pair_count",
                "destroy_total_abs",
                "birth_total_abs",
                "destroy_abs",
                "destroy_share",
                "birth_abs",
                "birth_share",
                *prototype_meaning.columns.tolist(),
            ]
        )

    table = pd.DataFrame.from_records(records).merge(
        prototype_meaning,
        on="proto_id",
        how="left",
        validate="many_to_one",
    )
    table["pair_family"] = pd.Categorical(
        table["pair_family"],
        categories=PAIR_FAMILY_ORDER,
        ordered=True,
    )
    return table.sort_values(["patient_id", "pair_family", "proto_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Global scalar transport summary
# ---------------------------------------------------------------------------


def build_global_transport_summary(pair_level_transport: pd.DataFrame) -> pd.DataFrame:
    """
    Build the public confirmatory global transport summary at patient scope.

    This output is confirmatory-only in its contrasts even if the underlying
    pair-level transport frame retains IM-PT for audit purposes.
    """

    output_columns = [
        "patient_id",
        "tc_im_ordered_pair_count",
        "tc_pt_ordered_pair_count",
        "tc_im_valid_uot_pair_count",
        "tc_pt_valid_uot_pair_count",
        "tc_im_median_U_abs",
        "tc_pt_median_U_abs",
        "patient_median_tc_pt_minus_tc_im_U_abs",
        "tc_im_median_transport_fraction",
        "tc_pt_median_transport_fraction",
        "patient_median_tc_pt_minus_tc_im_transport_fraction",
        "tc_im_median_unmatched_fraction",
        "tc_pt_median_unmatched_fraction",
        "patient_median_tc_pt_minus_tc_im_unmatched_fraction",
        "tc_im_median_M",
        "tc_pt_median_M",
        "patient_median_tc_pt_minus_tc_im_M",
        "tc_im_median_D_pos",
        "tc_pt_median_D_pos",
        "patient_median_tc_pt_minus_tc_im_D_pos",
        "tc_im_median_B_pos",
        "tc_pt_median_B_pos",
        "patient_median_tc_pt_minus_tc_im_B_pos",
        "tc_im_median_T_abs",
        "tc_pt_median_T_abs",
        "patient_median_tc_pt_minus_tc_im_T_abs",
        "tc_im_median_M_balanced",
        "tc_pt_median_M_balanced",
        "patient_median_tc_pt_minus_tc_im_M_balanced",
        "tc_im_median_balanced_minus_uot",
        "tc_pt_median_balanced_minus_uot",
        "patient_median_tc_pt_minus_tc_im_balanced_minus_uot",
    ]

    paired_patient_ids = _confirmatory_paired_patient_ids(pair_level_transport)
    confirmatory = pair_level_transport.loc[
        pair_level_transport["patient_id"].astype(str).isin(paired_patient_ids)
        & pair_level_transport["pair_family"].astype(str).isin(CONFIRMATORY_FAMILIES)
    ].copy()
    if confirmatory.empty:
        return pd.DataFrame(columns=output_columns)

    patient_family_summary = (
        confirmatory.groupby(["patient_id", "pair_family"], sort=True, observed=True)
        .agg(
            ordered_pair_count=("pair_id", "size"),
            valid_uot_pair_count=("uot_status", lambda s: int(pd.Series(s).astype(str).eq("ok").sum())),
            median_U_abs=("U_abs", "median"),
            median_transport_fraction=("transport_fraction", "median"),
            median_unmatched_fraction=("unmatched_fraction", "median"),
            median_M=("M", "median"),
            median_D_pos=("D_pos", "median"),
            median_B_pos=("B_pos", "median"),
            median_T_abs=("T_abs", "median"),
            median_M_balanced=("M_balanced", "median"),
            median_balanced_minus_uot=("balanced_minus_uot", "median"),
        )
        .reset_index()
    )
    wide = _pivot_confirmatory_family_values(
        patient_family_summary,
        index_columns=["patient_id"],
        value_columns=[
            "ordered_pair_count",
            "valid_uot_pair_count",
            "median_U_abs",
            "median_transport_fraction",
            "median_unmatched_fraction",
            "median_M",
            "median_D_pos",
            "median_B_pos",
            "median_T_abs",
            "median_M_balanced",
            "median_balanced_minus_uot",
        ],
    )
    for metric in (
        "U_abs",
        "transport_fraction",
        "unmatched_fraction",
        "M",
        "D_pos",
        "B_pos",
        "T_abs",
        "M_balanced",
        "balanced_minus_uot",
    ):
        wide[f"patient_median_tc_pt_minus_tc_im_{metric}"] = (
            pd.to_numeric(wide[f"tc_pt_median_{metric}"], errors="coerce").astype(float)
            - pd.to_numeric(wide[f"tc_im_median_{metric}"], errors="coerce").astype(float)
        )
    return wide.loc[:, output_columns].sort_values("patient_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# All-prototype patient-family transport/unmatched tables
# ---------------------------------------------------------------------------


def build_all_prototype_uot_transport_patient_family_table(
    inputs: LoadedArm2Inputs,
    computed: ComputedArm2Surfaces,
    prototype_meaning: pd.DataFrame,
) -> pd.DataFrame:
    """Build the all-prototype UOT transport table at patient-by-family-by-prototype scope."""

    del inputs
    return _aggregate_transport_surface(
        computed.pair_level_transport,
        computed.uot_transport_surface,
        prototype_meaning,
        transport_method="uot",
    )


def build_all_prototype_uot_unmatched_patient_family_table(
    inputs: LoadedArm2Inputs,
    computed: ComputedArm2Surfaces,
    prototype_meaning: pd.DataFrame,
) -> pd.DataFrame:
    """Build the all-prototype UOT unmatched table at patient-by-family-by-prototype scope."""

    del inputs
    return _aggregate_unmatched_surface(
        computed.pair_level_transport,
        computed.uot_unmatched_surface,
        prototype_meaning,
    )


def build_all_prototype_balanced_transport_patient_family_table(
    inputs: LoadedArm2Inputs,
    computed: ComputedArm2Surfaces,
    prototype_meaning: pd.DataFrame,
) -> pd.DataFrame:
    """Build the all-prototype Balanced-OT transport table at patient-by-family-by-prototype scope."""

    del inputs
    return _aggregate_transport_surface(
        computed.pair_level_transport,
        computed.balanced_transport_surface,
        prototype_meaning,
        transport_method="balanced",
    )


def build_all_prototype_ot_vs_uot_patient_family_delta_table(
    all_prototype_uot_transport_patient_family: pd.DataFrame,
    all_prototype_balanced_transport_patient_family: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the all-prototype OT-vs-UOT delta table at patient-by-family-by-
    prototype scope.

    This should remain a comparable transport-side delta layer only.
    """

    key_columns = [
        "patient_id",
        "pair_family",
        "pair_family_role",
        "proto_id",
        "ordered_pair_count",
        "dominant_cell_type",
        "dominant_cell_type_fraction",
        "top_cell_type_mix",
        "total_cells",
    ]
    if all_prototype_uot_transport_patient_family.empty or all_prototype_balanced_transport_patient_family.empty:
        return pd.DataFrame(
            columns=[
                *key_columns,
                "transport_method_uot",
                "transport_source_total_abs_uot",
                "transport_target_total_abs_uot",
                "transport_source_abs_uot",
                "transport_source_share_uot",
                "transport_target_abs_uot",
                "transport_target_share_uot",
                "transport_method_balanced",
                "transport_source_total_abs_balanced",
                "transport_target_total_abs_balanced",
                "transport_source_abs_balanced",
                "transport_source_share_balanced",
                "transport_target_abs_balanced",
                "transport_target_share_balanced",
                "delta_transport_source_total_abs",
                "delta_transport_target_total_abs",
                "delta_transport_source_abs",
                "delta_transport_source_share",
                "delta_transport_target_abs",
                "delta_transport_target_share",
            ]
        )
    merged = all_prototype_uot_transport_patient_family.merge(
        all_prototype_balanced_transport_patient_family,
        on=key_columns,
        how="inner",
        suffixes=("_uot", "_balanced"),
        validate="one_to_one",
    )
    for quantity in (
        "transport_source_total_abs",
        "transport_target_total_abs",
        "transport_source_abs",
        "transport_source_share",
        "transport_target_abs",
        "transport_target_share",
    ):
        merged[f"delta_{quantity}"] = (
            pd.to_numeric(merged[f"{quantity}_balanced"], errors="coerce").astype(float)
            - pd.to_numeric(merged[f"{quantity}_uot"], errors="coerce").astype(float)
        )
    merged["pair_family"] = pd.Categorical(
        merged["pair_family"],
        categories=PAIR_FAMILY_ORDER,
        ordered=True,
    )
    return merged.sort_values(["patient_id", "pair_family", "proto_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Transport validation and bundle assembly
# ---------------------------------------------------------------------------


def validate_transport_tables(
    inputs: LoadedArm2Inputs,
    computed: ComputedArm2Surfaces,
    prototype_meaning: pd.DataFrame,
    global_transport_summary: pd.DataFrame,
    all_prototype_uot_transport_patient_family: pd.DataFrame,
    all_prototype_uot_unmatched_patient_family: pd.DataFrame,
    all_prototype_balanced_transport_patient_family: pd.DataFrame,
    all_prototype_ot_vs_uot_patient_family_delta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate the transport-side all-prototype analysis outputs.

    Implementation notes:
    - Confirmatory contrasts must use only TC-IM and TC-PT.
    - IM-PT may remain available in audit/internal layers.
    - Balanced OT must remain transport-only with no unmatched semantics.
    """

    records: list[dict[str, object]] = []
    proto_ids = _expected_proto_ids(prototype_meaning)
    patient_family_count = int(
        computed.pair_level_transport.loc[:, ["patient_id", "pair_family"]]
        .drop_duplicates()
        .shape[0]
    )
    expected_rows = patient_family_count * proto_ids.size

    for check_name, frame in (
        ("all_prototype_coverage_uot_transport", all_prototype_uot_transport_patient_family),
        ("all_prototype_coverage_uot_unmatched", all_prototype_uot_unmatched_patient_family),
        ("all_prototype_coverage_balanced_transport", all_prototype_balanced_transport_patient_family),
        ("all_prototype_coverage_ot_vs_uot_delta", all_prototype_ot_vs_uot_patient_family_delta),
    ):
        observed_proto_ids = sorted(frame["proto_id"].astype(int).unique().tolist()) if not frame.empty else []
        records.append(
            {
                "check": check_name,
                "passed": bool(frame.shape[0] == expected_rows and observed_proto_ids == proto_ids.tolist()),
                "detail": (
                    f"rows={frame.shape[0]}, expected_rows={expected_rows}, "
                    f"observed_proto_ids={observed_proto_ids}, expected_proto_ids={proto_ids.tolist()}"
                ),
            }
        )

    paired_patient_ids = list(_confirmatory_paired_patient_ids(computed.pair_level_transport))
    confirmatory_columns_ok = bool(
        not any("im_pt" in str(column) for column in global_transport_summary.columns)
    )
    patient_filter_ok = (
        sorted(global_transport_summary["patient_id"].astype(str).tolist()) == paired_patient_ids
        if not global_transport_summary.empty
        else not paired_patient_ids
    )
    records.append(
        {
            "check": "global_summary_confirmatory_filtering",
            "passed": confirmatory_columns_ok and patient_filter_ok,
            "detail": (
                f"paired_patients={paired_patient_ids}, "
                f"summary_patients={global_transport_summary.get('patient_id', pd.Series(dtype=object)).astype(str).tolist()}, "
                f"has_im_pt_columns={not confirmatory_columns_ok}"
            ),
        }
    )

    combined = (
        pd.to_numeric(computed.pair_level_transport["transport_fraction"], errors="coerce").astype(float)
        + pd.to_numeric(computed.pair_level_transport["unmatched_fraction"], errors="coerce").astype(float)
    )
    valid_combined = combined.dropna().to_numpy(dtype=float)
    transport_fraction_ok = bool(np.allclose(valid_combined, 1.0, atol=1e-6))
    records.append(
        {
            "check": "pair_level_transport_fraction_partition",
            "passed": transport_fraction_ok,
            "detail": f"defined_row_count={int(valid_combined.shape[0])}",
        }
    )

    valid_uot = computed.pair_level_transport["uot_status"].astype(str).eq("ok")
    observed_u = pd.to_numeric(
        computed.pair_level_transport.loc[valid_uot, "U_abs"],
        errors="coerce",
    ).to_numpy(dtype=float)
    reconstructed_u = (
        pd.to_numeric(computed.pair_level_transport.loc[valid_uot, "B_pos"], errors="coerce").to_numpy(dtype=float)
        + pd.to_numeric(computed.pair_level_transport.loc[valid_uot, "D_pos"], errors="coerce").to_numpy(dtype=float)
    )
    records.append(
        {
            "check": "pair_level_u_abs_matches_birth_plus_destroy",
            "passed": bool(np.allclose(observed_u, reconstructed_u, atol=1e-9, equal_nan=True)),
            "detail": f"valid_uot_row_count={int(valid_uot.sum())}",
        }
    )

    forbidden_balanced_tokens = ("destroy", "birth", "unmatched", "U_abs", "D_pos", "B_pos")
    balanced_has_unmatched_terms = any(
        any(token in str(column) for token in forbidden_balanced_tokens)
        for column in all_prototype_balanced_transport_patient_family.columns.astype(str)
    )
    records.append(
        {
            "check": "balanced_transport_has_no_unmatched_semantics",
            "passed": not balanced_has_unmatched_terms,
            "detail": (
                f"columns={all_prototype_balanced_transport_patient_family.columns.astype(str).tolist()}"
            ),
        }
    )

    pair_counts = (
        computed.pair_level_transport.groupby(["patient_id", "pair_family"], sort=True, observed=True)
        .size()
        .rename("expected_ordered_pair_count")
        .reset_index()
    )
    pair_alignment_ok = True
    pair_alignment_details: list[str] = []
    for label, frame in (
        ("uot_transport", all_prototype_uot_transport_patient_family),
        ("uot_unmatched", all_prototype_uot_unmatched_patient_family),
        ("balanced_transport", all_prototype_balanced_transport_patient_family),
    ):
        merged = frame.merge(
            pair_counts,
            on=["patient_id", "pair_family"],
            how="left",
            validate="many_to_one",
        )
        aligned = bool(
            (
                pd.to_numeric(merged["ordered_pair_count"], errors="coerce").astype(float)
                == pd.to_numeric(merged["expected_ordered_pair_count"], errors="coerce").astype(float)
            ).all()
        )
        pair_alignment_ok = pair_alignment_ok and aligned
        pair_alignment_details.append(f"{label}={aligned}")
    records.append(
        {
            "check": "pair_level_to_patient_family_pair_count_alignment",
            "passed": pair_alignment_ok,
            "detail": ", ".join(pair_alignment_details),
        }
    )

    proto_axis_ok = all(
        sorted(frame["proto_id"].astype(int).unique().tolist()) == proto_ids.tolist()
        for frame in (
            all_prototype_uot_transport_patient_family,
            all_prototype_uot_unmatched_patient_family,
            all_prototype_balanced_transport_patient_family,
            all_prototype_ot_vs_uot_patient_family_delta,
        )
        if not frame.empty
    )
    records.append(
        {
            "check": "transport_table_proto_axis_alignment",
            "passed": proto_axis_ok,
            "detail": f"expected_proto_ids={proto_ids.tolist()}",
        }
    )

    proto_tensor_ok = bool(
        np.array_equal(np.asarray(computed.uot_transport_surface.proto_ids, dtype=int), proto_ids)
        and np.array_equal(np.asarray(computed.uot_unmatched_surface.proto_ids, dtype=int), proto_ids)
        and np.array_equal(np.asarray(computed.balanced_transport_surface.proto_ids, dtype=int), proto_ids)
        and computed.uot_transport_surface.source_abs.shape[1] == proto_ids.size
        and computed.uot_unmatched_surface.destroy_abs.shape[1] == proto_ids.size
        and computed.balanced_transport_surface.source_abs.shape[1] == proto_ids.size
        and np.array_equal(np.asarray(inputs.stage0.active_prototype_ids, dtype=int), proto_ids)
    )
    records.append(
        {
            "check": "proto_id_tensor_axis_consistency",
            "passed": proto_tensor_ok,
            "detail": (
                f"stage0_proto_ids={np.asarray(inputs.stage0.active_prototype_ids, dtype=int).tolist()}, "
                f"surface_proto_ids={proto_ids.tolist()}"
            ),
        }
    )

    validation = pd.DataFrame.from_records(records)
    failed = validation.loc[~validation["passed"].astype(bool)].copy()
    if not failed.empty:
        details = "; ".join(f"{row['check']}={row['detail']}" for _, row in failed.iterrows())
        raise ValueError(f"Transport-table validation failed: {details}")
    return validation


def build_transport_tables(
    inputs: LoadedArm2Inputs,
    computed: ComputedArm2Surfaces,
    prototype_meaning: pd.DataFrame,
) -> TransportAnalysisTables:
    """
    Assemble the full transport-side all-prototype analysis bundle.

    This top-level transport builder should be the only transport entrypoint
    used by recurrence, downstream views, and output assembly.
    """

    global_transport_summary = build_global_transport_summary(computed.pair_level_transport)
    all_prototype_uot_transport_patient_family = (
        build_all_prototype_uot_transport_patient_family_table(
            inputs=inputs,
            computed=computed,
            prototype_meaning=prototype_meaning,
        )
    )
    all_prototype_uot_unmatched_patient_family = (
        build_all_prototype_uot_unmatched_patient_family_table(
            inputs=inputs,
            computed=computed,
            prototype_meaning=prototype_meaning,
        )
    )
    all_prototype_balanced_transport_patient_family = (
        build_all_prototype_balanced_transport_patient_family_table(
            inputs=inputs,
            computed=computed,
            prototype_meaning=prototype_meaning,
        )
    )
    all_prototype_ot_vs_uot_patient_family_delta = (
        build_all_prototype_ot_vs_uot_patient_family_delta_table(
            all_prototype_uot_transport_patient_family=all_prototype_uot_transport_patient_family,
            all_prototype_balanced_transport_patient_family=all_prototype_balanced_transport_patient_family,
        )
    )
    transport_validation = validate_transport_tables(
        inputs=inputs,
        computed=computed,
        prototype_meaning=prototype_meaning,
        global_transport_summary=global_transport_summary,
        all_prototype_uot_transport_patient_family=all_prototype_uot_transport_patient_family,
        all_prototype_uot_unmatched_patient_family=all_prototype_uot_unmatched_patient_family,
        all_prototype_balanced_transport_patient_family=all_prototype_balanced_transport_patient_family,
        all_prototype_ot_vs_uot_patient_family_delta=all_prototype_ot_vs_uot_patient_family_delta,
    )
    return TransportAnalysisTables(
        global_transport_summary=global_transport_summary,
        all_prototype_uot_transport_patient_family=all_prototype_uot_transport_patient_family,
        all_prototype_uot_unmatched_patient_family=all_prototype_uot_unmatched_patient_family,
        all_prototype_balanced_transport_patient_family=all_prototype_balanced_transport_patient_family,
        all_prototype_ot_vs_uot_patient_family_delta=all_prototype_ot_vs_uot_patient_family_delta,
        transport_validation=transport_validation,
    )
