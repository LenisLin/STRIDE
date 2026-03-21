"""
Module: tasks.task_A.arm2.analysis_baseline

All-prototype-first baseline layer for the post-hoc Arm-II focused rewrite.

This module implements the first real Arm-II focused-analysis block:
- prototype biological meaning,
- pair-level baseline audit,
- raw ordered-pair x prototype baseline values,
- internal patient x family x prototype baseline aggregation,
- confirmatory baseline summaries,
- explicit baseline validations.

Transport, unmatched, recurrence, extracted views, and final full-package
writing remain intentionally out of scope for this step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis_contract import (
    BaselineAnalysisTables,
    CONFIRMATORY_FAMILIES,
    LoadedArm2Inputs,
    PAIR_FAMILY_ORDER,
    PROTOTYPE_ANNOTATION_COLUMNS,
    PROTOTYPE_ANNOTATION_VALUE_COLUMNS,
    Stage0AnalysisBundle,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide with explicit NaN on zero denominator to preserve scale semantics."""

    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=float),
        where=denominator > 0.0,
    )


def _format_fraction(value: float) -> str:
    """Render compact fraction labels for prototype cell-type mixtures."""

    text = f"{float(value):.3f}"
    return text.rstrip("0").rstrip(".")


def _format_label_fraction(value: float) -> str:
    """Render compact two-decimal fractions for prototype top-3 labels."""

    return f"{float(value):.2f}"


def _expected_prototype_ids(prototype_meaning: pd.DataFrame) -> np.ndarray:
    """Return the sorted prototype IDs represented by the meaning table."""

    return np.sort(prototype_meaning["proto_id"].astype(int).to_numpy())


def _validate_prototype_meaning_coverage(
    prototype_meaning: pd.DataFrame,
    active_prototype_ids: np.ndarray,
) -> None:
    """Require the biological-meaning table to span the active Stage-0 axis."""

    expected = sorted(np.asarray(active_prototype_ids, dtype=int).tolist())
    observed = sorted(prototype_meaning["proto_id"].astype(int).tolist())
    if observed != expected:
        raise ValueError(
            "Prototype-meaning table does not cover the active prototype axis: "
            f"expected={expected}, observed={observed}"
        )


def _confirmatory_paired_patient_ids(
    all_prototype_baseline_patient_family: pd.DataFrame,
) -> tuple[str, ...]:
    """Return patients with both confirmatory Arm-II families present."""

    confirmatory = all_prototype_baseline_patient_family.loc[
        all_prototype_baseline_patient_family["pair_family"].astype(str).isin(CONFIRMATORY_FAMILIES)
    ].copy()
    if confirmatory.empty:
        return ()
    family_counts = (
        confirmatory.groupby("patient_id", sort=True, observed=False)["pair_family"]
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


def _summarize_scale_relationships(frame: pd.DataFrame) -> tuple[bool, str]:
    """Validate that count-scale and share-scale columns remain mathematically separated."""

    checks = {
        "delta_count_reconstructs": np.allclose(
            pd.to_numeric(frame["delta_count"], errors="coerce").to_numpy(dtype=float),
            (
                pd.to_numeric(frame["target_count"], errors="coerce").to_numpy(dtype=float)
                - pd.to_numeric(frame["source_count"], errors="coerce").to_numpy(dtype=float)
            ),
            atol=1e-9,
            equal_nan=True,
        ),
        "abs_delta_count_reconstructs": np.allclose(
            pd.to_numeric(frame["abs_delta_count"], errors="coerce").to_numpy(dtype=float),
            np.abs(pd.to_numeric(frame["delta_count"], errors="coerce").to_numpy(dtype=float)),
            atol=1e-9,
            equal_nan=True,
        ),
        "delta_share_reconstructs": np.allclose(
            pd.to_numeric(frame["delta_share"], errors="coerce").to_numpy(dtype=float),
            (
                pd.to_numeric(frame["target_share"], errors="coerce").to_numpy(dtype=float)
                - pd.to_numeric(frame["source_share"], errors="coerce").to_numpy(dtype=float)
            ),
            atol=1e-9,
            equal_nan=True,
        ),
        "abs_delta_share_reconstructs": np.allclose(
            pd.to_numeric(frame["abs_delta_share"], errors="coerce").to_numpy(dtype=float),
            np.abs(pd.to_numeric(frame["delta_share"], errors="coerce").to_numpy(dtype=float)),
            atol=1e-9,
            equal_nan=True,
        ),
        "source_share_from_counts": np.allclose(
            pd.to_numeric(frame["source_share"], errors="coerce").to_numpy(dtype=float),
            _safe_divide(
                pd.to_numeric(frame["source_count"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(frame["source_total_count"], errors="coerce").to_numpy(dtype=float),
            ),
            atol=1e-9,
            equal_nan=True,
        ),
        "target_share_from_counts": np.allclose(
            pd.to_numeric(frame["target_share"], errors="coerce").to_numpy(dtype=float),
            _safe_divide(
                pd.to_numeric(frame["target_count"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(frame["target_total_count"], errors="coerce").to_numpy(dtype=float),
            ),
            atol=1e-9,
            equal_nan=True,
        ),
        "share_bounds_hold": bool(
            pd.to_numeric(frame["source_share"], errors="coerce").dropna().between(0.0, 1.0).all()
            and pd.to_numeric(frame["target_share"], errors="coerce").dropna().between(0.0, 1.0).all()
            and pd.to_numeric(frame["delta_share"], errors="coerce").dropna().between(-1.0, 1.0).all()
            and pd.to_numeric(frame["abs_delta_share"], errors="coerce").dropna().between(0.0, 1.0).all()
        ),
    }
    return bool(all(checks.values())), ", ".join(f"{name}={value}" for name, value in checks.items())


def _nonzero_abs_median(values: pd.Series) -> float:
    """Median absolute magnitude over nonzero entries only."""

    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float).abs()
    nonzero = numeric.loc[numeric > 0.0]
    if nonzero.empty:
        return 0.0
    return float(nonzero.median())


def _prop_nonzero(values: pd.Series) -> float:
    """Fraction of nonzero entries in a raw ordered-pair series."""

    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if numeric.empty:
        return np.nan
    return float((numeric.abs() > 0.0).mean())


def _summarize_internal_summary_table(frame: pd.DataFrame) -> tuple[bool, str]:
    """Validate the refined summary-over-pairs internal baseline table."""

    checks = {
        "ordered_pair_count_positive": bool(
            pd.to_numeric(frame["ordered_pair_count"], errors="coerce").ge(1).all()
        ),
        "share_summary_bounds_hold": bool(
            pd.to_numeric(frame["median_abs_delta_share"], errors="coerce").dropna().between(0.0, 1.0).all()
            and pd.to_numeric(frame["mean_abs_delta_share"], errors="coerce").dropna().between(0.0, 1.0).all()
            and pd.to_numeric(frame["median_abs_nonzero_delta_share"], errors="coerce").dropna().between(0.0, 1.0).all()
            and pd.to_numeric(frame["median_delta_share_context"], errors="coerce").dropna().between(-1.0, 1.0).all()
            and pd.to_numeric(frame["prop_nonzero_share"], errors="coerce").dropna().between(0.0, 1.0).all()
        ),
        "count_summaries_nonnegative": bool(
            pd.to_numeric(frame["median_abs_delta_count"], errors="coerce").ge(0.0).all()
            and pd.to_numeric(frame["mean_abs_delta_count"], errors="coerce").ge(0.0).all()
            and pd.to_numeric(frame["median_abs_nonzero_delta_count"], errors="coerce").ge(0.0).all()
            and pd.to_numeric(frame["prop_nonzero_count"], errors="coerce").dropna().between(0.0, 1.0).all()
        ),
        "nonzero_share_summary_not_below_all_pairs": bool(
            (
                pd.to_numeric(frame["median_abs_nonzero_delta_share"], errors="coerce").astype(float)
                + 1e-12
                >= pd.to_numeric(frame["median_abs_delta_share"], errors="coerce").astype(float)
            ).all()
        ),
        "nonzero_count_summary_not_below_all_pairs": bool(
            (
                pd.to_numeric(frame["median_abs_nonzero_delta_count"], errors="coerce").astype(float)
                + 1e-12
                >= pd.to_numeric(frame["median_abs_delta_count"], errors="coerce").astype(float)
            ).all()
        ),
    }
    return bool(all(checks.values())), ", ".join(f"{name}={value}" for name, value in checks.items())


# ---------------------------------------------------------------------------
# Prototype-meaning layer
# ---------------------------------------------------------------------------


def build_prototype_meaning_table(stage0_bundle: Stage0AnalysisBundle) -> pd.DataFrame:
    """
    Build the all-prototype biological meaning table from the frozen Stage-0
    bundle.

    Required columns:
    - `proto_id`
    - `dominant_cell_type`
    - `top_cell_type_mix`
    - `total_cells`

    `broad_group` is intentionally omitted in this step because the focused
    rewrite does not yet declare a separate stable grouping basis beyond the
    local cell-type composition table itself.
    """

    cell_type_table = stage0_bundle.prototype_cell_type_table.copy()
    required_columns = {
        "proto_id",
        "cell_type",
        "cell_count",
        "cell_type_fraction",
        "prototype_total_cells",
    }
    missing = sorted(required_columns - set(cell_type_table.columns))
    if missing:
        raise ValueError(
            "Stage-0 prototype cell-type table is missing required columns: "
            f"{missing}"
        )

    records: list[dict[str, object]] = []
    for proto_id, frame in cell_type_table.groupby("proto_id", sort=True, observed=False):
        ordered = frame.sort_values(
            ["cell_type_fraction", "cell_count", "cell_type"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        nonzero = ordered.loc[pd.to_numeric(ordered["cell_count"], errors="coerce").gt(0.0)].copy()
        if nonzero.empty:
            continue
        top_rows = nonzero.head(3).reset_index(drop=True)
        dominant = nonzero.iloc[0]
        top_mix = "; ".join(
            (
                f"{str(row['cell_type'])}:{_format_fraction(float(row['cell_type_fraction']))}"
            )
            for _, row in nonzero.head(3).iterrows()
        )
        top_cells = [None, None, None]
        top_fractions = [np.nan, np.nan, np.nan]
        for idx in range(min(3, top_rows.shape[0])):
            top_cells[idx] = str(top_rows.iloc[idx]["cell_type"])
            top_fractions[idx] = float(top_rows.iloc[idx]["cell_type_fraction"])
        top12_fraction_sum = float(
            np.nansum(np.asarray(top_fractions[:2], dtype=float), dtype=float)
        )
        prototype_label_parts = [
            f"{cell_type}({_format_label_fraction(fraction)})"
            for cell_type, fraction in zip(top_cells, top_fractions)
            if cell_type is not None and not pd.isna(fraction)
        ]
        records.append(
            {
                "proto_id": int(proto_id),
                "dominant_cell_type": str(dominant["cell_type"]),
                "dominant_cell_type_fraction": float(dominant["cell_type_fraction"]),
                "top_cell_type_mix": top_mix,
                "total_cells": int(round(float(dominant["prototype_total_cells"]))),
                "top1_cell_type": top_cells[0],
                "top1_fraction": top_fractions[0],
                "top2_cell_type": top_cells[1],
                "top2_fraction": top_fractions[1],
                "top3_cell_type": top_cells[2],
                "top3_fraction": top_fractions[2],
                "top12_fraction_sum": top12_fraction_sum,
                "prototype_label_top3": f"p{int(proto_id)} | " + " / ".join(prototype_label_parts),
            }
        )

    prototype_meaning = (
        pd.DataFrame.from_records(records)
        .loc[:, list(PROTOTYPE_ANNOTATION_COLUMNS)]
        .sort_values("proto_id")
        .reset_index(drop=True)
    )
    _validate_prototype_meaning_coverage(
        prototype_meaning=prototype_meaning,
        active_prototype_ids=stage0_bundle.active_prototype_ids,
    )
    return prototype_meaning


# ---------------------------------------------------------------------------
# Raw baseline layers
# ---------------------------------------------------------------------------


def build_baseline_pair_audit_table(inputs: LoadedArm2Inputs) -> pd.DataFrame:
    """
    Build the public pair-level baseline audit table on the exact Arm-II
    ordered pair set.

    This table is intentionally pre-transport. It provides ordered-pair audit
    truth and simple abundance/composition context before any transport or
    recurrence logic exists.
    """

    pair_metadata = inputs.pair_tensors.pair_metadata.copy()
    A = np.asarray(inputs.pair_tensors.A, dtype=float)
    B = np.asarray(inputs.pair_tensors.B, dtype=float)

    source_total = np.sum(A, axis=1, dtype=float)
    target_total = np.sum(B, axis=1, dtype=float)
    overlap_count = np.sum(np.minimum(A, B), axis=1, dtype=float)
    source_share = _safe_divide(A, source_total[:, None])
    target_share = _safe_divide(B, target_total[:, None])

    audit = pair_metadata.assign(
        confirmatory_family=pair_metadata["pair_family"].astype(str).isin(CONFIRMATORY_FAMILIES),
        source_total_cells=source_total,
        target_total_cells=target_total,
        total_cell_gap=target_total - source_total,
        abs_total_cell_gap=np.abs(target_total - source_total),
        baseline_l1_count_gap=np.sum(np.abs(A - B), axis=1, dtype=float),
        baseline_overlap_count=overlap_count,
        baseline_overlap_fraction=_safe_divide(
            2.0 * overlap_count,
            source_total + target_total,
        ),
        baseline_composition_l1=0.5 * np.sum(
            np.abs(source_share - target_share),
            axis=1,
            dtype=float,
        ),
    )
    columns = [
        "pair_id",
        "patient_id",
        "pair_family",
        "pair_family_role",
        "ordered_direction",
        "source_compartment",
        "target_compartment",
        "source_roi_id",
        "target_roi_id",
        "confirmatory_family",
        "source_total_cells",
        "target_total_cells",
        "total_cell_gap",
        "abs_total_cell_gap",
        "baseline_l1_count_gap",
        "baseline_overlap_count",
        "baseline_overlap_fraction",
        "baseline_composition_l1",
    ]
    return audit.loc[:, columns].reset_index(drop=True)


def build_pair_prototype_baseline_long_table(
    inputs: LoadedArm2Inputs,
    prototype_meaning: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the raw ordered-pair x prototype baseline table.

    Row unit:
    - ordered ROI pair x prototype

    This is the baseline-side raw surface that preserves count and share values
    separately before any patient-level collapsing.
    """

    _validate_prototype_meaning_coverage(
        prototype_meaning=prototype_meaning,
        active_prototype_ids=inputs.stage0.active_prototype_ids,
    )
    pair_metadata = inputs.pair_tensors.pair_metadata.copy()
    active_proto_ids = _expected_prototype_ids(prototype_meaning)
    A_active = np.asarray(inputs.pair_tensors.A[:, active_proto_ids], dtype=float)
    B_active = np.asarray(inputs.pair_tensors.B[:, active_proto_ids], dtype=float)

    source_total = np.sum(inputs.pair_tensors.A, axis=1, dtype=float)
    target_total = np.sum(inputs.pair_tensors.B, axis=1, dtype=float)
    source_share = _safe_divide(A_active, source_total[:, None])
    target_share = _safe_divide(B_active, target_total[:, None])
    delta_count = B_active - A_active
    delta_share = target_share - source_share

    n_pairs = pair_metadata.shape[0]
    n_prototypes = active_proto_ids.size
    return pd.DataFrame(
        {
            "pair_id": np.repeat(pair_metadata["pair_id"].astype(str).to_numpy(), n_prototypes),
            "patient_id": np.repeat(pair_metadata["patient_id"].astype(str).to_numpy(), n_prototypes),
            "pair_family": np.repeat(pair_metadata["pair_family"].astype(str).to_numpy(), n_prototypes),
            "pair_family_role": np.repeat(
                pair_metadata["pair_family_role"].astype(str).to_numpy(),
                n_prototypes,
            ),
            "ordered_direction": np.repeat(
                pair_metadata["ordered_direction"].astype(str).to_numpy(),
                n_prototypes,
            ),
            "proto_id": np.tile(active_proto_ids.astype(int), n_pairs),
            "source_total_count": np.repeat(source_total, n_prototypes),
            "target_total_count": np.repeat(target_total, n_prototypes),
            "source_count": A_active.ravel(),
            "target_count": B_active.ravel(),
            "delta_count": delta_count.ravel(),
            "abs_delta_count": np.abs(delta_count).ravel(),
            "source_share": source_share.ravel(),
            "target_share": target_share.ravel(),
            "delta_share": delta_share.ravel(),
            "abs_delta_share": np.abs(delta_share).ravel(),
        }
    )


def build_all_prototype_baseline_patient_family_table(
    pair_prototype_baseline_long: pd.DataFrame,
    prototype_meaning: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the stable internal baseline aggregation with row unit
    `(patient, family, prototype)`.

    This is the baseline-side internal memo boundary between raw ordered-pair
    values and later recurrence / extracted-view logic.

    Semantics:
    - each row summarizes the raw ordered-pair x prototype values within one
      `(patient, family, prototype)` group,
    - this is explicitly not a sum-collapsed pseudo-composition table,
    - absolute-magnitude summaries are the primary baseline signal,
    - signed summaries remain contextual only.
    """

    grouped = (
        pair_prototype_baseline_long.groupby(
            ["patient_id", "pair_family", "pair_family_role", "proto_id"],
            sort=True,
            observed=False,
        )
        .agg(
            ordered_pair_count=("pair_id", "nunique"),
            mean_source_total_count_context=("source_total_count", "mean"),
            mean_target_total_count_context=("target_total_count", "mean"),
            mean_source_count_context=("source_count", "mean"),
            mean_target_count_context=("target_count", "mean"),
            median_abs_delta_share=("abs_delta_share", "median"),
            mean_abs_delta_share=("abs_delta_share", "mean"),
            median_abs_nonzero_delta_share=("delta_share", _nonzero_abs_median),
            prop_nonzero_share=("delta_share", _prop_nonzero),
            median_abs_delta_count=("abs_delta_count", "median"),
            mean_abs_delta_count=("abs_delta_count", "mean"),
            median_abs_nonzero_delta_count=("delta_count", _nonzero_abs_median),
            prop_nonzero_count=("delta_count", _prop_nonzero),
            median_delta_share_context=("delta_share", "median"),
            median_delta_count_context=("delta_count", "median"),
        )
        .reset_index()
    )
    grouped = grouped.merge(prototype_meaning, on="proto_id", how="left")
    grouped["pair_family"] = pd.Categorical(
        grouped["pair_family"],
        categories=PAIR_FAMILY_ORDER,
        ordered=True,
    )
    annotation_columns = [column for column in prototype_meaning.columns if column != "proto_id"]
    columns = [
        "patient_id",
        "pair_family",
        "pair_family_role",
        "proto_id",
        *annotation_columns,
        "ordered_pair_count",
        "mean_source_total_count_context",
        "mean_target_total_count_context",
        "mean_source_count_context",
        "mean_target_count_context",
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
    ]
    return grouped.loc[:, columns].sort_values(
        ["patient_id", "pair_family", "proto_id"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Confirmatory baseline outputs
# ---------------------------------------------------------------------------


def build_baseline_patient_family_confirmatory_summary(
    all_prototype_baseline_patient_family: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the paired-patient confirmatory baseline summary.

    Row unit:
    - patient

    Construction rule:
    - only patients with both `TC-IM` and `TC-PT` are retained.

    Input semantics:
    - consume the patient-family-prototype summary-over-pairs table,
    - do not reconstruct a new baseline delta from sum-collapsed counts.

    Interpretation rule:
    - share-scale absolute magnitude is the primary signal,
    - count-scale absolute magnitude is parallel context,
    - signed summaries remain contextual only.
    """

    output_columns = [
        "patient_id",
        "tc_im_ordered_pair_count",
        "tc_pt_ordered_pair_count",
        "tc_im_prototype_count",
        "tc_pt_prototype_count",
        "tc_im_median_abs_delta_share",
        "tc_pt_median_abs_delta_share",
        "patient_median_tc_pt_minus_tc_im_abs_delta_share",
        "tc_im_mean_abs_delta_share",
        "tc_pt_mean_abs_delta_share",
        "patient_mean_tc_pt_minus_tc_im_abs_delta_share",
        "tc_im_median_abs_nonzero_delta_share",
        "tc_pt_median_abs_nonzero_delta_share",
        "patient_median_tc_pt_minus_tc_im_abs_nonzero_delta_share",
        "tc_im_mean_prop_nonzero_share",
        "tc_pt_mean_prop_nonzero_share",
        "patient_mean_tc_pt_minus_tc_im_prop_nonzero_share",
        "tc_im_median_abs_delta_count",
        "tc_pt_median_abs_delta_count",
        "patient_median_tc_pt_minus_tc_im_abs_delta_count",
        "tc_im_mean_abs_delta_count",
        "tc_pt_mean_abs_delta_count",
        "patient_mean_tc_pt_minus_tc_im_abs_delta_count",
        "tc_im_median_abs_nonzero_delta_count",
        "tc_pt_median_abs_nonzero_delta_count",
        "patient_median_tc_pt_minus_tc_im_abs_nonzero_delta_count",
        "tc_im_mean_prop_nonzero_count",
        "tc_pt_mean_prop_nonzero_count",
        "patient_mean_tc_pt_minus_tc_im_prop_nonzero_count",
        "tc_im_median_delta_share_context",
        "tc_pt_median_delta_share_context",
        "patient_median_tc_pt_minus_tc_im_delta_share_context",
        "tc_im_median_delta_count_context",
        "tc_pt_median_delta_count_context",
        "patient_median_tc_pt_minus_tc_im_delta_count_context",
    ]

    paired_patient_ids = _confirmatory_paired_patient_ids(all_prototype_baseline_patient_family)
    confirmatory = all_prototype_baseline_patient_family.loc[
        all_prototype_baseline_patient_family["patient_id"].astype(str).isin(paired_patient_ids)
        & all_prototype_baseline_patient_family["pair_family"].astype(str).isin(CONFIRMATORY_FAMILIES)
    ].copy()
    if confirmatory.empty:
        return pd.DataFrame(columns=output_columns)

    patient_family_summary = (
        confirmatory.groupby(["patient_id", "pair_family"], sort=True, observed=True)
        .agg(
            ordered_pair_count=("ordered_pair_count", "first"),
            prototype_count=("proto_id", "nunique"),
            median_abs_delta_share=("median_abs_delta_share", "median"),
            mean_abs_delta_share=("mean_abs_delta_share", "mean"),
            median_abs_nonzero_delta_share=("median_abs_nonzero_delta_share", "median"),
            mean_prop_nonzero_share=("prop_nonzero_share", "mean"),
            median_abs_delta_count=("median_abs_delta_count", "median"),
            mean_abs_delta_count=("mean_abs_delta_count", "mean"),
            median_abs_nonzero_delta_count=("median_abs_nonzero_delta_count", "median"),
            mean_prop_nonzero_count=("prop_nonzero_count", "mean"),
            median_delta_share_context=("median_delta_share_context", "median"),
            median_delta_count_context=("median_delta_count_context", "median"),
        )
        .reset_index()
    )
    wide = _pivot_confirmatory_family_values(
        patient_family_summary,
        index_columns=["patient_id"],
        value_columns=[
            "ordered_pair_count",
            "prototype_count",
            "median_abs_delta_share",
            "mean_abs_delta_share",
            "median_abs_nonzero_delta_share",
            "mean_prop_nonzero_share",
            "median_abs_delta_count",
            "mean_abs_delta_count",
            "median_abs_nonzero_delta_count",
            "mean_prop_nonzero_count",
            "median_delta_share_context",
            "median_delta_count_context",
        ],
    )
    wide["patient_median_tc_pt_minus_tc_im_abs_delta_share"] = (
        pd.to_numeric(wide["tc_pt_median_abs_delta_share"], errors="coerce").astype(float)
        - pd.to_numeric(wide["tc_im_median_abs_delta_share"], errors="coerce").astype(float)
    )
    wide["patient_mean_tc_pt_minus_tc_im_abs_delta_share"] = (
        pd.to_numeric(wide["tc_pt_mean_abs_delta_share"], errors="coerce").astype(float)
        - pd.to_numeric(wide["tc_im_mean_abs_delta_share"], errors="coerce").astype(float)
    )
    wide["patient_median_tc_pt_minus_tc_im_abs_nonzero_delta_share"] = (
        pd.to_numeric(wide["tc_pt_median_abs_nonzero_delta_share"], errors="coerce").astype(float)
        - pd.to_numeric(wide["tc_im_median_abs_nonzero_delta_share"], errors="coerce").astype(float)
    )
    wide["patient_mean_tc_pt_minus_tc_im_prop_nonzero_share"] = (
        pd.to_numeric(wide["tc_pt_mean_prop_nonzero_share"], errors="coerce").astype(float)
        - pd.to_numeric(wide["tc_im_mean_prop_nonzero_share"], errors="coerce").astype(float)
    )
    wide["patient_median_tc_pt_minus_tc_im_abs_delta_count"] = (
        pd.to_numeric(wide["tc_pt_median_abs_delta_count"], errors="coerce").astype(float)
        - pd.to_numeric(wide["tc_im_median_abs_delta_count"], errors="coerce").astype(float)
    )
    wide["patient_mean_tc_pt_minus_tc_im_abs_delta_count"] = (
        pd.to_numeric(wide["tc_pt_mean_abs_delta_count"], errors="coerce").astype(float)
        - pd.to_numeric(wide["tc_im_mean_abs_delta_count"], errors="coerce").astype(float)
    )
    wide["patient_median_tc_pt_minus_tc_im_abs_nonzero_delta_count"] = (
        pd.to_numeric(wide["tc_pt_median_abs_nonzero_delta_count"], errors="coerce").astype(float)
        - pd.to_numeric(wide["tc_im_median_abs_nonzero_delta_count"], errors="coerce").astype(float)
    )
    wide["patient_mean_tc_pt_minus_tc_im_prop_nonzero_count"] = (
        pd.to_numeric(wide["tc_pt_mean_prop_nonzero_count"], errors="coerce").astype(float)
        - pd.to_numeric(wide["tc_im_mean_prop_nonzero_count"], errors="coerce").astype(float)
    )
    wide["patient_median_tc_pt_minus_tc_im_delta_share_context"] = (
        pd.to_numeric(wide["tc_pt_median_delta_share_context"], errors="coerce").astype(float)
        - pd.to_numeric(wide["tc_im_median_delta_share_context"], errors="coerce").astype(float)
    )
    wide["patient_median_tc_pt_minus_tc_im_delta_count_context"] = (
        pd.to_numeric(wide["tc_pt_median_delta_count_context"], errors="coerce").astype(float)
        - pd.to_numeric(wide["tc_im_median_delta_count_context"], errors="coerce").astype(float)
    )
    return wide.loc[:, output_columns].sort_values("patient_id").reset_index(drop=True)


def build_baseline_prototype_confirmatory_summary(
    all_prototype_baseline_patient_family: pd.DataFrame,
    prototype_meaning: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the public confirmatory baseline prototype summary.

    Row unit:
    - prototype

    The ranking anchor is share-scale absolute magnitude. Count-scale values are
    preserved as same-scale context, and signed values remain contextual only.
    """

    annotation_columns = [column for column in prototype_meaning.columns if column != "proto_id"]
    output_columns = [
        "baseline_priority_rank",
        "proto_id",
        *annotation_columns,
        "paired_patient_count",
        "confirmatory_abs_share_anchor",
        "confirmatory_abs_nonzero_share_anchor",
        "tc_im_patient_count",
        "tc_pt_patient_count",
        "tc_im_median_abs_delta_share",
        "tc_pt_median_abs_delta_share",
        "tc_pt_minus_tc_im_summary_median_abs_delta_share",
        "tc_im_mean_abs_delta_share",
        "tc_pt_mean_abs_delta_share",
        "tc_pt_minus_tc_im_summary_mean_abs_delta_share",
        "tc_im_median_abs_nonzero_delta_share",
        "tc_pt_median_abs_nonzero_delta_share",
        "tc_pt_minus_tc_im_summary_median_abs_nonzero_delta_share",
        "tc_im_mean_prop_nonzero_share",
        "tc_pt_mean_prop_nonzero_share",
        "tc_pt_minus_tc_im_summary_mean_prop_nonzero_share",
        "patient_median_tc_pt_minus_tc_im_abs_delta_share",
        "patient_mean_tc_pt_minus_tc_im_abs_delta_share",
        "prop_paired_patients_tc_pt_gt_tc_im_abs_delta_share",
        "patient_median_tc_pt_minus_tc_im_abs_nonzero_delta_share",
        "patient_mean_tc_pt_minus_tc_im_prop_nonzero_share",
        "prop_paired_patients_tc_pt_gt_tc_im_abs_nonzero_delta_share",
        "tc_im_median_abs_delta_count",
        "tc_pt_median_abs_delta_count",
        "tc_pt_minus_tc_im_summary_median_abs_delta_count",
        "tc_im_mean_abs_delta_count",
        "tc_pt_mean_abs_delta_count",
        "tc_pt_minus_tc_im_summary_mean_abs_delta_count",
        "tc_im_median_abs_nonzero_delta_count",
        "tc_pt_median_abs_nonzero_delta_count",
        "tc_pt_minus_tc_im_summary_median_abs_nonzero_delta_count",
        "tc_im_mean_prop_nonzero_count",
        "tc_pt_mean_prop_nonzero_count",
        "tc_pt_minus_tc_im_summary_mean_prop_nonzero_count",
        "patient_median_tc_pt_minus_tc_im_abs_delta_count",
        "patient_mean_tc_pt_minus_tc_im_abs_delta_count",
        "prop_paired_patients_tc_pt_gt_tc_im_abs_delta_count",
        "patient_median_tc_pt_minus_tc_im_abs_nonzero_delta_count",
        "patient_mean_tc_pt_minus_tc_im_prop_nonzero_count",
        "prop_paired_patients_tc_pt_gt_tc_im_abs_nonzero_delta_count",
        "tc_im_median_delta_share_context",
        "tc_pt_median_delta_share_context",
        "tc_pt_minus_tc_im_summary_median_delta_share_context",
        "tc_im_median_delta_count_context",
        "tc_pt_median_delta_count_context",
        "tc_pt_minus_tc_im_summary_median_delta_count_context",
    ]
    paired_patient_ids = _confirmatory_paired_patient_ids(all_prototype_baseline_patient_family)
    confirmatory = all_prototype_baseline_patient_family.loc[
        all_prototype_baseline_patient_family["patient_id"].astype(str).isin(paired_patient_ids)
        & all_prototype_baseline_patient_family["pair_family"].astype(str).isin(CONFIRMATORY_FAMILIES)
    ].copy()
    if confirmatory.empty:
        empty = prototype_meaning.copy()
        empty["baseline_priority_rank"] = np.arange(1, empty.shape[0] + 1, dtype=int)
        zero_columns = {"paired_patient_count", "tc_im_patient_count", "tc_pt_patient_count"}
        for column in output_columns:
            if column in empty.columns:
                continue
            empty[column] = 0 if column in zero_columns else np.nan
        return empty.loc[:, output_columns].sort_values(
            ["baseline_priority_rank", "proto_id"]
        ).reset_index(drop=True)

    family_summary = (
        confirmatory.groupby(
            [
                "pair_family",
                "proto_id",
                *annotation_columns,
            ],
            sort=True,
            observed=True,
        )
        .agg(
            patient_count=("patient_id", "nunique"),
            median_abs_delta_share=("median_abs_delta_share", "median"),
            mean_abs_delta_share=("mean_abs_delta_share", "mean"),
            median_abs_nonzero_delta_share=("median_abs_nonzero_delta_share", "median"),
            mean_prop_nonzero_share=("prop_nonzero_share", "mean"),
            median_abs_delta_count=("median_abs_delta_count", "median"),
            mean_abs_delta_count=("mean_abs_delta_count", "mean"),
            median_abs_nonzero_delta_count=("median_abs_nonzero_delta_count", "median"),
            mean_prop_nonzero_count=("prop_nonzero_count", "mean"),
            median_delta_share_context=("median_delta_share_context", "median"),
            median_delta_count_context=("median_delta_count_context", "median"),
        )
        .reset_index()
    )
    summary = _pivot_confirmatory_family_values(
        family_summary,
        index_columns=[
            "proto_id",
            *annotation_columns,
        ],
        value_columns=[
            "patient_count",
            "median_abs_delta_share",
            "mean_abs_delta_share",
            "median_abs_nonzero_delta_share",
            "mean_prop_nonzero_share",
            "median_abs_delta_count",
            "mean_abs_delta_count",
            "median_abs_nonzero_delta_count",
            "mean_prop_nonzero_count",
            "median_delta_share_context",
            "median_delta_count_context",
        ],
    )
    summary["confirmatory_abs_share_anchor"] = summary[
        ["tc_im_median_abs_delta_share", "tc_pt_median_abs_delta_share"]
    ].max(axis=1)
    summary["confirmatory_abs_nonzero_share_anchor"] = summary[
        ["tc_im_median_abs_nonzero_delta_share", "tc_pt_median_abs_nonzero_delta_share"]
    ].max(axis=1)
    summary["tc_pt_minus_tc_im_summary_median_abs_delta_share"] = (
        pd.to_numeric(summary["tc_pt_median_abs_delta_share"], errors="coerce").astype(float)
        - pd.to_numeric(summary["tc_im_median_abs_delta_share"], errors="coerce").astype(float)
    )
    summary["tc_pt_minus_tc_im_summary_mean_abs_delta_share"] = (
        pd.to_numeric(summary["tc_pt_mean_abs_delta_share"], errors="coerce").astype(float)
        - pd.to_numeric(summary["tc_im_mean_abs_delta_share"], errors="coerce").astype(float)
    )
    summary["tc_pt_minus_tc_im_summary_median_abs_nonzero_delta_share"] = (
        pd.to_numeric(summary["tc_pt_median_abs_nonzero_delta_share"], errors="coerce").astype(float)
        - pd.to_numeric(summary["tc_im_median_abs_nonzero_delta_share"], errors="coerce").astype(float)
    )
    summary["tc_pt_minus_tc_im_summary_mean_prop_nonzero_share"] = (
        pd.to_numeric(summary["tc_pt_mean_prop_nonzero_share"], errors="coerce").astype(float)
        - pd.to_numeric(summary["tc_im_mean_prop_nonzero_share"], errors="coerce").astype(float)
    )
    summary["tc_pt_minus_tc_im_summary_median_abs_delta_count"] = (
        pd.to_numeric(summary["tc_pt_median_abs_delta_count"], errors="coerce").astype(float)
        - pd.to_numeric(summary["tc_im_median_abs_delta_count"], errors="coerce").astype(float)
    )
    summary["tc_pt_minus_tc_im_summary_mean_abs_delta_count"] = (
        pd.to_numeric(summary["tc_pt_mean_abs_delta_count"], errors="coerce").astype(float)
        - pd.to_numeric(summary["tc_im_mean_abs_delta_count"], errors="coerce").astype(float)
    )
    summary["tc_pt_minus_tc_im_summary_median_abs_nonzero_delta_count"] = (
        pd.to_numeric(summary["tc_pt_median_abs_nonzero_delta_count"], errors="coerce").astype(float)
        - pd.to_numeric(summary["tc_im_median_abs_nonzero_delta_count"], errors="coerce").astype(float)
    )
    summary["tc_pt_minus_tc_im_summary_mean_prop_nonzero_count"] = (
        pd.to_numeric(summary["tc_pt_mean_prop_nonzero_count"], errors="coerce").astype(float)
        - pd.to_numeric(summary["tc_im_mean_prop_nonzero_count"], errors="coerce").astype(float)
    )
    summary["tc_pt_minus_tc_im_summary_median_delta_share_context"] = (
        pd.to_numeric(summary["tc_pt_median_delta_share_context"], errors="coerce").astype(float)
        - pd.to_numeric(summary["tc_im_median_delta_share_context"], errors="coerce").astype(float)
    )
    summary["tc_pt_minus_tc_im_summary_median_delta_count_context"] = (
        pd.to_numeric(summary["tc_pt_median_delta_count_context"], errors="coerce").astype(float)
        - pd.to_numeric(summary["tc_im_median_delta_count_context"], errors="coerce").astype(float)
    )

    patient_wide = _pivot_confirmatory_family_values(
        confirmatory.loc[
            :,
            [
                "patient_id",
                "pair_family",
                "proto_id",
                "median_abs_delta_share",
                "median_abs_nonzero_delta_share",
                "prop_nonzero_share",
                "median_abs_delta_count",
                "median_abs_nonzero_delta_count",
                "prop_nonzero_count",
            ],
        ],
        index_columns=["patient_id", "proto_id"],
        value_columns=[
            "median_abs_delta_share",
            "median_abs_nonzero_delta_share",
            "prop_nonzero_share",
            "median_abs_delta_count",
            "median_abs_nonzero_delta_count",
            "prop_nonzero_count",
        ],
    )
    patient_wide["tc_pt_minus_tc_im_abs_delta_share"] = (
        pd.to_numeric(patient_wide["tc_pt_median_abs_delta_share"], errors="coerce").astype(float)
        - pd.to_numeric(patient_wide["tc_im_median_abs_delta_share"], errors="coerce").astype(float)
    )
    patient_wide["tc_pt_minus_tc_im_abs_nonzero_delta_share"] = (
        pd.to_numeric(
            patient_wide["tc_pt_median_abs_nonzero_delta_share"],
            errors="coerce",
        ).astype(float)
        - pd.to_numeric(
            patient_wide["tc_im_median_abs_nonzero_delta_share"],
            errors="coerce",
        ).astype(float)
    )
    patient_wide["tc_pt_minus_tc_im_prop_nonzero_share"] = (
        pd.to_numeric(patient_wide["tc_pt_prop_nonzero_share"], errors="coerce").astype(float)
        - pd.to_numeric(patient_wide["tc_im_prop_nonzero_share"], errors="coerce").astype(float)
    )
    patient_wide["tc_pt_minus_tc_im_abs_delta_count"] = (
        pd.to_numeric(patient_wide["tc_pt_median_abs_delta_count"], errors="coerce").astype(float)
        - pd.to_numeric(patient_wide["tc_im_median_abs_delta_count"], errors="coerce").astype(float)
    )
    patient_wide["tc_pt_minus_tc_im_abs_nonzero_delta_count"] = (
        pd.to_numeric(
            patient_wide["tc_pt_median_abs_nonzero_delta_count"],
            errors="coerce",
        ).astype(float)
        - pd.to_numeric(
            patient_wide["tc_im_median_abs_nonzero_delta_count"],
            errors="coerce",
        ).astype(float)
    )
    patient_wide["tc_pt_minus_tc_im_prop_nonzero_count"] = (
        pd.to_numeric(patient_wide["tc_pt_prop_nonzero_count"], errors="coerce").astype(float)
        - pd.to_numeric(patient_wide["tc_im_prop_nonzero_count"], errors="coerce").astype(float)
    )
    patient_contrast_summary = (
        patient_wide.groupby("proto_id", sort=True, observed=False)
        .agg(
            paired_patient_count=("patient_id", "nunique"),
            patient_median_tc_pt_minus_tc_im_abs_delta_share=("tc_pt_minus_tc_im_abs_delta_share", "median"),
            patient_mean_tc_pt_minus_tc_im_abs_delta_share=("tc_pt_minus_tc_im_abs_delta_share", "mean"),
            prop_paired_patients_tc_pt_gt_tc_im_abs_delta_share=(
                "tc_pt_minus_tc_im_abs_delta_share",
                lambda values: float(pd.to_numeric(values, errors="coerce").gt(0.0).mean()),
            ),
            patient_median_tc_pt_minus_tc_im_abs_nonzero_delta_share=(
                "tc_pt_minus_tc_im_abs_nonzero_delta_share",
                "median",
            ),
            patient_mean_tc_pt_minus_tc_im_prop_nonzero_share=(
                "tc_pt_minus_tc_im_prop_nonzero_share",
                "mean",
            ),
            prop_paired_patients_tc_pt_gt_tc_im_abs_nonzero_delta_share=(
                "tc_pt_minus_tc_im_abs_nonzero_delta_share",
                lambda values: float(pd.to_numeric(values, errors="coerce").gt(0.0).mean()),
            ),
            patient_median_tc_pt_minus_tc_im_abs_delta_count=("tc_pt_minus_tc_im_abs_delta_count", "median"),
            patient_mean_tc_pt_minus_tc_im_abs_delta_count=("tc_pt_minus_tc_im_abs_delta_count", "mean"),
            prop_paired_patients_tc_pt_gt_tc_im_abs_delta_count=(
                "tc_pt_minus_tc_im_abs_delta_count",
                lambda values: float(pd.to_numeric(values, errors="coerce").gt(0.0).mean()),
            ),
            patient_median_tc_pt_minus_tc_im_abs_nonzero_delta_count=(
                "tc_pt_minus_tc_im_abs_nonzero_delta_count",
                "median",
            ),
            patient_mean_tc_pt_minus_tc_im_prop_nonzero_count=(
                "tc_pt_minus_tc_im_prop_nonzero_count",
                "mean",
            ),
            prop_paired_patients_tc_pt_gt_tc_im_abs_nonzero_delta_count=(
                "tc_pt_minus_tc_im_abs_nonzero_delta_count",
                lambda values: float(pd.to_numeric(values, errors="coerce").gt(0.0).mean()),
            ),
        )
        .reset_index()
    )
    summary = summary.merge(patient_contrast_summary, on="proto_id", how="left")
    order = summary.sort_values(
        [
            "confirmatory_abs_share_anchor",
            "confirmatory_abs_nonzero_share_anchor",
            "tc_pt_median_abs_delta_share",
            "tc_im_median_abs_delta_share",
            "proto_id",
        ],
        ascending=[False, False, False, False, True],
    ).index
    summary.loc[order, "baseline_priority_rank"] = np.arange(1, len(order) + 1, dtype=int)
    summary["baseline_priority_rank"] = summary["baseline_priority_rank"].astype(int)
    return summary.loc[:, output_columns].sort_values(
        ["baseline_priority_rank", "proto_id"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Baseline validation and bundle assembly
# ---------------------------------------------------------------------------


def validate_baseline_tables(
    pair_prototype_baseline_long: pd.DataFrame,
    all_prototype_baseline_patient_family: pd.DataFrame,
    baseline_patient_family_confirmatory: pd.DataFrame,
    baseline_prototype_confirmatory: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate the baseline-side all-prototype analysis outputs.

    Required checks for this step:
    - count/share separation,
    - nonnegative absolute deltas,
    - all-prototype coverage,
    - confirmatory-family filtering,
    - hard failure if prototype share summaries collapse to zero despite
      nonzero underlying `abs_delta_share`.
    """

    records: list[dict[str, object]] = []

    raw_scale_ok, raw_scale_detail = _summarize_scale_relationships(
        pair_prototype_baseline_long,
    )
    internal_scale_ok, internal_scale_detail = _summarize_internal_summary_table(
        all_prototype_baseline_patient_family,
    )
    records.extend(
        [
            {
                "check": "raw_count_share_separation",
                "passed": raw_scale_ok,
                "detail": raw_scale_detail,
            },
            {
                "check": "internal_count_share_separation",
                "passed": internal_scale_ok,
                "detail": internal_scale_detail,
            },
        ]
    )

    raw_abs_ok = bool(
        pd.to_numeric(pair_prototype_baseline_long["abs_delta_count"], errors="coerce").ge(0.0).all()
        and pd.to_numeric(pair_prototype_baseline_long["abs_delta_share"], errors="coerce").ge(0.0).all()
    )
    internal_abs_ok = bool(
        pd.to_numeric(
            all_prototype_baseline_patient_family["median_abs_delta_count"],
            errors="coerce",
        ).ge(0.0).all()
        and pd.to_numeric(
            all_prototype_baseline_patient_family["mean_abs_delta_count"],
            errors="coerce",
        ).ge(0.0).all()
        and pd.to_numeric(
            all_prototype_baseline_patient_family["median_abs_nonzero_delta_count"],
            errors="coerce",
        ).ge(0.0).all()
        and pd.to_numeric(
            all_prototype_baseline_patient_family["median_abs_delta_share"],
            errors="coerce",
        ).ge(0.0).all()
        and pd.to_numeric(
            all_prototype_baseline_patient_family["mean_abs_delta_share"],
            errors="coerce",
        ).ge(0.0).all()
        and pd.to_numeric(
            all_prototype_baseline_patient_family["median_abs_nonzero_delta_share"],
            errors="coerce",
        ).ge(0.0).all()
    )
    records.extend(
        [
            {
                "check": "raw_absolute_deltas_nonnegative",
                "passed": raw_abs_ok,
                "detail": (
                    "min_abs_delta_count="
                    f"{pd.to_numeric(pair_prototype_baseline_long['abs_delta_count'], errors='coerce').min()}, "
                    "min_abs_delta_share="
                    f"{pd.to_numeric(pair_prototype_baseline_long['abs_delta_share'], errors='coerce').min()}"
                ),
            },
            {
                "check": "internal_absolute_deltas_nonnegative",
                "passed": internal_abs_ok,
                "detail": (
                    "min_median_abs_delta_count="
                    f"{pd.to_numeric(all_prototype_baseline_patient_family['median_abs_delta_count'], errors='coerce').min()}, "
                    "min_median_abs_delta_share="
                    f"{pd.to_numeric(all_prototype_baseline_patient_family['median_abs_delta_share'], errors='coerce').min()}"
                ),
            },
        ]
    )

    proto_ids = sorted(pair_prototype_baseline_long["proto_id"].astype(int).unique().tolist())
    pair_count = int(pair_prototype_baseline_long["pair_id"].astype(str).nunique())
    raw_coverage_ok = bool(pair_prototype_baseline_long.shape[0] == pair_count * len(proto_ids))

    patient_family_count = int(
        all_prototype_baseline_patient_family.loc[:, ["patient_id", "pair_family"]]
        .drop_duplicates()
        .shape[0]
    )
    internal_coverage_ok = bool(
        all_prototype_baseline_patient_family.shape[0] == patient_family_count * len(proto_ids)
    )

    if baseline_prototype_confirmatory.empty:
        prototype_coverage_ok = True
        prototype_coverage_detail = "prototype summary empty"
    else:
        prototype_coverage_ok = sorted(
            baseline_prototype_confirmatory["proto_id"].astype(int).tolist()
        ) == proto_ids
        prototype_coverage_detail = (
            f"expected={proto_ids}, "
            f"observed={sorted(baseline_prototype_confirmatory['proto_id'].astype(int).tolist())}"
        )
    records.extend(
        [
            {
                "check": "raw_all_prototype_coverage",
                "passed": raw_coverage_ok,
                "detail": f"rows={pair_prototype_baseline_long.shape[0]}, expected={pair_count * len(proto_ids)}",
            },
            {
                "check": "internal_all_prototype_coverage",
                "passed": internal_coverage_ok,
                "detail": (
                    f"rows={all_prototype_baseline_patient_family.shape[0]}, "
                    f"expected={patient_family_count * len(proto_ids)}"
                ),
            },
            {
                "check": "confirmatory_prototype_all_prototype_coverage",
                "passed": prototype_coverage_ok,
                "detail": prototype_coverage_detail,
            },
        ]
    )

    paired_patient_ids = _confirmatory_paired_patient_ids(all_prototype_baseline_patient_family)
    confirmatory_columns_ok = bool(
        not any("im_pt" in str(column) for column in baseline_patient_family_confirmatory.columns)
        and not any("im_pt" in str(column) for column in baseline_prototype_confirmatory.columns)
    )
    patient_filter_ok = sorted(
        baseline_patient_family_confirmatory["patient_id"].astype(str).tolist()
    ) == list(paired_patient_ids)
    records.append(
        {
            "check": "confirmatory_family_filtering",
            "passed": confirmatory_columns_ok and patient_filter_ok,
            "detail": (
                f"paired_patients={list(paired_patient_ids)}, "
                f"summary_patients={baseline_patient_family_confirmatory['patient_id'].astype(str).tolist()}, "
                f"has_im_pt_columns={not confirmatory_columns_ok}"
            ),
        }
    )

    paired_confirmatory_raw = pair_prototype_baseline_long.loc[
        pair_prototype_baseline_long["patient_id"].astype(str).isin(paired_patient_ids)
        & pair_prototype_baseline_long["pair_family"].astype(str).isin(CONFIRMATORY_FAMILIES)
    ].copy()
    if not paired_confirmatory_raw.empty:
        underlying_nonzero = (
            paired_confirmatory_raw.groupby("proto_id", sort=True, observed=False)["abs_delta_share"]
            .max()
            .astype(float)
        )
        share_summary_lookup = baseline_prototype_confirmatory.set_index("proto_id")
        collapsed_proto_ids: list[int] = []
        for proto_id, raw_max in underlying_nonzero.items():
            if raw_max <= 0.0:
                continue
            if int(proto_id) not in share_summary_lookup.index:
                collapsed_proto_ids.append(int(proto_id))
                continue
            share_values = pd.to_numeric(
                share_summary_lookup.loc[
                    int(proto_id),
                    [
                        "tc_im_median_abs_delta_share",
                        "tc_pt_median_abs_delta_share",
                        "tc_im_mean_abs_delta_share",
                        "tc_pt_mean_abs_delta_share",
                        "tc_im_median_abs_nonzero_delta_share",
                        "tc_pt_median_abs_nonzero_delta_share",
                    ],
                ],
                errors="coerce",
            ).fillna(0.0)
            if bool(np.isclose(share_values.to_numpy(dtype=float), 0.0, atol=1e-12).all()):
                collapsed_proto_ids.append(int(proto_id))
        if collapsed_proto_ids:
            raise ValueError(
                "Confirmatory prototype share summaries collapsed to zero despite nonzero "
                f"underlying abs_delta_share values for proto_id={collapsed_proto_ids}"
            )
    records.append(
        {
            "check": "confirmatory_prototype_share_summary_not_collapsed",
            "passed": True,
            "detail": "no collapsed-to-zero prototype share summaries detected",
        }
    )

    validation = pd.DataFrame.from_records(records)
    failed = validation.loc[~validation["passed"].astype(bool)].copy()
    if not failed.empty:
        details = "; ".join(f"{row['check']}={row['detail']}" for _, row in failed.iterrows())
        raise ValueError(f"Baseline validation failed: {details}")
    return validation


def build_baseline_tables(
    inputs: LoadedArm2Inputs,
    prototype_meaning: pd.DataFrame,
) -> BaselineAnalysisTables:
    """
    Assemble the full baseline-side all-prototype analysis bundle.

    This is the only baseline entrypoint the focused rewrite should use.
    Everything downstream should consume these stable tables rather than
    re-aggregating baseline logic ad hoc.
    """

    baseline_pair_audit = build_baseline_pair_audit_table(inputs)
    pair_prototype_baseline_long = build_pair_prototype_baseline_long_table(
        inputs=inputs,
        prototype_meaning=prototype_meaning,
    )
    all_prototype_baseline_patient_family = build_all_prototype_baseline_patient_family_table(
        pair_prototype_baseline_long=pair_prototype_baseline_long,
        prototype_meaning=prototype_meaning,
    )
    baseline_patient_family_confirmatory = build_baseline_patient_family_confirmatory_summary(
        all_prototype_baseline_patient_family=all_prototype_baseline_patient_family,
    )
    baseline_prototype_confirmatory = build_baseline_prototype_confirmatory_summary(
        all_prototype_baseline_patient_family=all_prototype_baseline_patient_family,
        prototype_meaning=prototype_meaning,
    )
    baseline_validation = validate_baseline_tables(
        pair_prototype_baseline_long=pair_prototype_baseline_long,
        all_prototype_baseline_patient_family=all_prototype_baseline_patient_family,
        baseline_patient_family_confirmatory=baseline_patient_family_confirmatory,
        baseline_prototype_confirmatory=baseline_prototype_confirmatory,
    )
    return BaselineAnalysisTables(
        prototype_meaning=prototype_meaning,
        baseline_pair_audit=baseline_pair_audit,
        pair_prototype_baseline_long=pair_prototype_baseline_long,
        all_prototype_baseline_patient_family=all_prototype_baseline_patient_family,
        baseline_patient_family_confirmatory=baseline_patient_family_confirmatory,
        baseline_prototype_confirmatory=baseline_prototype_confirmatory,
        baseline_validation=baseline_validation,
    )
