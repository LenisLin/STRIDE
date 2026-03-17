"""
Module: tasks.task_A.arm3.output

Phase 8 output builders for the Arm-3 runner.

Responsibilities:
- Build the Phase-8 prototype summary table from the prototype contrast prep
  parquet surface.
- Render a descriptive markdown summary for the existing Phase 4 / 7 / 8
  artifacts without assigning thresholded or biological verdicts.
- Write the Phase-8 parquet / csv / markdown outputs to the configured result
  root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Phase 8 output file names
# ---------------------------------------------------------------------------

ARM3_PHASE8_PROTOTYPE_STABILITY_BASENAME: str = "arm3_phase8_prototype_stability"
ARM3_PHASE8_MEMO_FILENAME: str = "arm3_phase8_memo.md"


_PROTO_CONTRAST_REQUIRED_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "coverage",
    "replicate_id",
    "prototype_k",
    "prototype_label",
    "Delta_U_k",
)


def _validate_proto_contrast_input(df_proto_contrast: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(_PROTO_CONTRAST_REQUIRED_COLUMNS) - set(df_proto_contrast.columns))
    if missing:
        raise ValueError(
            "build_prototype_stability_table: df_proto_contrast is missing required "
            f"columns: {missing}"
        )

    df = df_proto_contrast.copy()
    if df.empty:
        return df

    df["patient_id"] = df["patient_id"].astype(str)
    df["coverage"] = pd.to_numeric(df["coverage"], errors="raise")
    df["prototype_k"] = pd.to_numeric(df["prototype_k"], errors="raise").astype(int)
    return df


def _prototype_label_map(df_proto_contrast: pd.DataFrame) -> dict[int, str]:
    label_map: dict[int, str] = {}
    for proto_k, grp in df_proto_contrast.groupby("prototype_k", sort=True):
        labels = grp["prototype_label"].dropna().astype(str).unique().tolist()
        if not labels:
            label_map[int(proto_k)] = f"proto_{int(proto_k)}"
            continue
        if len(labels) > 1:
            raise ValueError(
                "build_prototype_stability_table: inconsistent prototype_label values "
                f"for prototype_k={int(proto_k)}: {labels}"
            )
        label_map[int(proto_k)] = labels[0]
    return label_map


def _sign_consistency_rate(reference_vals: pd.Series, bootstrap_vals: pd.Series) -> tuple[float, int]:
    eval_mask = (
        reference_vals.notna()
        & bootstrap_vals.notna()
        & (reference_vals != 0.0)
    )
    n_evaluable = int(eval_mask.sum())
    if n_evaluable == 0:
        return float("nan"), 0

    ref_sign = np.sign(reference_vals.loc[eval_mask].to_numpy(dtype=float))
    boot_sign = np.sign(bootstrap_vals.loc[eval_mask].to_numpy(dtype=float))
    n_match = int(np.sum(ref_sign == boot_sign))
    return float(n_match / n_evaluable), n_evaluable


def _correlation_or_nan(reference_vals: pd.Series, bootstrap_vals: pd.Series) -> tuple[float, int]:
    eval_mask = (
        reference_vals.notna()
        & bootstrap_vals.notna()
        & (reference_vals != 0.0)
    )
    n_patients = int(eval_mask.sum())
    if n_patients < 2:
        return float("nan"), n_patients

    ref_arr = reference_vals.loc[eval_mask].to_numpy(dtype=float)
    boot_arr = bootstrap_vals.loc[eval_mask].to_numpy(dtype=float)

    if np.all(ref_arr == ref_arr[0]) or np.all(boot_arr == boot_arr[0]):
        return float("nan"), n_patients

    corr = np.corrcoef(ref_arr, boot_arr)[0, 1]
    return float(corr), n_patients


def build_prototype_stability_table(df_proto_contrast: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Phase-8 prototype summary table from the prototype contrast prep surface.

    The output is purely descriptive and contains one row per prototype_k ×
    reduced coverage level. It intentionally excludes recurrence-like metrics or
    any thresholded interpretation fields.

    Correlation is computed between:
    - the full-reference patient-level Delta_U_k values (coverage == 1.0), and
    - the patient-level median bootstrap Delta_U_k values at the target reduced coverage.

    The evaluable patient cohort for both sign consistency and correlation is:
    - patients with non-missing full-reference and bootstrap-median values, and
    - patients whose full-reference Delta_U_k is not exactly 0.0.
    """
    df = _validate_proto_contrast_input(df_proto_contrast)
    columns = [
        "prototype_k",
        "prototype_label",
        "coverage",
        "sign_consistency_rate",
        "n_evaluable_patients",
        "n_zero_reference_patients",
        "correlation_to_full_cov",
        "n_correlation_patients",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    label_map = _prototype_label_map(df)

    df_full = df[df["coverage"] == 1.0].copy()
    if df_full.empty:
        raise ValueError(
            "build_prototype_stability_table: df_proto_contrast contains no "
            "full-reference rows at coverage == 1.0"
        )

    dup_full = df_full.duplicated(subset=["patient_id", "prototype_k"])
    if bool(dup_full.any()):
        raise ValueError(
            "build_prototype_stability_table: full-reference rows must be unique per "
            "(patient_id, prototype_k)"
        )

    full_lookup = (
        df_full.set_index(["patient_id", "prototype_k"])["Delta_U_k"]
        .sort_index()
    )
    zero_ref_counts = (
        df_full.groupby("prototype_k", sort=True)["Delta_U_k"]
        .apply(lambda s: int((s == 0.0).sum()))
        .to_dict()
    )

    reduced_coverages = sorted(
        cov for cov in df["coverage"].dropna().unique().tolist() if float(cov) != 1.0
    )

    records: list[dict[str, Any]] = []
    for coverage in reduced_coverages:
        df_cov = df[df["coverage"] == float(coverage)].copy()
        if df_cov.empty:
            continue

        boot_medians = (
            df_cov.groupby(["patient_id", "prototype_k"], sort=True)["Delta_U_k"]
            .median()
            .sort_index()
        )

        all_proto_ids = sorted(
            set(full_lookup.index.get_level_values("prototype_k").tolist())
            | set(boot_medians.index.get_level_values("prototype_k").tolist())
        )

        for proto_k in all_proto_ids:
            ref_series = full_lookup.xs(proto_k, level="prototype_k", drop_level=True)
            try:
                boot_series = boot_medians.xs(proto_k, level="prototype_k", drop_level=True)
            except KeyError:
                boot_series = pd.Series(dtype=float)

            merged = pd.DataFrame(
                {
                    "reference_delta_u_k": ref_series,
                    "bootstrap_median_delta_u_k": boot_series,
                }
            ).sort_index()

            sign_rate, n_evaluable = _sign_consistency_rate(
                merged["reference_delta_u_k"],
                merged["bootstrap_median_delta_u_k"],
            )
            corr, n_corr = _correlation_or_nan(
                merged["reference_delta_u_k"],
                merged["bootstrap_median_delta_u_k"],
            )

            records.append(
                {
                    "prototype_k": int(proto_k),
                    "prototype_label": label_map.get(int(proto_k), f"proto_{int(proto_k)}"),
                    "coverage": float(coverage),
                    "sign_consistency_rate": sign_rate,
                    "n_evaluable_patients": n_evaluable,
                    "n_zero_reference_patients": int(zero_ref_counts.get(int(proto_k), 0)),
                    "correlation_to_full_cov": corr,
                    "n_correlation_patients": n_corr,
                }
            )

    if not records:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame.from_records(records, columns=columns)
        .sort_values(["prototype_k", "coverage"], kind="stable")
        .reset_index(drop=True)
    )


def _format_number(value: object, digits: int = 3) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):.{digits}f}".rstrip("0").rstrip(".")


def build_arm3_memo(
    result_root: Path | str,
    df_degradation: pd.DataFrame,
    df_contrast: pd.DataFrame,
    df_prototype_stability: pd.DataFrame,
    calibration_record: dict[str, Any],
) -> str:
    """
    Render a descriptive Phase-8 markdown summary.

    This summary is intentionally limited to file context, row counts, coverage
    sets, prototype counts, and simple descriptive aggregates from upstream
    tables. It does not assign thresholded or biological outcome labels.
    """
    result_root = Path(result_root)
    stability_coverages = sorted(
        pd.to_numeric(df_prototype_stability.get("coverage"), errors="coerce")
        .dropna()
        .astype(float)
        .unique()
        .tolist()
    )
    prototype_count = int(
        df_prototype_stability["prototype_k"].nunique()
        if not df_prototype_stability.empty and "prototype_k" in df_prototype_stability.columns
        else 0
    )

    lines = [
        "# Arm-3 Phase 8 Pipeline Summary",
        "",
        "## Context",
        "",
        f"- Result root: `{result_root}`",
        f"- Phase 7 degradation rows: {len(df_degradation)}",
        f"- Phase 7 contrast rows: {len(df_contrast)}",
        f"- Phase 8 prototype summary rows: {len(df_prototype_stability)}",
        f"- Reduced coverage set: `{', '.join(_format_number(c, 2) for c in stability_coverages) if stability_coverages else 'NA'}`",
        f"- Prototype count in the Phase 8 table: {prototype_count}",
        "",
        "## Files Written By Phase 8",
        "",
        f"- `{ARM3_PHASE8_PROTOTYPE_STABILITY_BASENAME}.parquet`",
        f"- `{ARM3_PHASE8_PROTOTYPE_STABILITY_BASENAME}.csv`",
        f"- `{ARM3_PHASE8_MEMO_FILENAME}`",
        "",
    ]

    if not df_contrast.empty:
        contrast_summary = (
            df_contrast.groupby(["contrast_name", "coverage"], sort=True)
            .agg(
                median_abs_degradation=("abs_degradation", "median"),
                sign_consistency_rate=("sign_consistency_rate", "first"),
                n_evaluable=("n_evaluable", "first"),
            )
            .reset_index()
        )
        lines.extend(
            [
                "## Phase 7 Contrast Summary",
                "",
            ]
        )
        for _, row in contrast_summary.iterrows():
            lines.append(
                "- "
                f"{row['contrast_name']} at coverage={_format_number(row['coverage'], 2)}: "
                f"median_abs_degradation={_format_number(row['median_abs_degradation'])}, "
                f"sign_consistency_rate={_format_number(row['sign_consistency_rate'])}, "
                f"n_evaluable={int(row['n_evaluable'])}"
            )
        lines.append("")

    if not df_degradation.empty:
        degradation_summary = (
            df_degradation.groupby(["quantity", "coverage"], sort=True)
            .agg(
                median_abs_degradation=("median_abs_degradation", "median"),
                sign_consistency_rate=("sign_consistency_rate", "median"),
                floor_dominated_rate=("floor_dominated_rate", "median"),
            )
            .reset_index()
        )
        lines.extend(
            [
                "## Phase 7 Degradation Summary",
                "",
            ]
        )
        for _, row in degradation_summary.iterrows():
            lines.append(
                "- "
                f"{row['quantity']} at coverage={_format_number(row['coverage'], 2)}: "
                f"median_abs_degradation={_format_number(row['median_abs_degradation'])}, "
                f"median_sign_consistency_rate={_format_number(row['sign_consistency_rate'])}, "
                f"median_floor_dominated_rate={_format_number(row['floor_dominated_rate'])}"
            )
        lines.append("")

    if not df_prototype_stability.empty:
        stability_summary = (
            df_prototype_stability.groupby("coverage", sort=True)
            .agg(
                median_sign_consistency_rate=("sign_consistency_rate", "median"),
                median_correlation_to_full_cov=("correlation_to_full_cov", "median"),
                median_n_evaluable_patients=("n_evaluable_patients", "median"),
            )
            .reset_index()
        )
        lines.extend(
            [
                "## Phase 8 Prototype Summary Table",
                "",
            ]
        )
        for _, row in stability_summary.iterrows():
            lines.append(
                "- "
                f"coverage={_format_number(row['coverage'], 2)}: "
                f"median_sign_consistency_rate={_format_number(row['median_sign_consistency_rate'])}, "
                f"median_correlation_to_full_cov={_format_number(row['median_correlation_to_full_cov'])}, "
                f"median_n_evaluable_patients={_format_number(row['median_n_evaluable_patients'])}"
            )
        lines.append("")

    numeric_keys = [
        "run_utc",
        "lambda_dens",
        "tau_by_compartment",
        "tau_q",
        "lambda_grid_used",
        "target_alpha",
    ]
    numeric_meta = {key: calibration_record[key] for key in numeric_keys if key in calibration_record}
    if numeric_meta:
        lines.extend(
            [
                "## Frozen Calibration Metadata",
                "",
            ]
        )
        for key, value in numeric_meta.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")

    lines.extend(
        [
            "## Deferred Downstream Analysis",
            "",
            "- Thresholded interpretation is deferred to downstream analysis.",
            "- Biological prototype subsetting is deferred to downstream analysis.",
            "- This summary is descriptive only and does not assign outcome labels.",
            "",
        ]
    )

    return "\n".join(lines)


def write_phase8_outputs(
    result_root: Path | str,
    df_prototype_stability: pd.DataFrame,
    memo_text: str,
) -> None:
    """
    Write the Phase-8 output files to result_root.
    """
    result_root = Path(result_root)
    result_root.mkdir(parents=True, exist_ok=True)

    df_prototype_stability.to_parquet(
        result_root / f"{ARM3_PHASE8_PROTOTYPE_STABILITY_BASENAME}.parquet",
        index=False,
    )
    df_prototype_stability.to_csv(
        result_root / f"{ARM3_PHASE8_PROTOTYPE_STABILITY_BASENAME}.csv",
        index=False,
    )
    with open(result_root / ARM3_PHASE8_MEMO_FILENAME, "w", encoding="utf-8") as fh:
        fh.write(memo_text)
