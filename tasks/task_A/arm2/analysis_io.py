"""
Module: tasks.task_A.arm2.analysis_io

Load and validate frozen post-hoc inputs for the Arm-II focused rewrite.

This module is intentionally limited to:
- reading task-scoped config,
- loading the Arm-II metrics parquet,
- validating the locked startup-slice contract,
- loading the frozen Stage-0 bundle,
- reconstructing pair tensors on the shared prototype axis.

This module must not:
- rerun UOT or Balanced OT,
- compute baseline summaries,
- compute transport/unmatched summaries,
- write any outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml

from ..common import build_task_a_density_reference_from_arrays
from ..runtime_contract import load_task_a_config as load_shared_task_a_config
from .analysis_contract import (
    Arm2FocusedPaths,
    ARM_NAME,
    AUDIT_ONLY_FAMILIES,
    CONFIRMATORY_FAMILIES,
    LoadedArm2Inputs,
    PAIR_FAMILY_ORDER,
    PAIR_FAMILY_ROLE,
    PAIR_TYPE_ORDER,
    PAIR_TYPE_TO_FAMILY,
    PairTensorBundle,
    Stage0AnalysisBundle,
)

# ---------------------------------------------------------------------------
# Config and metrics loading
# ---------------------------------------------------------------------------


REQUIRED_METRIC_COLUMNS: tuple[str, ...] = (
    "arm",
    "pair_id",
    "patient_group_id",
    "patient_id",
    "patient_id_a",
    "patient_id_b",
    "compartment",
    "compartment_a",
    "compartment_b",
    "same_patient",
    "same_compartment",
    "pair_type",
    "pair_family",
    "roi_a",
    "roi_b",
    "lambda_pl",
    "lambda_mode",
    "tau_mode",
    "mass_mode",
    "uot_status",
)


def _decode_h5_strings(values: np.ndarray) -> np.ndarray:
    """Decode the byte-heavy string arrays emitted by `.h5ad` categorical storage."""

    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in values
        ],
        dtype=object,
    )


def _build_patient_roi_audit_table(roi_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct the stable patient-level ROI audit used across Arm-II summaries."""

    required = {"patient_id", "compartment", "roi_id"}
    missing = sorted(required - set(roi_table.columns))
    if missing:
        raise ValueError(f"ROI audit requires columns {missing}")

    dedup = (
        roi_table.loc[:, ["patient_id", "compartment", "roi_id"]]
        .drop_duplicates()
        .copy()
    )
    dedup["patient_id"] = dedup["patient_id"].astype(str)
    dedup["compartment"] = dedup["compartment"].astype(str)
    dedup["roi_id"] = dedup["roi_id"].astype(str)

    audit = (
        dedup.groupby(["patient_id", "compartment"], sort=True, observed=False)
        .size()
        .rename("n_roi")
        .reset_index()
        .pivot(index="patient_id", columns="compartment", values="n_roi")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    for compartment in ("TC", "IM", "PT"):
        if compartment not in audit.columns:
            audit[compartment] = 0

    audit = audit.loc[:, ["patient_id", "TC", "IM", "PT"]].copy()
    audit.rename(
        columns={"TC": "n_TC", "IM": "n_IM", "PT": "n_PT"},
        inplace=True,
    )
    audit["total_n_roi"] = audit[["n_TC", "n_IM", "n_PT"]].sum(axis=1)
    audit["is_nominal_3_3_3"] = (
        audit["n_TC"].eq(3) & audit["n_IM"].eq(3) & audit["n_PT"].eq(3)
    )
    audit["deviation_pattern"] = audit.apply(
        lambda row: (
            "nominal_3_3_3"
            if bool(row["is_nominal_3_3_3"])
            else f"TC={int(row['n_TC'])},IM={int(row['n_IM'])},PT={int(row['n_PT'])}"
        ),
        axis=1,
    )
    audit["ordered_rows_TC_IM"] = 2 * audit["n_TC"] * audit["n_IM"]
    audit["ordered_rows_IM_PT"] = 2 * audit["n_IM"] * audit["n_PT"]
    audit["ordered_rows_TC_PT"] = 2 * audit["n_TC"] * audit["n_PT"]
    audit = audit.sort_values("patient_id").reset_index(drop=True)

    deviation_pattern_counts = (
        audit["deviation_pattern"]
        .value_counts()
        .rename_axis("deviation_pattern")
        .reset_index(name="patient_count")
        .sort_values("deviation_pattern")
        .reset_index(drop=True)
    )
    return audit, deviation_pattern_counts


def _require_startup_checks_pass(validation: pd.DataFrame) -> None:
    """Promote any contract violation to a hard failure before analysis starts."""

    failed = validation.loc[~validation["passed"].astype(bool)].copy()
    if failed.empty:
        return
    details = "; ".join(
        f"{row['check']}={row['detail']}" for _, row in failed.iterrows()
    )
    raise ValueError(f"Arm-II startup contract validation failed: {details}")


def load_task_config(path: Path) -> dict[str, Any]:
    """Load the Task-A config needed by the post-hoc Arm-II focused rewrite."""

    loaded = load_shared_task_a_config(path)
    if "data" not in loaded or not isinstance(loaded["data"], dict):
        raise ValueError("Task-A config is missing the required 'data' mapping")
    if "k_full" not in loaded["data"]:
        raise ValueError("Task-A config is missing data.k_full")
    return loaded


def load_arm2_metrics(path: Path) -> pd.DataFrame:
    """Load the Arm-II metrics parquet and return the raw metrics table."""

    metrics_df = pd.read_parquet(path)
    missing = [column for column in REQUIRED_METRIC_COLUMNS if column not in metrics_df.columns]
    if missing:
        raise ValueError(f"Input parquet is missing required Arm-II columns: {missing}")

    metrics_df = metrics_df.loc[metrics_df["arm"].astype(str) == ARM_NAME].copy()
    if metrics_df.empty:
        raise ValueError(f"No rows with arm={ARM_NAME!r} were found in {path}")

    string_columns = (
        "arm",
        "pair_id",
        "patient_group_id",
        "patient_id",
        "patient_id_a",
        "patient_id_b",
        "compartment",
        "compartment_a",
        "compartment_b",
        "pair_type",
        "pair_family",
        "roi_a",
        "roi_b",
        "lambda_mode",
        "tau_mode",
        "mass_mode",
        "uot_status",
    )
    for column in string_columns:
        metrics_df[column] = metrics_df[column].astype(str)
    return metrics_df.reset_index(drop=True)


def validate_arm2_startup_contract(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the locked Arm-II startup-slice contract on the loaded metrics.

    Required scope comments for implementation:
    - confirmatory families: TC-IM, TC-PT
    - audit-only / exploratory family: IM-PT
    - no confirmatory claim or confirmatory contrast should be based on IM-PT
    """

    validation = pd.DataFrame.from_records(
        [
            {
                "check": "all_rows_arm2",
                "passed": bool((metrics_df["arm"].astype(str) == ARM_NAME).all()),
                "detail": f"unique_arms={sorted(metrics_df['arm'].astype(str).unique().tolist())}",
            },
            {
                "check": "same_patient_true",
                "passed": bool(metrics_df["same_patient"].astype(bool).all()),
                "detail": f"violations={int((~metrics_df['same_patient'].astype(bool)).sum())}",
            },
            {
                "check": "same_compartment_false",
                "passed": bool((~metrics_df["same_compartment"].astype(bool)).all()),
                "detail": f"violations={int(metrics_df['same_compartment'].astype(bool).sum())}",
            },
            {
                "check": "patient_ids_within_patient",
                "passed": bool(
                    (
                        metrics_df["patient_id_a"].astype(str)
                        == metrics_df["patient_id_b"].astype(str)
                    ).all()
                ),
                "detail": (
                    f"violations={int((metrics_df['patient_id_a'].astype(str) != metrics_df['patient_id_b'].astype(str)).sum())}"
                ),
            },
            {
                "check": "pair_type_vocab_valid",
                "passed": bool(metrics_df["pair_type"].astype(str).isin(PAIR_TYPE_ORDER).all()),
                "detail": (
                    "invalid="
                    f"{sorted(metrics_df.loc[~metrics_df['pair_type'].astype(str).isin(PAIR_TYPE_ORDER), 'pair_type'].astype(str).unique().tolist())}"
                ),
            },
            {
                "check": "pair_family_vocab_valid",
                "passed": bool(metrics_df["pair_family"].astype(str).isin(PAIR_FAMILY_ORDER).all()),
                "detail": (
                    "invalid="
                    f"{sorted(metrics_df.loc[~metrics_df['pair_family'].astype(str).isin(PAIR_FAMILY_ORDER), 'pair_family'].astype(str).unique().tolist())}"
                ),
            },
            {
                "check": "pair_family_matches_pair_type",
                "passed": bool(
                    (
                        metrics_df["pair_family"].astype(str)
                        == metrics_df["pair_type"].astype(str).map(PAIR_TYPE_TO_FAMILY).fillna("__invalid__")
                    ).all()
                ),
                "detail": (
                    f"violations={int((metrics_df['pair_family'].astype(str) != metrics_df['pair_type'].astype(str).map(PAIR_TYPE_TO_FAMILY).fillna('__invalid__')).sum())}"
                ),
            },
            {
                "check": "pair_id_unique",
                "passed": bool(metrics_df["pair_id"].astype(str).is_unique),
                "detail": f"duplicate_count={int(metrics_df['pair_id'].astype(str).duplicated().sum())}",
            },
            {
                "check": "patient_group_id_unique",
                "passed": bool(metrics_df["patient_group_id"].astype(str).is_unique),
                "detail": (
                    f"duplicate_count={int(metrics_df['patient_group_id'].astype(str).duplicated().sum())}"
                ),
            },
            {
                "check": "mass_mode_density",
                "passed": bool((metrics_df["mass_mode"].astype(str) == "density").all()),
                "detail": f"observed={metrics_df['mass_mode'].astype(str).value_counts(dropna=False).to_dict()}",
            },
            {
                "check": "tau_mode_unavailable",
                "passed": bool((metrics_df["tau_mode"].astype(str) == "unavailable").all()),
                "detail": f"observed={metrics_df['tau_mode'].astype(str).value_counts(dropna=False).to_dict()}",
            },
            {
                "check": "confirmatory_families_present",
                "passed": bool(set(CONFIRMATORY_FAMILIES).issubset(set(metrics_df["pair_family"].astype(str)))),
                "detail": (
                    f"observed_confirmatory={sorted(set(metrics_df['pair_family'].astype(str)) & set(CONFIRMATORY_FAMILIES))}"
                ),
            },
            {
                "check": "audit_family_present",
                "passed": bool(set(AUDIT_ONLY_FAMILIES).issubset(set(metrics_df["pair_family"].astype(str)))),
                "detail": f"observed_audit={sorted(set(metrics_df['pair_family'].astype(str)) & set(AUDIT_ONLY_FAMILIES))}",
            },
        ]
    )
    return validation


# ---------------------------------------------------------------------------
# Frozen Stage-0 bundle and pair tensors
# ---------------------------------------------------------------------------


def load_stage0_analysis_bundle(path: Path, expected_k: int) -> Stage0AnalysisBundle:
    """Load the frozen Stage-0 analysis bundle required by Arm-II analysis."""

    with h5py.File(path, "r") as handle:
        cost_matrix = np.asarray(handle["uns/cost_matrix"], dtype=float)
        if cost_matrix.shape != (expected_k, expected_k):
            raise ValueError(
                "Stage-0 cost_matrix shape does not match the configured shared prototype axis: "
                f"expected {(expected_k, expected_k)}, got {cost_matrix.shape}"
            )

        cost_scale = float(np.asarray(handle["uns/s_C"][()]).item())
        proto_ids = np.asarray(handle["obs/proto_id"], dtype=int)

        roi_codes = np.asarray(handle["obs/roi_id/codes"], dtype=int)
        roi_categories = _decode_h5_strings(handle["obs/roi_id/categories"][()])
        patient_codes = np.asarray(handle["obs/patient_id/codes"], dtype=int)
        patient_categories = _decode_h5_strings(handle["obs/patient_id/categories"][()])
        compartment_codes = np.asarray(handle["obs/compartment/codes"], dtype=int)
        compartment_categories = _decode_h5_strings(handle["obs/compartment/categories"][()])
        cell_type_codes = np.asarray(handle["obs/cell_type/codes"], dtype=int)
        cell_type_categories = _decode_h5_strings(handle["obs/cell_type/categories"][()])
        spatial_xy = np.asarray(handle["obsm/spatial"], dtype=float)

    valid_proto = (proto_ids >= 0) & (proto_ids < expected_k)
    if not valid_proto.any():
        raise ValueError("Stage-0 artifact does not contain any valid prototype assignments")

    valid_roi = valid_proto & (roi_codes >= 0) & (roi_codes < roi_categories.shape[0])
    roi_vectors_array = np.zeros((roi_categories.shape[0], expected_k), dtype=float)
    np.add.at(roi_vectors_array, (roi_codes[valid_roi], proto_ids[valid_roi]), 1.0)
    roi_vectors = {
        str(roi_categories[idx]): roi_vectors_array[idx].copy()
        for idx in range(roi_categories.shape[0])
    }
    roi_density_vectors, _roi_count_vectors, _roi_total_areas = build_task_a_density_reference_from_arrays(
        spatial_xy=spatial_xy[valid_roi],
        roi_ids=roi_categories[roi_codes[valid_roi]],
        proto_ids=proto_ids[valid_roi],
        k_full=expected_k,
    )

    roi_audit_df = pd.DataFrame(
        {
            "patient_id": [
                patient_categories[idx] if idx >= 0 else None
                for idx in patient_codes.tolist()
            ],
            "compartment": [
                compartment_categories[idx] if idx >= 0 else None
                for idx in compartment_codes.tolist()
            ],
            "roi_id": [
                roi_categories[idx] if idx >= 0 else None
                for idx in roi_codes.tolist()
            ],
        }
    ).dropna()
    patient_roi_audit_table, deviation_pattern_counts = _build_patient_roi_audit_table(
        roi_audit_df,
    )

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
    active_prototype_ids = np.flatnonzero(prototype_totals > 0.0)
    if active_prototype_ids.size == 0:
        raise ValueError("Stage-0 artifact does not contain any active prototype IDs")

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

    return Stage0AnalysisBundle(
        roi_vectors=roi_vectors,
        roi_density_vectors=roi_density_vectors,
        cost_matrix=cost_matrix,
        cost_scale=cost_scale,
        active_prototype_ids=active_prototype_ids.astype(int),
        patient_roi_audit_table=patient_roi_audit_table,
        deviation_pattern_counts=deviation_pattern_counts,
        prototype_cell_type_table=pd.DataFrame.from_records(prototype_records),
    )


def build_pair_metadata(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Build the ordered Arm-II pair metadata table used across the rewrite."""

    pair_metadata = metrics_df.loc[
        :,
        [
            "pair_id",
            "patient_id",
            "pair_family",
            "pair_type",
            "compartment_a",
            "compartment_b",
            "roi_a",
            "roi_b",
        ],
    ].copy()
    pair_metadata.rename(
        columns={
            "pair_type": "ordered_direction",
            "compartment_a": "source_compartment",
            "compartment_b": "target_compartment",
            "roi_a": "source_roi_id",
            "roi_b": "target_roi_id",
        },
        inplace=True,
    )
    pair_metadata["pair_family_role"] = pair_metadata["pair_family"].astype(str).map(PAIR_FAMILY_ROLE)
    pair_metadata["pair_family"] = pd.Categorical(
        pair_metadata["pair_family"],
        categories=PAIR_FAMILY_ORDER,
        ordered=True,
    )
    pair_metadata["ordered_direction"] = pd.Categorical(
        pair_metadata["ordered_direction"],
        categories=PAIR_TYPE_ORDER,
        ordered=True,
    )
    pair_metadata = pair_metadata.sort_values(
        ["patient_id", "ordered_direction", "source_roi_id", "target_roi_id", "pair_id"]
    ).reset_index(drop=True)
    return pair_metadata


def reconstruct_pair_tensors(
    metrics_df: pd.DataFrame,
    stage0_bundle: Stage0AnalysisBundle,
    k_full: int,
) -> PairTensorBundle:
    """
    Reconstruct ordered pair tensors `A` and `B` on the shared prototype axis.

    This function should only prepare frozen post-hoc tensors. It must not
    compute baseline summaries or invoke solver internals.
    """

    pair_metadata = build_pair_metadata(metrics_df)
    missing = sorted(
        set(pair_metadata["source_roi_id"].astype(str))
        .union(set(pair_metadata["target_roi_id"].astype(str)))
        - set(stage0_bundle.roi_vectors)
    )
    if missing:
        raise ValueError(
            "Arm-II rows reference ROI IDs that are absent from the Stage-0 artifact: "
            f"{missing}"
        )

    A = np.vstack(
        [
            stage0_bundle.roi_vectors[str(roi_id)]
            for roi_id in pair_metadata["source_roi_id"].astype(str)
        ]
    ).astype(float, copy=False)
    B = np.vstack(
        [
            stage0_bundle.roi_vectors[str(roi_id)]
            for roi_id in pair_metadata["target_roi_id"].astype(str)
        ]
    ).astype(float, copy=False)
    if A.shape != (pair_metadata.shape[0], k_full) or B.shape != (pair_metadata.shape[0], k_full):
        raise ValueError("Reconstructed Arm-II tensors do not match the expected [N, K] shape")
    A_density = np.vstack(
        [
            stage0_bundle.roi_density_vectors[str(roi_id)]
            for roi_id in pair_metadata["source_roi_id"].astype(str)
        ]
    ).astype(float, copy=False)
    B_density = np.vstack(
        [
            stage0_bundle.roi_density_vectors[str(roi_id)]
            for roi_id in pair_metadata["target_roi_id"].astype(str)
        ]
    ).astype(float, copy=False)
    if (
        A_density.shape != (pair_metadata.shape[0], k_full)
        or B_density.shape != (pair_metadata.shape[0], k_full)
    ):
        raise ValueError("Reconstructed Arm-II density tensors do not match the expected [N, K] shape")

    return PairTensorBundle(
        A=A,
        B=B,
        A_density=A_density,
        B_density=B_density,
        k_full=int(k_full),
        pair_metadata=pair_metadata,
    )


# ---------------------------------------------------------------------------
# Top-level input assembly
# ---------------------------------------------------------------------------


def load_inputs(paths: Arm2FocusedPaths) -> LoadedArm2Inputs:
    """
    Assemble the fully loaded frozen-input bundle for the Arm-II focused rewrite.

    This is the top-level IO boundary. No solver rerun or summary aggregation
    should be added here.
    """

    task_config = load_task_config(paths.task_config)
    metrics_df = load_arm2_metrics(paths.arm2_metrics_parquet)
    startup_validation = validate_arm2_startup_contract(metrics_df)
    _require_startup_checks_pass(startup_validation)

    k_full = int(task_config["data"]["k_full"])
    stage0_bundle = load_stage0_analysis_bundle(paths.stage0_h5ad, expected_k=k_full)
    pair_tensors = reconstruct_pair_tensors(
        metrics_df=metrics_df,
        stage0_bundle=stage0_bundle,
        k_full=k_full,
    )
    return LoadedArm2Inputs(
        paths=paths,
        task_config=task_config,
        metrics_df=metrics_df,
        stage0=stage0_bundle,
        pair_tensors=pair_tensors,
    )
