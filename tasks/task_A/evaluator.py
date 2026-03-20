"""
Module: tasks.task_A.evaluator
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from slotar.contracts import validate_metrics_table

SUPPORTED_ARMS: set[str] = {"A1_baseline", "A1_broken_reference", "A2_cross_compartment", "A3_uq_stress"}
FIXED_MODE = "task_fixed_by_compartment"
JOINT_MODE = "pair_specific_joint"
UNAVAILABLE_MODE = "unavailable"
DENSITY_MODE = "density"
ARM2_PAIR_TYPE_TO_FAMILY = {
    "TC->IM": "TC-IM",
    "IM->TC": "TC-IM",
    "IM->PT": "IM-PT",
    "PT->IM": "IM-PT",
    "TC->PT": "TC-PT",
    "PT->TC": "TC-PT",
}
ARM3_PAIR_TYPE_TO_FAMILY = {
    "TC->IM": "TC-IM",
    "TC->PT": "TC-PT",
}
MICRO_METRICS: tuple[str, ...] = ("T", "D_pos", "B_pos", "d_rel", "b_rel", "M", "R", "tau")
REQUIRED_COLUMNS: tuple[str, ...] = (
    "patient_group_id",
    "pair_id",
    "arm",
    "patient_id",
    "compartment",
    "patient_id_a",
    "patient_id_b",
    "compartment_a",
    "compartment_b",
    "same_patient",
    "same_compartment",
    "pair_type",
    "roi_a",
    "roi_b",
    "lambda_pl",
    "lambda_mode",
    "tau_mode",
    "mass_mode",
    "uot_status",
    "bypass_reason",
    "mass_pruned_ratio",
    "n_min_proto_used",
    "S0",
    "S1",
    "scale_ratio",
    "U",
    *MICRO_METRICS,
)


def evaluate_task_a(df_metrics: pd.DataFrame, config: Mapping[str, Any]) -> None:
    """
    Task-A evaluator covering the shared Arm-I / Arm-II / Arm-III operator surface.
    """
    if df_metrics.empty:
        raise AssertionError("Task-A produced an empty metrics table")

    missing = [column for column in REQUIRED_COLUMNS if column not in df_metrics.columns]
    if missing:
        raise AssertionError(f"Task-A metrics table is missing required columns: {missing}")

    validate_metrics_table(df_metrics)

    arm_values = set(df_metrics["arm"].astype(str))
    enabled_arms = config.get("enabled_arms", [])
    expected_arms = set(enabled_arms)
    if not arm_values.issubset(SUPPORTED_ARMS):
        raise AssertionError(f"Task-A found unsupported arms: {sorted(arm_values - SUPPORTED_ARMS)}")
    if arm_values != expected_arms:
        raise AssertionError(f"Task-A expected arms {sorted(expected_arms)}, found {sorted(arm_values)}")

    if not np.isfinite(df_metrics["lambda_pl"].to_numpy(dtype=float)).all():
        raise AssertionError("lambda_pl must be finite for all Task-A rows")
    if not (df_metrics["patient_id"].astype(str) == df_metrics["patient_id_a"].astype(str)).all():
        raise AssertionError("patient_id must mirror patient_id_a for all Task-A rows")
    if not (df_metrics["compartment"].astype(str) == df_metrics["compartment_a"].astype(str)).all():
        raise AssertionError("compartment must mirror compartment_a for all Task-A rows")

    _evaluate_arm1(df_metrics)
    _evaluate_arm2(df_metrics)
    _evaluate_arm3(df_metrics)
    _evaluate_ok_nonok_contract(df_metrics)


def _evaluate_arm1(df_metrics: pd.DataFrame) -> None:
    constrained_mask = df_metrics["arm"] == "A1_baseline"
    broken_mask = df_metrics["arm"] == "A1_broken_reference"
    arm1_mask = constrained_mask | broken_mask
    if not arm1_mask.any():
        return

    arm1 = df_metrics.loc[arm1_mask]
    if not (arm1["lambda_mode"] == FIXED_MODE).all():
        raise AssertionError("Arm-I lambda_mode must equal 'task_fixed_by_compartment'")
    if not (arm1["tau_mode"] == FIXED_MODE).all():
        raise AssertionError("Arm-I tau_mode must equal 'task_fixed_by_compartment'")
    if not (arm1["mass_mode"] == DENSITY_MODE).all():
        raise AssertionError("Arm-I mass_mode must equal 'density'")

    if constrained_mask.any():
        constrained = df_metrics.loc[constrained_mask]
        if not constrained["same_patient"].all():
            raise AssertionError("A1_baseline rows must have same_patient=True")
        if not constrained["same_compartment"].all():
            raise AssertionError("A1_baseline rows must have same_compartment=True")
        if not (constrained["patient_id_a"].astype(str) == constrained["patient_id_b"].astype(str)).all():
            raise AssertionError("A1_baseline rows must keep patient_id_b equal to patient_id_a")
        if not (constrained["compartment_a"].astype(str) == constrained["compartment_b"].astype(str)).all():
            raise AssertionError("A1_baseline rows must keep compartment_b equal to compartment_a")

    if broken_mask.any():
        broken = df_metrics.loc[broken_mask]
        if broken["same_patient"].any():
            raise AssertionError("A1_broken_reference rows must have same_patient=False")
        if broken["same_compartment"].any():
            raise AssertionError("A1_broken_reference rows must have same_compartment=False")
        if (broken["patient_id_a"].astype(str) == broken["patient_id_b"].astype(str)).any():
            raise AssertionError("A1_broken_reference rows must break patient locality on side B")
        if (broken["compartment_a"].astype(str) == broken["compartment_b"].astype(str)).any():
            raise AssertionError("A1_broken_reference rows must break compartment locality on side B")


def _evaluate_arm2(df_metrics: pd.DataFrame) -> None:
    arm2_mask = df_metrics["arm"] == "A2_cross_compartment"
    if not arm2_mask.any():
        return

    arm2 = df_metrics.loc[arm2_mask]
    for column in ("pair_family", "M_balanced"):
        if column not in arm2.columns:
            raise AssertionError(f"Arm-II metrics must include column {column!r}")

    if not (arm2["lambda_mode"] == JOINT_MODE).all():
        raise AssertionError("Arm-II lambda_mode must equal 'pair_specific_joint'")
    if not (arm2["tau_mode"] == UNAVAILABLE_MODE).all():
        raise AssertionError("Arm-II tau_mode must equal 'unavailable'")
    if not (arm2["mass_mode"] == DENSITY_MODE).all():
        raise AssertionError("Arm-II mass_mode must equal 'density'")
    if not arm2["same_patient"].all():
        raise AssertionError("Arm-II rows must have same_patient=True")
    if arm2["same_compartment"].any():
        raise AssertionError("Arm-II rows must have same_compartment=False")
    if not (arm2["patient_id_a"].astype(str) == arm2["patient_id_b"].astype(str)).all():
        raise AssertionError("Arm-II rows must remain within-patient")
    if (arm2["compartment_a"].astype(str) == arm2["compartment_b"].astype(str)).any():
        raise AssertionError("Arm-II rows must remain cross-compartment")

    invalid_pair_types = sorted(set(arm2["pair_type"].astype(str)) - set(ARM2_PAIR_TYPE_TO_FAMILY))
    if invalid_pair_types:
        raise AssertionError(f"Arm-II rows contain invalid pair_type values: {invalid_pair_types}")

    derived_families = arm2["pair_type"].astype(str).map(ARM2_PAIR_TYPE_TO_FAMILY)
    if not (derived_families.astype(str) == arm2["pair_family"].astype(str)).all():
        raise AssertionError("Arm-II pair_family values must match the declared pair_type")

    for pair_family, group in arm2.groupby("pair_family", sort=False):
        unique_lambda = np.unique(group["lambda_pl"].to_numpy(dtype=float))
        if unique_lambda.size != 1:
            raise AssertionError(
                f"Arm-II rows in pair_family {pair_family!r} must share one calibrated lambda_pl"
            )


def _evaluate_arm3(df_metrics: pd.DataFrame) -> None:
    arm3_mask = df_metrics["arm"] == "A3_uq_stress"
    if not arm3_mask.any():
        return

    arm3 = df_metrics.loc[arm3_mask]
    for column in (
        "pair_family",
        "lambda_dens",
        "U_abs_dens",
        "S_src",
        "S_tgt",
        "Delta_scale",
        "Q_src_dens",
        "Q_tgt_dens",
    ):
        if column not in arm3.columns:
            raise AssertionError(f"Arm-III metrics must include column {column!r}")

    if not (arm3["lambda_mode"] == JOINT_MODE).all():
        raise AssertionError("Arm-III lambda_mode must equal 'pair_specific_joint'")
    if not (arm3["tau_mode"] == FIXED_MODE).all():
        raise AssertionError("Arm-III tau_mode must equal 'task_fixed_by_compartment'")
    if not (arm3["mass_mode"] == DENSITY_MODE).all():
        raise AssertionError("Arm-III mass_mode must equal 'density'")
    if not arm3["same_patient"].all():
        raise AssertionError("Arm-III rows must have same_patient=True")
    if arm3["same_compartment"].any():
        raise AssertionError("Arm-III rows must have same_compartment=False")
    if not (arm3["patient_id_a"].astype(str) == arm3["patient_id_b"].astype(str)).all():
        raise AssertionError("Arm-III rows must remain within-patient")
    if (arm3["compartment_a"].astype(str) == arm3["compartment_b"].astype(str)).any():
        raise AssertionError("Arm-III rows must remain cross-compartment")

    invalid_pair_types = sorted(set(arm3["pair_type"].astype(str)) - set(ARM3_PAIR_TYPE_TO_FAMILY))
    if invalid_pair_types:
        raise AssertionError(f"Arm-III rows contain invalid pair_type values: {invalid_pair_types}")

    derived_families = arm3["pair_type"].astype(str).map(ARM3_PAIR_TYPE_TO_FAMILY)
    if not (derived_families.astype(str) == arm3["pair_family"].astype(str)).all():
        raise AssertionError("Arm-III pair_family values must match the declared pair_type")

    if not np.isfinite(arm3["lambda_dens"].to_numpy(dtype=float)).all():
        raise AssertionError("Arm-III lambda_dens must be finite for all rows")
    np.testing.assert_allclose(
        arm3["lambda_pl"].to_numpy(dtype=float),
        arm3["lambda_dens"].to_numpy(dtype=float),
        err_msg="Arm-III lambda_pl must mirror lambda_dens on the shared operator surface",
    )

    for pair_family, group in arm3.groupby("pair_family", sort=False):
        unique_lambda = np.unique(group["lambda_pl"].to_numpy(dtype=float))
        if unique_lambda.size != 1:
            raise AssertionError(
                f"Arm-III rows in pair_family {pair_family!r} must share one frozen lambda_pl"
            )


def _evaluate_ok_nonok_contract(df_metrics: pd.DataFrame) -> None:
    ok_mask = df_metrics["uot_status"] == "ok"
    non_ok_mask = ~ok_mask

    if ok_mask.any():
        ok_df = df_metrics.loc[ok_mask]
        np.testing.assert_allclose(
            ok_df["U"].to_numpy(dtype=float),
            ok_df["D_pos"].to_numpy(dtype=float) + ok_df["B_pos"].to_numpy(dtype=float),
            rtol=1e-8,
            atol=1e-10,
        )

    arm1_ok = ok_mask & df_metrics["arm"].isin({"A1_baseline", "A1_broken_reference"})
    if arm1_ok.any():
        arm1_ok_df = df_metrics.loc[arm1_ok]
        finite_columns = (
            "S0",
            "S1",
            "scale_ratio",
            "lambda_pl",
            "T",
            "D_pos",
            "B_pos",
            "d_rel",
            "b_rel",
            "M",
            "R",
            "tau",
        )
        for column in finite_columns:
            if not np.isfinite(arm1_ok_df[column].to_numpy(dtype=float)).all():
                raise AssertionError(f"Arm-I ok rows must have finite values for {column!r}")

    arm2_ok = ok_mask & (df_metrics["arm"] == "A2_cross_compartment")
    if arm2_ok.any():
        arm2_ok_df = df_metrics.loc[arm2_ok]
        finite_columns = (
            "S0",
            "S1",
            "scale_ratio",
            "lambda_pl",
            "T",
            "D_pos",
            "B_pos",
            "d_rel",
            "b_rel",
            "M",
            "M_balanced",
        )
        for column in finite_columns:
            if not np.isfinite(arm2_ok_df[column].to_numpy(dtype=float)).all():
                raise AssertionError(f"Arm-II ok rows must have finite values for {column!r}")
        if not arm2_ok_df["R"].isna().all():
            raise AssertionError("Arm-II ok rows must keep R as NaN")
        if not arm2_ok_df["tau"].isna().all():
            raise AssertionError("Arm-II ok rows must keep tau as NaN")

    arm3_ok = ok_mask & (df_metrics["arm"] == "A3_uq_stress")
    if arm3_ok.any():
        arm3_ok_df = df_metrics.loc[arm3_ok]
        finite_columns = (
            "S0",
            "S1",
            "scale_ratio",
            "lambda_pl",
            "lambda_dens",
            "T",
            "D_pos",
            "B_pos",
            "d_rel",
            "b_rel",
            "M",
            "R",
            "tau",
            "U_abs_dens",
            "S_src",
            "S_tgt",
            "Delta_scale",
            "Q_src_dens",
            "Q_tgt_dens",
        )
        for column in finite_columns:
            if not np.isfinite(arm3_ok_df[column].to_numpy(dtype=float)).all():
                raise AssertionError(f"Arm-III ok rows must have finite values for {column!r}")

    if non_ok_mask.any():
        non_ok_df = df_metrics.loc[non_ok_mask]
        for column in MICRO_METRICS:
            if not non_ok_df[column].isna().all():
                raise AssertionError(f"Non-ok rows must have NaN micro metrics for {column!r}")
        if not non_ok_df["U"].isna().all():
            raise AssertionError("Non-ok rows must have NaN for 'U'")

        arm2_non_ok = non_ok_df["arm"] == "A2_cross_compartment"
        if arm2_non_ok.any() and not non_ok_df.loc[arm2_non_ok, "M_balanced"].isna().all():
            raise AssertionError("Arm-II non-ok rows must keep M_balanced as NaN")

        arm3_non_ok = non_ok_df["arm"] == "A3_uq_stress"
        if arm3_non_ok.any():
            for column in ("U_abs_dens", "Q_src_dens", "Q_tgt_dens"):
                if not non_ok_df.loc[arm3_non_ok, column].isna().all():
                    raise AssertionError(f"Arm-III non-ok rows must keep {column!r} as NaN")

    ok_bypass = df_metrics.loc[ok_mask, "bypass_reason"]
    if not ok_bypass.isna().all():
        raise AssertionError("Ok rows must have null bypass_reason")
    if non_ok_mask.any() and df_metrics.loc[non_ok_mask, "bypass_reason"].isna().any():
        raise AssertionError("Non-ok rows must carry a task-level bypass_reason")

    if not np.isfinite(pd.to_numeric(df_metrics["mass_pruned_ratio"], errors="coerce").to_numpy(dtype=float)).all():
        raise AssertionError("Task-A rows must carry finite mass_pruned_ratio values")
    n_min_proto_used = pd.to_numeric(df_metrics["n_min_proto_used"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(n_min_proto_used).all():
        raise AssertionError("Task-A rows must carry finite n_min_proto_used values")
