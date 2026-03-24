"""
Module: tasks.task_A.arm2_spatial_gradient
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData

from slotar.contracts import COST_SCALE_ALIASES
from slotar.uot import UOTSolveConfig, calibrate_joint_lambda

from .arm1_noise_baseline import build_arm1_roi_table
from .common import (
    TaskARoiReferenceBundle,
    assemble_tensors,
    resolve_task_a_mass_mode,
    run_balanced_ot_batch,
    run_uot_batch_safe,
)

ARM_NAME = "A2_cross_compartment"
LAMBDA_MODE = "pair_specific_joint"
TAU_MODE = "unavailable"
ORDERED_PAIR_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("TC->IM", "TC-IM", "TC", "IM"),
    ("IM->TC", "TC-IM", "IM", "TC"),
    ("IM->PT", "IM-PT", "IM", "PT"),
    ("PT->IM", "IM-PT", "PT", "IM"),
    ("TC->PT", "TC-PT", "TC", "PT"),
    ("PT->TC", "TC-PT", "PT", "TC"),
)
PAIR_FAMILIES: tuple[str, ...] = ("TC-IM", "IM-PT", "TC-PT")


def run_arm2(
    adata: AnnData,
    config: Mapping[str, Any],
    uot_cfg: UOTSolveConfig,
    kernels: Sequence[np.ndarray],
    roi_references: TaskARoiReferenceBundle | None = None,
) -> pd.DataFrame:
    """
    Generate deterministic within-patient ordered cross-compartment pairs for
    the Task-A Arm-II startup slice, then run family-level shared-lambda UOT
    together with the same-pair Balanced OT comparator.
    """
    pair_meta = generate_cross_compartment_pairs(adata)
    if pair_meta.empty:
        raise ValueError("Arm II produced no eligible within-patient cross-compartment ROI pairs")

    k_full = int(config["data"]["k_full"])
    mass_mode = resolve_task_a_mass_mode(config, ARM_NAME)

    A, B, mass_gap = assemble_tensors(
        adata,
        pair_meta,
        k_full=k_full,
        mass_mode=mass_mode,
        roi_references=roi_references,
    )
    pair_meta = pair_meta.copy()
    pair_meta["mass_gap"] = mass_gap

    arm2_cfg = config["arm2"]
    lambda_grid = tuple(float(value) for value in arm2_cfg["lambda_grid"])
    target_alpha = float(arm2_cfg.get("target_alpha", 0.05))

    lambda_pl = np.zeros(pair_meta.shape[0], dtype=float)
    for pair_family in PAIR_FAMILIES:
        family_mask = pair_meta["pair_family"].astype(str) == pair_family
        if not family_mask.any():
            continue

        family_lambda = calibrate_joint_lambda(
            A=A[family_mask.to_numpy()],
            B=B[family_mask.to_numpy()],
            lambda_grid=lambda_grid,
            kernels=kernels,
            cfg=uot_cfg,
            target_alpha=target_alpha,
        )
        lambda_pl[family_mask.to_numpy()] = family_lambda

    if (lambda_pl <= 0.0).any():
        raise ValueError("Arm II failed to assign a positive shared lambda to every pair family")

    pair_meta["arm"] = ARM_NAME
    pair_meta["lambda_mode"] = LAMBDA_MODE
    pair_meta["tau_mode"] = TAU_MODE
    pair_meta["mass_mode"] = mass_mode
    pair_meta["group_mode"] = "provided"
    pair_meta["drift_mode"] = "unavailable"

    df_metrics = run_uot_batch_safe(
        A=A,
        B=B,
        lambda_pl=lambda_pl,
        kernels=kernels,
        uot_cfg=uot_cfg,
        pair_meta=pair_meta,
        tau_external=None,
    )

    balanced_cost = run_balanced_ot_batch(
        A=A,
        B=B,
        cost_matrix=_get_scaled_cost_matrix(adata),
        n_min_proto=uot_cfg.n_min_proto,
    )
    ok_mask = df_metrics["uot_status"] == "ok"
    df_metrics["M_balanced"] = np.where(ok_mask, balanced_cost, np.nan)
    return df_metrics


def generate_cross_compartment_pairs(adata: AnnData) -> pd.DataFrame:
    roi_table = build_arm1_roi_table(adata)
    records: list[dict[str, Any]] = []

    for patient_id, patient_df in roi_table.groupby("patient_id", sort=True):
        roi_by_compartment = {
            compartment: sorted(group["roi_id"].astype(str).tolist())
            for compartment, group in patient_df.groupby("compartment", sort=True)
        }

        for pair_type, pair_family, source_compartment, target_compartment in ORDERED_PAIR_SPECS:
            source_rois = roi_by_compartment.get(source_compartment, [])
            target_rois = roi_by_compartment.get(target_compartment, [])
            if not source_rois or not target_rois:
                continue

            for roi_a in source_rois:
                for roi_b in target_rois:
                    pair_id = f"{ARM_NAME}::{pair_type}::{patient_id}::{roi_a}::{roi_b}"
                    records.append(
                        {
                            "pair_id": pair_id,
                            "patient_group_id": pair_id,
                            "patient_id": patient_id,
                            "compartment": source_compartment,
                            "patient_id_a": patient_id,
                            "patient_id_b": patient_id,
                            "compartment_a": source_compartment,
                            "compartment_b": target_compartment,
                            "same_patient": True,
                            "same_compartment": False,
                            "pair_type": pair_type,
                            "pair_family": pair_family,
                            "roi_a": roi_a,
                            "roi_b": roi_b,
                        }
                    )

    return pd.DataFrame.from_records(records)


def _get_scaled_cost_matrix(adata: AnnData) -> np.ndarray:
    if "cost_matrix" not in adata.uns:
        raise ValueError("Validated AnnData is missing 'cost_matrix' for Arm II")

    scale = _resolve_cost_scale(adata)
    return np.asarray(adata.uns["cost_matrix"], dtype=float) / scale


def _resolve_cost_scale(adata: AnnData) -> float:
    if "s_C" in adata.uns:
        return float(adata.uns["s_C"])
    for alias in COST_SCALE_ALIASES:
        if alias in adata.uns:
            return float(adata.uns[alias])
    raise ValueError("Validated AnnData is missing a usable SLOTAR cost scale")
