"""
Module: tasks.task_A.arm1_noise_baseline
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData

from slotar.uot import UOTSolveConfig

from .common import assemble_tensors, run_uot_batch_safe

ARM_NAME = "A1_baseline"
FIXED_MODE = "task_fixed_by_compartment"


def build_arm1_roi_table(adata: AnnData) -> pd.DataFrame:
    obs = adata.obs.loc[:, ["patient_id", "compartment", "roi_id"]].drop_duplicates().copy()
    obs["patient_id"] = obs["patient_id"].astype(str)
    obs["compartment"] = obs["compartment"].astype(str)
    obs["roi_id"] = obs["roi_id"].astype(str)
    return obs.sort_values(["patient_id", "compartment", "roi_id"]).reset_index(drop=True)


def generate_anchored_arm1_slots(
    adata: AnnData,
    *,
    n_draws: int = 1,
    random_seed: int = 42,
) -> pd.DataFrame:
    obs = build_arm1_roi_table(adata)

    if n_draws <= 0:
        raise ValueError(f"n_draws must be a positive integer, got {n_draws}")

    rng = np.random.default_rng(random_seed)
    records: list[dict[str, Any]] = []
    groups: list[tuple[str, str, list[str]]] = []
    for (patient_id, compartment), group in obs.groupby(["patient_id", "compartment"], sort=True):
        roi_ids = sorted(group["roi_id"].tolist())
        if len(roi_ids) < 2:
            continue
        groups.append((patient_id, compartment, roi_ids))

    for draw_idx in range(1, n_draws + 1):
        draw_label = f"draw_{draw_idx:04d}"
        for slot_index, (patient_id, compartment, roi_ids) in enumerate(groups, start=1):
            roi_a, roi_b_local = rng.choice(roi_ids, size=2, replace=False).tolist()
            records.append(
                {
                    "draw_number": draw_idx,
                    "draw_label": draw_label,
                    "slot_index": slot_index,
                    "patient_id_a": patient_id,
                    "compartment_a": compartment,
                    "roi_a": roi_a,
                    "roi_b_local": roi_b_local,
                }
            )

    return pd.DataFrame.from_records(records)


def run_arm1(
    adata: AnnData,
    config: Mapping[str, Any],
    uot_cfg: UOTSolveConfig,
    kernels: Sequence[np.ndarray],
) -> pd.DataFrame:
    """
    Generate repeated within-patient, within-compartment ROI draws and
    execute the frozen batched UOT solver for the Arm-I null slice.
    """
    arm_cfg = config["arm1"]
    pair_meta = generate_within_compartment_pairs(
        adata,
        n_draws=int(arm_cfg["n_draws"]),
        random_seed=int(arm_cfg["random_seed"]),
    )
    if pair_meta.empty:
        raise ValueError("Arm I produced no eligible within-patient within-compartment ROI pairs")

    k_full = int(config["data"]["k_full"])
    mass_mode = str(config["data"]["mass_mode"])
    A, B, mass_gap = assemble_tensors(adata, pair_meta, k_full=k_full, mass_mode=mass_mode)
    pair_meta = pair_meta.copy()
    pair_meta["mass_gap"] = mass_gap

    # Task-A currently assigns fixed lambda/tau from compartment_a; for constrained
    # rows compartment_a == compartment_b.
    lambda_pl = expand_fixed_values(
        pair_meta["compartment_a"],
        fixed_by_compartment=arm_cfg["fixed_lambda_by_compartment"],
        field_name="fixed_lambda_by_compartment",
    )
    tau_external = expand_fixed_values(
        pair_meta["compartment_a"],
        fixed_by_compartment=arm_cfg["fixed_tau_by_compartment"],
        field_name="fixed_tau_by_compartment",
    )

    pair_meta["arm"] = ARM_NAME
    pair_meta["lambda_mode"] = FIXED_MODE
    pair_meta["tau_mode"] = FIXED_MODE
    pair_meta["mass_mode"] = mass_mode
    pair_meta["group_mode"] = "provided"
    pair_meta["drift_mode"] = "unavailable"

    return run_uot_batch_safe(
        A=A,
        B=B,
        lambda_pl=lambda_pl,
        tau_external=tau_external,
        kernels=kernels,
        uot_cfg=uot_cfg,
        pair_meta=pair_meta,
    )


def generate_within_compartment_pairs(
    adata: AnnData,
    *,
    n_draws: int = 1,
    random_seed: int = 42,
) -> pd.DataFrame:
    slots = generate_anchored_arm1_slots(adata, n_draws=n_draws, random_seed=random_seed)
    records: list[dict[str, Any]] = []
    for slot in slots.to_dict(orient="records"):
        pair_id = (
            f"{ARM_NAME}::{slot['draw_label']}::{slot['patient_id_a']}::{slot['compartment_a']}::"
            f"{slot['roi_a']}::{slot['roi_b_local']}"
        )
        records.append(
            {
                "pair_id": pair_id,
                "patient_group_id": pair_id,
                "draw_number": slot["draw_number"],
                "draw_label": slot["draw_label"],
                "slot_index": slot["slot_index"],
                "patient_id": slot["patient_id_a"],
                "compartment": slot["compartment_a"],
                "patient_id_a": slot["patient_id_a"],
                "patient_id_b": slot["patient_id_a"],
                "compartment_a": slot["compartment_a"],
                "compartment_b": slot["compartment_a"],
                "same_patient": True,
                "same_compartment": True,
                "pair_type": f"{slot['compartment_a']}-{slot['compartment_a']}",
                "roi_a": slot["roi_a"],
                "roi_b": slot["roi_b_local"],
            }
        )

    return pd.DataFrame.from_records(records)


def expand_fixed_values(
    compartments: pd.Series,
    *,
    fixed_by_compartment: Mapping[str, Any],
    field_name: str,
) -> np.ndarray:
    values: list[float] = []
    missing: set[str] = set()
    for compartment in compartments.astype(str):
        if compartment not in fixed_by_compartment:
            missing.add(compartment)
            continue
        values.append(float(fixed_by_compartment[compartment]))

    if missing:
        missing_sorted = sorted(missing)
        raise ValueError(f"{field_name} is missing compartments: {missing_sorted}")

    return np.asarray(values, dtype=float)
