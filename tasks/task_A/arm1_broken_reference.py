"""
Module: tasks.task_A.arm1_broken_reference
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData

from slotar.uot import UOTSolveConfig

from .arm1_noise_baseline import (
    FIXED_MODE,
    build_arm1_roi_table,
    expand_fixed_values,
    generate_anchored_arm1_slots,
)
from .common import (
    TaskARoiReferenceBundle,
    assemble_tensors,
    resolve_task_a_mass_mode,
    run_uot_batch_safe,
)

ARM_NAME = "A1_broken_reference"


def generate_broken_reference_pairs(
    adata: AnnData,
    *,
    n_draws: int = 1,
    random_seed: int = 42,
) -> pd.DataFrame:
    slots = generate_anchored_arm1_slots(adata, n_draws=n_draws, random_seed=random_seed)
    roi_pool = build_arm1_roi_table(adata)
    rng = np.random.default_rng(random_seed + 1)

    records: list[dict[str, Any]] = []
    for slot in slots.to_dict(orient="records"):
        candidates = roi_pool.loc[
            (roi_pool["roi_id"] != slot["roi_a"])
            & (roi_pool["patient_id"] != slot["patient_id_a"])
            & (roi_pool["compartment"] != slot["compartment_a"])
        ].reset_index(drop=True)
        if candidates.empty:
            raise ValueError(
                "Broken reference produced no valid B candidates for anchored slot "
                f"({slot['patient_id_a']}, {slot['compartment_a']}, {slot['roi_a']})"
            )

        chosen = candidates.iloc[int(rng.integers(0, candidates.shape[0]))]
        patient_id_b = str(chosen["patient_id"])
        compartment_b = str(chosen["compartment"])
        roi_b = str(chosen["roi_id"])
        pair_id = (
            f"{ARM_NAME}::{slot['draw_label']}::{slot['patient_id_a']}::{slot['compartment_a']}::"
            f"{slot['roi_a']}::{roi_b}"
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
                "patient_id_b": patient_id_b,
                "compartment_a": slot["compartment_a"],
                "compartment_b": compartment_b,
                "same_patient": False,
                "same_compartment": False,
                "pair_type": f"{slot['compartment_a']}-{compartment_b}",
                "roi_a": slot["roi_a"],
                "roi_b": roi_b,
            }
        )

    return pd.DataFrame.from_records(records)


def run_arm1(
    adata: AnnData,
    config: Mapping[str, Any],
    uot_cfg: UOTSolveConfig,
    kernels: Sequence[np.ndarray],
    roi_references: TaskARoiReferenceBundle | None = None,
) -> pd.DataFrame:
    """
    Generate anchored broken-locality ROI draws and execute the frozen batched UOT
    solver for the Arm-I negative-control slice.
    """
    arm_cfg = config["arm1"]
    pair_meta = generate_broken_reference_pairs(
        adata,
        n_draws=int(arm_cfg["n_draws"]),
        random_seed=int(arm_cfg["random_seed"]),
    )
    if pair_meta.empty:
        raise ValueError("Arm I broken reference produced no eligible ROI pairs")

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
