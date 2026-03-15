"""
Module: tasks.task_A.pipeline
"""
from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml

from slotar.contracts import COST_SCALE_ALIASES, validate_adata_inputs
from slotar.uot import UOTSolveConfig, precompute_logKernels

from .evaluator import evaluate_task_a

TEMPORARY_METRICS_FILENAME = "task_A_metrics.parquet"
SUPPORTED_ARM_MODULES = {
    "A1_baseline": ("tasks.task_A.arm1_noise_baseline", "run_arm1"),
    "A1_broken_reference": ("tasks.task_A.arm1_broken_reference", "run_arm1"),
    "A2_cross_compartment": ("tasks.task_A.arm2_spatial_gradient", "run_arm2"),
}


def _load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Task-A config must deserialize to a mapping")
    return loaded


def _resolve_cost_scale(adata: ad.AnnData) -> float:
    if "s_C" in adata.uns:
        return float(adata.uns["s_C"])
    for alias in COST_SCALE_ALIASES:
        if alias in adata.uns:
            return float(adata.uns[alias])
    raise ValueError("Validated AnnData is missing a usable SLOTAR cost scale")


def _load_arm_runner(arm_name: str):
    if arm_name not in SUPPORTED_ARM_MODULES:
        raise NotImplementedError(
            f"Patch-2 only supports enabled_arms drawn from {sorted(SUPPORTED_ARM_MODULES)}, got {arm_name!r}"
        )

    module_path, callable_name = SUPPORTED_ARM_MODULES[arm_name]
    module = import_module(module_path)
    return getattr(module, callable_name)


def _build_uot_config(config: dict[str, Any]) -> UOTSolveConfig:
    uot_params = config["uot_params"]
    return UOTSolveConfig(
        eps_schedule=tuple(float(eps) for eps in uot_params["eps_schedule"]),
        max_iter=int(uot_params["max_iter"]),
        tol=float(uot_params["tol"]),
        eta_floor=float(uot_params["eta_floor"]),
        n_min_proto=float(uot_params["n_min_proto"]),
        tau_q=float(uot_params.get("tau_q", 0.25)),
        tau_mode=str(uot_params.get("tau_mode", "external_fixed_by_task")),
    )


def _load_enabled_arms(config: dict[str, Any]) -> list[str]:
    enabled_arms = config.get("enabled_arms", [])
    if isinstance(enabled_arms, (str, bytes)) or not isinstance(enabled_arms, Sequence):
        raise ValueError("enabled_arms must be a list-like of strings")

    enabled_arms_list = list(enabled_arms)
    if any(not isinstance(arm_name, str) for arm_name in enabled_arms_list):
        raise ValueError("enabled_arms must be a list-like of strings")
    if not enabled_arms_list:
        raise ValueError("enabled_arms must contain at least one supported Task-A arm")
    if len(set(enabled_arms_list)) != len(enabled_arms_list):
        raise ValueError("enabled_arms must not contain duplicates")
    return enabled_arms_list


def main(config_path: str, data_path: str, output_dir: str) -> pd.DataFrame:
    config = _load_config(config_path)
    enabled_arms = _load_enabled_arms(config)
    unsupported = [arm_name for arm_name in enabled_arms if arm_name not in SUPPORTED_ARM_MODULES]
    if unsupported:
        raise NotImplementedError(
            "Patch-2 only supports enabled_arms drawn from "
            f"{sorted(SUPPORTED_ARM_MODULES)}, got {enabled_arms!r}"
        )

    adata = ad.read_h5ad(data_path)
    validate_adata_inputs(
        adata,
        require_prototypes=True,
        require_cost_scale=True,
        require_cost_matrix=True,
    )

    k_full = int(config["data"]["k_full"])
    cost_matrix = np.asarray(adata.uns["cost_matrix"], dtype=float)
    if cost_matrix.shape != (k_full, k_full):
        raise ValueError(
            "Task-A cost_matrix shape must match the declared shared prototype axis: "
            f"expected {(k_full, k_full)}, got {cost_matrix.shape}"
        )

    uot_cfg = _build_uot_config(config)
    kernels = precompute_logKernels(
        cost_matrix,
        uot_cfg.eps_schedule,
        s_C=_resolve_cost_scale(adata),
    )

    metrics_parts: list[pd.DataFrame] = []
    for arm_name in enabled_arms:
        run_arm = _load_arm_runner(arm_name)
        metrics_parts.append(run_arm(adata, config, uot_cfg, kernels))
    df_metrics = pd.concat(metrics_parts, ignore_index=True)
    evaluate_task_a(df_metrics, config)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Temporary direct write for Patch-2 smoke validation only. This is not the AVCP bridge path.
    df_metrics.to_parquet(out_dir / TEMPORARY_METRICS_FILENAME, index=False)
    return df_metrics


if __name__ == "__main__":  # pragma: no cover
    main("config.yaml", "data/cohort.h5ad", "outputs/")
