"""
Module: tasks.task_A.pipeline
"""
from __future__ import annotations

from importlib import import_module

import pandas as pd

from .evaluator import evaluate_task_a
from .runtime_contract import (
    TASK_A_METRICS_FILENAME,
    TASK_A_RUN_MANIFEST_FILENAME,
    TaskARuntimeBundle,
    prepare_task_a_runtime,
    resolve_task_a_arm_artifact_root,
    write_task_a_run_manifest,
)

TEMPORARY_METRICS_FILENAME = TASK_A_METRICS_FILENAME
TASK_A_MANIFEST_FILENAME = TASK_A_RUN_MANIFEST_FILENAME
ARM3_OPERATOR_NAME = "A3_uq_stress"
SUPPORTED_ARM_MODULES = {
    "A1_baseline": ("tasks.task_A.arm1_noise_baseline", "run_arm1"),
    "A1_broken_reference": ("tasks.task_A.arm1_broken_reference", "run_arm1"),
    "A2_cross_compartment": ("tasks.task_A.arm2_spatial_gradient", "run_arm2"),
    ARM3_OPERATOR_NAME: ("tasks.task_A.arm3_uq_stress", "run_arm3"),
}


def _load_arm_runner(arm_name: str):
    if arm_name not in SUPPORTED_ARM_MODULES:
        raise NotImplementedError(
            f"Patch-2 only supports enabled_arms drawn from {sorted(SUPPORTED_ARM_MODULES)}, got {arm_name!r}"
        )

    module_path, callable_name = SUPPORTED_ARM_MODULES[arm_name]
    module = import_module(module_path)
    return getattr(module, callable_name)


def _run_enabled_arm(
    arm_name: str,
    *,
    runtime: TaskARuntimeBundle,
) -> pd.DataFrame:
    run_arm = _load_arm_runner(arm_name)
    if arm_name == ARM3_OPERATOR_NAME:
        return run_arm(
            stage0_path=str(runtime.data_path),
            config=runtime.config,
            result_root=str(resolve_task_a_arm_artifact_root(runtime.artifact_layout, arm_name)),
        )

    if runtime.adata is None or runtime.uot_cfg is None or runtime.kernels is None:
        raise RuntimeError(f"Task-A arm {arm_name!r} requires an AnnData-backed runtime context")
    return run_arm(
        runtime.adata,
        runtime.config,
        runtime.uot_cfg,
        runtime.kernels,
        roi_references=runtime.roi_references,
    )


def main(config_path: str, data_path: str, output_dir: str) -> pd.DataFrame:
    runtime = prepare_task_a_runtime(
        config_path=config_path,
        data_path=data_path,
        output_dir=output_dir,
        supported_arms=SUPPORTED_ARM_MODULES,
        arm3_operator_name=ARM3_OPERATOR_NAME,
    )

    metrics_parts: list[pd.DataFrame] = []
    for arm_name in runtime.enabled_arms:
        metrics_parts.append(
            _run_enabled_arm(
                arm_name,
                runtime=runtime,
            )
        )
    df_metrics = pd.concat(metrics_parts, ignore_index=True)
    evaluate_task_a(df_metrics, runtime.config)

    df_metrics.to_parquet(runtime.artifact_layout.metrics_parquet, index=False)
    write_task_a_run_manifest(runtime)
    return df_metrics


if __name__ == "__main__":  # pragma: no cover
    main("config.yaml", "data/cohort.h5ad", "outputs/")
