"""Task-A workflow entrypoints for the current Task A layout.

Lazy-loaded to avoid pulling heavy dependencies (anndata, scipy, sklearn)
when only a subset of workflows is needed, and to allow individual modules
to be invoked via ``python -m tasks.task_A.workflows.<name>``.

The public surface keeps only the Stage 0 prepare path, stride adapter, and
the descriptive atlas plus block-local workflows visible.
"""
from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "build_task_a_family_observations",
    "build_task_a_real_data_crosswalk",
    "check_task_a_pre_block0_data_suitability",
    "describe_task_a_stage0_stride_mapping",
    "load_task_a_dataset_handle",
    "resolve_canonical_step3_inputs",
    "resolve_proxy_history_inputs",
    "validate_task_a_result_packet",
    "prepare_task_a_stage0_mapping",
    "write_task_a_descriptive_atlas",
    "write_canonical_step3_review",
    "write_task_a_result_packet",
    "run_block0_workflow",
    "run_block1_workflow",
    "run_block2_workflow",
    "run_task_a_family_core_fit_dry_run",
]

_LAZY_IMPORTS: dict[str, str] = {
    "check_task_a_pre_block0_data_suitability": ".check_data_suitability",
    "write_task_a_descriptive_atlas": ".descriptive_atlas",
    "write_task_a_result_packet": ".package_results",
    "validate_task_a_result_packet": ".package_results",
    "resolve_canonical_step3_inputs": ".review_canonical_step3",
    "resolve_proxy_history_inputs": ".review_canonical_step3",
    "write_canonical_step3_review": ".review_canonical_step3",
    "prepare_task_a_stage0_mapping": ".prepare",
    "run_block0_workflow": ".run_block0",
    "run_block1_workflow": ".run_block1",
    "run_block2_workflow": ".run_block2",
    "build_task_a_family_observations": ".stride_adapter",
    "build_task_a_real_data_crosswalk": ".stride_adapter",
    "describe_task_a_stage0_stride_mapping": ".stride_adapter",
    "load_task_a_dataset_handle": ".stride_adapter",
    "run_task_a_family_core_fit_dry_run": ".stride_adapter",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name], package=__name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
