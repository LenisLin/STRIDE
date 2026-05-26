"""Task-A workflow entrypoints for the current Task A layout.

Lazy-loaded to avoid pulling heavy dependencies (anndata, scipy, sklearn)
when only a subset of workflows is needed, and to allow individual modules
to be invoked via ``python -m tasks.task_A.workflows.<name>``.

The public surface keeps only the Stage 0 prepare path, stride adapter, and
block-local workflows visible. The descriptive atlas lives under
``tasks.task_A.descriptive``.
"""
from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "build_task_a_family_observations",
    "build_task_a_real_data_crosswalk",
    "describe_task_a_stage0_stride_mapping",
    "load_task_a_dataset_handle",
    "validate_task_a_result_packet",
    "prepare_task_a_stage0_mapping",
    "write_task_a_result_packet",
    "run_task_a_family_core_fit_dry_run",
]

_LAZY_IMPORTS: dict[str, str] = {
    "write_task_a_result_packet": ".package_results",
    "validate_task_a_result_packet": ".package_results",
    "prepare_task_a_stage0_mapping": ".prepare",
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
