"""Permutation-level worker helpers for Task A Block 0 execution caches."""
from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass

from .fit import extract_block0_fit_records, fit_block0_family
from .observations import (
    Block0ObservationBundle,
    build_null_tc_im_observations,
)
from .permutation import build_domain_label_permutation_assignments
from .schemas import FIT_LABEL_NULL, Block0FitRecord


@dataclass(frozen=True)
class Block0NullFitJob:
    """One independent null permutation fit request."""

    permutation_index: int


@dataclass(frozen=True)
class Block0NullFitResult:
    """Serializable result payload returned by a null permutation worker."""

    permutation_index: int
    records: tuple[Block0FitRecord, ...]
    warning_summary: Mapping[str, object]
    null_bundle_build_seconds: float
    null_fit_extract_seconds: float


def configure_worker_thread_environment(worker_cpu_threads: int) -> None:
    """Set thread-count environment variables before worker imports compute libraries."""
    threads = int(worker_cpu_threads)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        os.environ[name] = str(threads)


def configure_worker_threads(worker_cpu_threads: int) -> None:
    """Limit torch/BLAS thread pools inside a permutation worker process."""
    threads = int(worker_cpu_threads)
    configure_worker_thread_environment(threads)
    try:
        import torch

        torch.set_num_threads(threads)
    except Exception:
        pass


def fit_block0_null_permutation(
    job: Block0NullFitJob,
    *,
    real_bundle: Block0ObservationBundle,
    run_config: object,
    config_bundle: object,
    state_basis: object,
) -> Block0NullFitResult:
    """Build one null bundle, run full STRIDE, and return extracted records."""
    permutation_index = int(job.permutation_index)
    started = time.monotonic()
    assignments = build_domain_label_permutation_assignments(
        real_bundle.observations,
        permutation_index=permutation_index,
        master_seed=run_config.master_seed,
    )
    null_bundle = build_null_tc_im_observations(
        real_bundle,
        assignments,
        permutation_index=permutation_index,
    )
    null_bundle_build_seconds = max(0.0, time.monotonic() - started)

    started = time.monotonic()
    null_result = fit_block0_family(
        null_bundle,
        config_bundle=config_bundle,
        state_basis=state_basis,
        fit_label=FIT_LABEL_NULL,
        permutation_index=permutation_index,
    )
    records = extract_block0_fit_records(
        null_result,
        fit_label=FIT_LABEL_NULL,
        permutation_index=permutation_index,
    )
    null_fit_extract_seconds = max(0.0, time.monotonic() - started)
    return Block0NullFitResult(
        permutation_index=permutation_index,
        records=tuple(records),
        warning_summary=_fit_warning_summary(null_result),
        null_bundle_build_seconds=null_bundle_build_seconds,
        null_fit_extract_seconds=null_fit_extract_seconds,
    )


def _fit_warning_summary(result: object) -> dict[str, object]:
    warnings = _collect_fit_warnings(result)
    return {
        "warning_count": len(warnings),
        "warnings": list(warnings),
    }


def _collect_fit_warnings(result: object) -> tuple[str, ...]:
    collected: list[str] = []
    _extend_warning_payload(collected, getattr(result, "warnings", None))
    for mapping_name in ("diagnostics", "summaries", "metadata"):
        _extend_warning_mapping(collected, getattr(result, mapping_name, None))
    _extend_warning_object(collected, getattr(result, "objective", None))
    for ledger_name in ("final_ledger", "objective_ledger"):
        _extend_warning_object(collected, getattr(result, ledger_name, None))
    for patient_result in tuple(getattr(result, "patient_results", ()) or ()):
        _extend_warning_payload(collected, getattr(patient_result, "warnings", None))
        for mapping_name in ("diagnostics", "auxiliary", "metadata"):
            _extend_warning_mapping(collected, getattr(patient_result, mapping_name, None))
        _extend_warning_object(collected, getattr(patient_result, "objective", None))

    unique_warnings: list[str] = []
    seen: set[str] = set()
    for warning in collected:
        if warning in seen:
            continue
        seen.add(warning)
        unique_warnings.append(warning)
    return tuple(unique_warnings)


def _extend_warning_object(collected: list[str], value: object) -> None:
    if value is None:
        return
    _extend_warning_payload(collected, getattr(value, "warnings", None))
    for block_name in ("observation_blocks", "block_records"):
        for record in tuple(getattr(value, block_name, ()) or ()):
            _extend_warning_payload(collected, getattr(record, "warnings", None))
            _extend_warning_mapping(collected, getattr(record, "metadata", None))


def _extend_warning_mapping(collected: list[str], mapping: object) -> None:
    if not isinstance(mapping, Mapping):
        return
    for key, value in mapping.items():
        normalized_key = str(key).lower()
        if normalized_key in {"warning", "warnings", "warning_message", "warning_messages"}:
            _extend_warning_payload(collected, value)
        elif normalized_key == "warning_flags" and isinstance(value, Mapping):
            active_flags = tuple(
                str(flag_name)
                for flag_name, flag_value in value.items()
                if bool(flag_value) and str(flag_name) != "has_warnings"
            )
            if active_flags:
                collected.append(f"warning_flags={active_flags}")


def _extend_warning_payload(collected: list[str], value: object) -> None:
    if value is None:
        return
    if isinstance(value, str):
        warning = value.strip()
        if warning:
            collected.append(warning)
        return
    if isinstance(value, Mapping):
        for nested_value in value.values():
            _extend_warning_payload(collected, nested_value)
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            _extend_warning_payload(collected, item)
        return
    warning = str(value).strip()
    if warning:
        collected.append(warning)


__all__ = [
    "Block0NullFitJob",
    "Block0NullFitResult",
    "configure_worker_thread_environment",
    "configure_worker_threads",
    "fit_block0_null_permutation",
]
