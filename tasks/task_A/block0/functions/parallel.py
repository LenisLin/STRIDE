"""Permutation-level worker helpers for Task A Block 0 execution caches."""
from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass

from anndata import AnnData

from stride._schema import STRIDE_FOV_OBSERVATIONS_KEY, STRIDE_UNS_KEY
from stride.errors import ContractError

from .ann_data import build_null_tc_im_adata
from .permutation import (
    build_domain_label_permutation_assignments_from_fov_metadata,
)
from .progress import fit_runtime_summary, fit_warning_summary
from .schemas import FIT_LABEL_NULL, Block0FitRecord
from .stride_fit import extract_block0_fit_records, fit_block0_family


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
    runtime_summary: Mapping[str, object]
    null_adata_build_seconds: float
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
    real_adata: AnnData,
    run_config: object,
    config_bundle: object,
    device: object | None = None,
) -> Block0NullFitResult:
    """Build one null AnnData, run full STRIDE, and return extracted records."""
    permutation_index = int(job.permutation_index)
    started = time.monotonic()
    assignments = build_domain_label_permutation_assignments_from_fov_metadata(
        _fov_metadata(real_adata),
        permutation_index=permutation_index,
        master_seed=run_config.master_seed,
    )
    null_adata = build_null_tc_im_adata(
        real_adata,
        assignments,
        permutation_index=permutation_index,
    )
    null_adata_build_seconds = max(0.0, time.monotonic() - started)

    started = time.monotonic()
    null_result = fit_block0_family(
        null_adata,
        config_bundle=config_bundle,
        state_basis=None,
        fit_label=FIT_LABEL_NULL,
        permutation_index=permutation_index,
        device=device,
    )
    records = extract_block0_fit_records(
        null_result,
        source_adata=null_adata,
        fit_label=FIT_LABEL_NULL,
        permutation_index=permutation_index,
    )
    null_fit_extract_seconds = max(0.0, time.monotonic() - started)
    return Block0NullFitResult(
        permutation_index=permutation_index,
        records=tuple(records),
        warning_summary=fit_warning_summary(null_result),
        runtime_summary=fit_runtime_summary(null_result),
        null_adata_build_seconds=null_adata_build_seconds,
        null_fit_extract_seconds=null_fit_extract_seconds,
    )


def _fov_metadata(adata: AnnData) -> object:
    stride_uns = adata.uns.get(STRIDE_UNS_KEY)
    if not isinstance(stride_uns, Mapping):
        raise ContractError("Block 0 worker real AnnData is missing adata.uns['stride']")
    fov_observations = stride_uns.get(STRIDE_FOV_OBSERVATIONS_KEY)
    if not isinstance(fov_observations, Mapping):
        raise ContractError("Block 0 worker real AnnData is missing FOV observations")
    return fov_observations.get("metadata")


__all__ = [
    "Block0NullFitJob",
    "Block0NullFitResult",
    "configure_worker_thread_environment",
    "configure_worker_threads",
    "fit_block0_null_permutation",
]
