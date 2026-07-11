"""Canonical STRIDE fitting wrappers for Block 1."""
from __future__ import annotations

from anndata import AnnData

from stride.tl import FitResult, fit

from ...config import TaskAOrderedPairFamilySpec
from ...workflows.fit_adapter import (
    fit_task_a_pair,
    require_task_a_fit_support,
    summarize_task_a_fit,
)
from .schemas import RUN_SCOPE_FULL_COHORT


def fit_block1_family(
    adata: AnnData,
    *,
    family_spec: TaskAOrderedPairFamilySpec,
    device: object | None = None,
) -> FitResult:
    """Run canonical STRIDE for one Block 1 family."""
    return fit_task_a_pair(adata, device=device, estimator=fit)


def require_block1_fit_ok(
    fit_result: FitResult,
    *,
    pair_family: str,
    run_scope: str,
) -> None:
    """Fail fast when a formal expected Block 1 fit is not fully ok."""
    if run_scope != RUN_SCOPE_FULL_COHORT:
        return
    require_task_a_fit_support(
        fit_result,
        context=f"Block 1 full-cohort fit for {pair_family!r}",
    )


def summarize_fit_status_for_manifest(
    fit_result: FitResult,
    *,
    pair_family: str,
) -> dict[str, object]:
    """Return thin fit metadata for the execute manifest."""
    return summarize_task_a_fit(fit_result, pair_family=pair_family)


__all__ = [
    "fit_block1_family",
    "require_block1_fit_ok",
    "summarize_fit_status_for_manifest",
]
