"""Canonical STRIDE fitting wrappers for Block 1."""
from __future__ import annotations

from collections.abc import Sequence

from stride.api.fit import fit_stride
from stride.basis.contracts import StateBasis
from stride.errors import ContractError
from stride.geometry.state_geometry import StateGeometry
from stride.observation import FovObservation
from stride.outputs.fit_result import STRIDEFitResult

from .schemas import RUN_SCOPE_FULL_COHORT
from ...config import TaskAOrderedPairFamilySpec


def fit_block1_family(
    observations: Sequence[FovObservation],
    *,
    family_spec: TaskAOrderedPairFamilySpec,
    state_basis: StateBasis,
    geometry: StateGeometry | None = None,
    device: object | None = None,
) -> STRIDEFitResult:
    """Run canonical STRIDE for one Block 1 family."""
    return fit_stride(
        tuple(observations),
        source=family_spec.source_domain,
        target=family_spec.target_domain,
        K=int(state_basis.n_states),
        timepoint_order=family_spec.ordered_group_labels,
        state_basis=state_basis,
        geometry=geometry,
        device=device,
    )


def require_block1_fit_ok(
    fit_result: STRIDEFitResult,
    *,
    pair_family: str,
    run_scope: str,
) -> None:
    """Fail fast when a formal expected Block 1 fit is not fully ok."""
    if run_scope != RUN_SCOPE_FULL_COHORT:
        return
    if fit_result.fit_status != "ok":
        raise ContractError(f"Block 1 full-cohort fit for {pair_family!r} returned fit_status={fit_result.fit_status!r}")
    if any(patient_result.fit_status != "ok" for patient_result in fit_result.patient_results):
        raise ContractError(f"Block 1 full-cohort fit for {pair_family!r} contains non-ok patient results")
    if fit_result.cohort_relation.fit_status != "ok":
        raise ContractError(f"Block 1 full-cohort fit for {pair_family!r} has non-ok cohort recurrence")


def summarize_fit_status_for_manifest(
    fit_result: STRIDEFitResult,
    *,
    pair_family: str,
) -> dict[str, object]:
    """Return thin fit metadata for the execute manifest."""
    k_states = 0
    for patient_result in fit_result.patient_results:
        if patient_result.fit_status == "ok" and patient_result.state_ids is not None:
            k_states = len(patient_result.state_ids)
            break
    return {
        "pair_family": pair_family,
        "fit_status": str(fit_result.fit_status),
        "patient_count": int(len(fit_result.patient_results)),
        "k_states": int(k_states),
    }


__all__ = [
    "fit_block1_family",
    "require_block1_fit_ok",
    "summarize_fit_status_for_manifest",
]
