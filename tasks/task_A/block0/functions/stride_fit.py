"""STRIDE fit orchestration skeleton for Task A Block 0 calibration.

Block 0 asks whether real `TC-IM` STRIDE relation structure departs from a
within-patient count-preserving FOV domain-label permutation null. Fit
orchestration consumes Block 0 observation bundles derived from Task A config,
Stage 0 h5ad, run controls, and optional selectors. Extracted fit records are
the execution-cache payload consumed by later analysis, not biological
interpretation or downstream execution decisions. See `tasks/task_A/README.md`
and `tasks/task_A/contracts/artifact_contracts.md`.
"""
from __future__ import annotations

import numpy as np
from anndata import AnnData

from stride.errors import ContractError
from stride.tl import FitResult, fit

from ...config import TaskAConfigBundle
from ...workflows.fit_adapter import (
    extract_task_a_relations,
    fit_task_a_pair,
    require_task_a_fit_support,
)
from .ann_data import derive_patient_side_burdens
from .observations import Block0ObservationBundle
from .schemas import (
    FIT_LABEL_NULL,
    FIT_LABEL_REAL,
    Block0FitRecord,
)


def fit_block0_family(
    observation_bundle_or_adata: Block0ObservationBundle | AnnData,
    *,
    config_bundle: TaskAConfigBundle,
    state_basis: object | None = None,
    fit_label: str,
    permutation_index: int | None = None,
    device: object | None = None,
) -> FitResult:
    """Run the canonical STRIDE fit surface for one real or null Block 0 bundle."""
    if fit_label not in {FIT_LABEL_REAL, FIT_LABEL_NULL}:
        raise ContractError(f"Unsupported Block 0 fit_label: {fit_label!r}")
    if fit_label == FIT_LABEL_REAL and permutation_index is not None:
        raise ContractError("Real Block 0 fit must not carry permutation_index")
    if fit_label == FIT_LABEL_NULL and permutation_index is None:
        raise ContractError("Null Block 0 fit requires permutation_index")
    if isinstance(observation_bundle_or_adata, AnnData):
        return fit_task_a_pair(
            observation_bundle_or_adata,
            device=device,
            estimator=fit,
        )

    raise ContractError("legacy Block0ObservationBundle fitting is retired in new adapter path")


def require_all_patient_results_ok(result: FitResult, *, family_label: str) -> FitResult:
    """Validate that a full-calibration fit is interpretable for every patient."""
    return require_task_a_fit_support(result, context=f"Block 0 {family_label}")


def extract_block0_fit_records(
    result: FitResult,
    *,
    source_adata: AnnData,
    fit_label: str,
    permutation_index: int | None = None,
) -> tuple[Block0FitRecord, ...]:
    """Extract metric-ready Block 0 fit records from a canonical STRIDE fit result."""
    require_all_patient_results_ok(result, family_label=fit_label)
    if fit_label == FIT_LABEL_REAL and permutation_index is not None:
        raise ContractError("Real Block 0 fit records must not carry permutation_index")
    if fit_label == FIT_LABEL_NULL and permutation_index is None:
        raise ContractError("Null Block 0 fit records require permutation_index")
    if fit_label not in {FIT_LABEL_REAL, FIT_LABEL_NULL}:
        raise ContractError(f"Unsupported Block 0 fit label: {fit_label!r}")

    relations = extract_task_a_relations(result)
    if len(relations) != 1:
        raise ContractError(
            "Block 0 fit extraction requires exactly one realized relation; "
            f"observed={tuple(relation.relation_id for relation in relations)}"
        )
    relation = relations[0]
    patient_ids = relation.patient_ids
    A, d, e = relation.A, relation.d, relation.e

    burdens_by_patient = derive_patient_side_burdens(source_adata)
    missing_burdens = tuple(patient_id for patient_id in patient_ids if patient_id not in burdens_by_patient)
    if missing_burdens:
        raise ContractError(
            "Block 0 source AnnData lacks burden vectors for fitted patients: "
            f"{missing_burdens}"
        )

    records: list[Block0FitRecord] = []
    for patient_index, patient_id in enumerate(patient_ids):
        source_burden, target_burden = burdens_by_patient[patient_id]
        source_burden = np.asarray(source_burden, dtype=float)
        target_burden = np.asarray(target_burden, dtype=float)
        if source_burden.shape != (A.shape[1],) or target_burden.shape != (A.shape[1],):
            raise ContractError(
                f"Block 0 patient {patient_id!r} burden vectors must match the fitted K-state axis"
            )
        records.append(
            Block0FitRecord(
                patient_id=patient_id,
                fit_label=fit_label,
                permutation_index=permutation_index,
                A=A[patient_index].copy(),
                d=d[patient_index].copy(),
                e=e[patient_index].copy(),
                source_burden=source_burden.copy(),
                d_weights=source_burden.copy(),
                e_weights=target_burden.copy(),
            )
        )
    return tuple(records)


__all__ = [
    "extract_block0_fit_records",
    "fit_block0_family",
    "require_all_patient_results_ok",
]
