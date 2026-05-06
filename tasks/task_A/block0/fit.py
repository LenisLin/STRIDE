"""STRIDE fit orchestration skeleton for Task A Block 0 calibration.

Block 0 asks whether real `TC-IM` STRIDE relation structure departs from a
within-patient count-preserving FOV domain-label permutation null. Fit orchestration consumes only
Block 0 observation bundles derived from Task A config and Stage 0 h5ad plus
run controls; prepare outputs, descriptive-atlas artifacts, old suitability
outputs, and passed bundles are not evidence inputs. Extracted fit records are
the execution-cache payload consumed by later analysis, not biological
interpretation or downstream execution decisions. See `tasks/task_A/README.md`,
`tasks/task_A/contracts/artifact_contracts.md`, and
`tasks/task_A/contracts/design_freeze.py`.
"""
from __future__ import annotations

from stride.api.fit import STRIDEFitConfig, fit_stride
from stride.errors import ContractError
from stride.outputs.fit_result import STRIDEFitResult

from ..config import TaskAConfigBundle
from .observations import Block0ObservationBundle
from .schemas import (
    BLOCK_NAME,
    FIT_LABEL_NULL,
    FIT_LABEL_REAL,
    REAL_FAMILY,
    SOURCE_DOMAIN,
    TARGET_DOMAIN,
    Block0FitRecord,
)


def fit_block0_family(
    observation_bundle: Block0ObservationBundle,
    *,
    config_bundle: TaskAConfigBundle,
    state_basis: object,
    fit_label: str,
    permutation_index: int | None = None,
) -> STRIDEFitResult:
    """Run the canonical STRIDE fit surface for one real or null Block 0 bundle."""
    if fit_label != observation_bundle.label:
        raise ContractError("Block 0 fit_label must match the observation bundle label")
    if fit_label == FIT_LABEL_REAL and permutation_index is not None:
        raise ContractError("Real Block 0 fit must not carry permutation_index")
    if fit_label == FIT_LABEL_NULL and permutation_index != observation_bundle.permutation_index:
        raise ContractError("Null Block 0 fit permutation_index must match the observation bundle")
    if fit_label not in {FIT_LABEL_REAL, FIT_LABEL_NULL}:
        raise ContractError(f"Unsupported Block 0 fit_label: {fit_label!r}")
    metadata = {
        "task_block": BLOCK_NAME,
        "task_pair_family": REAL_FAMILY,
        "task_source_domain": SOURCE_DOMAIN,
        "task_target_domain": TARGET_DOMAIN,
        "task_fit_label": fit_label,
        "task_config_fingerprint": str(config_bundle.config_fingerprint),
    }
    if permutation_index is not None:
        metadata["block0_permutation_index"] = int(permutation_index)
    return fit_stride(
        observation_bundle.observations,
        state_basis=state_basis,
        config=STRIDEFitConfig(
            timepoint_order=(SOURCE_DOMAIN, TARGET_DOMAIN),
            metadata=metadata,
        ),
    )


def require_all_patient_results_ok(result: STRIDEFitResult, *, family_label: str) -> STRIDEFitResult:
    """Validate that a full-calibration fit is interpretable for every patient."""
    if result.fit_status != "ok":
        raise ContractError(f"Block 0 {family_label} fit_status must be 'ok'")
    if result.implementation_tier != "canonical_full":
        raise ContractError(f"Block 0 {family_label} fit must use canonical_full implementation")
    non_ok = [patient.patient_id for patient in result.patient_results if patient.fit_status != "ok"]
    if non_ok:
        raise ContractError(f"Block 0 {family_label} has non-ok patient fits: {non_ok}")
    return result


def extract_block0_fit_records(
    result: STRIDEFitResult,
    *,
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

    records: list[Block0FitRecord] = []
    for patient_result in result.patient_results:
        missing = tuple(
            field_name
            for field_name in ("A", "d", "e", "mu_minus", "mu_plus")
            if getattr(patient_result, field_name) is None
        )
        if missing:
            raise ContractError(
                f"Block 0 patient {patient_result.patient_id!r} fit lacks required fields: {missing}"
            )
        records.append(
            Block0FitRecord(
                patient_id=str(patient_result.patient_id),
                fit_label=fit_label,
                permutation_index=permutation_index,
                A=patient_result.A,
                d=patient_result.d,
                e=patient_result.e,
                source_burden=patient_result.mu_minus,
                d_weights=patient_result.mu_minus,
                e_weights=patient_result.mu_plus,
            )
        )
    return tuple(records)


__all__ = [
    "extract_block0_fit_records",
    "fit_block0_family",
    "require_all_patient_results_ok",
]
