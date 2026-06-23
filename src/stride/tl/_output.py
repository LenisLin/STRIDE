"""Public result containers and assembly helpers for STRIDE `.tl`.

This module owns `FitResult`, `RelationResult`, and compact provenance
assembly. It does not write AnnData, files, R objects, or native export
payloads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from stride.errors import ContractError

from ._optimizer import optimizer_handoff
from ._parameters import NUMERICAL_MIN_MASS, OFFDIAG_INIT_MASS
from ._resolve import RelationInput
from ._train import TrainingResult

RESULT_SCHEMA_VERSION = "stride_tl_result_v1"
PROVENANCE_SCHEMA_VERSION = "stride_fit_provenance.v1"
ROW_SIMPLEX_ATOL = 1e-8
E_BOUNDS = (0.0, 1.0)
POST_RECONSTRUCTION_FORM = "normalize(q_minus @ A + e)"
_OPTIMIZER_HANDOFF = optimizer_handoff()


@dataclass(frozen=True)
class CohortResult:
    """Cohort-supported consensus summary for one fitted relation."""

    relation_id: str
    patient_ids: tuple[str, ...]
    template_A: np.ndarray
    template_d: np.ndarray
    template_e: np.ndarray
    support_n_patients: int
    dispersion: float
    fit_status: str = "ok"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelationResult:
    """Public fitted output for one declared relation.

    relation_id: stable relation result key.
    patient_ids: patient ids aligned to axis `0` of `A`, `d`, and `e`.
    A: stacked patient transition arrays `[P, K, K]`.
    d: stacked patient source-row open channel arrays `[P, K]`.
    e: stacked patient target-side open tendency arrays `[P, K]`.
    support: relation and patient support summary.
    loss: compact final loss ledger or loss summary.
    provenance: compact evidence/operator/runtime provenance.
    warnings: structured relation-level warning records.
    """

    relation_id: str
    patient_ids: tuple[str, ...]
    A: Any
    d: Any
    e: Any
    support: Mapping[str, Any]
    loss: Any | None = None
    cohort: CohortResult | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class FitResult:
    """Public multi-relation `.tl.fit` result.

    relations: relation results keyed by stable `relation_id`.
    relation_ids: declared relation id order for result traversal.
    source: configured source side.
    target: configured target side.
    n_states: shared K-state basis size.
    warnings: structured fit-level warning records.
    provenance: compact package and fitting contract provenance.
    """

    relations: Mapping[str, RelationResult]
    relation_ids: tuple[str, ...]
    source: str
    target: str
    n_states: int
    warnings: tuple[Mapping[str, Any], ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)


def assemble_relation_result(
    relation: RelationInput,
    fit: TrainingResult,
) -> RelationResult:
    """Assemble one public relation result from training handoff.

    Purpose:
        Stack fitted `A/d/e`, attach support metadata, and assemble compact
        relation provenance.

    Key variables:
        patient_ids: public patient axis.
        A: fitted transition array `[P, K, K]`.
        d: fitted source-row open channel `[P, K]`.
        e: fitted target-side open tendency `[P, K]`.
        support: support counts and skipped-patient records.
        provenance: relation evidence and training handoff/runtime facts.
    """
    if fit.parameters is None:
        raise ContractError("fit.parameters is required for relation output assembly")
    if fit.loss_ledger is None:
        raise ContractError("fit.loss_ledger is required for relation output assembly")

    patient_ids = tuple(str(item) for item in fit.parameters.patient_ids)
    relation_patient_ids = tuple(str(item) for item in relation.patient_ids)
    if patient_ids != relation_patient_ids:
        raise ContractError("fit parameter patient_ids must match relation patient_ids")

    # Tensor-to-numpy conversion is the public output boundary.
    A = _detach_public_array(fit.parameters.A, name="A")
    d = _detach_public_array(fit.parameters.d, name="d")
    e = _detach_public_array(fit.parameters.e, name="e")

    support = _assemble_support(relation)
    loss = _assemble_loss_summary(fit.loss_ledger)
    cohort = _assemble_cohort_result(relation, A, d, e, loss)
    provenance = _assemble_provenance(relation, fit)
    warnings = _assemble_relation_warnings(relation, fit)

    result = RelationResult(
        relation_id=str(relation.relation_id),
        patient_ids=patient_ids,
        A=A,
        d=d,
        e=e,
        support=support,
        loss=loss,
        cohort=cohort,
        provenance=provenance,
        warnings=warnings,
    )
    _validate_relation_result(result)
    return result


def assemble_fit_result(
    *,
    relations: Sequence[RelationResult],
    warnings: Sequence[Mapping[str, Any]],
    source: str,
    target: str,
    n_states: int,
) -> FitResult:
    """Assemble the public multi-relation fit result.

    Purpose:
        Preserve realized relation order and provide stable relation-id access.

    Key variables:
        relation_ids: realized relation ids in declared traversal order.
        relation_map: mapping from relation id to public relation result.
        provenance: compact fit-level contract provenance.
    """
    K = _positive_int(n_states, name="n_states")
    for result in relations:
        _validate_relation_result(result)
        result_K = int(_as_public_array(result.A, name="A").shape[1])
        if result_K != K:
            raise ContractError("RelationResult K must match FitResult.n_states")

    relation_ids = tuple(str(result.relation_id) for result in relations)
    if len(set(relation_ids)) != len(relation_ids):
        raise ContractError("duplicate relation_id values are not allowed in FitResult")

    relation_map = dict(zip(relation_ids, relations, strict=True))
    provenance = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "n_relations": len(relation_ids),
        "relation_ids": relation_ids,
        "source": str(source),
        "target": str(target),
        "n_states": K,
    }
    return FitResult(
        relations=relation_map,
        relation_ids=relation_ids,
        source=str(source),
        target=str(target),
        n_states=K,
        warnings=tuple(dict(item) for item in warnings),
        provenance=provenance,
    )


def _assemble_provenance(
    relation: RelationInput,
    fit: TrainingResult,
) -> Mapping[str, Any]:
    """Assemble compact relation-level provenance.

    Purpose:
        Record evidence policy, domain policy, operator version, and training
        handoff/runtime facts without exposing beta object layers.

    Key variables:
        domain_policy: fixed `observation_layer_only` policy.
        block_construction_policy: evidence block construction policy.
        operator_version: canonical observation operator id.
        optimizer_protocol: fixed optimizer protocol id.
    """
    if fit.loss_ledger is None:
        raise ContractError("fit.loss_ledger is required for provenance assembly")
    metadata = fit.loss_ledger.metadata
    run_info = fit.run_info

    # Provenance mapping is frozen compact schema assembly, not audit expansion.
    K = _n_states_from_fit(fit)
    delta_init = _delta_init(K)
    observation_discrepancy = _scalarize_value(metadata.get("observation_discrepancy", {}))
    return {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "objective_contract_version": metadata.get("objective_contract_version"),
        "random_seed": run_info.random_seed,
        "objective_constants": _scalarize_value(metadata.get("objective_constants", {})),
        "objective_scale_initialization": {
            "policy": "identity_plus_small_open",
            "delta_init": delta_init,
            "K": K,
            "dtype": "float64",
        },
        "optimizer_start_initialization": {
            "policy": "offdiag_seeded_identity_plus_small_open",
            "delta_init": delta_init,
            "offdiag_init_mass": OFFDIAG_INIT_MASS,
            "numerical_min_mass": NUMERICAL_MIN_MASS,
            "K": K,
            "dtype": "float64",
        },
        "loss": _assemble_provenance_loss(fit.loss_ledger, relation),
        "e_bounds": list(E_BOUNDS),
        "post_reconstruction_form": POST_RECONSTRUCTION_FORM,
        "observation_comparison_plan": {
            "resolved_by": "task_layer",
            "n_evidence_blocks": len(relation.blocks),
            "domain_policy": "observation_layer_only",
            "block_construction_policy": relation.metadata.get("block_construction_policy"),
            "n_blocks_by_patient": _blocks_by_patient(relation),
        },
        "observation_discrepancy": observation_discrepancy,
        "state_geometry": _scalarize_value(metadata.get("state_geometry", {})),
        "optimizer": _assemble_optimizer_payload(run_info),
        "recurrence": {
            "support_n_patients": len(relation.patient_ids),
            "dispersion": _required_float_component(
                fit.loss_ledger.components,
                "recurrence_raw",
            ),
        },
        "detailed_optimizer_trace": fit.trace is not None,
    }


def _validate_relation_result(result: RelationResult) -> None:
    """Validate public relation result shape and hard parameter constraints.

    Purpose:
        Guard the local output contract for realized relation results.

    Key variables:
        A_shape: expected `[P, K, K]`.
        d_shape: expected `[P, K]`.
        e_shape: expected `[P, K]`.
        row_sums: `sum_j A[p, i, j] + d[p, i]`.
    """
    patient_ids = tuple(str(item) for item in result.patient_ids)
    P = len(patient_ids)
    if P <= 0:
        raise ContractError("RelationResult.patient_ids must be non-empty")
    if len(set(patient_ids)) != P:
        raise ContractError("RelationResult.patient_ids must not contain duplicates")

    A = _as_public_array(result.A, name="A")
    d = _as_public_array(result.d, name="d")
    e = _as_public_array(result.e, name="e")

    if A.ndim != 3:
        raise ContractError("RelationResult.A must be a [P, K, K] array")
    if d.ndim != 2:
        raise ContractError("RelationResult.d must be a [P, K] array")
    if e.ndim != 2:
        raise ContractError("RelationResult.e must be a [P, K] array")
    if A.shape[0] != P:
        raise ContractError("RelationResult.A first axis must align with patient_ids")
    K = int(A.shape[1])
    if K <= 0 or A.shape != (P, K, K):
        raise ContractError("RelationResult.A must have shape [P, K, K]")
    if d.shape != (P, K):
        raise ContractError("RelationResult.d must have shape [P, K]")
    if e.shape != (P, K):
        raise ContractError("RelationResult.e must have shape [P, K]")

    if not np.isfinite(A).all():
        raise ContractError("RelationResult.A must contain only finite values")
    if not np.isfinite(d).all():
        raise ContractError("RelationResult.d must contain only finite values")
    if not np.isfinite(e).all():
        raise ContractError("RelationResult.e must contain only finite values")
    if (A < 0.0).any():
        raise ContractError("RelationResult.A entries must be nonnegative")
    if (d < 0.0).any():
        raise ContractError("RelationResult.d entries must be nonnegative")
    if ((e < 0.0) | (e > 1.0)).any():
        raise ContractError("RelationResult.e entries must satisfy 0 <= e <= 1")

    row_sums = A.sum(axis=2) + d
    if not np.allclose(
        row_sums,
        np.ones((P, K), dtype=np.float64),
        rtol=ROW_SIMPLEX_ATOL,
        atol=ROW_SIMPLEX_ATOL,
    ):
        raise ContractError("RelationResult A/d row simplex constraint failed")

    if result.cohort is not None:
        _validate_cohort_result(result.cohort, relation=result)


def _detach_public_array(value: Any, *, name: str) -> np.ndarray:
    if not torch.is_tensor(value):
        raise ContractError(f"fit.parameters.{name} must be a torch tensor")
    return value.detach().cpu().numpy()


def _as_public_array(value: Any, *, name: str) -> np.ndarray:
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"RelationResult.{name} must be array-like") from exc


def _assemble_support(relation: RelationInput) -> Mapping[str, Any]:
    return {
        "support_counts": _mapping_to_plain_dict(relation.support_counts),
        "skipped_patient_ids": tuple(str(item) for item in relation.skipped_patient_ids),
        "n_evidence_blocks": len(relation.blocks),
        "n_blocks_by_patient": _blocks_by_patient(relation),
        "block_ids": tuple(str(block.block_id) for block in relation.blocks),
        "block_construction_policy": relation.metadata.get("block_construction_policy"),
    }


def _assemble_loss_summary(loss_ledger: Any) -> Mapping[str, Any]:
    return {
        "total": _scalarize_value(loss_ledger.total),
        "fit": _scalarize_value(loss_ledger.fit),
        "prior": _scalarize_value(loss_ledger.prior),
        "cohort": _scalarize_value(loss_ledger.cohort),
        "components": _scalarize_value(loss_ledger.components),
    }


def _assemble_cohort_result(
    relation: RelationInput,
    A: np.ndarray,
    d: np.ndarray,
    e: np.ndarray,
    loss: Mapping[str, Any],
) -> CohortResult:
    # Cohort template is the recurrence objective consensus summary.
    recurrence_raw = None
    components = loss.get("components")
    if isinstance(components, Mapping):
        recurrence_raw = components.get("recurrence_raw")
    if recurrence_raw is None:
        raise ContractError("loss components must include recurrence_raw for cohort output")
    return CohortResult(
        relation_id=str(relation.relation_id),
        patient_ids=tuple(str(item) for item in relation.patient_ids),
        template_A=A.mean(axis=0),
        template_d=d.mean(axis=0),
        template_e=e.mean(axis=0),
        support_n_patients=len(relation.patient_ids),
        dispersion=float(recurrence_raw),
        metadata={
            "summary": "recurrence_regularized_mean_consensus",
            "template_source": "mean_of_fitted_patient_relations",
            "dispersion_source": "loss.components.recurrence_raw",
        },
    )


def _assemble_relation_warnings(
    relation: RelationInput,
    fit: TrainingResult,
) -> tuple[Mapping[str, Any], ...]:
    warnings: list[Mapping[str, Any]] = []
    relation_warning = relation.metadata.get("warning")
    if isinstance(relation_warning, Mapping):
        warnings.append(dict(relation_warning))
    relation_warnings = relation.metadata.get("warnings")
    if isinstance(relation_warnings, Sequence) and not isinstance(
        relation_warnings,
        (str, bytes),
    ):
        warnings.extend(dict(item) for item in relation_warnings if isinstance(item, Mapping))
    if fit.loss_ledger is not None:
        warnings.extend(dict(item) for item in fit.loss_ledger.warnings)
    return tuple(warnings)


def _blocks_by_patient(relation: RelationInput) -> dict[str, int]:
    counts = {str(patient_id): 0 for patient_id in relation.patient_ids}
    for block in relation.blocks:
        patient_id = str(block.patient_id)
        counts[patient_id] = counts.get(patient_id, 0) + 1
    return counts


def _validate_cohort_result(
    cohort: CohortResult,
    *,
    relation: RelationResult,
) -> None:
    if cohort.relation_id != relation.relation_id:
        raise ContractError("CohortResult relation_id must match RelationResult")
    if tuple(cohort.patient_ids) != tuple(relation.patient_ids):
        raise ContractError("CohortResult patient_ids must match RelationResult")
    K = int(np.asarray(relation.d).shape[1])
    if np.asarray(cohort.template_A).shape != (K, K):
        raise ContractError("CohortResult.template_A must have shape [K, K]")
    if np.asarray(cohort.template_d).shape != (K,):
        raise ContractError("CohortResult.template_d must have shape [K]")
    if np.asarray(cohort.template_e).shape != (K,):
        raise ContractError("CohortResult.template_e must have shape [K]")
    if cohort.support_n_patients != len(relation.patient_ids):
        raise ContractError("CohortResult support_n_patients must align with patients")
    if cohort.fit_status == "ok" and cohort.support_n_patients <= 0:
        raise ContractError("ok CohortResult must have positive patient support")

    template_A = _as_cohort_array(cohort.template_A, name="template_A")
    template_d = _as_cohort_array(cohort.template_d, name="template_d")
    template_e = _as_cohort_array(cohort.template_e, name="template_e")
    if not np.isfinite(template_A).all():
        raise ContractError("CohortResult.template_A must contain only finite values")
    if not np.isfinite(template_d).all():
        raise ContractError("CohortResult.template_d must contain only finite values")
    if not np.isfinite(template_e).all():
        raise ContractError("CohortResult.template_e must contain only finite values")
    if not np.isfinite(float(cohort.dispersion)):
        raise ContractError("CohortResult.dispersion must be finite")
    if (template_A < 0.0).any():
        raise ContractError("CohortResult.template_A entries must be nonnegative")
    if (template_d < 0.0).any():
        raise ContractError("CohortResult.template_d entries must be nonnegative")
    if ((template_e < 0.0) | (template_e > 1.0)).any():
        raise ContractError("CohortResult.template_e entries must satisfy 0 <= e <= 1")

    row_sums = template_A.sum(axis=1) + template_d
    if not np.allclose(
        row_sums,
        np.ones((K,), dtype=np.float64),
        rtol=ROW_SIMPLEX_ATOL,
        atol=ROW_SIMPLEX_ATOL,
    ):
        raise ContractError("CohortResult template A/d row simplex constraint failed")


def _as_cohort_array(value: Any, *, name: str) -> np.ndarray:
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"CohortResult.{name} must be array-like") from exc


def _assemble_provenance_loss(loss_ledger: Any, relation: RelationInput) -> Mapping[str, Any]:
    components = loss_ledger.components
    scales = loss_ledger.metadata.get("loss_scales", {})
    obs_raw = _required_float_component(components, "obs_raw")
    obs_normalized = _required_float_component(components, "obs_normalized")
    open_raw = _required_float_component(components, "open_raw")
    geometry_raw = _required_float_component(components, "geometry_raw")
    geometry_normalized = _required_float_component(components, "geometry_normalized")
    geometry_effective = _required_float_component(components, "geometry_effective")
    consistency_raw = _required_float_component(components, "consistency_raw")
    recurrence_raw = _required_float_component(components, "recurrence_raw")
    return {
        "total": _scalarize_value(loss_ledger.total),
        "fit": _scalarize_value(loss_ledger.fit),
        "prior": _scalarize_value(loss_ledger.prior),
        "cohort": _scalarize_value(loss_ledger.cohort),
        "components": {
            "obs": {
                "raw": obs_raw,
                "scale": _required_float_mapping(scales, "obs_scale"),
                "normalized": obs_normalized,
                "floor_used": bool(scales.get("obs_scale_floor_used", False)),
            },
            "open": {
                "raw": open_raw,
                "normalized": open_raw,
            },
            "geometry": {
                "raw": geometry_raw,
                "scale": _required_float_mapping(scales, "geometry_scale"),
                "normalized": geometry_normalized,
                "effective": geometry_effective,
                "floor_used": bool(scales.get("geometry_scale_floor_used", False)),
            },
            "subbag_consistency": {
                "raw": consistency_raw,
                "effective": consistency_raw,
                "status": _subbag_consistency_status(relation),
            },
            "recurrence": {
                "raw": recurrence_raw,
                "cohort_scaled": _scalarize_value(loss_ledger.cohort),
            },
        },
    }


def _assemble_optimizer_payload(run_info: Any) -> Mapping[str, Any]:
    return {
        "framework": "torch",
        "algorithm": "AdamW",
        "weight_decay": _OPTIMIZER_HANDOFF.weight_decay,
        "protocol_name": _OPTIMIZER_HANDOFF.protocol_name,
        "exit_flag": run_info.optimizer_exit_flag,
        "reason": run_info.reason,
        "n_steps": run_info.n_steps,
        "initial_total": run_info.initial_total,
        "final_total": run_info.final_total,
        "absolute_improvement": run_info.absolute_improvement,
        "relative_improvement": run_info.relative_improvement,
        "warmup": {
            "steps": _OPTIMIZER_HANDOFF.warmup_steps,
            "steps_completed": run_info.warmup_steps_completed,
            "lr": _OPTIMIZER_HANDOFF.warmup_lr,
            "scheduler_policy": "none",
            "early_stop": "not_allowed",
        },
        "main": {
            "min_steps": _OPTIMIZER_HANDOFF.main_min_steps,
            "max_steps": _OPTIMIZER_HANDOFF.main_max_steps,
            "steps_completed": run_info.main_steps_completed,
            "lr": _OPTIMIZER_HANDOFF.main_lr,
            "scheduler_policy": _OPTIMIZER_HANDOFF.scheduler_policy,
            "early_stop": "main_after_min_steps",
        },
        "cosine": {
            "T_max": _OPTIMIZER_HANDOFF.cosine_T_max,
            "eta_min": _OPTIMIZER_HANDOFF.cosine_eta_min,
        },
        "early_stop_thresholds": {
            "min_relative_improvement": _OPTIMIZER_HANDOFF.min_relative_improvement,
            "convergence_tol": _OPTIMIZER_HANDOFF.convergence_tol,
            "patience": _OPTIMIZER_HANDOFF.patience,
        },
    }


def _n_states_from_fit(fit: TrainingResult) -> int:
    if fit.parameters is None:
        raise ContractError("fit.parameters is required for provenance assembly")
    return int(fit.parameters.A.shape[1])


def _delta_init(n_states: int) -> float:
    K = _positive_int(n_states, name="n_states")
    return min(0.05, 1.0 / float(K + 1))


def _subbag_consistency_status(relation: RelationInput) -> str:
    return (
        "ok"
        if any(count >= 2 for count in _blocks_by_patient(relation).values())
        else "insufficient_blocks"
    )


def _required_float_component(components: Mapping[str, Any], key: str) -> float:
    if key not in components:
        raise ContractError(f"loss components must include {key}")
    return _finite_float(_scalarize_value(components[key]), name=f"loss.components.{key}")


def _required_float_mapping(mapping: Mapping[str, Any], key: str) -> float:
    if key not in mapping:
        raise ContractError(f"loss metadata loss_scales must include {key}")
    return _finite_float(_scalarize_value(mapping[key]), name=f"loss_scales.{key}")


def _finite_float(value: Any, *, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ContractError(f"{name} must be finite")
    return scalar


def _scalarize_value(value: Any) -> Any:
    if torch.is_tensor(value):
        detached = value.detach().cpu()
        if detached.ndim == 0:
            return float(detached.item())
        return detached.numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return float(value.item())
        return value
    if isinstance(value, Mapping):
        return {str(key): _scalarize_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_scalarize_value(item) for item in value)
    if isinstance(value, list):
        return [_scalarize_value(item) for item in value]
    return value


def _mapping_to_plain_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _scalarize_value(item) for key, item in value.items()}


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return int(value)
