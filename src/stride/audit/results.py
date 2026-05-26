"""Post-fit STRIDE result assembly."""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..errors import ContractError
from ..latent.operators import CohortRelation, PatientRelationAudit
from ..losses import EvidenceBlock
from ..optimize import TrainResult
from ..outputs.fit_result import PatientRelationResult, STRIDEFitResult
from ..workflows._fit_inputs import _FitObservationGroup, _PatientFitInput
from ..workflows.config import TaskConfig
from .provenance import build_successful_provenance
from .status import FitRunStatus


FIT_RUNTIME_MODE = "fit_stride"


def _count_patient_statuses(results: Sequence[PatientRelationResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.fit_status] = counts.get(result.fit_status, 0) + 1
    return counts


def _group_state_matrix(observations: Sequence[object]) -> np.ndarray:
    return np.vstack(
        [np.asarray(observation.community_composition, dtype=float) for observation in observations]
    ).astype(float, copy=False)


def _mean_group_composition(group: _FitObservationGroup) -> np.ndarray:
    return np.mean(_group_state_matrix(group.observations), axis=0, dtype=float)


def _reconstruct_post_burden(A: np.ndarray, e: np.ndarray, mu_minus: np.ndarray) -> np.ndarray:
    return np.asarray(mu_minus, dtype=float) @ np.asarray(A, dtype=float) + np.asarray(e, dtype=float)


def _resolve_patient_state_ids(patient_input: _PatientFitInput) -> tuple[int, ...] | None:
    if patient_input.state_basis is not None:
        return tuple(patient_input.state_basis.resolved_state_ids)
    if patient_input.geometry is not None:
        return tuple(patient_input.geometry.state_ids)
    return None


def _shared_bridge_count_diagnostics(patient_input: _PatientFitInput) -> dict[str, Any]:
    return {
        "n_observations_by_group": dict(patient_input.n_observations_by_group),
        "n_observations_by_domain": dict(patient_input.n_observations_by_domain),
        "n_observations_by_group_and_domain": {
            group: dict(domain_counts)
            for group, domain_counts in patient_input.n_observations_by_group_and_domain.items()
        },
    }


def _shared_bridge_audit_metadata(patient_input: _PatientFitInput) -> dict[str, Any]:
    return {
        "ordered_group_labels": tuple(patient_input.ordered_group_labels),
        "mass_mode": patient_input.mass_mode,
        **dict(patient_input.metadata),
    }


def _tensor_array(value: object) -> np.ndarray:
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None  # type: ignore[assignment]
    if torch is not None and torch.is_tensor(value):
        return np.asarray(value.detach().cpu(), dtype=float)
    return np.asarray(value, dtype=float)


def _tensor_scalar(value: object) -> float:
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None  # type: ignore[assignment]
    if torch is not None and torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _reason_key(status: str) -> str:
    return "defer_reason" if status == "deferred" else "failure_reason"


def _optimizer_status_label(run_status: FitRunStatus) -> str:
    if run_status.stage == "optimizer":
        return run_status.status
    return "not_run"


def _optimizer_run_info_payload(train_result: TrainResult | None) -> dict[str, Any]:
    if train_result is None:
        return {}
    info = train_result.run_info
    payload: dict[str, Any] = {
        "optimizer_protocol": info.optimizer_protocol,
        "scheduler_policy": info.scheduler_policy,
        "ablation_mode": info.ablation_mode,
    }
    if info.optimizer_exit_flag is not None:
        payload["optimizer_exit_flag"] = info.optimizer_exit_flag
    if info.reason:
        if train_result.status == "ok":
            payload["completion_reason"] = info.reason
        else:
            payload[_reason_key(train_result.status)] = info.reason
    for field_name in (
        "n_steps",
        "warmup_steps_completed",
        "main_steps_completed",
        "step",
        "initial_total",
        "final_total",
        "absolute_improvement",
        "relative_improvement",
        "message",
    ):
        value = getattr(info, field_name)
        if value is not None:
            payload[field_name] = value
    return payload


def _fit_status_diagnostics(
    run_status: FitRunStatus,
    *,
    train_result: TrainResult | None,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "fit_status_stage": run_status.stage,
        "optimizer_status": _optimizer_status_label(run_status),
    }
    diagnostics.update(_optimizer_run_info_payload(train_result))
    if run_status.reason:
        if run_status.status == "ok":
            diagnostics.setdefault("completion_reason", run_status.reason)
        else:
            diagnostics.setdefault(_reason_key(run_status.status), run_status.reason)
    diagnostics.update(dict(run_status.context))
    return diagnostics


def _require_ok_train_result(train_result: TrainResult | None) -> TrainResult:
    if train_result is None or train_result.status != "ok":
        raise ContractError("ok fit assembly requires an ok TrainResult")
    if train_result.A is None or train_result.d is None or train_result.e is None:
        raise ContractError("ok TrainResult must carry detached A, d, and e")
    if train_result.loss_ledger is None:
        raise ContractError("ok TrainResult must carry a detached loss ledger")
    if train_result.train_config is None:
        raise ContractError("ok TrainResult must carry the resolved TrainConfig")
    return train_result


def _loss_ledger_observation_comparison_plan(train_result: TrainResult) -> dict[str, Any]:
    plan = dict(train_result.loss_ledger.metadata.get("observation_comparison_plan", {}))
    required = {"block_construction_policy", "n_blocks_by_patient", "n_evidence_blocks"}
    if not required.issubset(plan):
        raise ContractError("ok TrainResult loss ledger must carry observation comparison plan metadata")
    return plan


def _patient_evidence_block_count(
    comparison_plan: dict[str, Any],
    patient_id: str,
) -> int:
    blocks_by_patient = dict(comparison_plan["n_blocks_by_patient"])
    patient_key = str(patient_id)
    if patient_key not in blocks_by_patient:
        raise ContractError("observation comparison plan must include each realized patient")
    return int(blocks_by_patient[patient_key])


def build_cohort_relation(
    *,
    patient_ids: tuple[str, ...],
    run_status: FitRunStatus,
    train_result: TrainResult | None,
    K: int,
    state_ids: tuple[int, ...] | None,
) -> CohortRelation:
    """Assemble the result-level cohort common structure from detached training output."""
    if run_status.status != "ok":
        status_diagnostics = _fit_status_diagnostics(run_status, train_result=train_result)
        return CohortRelation(
            cohort_id="cohort",
            A=np.zeros((K, K), dtype=float),
            d=np.zeros(K, dtype=float),
            e=np.zeros(K, dtype=float),
            support_patient_ids=(),
            fit_status=run_status.status,
            state_ids=state_ids,
            metadata=status_diagnostics,
        )
    train_result = _require_ok_train_result(train_result)
    recurrence = train_result.loss_ledger.recurrence
    return CohortRelation(
        cohort_id="cohort",
        A=_tensor_array(recurrence.A_bar),
        d=_tensor_array(recurrence.d_bar),
        e=_tensor_array(recurrence.e_bar),
        support_patient_ids=patient_ids,
        fit_status="ok",
        dispersion=_tensor_scalar(recurrence.dispersion),
        state_ids=state_ids,
        metadata={
            "mode": "cohort_common_structure",
            "support_n_patients": int(recurrence.support_n_patients),
            "optimizer_status": train_result.status,
            **_optimizer_run_info_payload(train_result),
        },
    )


def build_patient_results(
    patient_inputs: Sequence[_PatientFitInput],
    *,
    run_status: FitRunStatus,
    train_result: TrainResult | None,
    evidence_blocks: Sequence[EvidenceBlock],
) -> tuple[PatientRelationResult, ...]:
    """Assemble patient result payloads from detached training output."""
    if run_status.status != "ok":
        status_diagnostics = _fit_status_diagnostics(run_status, train_result=train_result)
        reason_key = _reason_key(run_status.status)
        reason = str(status_diagnostics.get(reason_key, run_status.reason or "optimizer_failed"))
        return tuple(
            PatientRelationResult(
                patient_id=patient_input.patient_id,
                fit_status=run_status.status,
                state_ids=_resolve_patient_state_ids(patient_input),
                audit=PatientRelationAudit(
                    patient_id=patient_input.patient_id,
                    timepoint_order=patient_input.ordered_group_labels,
                    mass_mode=patient_input.mass_mode,
                    observation_fit_status=run_status.status,
                    relation_status=run_status.status,
                    metadata={
                        **_shared_bridge_audit_metadata(patient_input),
                        reason_key: reason,
                        "fit_status_stage": run_status.stage,
                    },
                ),
                diagnostics={
                    **_shared_bridge_count_diagnostics(patient_input),
                    "mode": f"canonical_full_{run_status.status}",
                    **status_diagnostics,
                },
                implementation_tier="canonical_full",
            )
            for patient_input in patient_inputs
        )

    train_result = _require_ok_train_result(train_result)
    A_all = _tensor_array(train_result.A)
    d_all = _tensor_array(train_result.d)
    e_all = _tensor_array(train_result.e)
    comparison_plan = _loss_ledger_observation_comparison_plan(train_result)
    block_construction_policy = str(comparison_plan["block_construction_policy"])
    block_counts = Counter(str(block.patient_id) for block in evidence_blocks)
    results: list[PatientRelationResult] = []
    for patient_index, patient_input in enumerate(patient_inputs):
        pre_group, post_group = patient_input.groups
        mu_minus = _mean_group_composition(pre_group)
        mu_plus = _mean_group_composition(post_group)
        patient_A = A_all[patient_index]
        patient_d = d_all[patient_index]
        patient_e = e_all[patient_index]
        transition_burden = np.asarray(patient_A, dtype=float) * mu_minus[:, None]
        emergence_scale = float(np.sum(mu_minus, dtype=float))
        patient_n_blocks = _patient_evidence_block_count(comparison_plan, patient_input.patient_id)
        if patient_n_blocks != int(block_counts[patient_input.patient_id]):
            raise ContractError("observation comparison plan must align with evidence_blocks")
        audit = PatientRelationAudit(
            patient_id=patient_input.patient_id,
            timepoint_order=patient_input.ordered_group_labels,
            mass_mode=patient_input.mass_mode,
            n_pre_observations=int(patient_input.n_observations_by_group[pre_group.group_label]),
            n_post_observations=int(patient_input.n_observations_by_group[post_group.group_label]),
            observation_fit_status="D_obs^BalancedSinkhornDivergence-v1",
            relation_status="ok",
            metadata={
                **_shared_bridge_audit_metadata(patient_input),
                "estimator_mode": FIT_RUNTIME_MODE,
                "optimizer_status": train_result.status,
                "n_evidence_blocks": patient_n_blocks,
                "block_construction_policy": block_construction_policy,
            },
        )
        results.append(
            PatientRelationResult(
                patient_id=patient_input.patient_id,
                fit_status="ok",
                A=patient_A,
                d=patient_d,
                e=patient_e,
                mu_minus=mu_minus,
                mu_plus=mu_plus,
                state_ids=_resolve_patient_state_ids(patient_input),
                audit=audit,
                diagnostics={
                    **_shared_bridge_count_diagnostics(patient_input),
                    "mode": FIT_RUNTIME_MODE,
                    "observation_fit_status": "D_obs^BalancedSinkhornDivergence-v1",
                    "optimizer_status": train_result.status,
                    "n_evidence_blocks": patient_n_blocks,
                    "block_construction_policy": block_construction_policy,
                },
                auxiliary={
                    "matched_transition_burden": transition_burden,
                    "raw_matched_transition_burden": transition_burden,
                    "source_unmatched_burden": np.asarray(patient_d, dtype=float) * mu_minus,
                    "target_unmatched_burden": np.asarray(patient_e, dtype=float) * emergence_scale,
                    "model_implied_mu_plus": _reconstruct_post_burden(patient_A, patient_e, mu_minus),
                },
                implementation_tier="canonical_full",
                objective=train_result.loss_ledger,
            )
        )
    return tuple(results)


def assemble_stride_fit_result(
    *,
    patient_inputs: tuple[_PatientFitInput, ...],
    task_config: TaskConfig,
    run_status: FitRunStatus,
    train_result: TrainResult | None,
    evidence_blocks: tuple[EvidenceBlock, ...],
    state_ids: tuple[int, ...] | None,
) -> STRIDEFitResult:
    """Assemble the public fit result from workflow and optimizer outputs."""
    if run_status.stage == "optimizer":
        if train_result is None:
            raise ContractError("optimizer fit status requires a TrainResult")
        if train_result.status != run_status.status:
            raise ContractError("FitRunStatus.status must match TrainResult.status")
    elif train_result is not None:
        raise ContractError("task feasibility status must not carry a TrainResult")

    patient_results = build_patient_results(
        patient_inputs,
        run_status=run_status,
        train_result=train_result,
        evidence_blocks=evidence_blocks,
    )
    cohort_relation = build_cohort_relation(
        patient_ids=tuple(patient_input.patient_id for patient_input in patient_inputs),
        run_status=run_status,
        train_result=train_result,
        K=task_config.K,
        state_ids=state_ids,
    )
    patient_status_counts = _count_patient_statuses(patient_results)
    status_diagnostics = _fit_status_diagnostics(run_status, train_result=train_result)
    objective = None
    provenance = None
    comparison_plan: dict[str, Any] = {}
    if run_status.status == "ok":
        ok_train_result = _require_ok_train_result(train_result)
        objective = ok_train_result.loss_ledger
        comparison_plan = _loss_ledger_observation_comparison_plan(ok_train_result)
        provenance = build_successful_provenance(
            ok_train_result.loss_ledger,
            train_config=ok_train_result.train_config,
            optimizer_run_info=ok_train_result.run_info,
        )
    metadata = {
        "implementation_tier": "canonical_full",
        "optimizer_status": _optimizer_status_label(run_status),
        "fit_status_stage": run_status.stage,
        "n_evidence_blocks": len(evidence_blocks),
        "source": task_config.source,
        "target": task_config.target,
        "K": task_config.K,
    }
    if comparison_plan:
        metadata.update(
            {
                "block_construction_policy": comparison_plan["block_construction_policy"],
                "n_blocks_by_patient": dict(comparison_plan["n_blocks_by_patient"]),
            }
        )
    summaries = {
        "n_patients": len(patient_results),
        "n_realized_patients": sum(1 for result in patient_results if result.fit_status == "ok"),
        "patient_status_counts": patient_status_counts,
        "cohort_fit_status": cohort_relation.fit_status,
        "cohort_support_n_patients": len(cohort_relation.support_patient_ids),
        "implementation_tier": "canonical_full",
        "optimizer_status": _optimizer_status_label(run_status),
        "n_evidence_blocks": len(evidence_blocks),
    }
    if comparison_plan:
        summaries.update(
            {
                "block_construction_policy": comparison_plan["block_construction_policy"],
                "n_blocks_by_patient": dict(comparison_plan["n_blocks_by_patient"]),
            }
        )
    diagnostics = {
        "mode": FIT_RUNTIME_MODE,
        "patient_status_counts": patient_status_counts,
        "cohort_support_patient_ids": tuple(cohort_relation.support_patient_ids),
        "implementation_tier": "canonical_full",
        "n_evidence_blocks": len(evidence_blocks),
        **status_diagnostics,
    }
    if comparison_plan:
        diagnostics.update(
            {
                "block_construction_policy": comparison_plan["block_construction_policy"],
                "n_blocks_by_patient": dict(comparison_plan["n_blocks_by_patient"]),
            }
        )
    if train_result is not None and train_result.trace is not None:
        diagnostics["optimizer_trace"] = train_result.trace
    return STRIDEFitResult(
        patient_inputs=patient_inputs,
        patient_results=patient_results,
        cohort_relation=cohort_relation,
        fit_status=run_status.status,
        implementation_tier="canonical_full",
        objective=objective,
        provenance=provenance,
        summaries=summaries,
        diagnostics=diagnostics,
        uncertainty=None,
        metadata=metadata,
    )


__all__ = [
    "assemble_stride_fit_result",
    "build_cohort_relation",
    "build_patient_results",
]
