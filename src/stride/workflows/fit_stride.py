"""Canonical bridge-input contracts and minimal STRIDE fit orchestration."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np

from ..basis.contracts import StateBasis, validate_state_basis
from ..errors import ContractError
from ..geometry.state_geometry import StateGeometry
from ..latent.operators import PatientRelationAudit, initialize_patient_relation
from ..latent.recurrence import (
    PatientRecurrenceEmbedding,
    RecurrenceConfig,
    RecurrenceParameters,
    RecurrenceResult,
)
from ..observation.contracts import (
    FovObservation,
    ObservationDiscrepancyConfig,
    validate_fov_observation,
)
from ..observation.discrepancy import compute_observation_discrepancy
from ..observation.measures import build_domain_stratified_measure
from ..outputs.fit_result import PatientBridgeResult, STRIDEFitResult
from ..outputs.uncertainty import (
    PatientBootstrapConfig,
    PatientBootstrapUncertaintyResult,
    STRIDEBootstrapUncertaintyResult,
    build_cohort_bootstrap_summary,
    summarize_bootstrap_array,
)


def _require_nonempty_identifier(value: str, *, field_name: str) -> str:
    normalized = str(value).strip()
    if normalized == "":
        raise ContractError(f"{field_name} must be a non-empty string")
    return normalized


def _observation_key(observation: FovObservation) -> tuple[str, str, str, str]:
    return (
        str(observation.patient_id),
        str(observation.timepoint),
        str(observation.fov_id),
        str(observation.domain_label),
    )


def _normalize_nested_counts(
    counts: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    return {
        str(group_label): {str(domain_label): int(count) for domain_label, count in domain_counts.items()}
        for group_label, domain_counts in counts.items()
    }


def _count_patient_statuses(results: tuple[PatientBridgeResult, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.fit_status] = counts.get(result.fit_status, 0) + 1
    return counts


def _count_uncertainty_statuses(
    statuses: tuple[str, ...],
) -> dict[str, int]:
    counts = Counter(str(status) for status in statuses)
    return {
        "ok": int(counts.get("ok", 0)),
        "deferred": int(counts.get("deferred", 0)),
        "failed": int(counts.get("failed", 0)),
    }


_MINIMAL_SUPPORTED_CASE = "two_group_uniform_patient_bridge"
_CANONICAL_BRIDGE_MODE = "observation_to_patient_bridge_v1"
_CANONICAL_BRIDGE_METHOD = "domain_stratified_cartesian_observation_discrepancy"
_PRIMARY_DEFER_REASON = "requires_exactly_two_ordered_groups"
_GEOMETRY_DEFER_REASON = "requires_shared_state_geometry"
_OBSERVATION_DEFER_REASON = "requires_supported_observation_discrepancy_rows"
_BRIDGE_DISCREPANCY_CONFIG = ObservationDiscrepancyConfig(
    eps_schedule=(1.0, 0.2),
    max_iter=4000,
    tol=1e-8,
    n_min_proto=0.0,
)


@dataclass(frozen=True)
class _GroupDomainSummary:
    domain_label: str
    n_observations: int
    share: float
    mean_composition: np.ndarray


@dataclass(frozen=True)
class _DomainCoupling:
    pre_domain_label: str
    post_domain_label: str
    weight: float


def _resolve_ordered_group_labels(
    observed_labels: tuple[str, ...],
    *,
    declared_order: tuple[str, ...],
) -> tuple[str, ...]:
    if not declared_order:
        return observed_labels

    extras = [label for label in observed_labels if label not in declared_order]
    if extras:
        raise ContractError(
            "Observed group labels are missing from STRIDEFitConfig.timepoint_order: "
            f"{tuple(extras)}"
        )

    return tuple(label for label in declared_order if label in observed_labels)


def _resolve_patient_state_ids(patient_input: "PatientBridgeInput") -> tuple[int, ...] | None:
    if patient_input.state_basis is not None:
        return patient_input.state_basis.resolved_state_ids
    if patient_input.geometry is not None:
        return tuple(int(state_id) for state_id in patient_input.geometry.state_ids)
    return None


def _shared_bridge_count_diagnostics(patient_input: "PatientBridgeInput") -> dict[str, Any]:
    return {
        "ordered_group_labels": patient_input.ordered_group_labels,
        "n_groups": len(patient_input.ordered_group_labels),
        "n_observations_by_group": dict(patient_input.n_observations_by_group),
        "n_observations_by_domain": dict(patient_input.n_observations_by_domain),
        "n_observations_by_group_and_domain": _normalize_nested_counts(
            patient_input.n_observations_by_group_and_domain
        ),
    }


def _shared_bridge_audit_metadata(patient_input: "PatientBridgeInput") -> dict[str, Any]:
    diagnostics = _shared_bridge_count_diagnostics(patient_input)
    return {
        "n_observations_by_group": diagnostics["n_observations_by_group"],
        "n_observations_by_domain": diagnostics["n_observations_by_domain"],
        "n_observations_by_group_and_domain": diagnostics["n_observations_by_group_and_domain"],
        "bridge_input_metadata": dict(patient_input.metadata),
    }


def _evaluate_patient_bridge_support(
    patient_input: "PatientBridgeInput",
) -> tuple[bool, str | None, str | None]:
    if patient_input.mass_mode != "uniform":
        return (
            False,
            "requires_uniform_mass_mode",
            "Bridge fitting remains deferred because only uniform-mass patient inputs are "
            "supported in the current pass.",
        )

    if len(patient_input.ordered_group_labels) != 2:
        return (
            False,
            _PRIMARY_DEFER_REASON,
            "Bridge fitting remains deferred because the current pass supports only "
            "patients with exactly two ordered groups.",
        )

    if patient_input.geometry is None:
        return (
            False,
            _GEOMETRY_DEFER_REASON,
            "Bridge fitting remains deferred because the current pass requires shared "
            "state geometry for geometry-aware off-diagonal estimation.",
        )

    return True, None, None


def _mean_group_composition(group: "BridgeObservationGroup") -> np.ndarray:
    compositions = np.vstack(
        [np.asarray(observation.community_composition, dtype=float) for observation in group.observations]
    ).astype(float, copy=False)
    return np.mean(compositions, axis=0, dtype=float)


def _clip_small_negative(
    values: np.ndarray,
    *,
    field_name: str,
    atol: float = 1e-12,
) -> np.ndarray:
    clipped = np.asarray(values, dtype=float).copy()
    if np.any(clipped < -float(atol)):
        raise ContractError(
            f"{field_name} became negative beyond tolerance during realized bridge estimation"
        )
    clipped[clipped < 0.0] = 0.0
    clipped[np.abs(clipped) <= float(atol)] = 0.0
    return clipped


def _normalize_burden_to_operator(
    matched_transition_burden: np.ndarray,
    source_total_burden: np.ndarray,
    *,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matched = _clip_small_negative(matched_transition_burden, field_name="matched_transition_burden", atol=atol)
    mu_minus = _clip_small_negative(source_total_burden, field_name="mu_minus", atol=atol)
    row_totals = np.sum(matched, axis=1, dtype=float)
    projected_matched = matched.copy()
    for row_idx, (row_total, row_budget) in enumerate(zip(row_totals, mu_minus, strict=True)):
        if row_budget <= float(atol) or row_total <= row_budget + float(atol):
            continue
        projected_matched[row_idx] *= float(row_budget / row_total)

    A = np.zeros_like(projected_matched, dtype=float)
    d = np.ones_like(mu_minus, dtype=float)
    supported_rows = mu_minus > float(atol)
    if np.any(supported_rows):
        A[supported_rows] = projected_matched[supported_rows] / mu_minus[supported_rows, None]
        d[supported_rows] = 1.0 - np.sum(A[supported_rows], axis=1, dtype=float)

    A = _clip_small_negative(A, field_name="A", atol=atol)
    d = _clip_small_negative(d, field_name="d", atol=atol)
    row_residual = 1.0 - (np.sum(A, axis=1, dtype=float) + d)
    if np.any(np.abs(row_residual) > 1e-8):
        raise ContractError("Canonical bridge construction failed the operator row-normalization check")
    return A, d, mu_minus, projected_matched


def _serialize_group_domain_summaries(
    summaries: tuple[_GroupDomainSummary, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "domain_label": summary.domain_label,
            "n_observations": summary.n_observations,
            "share": summary.share,
            "mean_composition": summary.mean_composition.tolist(),
        }
        for summary in summaries
    ]


def _serialize_domain_couplings(
    couplings: tuple[_DomainCoupling, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "pre_domain_label": coupling.pre_domain_label,
            "post_domain_label": coupling.post_domain_label,
            "weight": coupling.weight,
        }
        for coupling in couplings
    ]


def _summarize_group_domains(group: "BridgeObservationGroup") -> tuple[_GroupDomainSummary, ...]:
    total_observations = len(group.observations)
    summaries: list[_GroupDomainSummary] = []
    for domain_label in sorted(group.observations_by_domain):
        measure = build_domain_stratified_measure(
            group.observations_by_domain[domain_label],
            domain_label=domain_label,
        )
        summaries.append(
            _GroupDomainSummary(
                domain_label=str(domain_label),
                n_observations=len(measure.observations),
                share=float(len(measure.observations) / total_observations),
                mean_composition=np.mean(measure.state_matrix, axis=0, dtype=float),
            )
        )
    return tuple(summaries)


def _build_domain_coupling(
    pre_domains: tuple[_GroupDomainSummary, ...],
    post_domains: tuple[_GroupDomainSummary, ...],
    *,
    atol: float = 1e-12,
) -> tuple[_DomainCoupling, ...]:
    pre_remaining = {summary.domain_label: float(summary.share) for summary in pre_domains}
    post_remaining = {summary.domain_label: float(summary.share) for summary in post_domains}
    couplings: list[_DomainCoupling] = []

    for domain_label in sorted(set(pre_remaining) & set(post_remaining)):
        weight = min(pre_remaining[domain_label], post_remaining[domain_label])
        if weight <= float(atol):
            continue
        couplings.append(
            _DomainCoupling(
                pre_domain_label=domain_label,
                post_domain_label=domain_label,
                weight=float(weight),
            )
        )
        pre_remaining[domain_label] -= float(weight)
        post_remaining[domain_label] -= float(weight)

    for pre_label in sorted(pre_remaining):
        while pre_remaining[pre_label] > float(atol):
            progressed = False
            for post_label in sorted(post_remaining):
                if post_remaining[post_label] <= float(atol):
                    continue
                weight = min(pre_remaining[pre_label], post_remaining[post_label])
                if weight <= float(atol):
                    continue
                couplings.append(
                    _DomainCoupling(
                        pre_domain_label=pre_label,
                        post_domain_label=post_label,
                        weight=float(weight),
                    )
                )
                pre_remaining[pre_label] -= float(weight)
                post_remaining[post_label] -= float(weight)
                progressed = True
                if pre_remaining[pre_label] <= float(atol):
                    break
            if not progressed:
                break

    remaining_pre = sum(max(value, 0.0) for value in pre_remaining.values())
    remaining_post = sum(max(value, 0.0) for value in post_remaining.values())
    if remaining_pre > float(atol) or remaining_post > float(atol):
        raise ContractError("Domain coupling did not exhaust group shares in the realized bridge path")
    return tuple(couplings)


def _ordered_geometry_pairs(geometry: StateGeometry) -> tuple[tuple[int, int], ...]:
    cost_matrix = np.asarray(geometry.cost_matrix, dtype=float)
    adjacency_matrix = np.asarray(geometry.adjacency_matrix, dtype=float)
    similarity_graph = np.asarray(geometry.similarity_graph, dtype=float)
    if cost_matrix.shape != adjacency_matrix.shape or cost_matrix.shape != similarity_graph.shape:
        raise ContractError("StateGeometry arrays must share the same square [K, K] shape")

    normalized_cost = cost_matrix / float(geometry.cost_scale)
    ordered_pairs = [
        (float(normalized_cost[row_idx, col_idx]), -float(similarity_graph[row_idx, col_idx]), row_idx, col_idx)
        for row_idx in range(cost_matrix.shape[0])
        for col_idx in range(cost_matrix.shape[1])
        if adjacency_matrix[row_idx, col_idx] > 0.0
    ]
    ordered_pairs.sort()
    return tuple((row_idx, col_idx) for _cost, _neg_similarity, row_idx, col_idx in ordered_pairs)


def _geometry_greedy_transport(
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    geometry: StateGeometry,
    *,
    atol: float = 1e-12,
) -> np.ndarray:
    source = np.asarray(source_mass, dtype=float).reshape(-1)
    target = np.asarray(target_mass, dtype=float).reshape(-1)
    if source.shape != target.shape:
        raise ContractError("source_mass and target_mass must share the same [K] shape")

    cost_shape = np.asarray(geometry.cost_matrix, dtype=float).shape
    if cost_shape != (source.shape[0], source.shape[0]):
        raise ContractError("StateGeometry must align to the shared K-state axis")

    plan = np.zeros((source.shape[0], source.shape[0]), dtype=float)
    remaining_source = source.copy()
    remaining_target = target.copy()

    for row_idx, col_idx in _ordered_geometry_pairs(geometry):
        if remaining_source[row_idx] <= float(atol) or remaining_target[col_idx] <= float(atol):
            continue
        transported = min(remaining_source[row_idx], remaining_target[col_idx])
        if transported <= float(atol):
            continue
        plan[row_idx, col_idx] += float(transported)
        remaining_source[row_idx] -= float(transported)
        remaining_target[col_idx] -= float(transported)

    _clip_small_negative(remaining_source, field_name="remaining_source", atol=atol)
    _clip_small_negative(remaining_target, field_name="remaining_target", atol=atol)
    return _clip_small_negative(plan, field_name="A", atol=atol)


def _build_realized_patient_bridge_result(
    patient_input: "PatientBridgeInput",
) -> PatientBridgeResult:
    if patient_input.geometry is None:
        raise ContractError("Realized bridge estimation requires PatientBridgeInput.geometry")

    pre_group, post_group = patient_input.groups
    pre_label, post_label = patient_input.ordered_group_labels
    geometry = patient_input.geometry
    mu_minus_burden = _mean_group_composition(pre_group)
    mu_plus_burden = _mean_group_composition(post_group)
    n_states = np.asarray(geometry.cost_matrix, dtype=float).shape[0]
    matched_transition_burden = np.zeros((n_states, n_states), dtype=float)
    source_unmatched_burden = np.zeros(n_states, dtype=float)
    target_unmatched_burden = np.zeros(n_states, dtype=float)
    domain_pair_statuses: list[dict[str, Any]] = []

    total_pre_observations = len(pre_group.observations)
    total_post_observations = len(post_group.observations)
    for pre_domain_label, pre_domain_observations in sorted(pre_group.observations_by_domain.items()):
        pre_measure = build_domain_stratified_measure(
            pre_domain_observations,
            domain_label=pre_domain_label,
        )
        pre_share = float(len(pre_measure.observations) / total_pre_observations)
        for post_domain_label, post_domain_observations in sorted(post_group.observations_by_domain.items()):
            post_measure = build_domain_stratified_measure(
                post_domain_observations,
                domain_label=post_domain_label,
            )
            post_share = float(len(post_measure.observations) / total_post_observations)
            discrepancy = compute_observation_discrepancy(
                pre_measure,
                post_measure,
                match_penalty=np.asarray([1.0], dtype=float),
                geometry=geometry,
                cfg=_BRIDGE_DISCREPANCY_CONFIG,
                return_plan=True,
            )

            pair_status_values = tuple(str(status) for status in discrepancy.result.status.tolist())
            n_pairs = int(discrepancy.metadata["n_observation_pairs"])
            domain_pair_statuses.append(
                {
                    "pre_domain_label": str(pre_domain_label),
                    "post_domain_label": str(post_domain_label),
                    "n_pairs": n_pairs,
                    "pair_status_counts": dict(Counter(pair_status_values)),
                }
            )
            if any(status != "ok" for status in pair_status_values):
                return _build_deferred_patient_bridge_result(
                    patient_input,
                    defer_reason=_OBSERVATION_DEFER_REASON,
                    message=(
                        "Bridge fitting remains deferred because at least one domain-pair "
                        "observation discrepancy row did not produce an honest canonical "
                        "matching summary."
                    ),
                )

            pair_weight = (pre_share * post_share) / float(n_pairs)
            matching_plan = discrepancy.result.matching_plan
            if matching_plan is None:
                raise ContractError("Canonical bridge construction requires observation matching plans")

            matched_transition_burden += pair_weight * np.sum(matching_plan, axis=0, dtype=float)
            source_unmatched_burden += pair_weight * np.sum(
                discrepancy.result.details["source_unmatched_mass_by_state"],
                axis=0,
                dtype=float,
            )
            target_unmatched_burden += pair_weight * np.sum(
                discrepancy.result.details["target_unmatched_mass_by_state"],
                axis=0,
                dtype=float,
            )

    A, d, mu_minus, projected_transition_burden = _normalize_burden_to_operator(
        matched_transition_burden,
        mu_minus_burden,
    )
    mu_plus = _clip_small_negative(mu_plus_burden, field_name="mu_plus")
    emergence_scale = float(np.sum(mu_minus, dtype=float))
    if emergence_scale <= 0.0:
        raise ContractError("Canonical bridge construction requires positive pre-side burden support")
    e = _clip_small_negative(target_unmatched_burden / emergence_scale, field_name="e")

    estimator_metadata = {
        "supported_case": _MINIMAL_SUPPORTED_CASE,
        "estimator_mode": _CANONICAL_BRIDGE_MODE,
        "estimator_method": _CANONICAL_BRIDGE_METHOD,
            "observation_discrepancy_config": {
                "eps_schedule": tuple(_BRIDGE_DISCREPANCY_CONFIG.eps_schedule),
                "max_iter": int(_BRIDGE_DISCREPANCY_CONFIG.max_iter),
                "tol": float(_BRIDGE_DISCREPANCY_CONFIG.tol),
                "n_min_proto": float(_BRIDGE_DISCREPANCY_CONFIG.n_min_proto),
                "match_penalty": 1.0,
            },
            "domain_pair_statuses": domain_pair_statuses,
            "matched_transition_projection_applied": bool(
                not np.allclose(projected_transition_burden, matched_transition_burden)
            ),
        }

    audit = PatientRelationAudit(
        patient_id=patient_input.patient_id,
        timepoint_order=patient_input.ordered_group_labels,
        mass_mode=patient_input.mass_mode,
        n_pre_observations=int(patient_input.n_observations_by_group[pre_label]),
        n_post_observations=int(patient_input.n_observations_by_group[post_label]),
        observation_fit_status="observation_discrepancy_bridge",
        bridge_status="ok",
        metadata={
            **_shared_bridge_audit_metadata(patient_input),
            **estimator_metadata,
        },
    )
    relation = initialize_patient_relation(
        patient_id=patient_input.patient_id,
        A=A,
        d=d,
        e=e,
        mu_minus=mu_minus,
        mu_plus=mu_plus,
        state_ids=_resolve_patient_state_ids(patient_input),
        audit=audit,
    )
    return PatientBridgeResult(
        patient_id=patient_input.patient_id,
        fit_status="ok",
        A=relation.A,
        d=relation.d,
        e=relation.e,
        mu_minus=relation.mu_minus,
        mu_plus=relation.mu_plus,
        state_ids=relation.state_ids,
        audit=relation.audit,
        diagnostics={
            **_shared_bridge_count_diagnostics(patient_input),
            **estimator_metadata,
            "mode": "patient_bridge_realized",
            "matched_burden_mass": float(np.sum(projected_transition_burden, dtype=float)),
            "offdiagonal_operator_mass": float(np.sum(relation.A, dtype=float) - np.trace(relation.A)),
            "source_unmatched_burden_mass": float(np.sum(source_unmatched_burden, dtype=float)),
            "target_unmatched_burden_mass": float(np.sum(target_unmatched_burden, dtype=float)),
            "message": (
                "Applied the canonical two-group bridge using domain-stratified "
                "observation discrepancy summaries over ROI/FOV evidence."
            ),
        },
        auxiliary={
            "matched_transition_burden": projected_transition_burden,
            "raw_matched_transition_burden": matched_transition_burden,
            "source_unmatched_burden": source_unmatched_burden,
            "target_unmatched_burden": target_unmatched_burden,
        },
    )


def _build_deferred_patient_bridge_result(
    patient_input: "PatientBridgeInput",
    *,
    defer_reason: str,
    message: str,
) -> PatientBridgeResult:
    group_labels = patient_input.ordered_group_labels
    n_pre_observations = (
        int(patient_input.n_observations_by_group[group_labels[0]])
        if len(group_labels) >= 1
        else None
    )
    n_post_observations = (
        int(patient_input.n_observations_by_group[group_labels[1]])
        if len(group_labels) >= 2
        else None
    )
    audit = PatientRelationAudit(
        patient_id=patient_input.patient_id,
        timepoint_order=group_labels,
        mass_mode=patient_input.mass_mode,
        n_pre_observations=n_pre_observations,
        n_post_observations=n_post_observations,
        observation_fit_status="deferred",
        bridge_status="deferred",
        metadata={
            **_shared_bridge_audit_metadata(patient_input),
            "defer_reason": defer_reason,
        },
    )
    return PatientBridgeResult(
        patient_id=patient_input.patient_id,
        fit_status="deferred",
        state_ids=_resolve_patient_state_ids(patient_input),
        audit=audit,
        diagnostics={
            **_shared_bridge_count_diagnostics(patient_input),
            "mode": "deferred",
            "defer_reason": defer_reason,
            "message": message,
        },
    )


def _bootstrap_observation(
    observation: FovObservation,
    *,
    replicate_index: int,
    draw_index: int,
    source_fov_id: str | None = None,
) -> FovObservation:
    original_fov_id = str(source_fov_id or observation.fov_id)
    return replace(
        observation,
        fov_id=f"{original_fov_id}__boot_r{replicate_index}_d{draw_index}",
        metadata={
            **dict(observation.metadata),
            "bootstrap_source_fov_id": original_fov_id,
            "bootstrap_replicate": int(replicate_index),
            "bootstrap_draw_index": int(draw_index),
        },
    )


def _bootstrap_patient_bridge_input(
    patient_input: "PatientBridgeInput",
    *,
    rng: np.random.Generator,
    replicate_index: int,
    preserve_domain_stratification: bool,
) -> "PatientBridgeInput":
    bootstrap_groups: list[BridgeObservationGroup] = []
    counts_by_group: dict[str, int] = {}
    counts_by_domain: dict[str, int] = {}
    counts_by_group_and_domain: dict[str, dict[str, int]] = {}

    for group in patient_input.groups:
        sampled_group_observations: list[FovObservation] = []
        sampled_group_by_domain: dict[str, tuple[FovObservation, ...]] = {}

        if preserve_domain_stratification:
            for domain_label, domain_observations in group.observations_by_domain.items():
                sampled_domain_observations = tuple(
                    _bootstrap_observation(
                        domain_observations[int(sample_idx)],
                        replicate_index=replicate_index,
                        draw_index=draw_idx,
                    )
                    for draw_idx, sample_idx in enumerate(
                        rng.integers(
                            low=0,
                            high=len(domain_observations),
                            size=len(domain_observations),
                        )
                    )
                )
                sampled_group_by_domain[str(domain_label)] = sampled_domain_observations
                sampled_group_observations.extend(sampled_domain_observations)
        else:
            sampled_group_observations = [
                _bootstrap_observation(
                    group.observations[int(sample_idx)],
                    replicate_index=replicate_index,
                    draw_index=draw_idx,
                )
                for draw_idx, sample_idx in enumerate(
                    rng.integers(
                        low=0,
                        high=len(group.observations),
                        size=len(group.observations),
                    )
                )
            ]
            grouped_lists: dict[str, list[FovObservation]] = {}
            for sampled_observation in sampled_group_observations:
                grouped_lists.setdefault(str(sampled_observation.domain_label), []).append(sampled_observation)
            sampled_group_by_domain = {
                domain_label: tuple(domain_observations)
                for domain_label, domain_observations in grouped_lists.items()
            }

        counts_by_group[group.group_label] = len(sampled_group_observations)
        counts_by_group_and_domain[group.group_label] = {
            domain_label: len(domain_observations)
            for domain_label, domain_observations in sampled_group_by_domain.items()
        }
        for domain_label, domain_observations in sampled_group_by_domain.items():
            counts_by_domain[domain_label] = counts_by_domain.get(domain_label, 0) + len(domain_observations)

        bootstrap_groups.append(
            BridgeObservationGroup(
                group_label=group.group_label,
                observations=tuple(sampled_group_observations),
                observations_by_domain=sampled_group_by_domain,
            )
        )

    return PatientBridgeInput(
        patient_id=patient_input.patient_id,
        ordered_group_labels=patient_input.ordered_group_labels,
        groups=tuple(bootstrap_groups),
        n_observations_by_group=counts_by_group,
        n_observations_by_domain=counts_by_domain,
        n_observations_by_group_and_domain=counts_by_group_and_domain,
        state_basis=patient_input.state_basis,
        geometry=patient_input.geometry,
        mass_mode=patient_input.mass_mode,
        metadata={
            **dict(patient_input.metadata),
            "bootstrap_replicate": int(replicate_index),
            "preserve_domain_stratification": bool(preserve_domain_stratification),
        },
    )


def _build_deferred_patient_bootstrap_uncertainty(
    patient_input: "PatientBridgeInput",
    patient_result: PatientBridgeResult,
    *,
    config: PatientBootstrapConfig,
    message: str,
    defer_reason: str,
) -> PatientBootstrapUncertaintyResult:
    replicate_statuses = tuple("deferred" for _ in range(config.n_boot))
    replicate_diagnostics = tuple(
        {
            "replicate_index": replicate_index,
            "status": "deferred",
            "defer_reason": defer_reason,
            "message": message,
        }
        for replicate_index in range(config.n_boot)
    )
    return PatientBootstrapUncertaintyResult(
        patient_id=patient_input.patient_id,
        realized_fit_status=patient_result.fit_status,
        uncertainty_status="deferred",
        eligible=False,
        n_boot=config.n_boot,
        replicate_statuses=replicate_statuses,
        replicate_diagnostics=replicate_diagnostics,
        diagnostics={
            **_shared_bridge_count_diagnostics(patient_input),
            "mode": "patient_bridge_bootstrap_deferred",
            "defer_reason": defer_reason,
            "message": message,
        },
        metadata={
            "random_state": config.random_state,
            "preserve_domain_stratification": config.preserve_domain_stratification,
        },
    )


def _build_patient_bootstrap_uncertainty(
    patient_input: "PatientBridgeInput",
    patient_result: PatientBridgeResult,
    *,
    config: PatientBootstrapConfig,
    bootstrap_seed: int,
) -> PatientBootstrapUncertaintyResult:
    rng = np.random.default_rng(int(bootstrap_seed))
    replicate_statuses: list[str] = []
    replicate_diagnostics: list[dict[str, Any]] = []
    A_replicates: list[np.ndarray] = []
    d_replicates: list[np.ndarray] = []
    e_replicates: list[np.ndarray] = []

    for replicate_index in range(config.n_boot):
        bootstrap_input = _bootstrap_patient_bridge_input(
            patient_input,
            rng=rng,
            replicate_index=replicate_index,
            preserve_domain_stratification=config.preserve_domain_stratification,
        )
        try:
            bootstrap_result = _build_realized_patient_bridge_result(bootstrap_input)
        except ContractError as exc:
            replicate_statuses.append("failed")
            replicate_diagnostics.append(
                {
                    "replicate_index": replicate_index,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue

        replicate_statuses.append(bootstrap_result.fit_status)
        replicate_diagnostics.append(
            {
                "replicate_index": replicate_index,
                "status": bootstrap_result.fit_status,
                "defer_reason": bootstrap_result.diagnostics.get("defer_reason"),
                "message": bootstrap_result.diagnostics.get("message"),
                "n_observations_by_group": dict(bootstrap_input.n_observations_by_group),
                "n_observations_by_domain": dict(bootstrap_input.n_observations_by_domain),
                "n_observations_by_group_and_domain": _normalize_nested_counts(
                    bootstrap_input.n_observations_by_group_and_domain
                ),
            }
        )
        if bootstrap_result.fit_status != "ok":
            continue

        A_replicates.append(np.asarray(bootstrap_result.A, dtype=float))
        d_replicates.append(np.asarray(bootstrap_result.d, dtype=float))
        e_replicates.append(np.asarray(bootstrap_result.e, dtype=float))

    replicate_statuses_tuple = tuple(replicate_statuses)
    status_counts = _count_uncertainty_statuses(replicate_statuses_tuple)
    if A_replicates:
        uncertainty_status = "ok"
        A_summary = summarize_bootstrap_array(
            np.stack(A_replicates, axis=0),
            reference=np.asarray(patient_result.A, dtype=float),
        )
        d_summary = summarize_bootstrap_array(
            np.stack(d_replicates, axis=0),
            reference=np.asarray(patient_result.d, dtype=float),
        )
        e_summary = summarize_bootstrap_array(
            np.stack(e_replicates, axis=0),
            reference=np.asarray(patient_result.e, dtype=float),
        )
        message = (
            "Computed patient-level ROI/FOV bootstrap uncertainty for the canonical "
            "two-group realized bridge."
        )
    else:
        uncertainty_status = "deferred" if status_counts["deferred"] > 0 else "failed"
        A_summary = None
        d_summary = None
        e_summary = None
        message = (
            "Bootstrap uncertainty did not realize any successful bridge replicates for this patient."
        )

    return PatientBootstrapUncertaintyResult(
        patient_id=patient_input.patient_id,
        realized_fit_status=patient_result.fit_status,
        uncertainty_status=uncertainty_status,
        eligible=True,
        n_boot=config.n_boot,
        bootstrap_seed=int(bootstrap_seed),
        replicate_statuses=replicate_statuses_tuple,
        replicate_diagnostics=tuple(replicate_diagnostics),
        A_summary=A_summary,
        d_summary=d_summary,
        e_summary=e_summary,
        diagnostics={
            **_shared_bridge_count_diagnostics(patient_input),
            "mode": "patient_bridge_bootstrap",
            "status_counts": status_counts,
            "success_rate": float(status_counts["ok"] / config.n_boot),
            "message": message,
        },
        metadata={
            "random_state": config.random_state,
            "bootstrap_seed": int(bootstrap_seed),
            "preserve_domain_stratification": config.preserve_domain_stratification,
        },
    )


def _build_stride_bootstrap_uncertainty(
    patient_inputs: tuple["PatientBridgeInput", ...],
    patient_results: tuple[PatientBridgeResult, ...],
    *,
    config: PatientBootstrapConfig,
) -> STRIDEBootstrapUncertaintyResult:
    master_rng = np.random.default_rng(config.random_state)
    uncertainty_patient_results: list[PatientBootstrapUncertaintyResult] = []

    for patient_input, patient_result in zip(patient_inputs, patient_results, strict=True):
        is_supported, defer_reason, support_message = _evaluate_patient_bridge_support(patient_input)
        if patient_result.fit_status != "ok":
            uncertainty_patient_results.append(
                _build_deferred_patient_bootstrap_uncertainty(
                    patient_input,
                    patient_result,
                    config=config,
                    defer_reason=str(patient_result.diagnostics.get("defer_reason", "bridge_fit_deferred")),
                    message=(
                        "Bootstrap uncertainty was skipped because the realized bridge path "
                        "did not emit A, d, and e for this patient."
                    ),
                )
            )
            continue
        if not is_supported:
            uncertainty_patient_results.append(
                _build_deferred_patient_bootstrap_uncertainty(
                    patient_input,
                    patient_result,
                    config=config,
                    defer_reason=str(defer_reason),
                    message=(
                        "Bootstrap uncertainty was skipped because this patient does not "
                        "satisfy the current minimal supported bridge case. "
                        f"{support_message}"
                    ),
                )
            )
            continue

        bootstrap_seed = int(master_rng.integers(np.iinfo(np.int64).max))
        uncertainty_patient_results.append(
            _build_patient_bootstrap_uncertainty(
                patient_input,
                patient_result,
                config=config,
                bootstrap_seed=bootstrap_seed,
            )
        )

    patient_uncertainty_results = tuple(uncertainty_patient_results)
    return STRIDEBootstrapUncertaintyResult(
        config=config,
        patient_results=patient_uncertainty_results,
        cohort_summary=build_cohort_bootstrap_summary(patient_uncertainty_results),
        metadata={
            "random_state": config.random_state,
            "n_boot": config.n_boot,
            "preserve_domain_stratification": config.preserve_domain_stratification,
        },
    )


@dataclass(frozen=True)
class STRIDEFitConfig:
    """Configuration for the current deferred top-level STRIDE fit flow."""

    timepoint_order: tuple[str, ...] = ()
    recurrence_config: RecurrenceConfig | None = None
    uncertainty_config: PatientBootstrapConfig | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_labels = tuple(
            _require_nonempty_identifier(label, field_name="timepoint_order label")
            for label in self.timepoint_order
        )
        if len(set(normalized_labels)) != len(normalized_labels):
            raise ContractError("STRIDEFitConfig.timepoint_order must not contain duplicates")
        if self.uncertainty_config is not None and not isinstance(
            self.uncertainty_config,
            PatientBootstrapConfig,
        ):
            raise ContractError(
                "STRIDEFitConfig.uncertainty_config must be a PatientBootstrapConfig when provided"
            )


@dataclass(frozen=True)
class BridgeObservationGroup:
    """One ordered-side/timepoint group of observations for a patient bridge."""

    group_label: str
    observations: tuple[FovObservation, ...]
    observations_by_domain: Mapping[str, tuple[FovObservation, ...]]

    def __post_init__(self) -> None:
        validate_bridge_observation_group(self)


@dataclass(frozen=True)
class PatientBridgeInput:
    """Canonical per-patient bridge input bundle for future STRIDE fitting."""

    patient_id: str
    ordered_group_labels: tuple[str, ...]
    groups: tuple[BridgeObservationGroup, ...]
    n_observations_by_group: Mapping[str, int]
    n_observations_by_domain: Mapping[str, int]
    n_observations_by_group_and_domain: Mapping[str, Mapping[str, int]]
    state_basis: StateBasis | None = None
    geometry: StateGeometry | None = None
    mass_mode: str = "uniform"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_patient_bridge_input(self)

    @property
    def observations(self) -> tuple[FovObservation, ...]:
        """Return the flattened ordered observation sequence across groups."""
        return tuple(
            observation
            for group in self.groups
            for observation in group.observations
        )

    @property
    def groups_by_label(self) -> dict[str, BridgeObservationGroup]:
        """Return the grouped observations keyed by their declared label."""
        return {group.group_label: group for group in self.groups}


def validate_bridge_observation_group(group: BridgeObservationGroup) -> None:
    """Validate one ordered-side/timepoint observation group."""
    group_label = _require_nonempty_identifier(group.group_label, field_name="group_label")
    if len(group.observations) == 0:
        raise ContractError("BridgeObservationGroup.observations must be non-empty")
    if len(group.observations_by_domain) == 0:
        raise ContractError("BridgeObservationGroup.observations_by_domain must be non-empty")

    n_states: int | None = None
    observed_counter: Counter[tuple[str, str, str, str]] = Counter()
    observed_domain_counter: Counter[str] = Counter()
    for observation in group.observations:
        validate_fov_observation(observation)
        if observation.timepoint != group_label:
            raise ContractError(
                "All BridgeObservationGroup.observations must share the declared group_label"
            )
        if observation.domain_label is None:
            raise ContractError("BridgeObservationGroup observations must declare domain_label")

        observed_counter[_observation_key(observation)] += 1
        observed_domain_counter[str(observation.domain_label)] += 1
        current_n_states = int(np.asarray(observation.community_composition, dtype=float).shape[0])
        if n_states is None:
            n_states = current_n_states
        elif current_n_states != n_states:
            raise ContractError(
                "BridgeObservationGroup observations must share one K-state axis size"
            )

    grouped_counter: Counter[tuple[str, str, str, str]] = Counter()
    grouped_domain_counter: Counter[str] = Counter()
    for domain_label, domain_observations in group.observations_by_domain.items():
        normalized_domain = _require_nonempty_identifier(domain_label, field_name="domain_label")
        if len(domain_observations) == 0:
            raise ContractError(
                "BridgeObservationGroup.observations_by_domain entries must be non-empty"
            )
        for observation in domain_observations:
            validate_fov_observation(observation)
            if observation.timepoint != group_label:
                raise ContractError(
                    "Domain-grouped observations must share the BridgeObservationGroup.group_label"
                )
            if observation.domain_label != normalized_domain:
                raise ContractError(
                    "BridgeObservationGroup.observations_by_domain keys must match observation.domain_label"
                )
            grouped_counter[_observation_key(observation)] += 1
            grouped_domain_counter[normalized_domain] += 1

    if grouped_counter != observed_counter:
        raise ContractError(
            "BridgeObservationGroup.observations_by_domain must partition the declared observations"
        )
    if grouped_domain_counter != observed_domain_counter:
        raise ContractError(
            "BridgeObservationGroup.observations_by_domain counts must match observed domain counts"
        )


def validate_patient_bridge_input(patient_input: PatientBridgeInput) -> None:
    """Validate one canonical per-patient bridge input bundle."""
    patient_id = _require_nonempty_identifier(patient_input.patient_id, field_name="patient_id")
    if patient_input.mass_mode != "uniform":
        raise ContractError("PatientBridgeInput.mass_mode must be 'uniform' in the current pass")
    if len(patient_input.groups) == 0:
        raise ContractError("PatientBridgeInput.groups must be non-empty")

    expected_group_labels = tuple(group.group_label for group in patient_input.groups)
    if patient_input.ordered_group_labels != expected_group_labels:
        raise ContractError(
            "PatientBridgeInput.ordered_group_labels must align with PatientBridgeInput.groups"
        )
    if len(set(patient_input.ordered_group_labels)) != len(patient_input.ordered_group_labels):
        raise ContractError("PatientBridgeInput.ordered_group_labels must not contain duplicates")

    counts_by_group: dict[str, int] = {}
    counts_by_domain: dict[str, int] = {}
    counts_by_group_and_domain: dict[str, dict[str, int]] = {}
    n_states: int | None = None
    for group in patient_input.groups:
        validate_bridge_observation_group(group)
        counts_by_group[group.group_label] = len(group.observations)
        group_domain_counts: dict[str, int] = {}

        for observation in group.observations:
            if observation.patient_id != patient_id:
                raise ContractError("PatientBridgeInput observations must belong to one patient_id")
            if observation.mass_mode != patient_input.mass_mode:
                raise ContractError(
                    "PatientBridgeInput observations must share PatientBridgeInput.mass_mode"
                )

            current_n_states = int(np.asarray(observation.community_composition, dtype=float).shape[0])
            if n_states is None:
                n_states = current_n_states
            elif current_n_states != n_states:
                raise ContractError("PatientBridgeInput observations must share one K-state axis size")

            domain_label = str(observation.domain_label)
            counts_by_domain[domain_label] = counts_by_domain.get(domain_label, 0) + 1
            group_domain_counts[domain_label] = group_domain_counts.get(domain_label, 0) + 1

        counts_by_group_and_domain[group.group_label] = group_domain_counts

    if dict(patient_input.n_observations_by_group) != counts_by_group:
        raise ContractError(
            "PatientBridgeInput.n_observations_by_group does not match grouped observations"
        )
    if dict(patient_input.n_observations_by_domain) != counts_by_domain:
        raise ContractError(
            "PatientBridgeInput.n_observations_by_domain does not match grouped observations"
        )
    if _normalize_nested_counts(patient_input.n_observations_by_group_and_domain) != counts_by_group_and_domain:
        raise ContractError(
            "PatientBridgeInput.n_observations_by_group_and_domain does not match grouped observations"
        )

    if n_states is None:
        raise ContractError("PatientBridgeInput must contain at least one observation")

    if patient_input.state_basis is not None:
        validate_state_basis(patient_input.state_basis)
        if patient_input.state_basis.n_states != n_states:
            raise ContractError(
                "PatientBridgeInput.state_basis must align to the observation shared K-state axis"
            )

    if patient_input.geometry is not None:
        geometry_shape = np.asarray(patient_input.geometry.cost_matrix, dtype=float).shape
        if len(geometry_shape) != 2 or geometry_shape[0] != geometry_shape[1]:
            raise ContractError("PatientBridgeInput.geometry.cost_matrix must be square")
        if geometry_shape[0] != n_states:
            raise ContractError(
                "PatientBridgeInput.geometry must align to the observation shared K-state axis"
            )
        if len(patient_input.geometry.state_ids) != n_states:
            raise ContractError("PatientBridgeInput.geometry.state_ids must align to the shared K-state axis")

    if patient_input.state_basis is not None and patient_input.geometry is not None:
        if patient_input.state_basis.resolved_state_ids != tuple(patient_input.geometry.state_ids):
            raise ContractError(
                "PatientBridgeInput.state_basis and geometry must share the same declared state_ids"
            )


def build_patient_bridge_inputs(
    observations: tuple[FovObservation, ...] | list[FovObservation],
    *,
    state_basis: StateBasis | None = None,
    geometry: StateGeometry | None = None,
    config: STRIDEFitConfig | None = None,
) -> tuple[PatientBridgeInput, ...]:
    """Group canonical observations into per-patient bridge input bundles."""
    resolved_config = config or STRIDEFitConfig()
    observation_sequence = tuple(observations)
    if len(observation_sequence) == 0:
        raise ContractError("observations must contain at least one FovObservation")

    patient_order: list[str] = []
    observed_group_order: dict[str, list[str]] = {}
    patient_groups: dict[str, dict[str, list[FovObservation]]] = {}

    for observation in observation_sequence:
        validate_fov_observation(observation)
        patient_id = _require_nonempty_identifier(observation.patient_id, field_name="patient_id")
        group_label = _require_nonempty_identifier(observation.timepoint, field_name="timepoint")
        _require_nonempty_identifier(observation.fov_id, field_name="fov_id")
        if observation.domain_label is None:
            raise ContractError("All bridge observations must declare domain_label")

        if patient_id not in patient_groups:
            patient_order.append(patient_id)
            patient_groups[patient_id] = {}
            observed_group_order[patient_id] = []
        if group_label not in patient_groups[patient_id]:
            patient_groups[patient_id][group_label] = []
            observed_group_order[patient_id].append(group_label)
        patient_groups[patient_id][group_label].append(observation)

    bridge_inputs: list[PatientBridgeInput] = []
    for patient_id in patient_order:
        observed_labels = tuple(observed_group_order[patient_id])
        ordered_labels = _resolve_ordered_group_labels(
            observed_labels,
            declared_order=resolved_config.timepoint_order,
        )

        groups: list[BridgeObservationGroup] = []
        counts_by_group: dict[str, int] = {}
        counts_by_domain: dict[str, int] = {}
        counts_by_group_and_domain: dict[str, dict[str, int]] = {}
        for group_label in ordered_labels:
            group_observations = tuple(patient_groups[patient_id][group_label])
            observations_by_domain_lists: dict[str, list[FovObservation]] = {}
            for observation in group_observations:
                domain_label = str(observation.domain_label)
                observations_by_domain_lists.setdefault(domain_label, []).append(observation)
                counts_by_domain[domain_label] = counts_by_domain.get(domain_label, 0) + 1

            observations_by_domain = {
                domain_label: tuple(domain_observations)
                for domain_label, domain_observations in observations_by_domain_lists.items()
            }
            groups.append(
                BridgeObservationGroup(
                    group_label=group_label,
                    observations=group_observations,
                    observations_by_domain=observations_by_domain,
                )
            )
            counts_by_group[group_label] = len(group_observations)
            counts_by_group_and_domain[group_label] = {
                domain_label: len(domain_observations)
                for domain_label, domain_observations in observations_by_domain.items()
            }

        bridge_inputs.append(
            PatientBridgeInput(
                patient_id=patient_id,
                ordered_group_labels=ordered_labels,
                groups=tuple(groups),
                n_observations_by_group=counts_by_group,
                n_observations_by_domain=counts_by_domain,
                n_observations_by_group_and_domain=counts_by_group_and_domain,
                state_basis=state_basis,
                geometry=geometry,
                mass_mode="uniform",
                metadata={
                    "grouping_axis": "timepoint",
                    "declared_timepoint_order": tuple(resolved_config.timepoint_order),
                },
            )
        )

    return tuple(bridge_inputs)


def run_stride_fit(
    observations: tuple[FovObservation, ...] | list[FovObservation],
    *,
    state_basis: StateBasis | None = None,
    geometry: StateGeometry | None = None,
    config: STRIDEFitConfig | None = None,
) -> STRIDEFitResult:
    """Run the canonical STRIDE fit orchestration with minimal bridge realization."""
    resolved_config = config or STRIDEFitConfig()
    patient_inputs = build_patient_bridge_inputs(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        config=resolved_config,
    )

    patient_results: list[PatientBridgeResult] = []
    for patient_input in patient_inputs:
        is_supported, defer_reason, message = _evaluate_patient_bridge_support(patient_input)
        if is_supported:
            patient_results.append(_build_realized_patient_bridge_result(patient_input))
            continue

        patient_results.append(
            _build_deferred_patient_bridge_result(
                patient_input,
                defer_reason=str(defer_reason),
                message=str(message),
            )
        )

    patient_results_tuple = tuple(patient_results)
    recurrence_config = resolved_config.recurrence_config or RecurrenceConfig()
    recurrence = RecurrenceResult(
        patient_ids=tuple(patient_input.patient_id for patient_input in patient_inputs),
        families=(),
        fit_status="deferred",
        parameters=RecurrenceParameters(
            basis_dim=recurrence_config.basis_dim,
            loadings=None,
            metadata={"mode": recurrence_config.mode},
        ),
        embeddings=tuple(
            PatientRecurrenceEmbedding(
                patient_id=patient_input.patient_id,
                coordinates=np.full(recurrence_config.basis_dim, np.nan, dtype=float),
                fit_status="deferred",
            )
            for patient_input in patient_inputs
        ),
        metadata={
            "mode": recurrence_config.mode,
            "message": (
                "Canonical cohort-level recurrence remains deferred even when patient "
                "bridge estimation is available for supported patients."
            ),
            **dict(recurrence_config.metadata),
        },
    )

    patient_status_counts = _count_patient_statuses(patient_results_tuple)
    any_realized_bridges = any(patient_result.is_ok for patient_result in patient_results_tuple)
    uncertainty = (
        _build_stride_bootstrap_uncertainty(
            patient_inputs,
            patient_results_tuple,
            config=resolved_config.uncertainty_config,
        )
        if resolved_config.uncertainty_config is not None
        else None
    )
    metadata = dict(resolved_config.metadata)
    if resolved_config.timepoint_order:
        metadata["timepoint_order"] = tuple(resolved_config.timepoint_order)

    return STRIDEFitResult(
        patient_inputs=patient_inputs,
        patient_results=patient_results_tuple,
        recurrence=recurrence,
        fit_status="deferred",
        summaries={
            "n_patients": len(patient_results_tuple),
            "patient_status_counts": patient_status_counts,
            "recurrence_fit_status": recurrence.fit_status,
            **(
                {
                    "uncertainty_status": uncertainty.cohort_summary.uncertainty_status,
                    "n_patients_with_uncertainty": uncertainty.cohort_summary.n_realized_patients,
                }
                if uncertainty is not None
                else {}
            ),
        },
        diagnostics={
            "mode": (
                "patient_bridge_realized_recurrence_deferred"
                if any_realized_bridges
                else "deferred"
            ),
            "message": (
                "Supported patients may receive realized patient bridge estimates while "
                "cohort-level recurrence remains deferred."
                if any_realized_bridges
                else (
                    "No patients satisfied the minimal supported bridge-fitting case, "
                    "so patient bridge estimates remain deferred while cohort-level "
                    "recurrence also remains deferred."
                )
            ),
            "patient_status_counts": patient_status_counts,
            **(
                {"uncertainty_status": uncertainty.cohort_summary.uncertainty_status}
                if uncertainty is not None
                else {}
            ),
        },
        uncertainty=uncertainty,
        metadata=metadata,
    )


__all__ = [
    "BridgeObservationGroup",
    "PatientBridgeInput",
    "STRIDEFitConfig",
    "build_patient_bridge_inputs",
    "run_stride_fit",
    "validate_bridge_observation_group",
    "validate_patient_bridge_input",
]
