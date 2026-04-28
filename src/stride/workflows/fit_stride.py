"""Canonical STRIDE fit orchestration with benchmark-aware ablation controls.

Role:
    Provide the canonical patient-plus-cohort STRIDE fit path together with the
    compatibility proxy initializer and the benchmark controls consumed by Task
    A Block 3.

Local boundary:
    - This module owns bridge-input validation, proxy initialization, canonical
      cohort shrinkage, and benchmark-mode control of the fit output.
    - It does not define Task A routing, semisynthetic generation, or Block 3
      scoring rules.
    - It must preserve the distinction between the approximate proxy path and
      the canonical full path.

Primary contents:
    - Canonical bridge-input dataclasses and validators.
    - Proxy patient bridge fitting and canonical full STRIDE refinement.
    - Benchmark-mode controls for `reference`, `open_channel_ablation`, and
      `cohort_ablation`.

Why this module exists:
    Task A Block 3 needs ablations that change real inference outputs without
    inventing a task-local fit engine. Centralizing those controls here makes
    the ablation semantics auditable at the same layer where canonical STRIDE
    actually applies them.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np

from ..basis.contracts import StateBasis, validate_state_basis
from ..errors import ContractError
from ..geometry.state_geometry import StateGeometry
from ..latent.operators import PatientRelationAudit, initialize_patient_relation
from ..latent.recurrence import (
    RecurrenceConfig,
    build_deferred_recurrence_result,
    estimate_recurrence,
)
from ..objectives import LossBreakdown, LossWeights, aggregate_loss_breakdowns, evaluate_loss_bundle
from ..observation.contracts import (
    DomainStratifiedMeasure,
    FovObservation,
    ObservationDiscrepancyConfig,
    validate_fov_observation,
)
from ..observation.discrepancy import build_observation_kernels, match_observation_clouds
from ..observation.measures import build_domain_stratified_measure
from ..outputs.fit_result import PatientBridgeResult, STRIDEFitResult
from ..outputs.uncertainty import (
    PatientBootstrapConfig,
    PatientBootstrapUncertaintyResult,
    STRIDEBootstrapUncertaintyResult,
    build_cohort_bootstrap_summary,
    summarize_bootstrap_array,
)
from ..settings.runtime import RuntimeSettings


def _require_nonempty_identifier(value: str, *, field_name: str) -> str:
    """Normalize one identifier field and require non-empty string content."""

    normalized = str(value).strip()
    if normalized == "":
        raise ContractError(f"{field_name} must be a non-empty string")
    return normalized


def _observation_key(observation: FovObservation) -> tuple[str, str, str, str]:
    """Build a stable observation identity key for grouping and validation."""

    return (
        str(observation.patient_id),
        str(observation.timepoint),
        str(observation.fov_id),
        str(observation.domain_label),
    )


def _normalize_nested_counts(
    counts: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    """Convert nested count mappings into plain serializable dictionaries."""

    return {
        str(group_label): {str(domain_label): int(count) for domain_label, count in domain_counts.items()}
        for group_label, domain_counts in counts.items()
    }


def _count_patient_statuses(results: tuple[PatientBridgeResult, ...]) -> dict[str, int]:
    """Count patient fit statuses for summaries and diagnostics."""

    counts: dict[str, int] = {}
    for result in results:
        counts[result.fit_status] = counts.get(result.fit_status, 0) + 1
    return counts


def _count_uncertainty_statuses(
    statuses: tuple[str, ...],
) -> dict[str, int]:
    """Count uncertainty statuses into the canonical ok/deferred/failed map."""

    counts = Counter(str(status) for status in statuses)
    return {
        "ok": int(counts.get("ok", 0)),
        "deferred": int(counts.get("deferred", 0)),
        "failed": int(counts.get("failed", 0)),
    }


def _count_realized_patients(results: Sequence[PatientBridgeResult]) -> int:
    """Count patients with realized bridge fits."""

    return int(sum(int(result.is_ok) for result in results))


def _reconstruct_post_burden(
    A: np.ndarray,
    e: np.ndarray,
    mu_minus: np.ndarray,
) -> np.ndarray:
    """Reconstruct post-side burden from `A`, `e`, and the matched source mass."""

    transition_burden = np.sum(
        np.asarray(A, dtype=float) * np.asarray(mu_minus, dtype=float)[:, None],
        axis=0,
        dtype=float,
    )
    emergence_burden = np.asarray(e, dtype=float) * float(
        np.sum(np.asarray(mu_minus, dtype=float), dtype=float)
    )
    return transition_burden + emergence_burden


def _build_canonical_bridge_auxiliary(
    *,
    local_result: PatientBridgeResult,
    final_A: np.ndarray,
    final_d: np.ndarray,
    final_e: np.ndarray,
    mu_minus: np.ndarray,
    model_implied_mu_plus: np.ndarray,
    relation_refinement_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the auxiliary payload carried by canonical full patient results."""

    proxy_auxiliary = dict(local_result.auxiliary)
    canonical_auxiliary: dict[str, Any] = {}
    emergence_scale = float(np.sum(mu_minus, dtype=float))
    known_proxy_fields = (
        "matched_transition_burden",
        "raw_matched_transition_burden",
        "source_unmatched_burden",
        "target_unmatched_burden",
    )

    for field_name, value in proxy_auxiliary.items():
        if field_name in known_proxy_fields:
            canonical_auxiliary[f"proxy_initializer_{field_name}"] = value
            continue
        canonical_auxiliary[field_name] = value

    canonical_auxiliary.update(
        {
            "matched_transition_burden": np.asarray(final_A, dtype=float) * mu_minus[:, None],
            "raw_matched_transition_burden": np.asarray(final_A, dtype=float) * mu_minus[:, None],
            "source_unmatched_burden": np.asarray(final_d, dtype=float) * mu_minus,
            "target_unmatched_burden": np.asarray(final_e, dtype=float) * emergence_scale,
            "local_initializer_A": np.asarray(local_result.A, dtype=float),
            "local_initializer_d": np.asarray(local_result.d, dtype=float),
            "local_initializer_e": np.asarray(local_result.e, dtype=float),
            "model_implied_mu_plus": np.asarray(model_implied_mu_plus, dtype=float),
        }
    )
    if relation_refinement_metadata is not None:
        canonical_auxiliary.update(dict(relation_refinement_metadata))
    return canonical_auxiliary


def _shrink_relation_to_template(
    relation: PatientBridgeResult,
    *,
    template_A: np.ndarray,
    template_d: np.ndarray,
    template_e: np.ndarray,
    shrinkage_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shrink one realized patient relation toward the cohort template."""

    alpha = float(np.clip(shrinkage_weight, 0.0, 1.0))
    local_A = np.asarray(relation.A, dtype=float)
    local_d = np.asarray(relation.d, dtype=float)
    local_e = np.asarray(relation.e, dtype=float)
    shrunk_A = ((1.0 - alpha) * local_A) + (alpha * np.asarray(template_A, dtype=float))
    shrunk_d = ((1.0 - alpha) * local_d) + (alpha * np.asarray(template_d, dtype=float))
    shrunk_e = ((1.0 - alpha) * local_e) + (alpha * np.asarray(template_e, dtype=float))
    return shrunk_A, shrunk_d, shrunk_e


def _apply_benchmark_mode_to_relation(
    *,
    benchmark_mode: str,
    A: np.ndarray,
    d: np.ndarray,
    e: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize the realized relation after benchmark controls were applied upstream.

    Purpose:
        Return canonical relation arrays after any bridge-stage or
        shrinkage-stage benchmark controls have already taken effect.

    Inputs:
        benchmark_mode: One of `reference`, `open_channel_ablation`, or
            `cohort_ablation`.
        A/d/e: Canonical relation components after upstream inference controls.

    Returns:
        The normalized `(A, d, e)` tuple exposed to callers.

    Core flow:
        1. Accept the already-controlled relation from the upstream bridge and
           cohort stages.
        2. Normalize each array to float dtype without rewriting the inferred
           open channels.
        3. Preserve identical output semantics across benchmark modes at this
           projection layer.
    """
    del benchmark_mode
    return (
        np.asarray(A, dtype=float),
        np.asarray(d, dtype=float),
        np.asarray(e, dtype=float),
    )


def _resolve_bridge_match_penalty(*, benchmark_mode: str) -> float:
    """Resolve the observation-layer match penalty used by one benchmark mode."""

    if benchmark_mode == "open_channel_ablation":
        return float(_OPEN_CHANNEL_ABLATION_MATCH_PENALTY)
    return float(_DEFAULT_BRIDGE_MATCH_PENALTY)


def _resolve_benchmark_controls(
    config: "STRIDEFitConfig",
) -> tuple[LossWeights, float]:
    """Resolve objective-weight and shrinkage controls for one benchmark mode.

    This helper changes the optimization surface itself, not just the labels on
    the returned fit object.
    """
    objective_weights = config.objective_weights
    shrinkage_weight = float(config.cohort_shrinkage_weight)
    if config.benchmark_mode == "open_channel_ablation":
        objective_weights = replace(
            objective_weights,
            open_relation=0.0,
            open_channel_control=(
                0.0 if objective_weights.open_channel_control is not None else None
            ),
        )
    elif config.benchmark_mode == "cohort_ablation":
        objective_weights = replace(objective_weights, cohort_recurrence=0.0)
        shrinkage_weight = 0.0
    return objective_weights, shrinkage_weight


def _build_patient_loss_breakdown(
    patient_input: "PatientBridgeInput",
    *,
    local_result: PatientBridgeResult,
    final_A: np.ndarray,
    final_d: np.ndarray,
    final_e: np.ndarray,
    template_A: np.ndarray | None,
    template_d: np.ndarray | None,
    template_e: np.ndarray | None,
    weights: LossWeights,
) -> LossBreakdown:
    """Build the patient-level objective breakdown for a canonicalized result."""

    pre_group, post_group = patient_input.groups
    observed_pre = _mean_group_composition(pre_group)
    observed_post = _mean_group_composition(post_group)
    observed = np.concatenate([observed_pre, observed_post]).astype(float, copy=False)
    mu_minus = np.asarray(local_result.mu_minus, dtype=float)
    reconstructed_post = _reconstruct_post_burden(final_A, final_e, mu_minus)
    reconstructed = np.concatenate(
        [mu_minus, reconstructed_post],
    ).astype(float, copy=False)
    target_open_total = float(
        np.sum(np.asarray(local_result.d, dtype=float), dtype=float)
        + np.sum(np.asarray(local_result.e, dtype=float), dtype=float)
    )
    return evaluate_loss_bundle(
        observed=observed,
        reconstructed=reconstructed,
        A=final_A,
        d=final_d,
        e=final_e,
        reference_A=np.asarray(local_result.A, dtype=float),
        reference_d=np.asarray(local_result.d, dtype=float),
        reference_e=np.asarray(local_result.e, dtype=float),
        template_A=template_A,
        template_d=template_d,
        template_e=template_e,
        geometry_cost_matrix=(
            np.asarray(patient_input.geometry.cost_matrix, dtype=float)
            if patient_input.geometry is not None
            else None
        ),
        weights=weights,
        target_open_channel_total=target_open_total,
    )


def _project_relation_rows(A: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project relation rows back onto `sum_j A_ij + d_i = 1`."""

    projected_A = np.maximum(np.asarray(A, dtype=float), 0.0)
    projected_d = np.maximum(np.asarray(d, dtype=float).reshape(-1), 0.0)
    row_totals = np.sum(projected_A, axis=1, dtype=float) + projected_d
    for row_idx, row_total in enumerate(row_totals):
        if row_total <= 0.0:
            projected_A[row_idx, :] = 0.0
            projected_d[row_idx] = 1.0
            continue
        projected_A[row_idx, :] /= row_total
        projected_d[row_idx] /= row_total
    return projected_A, projected_d


def _pack_relation_variables(A: np.ndarray, d: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Pack relation arrays into one optimizer vector."""

    return np.concatenate(
        [
            np.asarray(A, dtype=float).reshape(-1),
            np.asarray(d, dtype=float).reshape(-1),
            np.asarray(e, dtype=float).reshape(-1),
        ]
    ).astype(float, copy=False)


def _unpack_relation_variables(
    values: np.ndarray,
    *,
    n_states: int,
    project_rows: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack one optimizer vector into relation arrays."""

    arr = np.asarray(values, dtype=float).reshape(-1)
    A_end = n_states * n_states
    d_end = A_end + n_states
    A = arr[:A_end].reshape(n_states, n_states)
    d = arr[A_end:d_end]
    e = arr[d_end : d_end + n_states]
    if project_rows:
        A, d = _project_relation_rows(A, d)
    return A, d, e


def _refine_relation_by_objective(
    patient_input: "PatientBridgeInput",
    *,
    local_result: PatientBridgeResult,
    initial_A: np.ndarray,
    initial_d: np.ndarray,
    initial_e: np.ndarray,
    template_A: np.ndarray | None,
    template_d: np.ndarray | None,
    template_e: np.ndarray | None,
    weights: LossWeights,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Actively refine `A/d/e` against the canonical objective when enabled."""

    initial_A, initial_d = _project_relation_rows(initial_A, initial_d)
    initial_e = np.maximum(np.asarray(initial_e, dtype=float), 0.0)
    n_states = int(initial_A.shape[0])
    initial_objective = _build_patient_loss_breakdown(
        patient_input,
        local_result=local_result,
        final_A=initial_A,
        final_d=initial_d,
        final_e=initial_e,
        template_A=template_A,
        template_d=template_d,
        template_e=template_e,
        weights=weights,
    )
    metadata: dict[str, Any] = {
        "relation_refinement_enabled": True,
        "relation_refinement_initial_objective": float(initial_objective.total),
        "relation_refinement_initial_geometry_structure": float(initial_objective.geometry_structure),
    }

    def _locality_candidate() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        candidate_A = np.zeros_like(initial_A, dtype=float)
        for row_idx in range(n_states):
            candidate_A[row_idx, row_idx] = float(np.sum(initial_A[row_idx], dtype=float))
        candidate_A, candidate_d = _project_relation_rows(candidate_A, initial_d)
        post_group = patient_input.groups[1]
        observed_post = _mean_group_composition(post_group)
        mu_minus = np.asarray(local_result.mu_minus, dtype=float)
        transition = np.sum(candidate_A * mu_minus[:, None], axis=0, dtype=float)
        emergence_scale = float(np.sum(mu_minus, dtype=float))
        if emergence_scale <= 0.0:
            candidate_e = initial_e
        else:
            candidate_e = np.maximum(observed_post - transition, 0.0) / emergence_scale
        return candidate_A, candidate_d, candidate_e

    def _candidate_objective(
        candidate_A: np.ndarray,
        candidate_d: np.ndarray,
        candidate_e: np.ndarray,
    ) -> LossBreakdown:
        return _build_patient_loss_breakdown(
            patient_input,
            local_result=local_result,
            final_A=candidate_A,
            final_d=candidate_d,
            final_e=candidate_e,
            template_A=template_A,
            template_d=template_d,
            template_e=template_e,
            weights=weights,
        )

    def _changed(
        candidate_A: np.ndarray,
        candidate_d: np.ndarray,
        candidate_e: np.ndarray,
    ) -> bool:
        return bool(
            np.max(np.abs(candidate_A - initial_A)) > 1e-8
            or np.max(np.abs(candidate_d - initial_d)) > 1e-8
            or np.max(np.abs(candidate_e - initial_e)) > 1e-8
        )

    try:
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover - scipy is a declared dependency
        metadata.update(
            {
                "relation_refinement_status": "dependency_unavailable",
                "relation_refinement_message": str(exc),
                "relation_refinement_final_objective": float(initial_objective.total),
                "relation_refinement_changed_Ade": False,
            }
        )
        return initial_A, initial_d, initial_e, metadata

    interior_eps = 1e-6
    start_A, start_d = _project_relation_rows(
        initial_A + interior_eps,
        initial_d + interior_eps,
    )
    start_e = np.maximum(initial_e, interior_eps)
    x0 = _pack_relation_variables(start_A, start_d, start_e)

    def _objective(values: np.ndarray) -> float:
        candidate_A, candidate_d, candidate_e = _unpack_relation_variables(
            values,
            n_states=n_states,
            project_rows=False,
        )
        return float(
            _build_patient_loss_breakdown(
                patient_input,
                local_result=local_result,
                final_A=candidate_A,
                final_d=candidate_d,
                final_e=candidate_e,
                template_A=template_A,
                template_d=template_d,
                template_e=template_e,
                weights=weights,
            ).total
        )

    constraints = tuple(
        {
            "type": "eq",
            "fun": (
                lambda values, row_idx=row_idx: float(
                    np.sum(values[row_idx * n_states : (row_idx + 1) * n_states], dtype=float)
                    + values[n_states * n_states + row_idx]
                    - 1.0
                )
            ),
        }
        for row_idx in range(n_states)
    )
    result = minimize(
        _objective,
        x0,
        method="SLSQP",
        bounds=[(0.0, None)] * len(x0),
        constraints=constraints,
        options={"maxiter": int(max_iter), "ftol": float(tol), "disp": False},
    )
    if not bool(result.success) or not np.isfinite(float(result.fun)):
        fallback_A, fallback_d, fallback_e = _locality_candidate()
        fallback_objective = _candidate_objective(fallback_A, fallback_d, fallback_e)
        if float(fallback_objective.total) < float(initial_objective.total):
            metadata.update(
                {
                    "relation_refinement_status": "ok",
                    "relation_refinement_message": (
                        "Accepted objective-improving locality candidate after optimizer did not converge: "
                        f"{result.message}"
                    ),
                    "relation_refinement_n_iter": int(getattr(result, "nit", 0)),
                    "relation_refinement_final_objective": float(fallback_objective.total),
                    "relation_refinement_final_geometry_structure": float(
                        fallback_objective.geometry_structure
                    ),
                    "relation_refinement_changed_Ade": _changed(
                        fallback_A,
                        fallback_d,
                        fallback_e,
                    ),
                }
            )
            return fallback_A, fallback_d, fallback_e, metadata
        metadata.update(
            {
                "relation_refinement_status": "failed",
                "relation_refinement_message": str(result.message),
                "relation_refinement_final_objective": float(initial_objective.total),
                "relation_refinement_changed_Ade": False,
            }
        )
        return initial_A, initial_d, initial_e, metadata

    refined_A, refined_d, refined_e = _unpack_relation_variables(
        np.asarray(result.x, dtype=float),
        n_states=n_states,
        project_rows=True,
    )
    refined_e = np.maximum(refined_e, 0.0)
    final_objective = _build_patient_loss_breakdown(
        patient_input,
        local_result=local_result,
        final_A=refined_A,
        final_d=refined_d,
        final_e=refined_e,
        template_A=template_A,
        template_d=template_d,
        template_e=template_e,
        weights=weights,
    )
    if float(final_objective.total) > float(initial_objective.total):
        fallback_A, fallback_d, fallback_e = _locality_candidate()
        fallback_objective = _candidate_objective(fallback_A, fallback_d, fallback_e)
        if float(fallback_objective.total) < float(final_objective.total):
            refined_A, refined_d, refined_e = fallback_A, fallback_d, fallback_e
            final_objective = fallback_objective
    metadata.update(
        {
            "relation_refinement_status": "ok",
            "relation_refinement_message": str(result.message),
            "relation_refinement_n_iter": int(getattr(result, "nit", 0)),
            "relation_refinement_final_objective": float(final_objective.total),
            "relation_refinement_final_geometry_structure": float(final_objective.geometry_structure),
            "relation_refinement_changed_Ade": _changed(refined_A, refined_d, refined_e),
        }
    )
    return refined_A, refined_d, refined_e, metadata


def _canonicalize_proxy_patient_result(
    patient_input: "PatientBridgeInput",
    *,
    local_result: PatientBridgeResult,
    objective_weights: LossWeights,
    benchmark_mode: str,
    template_A: np.ndarray | None = None,
    template_d: np.ndarray | None = None,
    template_e: np.ndarray | None = None,
    cohort_fit_status: str,
    shrinkage_weight: float,
    enable_relation_refinement: bool,
    relation_refinement_max_iter: int,
    relation_refinement_tol: float,
) -> PatientBridgeResult:
    """Project a proxy-initialized patient result onto the canonical full path.

    Purpose:
        Combine the local proxy initializer with cohort recurrence shrinkage and
        benchmark-mode controls to produce the patient result returned by
        canonical full STRIDE.

    Core flow:
        1. Preserve deferred/failed local statuses without fabricating new
           patient relations.
        2. Optionally shrink the realized local relation toward the cohort
           template.
        3. Apply benchmark-mode controls that alter the exposed output object.
        4. Rebuild diagnostics, auxiliary payloads, and loss breakdowns for the
           canonical result.
    """
    if local_result.fit_status != "ok":
        local_message = str(
            local_result.diagnostics.get(
                "message",
                "Canonical full STRIDE preserved the explicit deferred/failed patient "
                "status from the proxy initializer.",
            )
        )
        return PatientBridgeResult(
            patient_id=local_result.patient_id,
            fit_status=local_result.fit_status,
            state_ids=local_result.state_ids,
            audit=local_result.audit,
            diagnostics={
                **dict(local_result.diagnostics),
                "local_fit_status": local_result.fit_status,
                "cohort_fit_status": cohort_fit_status,
                "message": local_message,
                "proxy_initializer_message": local_message,
                "canonical_context_message": (
                    "Canonical full STRIDE preserved the explicit deferred/failed patient "
                    "status from the proxy initializer."
                ),
            },
            auxiliary=dict(local_result.auxiliary),
            implementation_tier="canonical_full",
        )

    final_A, final_d, final_e = (
        _shrink_relation_to_template(
            local_result,
            template_A=template_A,
            template_d=template_d,
            template_e=template_e,
            shrinkage_weight=shrinkage_weight,
        )
        if template_A is not None and template_d is not None and template_e is not None
        else (
            np.asarray(local_result.A, dtype=float),
            np.asarray(local_result.d, dtype=float),
            np.asarray(local_result.e, dtype=float),
        )
    )
    final_A, final_d, final_e = _apply_benchmark_mode_to_relation(
        benchmark_mode=benchmark_mode,
        A=final_A,
        d=final_d,
        e=final_e,
    )
    refinement_metadata: dict[str, Any] = {
        "relation_refinement_enabled": False,
        "relation_refinement_status": "disabled",
        "relation_refinement_changed_Ade": False,
    }
    if enable_relation_refinement:
        final_A, final_d, final_e, refinement_metadata = _refine_relation_by_objective(
            patient_input,
            local_result=local_result,
            initial_A=final_A,
            initial_d=final_d,
            initial_e=final_e,
            template_A=template_A,
            template_d=template_d,
            template_e=template_e,
            weights=objective_weights,
            max_iter=relation_refinement_max_iter,
            tol=relation_refinement_tol,
        )
    objective = _build_patient_loss_breakdown(
        patient_input,
        local_result=local_result,
        final_A=final_A,
        final_d=final_d,
        final_e=final_e,
        template_A=template_A,
        template_d=template_d,
        template_e=template_e,
        weights=objective_weights,
    )
    if not enable_relation_refinement:
        refinement_metadata.update(
            {
                "relation_refinement_initial_objective": float(objective.total),
                "relation_refinement_final_objective": float(objective.total),
                "relation_refinement_initial_geometry_structure": float(objective.geometry_structure),
                "relation_refinement_final_geometry_structure": float(objective.geometry_structure),
            }
        )
    mu_minus = (
        np.asarray(local_result.mu_minus, dtype=float)
        if local_result.mu_minus is not None
        else np.sum(np.asarray(final_A, dtype=float), axis=1, dtype=float) + np.asarray(final_d, dtype=float)
    )
    model_implied_mu_plus = _reconstruct_post_burden(final_A, final_e, mu_minus)
    resolved_mu_plus = (
        np.asarray(local_result.mu_plus, dtype=float)
        if local_result.mu_plus is not None
        else model_implied_mu_plus
    )
    template_shrinkage_applied = bool(
        template_A is not None and shrinkage_weight > 0.0 and cohort_fit_status == "ok"
    )
    return PatientBridgeResult(
        patient_id=local_result.patient_id,
        fit_status="ok",
        A=final_A,
        d=final_d,
        e=final_e,
        mu_minus=mu_minus,
        mu_plus=resolved_mu_plus,
        state_ids=local_result.state_ids,
        audit=local_result.audit,
        diagnostics={
            **dict(local_result.diagnostics),
            "local_fit_status": local_result.fit_status,
            "cohort_fit_status": cohort_fit_status,
            "template_shrinkage_applied": template_shrinkage_applied,
            "benchmark_mode": benchmark_mode,
            **refinement_metadata,
            "message": (
                "Canonical full STRIDE combined the proxy local initializer with the "
                "current cohort-level recurrence template."
                if template_shrinkage_applied
                else (
                    "Canonical full STRIDE preserved the local patient initializer because "
                    "cohort shrinkage was not applied."
                )
            ),
        },
        auxiliary=_build_canonical_bridge_auxiliary(
            local_result=local_result,
            final_A=final_A,
            final_d=final_d,
            final_e=final_e,
            mu_minus=mu_minus,
            model_implied_mu_plus=model_implied_mu_plus,
            relation_refinement_metadata=refinement_metadata,
        ),
        implementation_tier="canonical_full",
        objective=objective,
    )


def _align_recurrence_to_all_patients(
    patient_ids: tuple[str, ...],
    recurrence_result: object,
    *,
    basis_dim: int,
) -> object:
    patient_id_to_embedding = {
        str(embedding.patient_id): embedding
        for embedding in recurrence_result.embeddings
    }
    aligned_embeddings = []
    for patient_id in patient_ids:
        embedding = patient_id_to_embedding.get(str(patient_id))
        if embedding is not None:
            aligned_embeddings.append(embedding)
            continue
        aligned_embeddings.append(
            {
                "patient_id": str(patient_id),
                "coordinates": np.full(int(basis_dim), np.nan, dtype=float),
                "fit_status": "deferred",
            }
        )
    from ..latent.recurrence import PatientRecurrenceEmbedding, RecurrenceResult

    return RecurrenceResult(
        patient_ids=patient_ids,
        families=recurrence_result.families,
        fit_status=recurrence_result.fit_status,
        used_patient_ids=tuple(
            recurrence_result.used_patient_ids
            if recurrence_result.used_patient_ids
            else recurrence_result.patient_ids
        ),
        recurrence_unit=recurrence_result.recurrence_unit,
        parameters=recurrence_result.parameters,
        embeddings=tuple(
            embedding
            if isinstance(embedding, PatientRecurrenceEmbedding)
            else PatientRecurrenceEmbedding(
                patient_id=embedding["patient_id"],
                coordinates=embedding["coordinates"],
                fit_status=embedding["fit_status"],
            )
            for embedding in aligned_embeddings
        ),
        metadata={
            **dict(recurrence_result.metadata),
            "n_total_patients": len(patient_ids),
        },
    )


_MINIMAL_SUPPORTED_CASE = "two_group_uniform_patient_bridge"
_CANONICAL_BRIDGE_MODE = "observation_to_patient_bridge_v1"
_CANONICAL_BRIDGE_METHOD = "domain_stratified_cartesian_observation_discrepancy"
_PRIMARY_DEFER_REASON = "requires_exactly_two_ordered_groups"
_GEOMETRY_DEFER_REASON = "requires_shared_state_geometry"
_OBSERVATION_DEFER_REASON = "requires_supported_observation_discrepancy_rows"
_BRIDGE_PLAN_CHUNK_ELEMENTS = 250_000
_DEFAULT_BRIDGE_MATCH_PENALTY = 1.0
_OPEN_CHANNEL_ABLATION_MATCH_PENALTY = 50.0
_ALLOWED_BENCHMARK_MODES: tuple[str, ...] = (
    "reference",
    "open_channel_ablation",
    "cohort_ablation",
)
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


@dataclass(frozen=True)
class _DomainPairBridgeAccumulation:
    n_pairs: int
    pair_status_counts: Mapping[str, int]
    all_pairs_ok: bool
    matched_transition_burden: np.ndarray
    source_unmatched_burden: np.ndarray
    target_unmatched_burden: np.ndarray


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


def _bridge_discrepancy_config(runtime_settings: RuntimeSettings) -> ObservationDiscrepancyConfig:
    return replace(
        _BRIDGE_DISCREPANCY_CONFIG,
        runtime_settings=runtime_settings,
    )


def _clear_runtime_backend_cache(runtime_settings: RuntimeSettings) -> None:
    resolved_backend, resolved_device = runtime_settings.resolved_execution()
    if resolved_backend != "torch" or not str(resolved_device).startswith("cuda"):
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _accumulate_domain_pair_bridge(
    pre_measure: DomainStratifiedMeasure,
    post_measure: DomainStratifiedMeasure,
    *,
    kernels: Sequence[np.ndarray],
    cfg: ObservationDiscrepancyConfig,
    match_penalty: float,
) -> _DomainPairBridgeAccumulation:
    pre_state = np.asarray(pre_measure.state_matrix, dtype=float)
    post_state = np.asarray(post_measure.state_matrix, dtype=float)
    if pre_state.ndim != 2 or post_state.ndim != 2:
        raise ContractError("Domain-stratified measures must expose [N, K] state matrices")
    if pre_state.shape[1] != post_state.shape[1]:
        raise ContractError("Domain-stratified measures must share one K-state axis")

    n_pre = int(pre_state.shape[0])
    n_post = int(post_state.shape[0])
    n_states = int(pre_state.shape[1])
    n_pairs = n_pre * n_post
    if n_pairs <= 0:
        raise ContractError("Domain-pair bridge accumulation requires at least one observation pair")

    chunk_rows = cfg.runtime_settings.resolved_plan_chunk_rows(
        n_states,
        fallback_plan_chunk_elements=_BRIDGE_PLAN_CHUNK_ELEMENTS,
    )
    pair_status_counts: Counter[str] = Counter()
    all_pairs_ok = True
    matched_transition_burden = np.zeros((n_states, n_states), dtype=float)
    source_unmatched_burden = np.zeros(n_states, dtype=float)
    target_unmatched_burden = np.zeros(n_states, dtype=float)

    def _accumulate_plan_chunk(plan_chunk: np.ndarray) -> None:
        nonlocal matched_transition_burden
        matched_transition_burden += np.sum(plan_chunk, axis=0, dtype=float)

    for start in range(0, n_pairs, chunk_rows):
        stop = min(start + chunk_rows, n_pairs)
        pair_index = np.arange(start, stop, dtype=np.int64)
        pre_idx, post_idx = np.divmod(pair_index, n_post)
        chunk_result = match_observation_clouds(
            state_mass_pre=np.asarray(pre_state[pre_idx], dtype=float),
            state_mass_post=np.asarray(post_state[post_idx], dtype=float),
            match_penalty=np.full(pair_index.size, float(match_penalty), dtype=float),
            kernels=kernels,
            cfg=cfg,
            plan_consumer=_accumulate_plan_chunk,
        )

        chunk_status = tuple(str(status) for status in chunk_result.status.tolist())
        pair_status_counts.update(chunk_status)
        if all(status == "ok" for status in chunk_status):
            source_unmatched_burden += np.sum(
                chunk_result.details["source_unmatched_mass_by_state"],
                axis=0,
                dtype=float,
            )
            target_unmatched_burden += np.sum(
                chunk_result.details["target_unmatched_mass_by_state"],
                axis=0,
                dtype=float,
            )
        else:
            all_pairs_ok = False

        del pair_index, pre_idx, post_idx, chunk_result
        _clear_runtime_backend_cache(cfg.runtime_settings)

    return _DomainPairBridgeAccumulation(
        n_pairs=n_pairs,
        pair_status_counts=dict(pair_status_counts),
        all_pairs_ok=all_pairs_ok,
        matched_transition_burden=matched_transition_burden,
        source_unmatched_burden=source_unmatched_burden,
        target_unmatched_burden=target_unmatched_burden,
    )


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
    *,
    runtime_settings: RuntimeSettings,
    benchmark_mode: str = "reference",
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
    discrepancy_cfg = _bridge_discrepancy_config(runtime_settings)
    discrepancy_kernels = build_observation_kernels(
        geometry.cost_matrix,
        discrepancy_cfg.eps_schedule,
        cost_scale=geometry.cost_scale,
    )
    effective_match_penalty = _resolve_bridge_match_penalty(benchmark_mode=benchmark_mode)

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
            domain_pair_summary = _accumulate_domain_pair_bridge(
                pre_measure,
                post_measure,
                kernels=discrepancy_kernels,
                cfg=discrepancy_cfg,
                match_penalty=effective_match_penalty,
            )

            domain_pair_statuses.append(
                {
                    "pre_domain_label": str(pre_domain_label),
                    "post_domain_label": str(post_domain_label),
                    "n_pairs": domain_pair_summary.n_pairs,
                    "pair_status_counts": dict(domain_pair_summary.pair_status_counts),
                }
            )
            if not domain_pair_summary.all_pairs_ok:
                return _build_deferred_patient_bridge_result(
                    patient_input,
                    defer_reason=_OBSERVATION_DEFER_REASON,
                    message=(
                        "Bridge fitting remains deferred because at least one domain-pair "
                        "observation discrepancy row did not produce an honest canonical "
                        "matching summary."
                    ),
                    benchmark_mode=benchmark_mode,
                )

            pair_weight = (pre_share * post_share) / float(domain_pair_summary.n_pairs)
            matched_transition_burden += (
                pair_weight * domain_pair_summary.matched_transition_burden
            )
            source_unmatched_burden += (
                pair_weight * domain_pair_summary.source_unmatched_burden
            )
            target_unmatched_burden += (
                pair_weight * domain_pair_summary.target_unmatched_burden
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
    runtime_metadata = runtime_settings.execution_metadata()

    estimator_metadata = {
        "supported_case": _MINIMAL_SUPPORTED_CASE,
        "estimator_mode": _CANONICAL_BRIDGE_MODE,
        "estimator_method": _CANONICAL_BRIDGE_METHOD,
        "benchmark_mode": benchmark_mode,
        "effective_match_penalty": float(effective_match_penalty),
        "observation_discrepancy_config": {
            "eps_schedule": tuple(discrepancy_cfg.eps_schedule),
            "max_iter": int(discrepancy_cfg.max_iter),
            "tol": float(discrepancy_cfg.tol),
            "n_min_proto": float(discrepancy_cfg.n_min_proto),
            "match_penalty": float(effective_match_penalty),
            **runtime_metadata,
            "max_calibration_workers": runtime_settings.max_calibration_workers,
            "plan_chunk_elements": int(
                runtime_settings.resolved_plan_chunk_elements(
                    fallback=_BRIDGE_PLAN_CHUNK_ELEMENTS
                )
            ),
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
    benchmark_mode: str = "reference",
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
            "benchmark_mode": benchmark_mode,
            "effective_match_penalty": _resolve_bridge_match_penalty(
                benchmark_mode=benchmark_mode
            ),
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
    runtime_settings: RuntimeSettings,
    benchmark_mode: str,
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
            bootstrap_result = _build_realized_patient_bridge_result(
                bootstrap_input,
                runtime_settings=runtime_settings,
                benchmark_mode=benchmark_mode,
            )
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
    runtime_settings: RuntimeSettings,
    benchmark_mode: str,
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
                runtime_settings=runtime_settings,
                benchmark_mode=benchmark_mode,
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
    """Configuration for the canonical full STRIDE fit flow.

    Fields:
        benchmark_mode: Output/control regime used by downstream benchmark
            callers. `reference` keeps the full canonical path,
            `open_channel_ablation` disables explicit open-channel control, and
            `cohort_ablation` disables cohort recurrence shrinkage.
        objective_weights: Canonical loss weights before any benchmark-mode
            adjustments are applied.
        cohort_shrinkage_weight: Canonical shrinkage strength before benchmark
            controls optionally zero it out.
        enable_relation_refinement: Whether to run the active constrained
            relation refinement step after proxy initialization and cohort
            shrinkage.
    """

    timepoint_order: tuple[str, ...] = ()
    benchmark_mode: str = "reference"
    recurrence_config: RecurrenceConfig | None = None
    uncertainty_config: PatientBootstrapConfig | None = None
    objective_weights: LossWeights = field(default_factory=LossWeights)
    cohort_shrinkage_weight: float = 0.25
    enable_relation_refinement: bool = False
    relation_refinement_max_iter: int = 100
    relation_refinement_tol: float = 1e-8
    runtime_settings: RuntimeSettings = field(default_factory=RuntimeSettings)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_labels = tuple(
            _require_nonempty_identifier(label, field_name="timepoint_order label")
            for label in self.timepoint_order
        )
        benchmark_mode = _require_nonempty_identifier(
            self.benchmark_mode,
            field_name="benchmark_mode",
        )
        if len(set(normalized_labels)) != len(normalized_labels):
            raise ContractError("STRIDEFitConfig.timepoint_order must not contain duplicates")
        if benchmark_mode not in _ALLOWED_BENCHMARK_MODES:
            raise ContractError(
                "STRIDEFitConfig.benchmark_mode must be one of "
                f"{_ALLOWED_BENCHMARK_MODES}, got {benchmark_mode!r}"
            )
        if self.uncertainty_config is not None and not isinstance(
            self.uncertainty_config,
            PatientBootstrapConfig,
        ):
            raise ContractError(
                "STRIDEFitConfig.uncertainty_config must be a PatientBootstrapConfig when provided"
            )
        if not isinstance(self.runtime_settings, RuntimeSettings):
            raise ContractError(
                "STRIDEFitConfig.runtime_settings must be a RuntimeSettings object"
            )
        if not isinstance(self.objective_weights, LossWeights):
            raise ContractError(
                "STRIDEFitConfig.objective_weights must be a LossWeights object"
            )
        if not (0.0 <= float(self.cohort_shrinkage_weight) <= 1.0):
            raise ContractError(
                "STRIDEFitConfig.cohort_shrinkage_weight must lie in the closed interval [0, 1]"
            )
        if int(self.relation_refinement_max_iter) <= 0:
            raise ContractError("STRIDEFitConfig.relation_refinement_max_iter must be positive")
        if float(self.relation_refinement_tol) <= 0.0:
            raise ContractError("STRIDEFitConfig.relation_refinement_tol must be positive")


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
    """Group observations into canonical per-patient bridge input bundles.

    Purpose:
        Normalize raw observation rows into one validated `PatientBridgeInput`
        per patient for the canonical STRIDE fit path.

    Inputs:
        observations: Flat observation sequence supplied by callers.
        state_basis / geometry: Optional shared state-space metadata.
        config: Fit configuration whose `timepoint_order` controls group order.

    Returns:
        Ordered patient bridge inputs with domain-stratified group structure and
        count diagnostics.

    Core flow:
        1. Validate the incoming observations and group them by patient id.
        2. Resolve the effective ordered group labels from the configuration.
        3. Partition each patient's observations by ordered group and domain.
        4. Emit validated `PatientBridgeInput` objects with count diagnostics.
    """
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


def run_stride_proxy_fit(
    observations: tuple[FovObservation, ...] | list[FovObservation],
    *,
    state_basis: StateBasis | None = None,
    geometry: StateGeometry | None = None,
    config: STRIDEFitConfig | None = None,
) -> STRIDEFitResult:
    """Run the explicit approximate proxy fit path preserved for compatibility.

    This path may realize local patient bridges and apply bridge-stage
    benchmark controls, but it does not estimate the canonical cohort
    recurrence object and therefore remains
    `implementation_tier=approximate_proxy`.
    """
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
            patient_results.append(
                _build_realized_patient_bridge_result(
                    patient_input,
                    runtime_settings=resolved_config.runtime_settings,
                    benchmark_mode=resolved_config.benchmark_mode,
                )
            )
            continue

        patient_results.append(
            _build_deferred_patient_bridge_result(
                patient_input,
                defer_reason=str(defer_reason),
                message=str(message),
                benchmark_mode=resolved_config.benchmark_mode,
            )
        )

    patient_results_tuple = tuple(patient_results)
    recurrence_config = resolved_config.recurrence_config or RecurrenceConfig()
    recurrence = build_deferred_recurrence_result(
        tuple(patient_input.patient_id for patient_input in patient_inputs),
        used_patient_ids=tuple(
            result.patient_id for result in patient_results_tuple if result.is_ok
        ),
        config=recurrence_config,
        message=(
            "The preserved approximate proxy path does not estimate canonical cohort-level "
            "recurrence even when some local patient bridges are realized."
        ),
    )

    patient_status_counts = _count_patient_statuses(patient_results_tuple)
    any_realized_bridges = any(patient_result.is_ok for patient_result in patient_results_tuple)
    uncertainty = (
        _build_stride_bootstrap_uncertainty(
            patient_inputs,
            patient_results_tuple,
            config=resolved_config.uncertainty_config,
            runtime_settings=resolved_config.runtime_settings,
            benchmark_mode=resolved_config.benchmark_mode,
        )
        if resolved_config.uncertainty_config is not None
        else None
    )
    metadata = dict(resolved_config.metadata)
    if resolved_config.timepoint_order:
        metadata["timepoint_order"] = tuple(resolved_config.timepoint_order)
    metadata["benchmark_mode"] = resolved_config.benchmark_mode
    metadata["effective_match_penalty"] = _resolve_bridge_match_penalty(
        benchmark_mode=resolved_config.benchmark_mode
    )

    return STRIDEFitResult(
        patient_inputs=patient_inputs,
        patient_results=patient_results_tuple,
        recurrence=recurrence,
        fit_status="deferred",
        implementation_tier="approximate_proxy",
        summaries={
            "n_patients": len(patient_results_tuple),
            "patient_status_counts": patient_status_counts,
            "recurrence_fit_status": recurrence.fit_status,
            "implementation_tier": "approximate_proxy",
            "benchmark_mode": resolved_config.benchmark_mode,
            "effective_match_penalty": _resolve_bridge_match_penalty(
                benchmark_mode=resolved_config.benchmark_mode
            ),
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
                "proxy_patient_bridge_realized_recurrence_deferred"
                if any_realized_bridges
                else "deferred"
            ),
            "message": (
                "Supported patients may receive realized proxy patient bridge estimates while "
                "canonical cohort-level recurrence remains deferred on the preserved "
                "approximate proxy path."
                if any_realized_bridges
                else (
                    "No patients satisfied the minimal supported bridge-fitting case, "
                    "so proxy patient bridge estimates remain deferred while cohort-level "
                    "recurrence also remains deferred."
                )
            ),
            "patient_status_counts": patient_status_counts,
            "implementation_tier": "approximate_proxy",
            "benchmark_mode": resolved_config.benchmark_mode,
            "effective_match_penalty": _resolve_bridge_match_penalty(
                benchmark_mode=resolved_config.benchmark_mode
            ),
            **(
                {"uncertainty_status": uncertainty.cohort_summary.uncertainty_status}
                if uncertainty is not None
                else {}
            ),
        },
        uncertainty=uncertainty,
        metadata=metadata,
    )


def run_stride_fit(
    observations: tuple[FovObservation, ...] | list[FovObservation],
    *,
    state_basis: StateBasis | None = None,
    geometry: StateGeometry | None = None,
    config: STRIDEFitConfig | None = None,
) -> STRIDEFitResult:
    """Run the canonical full STRIDE path with patient-plus-cohort structure.

    Purpose:
        Upgrade the proxy-initialized patient bridges into canonical full STRIDE
        outputs by estimating cohort recurrence, shrinking patient relations
        toward the cohort template when allowed, while preserving any
        benchmark-mode bridge controls already applied upstream.

    Inputs:
        observations: Flat observation sequence for one cohort.
        state_basis / geometry: Optional shared state-space metadata.
        config: Canonical fit configuration, including benchmark-mode control.

    Returns:
        A `STRIDEFitResult` whose patient outputs and cohort recurrence reflect
        the canonical full path rather than the preserved proxy tier.

    Core flow:
        1. Run the preserved proxy initializer to get local patient relations.
        2. Estimate cohort recurrence from realized local relations when
           possible.
        3. Canonicalize each patient result with template shrinkage and
           benchmark-mode controls.
        4. Return canonical patient, cohort, summary, and diagnostic outputs.
    """
    resolved_config = config or STRIDEFitConfig()
    effective_objective_weights, effective_shrinkage_weight = _resolve_benchmark_controls(
        resolved_config
    )
    proxy_result = run_stride_proxy_fit(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        config=resolved_config,
    )
    recurrence_config = resolved_config.recurrence_config or RecurrenceConfig()
    realized_relations = tuple(
        result.relation
        for result in proxy_result.patient_results
        if result.relation is not None
    )
    realized_patient_ids = tuple(
        result.patient_id
        for result in proxy_result.patient_results
        if result.relation is not None
    )
    if realized_relations:
        recurrence = _align_recurrence_to_all_patients(
            proxy_result.patient_ids,
            estimate_recurrence(realized_relations, config=recurrence_config),
            basis_dim=recurrence_config.basis_dim,
        )
    else:
        recurrence = build_deferred_recurrence_result(
            proxy_result.patient_ids,
            used_patient_ids=(),
            config=recurrence_config,
            message=(
                "Canonical full STRIDE could not estimate cohort-level recurrence because "
                "no patient-level relations were realized by the local initializer."
            ),
        )

    template_family = recurrence.families[0] if recurrence.fit_status == "ok" and recurrence.families else None
    canonical_patient_results = tuple(
        _canonicalize_proxy_patient_result(
            patient_input,
            local_result=proxy_patient_result,
            objective_weights=effective_objective_weights,
            benchmark_mode=resolved_config.benchmark_mode,
            template_A=(template_family.template_A if template_family is not None else None),
            template_d=(template_family.template_d if template_family is not None else None),
            template_e=(template_family.template_e if template_family is not None else None),
            cohort_fit_status=recurrence.fit_status,
            shrinkage_weight=effective_shrinkage_weight,
            enable_relation_refinement=resolved_config.enable_relation_refinement,
            relation_refinement_max_iter=resolved_config.relation_refinement_max_iter,
            relation_refinement_tol=resolved_config.relation_refinement_tol,
        )
        for patient_input, proxy_patient_result in zip(
            proxy_result.patient_inputs,
            proxy_result.patient_results,
            strict=True,
        )
    )
    patient_status_counts = _count_patient_statuses(canonical_patient_results)
    fit_status = (
        "ok"
        if recurrence.fit_status == "ok"
        and all(result.fit_status == "ok" for result in canonical_patient_results)
        else "deferred"
    )
    objective = aggregate_loss_breakdowns(
        tuple(
            result.objective
            for result in canonical_patient_results
            if result.objective is not None
        )
    )
    metadata = {
        **dict(proxy_result.metadata),
        "benchmark_mode": resolved_config.benchmark_mode,
        "effective_match_penalty": _resolve_bridge_match_penalty(
            benchmark_mode=resolved_config.benchmark_mode
        ),
        "effective_cohort_shrinkage_weight": effective_shrinkage_weight,
        "effective_objective_weights": {
            "observation_data_fit": float(effective_objective_weights.resolved_observation_data_fit),
            "patient_consistency": float(effective_objective_weights.patient_consistency),
            "open_relation": float(effective_objective_weights.resolved_open_relation),
            "cohort_recurrence": float(effective_objective_weights.cohort_recurrence),
            "geometry_structure": float(effective_objective_weights.resolved_geometry_structure),
        },
        "proxy_initializer_tier": proxy_result.implementation_tier,
        "uncertainty_scope": (
            "proxy_initializer_bootstrap"
            if proxy_result.uncertainty is not None
            else "not_requested"
        ),
    }
    diagnostics = {
        "mode": (
            "canonical_full_joint_patient_cohort"
            if fit_status == "ok"
            else "canonical_full_patient_or_cohort_deferred"
        ),
        "message": (
            "Canonical full STRIDE returned patient-level relations together with an explicit "
            "cohort-level recurrence/common-structure layer."
            if fit_status == "ok"
            else (
                "Canonical full STRIDE returned explicit patient-level and cohort-level status, "
                "with at least one patient or cohort component remaining deferred."
            )
        ),
        "patient_status_counts": patient_status_counts,
        "proxy_initializer_patient_status_counts": dict(proxy_result.summaries.get("patient_status_counts", {})),
        "recurrence_used_patient_ids": tuple(recurrence.used_patient_ids or realized_patient_ids),
        "implementation_tier": "canonical_full",
        "benchmark_mode": resolved_config.benchmark_mode,
        "effective_match_penalty": _resolve_bridge_match_penalty(
            benchmark_mode=resolved_config.benchmark_mode
        ),
        **(
            {"uncertainty_status": proxy_result.uncertainty.cohort_summary.uncertainty_status}
            if proxy_result.uncertainty is not None
            else {}
        ),
    }
    summaries = {
        "n_patients": len(canonical_patient_results),
        "n_realized_patients": _count_realized_patients(canonical_patient_results),
        "patient_status_counts": patient_status_counts,
        "recurrence_fit_status": recurrence.fit_status,
        "n_recurrence_used_patients": len(recurrence.used_patient_ids),
        "implementation_tier": "canonical_full",
        "benchmark_mode": resolved_config.benchmark_mode,
        "effective_match_penalty": _resolve_bridge_match_penalty(
            benchmark_mode=resolved_config.benchmark_mode
        ),
        **(
            {
                "uncertainty_status": proxy_result.uncertainty.cohort_summary.uncertainty_status,
                "n_patients_with_uncertainty": proxy_result.uncertainty.cohort_summary.n_realized_patients,
            }
            if proxy_result.uncertainty is not None
            else {}
        ),
    }
    return STRIDEFitResult(
        patient_inputs=proxy_result.patient_inputs,
        patient_results=canonical_patient_results,
        recurrence=recurrence,
        fit_status=fit_status,
        implementation_tier="canonical_full",
        objective=objective,
        summaries=summaries,
        diagnostics=diagnostics,
        uncertainty=proxy_result.uncertainty,
        metadata=metadata,
    )


__all__ = [
    "BridgeObservationGroup",
    "PatientBridgeInput",
    "STRIDEFitConfig",
    "build_patient_bridge_inputs",
    "run_stride_proxy_fit",
    "run_stride_fit",
    "validate_bridge_observation_group",
    "validate_patient_bridge_input",
]
