"""Canonical STRIDE fit orchestration for the current Python interface.

Role:
    Provide the public patient-plus-cohort STRIDE fit path together with the
    bounded first-pass PyTorch/AdamW full-estimator implementation and the
    compatibility local initializer used outside the supported envelope.

Local boundary:
    - This module owns input validation, full-estimator fitting, local
      compatibility initialization, and the cohort recurrence layer used by
      `fit_stride`.
    - It does not define Task A routing, semisynthetic generation, or Block 3
      scoring rules.
    - It exposes Task A core ablations only through the private internal refit
      hook used by validation surfaces.

Primary contents:
    - Canonical input dataclasses and validators.
    - Full-estimator objective/optimizer orchestration.
    - Local patient initialization compatibility path.
    - Internal Task A ablation refit hooks.

Why this module exists:
    The Python API needs one audited path for `fit_stride(...)`. Task A may ask
    this layer for core STRIDE ablation refits through a private hook, keeping
    experiment controls out of the ordinary public configuration surface.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..basis.contracts import StateBasis
from ..errors import ContractError
from ..geometry.state_geometry import StateGeometry
from ..latent.operators import PatientRelationAudit, initialize_patient_relation
from ..latent.recurrence import (
    PatientRecurrenceEmbedding,
    RecurrenceConfig,
    RecurrenceFamily,
    RecurrenceParameters,
    RecurrenceResult,
    build_deferred_recurrence_result,
    estimate_recurrence,
)
from ..objectives import LossBreakdown, LossWeights, aggregate_loss_breakdowns, evaluate_loss_bundle
from ..objectives.full_estimator import FullEstimatorEvidenceBlock
from ..observation.contracts import (
    DomainStratifiedMeasure,
    FovObservation,
    ObservationDiscrepancyConfig,
)
from ..observation.discrepancy import build_observation_kernels, match_observation_clouds
from ..observation.measures import build_domain_stratified_measure
from ..optimize import FullEstimatorOptimizerConfig, optimize_full_estimator
from ..outputs.fit_result import PatientBridgeResult, STRIDEFitResult
from ..outputs.uncertainty import (
    PatientBootstrapConfig,
    PatientBootstrapUncertaintyResult,
    STRIDEBootstrapUncertaintyResult,
    build_cohort_bootstrap_summary,
    summarize_bootstrap_array,
)
from ..settings.runtime import RuntimeSettings
from ._fit_inputs import (
    _build_patient_fit_inputs as _build_patient_fit_inputs_for_order,
)
from ._fit_inputs import (
    _FitObservationGroup,
    _normalize_nested_counts,
    _PatientFitInput,
    _require_nonempty_identifier,
    _validate_fit_observation_group,  # noqa: F401 - retained as private module surface.
    _validate_patient_fit_input,  # noqa: F401 - retained as private module surface.
)


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

    local_auxiliary = dict(local_result.auxiliary)
    canonical_auxiliary: dict[str, Any] = {}
    emergence_scale = float(np.sum(mu_minus, dtype=float))
    known_local_fields = (
        "matched_transition_burden",
        "raw_matched_transition_burden",
        "source_unmatched_burden",
        "target_unmatched_burden",
    )

    for field_name, value in local_auxiliary.items():
        if field_name in known_local_fields:
            canonical_auxiliary[f"local_initializer_{field_name}"] = value
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


def _normalize_relation_arrays(
    *,
    A: np.ndarray,
    d: np.ndarray,
    e: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize relation arrays before attaching them to output objects."""
    return (
        np.asarray(A, dtype=float),
        np.asarray(d, dtype=float),
        np.asarray(e, dtype=float),
    )


def _build_patient_loss_breakdown(
    patient_input: _PatientFitInput,
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
    patient_input: _PatientFitInput,
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


def _canonicalize_local_patient_result(
    patient_input: _PatientFitInput,
    *,
    local_result: PatientBridgeResult,
    objective_weights: LossWeights,
    template_A: np.ndarray | None = None,
    template_d: np.ndarray | None = None,
    template_e: np.ndarray | None = None,
    cohort_fit_status: str,
    shrinkage_weight: float,
    enable_relation_refinement: bool,
    relation_refinement_max_iter: int,
    relation_refinement_tol: float,
) -> PatientBridgeResult:
    """Project a local initializer result onto the canonical full path.

    Purpose:
        Combine the local initializer with cohort recurrence shrinkage to
        produce the patient result returned by canonical full STRIDE.

    Core flow:
        1. Preserve deferred/failed local statuses without fabricating new
           patient relations.
        2. Optionally shrink the realized local relation toward the cohort
           template.
        3. Normalize relation arrays.
        4. Rebuild diagnostics, auxiliary payloads, and loss breakdowns for the
           canonical result.
    """
    if local_result.fit_status != "ok":
        local_message = str(
            local_result.diagnostics.get(
                "message",
                "Canonical full STRIDE preserved the explicit deferred/failed patient "
                "status from the local initializer.",
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
                "local_initializer_message": local_message,
                "canonical_context_message": (
                    "Canonical full STRIDE preserved the explicit deferred/failed patient "
                    "status from the local initializer."
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
    final_A, final_d, final_e = _normalize_relation_arrays(
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
            **refinement_metadata,
            "message": (
                "Canonical full STRIDE combined the current local initializer with the "
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
_ALLOWED_INTERNAL_ABLATION_MODES: tuple[str, ...] = (
    "none",
    "recurrence",
    "geometry",
    "consistency",
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


def _resolve_patient_state_ids(patient_input: _PatientFitInput) -> tuple[int, ...] | None:
    if patient_input.state_basis is not None:
        return patient_input.state_basis.resolved_state_ids
    if patient_input.geometry is not None:
        return tuple(int(state_id) for state_id in patient_input.geometry.state_ids)
    return None


def _shared_bridge_count_diagnostics(patient_input: _PatientFitInput) -> dict[str, Any]:
    return {
        "ordered_group_labels": patient_input.ordered_group_labels,
        "n_groups": len(patient_input.ordered_group_labels),
        "n_observations_by_group": dict(patient_input.n_observations_by_group),
        "n_observations_by_domain": dict(patient_input.n_observations_by_domain),
        "n_observations_by_group_and_domain": _normalize_nested_counts(
            patient_input.n_observations_by_group_and_domain
        ),
    }


def _shared_bridge_audit_metadata(patient_input: _PatientFitInput) -> dict[str, Any]:
    diagnostics = _shared_bridge_count_diagnostics(patient_input)
    return {
        "n_observations_by_group": diagnostics["n_observations_by_group"],
        "n_observations_by_domain": diagnostics["n_observations_by_domain"],
        "n_observations_by_group_and_domain": diagnostics["n_observations_by_group_and_domain"],
        "bridge_input_metadata": dict(patient_input.metadata),
    }


def _evaluate_patient_bridge_support(
    patient_input: _PatientFitInput,
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


def _mean_group_composition(group: _FitObservationGroup) -> np.ndarray:
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


def _summarize_group_domains(group: _FitObservationGroup) -> tuple[_GroupDomainSummary, ...]:
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
    patient_input: _PatientFitInput,
    *,
    runtime_settings: RuntimeSettings,
) -> PatientBridgeResult:
    if patient_input.geometry is None:
        raise ContractError("Realized bridge estimation requires _PatientFitInput.geometry")

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
    effective_match_penalty = float(_DEFAULT_BRIDGE_MATCH_PENALTY)

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
    patient_input: _PatientFitInput,
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
            "effective_match_penalty": float(_DEFAULT_BRIDGE_MATCH_PENALTY),
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
    patient_input: _PatientFitInput,
    *,
    rng: np.random.Generator,
    replicate_index: int,
    preserve_domain_stratification: bool,
) -> _PatientFitInput:
    bootstrap_groups: list[_FitObservationGroup] = []
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
            _FitObservationGroup(
                group_label=group.group_label,
                observations=tuple(sampled_group_observations),
                observations_by_domain=sampled_group_by_domain,
            )
        )

    return _PatientFitInput(
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
    patient_input: _PatientFitInput,
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
    patient_input: _PatientFitInput,
    patient_result: PatientBridgeResult,
    *,
    config: PatientBootstrapConfig,
    bootstrap_seed: int,
    runtime_settings: RuntimeSettings,
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
    patient_inputs: tuple[_PatientFitInput, ...],
    patient_results: tuple[PatientBridgeResult, ...],
    *,
    config: PatientBootstrapConfig,
    runtime_settings: RuntimeSettings,
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
        objective_weights: Canonical loss weights for the current reference
            path. These do not expose Task A ablation controls to normal users.
        cohort_shrinkage_weight: Canonical shrinkage strength for the current
            reference path.
        enable_relation_refinement: Whether to run the active constrained
            relation refinement step after local initialization and cohort
            shrinkage.
    """

    timepoint_order: tuple[str, ...] = ()
    recurrence_config: RecurrenceConfig | None = None
    uncertainty_config: PatientBootstrapConfig | None = None
    objective_weights: LossWeights = field(default_factory=LossWeights)
    cohort_shrinkage_weight: float = 0.25
    enable_relation_refinement: bool = False
    relation_refinement_max_iter: int = 100
    relation_refinement_tol: float = 1e-8
    runtime_settings: RuntimeSettings = field(default_factory=RuntimeSettings)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _ablation_mode: str = field(default="none", init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        normalized_labels = tuple(
            _require_nonempty_identifier(label, field_name="timepoint_order label")
            for label in self.timepoint_order
        )
        _validate_internal_ablation_mode(self._ablation_mode)
        if len(set(normalized_labels)) != len(normalized_labels):
            raise ContractError("STRIDEFitConfig.timepoint_order must not contain duplicates")
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


def _validate_internal_ablation_mode(value: object) -> str:
    normalized_mode = _require_nonempty_identifier(
        value,
        field_name="_ablation_mode",
    )
    if normalized_mode not in _ALLOWED_INTERNAL_ABLATION_MODES:
        raise ContractError(
            "Internal STRIDEFitConfig._ablation_mode must be one of "
            f"{_ALLOWED_INTERNAL_ABLATION_MODES}, got {normalized_mode!r}"
        )
    return normalized_mode


def _with_internal_ablation_mode(
    config: STRIDEFitConfig | None,
    ablation_mode: str,
) -> STRIDEFitConfig:
    """Return a config carrying a Task A-only core ablation request.

    The hook is intentionally private so recurrence/geometry/consistency refits
    remain experiment provenance rather than ordinary user-level fit controls.
    """

    resolved_config = config or STRIDEFitConfig()
    normalized_mode = _validate_internal_ablation_mode(ablation_mode)
    copied_config = replace(resolved_config)
    object.__setattr__(copied_config, "_ablation_mode", normalized_mode)
    return copied_config


def _build_patient_fit_inputs(
    observations: tuple[FovObservation, ...] | list[FovObservation],
    *,
    state_basis: StateBasis | None = None,
    geometry: StateGeometry | None = None,
    config: STRIDEFitConfig | None = None,
) -> tuple[_PatientFitInput, ...]:
    """Group observations into canonical per-patient fit input bundles.

    Purpose:
        Adapt the workflow-level `STRIDEFitConfig` surface into the canonical
        per-patient input bundles consumed by the current fit path.

    Core flow:
        1. Resolve the default fit config when callers omit one.
        2. Carry the declared `timepoint_order` into the input-grouping layer.
        3. Delegate grouping and validation to `stride.workflows._fit_inputs`.
    """
    resolved_config = config or STRIDEFitConfig()
    return _build_patient_fit_inputs_for_order(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        timepoint_order=tuple(resolved_config.timepoint_order),
    )


def _run_local_initializer_fit(
    observations: tuple[FovObservation, ...] | list[FovObservation],
    *,
    state_basis: StateBasis | None = None,
    geometry: StateGeometry | None = None,
    config: STRIDEFitConfig | None = None,
) -> STRIDEFitResult:
    """Run the current local patient initializer used by `fit_stride`.

    This initializer may realize local patient relations, but it does not
    estimate the canonical cohort recurrence object by itself.
    """
    resolved_config = config or STRIDEFitConfig()
    patient_inputs = _build_patient_fit_inputs(
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
                )
            )
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
    recurrence = build_deferred_recurrence_result(
        tuple(patient_input.patient_id for patient_input in patient_inputs),
        used_patient_ids=tuple(
            result.patient_id for result in patient_results_tuple if result.is_ok
        ),
        config=recurrence_config,
        message=(
            "The local patient initializer does not estimate canonical cohort-level "
            "recurrence by itself even when some patient relations are realized."
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
        )
        if resolved_config.uncertainty_config is not None
        else None
    )
    metadata = dict(resolved_config.metadata)
    if resolved_config.timepoint_order:
        metadata["timepoint_order"] = tuple(resolved_config.timepoint_order)
    metadata["effective_match_penalty"] = float(_DEFAULT_BRIDGE_MATCH_PENALTY)

    return STRIDEFitResult(
        patient_inputs=patient_inputs,
        patient_results=patient_results_tuple,
        recurrence=recurrence,
        fit_status="deferred",
        implementation_tier="local_initializer",
        summaries={
            "n_patients": len(patient_results_tuple),
            "patient_status_counts": patient_status_counts,
            "recurrence_fit_status": recurrence.fit_status,
            "implementation_tier": "local_initializer",
            "effective_match_penalty": float(_DEFAULT_BRIDGE_MATCH_PENALTY),
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
                "local_patient_relation_realized_recurrence_deferred"
                if any_realized_bridges
                else "deferred"
            ),
            "message": (
                "Supported patients may receive realized local patient relation estimates while "
                "canonical cohort-level recurrence remains deferred at the initializer layer."
                if any_realized_bridges
                else (
                    "No patients satisfied the minimal supported bridge-fitting case, "
                    "so local patient relation estimates remain deferred while cohort-level "
                    "recurrence also remains deferred."
                )
            ),
            "patient_status_counts": patient_status_counts,
            "implementation_tier": "local_initializer",
            "effective_match_penalty": float(_DEFAULT_BRIDGE_MATCH_PENALTY),
            **(
                {"uncertainty_status": uncertainty.cohort_summary.uncertainty_status}
                if uncertainty is not None
                else {}
            ),
        },
        uncertainty=uncertainty,
        metadata=metadata,
    )


def _build_deferred_ablation_fit(
    observations: tuple[FovObservation, ...] | list[FovObservation],
    *,
    state_basis: StateBasis | None,
    geometry: StateGeometry | None,
    config: STRIDEFitConfig,
) -> STRIDEFitResult:
    """Represent an internal core STRIDE ablation request as deferred refit work."""

    ablation_mode = _validate_internal_ablation_mode(config._ablation_mode)
    patient_inputs = _build_patient_fit_inputs(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        config=config,
    )
    message = (
        "Internal Task A core STRIDE ablation refit is deferred because this input "
        "does not satisfy the supported two-timepoint geometry-present full-estimator path."
    )
    patient_results = tuple(
        PatientBridgeResult(
            patient_id=patient_input.patient_id,
            fit_status="deferred",
            state_ids=_resolve_patient_state_ids(patient_input),
            audit=PatientRelationAudit(
                patient_id=patient_input.patient_id,
                timepoint_order=patient_input.ordered_group_labels,
                mass_mode=patient_input.mass_mode,
                observation_fit_status="deferred",
                bridge_status="deferred",
                metadata={
                    **_shared_bridge_audit_metadata(patient_input),
                    "ablation_mode": ablation_mode,
                    "defer_reason": "unsupported_full_estimator_input",
                },
            ),
            diagnostics={
                **_shared_bridge_count_diagnostics(patient_input),
                "mode": "core_ablation_refit_deferred",
                "ablation_mode": ablation_mode,
                "defer_reason": "unsupported_full_estimator_input",
                "message": message,
            },
            implementation_tier="canonical_full",
        )
        for patient_input in patient_inputs
    )
    recurrence_config = config.recurrence_config or RecurrenceConfig()
    recurrence = build_deferred_recurrence_result(
        tuple(patient_input.patient_id for patient_input in patient_inputs),
        used_patient_ids=(),
        config=recurrence_config,
        message=message,
    )
    patient_status_counts = _count_patient_statuses(patient_results)
    metadata = {
        **dict(config.metadata),
        "ablation_mode": ablation_mode,
        "ablation_status": "deferred_unsupported_full_estimator_input",
    }
    if config.timepoint_order:
        metadata["timepoint_order"] = tuple(config.timepoint_order)
    summaries = {
        "n_patients": len(patient_results),
        "n_realized_patients": 0,
        "patient_status_counts": patient_status_counts,
        "recurrence_fit_status": recurrence.fit_status,
        "n_recurrence_used_patients": 0,
        "implementation_tier": "canonical_full",
        "ablation_mode": ablation_mode,
        "ablation_status": "deferred_unsupported_full_estimator_input",
    }
    diagnostics = {
        "mode": "canonical_full_ablation_refit_deferred",
        "message": message,
        "patient_status_counts": patient_status_counts,
        "recurrence_used_patient_ids": (),
        "implementation_tier": "canonical_full",
        "ablation_mode": ablation_mode,
        "ablation_status": "deferred_unsupported_full_estimator_input",
    }
    return STRIDEFitResult(
        patient_inputs=patient_inputs,
        patient_results=patient_results,
        recurrence=recurrence,
        fit_status="deferred",
        implementation_tier="canonical_full",
        objective=None,
        summaries=summaries,
        diagnostics=diagnostics,
        uncertainty=None,
        metadata=metadata,
    )


def _group_state_matrix(observations: Sequence[FovObservation]) -> np.ndarray:
    return np.vstack(
        [np.asarray(observation.community_composition, dtype=float) for observation in observations]
    ).astype(float, copy=False)


def _build_full_estimator_evidence_blocks(
    patient_inputs: Sequence[_PatientFitInput],
) -> tuple[FullEstimatorEvidenceBlock, ...]:
    blocks: list[FullEstimatorEvidenceBlock] = []
    for patient_input in patient_inputs:
        source_group, target_group = patient_input.groups
        for source_domain, source_observations in sorted(
            source_group.observations_by_domain.items()
        ):
            for target_domain, target_observations in sorted(
                target_group.observations_by_domain.items()
            ):
                block_id = (
                    f"{patient_input.patient_id}:"
                    f"{source_group.group_label}:{source_domain}->"
                    f"{target_group.group_label}:{target_domain}"
                )
                blocks.append(
                    FullEstimatorEvidenceBlock(
                        patient_id=patient_input.patient_id,
                        source_bag=_group_state_matrix(source_observations),
                        target_bag=_group_state_matrix(target_observations),
                        block_id=block_id,
                    )
                )
    if not blocks:
        raise ContractError("Canonical full estimator requires at least one evidence block")
    return tuple(blocks)


def _all_patient_inputs_support_full_optimizer(
    patient_inputs: Sequence[_PatientFitInput],
) -> bool:
    return bool(patient_inputs) and all(
        _evaluate_patient_bridge_support(patient_input)[0]
        for patient_input in patient_inputs
    )


def _tensor_scalar(value: object) -> float:
    try:
        import torch
    except ImportError:  # pragma: no cover - optimizer path requires torch first
        torch = None  # type: ignore[assignment]
    if torch is not None and torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _tensor_array(value: object) -> np.ndarray:
    try:
        import torch
    except ImportError:  # pragma: no cover - optimizer path requires torch first
        torch = None  # type: ignore[assignment]
    if torch is not None and torch.is_tensor(value):
        return np.asarray(value.detach().cpu(), dtype=float)
    return np.asarray(value, dtype=float)


def _loss_breakdown_from_full_ledger(ledger: object) -> LossBreakdown:
    components = ledger.components

    def _effective(component_name: str) -> float:
        component = components[component_name]
        return _tensor_scalar(
            component.effective_normalized
            if component.effective_normalized is not None
            else component.normalized
        )

    return LossBreakdown(
        observation_data_fit=_effective("obs"),
        patient_consistency=_effective("consistency"),
        open_relation=_effective("open"),
        cohort_recurrence=_effective("recurrence"),
        geometry_structure=_effective("geometry"),
        total=_tensor_scalar(ledger.total),
    )


def _build_full_optimizer_recurrence(
    patient_ids: tuple[str, ...],
    ledger: object,
    *,
    recurrence_config: RecurrenceConfig,
) -> RecurrenceResult:
    recurrence = ledger.recurrence
    family = RecurrenceFamily(
        family_id="cohort_consensus",
        template_A=_tensor_array(recurrence.A_bar),
        template_d=_tensor_array(recurrence.d_bar),
        template_e=_tensor_array(recurrence.e_bar),
        support_n_patients=int(recurrence.support_n_patients),
        within_family_dispersion=_tensor_scalar(recurrence.dispersion),
        fit_status="ok",
        member_patient_ids=patient_ids,
    )
    embeddings = tuple(
        PatientRecurrenceEmbedding(
            patient_id=patient_id,
            coordinates=np.zeros(int(recurrence_config.basis_dim), dtype=float),
            fit_status="ok",
        )
        for patient_id in patient_ids
    )
    return RecurrenceResult(
        patient_ids=patient_ids,
        families=(family,),
        fit_status="ok",
        used_patient_ids=patient_ids,
        recurrence_unit=recurrence_config.recurrence_unit,
        parameters=RecurrenceParameters(
            basis_dim=recurrence_config.basis_dim,
            loadings=None,
            metadata={"mode": "full_estimator_consensus_v1"},
        ),
        embeddings=embeddings,
        metadata={
            "mode": "full_estimator_consensus_v1",
            "message": "Built the single cohort recurrence family from the full-estimator ledger.",
            "n_used_patients": len(patient_ids),
        },
    )


def _build_full_optimizer_patient_results(
    patient_inputs: Sequence[_PatientFitInput],
    *,
    optimizer_result: object,
    evidence_blocks: Sequence[FullEstimatorEvidenceBlock],
    objective: LossBreakdown,
) -> tuple[PatientBridgeResult, ...]:
    parameters = optimizer_result.parameters
    A_all = _tensor_array(parameters.A)
    d_all = _tensor_array(parameters.d)
    e_all = _tensor_array(parameters.e)
    block_counts = Counter(block.patient_id for block in evidence_blocks)
    results: list[PatientBridgeResult] = []
    for patient_index, patient_input in enumerate(patient_inputs):
        pre_group, post_group = patient_input.groups
        mu_minus = _mean_group_composition(pre_group)
        mu_plus = _mean_group_composition(post_group)
        audit = PatientRelationAudit(
            patient_id=patient_input.patient_id,
            timepoint_order=patient_input.ordered_group_labels,
            mass_mode=patient_input.mass_mode,
            n_pre_observations=int(patient_input.n_observations_by_group[pre_group.group_label]),
            n_post_observations=int(patient_input.n_observations_by_group[post_group.group_label]),
            observation_fit_status="D_obs^BalancedSinkhornDivergence-v1",
            bridge_status="ok",
            metadata={
                **_shared_bridge_audit_metadata(patient_input),
                "estimator_mode": "canonical_full_estimator_v1",
                "optimizer_status": optimizer_result.status,
                "n_evidence_blocks": int(block_counts[patient_input.patient_id]),
            },
        )
        results.append(
            PatientBridgeResult(
                patient_id=patient_input.patient_id,
                fit_status="ok",
                A=A_all[patient_index],
                d=d_all[patient_index],
                e=e_all[patient_index],
                mu_minus=mu_minus,
                mu_plus=mu_plus,
                state_ids=_resolve_patient_state_ids(patient_input),
                audit=audit,
                diagnostics={
                    **_shared_bridge_count_diagnostics(patient_input),
                    "mode": "canonical_full_estimator_v1",
                    "supported_case": _MINIMAL_SUPPORTED_CASE,
                    "estimator_mode": "canonical_full_estimator_v1",
                    "observation_fit_status": "D_obs^BalancedSinkhornDivergence-v1",
                    "optimizer_status": optimizer_result.status,
                    "n_evidence_blocks": int(block_counts[patient_input.patient_id]),
                    "message": (
                        "Fitted A, d, and e through the canonical PyTorch/AdamW "
                        "full-estimator objective."
                    ),
                },
                implementation_tier="canonical_full",
                objective=objective,
            )
        )
    return tuple(results)


def _build_non_ok_full_optimizer_fit(
    patient_inputs: tuple[_PatientFitInput, ...],
    *,
    optimizer_result: object,
    recurrence_config: RecurrenceConfig,
) -> STRIDEFitResult:
    fit_status = (
        optimizer_result.status
        if optimizer_result.status in {"deferred", "failed"}
        else "failed"
    )
    reason_key = "defer_reason" if fit_status == "deferred" else "failure_reason"
    reason = str(
        optimizer_result.diagnostics.get(
            reason_key,
            optimizer_result.diagnostics.get("failure_reason", "optimizer_failed"),
        )
    )
    message = str(
        optimizer_result.diagnostics.get(
            "message",
            reason,
        )
    )
    patient_results = tuple(
        PatientBridgeResult(
            patient_id=patient_input.patient_id,
            fit_status=fit_status,
            state_ids=_resolve_patient_state_ids(patient_input),
            audit=PatientRelationAudit(
                patient_id=patient_input.patient_id,
                timepoint_order=patient_input.ordered_group_labels,
                mass_mode=patient_input.mass_mode,
                observation_fit_status=fit_status,
                bridge_status=fit_status,
                metadata={
                    **_shared_bridge_audit_metadata(patient_input),
                    reason_key: reason,
                },
            ),
            diagnostics={
                **_shared_bridge_count_diagnostics(patient_input),
                "mode": f"canonical_full_estimator_{fit_status}",
                reason_key: reason,
                "message": message,
            },
            implementation_tier="canonical_full",
        )
        for patient_input in patient_inputs
    )
    recurrence = build_deferred_recurrence_result(
        tuple(patient_input.patient_id for patient_input in patient_inputs),
        used_patient_ids=(),
        config=recurrence_config,
        message=message,
    )
    status_counts = _count_patient_statuses(patient_results)
    return STRIDEFitResult(
        patient_inputs=patient_inputs,
        patient_results=patient_results,
        recurrence=recurrence,
        fit_status=fit_status,
        implementation_tier="canonical_full",
        summaries={
            "n_patients": len(patient_results),
            "n_realized_patients": 0,
            "patient_status_counts": status_counts,
            "recurrence_fit_status": recurrence.fit_status,
            "implementation_tier": "canonical_full",
            "optimizer_status": optimizer_result.status,
        },
        diagnostics={
            "mode": f"canonical_full_estimator_{fit_status}",
            "message": message,
            "patient_status_counts": status_counts,
            "implementation_tier": "canonical_full",
            "optimizer_status": optimizer_result.status,
            **dict(optimizer_result.diagnostics),
        },
        metadata={
            "implementation_tier": "canonical_full",
            "optimizer_status": optimizer_result.status,
            reason_key: reason,
        },
    )


def _run_full_optimizer_fit(
    patient_inputs: tuple[_PatientFitInput, ...],
    *,
    geometry: StateGeometry,
    config: STRIDEFitConfig,
    ablation_mode: str,
) -> STRIDEFitResult:
    evidence_blocks = _build_full_estimator_evidence_blocks(patient_inputs)
    patient_ids = tuple(patient_input.patient_id for patient_input in patient_inputs)
    K = int(np.asarray(geometry.cost_matrix, dtype=float).shape[0])
    optimizer_result = optimize_full_estimator(
        patient_ids=patient_ids,
        K=K,
        evidence_blocks=evidence_blocks,
        geometry=geometry,
        config=FullEstimatorOptimizerConfig(
            max_steps=4,
            min_steps=1,
            learning_rate=0.02,
            min_relative_improvement=1e-12,
            gradient_norm_tol=0.06,
            ablation_mode=ablation_mode,
        ),
    )
    recurrence_config = config.recurrence_config or RecurrenceConfig()
    if optimizer_result.status != "ok":
        return _build_non_ok_full_optimizer_fit(
            patient_inputs,
            optimizer_result=optimizer_result,
            recurrence_config=recurrence_config,
        )
    if (
        optimizer_result.parameters is None
        or optimizer_result.final_ledger is None
        or optimizer_result.provenance is None
    ):
        raise ContractError("ok full-estimator optimizer result must carry parameters, ledger, and provenance")

    objective = _loss_breakdown_from_full_ledger(optimizer_result.final_ledger)
    recurrence = _build_full_optimizer_recurrence(
        patient_ids,
        optimizer_result.final_ledger,
        recurrence_config=recurrence_config,
    )
    patient_results = _build_full_optimizer_patient_results(
        patient_inputs,
        optimizer_result=optimizer_result,
        evidence_blocks=evidence_blocks,
        objective=objective,
    )
    patient_status_counts = _count_patient_statuses(patient_results)
    metadata = {
        **dict(config.metadata),
        "implementation_tier": "canonical_full",
        "optimizer_status": optimizer_result.status,
        "n_evidence_blocks": len(evidence_blocks),
        "recurrence_support_n_patients": len(recurrence.used_patient_ids),
        "detailed_optimizer_trace": False,
    }
    if config.timepoint_order:
        metadata["timepoint_order"] = tuple(config.timepoint_order)
    if ablation_mode != "none":
        metadata["ablation_mode"] = ablation_mode
        metadata["ablation_status"] = "ok"
    summaries = {
        "n_patients": len(patient_results),
        "n_realized_patients": _count_realized_patients(patient_results),
        "patient_status_counts": patient_status_counts,
        "recurrence_fit_status": recurrence.fit_status,
        "n_recurrence_used_patients": len(recurrence.used_patient_ids),
        "implementation_tier": "canonical_full",
        "optimizer_status": optimizer_result.status,
        "n_evidence_blocks": len(evidence_blocks),
        **({"ablation_mode": ablation_mode, "ablation_status": "ok"} if ablation_mode != "none" else {}),
    }
    diagnostics = {
        "mode": "canonical_full_estimator_v1",
        "message": (
            "Canonical full STRIDE fitted patient relations through the PyTorch/AdamW "
            "full-estimator objective."
        ),
        "patient_status_counts": patient_status_counts,
        "recurrence_used_patient_ids": tuple(recurrence.used_patient_ids),
        "implementation_tier": "canonical_full",
        "optimizer_status": optimizer_result.status,
        "n_evidence_blocks": len(evidence_blocks),
        "recurrence_support_n_patients": len(recurrence.used_patient_ids),
        **dict(optimizer_result.diagnostics),
        **({"ablation_mode": ablation_mode, "ablation_status": "ok"} if ablation_mode != "none" else {}),
    }
    return STRIDEFitResult(
        patient_inputs=patient_inputs,
        patient_results=patient_results,
        recurrence=recurrence,
        fit_status="ok",
        implementation_tier="canonical_full",
        objective=objective,
        provenance=optimizer_result.provenance,
        summaries=summaries,
        diagnostics=diagnostics,
        uncertainty=None,
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
        Fit the supported full-estimator path through the PyTorch/AdamW
        objective when geometry and patient inputs satisfy the bounded
        first-pass envelope. Non-ablation compatibility inputs outside that
        envelope may still route through the local initializer fallback.

    Inputs:
        observations: Flat observation sequence for one cohort.
        state_basis / geometry: Optional shared state-space metadata.
        config: Canonical fit configuration.

    Returns:
        A `STRIDEFitResult` whose patient outputs and cohort recurrence reflect
        the current canonical full path or an explicit non-`ok` status when the
        full optimizer cannot complete.

    Core flow:
        1. Resolve patient fit inputs under the requested source/target order.
        2. Run the full optimizer when the bounded support envelope is met.
        3. Defer unsupported internal ablations rather than masking outputs.
        4. Route non-ablation compatibility inputs through the local initializer
           fallback.
    """
    resolved_config = config or STRIDEFitConfig()
    ablation_mode = _validate_internal_ablation_mode(resolved_config._ablation_mode)
    patient_inputs = _build_patient_fit_inputs(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        config=resolved_config,
    )
    if geometry is not None and _all_patient_inputs_support_full_optimizer(patient_inputs):
        return _run_full_optimizer_fit(
            patient_inputs,
            geometry=geometry,
            config=resolved_config,
            ablation_mode=ablation_mode,
        )
    if ablation_mode != "none":
        return _build_deferred_ablation_fit(
            observations,
            state_basis=state_basis,
            geometry=geometry,
            config=resolved_config,
        )
    effective_objective_weights = resolved_config.objective_weights
    effective_shrinkage_weight = float(resolved_config.cohort_shrinkage_weight)
    local_initializer_result = _run_local_initializer_fit(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        config=resolved_config,
    )
    recurrence_config = resolved_config.recurrence_config or RecurrenceConfig()
    realized_relations = tuple(
        result.relation
        for result in local_initializer_result.patient_results
        if result.relation is not None
    )
    realized_patient_ids = tuple(
        result.patient_id
        for result in local_initializer_result.patient_results
        if result.relation is not None
    )
    if realized_relations:
        recurrence = _align_recurrence_to_all_patients(
            local_initializer_result.patient_ids,
            estimate_recurrence(realized_relations, config=recurrence_config),
            basis_dim=recurrence_config.basis_dim,
        )
    else:
        recurrence = build_deferred_recurrence_result(
            local_initializer_result.patient_ids,
            used_patient_ids=(),
            config=recurrence_config,
            message=(
                "Canonical full STRIDE could not estimate cohort-level recurrence because "
                "no patient-level relations were realized by the local initializer."
            ),
        )

    template_family = recurrence.families[0] if recurrence.fit_status == "ok" and recurrence.families else None
    canonical_patient_results = tuple(
        _canonicalize_local_patient_result(
            patient_input,
            local_result=local_patient_result,
            objective_weights=effective_objective_weights,
            template_A=(template_family.template_A if template_family is not None else None),
            template_d=(template_family.template_d if template_family is not None else None),
            template_e=(template_family.template_e if template_family is not None else None),
            cohort_fit_status=recurrence.fit_status,
            shrinkage_weight=effective_shrinkage_weight,
            enable_relation_refinement=resolved_config.enable_relation_refinement,
            relation_refinement_max_iter=resolved_config.relation_refinement_max_iter,
            relation_refinement_tol=resolved_config.relation_refinement_tol,
        )
        for patient_input, local_patient_result in zip(
            local_initializer_result.patient_inputs,
            local_initializer_result.patient_results,
            strict=True,
        )
    )
    patient_status_counts = _count_patient_statuses(canonical_patient_results)
    patient_and_recurrence_realized = (
        recurrence.fit_status == "ok"
        and all(result.fit_status == "ok" for result in canonical_patient_results)
    )
    compact_successful_fit_provenance = None
    compatibility_result_without_provenance = (
        patient_and_recurrence_realized and compact_successful_fit_provenance is None
    )
    fit_status = (
        "ok"
        if patient_and_recurrence_realized and compact_successful_fit_provenance is not None
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
        **dict(local_initializer_result.metadata),
        "effective_match_penalty": float(_DEFAULT_BRIDGE_MATCH_PENALTY),
        "effective_cohort_shrinkage_weight": effective_shrinkage_weight,
        "effective_objective_weights": {
            "observation_data_fit": float(effective_objective_weights.resolved_observation_data_fit),
            "patient_consistency": float(effective_objective_weights.patient_consistency),
            "open_relation": float(effective_objective_weights.resolved_open_relation),
            "cohort_recurrence": float(effective_objective_weights.cohort_recurrence),
            "geometry_structure": float(effective_objective_weights.resolved_geometry_structure),
        },
        "local_initializer_tier": local_initializer_result.implementation_tier,
        "uncertainty_scope": (
            "local_initializer_bootstrap"
            if local_initializer_result.uncertainty is not None
            else "not_requested"
        ),
    }
    diagnostics = {
        "mode": (
            "canonical_full_joint_patient_cohort"
            if fit_status == "ok"
            else "canonical_full_compatibility_without_provenance"
            if compatibility_result_without_provenance
            else "canonical_full_patient_or_cohort_deferred"
        ),
        "message": (
            "Canonical full STRIDE returned patient-level relations together with an explicit "
            "cohort-level recurrence/common-structure layer."
            if fit_status == "ok"
            else (
                "Canonical namespace STRIDE returned realized patient-level relations and "
                "cohort-level recurrence, but compact successful-fit provenance is not available "
                "because this compatibility fallback did not run the supported full-estimator "
                "optimizer path."
            )
            if compatibility_result_without_provenance
            else (
                "Canonical full STRIDE returned explicit patient-level and cohort-level status, "
                "with at least one patient or cohort component remaining deferred."
            )
        ),
        "patient_status_counts": patient_status_counts,
        "local_initializer_patient_status_counts": dict(
            local_initializer_result.summaries.get("patient_status_counts", {})
        ),
        "recurrence_used_patient_ids": tuple(recurrence.used_patient_ids or realized_patient_ids),
        "implementation_tier": "canonical_full",
        "effective_match_penalty": float(_DEFAULT_BRIDGE_MATCH_PENALTY),
        **(
            {
                "uncertainty_status": (
                    local_initializer_result.uncertainty.cohort_summary.uncertainty_status
                )
            }
            if local_initializer_result.uncertainty is not None
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
        "effective_match_penalty": float(_DEFAULT_BRIDGE_MATCH_PENALTY),
        **(
            {
                "uncertainty_status": (
                    local_initializer_result.uncertainty.cohort_summary.uncertainty_status
                ),
                "n_patients_with_uncertainty": (
                    local_initializer_result.uncertainty.cohort_summary.n_realized_patients
                ),
            }
            if local_initializer_result.uncertainty is not None
            else {}
        ),
    }
    return STRIDEFitResult(
        patient_inputs=local_initializer_result.patient_inputs,
        patient_results=canonical_patient_results,
        recurrence=recurrence,
        fit_status=fit_status,
        implementation_tier="canonical_full",
        objective=objective,
        summaries=summaries,
        diagnostics=diagnostics,
        uncertainty=local_initializer_result.uncertainty,
        metadata=metadata,
    )


__all__ = [
    "STRIDEFitConfig",
    "run_stride_fit",
]
