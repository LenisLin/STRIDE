"""PyTorch/AdamW optimizer protocol for the canonical full estimator."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ..errors import ContractError
from ..geometry.state_geometry import StateGeometry
from ..objectives.full_estimator import (
    ABLATION_TERM_HANDLING,
    EPSILON_NORM,
    FullEstimatorEvidenceBlock,
    FullEstimatorObjectiveFixedCache,
    FullEstimatorObjectiveLedger,
    FullEstimatorParameters,
    FullEstimatorUnconstrainedParameters,
    compute_full_estimator_objective,
    parameters_from_unconstrained,
    unconstrained_from_initialization,
)
from ..observation.balanced_sinkhorn import BalancedSinkhornDivergenceConfig
from ..outputs.provenance import (
    STRIDE_FIT_PROVENANCE_SCHEMA_VERSION,
    STRIDEFitProvenance,
    build_stride_fit_provenance,
)

try:  # pragma: no cover - exercised through optimize_full_estimator when absent
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


OptimizerStatus = Literal["ok", "failed", "deferred"]
SchedulerPolicy = Literal["none", "ReduceLROnPlateau_on_total_objective"]


@dataclass(frozen=True)
class FullEstimatorOptimizerConfig:
    """Fixed v1 optimizer protocol for full-estimator refits."""

    alpha: float = 0.5
    learning_rate: float = 0.05
    max_steps: int = 3
    min_steps: int = 1
    convergence_tol: float = 0.0
    patience: int = 3
    min_relative_improvement: float = 1e-8
    gradient_norm_tol: float = 1e-6
    epsilon_norm: float = EPSILON_NORM
    scheduler_policy: SchedulerPolicy = "none"
    detailed_optimizer_trace: bool = False
    random_seed: int | None = None
    ablation_mode: str = "none"
    observation_config: BalancedSinkhornDivergenceConfig | None = None


@dataclass(frozen=True)
class FullEstimatorOptimizerResult:
    """Structured optimizer status/result surface."""

    status: OptimizerStatus
    parameters: FullEstimatorParameters | None = None
    final_ledger: FullEstimatorObjectiveLedger | None = None
    provenance: STRIDEFitProvenance | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    trace: Mapping[str, Any] | None = None


def _require_torch() -> Any:
    if torch is None:  # pragma: no cover - depends on optional runtime
        raise ContractError("canonical full-estimator optimizer requires torch")
    return torch


def _require_positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{field_name} must be a positive integer")
    return int(value)


def _require_finite_float(
    value: Any,
    *,
    field_name: str,
    nonnegative: bool = False,
    positive: bool = False,
) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{field_name} must be a finite float")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field_name} must be a finite float") from exc
    if not math.isfinite(resolved):
        raise ContractError(f"{field_name} must be finite")
    if nonnegative and resolved < 0.0:
        raise ContractError(f"{field_name} must be nonnegative")
    if positive and resolved <= 0.0:
        raise ContractError(f"{field_name} must be positive")
    return resolved


def _validate_scheduler_policy(value: Any) -> SchedulerPolicy:
    policy = str(value)
    if policy not in {"none", "ReduceLROnPlateau_on_total_objective"}:
        raise ContractError(
            "scheduler_policy must be one of 'none' or "
            "'ReduceLROnPlateau_on_total_objective'"
        )
    return policy  # type: ignore[return-value]


def _validate_config(config: FullEstimatorOptimizerConfig | None) -> FullEstimatorOptimizerConfig:
    resolved = config or FullEstimatorOptimizerConfig()
    if not isinstance(resolved, FullEstimatorOptimizerConfig):
        raise ContractError("config must be a FullEstimatorOptimizerConfig object")
    _require_finite_float(resolved.alpha, field_name="alpha", nonnegative=True)
    if float(resolved.alpha) > 1.0:
        raise ContractError("alpha must be finite and in [0, 1]")
    _require_finite_float(resolved.learning_rate, field_name="learning_rate", positive=True)
    max_steps = _require_positive_int(resolved.max_steps, field_name="max_steps")
    min_steps = _require_positive_int(resolved.min_steps, field_name="min_steps")
    if min_steps > max_steps:
        raise ContractError("min_steps must be less than or equal to max_steps")
    _require_positive_int(resolved.patience, field_name="patience")
    _require_finite_float(
        resolved.convergence_tol,
        field_name="convergence_tol",
        nonnegative=True,
    )
    _require_finite_float(
        resolved.min_relative_improvement,
        field_name="min_relative_improvement",
        nonnegative=True,
    )
    _require_finite_float(
        resolved.gradient_norm_tol,
        field_name="gradient_norm_tol",
        nonnegative=True,
    )
    _require_finite_float(resolved.epsilon_norm, field_name="epsilon_norm", positive=True)
    _validate_scheduler_policy(resolved.scheduler_policy)
    if not isinstance(resolved.detailed_optimizer_trace, bool):
        raise ContractError("detailed_optimizer_trace must be a bool")
    if resolved.random_seed is not None and (
        isinstance(resolved.random_seed, bool) or not isinstance(resolved.random_seed, int)
    ):
        raise ContractError("random_seed must be an int or None")
    return resolved


def _scalar(value: Any) -> float:
    torch_module = _require_torch()
    if torch_module.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _component_payload(ledger: FullEstimatorObjectiveLedger, component_name: str) -> dict[str, Any]:
    component = ledger.components[component_name]
    return {
        "raw": _scalar(component.raw),
        "scale": _scalar(component.scale),
        "normalized": _scalar(component.normalized),
        "floor_used": bool(component.floor_used),
    }


def _build_successful_provenance(
    ledger: FullEstimatorObjectiveLedger,
    *,
    optimizer_config: FullEstimatorOptimizerConfig,
) -> STRIDEFitProvenance:
    initialization = ledger.initialization
    state_geometry = dict(ledger.metadata.get("state_geometry", {}))
    provenance_payload: dict[str, Any] = {
        "provenance_schema_version": STRIDE_FIT_PROVENANCE_SCHEMA_VERSION,
        "alpha": float(ledger.alpha),
        "random_seed": optimizer_config.random_seed,
        "initialization": {
            "policy": "identity_plus_small_open",
            "delta_init": float(initialization.delta_init),
            "K": int(initialization.K),
            "dtype": str(initialization.dtype),
        },
        "loss": {
            "total": _scalar(ledger.total),
            "local": _scalar(ledger.local),
            "regularization": _scalar(ledger.regularization),
            "epsilon_norm": float(ledger.epsilon_norm),
            "local_denominator": 3,
            "regularization_denominator": 2,
            "components": {
                name: _component_payload(ledger, name)
                for name in ("obs", "open", "geometry", "consistency", "recurrence")
            },
        },
        "e_bounds": list(ledger.metadata.get("e_bounds", (0.0, 1.0))),
        "post_reconstruction_form": ledger.metadata.get(
            "post_reconstruction_form",
            "normalize(q_minus @ A + e)",
        ),
        "observation_comparison_plan": dict(ledger.metadata["observation_comparison_plan"]),
        "observation_discrepancy": dict(ledger.metadata["observation_discrepancy"]),
        "state_geometry": {
            "normalization": state_geometry.get("normalization", "C_norm = C_raw / s_C"),
            "s_C": float(state_geometry["s_C"]),
        },
        "optimizer": {
            "framework": "torch",
            "algorithm": "AdamW",
            "weight_decay": 0.0,
            "scheduler_policy": str(optimizer_config.scheduler_policy),
        },
        "recurrence": {
            "support_n_patients": int(ledger.recurrence.support_n_patients),
            "dispersion": _scalar(ledger.recurrence.dispersion),
        },
        "detailed_optimizer_trace": bool(optimizer_config.detailed_optimizer_trace),
    }
    if optimizer_config.detailed_optimizer_trace:
        provenance_payload["optimizer_trace_ref"] = "optimizer_trace:result.trace"
    if ledger.ablation_mode != "none":
        provenance_payload.update(
            {
                "ablation_mode": ledger.ablation_mode,
                "ablation_term_handling": ledger.ablation_term_handling
                or ABLATION_TERM_HANDLING,
                "ablation_denominator_policy": ledger.metadata[
                    "ablation_denominator_policy"
                ],
            }
        )
    return build_stride_fit_provenance(provenance_payload)


def _finite_loss(value: Any) -> bool:
    torch_module = _require_torch()
    return bool(torch_module.isfinite(value.detach()).cpu().item())


def _gradient_norm(parameters: Sequence[Any]) -> float | None:
    torch_module = _require_torch()
    squared_norm = torch_module.zeros((), dtype=torch_module.float64)
    saw_gradient = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        saw_gradient = True
        grad = parameter.grad.detach().to(dtype=torch_module.float64)
        if not bool(torch_module.isfinite(grad).all().cpu().item()):
            return float("nan")
        squared_norm = squared_norm.to(device=grad.device) + torch_module.sum(grad * grad)
    if not saw_gradient:
        return None
    return float(torch_module.sqrt(squared_norm).detach().cpu().item())


def _completion_reason(
    *,
    absolute_improvement: float,
    relative_improvement: float,
    gradient_norm: float | None,
    config: FullEstimatorOptimizerConfig,
) -> str | None:
    if (
        gradient_norm is not None
        and float(config.gradient_norm_tol) > 0.0
        and gradient_norm <= float(config.gradient_norm_tol)
    ):
        return "gradient_norm_met"
    if float(config.convergence_tol) > 0.0 and abs(absolute_improvement) <= float(
        config.convergence_tol
    ):
        return "absolute_objective_delta_met"
    if float(config.min_relative_improvement) > 0.0 and abs(
        relative_improvement
    ) <= float(config.min_relative_improvement):
        return "relative_objective_delta_met"
    return None


def _clone_leaf_unconstrained(
    unconstrained: FullEstimatorUnconstrainedParameters,
) -> FullEstimatorUnconstrainedParameters:
    row_logits = unconstrained.row_logits.clone().detach().requires_grad_(True)
    e_logits = unconstrained.e_logits.clone().detach().requires_grad_(True)
    return FullEstimatorUnconstrainedParameters(
        patient_ids=unconstrained.patient_ids,
        row_logits=row_logits,
        e_logits=e_logits,
    )


def optimize_full_estimator(
    *,
    patient_ids: Sequence[str],
    K: int,
    evidence_blocks: Sequence[FullEstimatorEvidenceBlock],
    geometry: StateGeometry,
    config: FullEstimatorOptimizerConfig | None = None,
) -> FullEstimatorOptimizerResult:
    """Run the canonical full estimator with PyTorch AdamW."""
    torch_module = _require_torch()
    resolved_config = _validate_config(config)
    if resolved_config.random_seed is not None:
        torch_module.manual_seed(int(resolved_config.random_seed))

    unconstrained = _clone_leaf_unconstrained(
        unconstrained_from_initialization(patient_ids, K)
    )
    optimizer = torch_module.optim.AdamW(
        [unconstrained.row_logits, unconstrained.e_logits],
        lr=float(resolved_config.learning_rate),
        weight_decay=0.0,
    )
    fixed_cache = FullEstimatorObjectiveFixedCache.build(
        params=parameters_from_unconstrained(unconstrained),
        evidence_blocks=evidence_blocks,
        geometry=geometry,
        epsilon_norm=float(resolved_config.epsilon_norm),
        config=resolved_config.observation_config,
    )
    scheduler = None
    if resolved_config.scheduler_policy == "ReduceLROnPlateau_on_total_objective":
        scheduler = torch_module.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
        )

    trace_steps: list[dict[str, Any]] = []
    initial_total: float | None = None
    previous_total: float | None = None
    completed_steps = 0
    non_improving_steps = 0
    try:
        while True:
            optimizer.zero_grad(set_to_none=True)
            params = parameters_from_unconstrained(unconstrained)
            ledger = compute_full_estimator_objective(
                params,
                evidence_blocks,
                geometry,
                alpha=float(resolved_config.alpha),
                epsilon_norm=float(resolved_config.epsilon_norm),
                ablation_mode=str(resolved_config.ablation_mode),
                config=resolved_config.observation_config,
                fixed_cache=fixed_cache,
            )
            if not _finite_loss(ledger.total):
                return FullEstimatorOptimizerResult(
                    status="failed",
                    diagnostics={
                        "optimizer_status": "failed",
                        "failure_reason": "nonfinite_total_objective",
                        "step": completed_steps,
                    },
                )
            current_total = _scalar(ledger.total)
            if initial_total is None:
                initial_total = current_total
            trace_record = {
                "step": completed_steps,
                "total": current_total,
                "local": _scalar(ledger.local),
                "regularization": _scalar(ledger.regularization),
            }
            if resolved_config.detailed_optimizer_trace:
                trace_steps.append(trace_record)

            absolute_improvement = 0.0
            relative_improvement = 0.0
            if previous_total is not None:
                absolute_improvement = previous_total - current_total
                relative_improvement = absolute_improvement / max(abs(previous_total), 1e-12)
                if completed_steps >= int(resolved_config.min_steps):
                    reason = _completion_reason(
                        absolute_improvement=absolute_improvement,
                        relative_improvement=relative_improvement,
                        gradient_norm=None,
                        config=resolved_config,
                    )
                    if reason is not None:
                        provenance = _build_successful_provenance(
                            ledger,
                            optimizer_config=resolved_config,
                        )
                        trace: Mapping[str, Any] = {
                            "initial_total": initial_total,
                            "final_total": current_total,
                            "n_steps": completed_steps,
                        }
                        if resolved_config.detailed_optimizer_trace:
                            trace = {**trace, "steps": tuple(trace_steps)}
                        return FullEstimatorOptimizerResult(
                            status="ok",
                            parameters=params,
                            final_ledger=ledger,
                            provenance=provenance,
                            diagnostics={
                                "optimizer_status": "ok",
                                "completion_reason": reason,
                                "scheduler_policy": str(resolved_config.scheduler_policy),
                                "n_steps": completed_steps,
                                "initial_total": initial_total,
                                "final_total": current_total,
                                "absolute_improvement": absolute_improvement,
                                "relative_improvement": relative_improvement,
                                "ablation_mode": str(resolved_config.ablation_mode),
                            },
                            trace=trace,
                        )
                if absolute_improvement <= 0.0:
                    non_improving_steps += 1
                else:
                    non_improving_steps = 0
                if (
                    completed_steps >= int(resolved_config.min_steps)
                    and non_improving_steps >= int(resolved_config.patience)
                ):
                    return FullEstimatorOptimizerResult(
                        status="deferred",
                        diagnostics={
                            "optimizer_status": "deferred",
                            "defer_reason": "insufficient_objective_improvement",
                            "n_steps": completed_steps,
                            "initial_total": initial_total,
                            "final_total": current_total,
                        },
                        trace={
                            "initial_total": initial_total,
                            "final_total": current_total,
                            "n_steps": completed_steps,
                            **(
                                {"steps": tuple(trace_steps)}
                                if resolved_config.detailed_optimizer_trace
                                else {}
                            ),
                        },
                    )

            if completed_steps >= int(resolved_config.max_steps):
                return FullEstimatorOptimizerResult(
                    status="deferred",
                    diagnostics={
                        "optimizer_status": "deferred",
                        "defer_reason": "max_steps_exhausted_without_completion",
                        "n_steps": completed_steps,
                        "initial_total": initial_total,
                        "final_total": current_total,
                        "absolute_improvement": absolute_improvement,
                        "relative_improvement": relative_improvement,
                    },
                    trace={
                        "initial_total": initial_total,
                        "final_total": current_total,
                        "n_steps": completed_steps,
                        **(
                            {"steps": tuple(trace_steps)}
                            if resolved_config.detailed_optimizer_trace
                            else {}
                        ),
                    },
                )

            ledger.total.backward()
            grad_norm = _gradient_norm([unconstrained.row_logits, unconstrained.e_logits])
            if grad_norm is None or grad_norm <= 0.0:
                return FullEstimatorOptimizerResult(
                    status="failed",
                    diagnostics={
                        "optimizer_status": "failed",
                        "failure_reason": "missing_or_zero_gradient",
                        "n_steps": completed_steps,
                        "initial_total": initial_total,
                        "final_total": current_total,
                    },
                )
            if not math.isfinite(grad_norm):
                return FullEstimatorOptimizerResult(
                    status="failed",
                    diagnostics={
                        "optimizer_status": "failed",
                        "failure_reason": "nonfinite_gradient",
                        "n_steps": completed_steps,
                        "initial_total": initial_total,
                        "final_total": current_total,
                    },
                )
            if completed_steps >= int(resolved_config.min_steps):
                reason = _completion_reason(
                    absolute_improvement=absolute_improvement,
                    relative_improvement=relative_improvement,
                    gradient_norm=grad_norm,
                    config=resolved_config,
                )
                if reason == "gradient_norm_met":
                    provenance = _build_successful_provenance(
                        ledger,
                        optimizer_config=resolved_config,
                    )
                    trace: Mapping[str, Any] = {
                        "initial_total": initial_total,
                        "final_total": current_total,
                        "n_steps": completed_steps,
                    }
                    if resolved_config.detailed_optimizer_trace:
                        trace = {**trace, "steps": tuple(trace_steps)}
                    return FullEstimatorOptimizerResult(
                        status="ok",
                        parameters=params,
                        final_ledger=ledger,
                        provenance=provenance,
                        diagnostics={
                            "optimizer_status": "ok",
                            "completion_reason": reason,
                            "scheduler_policy": str(resolved_config.scheduler_policy),
                            "n_steps": completed_steps,
                            "initial_total": initial_total,
                            "final_total": current_total,
                            "gradient_norm": grad_norm,
                            "ablation_mode": str(resolved_config.ablation_mode),
                        },
                        trace=trace,
                    )
            optimizer.step()
            completed_steps += 1
            previous_total = current_total
            if scheduler is not None:
                scheduler.step(float(ledger.total.detach().cpu()))
    except RuntimeError as exc:
        return FullEstimatorOptimizerResult(
            status="failed",
            diagnostics={
                "optimizer_status": "failed",
                "failure_reason": "runtime_error",
                "message": str(exc),
            },
        )



__all__ = [
    "FullEstimatorOptimizerConfig",
    "FullEstimatorOptimizerResult",
    "optimize_full_estimator",
]
