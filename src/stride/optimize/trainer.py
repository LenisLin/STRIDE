"""Torch optimizer runtime for STRIDE relation fitting."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
from typing import Any

from ..geometry.state_geometry import StateGeometry
from ..losses.assembly import ADEState, EvidenceBlock, LossLedger, ObjectiveContext, compute_loss_ledger
from .config import TrainConfig, validate_train_config
from .model import RelationModel, require_torch
from .result import OptimizerRunInfo, TrainResult


def _scalar(value: Any) -> float:
    torch_module = require_torch()
    if torch_module.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _finite_loss(value: Any) -> bool:
    try:
        return math.isfinite(_scalar(value))
    except (TypeError, ValueError, RuntimeError):
        return False


def _plateau_condition_met(
    *,
    absolute_improvement: float,
    relative_improvement: float,
    config: TrainConfig,
) -> bool:
    if float(config.convergence_tol) > 0.0 and abs(absolute_improvement) <= float(config.convergence_tol):
        return True
    if (
        float(config.min_relative_improvement) > 0.0
        and abs(relative_improvement) <= float(config.min_relative_improvement)
    ):
        return True
    return False


def _detach_state(state: ADEState) -> tuple[Any, Any, Any]:
    return (
        state.A.detach().clone(),
        state.d.detach().clone(),
        state.e.detach().clone(),
    )


def _detach_value(value: Any) -> Any:
    torch_module = require_torch()
    if torch_module.is_tensor(value):
        return value.detach().clone()
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                field.name: _detach_value(getattr(value, field.name))
                for field in fields(value)
            },
        )
    if isinstance(value, Mapping):
        return {key: _detach_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_value(item) for item in value)
    if isinstance(value, list):
        return [_detach_value(item) for item in value]
    return value


def _detach_ledger(ledger: LossLedger) -> LossLedger:
    return _detach_value(ledger)


def _trace_payload(
    *,
    initial_total: float | None,
    final_total: float | None,
    n_steps: int,
    warmup_steps_completed: int,
    main_steps_completed: int,
    trace_steps: Sequence[Mapping[str, Any]],
    config: TrainConfig,
) -> Mapping[str, Any]:
    trace: Mapping[str, Any] = {
        "initial_total": initial_total,
        "final_total": final_total,
        "n_steps": n_steps,
        "warmup_steps_completed": warmup_steps_completed,
        "main_steps_completed": main_steps_completed,
        "optimizer_protocol": config.schedule.protocol_name,
    }
    if config.detailed_trace:
        trace = {**trace, "steps": tuple(trace_steps)}
    return trace


def _run_info(
    *,
    reason: str,
    config: TrainConfig,
    optimizer_exit_flag: str | None = None,
    n_steps: int | None = None,
    warmup_steps_completed: int | None = None,
    main_steps_completed: int | None = None,
    step: int | None = None,
    initial_total: float | None = None,
    final_total: float | None = None,
    absolute_improvement: float | None = None,
    relative_improvement: float | None = None,
    message: str | None = None,
) -> OptimizerRunInfo:
    return OptimizerRunInfo(
        reason=reason,
        optimizer_exit_flag=optimizer_exit_flag,
        n_steps=n_steps,
        warmup_steps_completed=warmup_steps_completed,
        main_steps_completed=main_steps_completed,
        step=step,
        initial_total=initial_total,
        final_total=final_total,
        absolute_improvement=absolute_improvement,
        relative_improvement=relative_improvement,
        optimizer_protocol=str(config.schedule.protocol_name),
        scheduler_policy=str(config.schedule.main_stage.scheduler_policy),
        ablation_mode=str(config.ablation_mode),
        message=message,
    )


def _set_optimizer_lr(optimizer: Any, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _build_main_scheduler(optimizer: Any, config: TrainConfig) -> Any:
    torch_module = require_torch()
    cosine = config.schedule.cosine
    return torch_module.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cosine.T_max),
        eta_min=float(cosine.eta_min),
    )


def _compute_ledger(
    *,
    model: RelationModel,
    evidence_blocks: Sequence[EvidenceBlock],
    geometry: StateGeometry,
    config: TrainConfig,
    objective_context: ObjectiveContext,
) -> tuple[ADEState, LossLedger]:
    state = model.state()
    ledger = compute_loss_ledger(
        state,
        evidence_blocks,
        geometry,
        objective_weights=config.objective_weights,
        epsilon_norm=float(config.epsilon_norm),
        ablation_mode=str(config.ablation_mode),
        config=config.observation_config,
        objective_context=objective_context,
    )
    return state, ledger


def _trace_step_payload(
    *,
    global_step: int,
    stage_name: str,
    stage_step: int,
    total: float,
    ledger: LossLedger,
) -> dict[str, Any]:
    return {
        "step": global_step,
        "stage": stage_name,
        "stage_step": stage_step,
        "total": total,
        "fit": _scalar(ledger.fit),
        "prior": _scalar(ledger.prior),
        "cohort": _scalar(ledger.cohort),
    }


def _completed_result(
    *,
    status: str,
    reason: str,
    config: TrainConfig,
    optimizer_exit_flag: str,
    state: ADEState,
    ledger: LossLedger,
    total_steps_completed: int,
    warmup_steps_completed: int,
    main_steps_completed: int,
    initial_total: float | None,
    final_total: float | None,
    trace_steps: Sequence[Mapping[str, Any]],
    absolute_improvement: float | None = None,
    relative_improvement: float | None = None,
) -> TrainResult:
    A, d, e = _detach_state(state)
    return TrainResult(
        status=status,
        A=A if status == "ok" else None,
        d=d if status == "ok" else None,
        e=e if status == "ok" else None,
        loss_ledger=_detach_ledger(ledger) if status == "ok" else None,
        train_config=config,
        run_info=_run_info(
            reason=reason,
            config=config,
            optimizer_exit_flag=optimizer_exit_flag,
            n_steps=total_steps_completed,
            warmup_steps_completed=warmup_steps_completed,
            main_steps_completed=main_steps_completed,
            initial_total=initial_total,
            final_total=final_total,
            absolute_improvement=absolute_improvement,
            relative_improvement=relative_improvement,
        ),
        trace=_trace_payload(
            initial_total=initial_total,
            final_total=final_total,
            n_steps=total_steps_completed,
            warmup_steps_completed=warmup_steps_completed,
            main_steps_completed=main_steps_completed,
            trace_steps=trace_steps,
            config=config,
        ),
    )


def run_training(
    *,
    patient_ids: Sequence[str],
    K: int,
    evidence_blocks: Sequence[EvidenceBlock],
    geometry: StateGeometry,
    config: TrainConfig | None = None,
) -> TrainResult:
    """Run the canonical STRIDE optimizer with the fixed two-phase reference protocol."""
    torch_module = require_torch()
    resolved_config = validate_train_config(config)
    if resolved_config.seed is not None:
        torch_module.manual_seed(int(resolved_config.seed))

    schedule = resolved_config.schedule
    model = RelationModel(patient_ids=patient_ids, K=K, device=resolved_config.device)
    optimizer = torch_module.optim.AdamW(
        [model.row_logits, model.e_logits],
        lr=float(schedule.warmup_stage.lr),
        weight_decay=0.0,
    )
    objective_context = ObjectiveContext.build(
        params=model.state(),
        evidence_blocks=evidence_blocks,
        geometry=geometry,
        epsilon_norm=float(resolved_config.epsilon_norm),
        config=resolved_config.observation_config,
    )

    trace_steps: list[dict[str, Any]] = []
    initial_total: float | None = None
    total_steps_completed = 0
    warmup_steps_completed = 0
    main_steps_completed = 0
    non_improving_main_steps = 0
    previous_main_total: float | None = None
    last_total: float | None = None

    try:
        for warmup_step in range(1, int(schedule.warmup_stage.max_steps) + 1):
            optimizer.zero_grad(set_to_none=True)
            state, ledger = _compute_ledger(
                model=model,
                evidence_blocks=evidence_blocks,
                geometry=geometry,
                config=resolved_config,
                objective_context=objective_context,
            )
            if not _finite_loss(ledger.total):
                return TrainResult(
                    status="failed",
                    train_config=resolved_config,
                    run_info=_run_info(
                        reason="nonfinite_total_objective",
                        config=resolved_config,
                        n_steps=total_steps_completed,
                        warmup_steps_completed=warmup_steps_completed,
                        main_steps_completed=main_steps_completed,
                        step=total_steps_completed,
                        initial_total=initial_total,
                        final_total=last_total,
                    ),
                )

            current_total = _scalar(ledger.total)
            if initial_total is None:
                initial_total = current_total
            if resolved_config.detailed_trace:
                trace_steps.append(
                    _trace_step_payload(
                        global_step=total_steps_completed,
                        stage_name=schedule.warmup_stage.name,
                        stage_step=warmup_step,
                        total=current_total,
                        ledger=ledger,
                    )
                )

            ledger.total.backward()
            optimizer.step()
            total_steps_completed += 1
            warmup_steps_completed += 1
            last_total = current_total

        _set_optimizer_lr(optimizer, float(schedule.main_stage.lr))
        scheduler = _build_main_scheduler(optimizer, resolved_config)
        for main_step in range(1, int(schedule.main_stage.max_steps) + 1):
            optimizer.zero_grad(set_to_none=True)
            state, ledger = _compute_ledger(
                model=model,
                evidence_blocks=evidence_blocks,
                geometry=geometry,
                config=resolved_config,
                objective_context=objective_context,
            )
            if not _finite_loss(ledger.total):
                return TrainResult(
                    status="failed",
                    train_config=resolved_config,
                    run_info=_run_info(
                        reason="nonfinite_total_objective",
                        config=resolved_config,
                        n_steps=total_steps_completed,
                        warmup_steps_completed=warmup_steps_completed,
                        main_steps_completed=main_steps_completed,
                        step=total_steps_completed,
                        initial_total=initial_total,
                        final_total=last_total,
                    ),
                )

            current_total = _scalar(ledger.total)
            if initial_total is None:
                initial_total = current_total
            if resolved_config.detailed_trace:
                trace_steps.append(
                    _trace_step_payload(
                        global_step=total_steps_completed,
                        stage_name=schedule.main_stage.name,
                        stage_step=main_step,
                        total=current_total,
                        ledger=ledger,
                    )
                )

            absolute_improvement: float | None = None
            relative_improvement: float | None = None
            if previous_main_total is not None:
                absolute_improvement = previous_main_total - current_total
                relative_improvement = absolute_improvement / max(abs(previous_main_total), 1e-12)
                if main_steps_completed >= int(schedule.main_stage.min_steps):
                    plateau_condition_met = _plateau_condition_met(
                        absolute_improvement=absolute_improvement,
                        relative_improvement=relative_improvement,
                        config=resolved_config,
                    )
                    if plateau_condition_met:
                        non_improving_main_steps += 1
                    else:
                        non_improving_main_steps = 0

                    if non_improving_main_steps >= int(resolved_config.patience):
                        return _completed_result(
                            status="ok",
                            reason="plateau_patience",
                            config=resolved_config,
                            optimizer_exit_flag="plateau_patience",
                            state=state,
                            ledger=ledger,
                            total_steps_completed=total_steps_completed,
                            warmup_steps_completed=warmup_steps_completed,
                            main_steps_completed=main_steps_completed,
                            initial_total=initial_total,
                            final_total=current_total,
                            trace_steps=trace_steps,
                            absolute_improvement=absolute_improvement,
                            relative_improvement=relative_improvement,
                        )

            ledger.total.backward()
            optimizer.step()
            scheduler.step()
            total_steps_completed += 1
            main_steps_completed += 1
            last_total = current_total
            previous_main_total = current_total

        with torch_module.no_grad():
            final_state, final_ledger = _compute_ledger(
                model=model,
                evidence_blocks=evidence_blocks,
                geometry=geometry,
                config=resolved_config,
                objective_context=objective_context,
            )
        if not _finite_loss(final_ledger.total):
            return TrainResult(
                status="failed",
                train_config=resolved_config,
                run_info=_run_info(
                    reason="nonfinite_total_objective_after_final_step",
                    config=resolved_config,
                    n_steps=total_steps_completed,
                    warmup_steps_completed=warmup_steps_completed,
                    main_steps_completed=main_steps_completed,
                    initial_total=initial_total,
                    final_total=last_total,
                ),
            )

        return _completed_result(
            status="ok",
            reason="max_steps_exhausted_finite",
            config=resolved_config,
            optimizer_exit_flag="max_steps_exhausted_finite",
            state=final_state,
            ledger=final_ledger,
            total_steps_completed=total_steps_completed,
            warmup_steps_completed=warmup_steps_completed,
            main_steps_completed=main_steps_completed,
            initial_total=initial_total,
            final_total=_scalar(final_ledger.total),
            trace_steps=trace_steps,
        )
    except RuntimeError as exc:
        return TrainResult(
            status="failed",
            train_config=resolved_config,
            run_info=_run_info(
                reason="runtime_error",
                config=resolved_config,
                n_steps=total_steps_completed,
                warmup_steps_completed=warmup_steps_completed,
                main_steps_completed=main_steps_completed,
                initial_total=initial_total,
                final_total=last_total,
                message=str(exc),
            ),
        )


__all__ = ["run_training"]
