"""One-relation training runtime for STRIDE `.tl`.

This module owns the one-relation training runtime scaffold around the canonical
objective. It uses `_optimizer.py` for AdamW, scheduler primitives, and fixed
protocol facts, `_parameters.py` for trainable constrained variables, and
`_losses.py` for objective evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from stride.errors import ContractError

from ._losses import (
    EPSILON_NORM,
    LossContext,
    LossLedger,
    _compute_geometry_loss,
    compute_total_loss,
)
from ._optimizer import (
    MAIN_LR,
    MAIN_MAX_STEPS,
    MAIN_MIN_STEPS,
    PLATEAU_PATIENCE,
    WARMUP_LR,
    WARMUP_STEPS,
    create_adamw,
    create_main_scheduler,
    optimizer_handoff,
)
from ._parameters import (
    ParameterLogits,
    RelationParameters,
    constrain_parameters,
    initialize_parameters,
    predict_target_composition,
)
from ._resolve import EvidenceBlock, RelationInput
from ._sinkhorn import (
    SinkhornConfig,
    compute_fov_ground_cost_matrix,
    compute_observed_self_ground_cost,
)

_OPTIMIZER_HANDOFF = optimizer_handoff()


@dataclass(frozen=True)
class TrainingRunInfo:
    """Training-loop terminal facts; this is not an audit record."""

    reason: str | None = None
    optimizer_exit_flag: str | None = None
    n_steps: int | None = None
    warmup_steps_completed: int | None = None
    main_steps_completed: int | None = None
    initial_total: float | None = None
    final_total: float | None = None
    absolute_improvement: float | None = None
    relative_improvement: float | None = None
    optimizer_protocol: str = _OPTIMIZER_HANDOFF.protocol_name
    scheduler_policy: str = _OPTIMIZER_HANDOFF.scheduler_policy
    random_seed: int | None = None
    message: str | None = None


@dataclass(frozen=True)
class TrainingResult:
    """Internal handoff from training runtime to output assembly."""

    parameters: RelationParameters | None = None
    loss_ledger: LossLedger | None = None
    run_info: TrainingRunInfo = field(default_factory=TrainingRunInfo)
    trace: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _PreparedRelationFit:
    """One-time setup handoff for a one-relation training run."""

    relation: RelationInput
    blocks: tuple[EvidenceBlock, ...]
    cost_matrix: torch.Tensor
    cost_scale: float
    context: LossContext


class RelationModel(torch.nn.Module):
    """Trainable logits whose forward pass returns constrained `A`, `d`, and `e`.

    patient_ids: patient ids aligned to training axis `P`.
    row_logits: trainable logits for row-simplex `[A_i,* , d_i]`.
    e_logits: trainable logits for bounded `e`.
    """

    def __init__(self, initial_logits: ParameterLogits) -> None:
        """Register trainable relation logits from a prepared initialization."""
        super().__init__()
        if not isinstance(initial_logits, ParameterLogits):
            raise TypeError("initial_logits must be a ParameterLogits object")

        self.patient_ids = tuple(initial_logits.patient_ids)
        self.row_logits = torch.nn.Parameter(
            initial_logits.row_logits.detach().clone().to(dtype=torch.float64)
        )
        self.e_logits = torch.nn.Parameter(
            initial_logits.e_logits.detach()
            .clone()
            .to(
                device=self.row_logits.device,
                dtype=torch.float64,
            )
        )

    def logit_state(self) -> ParameterLogits:
        """Return the current unconstrained training state."""
        return ParameterLogits(
            patient_ids=self.patient_ids,
            row_logits=self.row_logits,
            e_logits=self.e_logits,
        )

    def state(self) -> RelationParameters:
        """Return constrained patient-level `A/d/e` parameters."""
        return constrain_parameters(self.logit_state())

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return constrained `A`, `d`, and `e` tensors for torch optimization."""
        parameters = self.state()
        return parameters.A, parameters.d, parameters.e


def train_relation(
    relation: RelationInput,
    cost_matrix: torch.Tensor,
    cost_scale: float,
    *,
    device: Any = "cuda:0",
) -> TrainingResult:
    """Train one resolved relation and return an output handoff.

    Purpose:
        Build the one-time relation-fit setup, use `_optimizer.py` primitives
        for AdamW and scheduler control, and return a `TrainingResult` handoff
        for `_output.py`.

    Key variables:
        runtime_device: fixed training runtime device.
        model: trainable relation logits for this relation.
        optimizer: AdamW optimizer with `weight_decay=0.0`.
        loss_ledger: current canonical objective ledger.
        run_info: training-loop terminal facts.

    Boundary:
        This function does not assemble public results.
    """
    # relation: resolved evidence blocks for exactly one declared relation.
    runtime_device = _resolve_runtime_device(device)

    materialized_relation, blocks, cost, scale = _materialize_relation_inputs_once(
        relation,
        cost_matrix,
        cost_scale,
        device=runtime_device,
    )
    context = _build_loss_context_once(
        relation=materialized_relation,
        blocks=blocks,
        cost_matrix=cost,
        cost_scale=scale,
    )
    prepared = _PreparedRelationFit(
        relation=materialized_relation,
        blocks=blocks,
        cost_matrix=cost,
        cost_scale=scale,
        context=context,
    )

    K = int(prepared.cost_matrix.shape[0])
    initial_logits = initialize_parameters(
        prepared.relation.patient_ids,
        K,
        device=prepared.cost_matrix.device,
    )
    model = RelationModel(initial_logits)
    optimizer = create_adamw(model.parameters(), lr=WARMUP_LR)

    initial_total: float | None = None
    latest_total: float | None = None
    total_steps = 0

    for warmup_step in range(1, int(WARMUP_STEPS) + 1):
        ledger, total = _training_step(
            model,
            optimizer,
            prepared,
            stage="warmup",
            step_index=warmup_step,
        )
        if initial_total is None:
            initial_total = total
        latest_total = total
        total_steps += 1

    _set_optimizer_lr(optimizer, MAIN_LR)
    scheduler = create_main_scheduler(optimizer)
    plateau_count = 0
    previous_total = latest_total
    terminal_reason = "max_steps"
    optimizer_exit_flag = "max_steps_exhausted_finite"
    main_steps_completed = 0

    for _main_step in range(1, int(MAIN_MAX_STEPS) + 1):
        optimizer.zero_grad(set_to_none=True)
        params = model.state()
        ledger = compute_total_loss(
            params,
            prepared.blocks,
            prepared.cost_matrix,
            prepared.cost_scale,
            context=prepared.context,
        )
        total = _scalar_total(ledger.total, name="training total objective")
        if initial_total is None:
            initial_total = total
        ledger.total.backward()
        optimizer.step()
        scheduler.step()

        absolute_improvement, relative_improvement = _step_improvements(
            previous_total,
            total,
        )
        latest_total = total
        previous_total = total
        total_steps += 1
        main_steps_completed += 1

        if main_steps_completed >= int(MAIN_MIN_STEPS) and _plateau_condition_met(
            absolute_improvement=absolute_improvement,
            relative_improvement=relative_improvement,
        ):
            plateau_count += 1
        else:
            plateau_count = 0

        if plateau_count >= int(PLATEAU_PATIENCE):
            terminal_reason = "plateau_patience"
            optimizer_exit_flag = "plateau_patience"
            break

    if latest_total is None or initial_total is None:
        raise ContractError("training produced no objective evaluation")

    final_params, final_ledger, final_total = _final_state(model, prepared)
    absolute_improvement, relative_improvement = _run_improvements(
        initial_total,
        final_total,
    )
    run_info = TrainingRunInfo(
        reason=terminal_reason,
        optimizer_exit_flag=optimizer_exit_flag,
        n_steps=total_steps,
        warmup_steps_completed=int(WARMUP_STEPS),
        main_steps_completed=main_steps_completed,
        initial_total=initial_total,
        final_total=final_total,
        absolute_improvement=absolute_improvement,
        relative_improvement=relative_improvement,
        random_seed=None,
        message=(
            "Finite training objective reached plateau patience"
            if terminal_reason == "plateau_patience"
            else "Finite training objective reached main-step cap"
        ),
    )
    return TrainingResult(
        parameters=final_params,
        loss_ledger=final_ledger,
        run_info=run_info,
        trace=None,
    )


def _resolve_runtime_device(device: Any) -> torch.device:
    """Resolve the requested runtime device once before relation materialization."""
    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ContractError(f"invalid runtime device: {device!r}") from exc

    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ContractError(f"requested runtime device is unavailable: {resolved}")
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise ContractError(f"requested runtime device is unavailable: {resolved}")
        return resolved

    if resolved.type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ContractError(f"requested runtime device is unavailable: {resolved}")
        try:
            torch.empty((), dtype=torch.float64, device=resolved)
        except (TypeError, RuntimeError) as exc:
            raise ContractError(
                f"requested runtime device does not support float64 tensors: {resolved}"
            ) from exc
        return resolved

    return resolved


def _materialize_relation_inputs_once(
    relation: RelationInput,
    cost_matrix: torch.Tensor,
    cost_scale: float,
    *,
    device: Any | None,
) -> tuple[RelationInput, tuple[EvidenceBlock, ...], torch.Tensor, float]:
    """Materialize one relation's evidence and cost once for training setup.

    This is a boundary check and device placement helper, not an audit module.
    It returns new evidence blocks so repeated objective calls no longer move
    relation evidence across devices.
    """
    if not isinstance(relation, RelationInput):
        raise ContractError("relation must be a RelationInput object")
    if len(relation.patient_ids) == 0:
        raise ContractError("relation.patient_ids must be non-empty")
    if len(relation.blocks) == 0:
        raise ContractError("relation.blocks must be non-empty")

    scale = _positive_finite_float(cost_scale, name="cost_scale")
    try:
        cost = torch.as_tensor(cost_matrix, dtype=torch.float64, device=device)
    except (TypeError, ValueError) as exc:
        raise ContractError("cost_matrix must be coercible to a float64 tensor") from exc
    if cost.ndim != 2 or cost.shape[0] <= 0 or cost.shape[0] != cost.shape[1]:
        raise ContractError("cost_matrix must be a non-empty square [K, K] tensor")

    materialized_blocks: list[EvidenceBlock] = []
    K = int(cost.shape[0])
    for block in relation.blocks:
        if not isinstance(block, EvidenceBlock):
            raise ContractError("relation.blocks must contain EvidenceBlock objects")
        source = _as_float64_matrix_on_device(
            block.source_bag,
            name="source_bag",
            device=cost.device,
        )
        target = _as_float64_matrix_on_device(
            block.target_bag,
            name="target_bag",
            device=cost.device,
        )
        if source.shape[1] != target.shape[1]:
            raise ContractError("source_bag and target_bag must share the K-state axis")
        if source.shape[1] != K:
            raise ContractError("evidence bags must align with cost_matrix K-state axis")
        materialized_blocks.append(
            EvidenceBlock(
                patient_id=str(block.patient_id),
                source_bag=source,
                target_bag=target,
                block_id=str(block.block_id),
                metadata=block.metadata,
            )
        )

    materialized_relation = RelationInput(
        relation_id=relation.relation_id,
        source_timepoint=relation.source_timepoint,
        target_timepoint=relation.target_timepoint,
        source_domain=relation.source_domain,
        target_domain=relation.target_domain,
        patient_ids=tuple(str(item) for item in relation.patient_ids),
        support_counts=relation.support_counts,
        skipped_patient_ids=relation.skipped_patient_ids,
        blocks=tuple(materialized_blocks),
        metadata=relation.metadata,
    )
    return materialized_relation, tuple(materialized_blocks), cost, scale


def _scale_initial_parameters(
    patient_ids: Sequence[str],
    n_states: int,
    *,
    device: Any,
) -> RelationParameters:
    """Return objective-scale initialization, not optimizer-start logits.

    This deterministic identity-plus-small-open object has no off-diagonal
    seed. It is used only for objective scale computation and fixed setup
    quantities such as `s_G_init`.
    """
    normalized_patient_ids = _normalize_patient_ids(patient_ids)
    K = _positive_int(n_states, name="n_states")
    P = len(normalized_patient_ids)
    delta_init = min(0.05, 1.0 / float(K + 1))

    A = (1.0 - delta_init) * torch.eye(K, dtype=torch.float64, device=device)
    A = A.unsqueeze(0).repeat(P, 1, 1)
    d = torch.full((P, K), delta_init, dtype=torch.float64, device=device)
    e = torch.full((P, K), delta_init / float(K), dtype=torch.float64, device=device)
    return RelationParameters(patient_ids=normalized_patient_ids, A=A, d=d, e=e)


def _compute_block_fov_cost_scale(
    block: EvidenceBlock,
    *,
    parameters: RelationParameters,
    cost_matrix: torch.Tensor,
    cost_scale: float,
    sinkhorn_config: SinkhornConfig,
) -> tuple[float, bool, torch.Tensor, bool]:
    """Compute fixed per-block `s_G_init` and observed-self cache."""
    patient_id = str(block.patient_id)
    try:
        pidx = parameters.patient_ids.index(patient_id)
    except ValueError as exc:
        raise ContractError("block patient_id must align with relation patient_ids") from exc

    predicted_init = predict_target_composition(
        block.source_bag,
        parameters.A[pidx],
        parameters.e[pidx],
    )
    ground = compute_fov_ground_cost_matrix(
        predicted_init,
        block.target_bag,
        cost_matrix,
        cost_scale,
        validate_inputs=True,
        collect_warnings=False,
        config=sinkhorn_config,
    )
    positive = ground.value[(ground.value > 0.0) & torch.isfinite(ground.value)]
    if positive.numel() == 0:
        fov_cost_scale = 1.0
        floor_used = True
    else:
        # Setup-only scalar synchronization for fixed `s_G_init`.
        fov_cost_scale = float(torch.quantile(positive.detach(), 0.5).cpu())
        floor_used = False
        if not math.isfinite(fov_cost_scale) or fov_cost_scale <= 0.0:
            raise ContractError("computed fov_cost_scale must be finite and positive")

    observed_self = compute_observed_self_ground_cost(
        block.target_bag,
        cost_matrix,
        cost_scale,
        validate_inputs=True,
        collect_warnings=False,
        config=sinkhorn_config,
    )
    return (
        fov_cost_scale,
        floor_used,
        observed_self.value,
        bool(observed_self.metadata["clipped_negative"]),
    )


def _build_loss_context_once(
    *,
    relation: RelationInput,
    blocks: Sequence[EvidenceBlock],
    cost_matrix: torch.Tensor,
    cost_scale: float,
    sinkhorn_config: SinkhornConfig | None = None,
) -> LossContext:
    """Build fixed objective scales and caches once for one relation fit."""
    resolved_sinkhorn = sinkhorn_config or SinkhornConfig()
    K = int(cost_matrix.shape[0])
    scale_params = _scale_initial_parameters(
        relation.patient_ids,
        K,
        device=cost_matrix.device,
    )

    fov_cost_scales: dict[str, float] = {}
    fov_cost_scale_floor_used: dict[str, bool] = {}
    observed_self_ground_costs: dict[str, torch.Tensor] = {}
    observed_self_clipped_negative: dict[str, bool] = {}
    for block in blocks:
        (
            fov_cost_scale,
            floor_used,
            observed_self_ground_cost,
            clipped_negative,
        ) = _compute_block_fov_cost_scale(
            block,
            parameters=scale_params,
            cost_matrix=cost_matrix,
            cost_scale=cost_scale,
            sinkhorn_config=resolved_sinkhorn,
        )
        fov_cost_scales[block.block_id] = fov_cost_scale
        fov_cost_scale_floor_used[block.block_id] = floor_used
        observed_self_ground_costs[block.block_id] = observed_self_ground_cost
        observed_self_clipped_negative[block.block_id] = clipped_negative

    unit = torch.tensor(1.0, dtype=torch.float64, device=cost_matrix.device)
    temp_context = LossContext(
        obs_scale=unit,
        geometry_scale=unit,
        fov_cost_scales=fov_cost_scales,
        fov_cost_scale_floor_used=fov_cost_scale_floor_used,
        sinkhorn_config=resolved_sinkhorn,
        observed_self_ground_costs=observed_self_ground_costs,
        observed_self_clipped_negative=observed_self_clipped_negative,
    )
    baseline = compute_total_loss(
        scale_params,
        blocks,
        cost_matrix,
        cost_scale,
        context=temp_context,
    )
    obs_raw = baseline.components["obs_raw"]
    geometry_raw = _compute_geometry_loss(scale_params, cost_matrix, cost_scale)
    _ensure_finite_tensor_scalar(obs_raw, name="baseline obs_raw")
    _ensure_finite_tensor_scalar(geometry_raw, name="baseline geometry_raw")

    epsilon = torch.tensor(EPSILON_NORM, dtype=torch.float64, device=cost_matrix.device)
    obs_scale_floor_used = _tensor_less_than(obs_raw, epsilon)
    geometry_scale_floor_used = _tensor_less_than(geometry_raw, epsilon)
    obs_scale = torch.maximum(obs_raw.detach(), epsilon)
    geometry_scale = torch.maximum(geometry_raw.detach(), epsilon)

    return LossContext(
        obs_scale=obs_scale,
        geometry_scale=geometry_scale,
        fov_cost_scales=fov_cost_scales,
        obs_scale_floor_used=obs_scale_floor_used,
        geometry_scale_floor_used=geometry_scale_floor_used,
        fov_cost_scale_floor_used=fov_cost_scale_floor_used,
        sinkhorn_config=resolved_sinkhorn,
        observed_self_ground_costs=observed_self_ground_costs,
        observed_self_clipped_negative=observed_self_clipped_negative,
    )


def _training_step(
    model: RelationModel,
    optimizer: torch.optim.Optimizer,
    prepared: _PreparedRelationFit,
    *,
    stage: str,
    step_index: int,
) -> tuple[LossLedger, float]:
    """Run one full-batch objective/backward/optimizer step."""
    optimizer.zero_grad(set_to_none=True)
    params = model.state()
    ledger = compute_total_loss(
        params,
        prepared.blocks,
        prepared.cost_matrix,
        prepared.cost_scale,
        context=prepared.context,
    )
    total = _scalar_total(ledger.total, name="training total objective")
    ledger.total.backward()
    optimizer.step()
    return ledger, total


def _scalar_total(value: torch.Tensor, *, name: str) -> float:
    """Synchronize a scalar total objective for training control only."""
    if not torch.is_tensor(value) or value.ndim != 0:
        raise ContractError(f"{name} must be a scalar tensor")
    scalar = float(value.detach().cpu())
    if not math.isfinite(scalar):
        raise ContractError(f"{name} must be finite")
    return scalar


def _plateau_condition_met(
    *,
    absolute_improvement: float,
    relative_improvement: float,
) -> bool:
    """Return whether one main-step improvement satisfies plateau criteria."""
    relative_threshold = float(_OPTIMIZER_HANDOFF.min_relative_improvement)
    return abs(absolute_improvement) <= float(_OPTIMIZER_HANDOFF.convergence_tol) or (
        relative_threshold > 0.0 and abs(relative_improvement) <= relative_threshold
    )


def _final_state(
    model: RelationModel,
    prepared: _PreparedRelationFit,
) -> tuple[RelationParameters, LossLedger, float]:
    with torch.no_grad():
        params = _detach_parameters(model.state())
        ledger = compute_total_loss(
            params,
            prepared.blocks,
            prepared.cost_matrix,
            prepared.cost_scale,
            context=prepared.context,
        )
        total = _scalar_total(ledger.total, name="final training total objective")
    return params, ledger, total


def _detach_parameters(parameters: RelationParameters) -> RelationParameters:
    return RelationParameters(
        patient_ids=parameters.patient_ids,
        A=parameters.A.detach().clone(),
        d=parameters.d.detach().clone(),
        e=parameters.e.detach().clone(),
    )


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _step_improvements(
    previous_total: float | None,
    current_total: float,
) -> tuple[float, float]:
    if previous_total is None:
        return math.inf, math.inf
    absolute = float(previous_total) - float(current_total)
    denominator = max(abs(float(previous_total)), EPSILON_NORM)
    return absolute, absolute / denominator


def _run_improvements(initial_total: float, final_total: float) -> tuple[float, float]:
    absolute = float(initial_total) - float(final_total)
    denominator = max(abs(float(initial_total)), EPSILON_NORM)
    return absolute, absolute / denominator


def _as_float64_matrix_on_device(
    value: Any,
    *,
    name: str,
    device: Any,
) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64, device=device)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be coercible to a float64 tensor") from exc
    if tensor.ndim != 2:
        raise ContractError(f"{name} must be a 2D tensor")
    if tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
        raise ContractError(f"{name} must be non-empty")
    return tensor


def _positive_finite_float(value: Any, *, name: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be finite and strictly positive") from exc
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ContractError(f"{name} must be finite and strictly positive")
    return resolved


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return int(value)


def _normalize_patient_ids(patient_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(patient_ids, (str, bytes)):
        raise ContractError("patient_ids must be a sequence, not a string")
    normalized = tuple(str(item).strip() for item in patient_ids)
    if len(normalized) == 0:
        raise ContractError("patient_ids must be non-empty")
    if any(item == "" for item in normalized):
        raise ContractError("patient_ids must contain non-empty identifiers")
    if len(set(normalized)) != len(normalized):
        raise ContractError("patient_ids must not contain duplicates")
    return normalized


def _ensure_finite_tensor_scalar(value: torch.Tensor, *, name: str) -> None:
    if not torch.is_tensor(value) or value.ndim != 0:
        raise ContractError(f"{name} must be a scalar tensor")
    if not bool(torch.isfinite(value.detach()).cpu()):
        raise ContractError(f"{name} must be finite")


def _tensor_less_than(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool((left.detach() < right.detach()).cpu())
