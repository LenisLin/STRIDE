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

from ._fit_setup import (
    materialize_relation_inputs as _materialize_relation_inputs_once,
)
from ._fit_setup import (
    resolve_runtime_device as _resolve_runtime_device,
)
from ._fit_setup import (
    scale_initial_parameters as _scale_initial_parameters,
)
from ._losses import (
    EPSILON_NORM,
    LossContext,
    LossLedger,
    _batched_pairwise_composition_ground_cost_result,
    _compute_geometry_loss,
    _group_blocks_by_fov_shape,
    _ObservationBatchFallback,
    compute_total_loss,
)
from ._objective import REFERENCE_OBJECTIVE_POLICY, ObjectivePolicy
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
    _normalized_cost_matrix,
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
    objective_policy: ObjectivePolicy = REFERENCE_OBJECTIVE_POLICY,
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

    for _warmup_step in range(1, int(WARMUP_STEPS) + 1):
        ledger, total = _training_step(
            model,
            optimizer,
            prepared,
            objective_policy=objective_policy,
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
            objective_policy=objective_policy,
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
        main_steps_completed += 1
        total_steps += 1

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

    final_params, final_ledger, final_total = _final_state(
        model,
        prepared,
        objective_policy=objective_policy,
    )
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


def _compute_block_fov_cost_scale(
    block: EvidenceBlock,
    *,
    parameters: RelationParameters,
    cost_matrix: torch.Tensor,
    cost_scale: float,
    sinkhorn_config: SinkhornConfig,
) -> tuple[float, bool, torch.Tensor, bool]:
    """Compute fixed per-block scale without changing observation weighting.

    This compatibility entry remains in `_train.py` because task-local tests
    patch its Sinkhorn helpers directly. The calculation is setup-only and is
    not part of the optimizer hot path.
    """
    patient_id = str(block.patient_id)
    try:
        patient_index = parameters.patient_ids.index(patient_id)
    except ValueError as exc:
        raise ContractError("block patient_id must align with relation patient_ids") from exc

    predicted_init = predict_target_composition(
        block.source_bag,
        parameters.A[patient_index],
        parameters.e[patient_index],
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
    patient_index_by_id = {
        str(patient_id): index for index, patient_id in enumerate(relation.patient_ids)
    }
    try:
        _populate_batched_block_context(
            blocks=blocks,
            parameters=scale_params,
            cost_matrix=cost_matrix,
            cost_scale=cost_scale,
            sinkhorn_config=resolved_sinkhorn,
            fov_cost_scales=fov_cost_scales,
            fov_cost_scale_floor_used=fov_cost_scale_floor_used,
            observed_self_ground_costs=observed_self_ground_costs,
            observed_self_clipped_negative=observed_self_clipped_negative,
        )
    except _ObservationBatchFallback:
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
        patient_index_by_id=patient_index_by_id,
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
        patient_index_by_id=patient_index_by_id,
        normalized_cost_matrix=_normalized_cost_matrix(
            cost_matrix,
            cost_scale,
            n_states=K,
            device=cost_matrix.device,
            validate=False,
        ).detach(),
    )


def _populate_batched_block_context(
    *,
    blocks: Sequence[EvidenceBlock],
    parameters: RelationParameters,
    cost_matrix: torch.Tensor,
    cost_scale: float,
    sinkhorn_config: SinkhornConfig,
    fov_cost_scales: dict[str, float],
    fov_cost_scale_floor_used: dict[str, bool],
    observed_self_ground_costs: dict[str, torch.Tensor],
    observed_self_clipped_negative: dict[str, bool],
) -> None:
    """Populate fixed setup scales/caches in shape-homogeneous batches.

    This preserves the scalar setup contract:
    - predicted/observed FOV ground costs use the same normalized state geometry;
    - fov_cost_scale is still the per-block median of positive ground costs;
    - floor flags remain per block;
    - observed-self ground costs and clipped-negative flags remain per block.
    """
    C_norm = _normalized_cost_matrix(
        cost_matrix,
        cost_scale,
        n_states=int(cost_matrix.shape[0]),
        device=cost_matrix.device,
        validate=True,
    )
    patient_index_by_id = {
        str(patient_id): index for index, patient_id in enumerate(parameters.patient_ids)
    }
    for group in _group_blocks_by_fov_shape(tuple(blocks)):
        predicted_init: list[torch.Tensor] = []
        target_bags: list[torch.Tensor] = []
        for block in group.blocks:
            try:
                pidx = patient_index_by_id[str(block.patient_id)]
            except KeyError as exc:
                raise _ObservationBatchFallback from exc
            predicted_init.append(
                predict_target_composition(
                    block.source_bag,
                    parameters.A[pidx],
                    parameters.e[pidx],
                )
            )
            target_bags.append(block.target_bag)

        predicted = torch.stack(predicted_init).to(device=cost_matrix.device, dtype=torch.float64)
        observed = torch.stack(target_bags).to(device=cost_matrix.device, dtype=torch.float64)
        ground_result = _batched_pairwise_composition_ground_cost_result(
            predicted,
            observed,
            C_norm,
            config=sinkhorn_config,
            label="inner_composition_distance.fov_ground_cost.batched_setup",
            runtime_checks=True,
        )
        observed_self_result = _batched_pairwise_composition_ground_cost_result(
            observed,
            observed,
            C_norm,
            config=sinkhorn_config,
            label="inner_composition_distance.observed_self.batched_setup",
            runtime_checks=True,
        )
        ground = ground_result.value.detach()
        observed_self = observed_self_result.value.detach()

        for index, block in enumerate(group.blocks):
            positive = ground[index][(ground[index] > 0.0) & torch.isfinite(ground[index])]
            if positive.numel() == 0:
                fov_cost_scales[block.block_id] = 1.0
                fov_cost_scale_floor_used[block.block_id] = True
            else:
                fov_cost_scale = float(torch.quantile(positive.detach(), 0.5).cpu())
                if not math.isfinite(fov_cost_scale) or fov_cost_scale <= 0.0:
                    raise ContractError("computed fov_cost_scale must be finite and positive")
                fov_cost_scales[block.block_id] = fov_cost_scale
                fov_cost_scale_floor_used[block.block_id] = False
            # `s_G_init` quantiles and observed-self clipped flags remain per block.
            observed_self_ground_costs[block.block_id] = observed_self[index].detach()
            observed_self_clipped_negative[block.block_id] = (
                observed_self_result.clipped_negative[index]
            )


def _training_step(
    model: RelationModel,
    optimizer: torch.optim.Optimizer,
    prepared: _PreparedRelationFit,
    *,
    objective_policy: ObjectivePolicy,
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
        objective_policy=objective_policy,
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
    *,
    objective_policy: ObjectivePolicy,
) -> tuple[RelationParameters, LossLedger, float]:
    with torch.no_grad():
        params = _detach_parameters(model.state())
        ledger = compute_total_loss(
            params,
            prepared.blocks,
            prepared.cost_matrix,
            prepared.cost_scale,
            context=prepared.context,
            objective_policy=objective_policy,
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


def _ensure_finite_tensor_scalar(value: torch.Tensor, *, name: str) -> None:
    if not torch.is_tensor(value) or value.ndim != 0:
        raise ContractError(f"{name} must be a scalar tensor")
    if not bool(torch.isfinite(value.detach()).cpu()):
        raise ContractError(f"{name} must be finite")


def _tensor_less_than(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool((left.detach() < right.detach()).cpu())
