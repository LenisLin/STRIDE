"""Canonical STRIDE v1 objective assembly for `.tl`.

This module declares the full three-block objective ledger: fit, prior, and
cohort. It records observation, open, geometry, consistency, and recurrence
components without converting observation-layer diagnostics into biological
`d/e`.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from ._objective import REFERENCE_OBJECTIVE_POLICY, ObjectivePolicy
from ._parameters import RelationParameters, predict_target_composition
from ._resolve import EvidenceBlock
from ._sinkhorn import (
    SinkhornConfig,
    _apply_small_negative_rule,
    _as_float64_matrix,
    _batched_sinkhorn_divergence_value,
    _batched_sinkhorn_transport_value,
    _normalized_cost_matrix,
    _resolve_sinkhorn_config,
    _validate_fov_cost_scale,
    compute_sinkhorn_divergence,
)

OBJECTIVE_CONTRACT_VERSION = "stride_full_estimator_three_block_v1"
EPSILON_NORM = 1e-2
RHO_SUBBAG = 1.0
GEOMETRY_EFFECTIVE_WEIGHT = 1e-2
S_COHORT = 1e-2


@dataclass(frozen=True)
class LossContext:
    """Fixed objective quantities for repeated one-relation loss evaluation.

    These values are prepared once before optimizer steps. The loss hot path
    consumes them without revalidating evidence or recomputing baseline scales.
    Scale tensors remain tensors here; `_output.py` scalarizes final values
    once during public output assembly to avoid hot-path CPU synchronization.
    """

    obs_scale: torch.Tensor
    geometry_scale: torch.Tensor
    fov_cost_scales: Mapping[str, float]
    obs_scale_floor_used: bool = False
    geometry_scale_floor_used: bool = False
    fov_cost_scale_floor_used: Mapping[str, bool] = field(default_factory=dict)
    sinkhorn_config: SinkhornConfig | None = None
    observed_self_ground_costs: Mapping[str, torch.Tensor] = field(default_factory=dict)
    observed_self_clipped_negative: Mapping[str, bool] = field(default_factory=dict)
    # Execution-only lookup cache. Patient-balanced aggregation still follows
    # `RelationParameters.patient_ids` and is not weighted by block count.
    patient_index_by_id: Mapping[str, int] = field(default_factory=dict)
    normalized_cost_matrix: torch.Tensor | None = None


@dataclass(frozen=True)
class _ObservationLoss:
    """Observation loss values shared by L_obs and subbag consistency."""

    raw: torch.Tensor
    normalized: torch.Tensor
    block_values: torch.Tensor
    normalized_block_values: torch.Tensor
    block_patient_ids: tuple[str, ...]
    warnings: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class _BatchedGroundCostValue:
    """Batched composition ground costs plus scalar-path validation metadata."""

    value: torch.Tensor
    clipped_negative: tuple[bool, ...]
    warnings: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class LossLedger:
    """Full objective ledger for one relation fit.

    total: scalar tensor optimized by AdamW.
    fit: fit block contribution.
    prior: prior block contribution.
    cohort: cohort recurrence contribution.
    components: raw, normalized, and effective objective component tensors.
    metadata: objective constants and operator metadata.
    warnings: structured objective or convergence warning records.
    """

    total: torch.Tensor
    fit: torch.Tensor
    prior: torch.Tensor
    cohort: torch.Tensor
    components: Mapping[str, torch.Tensor]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[Mapping[str, Any], ...] = ()


def compute_total_loss(
    parameters: RelationParameters,
    blocks: Sequence[EvidenceBlock],
    cost_matrix: torch.Tensor,
    cost_scale: float,
    *,
    context: Mapping[str, Any] | LossContext | None = None,
    objective_policy: ObjectivePolicy = REFERENCE_OBJECTIVE_POLICY,
) -> LossLedger:
    """Compute the canonical three-block STRIDE objective.

    Purpose:
        Assemble `L_total = mean(L_fit, L_prior, L_cohort)` for one relation.

    Key variables:
        observation_loss: normalized canonical `L_obs` contribution.
        open_loss: tendency-level cost from `d/e`.
        geometry_loss: raw `A` geometry penalty under `cost_matrix / cost_scale`.
        recurrence_loss: patient-level recurrence component.
        total: scalar optimizer objective.
    """
    # total: scalar objective used by the formal optimizer.
    ctx = _coerce_loss_context(context)

    obs = _compute_observation_loss(
        parameters,
        blocks,
        cost_matrix,
        cost_scale,
        context=ctx,
    )
    open_raw = _compute_open_loss(parameters)
    geometry_raw = _compute_geometry_loss(parameters, cost_matrix, cost_scale)
    recurrence_raw = _compute_recurrence_loss(parameters)

    if not isinstance(objective_policy, ObjectivePolicy):
        raise TypeError("objective_policy must be an ObjectivePolicy")

    geometry_normalized = geometry_raw / ctx.geometry_scale
    geometry_nominal = GEOMETRY_EFFECTIVE_WEIGHT * geometry_normalized
    geometry_effective = objective_policy.geometry_weight * geometry_nominal
    consistency_raw = _compute_subbag_consistency(
        patient_ids=parameters.patient_ids,
        block_patient_ids=obs.block_patient_ids,
        normalized_block_losses=obs.normalized_block_values,
    )

    consistency_effective = (
        objective_policy.consistency_weight * RHO_SUBBAG * consistency_raw
    )
    recurrence_scaled = recurrence_raw / S_COHORT
    recurrence_effective = objective_policy.recurrence_weight * recurrence_scaled

    fit = obs.normalized + consistency_effective
    prior = (open_raw + geometry_effective) / 2.0
    cohort = recurrence_effective
    total = (fit + prior + cohort) / 3.0

    return LossLedger(
        total=total,
        fit=fit,
        prior=prior,
        cohort=cohort,
        components={
            "obs_raw": obs.raw,
            "obs_normalized": obs.normalized,
            "open_raw": open_raw,
            "geometry_raw": geometry_raw,
            "geometry_normalized": geometry_normalized,
            "geometry_nominal": geometry_nominal,
            "geometry_effective": geometry_effective,
            "consistency_raw": consistency_raw,
            "consistency_effective": consistency_effective,
            "recurrence_raw": recurrence_raw,
            "recurrence_scaled": recurrence_scaled,
            "recurrence_effective": recurrence_effective,
        },
        metadata={
            "objective_contract_version": OBJECTIVE_CONTRACT_VERSION,
            "objective_policy": {
                "name": objective_policy.name,
                "consistency_weight": objective_policy.consistency_weight,
                "geometry_weight": objective_policy.geometry_weight,
                "recurrence_weight": objective_policy.recurrence_weight,
                "fixed_block_denominators": True,
            },
            "objective_constants": {
                "rho_subbag": RHO_SUBBAG,
                "geometry_effective_weight": GEOMETRY_EFFECTIVE_WEIGHT,
                "s_cohort": S_COHORT,
                "epsilon_norm": EPSILON_NORM,
            },
            "loss_scales": {
                "obs_scale": ctx.obs_scale,
                "obs_scale_floor_used": ctx.obs_scale_floor_used,
                "geometry_scale": ctx.geometry_scale,
                "geometry_scale_floor_used": ctx.geometry_scale_floor_used,
                "fov_cost_scales": dict(ctx.fov_cost_scales),
                "fov_cost_scale_floor_used": dict(ctx.fov_cost_scale_floor_used),
            },
            "observation_discrepancy": (
                ctx.sinkhorn_config or SinkhornConfig()
            ).metadata(),
            "state_geometry": {
                "normalization": "C_norm = C_raw / s_C",
                "s_C": float(cost_scale),
            },
        },
        warnings=obs.warnings,
    )


def _compute_observation_loss(
    parameters: RelationParameters,
    blocks: Sequence[EvidenceBlock],
    cost_matrix: torch.Tensor,
    cost_scale: float,
    *,
    context: Mapping[str, Any] | LossContext | None = None,
) -> _ObservationLoss:
    """Compute the observation-fit component over evidence blocks.

    Purpose:
        Compare predicted and observed target-side FOV bags using canonical
        `D_obs^BalancedSinkhornDivergence-v1`.

    Key variables:
        source_bag: source-side FOV composition bag.
        target_bag: observed target-side FOV composition bag.
        predicted_bag: `normalize(source_bag @ A + e)`.
        block_losses: per-evidence-block observation discrepancies.
    """
    ctx = _coerce_loss_context(context)
    resolved_config = _resolve_sinkhorn_config(ctx.sinkhorn_config)
    try:
        return _compute_observation_loss_batched_path(
            parameters,
            blocks,
            cost_matrix,
            cost_scale,
            context=ctx,
            sinkhorn_config=resolved_config,
        )
    except _ObservationBatchFallback:
        return _compute_observation_loss_single_block_path(
            parameters,
            blocks,
            cost_matrix,
            cost_scale,
            context=ctx,
        )


def _compute_observation_loss_single_block_path(
    parameters: RelationParameters,
    blocks: Sequence[EvidenceBlock],
    cost_matrix: torch.Tensor,
    cost_scale: float,
    *,
    context: Mapping[str, Any] | LossContext | None = None,
) -> _ObservationLoss:
    """Compute observation loss with the original per-block Sinkhorn route."""
    ctx = _coerce_loss_context(context)
    block_values: list[torch.Tensor] = []
    block_patient_ids: list[str] = []
    warnings: list[Mapping[str, Any]] = []

    for block in blocks:
        patient_id = str(block.patient_id)
        pidx = _patient_index(parameters, patient_id, ctx)
        predicted = predict_target_composition(
            block.source_bag,
            parameters.A[pidx],
            parameters.e[pidx],
        )
        result = compute_sinkhorn_divergence(
            predicted,
            block.target_bag,
            cost_matrix,
            cost_scale,
            fov_cost_scale=ctx.fov_cost_scales[block.block_id],
            fov_cost_scale_floor_used=ctx.fov_cost_scale_floor_used.get(
                block.block_id,
                False,
            ),
            observed_self_ground_cost=ctx.observed_self_ground_costs.get(block.block_id),
            observed_self_clipped_negative=ctx.observed_self_clipped_negative.get(
                block.block_id,
                False,
            ),
            validate_inputs=False,
            collect_warnings=False,
            config=ctx.sinkhorn_config,
        )
        block_values.append(result.value)
        block_patient_ids.append(patient_id)
        warnings.extend(result.warnings)

    block_tensor = torch.stack(block_values)
    patient_means: list[torch.Tensor] = []
    for patient_id in parameters.patient_ids:
        idx = [i for i, item in enumerate(block_patient_ids) if item == patient_id]
        selected = block_tensor[torch.as_tensor(idx, device=block_tensor.device)]
        # Contract averaging is patient-balanced rather than block-count weighted.
        patient_means.append(selected.mean())

    raw = torch.stack(patient_means).mean()
    normalized_block = block_tensor / ctx.obs_scale
    normalized = raw / ctx.obs_scale
    return _ObservationLoss(
        raw=raw,
        normalized=normalized,
        block_values=block_tensor,
        normalized_block_values=normalized_block,
        block_patient_ids=tuple(block_patient_ids),
        warnings=tuple(warnings),
    )


class _ObservationBatchFallback(Exception):
    """Internal marker for falling back to the single-block observation path."""


def _compute_observation_loss_batched_path(
    parameters: RelationParameters,
    blocks: Sequence[EvidenceBlock],
    cost_matrix: torch.Tensor,
    cost_scale: float,
    *,
    context: LossContext,
    sinkhorn_config: SinkhornConfig,
) -> _ObservationLoss:
    """Compute observation loss by batching evidence blocks with matching FOV shapes.

    This is an execution-shape optimization only. Blocks are grouped by
    (source_fov_count, target_fov_count), evaluated in batches, then restored to
    the original block order before patient-balanced averaging.
    """
    block_sequence = tuple(blocks)
    if not block_sequence:
        raise _ObservationBatchFallback

    cost = context.normalized_cost_matrix
    if cost is None:
        cost = _normalized_cost_matrix(
            cost_matrix,
            cost_scale,
            n_states=int(parameters.A.shape[1]),
            device=parameters.A.device,
            validate=False,
        )
    elif (
        cost.shape != cost_matrix.shape
        or cost.device != parameters.A.device
        or cost.dtype != torch.float64
    ):
        raise _ObservationBatchFallback

    block_values_by_index: dict[int, torch.Tensor] = {}
    warnings: list[Mapping[str, Any]] = []

    # Blocks are batched only within identical FOV shapes, then restored to
    # original order before patient-balanced averaging.
    for group in _group_blocks_by_fov_shape(block_sequence):
        values, group_warnings = _compute_observation_block_group(
            parameters,
            group,
            cost,
            context=context,
            sinkhorn_config=sinkhorn_config,
        )
        warnings.extend(group_warnings)
        for offset, value in zip(group.indices, values, strict=True):
            block_values_by_index[int(offset)] = value

    block_values = [block_values_by_index[index] for index in range(len(block_sequence))]
    return _assemble_observation_loss_from_block_values(
        block_values=block_values,
        block_patient_ids=tuple(str(block.patient_id) for block in block_sequence),
        patient_ids=parameters.patient_ids,
        obs_scale=context.obs_scale,
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class _BlockShapeGroup:
    indices: tuple[int, ...]
    blocks: tuple[EvidenceBlock, ...]
    n_source_fov: int
    n_target_fov: int


def _group_blocks_by_fov_shape(blocks: Sequence[EvidenceBlock]) -> tuple[_BlockShapeGroup, ...]:
    grouped: dict[tuple[int, int], list[tuple[int, EvidenceBlock]]] = {}
    for index, block in enumerate(blocks):
        key = (int(block.source_bag.shape[0]), int(block.target_bag.shape[0]))
        grouped.setdefault(key, []).append((index, block))
    return tuple(
        _BlockShapeGroup(
            indices=tuple(index for index, _block in items),
            blocks=tuple(block for _index, block in items),
            n_source_fov=shape[0],
            n_target_fov=shape[1],
        )
        for shape, items in grouped.items()
    )


def _compute_observation_block_group(
    parameters: RelationParameters,
    group: _BlockShapeGroup,
    C_norm: torch.Tensor,
    *,
    context: LossContext,
    sinkhorn_config: SinkhornConfig,
) -> tuple[torch.Tensor, tuple[Mapping[str, Any], ...]]:
    predicted_bags: list[torch.Tensor] = []
    target_bags: list[torch.Tensor] = []
    fov_scales: list[float] = []
    observed_self_costs: list[torch.Tensor] = []

    for block in group.blocks:
        patient_id = str(block.patient_id)
        pidx = _patient_index(parameters, patient_id, context)
        predicted_bags.append(
            predict_target_composition(
                block.source_bag,
                parameters.A[pidx],
                parameters.e[pidx],
            )
        )
        target_bags.append(_as_float64_matrix(block.target_bag, name="target_bag"))
        fov_scales.append(
            _validate_fov_cost_scale(
                context.fov_cost_scales[block.block_id],
                floor_used=context.fov_cost_scale_floor_used.get(block.block_id, False),
            )
        )
        observed_self = context.observed_self_ground_costs.get(block.block_id)
        if observed_self is None:
            raise _ObservationBatchFallback
        observed_self_costs.append(_as_float64_matrix(observed_self, name="observed_self_ground_cost"))

    predicted = torch.stack(predicted_bags)
    observed = torch.stack(target_bags).to(device=predicted.device, dtype=torch.float64)
    scales = torch.as_tensor(fov_scales, dtype=torch.float64, device=predicted.device)
    G_cross = _batched_pairwise_composition_ground_cost_value(
        predicted,
        observed,
        C_norm,
        config=sinkhorn_config,
        label="inner_composition_distance.cross",
    )
    G_pred = _batched_pairwise_composition_ground_cost_value(
        predicted,
        predicted,
        C_norm,
        config=sinkhorn_config,
        label="inner_composition_distance.predicted_self",
    )
    G_obs = torch.stack(observed_self_costs).to(device=predicted.device, dtype=torch.float64)
    left_mass = torch.full(
        (len(group.blocks), group.n_source_fov),
        1.0 / float(group.n_source_fov),
        dtype=torch.float64,
        device=predicted.device,
    )
    right_mass = torch.full(
        (len(group.blocks), group.n_target_fov),
        1.0 / float(group.n_target_fov),
        dtype=torch.float64,
        device=predicted.device,
    )
    outer = _batched_sinkhorn_divergence_value(
        left_mass,
        right_mass,
        G_cross / scales[:, None, None],
        G_pred / scales[:, None, None],
        G_obs / scales[:, None, None],
        epsilon_schedule=sinkhorn_config.outer_epsilon_schedule,
        config=sinkhorn_config,
        label="outer_fov_bag_divergence.batched_blocks",
        collect_warnings=False,
        runtime_checks=False,
    )
    values, _outer_clipped, outer_warnings = _apply_small_negative_rule(
        outer.value,
        label="outer_fov_bag_divergence.batched_blocks",
        runtime_checks=False,
    )
    return values, tuple(outer.warnings + outer_warnings)


def _batched_pairwise_composition_ground_cost_value(
    left: torch.Tensor,
    right: torch.Tensor,
    C_norm: torch.Tensor,
    *,
    config: SinkhornConfig,
    label: str,
) -> torch.Tensor:
    return _batched_pairwise_composition_ground_cost_result(
        left,
        right,
        C_norm,
        config=config,
        label=label,
        runtime_checks=False,
    ).value


def _batched_pairwise_composition_ground_cost_result(
    left: torch.Tensor,
    right: torch.Tensor,
    C_norm: torch.Tensor,
    *,
    config: SinkhornConfig,
    label: str,
    runtime_checks: bool,
) -> _BatchedGroundCostValue:
    """Return batched FOV-pair composition costs.

    runtime_checks=True is used by setup/context construction and must preserve
    scalar-path validation, small-negative clipping, and per-block clipped flags.
    runtime_checks=False is used by the training hot path and matches the existing
    non-diagnostic scalar execution semantics.
    """
    if left.ndim != 3 or right.ndim != 3:
        raise _ObservationBatchFallback
    if left.shape[0] != right.shape[0] or left.shape[2] != right.shape[2]:
        raise _ObservationBatchFallback
    batch_size = int(left.shape[0])
    n_left = int(left.shape[1])
    n_right = int(right.shape[1])
    n_states = int(left.shape[2])
    if n_left <= 0 or n_right <= 0:
        raise _ObservationBatchFallback

    left_batch = (
        left[:, :, None, :]
        .expand(batch_size, n_left, n_right, n_states)
        .reshape(-1, n_states)
    )
    right_batch = (
        right[:, None, :, :]
        .expand(batch_size, n_left, n_right, n_states)
        .reshape(-1, n_states)
    )
    n_pairs = int(left_batch.shape[0])
    C_batch = C_norm.expand(n_pairs, -1, -1)
    C_t_batch = C_norm.T.expand(n_pairs, -1, -1)

    # The four concatenated transports implement the same debiased composition
    # divergence formula as the scalar path for each FOV pair.
    all_left = torch.cat((left_batch, right_batch, left_batch, right_batch), dim=0)
    all_right = torch.cat((right_batch, left_batch, left_batch, right_batch), dim=0)
    all_cost = torch.cat((C_batch, C_t_batch, C_batch, C_batch), dim=0)
    transport = _batched_sinkhorn_transport_value(
        all_left,
        all_right,
        all_cost,
        epsilon_schedule=config.inner_epsilon_schedule,
        config=config,
        runtime_checks=runtime_checks,
    )
    values = transport.value.reshape(4, batch_size, n_left * n_right)
    divergence = 0.5 * (values[0] + values[1]) - 0.5 * values[2] - 0.5 * values[3]
    if runtime_checks:
        # Setup-time validation must match the scalar path per evidence block:
        # invalid values raise, tiny negatives are clipped, and each block keeps
        # its own clipped flag. The expensive Sinkhorn work remains batched.
        checked_values: list[torch.Tensor] = []
        clipped_flags: list[bool] = []
        warnings: list[Mapping[str, Any]] = []
        for index in range(batch_size):
            checked, clipped_negative, block_warnings = _apply_small_negative_rule(
                divergence[index],
                label=f"{label}[{index}]",
                runtime_checks=True,
            )
            checked_values.append(checked)
            clipped_flags.append(clipped_negative)
            warnings.extend(block_warnings)
        return _BatchedGroundCostValue(
            value=torch.stack(checked_values).reshape(batch_size, n_left, n_right),
            clipped_negative=tuple(clipped_flags),
            warnings=tuple(warnings),
        )

    clipped_values, _clipped_negative, _clip_warnings = _apply_small_negative_rule(
        divergence,
        label=label,
        runtime_checks=False,
    )
    return _BatchedGroundCostValue(
        value=clipped_values.reshape(batch_size, n_left, n_right),
        clipped_negative=tuple(False for _ in range(batch_size)),
    )


def _assemble_observation_loss_from_block_values(
    *,
    block_values: Sequence[torch.Tensor],
    block_patient_ids: Sequence[str],
    patient_ids: Sequence[str],
    obs_scale: torch.Tensor,
    warnings: tuple[Mapping[str, Any], ...],
) -> _ObservationLoss:
    block_tensor = torch.stack(tuple(block_values))
    patient_means: list[torch.Tensor] = []
    for patient_id in patient_ids:
        idx = [i for i, item in enumerate(block_patient_ids) if item == patient_id]
        selected = block_tensor[torch.as_tensor(idx, device=block_tensor.device)]
        patient_means.append(selected.mean())
    raw = torch.stack(patient_means).mean()

    normalized_block = block_tensor / obs_scale
    normalized = raw / obs_scale
    return _ObservationLoss(
        raw=raw,
        normalized=normalized,
        block_values=block_tensor,
        normalized_block_values=normalized_block,
        block_patient_ids=tuple(block_patient_ids),
        warnings=warnings,
    )


def _compute_subbag_consistency(
    *,
    patient_ids: Sequence[str],
    block_patient_ids: Sequence[str],
    normalized_block_losses: torch.Tensor,
) -> torch.Tensor:
    """Return mean within-patient variance of normalized observation block losses."""

    values: list[torch.Tensor] = []
    for patient_id in patient_ids:
        idx = [i for i, item in enumerate(block_patient_ids) if item == patient_id]
        if len(idx) < 2:
            values.append(
                torch.zeros(
                    (),
                    dtype=normalized_block_losses.dtype,
                    device=normalized_block_losses.device,
                )
            )
            continue
        patient_losses = normalized_block_losses[
            torch.as_tensor(idx, device=normalized_block_losses.device)
        ]
        values.append(torch.mean((patient_losses - patient_losses.mean()) ** 2))
    return torch.stack(values).mean()


def _compute_open_loss(parameters: RelationParameters) -> torch.Tensor:
    """Compute the tendency-level open-channel cost.

    Purpose:
        Declare the open prior over fitted biological `d/e`.

    Key variables:
        d: source-row open channel tensor.
        e: target-side additive open tendency tensor.
        open_loss: scalar tendency-level open cost.
    """
    # open_loss: tendency-level cost, not observation unmatched residual.
    return parameters.d.mean() + parameters.e.mean()


def _compute_geometry_loss(
    parameters: RelationParameters,
    cost_matrix: torch.Tensor,
    cost_scale: float,
) -> torch.Tensor:
    """Compute the raw-`A` geometry prior.

    Purpose:
        Penalize distant remodeling under the shared-state geometry.

    Key variables:
        C_norm: normalized state cost matrix.
        A: transition tensor `[P, K, K]`.
        geometry_loss: scalar geometry prior.
    """
    # C_norm: `cost_matrix / cost_scale` on the shared K-state basis.
    C_norm = cost_matrix.to(device=parameters.A.device, dtype=torch.float64) / float(cost_scale)
    K = int(parameters.A.shape[1])
    per_patient = (parameters.A * C_norm.unsqueeze(0)).sum(dim=(1, 2)) / float(K)
    return per_patient.mean()


def _compute_recurrence_loss(parameters: RelationParameters) -> torch.Tensor:
    """Compute patient-level recurrence regularity.

    Purpose:
        Declare the cohort block over patient-level fitted relations.

    Key variables:
        patient_ids: realized patient axis.
        A: transition tensor.
        d: source-row open channel tensor.
        e: target-side open tendency tensor.
        recurrence_loss: scalar cohort recurrence component.
    """
    # recurrence_loss: patient-level relation regularity term.
    T = torch.cat([parameters.A, parameters.d.unsqueeze(2)], dim=2)
    T_bar = T.mean(dim=0)
    e_bar = parameters.e.mean(dim=0)

    per_patient_T = ((T - T_bar.unsqueeze(0)) ** 2).sum(dim=2).mean(dim=1)
    per_patient_e = ((parameters.e - e_bar.unsqueeze(0)) ** 2).mean(dim=1)
    return per_patient_T.mean() + per_patient_e.mean()


def _coerce_loss_context(context: Mapping[str, Any] | LossContext | None) -> LossContext:
    """Return fixed loss context for objective assembly.

    Training code should pass LossContext. Mapping support is kept only as an
    internal scaffold bridge while neighboring modules are still being filled.
    """

    if isinstance(context, LossContext):
        return context
    if context is None or not isinstance(context, Mapping):
        raise TypeError("context must be a LossContext or mapping with fixed objective scales")

    obs_scale_value = context["obs_scale"]
    geometry_scale_value = context["geometry_scale"]
    device = None
    if torch.is_tensor(obs_scale_value):
        device = obs_scale_value.device
    elif torch.is_tensor(geometry_scale_value):
        device = geometry_scale_value.device

    obs_scale = torch.as_tensor(obs_scale_value, dtype=torch.float64, device=device)
    geometry_scale = torch.as_tensor(
        geometry_scale_value,
        dtype=torch.float64,
        device=obs_scale.device,
    )
    return LossContext(
        obs_scale=obs_scale,
        geometry_scale=geometry_scale,
        fov_cost_scales=context["fov_cost_scales"],
        obs_scale_floor_used=bool(context.get("obs_scale_floor_used", False)),
        geometry_scale_floor_used=bool(context.get("geometry_scale_floor_used", False)),
        fov_cost_scale_floor_used=context.get("fov_cost_scale_floor_used", {}),
        sinkhorn_config=context.get("sinkhorn_config"),
        observed_self_ground_costs=context.get("observed_self_ground_costs", {}),
        observed_self_clipped_negative=context.get("observed_self_clipped_negative", {}),
        patient_index_by_id=context.get("patient_index_by_id", {}),
        normalized_cost_matrix=context.get("normalized_cost_matrix"),
    )


def _patient_index(
    parameters: RelationParameters,
    patient_id: str,
    context: LossContext,
) -> int:
    """Resolve one patient-axis index, preferring the fixed fit-time cache."""
    cached = context.patient_index_by_id.get(patient_id)
    if cached is not None:
        return int(cached)
    return parameters.patient_ids.index(patient_id)
