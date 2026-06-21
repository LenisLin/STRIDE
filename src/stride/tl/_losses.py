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

from ._parameters import RelationParameters, predict_target_composition
from ._resolve import EvidenceBlock
from ._sinkhorn import SinkhornConfig, compute_sinkhorn_divergence

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

    geometry_normalized = geometry_raw / ctx.geometry_scale
    geometry_effective = GEOMETRY_EFFECTIVE_WEIGHT * geometry_normalized
    consistency_raw = _compute_subbag_consistency(
        patient_ids=parameters.patient_ids,
        block_patient_ids=obs.block_patient_ids,
        normalized_block_losses=obs.normalized_block_values,
    )

    fit = obs.normalized + RHO_SUBBAG * consistency_raw
    prior = (open_raw + geometry_effective) / 2.0
    cohort = recurrence_raw / S_COHORT
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
            "geometry_effective": geometry_effective,
            "consistency_raw": consistency_raw,
            "recurrence_raw": recurrence_raw,
        },
        metadata={
            "objective_contract_version": OBJECTIVE_CONTRACT_VERSION,
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
    # block_losses: canonical observation discrepancies by evidence block.
    ctx = _coerce_loss_context(context)
    block_values: list[torch.Tensor] = []
    block_patient_ids: list[str] = []
    warnings: list[Mapping[str, Any]] = []

    for block in blocks:
        patient_id = str(block.patient_id)
        pidx = parameters.patient_ids.index(patient_id)
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
    )
