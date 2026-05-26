"""Fit-block loss helpers for STRIDE.

Task: expose the observation and subbag-consistency pieces of
``L_fit = normalized_L_obs + rho_subbag * L_subbag_consistency``.
Reference: ``docs/stride_design_freeze.md`` defines source/target observation
fit over task-declared evidence blocks and defines subbag consistency as the
within-patient variance of normalized observation block losses.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ..errors import ContractError
from ..observation.balanced_sinkhorn import (
    SMALL_NEGATIVE_TOL,
    BalancedSinkhornDivergenceConfig,
    _apply_small_negative_rule,
    _batched_pairwise_composition_ground_cost_value,
    _batched_sinkhorn_divergence_value,
)
from ._constants import RHO_SUBBAG
from ._initialization import EvidenceBlock, FovCostScale
from ._parameters import (
    _as_float64_tensor,
    _ensure_distribution_matrix,
    _ensure_finite_tensor,
    _normalize_patient_ids,
    _normalized_geometry_cost,
    _patient_index,
    _require_torch,
    _validate_parameters,
    _validate_raw_loss,
    post_reconstruct,
)
from ._totals import ConsistencyPatientLedger, ObservationBlockLedger, _ObservationRawResult

if TYPE_CHECKING:  # pragma: no cover
    from ..geometry.state_geometry import StateGeometry
    from ._parameters import ADEState


def _resolve_observation_config(
    config: BalancedSinkhornDivergenceConfig | None,
) -> BalancedSinkhornDivergenceConfig:
    if config is None:
        return BalancedSinkhornDivergenceConfig()
    if not isinstance(config, BalancedSinkhornDivergenceConfig):
        raise ContractError("config must be None or a BalancedSinkhornDivergenceConfig instance")
    return config


def _observation_metadata(config: BalancedSinkhornDivergenceConfig, *, s_C: float) -> dict[str, Any]:
    return {
        **config.metadata(),
        "state_geometry": {
            "normalization": "C_norm = C_raw / s_C",
            "s_C": float(s_C),
        },
    }


def compute_observation_raw(
    params: ADEState,
    evidence_blocks: Sequence[EvidenceBlock],
    geometry: StateGeometry,
    *,
    fov_cost_scales: Sequence[FovCostScale],
    config: BalancedSinkhornDivergenceConfig | None,
    observed_self_ground_cost_cache: dict[tuple[object, ...], Any] | None = None,
) -> _ObservationRawResult:
    """Return normalized-observation block inputs and cohort mean raw loss."""
    torch_module = _require_torch()
    A, _, e, patient_ids = _validate_parameters(params)
    if len(evidence_blocks) == 0:
        raise ContractError("loss assembly objective requires at least one evidence block")
    if len(fov_cost_scales) != len(evidence_blocks):
        raise ContractError("fov_cost_scales must align with evidence_blocks")

    resolved_config = _resolve_observation_config(config)
    C_norm = _normalized_geometry_cost(geometry, K=int(A.shape[1]), device=A.device)
    metadata = _observation_metadata(resolved_config, s_C=float(geometry.cost_scale))
    block_values_list: list[Any | None] = [None] * len(evidence_blocks)
    block_patient_ids: list[str] = []
    records: list[ObservationBlockLedger | None] = [None] * len(evidence_blocks)
    groups: dict[tuple[int, int, int], list[int]] = {}
    validated_sources: list[Any] = []
    validated_targets: list[Any] = []

    for block_index, (block, fov_cost_scale) in enumerate(
        zip(evidence_blocks, fov_cost_scales, strict=True)
    ):
        if not isinstance(block, EvidenceBlock):
            raise ContractError("evidence_blocks must contain EvidenceBlock objects")
        if not isinstance(fov_cost_scale, FovCostScale):
            raise ContractError("fov_cost_scales must contain FovCostScale objects")
        patient_id = str(block.patient_id).strip()
        if patient_id == "":
            raise ContractError("EvidenceBlock.patient_id must be non-empty")
        source = _as_float64_tensor(block.source_bag, name="source_bag", device=A.device)
        target = _as_float64_tensor(block.target_bag, name="target_bag", device=A.device)
        _ensure_distribution_matrix(source, name="source_bag")
        _ensure_distribution_matrix(target, name="target_bag")
        if int(source.shape[1]) != int(A.shape[1]) or int(target.shape[1]) != int(A.shape[1]):
            raise ContractError("evidence block bags must align with K")
        validated_sources.append(source)
        validated_targets.append(target)
        block_patient_ids.append(patient_id)
        groups.setdefault((int(source.shape[0]), int(target.shape[0]), int(source.shape[1])), []).append(
            block_index
        )
        block_id = block.block_id or f"block_{block_index}"
        records[block_index] = (
            ObservationBlockLedger(
                block_id=block_id,
                patient_id=patient_id,
                raw=torch_module.full((), torch_module.nan, dtype=torch_module.float64, device=A.device),
                normalized=torch_module.full((), torch_module.nan, dtype=torch_module.float64, device=A.device),
                status="ok",
                fov_cost_scale=fov_cost_scale.value,
                fov_cost_scale_floor_used=fov_cost_scale.floor_used,
                metadata={},
                warnings=(),
            )
        )

    for group_key, block_indices in groups.items():
        n_source, n_target, _n_states = group_key
        source_group = torch_module.stack([validated_sources[index] for index in block_indices], dim=0)
        target_group = torch_module.stack([validated_targets[index] for index in block_indices], dim=0)
        patient_index_group = torch_module.as_tensor(
            [_patient_index(patient_ids, block_patient_ids[index]) for index in block_indices],
            dtype=torch_module.long,
            device=A.device,
        )
        A_group = A[patient_index_group]
        e_group = e[patient_index_group]
        raw_post = torch_module.bmm(source_group, A_group) + e_group.unsqueeze(1)
        _ensure_finite_tensor(raw_post, name="raw_post")
        if bool((raw_post < 0.0).any().detach().cpu().item()):
            raise ContractError("raw_post entries must be nonnegative")
        row_sums = raw_post.sum(dim=2, keepdim=True)
        if bool((row_sums <= 0.0).any().detach().cpu().item()):
            raise ContractError("raw_post rows must have positive mass")
        predicted_group = raw_post / row_sums

        G_cross = _batched_pairwise_composition_ground_cost_value(
            predicted_group,
            target_group,
            C_norm,
            config=resolved_config,
            label="inner_composition_distance.cross",
        )
        G_pred = _batched_pairwise_composition_ground_cost_value(
            predicted_group,
            predicted_group,
            C_norm,
            config=resolved_config,
            label="inner_composition_distance.predicted_self",
        )
        G_obs_values: list[Any | None] = []
        G_obs_clipped: list[bool | None] = []
        missing_obs = False
        for block_index in block_indices:
            cache_key = (
                "observed_self_ground_cost_value_v1",
                block_index,
                str(A.device),
            )
            cached = None if observed_self_ground_cost_cache is None else observed_self_ground_cost_cache.get(cache_key)
            if cached is None:
                G_obs_values.append(None)
                G_obs_clipped.append(None)
            else:
                if isinstance(cached, tuple) and len(cached) == 2:
                    cached_value, cached_clipped = cached
                    G_obs_values.append(cached_value)
                    G_obs_clipped.append(bool(cached_clipped))
                else:
                    G_obs_values.append(cached)
                    G_obs_clipped.append(False)
            missing_obs = missing_obs or cached is None
        if missing_obs:
            G_obs_group = _batched_pairwise_composition_ground_cost_value(
                target_group,
                target_group,
                C_norm,
                config=resolved_config,
                label="inner_composition_distance.observed_self",
            )
            G_obs_values = [G_obs_group.value[position] for position in range(len(block_indices))]
            G_obs_clipped = [
                bool(G_obs_group.clipped_mask[position].any().detach().cpu().item())
                for position in range(len(block_indices))
            ]
            if observed_self_ground_cost_cache is not None:
                for position, block_index in enumerate(block_indices):
                    cache_key = (
                        "observed_self_ground_cost_value_v1",
                        block_index,
                        str(A.device),
                    )
                    observed_self_ground_cost_cache[cache_key] = (
                        G_obs_values[position],
                        G_obs_clipped[position],
                    )
        G_obs = torch_module.stack([item for item in G_obs_values if item is not None], dim=0)
        fov_scales = torch_module.as_tensor(
            [fov_cost_scales[index].value for index in block_indices],
            dtype=torch_module.float64,
            device=A.device,
        )
        predicted_fov_mass = torch_module.full(
            (len(block_indices), n_source),
            1.0 / float(n_source),
            dtype=torch_module.float64,
            device=A.device,
        )
        observed_fov_mass = torch_module.full(
            (len(block_indices), n_target),
            1.0 / float(n_target),
            dtype=torch_module.float64,
            device=A.device,
        )
        outer = _batched_sinkhorn_divergence_value(
            predicted_fov_mass,
            observed_fov_mass,
            G_cross.value / fov_scales[:, None, None],
            G_pred.value / fov_scales[:, None, None],
            G_obs / fov_scales[:, None, None],
            epsilon_schedule=resolved_config.outer_epsilon_schedule,
            config=resolved_config,
        )
        outer_clipped_mask = (outer.value.detach() < 0.0) & (
            outer.value.detach() >= -SMALL_NEGATIVE_TOL
        )
        values, _, _ = _apply_small_negative_rule(
            outer.value,
            label="outer_fov_bag_divergence",
        )
        for position, block_index in enumerate(block_indices):
            value = values[position]
            _validate_raw_loss(value, name="L_obs_pair_raw")
            record = records[block_index]
            if record is None:
                raise ContractError("internal observation records must align with evidence blocks")
            block_values_list[block_index] = value
            block_clipped_negative = any(
                (
                    bool(G_cross.clipped_mask[position].any().detach().cpu().item()),
                    bool(G_pred.clipped_mask[position].any().detach().cpu().item()),
                    bool(G_obs_clipped[position]),
                    bool(outer_clipped_mask[position].detach().cpu().item()),
                )
            )
            records[block_index] = ObservationBlockLedger(
                block_id=record.block_id,
                patient_id=record.patient_id,
                raw=value,
                normalized=record.normalized,
                status="ok_with_warnings" if block_clipped_negative else "ok",
                fov_cost_scale=record.fov_cost_scale,
                fov_cost_scale_floor_used=record.fov_cost_scale_floor_used,
                metadata={"clipped_tiny_negative": True} if block_clipped_negative else {},
                warnings=(),
            )

    values = [value for value in block_values_list if value is not None]
    if len(values) != len(evidence_blocks):
        raise ContractError("internal observation block values must align with evidence blocks")
    block_values = torch_module.stack(values)
    patient_means: list[Any] = []
    for patient_id in patient_ids:
        indices = [idx for idx, block_patient_id in enumerate(block_patient_ids) if block_patient_id == patient_id]
        if len(indices) == 0:
            raise ContractError("each fitted patient must have at least one evidence block")
        patient_means.append(
            block_values[
                torch_module.as_tensor(indices, dtype=torch_module.long, device=block_values.device)
            ].mean()
        )
    raw = torch_module.stack(patient_means).mean()
    _validate_raw_loss(raw, name="L_obs_raw")
    return _ObservationRawResult(
        raw=raw,
        block_values=block_values,
        block_records=tuple(record for record in records if record is not None),
        metadata=metadata,
    )


def compute_consistency_raw_from_block_losses(
    *,
    patient_ids: Sequence[str],
    block_patient_ids: Sequence[str],
    normalized_block_losses: Any,
) -> tuple[Any, Mapping[str, ConsistencyPatientLedger]]:
    """Return cohort mean patient consistency from normalized obs block losses."""
    torch_module = _require_torch()
    normalized_patients = _normalize_patient_ids(patient_ids)
    block_patients = tuple(str(item).strip() for item in block_patient_ids)
    losses = _as_float64_tensor(
        normalized_block_losses,
        name="normalized_block_losses",
    )
    if losses.ndim != 1:
        raise ContractError("normalized_block_losses must be a 1D tensor")
    if len(block_patients) != int(losses.shape[0]):
        raise ContractError("block_patient_ids must align with normalized_block_losses")
    _ensure_finite_tensor(losses, name="normalized_block_losses")
    patient_values: list[Any] = []
    records: dict[str, ConsistencyPatientLedger] = {}
    for patient_id in normalized_patients:
        indices = [idx for idx, item in enumerate(block_patients) if item == patient_id]
        if len(indices) < 2:
            raw = torch_module.zeros((), dtype=torch_module.float64, device=losses.device)
            status = "insufficient_blocks"
        else:
            values = losses[
                torch_module.as_tensor(indices, dtype=torch_module.long, device=losses.device)
            ]
            raw = torch_module.mean((values - values.mean()) ** 2)
            status = "ok"
        _validate_raw_loss(raw, name=f"L_consistency_raw[{patient_id}]")
        patient_values.append(raw)
        records[patient_id] = ConsistencyPatientLedger(
            patient_id=patient_id,
            raw=raw,
            n_blocks=len(indices),
            status=status,
        )
    cohort_raw = torch_module.stack(patient_values).mean()
    _validate_raw_loss(cohort_raw, name="L_consistency_raw")
    return cohort_raw, records

__all__ = [
    "RHO_SUBBAG",
    "ObservationBlockLedger",
    "compute_consistency_raw_from_block_losses",
    "compute_observation_raw",
    "post_reconstruct",
]
