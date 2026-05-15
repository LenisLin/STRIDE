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
from ..observation.balanced_sinkhorn import compute_balanced_sinkhorn_observation_discrepancy
from ._constants import RHO_SUBBAG
from ._initialization import EvidenceBlock, FovCostScale
from ._parameters import (
    _as_float64_tensor,
    _ensure_finite_tensor,
    _normalize_patient_ids,
    _patient_index,
    _require_torch,
    _validate_parameters,
    _validate_raw_loss,
    post_reconstruct,
)
from ._totals import ConsistencyPatientLedger, ObservationBlockLedger, _ObservationRawResult

if TYPE_CHECKING:  # pragma: no cover
    from ..geometry.state_geometry import StateGeometry
    from ..observation.balanced_sinkhorn import BalancedSinkhornDivergenceConfig
    from ._parameters import ADEState


def compute_observation_raw(
    params: "ADEState",
    evidence_blocks: Sequence["EvidenceBlock"],
    geometry: "StateGeometry",
    *,
    fov_cost_scales: Sequence["FovCostScale"],
    config: "BalancedSinkhornDivergenceConfig | None",
    observed_self_ground_cost_cache: dict[tuple[object, ...], Any] | None = None,
) -> "_ObservationRawResult":
    """Return normalized-observation block inputs and cohort mean raw loss."""
    torch_module = _require_torch()
    A, _, e, patient_ids = _validate_parameters(params)
    if len(evidence_blocks) == 0:
        raise ContractError("loss assembly objective requires at least one evidence block")
    if len(fov_cost_scales) != len(evidence_blocks):
        raise ContractError("fov_cost_scales must align with evidence_blocks")

    values: list[Any] = []
    records: list[ObservationBlockLedger] = []
    metadata: Mapping[str, Any] | None = None
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
        patient_idx = _patient_index(patient_ids, patient_id)
        predicted = post_reconstruct(block.source_bag, A[patient_idx], e[patient_idx])
        result = compute_balanced_sinkhorn_observation_discrepancy(
            predicted,
            block.target_bag,
            geometry,
            fov_cost_scale=fov_cost_scale.value,
            fov_cost_scale_floor_used=fov_cost_scale.floor_used,
            config=config,
            observed_self_ground_cost_cache=observed_self_ground_cost_cache,
        )
        value = result.value
        _validate_raw_loss(value, name="L_obs_pair_raw")
        block_id = block.block_id or f"block_{block_index}"
        metadata = metadata or result.metadata
        values.append(value)
        records.append(
            ObservationBlockLedger(
                block_id=block_id,
                patient_id=patient_id,
                raw=value,
                normalized=torch_module.full_like(value, torch_module.nan),
                status=str(result.status),
                fov_cost_scale=fov_cost_scale.value,
                fov_cost_scale_floor_used=fov_cost_scale.floor_used,
                metadata=dict(result.metadata),
                warnings=tuple(str(item) for item in result.warnings),
            )
        )
    block_values = torch_module.stack(values)
    patient_means: list[Any] = []
    for patient_id in patient_ids:
        indices = [idx for idx, record in enumerate(records) if record.patient_id == patient_id]
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
        block_records=tuple(records),
        metadata=dict(metadata or {}),
    )


def compute_consistency_raw_from_block_losses(
    *,
    patient_ids: Sequence[str],
    block_patient_ids: Sequence[str],
    normalized_block_losses: Any,
) -> tuple[Any, Mapping[str, "ConsistencyPatientLedger"]]:
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
