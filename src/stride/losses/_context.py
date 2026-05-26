"""Private objective context construction for STRIDE loss assembly."""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import ContractError
from ..geometry.state_geometry import StateGeometry
from ..observation.balanced_sinkhorn import BalancedSinkhornDivergenceConfig
from ._constants import EPSILON_NORM
from ._initialization import (
    EvidenceBlock,
    FovCostScale,
    ScaleInit,
    _initial_parameters_for,
    _resolve_block_fov_cost_scale,
    _resolve_observation_config,
)
from ._parameters import (
    ADEState,
    _as_float64_tensor,
    _device_key,
    _finite_scalar_bool,
    _require_torch,
    _tensor_data_ptr,
    _tensor_mutation_version,
    _validate_epsilon_norm,
    _validate_parameter_shapes,
    _validate_parameters,
)
from ._totals import CohortLossLedger, _ObservationRawResult, _scale_from_baseline
from .cohort import compute_recurrence_raw
from .fit import (
    compute_consistency_raw_from_block_losses,
)
from .fit import (
    compute_observation_raw as _compute_observation_raw,
)
from .prior import compute_geometry_raw


@dataclass(frozen=True)
class ObjectiveContext:
    """Fixed loss assembly baseline quantities for one objective input surface."""

    patient_ids: tuple[str, ...]
    K: int
    device: str
    evidence_block_keys: tuple[tuple[object, ...], ...]
    geometry_cost_scale: float
    geometry_cost_matrix: Any
    observation_config: BalancedSinkhornDivergenceConfig
    epsilon_norm: float
    initialization: ScaleInit
    init_params: ADEState
    fov_cost_scales: tuple[FovCostScale, ...]
    observation_ground_cost_cache: dict[tuple[object, ...], Any]
    baseline_obs: _ObservationRawResult
    obs_scale: Any
    baseline_consistency_raw: Any
    baseline_recurrence: CohortLossLedger
    baseline_geometry_raw: Any

    @classmethod
    def build(
        cls,
        *,
        params: ADEState,
        evidence_blocks: Sequence[EvidenceBlock],
        geometry: StateGeometry,
        epsilon_norm: float = EPSILON_NORM,
        config: BalancedSinkhornDivergenceConfig | None = None,
    ) -> ObjectiveContext:
        """Compute fixed baseline quantities for repeated objective evaluations."""
        _, _, _, patient_ids = _validate_parameters(params)
        init, init_params = _initial_parameters_for(params)
        resolved_config = _resolve_observation_config(config)
        resolved_epsilon = _validate_epsilon_norm(epsilon_norm)
        fov_cost_scales = tuple(
            _resolve_block_fov_cost_scale(
                block,
                geometry,
                K=init.K,
                config=resolved_config,
                device=init.A.device,
            )
            for block in evidence_blocks
        )
        observation_ground_cost_cache: dict[tuple[object, ...], Any] = {}
        baseline_obs = _compute_observation_raw(
            init_params,
            evidence_blocks,
            geometry,
            fov_cost_scales=fov_cost_scales,
            config=resolved_config,
            observed_self_ground_cost_cache=observation_ground_cost_cache,
        )
        obs_scale, _ = _scale_from_baseline(
            baseline_obs.raw,
            epsilon_norm=resolved_epsilon,
            name="L_obs",
        )
        block_patient_ids = tuple(block.patient_id for block in evidence_blocks)
        baseline_normalized_block_losses = baseline_obs.block_values / obs_scale
        baseline_consistency_raw, _ = compute_consistency_raw_from_block_losses(
            patient_ids=patient_ids,
            block_patient_ids=block_patient_ids,
            normalized_block_losses=baseline_normalized_block_losses,
        )
        baseline_recurrence = compute_recurrence_raw(init_params)
        baseline_geometry_raw = compute_geometry_raw(init_params, geometry)
        return cls(
            patient_ids=patient_ids,
            K=init.K,
            device=_device_key(init.A.device),
            evidence_block_keys=_evidence_block_keys(evidence_blocks),
            geometry_cost_scale=float(geometry.cost_scale),
            geometry_cost_matrix=_as_float64_tensor(
                geometry.cost_matrix,
                name="StateGeometry.cost_matrix",
                device=init.A.device,
            ).detach().clone(),
            observation_config=resolved_config,
            epsilon_norm=resolved_epsilon,
            initialization=init,
            init_params=init_params,
            fov_cost_scales=fov_cost_scales,
            observation_ground_cost_cache=observation_ground_cost_cache,
            baseline_obs=baseline_obs,
            obs_scale=obs_scale,
            baseline_consistency_raw=baseline_consistency_raw,
            baseline_recurrence=baseline_recurrence,
            baseline_geometry_raw=baseline_geometry_raw,
        )

def _evidence_block_keys(
    evidence_blocks: Sequence[EvidenceBlock],
) -> tuple[tuple[object, ...], ...]:
    keys: list[tuple[object, ...]] = []

    def _array_content_key(value: Any) -> tuple[object, ...]:
        torch_module = _require_torch()
        if torch_module.is_tensor(value):
            tensor = value.detach()
            array = np.ascontiguousarray(tensor.cpu().numpy())
            digest = hashlib.sha256(array.view(np.uint8)).hexdigest()
            return (
                tuple(int(item) for item in tensor.shape),
                str(tensor.dtype),
                str(tensor.device),
                _tensor_data_ptr(value),
                _tensor_mutation_version(value),
                digest,
            )
        array = np.ascontiguousarray(np.asarray(value))
        digest = hashlib.sha256(array.view(np.uint8)).hexdigest()
        return (
            tuple(int(item) for item in array.shape),
            str(array.dtype),
            "numpy",
            None,
            None,
            digest,
        )

    for block in evidence_blocks:
        if not isinstance(block, EvidenceBlock):
            raise ContractError("evidence_blocks must contain EvidenceBlock objects")
        fov_cost_scale = (
            None if block.fov_cost_scale is None else float(block.fov_cost_scale)
        )
        keys.append(
            (
                str(block.patient_id).strip(),
                block.block_id,
                id(block.source_bag),
                id(block.target_bag),
                _array_content_key(block.source_bag),
                _array_content_key(block.target_bag),
                fov_cost_scale,
                bool(block.fov_cost_scale_floor_used),
            )
        )
    return tuple(keys)

def _validate_objective_context(
    objective_context: ObjectiveContext,
    *,
    params: ADEState,
    evidence_blocks: Sequence[EvidenceBlock],
    geometry: StateGeometry,
    epsilon_norm: float,
    config: BalancedSinkhornDivergenceConfig | None,
) -> None:
    torch_module = _require_torch()
    A, _, _, patient_ids = _validate_parameter_shapes(params)
    if not isinstance(objective_context, ObjectiveContext):
        raise ContractError("objective_context must be a ObjectiveContext object")
    if objective_context.patient_ids != patient_ids:
        raise ContractError("objective_context patient_ids do not match current objective parameters")
    if int(objective_context.K) != int(A.shape[1]):
        raise ContractError("objective_context K does not match current objective parameters")
    if objective_context.device != _device_key(A.device):
        raise ContractError("objective_context device does not match current objective parameters")
    if objective_context.evidence_block_keys != _evidence_block_keys(evidence_blocks):
        raise ContractError("objective_context evidence_blocks do not match current objective inputs")
    if float(objective_context.geometry_cost_scale) != float(geometry.cost_scale):
        raise ContractError("objective_context geometry does not match current objective inputs")
    current_geometry = _as_float64_tensor(
        geometry.cost_matrix,
        name="StateGeometry.cost_matrix",
        device=A.device,
    )
    cached_geometry = _as_float64_tensor(
        objective_context.geometry_cost_matrix,
        name="objective_context.geometry_cost_matrix",
        device=A.device,
    )
    if current_geometry.shape != cached_geometry.shape or not _finite_scalar_bool(
        torch_module.allclose(current_geometry, cached_geometry, rtol=0.0, atol=0.0)
    ):
        raise ContractError("objective_context geometry does not match current objective inputs")
    if objective_context.observation_config != _resolve_observation_config(config):
        raise ContractError(
            "objective_context observation_config does not match current objective inputs"
        )
    if float(objective_context.epsilon_norm) != _validate_epsilon_norm(epsilon_norm):
        raise ContractError("objective_context epsilon_norm does not match current objective inputs")

__all__ = [
    "ObjectiveContext",
    "_evidence_block_keys",
    "_validate_objective_context",
]
