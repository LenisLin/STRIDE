"""Private initialization and observation-scale helpers for STRIDE losses."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..errors import ContractError
from ..geometry.state_geometry import StateGeometry
from ..observation.balanced_sinkhorn import (
    BalancedSinkhornDivergenceConfig,
    _pairwise_composition_ground_cost_value,
)
from ._constants import S_G_INIT_ATOL, S_G_INIT_RTOL
from ._parameters import (
    ADEState,
    _as_float64_tensor,
    _ensure_distribution_matrix,
    _ensure_finite_tensor,
    _normalized_geometry_cost,
    _require_positive_int,
    _require_torch,
    _validate_parameter_shapes,
    post_reconstruct,
)


@dataclass(frozen=True)
class ScaleInit:
    """Deterministic identity-plus-small-open starting point."""

    delta_init: float
    A: Any
    d: Any
    e: Any
    K: int
    dtype: str = "float64"

@dataclass(frozen=True)
class EvidenceBlock:
    """One task-resolved source/target observation evidence block."""

    patient_id: str
    source_bag: Any
    target_bag: Any
    fov_cost_scale: float | None = None
    fov_cost_scale_floor_used: bool = False
    block_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FovCostScale:
    """Canonical per-evidence-block ``s_G_init`` scale."""

    value: float
    floor_used: bool
    positive_cost_count: int


def identity_plus_small_open_initialization(
    K: int,
    *,
    device: Any | None = None,
) -> ScaleInit:
    """Return the deterministic identity-plus-small-open initialization."""
    torch_module = _require_torch()
    n_states = _require_positive_int(K, name="K")
    delta_init = min(0.05, 1.0 / float(n_states + 1))
    A = (1.0 - delta_init) * torch_module.eye(
        n_states,
        dtype=torch_module.float64,
        device=device,
    )
    d = torch_module.full((n_states,), delta_init, dtype=torch_module.float64, device=device)
    e = torch_module.full(
        (n_states,),
        delta_init / float(n_states),
        dtype=torch_module.float64,
        device=device,
    )
    return ScaleInit(
        delta_init=delta_init,
        A=A,
        d=d,
        e=e,
        K=n_states,
    )


def _resolve_observation_config(
    config: BalancedSinkhornDivergenceConfig | None,
) -> BalancedSinkhornDivergenceConfig:
    if config is None:
        return BalancedSinkhornDivergenceConfig()
    if not isinstance(config, BalancedSinkhornDivergenceConfig):
        raise ContractError(
            "config must be None or a BalancedSinkhornDivergenceConfig instance"
        )
    return config


def compute_init_fov_cost_scale(
    block: EvidenceBlock,
    geometry: StateGeometry,
    *,
    K: int,
    config: BalancedSinkhornDivergenceConfig | None = None,
    device: Any | None = None,
) -> FovCostScale:
    """Compute canonical ``s_G_init`` from identity-plus-small-open FOV costs."""
    torch_module = _require_torch()
    if not isinstance(block, EvidenceBlock):
        raise ContractError("block must be a EvidenceBlock object")
    n_states = _require_positive_int(K, name="K")
    resolved_config = _resolve_observation_config(config)
    init = identity_plus_small_open_initialization(n_states, device=device)
    source = _as_float64_tensor(block.source_bag, name="source_bag", device=init.A.device)
    target = _as_float64_tensor(block.target_bag, name="target_bag", device=source.device)
    _ensure_distribution_matrix(source, name="source_bag")
    _ensure_distribution_matrix(target, name="target_bag")
    if source.shape[1] != n_states or target.shape[1] != n_states:
        raise ContractError("evidence block bags must align with K")

    predicted = post_reconstruct(source, init.A, init.e)
    C_norm = _normalized_geometry_cost(geometry, K=n_states, device=predicted.device)
    ground_cost = _pairwise_composition_ground_cost_value(
        predicted,
        target,
        C_norm,
        config=resolved_config,
        label="s_G_init.inner_composition_distance",
    ).value
    _ensure_finite_tensor(ground_cost, name="s_G_init FOV-level costs")
    positive = ground_cost[(ground_cost > 0.0) & torch_module.isfinite(ground_cost)]
    if positive.numel() == 0:
        return FovCostScale(
            value=1.0,
            floor_used=True,
            positive_cost_count=0,
        )
    scale = float(torch_module.quantile(positive.detach(), 0.5).cpu())
    if not math.isfinite(scale) or scale <= 0.0:
        raise ContractError("s_G_init must be finite and strictly positive")
    return FovCostScale(
        value=scale,
        floor_used=False,
        positive_cost_count=int(positive.numel()),
    )


def _resolve_block_fov_cost_scale(
    block: EvidenceBlock,
    geometry: StateGeometry,
    *,
    K: int,
    config: BalancedSinkhornDivergenceConfig | None,
    device: Any | None = None,
) -> FovCostScale:
    computed = compute_init_fov_cost_scale(block, geometry, K=K, config=config, device=device)
    if block.fov_cost_scale is None:
        if block.fov_cost_scale_floor_used:
            raise ContractError(
                "s_G_init_floor_used may be provided only with a precomputed s_G_init"
            )
        return computed

    provided = float(block.fov_cost_scale)
    if not math.isfinite(provided) or provided <= 0.0:
        raise ContractError("s_G_init must be finite and strictly positive")
    if not math.isclose(
        provided,
        computed.value,
        rel_tol=S_G_INIT_RTOL,
        abs_tol=S_G_INIT_ATOL,
    ):
        raise ContractError("provided s_G_init does not match identity-plus-small-open scale")
    if bool(block.fov_cost_scale_floor_used) != computed.floor_used:
        raise ContractError("provided s_G_init_floor_used does not match computed floor usage")
    return computed

def _initial_parameters_for(params: ADEState) -> tuple[ScaleInit, ADEState]:
    A, _, _, patient_ids = _validate_parameter_shapes(params)
    init = identity_plus_small_open_initialization(int(A.shape[1]), device=A.device)
    init_params = ADEState(
        patient_ids=patient_ids,
        A=init.A.expand(len(patient_ids), init.K, init.K).clone(),
        d=init.d.expand(len(patient_ids), init.K).clone(),
        e=init.e.expand(len(patient_ids), init.K).clone(),
    )
    return init, init_params

__all__ = [
    "EvidenceBlock",
    "FovCostScale",
    "ScaleInit",
    "compute_init_fov_cost_scale",
    "identity_plus_small_open_initialization",
    "_initial_parameters_for",
    "_resolve_block_fov_cost_scale",
    "_resolve_observation_config",
]
