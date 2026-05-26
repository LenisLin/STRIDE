"""Private loss total and ledger containers for STRIDE losses."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..errors import ContractError
from ._constants import (
    ABLATION_TERM_HANDLING,
    EPSILON_NORM,
    GEOMETRY_EFFECTIVE_WEIGHT,
    RHO_SUBBAG,
    S_COHORT,
)
from ._initialization import ScaleInit
from ._parameters import (
    _component_tensor,
    _require_torch,
    _validate_ablation_mode,
    _validate_epsilon_norm,
    _validate_objective_weights,
    _validate_raw_loss,
)

@dataclass(frozen=True)
class LossComponent:
    """Canonical raw, normalized, and optimizer-effective component value."""

    raw: Any
    scale: Any
    normalized: Any
    floor_used: bool
    effective_normalized: Any | None = None


@dataclass(frozen=True)
class LossTotals:
    """Three-block objective totals for ``L_fit``, ``L_prior``, and ``L_cohort``."""

    total: Any
    fit: Any
    prior: Any
    cohort: Any
    components: Mapping[str, LossComponent]
    objective_weights: tuple[float, float, float]
    epsilon_norm: float
    rho_subbag: float = RHO_SUBBAG
    geometry_effective_weight: float = GEOMETRY_EFFECTIVE_WEIGHT
    s_cohort: float = S_COHORT
    ablation_mode: str = "none"
    ablation_term_handling: str | None = None
    ablation_denominator_policy: str = "fixed_denominator_no_reweighting"

@dataclass(frozen=True)
class ObservationBlockLedger:
    """Per-evidence-block observation loss record."""

    block_id: str
    patient_id: str
    raw: Any
    normalized: Any
    status: str
    fov_cost_scale: float
    fov_cost_scale_floor_used: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConsistencyPatientLedger:
    """Per-patient consistency support record."""

    patient_id: str
    raw: Any
    n_blocks: int
    status: str


@dataclass(frozen=True)
class CohortLossLedger:
    """Single-cohort recurrence consensus and dispersion record."""

    raw: Any
    L_T: Any
    L_e_rec: Any
    cohort_scaled: Any
    dispersion: Any
    support_n_patients: int
    T_bar: Any
    A_bar: Any
    d_bar: Any
    e_bar: Any
    per_patient_dispersion: Any
    status: str = "ok"


@dataclass(frozen=True)
class LossLedger:
    """Complete objective ledger for downstream optimizer/provenance wiring."""

    total: Any
    fit: Any
    prior: Any
    cohort: Any
    components: Mapping[str, LossComponent]
    objective_weights: tuple[float, float, float]
    epsilon_norm: float
    rho_subbag: float
    geometry_effective_weight: float
    s_cohort: float
    offdiag_init_mass: float
    numerical_min_mass: float
    initialization: ScaleInit
    observation_blocks: tuple[ObservationBlockLedger, ...]
    consistency_patients: Mapping[str, ConsistencyPatientLedger]
    recurrence: CohortLossLedger
    ablation_mode: str = "none"
    ablation_term_handling: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ObservationRawResult:
    raw: Any
    block_values: Any
    block_records: tuple[ObservationBlockLedger, ...]
    metadata: Mapping[str, Any]


def _scale_from_baseline(
    baseline_raw: Any,
    *,
    epsilon_norm: float,
    name: str,
) -> tuple[Any, bool]:
    torch_module = _require_torch()
    _validate_raw_loss(baseline_raw, name=f"{name}_baseline_raw")
    eps = torch_module.as_tensor(
        float(epsilon_norm),
        dtype=torch_module.float64,
        device=baseline_raw.device,
    )
    scale = torch_module.maximum(baseline_raw, eps)
    return scale, bool((baseline_raw.detach().cpu() < float(epsilon_norm)).item())


def assemble_loss_totals(
    *,
    raw_components: Mapping[str, Any],
    baseline_components: Mapping[str, Any],
    objective_weights: Sequence[float] = (1.0, 1.0, 1.0),
    epsilon_norm: float = EPSILON_NORM,
    ablation_mode: str = "none",
) -> LossTotals:
    """Normalize components and assemble fixed-denominator objective totals."""
    torch_module = _require_torch()
    resolved_weights = _validate_objective_weights(objective_weights)
    resolved_epsilon = _validate_epsilon_norm(epsilon_norm)
    resolved_ablation = _validate_ablation_mode(ablation_mode)
    required_raw = ("obs", "open", "geometry", "consistency", "recurrence")
    missing_raw = [name for name in required_raw if name not in raw_components]
    if missing_raw:
        raise ContractError(f"raw_components missing required keys: {tuple(missing_raw)}")
    required_baseline = ("obs", "geometry")
    missing_baseline = [name for name in required_baseline if name not in baseline_components]
    if missing_baseline:
        raise ContractError(
            f"baseline_components missing required keys: {tuple(missing_baseline)}"
        )

    first_tensor = next(
        (
            value
            for value in raw_components.values()
            if torch_module.is_tensor(value)
        ),
        None,
    )
    device = first_tensor.device if first_tensor is not None else None
    components: dict[str, LossComponent] = {}
    for name in required_raw:
        raw = _component_tensor(raw_components[name], name=f"L_{name}_raw", device=device)
        _validate_raw_loss(raw, name=f"L_{name}_raw")
        if name in {"open", "consistency"}:
            scale = torch_module.ones((), dtype=torch_module.float64, device=raw.device)
            floor_used = False
        elif name == "recurrence":
            scale = torch_module.as_tensor(S_COHORT, dtype=torch_module.float64, device=raw.device)
            floor_used = False
        else:
            baseline = _component_tensor(
                baseline_components[name],
                name=f"L_{name}_baseline_raw",
                device=raw.device,
            )
            scale, floor_used = _scale_from_baseline(
                baseline,
                epsilon_norm=resolved_epsilon,
                name=f"L_{name}",
            )
        normalized = raw / scale
        _validate_raw_loss(normalized, name=f"normalized_L_{name}")
        if name == "geometry":
            effective = GEOMETRY_EFFECTIVE_WEIGHT * normalized
        elif name == "consistency":
            effective = RHO_SUBBAG * normalized
        else:
            effective = normalized
        if resolved_ablation == name:
            effective = torch_module.zeros_like(effective)
        components[name] = LossComponent(
            raw=raw,
            scale=scale,
            normalized=normalized,
            floor_used=floor_used,
            effective_normalized=effective,
        )

    obs_eff = components["obs"].effective_normalized
    open_eff = components["open"].effective_normalized
    geometry_eff = components["geometry"].effective_normalized
    consistency_eff = components["consistency"].effective_normalized
    recurrence_eff = components["recurrence"].effective_normalized
    fit = obs_eff + consistency_eff
    prior = (open_eff + geometry_eff) / 2.0
    cohort = recurrence_eff
    weights_tensor = torch_module.as_tensor(
        resolved_weights,
        dtype=torch_module.float64,
        device=fit.device,
    )
    total = (
        weights_tensor[0] * fit
        + weights_tensor[1] * prior
        + weights_tensor[2] * cohort
    ) / weights_tensor.sum()
    for name, value in (("L_fit", fit), ("L_prior", prior), ("L_cohort", cohort), ("L_total", total)):
        _validate_raw_loss(value, name=name)
    return LossTotals(
        total=total,
        fit=fit,
        prior=prior,
        cohort=cohort,
        components=components,
        objective_weights=resolved_weights,
        epsilon_norm=resolved_epsilon,
        ablation_mode=resolved_ablation,
        ablation_term_handling=(
            ABLATION_TERM_HANDLING if resolved_ablation != "none" else None
        ),
    )

__all__ = [
    "CohortLossLedger",
    "ConsistencyPatientLedger",
    "LossComponent",
    "LossLedger",
    "LossTotals",
    "ObservationBlockLedger",
    "_ObservationRawResult",
    "_scale_from_baseline",
    "assemble_loss_totals",
]
