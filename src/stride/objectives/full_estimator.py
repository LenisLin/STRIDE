"""Canonical full-estimator objective core for STRIDE v1.

This module is an internal optimizer-ready surface used by the canonical
PyTorch/AdamW full-estimator workflow for supported ``fit_stride(...)`` inputs.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..errors import ContractError
from ..geometry.state_geometry import StateGeometry
from ..observation.balanced_sinkhorn import (
    BalancedSinkhornDivergenceConfig,
    _pairwise_composition_ground_cost,
    compute_balanced_sinkhorn_observation_discrepancy,
)

try:  # pragma: no cover - exercised through _require_torch when unavailable
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


EPSILON_NORM = 1e-2
ABLATION_MODES = frozenset({"none", "geometry", "recurrence", "consistency"})
ABLATION_TERM_HANDLING = "zero_weight"
OPTIMIZER_INIT_MIN_MASS = 1e-12
S_G_INIT_RTOL = 1e-7
S_G_INIT_ATOL = 1e-10


@dataclass(frozen=True)
class FullEstimatorInitialization:
    """Deterministic identity-plus-small-open starting point."""

    delta_init: float
    A: Any
    d: Any
    e: Any
    K: int
    dtype: str = "float64"


@dataclass
class FullEstimatorUnconstrainedParameters:
    """Unconstrained optimizer variables for patient-level ``A/d/e``."""

    patient_ids: tuple[str, ...]
    row_logits: Any
    e_logits: Any


@dataclass(frozen=True)
class FullEstimatorParameters:
    """Constrained patient-level ``A/d/e`` tensors."""

    patient_ids: tuple[str, ...]
    A: Any
    d: Any
    e: Any


@dataclass(frozen=True)
class FullEstimatorEvidenceBlock:
    """One task-resolved source/target observation evidence block."""

    patient_id: str
    source_bag: Any
    target_bag: Any
    fov_cost_scale: float | None = None
    fov_cost_scale_floor_used: bool = False
    block_id: str | None = None


@dataclass(frozen=True)
class FullEstimatorFovCostScale:
    """Canonical per-evidence-block ``s_G_init`` scale."""

    value: float
    floor_used: bool
    positive_cost_count: int


@dataclass(frozen=True)
class FullEstimatorComponentLedger:
    """Raw, scale, normalized, and effective normalized component value."""

    raw: Any
    scale: Any
    normalized: Any
    floor_used: bool
    effective_normalized: Any | None = None


@dataclass(frozen=True)
class FullEstimatorTotals:
    """Fixed-denominator grouped objective totals."""

    total: Any
    local: Any
    regularization: Any
    components: Mapping[str, FullEstimatorComponentLedger]
    alpha: float
    epsilon_norm: float
    local_denominator: int = 3
    regularization_denominator: int = 2
    ablation_mode: str = "none"
    ablation_term_handling: str | None = None
    ablation_denominator_policy: str = "fixed_denominator_no_reweighting"


@dataclass(frozen=True)
class FullEstimatorObservationBlockLedger:
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
class FullEstimatorConsistencyPatientLedger:
    """Per-patient consistency support record."""

    patient_id: str
    raw: Any
    n_blocks: int
    status: str


@dataclass(frozen=True)
class FullEstimatorRecurrenceLedger:
    """Single-cohort recurrence consensus and dispersion record."""

    raw: Any
    dispersion: Any
    support_n_patients: int
    A_bar: Any
    d_bar: Any
    e_bar: Any
    per_patient_dispersion: Any
    status: str = "ok"


@dataclass(frozen=True)
class FullEstimatorObjectiveLedger:
    """Complete objective ledger for downstream optimizer/provenance wiring."""

    total: Any
    local: Any
    regularization: Any
    components: Mapping[str, FullEstimatorComponentLedger]
    alpha: float
    epsilon_norm: float
    initialization: FullEstimatorInitialization
    observation_blocks: tuple[FullEstimatorObservationBlockLedger, ...]
    consistency_patients: Mapping[str, FullEstimatorConsistencyPatientLedger]
    recurrence: FullEstimatorRecurrenceLedger
    ablation_mode: str = "none"
    ablation_term_handling: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ObservationRawResult:
    raw: Any
    block_values: Any
    block_records: tuple[FullEstimatorObservationBlockLedger, ...]
    metadata: Mapping[str, Any]


def _require_torch() -> Any:
    if torch is None:  # pragma: no cover - depends on optional runtime
        raise ContractError("canonical full-estimator objective requires torch")
    return torch


def _require_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return int(value)


def _normalize_patient_ids(patient_ids: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(item).strip() for item in patient_ids)
    if len(normalized) == 0:
        raise ContractError("patient_ids must be non-empty")
    if any(item == "" for item in normalized):
        raise ContractError("patient_ids must contain non-empty identifiers")
    if len(set(normalized)) != len(normalized):
        raise ContractError("patient_ids must not contain duplicates")
    return normalized


def _as_float64_tensor(value: Any, *, name: str, device: Any | None = None) -> Any:
    torch_module = _require_torch()
    if torch_module.is_tensor(value):
        resolved_device = value.device if device is None else device
        return value.to(device=resolved_device, dtype=torch_module.float64)
    return torch_module.as_tensor(value, dtype=torch_module.float64, device=device)


def _finite_scalar_bool(value: Any) -> bool:
    torch_module = _require_torch()
    return bool(value.detach().cpu().item()) if torch_module.is_tensor(value) else bool(value)


def _ensure_finite_tensor(value: Any, *, name: str) -> None:
    torch_module = _require_torch()
    if not _finite_scalar_bool(torch_module.isfinite(value).all()):
        raise ContractError(f"{name} must contain only finite values")


def _ensure_distribution_matrix(matrix: Any, *, name: str) -> None:
    torch_module = _require_torch()
    if matrix.ndim != 2:
        raise ContractError(f"{name} must be a 2D [N, K] distribution matrix")
    if matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise ContractError(f"{name} must be a non-empty [N, K] distribution matrix")
    _ensure_finite_tensor(matrix, name=name)
    if _finite_scalar_bool((matrix < 0.0).any()):
        raise ContractError(f"{name} entries must be nonnegative")
    row_sums = torch_module.sum(matrix, dim=1)
    if not _finite_scalar_bool(
        torch_module.allclose(
            row_sums,
            torch_module.ones_like(row_sums),
            rtol=1e-8,
            atol=1e-8,
        )
    ):
        raise ContractError(f"{name} rows must sum to 1.0")


def _validate_alpha(alpha: float) -> float:
    try:
        resolved = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ContractError("alpha must be finite and in [0, 1]") from exc
    if not math.isfinite(resolved) or resolved < 0.0 or resolved > 1.0:
        raise ContractError("alpha must be finite and in [0, 1]")
    return resolved


def _validate_ablation_mode(ablation_mode: str) -> str:
    mode = str(ablation_mode)
    if mode not in ABLATION_MODES:
        raise ContractError(
            "ablation_mode must be one of 'none', 'geometry', 'recurrence', 'consistency'"
        )
    return mode


def _validate_epsilon_norm(epsilon_norm: float) -> float:
    try:
        resolved = float(epsilon_norm)
    except (TypeError, ValueError) as exc:
        raise ContractError("epsilon_norm must be finite and strictly positive") from exc
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ContractError("epsilon_norm must be finite and strictly positive")
    return resolved


def _validate_raw_loss(value: Any, *, name: str) -> None:
    torch_module = _require_torch()
    if value.ndim != 0:
        raise ContractError(f"{name} must be a scalar loss")
    if not _finite_scalar_bool(torch_module.isfinite(value)):
        raise ContractError(f"{name} must be finite")
    if _finite_scalar_bool(value < -1e-10):
        raise ContractError(f"{name} must be nonnegative")


def _component_tensor(value: Any, *, name: str, device: Any | None = None) -> Any:
    tensor = _as_float64_tensor(value, name=name, device=device)
    if tensor.ndim != 0:
        raise ContractError(f"{name} must be scalar")
    return tensor


def identity_plus_small_open_initialization(
    K: int,
    *,
    device: Any | None = None,
) -> FullEstimatorInitialization:
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
    return FullEstimatorInitialization(
        delta_init=delta_init,
        A=A,
        d=d,
        e=e,
        K=n_states,
    )


def unconstrained_from_constrained(
    patient_ids: Sequence[str],
    A: Any,
    d: Any,
    e: Any,
) -> FullEstimatorUnconstrainedParameters:
    """Build softmax/sigmoid logits from feasible constrained tensors."""
    torch_module = _require_torch()
    normalized_patient_ids = _normalize_patient_ids(patient_ids)
    A_tensor = _as_float64_tensor(A, name="A")
    d_tensor = _as_float64_tensor(d, name="d", device=A_tensor.device)
    e_tensor = _as_float64_tensor(e, name="e", device=A_tensor.device)
    params = FullEstimatorParameters(
        patient_ids=normalized_patient_ids,
        A=A_tensor,
        d=d_tensor,
        e=e_tensor,
    )
    _validate_parameters(params)

    row_simplex = torch_module.cat([A_tensor, d_tensor.unsqueeze(2)], dim=2)
    row_simplex = torch_module.clamp(row_simplex, min=OPTIMIZER_INIT_MIN_MASS)
    row_simplex = row_simplex / row_simplex.sum(dim=2, keepdim=True)
    row_logits = torch_module.log(row_simplex)
    e_logits = torch_module.logit(
        torch_module.clamp(
            e_tensor,
            min=OPTIMIZER_INIT_MIN_MASS,
            max=1.0 - OPTIMIZER_INIT_MIN_MASS,
        )
    )
    return FullEstimatorUnconstrainedParameters(
        patient_ids=normalized_patient_ids,
        row_logits=row_logits,
        e_logits=e_logits,
    )


def unconstrained_from_initialization(
    patient_ids: Sequence[str],
    K: int,
    *,
    device: Any | None = None,
) -> FullEstimatorUnconstrainedParameters:
    """Return unconstrained parameters whose transform equals canonical init."""
    torch_module = _require_torch()
    normalized_patient_ids = _normalize_patient_ids(patient_ids)
    init = identity_plus_small_open_initialization(K, device=device)
    A = init.A.expand(len(normalized_patient_ids), init.K, init.K).clone()
    d = init.d.expand(len(normalized_patient_ids), init.K).clone()
    e = init.e.expand(len(normalized_patient_ids), init.K).clone()
    if init.K > 1:
        off_diagonal = ~torch_module.eye(init.K, dtype=torch_module.bool, device=device)
        A[:, off_diagonal] = OPTIMIZER_INIT_MIN_MASS
        diagonal_delta = OPTIMIZER_INIT_MIN_MASS * float(init.K - 1)
        diagonal = torch_module.arange(init.K, device=device)
        A[:, diagonal, diagonal] = A[:, diagonal, diagonal] - diagonal_delta
    row_simplex = torch_module.cat([A, d.unsqueeze(2)], dim=2)
    row_logits = torch_module.log(row_simplex)
    e_logits = torch_module.logit(e)
    return FullEstimatorUnconstrainedParameters(
        patient_ids=normalized_patient_ids,
        row_logits=row_logits,
        e_logits=e_logits,
    )


def parameters_from_unconstrained(
    unconstrained: FullEstimatorUnconstrainedParameters,
) -> FullEstimatorParameters:
    """Transform unconstrained logits into feasible constrained ``A/d/e``."""
    torch_module = _require_torch()
    if not isinstance(unconstrained, FullEstimatorUnconstrainedParameters):
        raise ContractError("unconstrained must be a FullEstimatorUnconstrainedParameters object")
    patient_ids = _normalize_patient_ids(unconstrained.patient_ids)
    row_logits = _as_float64_tensor(unconstrained.row_logits, name="row_logits")
    e_logits = _as_float64_tensor(unconstrained.e_logits, name="e_logits", device=row_logits.device)
    if row_logits.ndim != 3:
        raise ContractError("row_logits must be a [P, K, K+1] tensor")
    if e_logits.ndim != 2:
        raise ContractError("e_logits must be a [P, K] tensor")
    if row_logits.shape[0] != len(patient_ids):
        raise ContractError("row_logits first dimension must align with patient_ids")
    K = int(row_logits.shape[1])
    if K <= 0 or row_logits.shape[2] != K + 1:
        raise ContractError("row_logits must have shape [P, K, K+1]")
    if e_logits.shape != (len(patient_ids), K):
        raise ContractError("e_logits must align with patient_ids and K")
    if not _finite_scalar_bool(torch_module.isfinite(row_logits).all()):
        raise ContractError("row_logits must contain only finite values")
    if not _finite_scalar_bool(torch_module.isfinite(e_logits).all()):
        raise ContractError("e_logits must contain only finite values")

    row_simplex = torch_module.softmax(row_logits, dim=2)
    return FullEstimatorParameters(
        patient_ids=patient_ids,
        A=row_simplex[:, :, :K],
        d=row_simplex[:, :, K],
        e=torch_module.sigmoid(e_logits),
    )


def _validate_parameters(params: FullEstimatorParameters) -> tuple[Any, Any, Any, tuple[str, ...]]:
    torch_module = _require_torch()
    if not isinstance(params, FullEstimatorParameters):
        raise ContractError("params must be a FullEstimatorParameters object")
    patient_ids = _normalize_patient_ids(params.patient_ids)
    A = _as_float64_tensor(params.A, name="A")
    d = _as_float64_tensor(params.d, name="d", device=A.device)
    e = _as_float64_tensor(params.e, name="e", device=A.device)
    if A.ndim != 3:
        raise ContractError("A must be a [P, K, K] tensor")
    if A.shape[0] != len(patient_ids):
        raise ContractError("A first dimension must align with patient_ids")
    P = len(patient_ids)
    K = int(A.shape[1])
    if K <= 0 or A.shape[2] != K:
        raise ContractError("A must be a [P, K, K] tensor")
    if d.shape != (P, K):
        raise ContractError("d must be a [P, K] tensor aligned with A")
    if e.shape != (P, K):
        raise ContractError("e must be a [P, K] tensor aligned with A")
    for name, value in (("A", A), ("d", d), ("e", e)):
        _ensure_finite_tensor(value, name=name)
        if _finite_scalar_bool((value < 0.0).any()):
            raise ContractError(f"{name} entries must be nonnegative")
    if _finite_scalar_bool((e > 1.0).any()):
        raise ContractError("e entries must be bounded in [0, 1]")
    row_sums = A.sum(dim=2) + d
    if not _finite_scalar_bool(
        torch_module.allclose(
            row_sums,
            torch_module.ones_like(row_sums),
            rtol=1e-8,
            atol=1e-8,
        )
    ):
        raise ContractError("each source row [A_i,*, d_i] must lie on a simplex")
    return A, d, e, patient_ids


def _patient_index(patient_ids: tuple[str, ...], patient_id: str) -> int:
    try:
        return patient_ids.index(str(patient_id))
    except ValueError as exc:
        raise ContractError(f"evidence block patient_id {patient_id!r} is not fitted") from exc


def _normalized_geometry_cost(
    geometry: StateGeometry,
    *,
    K: int,
    device: Any,
) -> Any:
    torch_module = _require_torch()
    if not isinstance(geometry, StateGeometry):
        raise ContractError("geometry must be a StateGeometry object")
    cost_scale = float(geometry.cost_scale)
    if not math.isfinite(cost_scale) or cost_scale <= 0.0:
        raise ContractError("StateGeometry.cost_scale must be finite and strictly positive")
    C_raw = torch_module.as_tensor(geometry.cost_matrix, dtype=torch_module.float64, device=device)
    if C_raw.ndim != 2 or C_raw.shape != (K, K):
        raise ContractError("StateGeometry.cost_matrix must align to the input K-state axis")
    if not _finite_scalar_bool(torch_module.isfinite(C_raw).all()):
        raise ContractError("StateGeometry.cost_matrix must contain only finite values")
    if _finite_scalar_bool((C_raw < 0.0).any()):
        raise ContractError("StateGeometry.cost_matrix must be nonnegative")
    if not _finite_scalar_bool(torch_module.allclose(C_raw, C_raw.T, rtol=0.0, atol=1e-12)):
        raise ContractError("StateGeometry.cost_matrix must be symmetric")
    if not _finite_scalar_bool(
        torch_module.allclose(
            torch_module.diagonal(C_raw),
            torch_module.zeros(K, dtype=torch_module.float64, device=device),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ContractError("StateGeometry.cost_matrix diagonal must be zero")
    off_diagonal = ~torch_module.eye(K, dtype=torch_module.bool, device=device)
    if not _finite_scalar_bool(((C_raw > 0.0) & off_diagonal).any()):
        raise ContractError("StateGeometry.cost_matrix must contain a positive off-diagonal cost")
    C_norm = C_raw / cost_scale
    _ensure_finite_tensor(C_norm, name="C_norm = C_raw / s_C")
    return C_norm


def post_reconstruct(q_minus: Any, A: Any, e: Any) -> Any:
    """Return ``normalize(q_minus @ A + e)`` for source FOV bags."""
    q = _as_float64_tensor(q_minus, name="q_minus")
    A_tensor = _as_float64_tensor(A, name="A", device=q.device)
    e_tensor = _as_float64_tensor(e, name="e", device=q.device)
    _ensure_distribution_matrix(q, name="q_minus")
    K = int(q.shape[1])
    if A_tensor.shape != (K, K):
        raise ContractError("A must be a [K, K] tensor aligned with q_minus")
    if e_tensor.shape != (K,):
        raise ContractError("e must be a [K] tensor aligned with q_minus")
    for name, value in (("A", A_tensor), ("e", e_tensor)):
        _ensure_finite_tensor(value, name=name)
        if _finite_scalar_bool((value < 0.0).any()):
            raise ContractError(f"{name} entries must be nonnegative")
    raw_post = q @ A_tensor + e_tensor
    _ensure_finite_tensor(raw_post, name="raw_post")
    if _finite_scalar_bool((raw_post < 0.0).any()):
        raise ContractError("raw_post entries must be nonnegative")
    row_sums = raw_post.sum(dim=1, keepdim=True)
    if _finite_scalar_bool((row_sums <= 0.0).any()):
        raise ContractError("raw_post rows must have positive mass")
    return raw_post / row_sums


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
    block: FullEstimatorEvidenceBlock,
    geometry: StateGeometry,
    *,
    K: int,
    config: BalancedSinkhornDivergenceConfig | None = None,
) -> FullEstimatorFovCostScale:
    """Compute canonical ``s_G_init`` from identity-plus-small-open FOV costs."""
    torch_module = _require_torch()
    if not isinstance(block, FullEstimatorEvidenceBlock):
        raise ContractError("block must be a FullEstimatorEvidenceBlock object")
    n_states = _require_positive_int(K, name="K")
    resolved_config = _resolve_observation_config(config)
    init = identity_plus_small_open_initialization(n_states)
    source = _as_float64_tensor(block.source_bag, name="source_bag", device=init.A.device)
    target = _as_float64_tensor(block.target_bag, name="target_bag", device=source.device)
    _ensure_distribution_matrix(source, name="source_bag")
    _ensure_distribution_matrix(target, name="target_bag")
    if source.shape[1] != n_states or target.shape[1] != n_states:
        raise ContractError("evidence block bags must align with K")

    predicted = post_reconstruct(source, init.A, init.e)
    C_norm = _normalized_geometry_cost(geometry, K=n_states, device=predicted.device)
    ground_cost = _pairwise_composition_ground_cost(
        predicted,
        target,
        C_norm,
        config=resolved_config,
        label="s_G_init.inner_composition_distance",
    ).value
    _ensure_finite_tensor(ground_cost, name="s_G_init FOV-level costs")
    positive = ground_cost[(ground_cost > 0.0) & torch_module.isfinite(ground_cost)]
    if positive.numel() == 0:
        return FullEstimatorFovCostScale(
            value=1.0,
            floor_used=True,
            positive_cost_count=0,
        )
    scale = float(torch_module.quantile(positive.detach(), 0.5).cpu())
    if not math.isfinite(scale) or scale <= 0.0:
        raise ContractError("s_G_init must be finite and strictly positive")
    return FullEstimatorFovCostScale(
        value=scale,
        floor_used=False,
        positive_cost_count=int(positive.numel()),
    )


def _resolve_block_fov_cost_scale(
    block: FullEstimatorEvidenceBlock,
    geometry: StateGeometry,
    *,
    K: int,
    config: BalancedSinkhornDivergenceConfig | None,
) -> FullEstimatorFovCostScale:
    computed = compute_init_fov_cost_scale(block, geometry, K=K, config=config)
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


def compute_open_raw(params: FullEstimatorParameters) -> Any:
    """Return ``mean(d_p) + mean(e_p)`` over fitted patients/components."""
    _, d, e, _ = _validate_parameters(params)
    raw = d.mean() + e.mean()
    _validate_raw_loss(raw, name="L_open_raw")
    return raw


def _validate_parameter_shapes(params: FullEstimatorParameters) -> tuple[Any, Any, Any, tuple[str, ...]]:
    if not isinstance(params, FullEstimatorParameters):
        raise ContractError("params must be a FullEstimatorParameters object")
    patient_ids = _normalize_patient_ids(params.patient_ids)
    A = _as_float64_tensor(params.A, name="A")
    d = _as_float64_tensor(params.d, name="d", device=A.device)
    e = _as_float64_tensor(params.e, name="e", device=A.device)
    if A.ndim != 3 or A.shape[1] <= 0 or A.shape[2] != A.shape[1]:
        raise ContractError("A must be a [P, K, K] tensor")
    if A.shape[0] != len(patient_ids):
        raise ContractError("A first dimension must align with patient_ids")
    if d.shape != (len(patient_ids), int(A.shape[1])):
        raise ContractError("d must be a [P, K] tensor aligned with A")
    if e.shape != (len(patient_ids), int(A.shape[1])):
        raise ContractError("e must be a [P, K] tensor aligned with A")
    for name, value in (("A", A), ("d", d), ("e", e)):
        _ensure_finite_tensor(value, name=name)
    return A, d, e, patient_ids


def compute_geometry_raw(params: FullEstimatorParameters, geometry: StateGeometry) -> Any:
    """Return cohort mean ``(1/K) * sum_i sum_j A_p[i,j] * C_norm[i,j]``."""
    A, _, _, _ = _validate_parameters(params)
    K = int(A.shape[1])
    C_norm = _normalized_geometry_cost(geometry, K=K, device=A.device)
    per_patient = (A * C_norm.unsqueeze(0)).sum(dim=(1, 2)) / float(K)
    raw = per_patient.mean()
    _validate_raw_loss(raw, name="L_geometry_raw")
    return raw


def compute_recurrence_raw(params: FullEstimatorParameters) -> FullEstimatorRecurrenceLedger:
    """Return single-cohort consensus recurrence raw loss and support fields."""
    A, d, e, patient_ids = _validate_parameters(params)
    A_bar = A.mean(dim=0)
    d_bar = d.mean(dim=0)
    e_bar = e.mean(dim=0)
    per_patient = (
        ((A - A_bar.unsqueeze(0)) ** 2).mean(dim=(1, 2))
        + ((d - d_bar.unsqueeze(0)) ** 2).mean(dim=1)
        + ((e - e_bar.unsqueeze(0)) ** 2).mean(dim=1)
    )
    raw = per_patient.mean()
    _validate_raw_loss(raw, name="L_recurrence_raw")
    return FullEstimatorRecurrenceLedger(
        raw=raw,
        dispersion=raw,
        support_n_patients=len(patient_ids),
        A_bar=A_bar,
        d_bar=d_bar,
        e_bar=e_bar,
        per_patient_dispersion=per_patient,
    )


def compute_consistency_raw_from_block_losses(
    *,
    patient_ids: Sequence[str],
    block_patient_ids: Sequence[str],
    normalized_block_losses: Any,
) -> tuple[Any, Mapping[str, FullEstimatorConsistencyPatientLedger]]:
    """Return cohort mean patient consistency from normalized obs block losses."""
    torch_module = _require_torch()
    normalized_patients = _normalize_patient_ids(patient_ids)
    block_patients = tuple(str(item).strip() for item in block_patient_ids)
    losses = _as_float64_tensor(normalized_block_losses, name="normalized_block_losses")
    if losses.ndim != 1:
        raise ContractError("normalized_block_losses must be a 1D tensor")
    if len(block_patients) != int(losses.shape[0]):
        raise ContractError("block_patient_ids must align with normalized_block_losses")
    _ensure_finite_tensor(losses, name="normalized_block_losses")
    patient_values: list[Any] = []
    records: dict[str, FullEstimatorConsistencyPatientLedger] = {}
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
        records[patient_id] = FullEstimatorConsistencyPatientLedger(
            patient_id=patient_id,
            raw=raw,
            n_blocks=len(indices),
            status=status,
        )
    cohort_raw = torch_module.stack(patient_values).mean()
    _validate_raw_loss(cohort_raw, name="L_consistency_raw")
    return cohort_raw, records


def _compute_observation_raw(
    params: FullEstimatorParameters,
    evidence_blocks: Sequence[FullEstimatorEvidenceBlock],
    geometry: StateGeometry,
    *,
    fov_cost_scales: Sequence[FullEstimatorFovCostScale],
    config: BalancedSinkhornDivergenceConfig | None,
) -> _ObservationRawResult:
    torch_module = _require_torch()
    A, _, e, patient_ids = _validate_parameters(params)
    if len(evidence_blocks) == 0:
        raise ContractError("full-estimator objective requires at least one evidence block")
    if len(fov_cost_scales) != len(evidence_blocks):
        raise ContractError("fov_cost_scales must align with evidence_blocks")

    values: list[Any] = []
    records: list[FullEstimatorObservationBlockLedger] = []
    metadata: Mapping[str, Any] | None = None
    for block_index, (block, fov_cost_scale) in enumerate(
        zip(evidence_blocks, fov_cost_scales, strict=True)
    ):
        if not isinstance(block, FullEstimatorEvidenceBlock):
            raise ContractError("evidence_blocks must contain FullEstimatorEvidenceBlock objects")
        if not isinstance(fov_cost_scale, FullEstimatorFovCostScale):
            raise ContractError("fov_cost_scales must contain FullEstimatorFovCostScale objects")
        patient_id = str(block.patient_id).strip()
        if patient_id == "":
            raise ContractError("FullEstimatorEvidenceBlock.patient_id must be non-empty")
        patient_idx = _patient_index(patient_ids, patient_id)
        predicted = post_reconstruct(block.source_bag, A[patient_idx], e[patient_idx])
        result = compute_balanced_sinkhorn_observation_discrepancy(
            predicted,
            block.target_bag,
            geometry,
            fov_cost_scale=fov_cost_scale.value,
            fov_cost_scale_floor_used=fov_cost_scale.floor_used,
            config=config,
        )
        value = result.value
        _validate_raw_loss(value, name="L_obs_pair_raw")
        block_id = block.block_id or f"block_{block_index}"
        metadata = metadata or result.metadata
        values.append(value)
        records.append(
            FullEstimatorObservationBlockLedger(
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
            block_values[torch_module.as_tensor(indices, dtype=torch_module.long, device=block_values.device)].mean()
        )
    raw = torch_module.stack(patient_means).mean()
    _validate_raw_loss(raw, name="L_obs_raw")
    return _ObservationRawResult(
        raw=raw,
        block_values=block_values,
        block_records=tuple(records),
        metadata=dict(metadata or {}),
    )


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


def assemble_full_estimator_totals(
    *,
    raw_components: Mapping[str, Any],
    baseline_components: Mapping[str, Any],
    alpha: float = 0.5,
    epsilon_norm: float = EPSILON_NORM,
    ablation_mode: str = "none",
) -> FullEstimatorTotals:
    """Normalize components and assemble fixed-denominator objective totals."""
    torch_module = _require_torch()
    resolved_alpha = _validate_alpha(alpha)
    resolved_epsilon = _validate_epsilon_norm(epsilon_norm)
    resolved_ablation = _validate_ablation_mode(ablation_mode)
    required_raw = ("obs", "open", "geometry", "consistency", "recurrence")
    missing_raw = [name for name in required_raw if name not in raw_components]
    if missing_raw:
        raise ContractError(f"raw_components missing required keys: {tuple(missing_raw)}")
    required_baseline = ("obs", "geometry", "consistency", "recurrence")
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
    components: dict[str, FullEstimatorComponentLedger] = {}
    for name in required_raw:
        raw = _component_tensor(raw_components[name], name=f"L_{name}_raw", device=device)
        _validate_raw_loss(raw, name=f"L_{name}_raw")
        if name == "open":
            scale = torch_module.ones((), dtype=torch_module.float64, device=raw.device)
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
        effective = torch_module.zeros_like(normalized) if resolved_ablation == name else normalized
        components[name] = FullEstimatorComponentLedger(
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
    local = (obs_eff + open_eff + geometry_eff) / 3.0
    regularization = (consistency_eff + recurrence_eff) / 2.0
    total = (1.0 - resolved_alpha) * local + resolved_alpha * regularization
    for name, value in (("L_local", local), ("L_regularization", regularization), ("L_total", total)):
        _validate_raw_loss(value, name=name)
    return FullEstimatorTotals(
        total=total,
        local=local,
        regularization=regularization,
        components=components,
        alpha=resolved_alpha,
        epsilon_norm=resolved_epsilon,
        ablation_mode=resolved_ablation,
        ablation_term_handling=(
            ABLATION_TERM_HANDLING if resolved_ablation != "none" else None
        ),
    )


def _initial_parameters_for(params: FullEstimatorParameters) -> tuple[FullEstimatorInitialization, FullEstimatorParameters]:
    A, _, _, patient_ids = _validate_parameter_shapes(params)
    init = identity_plus_small_open_initialization(int(A.shape[1]), device=A.device)
    init_params = FullEstimatorParameters(
        patient_ids=patient_ids,
        A=init.A.expand(len(patient_ids), init.K, init.K).clone(),
        d=init.d.expand(len(patient_ids), init.K).clone(),
        e=init.e.expand(len(patient_ids), init.K).clone(),
    )
    return init, init_params


def compute_full_estimator_objective(
    params: FullEstimatorParameters,
    evidence_blocks: Sequence[FullEstimatorEvidenceBlock],
    geometry: StateGeometry,
    *,
    alpha: float = 0.5,
    epsilon_norm: float = EPSILON_NORM,
    ablation_mode: str = "none",
    config: BalancedSinkhornDivergenceConfig | None = None,
) -> FullEstimatorObjectiveLedger:
    """Evaluate the canonical full-estimator objective and normalization ledger."""
    _, _, _, patient_ids = _validate_parameters(params)
    init, init_params = _initial_parameters_for(params)
    fov_cost_scales = tuple(
        _resolve_block_fov_cost_scale(block, geometry, K=init.K, config=config)
        for block in evidence_blocks
    )

    current_obs = _compute_observation_raw(
        params,
        evidence_blocks,
        geometry,
        fov_cost_scales=fov_cost_scales,
        config=config,
    )
    baseline_obs = _compute_observation_raw(
        init_params,
        evidence_blocks,
        geometry,
        fov_cost_scales=fov_cost_scales,
        config=config,
    )
    obs_scale, _ = _scale_from_baseline(
        baseline_obs.raw,
        epsilon_norm=_validate_epsilon_norm(epsilon_norm),
        name="L_obs",
    )
    current_normalized_block_losses = current_obs.block_values / obs_scale
    baseline_normalized_block_losses = baseline_obs.block_values / obs_scale
    block_patient_ids = tuple(block.patient_id for block in evidence_blocks)
    consistency_raw, consistency_records = compute_consistency_raw_from_block_losses(
        patient_ids=patient_ids,
        block_patient_ids=block_patient_ids,
        normalized_block_losses=current_normalized_block_losses,
    )
    baseline_consistency_raw, _ = compute_consistency_raw_from_block_losses(
        patient_ids=patient_ids,
        block_patient_ids=block_patient_ids,
        normalized_block_losses=baseline_normalized_block_losses,
    )

    recurrence = compute_recurrence_raw(params)
    baseline_recurrence = compute_recurrence_raw(init_params)
    totals = assemble_full_estimator_totals(
        raw_components={
            "obs": current_obs.raw,
            "open": compute_open_raw(params),
            "geometry": compute_geometry_raw(params, geometry),
            "consistency": consistency_raw,
            "recurrence": recurrence.raw,
        },
        baseline_components={
            "obs": baseline_obs.raw,
            "geometry": compute_geometry_raw(init_params, geometry),
            "consistency": baseline_consistency_raw,
            "recurrence": baseline_recurrence.raw,
        },
        alpha=alpha,
        epsilon_norm=epsilon_norm,
        ablation_mode=ablation_mode,
    )
    observation_records = tuple(
        FullEstimatorObservationBlockLedger(
            block_id=record.block_id,
            patient_id=record.patient_id,
            raw=record.raw,
            normalized=current_normalized_block_losses[index],
            status=record.status,
            fov_cost_scale=record.fov_cost_scale,
            fov_cost_scale_floor_used=record.fov_cost_scale_floor_used,
            metadata=record.metadata,
            warnings=record.warnings,
        )
        for index, record in enumerate(current_obs.block_records)
    )
    observation_metadata = dict(current_obs.metadata)
    observation_discrepancy = {
        key: observation_metadata[key]
        for key in (
            "operator_version",
            "backend",
            "dtype",
            "inner_epsilon_schedule",
            "outer_epsilon_schedule",
            "max_iter",
            "tol",
            "warning_tol",
        )
        if key in observation_metadata
    }
    state_geometry = dict(observation_metadata.get("state_geometry", {}))
    metadata = {
        "e_bounds": (0.0, 1.0),
        "post_reconstruction_form": "normalize(q_minus @ A + e)",
        "observation_comparison_plan": {
            "resolved_by": "task_layer",
            "n_evidence_blocks": len(evidence_blocks),
            "domain_policy": "observation_layer_only",
        },
        "observation_discrepancy": observation_discrepancy,
        "state_geometry": state_geometry,
        "fov_cost_scales": [
            {
                "s_G_init": item.value,
                "s_G_init_floor_used": item.floor_used,
                "positive_cost_count": item.positive_cost_count,
            }
            for item in fov_cost_scales
        ],
        "ablation_denominator_policy": totals.ablation_denominator_policy,
    }
    return FullEstimatorObjectiveLedger(
        total=totals.total,
        local=totals.local,
        regularization=totals.regularization,
        components=totals.components,
        alpha=totals.alpha,
        epsilon_norm=totals.epsilon_norm,
        initialization=init,
        observation_blocks=observation_records,
        consistency_patients=consistency_records,
        recurrence=recurrence,
        ablation_mode=totals.ablation_mode,
        ablation_term_handling=totals.ablation_term_handling,
        metadata=metadata,
    )


__all__ = [
    "ABLATION_MODES",
    "ABLATION_TERM_HANDLING",
    "EPSILON_NORM",
    "FullEstimatorComponentLedger",
    "FullEstimatorConsistencyPatientLedger",
    "FullEstimatorEvidenceBlock",
    "FullEstimatorFovCostScale",
    "FullEstimatorInitialization",
    "FullEstimatorObjectiveLedger",
    "FullEstimatorObservationBlockLedger",
    "FullEstimatorParameters",
    "FullEstimatorRecurrenceLedger",
    "FullEstimatorTotals",
    "FullEstimatorUnconstrainedParameters",
    "assemble_full_estimator_totals",
    "compute_consistency_raw_from_block_losses",
    "compute_full_estimator_objective",
    "compute_geometry_raw",
    "compute_init_fov_cost_scale",
    "compute_open_raw",
    "compute_recurrence_raw",
    "identity_plus_small_open_initialization",
    "parameters_from_unconstrained",
    "post_reconstruct",
    "unconstrained_from_constrained",
    "unconstrained_from_initialization",
]
