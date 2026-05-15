"""Private parameter transforms and tensor validators for STRIDE losses."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..errors import ContractError
from ..geometry.state_geometry import StateGeometry
from ._constants import ABLATION_MODES, EPSILON_NORM, NUMERICAL_MIN_MASS, OFFDIAG_INIT_MASS

try:  # pragma: no cover - exercised through _require_torch when unavailable
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

@dataclass
class LogitState:
    """Unconstrained optimizer variables for patient-level ``A/d/e``."""

    patient_ids: tuple[str, ...]
    row_logits: Any
    e_logits: Any


@dataclass(frozen=True)
class ADEState:
    """Constrained patient-level ``A/d/e`` tensors."""

    patient_ids: tuple[str, ...]
    A: Any
    d: Any
    e: Any


def _require_torch() -> Any:
    if torch is None:  # pragma: no cover - depends on optional runtime
        raise ContractError("canonical loss assembly objective requires torch")
    return torch


def _device_key(device: Any) -> str:
    return str(device)


def _tensor_data_ptr(value: Any) -> int | None:
    torch_module = _require_torch()
    if not torch_module.is_tensor(value):
        return None
    return int(value.data_ptr())


def _tensor_mutation_version(value: Any) -> int | None:
    torch_module = _require_torch()
    if not torch_module.is_tensor(value):
        return None
    version = getattr(value, "_version", None)
    return None if version is None else int(version)


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


def _validate_objective_weights(weights: Sequence[float]) -> tuple[float, float, float]:
    weights_tuple = tuple(weights) if not isinstance(weights, (str, bytes)) else ()
    if len(weights_tuple) != 3:
        raise ContractError("objective_weights must be a three-item sequence ordered as (fit, prior, cohort)")
    resolved = tuple(float(item) for item in weights_tuple)
    if any(not math.isfinite(item) or item < 0.0 for item in resolved):
        raise ContractError("objective_weights entries must be finite and nonnegative")
    if sum(resolved) <= 0.0:
        raise ContractError("objective_weights must contain at least one positive entry")
    if resolved != (1.0, 1.0, 1.0):
        raise ContractError(
            "objective_weights is fixed to (1.0, 1.0, 1.0) for the "
            "stride_full_estimator_three_block_v1 reference objective"
        )
    return resolved  # type: ignore[return-value]


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
    if not math.isclose(resolved, EPSILON_NORM, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError(
            "epsilon_norm is fixed to 0.01 for the "
            "stride_full_estimator_three_block_v1 reference objective"
        )
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



def unconstrained_from_constrained(
    patient_ids: Sequence[str],
    A: Any,
    d: Any,
    e: Any,
) -> LogitState:
    """Build softmax/sigmoid logits from feasible constrained tensors."""
    torch_module = _require_torch()
    normalized_patient_ids = _normalize_patient_ids(patient_ids)
    A_tensor = _as_float64_tensor(A, name="A")
    d_tensor = _as_float64_tensor(d, name="d", device=A_tensor.device)
    e_tensor = _as_float64_tensor(e, name="e", device=A_tensor.device)
    params = ADEState(
        patient_ids=normalized_patient_ids,
        A=A_tensor,
        d=d_tensor,
        e=e_tensor,
    )
    _validate_parameters(params)

    row_simplex = torch_module.cat([A_tensor, d_tensor.unsqueeze(2)], dim=2)
    row_simplex = torch_module.clamp(row_simplex, min=NUMERICAL_MIN_MASS)
    row_simplex = row_simplex / row_simplex.sum(dim=2, keepdim=True)
    row_logits = torch_module.log(row_simplex)
    e_logits = torch_module.logit(
        torch_module.clamp(
            e_tensor,
            min=NUMERICAL_MIN_MASS,
            max=1.0 - NUMERICAL_MIN_MASS,
        )
    )
    return LogitState(
        patient_ids=normalized_patient_ids,
        row_logits=row_logits,
        e_logits=e_logits,
    )


def unconstrained_from_initialization(
    patient_ids: Sequence[str],
    K: int,
    *,
    device: Any | None = None,
) -> LogitState:
    """Return unconstrained parameters whose transform equals canonical init."""
    torch_module = _require_torch()
    normalized_patient_ids = _normalize_patient_ids(patient_ids)
    from ._initialization import identity_plus_small_open_initialization

    init = identity_plus_small_open_initialization(K, device=device)
    A = init.A.expand(len(normalized_patient_ids), init.K, init.K).clone()
    d = init.d.expand(len(normalized_patient_ids), init.K).clone()
    e = init.e.expand(len(normalized_patient_ids), init.K).clone()
    if init.K > 1:
        diagonal_start = 1.0 - init.delta_init - float(init.K - 1) * OFFDIAG_INIT_MASS
        if diagonal_start <= 0.0:
            raise ContractError(
                "offdiag optimizer initialization is invalid for K: "
                "1 - delta_init - (K - 1) * offdiag_init_mass must be positive"
            )
        off_diagonal = ~torch_module.eye(init.K, dtype=torch_module.bool, device=device)
        A[:, off_diagonal] = OFFDIAG_INIT_MASS
        diagonal = torch_module.arange(init.K, device=device)
        A[:, diagonal, diagonal] = diagonal_start
    row_simplex = torch_module.cat([A, d.unsqueeze(2)], dim=2)
    row_logits = torch_module.log(row_simplex)
    e_logits = torch_module.logit(e)
    return LogitState(
        patient_ids=normalized_patient_ids,
        row_logits=row_logits,
        e_logits=e_logits,
    )


def parameters_from_unconstrained(
    unconstrained: LogitState,
) -> ADEState:
    """Transform unconstrained logits into feasible constrained ``A/d/e``."""
    torch_module = _require_torch()
    if not isinstance(unconstrained, LogitState):
        raise ContractError("unconstrained must be a LogitState object")
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
    return ADEState(
        patient_ids=patient_ids,
        A=row_simplex[:, :, :K],
        d=row_simplex[:, :, K],
        e=torch_module.sigmoid(e_logits),
    )


def _validate_parameters(params: ADEState) -> tuple[Any, Any, Any, tuple[str, ...]]:
    torch_module = _require_torch()
    if not isinstance(params, ADEState):
        raise ContractError("params must be a ADEState object")
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

def _validate_parameter_shapes(params: ADEState) -> tuple[Any, Any, Any, tuple[str, ...]]:
    if not isinstance(params, ADEState):
        raise ContractError("params must be a ADEState object")
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

__all__ = [
    "ADEState",
    "LogitState",
    "parameters_from_unconstrained",
    "post_reconstruct",
    "unconstrained_from_constrained",
    "unconstrained_from_initialization",
    "_as_float64_tensor",
    "_component_tensor",
    "_device_key",
    "_ensure_distribution_matrix",
    "_ensure_finite_tensor",
    "_finite_scalar_bool",
    "_normalized_geometry_cost",
    "_normalize_patient_ids",
    "_patient_index",
    "_require_positive_int",
    "_require_torch",
    "_validate_ablation_mode",
    "_validate_epsilon_norm",
    "_validate_objective_weights",
    "_validate_parameter_shapes",
    "_validate_parameters",
    "_validate_raw_loss",
]
