"""Parameter containers and transforms for patient-level STRIDE relations.

This module owns the formal torch parameterization for constrained
patient-level `A/d/e`. It declares unconstrained optimizer logits, the
constrained relation state, and the target-composition prediction form
`normalize(v_source @ A + e)`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from stride.errors import ContractError

NUMERICAL_MIN_MASS = 1e-12
OFFDIAG_INIT_MASS = 1e-2


@dataclass(frozen=True)
class ParameterLogits:
    """Unconstrained optimizer variables for relation parameters.

    patient_ids: patient ids aligned to optimizer axis `P`.
    row_logits: tensor `[P, K, K + 1]` for each `[A_i,* , d_i]` simplex.
    e_logits: tensor `[P, K]` for bounded target-side `e`.
    """

    patient_ids: tuple[str, ...]
    row_logits: torch.Tensor
    e_logits: torch.Tensor


@dataclass(frozen=True)
class RelationParameters:
    """Constrained patient-level relation parameters.

    patient_ids: patient ids aligned to axis `P`.
    A: transition tensor `[P, K, K]`.
    d: source-row open channel tensor `[P, K]`.
    e: target-side additive open tendency tensor `[P, K]`.
    """

    patient_ids: tuple[str, ...]
    A: torch.Tensor
    d: torch.Tensor
    e: torch.Tensor


def initialize_parameters(
    patient_ids: Sequence[str],
    n_states: int,
    *,
    device: Any | None = None,
) -> ParameterLogits:
    """Create canonical optimizer-start logits for one relation.

    Purpose:
        Declare the off-diagonal-seeded identity-plus-small-open initialization
        used by the formal optimizer.

    Key variables:
        normalized_patient_ids: patient ids aligned to optimizer axis `P`.
        delta_init: open-channel initialization mass.
        row_logits: unconstrained logits for `[A_i,* , d_i]`.
        e_logits: unconstrained logits for bounded `e`.
    """
    normalized_patient_ids = _normalize_patient_ids(patient_ids)
    K = _require_positive_int(n_states, name="n_states")
    P = len(normalized_patient_ids)
    delta_init = min(0.05, 1.0 / float(K + 1))

    A = torch.eye(K, dtype=torch.float64, device=device).unsqueeze(0).repeat(P, 1, 1)
    d = torch.full((P, K), delta_init, dtype=torch.float64, device=device)
    e = torch.full((P, K), delta_init / float(K), dtype=torch.float64, device=device)

    if K > 1:
        diagonal_start = 1.0 - delta_init - float(K - 1) * OFFDIAG_INIT_MASS
        if diagonal_start <= 0.0:
            raise ContractError(
                "offdiag optimizer initialization is invalid for n_states: "
                "1 - delta_init - (K - 1) * offdiag_init_mass must be positive"
            )
        A.fill_(OFFDIAG_INIT_MASS)
        diagonal = torch.arange(K, device=device)
        A[:, diagonal, diagonal] = diagonal_start
    else:
        A[:, 0, 0] = 1.0 - delta_init

    return ParameterLogits(
        patient_ids=normalized_patient_ids,
        row_logits=torch.log(torch.cat([A, d.unsqueeze(2)], dim=2)),
        e_logits=torch.logit(e),
    )


def constrain_parameters(logits: ParameterLogits) -> RelationParameters:
    """Transform optimizer logits into constrained `A/d/e`.

    Purpose:
        Apply row-wise softmax to `[A_i,* , d_i]` and sigmoid to `e`.

    Key variables:
        row_simplex: feasible `[P, K, K + 1]` tensor.
        A: transition tensor `[P, K, K]`.
        d: source-row open channel `[P, K]`.
        e: bounded target-side open tendency `[P, K]`.
    """
    if not isinstance(logits, ParameterLogits):
        raise ContractError("logits must be a ParameterLogits object")
    patient_ids = _normalize_patient_ids(logits.patient_ids)
    row_logits = _as_float64_tensor(logits.row_logits, name="row_logits")
    e_logits = _as_float64_tensor(logits.e_logits, name="e_logits", device=row_logits.device)

    if row_logits.ndim != 3:
        raise ContractError("row_logits must be a [P, K, K + 1] tensor")
    P = len(patient_ids)
    if row_logits.shape[0] != P:
        raise ContractError("row_logits first dimension must align with patient_ids")
    K = int(row_logits.shape[1])
    if K <= 0 or row_logits.shape[2] != K + 1:
        raise ContractError("row_logits must have shape [P, K, K + 1]")
    if e_logits.shape != (P, K):
        raise ContractError("e_logits must have shape [P, K] aligned with row_logits")

    row_simplex = torch.softmax(row_logits, dim=2)
    return RelationParameters(
        patient_ids=patient_ids,
        A=row_simplex[:, :, :K],
        d=row_simplex[:, :, K],
        e=torch.sigmoid(e_logits),
    )


def predict_target_composition(
    source_bag: torch.Tensor,
    A: torch.Tensor,
    e: torch.Tensor,
) -> torch.Tensor:
    """Return `normalize(source_bag @ A + e)` for source FOV bags.

    Purpose:
        Declare the frozen target-side FOV reconstruction form.

    Key variables:
        raw_post: unnormalized predicted target composition.
        row_sums: per-FOV normalization denominator.
        predicted: normalized predicted target FOV composition.
    """
    A_tensor = _as_float64_tensor(A, name="A")
    source = _as_float64_tensor(source_bag, name="source_bag", device=A_tensor.device)
    e_tensor = _as_float64_tensor(e, name="e", device=A_tensor.device)

    _ensure_non_empty_matrix(source, name="source_bag")
    K = int(source.shape[1])
    if A_tensor.shape != (K, K):
        raise ContractError("A must be a [K, K] tensor aligned with source_bag")
    if e_tensor.shape != (K,):
        raise ContractError("e must be a [K] tensor aligned with source_bag")

    raw_post = source @ A_tensor + e_tensor
    row_sums = raw_post.sum(dim=1, keepdim=True)
    return raw_post / row_sums


def _normalize_patient_ids(patient_ids: Sequence[str]) -> tuple[str, ...]:
    """Return normalized non-empty patient ids.

    Purpose:
        Define the patient axis order shared by logits, parameters, and output.

    Key variables:
        normalized: stripped string patient identifiers.
        duplicates: repeated patient ids that would make axis `P` ambiguous.
    """
    if isinstance(patient_ids, (str, bytes)):
        raise ContractError("patient_ids must be a sequence, not a string")
    normalized = tuple(str(item).strip() for item in patient_ids)
    if len(normalized) == 0:
        raise ContractError("patient_ids must be non-empty")
    if any(item == "" for item in normalized):
        raise ContractError("patient_ids must contain non-empty identifiers")
    if len(set(normalized)) != len(normalized):
        raise ContractError("patient_ids must not contain duplicates")
    return normalized


def _require_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return int(value)


def _as_float64_tensor(
    value: Any,
    *,
    name: str,
    device: Any | None = None,
) -> torch.Tensor:
    try:
        if torch.is_tensor(value):
            resolved_device = value.device if device is None else device
            return value.to(device=resolved_device, dtype=torch.float64)
        return torch.as_tensor(value, dtype=torch.float64, device=device)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be coercible to a float64 tensor") from exc


def _finite_scalar_bool(value: Any) -> bool:
    scalar = value.detach().cpu().item() if torch.is_tensor(value) else value
    return bool(math.isfinite(float(scalar)) and scalar)


def _ensure_finite_tensor(value: torch.Tensor, *, name: str) -> None:
    if not _finite_scalar_bool(torch.isfinite(value).all()):
        raise ContractError(f"{name} must contain only finite values")


def _ensure_distribution_matrix(matrix: torch.Tensor, *, name: str) -> None:
    _ensure_non_empty_matrix(matrix, name=name)
    _ensure_finite_tensor(matrix, name=name)
    if _finite_scalar_bool((matrix < 0.0).any()):
        raise ContractError(f"{name} entries must be nonnegative")
    row_sums = matrix.sum(dim=1)
    if not _finite_scalar_bool(
        torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            rtol=1e-8,
            atol=1e-8,
        )
    ):
        raise ContractError(f"{name} rows must sum to 1.0")


def _ensure_non_empty_matrix(matrix: torch.Tensor, *, name: str) -> None:
    if matrix.ndim != 2:
        raise ContractError(f"{name} must be a 2D [N, K] distribution matrix")
    if matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise ContractError(f"{name} must be a non-empty [N, K] distribution matrix")
