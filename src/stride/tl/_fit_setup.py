"""One-time input and scale preparation for STRIDE relation fitting.

This module owns device resolution, evidence materialization, and the fixed
identity-plus-small-open scale reference used before optimization. It does not
run optimizer steps or assemble public results.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from stride.errors import ContractError

from ._parameters import RelationParameters
from ._resolve import EvidenceBlock, RelationInput


def resolve_runtime_device(device: Any) -> torch.device:
    """Resolve an available runtime device supporting the float64 fit contract."""
    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ContractError(f"invalid runtime device: {device!r}") from exc

    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ContractError(f"requested runtime device is unavailable: {resolved}")
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise ContractError(f"requested runtime device is unavailable: {resolved}")
        return resolved

    if resolved.type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ContractError(f"requested runtime device is unavailable: {resolved}")
        try:
            torch.empty((), dtype=torch.float64, device=resolved)
        except (TypeError, RuntimeError) as exc:
            raise ContractError(
                f"requested runtime device does not support float64 tensors: {resolved}"
            ) from exc
    return resolved


def materialize_relation_inputs(
    relation: RelationInput,
    cost_matrix: torch.Tensor,
    cost_scale: float,
    *,
    device: Any | None,
) -> tuple[RelationInput, tuple[EvidenceBlock, ...], torch.Tensor, float]:
    """Validate and place fixed relation evidence on the fit device once."""
    if not isinstance(relation, RelationInput):
        raise ContractError("relation must be a RelationInput object")
    if not relation.patient_ids:
        raise ContractError("relation.patient_ids must be non-empty")
    if not relation.blocks:
        raise ContractError("relation.blocks must be non-empty")

    scale = _positive_finite_float(cost_scale, name="cost_scale")
    try:
        cost = torch.as_tensor(cost_matrix, dtype=torch.float64, device=device)
    except (TypeError, ValueError) as exc:
        raise ContractError("cost_matrix must be coercible to a float64 tensor") from exc
    if cost.ndim != 2 or cost.shape[0] <= 0 or cost.shape[0] != cost.shape[1]:
        raise ContractError("cost_matrix must be a non-empty square [K, K] tensor")

    materialized_blocks: list[EvidenceBlock] = []
    n_states = int(cost.shape[0])
    for block in relation.blocks:
        if not isinstance(block, EvidenceBlock):
            raise ContractError("relation.blocks must contain EvidenceBlock objects")
        source = _as_float64_matrix(block.source_bag, name="source_bag", device=cost.device)
        target = _as_float64_matrix(block.target_bag, name="target_bag", device=cost.device)
        if source.shape[1] != target.shape[1]:
            raise ContractError("source_bag and target_bag must share the K-state axis")
        if source.shape[1] != n_states:
            raise ContractError("evidence bags must align with cost_matrix K-state axis")
        materialized_blocks.append(
            EvidenceBlock(
                patient_id=str(block.patient_id),
                source_bag=source,
                target_bag=target,
                block_id=str(block.block_id),
                metadata=block.metadata,
            )
        )

    blocks = tuple(materialized_blocks)
    materialized_relation = RelationInput(
        relation_id=relation.relation_id,
        source_timepoint=relation.source_timepoint,
        target_timepoint=relation.target_timepoint,
        source_domain=relation.source_domain,
        target_domain=relation.target_domain,
        patient_ids=tuple(str(item) for item in relation.patient_ids),
        support_counts=relation.support_counts,
        skipped_patient_ids=relation.skipped_patient_ids,
        blocks=blocks,
        metadata=relation.metadata,
    )
    return materialized_relation, blocks, cost, scale


def scale_initial_parameters(
    patient_ids: Sequence[str],
    n_states: int,
    *,
    device: Any,
) -> RelationParameters:
    """Build the fixed objective-scale reference, not optimizer-start logits.

    The identity-plus-small-open reference has no off-diagonal seed and is used
    only to normalize objective components and derive fixed observation scales.
    """
    normalized_patient_ids = _normalize_patient_ids(patient_ids)
    n_states = _positive_int(n_states, name="n_states")
    n_patients = len(normalized_patient_ids)
    delta_init = min(0.05, 1.0 / float(n_states + 1))

    A = (1.0 - delta_init) * torch.eye(n_states, dtype=torch.float64, device=device)
    A = A.unsqueeze(0).repeat(n_patients, 1, 1)
    d = torch.full((n_patients, n_states), delta_init, dtype=torch.float64, device=device)
    e = torch.full(
        (n_patients, n_states),
        delta_init / float(n_states),
        dtype=torch.float64,
        device=device,
    )
    return RelationParameters(patient_ids=normalized_patient_ids, A=A, d=d, e=e)


def _as_float64_matrix(value: Any, *, name: str, device: Any) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64, device=device)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be coercible to a float64 tensor") from exc
    if tensor.ndim != 2:
        raise ContractError(f"{name} must be a 2D tensor")
    if tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
        raise ContractError(f"{name} must be non-empty")
    return tensor


def _positive_finite_float(value: Any, *, name: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be finite and strictly positive") from exc
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ContractError(f"{name} must be finite and strictly positive")
    return resolved


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return int(value)


def _normalize_patient_ids(patient_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(patient_ids, (str, bytes)):
        raise ContractError("patient_ids must be a sequence, not a string")
    normalized = tuple(str(item).strip() for item in patient_ids)
    if not normalized:
        raise ContractError("patient_ids must be non-empty")
    if any(item == "" for item in normalized):
        raise ContractError("patient_ids must contain non-empty identifiers")
    if len(set(normalized)) != len(normalized):
        raise ContractError("patient_ids must not contain duplicates")
    return normalized
