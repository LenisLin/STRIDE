"""Shared pure array and axis validation helpers for STRIDE internals.

This module contains only representation-level checks that have identical
meaning across namespaces. Scientific shape and alignment rules remain with
their owning `.tl`, `.da`, `.io`, or `.pl` surfaces.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from stride.errors import ContractError


def as_finite_float_array(
    value: object,
    *,
    name: str,
    ndim: int | None = None,
) -> np.ndarray:
    """Return a finite float array with an optional representation dimension."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be numeric") from exc
    if ndim is not None and array.ndim != ndim:
        raise ContractError(f"{name} must be a {ndim}D array")
    if not np.isfinite(array).all():
        raise ContractError(f"{name} must contain only finite values")
    return array


def resolve_full_axis_order(
    size: int,
    order: Sequence[int] | None,
    *,
    name: str,
) -> tuple[int, ...]:
    """Resolve a complete permutation of an integer axis."""
    if order is None:
        return tuple(range(size))
    resolved = tuple(int(value) for value in order)
    if sorted(resolved) != list(range(size)):
        raise ContractError(f"{name} must be a full permutation of 0..{size - 1}")
    return resolved


def resolve_axis_labels(
    size: int,
    labels: Sequence[str] | None,
    *,
    name: str,
    prefix: str,
) -> tuple[str, ...]:
    """Resolve fixed-length string labels for one axis."""
    if labels is None:
        return tuple(f"{prefix}{index}" for index in range(size))
    resolved = tuple(str(value) for value in labels)
    if len(resolved) != size:
        raise ContractError(f"{name} length must match axis size")
    return resolved
