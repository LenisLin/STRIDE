"""Thin validation facade for the canonical STRIDE data-layer contract."""
from __future__ import annotations

from .longitudinal import validate_longitudinal_adata


def validate_dataset(*args: object, **kwargs: object) -> None:
    """Validate dataset inputs through the current longitudinal AnnData contract."""
    validate_longitudinal_adata(*args, **kwargs)  # type: ignore[arg-type]


__all__ = ["validate_dataset", "validate_longitudinal_adata"]
