"""STRIDE public package namespace."""
from __future__ import annotations

from ._version import __version__
from .errors import ContractError
from .tl import CohortResult, FitResult, RelationResult, fit

__all__ = (
    "__version__",
    "ContractError",
    "fit",
    "FitResult",
    "RelationResult",
    "CohortResult",
)
