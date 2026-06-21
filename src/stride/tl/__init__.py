"""Formal STRIDE fitting namespace.

The public `.tl` surface is intentionally small: `fit` runs the formal fitting
workflow, and result containers expose fitted relation outputs. Implementation
helpers remain in private modules.
"""
from __future__ import annotations

from ._output import CohortResult, FitResult, RelationResult
from ._run import fit

__all__ = (
    "fit",
    "FitResult",
    "RelationResult",
    "CohortResult",
)
