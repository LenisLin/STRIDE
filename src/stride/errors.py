"""Shared exception types for STRIDE contracts."""
from __future__ import annotations


class ContractError(ValueError):
    """Raised when a declared STRIDE interface contract is violated."""


__all__ = ["ContractError"]
