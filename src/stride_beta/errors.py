"""Shared exception types for canonical STRIDE contracts and adapters."""
from __future__ import annotations


class ContractError(ValueError):
    """Raised when a declared STRIDE interface contract is violated."""


class UOTInputError(ValueError):
    """Raised for programmer misuse of observation-layer adapter entrypoints."""


__all__ = ["ContractError", "UOTInputError"]
