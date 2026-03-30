"""Shared exception types for canonical STRIDE contracts and adapters."""
from __future__ import annotations


class ContractError(ValueError):
    """Raised when a declared STRIDE interface contract is violated."""


# Compatibility alias retained for legacy validation-facing call sites.
DataContractError = ContractError


class UOTInputError(ValueError):
    """Raised for programmer misuse of observation-layer adapter entrypoints."""


__all__ = ["ContractError", "DataContractError", "UOTInputError"]
