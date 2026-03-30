"""Transition wrapper for canonical STRIDE error types."""
from __future__ import annotations

from stride.errors import ContractError, DataContractError, UOTInputError

__all__ = ["ContractError", "DataContractError", "UOTInputError"]
