"""Compatibility exception and status exports for migration-era imports."""
from __future__ import annotations

from ..errors import ContractError, DataContractError, UOTInputError
from .status import (
    ERR_UOT_EMPTY_MASS_SOURCE,
    ERR_UOT_EMPTY_MASS_TARGET,
    ERR_UOT_EMPTY_SUPPORT,
    ERR_UOT_NUMERICAL,
)

__all__ = [
    "ContractError",
    "DataContractError",
    "ERR_UOT_EMPTY_MASS_SOURCE",
    "ERR_UOT_EMPTY_MASS_TARGET",
    "ERR_UOT_EMPTY_SUPPORT",
    "ERR_UOT_NUMERICAL",
    "UOTInputError",
]
