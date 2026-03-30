"""Canonical STRIDE package root with a deliberately small stable surface."""
from __future__ import annotations

from .api import BasisSpec, DatasetHandle, build_patient_relation, fit_stride, summarize_fit
from .errors import ContractError, DataContractError

__all__ = [
    "BasisSpec",
    "ContractError",
    "DataContractError",
    "DatasetHandle",
    "build_patient_relation",
    "fit_stride",
    "summarize_fit",
]
