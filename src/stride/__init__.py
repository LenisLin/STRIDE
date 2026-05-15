"""Canonical STRIDE package root with a deliberately small stable surface."""
from __future__ import annotations

from ._version import __version__
from .api import BasisSpec, DatasetHandle, build_patient_relation, fit_stride, summarize_fit
from .errors import ContractError

__all__ = [
    "BasisSpec",
    "ContractError",
    "DatasetHandle",
    "__version__",
    "build_patient_relation",
    "fit_stride",
    "summarize_fit",
]
