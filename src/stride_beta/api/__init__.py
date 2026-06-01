"""Stable public facades for the canonical STRIDE package."""
from __future__ import annotations

from .basis import BasisSpec
from .dataset import DatasetHandle
from .fit import build_patient_relation, fit_stride
from .summary import summarize_fit

__all__ = [
    "BasisSpec",
    "DatasetHandle",
    "build_patient_relation",
    "fit_stride",
    "summarize_fit",
]
