"""Stable first-pass public facades for the canonical STRIDE package.

Package-root stable exports live here. Expert-only, compatibility, and
explicitly deferred surfaces remain available from narrower submodules.
"""
from __future__ import annotations

from .basis import BasisSpec
from .dataset import DatasetHandle
from .fit import BridgeConfig, STRIDEFitConfig, build_patient_relation, fit_stride
from .summary import summarize_fit

__all__ = [
    "BasisSpec",
    "BridgeConfig",
    "DatasetHandle",
    "STRIDEFitConfig",
    "build_patient_relation",
    "fit_stride",
    "summarize_fit",
]
