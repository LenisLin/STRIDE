"""Explicit preserved proxy entrypoints for the narrow STRIDE compatibility path."""
from __future__ import annotations

from .fit import ProxySTRIDEFitConfig, fit_stride_proxy

__all__ = [
    "ProxySTRIDEFitConfig",
    "fit_stride_proxy",
]
