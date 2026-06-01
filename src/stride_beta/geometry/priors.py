"""Compatibility alias for geometry-layer prior objects on a shared state basis."""
from __future__ import annotations

from .state_geometry import StateGeometry

# Compatibility alias retained for callers that speak in terms of geometry priors.
GeometryPrior = StateGeometry

__all__ = ["GeometryPrior"]
