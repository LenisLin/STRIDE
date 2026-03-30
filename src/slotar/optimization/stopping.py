"""Transition wrapper for canonical STRIDE stopping and pathology helpers."""
from __future__ import annotations

from stride.optimize.stopping import PathologyCheck, StoppingCriteria, detect_pathologies, should_stop

__all__ = ["PathologyCheck", "StoppingCriteria", "detect_pathologies", "should_stop"]
