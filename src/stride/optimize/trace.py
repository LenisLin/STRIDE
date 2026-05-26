"""Optimizer trace type aliases for STRIDE training."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

TracePayload = Mapping[str, Any]

__all__ = ["TracePayload"]
