"""Transition wrapper for canonical STRIDE pathology diagnostics."""
from __future__ import annotations

from stride.outputs.diagnostics import (
    PathologyDiagnostic,
    audit_failure_modes,
    reject_pathological_relation,
)

__all__ = ["PathologyDiagnostic", "audit_failure_modes", "reject_pathological_relation"]
