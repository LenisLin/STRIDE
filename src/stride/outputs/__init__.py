"""Stable contract re-exports for STRIDE outputs.

Package-root exports intentionally stay narrow. Diagnostics, summary helpers,
and lower-level bootstrap utilities remain available from their submodules.
"""
from __future__ import annotations

from .fit_result import (
    PatientBridgeResult,
    PatientRelationFitResult,
    STRIDEFitResult,
    validate_patient_bridge_result,
    validate_stride_fit_result,
)
from .provenance import (
    STRIDE_FIT_PROVENANCE_SCHEMA_VERSION,
    STRIDEFitProvenance,
    build_stride_fit_provenance,
    validate_stride_fit_provenance,
)
from .r_export import EVENTS_FILENAME, META_FILENAME, METRICS_FILENAME, write_r_handover
from .uncertainty import (
    BootstrapArraySummary,
    BootstrapConfig,
    CohortBootstrapUncertaintySummary,
    PatientBootstrapConfig,
    PatientBootstrapUncertaintyResult,
    STRIDEBootstrapUncertaintyResult,
)

__all__ = [
    "BootstrapArraySummary",
    "BootstrapConfig",
    "CohortBootstrapUncertaintySummary",
    "EVENTS_FILENAME",
    "META_FILENAME",
    "METRICS_FILENAME",
    "PatientBridgeResult",
    "PatientBootstrapConfig",
    "PatientBootstrapUncertaintyResult",
    "PatientRelationFitResult",
    "STRIDEBootstrapUncertaintyResult",
    "STRIDEFitProvenance",
    "STRIDEFitResult",
    "STRIDE_FIT_PROVENANCE_SCHEMA_VERSION",
    "build_stride_fit_provenance",
    "validate_patient_bridge_result",
    "validate_stride_fit_provenance",
    "validate_stride_fit_result",
    "write_r_handover",
]
