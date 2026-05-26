"""Stable contract re-exports for STRIDE outputs.

Package-root exports intentionally stay narrow. Diagnostics, summary helpers,
and lower-level bootstrap utilities remain available from their submodules.
"""
from __future__ import annotations

from .fit_result import (
    PatientRelationResult,
    STRIDEFitResult,
    validate_patient_relation_result,
    validate_stride_fit_result,
)
from .fit_export import (
    CohortRelationRecord,
    NativeRelationExport,
    NativeRelationExportManifest,
    PATIENT_ARRAYS_FILENAME,
    PATIENT_INDEX_FILENAME,
    PatientRelationRecord,
    STRIDE_NATIVE_RELATION_EXPORT_VERSION,
    read_stride_native_relation_export,
    validate_stride_native_relation_export,
    write_stride_native_relation_export,
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
    "CohortRelationRecord",
    "NativeRelationExport",
    "NativeRelationExportManifest",
    "PATIENT_ARRAYS_FILENAME",
    "PATIENT_INDEX_FILENAME",
    "PatientRelationResult",
    "PatientBootstrapConfig",
    "PatientBootstrapUncertaintyResult",
    "PatientRelationRecord",
    "STRIDEBootstrapUncertaintyResult",
    "STRIDEFitProvenance",
    "STRIDEFitResult",
    "STRIDE_FIT_PROVENANCE_SCHEMA_VERSION",
    "STRIDE_NATIVE_RELATION_EXPORT_VERSION",
    "build_stride_fit_provenance",
    "read_stride_native_relation_export",
    "validate_patient_relation_result",
    "validate_stride_fit_provenance",
    "validate_stride_native_relation_export",
    "validate_stride_fit_result",
    "write_stride_native_relation_export",
    "write_r_handover",
]
