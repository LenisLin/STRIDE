"""Task A Block 0 `TC-IM` execution-cache and analysis package.

Block 0 execution runs real/null full STRIDE fits and writes a reusable
`A,d,e,mu` fit cache. Calibration tables are derived later by the analysis
surface. Block 0 consumes Stage 0 h5ad, Task A config, run controls, and
optional selectors; it does not emit biological interpretation or downstream
execution decisions. See
`tasks/task_A/README.md`, `tasks/task_A/contracts/artifact_contracts.md`, and
`tasks/task_A/contracts/design_freeze.py`.
"""
from __future__ import annotations

from .analyze import run_block0_analyze
from .fit import (
    BLOCK_NAME,
    CALIBRATION_MANIFEST_FILENAME,
    DEFAULT_N_PERMUTATIONS,
    EXECUTION_MANIFEST_FILENAME,
    FIT_CACHE_FILENAME,
    FIT_CACHE_INDEX_FILENAME,
    METRIC_SUMMARY_FILENAME,
    NULL_FAMILY,
    PATIENT_CALIBRATION_FILENAME,
    parse_args,
    run_block0_execute,
)

__all__ = [
    "BLOCK_NAME",
    "CALIBRATION_MANIFEST_FILENAME",
    "DEFAULT_N_PERMUTATIONS",
    "EXECUTION_MANIFEST_FILENAME",
    "FIT_CACHE_FILENAME",
    "FIT_CACHE_INDEX_FILENAME",
    "METRIC_SUMMARY_FILENAME",
    "NULL_FAMILY",
    "PATIENT_CALIBRATION_FILENAME",
    "parse_args",
    "run_block0_analyze",
    "run_block0_execute",
]
