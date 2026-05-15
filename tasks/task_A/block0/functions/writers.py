"""Artifact writers for Task A Block 0 execution and analysis outputs.

Execution writers record fit-cache provenance; analysis writers record
cache-derived calibration tables. They must not create biology interpretation
prose, pass/fail gates, or downstream execution decisions. See
`tasks/task_A/README.md`, `tasks/task_A/contracts/artifact_contracts.md`, and
`tasks/task_A/contracts/design_freeze.py`.
"""
from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from stride.errors import ContractError

from .schemas import (
    CALIBRATION_MANIFEST_FILENAME,
    EXECUTION_MANIFEST_FILENAME,
    EXECUTION_MANIFEST_REQUIRED_FIELDS,
    MANIFEST_REQUIRED_FIELDS,
    METRIC_SUMMARY_COLUMNS,
    METRIC_SUMMARY_FILENAME,
    PATIENT_CALIBRATION_COLUMNS,
    PATIENT_CALIBRATION_FILENAME,
)


def validate_block0_frame_columns(
    frame: pd.DataFrame,
    expected_columns: Sequence[str],
    *,
    label: str,
) -> pd.DataFrame:
    """Validate that a formal Block 0 CSV frame exactly matches its schema."""
    observed_columns = tuple(str(column) for column in frame.columns)
    expected = tuple(expected_columns)
    if observed_columns != expected:
        missing = tuple(column for column in expected if column not in observed_columns)
        extra = tuple(column for column in observed_columns if column not in expected)
        raise ContractError(
            f"{label} columns do not match the Block 0 schema; "
            f"missing={missing}, extra={extra}, observed={observed_columns}"
        )
    return frame


def _validate_manifest_payload(manifest_payload: Mapping[str, object]) -> dict[str, object]:
    missing = tuple(field for field in MANIFEST_REQUIRED_FIELDS if field not in manifest_payload)
    extra = tuple(field for field in manifest_payload if field not in MANIFEST_REQUIRED_FIELDS)
    if missing or extra:
        raise ContractError(
            "Block 0 manifest payload does not match the Block 0 schema; "
            f"missing={missing}, extra={extra}"
        )
    return dict(manifest_payload)


def _validate_execution_manifest_payload(manifest_payload: Mapping[str, object]) -> dict[str, object]:
    missing = tuple(field for field in EXECUTION_MANIFEST_REQUIRED_FIELDS if field not in manifest_payload)
    extra = tuple(field for field in manifest_payload if field not in EXECUTION_MANIFEST_REQUIRED_FIELDS)
    if missing or extra:
        raise ContractError(
            "Block 0 execution manifest payload does not match the Block 0 schema; "
            f"missing={missing}, extra={extra}"
        )
    return dict(manifest_payload)


def write_block0_execution_outputs(
    *,
    output_dir: str | Path,
    manifest_payload: Mapping[str, object],
) -> dict[str, Path]:
    """Write the Block 0 execution manifest after the fit cache exists."""
    manifest = _validate_execution_manifest_payload(manifest_payload)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / EXECUTION_MANIFEST_FILENAME

    with tempfile.TemporaryDirectory(
        dir=output_path,
        prefix=".block0_execution_",
        suffix=".tmp",
    ) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        temp_manifest_path = temp_dir / EXECUTION_MANIFEST_FILENAME
        with temp_manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with temp_manifest_path.open("r", encoding="utf-8") as handle:
            _validate_execution_manifest_payload(json.load(handle))
        temp_manifest_path.replace(manifest_path)

    return {"manifest": manifest_path}


def write_block0_analysis_outputs(
    *,
    output_dir: str | Path,
    manifest_payload: Mapping[str, object],
    patient_calibration: pd.DataFrame,
    metric_summary: pd.DataFrame,
) -> dict[str, Path]:
    """Write the three derived Block 0 analysis artifacts.

    These artifacts are derived from an existing execution cache; this writer
    must not be called by the fit/permutation execution path.
    """
    manifest = _validate_manifest_payload(manifest_payload)
    patient_frame = validate_block0_frame_columns(
        patient_calibration,
        PATIENT_CALIBRATION_COLUMNS,
        label=PATIENT_CALIBRATION_FILENAME,
    )
    metric_frame = validate_block0_frame_columns(
        metric_summary,
        METRIC_SUMMARY_COLUMNS,
        label=METRIC_SUMMARY_FILENAME,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_path = output_path / CALIBRATION_MANIFEST_FILENAME
    patient_path = output_path / PATIENT_CALIBRATION_FILENAME
    metric_path = output_path / METRIC_SUMMARY_FILENAME

    with tempfile.TemporaryDirectory(
        dir=output_path,
        prefix=".block0_calibration_",
        suffix=".tmp",
    ) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        temp_manifest_path = temp_dir / CALIBRATION_MANIFEST_FILENAME
        temp_patient_path = temp_dir / PATIENT_CALIBRATION_FILENAME
        temp_metric_path = temp_dir / METRIC_SUMMARY_FILENAME

        with temp_manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        patient_frame.to_csv(temp_patient_path, index=False)
        metric_frame.to_csv(temp_metric_path, index=False)

        _validate_written_outputs(
            manifest_path=temp_manifest_path,
            patient_path=temp_patient_path,
            metric_path=temp_metric_path,
        )
        temp_patient_path.replace(patient_path)
        temp_metric_path.replace(metric_path)
        temp_manifest_path.replace(manifest_path)

    return {
        "manifest": manifest_path,
        "patient_calibration": patient_path,
        "metric_summary": metric_path,
    }


def _validate_written_outputs(
    *,
    manifest_path: Path,
    patient_path: Path,
    metric_path: Path,
) -> None:
    with manifest_path.open("r", encoding="utf-8") as handle:
        _validate_manifest_payload(json.load(handle))
    validate_block0_frame_columns(
        pd.read_csv(patient_path, nrows=0),
        PATIENT_CALIBRATION_COLUMNS,
        label=PATIENT_CALIBRATION_FILENAME,
    )
    validate_block0_frame_columns(
        pd.read_csv(metric_path, nrows=0),
        METRIC_SUMMARY_COLUMNS,
        label=METRIC_SUMMARY_FILENAME,
    )


__all__ = [
    "CALIBRATION_MANIFEST_FILENAME",
    "METRIC_SUMMARY_FILENAME",
    "PATIENT_CALIBRATION_FILENAME",
    "validate_block0_frame_columns",
    "write_block0_analysis_outputs",
    "write_block0_execution_outputs",
]
