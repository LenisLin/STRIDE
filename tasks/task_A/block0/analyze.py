"""Cache-derived analysis entrypoint for Task A Block 0 calibration.

Block 0 analysis reads an existing real/null STRIDE fit cache and derives the
fixed family-summary calibration tables. It does not rerun `fit_stride`, build
new null permutations, emit biological interpretation, or create downstream
execution decisions.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from stride.errors import ContractError

from .functions.cache import read_block0_fit_cache, sha256_file
from .functions.metrics import build_block0_calibration_frames
from .functions.schemas import (
    BLOCK0_ANALYSIS_SPEC_VERSION,
    BLOCK_NAME,
    CALIBRATION_MANIFEST_FILENAME,
    EXECUTION_MANIFEST_FILENAME,
    FIT_LABEL_NULL,
    FIT_LABEL_REAL,
    METRIC_SUMMARY_FILENAME,
    NULL_FAMILY,
    PATIENT_CALIBRATION_FILENAME,
    REAL_FAMILY,
    SUMMARY_ROLES,
    Block0FitRecord,
)
from .functions.writers import write_block0_analysis_outputs

_BLOCK0_ANALYSIS_OUTPUTS = (
    CALIBRATION_MANIFEST_FILENAME,
    PATIENT_CALIBRATION_FILENAME,
    METRIC_SUMMARY_FILENAME,
)


def run_block0_analyze(
    *,
    fit_cache: str | Path,
    fit_cache_index: str | Path,
    output_dir: str | Path,
    execution_manifest: str | Path | None = None,
) -> dict[str, Path]:
    """Derive fixed family-summary calibration tables from an execution cache."""
    output_root = Path(output_dir)
    _guard_no_existing_analysis_outputs(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    fit_cache_path = Path(fit_cache)
    fit_cache_index_path = Path(fit_cache_index)
    execution_manifest_path = (
        Path(execution_manifest)
        if execution_manifest is not None
        else fit_cache_path.parent / EXECUTION_MANIFEST_FILENAME
    )
    execution_payload = _load_execution_manifest(execution_manifest_path)
    if str(execution_payload.get("fit_cache_sha256")) != sha256_file(fit_cache_path):
        raise ContractError("Block 0 fit cache SHA-256 does not match the execution manifest")
    if str(execution_payload.get("fit_cache_index_sha256")) != sha256_file(fit_cache_index_path):
        raise ContractError("Block 0 fit cache index SHA-256 does not match the execution manifest")
    records = read_block0_fit_cache(fit_cache_path, fit_cache_index_path)
    n_permutations = _n_permutations_from_records(records)
    if int(execution_payload["n_permutations"]) != int(n_permutations):
        raise ContractError("Block 0 cache records do not match the execution manifest permutation count")

    real_records = tuple(record for record in records if record.fit_label == FIT_LABEL_REAL)
    null_records = tuple(record for record in records if record.fit_label == FIT_LABEL_NULL)
    patient_frame, metric_frame = build_block0_calibration_frames(
        real_records,
        null_records,
        run_scope=str(execution_payload["run_scope"]),
        n_permutations=n_permutations,
        readiness_status=str(execution_payload["readiness_status"]),
    )
    manifest_payload = _build_analysis_manifest_payload(
        execution_payload=execution_payload,
        execution_manifest_path=execution_manifest_path,
        fit_cache_path=fit_cache_path,
        fit_cache_index_path=fit_cache_index_path,
        output_root=output_root,
    )
    return write_block0_analysis_outputs(
        output_dir=output_root,
        manifest_payload=manifest_payload,
        patient_calibration=patient_frame,
        metric_summary=metric_frame,
    )


def _guard_no_existing_analysis_outputs(output_root: Path) -> None:
    existing = tuple(name for name in _BLOCK0_ANALYSIS_OUTPUTS if (output_root / name).exists())
    if existing:
        raise ContractError(
            "Block 0 analysis output_dir already contains analysis artifacts: "
            f"{existing}. Use a clean output directory."
        )


def _load_execution_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        raise ContractError(f"Block 0 execution manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ContractError("Block 0 execution manifest must be a JSON object")
    return dict(payload)


def _n_permutations_from_records(records: tuple[Block0FitRecord, ...]) -> int:
    indices = sorted(
        {
            int(record.permutation_index)
            for record in records
            if record.fit_label == FIT_LABEL_NULL
        }
    )
    if not indices:
        raise ContractError("Block 0 analysis requires null fit records")
    expected = list(range(max(indices) + 1))
    if indices != expected:
        raise ContractError("Block 0 analysis null records must cover consecutive permutation indices")
    return len(indices)


def _build_analysis_manifest_payload(
    *,
    execution_payload: Mapping[str, object],
    execution_manifest_path: Path,
    fit_cache_path: Path,
    fit_cache_index_path: Path,
    output_root: Path,
) -> dict[str, object]:
    return {
        "task_name": BLOCK_NAME,
        "config_path": str(execution_payload["config_path"]),
        "stage0_h5ad": str(execution_payload["stage0_h5ad"]),
        "run_scope": str(execution_payload["run_scope"]),
        "n_permutations": int(execution_payload["n_permutations"]),
        "master_seed": int(execution_payload["master_seed"]),
        "seed_derivation_policy": str(execution_payload["seed_derivation_policy"]),
        "real_family": REAL_FAMILY,
        "null_family": NULL_FAMILY,
        "permutation_policy": str(execution_payload["permutation_policy"]),
        "summary_roles": dict(SUMMARY_ROLES),
        "fit_status": str(execution_payload["fit_status"]),
        "readiness_status": str(execution_payload["readiness_status"]),
        "analysis_spec_version": BLOCK0_ANALYSIS_SPEC_VERSION,
        "source_execution_manifest_path": str(execution_manifest_path),
        "source_fit_cache_path": str(fit_cache_path),
        "source_fit_cache_index_path": str(fit_cache_index_path),
        "source_fit_cache_sha256": sha256_file(fit_cache_path),
        "source_fit_cache_index_sha256": sha256_file(fit_cache_index_path),
        "patient_calibration_path": str(output_root / PATIENT_CALIBRATION_FILENAME),
        "metric_summary_path": str(output_root / METRIC_SUMMARY_FILENAME),
    }


__all__ = ["run_block0_analyze"]
