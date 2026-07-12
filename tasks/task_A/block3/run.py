"""Resumable formal coordinator for Task A Block 3."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stride.errors import ContractError

from . import execution
from .contracts import Block3SubexperimentId

_FORMAL_RERUNS = 10
_FORMAL_TEST_PATIENTS = 8
_STATUS_VALUES = {"pending", "running", "complete", "failed", "awaiting_generator_review"}
_SECTION_NAMES = {
    "3A": "generator_validation",
    "3B-1": "a_benchmark",
    "3B-2": "de_benchmark",
    "3C-1": "subbag_consistency_ablation",
    "3C-2": "geometry_ablation",
    "3C-3": "recurrence_ablation",
}
_EXPECTED_PATIENT_ROWS = {
    "3B-1": 7200,
    "3B-2": 5760,
    "3C-1": 2880,
    "3C-2": 2880,
    "3C-3": 2880,
}
_EXPECTED_SUMMARY_ROWS = {
    "3B-1": 90,
    "3B-2": 72,
    "3C-1": 36,
    "3C-2": 36,
    "3C-3": 36,
}
_RELEVANT_GIT_PATHS = (
    "AGENTS.md",
    "docs/task_A",
    "docs/visualization",
    "src/stride",
    "tasks/task_A",
    "tests",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if payload.get("status") is not None and payload["status"] not in _STATUS_VALUES:
        raise ValueError(f"unsupported run status: {payload['status']!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_progress(output_dir: Path, payload: dict[str, object]) -> None:
    record = {"timestamp_utc": _utc_now(), **payload}
    with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _command_output(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _preflight(device: str) -> dict[str, Any]:
    import torch

    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Block3 requires an available CUDA device")
    if resolved.index is not None and resolved.index >= torch.cuda.device_count():
        raise RuntimeError(f"requested CUDA device is unavailable: {resolved}")
    gpu_name = torch.cuda.get_device_name(resolved)
    if "RTX 4090" not in gpu_name:
        raise RuntimeError(f"formal Block3 requires RTX 4090; found {gpu_name!r}")
    probe = float((torch.tensor([2.0], dtype=torch.float64, device=resolved) ** 2).cpu()[0])
    if probe != 4.0:
        raise RuntimeError("CUDA float64 probe failed")
    python_packages = {}
    for name in ("anndata", "numpy", "pandas", "POT", "scipy", "torch"):
        try:
            python_packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            python_packages[name] = "missing"
    rscript = Path(sys.executable).with_name("Rscript")
    r_packages: dict[str, str] = {}
    if rscript.exists():
        expression = (
            "pkgs <- c('tidyverse','dittoSeq','ComplexHeatmap','circlize','rhdf5'); "
            "cat(paste(pkgs, vapply(pkgs, function(x) as.character(packageVersion(x)), "
            "character(1)), sep='=', collapse='\\n'))"
        )
        for line in _command_output([str(rscript), "-e", expression]).splitlines():
            package, version = line.split("=", 1)
            r_packages[package] = version
    return {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "python_packages": python_packages,
        "rscript": str(rscript),
        "r_packages": r_packages,
        "torch_cuda_version": torch.version.cuda,
        "device": str(resolved),
        "gpu_name": gpu_name,
        "cuda_available": True,
        "nvidia_smi": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }


def _git_provenance() -> dict[str, Any]:
    return {
        "commit": _command_output(["git", "rev-parse", "HEAD"]),
        "dirty": bool(_command_output(["git", "status", "--short"])),
    }


def _resume_git_provenance() -> dict[str, Any]:
    relevant_status = _command_output(
        ["git", "status", "--short", "--", *_RELEVANT_GIT_PATHS]
    )
    if relevant_status:
        raise ContractError(
            "Block3-related paths must be committed before formal resume"
        )
    index_records = _command_output(
        ["git", "ls-files", "-s", "--", *_RELEVANT_GIT_PATHS]
    )
    return {
        **_git_provenance(),
        "relevant_paths_clean": True,
        "relevant_path_index_sha256": hashlib.sha256(
            index_records.encode("utf-8")
        ).hexdigest(),
        "relevant_paths": list(_RELEVANT_GIT_PATHS),
    }


def _initialize_manifest(
    *,
    output_dir: Path,
    task_config: Path,
    stage0_h5ad: Path,
    device: str,
    environment: dict[str, Any],
    n_reruns: int,
    n_test: int,
) -> dict[str, Any]:
    manifest = {
        "schema_version": "task_a_block3_run.v1",
        "status": "running",
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "fit_surface": "stride.tl.fit",
        "device": device,
        "execution_scope": (
            "formal_full_data"
            if n_reruns == _FORMAL_RERUNS and n_test == _FORMAL_TEST_PATIENTS
            else "subset_engineering_test"
        ),
        "n_reruns": n_reruns,
        "n_train_patients": 24,
        "n_test_patients": n_test,
        "inputs": {
            "task_config": str(task_config),
            "task_config_sha256": _sha256(task_config),
            "stage0_h5ad": str(stage0_h5ad),
            "stage0_h5ad_sha256": _sha256(stage0_h5ad),
        },
        "environment": environment,
        "git": _git_provenance(),
        "sections": {name: "pending" for name in _SECTION_NAMES.values()},
        "generator_review": {"approved": False},
    }
    _atomic_json(output_dir / "block3_run_manifest.json", manifest)
    return manifest


def _update_manifest(output_dir: Path, manifest: dict[str, Any], **updates: Any) -> None:
    manifest.update(updates)
    manifest["updated_at_utc"] = _utc_now()
    _atomic_json(output_dir / "block3_run_manifest.json", manifest)


def _write_generator_bundle(
    path: Path,
    *,
    reruns: tuple[execution.Block3GeneratorRerun, ...],
    cohort: execution.Block3CohortInputs,
) -> None:
    if path.exists():
        raise ContractError(f"generator bundle must not be overwritten: {path}")
    n_reruns = len(reruns)
    n_test = len(reruns[0].test_patient_ids)
    patient_ids = np.empty((n_reruns, n_test), dtype="U128")
    train_ids = np.empty((n_reruns, 24), dtype="U128")
    x = np.empty((n_reruns, n_test, len(cohort.state_ids)), dtype=float)
    y = np.empty_like(x)
    A = np.empty((n_reruns, n_test, len(cohort.state_ids), len(cohort.state_ids)), dtype=float)
    d = np.empty_like(x)
    e = np.empty_like(x)
    row_imputed = np.empty_like(x, dtype=bool)
    open_mass = np.empty((n_reruns, n_test), dtype=float)
    endpoint_closure = np.empty((n_reruns, n_test), dtype=float)
    sampled_templates = np.empty((n_reruns, n_test), dtype="U128")
    diagnostics_json = np.empty((n_reruns, n_test), dtype="U4096")
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    source_offsets = [0]
    target_offsets = [0]
    for rerun_index, rerun in enumerate(reruns):
        train_ids[rerun_index] = np.asarray(rerun.train_patient_ids, dtype=str)
        for patient_index, patient_id in enumerate(rerun.test_patient_ids):
            truth = rerun.generator_truths[patient_id]
            if truth.source_fovs is None or truth.target_fovs is None:
                raise ContractError("generator bundle requires source and target FOV arrays")
            patient_ids[rerun_index, patient_index] = patient_id
            x[rerun_index, patient_index] = truth.x
            y[rerun_index, patient_index] = truth.y
            A[rerun_index, patient_index] = truth.A
            d[rerun_index, patient_index] = truth.d
            e[rerun_index, patient_index] = truth.e
            row_imputed[rerun_index, patient_index] = np.asarray(
                truth.row_imputed_mask, dtype=bool
            )
            open_mass[rerun_index, patient_index] = truth.open_mass
            endpoint_closure[rerun_index, patient_index] = float(truth.endpoint_closure_l1)
            sampled_templates[rerun_index, patient_index] = truth.sampled_template_patient_id or ""
            diagnostics_json[rerun_index, patient_index] = json.dumps(
                truth.generator_diagnostics or {}, sort_keys=True
            )
            source_parts.append(np.asarray(truth.source_fovs, dtype=float))
            target_parts.append(np.asarray(truth.target_fovs, dtype=float))
            source_offsets.append(source_offsets[-1] + source_parts[-1].shape[0])
            target_offsets.append(target_offsets[-1] + target_parts[-1].shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            rerun_ids=np.asarray([rerun.rerun_id for rerun in reruns], dtype=str),
            split_seeds=np.asarray([rerun.split_seed for rerun in reruns], dtype=np.int64),
            train_patient_ids=train_ids,
            patient_ids=patient_ids,
            x=x,
            y=y,
            A=A,
            d=d,
            e=e,
            row_imputed_mask=row_imputed,
            open_mass=open_mass,
            endpoint_closure_l1=endpoint_closure,
            sampled_template_ids=sampled_templates,
            medoid_template_ids=np.asarray(
                [rerun.template_medoid_patient_id or "" for rerun in reruns], dtype=str
            ),
            generator_parameters_json=np.asarray(
                [json.dumps(rerun.generator_parameters or {}, sort_keys=True) for rerun in reruns],
                dtype=str,
            ),
            diagnostics_json=diagnostics_json,
            source_fov_values=np.concatenate(source_parts, axis=0),
            source_fov_offsets=np.asarray(source_offsets, dtype=np.int64),
            target_fov_values=np.concatenate(target_parts, axis=0),
            target_fov_offsets=np.asarray(target_offsets, dtype=np.int64),
            cost_matrix=np.asarray(cohort.cost_matrix, dtype=float),
            cost_scale=np.asarray([cohort.cost_scale], dtype=float),
            state_ids=np.asarray(cohort.state_ids, dtype=np.int64),
        )
    temporary.replace(path)
    _atomic_text(path.with_suffix(path.suffix + ".sha256"), _sha256(path) + "\n")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _load_generator_bundle(
    path: Path,
    *,
    cohort: execution.Block3CohortInputs,
) -> tuple[execution.Block3GeneratorRerun, ...]:
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() or not checksum_path.exists() or checksum_path.read_text().strip() != _sha256(path):
        raise ContractError("generator bundle checksum validation failed")
    with np.load(path, allow_pickle=False) as payload:
        if not np.array_equal(payload["state_ids"], np.asarray(cohort.state_ids, dtype=np.int64)):
            raise ContractError("generator bundle state axis mismatch")
        if not np.array_equal(payload["cost_matrix"], np.asarray(cohort.cost_matrix, dtype=float)):
            raise ContractError("generator bundle cost matrix mismatch")
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    reruns = []
    source_values = arrays["source_fov_values"]
    target_values = arrays["target_fov_values"]
    source_offsets = arrays["source_fov_offsets"]
    target_offsets = arrays["target_fov_offsets"]
    flat_index = 0
    for rerun_index, rerun_id_value in enumerate(arrays["rerun_ids"]):
        rerun_id = str(rerun_id_value)
        test_ids = tuple(str(item) for item in arrays["patient_ids"][rerun_index].tolist())
        truths = {}
        for patient_index, patient_id in enumerate(test_ids):
            source_slice = slice(source_offsets[flat_index], source_offsets[flat_index + 1])
            target_slice = slice(target_offsets[flat_index], target_offsets[flat_index + 1])
            truths[patient_id] = execution.Block3PatientTruth(
                rerun_id=rerun_id,
                patient_id=patient_id,
                x=arrays["x"][rerun_index, patient_index],
                y=arrays["y"][rerun_index, patient_index],
                A=arrays["A"][rerun_index, patient_index],
                d=arrays["d"][rerun_index, patient_index],
                e=arrays["e"][rerun_index, patient_index],
                open_mass=float(arrays["open_mass"][rerun_index, patient_index]),
                y_endpoint=arrays["y"][rerun_index, patient_index],
                source_fovs=source_values[source_slice],
                target_fovs=target_values[target_slice],
                sampled_template_patient_id=(
                    str(arrays["sampled_template_ids"][rerun_index, patient_index]) or None
                ),
                medoid_template_patient_id=(str(arrays["medoid_template_ids"][rerun_index]) or None),
                row_imputed_mask=arrays["row_imputed_mask"][rerun_index, patient_index],
                endpoint_closure_l1=float(
                    arrays["endpoint_closure_l1"][rerun_index, patient_index]
                ),
                generator_diagnostics=json.loads(
                    str(arrays["diagnostics_json"][rerun_index, patient_index])
                ),
            )
            flat_index += 1
        reruns.append(
            execution.Block3GeneratorRerun(
                rerun_id=rerun_id,
                split_seed=int(arrays["split_seeds"][rerun_index]),
                train_patient_ids=tuple(
                    str(item) for item in arrays["train_patient_ids"][rerun_index].tolist()
                ),
                test_patient_ids=test_ids,
                hidden_relation_condition_id=execution._SHARED_GENERATOR_TRUTH_CONDITION_ID,
                open_mass_scale=1.0,
                generator_truths=truths,
                baseline_truths={
                    execution._A_BENCHMARK_CONDITION_ID: dict(truths),
                    execution._DE_BENCHMARK_CONDITION_ID: dict(truths),
                },
                template_medoid_patient_id=(
                    str(arrays["medoid_template_ids"][rerun_index]) or None
                ),
                generator_parameters=json.loads(
                    str(arrays["generator_parameters_json"][rerun_index])
                ),
            )
        )
    return tuple(reruns)


def _write_section(
    *,
    output_dir: Path,
    plan: execution.Block3Stage0ExecutionPlan,
    subexperiment_id: str,
    raw_rows: Any,
    review_rows: Any,
) -> None:
    result = execution._write_internal_block3_outputs(
        output_dir=output_dir,
        plan=plan,
        subexperiment_id=subexperiment_id,
        raw_rows=raw_rows,
        review_rows=review_rows,
    )
    semantic_name = _SECTION_NAMES[subexperiment_id]
    preserved_shared = _preserve_section_shared_tables(
        output_dir=output_dir,
        semantic_name=semantic_name,
        artifact_paths=result.raw_artifact_paths,
    )
    section_manifest = {
        "schema_version": "task_a_block3_section.v1",
        "status": "complete",
        "experiment_name": semantic_name,
        "subexperiment_id": subexperiment_id,
        "raw_artifacts": _artifact_records(
            output_dir,
            (
                path
                for role, path in result.raw_artifact_paths.items()
                if role not in {"patient_truth_store", "method_native_output_store"}
            ),
        ),
        "preserved_shared_artifacts": _artifact_records(output_dir, preserved_shared),
        "review_artifacts": _artifact_records(output_dir, result.review_artifact_paths.values()),
    }
    _atomic_json(output_dir / "manifests" / f"{semantic_name}.json", section_manifest)


def _preserve_section_shared_tables(
    *,
    output_dir: Path,
    semantic_name: str,
    artifact_paths: dict[str, Path],
) -> tuple[Path, ...]:
    preserved = []
    for role in ("patient_truth_store", "method_native_output_store"):
        source = artifact_paths.get(role)
        if source is None or not source.exists():
            continue
        destination = output_dir / "raw" / semantic_name / f"{role}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".csv.tmp")
        temporary.write_bytes(source.read_bytes())
        temporary.replace(destination)
        preserved.append(destination)
    return tuple(preserved)


def _combine_shared_tables(output_dir: Path) -> None:
    for role in ("patient_truth_store", "method_native_output_store"):
        frames = []
        for semantic_name in _SECTION_NAMES.values():
            path = output_dir / "raw" / semantic_name / f"{role}.csv"
            if path.exists():
                frames.append(pd.read_csv(path))
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        destination = output_dir / "raw" / f"{role}.csv"
        temporary = destination.with_suffix(".csv.tmp")
        combined.to_csv(temporary, index=False)
        temporary.replace(destination)


def _artifact_records(output_dir: Path, paths: Iterable[Path]) -> list[dict[str, object]]:
    return [
        {
            "relative_path": str(path.relative_to(output_dir)),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]


def _write_artifact_index(output_dir: Path) -> None:
    excluded = {
        "block3_artifact_index.csv",
        "block3_run_manifest.json",
        "progress.jsonl",
    }
    records = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        relative = str(path.relative_to(output_dir))
        if relative in excluded or relative.endswith(".tmp"):
            continue
        records.append(
            {
                "relative_path": relative,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    destination = output_dir / "block3_artifact_index.csv"
    temporary = destination.with_suffix(".csv.tmp")
    pd.DataFrame.from_records(records).to_csv(temporary, index=False)
    temporary.replace(destination)


def _validate_formal_rows(output_dir: Path) -> None:
    for subexperiment_id, expected in _EXPECTED_PATIENT_ROWS.items():
        name = _SECTION_NAMES[subexperiment_id]
        patient_path = output_dir / "raw" / name / "patient_metrics.csv"
        summary_path = output_dir / "raw" / name / "condition_summary.csv"
        patient_frame = pd.read_csv(patient_path)
        summary_frame = pd.read_csv(summary_path)
        if len(patient_frame) != expected:
            raise ContractError(f"{subexperiment_id} patient row count mismatch")
        if len(summary_frame) != _EXPECTED_SUMMARY_ROWS[subexperiment_id]:
            raise ContractError(f"{subexperiment_id} summary row count mismatch")
    native = pd.read_csv(output_dir / "raw" / "method_native_output_store.csv")
    if "fit_status" in native and not native["fit_status"].eq("ok").all():
        raise ContractError("formal Block3 contains non-ok method status")


def _write_generator_review_decision(
    output_dir: Path,
    *,
    review_note: str,
) -> Path:
    note = str(review_note).strip()
    if not note:
        raise ContractError("generator review note must be non-empty")
    diagnostics_path = output_dir / "raw" / "generator_diagnostics.csv"
    diagnostics = pd.read_csv(diagnostics_path)
    ordering_counts = {
        str(key): int(value)
        for key, value in diagnostics["burden_ordering_status"]
        .value_counts(dropna=False)
        .items()
    }
    truth_finite = diagnostics["truth_finite"].astype(str).str.lower().eq("true")
    payload = {
        "schema_version": "task_a_block3_generator_review.v1",
        "status": "complete",
        "decision": "accepted_for_downstream_execution",
        "decided_at_utc": _utc_now(),
        "approval_source": "--approve-generator-review",
        "review_note": note,
        "n_truth_records": int(len(diagnostics)),
        "n_truth_finite": int(truth_finite.sum()),
        "burden_ordering_counts": ordering_counts,
        "generator_retuned": False,
        "interpretation_boundary": (
            "Manual continuation decision only; not a pass/fail scientific result."
        ),
    }
    destination = output_dir / "review" / "generator_review_decision.json"
    _atomic_json(destination, payload)
    return destination


def _write_execution_audit(
    output_dir: Path,
    *,
    figure_dir: Path | None = None,
) -> Path:
    manifest = json.loads(
        (output_dir / "block3_run_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("status") != "complete":
        raise ContractError("execution audit requires a complete Block3 manifest")
    cache_counts = {}
    cache_checksums_valid = True
    for cache_name in (
        "reference",
        "consistency_ablation",
        "geometry_ablation",
        "recurrence_ablation",
    ):
        cache_dir = output_dir / "cache" / cache_name
        cache_files = sorted(cache_dir.glob("rerun_*.npz"))
        cache_counts[cache_name] = len(cache_files)
        for cache_path in cache_files:
            checksum_path = cache_path.with_suffix(cache_path.suffix + ".sha256")
            cache_checksums_valid = cache_checksums_valid and (
                checksum_path.exists()
                and checksum_path.read_text(encoding="ascii").strip()
                == _sha256(cache_path)
            )
    row_counts = {}
    method_sets = {}
    native_frames = {}
    for subexperiment_id in ("3B-1", "3B-2", "3C-1", "3C-2", "3C-3"):
        name = _SECTION_NAMES[subexperiment_id]
        patient_frame = pd.read_csv(output_dir / "raw" / name / "patient_metrics.csv")
        summary_frame = pd.read_csv(output_dir / "raw" / name / "condition_summary.csv")
        native_frames[name] = pd.read_csv(
            output_dir / "raw" / name / "method_native_output_store.csv"
        )
        row_counts[name] = {
            "patient_metrics": int(len(patient_frame)),
            "condition_summary": int(len(summary_frame)),
        }
        method_sets[name] = sorted(str(item) for item in patient_frame["method_name"].unique())
    native = pd.read_csv(output_dir / "raw" / "method_native_output_store.csv")
    reference_columns = ("rerun_id", "patient_id", "A_json", "d_json", "e_json")
    reference_frames = []
    for frame in native_frames.values():
        reference_frames.append(
            frame.loc[frame["method_name"] == "stride_reference", reference_columns]
            .sort_values(["rerun_id", "patient_id"])
            .reset_index(drop=True)
        )
    reference_arrays_identical = all(
        reference_frames[0].equals(frame) for frame in reference_frames[1:]
    )
    recurrence_native = native_frames["recurrence_ablation"]
    recurrence_metadata = [
        json.loads(value)
        for value in recurrence_native.loc[
            recurrence_native["method_name"] == "recurrence_ablation",
            "metadata_json",
        ]
    ]
    recurrence_provenance_valid = bool(recurrence_metadata) and all(
        item.get("zeroed_objective_term") == "recurrence"
        and item.get("fixed_denominator_policy") is True
        for item in recurrence_metadata
    )
    pdf_records = []
    if figure_dir is not None:
        pdf_paths = sorted(figure_dir.glob("*.pdf"))
        if len(pdf_paths) != 20:
            raise ContractError(f"execution audit expected 20 PDFs; found {len(pdf_paths)}")
        for pdf_path in pdf_paths:
            result = subprocess.run(
                ["pdfinfo", str(pdf_path)],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0 or pdf_path.stat().st_size <= 0:
                raise ContractError(f"invalid final PDF: {pdf_path}")
            pdf_records.append(
                {
                    "name": pdf_path.name,
                    "sha256": _sha256(pdf_path),
                    "size_bytes": pdf_path.stat().st_size,
                }
            )
    payload = {
        "schema_version": "task_a_block3_execution_audit.v1",
        "status": "complete",
        "audited_at_utc": _utc_now(),
        "run_status": manifest["status"],
        "execution_scope": manifest["execution_scope"],
        "device": manifest["device"],
        "gpu_name": manifest["environment"]["gpu_name"],
        "sections": manifest["sections"],
        "generator_review": manifest["generator_review"],
        "cache_counts": cache_counts,
        "cache_checksums_valid": bool(cache_checksums_valid),
        "row_counts": row_counts,
        "method_sets": method_sets,
        "all_method_status_ok": bool(native["fit_status"].eq("ok").all()),
        "reference_arrays_identical_across_sections": reference_arrays_identical,
        "recurrence_ablation_provenance": {
            "valid": recurrence_provenance_valid,
            "zeroed_objective_term": "recurrence",
            "fixed_denominator_policy": True,
            "cohort_semantics": "unregularized_cohort_diagnostic",
        },
        "pdfs": pdf_records,
    }
    destination = output_dir / "block3_execution_audit.json"
    _atomic_json(destination, payload)
    return destination


def _run_first_phase(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("new Block3 output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = _preflight(args.device)
    task_config = Path(args.task_config).expanduser().resolve()
    stage0_h5ad = Path(args.stage0_h5ad).expanduser().resolve()
    manifest = _initialize_manifest(
        output_dir=output_dir,
        task_config=task_config,
        stage0_h5ad=stage0_h5ad,
        device=args.device,
        environment=environment,
        n_reruns=args.max_reruns or _FORMAL_RERUNS,
        n_test=args.n_test or _FORMAL_TEST_PATIENTS,
    )
    try:
        n_reruns = int(manifest["n_reruns"])
        n_test = int(manifest["n_test_patients"])
        cohort = execution._build_block3_cohort_inputs_from_stage0(
            stage0_h5ad=stage0_h5ad,
            config_path=task_config,
            output_dir=output_dir,
            n_test_patients=n_test,
        )
        reruns = execution._build_generator_reruns(
            cohort_inputs=cohort,
            n_reruns=n_reruns,
            n_test_patients=n_test,
        )
        _write_generator_bundle(
            output_dir / "cache" / "generator_bundle.npz",
            reruns=reruns,
            cohort=cohort,
        )
        plan = execution.build_stage0_execution_plan(
            output_dir=output_dir,
            n_generator_reruns=n_reruns,
            n_test_patients=n_test,
        )
        raw_rows, review_rows = execution._build_rows_for_subexperiment(
            subexperiment_id=Block3SubexperimentId.GENERATOR_VALIDATION.value,
            reruns=reruns,
            cohort_inputs=cohort,
            runtime=execution.Block3RuntimeControls(device=args.device),
        )
        _write_section(
            output_dir=output_dir,
            plan=plan,
            subexperiment_id="3A",
            raw_rows=raw_rows,
            review_rows=review_rows,
        )
        manifest["sections"]["generator_validation"] = "complete"
        _append_progress(output_dir, {"unit": "generator_validation", "status": "complete"})
        _write_artifact_index(output_dir)
        _update_manifest(output_dir, manifest, status="awaiting_generator_review")
        return 0
    except Exception as exc:
        _append_progress(output_dir, {"unit": "first_phase", "status": "failed", "error": str(exc)})
        _update_manifest(output_dir, manifest, status="failed", failure=str(exc))
        raise


def _run_resume(args: argparse.Namespace) -> int:
    if not args.approve_generator_review:
        raise ContractError("resume requires --approve-generator-review")
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_dir / "block3_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior_status = manifest.get("status")
    if prior_status not in {"awaiting_generator_review", "failed"}:
        raise ContractError(
            "resume requires status='awaiting_generator_review' or a failed approved resume"
        )
    task_config = Path(manifest["inputs"]["task_config"])
    stage0_h5ad = Path(manifest["inputs"]["stage0_h5ad"])
    if _sha256(task_config) != manifest["inputs"]["task_config_sha256"]:
        raise ContractError("Task A config checksum changed since generator phase")
    if _sha256(stage0_h5ad) != manifest["inputs"]["stage0_h5ad_sha256"]:
        raise ContractError("Stage0 checksum changed since generator phase")
    resume_environment = _preflight(args.device)
    resume_provenance = _resume_git_provenance()
    if prior_status == "awaiting_generator_review":
        review_decision_path = _write_generator_review_decision(
            output_dir,
            review_note=args.generator_review_note,
        )
        manifest["generator_review"] = {
            "approved": True,
            "approved_at_utc": _utc_now(),
            "approval_source": "--approve-generator-review",
            "review_note": args.generator_review_note,
            "decision_path": str(review_decision_path.relative_to(output_dir)),
        }
    elif not manifest.get("generator_review", {}).get("approved"):
        raise ContractError("failed generator phase cannot be resumed; clear output and restart")
    manifest.pop("failure", None)
    manifest["resume_provenance"] = {
        "environment": resume_environment,
        "git": resume_provenance,
    }
    _update_manifest(output_dir, manifest, status="running")
    try:
        n_reruns = int(manifest["n_reruns"])
        n_test = int(manifest["n_test_patients"])
        cohort = execution._build_block3_cohort_inputs_from_stage0(
            stage0_h5ad=stage0_h5ad,
            config_path=task_config,
            output_dir=output_dir,
            n_test_patients=n_test,
        )
        reruns = _load_generator_bundle(
            output_dir / "cache" / "generator_bundle.npz",
            cohort=cohort,
        )
        plan = execution.build_stage0_execution_plan(
            output_dir=output_dir,
            n_generator_reruns=n_reruns,
            n_test_patients=n_test,
        )
        runtime = execution.Block3RuntimeControls(
            device=args.device,
            cache_dir=output_dir / "cache",
            progress_callback=lambda record: _append_progress(output_dir, record),
        )
        for subexperiment_id in ("3B-1", "3B-2", "3C-1", "3C-2", "3C-3"):
            section_name = _SECTION_NAMES[subexperiment_id]
            if manifest["sections"].get(section_name) == "complete":
                _append_progress(
                    output_dir,
                    {"unit": section_name, "status": "complete", "action": "skip"},
                )
                continue
            raw_rows, review_rows = execution._build_rows_for_subexperiment(
                subexperiment_id=subexperiment_id,
                reruns=reruns,
                cohort_inputs=cohort,
                runtime=runtime,
            )
            _write_section(
                output_dir=output_dir,
                plan=plan,
                subexperiment_id=subexperiment_id,
                raw_rows=raw_rows,
                review_rows=review_rows,
            )
            manifest["sections"][section_name] = "complete"
            _append_progress(output_dir, {"unit": section_name, "status": "complete"})
            _update_manifest(output_dir, manifest, status="running")
        _combine_shared_tables(output_dir)
        if manifest["execution_scope"] == "formal_full_data":
            _validate_formal_rows(output_dir)
        _update_manifest(output_dir, manifest, status="complete")
        _write_execution_audit(output_dir)
        _write_artifact_index(output_dir)
        return 0
    except Exception as exc:
        _append_progress(output_dir, {"unit": "resume", "status": "failed", "error": str(exc)})
        _update_manifest(output_dir, manifest, status="failed", failure=str(exc))
        raise


def _finalize_audit(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    figure_dir = (
        Path(args.figure_dir).expanduser().resolve()
        if args.figure_dir
        else output_dir.parent / "figures"
    )
    _write_execution_audit(output_dir, figure_dir=figure_dir)
    _write_artifact_index(output_dir)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m tasks.task_A.block3.run")
    parser.add_argument("--task-config")
    parser.add_argument("--stage0-h5ad")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--approve-generator-review", action="store_true")
    parser.add_argument("--generator-review-note")
    parser.add_argument("--finalize-audit", action="store_true")
    parser.add_argument("--figure-dir")
    parser.add_argument("--max-reruns", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    args = parser.parse_args(argv)
    if args.resume and not args.generator_review_note:
        parser.error("--generator-review-note is required for resume")
    if not args.resume and not args.finalize_audit and (
        not args.task_config or not args.stage0_h5ad
    ):
        parser.error("--task-config and --stage0-h5ad are required for a new run")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.finalize_audit:
        return _finalize_audit(args)
    return _run_resume(args) if args.resume else _run_first_phase(args)


if __name__ == "__main__":
    raise SystemExit(main())
