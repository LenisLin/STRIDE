"""Task A Block 0 `TC-IM` execution-cache and analysis entrypoint.

Block 0 execution runs real/null full STRIDE fits and writes reusable
`A,d,e,mu` records. Block 0 analysis derives calibration tables from an
existing cache. Block 0 consumes Stage 0 h5ad, Task A config, run controls,
and optional selectors; it does not emit biological interpretation or
downstream execution decisions.
See `tasks/task_A/README.md`, `tasks/task_A/contracts/artifact_contracts.md`,
and `tasks/task_A/contracts/design_freeze.py`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections.abc import Mapping
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path

import numpy as np

from stride.errors import ContractError

from ..config import load_task_a_config_bundle
from ..real_data.demo_subset import resolve_demo_subset
from ..workflows.stride_adapter import (
    load_task_a_dataset_handle,
    resolve_task_a_state_basis,
)
from .analyze import run_block0_analyze
from .functions.cache import sha256_file, write_block0_fit_cache
from .functions.stride_fit import (
    extract_block0_fit_records,
    fit_block0_family,
)
from .functions.observations import (
    Block0ObservationBundle,
    build_null_tc_im_observations,
    build_real_tc_im_observations,
    resolve_block0_run_config,
)
from .functions.parallel import (
    Block0NullFitJob,
    configure_worker_thread_environment,
    configure_worker_threads,
    fit_block0_null_permutation,
)
from .functions.permutation import build_domain_label_permutation_assignments
from .functions.progress import fit_warning_summary
from .functions.schemas import (
    BLOCK_NAME,
    CALIBRATION_MANIFEST_FILENAME,
    EXECUTION_MANIFEST_FILENAME,
    EXECUTION_NAME,
    EXECUTION_PROGRESS_FILENAME,
    FIT_CACHE_FILENAME,
    FIT_CACHE_INDEX_FILENAME,
    FIT_CACHE_SCHEMA_VERSION,
    FULL_CALIBRATION_N_PERMUTATIONS,
    METRIC_SUMMARY_FILENAME,
    MIN_N_PERMUTATIONS,
    NULL_FAMILY,
    PATIENT_CALIBRATION_FILENAME,
    REAL_FAMILY,
    FIT_LABEL_NULL,
    FIT_LABEL_REAL,
    Block0FitRecord,
)
from .functions.writers import write_block0_execution_outputs


DEFAULT_N_PERMUTATIONS = FULL_CALIBRATION_N_PERMUTATIONS
BLOCK0_PROGRESS_FILENAME = EXECUTION_PROGRESS_FILENAME
BLOCK0_RESUME_CHECKPOINT_FILENAME = ".block0_calibration_resume_records.json"
_BLOCK0_EXECUTION_AND_PROGRESS_OUTPUTS = (
    EXECUTION_MANIFEST_FILENAME,
    FIT_CACHE_FILENAME,
    FIT_CACHE_INDEX_FILENAME,
    CALIBRATION_MANIFEST_FILENAME,
    PATIENT_CALIBRATION_FILENAME,
    METRIC_SUMMARY_FILENAME,
    BLOCK0_PROGRESS_FILENAME,
    BLOCK0_RESUME_CHECKPOINT_FILENAME,
)
_BLOCK0_EXECUTION_OUTPUTS = (
    EXECUTION_MANIFEST_FILENAME,
    FIT_CACHE_FILENAME,
    FIT_CACHE_INDEX_FILENAME,
    CALIBRATION_MANIFEST_FILENAME,
    PATIENT_CALIBRATION_FILENAME,
    METRIC_SUMMARY_FILENAME,
)
def _permutation_count(value: str) -> int:
    count = int(value)
    if count < MIN_N_PERMUTATIONS:
        raise argparse.ArgumentTypeError(
            f"n-permutations must be >= {MIN_N_PERMUTATIONS}"
        )
    return count


def _parallel_permutation_count(value: int | str) -> int:
    count = int(value)
    if count < 1:
        raise ContractError("parallel_permutations must be >= 1")
    return count


def _worker_cpu_thread_count(value: int | str) -> int:
    count = int(value)
    if count < 1 or count > 6:
        raise ContractError("worker_cpu_threads must be in [1, 6]")
    return count


def _guard_block0_cpu_budget(
    *,
    parallel_permutations: int,
    worker_cpu_threads: int,
    allow_cpu_oversubscription: bool,
) -> None:
    cpu_count = os.cpu_count()
    requested = int(parallel_permutations) * int(worker_cpu_threads)
    if cpu_count is not None and requested > int(cpu_count) and not allow_cpu_oversubscription:
        raise ContractError(
            "Block 0 parallel CPU budget exceeds available CPUs; "
            f"parallel_permutations={parallel_permutations}, "
            f"worker_cpu_threads={worker_cpu_threads}, cpu_count={cpu_count}. "
            "Reduce worker count/thread count or pass --allow-cpu-oversubscription."
        )


def run_block0_execute(
    *,
    config_path: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    master_seed: int | None = None,
    patient_ids: tuple[str, ...] | None = None,
    demo_subset_name: str | None = None,
    resume: bool = False,
    parallel_permutations: int = 1,
    worker_cpu_threads: int = 4,
    allow_cpu_oversubscription: bool = False,
    device: object | None = None,
) -> dict[str, Path]:
    """Run Block 0 real/null fits and write a reusable fit cache."""
    if master_seed is None:
        raise ContractError("Task A Block 0 requires an explicit master_seed")
    resolved_parallel_permutations = _parallel_permutation_count(parallel_permutations)
    resolved_worker_cpu_threads = _worker_cpu_thread_count(worker_cpu_threads)
    _guard_block0_cpu_budget(
        parallel_permutations=resolved_parallel_permutations,
        worker_cpu_threads=resolved_worker_cpu_threads,
        allow_cpu_oversubscription=bool(allow_cpu_oversubscription),
    )
    run_config = resolve_block0_run_config(
        config_path=config_path,
        data_path=data_path,
        output_dir=output_dir,
        n_permutations=n_permutations,
        master_seed=master_seed,
        patient_ids=patient_ids,
        demo_subset_name=demo_subset_name,
    )
    output_root = run_config.output_dir
    _guard_no_existing_execution_outputs(output_root, resume=bool(resume))
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / BLOCK0_PROGRESS_FILENAME
    if bool(resume) and progress_path.exists():
        pass
    else:
        _initialize_progress_file(progress_path)

    stage_started_at = _utc_timestamp()
    stage_started_monotonic = time.monotonic()
    config_bundle = load_task_a_config_bundle(run_config.config_path)
    handle = load_task_a_dataset_handle(run_config.data_path, backed=True)
    selected_patient_ids = _selected_patient_ids(run_config)
    _append_stage_progress_event(
        progress_path,
        stage="input_config_data_loading",
        started_at=stage_started_at,
        duration_seconds=_elapsed_seconds(stage_started_monotonic),
        status="ok",
    )

    stage_started_at = _utc_timestamp()
    stage_started_monotonic = time.monotonic()
    real_bundle = build_real_tc_im_observations(
        handle,
        config_bundle,
        patient_ids=selected_patient_ids,
    )
    _append_stage_progress_event(
        progress_path,
        stage="real_observation_construction",
        started_at=stage_started_at,
        duration_seconds=_elapsed_seconds(stage_started_monotonic),
        status="ok",
    )
    real_observation_signature = _observation_bundle_signature(real_bundle)
    resume_state = _load_resume_checkpoint(
        output_root=output_root,
        run_config=run_config,
        config_fingerprint=str(config_bundle.config_fingerprint),
        real_observation_signature=real_observation_signature,
    ) if bool(resume) else _empty_resume_state()

    stage_started_at = _utc_timestamp()
    stage_started_monotonic = time.monotonic()
    state_basis = resolve_task_a_state_basis(handle)
    _append_stage_progress_event(
        progress_path,
        stage="state_basis_resolution",
        started_at=stage_started_at,
        duration_seconds=_elapsed_seconds(stage_started_monotonic),
        status="ok",
    )
    _append_stage_progress_event(
        progress_path,
        stage="parallel_resource_config",
        started_at=_utc_timestamp(),
        duration_seconds=0.0,
        status="ok",
        extra={
            "parallel_permutations": resolved_parallel_permutations,
            "worker_cpu_threads": resolved_worker_cpu_threads,
            "allow_cpu_oversubscription": bool(allow_cpu_oversubscription),
            "cpu_count": os.cpu_count(),
            "requested_device": None if device is None else str(device),
        },
    )

    real_records = tuple(resume_state["real_records"])
    if real_records:
        _append_progress_event(
            progress_path,
            event="fit_skipped",
            fit_label=FIT_LABEL_REAL,
            permutation_index=None,
            started_at=_utc_timestamp(),
            finished_at=_utc_timestamp(),
            status="skipped",
            duration_seconds=0.0,
        )
    else:
        real_started_at = _utc_timestamp()
        real_started_monotonic = time.monotonic()
        _append_progress_event(
            progress_path,
            event="fit_started",
            fit_label=FIT_LABEL_REAL,
            permutation_index=None,
            started_at=real_started_at,
            finished_at=None,
            status="running",
            duration_seconds=None,
        )
        real_result = None
        try:
            real_result = fit_block0_family(
                real_bundle,
                config_bundle=config_bundle,
                state_basis=state_basis,
                fit_label=FIT_LABEL_REAL,
                device=device,
            )
            real_records = extract_block0_fit_records(real_result, fit_label=FIT_LABEL_REAL)
        except Exception as exc:
            duration_seconds = _elapsed_seconds(real_started_monotonic)
            _append_progress_event(
                progress_path,
                event="fit_failed",
                fit_label=FIT_LABEL_REAL,
                permutation_index=None,
                started_at=real_started_at,
                finished_at=_utc_timestamp(),
                status="failed",
                duration_seconds=duration_seconds,
                error=exc,
            )
            _append_stage_progress_event(
                progress_path,
                stage="real_fit_extract",
                started_at=real_started_at,
                duration_seconds=duration_seconds,
                status="failed",
                error=exc,
            )
            raise ContractError(
                f"Block 0 real fit/extraction failed for fit_label=real: {exc}"
            ) from exc
        else:
            resume_state["real_records"] = tuple(real_records)
            _write_resume_checkpoint(
                output_root=output_root,
                run_config=run_config,
                config_fingerprint=str(config_bundle.config_fingerprint),
                real_observation_signature=real_observation_signature,
                resume_state=resume_state,
            )
            duration_seconds = _elapsed_seconds(real_started_monotonic)
            _append_progress_event(
                progress_path,
                event="fit_finished",
                fit_label=FIT_LABEL_REAL,
                permutation_index=None,
                started_at=real_started_at,
                finished_at=_utc_timestamp(),
                status="ok",
                duration_seconds=duration_seconds,
                warning_summary=fit_warning_summary(real_result),
            )
            _append_stage_progress_event(
                progress_path,
                stage="real_fit_extract",
                started_at=real_started_at,
                duration_seconds=duration_seconds,
                status="ok",
            )
        finally:
            del real_result

    if resolved_parallel_permutations == 1:
        null_records_list = _run_serial_null_permutation_fits(
            real_bundle=real_bundle,
            run_config=run_config,
            config_bundle=config_bundle,
            state_basis=state_basis,
            output_root=output_root,
            progress_path=progress_path,
            config_fingerprint=str(config_bundle.config_fingerprint),
            real_observation_signature=real_observation_signature,
            resume_state=resume_state,
            device=device,
        )
    else:
        null_records_list = _run_parallel_null_permutation_fits(
            real_bundle=real_bundle,
            run_config=run_config,
            config_bundle=config_bundle,
            state_basis=state_basis,
            output_root=output_root,
            progress_path=progress_path,
            config_fingerprint=str(config_bundle.config_fingerprint),
            real_observation_signature=real_observation_signature,
            resume_state=resume_state,
            parallel_permutations=resolved_parallel_permutations,
            worker_cpu_threads=resolved_worker_cpu_threads,
            device=device,
        )

    null_records = tuple(null_records_list)
    del null_records_list
    stage_started_at = _utc_timestamp()
    stage_started_monotonic = time.monotonic()
    cache_info = write_block0_fit_cache(
        (*real_records, *null_records),
        output_dir=output_root,
    )
    _append_stage_progress_event(
        progress_path,
        stage="fit_cache_write",
        started_at=stage_started_at,
        duration_seconds=_elapsed_seconds(stage_started_monotonic),
        status="ok",
    )
    manifest_payload = _build_execution_manifest_payload(
        run_config=run_config,
        config_path=config_bundle.config_path,
        output_root=output_root,
        cache_info=cache_info,
        progress_path=progress_path,
    )
    stage_started_at = _utc_timestamp()
    stage_started_monotonic = time.monotonic()
    output_paths = write_block0_execution_outputs(
        output_dir=output_root,
        manifest_payload=manifest_payload,
    )
    _append_stage_progress_event(
        progress_path,
        stage="execution_manifest_write",
        started_at=stage_started_at,
        duration_seconds=_elapsed_seconds(stage_started_monotonic),
        status="ok",
    )
    _clear_resume_checkpoint(output_root)
    return output_paths


def _selected_patient_ids(run_config) -> tuple[str, ...] | None:
    if run_config.demo_subset_name is None:
        return run_config.patient_ids
    try:
        return resolve_demo_subset(run_config.demo_subset_name).patient_ids
    except KeyError as exc:
        raise ContractError(str(exc)) from exc


def _observation_bundle_signature(bundle: Block0ObservationBundle) -> dict[str, object]:
    digest = hashlib.sha256()
    patient_ids = tuple(str(patient_id) for patient_id in bundle.patient_ids)
    digest.update(str(bundle.label).encode("utf-8"))
    digest.update(json.dumps(patient_ids, separators=(",", ":")).encode("utf-8"))
    observations = tuple(
        sorted(
            bundle.observations,
            key=lambda observation: (
                str(observation.patient_id),
                str(observation.fov_id),
                "" if observation.domain_label is None else str(observation.domain_label),
                str(observation.timepoint),
            ),
        )
    )
    for observation in observations:
        composition = np.ascontiguousarray(
            np.asarray(observation.community_composition, dtype=float)
        )
        descriptor = {
            "patient_id": str(observation.patient_id),
            "fov_id": str(observation.fov_id),
            "timepoint": str(observation.timepoint),
            "domain_label": (
                None
                if observation.domain_label is None
                else str(observation.domain_label)
            ),
            "mass": float(observation.mass),
            "mass_mode": str(observation.mass_mode),
            "composition_shape": [int(item) for item in composition.shape],
        }
        digest.update(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(composition.tobytes(order="C"))
    return {
        "label": str(bundle.label),
        "patient_ids": list(patient_ids),
        "n_observations": int(len(observations)),
        "digest": digest.hexdigest(),
    }


def _guard_no_existing_execution_outputs(output_root: Path, *, resume: bool = False) -> None:
    guarded_outputs = _BLOCK0_EXECUTION_OUTPUTS if resume else _BLOCK0_EXECUTION_AND_PROGRESS_OUTPUTS
    existing_outputs = tuple(
        filename for filename in guarded_outputs if (output_root / filename).exists()
    )
    if existing_outputs:
        raise ContractError(
            "Block 0 execution output dir contains existing stale output artifacts; "
            "choose an empty output dir or remove stale outputs before rerun: "
            f"{existing_outputs}"
        )


def _initialize_progress_file(progress_path: Path) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text("", encoding="utf-8")


def _resume_checkpoint_path(output_root: Path) -> Path:
    return output_root / BLOCK0_RESUME_CHECKPOINT_FILENAME


def _empty_resume_state() -> dict[str, object]:
    return {
        "real_records": (),
        "null_records_by_permutation": {},
    }


def _resume_meta_payload(
    *,
    run_config,
    config_fingerprint: str,
    real_observation_signature: Mapping[str, object],
) -> dict[str, object]:
    return {
        "config_path": str(run_config.config_path),
        "stage0_h5ad": str(run_config.data_path),
        "run_scope": run_config.run_scope,
        "n_permutations": int(run_config.n_permutations),
        "master_seed": int(run_config.master_seed),
        "patient_ids": None if run_config.patient_ids is None else list(run_config.patient_ids),
        "demo_subset_name": run_config.demo_subset_name,
        "config_fingerprint": str(config_fingerprint),
        "real_observation_signature": dict(real_observation_signature),
    }


def _write_resume_checkpoint(
    *,
    output_root: Path,
    run_config,
    config_fingerprint: str,
    real_observation_signature: Mapping[str, object],
    resume_state: Mapping[str, object],
) -> None:
    payload = {
        "meta": _resume_meta_payload(
            run_config=run_config,
            config_fingerprint=config_fingerprint,
            real_observation_signature=real_observation_signature,
        ),
        "real_records": _serialize_fit_records(
            tuple(resume_state.get("real_records", ())),
        ),
        "null_records_by_permutation": {
            str(int(permutation_index)): _serialize_fit_records(tuple(records))
            for permutation_index, records in dict(
                resume_state.get("null_records_by_permutation", {})
            ).items()
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _resume_checkpoint_path(output_root)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output_root,
        prefix=".block0_resume_",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp_path.replace(checkpoint_path)


def _load_resume_checkpoint(
    *,
    output_root: Path,
    run_config,
    config_fingerprint: str,
    real_observation_signature: Mapping[str, object],
) -> dict[str, object]:
    checkpoint_path = _resume_checkpoint_path(output_root)
    if not checkpoint_path.exists():
        return _empty_resume_state()
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ContractError("Existing Block 0 resume checkpoint must be a JSON object")
    expected_meta = _resume_meta_payload(
        run_config=run_config,
        config_fingerprint=config_fingerprint,
        real_observation_signature=real_observation_signature,
    )
    observed_meta = payload.get("meta")
    if observed_meta != expected_meta:
        raise ContractError(
            "Existing Block 0 resume checkpoint does not match the current run configuration. "
            "Clear the output directory or rerun with matching inputs."
        )
    real_records = _deserialize_fit_records(payload.get("real_records", ()))
    null_payload = payload.get("null_records_by_permutation", {})
    if not isinstance(null_payload, Mapping):
        raise ContractError("Existing Block 0 resume checkpoint null records must be a JSON object")
    null_records_by_permutation = {
        int(permutation_index): _deserialize_fit_records(records)
        for permutation_index, records in null_payload.items()
    }
    expected_indices = set(range(int(run_config.n_permutations)))
    extra_indices = set(null_records_by_permutation) - expected_indices
    if extra_indices:
        raise ContractError(
            "Existing Block 0 resume checkpoint contains unexpected permutation indices: "
            f"{tuple(sorted(extra_indices))}"
        )
    _validate_resume_records(
        real_records=real_records,
        null_records_by_permutation=null_records_by_permutation,
        real_observation_signature=real_observation_signature,
    )
    return {
        "real_records": real_records,
        "null_records_by_permutation": null_records_by_permutation,
    }


def _validate_resume_records(
    *,
    real_records: tuple[Block0FitRecord, ...],
    null_records_by_permutation: Mapping[int, tuple[Block0FitRecord, ...]],
    real_observation_signature: Mapping[str, object],
) -> None:
    expected_patient_ids = tuple(
        str(patient_id)
        for patient_id in real_observation_signature.get("patient_ids", ())
    )
    if not real_records and null_records_by_permutation:
        raise ContractError(
            "Existing Block 0 resume checkpoint has null records without real records"
        )
    if real_records:
        _validate_resume_record_family(
            records=real_records,
            expected_patient_ids=expected_patient_ids,
            expected_fit_label=FIT_LABEL_REAL,
            expected_permutation_index=None,
        )
    for permutation_index, records in null_records_by_permutation.items():
        _validate_resume_record_family(
            records=records,
            expected_patient_ids=expected_patient_ids,
            expected_fit_label=FIT_LABEL_NULL,
            expected_permutation_index=int(permutation_index),
        )


def _validate_resume_record_family(
    *,
    records: tuple[Block0FitRecord, ...],
    expected_patient_ids: tuple[str, ...],
    expected_fit_label: str,
    expected_permutation_index: int | None,
) -> None:
    observed_patient_ids = tuple(sorted(str(record.patient_id) for record in records))
    if observed_patient_ids != tuple(sorted(expected_patient_ids)):
        raise ContractError(
            "Existing Block 0 resume checkpoint records do not match current real observations"
        )
    for record in records:
        if record.fit_label != expected_fit_label:
            raise ContractError(
                "Existing Block 0 resume checkpoint records have unexpected fit_label"
            )
        if record.permutation_index != expected_permutation_index:
            raise ContractError(
                "Existing Block 0 resume checkpoint records have unexpected permutation_index"
            )


def _clear_resume_checkpoint(output_root: Path) -> None:
    checkpoint_path = _resume_checkpoint_path(output_root)
    if checkpoint_path.exists():
        checkpoint_path.unlink()


def _serialize_fit_records(records: tuple[Block0FitRecord, ...]) -> list[dict[str, object]]:
    return [
        {
            **{
                key: value
                for key, value in asdict(record).items()
                if key not in {"A", "d", "e", "source_burden", "d_weights", "e_weights"}
            },
            "A": np.asarray(record.A, dtype=float).tolist(),
            "d": np.asarray(record.d, dtype=float).tolist(),
            "e": np.asarray(record.e, dtype=float).tolist(),
            "source_burden": np.asarray(record.source_burden, dtype=float).tolist(),
            "d_weights": np.asarray(record.d_weights, dtype=float).tolist(),
            "e_weights": np.asarray(record.e_weights, dtype=float).tolist(),
        }
        for record in records
    ]


def _deserialize_fit_records(records_payload: object) -> tuple[Block0FitRecord, ...]:
    if not isinstance(records_payload, list):
        raise ContractError("Existing Block 0 resume checkpoint records must be a JSON list")
    records: list[Block0FitRecord] = []
    for item in records_payload:
        if not isinstance(item, Mapping):
            raise ContractError("Existing Block 0 resume checkpoint record must be a JSON object")
        records.append(
            Block0FitRecord(
                patient_id=str(item["patient_id"]),
                fit_label=str(item["fit_label"]),
                permutation_index=(
                    None
                    if item.get("permutation_index") is None
                    else int(item["permutation_index"])
                ),
                fit_status=str(item.get("fit_status", "ok")),
                A=np.asarray(item["A"], dtype=float),
                d=np.asarray(item["d"], dtype=float),
                e=np.asarray(item["e"], dtype=float),
                source_burden=np.asarray(item["source_burden"], dtype=float),
                d_weights=np.asarray(item["d_weights"], dtype=float),
                e_weights=np.asarray(item["e_weights"], dtype=float),
            )
        )
    return tuple(records)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _elapsed_seconds(started_monotonic: float) -> float:
    return max(0.0, float(time.monotonic() - started_monotonic))


def _append_stage_progress_event(
    progress_path: Path,
    *,
    stage: str,
    started_at: str,
    duration_seconds: float,
    status: str,
    permutation_index: int | None = None,
    error: BaseException | None = None,
    extra: Mapping[str, object] | None = None,
) -> None:
    finished_at = _utc_timestamp()
    payload: dict[str, object] = {
        "event": "stage_finished",
        "stage": stage,
        "permutation_index": permutation_index,
        "fit_label": None,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
        "status": status,
    }
    if extra is not None:
        payload.update(dict(extra))
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_message"] = str(error)
    with progress_path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()


def _append_progress_event(
    progress_path: Path,
    *,
    event: str,
    fit_label: str,
    permutation_index: int | None,
    started_at: str,
    finished_at: str | None,
    status: str,
    duration_seconds: float | None,
    error: BaseException | None = None,
    warning_summary: Mapping[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "event": event,
        "permutation_index": permutation_index,
        "fit_label": fit_label,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
        "status": status,
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_message"] = str(error)
    if warning_summary is not None:
        payload["warning_count"] = int(warning_summary.get("warning_count", 0))
        payload["warning_categories"] = dict(warning_summary.get("warning_categories", {}))
        payload["warning_examples"] = list(warning_summary.get("warning_examples", ()))
    with progress_path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()


def _run_serial_null_permutation_fits(
    *,
    real_bundle: Block0ObservationBundle,
    run_config,
    config_bundle,
    state_basis,
    output_root: Path,
    progress_path: Path,
    config_fingerprint: str,
    real_observation_signature: Mapping[str, object],
    resume_state: dict[str, object],
    device: object | None,
) -> list[Block0FitRecord]:
    null_records_list: list[Block0FitRecord] = []
    for permutation_index in range(run_config.n_permutations):
        resumed_null_records = resume_state["null_records_by_permutation"].get(
            int(permutation_index),
        )
        if resumed_null_records is not None:
            null_records_list.extend(resumed_null_records)
            _append_progress_event(
                progress_path,
                event="fit_skipped",
                fit_label=FIT_LABEL_NULL,
                permutation_index=permutation_index,
                started_at=_utc_timestamp(),
                finished_at=_utc_timestamp(),
                status="skipped",
                duration_seconds=0.0,
            )
            continue
        permutation_started_at = _utc_timestamp()
        permutation_started_monotonic = time.monotonic()
        _append_progress_event(
            progress_path,
            event="fit_started",
            fit_label=FIT_LABEL_NULL,
            permutation_index=permutation_index,
            started_at=permutation_started_at,
            finished_at=None,
            status="running",
            duration_seconds=None,
        )
        null_bundle = None
        null_result = None
        try:
            stage_started_at = _utc_timestamp()
            stage_started_monotonic = time.monotonic()
            null_bundle = _build_null_observation_bundle(
                real_bundle,
                run_config,
                permutation_index,
            )
            _append_stage_progress_event(
                progress_path,
                stage="null_bundle_build",
                started_at=stage_started_at,
                duration_seconds=_elapsed_seconds(stage_started_monotonic),
                status="ok",
                permutation_index=permutation_index,
            )
            stage_started_at = _utc_timestamp()
            stage_started_monotonic = time.monotonic()
            null_result = fit_block0_family(
                null_bundle,
                config_bundle=config_bundle,
                state_basis=state_basis,
                fit_label=FIT_LABEL_NULL,
                permutation_index=permutation_index,
                device=device,
            )
            extracted_null_records = extract_block0_fit_records(
                null_result,
                fit_label=FIT_LABEL_NULL,
                permutation_index=permutation_index,
            )
            null_records_list.extend(extracted_null_records)
            resume_state["null_records_by_permutation"][int(permutation_index)] = tuple(
                extracted_null_records
            )
            _write_resume_checkpoint(
                output_root=output_root,
                run_config=run_config,
                config_fingerprint=config_fingerprint,
                real_observation_signature=real_observation_signature,
                resume_state=resume_state,
            )
            _append_stage_progress_event(
                progress_path,
                stage="null_fit_extract",
                started_at=stage_started_at,
                duration_seconds=_elapsed_seconds(stage_started_monotonic),
                status="ok",
                permutation_index=permutation_index,
            )
        except Exception as exc:
            duration_seconds = _elapsed_seconds(permutation_started_monotonic)
            _append_progress_event(
                progress_path,
                event="fit_failed",
                fit_label=FIT_LABEL_NULL,
                permutation_index=permutation_index,
                started_at=permutation_started_at,
                finished_at=_utc_timestamp(),
                status="failed",
                duration_seconds=duration_seconds,
                error=exc,
            )
            raise ContractError(
                "Block 0 null fit/extraction failed for "
                f"permutation_index={permutation_index}: {exc}"
            ) from exc
        else:
            duration_seconds = _elapsed_seconds(permutation_started_monotonic)
            _append_progress_event(
                progress_path,
                event="fit_finished",
                fit_label=FIT_LABEL_NULL,
                permutation_index=permutation_index,
                started_at=permutation_started_at,
                finished_at=_utc_timestamp(),
                status="ok",
                duration_seconds=duration_seconds,
                warning_summary=fit_warning_summary(null_result),
            )
        finally:
            del null_bundle
            del null_result
    return null_records_list


def _run_parallel_null_permutation_fits(
    *,
    real_bundle: Block0ObservationBundle,
    run_config,
    config_bundle,
    state_basis,
    output_root: Path,
    progress_path: Path,
    config_fingerprint: str,
    real_observation_signature: Mapping[str, object],
    resume_state: dict[str, object],
    parallel_permutations: int,
    worker_cpu_threads: int,
    device: object | None,
) -> list[Block0FitRecord]:
    records_by_permutation: dict[int, tuple[Block0FitRecord, ...]] = {}
    pending_indices: list[int] = []
    for permutation_index in range(run_config.n_permutations):
        resumed_null_records = resume_state["null_records_by_permutation"].get(
            int(permutation_index),
        )
        if resumed_null_records is not None:
            records_by_permutation[int(permutation_index)] = tuple(resumed_null_records)
            _append_progress_event(
                progress_path,
                event="fit_skipped",
                fit_label=FIT_LABEL_NULL,
                permutation_index=permutation_index,
                started_at=_utc_timestamp(),
                finished_at=_utc_timestamp(),
                status="skipped",
                duration_seconds=0.0,
            )
        else:
            pending_indices.append(int(permutation_index))

    started_by_index: dict[int, tuple[str, float]] = {}
    configure_worker_thread_environment(int(worker_cpu_threads))
    with ProcessPoolExecutor(
        max_workers=int(parallel_permutations),
        mp_context=_block0_parallel_context(),
        initializer=configure_worker_threads,
        initargs=(int(worker_cpu_threads),),
    ) as executor:
        future_to_index = {}
        for permutation_index in pending_indices:
            started_at = _utc_timestamp()
            started_monotonic = time.monotonic()
            started_by_index[permutation_index] = (started_at, started_monotonic)
            _append_progress_event(
                progress_path,
                event="fit_started",
                fit_label=FIT_LABEL_NULL,
                permutation_index=permutation_index,
                started_at=started_at,
                finished_at=None,
                status="running",
                duration_seconds=None,
            )
            future = executor.submit(
                fit_block0_null_permutation,
                Block0NullFitJob(permutation_index),
                real_bundle=real_bundle,
                run_config=run_config,
                config_bundle=config_bundle,
                state_basis=state_basis,
                device=device,
            )
            future_to_index[future] = permutation_index

        for future in as_completed(future_to_index):
            permutation_index = int(future_to_index[future])
            started_at, started_monotonic = started_by_index[permutation_index]
            try:
                result = future.result()
            except Exception as exc:
                duration_seconds = _elapsed_seconds(started_monotonic)
                _append_progress_event(
                    progress_path,
                    event="fit_failed",
                    fit_label=FIT_LABEL_NULL,
                    permutation_index=permutation_index,
                    started_at=started_at,
                    finished_at=_utc_timestamp(),
                    status="failed",
                    duration_seconds=duration_seconds,
                    error=exc,
                )
                for pending_future in future_to_index:
                    if pending_future is not future:
                        pending_future.cancel()
                raise ContractError(
                    "Block 0 null fit/extraction failed for "
                    f"permutation_index={permutation_index}: {exc}"
                ) from exc

            records_by_permutation[int(result.permutation_index)] = tuple(result.records)
            resume_state["null_records_by_permutation"][int(result.permutation_index)] = tuple(
                result.records
            )
            _write_resume_checkpoint(
                output_root=output_root,
                run_config=run_config,
                config_fingerprint=config_fingerprint,
                real_observation_signature=real_observation_signature,
                resume_state=resume_state,
            )
            _append_stage_progress_event(
                progress_path,
                stage="null_bundle_build",
                started_at=started_at,
                duration_seconds=float(result.null_bundle_build_seconds),
                status="ok",
                permutation_index=int(result.permutation_index),
            )
            _append_stage_progress_event(
                progress_path,
                stage="null_fit_extract",
                started_at=started_at,
                duration_seconds=float(result.null_fit_extract_seconds),
                status="ok",
                permutation_index=int(result.permutation_index),
            )
            _append_progress_event(
                progress_path,
                event="fit_finished",
                fit_label=FIT_LABEL_NULL,
                permutation_index=int(result.permutation_index),
                started_at=started_at,
                finished_at=_utc_timestamp(),
                status="ok",
                duration_seconds=_elapsed_seconds(started_monotonic),
                warning_summary=result.warning_summary,
            )

    null_records_list: list[Block0FitRecord] = []
    for permutation_index in range(run_config.n_permutations):
        try:
            null_records_list.extend(records_by_permutation[int(permutation_index)])
        except KeyError as exc:
            raise ContractError(
                "Block 0 parallel null fitting did not produce records for "
                f"permutation_index={permutation_index}"
            ) from exc
    return null_records_list


def _block0_parallel_context() -> mp.context.BaseContext:
    """Use spawn to avoid forking after parent STRIDE/Torch/OpenMP real fit."""
    return mp.get_context("spawn")


def _build_null_observation_bundle(
    real_bundle: Block0ObservationBundle,
    run_config,
    permutation_index: int,
) -> Block0ObservationBundle:
    assignments = build_domain_label_permutation_assignments(
        real_bundle.observations,
        permutation_index=permutation_index,
        master_seed=run_config.master_seed,
    )
    return build_null_tc_im_observations(
        real_bundle,
        assignments,
        permutation_index=permutation_index,
    )


def _build_execution_manifest_payload(
    *,
    run_config,
    config_path: Path,
    output_root: Path,
    cache_info: Mapping[str, object],
    progress_path: Path,
) -> dict[str, object]:
    return {
        "task_name": EXECUTION_NAME,
        "config_path": str(config_path),
        "stage0_h5ad": str(run_config.data_path),
        "run_scope": run_config.run_scope,
        "n_permutations": int(run_config.n_permutations),
        "master_seed": int(run_config.master_seed),
        "seed_derivation_policy": (
            "sha256(namespace|master_seed|patient_id|permutation_index) "
            "truncated to uint32"
        ),
        "real_family": REAL_FAMILY,
        "null_family": NULL_FAMILY,
        "permutation_policy": (
            "within-patient FOV/ROI domain-label permutation preserving patient_id, "
            "FOV composition, FOV count structure, and exact per-patient TC/IM counts; "
            "identity permutations allowed; cross-patient borrowing disallowed"
        ),
        "fit_status": "ok",
        "readiness_status": run_config.readiness_status,
        "patient_count": int(cache_info["patient_count"]),
        "record_count": int(cache_info["record_count"]),
        "k_states": int(cache_info["k_states"]),
        "fit_cache_schema_version": FIT_CACHE_SCHEMA_VERSION,
        "fit_cache_path": str(output_root / FIT_CACHE_FILENAME),
        "fit_cache_index_path": str(output_root / FIT_CACHE_INDEX_FILENAME),
        "fit_cache_sha256": str(cache_info["fit_cache_sha256"]),
        "fit_cache_index_sha256": str(cache_info["fit_cache_index_sha256"]),
        "progress_path": str(progress_path),
    }


def run_block0(*_args: object, **_kwargs: object) -> dict[str, Path]:
    """Retired monolithic runner; use `execute` then `analyze`."""
    raise ContractError(
        "Block 0 monolithic run is retired; use run_block0_execute and "
        "run_block0_analyze, or the CLI subcommands `execute` and `analyze`."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the Block 0 execution/analyze CLI surface."""
    parser = argparse.ArgumentParser(
        description="Task A Block 0 execution cache and cache-derived analysis.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    execute = subparsers.add_parser(
        "execute",
        help="Run real/null full STRIDE fits and write a reusable fit cache.",
    )
    execute.add_argument("--task-config", required=True)
    execute.add_argument("--stage0-h5ad", required=True)
    execute.add_argument("--output-dir", required=True)
    execute.add_argument(
        "--n-permutations",
        type=_permutation_count,
        default=DEFAULT_N_PERMUTATIONS,
        help=(
            "Number of empirical-null permutations; "
            f"full calibration uses {DEFAULT_N_PERMUTATIONS}."
        ),
    )
    execute.add_argument(
        "--master-seed",
        type=int,
        required=True,
        help="Master seed used for deterministic patient/permutation seed derivation.",
    )
    scope_group = execute.add_mutually_exclusive_group()
    scope_group.add_argument(
        "--patient-id",
        action="append",
        default=None,
        help="Optional diagnostic patient subset selector; may be repeated.",
    )
    scope_group.add_argument(
        "--demo-subset",
        default=None,
        help="Optional named diagnostic subset selector.",
    )
    execute.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted Block 0 run from checkpointed fit records.",
    )
    execute.add_argument(
        "--parallel-permutations",
        type=int,
        default=1,
        help="Number of null permutation fits to run concurrently; formal runs commonly use 8.",
    )
    execute.add_argument(
        "--worker-cpu-threads",
        type=int,
        default=4,
        help="Torch/BLAS CPU thread limit inside each permutation worker; allowed range is 1..6.",
    )
    execute.add_argument(
        "--allow-cpu-oversubscription",
        action="store_true",
        help="Allow parallel_permutations * worker_cpu_threads to exceed os.cpu_count().",
    )
    execute.add_argument(
        "--device",
        default=None,
        help="Optional torch device forwarded to fit_stride, e.g. 'cuda' or 'cpu'.",
    )

    analyze = subparsers.add_parser(
        "analyze",
        help="Derive fixed family-summary calibration tables from an existing Block 0 fit cache.",
    )
    analyze.add_argument("--fit-cache", required=True)
    analyze.add_argument("--fit-cache-index", required=True)
    analyze.add_argument("--output-dir", required=True)
    analyze.add_argument(
        "--execution-manifest",
        default=None,
        help="Optional execution manifest path; defaults to the fit-cache directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the pending Block 0 calibration CLI."""
    try:
        args = parse_args(argv)
        if args.command == "execute":
            run_block0_execute(
                config_path=args.task_config,
                data_path=args.stage0_h5ad,
                output_dir=args.output_dir,
                n_permutations=int(args.n_permutations),
                master_seed=int(args.master_seed),
                patient_ids=None if args.patient_id is None else tuple(args.patient_id),
                demo_subset_name=args.demo_subset,
                resume=bool(args.resume),
                parallel_permutations=int(args.parallel_permutations),
                worker_cpu_threads=int(args.worker_cpu_threads),
                allow_cpu_oversubscription=bool(args.allow_cpu_oversubscription),
                device=args.device,
            )
        elif args.command == "analyze":
            run_block0_analyze(
                fit_cache=args.fit_cache,
                fit_cache_index=args.fit_cache_index,
                output_dir=args.output_dir,
                execution_manifest=args.execution_manifest,
            )
        else:
            raise ContractError(f"Unsupported Block 0 command: {args.command!r}")
    except (ContractError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


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
