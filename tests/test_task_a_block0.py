from __future__ import annotations

import gc
import inspect
import json
import os
import subprocess
import sys
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


SKELETON_BLOCK0_FILES = {
    "__init__.py",
    "__main__.py",
    "cli.py",
    "schemas.py",
    "observations.py",
    "permutation.py",
    "fit.py",
    "cache.py",
    "parallel.py",
    "metrics.py",
    "writers.py",
}


def test_block0_package_is_executable_calibration_surface() -> None:
    import tasks.task_A.block0 as block0

    assert block0.BLOCK_NAME == "block0_calibration"
    assert block0.CALIBRATION_MANIFEST_FILENAME == "block0_calibration_manifest.json"
    assert block0.PATIENT_CALIBRATION_FILENAME == "block0_patient_calibration.csv"
    assert block0.METRIC_SUMMARY_FILENAME == "block0_metric_summary.csv"
    assert block0.DEFAULT_N_PERMUTATIONS == 199
    assert block0.NULL_FAMILY == "TC-IM_within_patient_domain_label_permutation_null"
    assert set(block0.__all__) == {
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
        "run_block0",
        "run_block0_analyze",
        "run_block0_execute",
    }


def test_block0_full_calibration_readiness_uses_b199(tmp_path: Path) -> None:
    from tasks.task_A.block0.schemas import (
        CALIBRATION_READY_STATUS,
        DIAGNOSTIC_READINESS_STATUS,
        FULL_COHORT_SCOPE,
        Block0RunConfig,
    )

    base = {
        "config_path": tmp_path / "config.yaml",
        "data_path": tmp_path / "stage0.h5ad",
        "output_dir": tmp_path / "block0",
        "run_scope": FULL_COHORT_SCOPE,
        "master_seed": 7,
    }

    assert Block0RunConfig(n_permutations=199, **base).readiness_status == CALIBRATION_READY_STATUS
    assert Block0RunConfig(n_permutations=198, **base).readiness_status == DIAGNOSTIC_READINESS_STATUS


def test_block0_parallel_cpu_budget_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    from stride.errors import ContractError
    import tasks.task_A.block0.cli as cli_module

    monkeypatch.setattr(cli_module.os, "cpu_count", lambda: 32)

    cli_module._guard_block0_cpu_budget(
        parallel_permutations=8,
        worker_cpu_threads=4,
        allow_cpu_oversubscription=False,
    )
    with pytest.raises(ContractError, match="CPU budget"):
        cli_module._guard_block0_cpu_budget(
            parallel_permutations=8,
            worker_cpu_threads=6,
            allow_cpu_oversubscription=False,
        )
    cli_module._guard_block0_cpu_budget(
        parallel_permutations=8,
        worker_cpu_threads=6,
        allow_cpu_oversubscription=True,
    )


def test_block0_file_layout_matches_calibration_skeleton() -> None:
    block0_dir = ROOT / "tasks" / "task_A" / "block0"
    python_files = {path.name for path in block0_dir.glob("*.py")}

    assert python_files == SKELETON_BLOCK0_FILES


@pytest.mark.parametrize(
    "module_name, public_names",
    [
        (
            "schemas",
            (
                "Block0DomainLabelPermutationAssignment",
                "Block0FitRecord",
                "Block0PatientDomainCounts",
                "Block0RunConfig",
                "Block0PatientCalibrationRow",
            ),
        ),
        ("observations", ("Block0ObservationBundle", "build_null_tc_im_observations", "build_real_tc_im_observations")),
        (
            "permutation",
            (
                "derive_block0_seed",
                "build_patient_domain_counts",
                "build_domain_label_permutation_assignments",
            ),
        ),
        ("fit", ("extract_block0_fit_records", "fit_block0_family", "require_all_patient_results_ok")),
        ("cache", ("read_block0_fit_cache", "sha256_file", "write_block0_fit_cache")),
        ("parallel", ("Block0NullFitJob", "Block0NullFitResult", "fit_block0_null_permutation")),
        ("metrics", ("build_block0_calibration_frames", "effect_ratio", "empirical_tail_p_value", "family_summary_values", "tail_null_fraction")),
        ("writers", ("validate_block0_frame_columns", "write_block0_analysis_outputs", "write_block0_execution_outputs")),
    ],
)
def test_block0_skeleton_modules_expose_planned_names(
    module_name: str,
    public_names: tuple[str, ...],
) -> None:
    module = __import__(f"tasks.task_A.block0.{module_name}", fromlist=["*"])

    assert isinstance(module, ModuleType)
    for public_name in public_names:
        assert hasattr(module, public_name)


def test_block0_real_observation_builder_does_not_accept_external_state_basis() -> None:
    from tasks.task_A.block0.observations import build_real_tc_im_observations

    signature = inspect.signature(build_real_tc_im_observations)

    assert "state_basis" not in signature.parameters
    assert tuple(signature.parameters) == ("handle", "config_bundle", "patient_ids")


def test_block0_execute_writes_fit_cache_from_fit_records(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tasks.task_A.block0 import run_block0_execute
    from tasks.task_A.block0.cache import read_block0_fit_cache
    from tasks.task_A.block0.observations import Block0ObservationBundle
    from tasks.task_A.block0.schemas import FIT_LABEL_NULL, FIT_LABEL_REAL

    real_bundle = Block0ObservationBundle(
        label=FIT_LABEL_REAL,
        observations=(
            _fov_observation(patient_id="P01", fov_id="P01_TC_1", domain_label="TC"),
            _fov_observation(patient_id="P01", fov_id="P01_IM_1", domain_label="IM"),
        ),
        patient_ids=("P01",),
    )
    config_bundle = SimpleNamespace(
        config_path=tmp_path / "config.yaml",
        config_fingerprint="fixture-fingerprint",
        data=SimpleNamespace(mass_mode="uniform"),
    )
    fit_calls: list[tuple[str, int | None, object]] = []
    load_calls: list[tuple[object, bool]] = []
    live_null_refs: list[weakref.ReferenceType] = []

    class TrackedFitResult:
        def __init__(self, fit_label: str, permutation_index: int | None) -> None:
            self.fit_label = fit_label
            self.permutation_index = permutation_index
            self.warnings = (
                (f"{fit_label} fit warning",)
                if fit_label == FIT_LABEL_REAL or permutation_index == 1
                else ()
            )

    monkeypatch.setattr("tasks.task_A.block0.cli.load_task_a_config_bundle", lambda _path: config_bundle)

    def fake_load_dataset(path, *, backed=False):
        load_calls.append((path, backed))
        return object()

    monkeypatch.setattr("tasks.task_A.block0.cli.load_task_a_dataset_handle", fake_load_dataset)
    monkeypatch.setattr("tasks.task_A.block0.cli.resolve_task_a_state_basis", lambda _handle: "state-basis")
    monkeypatch.setattr(
        "tasks.task_A.block0.cli.build_real_tc_im_observations",
        lambda _handle, _config_bundle, patient_ids=None: real_bundle,
    )

    def fake_fit(bundle, *, config_bundle, state_basis, fit_label, permutation_index=None):
        fit_calls.append((fit_label, permutation_index, state_basis))
        if fit_label == FIT_LABEL_NULL:
            gc.collect()
            assert all(ref() is None for ref in live_null_refs), "previous null fit results were retained"
            result = TrackedFitResult(fit_label, permutation_index)
            live_null_refs.append(weakref.ref(result))
            return result
        return TrackedFitResult(fit_label, permutation_index)

    def fake_extract(result, *, fit_label, permutation_index=None):
        if fit_label == FIT_LABEL_REAL:
            return (
                _block0_fit_record(
                    fit_label=FIT_LABEL_REAL,
                    permutation_index=None,
                    variant=0.0,
                ),
            )
        return (
            _block0_fit_record(
                fit_label=FIT_LABEL_NULL,
                permutation_index=permutation_index,
                variant=float(permutation_index + 1),
            ),
        )

    monkeypatch.setattr("tasks.task_A.block0.cli.fit_block0_family", fake_fit)
    monkeypatch.setattr(
        "tasks.task_A.block0.cli.fit_block0_null_batch",
        lambda *_args, **_kwargs: pytest.fail("run_block0_execute must stream null fits instead of using null batch"),
        raising=False,
    )
    monkeypatch.setattr("tasks.task_A.block0.cli.extract_block0_fit_records", fake_extract)

    output_dir = tmp_path / "block0"
    output_dir.mkdir()
    paths = run_block0_execute(
        config_path=tmp_path / "config.yaml",
        data_path=tmp_path / "stage0.h5ad",
        output_dir=output_dir,
        n_permutations=3,
        master_seed=7,
    )

    assert {path.name for path in paths.values()} == {
        "block0_execution_manifest.json",
    }
    assert {path.name for path in output_dir.iterdir()} == {
        "block0_execution_manifest.json",
        "block0_execution_progress.jsonl",
        "block0_fit_cache.npz",
        "block0_fit_cache_index.csv",
    }
    manifest = json.loads((output_dir / "block0_execution_manifest.json").read_text(encoding="utf-8"))
    assert manifest["task_name"] == "block0_execution_cache"
    assert manifest["fit_status"] == "ok"
    assert manifest["readiness_status"] == "diagnostic"
    assert manifest["n_permutations"] == 3
    assert manifest["record_count"] == 4
    assert manifest["fit_cache_path"].endswith("block0_fit_cache.npz")
    assert [call[:2] for call in fit_calls] == [
        (FIT_LABEL_REAL, None),
        (FIT_LABEL_NULL, 0),
        (FIT_LABEL_NULL, 1),
        (FIT_LABEL_NULL, 2),
    ]
    assert all(call[2] == "state-basis" for call in fit_calls)
    assert load_calls == [(tmp_path / "stage0.h5ad", True)]
    assert "progress" not in paths
    assert manifest["progress_path"].endswith("block0_execution_progress.jsonl")
    assert "warning_count" not in manifest
    assert "warnings" not in manifest
    cache_records = read_block0_fit_cache(
        output_dir / "block0_fit_cache.npz",
        output_dir / "block0_fit_cache_index.csv",
    )
    assert len(cache_records) == 4
    assert [record.fit_label for record in cache_records] == [
        FIT_LABEL_REAL,
        FIT_LABEL_NULL,
        FIT_LABEL_NULL,
        FIT_LABEL_NULL,
    ]

    progress_lines = [
        json.loads(line)
        for line in (output_dir / "block0_execution_progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    real_finished_rows = [
        row for row in progress_lines if row["fit_label"] == FIT_LABEL_REAL and row["status"] == "ok"
    ]
    assert len(real_finished_rows) == 1
    assert real_finished_rows[0]["warning_count"] == 1
    assert real_finished_rows[0]["warnings"] == ["real fit warning"]
    finished_null_rows = [
        row for row in progress_lines if row["fit_label"] == FIT_LABEL_NULL and row["status"] == "ok"
    ]
    assert [row["permutation_index"] for row in finished_null_rows] == [0, 1, 2]
    assert all(row["started_at"] and row["finished_at"] for row in finished_null_rows)
    assert all("warning_count" in row and "warnings" in row for row in finished_null_rows)
    assert [row["warning_count"] for row in finished_null_rows] == [0, 1, 0]
    assert finished_null_rows[1]["warnings"] == ["null fit warning"]
    assert not (output_dir / "block0_patient_calibration.csv").exists()
    assert not (output_dir / "block0_metric_summary.csv").exists()
    gc.collect()
    assert all(ref() is None for ref in live_null_refs)


def test_block0_progress_sidecar_records_timing_without_formal_output_timing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tasks.task_A.block0 import run_block0_execute
    from tasks.task_A.block0.observations import Block0ObservationBundle
    from tasks.task_A.block0.schemas import FIT_LABEL_NULL, FIT_LABEL_REAL

    real_bundle = Block0ObservationBundle(
        label=FIT_LABEL_REAL,
        observations=(
            _fov_observation(patient_id="P01", fov_id="P01_TC_1", domain_label="TC"),
            _fov_observation(patient_id="P01", fov_id="P01_IM_1", domain_label="IM"),
        ),
        patient_ids=("P01",),
    )
    config_bundle = SimpleNamespace(
        config_path=tmp_path / "config.yaml",
        config_fingerprint="fixture-fingerprint",
        data=SimpleNamespace(mass_mode="uniform"),
    )

    monkeypatch.setattr("tasks.task_A.block0.cli.load_task_a_config_bundle", lambda _path: config_bundle)
    monkeypatch.setattr("tasks.task_A.block0.cli.load_task_a_dataset_handle", lambda _path, *, backed=False: object())
    monkeypatch.setattr("tasks.task_A.block0.cli.resolve_task_a_state_basis", lambda _handle: "state-basis")
    monkeypatch.setattr(
        "tasks.task_A.block0.cli.build_real_tc_im_observations",
        lambda _handle, _config_bundle, patient_ids=None: real_bundle,
    )

    def fake_fit(bundle, *, config_bundle, state_basis, fit_label, permutation_index=None):
        return SimpleNamespace(fit_label=fit_label, permutation_index=permutation_index)

    def fake_extract(result, *, fit_label, permutation_index=None):
        if fit_label == FIT_LABEL_REAL:
            return (_block0_fit_record(fit_label=FIT_LABEL_REAL, permutation_index=None, variant=0.0),)
        return (
            _block0_fit_record(
                fit_label=FIT_LABEL_NULL,
                permutation_index=permutation_index,
                variant=float(permutation_index + 1),
            ),
        )

    monkeypatch.setattr("tasks.task_A.block0.cli.fit_block0_family", fake_fit)
    monkeypatch.setattr("tasks.task_A.block0.cli.extract_block0_fit_records", fake_extract)

    output_dir = tmp_path / "block0"
    paths = run_block0_execute(
        config_path=tmp_path / "config.yaml",
        data_path=tmp_path / "stage0.h5ad",
        output_dir=output_dir,
        n_permutations=3,
        master_seed=7,
    )

    assert set(paths) == {"manifest"}
    manifest = json.loads((output_dir / "block0_execution_manifest.json").read_text(encoding="utf-8"))
    assert not any("duration" in key or "elapsed" in key or "timing" in key for key in manifest)
    for csv_name in ("block0_fit_cache_index.csv",):
        header = (output_dir / csv_name).read_text(encoding="utf-8").splitlines()[0]
        assert "duration" not in header
        assert "elapsed" not in header
        assert "timing" not in header

    progress_rows = [
        json.loads(line)
        for line in (output_dir / "block0_execution_progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    fit_started_rows = [row for row in progress_rows if row["event"] == "fit_started"]
    fit_terminal_rows = [row for row in progress_rows if row["event"] in {"fit_finished", "fit_failed"}]
    assert fit_started_rows
    assert all(row["duration_seconds"] is None for row in fit_started_rows)
    assert fit_terminal_rows
    assert all(isinstance(row["duration_seconds"], float) for row in fit_terminal_rows)
    assert all(row["duration_seconds"] >= 0.0 for row in fit_terminal_rows)

    stage_rows = [row for row in progress_rows if row["event"] == "stage_finished"]
    assert {row["stage"] for row in stage_rows} >= {
        "input_config_data_loading",
        "real_observation_construction",
        "state_basis_resolution",
        "real_fit_extract",
        "null_bundle_build",
        "null_fit_extract",
        "fit_cache_write",
        "execution_manifest_write",
    }
    assert all(row["status"] == "ok" for row in stage_rows)
    assert all(isinstance(row["duration_seconds"], float) for row in stage_rows)
    assert all(row["duration_seconds"] >= 0.0 for row in stage_rows)


def test_block0_parallel_null_fit_results_are_sorted_and_checkpointed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import tasks.task_A.block0.cli as cli_module
    from tasks.task_A.block0.observations import Block0ObservationBundle
    from tasks.task_A.block0.parallel import Block0NullFitResult
    from tasks.task_A.block0.schemas import FIT_LABEL_NULL, FIT_LABEL_REAL, FULL_COHORT_SCOPE

    real_bundle = Block0ObservationBundle(
        label=FIT_LABEL_REAL,
        observations=(
            _fov_observation(patient_id="P01", fov_id="P01_TC_1", domain_label="TC"),
            _fov_observation(patient_id="P01", fov_id="P01_IM_1", domain_label="IM"),
        ),
        patient_ids=("P01",),
    )
    run_config = SimpleNamespace(
        config_path=tmp_path / "config.yaml",
        data_path=tmp_path / "stage0.h5ad",
        output_dir=tmp_path / "block0",
        run_scope=FULL_COHORT_SCOPE,
        n_permutations=3,
        master_seed=7,
        patient_ids=None,
        demo_subset_name=None,
    )
    progress_path = run_config.output_dir / "block0_execution_progress.jsonl"
    progress_path.parent.mkdir()
    progress_path.write_text("", encoding="utf-8")
    resume_state = {"real_records": (), "null_records_by_permutation": {}}
    real_signature = {
        "label": FIT_LABEL_REAL,
        "patient_ids": ["P01"],
        "n_observations": 2,
        "digest": "fixture",
    }

    def fake_worker(job, **_kwargs):
        return Block0NullFitResult(
            permutation_index=job.permutation_index,
            records=(
                _block0_fit_record(
                    fit_label=FIT_LABEL_NULL,
                    permutation_index=job.permutation_index,
                    variant=float(job.permutation_index + 1),
                ),
            ),
            warning_summary={"warning_count": 0, "warnings": []},
            null_bundle_build_seconds=0.1,
            null_fit_extract_seconds=0.2,
        )

    class FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

        def cancel(self):
            return False

    class FakeExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return FakeFuture(fn(*args, **kwargs))

    monkeypatch.setattr(cli_module, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(cli_module, "as_completed", lambda futures: tuple(reversed(tuple(futures))))
    monkeypatch.setattr(cli_module, "fit_block0_null_permutation", fake_worker)

    records = cli_module._run_parallel_null_permutation_fits(
        real_bundle=real_bundle,
        run_config=run_config,
        config_bundle=SimpleNamespace(config_fingerprint="fixture-fingerprint"),
        state_basis="state-basis",
        output_root=run_config.output_dir,
        progress_path=progress_path,
        config_fingerprint="fixture-fingerprint",
        real_observation_signature=real_signature,
        resume_state=resume_state,
        parallel_permutations=2,
        worker_cpu_threads=4,
    )

    assert [record.permutation_index for record in records] == [0, 1, 2]
    checkpoint = json.loads(
        (run_config.output_dir / cli_module.BLOCK0_RESUME_CHECKPOINT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert sorted(checkpoint["null_records_by_permutation"]) == ["0", "1", "2"]
    progress_rows = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
    ]
    finished = [row for row in progress_rows if row["event"] == "fit_finished"]
    assert sorted(row["permutation_index"] for row in finished) == [0, 1, 2]


def test_block0_parallel_executor_uses_spawn_context_after_real_fit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import tasks.task_A.block0.cli as cli_module
    from tasks.task_A.block0.observations import Block0ObservationBundle
    from tasks.task_A.block0.parallel import Block0NullFitResult
    from tasks.task_A.block0.schemas import FIT_LABEL_NULL, FIT_LABEL_REAL, FULL_COHORT_SCOPE

    real_bundle = Block0ObservationBundle(
        label=FIT_LABEL_REAL,
        observations=(
            _fov_observation(patient_id="P01", fov_id="P01_TC_1", domain_label="TC"),
            _fov_observation(patient_id="P01", fov_id="P01_IM_1", domain_label="IM"),
        ),
        patient_ids=("P01",),
    )
    run_config = SimpleNamespace(
        config_path=tmp_path / "config.yaml",
        data_path=tmp_path / "stage0.h5ad",
        output_dir=tmp_path / "block0",
        run_scope=FULL_COHORT_SCOPE,
        n_permutations=1,
        master_seed=7,
        patient_ids=None,
        demo_subset_name=None,
    )
    progress_path = run_config.output_dir / "block0_execution_progress.jsonl"
    progress_path.parent.mkdir()
    progress_path.write_text("", encoding="utf-8")
    executor_kwargs: list[dict[str, object]] = []

    class FakeFuture:
        def result(self):
            return Block0NullFitResult(
                permutation_index=0,
                records=(
                    _block0_fit_record(
                        fit_label=FIT_LABEL_NULL,
                        permutation_index=0,
                        variant=1.0,
                    ),
                ),
                warning_summary={"warning_count": 0, "warnings": []},
                null_bundle_build_seconds=0.1,
                null_fit_extract_seconds=0.2,
            )

        def cancel(self):
            return False

    class FakeExecutor:
        def __init__(self, *args, **kwargs):
            executor_kwargs.append(dict(kwargs))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, *_args, **_kwargs):
            return FakeFuture()

    monkeypatch.setattr(cli_module, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(cli_module, "as_completed", lambda futures: tuple(futures))

    cli_module._run_parallel_null_permutation_fits(
        real_bundle=real_bundle,
        run_config=run_config,
        config_bundle=SimpleNamespace(config_fingerprint="fixture-fingerprint"),
        state_basis="state-basis",
        output_root=run_config.output_dir,
        progress_path=progress_path,
        config_fingerprint="fixture-fingerprint",
        real_observation_signature={
            "label": FIT_LABEL_REAL,
            "patient_ids": ["P01"],
            "n_observations": 2,
            "digest": "fixture",
        },
        resume_state={"real_records": (), "null_records_by_permutation": {}},
        parallel_permutations=2,
        worker_cpu_threads=4,
    )

    assert executor_kwargs
    mp_context = executor_kwargs[0]["mp_context"]
    assert mp_context.get_start_method() == "spawn"


@pytest.mark.parametrize(
    "existing_name",
    [
        "block0_execution_manifest.json",
        "block0_fit_cache.npz",
        "block0_fit_cache_index.csv",
        "block0_calibration_manifest.json",
        "block0_patient_calibration.csv",
        "block0_metric_summary.csv",
        "block0_execution_progress.jsonl",
    ],
)
def test_block0_run_rejects_existing_formal_or_progress_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    existing_name: str,
) -> None:
    from stride.errors import ContractError
    from tasks.task_A.block0 import run_block0_execute

    output_dir = tmp_path / "block0"
    output_dir.mkdir()
    (output_dir / existing_name).write_text("stale previous run\n", encoding="utf-8")

    monkeypatch.setattr(
        "tasks.task_A.block0.cli.load_task_a_config_bundle",
        lambda _path: pytest.fail("run_block0_execute must fail before loading inputs when stale outputs exist"),
    )

    with pytest.raises(ContractError, match="existing stale.*output"):
        run_block0_execute(
            config_path=tmp_path / "config.yaml",
            data_path=tmp_path / "stage0.h5ad",
            output_dir=output_dir,
            n_permutations=3,
            master_seed=7,
        )


def test_block0_null_failure_reports_permutation_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from stride.errors import ContractError
    from tasks.task_A.block0 import run_block0_execute
    from tasks.task_A.block0.observations import Block0ObservationBundle
    from tasks.task_A.block0.schemas import FIT_LABEL_NULL, FIT_LABEL_REAL

    real_bundle = Block0ObservationBundle(
        label=FIT_LABEL_REAL,
        observations=(
            _fov_observation(patient_id="P01", fov_id="P01_TC_1", domain_label="TC"),
            _fov_observation(patient_id="P01", fov_id="P01_IM_1", domain_label="IM"),
        ),
        patient_ids=("P01",),
    )
    config_bundle = SimpleNamespace(
        config_path=tmp_path / "config.yaml",
        config_fingerprint="fixture-fingerprint",
        data=SimpleNamespace(mass_mode="uniform"),
    )

    monkeypatch.setattr("tasks.task_A.block0.cli.load_task_a_config_bundle", lambda _path: config_bundle)
    monkeypatch.setattr("tasks.task_A.block0.cli.load_task_a_dataset_handle", lambda _path, *, backed=False: object())
    monkeypatch.setattr("tasks.task_A.block0.cli.resolve_task_a_state_basis", lambda _handle: "state-basis")
    monkeypatch.setattr(
        "tasks.task_A.block0.cli.build_real_tc_im_observations",
        lambda _handle, _config_bundle, patient_ids=None: real_bundle,
    )

    def fake_fit(bundle, *, config_bundle, state_basis, fit_label, permutation_index=None):
        if fit_label == FIT_LABEL_NULL and permutation_index == 1:
            raise ContractError("simulated canonical fit failure")
        return SimpleNamespace(fit_label=fit_label, permutation_index=permutation_index)

    def fake_extract(result, *, fit_label, permutation_index=None):
        if fit_label == FIT_LABEL_REAL:
            return (_block0_fit_record(fit_label=FIT_LABEL_REAL, permutation_index=None, variant=0.0),)
        return (
            _block0_fit_record(
                fit_label=FIT_LABEL_NULL,
                permutation_index=permutation_index,
                variant=float(permutation_index + 1),
            ),
        )

    monkeypatch.setattr("tasks.task_A.block0.cli.fit_block0_family", fake_fit)
    monkeypatch.setattr("tasks.task_A.block0.cli.extract_block0_fit_records", fake_extract)

    output_dir = tmp_path / "block0"
    with pytest.raises(ContractError, match="permutation_index=1"):
        run_block0_execute(
            config_path=tmp_path / "config.yaml",
            data_path=tmp_path / "stage0.h5ad",
            output_dir=output_dir,
            n_permutations=3,
            master_seed=7,
        )

    progress_lines = [
        json.loads(line)
        for line in (output_dir / "block0_execution_progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    failed_rows = [row for row in progress_lines if row["status"] == "failed"]
    assert failed_rows[-1]["permutation_index"] == 1
    assert failed_rows[-1]["fit_label"] == FIT_LABEL_NULL
    assert failed_rows[-1]["error_type"] == "ContractError"
    assert "simulated canonical fit failure" in failed_rows[-1]["error_message"]


def test_block0_resume_reuses_completed_fit_record_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from stride.errors import ContractError
    import tasks.task_A.block0.cli as cli_module
    from tasks.task_A.block0 import run_block0_execute
    from tasks.task_A.block0.observations import Block0ObservationBundle
    from tasks.task_A.block0.schemas import FIT_LABEL_NULL, FIT_LABEL_REAL

    real_bundle = Block0ObservationBundle(
        label=FIT_LABEL_REAL,
        observations=(
            _fov_observation(patient_id="P01", fov_id="P01_TC_1", domain_label="TC"),
            _fov_observation(patient_id="P01", fov_id="P01_IM_1", domain_label="IM"),
        ),
        patient_ids=("P01",),
    )
    config_bundle = SimpleNamespace(
        config_path=tmp_path / "config.yaml",
        config_fingerprint="fixture-fingerprint",
        data=SimpleNamespace(mass_mode="uniform"),
    )
    fit_calls: list[tuple[str, int | None]] = []
    fail_once = {"active": True}

    monkeypatch.setattr("tasks.task_A.block0.cli.load_task_a_config_bundle", lambda _path: config_bundle)
    monkeypatch.setattr("tasks.task_A.block0.cli.load_task_a_dataset_handle", lambda _path, *, backed=False: object())
    monkeypatch.setattr("tasks.task_A.block0.cli.resolve_task_a_state_basis", lambda _handle: "state-basis")
    monkeypatch.setattr(
        "tasks.task_A.block0.cli.build_real_tc_im_observations",
        lambda _handle, _config_bundle, patient_ids=None: real_bundle,
    )

    def fake_fit(bundle, *, config_bundle, state_basis, fit_label, permutation_index=None):
        fit_calls.append((fit_label, permutation_index))
        if fit_label == FIT_LABEL_NULL and permutation_index == 1 and fail_once["active"]:
            fail_once["active"] = False
            raise RuntimeError("synthetic_interrupt")
        return SimpleNamespace(fit_label=fit_label, permutation_index=permutation_index)

    def fake_extract(result, *, fit_label, permutation_index=None):
        if fit_label == FIT_LABEL_REAL:
            return (_block0_fit_record(fit_label=FIT_LABEL_REAL, permutation_index=None, variant=0.0),)
        return (
            _block0_fit_record(
                fit_label=FIT_LABEL_NULL,
                permutation_index=permutation_index,
                variant=float(permutation_index + 1),
            ),
        )

    monkeypatch.setattr("tasks.task_A.block0.cli.fit_block0_family", fake_fit)
    monkeypatch.setattr("tasks.task_A.block0.cli.extract_block0_fit_records", fake_extract)

    output_dir = tmp_path / "block0"
    with pytest.raises(ContractError, match="permutation_index=1"):
        run_block0_execute(
            config_path=tmp_path / "config.yaml",
            data_path=tmp_path / "stage0.h5ad",
            output_dir=output_dir,
            n_permutations=3,
            master_seed=7,
        )

    checkpoint_path = output_dir / cli_module.BLOCK0_RESUME_CHECKPOINT_FILENAME
    assert checkpoint_path.exists()

    paths = run_block0_execute(
        config_path=tmp_path / "config.yaml",
        data_path=tmp_path / "stage0.h5ad",
        output_dir=output_dir,
        n_permutations=3,
        master_seed=7,
        resume=True,
    )

    assert {path.name for path in paths.values()} == {
        "block0_execution_manifest.json",
    }
    assert (output_dir / "block0_fit_cache.npz").exists()
    assert (output_dir / "block0_fit_cache_index.csv").exists()
    assert not checkpoint_path.exists()
    assert fit_calls == [
        (FIT_LABEL_REAL, None),
        (FIT_LABEL_NULL, 0),
        (FIT_LABEL_NULL, 1),
        (FIT_LABEL_NULL, 1),
        (FIT_LABEL_NULL, 2),
    ]


def test_block0_resume_rejects_stale_real_observation_surface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from stride.errors import ContractError
    import tasks.task_A.block0.cli as cli_module
    from tasks.task_A.block0 import run_block0_execute
    from tasks.task_A.block0.observations import Block0ObservationBundle
    from tasks.task_A.block0.schemas import FIT_LABEL_NULL, FIT_LABEL_REAL

    real_bundle_p01 = Block0ObservationBundle(
        label=FIT_LABEL_REAL,
        observations=(
            _fov_observation(patient_id="P01", fov_id="P01_TC_1", domain_label="TC"),
            _fov_observation(patient_id="P01", fov_id="P01_IM_1", domain_label="IM"),
        ),
        patient_ids=("P01",),
    )
    real_bundle_p02 = Block0ObservationBundle(
        label=FIT_LABEL_REAL,
        observations=(
            _fov_observation(patient_id="P02", fov_id="P02_TC_1", domain_label="TC"),
            _fov_observation(patient_id="P02", fov_id="P02_IM_1", domain_label="IM"),
        ),
        patient_ids=("P02",),
    )
    active_bundle = {"value": real_bundle_p01}
    config_bundle = SimpleNamespace(
        config_path=tmp_path / "config.yaml",
        config_fingerprint="fixture-fingerprint",
        data=SimpleNamespace(mass_mode="uniform"),
    )
    fit_calls: list[tuple[str, int | None]] = []

    monkeypatch.setattr("tasks.task_A.block0.cli.load_task_a_config_bundle", lambda _path: config_bundle)
    monkeypatch.setattr("tasks.task_A.block0.cli.load_task_a_dataset_handle", lambda _path, *, backed=False: object())
    monkeypatch.setattr("tasks.task_A.block0.cli.resolve_task_a_state_basis", lambda _handle: "state-basis")
    monkeypatch.setattr(
        "tasks.task_A.block0.cli.build_real_tc_im_observations",
        lambda _handle, _config_bundle, patient_ids=None: active_bundle["value"],
    )

    def fake_fit(bundle, *, config_bundle, state_basis, fit_label, permutation_index=None):
        fit_calls.append((fit_label, permutation_index))
        if fit_label == FIT_LABEL_NULL and permutation_index == 1:
            raise RuntimeError("synthetic_interrupt")
        return SimpleNamespace(fit_label=fit_label, permutation_index=permutation_index)

    def fake_extract(result, *, fit_label, permutation_index=None):
        if fit_label == FIT_LABEL_REAL:
            return (_block0_fit_record(fit_label=FIT_LABEL_REAL, permutation_index=None, variant=0.0),)
        return (
            _block0_fit_record(
                fit_label=FIT_LABEL_NULL,
                permutation_index=permutation_index,
                variant=float(permutation_index + 1),
            ),
        )

    monkeypatch.setattr("tasks.task_A.block0.cli.fit_block0_family", fake_fit)
    monkeypatch.setattr("tasks.task_A.block0.cli.extract_block0_fit_records", fake_extract)

    output_dir = tmp_path / "block0"
    with pytest.raises(ContractError, match="permutation_index=1"):
        run_block0_execute(
            config_path=tmp_path / "config.yaml",
            data_path=tmp_path / "stage0.h5ad",
            output_dir=output_dir,
            n_permutations=3,
            master_seed=7,
        )

    assert (output_dir / cli_module.BLOCK0_RESUME_CHECKPOINT_FILENAME).exists()
    active_bundle["value"] = real_bundle_p02

    with pytest.raises(ContractError, match="checkpoint does not match"):
        run_block0_execute(
            config_path=tmp_path / "config.yaml",
            data_path=tmp_path / "stage0.h5ad",
            output_dir=output_dir,
            n_permutations=3,
            master_seed=7,
            resume=True,
        )

    assert fit_calls == [
        (FIT_LABEL_REAL, None),
        (FIT_LABEL_NULL, 0),
        (FIT_LABEL_NULL, 1),
    ]


def test_block0_real_failure_reports_fit_label(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from stride.errors import ContractError
    from tasks.task_A.block0 import run_block0_execute
    from tasks.task_A.block0.observations import Block0ObservationBundle
    from tasks.task_A.block0.schemas import FIT_LABEL_REAL

    real_bundle = Block0ObservationBundle(
        label=FIT_LABEL_REAL,
        observations=(
            _fov_observation(patient_id="P01", fov_id="P01_TC_1", domain_label="TC"),
            _fov_observation(patient_id="P01", fov_id="P01_IM_1", domain_label="IM"),
        ),
        patient_ids=("P01",),
    )
    config_bundle = SimpleNamespace(
        config_path=tmp_path / "config.yaml",
        config_fingerprint="fixture-fingerprint",
        data=SimpleNamespace(mass_mode="uniform"),
    )

    monkeypatch.setattr("tasks.task_A.block0.cli.load_task_a_config_bundle", lambda _path: config_bundle)
    monkeypatch.setattr("tasks.task_A.block0.cli.load_task_a_dataset_handle", lambda _path, *, backed=False: object())
    monkeypatch.setattr("tasks.task_A.block0.cli.resolve_task_a_state_basis", lambda _handle: "state-basis")
    monkeypatch.setattr(
        "tasks.task_A.block0.cli.build_real_tc_im_observations",
        lambda _handle, _config_bundle, patient_ids=None: real_bundle,
    )
    monkeypatch.setattr(
        "tasks.task_A.block0.cli.fit_block0_family",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ContractError("simulated real failure")),
    )

    with pytest.raises(ContractError, match="fit_label=real"):
        run_block0_execute(
            config_path=tmp_path / "config.yaml",
            data_path=tmp_path / "stage0.h5ad",
            output_dir=tmp_path / "block0",
            n_permutations=3,
            master_seed=7,
        )


def test_block0_cli_requires_master_seed() -> None:
    from tasks.task_A.block0 import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            [
                "execute",
                "--task-config",
                "tasks/task_A/config.yaml",
                "--stage0-h5ad",
                "stage0.h5ad",
                "--output-dir",
                "/tmp/block0",
            ]
        )


def test_block0_cli_rejects_too_few_permutations() -> None:
    from tasks.task_A.block0 import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            [
                "execute",
                "--task-config",
                "tasks/task_A/config.yaml",
                "--stage0-h5ad",
                "stage0.h5ad",
                "--output-dir",
                "/tmp/block0",
                "--master-seed",
                "7",
                "--n-permutations",
                "1",
            ]
        )


def test_block0_analyze_cli_does_not_accept_analysis_spec() -> None:
    from tasks.task_A.block0 import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            [
                "analyze",
                "--fit-cache",
                "block0_fit_cache.npz",
                "--fit-cache-index",
                "block0_fit_cache_index.csv",
                "--output-dir",
                "/tmp/block0",
                "--analysis-spec",
                "legacy_a_d_e_distance_v1",
            ]
        )


def test_block0_cli_help_exposes_execution_and_analysis_subcommands() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tasks.task_A.block0",
            "--help",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": f"{ROOT}{os.pathsep}{SRC}"},
        timeout=60,
    )

    assert result.returncode == 0
    assert "execution cache" in result.stdout
    assert "execute" in result.stdout
    assert "analyze" in result.stdout


def test_block0_seed_derivation_is_stable_and_namespaced() -> None:
    from tasks.task_A.block0.permutation import derive_block0_seed

    seed_a = derive_block0_seed(master_seed=7, patient_id="P01", permutation_index=3)
    seed_b = derive_block0_seed(master_seed=7, patient_id="P01", permutation_index=3)
    seed_other_patient = derive_block0_seed(master_seed=7, patient_id="P02", permutation_index=3)
    seed_other_permutation = derive_block0_seed(master_seed=7, patient_id="P01", permutation_index=4)

    assert seed_a == seed_b
    assert seed_a != seed_other_patient
    assert seed_a != seed_other_permutation
    assert 0 <= seed_a < 2**32


def _fov_observation(
    *,
    patient_id: str,
    fov_id: str,
    domain_label: str,
    variant: float = 0.0,
):
    from stride.observation.contracts import FovObservation

    return FovObservation(
        patient_id=patient_id,
        timepoint=domain_label,
        fov_id=fov_id,
        community_composition=np.asarray([0.75 - variant, 0.25 + variant], dtype=float),
        mass=1.0,
        mass_mode="uniform",
        domain_label=domain_label,
        metadata={"source": "test"},
    )


def test_block0_domain_label_permutation_preserves_patient_and_counts() -> None:
    from collections import Counter, defaultdict

    from tasks.task_A.block0.permutation import (
        build_domain_label_permutation_assignments,
        build_patient_domain_counts,
    )

    observations = (
        _fov_observation(patient_id="P01", fov_id="P01_TC_1", domain_label="TC"),
        _fov_observation(patient_id="P01", fov_id="P01_TC_2", domain_label="TC"),
        _fov_observation(patient_id="P01", fov_id="P01_IM_1", domain_label="IM"),
        _fov_observation(patient_id="P02", fov_id="P02_TC_1", domain_label="TC"),
        _fov_observation(patient_id="P02", fov_id="P02_IM_1", domain_label="IM"),
        _fov_observation(patient_id="P02", fov_id="P02_IM_2", domain_label="IM"),
    )

    counts = build_patient_domain_counts(observations)
    assignments = build_domain_label_permutation_assignments(
        observations,
        permutation_index=5,
        master_seed=11,
    )

    assert [(row.patient_id, row.n_TC, row.n_IM) for row in counts] == [
        ("P01", 2, 1),
        ("P02", 1, 2),
    ]
    assert {assignment.patient_id for assignment in assignments} == {"P01", "P02"}
    assert {assignment.fov_id for assignment in assignments} == {
        observation.fov_id for observation in observations
    }
    assert all(not hasattr(assignment, "donor_patient_id") for assignment in assignments)
    assert all(hasattr(assignment, "original_domain_label") for assignment in assignments)
    assert all(hasattr(assignment, "permuted_domain_label") for assignment in assignments)
    assert all(assignment.permutation_index == 5 for assignment in assignments)

    original_counts: dict[str, Counter[str]] = defaultdict(Counter)
    permuted_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for observation in observations:
        original_counts[observation.patient_id][observation.domain_label] += 1
    for assignment in assignments:
        permuted_counts[assignment.patient_id][assignment.permuted_domain_label] += 1
    assert original_counts == permuted_counts


def test_block0_missing_patient_domain_fails_fast() -> None:
    from stride.errors import ContractError
    from tasks.task_A.block0.permutation import build_domain_label_permutation_assignments

    with pytest.raises(ContractError, match="both TC and IM"):
        build_domain_label_permutation_assignments(
            (
                _fov_observation(patient_id="P01", fov_id="P01_TC_1", domain_label="TC"),
                _fov_observation(patient_id="P01", fov_id="P01_TC_2", domain_label="TC"),
            ),
            permutation_index=0,
            master_seed=11,
        )


def test_block0_null_observations_rewrite_only_domain_labels() -> None:
    from tasks.task_A.block0.observations import (
        Block0ObservationBundle,
        build_null_tc_im_observations,
    )
    from tasks.task_A.block0.schemas import (
        FIT_LABEL_REAL,
        Block0DomainLabelPermutationAssignment,
    )

    tc_observation = _fov_observation(
        patient_id="P01",
        fov_id="P01_TC_1",
        domain_label="TC",
        variant=0.10,
    )
    im_observation = _fov_observation(
        patient_id="P01",
        fov_id="P01_IM_1",
        domain_label="IM",
        variant=0.05,
    )
    real_bundle = Block0ObservationBundle(
        label=FIT_LABEL_REAL,
        observations=(tc_observation, im_observation),
        patient_ids=("P01",),
    )
    assignments = (
        Block0DomainLabelPermutationAssignment(
            permutation_index=3,
            patient_id="P01",
            fov_id="P01_TC_1",
            original_domain_label="TC",
            permuted_domain_label="IM",
            seed=101,
        ),
        Block0DomainLabelPermutationAssignment(
            permutation_index=3,
            patient_id="P01",
            fov_id="P01_IM_1",
            original_domain_label="IM",
            permuted_domain_label="TC",
            seed=101,
        ),
    )

    null_bundle = build_null_tc_im_observations(
        real_bundle,
        assignments,
        permutation_index=3,
    )

    null_tc_observation, null_im_observation = null_bundle.observations
    assert null_bundle.patient_ids == ("P01",)
    assert null_bundle.permutation_index == 3
    assert null_tc_observation.patient_id == tc_observation.patient_id
    assert null_tc_observation.fov_id == tc_observation.fov_id
    assert null_tc_observation.community_composition is tc_observation.community_composition
    assert null_tc_observation.mass == tc_observation.mass
    assert null_tc_observation.mass_mode == tc_observation.mass_mode
    assert null_tc_observation.timepoint == "IM"
    assert null_tc_observation.domain_label == "IM"
    assert null_tc_observation.metadata["block0_original_domain_label"] == "TC"
    assert null_tc_observation.metadata["block0_permuted_domain_label"] == "IM"
    assert null_im_observation.timepoint == "TC"
    assert null_im_observation.domain_label == "TC"


def test_block0_empirical_tail_p_value_uses_plus_one_correction() -> None:
    from tasks.task_A.block0.metrics import empirical_tail_p_value

    assert empirical_tail_p_value(0.2, [0.1, 0.5, 0.8], tail="right") == pytest.approx(0.75)
    assert empirical_tail_p_value(0.2, [0.1, 0.5, 0.8], tail="left") == pytest.approx(0.5)


def test_block0_tail_effect_and_family_summary_helpers_are_explicit() -> None:
    from tasks.task_A.block0.metrics import effect_ratio, family_summary_values, tail_null_fraction
    from tasks.task_A.block0.schemas import FIT_LABEL_REAL

    record = _block0_fit_record(
        fit_label=FIT_LABEL_REAL,
        permutation_index=None,
        variant=0.0,
    )
    values = family_summary_values(record)

    assert values[("self_retention", "burden_weighted")] == pytest.approx(0.76)
    assert values[("self_retention", "community_mean")] == pytest.approx(0.775)
    assert values[("depletion", "burden_weighted")] == pytest.approx(0.13)
    assert values[("off_diagonal_remodeling", "burden_weighted")] == pytest.approx(0.24)
    assert values[("emergence", "burden_weighted")] == pytest.approx(0.175)
    assert values[("emergence", "community_mean")] == pytest.approx(0.225)
    assert tail_null_fraction(0.2, [0.1, 0.5, 0.8], tail="right") == pytest.approx(2 / 3)
    assert tail_null_fraction(0.2, [0.1, 0.5, 0.8], tail="left") == pytest.approx(1 / 3)
    assert effect_ratio(2.0, 4.0) == (0.5, "estimable")
    assert effect_ratio(2.0, 0.0) == (None, "not_estimable")


def test_block0_rows_require_exact_declared_schema(tmp_path: Path) -> None:
    from stride.errors import ContractError
    from tasks.task_A.block0.schemas import (
        MANIFEST_REQUIRED_FIELDS,
        METRIC_SUMMARY_COLUMNS,
        PATIENT_CALIBRATION_COLUMNS,
        Block0MetricSummaryRow,
        Block0PatientCalibrationRow,
    )
    from tasks.task_A.block0.writers import write_block0_analysis_outputs

    patient_values = {column: None for column in PATIENT_CALIBRATION_COLUMNS}
    assert tuple(Block0PatientCalibrationRow(patient_values).to_csv_row()) == PATIENT_CALIBRATION_COLUMNS

    missing_patient_values = dict(patient_values)
    missing_patient_values.pop("null_reference")
    with pytest.raises(ContractError, match="missing"):
        Block0PatientCalibrationRow(missing_patient_values).to_csv_row()

    metric_values = {column: None for column in METRIC_SUMMARY_COLUMNS}
    assert tuple(Block0MetricSummaryRow(metric_values).to_csv_row()) == METRIC_SUMMARY_COLUMNS

    extra_metric_values = {**metric_values, "unexpected_extra_field": "not allowed"}
    with pytest.raises(ContractError, match="extra"):
        Block0MetricSummaryRow(extra_metric_values).to_csv_row()

    import pandas as pd

    manifest_values = {field: None for field in MANIFEST_REQUIRED_FIELDS}
    with pytest.raises(ContractError, match="extra"):
        write_block0_analysis_outputs(
            output_dir=tmp_path,
            manifest_payload={**manifest_values, "unexpected_extra_field": "not allowed"},
            patient_calibration=pd.DataFrame(columns=PATIENT_CALIBRATION_COLUMNS),
            metric_summary=pd.DataFrame(columns=METRIC_SUMMARY_COLUMNS),
        )


def _block0_fit_record(
    *,
    fit_label: str,
    permutation_index: int | None,
    variant: float,
    patient_id: str = "P01",
):
    from tasks.task_A.block0.schemas import Block0FitRecord

    return Block0FitRecord(
        patient_id=patient_id,
        fit_label=fit_label,
        permutation_index=permutation_index,
        A=np.asarray(
            [
                [0.75 - 0.10 * variant, 0.25 + 0.10 * variant],
                [0.20 + 0.05 * variant, 0.80 - 0.05 * variant],
            ],
            dtype=float,
        ),
        d=np.asarray([0.10 + 0.04 * variant, 0.25 + 0.02 * variant], dtype=float),
        e=np.asarray([0.30 + 0.03 * variant, 0.15 + 0.08 * variant], dtype=float),
        source_burden=np.asarray([4.0, 1.0], dtype=float),
        d_weights=np.asarray([4.0, 1.0], dtype=float),
        e_weights=np.asarray([1.0, 5.0], dtype=float),
    )


def test_block0_calibration_frames_derive_family_summary_long_tables() -> None:
    from tasks.task_A.block0.metrics import (
        build_block0_calibration_frames,
        empirical_tail_p_value,
        family_summary_values,
        tail_null_fraction,
    )
    from tasks.task_A.block0.schemas import (
        CALIBRATION_READY_STATUS,
        FAMILY_SUMMARY_SCALES,
        FIT_LABEL_NULL,
        FIT_LABEL_REAL,
        METRIC_SUMMARY_COLUMNS,
        PATIENT_CALIBRATION_COLUMNS,
        SUMMARY_NAMES,
    )

    real_record = _block0_fit_record(fit_label=FIT_LABEL_REAL, permutation_index=None, variant=0.0)
    real_record_p02 = _block0_fit_record(
        patient_id="P02",
        fit_label=FIT_LABEL_REAL,
        permutation_index=None,
        variant=0.5,
    )
    null_records = tuple(
        _block0_fit_record(fit_label=FIT_LABEL_NULL, permutation_index=index, variant=float(index + 1))
        for index in range(3)
    ) + tuple(
        _block0_fit_record(
            patient_id="P02",
            fit_label=FIT_LABEL_NULL,
            permutation_index=index,
            variant=float(index + 2),
        )
        for index in range(3)
    )

    patient_frame, metric_frame = build_block0_calibration_frames(
        (real_record, real_record_p02),
        null_records,
        run_scope="full_cohort",
        n_permutations=3,
        readiness_status=CALIBRATION_READY_STATUS,
    )

    assert tuple(patient_frame.columns) == PATIENT_CALIBRATION_COLUMNS
    assert tuple(metric_frame.columns) == METRIC_SUMMARY_COLUMNS
    assert len(patient_frame) == 2 * len(SUMMARY_NAMES) * len(FAMILY_SUMMARY_SCALES) * 2
    assert len(metric_frame) == len(SUMMARY_NAMES) * len(FAMILY_SUMMARY_SCALES) * 2
    assert set(patient_frame["summary_name"]) == set(SUMMARY_NAMES)
    assert set(metric_frame["summary_role"]) == {
        "proof_carrying",
        "diagnostic_supportive",
        "supportive",
    }
    assert set(metric_frame["effect_ratio_status"]).issubset({"estimable", "not_estimable"})

    p01_nulls = tuple(record for record in null_records if record.patient_id == "P01")
    p01_real_value = family_summary_values(real_record)[("emergence", "burden_weighted")]
    p01_reference_distribution = np.asarray(
        [
            family_summary_values(null_record)[("emergence", "burden_weighted")]
            for null_record in p01_nulls
        ],
        dtype=float,
    )
    p01_row = patient_frame[
        (patient_frame["patient_id"] == "P01")
        & (patient_frame["summary_name"] == "emergence")
        & (patient_frame["scale"] == "burden_weighted")
        & (patient_frame["reference_stat"] == "median")
    ].iloc[0]
    assert p01_row["expected_tail"] == "right"
    assert p01_row["real_value"] == pytest.approx(p01_real_value)
    assert p01_row["null_reference"] == pytest.approx(float(np.median(p01_reference_distribution)))
    assert p01_row["empirical_p_value"] == pytest.approx(
        empirical_tail_p_value(p01_real_value, p01_reference_distribution, tail="right")
    )
    assert p01_row["primary_tail_fraction"] == pytest.approx(
        tail_null_fraction(p01_real_value, p01_reference_distribution, tail="right")
    )
    assert p01_row["opposite_tail_fraction"] == pytest.approx(
        tail_null_fraction(p01_real_value, p01_reference_distribution, tail="left")
    )

    self_retention_row = patient_frame[
        (patient_frame["patient_id"] == "P01")
        & (patient_frame["summary_name"] == "self_retention")
        & (patient_frame["scale"] == "burden_weighted")
        & (patient_frame["reference_stat"] == "median")
    ].iloc[0]
    assert self_retention_row["expected_tail"] == "left"

    p02_nulls = tuple(record for record in null_records if record.patient_id == "P02")
    cohort_real = float(
        np.median(
            [
                family_summary_values(real_record)[("emergence", "burden_weighted")],
                family_summary_values(real_record_p02)[("emergence", "burden_weighted")],
            ]
        )
    )
    cohort_reference = np.median(
        np.vstack(
            [
                np.asarray(
                    [
                        family_summary_values(record)[("emergence", "burden_weighted")]
                        for record in p01_nulls
                    ],
                    dtype=float,
                ),
                np.asarray(
                    [
                        family_summary_values(record)[("emergence", "burden_weighted")]
                        for record in p02_nulls
                    ],
                    dtype=float,
                ),
            ]
        ),
        axis=0,
    )
    e_median_row = metric_frame[
        (metric_frame["summary_name"] == "emergence")
        & (metric_frame["scale"] == "burden_weighted")
        & (metric_frame["cohort_stat"] == "median")
    ].iloc[0]

    assert e_median_row["expected_tail"] == "right"
    assert e_median_row["real_value"] == pytest.approx(cohort_real)
    assert e_median_row["null_reference"] == pytest.approx(
        float(np.median(cohort_reference))
    )
    assert e_median_row["empirical_p_value"] == pytest.approx(
        empirical_tail_p_value(cohort_real, cohort_reference, tail="right")
    )
    assert e_median_row["primary_tail_fraction"] == pytest.approx(
        tail_null_fraction(cohort_real, cohort_reference, tail="right")
    )
    assert int(e_median_row["n_patient_delta_negative"]) == 2
    assert int(e_median_row["n_patient_delta_positive"]) == 0


def test_block0_fit_records_fail_fast_on_missing_metric_specific_payload() -> None:
    from stride.errors import ContractError
    from tasks.task_A.block0.schemas import FIT_LABEL_REAL, Block0FitRecord

    with pytest.raises(ContractError, match="permutation_index"):
        _block0_fit_record(fit_label=FIT_LABEL_REAL, permutation_index=0, variant=0.0)

    with pytest.raises(ContractError, match="e_weights"):
        Block0FitRecord(
            patient_id="P01",
            fit_label=FIT_LABEL_REAL,
            permutation_index=None,
            A=np.eye(2),
            d=np.ones(2),
            e=np.ones(2),
            source_burden=np.ones(2),
            d_weights=np.ones(2),
            e_weights=None,
        )


def test_block0_real_observations_fail_fast_for_missing_selector_patient() -> None:
    from stride import DatasetHandle
    from stride.errors import ContractError
    from tasks.task_A.block0.observations import build_real_tc_im_observations
    from tasks.task_A.config import load_task_a_config_bundle
    from tests.helpers_task_a_fixture import build_task_a_fixture

    handle = DatasetHandle(adata=build_task_a_fixture())
    config_bundle = load_task_a_config_bundle(ROOT / "tasks" / "task_A" / "config.yaml")

    with pytest.raises(ContractError, match="requested patients"):
        build_real_tc_im_observations(
            handle,
            config_bundle,
            patient_ids=("P99",),
        )


def test_block0_real_observations_fail_fast_for_missing_tc_or_im() -> None:
    from stride import DatasetHandle
    from stride.errors import ContractError
    from tasks.task_A.block0.observations import build_real_tc_im_observations
    from tasks.task_A.config import load_task_a_config_bundle
    from tests.helpers_task_a_fixture import build_task_a_fixture

    adata = build_task_a_fixture()
    keep = ~(
        (adata.obs["patient_id"].astype(str) == "P01")
        & (adata.obs["compartment"].astype(str) == "IM")
    )
    handle = DatasetHandle(adata=adata[keep].copy())
    config_bundle = load_task_a_config_bundle(ROOT / "tasks" / "task_A" / "config.yaml")

    with pytest.raises(ContractError, match="both TC and IM"):
        build_real_tc_im_observations(
            handle,
            config_bundle,
            patient_ids=("P01",),
        )


def test_block0_fit_invokes_canonical_stride_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    from tasks.task_A.block0.fit import fit_block0_family
    from tasks.task_A.block0.observations import Block0ObservationBundle
    from tasks.task_A.block0.schemas import FIT_LABEL_REAL

    real_bundle = Block0ObservationBundle(
        label=FIT_LABEL_REAL,
        observations=(
            _fov_observation(patient_id="P01", fov_id="P01_TC_1", domain_label="TC"),
            _fov_observation(patient_id="P01", fov_id="P01_IM_1", domain_label="IM"),
        ),
        patient_ids=("P01",),
    )
    config_bundle = SimpleNamespace(config_fingerprint="fixture-fingerprint")
    captured: dict[str, object] = {}

    def fake_fit_stride(data, *, state_basis, config):
        captured["data"] = data
        captured["state_basis"] = state_basis
        captured["timepoint_order"] = config.timepoint_order
        captured["metadata"] = dict(config.metadata)
        return "fit-result"

    monkeypatch.setattr("tasks.task_A.block0.fit.fit_stride", fake_fit_stride)

    result = fit_block0_family(
        real_bundle,
        config_bundle=config_bundle,
        state_basis="state-basis",
        fit_label=FIT_LABEL_REAL,
    )

    assert result == "fit-result"
    assert captured["data"] == real_bundle.observations
    assert captured["state_basis"] == "state-basis"
    assert captured["timepoint_order"] == ("TC", "IM")
    assert captured["metadata"]["task_block"] == "block0_calibration"
    assert captured["metadata"]["task_pair_family"] == "TC-IM"


def test_block0_does_not_expose_null_batch_public_api() -> None:
    import tasks.task_A.block0.cli as cli_module
    import tasks.task_A.block0.fit as fit_module

    assert not hasattr(cli_module, "fit_block0_null_batch")
    assert not hasattr(fit_module, "fit_block0_null_batch")
    assert "fit_block0_null_batch" not in fit_module.__all__


def test_block0_writer_enforces_three_formal_outputs_and_strict_columns(tmp_path: Path) -> None:
    import pandas as pd
    from stride.errors import ContractError
    from tasks.task_A.block0.schemas import (
        CALIBRATION_MANIFEST_FILENAME,
        MANIFEST_REQUIRED_FIELDS,
        METRIC_SUMMARY_COLUMNS,
        METRIC_SUMMARY_FILENAME,
        PATIENT_CALIBRATION_COLUMNS,
        PATIENT_CALIBRATION_FILENAME,
    )
    from tasks.task_A.block0.writers import write_block0_analysis_outputs

    manifest = {field: f"value-for-{field}" for field in MANIFEST_REQUIRED_FIELDS}
    patient_frame = pd.DataFrame(columns=PATIENT_CALIBRATION_COLUMNS)
    metric_frame = pd.DataFrame(columns=METRIC_SUMMARY_COLUMNS)

    paths = write_block0_analysis_outputs(
        output_dir=tmp_path,
        manifest_payload=manifest,
        patient_calibration=patient_frame,
        metric_summary=metric_frame,
    )

    assert {path.name for path in paths.values()} == {
        CALIBRATION_MANIFEST_FILENAME,
        PATIENT_CALIBRATION_FILENAME,
        METRIC_SUMMARY_FILENAME,
    }
    assert {path.name for path in tmp_path.iterdir()} == {
        CALIBRATION_MANIFEST_FILENAME,
        PATIENT_CALIBRATION_FILENAME,
        METRIC_SUMMARY_FILENAME,
    }

    bad_output_dir = tmp_path / "bad"
    with pytest.raises(ContractError, match="columns"):
        write_block0_analysis_outputs(
            output_dir=bad_output_dir,
            manifest_payload=manifest,
            patient_calibration=patient_frame.assign(unexpected_extra_field=[]),
            metric_summary=metric_frame,
        )
    assert not (bad_output_dir / CALIBRATION_MANIFEST_FILENAME).exists()
    assert not (bad_output_dir / PATIENT_CALIBRATION_FILENAME).exists()
    assert not (bad_output_dir / METRIC_SUMMARY_FILENAME).exists()


def test_block0_writer_uses_atomic_replace_after_temp_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pandas as pd
    from tasks.task_A.block0.schemas import (
        CALIBRATION_MANIFEST_FILENAME,
        MANIFEST_REQUIRED_FIELDS,
        METRIC_SUMMARY_COLUMNS,
        METRIC_SUMMARY_FILENAME,
        PATIENT_CALIBRATION_COLUMNS,
        PATIENT_CALIBRATION_FILENAME,
    )
    from tasks.task_A.block0.writers import write_block0_analysis_outputs

    manifest = {field: f"value-for-{field}" for field in MANIFEST_REQUIRED_FIELDS}
    patient_frame = pd.DataFrame(columns=PATIENT_CALIBRATION_COLUMNS)
    metric_frame = pd.DataFrame(columns=METRIC_SUMMARY_COLUMNS)
    original_replace = Path.replace
    replace_calls: list[tuple[Path, Path]] = []

    def tracked_replace(self: Path, target: str | Path) -> Path:
        target_path = Path(target)
        replace_calls.append((self, target_path))
        assert self.exists()
        assert self.parent != target_path.parent
        assert self.name == target_path.name
        return original_replace(self, target_path)

    monkeypatch.setattr(Path, "replace", tracked_replace)

    paths = write_block0_analysis_outputs(
        output_dir=tmp_path,
        manifest_payload=manifest,
        patient_calibration=patient_frame,
        metric_summary=metric_frame,
    )

    assert {path.name for path in paths.values()} == {
        CALIBRATION_MANIFEST_FILENAME,
        PATIENT_CALIBRATION_FILENAME,
        METRIC_SUMMARY_FILENAME,
    }
    assert [target.name for _source, target in replace_calls] == [
        PATIENT_CALIBRATION_FILENAME,
        METRIC_SUMMARY_FILENAME,
        CALIBRATION_MANIFEST_FILENAME,
    ]
    assert not any(path.name.endswith(".tmp") or ".tmp" in path.name for path in tmp_path.iterdir())


def test_block0_contract_registry_matches_calibration_outputs() -> None:
    from tasks.task_A.contracts import TASK_A_ARTIFACT_SPEC_BY_NAME, TASK_A_SURFACE_SPEC_BY_NAME

    surface = TASK_A_SURFACE_SPEC_BY_NAME["run_block0_execute"]
    assert surface.owner == "tasks.task_A.block0"
    assert surface.execution_status == "executable"
    assert surface.produces == (
        "block0_execution_manifest.json",
        "block0_fit_cache.npz",
        "block0_fit_cache_index.csv",
    )
    assert "block0_execution_progress.jsonl" not in surface.produces
    assert "Derive calibration metrics or p-values" in surface.does_not_do
    assert "Emit downstream execution decisions" in surface.does_not_do
    analysis_surface = TASK_A_SURFACE_SPEC_BY_NAME["analyze_block0_cache"]
    assert analysis_surface.produces == (
        "block0_calibration_manifest.json",
        "block0_patient_calibration.csv",
        "block0_metric_summary.csv",
    )
    assert "Call fit_stride" in analysis_surface.does_not_do

    execution_manifest = TASK_A_ARTIFACT_SPEC_BY_NAME["block0_execution_manifest"]
    manifest = TASK_A_ARTIFACT_SPEC_BY_NAME["block0_calibration_manifest"]
    cache = TASK_A_ARTIFACT_SPEC_BY_NAME["block0_fit_cache"]
    patient = TASK_A_ARTIFACT_SPEC_BY_NAME["block0_patient_calibration"]
    summary = TASK_A_ARTIFACT_SPEC_BY_NAME["block0_metric_summary"]
    assert "block0_calibration_progress" not in TASK_A_ARTIFACT_SPEC_BY_NAME
    assert execution_manifest.readiness_classification == "calibration_ready_or_diagnostic"
    assert "fit_cache_path" in execution_manifest.minimum_fields
    assert cache.filename == "block0_fit_cache.npz"
    assert manifest.readiness_classification == "calibration_ready_or_diagnostic"
    assert "analysis_spec_version" in manifest.minimum_fields
    assert "summary_roles" in manifest.minimum_fields
    assert patient.artifact_state_location == "block0_calibration_manifest.json.readiness_status"
    assert "summary_name" in patient.minimum_fields
    assert "reference_stat" in patient.minimum_fields
    assert "primary_tail_fraction" in patient.minimum_fields
    assert "effect_ratio_status" in patient.minimum_fields
    assert summary.filename == "block0_metric_summary.csv"
    assert "expected_tail" in summary.minimum_fields
    assert "primary_tail_fraction" in summary.minimum_fields
    assert "effect_ratio_status" in summary.minimum_fields
