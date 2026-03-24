from __future__ import annotations

# ruff: noqa: E402, I001

import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ANNDATA_AVAILABLE = importlib.util.find_spec("anndata") is not None
pytestmark = pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not installed")


def _write_config(path: Path, *, enabled_arms: list[str]) -> Path:
    from tests.helpers_task_a_fixture import K_FULL

    config = {
        "task_name": "Task A Patch-2 smoke test",
        "enabled_arms": enabled_arms,
        "data": {"mass_mode": "density", "k_full": K_FULL},
        "uot_params": {
            "eps_schedule": [1.0, 0.5, 0.1],
            "max_iter": 4000,
            "tol": 1.0e-8,
            "eta_floor": 1.0e-12,
            "n_min_proto": 0.0,
            "tau_mode": "external_fixed_by_task",
        },
        "arm1": {
            "n_draws": 2,
            "random_seed": 7,
            "fixed_lambda_by_compartment": {"TC": 1.0, "IM": 1.5, "PT": 2.0},
            "fixed_tau_by_compartment": {"TC": 0.5, "IM": 0.75, "PT": 1.0},
        },
        "arm2": {
            "target_alpha": 0.05,
            "lambda_grid": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_pipeline_rejects_unsupported_arm_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tasks.task_A.pipeline import main

    config_path = _write_config(tmp_path / "config.yaml", enabled_arms=["A9_unknown"])
    attempted_imports: list[str] = []

    def fail_if_called(name: str):
        attempted_imports.append(name)
        raise AssertionError("Deferred arm imports should not run for unsupported-arm rejection")

    monkeypatch.setattr("tasks.task_A.pipeline.import_module", fail_if_called)

    with pytest.raises(NotImplementedError, match="Patch-2 only supports"):
        main(str(config_path), str(tmp_path / "missing_fixture.h5ad"), str(tmp_path / "out"))

    assert attempted_imports == []


def test_pipeline_smoke_runs_arm1_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tasks.task_A.pipeline import TASK_A_MANIFEST_FILENAME, TEMPORARY_METRICS_FILENAME, main
    from tests.helpers_task_a_fixture import write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(tmp_path / "config.yaml", enabled_arms=["A1_baseline"])
    output_dir = tmp_path / "outputs"
    imported_modules: list[str] = []

    def tracking_import(name: str):
        imported_modules.append(name)
        return importlib.import_module(name)

    monkeypatch.setattr("tasks.task_A.pipeline.import_module", tracking_import)

    df_metrics = main(str(config_path), str(fixture_path), str(output_dir))

    assert imported_modules == ["tasks.task_A.arm1_noise_baseline"]
    assert not df_metrics.empty
    assert df_metrics.shape[0] == 12
    assert set(df_metrics["arm"].astype(str)) == {"A1_baseline"}
    assert df_metrics["patient_group_id"].is_unique
    assert (df_metrics["roi_a"] != df_metrics["roi_b"]).all()
    assert (df_metrics["lambda_mode"] == "task_fixed_by_compartment").all()
    assert (df_metrics["tau_mode"] == "task_fixed_by_compartment").all()

    ok_mask = df_metrics["uot_status"] == "ok"
    if ok_mask.any():
        np.testing.assert_allclose(
            df_metrics.loc[ok_mask, "U"].to_numpy(dtype=float),
            df_metrics.loc[ok_mask, "D_pos"].to_numpy(dtype=float)
            + df_metrics.loc[ok_mask, "B_pos"].to_numpy(dtype=float),
            rtol=1e-8,
            atol=1e-10,
        )

    assert (output_dir / TEMPORARY_METRICS_FILENAME).exists()
    manifest = json.loads((output_dir / TASK_A_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["metrics_parquet"] == str((output_dir / TEMPORARY_METRICS_FILENAME).resolve())


def test_pipeline_smoke_runs_constrained_and_broken_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tasks.task_A.pipeline import TASK_A_MANIFEST_FILENAME, TEMPORARY_METRICS_FILENAME, main
    from tests.helpers_task_a_fixture import write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(
        tmp_path / "config.yaml",
        enabled_arms=["A1_baseline", "A1_broken_reference"],
    )
    output_dir = tmp_path / "outputs"
    imported_modules: list[str] = []

    def tracking_import(name: str):
        imported_modules.append(name)
        return importlib.import_module(name)

    monkeypatch.setattr("tasks.task_A.pipeline.import_module", tracking_import)

    df_metrics = main(str(config_path), str(fixture_path), str(output_dir))

    assert imported_modules == [
        "tasks.task_A.arm1_noise_baseline",
        "tasks.task_A.arm1_broken_reference",
    ]
    assert not df_metrics.empty
    assert df_metrics.shape[0] == 24
    assert set(df_metrics["arm"].astype(str)) == {"A1_baseline", "A1_broken_reference"}
    assert df_metrics["patient_group_id"].is_unique
    assert (df_metrics["patient_id"] == df_metrics["patient_id_a"]).all()
    assert (df_metrics["compartment"] == df_metrics["compartment_a"]).all()
    assert df_metrics.loc[df_metrics["arm"] == "A1_baseline", "same_patient"].all()
    assert df_metrics.loc[df_metrics["arm"] == "A1_baseline", "same_compartment"].all()
    assert (~df_metrics.loc[df_metrics["arm"] == "A1_broken_reference", "same_patient"]).all()
    assert (~df_metrics.loc[df_metrics["arm"] == "A1_broken_reference", "same_compartment"]).all()
    assert (output_dir / TEMPORARY_METRICS_FILENAME).exists()
    manifest = json.loads((output_dir / TASK_A_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["metrics_parquet"] == str((output_dir / TEMPORARY_METRICS_FILENAME).resolve())


def test_pipeline_smoke_runs_arm2_startup_slice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tasks.task_A.pipeline import TASK_A_MANIFEST_FILENAME, TEMPORARY_METRICS_FILENAME, main
    from tests.helpers_task_a_fixture import write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(tmp_path / "config.yaml", enabled_arms=["A2_cross_compartment"])
    output_dir = tmp_path / "outputs"
    imported_modules: list[str] = []

    def tracking_import(name: str):
        imported_modules.append(name)
        return importlib.import_module(name)

    monkeypatch.setattr("tasks.task_A.pipeline.import_module", tracking_import)

    df_metrics = main(str(config_path), str(fixture_path), str(output_dir))

    assert imported_modules == ["tasks.task_A.arm2_spatial_gradient"]
    assert not df_metrics.empty
    assert df_metrics.shape[0] == 48
    assert set(df_metrics["arm"].astype(str)) == {"A2_cross_compartment"}
    assert df_metrics["patient_group_id"].is_unique
    assert (df_metrics["patient_id"] == df_metrics["patient_id_a"]).all()
    assert (df_metrics["compartment"] == df_metrics["compartment_a"]).all()
    assert df_metrics["same_patient"].all()
    assert (~df_metrics["same_compartment"]).all()
    assert (df_metrics["patient_id_a"] == df_metrics["patient_id_b"]).all()
    assert (df_metrics["compartment_a"] != df_metrics["compartment_b"]).all()
    assert set(df_metrics["pair_family"].astype(str)) == {"TC-IM", "IM-PT", "TC-PT"}
    assert set(df_metrics["pair_type"].astype(str)) == {
        "TC->IM",
        "IM->TC",
        "IM->PT",
        "PT->IM",
        "TC->PT",
        "PT->TC",
    }
    assert (df_metrics["mass_mode"] == "density").all()
    assert (df_metrics["lambda_mode"] == "pair_specific_joint").all()
    assert (df_metrics["tau_mode"] == "unavailable").all()
    assert df_metrics["tau"].isna().all()
    assert df_metrics["R"].isna().all()

    ok_mask = df_metrics["uot_status"] == "ok"
    if ok_mask.any():
        assert np.isfinite(df_metrics.loc[ok_mask, "M_balanced"].to_numpy(dtype=float)).all()

    assert (output_dir / TEMPORARY_METRICS_FILENAME).exists()
    manifest = json.loads((output_dir / TASK_A_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["metrics_parquet"] == str((output_dir / TEMPORARY_METRICS_FILENAME).resolve())


def test_pipeline_routes_arm3_through_shared_full_coverage_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tasks.task_A.pipeline as pipeline

    config_path = _write_config(tmp_path / "config.yaml", enabled_arms=["A3_uq_stress"])
    output_dir = tmp_path / "outputs"
    imported_modules: list[str] = []
    captured: dict[str, object] = {}

    def fake_run_arm3(stage0_path: str, config: dict, result_root: str) -> pd.DataFrame:
        captured["stage0_path"] = stage0_path
        captured["result_root"] = result_root
        captured["enabled_arms"] = list(config["enabled_arms"])
        return pd.DataFrame(
            [
                {
                    "patient_group_id": "A3_uq_stress::TC->IM::P01::P01_TC_01::P01_IM_01",
                    "pair_id": "A3_uq_stress::TC->IM::P01::P01_TC_01::P01_IM_01",
                    "arm": "A3_uq_stress",
                    "patient_id": "P01",
                    "compartment": "TC",
                    "patient_id_a": "P01",
                    "patient_id_b": "P01",
                    "compartment_a": "TC",
                    "compartment_b": "IM",
                    "same_patient": True,
                    "same_compartment": False,
                    "pair_type": "TC->IM",
                    "roi_a": "P01_TC_01",
                    "roi_b": "P01_IM_01",
                    "lambda_pl": 10.0,
                    "lambda_mode": "pair_specific_joint",
                    "tau_mode": "task_fixed_by_compartment",
                    "mass_mode": "density",
                    "uot_status": "ok",
                    "bypass_reason": None,
                    "mass_pruned_ratio": 0.1,
                    "n_min_proto_used": 0.0,
                    "S0": 2.0,
                    "S1": 1.5,
                    "scale_ratio": 0.75,
                    "U": 0.3,
                    "T": 1.0,
                    "D_pos": 0.1,
                    "B_pos": 0.2,
                    "d_rel": 0.05,
                    "b_rel": 0.1,
                    "M": 0.2,
                    "R": 0.8,
                    "tau": 0.25,
                    "pair_family": "TC-IM",
                    "lambda_dens": 10.0,
                    "U_abs_dens": 0.3,
                    "S_src": 2.0,
                    "S_tgt": 1.5,
                    "Delta_scale": -0.5,
                    "Q_src_dens": 0.5,
                    "Q_tgt_dens": 2.0 / 3.0,
                }
            ]
        )

    def tracking_import(name: str):
        imported_modules.append(name)
        if name != "tasks.task_A.arm3_uq_stress":
            raise AssertionError(f"unexpected import {name}")
        return types.SimpleNamespace(run_arm3=fake_run_arm3)

    monkeypatch.setattr("tasks.task_A.pipeline.import_module", tracking_import)
    monkeypatch.setattr(
        "tasks.task_A.runtime_contract.ad",
        types.SimpleNamespace(
            read_h5ad=lambda path: (_ for _ in ()).throw(AssertionError(f"unexpected AnnData load {path}"))
        ),
    )

    df_metrics = pipeline.main(str(config_path), str(tmp_path / "fixture.h5ad"), str(output_dir))

    assert imported_modules == ["tasks.task_A.arm3_uq_stress"]
    assert captured["stage0_path"] == str(tmp_path / "fixture.h5ad")
    assert captured["result_root"] == str(output_dir)
    assert captured["enabled_arms"] == ["A3_uq_stress"]
    assert set(df_metrics["arm"].astype(str)) == {"A3_uq_stress"}
    assert set(df_metrics["pair_type"].astype(str)) == {"TC->IM"}
    assert set(df_metrics["pair_family"].astype(str)) == {"TC-IM"}
    assert (df_metrics["tau_mode"] == "task_fixed_by_compartment").all()
    assert (output_dir / pipeline.TEMPORARY_METRICS_FILENAME).exists()
    manifest = json.loads((output_dir / pipeline.TASK_A_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["arm_artifact_roots"]["A3_uq_stress"] == str(output_dir.resolve())


def test_pipeline_builds_shared_roi_references_once_for_multi_arm_density_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tasks.task_A.runtime_contract as runtime_contract
    from tasks.task_A.pipeline import main
    from tests.helpers_task_a_fixture import write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(
        tmp_path / "config.yaml",
        enabled_arms=["A1_baseline", "A2_cross_compartment"],
    )
    output_dir = tmp_path / "outputs"

    original_builder = runtime_contract.build_task_a_roi_reference_bundle_from_adata
    call_count = {"n": 0}

    def tracking_builder(adata, *, k_full: int):
        call_count["n"] += 1
        return original_builder(adata, k_full=k_full)

    monkeypatch.setattr(runtime_contract, "build_task_a_roi_reference_bundle_from_adata", tracking_builder)

    df_metrics = main(str(config_path), str(fixture_path), str(output_dir))

    assert call_count["n"] == 1
    assert set(df_metrics["arm"].astype(str)) == {"A1_baseline", "A2_cross_compartment"}


# ---------------------------------------------------------------------------
# Arm3 calibration + bootstrap chain smoke tests
# ---------------------------------------------------------------------------


def _build_arm3_smoke_fixture(k_full: int = 25) -> tuple[dict, dict, dict, dict, pd.DataFrame]:
    """
    Build a minimal synthetic Arm3 fixture with all three pair families:
      - 12 ROIs (2 patients × 3 compartments × 2 ROIs each)
      - 10 blocks per ROI
    Returns roi_block_summary, roi_density_vectors, roi_compartment_map,
    roi_patient_map, pair_meta.
    """
    rng = np.random.default_rng(99)
    count_cols = [f"count_k{k}" for k in range(k_full)]
    specs = [
        ("P01", "TC", "P01_TC_01"), ("P01", "TC", "P01_TC_02"),
        ("P01", "IM", "P01_IM_01"), ("P01", "IM", "P01_IM_02"),
        ("P01", "PT", "P01_PT_01"), ("P01", "PT", "P01_PT_02"),
        ("P02", "TC", "P02_TC_01"), ("P02", "TC", "P02_TC_02"),
        ("P02", "IM", "P02_IM_01"), ("P02", "IM", "P02_IM_02"),
        ("P02", "PT", "P02_PT_01"), ("P02", "PT", "P02_PT_02"),
    ]
    n_blocks = 10
    roi_block_summary: dict[str, pd.DataFrame] = {}
    roi_density_vectors: dict[str, np.ndarray] = {}
    roi_compartment_map: dict[str, str] = {}
    roi_patient_map: dict[str, str] = {}

    for patient_id, compartment, roi_id in specs:
        roi_compartment_map[roi_id] = compartment
        roi_patient_map[roi_id] = patient_id
        block_ids = [f"{roi_id}_b{b}" for b in range(n_blocks)]
        areas = rng.uniform(0.01, 0.05, size=n_blocks)
        counts = rng.integers(1, 15, size=(n_blocks, k_full)).astype(float)
        df = pd.DataFrame({"block_id": block_ids, "block_area_mm2": areas})
        for k, col in enumerate(count_cols):
            df[col] = counts[:, k]
        roi_block_summary[roi_id] = df
        roi_density_vectors[roi_id] = counts.sum(axis=0) / float(areas.sum())

    direction_to_family = {
        ("TC", "IM"): "TC-IM", ("IM", "TC"): "TC-IM",
        ("IM", "PT"): "IM-PT", ("PT", "IM"): "IM-PT",
        ("TC", "PT"): "TC-PT", ("PT", "TC"): "TC-PT",
    }
    pair_records = []
    for patient_id in ("P01", "P02"):
        comp_rois: dict[str, list[str]] = {}
        for pid, comp, roi_id in specs:
            if pid == patient_id:
                comp_rois.setdefault(comp, []).append(roi_id)
        for (ca, cb), family in direction_to_family.items():
            for roi_a in comp_rois.get(ca, []):
                for roi_b in comp_rois.get(cb, []):
                    pair_records.append({
                        "pair_id": f"smoke::{patient_id}::{roi_a}::{roi_b}",
                        "patient_id": patient_id,
                        "roi_a": roi_a,
                        "roi_b": roi_b,
                        "pair_family": family,
                        "compartment_a": ca,
                        "compartment_b": cb,
                    })
    pair_meta = pd.DataFrame.from_records(pair_records)
    return roi_block_summary, roi_density_vectors, roi_compartment_map, roi_patient_map, pair_meta


def test_pipeline_smoke_calibrate_arm3_lambda_and_tau() -> None:
    """Arm3 Phase 4: calibrate_lambda_dens + calibrate_tau_by_compartment return valid dicts."""
    from slotar.uot import UOTSolveConfig, precompute_logKernels
    from tasks.task_A.arm3.calibrate import calibrate_lambda_dens, calibrate_tau_by_compartment

    k_full = 25
    eps_schedule = [1.0, 0.5]
    lambda_grid = (0.1, 0.5, 1.0, 2.0)
    C = np.abs(np.arange(k_full, dtype=float)[:, None] - np.arange(k_full, dtype=float)[None, :])
    kernels = precompute_logKernels(C, eps_schedule, s_C=1.0)
    cfg = UOTSolveConfig(eps_schedule=eps_schedule, max_iter=200, tol=1e-5)

    roi_block_summary, roi_density_vectors, roi_compartment_map, roi_patient_map, pair_meta = (
        _build_arm3_smoke_fixture(k_full)
    )

    # Phase 4a: calibrate_lambda_dens across all three families
    lambda_dens = calibrate_lambda_dens(
        roi_density_vectors=roi_density_vectors,
        pair_meta=pair_meta,
        k_full=k_full,
        lambda_grid=lambda_grid,
        uot_cfg=cfg,
        kernels=kernels,
        target_alpha=0.05,
    )
    assert set(lambda_dens.keys()) == {"TC-IM", "IM-PT", "TC-PT"}
    for fam, val in lambda_dens.items():
        assert np.isfinite(val), f"lambda_dens[{fam!r}] is not finite: {val}"
        assert val > 0.0, f"lambda_dens[{fam!r}] is not positive: {val}"

    # Phase 4b: calibrate_tau_by_compartment (each patient has 2 ROIs per compartment
    # so within-patient same-compartment pairs are available for TC, IM, PT).
    tau_by_compartment = calibrate_tau_by_compartment(
        roi_density_vectors=roi_density_vectors,
        roi_compartment_map=roi_compartment_map,
        roi_patient_map=roi_patient_map,
        k_full=k_full,
        scaled_cost_matrix=C,
        frozen_lambdas=lambda_dens,
        uot_cfg=cfg,
        kernels=kernels,
    )
    assert set(tau_by_compartment.keys()) == {"TC", "IM", "PT"}
    for comp, val in tau_by_compartment.items():
        assert np.isfinite(val), f"tau[{comp!r}] is not finite: {val}"
        assert val > 0.0, f"tau[{comp!r}] is not positive: {val}"


def test_pipeline_smoke_arm3_bootstrap_pass() -> None:
    """Arm3 Phase 5: run_bootstrap_pass returns correct shapes and audit frame."""
    from tasks.task_A.arm3.pseudo_roi import run_bootstrap_pass

    k_full = 25
    n_reps = 5
    coverage = 0.75

    roi_block_summary, _, _, _, pair_meta = _build_arm3_smoke_fixture(k_full)
    n_pairs = len(pair_meta)

    A_reps, B_reps, pseudo_meta = run_bootstrap_pass(
        roi_block_summary=roi_block_summary,
        pair_meta=pair_meta,
        coverage=coverage,
        n_reps=n_reps,
        k_full=k_full,
        rng_seed=7,
    )

    assert A_reps.shape == (n_reps, n_pairs, k_full)
    assert B_reps.shape == (n_reps, n_pairs, k_full)
    assert np.all(np.isfinite(A_reps))
    assert np.all(np.isfinite(B_reps))
    assert np.all(A_reps >= 0.0)
    assert np.all(B_reps >= 0.0)
    assert len(pseudo_meta) == n_reps * n_pairs
    assert set(pseudo_meta.columns) >= {
        "replicate_id", "pair_id", "coverage",
        "pseudo_area_a_mm2", "pseudo_area_b_mm2",
        "n_blocks_sampled_a", "n_blocks_sampled_b",
    }
    assert (pseudo_meta["coverage"] == coverage).all()


def test_pipeline_smoke_arm2_batched_solve() -> None:
    """Arm2 smoke: batched_uot_solve on a 12-pair fixture returns correct shapes and status."""
    from slotar.uot import UOTSolveConfig, batched_uot_solve, precompute_logKernels
    from tests.helpers_task_a_fixture import expected_arm2_pair_records, expected_roi_vectors

    k_full = 25
    eps_schedule = [1.0, 0.5, 0.1]
    C = np.abs(np.arange(k_full, dtype=float)[:, None] - np.arange(k_full, dtype=float)[None, :])
    kernels = precompute_logKernels(C, eps_schedule, s_C=1.0)
    cfg = UOTSolveConfig(eps_schedule=eps_schedule, max_iter=300, tol=1e-6)

    roi_vectors = expected_roi_vectors(k_full)
    pair_records = expected_arm2_pair_records()
    pair_meta = pd.DataFrame.from_records(pair_records)
    n = len(pair_meta)

    A = np.stack([roi_vectors[str(r)] for r in pair_meta["roi_a"]], axis=0).astype(float)
    B = np.stack([roi_vectors[str(r)] for r in pair_meta["roi_b"]], axis=0).astype(float)
    lambda_pl = np.full(n, 1.0, dtype=float)

    metrics, details, status = batched_uot_solve(
        A=A, B=B, lambda_pl=lambda_pl, kernels=kernels, cfg=cfg
    )

    assert status.shape == (n,)
    for key in ("T", "D_pos", "B_pos", "d_rel", "b_rel", "M"):
        assert metrics[key].shape == (n,)
    ok_mask = status == "ok"
    assert ok_mask.any(), "Expected at least some successful solves on the Arm2 fixture"
    assert np.all(metrics["T"][ok_mask] > 0.0)
    assert np.all(np.isfinite(metrics["M"][ok_mask]))
