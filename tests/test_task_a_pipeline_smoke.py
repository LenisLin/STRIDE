from __future__ import annotations

# ruff: noqa: E402, I001

import importlib
import importlib.util
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
    from tasks.task_A.pipeline import TEMPORARY_METRICS_FILENAME, main
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


def test_pipeline_smoke_runs_constrained_and_broken_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tasks.task_A.pipeline import TEMPORARY_METRICS_FILENAME, main
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


def test_pipeline_smoke_runs_arm2_startup_slice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tasks.task_A.pipeline import TEMPORARY_METRICS_FILENAME, main
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
        pipeline,
        "ad",
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
