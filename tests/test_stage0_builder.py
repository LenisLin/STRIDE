from __future__ import annotations

# ruff: noqa: E402, I001

import importlib.util
import json
import sys
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


def _build_small_stage0_cell_table() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    patients = ["P01", "P02"]
    compartments = ["TC", "IM", "PT"]
    subtypes = ["SubtypeA", "SubtypeB", "SubtypeC", "SubtypeD", "SubtypeE"]
    cell_id = 0

    for patient_idx, patient_id in enumerate(patients):
        for compartment_idx, compartment in enumerate(compartments):
            for roi_num in range(2):
                roi_id = f"{patient_id}_{compartment}_{roi_num + 1:02d}"
                for local_idx in range(8):
                    records.append(
                        {
                            "ID": roi_id,
                            "PID": patient_id,
                            "Tissue": compartment,
                            "SubType": subtypes[(patient_idx + compartment_idx + local_idx) % len(subtypes)],
                            "Area": float(20 + local_idx),
                            "x": float(local_idx),
                            "y": float(compartment_idx * 20 + roi_num * 5 + local_idx / 10.0),
                            "CellID": f"cell_{cell_id}",
                        }
                    )
                    cell_id += 1

    return pd.DataFrame.from_records(records)


def _build_small_expression_bundle(n_cells: int) -> tuple[np.ndarray, list[str]]:
    marker_names = ["M01", "M02", "M03", "M04", "M05"]
    base = np.arange(n_cells, dtype=np.float32)
    expression_matrix = np.column_stack(
        [
            base / max(n_cells - 1, 1),
            ((base % 7) / 6.0).astype(np.float32),
            ((base % 11) / 10.0).astype(np.float32),
            ((base % 13) / 12.0).astype(np.float32),
            ((base % 17) / 16.0).astype(np.float32),
        ]
    ).astype(np.float32)
    return expression_matrix, marker_names


def _write_task_a_config(path: Path, *, enabled_blocks: list[str]) -> Path:
    config = {
        "task_name": "Task A Stage 0 smoke",
        "enabled_blocks": enabled_blocks,
        "data": {
            "mass_mode": "uniform",
            "k_full": 25,
        },
        "observation_match": {
            "eps_schedule": [1.0, 0.5, 0.1],
            "max_iter": 400,
            "tol": 1.0e-8,
            "eta_floor": 1.0e-12,
            "n_min_proto": 0.0,
        },
        "block0": {
            "n_draws": 3,
            "random_seed": 11,
            "match_penalty_by_compartment": {"TC": 1.0, "IM": 1.5, "PT": 2.0},
            "retention_threshold_by_compartment": {"TC": 0.4, "IM": 0.6, "PT": 0.8},
        },
        "block1": {
            "target_alpha": 0.05,
            "lambda_grid": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_stage0_h5ad_meets_taska_minimum_contract(tmp_path: Path) -> None:
    import anndata as ad
    from tasks.task_A.stage0.build_artifacts import (
        build_stage0_adata_from_cell_table,
        validate_stage0_minimum_contract,
    )

    cell_table = _build_small_stage0_cell_table()
    expression_matrix, marker_names = _build_small_expression_bundle(cell_table.shape[0])
    adata = build_stage0_adata_from_cell_table(
        cell_table,
        expression_matrix=expression_matrix,
        marker_names=marker_names,
        k=25,
        knn=20,
        n_bal=8,
    )
    h5ad_path = tmp_path / "stage0_small.h5ad"
    adata.write_h5ad(h5ad_path)

    reloaded = ad.read_h5ad(h5ad_path)
    validate_stage0_minimum_contract(reloaded)
    assert set(reloaded.obs["compartment"].astype(str)) == {"TC", "IM", "PT"}
    assert np.asarray(reloaded.uns["cost_matrix"], dtype=float).shape == (25, 25)
    np.testing.assert_allclose(np.asarray(reloaded.X), expression_matrix)


def test_stage0_pre_block0_suitability_prepare_and_block1_bundle(tmp_path: Path) -> None:
    from tasks.task_A.stage0.build_artifacts import build_stage0_adata_from_cell_table
    from tasks.task_A.workflows.check_data_suitability import check_task_a_pre_block0_data_suitability
    from tasks.task_A.workflows.run_block1 import run_block1_workflow
    from tasks.task_A.workflows.prepare import prepare_task_a_stage0_mapping
    from tests.helpers_task_a_fixture import write_passed_block0_bundle

    cell_table = _build_small_stage0_cell_table()
    expression_matrix, marker_names = _build_small_expression_bundle(cell_table.shape[0])
    adata = build_stage0_adata_from_cell_table(
        cell_table,
        expression_matrix=expression_matrix,
        marker_names=marker_names,
        k=25,
        knn=20,
        n_bal=8,
    )
    h5ad_path = tmp_path / "stage0_pipeline.h5ad"
    adata.write_h5ad(h5ad_path)

    config_path = _write_task_a_config(
        tmp_path / "task_a_tc.yaml",
        enabled_blocks=["block0_locality_gate", "block1_continuity_backbone"],
    )

    prepare_manifest = prepare_task_a_stage0_mapping(
        config_path=str(config_path),
        data_path=str(h5ad_path),
        output_dir=str(tmp_path / "prepare"),
        patient_ids=("P01",),
    )
    suitability_report = check_task_a_pre_block0_data_suitability(
        config_path=str(config_path),
        data_path=str(h5ad_path),
        output_dir=str(tmp_path / "suitability"),
    )
    block0_bundle = write_passed_block0_bundle(
        tmp_path / "block0" / "block0_bundle.json",
        config_path=config_path,
        data_path=h5ad_path,
    )
    block1_bundle = run_block1_workflow(
        config_path=str(config_path),
        data_path=str(h5ad_path),
        block0_bundle=str(block0_bundle),
        output_dir=str(tmp_path / "block1"),
    )

    assert Path(prepare_manifest["mapping_manifest"]).exists()
    assert prepare_manifest["run_scope"] == "patient_subset"
    assert prepare_manifest["artifact_state"] == "scaffold_active"
    assert Path(suitability_report).exists()
    suitability_payload = json.loads(Path(suitability_report).read_text(encoding="utf-8"))
    assert suitability_payload["artifact_state"] == "contract_passed"
    assert Path(block0_bundle).exists()
    assert Path(block1_bundle).exists()


def test_stage0_build_artifacts_does_not_persist_roi_clinical_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import anndata as ad

    from tasks.task_A.stage0 import build_artifacts as stage0_build

    cell_table = pd.DataFrame.from_records(
        [
            {"ID": "R1", "PID": "P1", "Tissue": "TC", "SubType": "A", "Area": 10.0, "x": 0.0, "y": 0.0},
            {"ID": "R1", "PID": "P1", "Tissue": "TC", "SubType": "B", "Area": 11.0, "x": 1.0, "y": 1.0},
            {"ID": "R2", "PID": "P1", "Tissue": "IM", "SubType": "A", "Area": 12.0, "x": 2.0, "y": 2.0},
            {"ID": "R2", "PID": "P1", "Tissue": "IM", "SubType": "C", "Area": 13.0, "x": 3.0, "y": 3.0},
        ]
    )
    roi_clinical_table = pd.DataFrame.from_records(
        [
            {"ID": "R1", "PID": "P1", "Tissue": "TC", "Batch": "B1"},
            {"ID": "R2", "PID": "P1", "Tissue": "IM", "Batch": "B1"},
        ]
    )

    monkeypatch.setattr(stage0_build, "run_r_extraction", lambda *args, **kwargs: None)
    monkeypatch.setattr(stage0_build, "load_extracted_tables", lambda **kwargs: (cell_table, roi_clinical_table))
    monkeypatch.setattr(stage0_build, "load_marker_names", lambda *_args, **_kwargs: ["M01", "M02"])
    monkeypatch.setattr(
        stage0_build,
        "load_expression_matrix",
        lambda *_args, **_kwargs: np.asarray(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.5, 0.5],
                [0.25, 0.75],
            ],
            dtype=np.float32,
        ),
    )

    def _forbid_to_parquet(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("roi_clinical parquet persistence is forbidden")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _forbid_to_parquet, raising=True)

    outputs = stage0_build.build_stage0_artifacts(
        rds_path="dummy.rds",
        output_dir=tmp_path / "stage0",
        k=2,
        knn=1,
        n_bal=2,
        require_all_proto_ids=False,
    )

    assert set(outputs) == {"h5ad_path", "validation_path"}
    assert outputs["h5ad_path"].exists()
    assert outputs["validation_path"].exists()
    assert not (tmp_path / "stage0" / "task_A_stage0_roi_clinical.parquet").exists()
