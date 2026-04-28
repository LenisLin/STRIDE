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


def _build_mock_cell_table() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    cell_types = ["A", "B", "C", "D", "E"]
    roi_specs = [
        ("P01", "TC", "P01_TC_01", 0.0),
        ("P01", "TC", "P01_TC_02", 10.0),
        ("P01", "IM", "P01_IM_01", 20.0),
        ("P01", "IM", "P01_IM_02", 30.0),
        ("P01", "PT", "P01_PT_01", 40.0),
        ("P01", "PT", "P01_PT_02", 50.0),
        ("P02", "TC", "P02_TC_01", 60.0),
        ("P02", "TC", "P02_TC_02", 70.0),
        ("P02", "IM", "P02_IM_01", 80.0),
        ("P02", "IM", "P02_IM_02", 90.0),
        ("P02", "PT", "P02_PT_01", 100.0),
        ("P02", "PT", "P02_PT_02", 110.0),
    ]
    for patient_id, compartment, roi_id, base_x in roi_specs:
        for cell_idx in range(6):
            records.append(
                {
                    "ID": roi_id,
                    "PID": patient_id,
                    "Tissue": compartment,
                    "SubType": cell_types[(cell_idx + int(base_x // 10)) % len(cell_types)],
                    "Area": float(50 + cell_idx),
                    "x": base_x + float(cell_idx),
                    "y": float((cell_idx * 3) + (int(base_x) % 7)),
                    "CellID": f"{roi_id}_cell_{cell_idx}",
                }
            )
    return pd.DataFrame.from_records(records)


def _build_mock_expression_bundle(n_cells: int) -> tuple[np.ndarray, list[str]]:
    marker_names = ["CLEC9A", "CD169", "CD14", "CD8a"]
    base = np.arange(n_cells, dtype=np.float32)
    expression_matrix = np.column_stack(
        [
            base / max(n_cells - 1, 1),
            ((base % 5) / 4.0).astype(np.float32),
            ((base % 7) / 6.0).astype(np.float32),
            ((base % 9) / 8.0).astype(np.float32),
        ]
    ).astype(np.float32)
    return expression_matrix, marker_names


def _write_task_a_config(path: Path, *, k_full: int) -> Path:
    config = {
        "task_name": "Task A stage0 builder test",
        "enabled_blocks": ["block0_locality_gate", "block1_continuity_backbone"],
        "data": {"mass_mode": "uniform", "k_full": k_full},
        "observation_match": {
            "eps_schedule": [1.0, 0.5],
            "max_iter": 100,
            "tol": 1.0e-8,
            "eta_floor": 1.0e-12,
            "n_min_proto": 0.0,
        },
        "block1": {
            "target_alpha": 0.05,
            "lambda_grid": [0.05, 0.1, 0.5, 1.0],
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_stage0_builder_preserves_stride_and_source_alias_contracts(tmp_path: Path) -> None:
    import anndata as ad

    from tasks.task_A.stage0.build_artifacts import (
        build_stage0_adata_from_cell_table,
        validate_representation_completeness,
        validate_stage0_minimum_contract,
    )
    from tasks.task_A.workflows.stride_adapter import load_task_a_dataset_handle, resolve_task_a_state_basis

    cell_table = _build_mock_cell_table()
    expression_matrix, marker_names = _build_mock_expression_bundle(cell_table.shape[0])
    adata = build_stage0_adata_from_cell_table(
        cell_table,
        expression_matrix=expression_matrix,
        marker_names=marker_names,
        k=6,
        knn=3,
        n_bal=6,
        random_state=0,
    )

    out_path = tmp_path / "stage0_fixture.h5ad"
    adata.write_h5ad(out_path)
    reloaded = ad.read_h5ad(out_path)

    validate_stage0_minimum_contract(reloaded)
    validate_representation_completeness(reloaded)
    np.testing.assert_allclose(np.asarray(reloaded.X), expression_matrix)
    assert set(("fov_id", "domain_label", "cell_subtype_label", "roi_id", "compartment", "cell_type")).issubset(
        reloaded.obs.columns
    )
    assert reloaded.obs["roi_id"].astype(str).equals(reloaded.obs["fov_id"].astype(str))
    assert reloaded.obs["compartment"].astype(str).equals(reloaded.obs["domain_label"].astype(str))
    assert reloaded.obs["cell_type"].astype(str).equals(reloaded.obs["cell_subtype_label"].astype(str))
    assert np.asarray(reloaded.obsm["community_features"]).shape[0] == reloaded.n_obs
    assert np.asarray(reloaded.uns["prototype_centroids"]).shape == (6, reloaded.obsm["community_features"].shape[1])
    assert np.asarray(reloaded.uns["cost_matrix"]).shape == (6, 6)
    proto_ids = reloaded.obs["proto_id"].astype(int).to_numpy()
    assert np.array_equal(np.unique(proto_ids), np.arange(6))
    assert float(reloaded.uns["s_C"]) > 0.0

    handle = load_task_a_dataset_handle(reloaded)
    state_basis = resolve_task_a_state_basis(handle)
    assert state_basis.n_states == 6


def test_stage0_prepare_workflow_accepts_rewritten_builder_output(tmp_path: Path) -> None:
    from tasks.task_A.workflows.prepare import prepare_task_a_stage0_mapping
    from tasks.task_A.stage0.build_artifacts import build_stage0_adata_from_cell_table

    cell_table = _build_mock_cell_table()
    expression_matrix, marker_names = _build_mock_expression_bundle(cell_table.shape[0])
    adata = build_stage0_adata_from_cell_table(
        cell_table,
        expression_matrix=expression_matrix,
        marker_names=marker_names,
        k=6,
        knn=3,
        n_bal=6,
        random_state=0,
    )
    stage0_path = tmp_path / "stage0.h5ad"
    adata.write_h5ad(stage0_path)
    config_path = _write_task_a_config(tmp_path / "task_a.yaml", k_full=6)

    manifest = prepare_task_a_stage0_mapping(
        config_path=config_path,
        data_path=stage0_path,
        output_dir=tmp_path / "prepare",
        patient_ids=("P01",),
    )

    mapping_payload = json.loads(Path(manifest["mapping_manifest"]).read_text(encoding="utf-8"))
    assert mapping_payload["field_mapping"]["fov_key"] == "fov_id"
    assert mapping_payload["field_mapping"]["domain_key"] == "domain_label"
    assert mapping_payload["field_mapping"]["cell_subtype_key"] == "cell_subtype_label"
    assert manifest["run_scope"] == "patient_subset"
    assert manifest["artifact_state"] == "scaffold_active"


def test_stage0_validation_accepts_canonical_only_builder_output(tmp_path: Path) -> None:
    from tasks.task_A.stage0.build_artifacts import (
        build_stage0_adata_from_cell_table,
        build_stage0_h5ad_validation_report,
    )

    cell_table = _build_mock_cell_table()
    expression_matrix, marker_names = _build_mock_expression_bundle(cell_table.shape[0])
    adata = build_stage0_adata_from_cell_table(
        cell_table,
        expression_matrix=expression_matrix,
        marker_names=marker_names,
        k=6,
        knn=3,
        n_bal=6,
        random_state=0,
    )

    canonical = adata.copy()
    canonical.obs = canonical.obs.drop(columns=["roi_id", "compartment", "cell_type", "proto_id"])
    del canonical.obsm["community_features"]
    del canonical.uns["prototype_centroids"]
    del canonical.uns["s_C"]
    del canonical.uns["global_cost_scale"]
    del canonical.uns["scaler_params"]

    report = build_stage0_h5ad_validation_report(
        canonical,
        require_all_proto_ids=False,
    )
    assert report["artifact_state"] == "contract_passed"
    assert report["counts"]["n_rois"] == 12
    assert report["counts"]["n_compartments"] == 3
    assert report["counts"]["n_unique_proto_ids"] == 6
