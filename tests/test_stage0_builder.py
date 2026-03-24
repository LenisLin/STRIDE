from __future__ import annotations

# ruff: noqa: E402, I001

import importlib.util
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


def _write_task_a_config(path: Path, *, enabled_arms: list[str]) -> Path:
    config = {
        "task_name": "Task A Stage 0 smoke",
        "enabled_arms": enabled_arms,
        "data": {"mass_mode": "count", "k_full": 25},
        "uot_params": {
            "eps_schedule": [1.0, 0.5, 0.1],
            "max_iter": 400,
            "tol": 1.0e-8,
            "eta_floor": 1.0e-12,
            "n_min_proto": 0.0,
            "tau_mode": "external_fixed_by_task",
        },
        "arm1": {
            "n_draws": 3,
            "random_seed": 11,
            "fixed_lambda_by_compartment": {"TC": 1.0, "IM": 1.5, "PT": 2.0},
            "fixed_tau_by_compartment": {"TC": 0.4, "IM": 0.6, "PT": 0.8},
        },
        "arm2": {
            "target_alpha": 0.05,
            "lambda_grid": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_representation_builds_k25_outputs() -> None:
    from anndata import AnnData
    from slotar.representation import build_community_features, learn_global_prototypes

    cell_table = _build_small_stage0_cell_table()
    expression_matrix, marker_names = _build_small_expression_bundle(cell_table.shape[0])
    obs = pd.DataFrame(
        {
            "patient_id": cell_table["PID"].astype(str).to_numpy(),
            "timepoint": np.zeros(cell_table.shape[0], dtype=int),
            "roi_id": cell_table["ID"].astype(str).to_numpy(),
            "compartment": cell_table["Tissue"].astype(str).to_numpy(),
            "cell_type": cell_table["SubType"].astype(str).to_numpy(),
        },
        index=pd.Index(cell_table["CellID"].astype(str).to_numpy(), dtype="object"),
    )
    var = pd.DataFrame(index=pd.Index(marker_names, dtype="object", name="marker_name"))
    adata = AnnData(X=expression_matrix, obs=obs, var=var)
    adata.obsm["spatial"] = cell_table.loc[:, ["x", "y"]].to_numpy(dtype=float)
    adata.uns["roi_areas"] = {roi_id: 1.0 for roi_id in sorted(obs["roi_id"].unique())}

    build_community_features(adata, k=20)
    learn_global_prototypes(adata, n_bal=8, K=25, random_state=42)

    proto_id = adata.obs["proto_id"].astype(int)
    assert ((proto_id >= 0) & (proto_id < 25)).all()
    assert np.asarray(adata.obsm["community_features"]).shape[0] == obs.shape[0]
    assert np.asarray(adata.uns["cost_matrix"], dtype=float).shape == (25, 25)
    assert np.isfinite(np.asarray(adata.uns["cost_matrix"], dtype=float)).all()
    assert np.isfinite(float(adata.uns["s_C"]))
    assert float(adata.uns["s_C"]) > 0.0


def test_stage0_h5ad_meets_taska_minimum_contract(tmp_path: Path) -> None:
    import anndata as ad
    from tasks.task_A.build_stage0_artifacts import (
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

    required_fields = {"patient_id", "timepoint", "roi_id", "compartment", "cell_type", "proto_id"}
    assert required_fields.issubset(set(reloaded.obs.columns))
    assert set(reloaded.obs["compartment"].astype(str)) == {"TC", "IM", "PT"}
    assert np.asarray(reloaded.uns["cost_matrix"], dtype=float).shape == (25, 25)
    assert reloaded.n_vars == len(marker_names)
    assert reloaded.var_names.tolist() == marker_names
    np.testing.assert_allclose(np.asarray(reloaded.X), expression_matrix)


def test_stage0_representation_fields_if_present(tmp_path: Path) -> None:
    import anndata as ad
    from tasks.task_A.build_stage0_artifacts import (
        build_stage0_adata_from_cell_table,
        validate_representation_completeness,
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
    h5ad_path = tmp_path / "stage0_with_representation.h5ad"
    adata.write_h5ad(h5ad_path)

    reloaded = ad.read_h5ad(h5ad_path)
    validate_representation_completeness(reloaded)

    assert "cell_type" in reloaded.obs.columns
    assert "community_features" in reloaded.obsm
    assert "scaler_params" in reloaded.uns
    assert "prototype_centroids" in reloaded.uns
    assert reloaded.var_names.tolist() == marker_names
    np.testing.assert_allclose(np.asarray(reloaded.X), expression_matrix)


def test_load_expression_matrix_binary_roundtrip(tmp_path: Path) -> None:
    from tasks.task_A.build_stage0_artifacts import load_expression_matrix, load_marker_names

    marker_names = ["CD3", "CD8", "CD20"]
    expected = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
        ],
        dtype=np.float32,
    )
    marker_path = tmp_path / "markers.txt"
    marker_path.write_text("\n".join(marker_names) + "\n", encoding="utf-8")
    expr_path = tmp_path / "expression.bin"
    expected.T.astype(np.float32).flatten(order="F").tofile(expr_path)

    loaded_markers = load_marker_names(marker_path)
    loaded_expression = load_expression_matrix(
        expr_path,
        n_cells=expected.shape[0],
        marker_names=loaded_markers,
    )

    assert loaded_markers == marker_names
    np.testing.assert_allclose(loaded_expression, expected)


def test_stage0_pipeline_smoke_arm1(tmp_path: Path) -> None:
    from tasks.task_A.build_stage0_artifacts import build_stage0_adata_from_cell_table
    from tasks.task_A.pipeline import TASK_A_MANIFEST_FILENAME, TEMPORARY_METRICS_FILENAME, main

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
        enabled_arms=["A1_baseline", "A1_broken_reference"],
    )
    output_dir = tmp_path / "outputs"
    df_metrics = main(str(config_path), str(h5ad_path), str(output_dir))

    assert not df_metrics.empty
    assert df_metrics.shape[0] == 36
    assert set(df_metrics["arm"].astype(str)) == {"A1_baseline", "A1_broken_reference"}
    assert set(df_metrics["compartment"].astype(str)) == {"TC", "IM", "PT"}
    assert df_metrics["patient_group_id"].is_unique
    assert (df_metrics["roi_a"] != df_metrics["roi_b"]).all()
    assert df_metrics.loc[df_metrics["arm"] == "A1_baseline", "same_patient"].all()
    assert df_metrics.loc[df_metrics["arm"] == "A1_baseline", "same_compartment"].all()
    assert (~df_metrics.loc[df_metrics["arm"] == "A1_broken_reference", "same_patient"]).all()
    assert (~df_metrics.loc[df_metrics["arm"] == "A1_broken_reference", "same_compartment"]).all()
    assert (output_dir / TEMPORARY_METRICS_FILENAME).exists()
    assert (output_dir / TASK_A_MANIFEST_FILENAME).exists()


def test_stage0_pipeline_smoke_arm2(tmp_path: Path) -> None:
    from tasks.task_A.build_stage0_artifacts import build_stage0_adata_from_cell_table
    from tasks.task_A.pipeline import TASK_A_MANIFEST_FILENAME, TEMPORARY_METRICS_FILENAME, main

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
    h5ad_path = tmp_path / "stage0_pipeline_arm2.h5ad"
    adata.write_h5ad(h5ad_path)

    config_path = _write_task_a_config(
        tmp_path / "task_a_arm2.yaml",
        enabled_arms=["A2_cross_compartment"],
    )
    output_dir = tmp_path / "outputs_arm2"
    df_metrics = main(str(config_path), str(h5ad_path), str(output_dir))

    assert not df_metrics.empty
    assert df_metrics.shape[0] == 48
    assert set(df_metrics["arm"].astype(str)) == {"A2_cross_compartment"}
    assert set(df_metrics["pair_type"].astype(str)) == {
        "TC->IM",
        "IM->TC",
        "IM->PT",
        "PT->IM",
        "TC->PT",
        "PT->TC",
    }
    assert set(df_metrics["pair_family"].astype(str)) == {"TC-IM", "IM-PT", "TC-PT"}
    assert df_metrics["patient_group_id"].is_unique
    assert df_metrics["same_patient"].all()
    assert (~df_metrics["same_compartment"]).all()
    assert df_metrics["tau"].isna().all()
    assert df_metrics["R"].isna().all()

    ok_mask = df_metrics["uot_status"] == "ok"
    if ok_mask.any():
        assert np.isfinite(df_metrics.loc[ok_mask, "M_balanced"].to_numpy(dtype=float)).all()

    assert (output_dir / TEMPORARY_METRICS_FILENAME).exists()
    assert (output_dir / TASK_A_MANIFEST_FILENAME).exists()
