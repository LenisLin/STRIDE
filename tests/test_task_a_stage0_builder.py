from __future__ import annotations

# ruff: noqa: E402, I001

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

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


def test_stage0_builder_produces_task_a_ready_h5ad(tmp_path: Path) -> None:
    import anndata as ad
    from tasks.task_A.arm1_noise_baseline import generate_within_compartment_pairs
    from tasks.task_A.build_stage0_artifacts import (
        build_stage0_adata_from_cell_table,
        validate_stage0_minimum_contract,
    )
    from tasks.task_A.common import assemble_tensors

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        cell_table = _build_mock_cell_table()
        expression_matrix, marker_names = _build_mock_expression_bundle(cell_table.shape[0])
        adata = build_stage0_adata_from_cell_table(
            cell_table,
            expression_matrix=expression_matrix,
            marker_names=marker_names,
            k=25,
            knn=3,
            n_bal=6,
            random_state=0,
        )

    out_path = tmp_path / "stage0_fixture.h5ad"
    adata.write_h5ad(out_path)
    reloaded = ad.read_h5ad(out_path)

    validate_stage0_minimum_contract(reloaded)
    assert np.asarray(reloaded.uns["cost_matrix"]).shape == (25, 25)
    proto_ids = reloaded.obs["proto_id"].astype(int).to_numpy()
    assert np.all(proto_ids >= 0)
    assert np.all(proto_ids < 25)
    assert reloaded.n_vars == len(marker_names)
    assert reloaded.var_names.tolist() == marker_names
    np.testing.assert_allclose(np.asarray(reloaded.X), expression_matrix)

    pair_meta = generate_within_compartment_pairs(reloaded, n_draws=3, random_seed=11)
    assert pair_meta.shape[0] == 18
    assert pair_meta["patient_group_id"].is_unique
    assert (pair_meta["roi_a"] != pair_meta["roi_b"]).all()
    A, B, mass_gap = assemble_tensors(reloaded, pair_meta, k_full=25, mass_mode="count")
    assert A.shape == (18, 25)
    assert B.shape == (18, 25)
    assert mass_gap.shape == (18,)
