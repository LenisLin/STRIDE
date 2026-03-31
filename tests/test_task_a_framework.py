from __future__ import annotations

# ruff: noqa: E402, I001

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest
import yaml
from sklearn.exceptions import ConvergenceWarning

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
    subtypes = ["SubtypeA", "SubtypeB", "SubtypeC", "SubtypeD"]
    cell_id = 0

    for patient_idx, patient_id in enumerate(patients):
        for compartment_idx, compartment in enumerate(compartments):
            for roi_num in range(2):
                roi_id = f"{patient_id}_{compartment}_{roi_num + 1:02d}"
                for local_idx in range(6):
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


def _build_small_expression_bundle(n_cells: int) -> tuple[list[str], object]:
    import numpy as np

    marker_names = ["M01", "M02", "M03", "M04"]
    base = np.arange(n_cells, dtype=float)
    expression_matrix = np.column_stack(
        [
            base / max(n_cells - 1, 1),
            (base % 5) / 4.0,
            (base % 7) / 6.0,
            (base % 11) / 10.0,
        ]
    ).astype("float32")
    return marker_names, expression_matrix


def _build_stage0_adata():
    from tasks.task_A.stage0.build_artifacts import build_stage0_adata_from_cell_table

    cell_table = _build_small_stage0_cell_table()
    marker_names, expression_matrix = _build_small_expression_bundle(cell_table.shape[0])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        return build_stage0_adata_from_cell_table(
            cell_table,
            expression_matrix=expression_matrix,
            marker_names=marker_names,
            k=6,
            knn=3,
            n_bal=5,
            random_state=0,
        )


def _write_task_a_config(path: Path) -> Path:
    config = {
        "task_name": "Task A framework smoke",
        "enabled_blocks": ["block1_continuity_backbone"],
        "data": {
            "mass_mode": "density",
            "k_full": 6,
        },
        "uot_params": {
            "eps_schedule": [1.0, 0.5],
            "max_iter": 100,
            "tol": 1.0e-8,
            "eta_floor": 1.0e-12,
            "n_min_proto": 0.0,
            "tau_mode": "external_fixed_by_task",
        },
        "block0": {
            "n_draws": 2,
            "random_seed": 7,
            "fixed_lambda_by_compartment": {"TC": 1.0, "IM": 1.5, "PT": 2.0},
            "fixed_tau_by_compartment": {"TC": 0.5, "IM": 0.75, "PT": 1.0},
        },
        "block1": {
            "target_alpha": 0.05,
            "lambda_grid": [0.05, 0.1, 0.5, 1.0],
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_task_a_config_bundle_populates_framework_defaults(tmp_path: Path) -> None:
    from tasks.task_A.config import load_task_a_config_bundle

    config_path = _write_task_a_config(tmp_path / "task_a.yaml")
    bundle = load_task_a_config_bundle(config_path)

    assert bundle.enabled_blocks == ("block1_continuity_backbone",)
    assert bundle.ordered_pair_family_names == ("TC-IM", "TC-PT", "IM-PT")
    assert bundle.exports.mapping_manifest_filename == "task_a_stride_mapping.json"
    assert bundle.benchmarks.default_n_patients == 6


def test_prepare_workflow_writes_stride_mapping_and_core_fit_dry_run(tmp_path: Path) -> None:
    from tasks.task_A.workflows.prepare import prepare_task_a_stage0_mapping

    adata = _build_stage0_adata()
    stage0_path = tmp_path / "stage0.h5ad"
    adata.write_h5ad(stage0_path)
    config_path = _write_task_a_config(tmp_path / "task_a.yaml")

    manifest = prepare_task_a_stage0_mapping(
        config_path=config_path,
        data_path=stage0_path,
        output_dir=tmp_path / "prepare",
    )

    mapping_path = Path(manifest["mapping_manifest"])
    dry_run_path = Path(manifest["core_fit_dry_run"])
    assert mapping_path.exists()
    assert dry_run_path.exists()

    mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert mapping_payload["field_mapping"]["domain_key"] == "domain_label"
    assert {summary["pair_family"] for summary in mapping_payload["family_summaries"]} == {
        "TC-IM",
        "TC-PT",
        "IM-PT",
    }

    dry_run_df = pd.read_csv(dry_run_path)
    assert set(dry_run_df["pair_family"].astype(str)) == {"TC-IM", "TC-PT"}
    assert set(dry_run_df["fit_status"].astype(str)).issubset({"ok", "deferred", "failed"})


def test_new_task_a_workflow_modules_import_cleanly() -> None:
    import tasks.task_A.config
    import tasks.task_A.contracts
    import tasks.task_A.real_data
    import tasks.task_A.workflows

    assert hasattr(tasks.task_A.config, "load_task_a_config_bundle")
    assert hasattr(tasks.task_A.workflows, "prepare_task_a_stage0_mapping")
    assert hasattr(tasks.task_A.workflows, "run_block0_workflow")
    assert hasattr(tasks.task_A.workflows, "run_block1_workflow")
    assert hasattr(tasks.task_A.workflows, "run_block2_workflow")
    assert hasattr(tasks.task_A.real_data, "resolve_demo_subset")
    assert not hasattr(tasks.task_A.workflows, "run_real_data_workflow")
    assert not hasattr(tasks.task_A.workflows, "run_semisynthetic_workflow")
    assert not hasattr(tasks.task_A.workflows, "build_task_a_export_index")


def test_task_a_deferred_block0_surface_writes_bundle(tmp_path: Path) -> None:
    from tasks.task_A.workflows.run_block0 import run_block0_workflow
    from tests.helpers_task_a_fixture import write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = tmp_path / "task_a.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task_name": "Task A framework smoke",
                "enabled_blocks": ["block0_locality_gate"],
                "data": {"mass_mode": "density", "k_full": 6},
                "uot_params": {
                    "eps_schedule": [1.0, 0.5],
                    "max_iter": 100,
                    "tol": 1.0e-8,
                    "eta_floor": 1.0e-12,
                    "n_min_proto": 0.0,
                    "tau_mode": "external_fixed_by_task",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    bundle_path = run_block0_workflow(
        config_path=config_path,
        data_path=fixture_path,
        output_dir=tmp_path / "block0",
    )

    payload = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    assert payload["status"] == "deferred"
    assert payload["block"] == "block0_locality_gate"


def test_task_a_framework_adds_no_task_specific_leakage_to_stride_core() -> None:
    offenders: list[str] = []
    for path in sorted((ROOT / "src" / "stride").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "task_A" in text or "Task A" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
