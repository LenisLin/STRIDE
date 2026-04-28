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

from stride.errors import ContractError

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
        "enabled_blocks": ["block0_locality_gate", "block1_continuity_backbone"],
        "data": {
            "mass_mode": "uniform",
            "k_full": 6,
        },
        "block0": {
            "random_seed": 7,
        },
        "block1": {
            "target_alpha": 0.05,
            "lambda_grid": [0.05, 0.1, 0.5, 1.0],
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_task_a_config_bundle_populates_framework_defaults(tmp_path: Path) -> None:
    from tasks.task_A.config import (
        DEFAULT_BLOCK2_PRIMARY_ROUTES,
        DEFAULT_BLOCK2_PRIMARY_SOURCE_COMMUNITIES,
        DEFAULT_BLOCK2_PRIMARY_TARGET_COMMUNITIES,
        DEFAULT_BLOCK3_ENABLED_SUBEXPERIMENTS,
        load_task_a_config_bundle,
    )

    config_path = _write_task_a_config(tmp_path / "task_a.yaml")
    bundle = load_task_a_config_bundle(config_path)

    assert bundle.enabled_blocks == ("block0_locality_gate", "block1_continuity_backbone")
    assert bundle.ordered_pair_family_names == ("TC-IM", "TC-PT", "IM-PT")
    assert bundle.data.mass_mode == "uniform"
    assert bundle.block1.target_alpha == pytest.approx(0.05)
    assert bundle.block1.lambda_grid == (0.05, 0.1, 0.5, 1.0)
    assert bundle.block2.primary_routes == DEFAULT_BLOCK2_PRIMARY_ROUTES
    assert bundle.block2.primary_source_communities == DEFAULT_BLOCK2_PRIMARY_SOURCE_COMMUNITIES
    assert bundle.block2.primary_target_communities == DEFAULT_BLOCK2_PRIMARY_TARGET_COMMUNITIES
    assert bundle.block3.enabled_subexperiments == DEFAULT_BLOCK3_ENABLED_SUBEXPERIMENTS
    assert bundle.exports.mapping_manifest_filename == "task_a_stride_mapping.json"
    assert bundle.benchmarks.default_n_patients == 6


def test_task_a_config_bundle_uses_default_block1_controls_when_section_missing(tmp_path: Path) -> None:
    from tasks.task_A.config import (
        DEFAULT_BLOCK1_LAMBDA_GRID,
        DEFAULT_BLOCK1_TARGET_ALPHA,
        load_task_a_config_bundle,
    )

    config_path = tmp_path / "task_a_defaults.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task_name": "Task A default block1 config",
                "enabled_blocks": ["block0_locality_gate", "block1_continuity_backbone"],
                "data": {
                    "mass_mode": "uniform",
                    "k_full": 6,
                },
                "block0": {
                    "random_seed": 7,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    bundle = load_task_a_config_bundle(config_path)

    assert bundle.block1.target_alpha == pytest.approx(DEFAULT_BLOCK1_TARGET_ALPHA)
    assert bundle.block1.lambda_grid == DEFAULT_BLOCK1_LAMBDA_GRID


@pytest.mark.parametrize(
    ("block1_payload", "match"),
    [
        ({"target_alpha": 0.0, "lambda_grid": [0.05, 0.1]}, "must lie strictly between 0 and 1"),
        ({"target_alpha": float("nan"), "lambda_grid": [0.05, 0.1]}, "must be a finite float"),
        ({"target_alpha": 0.05, "lambda_grid": []}, "must not be empty"),
        (
            {"target_alpha": 0.05, "lambda_grid": [0.05, float("nan")]},
            "positive finite floats",
        ),
    ],
)
def test_task_a_config_bundle_rejects_invalid_block1_controls(
    tmp_path: Path,
    block1_payload: dict[str, object],
    match: str,
) -> None:
    from tasks.task_A.config import load_task_a_config_bundle

    config_path = tmp_path / "task_a_invalid_block1.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task_name": "Task A invalid block1 config",
                "enabled_blocks": ["block0_locality_gate", "block1_continuity_backbone"],
                "data": {
                    "mass_mode": "uniform",
                    "k_full": 6,
                },
                "block0": {
                    "random_seed": 7,
                },
                "block1": block1_payload,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=match):
        load_task_a_config_bundle(config_path)


def test_task_a_config_bundle_rejects_block1_without_block0(tmp_path: Path) -> None:
    from tasks.task_A.config import load_task_a_config_bundle

    config = {
        "task_name": "Task A invalid config",
        "enabled_blocks": ["block1_continuity_backbone"],
        "data": {"mass_mode": "uniform", "k_full": 6},
        "observation_match": {
            "eps_schedule": [1.0, 0.5],
            "max_iter": 100,
            "tol": 1.0e-8,
            "eta_floor": 1.0e-12,
            "n_min_proto": 0.0,
        },
    }
    config_path = tmp_path / "task_a_invalid.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="block1_continuity_backbone depends on block0_locality_gate"):
        load_task_a_config_bundle(config_path)


def test_task_a_config_bundle_rejects_block2_without_full_gate_chain(tmp_path: Path) -> None:
    from tasks.task_A.config import load_task_a_config_bundle

    config = {
        "task_name": "Task A invalid block2 config",
        "enabled_blocks": ["block0_locality_gate", "block2_bounded_audit"],
        "data": {"mass_mode": "uniform", "k_full": 6},
        "observation_match": {
            "eps_schedule": [1.0, 0.5],
            "max_iter": 100,
            "tol": 1.0e-8,
            "eta_floor": 1.0e-12,
            "n_min_proto": 0.0,
        },
    }
    config_path = tmp_path / "task_a_invalid_block2.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="block2_bounded_audit depends on block1_continuity_backbone"):
        load_task_a_config_bundle(config_path)


def test_task_a_config_bundle_rejects_block3_as_unsupported_active_block(tmp_path: Path) -> None:
    from tasks.task_A.config import load_task_a_config_bundle

    config = {
        "task_name": "Task A invalid block3 config",
        "enabled_blocks": [
            "block0_locality_gate",
            "block1_continuity_backbone",
            "block3_method_validation",
        ],
        "data": {"mass_mode": "uniform", "k_full": 6},
    }
    config_path = tmp_path / "task_a_invalid_block3.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported Task-A blocks"):
        load_task_a_config_bundle(config_path)


def test_task_a_block1_summary_contract_constants_are_frozen() -> None:
    from tasks.task_A.block1.summaries import (
        FAMILY_SUMMARY_FILENAME,
        FAMILY_SUMMARY_SCALES,
        PROOF_CARRYING_SUMMARY_NAMES,
        SOURCE_COMMUNITY_SUMMARY_FILENAME,
        SOURCE_ELIGIBILITY_RULE,
        SUMMARY_CONTRACT_VERSION,
        SUPPORTIVE_SUMMARY_NAMES,
        TARGET_COMMUNITY_SUMMARY_FILENAME,
        TARGET_ELIGIBILITY_RULE,
    )

    assert SUMMARY_CONTRACT_VERSION == "task_a_block1_summary_v1"
    assert FAMILY_SUMMARY_FILENAME == "block1_family_summary.csv"
    assert SOURCE_COMMUNITY_SUMMARY_FILENAME == "block1_source_community_summary.csv"
    assert TARGET_COMMUNITY_SUMMARY_FILENAME == "block1_target_community_summary.csv"
    assert PROOF_CARRYING_SUMMARY_NAMES == ("self_retention", "depletion")
    assert SUPPORTIVE_SUMMARY_NAMES == ("off_diagonal_remodeling", "emergence")
    assert FAMILY_SUMMARY_SCALES == ("burden_weighted", "community_mean")
    assert SOURCE_ELIGIBILITY_RULE == "mu_minus > 0"
    assert TARGET_ELIGIBILITY_RULE == "mu_plus > 0"


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
        patient_ids=("P01",),
    )

    mapping_path = Path(manifest["mapping_manifest"])
    dry_run_path = Path(manifest["core_fit_dry_run"])
    assert mapping_path.exists()
    assert dry_run_path.exists()
    assert manifest["run_scope"] == "patient_subset"
    assert manifest["artifact_state"] == "scaffold_active"
    assert manifest["mass_mode"] == "uniform"
    assert manifest["scientific_interpretation_allowed"] is False

    mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert mapping_payload["field_mapping"]["domain_key"] == "domain_label"
    assert {summary["pair_family"] for summary in mapping_payload["family_summaries"]} == {
        "TC-IM",
        "TC-PT",
        "IM-PT",
    }

    dry_run_df = pd.read_csv(dry_run_path)
    assert set(dry_run_df["pair_family"].astype(str)) == {"TC-IM", "TC-PT"}
    assert set(dry_run_df["patient_id"].astype(str)) == {"P01"}
    assert set(dry_run_df["fit_status"].astype(str)).issubset({"ok", "deferred", "failed"})


def test_new_task_a_workflow_modules_import_cleanly() -> None:
    import tasks.task_A.config
    import tasks.task_A.contracts
    import tasks.task_A.real_data
    import tasks.task_A.workflows

    assert hasattr(tasks.task_A.config, "load_task_a_config_bundle")
    assert hasattr(tasks.task_A.workflows, "check_task_a_pre_block0_data_suitability")
    assert hasattr(tasks.task_A.workflows, "prepare_task_a_stage0_mapping")
    assert hasattr(tasks.task_A.workflows, "run_block0_workflow")
    assert hasattr(tasks.task_A.workflows, "run_block1_workflow")
    assert hasattr(tasks.task_A.workflows, "run_block2_workflow")
    assert hasattr(tasks.task_A.real_data, "resolve_demo_subset")
    assert not hasattr(tasks.task_A.workflows, "run_real_data_workflow")
    assert not hasattr(tasks.task_A.workflows, "run_semisynthetic_workflow")
    assert not hasattr(tasks.task_A.workflows, "build_task_a_export_index")


def test_task_a_legacy_modules_are_absent() -> None:
    legacy_modules = (
        "tasks.task_A.pipeline",
        "tasks.task_A.runtime_contract",
        "tasks.task_A.block1.focused",
        "tasks.task_A.block1.real_data_mirror",
        "tasks.task_A.real_data.block1_mirror",
        "tasks.task_A.block1.analysis",
    )
    for module_name in legacy_modules:
        assert importlib.util.find_spec(module_name) is None


def test_task_a_crosswalk_does_not_expose_roi_clinical_sidecar() -> None:
    from tasks.task_A.contracts.stride_mapping import TaskARealDataCrosswalk

    crosswalk = TaskARealDataCrosswalk()
    as_dict = crosswalk.to_json_dict()

    assert "unmapped_sidecar_fields" not in as_dict
    assert not hasattr(crosswalk, "unmapped_sidecar_fields")


def test_task_a_block0_surface_writes_real_bundle(tmp_path: Path) -> None:
    from tasks.task_A.workflows.run_block0 import run_block0_workflow

    adata = _build_stage0_adata()
    fixture_path = tmp_path / "fixture.h5ad"
    adata.write_h5ad(fixture_path)
    config_path = _write_task_a_config(tmp_path / "task_a.yaml")

    bundle_path = run_block0_workflow(
        config_path=config_path,
        data_path=fixture_path,
        output_dir=tmp_path / "block0",
    )
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert payload["status"] in {"passed", "failed", "deferred"}
    assert payload["pair_metrics_path"]
    assert Path(payload["pair_metrics_path"]).exists()


def test_task_a_framework_adds_no_task_specific_leakage_to_stride_core() -> None:
    offenders: list[str] = []
    for path in sorted((ROOT / "src" / "stride").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "task_A" in text or "Task A" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
