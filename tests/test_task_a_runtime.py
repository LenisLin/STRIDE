from __future__ import annotations

# ruff: noqa: E402, I001

import importlib.util
import json
import sys
from pathlib import Path

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


def _write_config(path: Path, *, enabled_blocks: list[str]) -> Path:
    from tests.helpers_task_a_fixture import K_FULL

    config = {
        "task_name": "Task A block smoke test",
        "enabled_blocks": enabled_blocks,
        "data": {
            "mass_mode": "density",
            "k_full": K_FULL,
        },
        "uot_params": {
            "eps_schedule": [1.0, 0.5, 0.1],
            "max_iter": 4000,
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
            "lambda_grid": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_block0_workflow_writes_deferred_bundle(tmp_path: Path) -> None:
    from tasks.task_A.workflows.run_block0 import run_block0_workflow
    from tests.helpers_task_a_fixture import write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(tmp_path / "config.yaml", enabled_blocks=["block0_locality_gate"])
    output_dir = tmp_path / "block0"

    bundle_path = run_block0_workflow(
        config_path=str(config_path),
        data_path=str(fixture_path),
        output_dir=str(output_dir),
    )

    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert payload["block"] == "block0_locality_gate"
    assert payload["status"] == "deferred"
    assert "locality-gate" in payload["blocker_reason"]
    assert (output_dir / "block0_workflow_manifest.json").exists()


def test_block1_workflow_writes_block_local_bundle(tmp_path: Path) -> None:
    from tasks.task_A.workflows.run_block1 import run_block1_workflow
    from tests.helpers_task_a_fixture import write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(tmp_path / "config.yaml", enabled_blocks=["block1_continuity_backbone"])
    output_dir = tmp_path / "block1"

    bundle_path = run_block1_workflow(
        config_path=str(config_path),
        data_path=str(fixture_path),
        output_dir=str(output_dir),
    )

    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    dry_run_path = Path(payload["core_fit_dry_run_path"])
    mapping_path = Path(payload["mapping_manifest_path"])
    assert payload["block"] == "block1_continuity_backbone"
    assert payload["status"] == "active"
    assert payload["confirmatory_pair_families"] == ["TC-IM", "TC-PT"]
    assert dry_run_path.exists()
    assert mapping_path.exists()

    dry_run_df = pd.read_csv(dry_run_path)
    assert set(dry_run_df["pair_family"].astype(str)) == {"TC-IM", "TC-PT"}
    assert set(dry_run_df["fit_status"].astype(str)).issubset({"ok", "deferred", "failed"})


def test_block2_workflow_consumes_block1_bundle(tmp_path: Path) -> None:
    from tasks.task_A.workflows.run_block1 import run_block1_workflow
    from tasks.task_A.workflows.run_block2 import run_block2_workflow
    from tests.helpers_task_a_fixture import write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(
        tmp_path / "config.yaml",
        enabled_blocks=["block1_continuity_backbone", "block2_bounded_audit"],
    )
    block1_root = tmp_path / "block1"
    block2_root = tmp_path / "block2"

    bundle_path = run_block1_workflow(
        config_path=str(config_path),
        data_path=str(fixture_path),
        output_dir=str(block1_root),
    )
    manifest_path = run_block2_workflow(
        block1_bundle=str(bundle_path),
        output_dir=str(block2_root),
    )

    assert manifest_path.exists()
    assert (block2_root / "block2_bounded_audit_summary.csv").exists()
    assert (block2_root / "block2_contract_audit.csv").exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["block"] == "block2_bounded_audit"
    assert Path(payload["block1_bundle_path"]).exists()
    assert Path(payload["summary_path"]).exists()
