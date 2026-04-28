from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from collections import defaultdict
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

from stride.errors import ContractError
from stride.outputs.fit_result import PatientBridgeResult

ANNDATA_AVAILABLE = importlib.util.find_spec("anndata") is not None
pytestmark = pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not installed")


def _write_config(path: Path) -> Path:
    config = {
        "task_name": "Task A Block 0 test",
        "enabled_blocks": ["block0_locality_gate"],
        "data": {"mass_mode": "uniform", "k_full": 25},
        "block0": {
            "random_seed": 7,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _counts_by_patient_and_group(observations: tuple[object, ...]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for observation in observations:
        counts[str(observation.patient_id)][str(observation.timepoint)] += 1
    return {
        patient_id: dict(sorted(group_counts.items()))
        for patient_id, group_counts in sorted(counts.items())
    }


def test_block0_real_family_construction_uses_tc_im_only(tmp_path: Path) -> None:
    from tasks.task_A.block0.locality_gate import build_block0_real_family_observations
    from tasks.task_A.config import load_task_a_config_bundle
    from tests.helpers_task_a_fixture import write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(tmp_path / "config.yaml")
    config_bundle = load_task_a_config_bundle(config_path)

    observations = build_block0_real_family_observations(
        fixture_path,
        config_bundle=config_bundle,
    )

    assert observations
    assert {str(observation.timepoint) for observation in observations} == {"TC", "IM"}
    assert {str(observation.domain_label) for observation in observations} == {"TC", "IM"}
    assert _counts_by_patient_and_group(observations) == {
        "P01": {"IM": 2, "TC": 2},
        "P02": {"IM": 2, "TC": 2},
    }


def test_block0_real_family_construction_accepts_canonical_stride_field_names(tmp_path: Path) -> None:
    import anndata as ad

    from tasks.task_A.block0.locality_gate import build_block0_real_family_observations
    from tasks.task_A.config import load_task_a_config_bundle
    from tests.helpers_task_a_fixture import build_task_a_fixture

    legacy = build_task_a_fixture()
    canonical = ad.AnnData(
        X=legacy.X,
        obs=legacy.obs.rename(
            columns={
                "roi_id": "fov_id",
                "compartment": "domain_label",
                "cell_type": "cell_subtype_label",
                "proto_id": "state_id",
            }
        ),
        obsm={
            "spatial": legacy.obsm["spatial"],
            "local_state_features": legacy.obsm["community_features"],
        },
        uns={
            "roi_areas": dict(legacy.uns["roi_areas"]),
            "cost_matrix": legacy.uns["cost_matrix"],
            "state_centroids": legacy.uns["prototype_centroids"],
            "cost_scale": legacy.uns["s_C"],
        },
    )
    config_path = _write_config(tmp_path / "config.yaml")
    config_bundle = load_task_a_config_bundle(config_path)

    observations = build_block0_real_family_observations(
        canonical,
        config_bundle=config_bundle,
    )

    assert _counts_by_patient_and_group(observations) == {
        "P01": {"IM": 2, "TC": 2},
        "P02": {"IM": 2, "TC": 2},
    }


def test_block0_null_family_construction_is_seeded_and_count_preserving(tmp_path: Path) -> None:
    from tasks.task_A.block0.locality_gate import (
        build_block0_null_assignments,
        build_block0_null_family_observations,
        build_block0_real_family_observations,
        load_block0_runtime_config,
    )
    from tasks.task_A.config import load_task_a_config_bundle
    from tests.helpers_task_a_fixture import write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(tmp_path / "config.yaml")
    config_bundle = load_task_a_config_bundle(config_path)
    runtime_config = load_block0_runtime_config(config_bundle)
    real_observations = build_block0_real_family_observations(
        fixture_path,
        config_bundle=config_bundle,
    )

    assignments_a = build_block0_null_assignments(real_observations, runtime_config)
    assignments_b = build_block0_null_assignments(real_observations, runtime_config)
    assert assignments_a == assignments_b
    assert {assignment.anchor_patient_id for assignment in assignments_a} == {"P01", "P02"}
    assert all(assignment.assignment_status == "assigned" for assignment in assignments_a)
    assert all(assignment.anchor_patient_id != assignment.donor_patient_id for assignment in assignments_a)
    assert all(assignment.count_stratum_key == "TC:2|IM:2" for assignment in assignments_a)

    null_observations = build_block0_null_family_observations(
        real_observations,
        runtime_config,
        assignments_a,
    )
    assert _counts_by_patient_and_group(null_observations) == _counts_by_patient_and_group(real_observations)


def test_block0_singleton_stratum_defers_null_assignment(tmp_path: Path) -> None:
    from tasks.task_A.block0.locality_gate import (
        build_block0_null_assignments,
        build_block0_null_family_observations,
        build_block0_real_family_observations,
        load_block0_runtime_config,
    )
    from tasks.task_A.config import load_task_a_config_bundle
    from tests.helpers_task_a_fixture import build_task_a_fixture

    adata = build_task_a_fixture()
    single_patient = adata[adata.obs["patient_id"].astype(str) == "P01"].copy()
    config_path = _write_config(tmp_path / "config.yaml")
    config_bundle = load_task_a_config_bundle(config_path)
    runtime_config = load_block0_runtime_config(config_bundle)

    real_observations = build_block0_real_family_observations(
        single_patient,
        config_bundle=config_bundle,
    )
    assignments = build_block0_null_assignments(real_observations, runtime_config)
    assert len(assignments) == 1
    assert assignments[0].assignment_status == "deferred"
    assert assignments[0].assignment_reason == "count_stratum_has_fewer_than_two_patients"

    null_observations = build_block0_null_family_observations(
        real_observations,
        runtime_config,
        assignments,
    )
    assert _counts_by_patient_and_group(null_observations) == {"P01": {"TC": 2}}


def test_block0_summary_computation_from_a_d_e() -> None:
    from tasks.task_A.block0.locality_gate import summarize_block0_bridge_result_totals

    result = PatientBridgeResult(
        patient_id="P01",
        fit_status="ok",
        A=np.asarray([[0.6, 0.1], [0.2, 0.5]], dtype=float),
        d=np.asarray([0.3, 0.3], dtype=float),
        e=np.asarray([0.05, 0.15], dtype=float),
    )

    summary = summarize_block0_bridge_result_totals(result)
    assert summary == {
        "total_continuity_mass": pytest.approx(1.4),
        "total_depletion_mass": pytest.approx(0.6),
        "total_emergence_mass": pytest.approx(0.2),
    }


def test_block0_workflow_writes_stride_native_bundle_and_schema(tmp_path: Path) -> None:
    from tasks.task_A.workflows.run_block0 import run_block0_workflow
    from tests.helpers_task_a_fixture import write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(tmp_path / "config.yaml")

    bundle_path = run_block0_workflow(
        config_path=config_path,
        data_path=fixture_path,
        output_dir=tmp_path / "block0",
    )

    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    pair_metrics_df = pd.read_csv(payload["pair_metrics_path"])

    assert payload["status"] == "deferred"
    assert payload["artifact_state"] == "scaffold_active"
    assert payload["implementation_tier"] == "canonical_full"
    assert payload["evidence_lineage"] == "canonical_rerun"
    assert payload["block0_passed"] is False
    assert payload["real_families"] == ["TC-IM"]
    assert payload["null_families"] == ["TC-IM_randomized_target"]
    assert "confirmatory_pair_families" not in payload
    assert "control_families" not in payload
    assert set(pair_metrics_df["pair_family"].astype(str)) == {"TC-IM"}
    assert set(pair_metrics_df["null_family"].astype(str)) == {"TC-IM_randomized_target"}
    assert {
        "real_fit_status",
        "null_fit_status",
        "delta_total_continuity_mass",
        "delta_total_emergence_mass",
        "count_stratum_key",
    }.issubset(pair_metrics_df.columns)
    assert not {"R", "M", "tau", "D_pos", "B_pos", "d_rel", "b_rel"}.intersection(pair_metrics_df.columns)
    assert payload["metrics_summary"]["eligible_patients"] == 2
    assert payload["metrics_summary"]["required_support"] == 6
    assert payload["metrics_summary"]["gate_summary_quantities"] == [
        "delta_total_continuity_mass",
        "delta_total_emergence_mass",
    ]
    assert payload["inputs"]["real_family_definition"]["fit_surface"] == "fit_stride"
    assert "paired_support_below_threshold" in payload["failure_reasons"]


def test_block0_subset_sidecar_runs_but_cannot_contract_pass(tmp_path: Path) -> None:
    from tasks.task_A.workflows.run_block0 import run_block0_workflow
    from tests.helpers_task_a_fixture import write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(tmp_path / "config.yaml")

    bundle_path = run_block0_workflow(
        config_path=config_path,
        data_path=fixture_path,
        output_dir=tmp_path / "block0_subset",
        patient_ids=("P01",),
    )
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    pair_metrics_df = pd.read_csv(payload["pair_metrics_path"])

    assert payload["run_scope"] == "patient_subset"
    assert payload["status"] == "deferred"
    assert payload["artifact_state"] == "scaffold_active"
    assert payload["block0_passed"] is False
    assert payload["metrics_summary"]["paired_comparisons"]["paired_support"] == 0
    assert set(pair_metrics_df["null_fit_status"].astype(str)) == {"deferred"}


def test_require_block0_passed_contract_accepts_written_pass_bundle_and_rejects_non_pass(
    tmp_path: Path,
) -> None:
    from tasks.task_A.block0.bundle import require_block0_passed_contract
    from tasks.task_A.workflows.run_block0 import run_block0_workflow
    from tests.helpers_task_a_fixture import write_passed_block0_bundle, write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(tmp_path / "config.yaml")

    passed_bundle_path = write_passed_block0_bundle(
        tmp_path / "block0_passed" / "block0_bundle.json",
        config_path=config_path,
        data_path=fixture_path,
    )
    contract = require_block0_passed_contract(
        passed_bundle_path,
        config_path=config_path,
        data_path=fixture_path,
    )
    assert contract.status == "passed"
    assert contract.artifact_state == "contract_passed"
    assert contract.real_families == ("TC-IM",)
    assert contract.null_families == ("TC-IM_randomized_target",)

    subset_bundle_path = run_block0_workflow(
        config_path=config_path,
        data_path=fixture_path,
        output_dir=tmp_path / "block0_subset",
        patient_ids=("P01",),
    )
    with pytest.raises(ContractError, match="requires a passed, contract-passed Block 0 locality gate bundle"):
        require_block0_passed_contract(
            subset_bundle_path,
            config_path=config_path,
            data_path=fixture_path,
        )


def test_require_block0_passed_contract_allows_downstream_block2_block3_config_drift(
    tmp_path: Path,
) -> None:
    from tasks.task_A.block0.bundle import require_block0_passed_contract
    from tests.helpers_task_a_fixture import write_passed_block0_bundle, write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task_name": "Task A Block 0 test",
                "enabled_blocks": [
                    "block0_locality_gate",
                    "block1_continuity_backbone",
                ],
                "data": {"mass_mode": "uniform", "k_full": 25},
                "block0": {"random_seed": 7},
                "block1": {
                    "target_alpha": 0.05,
                    "lambda_grid": [0.05, 0.1, 0.5, 1.0],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    passed_bundle_path = write_passed_block0_bundle(
        tmp_path / "block0_passed" / "block0_bundle.json",
        config_path=config_path,
        data_path=fixture_path,
    )

    config_path.write_text(
        yaml.safe_dump(
                {
                    "task_name": "Task A Block 0 test",
                    "enabled_blocks": [
                        "block0_locality_gate",
                        "block1_continuity_backbone",
                        "block2_bounded_audit",
                    ],
                    "data": {"mass_mode": "uniform", "k_full": 25},
                    "block0": {"random_seed": 7},
                    "block1": {
                        "target_alpha": 0.05,
                        "lambda_grid": [0.05, 0.1, 0.5, 1.0],
                    },
                    "block2": {
                        "patient_subsample_replicates": 2,
                        "leave_some_out_replicates": 2,
                    "seed_rerun_replicates": 1,
                    "roi_drop_replicates": 1,
                    "patient_subsample_min_patients": 1,
                    "leave_some_out_min_patients": 1,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    contract = require_block0_passed_contract(
        passed_bundle_path,
        config_path=config_path,
        data_path=fixture_path,
    )
    assert contract.status == "passed"
    assert contract.artifact_state == "contract_passed"


def test_block0_cli_parser_rejects_combined_subset_flags() -> None:
    from tasks.task_A.workflows.run_block0 import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--task-config",
                "tasks/task_A/config.yaml",
                "--stage0-h5ad",
                "fixture.h5ad",
                "--output-dir",
                "/tmp/out",
                "--patient-id",
                "P01",
                "--demo-subset",
                "alignment_v1",
            ]
        )


def test_block0_cli_writes_bundle_and_pair_metrics(tmp_path: Path) -> None:
    from tests.helpers_task_a_fixture import write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(tmp_path / "config.yaml")
    output_dir = tmp_path / "cli_block0"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tasks.task_A.workflows.run_block0",
            "--task-config",
            str(config_path),
            "--stage0-h5ad",
            str(fixture_path),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": f"{ROOT}{os.pathsep}{SRC}"},
        timeout=120,
    )

    assert result.returncode == 0, (
        f"run_block0 CLI exited with code {result.returncode}\n"
        f"STDOUT: {result.stdout}\n"
        f"STDERR: {result.stderr}"
    )
    assert (output_dir / "block0_bundle.json").exists()
    assert (output_dir / "block0_pair_metrics.csv").exists()
