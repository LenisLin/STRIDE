from __future__ import annotations

# ruff: noqa: E402, I001

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
from stride.latent.recurrence import RecurrenceResult
from stride.outputs.fit_result import PatientBridgeResult, STRIDEFitResult

ANNDATA_AVAILABLE = importlib.util.find_spec("anndata") is not None
pytestmark = pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not installed")


def _write_config(path: Path, *, enabled_blocks: list[str]) -> Path:
    from tests.helpers_task_a_fixture import K_FULL

    config = {
        "task_name": "Task A block smoke test",
        "enabled_blocks": enabled_blocks,
        "data": {
            "mass_mode": "uniform",
            "k_full": K_FULL,
        },
        "block0": {
            "random_seed": 7,
        },
        "block1": {
            "target_alpha": 0.05,
            "lambda_grid": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        },
        "block2": {
            "patient_subsample_replicates": 2,
            "leave_some_out_replicates": 2,
            "seed_rerun_replicates": 1,
            "roi_drop_replicates": 1,
            "patient_subsample_min_patients": 1,
            "leave_some_out_min_patients": 1,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_block0_workflow_writes_real_bundle(tmp_path: Path) -> None:
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
    assert payload["artifact_state"] == "scaffold_active"
    assert payload["implementation_tier"] == "canonical_full"
    assert payload["evidence_lineage"] == "canonical_rerun"
    assert payload["block0_passed"] is False
    assert payload["real_families"] == ["TC-IM"]
    assert payload["null_families"] == ["TC-IM_randomized_target"]
    assert Path(payload["pair_metrics_path"]).exists()


def test_block1_workflow_writes_block_local_bundle(tmp_path: Path) -> None:
    from tasks.task_A.workflows.run_block1 import run_block1_workflow
    from tests.helpers_task_a_fixture import write_passed_block0_bundle, write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(
        tmp_path / "config.yaml",
        enabled_blocks=["block0_locality_gate", "block1_continuity_backbone"],
    )
    output_dir = tmp_path / "block1"
    block0_bundle_path = write_passed_block0_bundle(
        tmp_path / "block0" / "block0_bundle.json",
        config_path=config_path,
        data_path=fixture_path,
    )

    bundle_path = run_block1_workflow(
        config_path=str(config_path),
        data_path=str(fixture_path),
        block0_bundle=str(block0_bundle_path),
        output_dir=str(output_dir),
    )

    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    dry_run_path = Path(payload["core_fit_dry_run_path"])
    mapping_path = Path(payload["mapping_manifest_path"])
    family_summary_path = Path(payload["family_summary_path"])
    source_summary_path = Path(payload["source_community_summary_path"])
    target_summary_path = Path(payload["target_community_summary_path"])
    recurrence_summary_path = Path(payload["recurrence_summary_path"])
    recurrence_families_path = Path(payload["recurrence_families_path"])
    recurrence_embeddings_path = Path(payload["recurrence_embeddings_path"])
    family_comparison_path = Path(payload["confirmatory_family_comparison_path"])
    source_comparison_path = Path(payload["exploratory_source_community_comparison_path"])
    target_comparison_path = Path(payload["exploratory_target_community_comparison_path"])
    correspondence_manifest_path = Path(payload["community_correspondence_manifest_path"])
    correspondence_index_path = Path(payload["community_correspondence_index_path"])
    assert payload["block"] == "block1_continuity_backbone"
    assert payload["scientific_role"] == "real_data_biological_discovery"
    assert payload["status"] == "active"
    assert payload["artifact_state"] == "evidence_ready"
    assert payload["implementation_tier"] == "canonical_full"
    assert payload["evidence_lineage"] == "canonical_rerun"
    assert payload["fit_surface"] == "fit_stride"
    assert Path(payload["block0_bundle_path"]) == block0_bundle_path
    assert payload["block0_gate_status"] == "passed"
    assert payload["confirmatory_pair_families"] == ["TC-IM", "TC-PT"]
    assert payload["summary_contract_version"] == "task_a_block1_summary_v1"
    assert payload["paired_comparison_contract_version"] == "task_a_block1_paired_comparison_v1"
    assert payload["proof_carrying_family_summaries"] == ["self_retention", "depletion"]
    assert payload["supportive_family_summaries"] == ["off_diagonal_remodeling", "emergence"]
    assert payload["family_summary_scales"] == ["burden_weighted", "community_mean"]
    assert payload["source_eligibility_rule"] == "mu_minus > 0"
    assert payload["target_eligibility_rule"] == "mu_plus > 0"
    assert dry_run_path.exists()
    assert mapping_path.exists()
    assert recurrence_summary_path.exists()
    assert recurrence_families_path.exists()
    assert recurrence_embeddings_path.exists()
    assert family_summary_path.exists()
    assert source_summary_path.exists()
    assert target_summary_path.exists()
    assert family_comparison_path.exists()
    assert source_comparison_path.exists()
    assert target_comparison_path.exists()
    assert correspondence_manifest_path.exists()
    assert correspondence_index_path.exists()

    dry_run_df = pd.read_csv(dry_run_path)
    assert set(dry_run_df["pair_family"].astype(str)) == {"TC-IM", "TC-PT"}
    assert set(dry_run_df["fit_status"].astype(str)).issubset({"ok", "deferred", "failed"})
    assert set(dry_run_df["implementation_tier"].astype(str)) == {"canonical_full"}
    assert set(dry_run_df["fit_surface"].astype(str)) == {"fit_stride"}
    family_summary_df = pd.read_csv(family_summary_path)
    assert set(family_summary_df["summary_name"].astype(str)) == {
        "self_retention",
        "depletion",
        "off_diagonal_remodeling",
        "emergence",
    }


def test_block1_cli_writes_bundle_and_artifacts(tmp_path: Path) -> None:
    from tests.helpers_task_a_fixture import write_passed_block0_bundle, write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(
        tmp_path / "config.yaml",
        enabled_blocks=["block0_locality_gate", "block1_continuity_backbone"],
    )
    output_dir = tmp_path / "cli_block1"
    block0_bundle_path = write_passed_block0_bundle(
        tmp_path / "block0" / "block0_bundle.json",
        config_path=config_path,
        data_path=fixture_path,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tasks.task_A.workflows.run_block1",
            "--task-config",
            str(config_path),
            "--stage0-h5ad",
            str(fixture_path),
            "--block0-bundle",
            str(block0_bundle_path),
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
        f"run_block1 CLI exited with code {result.returncode}\n"
        f"STDOUT: {result.stdout}\n"
        f"STDERR: {result.stderr}"
    )
    assert (output_dir / "block1_bundle.json").exists()
    assert (output_dir / "block1_workflow_manifest.json").exists()
    assert (output_dir / "community_correspondence" / "block1_community_correspondence_manifest.json").exists()


def test_block1_workflow_writes_comparison_and_correspondence_packet(tmp_path: Path) -> None:
    from tasks.task_A.workflows.run_block1 import run_block1_workflow
    from tests.helpers_task_a_fixture import write_passed_block0_bundle, write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(
        tmp_path / "config.yaml",
        enabled_blocks=["block0_locality_gate", "block1_continuity_backbone"],
    )
    output_dir = tmp_path / "block1"
    block0_bundle_path = write_passed_block0_bundle(
        tmp_path / "block0" / "block0_bundle.json",
        config_path=config_path,
        data_path=fixture_path,
    )

    bundle_path = run_block1_workflow(
        config_path=str(config_path),
        data_path=str(fixture_path),
        block0_bundle=str(block0_bundle_path),
        output_dir=str(output_dir),
    )
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))

    family_comparison = pd.read_csv(payload["confirmatory_family_comparison_path"])
    source_comparison = pd.read_csv(payload["exploratory_source_community_comparison_path"])
    target_comparison = pd.read_csv(payload["exploratory_target_community_comparison_path"])
    correspondence_manifest = json.loads(
        Path(payload["community_correspondence_manifest_path"]).read_text(encoding="utf-8")
    )
    correspondence_index = pd.read_csv(payload["community_correspondence_index_path"])

    assert len(family_comparison) == 16
    assert set(family_comparison["comparison_status"].astype(str)) == {"estimable"}
    assert set(family_comparison["pair_family_left"].astype(str)) == {"TC-IM"}
    assert set(family_comparison["pair_family_right"].astype(str)) == {"TC-PT"}
    assert set(family_comparison["comparison_scope_role"].astype(str)) == {"confirmatory"}
    assert set(family_comparison["summary_name"].astype(str)) == {
        "self_retention",
        "depletion",
        "off_diagonal_remodeling",
        "emergence",
    }

    assert not source_comparison.empty
    assert set(source_comparison["comparison_scope_role"].astype(str)) == {"exploratory_supportive"}
    assert set(source_comparison["summary_name"].astype(str)) == {
        "self_retention",
        "depletion",
        "off_diagonal_remodeling",
    }

    assert not target_comparison.empty
    assert set(target_comparison["comparison_scope_role"].astype(str)) == {"exploratory_supportive"}
    assert set(target_comparison["summary_name"].astype(str)) == {
        "incoming_matched_operator",
        "emergence_tendency",
    }

    assert correspondence_manifest["packet_role"] == "objective_community_correspondence"
    assert correspondence_manifest["artifact_state"] == "evidence_ready"
    assert correspondence_manifest["scientific_interpretation_allowed"] is False
    assert set(correspondence_index["category"].astype(str)) >= {
        "community_cell_subtype",
        "community_crosswalk",
        "comparison_reference",
        "source_burden_components",
        "source_major_targets",
        "summary_reference",
        "target_burden_components",
    }
    assert {
        "block1_source_community_summary.csv",
        "block1_target_community_summary.csv",
        "block1_confirmatory_family_comparison.csv",
        "block1_exploratory_source_community_comparison.csv",
        "block1_exploratory_target_community_comparison.csv",
        "community_correspondence/tables/community_id_crosswalk.csv",
    }.issubset(set(correspondence_index["relative_path"].astype(str)))


def test_block1_workflow_writes_manifest_consistent_with_bundle(tmp_path: Path) -> None:
    from tasks.task_A.block1.bundle import WORKFLOW_MANIFEST_FILENAME
    from tasks.task_A.workflows.run_block1 import run_block1_workflow
    from tests.helpers_task_a_fixture import write_passed_block0_bundle, write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(
        tmp_path / "config.yaml",
        enabled_blocks=["block0_locality_gate", "block1_continuity_backbone"],
    )
    output_dir = tmp_path / "block1"
    block0_bundle_path = write_passed_block0_bundle(
        tmp_path / "block0" / "block0_bundle.json",
        config_path=config_path,
        data_path=fixture_path,
    )

    bundle_path = run_block1_workflow(
        config_path=str(config_path),
        data_path=str(fixture_path),
        block0_bundle=str(block0_bundle_path),
        output_dir=str(output_dir),
    )

    bundle_payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    manifest_payload = json.loads((output_dir / WORKFLOW_MANIFEST_FILENAME).read_text(encoding="utf-8"))

    assert manifest_payload["bundle_path"] == str(bundle_path)
    for field_name in (
        "block",
        "scientific_role",
        "status",
        "artifact_state",
        "implementation_tier",
        "evidence_lineage",
        "fit_surface",
        "block0_bundle_path",
        "block0_gate_status",
        "config_fingerprint",
        "core_fit_dry_run_path",
        "mapping_manifest_path",
        "recurrence_summary_path",
        "recurrence_families_path",
        "recurrence_embeddings_path",
        "family_summary_path",
        "source_community_summary_path",
        "target_community_summary_path",
        "confirmatory_family_comparison_path",
        "exploratory_source_community_comparison_path",
        "exploratory_target_community_comparison_path",
        "community_correspondence_manifest_path",
        "community_correspondence_index_path",
        "summary_contract_version",
        "paired_comparison_contract_version",
        "proof_carrying_family_summaries",
        "supportive_family_summaries",
        "family_summary_scales",
    ):
        assert manifest_payload[field_name] == bundle_payload[field_name]


def test_block1_workflow_passes_block1_controls_into_fit_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tasks.task_A.workflows import stride_adapter
    from tasks.task_A.workflows.run_block1 import run_block1_workflow
    from tests.helpers_task_a_fixture import write_passed_block0_bundle, write_task_a_fixture

    captured_metadata: list[dict[str, object]] = []

    def fake_fit_stride(observations, state_basis, config):  # type: ignore[no-untyped-def]
        del state_basis
        patient_ids = tuple(sorted({str(observation.patient_id) for observation in observations}))
        captured_metadata.append(dict(config.metadata))
        return STRIDEFitResult(
            patient_inputs=tuple(SimpleNamespace(patient_id=patient_id) for patient_id in patient_ids),
            patient_results=tuple(
                PatientBridgeResult(patient_id=patient_id, fit_status="deferred")
                for patient_id in patient_ids
            ),
            recurrence=RecurrenceResult(patient_ids=patient_ids, families=(), fit_status="deferred"),
            fit_status="deferred",
            metadata=dict(config.metadata),
        )

    monkeypatch.setattr(stride_adapter, "fit_stride", fake_fit_stride)

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(
        tmp_path / "config.yaml",
        enabled_blocks=["block0_locality_gate", "block1_continuity_backbone"],
    )
    block0_bundle_path = write_passed_block0_bundle(
        tmp_path / "block0" / "block0_bundle.json",
        config_path=config_path,
        data_path=fixture_path,
    )

    run_block1_workflow(
        config_path=str(config_path),
        data_path=str(fixture_path),
        block0_bundle=str(block0_bundle_path),
        output_dir=str(tmp_path / "block1"),
    )

    assert len(captured_metadata) == 2
    assert {str(metadata["task_pair_family"]) for metadata in captured_metadata} == {"TC-IM", "TC-PT"}
    assert {str(metadata["task_claim_role"]) for metadata in captured_metadata} == {"confirmatory"}
    assert {str(metadata["task_source_domain"]) for metadata in captured_metadata} == {"TC"}
    assert {str(metadata["task_target_domain"]) for metadata in captured_metadata} == {"IM", "PT"}
    for metadata in captured_metadata:
        assert metadata["task_block"] == "block1_continuity_backbone"
        assert metadata["task_block1_target_alpha"] == pytest.approx(0.05)
        assert metadata["task_block1_lambda_grid"] == [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def test_block1_workflow_rejects_disabled_block_in_config(tmp_path: Path) -> None:
    from tasks.task_A.workflows.run_block1 import run_block1_workflow
    from tests.helpers_task_a_fixture import write_passed_block0_bundle, write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(tmp_path / "config.yaml", enabled_blocks=["block0_locality_gate"])
    block0_bundle_path = write_passed_block0_bundle(
        tmp_path / "block0" / "block0_bundle.json",
        config_path=config_path,
        data_path=fixture_path,
    )

    with pytest.raises(ContractError, match="does not enable block1_continuity_backbone"):
        run_block1_workflow(
            config_path=str(config_path),
            data_path=str(fixture_path),
            block0_bundle=str(block0_bundle_path),
            output_dir=str(tmp_path / "block1"),
        )


def test_block1_workflow_rejects_config_mutation_after_block0(tmp_path: Path) -> None:
    from tasks.task_A.workflows.run_block1 import run_block1_workflow
    from tests.helpers_task_a_fixture import write_passed_block0_bundle, write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(
        tmp_path / "config.yaml",
        enabled_blocks=["block0_locality_gate", "block1_continuity_backbone"],
    )
    block0_bundle_path = write_passed_block0_bundle(
        tmp_path / "block0" / "block0_bundle.json",
        config_path=config_path,
        data_path=fixture_path,
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                **yaml.safe_load(config_path.read_text(encoding="utf-8")),
                "block1": {
                    "target_alpha": 0.01,
                    "lambda_grid": [0.5, 1.0],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="config content changed after the bundle was produced"):
        run_block1_workflow(
            config_path=str(config_path),
            data_path=str(fixture_path),
            block0_bundle=str(block0_bundle_path),
            output_dir=str(tmp_path / "block1"),
        )


def test_block2_workflow_consumes_block1_bundle(tmp_path: Path) -> None:
    from tasks.task_A.workflows.run_block1 import run_block1_workflow
    from tasks.task_A.workflows.run_block2 import run_block2_workflow
    from tests.helpers_task_a_fixture import write_passed_block0_bundle, write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(
        tmp_path / "config.yaml",
        enabled_blocks=["block0_locality_gate", "block1_continuity_backbone", "block2_bounded_audit"],
    )
    block1_root = tmp_path / "block1"
    block2_root = tmp_path / "block2"
    block0_bundle_path = write_passed_block0_bundle(
        tmp_path / "block0" / "block0_bundle.json",
        config_path=config_path,
        data_path=fixture_path,
    )

    bundle_path = run_block1_workflow(
        config_path=str(config_path),
        data_path=str(fixture_path),
        block0_bundle=str(block0_bundle_path),
        output_dir=str(block1_root),
    )
    manifest_path = run_block2_workflow(
        block1_bundle=str(bundle_path),
        output_dir=str(block2_root),
    )

    assert manifest_path.exists()
    assert (block2_root / "block2_bounded_audit_summary.csv").exists()
    assert (block2_root / "block2_contract_audit.csv").exists()
    assert (block2_root / "block2_replicate_manifest.csv").exists()
    assert (block2_root / "block2_family_robustness.csv").exists()
    assert (block2_root / "block2_source_community_robustness.csv").exists()
    assert (block2_root / "block2_target_community_robustness.csv").exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["block"] == "block2_bounded_audit"
    assert payload["scientific_role"] == "robustness_over_block1_summaries"
    assert payload["artifact_state"] == "evidence_ready"
    assert Path(payload["block1_bundle_path"]).exists()
    assert Path(payload["block0_bundle_path"]) == block0_bundle_path
    assert payload["claim_scope"] == "block1_summary_robustness"
    assert Path(payload["block1_family_summary_path"]).exists()
    assert Path(payload["block1_source_community_summary_path"]).exists()
    assert Path(payload["block1_target_community_summary_path"]).exists()
    assert Path(payload["replicate_manifest_path"]).exists()
    assert Path(payload["family_robustness_path"]).exists()
    assert Path(payload["source_community_robustness_path"]).exists()
    assert Path(payload["target_community_robustness_path"]).exists()
    assert Path(payload["summary_path"]).exists()

    summary_df = pd.read_csv(payload["summary_path"])
    assert "overall_robustness_call" in summary_df.columns
    replicate_df = pd.read_csv(payload["replicate_manifest_path"])
    assert set(replicate_df["route_name"].astype(str)) == {
        "patient_subsample",
        "leave_some_out",
        "seed_rerun",
        "roi_drop_one_per_domain",
    }


def test_block2_cli_writes_bundle_and_artifacts(tmp_path: Path) -> None:
    from tasks.task_A.workflows.run_block1 import run_block1_workflow
    from tests.helpers_task_a_fixture import write_passed_block0_bundle, write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(
        tmp_path / "config.yaml",
        enabled_blocks=["block0_locality_gate", "block1_continuity_backbone", "block2_bounded_audit"],
    )
    block0_bundle_path = write_passed_block0_bundle(
        tmp_path / "block0" / "block0_bundle.json",
        config_path=config_path,
        data_path=fixture_path,
    )
    block1_bundle_path = run_block1_workflow(
        config_path=str(config_path),
        data_path=str(fixture_path),
        block0_bundle=str(block0_bundle_path),
        output_dir=str(tmp_path / "block1"),
    )
    output_dir = tmp_path / "cli_block2"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tasks.task_A.workflows.run_block2",
            "--block1-bundle",
            str(block1_bundle_path),
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
        f"run_block2 CLI exited with code {result.returncode}\n"
        f"STDOUT: {result.stdout}\n"
        f"STDERR: {result.stderr}"
    )
    assert (output_dir / "block2_bounded_audit_manifest.json").exists()
    assert (output_dir / "block2_replicate_manifest.csv").exists()
    assert (output_dir / "block2_family_robustness.csv").exists()


def test_block2_workflow_rejects_config_mutation_after_block1(tmp_path: Path) -> None:
    from tasks.task_A.workflows.run_block1 import run_block1_workflow
    from tasks.task_A.workflows.run_block2 import run_block2_workflow
    from tests.helpers_task_a_fixture import write_passed_block0_bundle, write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(
        tmp_path / "config.yaml",
        enabled_blocks=["block0_locality_gate", "block1_continuity_backbone", "block2_bounded_audit"],
    )
    block0_bundle_path = write_passed_block0_bundle(
        tmp_path / "block0" / "block0_bundle.json",
        config_path=config_path,
        data_path=fixture_path,
    )
    bundle_path = run_block1_workflow(
        config_path=str(config_path),
        data_path=str(fixture_path),
        block0_bundle=str(block0_bundle_path),
        output_dir=str(tmp_path / "block1"),
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                **yaml.safe_load(config_path.read_text(encoding="utf-8")),
                "block1": {
                    "target_alpha": 0.01,
                    "lambda_grid": [0.5, 1.0],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="config content changed after the bundle was produced"):
        run_block2_workflow(
            block1_bundle=str(bundle_path),
            output_dir=str(tmp_path / "block2"),
        )
