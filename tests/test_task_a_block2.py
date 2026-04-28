from __future__ import annotations

import json
from pathlib import Path
import sys

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

from tasks.task_A.block2.bundle import (
    BLOCK_NAME,
    build_block2_summary,
    load_block1_bundle_contract,
    write_block2_bundle,
)
from tasks.task_A.workflows.run_block1 import run_block1_workflow
from tests.helpers_task_a_fixture import write_passed_block0_bundle, write_task_a_fixture


def _write_config(path: Path, *, enabled_blocks: list[str]) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "task_name": "Task A block2 test config",
                "enabled_blocks": enabled_blocks,
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
    return path


def test_build_block2_summary_aggregates_primary_route_calls() -> None:
    family_robustness_df = pd.DataFrame(
        [
            {
                "finding_id": "family::self_retention::burden_weighted",
                "route_name": "patient_subsample",
                "route_group": "primary",
                "summary_scope": "family",
                "finding_priority": "primary",
                "summary_name": "self_retention",
                "scale": "burden_weighted",
                "community_id": pd.NA,
                "full_data_direction": "tc_im_gt_tc_pt",
                "full_data_support_fraction": 1.0,
                "full_data_median_delta": 0.2,
                "robustness_call": "robust",
                "direction_recovery_rate": 1.0,
                "estimable_replicate_fraction": 1.0,
                "median_replicate_support_fraction": 1.0,
            },
            {
                "finding_id": "family::self_retention::burden_weighted",
                "route_name": "leave_some_out",
                "route_group": "primary",
                "summary_scope": "family",
                "finding_priority": "primary",
                "summary_name": "self_retention",
                "scale": "burden_weighted",
                "community_id": pd.NA,
                "full_data_direction": "tc_im_gt_tc_pt",
                "full_data_support_fraction": 1.0,
                "full_data_median_delta": 0.2,
                "robustness_call": "robust",
                "direction_recovery_rate": 0.8,
                "estimable_replicate_fraction": 1.0,
                "median_replicate_support_fraction": 0.75,
            },
        ]
    )
    source_robustness_df = pd.DataFrame(
        [
            {
                "finding_id": "source_community::1::self_retention",
                "route_name": "patient_subsample",
                "route_group": "primary",
                "summary_scope": "source_community",
                "finding_priority": "primary",
                "summary_name": "self_retention",
                "scale": "community",
                "community_id": 1,
                "full_data_direction": "tc_im_gt_tc_pt",
                "full_data_support_fraction": 0.9,
                "full_data_median_delta": 0.15,
                "robustness_call": "robust",
                "direction_recovery_rate": 0.75,
                "estimable_replicate_fraction": 0.8,
                "median_replicate_support_fraction": 0.7,
            },
            {
                "finding_id": "source_community::1::self_retention",
                "route_name": "leave_some_out",
                "route_group": "primary",
                "summary_scope": "source_community",
                "finding_priority": "primary",
                "summary_name": "self_retention",
                "scale": "community",
                "community_id": 1,
                "full_data_direction": "tc_im_gt_tc_pt",
                "full_data_support_fraction": 0.9,
                "full_data_median_delta": 0.15,
                "robustness_call": "failure",
                "direction_recovery_rate": 0.25,
                "estimable_replicate_fraction": 0.75,
                "median_replicate_support_fraction": 0.55,
            },
        ]
    )

    summary = build_block2_summary(
        family_robustness_df=family_robustness_df,
        source_robustness_df=source_robustness_df,
        target_robustness_df=pd.DataFrame(),
    )

    assert set(summary["block"].astype(str)) == {BLOCK_NAME}
    assert set(summary["summary_scope"].astype(str)) == {"family", "source_community"}
    family_row = summary.loc[summary["summary_scope"] == "family"].iloc[0]
    assert family_row["overall_robustness_call"] == "robust"
    assert int(family_row["primary_routes_robust"]) == 2
    source_row = summary.loc[summary["summary_scope"] == "source_community"].iloc[0]
    assert source_row["overall_robustness_call"] == "failure"
    assert float(source_row["worst_direction_recovery_rate"]) == pytest.approx(0.25)


def test_block2_bundle_reads_block1_bundle_contract_and_writes_robustness_outputs(
    tmp_path: Path,
) -> None:
    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(
        tmp_path / "config.yaml",
        enabled_blocks=[
            "block0_locality_gate",
            "block1_continuity_backbone",
            "block2_bounded_audit",
        ],
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

    contract = load_block1_bundle_contract(block1_bundle_path)
    assert contract.block == "block1_continuity_backbone"
    assert contract.scientific_role == "real_data_biological_discovery"
    assert contract.artifact_state == "evidence_ready"
    assert contract.implementation_tier == "canonical_full"
    assert contract.evidence_lineage == "canonical_rerun"
    assert contract.fit_surface == "fit_stride"
    assert contract.block0_gate_status == "passed"
    assert contract.confirmatory_pair_families == ("TC-IM", "TC-PT")
    assert contract.recurrence_summary_path is not None and contract.recurrence_summary_path.exists()
    assert contract.recurrence_families_path is not None and contract.recurrence_families_path.exists()
    assert contract.recurrence_embeddings_path is not None and contract.recurrence_embeddings_path.exists()
    assert contract.family_summary_path.exists()
    assert contract.source_community_summary_path.exists()
    assert contract.target_community_summary_path.exists()
    assert contract.confirmatory_family_comparison_path.exists()
    assert contract.exploratory_source_community_comparison_path.exists()
    assert contract.exploratory_target_community_comparison_path.exists()

    manifest_path = write_block2_bundle(
        block1_bundle_path=block1_bundle_path,
        output_dir=tmp_path / "block2",
    )
    assert manifest_path.exists()
    assert (tmp_path / "block2" / "block2_bounded_audit_summary.csv").exists()
    assert (tmp_path / "block2" / "block2_contract_audit.csv").exists()
    assert (tmp_path / "block2" / "block2_replicate_manifest.csv").exists()
    assert (tmp_path / "block2" / "block2_family_robustness.csv").exists()
    assert (tmp_path / "block2" / "block2_source_community_robustness.csv").exists()
    assert (tmp_path / "block2" / "block2_target_community_robustness.csv").exists()

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["artifact_state"] == "evidence_ready"
    assert manifest_payload["scientific_role"] == "robustness_over_block1_summaries"
    assert manifest_payload["claim_scope"] == "block1_summary_robustness"
    assert manifest_payload["implementation_tier"] == "canonical_full"
    assert manifest_payload["evidence_lineage"] == "canonical_rerun"
    assert manifest_payload["fit_surface"] == "fit_stride"
    assert manifest_payload["primary_routes"] == ["patient_subsample", "leave_some_out"]
    assert Path(manifest_payload["block1_recurrence_summary_path"]).exists()
    assert Path(manifest_payload["block1_recurrence_families_path"]).exists()
    assert Path(manifest_payload["block1_recurrence_embeddings_path"]).exists()
    assert Path(manifest_payload["block1_family_summary_path"]).exists()
    assert Path(manifest_payload["replicate_manifest_path"]).exists()
    assert Path(manifest_payload["family_robustness_path"]).exists()
    assert Path(manifest_payload["source_community_robustness_path"]).exists()
    assert Path(manifest_payload["target_community_robustness_path"]).exists()

    summary_df = pd.read_csv(manifest_payload["summary_path"])
    assert "overall_robustness_call" in summary_df.columns
    assert not summary_df.empty
    replicate_df = pd.read_csv(manifest_payload["replicate_manifest_path"])
    assert set(replicate_df["route_status"].astype(str)).issubset({"executed", "failed"})


def test_block2_bundle_rejects_disabled_block_in_config(tmp_path: Path) -> None:
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
    block1_bundle_path = run_block1_workflow(
        config_path=str(config_path),
        data_path=str(fixture_path),
        block0_bundle=str(block0_bundle_path),
        output_dir=str(tmp_path / "block1"),
    )

    with pytest.raises(ContractError, match="does not enable block2_bounded_audit"):
        write_block2_bundle(
            block1_bundle_path=block1_bundle_path,
            output_dir=tmp_path / "block2",
        )


def test_block2_bundle_rejects_proxy_tier_block1_bundle(tmp_path: Path) -> None:
    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(
        tmp_path / "config.yaml",
        enabled_blocks=[
            "block0_locality_gate",
            "block1_continuity_backbone",
            "block2_bounded_audit",
        ],
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

    payload = json.loads(Path(block1_bundle_path).read_text(encoding="utf-8"))
    payload["implementation_tier"] = "proxy_tier"
    Path(block1_bundle_path).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="canonical-full Block 1 outputs"):
        write_block2_bundle(
            block1_bundle_path=block1_bundle_path,
            output_dir=tmp_path / "block2",
        )


def test_block2_bundle_rejects_missing_block1_recurrence_status_contract(tmp_path: Path) -> None:
    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(
        tmp_path / "config.yaml",
        enabled_blocks=[
            "block0_locality_gate",
            "block1_continuity_backbone",
            "block2_bounded_audit",
        ],
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

    payload = json.loads(Path(block1_bundle_path).read_text(encoding="utf-8"))
    payload["cohort_recurrence_fit_status"] = ""
    payload["cohort_recurrence_fit_status_by_pair_family"] = {}
    Path(block1_bundle_path).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="cohort_recurrence_fit_status"):
        write_block2_bundle(
            block1_bundle_path=block1_bundle_path,
            output_dir=tmp_path / "block2",
        )


def test_block2_bundle_can_resume_interrupted_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tasks.task_A.block2.bundle as block2_bundle_module

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(
        tmp_path / "config.yaml",
        enabled_blocks=[
            "block0_locality_gate",
            "block1_continuity_backbone",
            "block2_bounded_audit",
        ],
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
    block2_output_dir = tmp_path / "block2_resume"

    real_runner = block2_bundle_module.run_block1_reestimate_for_block2
    interrupted_call_count = {"value": 0}

    def interrupting_runner(*args, **kwargs):  # type: ignore[no-untyped-def]
        interrupted_call_count["value"] += 1
        if interrupted_call_count["value"] == 2:
            raise RuntimeError("synthetic_interrupt")
        return real_runner(*args, **kwargs)

    monkeypatch.setattr(block2_bundle_module, "run_block1_reestimate_for_block2", interrupting_runner)

    with pytest.raises(RuntimeError, match="synthetic_interrupt"):
        write_block2_bundle(
            block1_bundle_path=block1_bundle_path,
            output_dir=block2_output_dir,
            resume=True,
        )

    checkpoint_manifest_path = block2_output_dir / ".block2_resume_replicate_manifest.csv"
    checkpoint_meta_path = block2_output_dir / ".block2_resume_meta.json"
    checkpoint_assessments_path = block2_output_dir / ".block2_resume_assessments.json"
    assert checkpoint_manifest_path.exists()
    assert checkpoint_meta_path.exists()
    assert checkpoint_assessments_path.exists()

    checkpoint_manifest_df = pd.read_csv(checkpoint_manifest_path, keep_default_na=False)
    pending_replicates = int((checkpoint_manifest_df["route_status"].astype(str) == "pending").sum())
    assert pending_replicates > 0
    assert int((checkpoint_manifest_df["route_status"].astype(str) == "executed").sum()) >= 1

    resumed_call_count = {"value": 0}

    def counting_runner(*args, **kwargs):  # type: ignore[no-untyped-def]
        resumed_call_count["value"] += 1
        return real_runner(*args, **kwargs)

    monkeypatch.setattr(block2_bundle_module, "run_block1_reestimate_for_block2", counting_runner)

    manifest_path = write_block2_bundle(
        block1_bundle_path=block1_bundle_path,
        output_dir=block2_output_dir,
        resume=True,
    )

    assert manifest_path.exists()
    assert resumed_call_count["value"] == pending_replicates
    assert not checkpoint_manifest_path.exists()
    assert not checkpoint_meta_path.exists()
    assert not checkpoint_assessments_path.exists()

    replicate_df = pd.read_csv(block2_output_dir / "block2_replicate_manifest.csv")
    assert "pending" not in set(replicate_df["route_status"].astype(str))
