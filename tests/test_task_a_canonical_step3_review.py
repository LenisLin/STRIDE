from __future__ import annotations

# ruff: noqa: E402, I001

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stride.errors import ContractError

from tasks.task_A.workflows.review_canonical_step3 import (
    CanonicalStep3Inputs,
    ProxyHistoryInputs,
    write_canonical_step3_review,
)
from tests.test_task_a_result_packet import (
    _write_atlas_bundle,
    _write_block1_bundle,
    _write_block2_manifest,
    _write_current_contract_block0_run,
    _write_csv,
    _write_json,
    _write_prepare_bundle,
)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _rewrite_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _drop_keys(path: Path, *, keys: tuple[str, ...]) -> None:
    payload = _load_json(path)
    for key in keys:
        payload.pop(key, None)
    _rewrite_json(path, payload)


def _patch_field(path: Path, *, key: str, value: object) -> None:
    payload = _load_json(path)
    payload[key] = value
    _rewrite_json(path, payload)


def _build_canonical_inputs(tmp_path: Path) -> CanonicalStep3Inputs:
    prepare_manifest_path = _write_prepare_bundle(tmp_path / "canonical_prepare")
    atlas_manifest_path = _write_atlas_bundle(tmp_path / "canonical_atlas")

    _, block0_bundle_path, suitability_path = _write_current_contract_block0_run(
        tmp_path / "canonical_block0"
    )
    block1_bundle_path = _write_block1_bundle(tmp_path / "canonical_block1")
    _patch_field(
        block1_bundle_path,
        key="block0_bundle_path",
        value=str(block0_bundle_path),
    )
    _patch_field(
        block1_bundle_path.parent / "block1_workflow_manifest.json",
        key="block0_bundle_path",
        value=str(block0_bundle_path),
    )
    block2_manifest_path = _write_block2_manifest(
        tmp_path / "canonical_block2",
        block1_bundle_path=block1_bundle_path,
    )
    block2_payload = _load_json(block2_manifest_path)
    block2_payload["executed_replicates"] = 1
    block2_payload["failed_replicates"] = 0
    _rewrite_json(block2_manifest_path, block2_payload)
    return CanonicalStep3Inputs(
        canonical_root=tmp_path / "canonical_root",
        prepare_manifest_path=prepare_manifest_path,
        atlas_manifest_path=atlas_manifest_path,
        block0_bundle_path=block0_bundle_path,
        block0_suitability_report_path=suitability_path,
        block1_bundle_path=block1_bundle_path,
        block2_manifest_path=block2_manifest_path,
    )


def _build_proxy_inputs(tmp_path: Path) -> ProxyHistoryInputs:
    proxy_root = tmp_path / "proxy_history"
    _, proxy_block0_bundle_path, _ = _write_current_contract_block0_run(proxy_root / "block0")
    _drop_keys(
        proxy_block0_bundle_path,
        keys=("implementation_tier", "evidence_lineage"),
    )
    proxy_block1_bundle_path = _write_block1_bundle(proxy_root / "block1")
    proxy_block2_manifest_path = _write_block2_manifest(
        proxy_root / "block2",
        block1_bundle_path=proxy_block1_bundle_path,
    )
    for path in (
        proxy_block1_bundle_path,
        proxy_block1_bundle_path.parent / "block1_workflow_manifest.json",
    ):
        _drop_keys(
            path,
            keys=(
                "implementation_tier",
                "evidence_lineage",
                "fit_surface",
                "recurrence_summary_path",
                "recurrence_families_path",
                "recurrence_embeddings_path",
                "cohort_recurrence_fit_status",
                "cohort_recurrence_fit_status_by_pair_family",
                "cohort_recurrence_family_count",
                "cohort_recurrence_family_count_by_pair_family",
                "n_recurrence_used_patients",
                "n_recurrence_used_patients_by_pair_family",
            ),
        )
    for filename in (
        "block1_recurrence_summary.json",
        "block1_recurrence_families.json",
        "block1_recurrence_embeddings.csv",
    ):
        candidate = proxy_block1_bundle_path.parent / filename
        if candidate.exists():
            candidate.unlink()
    _drop_keys(
        proxy_block2_manifest_path,
        keys=(
            "implementation_tier",
            "evidence_lineage",
            "fit_surface",
            "block1_cohort_recurrence_fit_status",
            "block1_cohort_recurrence_fit_status_by_pair_family",
            "block1_cohort_recurrence_family_count",
            "block1_cohort_recurrence_family_count_by_pair_family",
            "block1_n_recurrence_used_patients",
            "block1_n_recurrence_used_patients_by_pair_family",
            "block1_recurrence_summary_path",
            "block1_recurrence_families_path",
            "block1_recurrence_embeddings_path",
        ),
    )

    stale_packet_root = tmp_path / "result_packets" / "2026-04-05_block2_objective_review_packet"
    stale_index_path = _write_csv(
        stale_packet_root / "task_a_result_packet_index.csv",
        [
            {
                "layer": "atlas",
                "artifact_name": "task_a_descriptive_atlas_manifest.json",
                "expected_relative_path": "task_a_descriptive_atlas_manifest.json",
                "packet_relative_path": "atlas/bundle/task_a_descriptive_atlas_manifest.json",
                "source_path": str(proxy_root / "atlas" / "task_a_descriptive_atlas_manifest.json"),
                "artifact_kind": "manifest",
                "artifact_status": "available",
                "contract_alignment": "legacy_packet",
                "format": "json",
                "n_rows": 1,
                "n_columns": 3,
                "observed_columns": "",
                "rows_represent": "Legacy proxy packet manifest row",
                "columns_represent": "Legacy proxy packet fields",
                "claim_scope": "provenance",
                "review_role": "provenance",
                "analysis_level": "mixed",
                "family_surface_role": "not_applicable",
                "is_proof_carrying": False,
                "proof_carrying_status": "none",
                "source_workflow": "legacy_packet",
                "source_manifest_or_bundle": str(stale_packet_root / "task_a_result_packet_manifest.json"),
                "sha256": "",
                "notes": "legacy stale packet",
                "review_rank": 1,
            }
        ],
    )
    stale_manifest_path = _write_json(
        stale_packet_root / "task_a_result_packet_manifest.json",
        {
            "workflow_name": "write_task_a_result_packet",
            "packet_role": "objective_task_a_result_packet",
            "packet_root": str(stale_packet_root),
            "central_index_path": str(stale_index_path),
            "input_sources": {
                "block1_bundle_path": str(proxy_block1_bundle_path),
                "block2_manifest_path": str(proxy_block2_manifest_path),
            },
        },
    )
    return ProxyHistoryInputs(
        proxy_root=proxy_root,
        block0_bundle_path=proxy_block0_bundle_path,
        block1_bundle_path=proxy_block1_bundle_path,
        block2_manifest_path=proxy_block2_manifest_path,
        stale_packet_manifest_path=stale_manifest_path,
        stale_packet_index_path=stale_index_path,
    )


def test_write_canonical_step3_review_builds_packet_and_audit_outputs(tmp_path: Path) -> None:
    canonical_inputs = _build_canonical_inputs(tmp_path)
    proxy_inputs = _build_proxy_inputs(tmp_path)

    review = write_canonical_step3_review(
        canonical_inputs=canonical_inputs,
        proxy_inputs=proxy_inputs,
        packet_output_dir=tmp_path / "result_packets" / "2026-04-06_canonical_step3_objective_packet",
        audit_output_dir=tmp_path / "result_packets" / "2026-04-06_canonical_step3_audit",
        results_packet_root=tmp_path / "result_packets",
    )

    assert review.packet.manifest_path.exists()
    assert review.packet.index_path.exists()
    assert review.completion_status_path.exists()
    assert review.primary_review_index_path.exists()
    assert review.delta_index_path.exists()
    assert review.block0_review_table_copy_path.exists()
    assert review.block1_review_index_copy_path.exists()
    assert review.block2_review_index_copy_path.exists()

    status_payload = _load_json(review.completion_status_path)
    assert status_payload["answers"]["did_block2_finish"] is True
    assert status_payload["answers"]["is_block2_manifest_valid"] is True
    assert status_payload["answers"]["step3_complete_before_run"] is False
    assert status_payload["answers"]["step3_complete_after_run"] is True
    assert status_payload["answers"]["step3_packet_action"] == "generated_now"

    primary_df = pd.read_csv(review.primary_review_index_path)
    assert "evidence_surface_class" in primary_df.columns
    assert "proof-carrying" in set(primary_df["evidence_surface_class"].astype(str))
    assert "descriptive" in set(primary_df["evidence_surface_class"].astype(str))

    delta_df = pd.read_csv(review.delta_index_path)
    recurrence_row = delta_df.loc[
        delta_df["artifact_name"].astype(str) == "block1_recurrence_summary.json"
    ].iloc[0]
    assert str(recurrence_row["proxy_present"]).lower() == "false"
    assert str(recurrence_row["canonical_present"]).lower() == "true"
    assert recurrence_row["difference_category"] == "added_surface"

    packet_manifest_payload = _load_json(review.packet.manifest_path)
    assert packet_manifest_payload["included_layers"] == ["atlas", "block0", "block1", "block2"]
    assert packet_manifest_payload["deferred_layers"] == ["block3"]


def test_write_canonical_step3_review_rejects_noncanonical_block2_lineage(tmp_path: Path) -> None:
    canonical_inputs = _build_canonical_inputs(tmp_path)
    proxy_inputs = _build_proxy_inputs(tmp_path)
    _patch_field(
        canonical_inputs.block2_manifest_path,
        key="evidence_lineage",
        value="proxy_history",
    )

    with pytest.raises(ContractError, match="block2_evidence_lineage_canonical_rerun"):
        write_canonical_step3_review(
            canonical_inputs=canonical_inputs,
            proxy_inputs=proxy_inputs,
            packet_output_dir=tmp_path / "result_packets" / "2026-04-06_canonical_step3_objective_packet",
            audit_output_dir=tmp_path / "result_packets" / "2026-04-06_canonical_step3_audit",
            results_packet_root=tmp_path / "result_packets",
        )
