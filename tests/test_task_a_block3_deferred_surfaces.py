from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stride.errors import ContractError


def test_run_block3_workflow_is_fail_fast_deprecated() -> None:
    from tasks.task_A.workflows.run_block3 import (
        DEPRECATION_MESSAGE,
        run_block3_workflow,
    )

    with pytest.raises(ContractError, match="retired from the active path"):
        run_block3_workflow(
            block2_manifest="/tmp/block2_bounded_audit_manifest.json",
            output_dir="/tmp/task_a_block3",
        )

    assert "public Block 3 runner is retired" in DEPRECATION_MESSAGE
    assert "internal non-authority Phase 3 package remains on disk" in DEPRECATION_MESSAGE


def test_workflow_public_exports_drop_active_block3_runner() -> None:
    import tasks.task_A.workflows as workflow_module

    assert "run_block3_workflow" not in set(workflow_module.__all__)
    assert "write_block3_review" not in set(workflow_module.__all__)


def test_result_packet_builder_rejects_block3_manifest_directly(tmp_path: Path) -> None:
    from tasks.task_A.result_packet import write_task_a_result_packet
    from tests.test_task_a_result_packet import _write_atlas_bundle, _write_block0_run, _write_json

    atlas_manifest_path = _write_atlas_bundle(tmp_path / "atlas_source")
    prepare_manifest_path, block0_bundle_path, suitability_path = _write_block0_run(
        tmp_path / "block0_run"
    )
    block3_manifest_path = _write_json(
        tmp_path / "block3" / "block3_method_validation_manifest.json",
        {"block": "block3_method_validation", "artifact_state": "evidence_ready"},
    )

    with pytest.raises(
        ContractError,
        match="Block 3 packet integration is deferred / non-authority / pending clean bridge spec",
    ):
        write_task_a_result_packet(
            atlas_manifest_path=atlas_manifest_path,
            prepare_manifest_path=prepare_manifest_path,
            block0_bundle_path=block0_bundle_path,
            block0_suitability_report_path=suitability_path,
            block3_manifest_path=block3_manifest_path,
            output_dir=tmp_path / "packet",
        )
