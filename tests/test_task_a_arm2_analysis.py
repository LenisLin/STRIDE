from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tasks.task_A.analyze_arm2_bioinformed import build_paths_from_args as build_bio_paths
import tasks.task_A.analyze_arm2_results as focused_results_mod
from tasks.task_A.analyze_arm2_results import build_paths_from_args as build_focused_paths
from tasks.task_A.analyze_arm2_results import discover_arm2_parquet
from tasks.task_A.arm2.analysis_contract import ARM_NAME
from tasks.task_A.runtime_contract import (
    TASK_A_MANIFEST_SCHEMA_VERSION,
    resolve_task_a_arm_bioinformed_output_dir,
    resolve_task_a_arm_focused_output_dir,
)


def _write_manifest(run_root: Path) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    metrics_path = run_root / "task_A_metrics.parquet"
    pd.DataFrame([{"arm": ARM_NAME, "pair_id": "p1"}]).to_parquet(metrics_path, index=False)

    stage0_path = run_root / "fixture.h5ad"
    stage0_path.write_text("placeholder\n", encoding="utf-8")

    config_path = run_root / "config.yaml"
    config_path.write_text("data:\n  k_full: 25\n", encoding="utf-8")

    analysis_root = (run_root / "analysis").resolve()
    manifest_path = run_root / "task_a_run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": TASK_A_MANIFEST_SCHEMA_VERSION,
                "task_name": "Task A",
                "config_path": str(config_path.resolve()),
                "stage0_h5ad": str(stage0_path.resolve()),
                "run_root": str(run_root.resolve()),
                "metrics_parquet": str(metrics_path.resolve()),
                "enabled_arms": [ARM_NAME],
                "arm_artifact_roots": {ARM_NAME: str(run_root.resolve())},
                "arm_analysis_roots": {ARM_NAME: str(analysis_root)},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_build_focused_paths_prefers_task_a_manifest(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "arm2_run")
    args = Namespace(
        task_a_manifest=str(manifest_path),
        task_a_run_root=None,
        input_parquet=None,
        search_root=str(tmp_path),
        stage0_h5ad=None,
        task_config=None,
        output_dir=None,
        prototype_view_ids=[5, 2, 5],
    )

    paths = build_focused_paths(args)

    assert paths.arm2_metrics_parquet == (manifest_path.parent / "task_A_metrics.parquet").resolve()
    assert paths.stage0_h5ad == (manifest_path.parent / "fixture.h5ad").resolve()
    assert paths.task_config == (manifest_path.parent / "config.yaml").resolve()
    assert paths.output_dir == resolve_task_a_arm_focused_output_dir(
        type("ManifestLike", (), {"arm_analysis_roots": {ARM_NAME: (manifest_path.parent / "analysis").resolve()}})(),
        ARM_NAME,
    )
    assert paths.prototype_view_ids == (2, 5)


def test_build_bio_paths_prefers_task_a_run_root(tmp_path: Path) -> None:
    run_root = tmp_path / "arm2_run"
    _write_manifest(run_root)
    args = Namespace(
        task_a_manifest=None,
        task_a_run_root=str(run_root),
        arm2_metrics_parquet=None,
        stage0_h5ad=None,
        task_config=None,
        output_dir=None,
    )

    paths = build_bio_paths(args)

    assert paths.arm2_metrics_parquet == (run_root / "task_A_metrics.parquet").resolve()
    assert paths.stage0_h5ad == (run_root / "fixture.h5ad").resolve()
    assert paths.task_config == (run_root / "config.yaml").resolve()
    assert paths.output_dir == resolve_task_a_arm_bioinformed_output_dir(
        type("ManifestLike", (), {"arm_analysis_roots": {ARM_NAME: (run_root / "analysis").resolve()}})(),
        ARM_NAME,
    )


def test_discover_arm2_parquet_requires_explicit_path_when_search_is_ambiguous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(focused_results_mod, "DEFAULT_ARM2_CANDIDATES", ())
    for idx in range(2):
        path = tmp_path / f"candidate_{idx}.parquet"
        pd.DataFrame([{"arm": ARM_NAME, "pair_id": f"p{idx}"}]).to_parquet(path, index=False)

    with pytest.raises(FileExistsError, match="Multiple Arm-II parquets"):
        discover_arm2_parquet(None, tmp_path)
