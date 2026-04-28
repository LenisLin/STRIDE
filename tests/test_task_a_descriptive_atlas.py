from __future__ import annotations

# ruff: noqa: E402, I001

import importlib.util
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

ANNDATA_AVAILABLE = importlib.util.find_spec("anndata") is not None
pytestmark = pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not installed")


def _write_prepare_manifest(
    tmp_path: Path,
    *,
    patient_ids: tuple[str, ...] | None = None,
) -> tuple[Path, dict[str, object]]:
    from tasks.task_A.workflows.prepare import prepare_task_a_stage0_mapping
    from tests.helpers_task_a_fixture import write_task_a_fixture

    stage0_path = write_task_a_fixture(tmp_path / "stage0.h5ad")
    prepare_dir = tmp_path / "prepare"
    manifest = prepare_task_a_stage0_mapping(
        config_path=ROOT / "tasks" / "task_A" / "config.yaml",
        data_path=stage0_path,
        output_dir=prepare_dir,
        patient_ids=patient_ids,
    )
    return prepare_dir / "task_a_prepare_manifest.json", manifest


def test_descriptive_atlas_workflow_is_exported() -> None:
    import tasks.task_A.workflows

    assert hasattr(tasks.task_A.workflows, "write_task_a_descriptive_atlas")


def test_descriptive_atlas_writes_manifest_and_index(tmp_path: Path) -> None:
    from tasks.task_A.workflows.descriptive_atlas import write_task_a_descriptive_atlas

    prepare_manifest_path, _prepare_manifest = _write_prepare_manifest(tmp_path)
    output_dir = tmp_path / "atlas"
    manifest = write_task_a_descriptive_atlas(
        prepare_manifest_path=prepare_manifest_path,
        output_dir=output_dir,
        max_overlay_communities=2,
    )

    manifest_path = output_dir / "task_a_descriptive_atlas_manifest.json"
    output_index_path = output_dir / "task_a_descriptive_atlas_output_index.csv"
    assert manifest_path.exists()
    assert output_index_path.exists()

    written_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["atlas_role"] == "descriptive_only"
    assert written_manifest["atlas_role"] == "descriptive_only"
    assert written_manifest["scientific_interpretation_allowed"] is False
    assert written_manifest["artifact_state"] == "contract_passed"
    assert written_manifest["run_scope"] == "full_cohort_alignment_check"
    assert written_manifest["n_patients"] == 2
    assert written_manifest["n_observed_communities"] == 8

    output_index = pd.read_csv(output_index_path)
    expected_paths = {
        "task_a_descriptive_atlas_manifest.json",
        "task_a_descriptive_atlas_output_index.csv",
        "tables/community_cell_subtype_counts.csv",
        "tables/community_cell_subtype_row_fractions.csv",
        "tables/community_domain_distribution.csv",
        "tables/community_domain_roi_prevalence.csv",
        "tables/community_patient_occurrence_summary.csv",
        "tables/community_patient_occurrence_matrix.csv",
        "tables/representative_overlay_selection.csv",
        "figures/community_by_cell_subtype_heatmap.svg",
        "figures/community_domain_abundance_heatmap.svg",
        "figures/community_domain_roi_prevalence_heatmap.svg",
        "figures/patient_level_community_prevalence.svg",
        "figures/representative_spatial_overlays/community_04_overlay.svg",
        "figures/representative_spatial_overlays/community_05_overlay.svg",
    }
    assert expected_paths.issubset(set(output_index["relative_path"].astype(str)))
    for relative_path in output_index["relative_path"].astype(str):
        assert (output_dir / relative_path).exists(), f"Indexed atlas artifact missing: {relative_path}"


def test_descriptive_atlas_rejects_missing_prepare_manifest_fields(tmp_path: Path) -> None:
    from tasks.task_A.workflows.descriptive_atlas import write_task_a_descriptive_atlas

    prepare_manifest_path, _prepare_manifest = _write_prepare_manifest(tmp_path)
    broken_path = tmp_path / "broken_prepare_manifest.json"
    payload = json.loads(prepare_manifest_path.read_text(encoding="utf-8"))
    payload.pop("mapping_manifest")
    broken_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ContractError, match="missing required fields"):
        write_task_a_descriptive_atlas(
            prepare_manifest_path=broken_path,
            output_dir=tmp_path / "atlas_broken",
        )


def test_descriptive_atlas_rejects_non_descriptive_prepare_manifest(tmp_path: Path) -> None:
    from tasks.task_A.workflows.descriptive_atlas import write_task_a_descriptive_atlas

    prepare_manifest_path, _prepare_manifest = _write_prepare_manifest(tmp_path)
    dishonest_path = tmp_path / "dishonest_prepare_manifest.json"
    payload = json.loads(prepare_manifest_path.read_text(encoding="utf-8"))
    payload["scientific_interpretation_allowed"] = True
    dishonest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ContractError, match="scientific_interpretation_allowed=false"):
        write_task_a_descriptive_atlas(
            prepare_manifest_path=dishonest_path,
            output_dir=tmp_path / "atlas_dishonest",
        )


def test_descriptive_atlas_subset_and_overlay_selection_are_deterministic(tmp_path: Path) -> None:
    from tasks.task_A.workflows.descriptive_atlas import write_task_a_descriptive_atlas

    subset_manifest_path, _prepare_manifest = _write_prepare_manifest(tmp_path / "subset", patient_ids=("P01",))
    subset_output_dir = tmp_path / "atlas_subset"
    subset_written = write_task_a_descriptive_atlas(
        prepare_manifest_path=subset_manifest_path,
        output_dir=subset_output_dir,
        max_overlay_communities=2,
    )
    assert subset_written["run_scope"] == "patient_subset"
    assert subset_written["patient_subset"] == ["P01"]
    assert subset_written["n_patients"] == 1

    full_manifest_path, _prepare_manifest = _write_prepare_manifest(tmp_path / "full")
    full_output_dir = tmp_path / "atlas_full"
    write_task_a_descriptive_atlas(
        prepare_manifest_path=full_manifest_path,
        output_dir=full_output_dir,
        max_overlay_communities=2,
    )
    selection = pd.read_csv(full_output_dir / "tables" / "representative_overlay_selection.csv")
    selection = selection.sort_values("community_total_cells", ascending=False, kind="mergesort").reset_index(drop=True)

    assert selection.loc[0, "community_id"] == 5
    assert selection.loc[0, "patient_id"] == "P01"
    assert selection.loc[0, "domain_label"] == "PT"
    assert selection.loc[0, "fov_id"] == "P01_PT_01"
    assert selection.loc[0, "community_cells"] == 4
    assert selection.loc[0, "roi_total_cells"] == 4
    assert selection.loc[0, "community_fraction_in_roi"] == pytest.approx(1.0)

    assert selection.loc[1, "community_id"] == 4
    assert selection.loc[1, "patient_id"] == "P01"
    assert selection.loc[1, "domain_label"] == "IM"
    assert selection.loc[1, "fov_id"] == "P01_IM_02"
    assert selection.loc[1, "community_fraction_in_roi"] == pytest.approx(0.75)
