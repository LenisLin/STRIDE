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

ANNDATA_AVAILABLE = importlib.util.find_spec("anndata") is not None
pytestmark = pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not installed")

TASK_CONFIG = ROOT / "tasks" / "task_A" / "config.yaml"


def _write_stage0_fixture(tmp_path: Path) -> Path:
    from tests.helpers_task_a_fixture import write_task_a_fixture

    return write_task_a_fixture(tmp_path / "stage0.h5ad")


def test_descriptive_atlas_is_stage0_entrypoint_not_workflow_export() -> None:
    import tasks.task_A.descriptive
    import tasks.task_A.workflows

    assert hasattr(tasks.task_A.descriptive, "write_task_a_descriptive_atlas")
    assert not hasattr(tasks.task_A.workflows, "write_task_a_descriptive_atlas")


def test_descriptive_atlas_writes_manifest_and_index_from_stage0(tmp_path: Path) -> None:
    from tests.helpers_task_a_fixture import K_FULL
    from tasks.task_A.descriptive import write_task_a_descriptive_atlas

    stage0_path = _write_stage0_fixture(tmp_path)
    output_dir = tmp_path / "atlas"
    manifest = write_task_a_descriptive_atlas(
        config_path=TASK_CONFIG,
        stage0_h5ad=stage0_path,
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
    assert written_manifest["claim_scope"] == "descriptive_only"
    assert written_manifest["scientific_interpretation_allowed"] is False
    assert written_manifest["config_path"] == str(TASK_CONFIG.resolve())
    assert written_manifest["stage0_h5ad"] == str(stage0_path.resolve())
    assert written_manifest["patient_id_key"] == "patient_id"
    assert written_manifest["fov_key"] == "roi_id"
    assert written_manifest["domain_key"] == "compartment"
    assert written_manifest["cell_subtype_key"] == "cell_type"
    assert written_manifest["community_id_key"] == "proto_id"
    assert written_manifest["spatial_key"] == "spatial"
    assert written_manifest["configured_community_ids"] == list(range(K_FULL))
    assert written_manifest["observed_community_ids"] == list(range(8))
    assert written_manifest["patient_ids"] == ["P01", "P02"]
    assert written_manifest["n_patients"] == 2
    assert written_manifest["n_observed_communities"] == 8

    removed_manifest_fields = {
        "artifact_state",
        "block0_gate_status",
        "prepare_manifest_path",
        "mapping_manifest_path",
        "fit_surface",
        "core_fit_dry_run",
        "implementation_tier",
        "evidence_lineage",
    }
    assert removed_manifest_fields.isdisjoint(written_manifest)

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


def test_descriptive_atlas_subset_and_overlay_selection_are_deterministic(tmp_path: Path) -> None:
    from tasks.task_A.descriptive import write_task_a_descriptive_atlas

    stage0_path = _write_stage0_fixture(tmp_path)
    subset_output_dir = tmp_path / "atlas_subset"
    subset_written = write_task_a_descriptive_atlas(
        config_path=TASK_CONFIG,
        stage0_h5ad=stage0_path,
        output_dir=subset_output_dir,
        patient_ids=("P01",),
        max_overlay_communities=2,
    )
    assert subset_written["patient_ids"] == ["P01"]
    assert subset_written["n_patients"] == 1

    full_output_dir = tmp_path / "atlas_full"
    write_task_a_descriptive_atlas(
        config_path=TASK_CONFIG,
        stage0_h5ad=stage0_path,
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


def test_descriptive_atlas_rejects_missing_requested_patient_ids(tmp_path: Path) -> None:
    from tasks.task_A.descriptive import DescriptiveAtlasContractError, write_task_a_descriptive_atlas

    stage0_path = _write_stage0_fixture(tmp_path)
    with pytest.raises(DescriptiveAtlasContractError, match="missing.*MISSING"):
        write_task_a_descriptive_atlas(
            config_path=TASK_CONFIG,
            stage0_h5ad=stage0_path,
            output_dir=tmp_path / "atlas_missing_patient",
            patient_ids=("P01", "MISSING"),
            max_overlay_communities=1,
        )


def test_descriptive_atlas_rejects_unconfigured_stage0_community_ids(tmp_path: Path) -> None:
    import anndata as ad
    from tests.helpers_task_a_fixture import K_FULL
    from tasks.task_A.descriptive import DescriptiveAtlasContractError, write_task_a_descriptive_atlas

    stage0_path = _write_stage0_fixture(tmp_path)
    adata = ad.read_h5ad(stage0_path)
    adata.obs.loc[adata.obs.index[0], "proto_id"] = K_FULL
    invalid_stage0_path = tmp_path / "stage0_invalid_community.h5ad"
    adata.write_h5ad(invalid_stage0_path)

    with pytest.raises(DescriptiveAtlasContractError, match="unconfigured.*community.*25"):
        write_task_a_descriptive_atlas(
            config_path=TASK_CONFIG,
            stage0_h5ad=invalid_stage0_path,
            output_dir=tmp_path / "atlas_invalid_community",
            max_overlay_communities=1,
        )


def test_descriptive_package_has_no_stride_or_prepare_import_boundary() -> None:
    package_root = ROOT / "tasks" / "task_A" / "descriptive"
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(package_root.glob("*.py"))
    )
    forbidden_tokens = {
        "from stride",
        "import stride",
        "fit_stride",
        "prepare_task_a_stage0_mapping",
        "core_fit_dry_run",
        "load_task_a_dataset_handle",
    }
    for token in forbidden_tokens:
        assert token not in combined
