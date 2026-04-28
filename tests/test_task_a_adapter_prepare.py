"""Tests for Task-A stride_adapter crosswalk, ordered-group semantics,
and the prepare workflow CLI entrypoint.

Fixture-based tests use ``build_task_a_fixture`` (tasks/tests/helpers_task_a_fixture.py)
which produces an AnnData with the canonical Stage-0 field layout used by the real
cohort (patient_id, timepoint=0, roi_id, compartment, cell_type, proto_id, obsm[spatial],
uns[cost_matrix], uns[s_C]).
"""
from __future__ import annotations

# ruff: noqa: E402, I001

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_minimal_config(path: Path) -> Path:
    config = {
        "task_name": "adapter-prepare test",
        "enabled_blocks": ["block0_locality_gate"],
        "data": {"k_full": 25, "mass_mode": "uniform"},
        "observation_match": {
            "eps_schedule": [1.0, 0.5],
            "max_iter": 200,
            "tol": 1.0e-8,
            "eta_floor": 1.0e-12,
            "n_min_proto": 0.0,
        },
        "block0": {
            "n_draws": 1,
            "random_seed": 42,
            "match_penalty_by_compartment": {"TC": 1.0, "IM": 1.5, "PT": 2.0},
            "retention_threshold_by_compartment": {"TC": 0.4, "IM": 0.6, "PT": 0.8},
        },
        "block1": {
            "target_alpha": 0.05,
            "lambda_grid": [0.1, 1.0, 10.0],
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Crosswalk field assertions
# ---------------------------------------------------------------------------


def test_adapter_crosswalk_has_expected_raw_keys() -> None:
    """``build_task_a_real_data_crosswalk`` should resolve every alias correctly
    for a fixture AnnData whose uns uses ``s_C`` and obs uses ``roi_id`` /
    ``compartment`` / ``cell_type`` / ``proto_id``."""
    from tests.helpers_task_a_fixture import build_task_a_fixture
    from tasks.task_A.workflows.stride_adapter import (
        build_task_a_real_data_crosswalk,
        load_task_a_dataset_handle,
    )

    adata = build_task_a_fixture()
    handle = load_task_a_dataset_handle(adata)
    crosswalk = build_task_a_real_data_crosswalk(handle)

    # obs-layer direct / alias mappings
    assert crosswalk.patient_id_raw == "patient_id"
    assert crosswalk.patient_id_canonical == "patient_id"
    assert crosswalk.patient_id_mapping == "direct"

    assert crosswalk.fov_raw == handle.fov_key        # "roi_id" in fixture
    assert crosswalk.fov_canonical == "fov_id"
    assert crosswalk.fov_mapping == "alias"           # roi_id != fov_id

    assert crosswalk.domain_raw == handle.domain_key  # "compartment"
    assert crosswalk.domain_canonical == "domain_label"
    assert crosswalk.domain_mapping == "alias"

    assert crosswalk.cell_subtype_raw == handle.cell_subtype_key
    assert crosswalk.cell_subtype_canonical == "cell_subtype_label"

    assert crosswalk.state_id_raw == handle.state_id_key
    assert crosswalk.state_id_canonical == "state_id"

    # uns-layer: fixture uses s_C, not cost_scale → alias
    assert crosswalk.cost_scale_raw == "s_C"
    assert crosswalk.cost_scale_canonical == "cost_scale"
    assert crosswalk.cost_scale_mapping == "alias"

    assert crosswalk.cost_matrix_key == "cost_matrix"
    assert crosswalk.cost_matrix_mapping == "direct"

    assert crosswalk.spatial_key == "spatial"
    assert crosswalk.spatial_mapping == "direct"

    # derived semantics
    assert crosswalk.mass_value == 1.0
    assert crosswalk.mass_mode == "uniform"

    # inert raw timepoint: fixture has timepoint=0 only
    assert crosswalk.timepoint_inert is True
    assert crosswalk.timepoint_raw_observed_values == ("0",)

    # ordered-group derivation source must be compartment
    assert crosswalk.ordered_group_source == "compartment"
    assert crosswalk.ordered_group_mapping == "derived"

    # explicitly unmapped fields must be declared
    assert "block_id" in crosswalk.unmapped_obs_fields
    assert "cell_area" in crosswalk.unmapped_obs_fields

    # downstream-only non-canonical fields must be deferred, not mapped
    for deferred_field in ("comparison_id", "count_stratum_key", "real_fit_status", "null_fit_status"):
        assert deferred_field in crosswalk.deferred_downstream_fields


def test_adapter_crosswalk_json_roundtrip() -> None:
    """``to_json_dict()`` must be JSON-serialisable and preserve key entries."""
    from tests.helpers_task_a_fixture import build_task_a_fixture
    from tasks.task_A.workflows.stride_adapter import (
        build_task_a_real_data_crosswalk,
        load_task_a_dataset_handle,
    )

    adata = build_task_a_fixture()
    handle = load_task_a_dataset_handle(adata)
    crosswalk = build_task_a_real_data_crosswalk(handle)
    as_dict = crosswalk.to_json_dict()

    # Must be JSON-serialisable
    raw_json = json.dumps(as_dict)
    restored = json.loads(raw_json)

    # crosswalk entries should include every obs-layer field
    crosswalk_entries = restored["crosswalk"]
    raw_fields = {entry["raw"] for entry in crosswalk_entries}
    assert "patient_id" in raw_fields
    assert "roi_id" in raw_fields or "fov_id" in raw_fields
    assert "compartment" in raw_fields

    # derived semantics block
    assert restored["derived_semantics"]["mass"] == 1.0
    assert restored["derived_semantics"]["mass_mode"] == "uniform"

    # deferred downstream surfaces must be present
    assert "comparison_id" in restored["deferred_downstream_fields"]


# ---------------------------------------------------------------------------
# Ordered-group semantics: FovObservation.timepoint must come from compartment
# ---------------------------------------------------------------------------


def test_adapter_observations_use_domain_as_timepoint(tmp_path: Path) -> None:
    """``build_task_a_family_observations`` must set ``FovObservation.timepoint``
    from the compartment/domain field, NOT from the raw inert timepoint column.

    The fixture has ``obs.timepoint == 0`` for every cell; the ordered-group
    labels for TC-IM are ("TC", "IM").  Correct observations should have
    ``timepoint in {"TC", "IM"}``, not ``"0"``.
    """
    from tests.helpers_task_a_fixture import build_task_a_fixture
    from tasks.task_A.config import TaskAOrderedPairFamilySpec
    from tasks.task_A.workflows.stride_adapter import (
        build_task_a_family_observations,
        load_task_a_dataset_handle,
        resolve_task_a_state_basis,
    )

    adata = build_task_a_fixture()
    handle = load_task_a_dataset_handle(adata)
    state_basis = resolve_task_a_state_basis(handle)

    family_spec = TaskAOrderedPairFamilySpec(
        name="TC-IM",
        source_domain="TC",
        target_domain="IM",
        claim_role="confirmatory",
        pair_types=("TC->IM", "IM->TC"),
    )
    observations = build_task_a_family_observations(
        handle,
        family_spec,
        state_basis=state_basis,
    )

    assert len(observations) > 0, "Expected at least one observation for TC-IM family"

    for obs in observations:
        # timepoint must be the domain label ("TC" or "IM"), never the raw "0"
        assert obs.timepoint in {"TC", "IM"}, (
            f"FovObservation.timepoint should be a domain label, got {obs.timepoint!r}"
        )
        assert obs.timepoint != "0", (
            "FovObservation.timepoint must not be the raw inert timepoint value '0'"
        )
        # domain_label must match timepoint (they are set from the same field)
        assert obs.domain_label == obs.timepoint


def test_adapter_assert_timepoint_inert_raises_on_multiple_values(tmp_path: Path) -> None:
    """``_assert_timepoint_inert`` must raise if the raw timepoint has > 1 unique value."""
    import anndata as ad
    import pandas as pd
    from stride.errors import ContractError
    from tasks.task_A.workflows.stride_adapter import (
        load_task_a_dataset_handle,
        _assert_timepoint_inert,  # tested directly: the guard is a silent-corruption risk; public surface (build_task_a_real_data_crosswalk) also calls it, but isolating the guard lets the test assert on the specific failure mode without triggering unrelated ContractErrors
    )
    from tests.helpers_task_a_fixture import build_task_a_fixture

    adata = build_task_a_fixture()
    # Tamper with the timepoint column to introduce a second distinct value
    obs = adata.obs.copy()
    obs["timepoint"] = obs["timepoint"].astype(object)
    obs.loc[obs.index[0], "timepoint"] = "1"  # now we have both "0" and "1"
    adata = ad.AnnData(X=adata.X, obs=obs, obsm=dict(adata.obsm), uns=dict(adata.uns))

    handle = load_task_a_dataset_handle(adata)
    with pytest.raises(ContractError, match="Task A semantic alignment failed:"):
        _assert_timepoint_inert(handle)


def test_task_a_config_rejects_legacy_mass_mode(tmp_path: Path) -> None:
    from tasks.task_A.config import load_task_a_config_bundle

    config_path = tmp_path / "legacy_mass_mode.yaml"
    config = yaml.safe_load(_write_minimal_config(config_path).read_text(encoding="utf-8"))
    config["data"]["mass_mode"] = "density"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="data.mass_mode='uniform'"):
        load_task_a_config_bundle(config_path)


def test_adapter_full_mapping_summary_structure(tmp_path: Path) -> None:
    """``describe_task_a_stage0_stride_mapping`` must produce a well-structured
    summary that records crosswalk, patient_ids, and per-family stats."""
    from tests.helpers_task_a_fixture import build_task_a_fixture
    from tasks.task_A.config import load_task_a_config_bundle

    config_path = _write_minimal_config(tmp_path / "config.yaml")
    config_bundle = load_task_a_config_bundle(config_path)

    from tasks.task_A.workflows.stride_adapter import describe_task_a_stage0_stride_mapping

    adata = build_task_a_fixture()
    summary = describe_task_a_stage0_stride_mapping(adata, config_bundle=config_bundle)

    # Patient IDs from fixture: P01, P02
    assert set(summary.patient_ids) == {"P01", "P02"}

    # Field mapping should have correct keys
    assert summary.field_mapping.patient_id_key == "patient_id"
    assert summary.field_mapping.n_states == 25

    # Crosswalk must be attached
    assert summary.real_data_crosswalk is not None
    cw = summary.real_data_crosswalk
    assert cw.ordered_group_source == "compartment"
    assert cw.timepoint_inert is True

    # Family summaries: TC-IM, TC-PT, IM-PT (from config defaults)
    family_names = {fs.pair_family for fs in summary.family_summaries}
    assert "TC-IM" in family_names
    assert "TC-PT" in family_names

    # Both patients should be eligible for TC-IM and TC-PT (fixture has all 3 compartments)
    for fs in summary.family_summaries:
        if fs.pair_family in {"TC-IM", "TC-PT"}:
            assert set(fs.eligible_patients) == {"P01", "P02"}
            assert fs.skipped_patients == ()
            assert fs.n_observations > 0

    # JSON round-trip
    as_json = json.dumps(summary.to_json_dict())
    restored = json.loads(as_json)
    assert restored["patient_ids"] == ["P01", "P02"]
    assert "real_data_crosswalk" in restored


# ---------------------------------------------------------------------------
# Subprocess regression: prepare CLI writes all three expected outputs
# ---------------------------------------------------------------------------


def _subprocess_prepare(
    *extra_args: str,
    tmp_path: Path,
    h5ad_path: Path | None = None,
    output_subdir: str = "prepare_out",
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Helper: run the prepare CLI and return (result, output_dir)."""
    from tests.helpers_task_a_fixture import write_task_a_fixture

    if h5ad_path is None:
        h5ad_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_minimal_config(tmp_path / "config.yaml")
    output_dir = tmp_path / output_subdir
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tasks.task_A.workflows.prepare",
            "--task-config",
            str(config_path),
            "--stage0-h5ad",
            str(h5ad_path),
            "--output-dir",
            str(output_dir),
            *extra_args,
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": f"{ROOT}{os.pathsep}{SRC}"},
        timeout=120,
    )
    return result, output_dir


def _subprocess_check_data_suitability(
    *,
    tmp_path: Path,
    h5ad_path: Path | None = None,
    output_subdir: str = "suitability_out",
) -> tuple[subprocess.CompletedProcess[str], Path]:
    from tests.helpers_task_a_fixture import write_task_a_fixture

    if h5ad_path is None:
        h5ad_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_minimal_config(tmp_path / "config.yaml")
    output_dir = tmp_path / output_subdir
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tasks.task_A.workflows.check_data_suitability",
            "--task-config",
            str(config_path),
            "--stage0-h5ad",
            str(h5ad_path),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": f"{ROOT}{os.pathsep}{SRC}"},
        timeout=120,
    )
    return result, output_dir


def test_prepare_workflow_subprocess_full_cohort_writes_three_outputs(tmp_path: Path) -> None:
    result, output_dir = _subprocess_prepare(tmp_path=tmp_path)

    assert result.returncode == 0, (
        f"prepare CLI exited with code {result.returncode}\n"
        f"STDOUT: {result.stdout}\n"
        f"STDERR: {result.stderr}"
    )

    mapping_json = output_dir / "task_a_stride_mapping.json"
    dry_run_csv = output_dir / "task_a_core_fit_dry_run.csv"
    prepare_manifest_json = output_dir / "task_a_prepare_manifest.json"

    assert mapping_json.exists(), f"Expected {mapping_json} to be written"
    assert dry_run_csv.exists(), f"Expected {dry_run_csv} to be written"
    assert prepare_manifest_json.exists(), f"Expected {prepare_manifest_json} to be written"

    mapping_data = json.loads(mapping_json.read_text(encoding="utf-8"))
    assert set(mapping_data["patient_ids"]) == {"P01", "P02"}

    import pandas as pd
    dry_run_df = pd.read_csv(dry_run_csv)
    assert set(dry_run_df["patient_id"].astype(str)) == {"P01", "P02"}
    assert {"implementation_tier", "fit_surface", "cohort_recurrence_fit_status"}.issubset(
        dry_run_df.columns
    )
    assert set(dry_run_df["implementation_tier"].astype(str)) == {"canonical_full"}
    assert set(dry_run_df["fit_surface"].astype(str)) == {"fit_stride"}

    manifest_data = json.loads(prepare_manifest_json.read_text(encoding="utf-8"))
    assert manifest_data["run_scope"] == "full_cohort_alignment_check"
    assert manifest_data["artifact_state"] == "contract_passed"
    assert manifest_data["mass_mode"] == "uniform"
    assert manifest_data["fit_surface"] == "fit_stride"
    assert manifest_data["implementation_tier"] == "canonical_full"
    assert manifest_data["evidence_lineage"] == "canonical_rerun"
    assert "patient_subset" not in manifest_data


def test_prepare_workflow_subprocess_patient_subset_writes_three_outputs(tmp_path: Path) -> None:
    result, output_dir = _subprocess_prepare("--patient-id", "P01", tmp_path=tmp_path)

    assert result.returncode == 0, (
        f"prepare CLI exited with code {result.returncode}\n"
        f"STDOUT: {result.stdout}\n"
        f"STDERR: {result.stderr}"
    )

    mapping_json = output_dir / "task_a_stride_mapping.json"
    dry_run_csv = output_dir / "task_a_core_fit_dry_run.csv"
    prepare_manifest_json = output_dir / "task_a_prepare_manifest.json"

    assert mapping_json.exists(), f"Expected {mapping_json} to be written"
    assert dry_run_csv.exists(), f"Expected {dry_run_csv} to be written"
    assert prepare_manifest_json.exists(), f"Expected {prepare_manifest_json} to be written"

    mapping_data = json.loads(mapping_json.read_text(encoding="utf-8"))
    assert "real_data_crosswalk" in mapping_data, "mapping JSON must include real_data_crosswalk"
    assert "patient_ids" in mapping_data

    import pandas as pd
    dry_run_df = pd.read_csv(dry_run_csv)
    expected_cols = {
        "pair_family",
        "patient_id",
        "implementation_tier",
        "fit_surface",
        "fit_status",
        "bridge_realized",
        "cohort_recurrence_fit_status",
        "n_recurrence_families",
        "n_recurrence_used_patients",
    }
    assert expected_cols.issubset(dry_run_df.columns), (
        f"dry_run CSV missing columns: {expected_cols - set(dry_run_df.columns)}"
    )
    assert set(dry_run_df["patient_id"].astype(str)) == {"P01"}
    assert set(dry_run_df["implementation_tier"].astype(str)) == {"canonical_full"}
    assert set(dry_run_df["fit_surface"].astype(str)) == {"fit_stride"}

    manifest_data = json.loads(prepare_manifest_json.read_text(encoding="utf-8"))
    assert "mapping_manifest" in manifest_data
    assert "core_fit_dry_run" in manifest_data
    assert "pair_families" in manifest_data
    assert manifest_data["run_scope"] == "patient_subset"
    assert manifest_data["artifact_state"] == "scaffold_active"
    assert manifest_data["mass_mode"] == "uniform"
    assert manifest_data["scientific_interpretation_allowed"] is False
    assert manifest_data["fit_surface"] == "fit_stride"
    assert manifest_data["implementation_tier"] == "canonical_full"
    assert manifest_data["evidence_lineage"] == "canonical_rerun"


def test_prepare_workflow_subprocess_patient_id_filter(tmp_path: Path) -> None:
    """``--patient-id P01`` filters the h5ad and records ``patient_subset`` in the manifest."""
    result, output_dir = _subprocess_prepare("--patient-id", "P01", tmp_path=tmp_path, output_subdir="filter_out")

    assert result.returncode == 0, (
        f"prepare CLI exited {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    manifest_data = json.loads(
        (output_dir / "task_a_prepare_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_data["patient_subset"] == ["P01"]
    assert manifest_data["run_scope"] == "patient_subset"
    assert manifest_data["artifact_state"] == "scaffold_active"
    assert "demo_subset_name" not in manifest_data


def test_prepare_workflow_subprocess_rejects_effective_full_cohort_subset(tmp_path: Path) -> None:
    result, output_dir = _subprocess_prepare(
        "--patient-id", "P01",
        "--patient-id", "P02",
        "--patient-id", "DOES_NOT_EXIST",
        tmp_path=tmp_path,
        output_subdir="effective_full_out",
    )

    assert result.returncode != 0
    assert "full-cohort alignment checks must omit subset selectors" in (result.stdout + result.stderr)
    assert not (output_dir / "task_a_prepare_manifest.json").exists()


def test_prepare_workflow_subprocess_rejects_unknown_demo_subset(tmp_path: Path) -> None:
    result, output_dir = _subprocess_prepare(
        "--demo-subset", "not_a_real_subset",
        tmp_path=tmp_path,
        output_subdir="bad_demo_subset_out",
    )

    assert result.returncode != 0
    assert "Unknown demo subset" in (result.stdout + result.stderr)
    assert not (output_dir / "task_a_prepare_manifest.json").exists()


def test_prepare_workflow_subprocess_demo_subset_blocks_combined_flags(tmp_path: Path) -> None:
    """``--demo-subset`` and ``--patient-id`` must be mutually exclusive.

    The CLI must exit non-zero and print the guard message when both are supplied,
    confirming that ``--demo-subset`` is wired into the argument parser and
    mutually-exclusive validation fires at runtime.
    """
    result, _ = _subprocess_prepare(
        "--demo-subset", "alignment_v1",
        "--patient-id", "P01",
        tmp_path=tmp_path,
        output_subdir="guard_out",
    )
    assert result.returncode != 0, (
        "Expected non-zero exit when --demo-subset and --patient-id are combined"
    )
    combined_output = result.stdout + result.stderr
    assert "Cannot combine" in combined_output, (
        f"Expected guard message in output; got:\n{combined_output}"
    )


def test_prepare_workflow_function_demo_subset_in_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``prepare_task_a_stage0_mapping`` records ``demo_subset_name`` and
    ``demo_subset_rationale`` in the written manifest when supplied."""
    from tests.helpers_task_a_fixture import write_task_a_fixture
    from tasks.task_A.real_data.demo_subset import TaskADemoSubset
    from tasks.task_A.workflows.prepare import prepare_task_a_stage0_mapping

    monkeypatch.setattr(
        "tasks.task_A.workflows.prepare.resolve_demo_subset",
        lambda name: TaskADemoSubset(
            name=str(name),
            patient_ids=("P01",),
            rationale="fixture subset rationale",
        ),
    )

    h5ad_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_minimal_config(tmp_path / "config.yaml")
    output_dir = tmp_path / "out"

    manifest = prepare_task_a_stage0_mapping(
        config_path=config_path,
        data_path=h5ad_path,
        output_dir=output_dir,
        demo_subset_name="fixture_subset",
    )

    assert manifest["demo_subset_name"] == "fixture_subset"
    assert manifest["demo_subset_rationale"] == "fixture subset rationale"
    assert manifest["patient_subset"] == ["P01"]
    assert manifest["run_scope"] == "demo_subset"
    assert manifest["artifact_state"] == "scaffold_active"
    assert manifest["mass_mode"] == "uniform"
    assert manifest["scientific_interpretation_allowed"] is False

    # Verify the written file also has the fields
    manifest_path = output_dir / "task_a_prepare_manifest.json"
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written["demo_subset_name"] == "fixture_subset"
    assert written["demo_subset_rationale"] == "fixture subset rationale"
    assert written["run_scope"] == "demo_subset"
    assert written["artifact_state"] == "scaffold_active"


def test_prepare_workflow_function_rejects_demo_subset_provenance_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stride.errors import ContractError
    from tests.helpers_task_a_fixture import write_task_a_fixture
    from tasks.task_A.real_data.demo_subset import TaskADemoSubset
    from tasks.task_A.workflows.prepare import prepare_task_a_stage0_mapping

    monkeypatch.setattr(
        "tasks.task_A.workflows.prepare.resolve_demo_subset",
        lambda name: TaskADemoSubset(
            name=str(name),
            patient_ids=("P01", "P02"),
            rationale="fixture full subset",
        ),
    )

    h5ad_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_minimal_config(tmp_path / "config.yaml")

    with pytest.raises(ContractError, match="may label a run as demo_subset only when patient_ids exactly match"):
        prepare_task_a_stage0_mapping(
            config_path=config_path,
            data_path=h5ad_path,
            output_dir=tmp_path / "out",
            patient_ids=("P01",),
            demo_subset_name="fixture_subset",
        )


def test_check_data_suitability_cli_writes_pre_block0_report(tmp_path: Path) -> None:
    result, output_dir = _subprocess_check_data_suitability(tmp_path=tmp_path)

    assert result.returncode == 0, (
        f"data suitability CLI exited with code {result.returncode}\n"
        f"STDOUT: {result.stdout}\n"
        f"STDERR: {result.stderr}"
    )

    report_path = output_dir / "task_a_pre_block0_data_suitability.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["report_scope"] == "pre_block0_data_suitability"
    assert report["run_scope"] == "full_cohort_alignment_check"
    assert report["artifact_state"] == "contract_passed"
    assert report["scientific_interpretation_allowed"] is False
    assert report["block0_gate_status"] == "not_passed"
    assert report["mass_mode"] == "uniform"
    assert report["confirmatory_pair_families"] == ["TC-IM", "TC-PT"]
    assert report["stage0_validation"]["taska_minimum_contract"]["ok"] is True


def test_data_suitability_report_fails_on_semantic_misalignment(tmp_path: Path) -> None:
    import anndata as ad
    from stride.errors import ContractError
    from tests.helpers_task_a_fixture import build_task_a_fixture
    from tasks.task_A.workflows.prepare import build_task_a_pre_block0_data_suitability_report

    adata = build_task_a_fixture()
    obs = adata.obs.copy()
    obs["timepoint"] = obs["timepoint"].astype(str)
    obs.loc[obs.index[0], "timepoint"] = "1"
    broken = ad.AnnData(X=adata.X, obs=obs, obsm=dict(adata.obsm), uns=dict(adata.uns))
    h5ad_path = tmp_path / "broken_timepoint.h5ad"
    broken.write_h5ad(h5ad_path)
    config_path = _write_minimal_config(tmp_path / "config.yaml")

    with pytest.raises(ContractError, match="Task A semantic alignment failed:"):
        build_task_a_pre_block0_data_suitability_report(
            config_path=config_path,
            data_path=h5ad_path,
        )


def test_check_data_suitability_cli_writes_partial_report_for_missing_proto_id(tmp_path: Path) -> None:
    import anndata as ad
    from tests.helpers_task_a_fixture import build_task_a_fixture

    adata = build_task_a_fixture()
    obs = adata.obs.drop(columns=["proto_id"]).copy()
    broken = ad.AnnData(X=adata.X, obs=obs, obsm=dict(adata.obsm), uns=dict(adata.uns))
    h5ad_path = tmp_path / "missing_proto_id.h5ad"
    broken.write_h5ad(h5ad_path)

    result, output_dir = _subprocess_check_data_suitability(
        tmp_path=tmp_path,
        h5ad_path=h5ad_path,
        output_subdir="missing_proto_suitability",
    )

    assert result.returncode == 0, (
        f"data suitability CLI exited with code {result.returncode}\n"
        f"STDOUT: {result.stdout}\n"
        f"STDERR: {result.stderr}"
    )
    report_path = output_dir / "task_a_pre_block0_data_suitability.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["stage0_validation"]["taska_minimum_contract"]["ok"] is False
    assert report["stage0_validation"]["counts"]["n_unique_proto_ids"] is None
    assert report["artifact_state"] == "scaffold_active"
    assert report["mapping_summary"] is None
    assert report["mapping_summary_error"]
