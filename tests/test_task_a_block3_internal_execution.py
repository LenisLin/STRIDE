from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stride.errors import ContractError

ANNDATA_AVAILABLE = importlib.util.find_spec("anndata") is not None
pytestmark = pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not installed")


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_task_a_config(path: Path) -> Path:
    from tests.helpers_task_a_fixture import K_FULL

    payload = {
        "task_name": "Task A block3 internal execution fixture",
        "enabled_blocks": [
            "block0_locality_gate",
            "block1_continuity_backbone",
            "block2_bounded_audit",
        ],
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
            "master_seed": 17,
            "patient_subsample_replicates": 1,
            "leave_some_out_replicates": 1,
            "seed_rerun_replicates": 1,
            "roi_drop_replicates": 1,
            "patient_subsample_min_patients": 1,
            "leave_some_out_min_patients": 1,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_block3_upstream_fixture(
    base: Path,
    *,
    fixture_variant: str,
) -> Path:
    from tasks.task_A.workflows.run_block1 import run_block1_workflow
    from tests.helpers_task_a_fixture import (
        FIXTURE_VARIANT_BLOCK3_PHASE3,
        write_block3_phase3_task_a_fixture,
        write_passed_block0_bundle,
        write_task_a_fixture,
    )

    config_path = _write_task_a_config(base / "config.yaml")
    if fixture_variant == FIXTURE_VARIANT_BLOCK3_PHASE3:
        data_path = write_block3_phase3_task_a_fixture(base / "stage0.h5ad")
    else:
        data_path = write_task_a_fixture(base / "stage0.h5ad", variant=fixture_variant)

    block0_bundle_path = write_passed_block0_bundle(
        base / "block0" / "block0_bundle.json",
        config_path=config_path,
        data_path=data_path,
    )
    block1_bundle_path = run_block1_workflow(
        config_path=str(config_path),
        data_path=str(data_path),
        block0_bundle=str(block0_bundle_path),
        output_dir=str(base / "block1"),
    )
    return _write_json(
        base / "block2" / "block2_bounded_audit_manifest.json",
        {
            "block": "block2_bounded_audit",
            "scientific_role": "robustness_of_frozen_block1_findings",
            "artifact_state": "evidence_ready",
            "implementation_tier": "canonical_full",
            "evidence_lineage": "canonical_rerun",
            "block1_bundle_path": str(block1_bundle_path),
        },
    )


def _make_minimal_block3_cohort_inputs() -> object:
    from tasks.task_A.block3.execution import Block3CohortInputs

    cost_matrix = np.asarray(
        [
            [0.0, 0.2, 0.8],
            [0.2, 0.0, 0.3],
            [0.8, 0.3, 0.0],
        ],
        dtype=float,
    )
    patient_source_profiles: dict[str, np.ndarray] = {}
    patient_target_profiles: dict[str, np.ndarray] = {}
    for patient_index in range(32):
        source = np.asarray(
            [
                0.46 + 0.01 * (patient_index % 4),
                0.34 - 0.005 * (patient_index % 3),
                0.20 - 0.01 * (patient_index % 4) + 0.005 * (patient_index % 3),
            ],
            dtype=float,
        )
        source = source / np.sum(source, dtype=float)
        target = np.asarray(
            [
                source[0] - (0.03 + 0.002 * (patient_index % 3)),
                source[1] + (0.01 + 0.001 * (patient_index % 2)),
                source[2] + (0.02 + 0.002 * (patient_index % 3) - 0.001 * (patient_index % 2)),
            ],
            dtype=float,
        )
        patient_id = f"P{patient_index + 1:02d}"
        patient_source_profiles[patient_id] = source
        patient_target_profiles[patient_id] = target
    return Block3CohortInputs(
        stage0_h5ad=Path("/tmp/stage0.h5ad"),
        config_path=Path("/tmp/config.yaml"),
        output_dir=Path("/tmp/block1"),
        master_seed=17,
        state_ids=(0, 1, 2),
        state_basis=None,
        geometry=None,
        identity_vectors=np.eye(3, dtype=float),
        cost_matrix=cost_matrix,
        patient_source_profiles=patient_source_profiles,
        patient_target_profiles=patient_target_profiles,
    )


def test_build_identity_cost_matrix_uses_positive_offdiag_median_scale() -> None:
    from tasks.task_A.block3.execution import _build_identity_cost_matrix, _js_divergence

    identity_vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.7, 0.3, 0.0],
            [0.0, 0.4, 0.6],
        ],
        dtype=float,
    )

    cost_matrix = _build_identity_cost_matrix(identity_vectors)

    cost_raw = np.zeros((identity_vectors.shape[0], identity_vectors.shape[0]), dtype=float)
    for left_index in range(identity_vectors.shape[0]):
        for right_index in range(left_index + 1, identity_vectors.shape[0]):
            value = float(np.sqrt(_js_divergence(identity_vectors[left_index], identity_vectors[right_index])))
            cost_raw[left_index, right_index] = value
            cost_raw[right_index, left_index] = value
    positive = cost_raw[cost_raw > 0.0]
    expected = cost_raw / float(np.median(positive))

    np.testing.assert_allclose(cost_matrix, expected)


def test_load_identity_vectors_fails_on_missing_state_id(tmp_path: Path) -> None:
    from tasks.task_A.block3.execution import _load_identity_vectors

    output_dir = tmp_path / "block1"
    fractions_path = output_dir / "community_correspondence" / "tables" / "community_cell_subtype_row_fractions.csv"
    fractions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [[0.8, 0.2], [0.3, 0.7]],
        index=pd.Index([0, 1], name="community_id"),
        columns=["SubtypeA", "SubtypeB"],
    ).to_csv(fractions_path)

    with pytest.raises(ContractError, match="missing community identity vectors"):
        _load_identity_vectors(
            block1_payload={"output_dir": str(output_dir)},
            state_ids=(0, 1, 2),
        )


def test_load_identity_vectors_fails_on_zero_sum_row(tmp_path: Path) -> None:
    from tasks.task_A.block3.execution import _load_identity_vectors

    output_dir = tmp_path / "block1"
    fractions_path = output_dir / "community_correspondence" / "tables" / "community_cell_subtype_row_fractions.csv"
    fractions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [[0.8, 0.2], [0.0, 0.0], [0.1, 0.9]],
        index=pd.Index([0, 1, 2], name="community_id"),
        columns=["SubtypeA", "SubtypeB"],
    ).to_csv(fractions_path)

    with pytest.raises(ContractError, match="zero total mass"):
        _load_identity_vectors(
            block1_payload={"output_dir": str(output_dir)},
            state_ids=(0, 1, 2),
        )


def test_estimate_kappa_uses_robust_median_total_variation_rule() -> None:
    from tasks.task_A.block3.execution import _estimate_kappa

    profiles = [
        np.asarray([0.8, 0.2, 0.0], dtype=float),
        np.asarray([0.6, 0.3, 0.1], dtype=float),
        np.asarray([0.5, 0.2, 0.3], dtype=float),
    ]
    centroid = np.mean(np.vstack(profiles), axis=0, dtype=float)
    deviations = [0.5 * float(np.sum(np.abs(profile - centroid), dtype=float)) for profile in profiles]
    expected = float(np.clip(1.0 / max(float(np.median(deviations)), 1e-8), 1.0, 200.0))

    assert _estimate_kappa(profiles) == pytest.approx(expected)


def test_sample_capped_depletion_redistributes_remaining_mass_when_base_support_exhausts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tasks.task_A.block3 import execution as block3_execution

    monkeypatch.setattr(
        block3_execution,
        "_sample_gamma_simplex",
        lambda *, base, kappa, rng: np.asarray([1.0, 0.0, 0.0], dtype=float),
    )

    depletion = block3_execution._sample_capped_depletion(
        x=np.asarray([0.2, 0.3, 0.5], dtype=float),
        mass=0.6,
        base=np.asarray([1.0, 0.0, 0.0], dtype=float),
        kappa=20.0,
        rng=np.random.default_rng(0),
    )

    np.testing.assert_allclose(np.sum(depletion, dtype=float), 0.6)
    assert depletion[0] == pytest.approx(0.2)
    assert np.all(depletion <= np.asarray([0.2, 0.3, 0.5], dtype=float) + 1e-12)
    assert depletion[1] > 0.0 or depletion[2] > 0.0


def test_relation_motif_probe_breaks_nearest_neighbor_truth_rule() -> None:
    from tasks.task_A.block3.execution import _build_truth_for_condition

    x = np.asarray([0.6, 0.4, 0.0], dtype=float)
    delta_minus = np.asarray([0.0, 0.0, 0.0], dtype=float)
    delta_plus = np.asarray([0.0, 0.0, 0.0], dtype=float)
    cost_matrix = np.asarray(
        [
            [0.0, 0.1, 0.9],
            [0.1, 0.0, 0.2],
            [0.9, 0.2, 0.0],
        ],
        dtype=float,
    )
    relation_motif = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    legacy_truth = _build_truth_for_condition(
        rerun_id="rerun_01",
        patient_id="P01",
        x=x,
        delta_minus=delta_minus,
        delta_plus=delta_plus,
        condition_id="relation_weak",
        cost_matrix=cost_matrix,
    )
    motif_truth = _build_truth_for_condition(
        rerun_id="rerun_01",
        patient_id="P01",
        x=x,
        delta_minus=delta_minus,
        delta_plus=delta_plus,
        condition_id="relation_weak",
        cost_matrix=cost_matrix,
        support_mode="relation_motif_probe_v1",
        relation_motif=relation_motif,
    )

    assert legacy_truth.A[0, 1] > 0.0
    assert legacy_truth.A[0, 2] == pytest.approx(0.0)
    assert motif_truth.A[0, 1] == pytest.approx(0.0)
    assert motif_truth.A[0, 2] > 0.0


def test_relation_motif_probe_zero_row_falls_back_deterministically() -> None:
    from tasks.task_A.block3.execution import _build_truth_for_condition

    x = np.asarray([0.5, 0.3, 0.2], dtype=float)
    delta_minus = np.asarray([0.0, 0.0, 0.0], dtype=float)
    delta_plus = np.asarray([0.0, 0.0, 0.0], dtype=float)
    cost_matrix = np.asarray(
        [
            [0.0, 0.2, 0.7],
            [0.2, 0.0, 0.1],
            [0.7, 0.1, 0.0],
        ],
        dtype=float,
    )
    relation_motif = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.6, 0.0, 0.4],
            [0.3, 0.7, 0.0],
        ],
        dtype=float,
    )

    legacy_truth = _build_truth_for_condition(
        rerun_id="rerun_01",
        patient_id="P01",
        x=x,
        delta_minus=delta_minus,
        delta_plus=delta_plus,
        condition_id="relation_strong",
        cost_matrix=cost_matrix,
    )
    motif_truth_first = _build_truth_for_condition(
        rerun_id="rerun_01",
        patient_id="P01",
        x=x,
        delta_minus=delta_minus,
        delta_plus=delta_plus,
        condition_id="relation_strong",
        cost_matrix=cost_matrix,
        support_mode="relation_motif_probe_v1",
        relation_motif=relation_motif,
    )
    motif_truth_second = _build_truth_for_condition(
        rerun_id="rerun_01",
        patient_id="P01",
        x=x,
        delta_minus=delta_minus,
        delta_plus=delta_plus,
        condition_id="relation_strong",
        cost_matrix=cost_matrix,
        support_mode="relation_motif_probe_v1",
        relation_motif=relation_motif,
    )

    np.testing.assert_allclose(motif_truth_first.A[0], legacy_truth.A[0])
    np.testing.assert_allclose(motif_truth_second.A[0], legacy_truth.A[0])
    np.testing.assert_allclose(motif_truth_first.A, motif_truth_second.A)


def test_relation_motif_probe_keeps_relation_null_and_open_scaffold_unchanged() -> None:
    from tasks.task_A.block3.execution import _build_truth_for_condition

    x = np.asarray([0.5, 0.3, 0.2], dtype=float)
    delta_minus = np.asarray([0.10, 0.00, 0.05], dtype=float)
    delta_plus = np.asarray([0.02, 0.03, 0.10], dtype=float)
    cost_matrix = np.asarray(
        [
            [0.0, 0.1, 0.5],
            [0.1, 0.0, 0.2],
            [0.5, 0.2, 0.0],
        ],
        dtype=float,
    )
    relation_motif = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.8, 0.0, 0.2],
            [0.3, 0.7, 0.0],
        ],
        dtype=float,
    )

    legacy_truth = _build_truth_for_condition(
        rerun_id="rerun_01",
        patient_id="P01",
        x=x,
        delta_minus=delta_minus,
        delta_plus=delta_plus,
        condition_id="relation_null",
        cost_matrix=cost_matrix,
    )
    motif_truth = _build_truth_for_condition(
        rerun_id="rerun_01",
        patient_id="P01",
        x=x,
        delta_minus=delta_minus,
        delta_plus=delta_plus,
        condition_id="relation_null",
        cost_matrix=cost_matrix,
        support_mode="relation_motif_probe_v1",
        relation_motif=relation_motif,
    )

    np.testing.assert_allclose(motif_truth.A, legacy_truth.A)
    np.testing.assert_allclose(motif_truth.d, legacy_truth.d)
    np.testing.assert_allclose(motif_truth.e, legacy_truth.e)
    assert motif_truth.open_mass == pytest.approx(legacy_truth.open_mass)
    offdiag = motif_truth.A - np.diag(np.diag(motif_truth.A))
    np.testing.assert_allclose(offdiag, np.zeros_like(offdiag))


def test_build_generator_reruns_open_mass_scale_rescales_same_open_realization() -> None:
    from tasks.task_A.block3.execution import _build_generator_reruns

    cohort_inputs = _make_minimal_block3_cohort_inputs()

    full_reruns = _build_generator_reruns(
        cohort_inputs=cohort_inputs,
        support_mode="legacy_nearest_c",
        open_mass_scale=1.0,
    )
    half_reruns = _build_generator_reruns(
        cohort_inputs=cohort_inputs,
        support_mode="legacy_nearest_c",
        open_mass_scale=0.5,
    )

    assert tuple(rerun.rerun_id for rerun in full_reruns) == tuple(rerun.rerun_id for rerun in half_reruns)
    for full_rerun, half_rerun in zip(full_reruns, half_reruns, strict=True):
        assert full_rerun.train_patient_ids == half_rerun.train_patient_ids
        assert full_rerun.test_patient_ids == half_rerun.test_patient_ids
        assert full_rerun.hidden_relation_condition_id == half_rerun.hidden_relation_condition_id
        for condition_id in ("relation_null", "relation_mid", "relation_strong"):
            for patient_id in full_rerun.test_patient_ids:
                full_truth = full_rerun.baseline_truths[condition_id][patient_id]
                half_truth = half_rerun.baseline_truths[condition_id][patient_id]
                np.testing.assert_allclose(half_truth.x, full_truth.x)
                np.testing.assert_allclose(half_truth.d, 0.5 * full_truth.d, atol=1e-12)
                np.testing.assert_allclose(half_truth.e, 0.5 * full_truth.e, atol=1e-12)
                assert half_truth.open_mass == pytest.approx(0.5 * full_truth.open_mass)
                np.testing.assert_array_equal(full_truth.A > 1e-12, half_truth.A > 1e-12)
                assert np.sum(half_truth.y, dtype=float) == pytest.approx(1.0)


def test_internal_3b1_open_mass_sensitivity_accepts_quarter_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tasks.task_A.block3.execution as execution
    from tasks.task_A.block3.calibration import UOTCalibrationResult

    def _fake_outputs(*, truths: list[object], **_kwargs: object) -> dict[str, object]:
        return {
            truth.patient_id: execution.Block3MethodOutput(
                patient_id=truth.patient_id,
                fit_status="ok",
                A=np.asarray(truth.A, dtype=float),
                d=np.asarray(truth.d, dtype=float),
                e=np.asarray(truth.e, dtype=float),
                mu_minus=np.asarray(truth.x, dtype=float),
                mu_plus=np.asarray(truth.y, dtype=float),
            )
            for truth in truths
        }

    monkeypatch.setattr(
        execution,
        "_calibrated_uot_lambda_for_train",
        lambda **_kwargs: UOTCalibrationResult(
            selected_lambda=1.0,
            target_overlap=1.0,
            achieved_by_lambda={1.0: 1.0},
            absolute_error_by_lambda={1.0: 0.0},
            boundary_hit=False,
        ),
    )
    monkeypatch.setattr(execution, "_run_stride_method", _fake_outputs)
    monkeypatch.setattr(execution, "_run_balanced_ot_baseline", _fake_outputs)
    monkeypatch.setattr(execution, "_run_uot_baseline", _fake_outputs)
    monkeypatch.setattr(execution, "_run_partial_ot_baseline", _fake_outputs)
    monkeypatch.setattr(execution, "_run_diagonal_transport_baseline", _fake_outputs)

    cohort_inputs = _make_minimal_block3_cohort_inputs()
    reruns = execution._build_generator_reruns(
        cohort_inputs=cohort_inputs,
        support_mode=execution._SUPPORT_MODE_LEGACY_NEAREST_C,
        open_mass_scale=0.25,
    )
    raw_rows, _review_rows = execution._build_3b1_rows(
        reruns=reruns,
        cohort_inputs=cohort_inputs,
    )

    assert {rerun.open_mass_scale for rerun in reruns} == {0.25}
    assert {row.open_mass_scale for row in raw_rows.patient_metrics} == {0.25}
    truth_rows = raw_rows.shared_tables["patient_truth_store"]
    assert {float(row["open_mass_scale"]) for row in truth_rows} == {0.25}


def test_internal_3b_diagnostic_matrix_writes_internal_sidecar_artifacts(tmp_path: Path) -> None:
    from tasks.task_A.block3.execution import execute_internal_block3_3b_diagnostic_matrix
    from tests.helpers_task_a_fixture import FIXTURE_VARIANT_BLOCK3_PHASE3

    block2_manifest_path = _write_block3_upstream_fixture(
        tmp_path / "phase3_fixture_matrix",
        fixture_variant=FIXTURE_VARIANT_BLOCK3_PHASE3,
    )

    result = execute_internal_block3_3b_diagnostic_matrix(
        block2_manifest_path=block2_manifest_path,
        output_dir=tmp_path / "block3_3b_diagnostic_matrix",
    )

    assert len(result.arm_results) == 6
    assert {
        (arm.support_mode, arm.open_mass_scale)
        for arm in result.arm_results
    } == {
        ("legacy_nearest_c", 0.1),
        ("legacy_nearest_c", 0.25),
        ("legacy_nearest_c", 0.5),
        ("relation_motif_probe_v1", 0.1),
        ("relation_motif_probe_v1", 0.25),
        ("relation_motif_probe_v1", 0.5),
    }
    for arm in result.arm_results:
        assert "3b1_patient_metrics" in arm.execution_result.raw_artifact_paths
        assert "3b1_condition_summary" in arm.execution_result.raw_artifact_paths
        assert arm.execution_result.raw_artifact_paths["3b1_patient_metrics"].exists()
        assert arm.execution_result.raw_artifact_paths["3b1_condition_summary"].exists()

    truth_budget_df = pd.read_csv(result.summary_artifact_paths["3b_truth_budget_summary"])
    assert len(truth_budget_df) == 24
    assert set(truth_budget_df["support_mode"].astype(str)) == {
        "legacy_nearest_c",
        "relation_motif_probe_v1",
    }
    assert set(truth_budget_df["open_mass_scale"].astype(float)) == {0.1, 0.25, 0.5}
    assert set(truth_budget_df["condition_id"].astype(str)) == {
        "relation_null",
        "relation_weak",
        "relation_mid",
        "relation_strong",
    }

    method_budget_df = pd.read_csv(result.summary_artifact_paths["3b_method_budget_summary"])
    assert len(method_budget_df) == 120
    assert set(method_budget_df["method_name"].astype(str)) == {
        "stride_reference",
        "balanced_ot_baseline",
        "uot_baseline",
        "partial_ot_baseline",
        "diagonal_transport_baseline",
    }

    rank_df = pd.read_csv(result.summary_artifact_paths["3b_rank_concordance_summary"])
    condition_rows = rank_df.loc[rank_df["row_type"].astype(str) == "condition"].copy()
    assert len(condition_rows) == 24
    assert set(condition_rows["formal_rank_consistent_with_candidate"].astype(str)) <= {"True", "False"}
    verdict_rows = rank_df.loc[rank_df["row_type"].astype(str) == "matrix_verdict"].copy()
    assert len(verdict_rows) == 1
    assert verdict_rows["verdict_label"].iloc[0] in {
        "generator_open_dominance",
        "generator_support_isomorphism",
        "input_surface_mismatch",
        "implementation_defect",
    }


def test_run_balanced_ot_baseline_uses_nonredundant_transport_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tasks.task_A.block3.execution import (
        Block3CohortInputs,
        Block3PatientTruth,
        _run_balanced_ot_baseline,
    )

    x = np.asarray([0.6, 0.3, 0.1], dtype=float)
    y = np.asarray([0.2, 0.5, 0.3], dtype=float)
    cost_matrix = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    cohort_inputs = Block3CohortInputs(
        stage0_h5ad=Path("/tmp/stage0.h5ad"),
        config_path=Path("/tmp/config.yaml"),
        output_dir=Path("/tmp/block1"),
        master_seed=17,
        state_ids=(0, 1, 2),
        state_basis=None,
        geometry=None,
        identity_vectors=np.eye(3, dtype=float),
        cost_matrix=cost_matrix,
        patient_source_profiles={"P01": x},
        patient_target_profiles={"P01": y},
    )
    truths = [
        Block3PatientTruth(
            rerun_id="rerun_01",
            patient_id="P01",
            x=x,
            y=y,
            A=np.eye(3, dtype=float),
            d=np.zeros(3, dtype=float),
            e=np.zeros(3, dtype=float),
            open_mass=0.0,
        )
    ]

    def _fake_linprog(*, c, A_eq, b_eq, bounds, method):
        assert A_eq.shape == (5, 9)
        np.testing.assert_allclose(b_eq, np.concatenate([x, y[:-1]]))
        return SimpleNamespace(
            success=True,
            status=0,
            message="ok",
            x=np.outer(x, y).reshape(-1),
        )

    import scipy.optimize

    monkeypatch.setattr(scipy.optimize, "linprog", _fake_linprog)

    outputs = _run_balanced_ot_baseline(
        cohort_inputs=cohort_inputs,
        truths=truths,
    )

    output = outputs["P01"]
    np.testing.assert_allclose(np.sum(x[:, None] * output.A, axis=0, dtype=float), y)
    np.testing.assert_allclose(output.mu_plus, y)


def test_run_balanced_ot_baseline_keeps_zero_source_rows_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tasks.task_A.block3.execution import (
        Block3CohortInputs,
        Block3PatientTruth,
        _run_balanced_ot_baseline,
    )

    x = np.asarray([0.7, 0.3, 0.0], dtype=float)
    y = np.asarray([0.2, 0.3, 0.5], dtype=float)
    cost_matrix = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    cohort_inputs = Block3CohortInputs(
        stage0_h5ad=Path("/tmp/stage0.h5ad"),
        config_path=Path("/tmp/config.yaml"),
        output_dir=Path("/tmp/block1"),
        master_seed=17,
        state_ids=(0, 1, 2),
        state_basis=None,
        geometry=None,
        identity_vectors=np.eye(3, dtype=float),
        cost_matrix=cost_matrix,
        patient_source_profiles={"P01": x},
        patient_target_profiles={"P01": y},
    )
    truths = [
        Block3PatientTruth(
            rerun_id="rerun_01",
            patient_id="P01",
            x=x,
            y=y,
            A=np.eye(3, dtype=float),
            d=np.zeros(3, dtype=float),
            e=np.zeros(3, dtype=float),
            open_mass=0.0,
        )
    ]

    def _fake_linprog(*, c, A_eq, b_eq, bounds, method):
        del c, A_eq, b_eq, bounds, method
        return SimpleNamespace(
            success=True,
            status=0,
            message="ok",
            x=np.outer(x, y).reshape(-1),
        )

    import scipy.optimize

    monkeypatch.setattr(scipy.optimize, "linprog", _fake_linprog)

    outputs = _run_balanced_ot_baseline(
        cohort_inputs=cohort_inputs,
        truths=truths,
    )

    output = outputs["P01"]
    np.testing.assert_allclose(output.A[2], np.zeros(3, dtype=float))
    np.testing.assert_allclose(np.sum(x[:, None] * output.A, axis=0, dtype=float), y)


def test_internal_block3_execution_requires_frozen_24_8_10_cohort_design(tmp_path: Path) -> None:
    from tasks.task_A.block3.execution import execute_internal_block3_subexperiment
    from tests.helpers_task_a_fixture import FIXTURE_VARIANT_DEFAULT

    block2_manifest_path = _write_block3_upstream_fixture(
        tmp_path / "small_fixture",
        fixture_variant=FIXTURE_VARIANT_DEFAULT,
    )

    with pytest.raises(ContractError, match="requires at least 32 eligible patients"):
        execute_internal_block3_subexperiment(
            block2_manifest_path=block2_manifest_path,
            output_dir=tmp_path / "block3_output",
            subexperiment_id="3A",
        )


def test_internal_block3_execution_writes_real_phase3_internal_artifacts(tmp_path: Path) -> None:
    from tasks.task_A.block3.execution import execute_internal_block3_subexperiment
    from tests.helpers_task_a_fixture import FIXTURE_VARIANT_BLOCK3_PHASE3

    block2_manifest_path = _write_block3_upstream_fixture(
        tmp_path / "phase3_fixture",
        fixture_variant=FIXTURE_VARIANT_BLOCK3_PHASE3,
    )

    result_3a = execute_internal_block3_subexperiment(
        block2_manifest_path=block2_manifest_path,
        output_dir=tmp_path / "block3_3a",
        subexperiment_id="3A",
    )
    result_3b = execute_internal_block3_subexperiment(
        block2_manifest_path=block2_manifest_path,
        output_dir=tmp_path / "block3_3b1",
        subexperiment_id="3B-1",
    )
    result_3b2 = execute_internal_block3_subexperiment(
        block2_manifest_path=block2_manifest_path,
        output_dir=tmp_path / "block3_3b2",
        subexperiment_id="3B-2",
    )
    result_3c1 = execute_internal_block3_subexperiment(
        block2_manifest_path=block2_manifest_path,
        output_dir=tmp_path / "block3_3c1",
        subexperiment_id="3C-1",
    )
    result_3c2 = execute_internal_block3_subexperiment(
        block2_manifest_path=block2_manifest_path,
        output_dir=tmp_path / "block3_3c2",
        subexperiment_id="3C-2",
    )

    for result in (result_3a, result_3b, result_3b2, result_3c1, result_3c2):
        manifest_payload = json.loads(result.raw_manifest_path.read_text(encoding="utf-8"))
        review_manifest_payload = json.loads(result.review_manifest_path.read_text(encoding="utf-8"))
        assert manifest_payload["artifact_state"] == "scaffold_active"
        assert manifest_payload["scientific_interpretation_allowed"] is False
        assert manifest_payload["packet_bridge_enabled"] is False
        assert manifest_payload["packet_bridge_policy"] == "deferred_non_authority_pending_clean_bridge_spec"
        assert review_manifest_payload["artifact_state"] == "scaffold_active"
        assert review_manifest_payload["scientific_interpretation_allowed"] is False
        assert review_manifest_payload["packet_bridge_enabled"] is False
        assert review_manifest_payload["packet_bridge_policy"] == "deferred_non_authority_pending_clean_bridge_spec"

    raw_index_3a = pd.read_csv(result_3a.raw_index_path, keep_default_na=False)
    assert set(raw_index_3a["artifact_role"].astype(str)) == {
        "bundle_manifest",
        "raw_index",
        "generator_rerun_registry",
        "generator_split_registry",
        "patient_truth_store",
        "3a_object_scores",
        "3a_rerun_stability",
    }
    object_scores_df = pd.read_csv(result_3a.raw_artifact_paths["3a_object_scores"])
    assert set(object_scores_df["metric_name"].astype(str)) == {
        "Pearson correlation",
        "MAE",
        "MSE",
        "JS divergence",
    }
    assert object_scores_df["rerun_id"].nunique() == 10
    split_registry_df = pd.read_csv(result_3a.raw_artifact_paths["generator_split_registry"])
    assert split_registry_df["rerun_id"].nunique() == 10
    assert set(split_registry_df["split_role"].astype(str)) == {"train", "test"}
    assert set(split_registry_df.groupby("rerun_id")["split_role"].value_counts().astype(int)) == {8, 24}

    raw_index_3b = pd.read_csv(result_3b.raw_index_path, keep_default_na=False)
    assert set(raw_index_3b["artifact_role"].astype(str)) == {
        "bundle_manifest",
        "raw_index",
        "generator_rerun_registry",
        "generator_split_registry",
        "patient_truth_store",
        "method_native_output_store",
        "3b1_patient_metrics",
        "3b1_condition_summary",
    }
    patient_metrics_3b = pd.read_csv(result_3b.raw_artifact_paths["3b1_patient_metrics"])
    assert set(patient_metrics_3b["method_name"].astype(str)) == {
        "stride_reference",
        "balanced_ot_baseline",
        "uot_baseline",
        "partial_ot_baseline",
        "diagonal_transport_baseline",
    }
    assert set(patient_metrics_3b["condition_id"].astype(str)) == {
        "relation_null",
        "relation_weak",
        "relation_mid",
        "relation_strong",
    }
    assert patient_metrics_3b["rerun_id"].nunique() == 10
    null_recall_rows = patient_metrics_3b.loc[
        (patient_metrics_3b["condition_id"].astype(str) == "relation_null")
        & (patient_metrics_3b["metric_name"].astype(str) == "target_recall_at_k")
    ].copy()
    assert not null_recall_rows.empty
    assert set(null_recall_rows["metric_status"].astype(str)) == {"not_applicable"}
    assert null_recall_rows["reported_value"].isna().all()

    truth_store_3b = pd.read_csv(result_3b.raw_artifact_paths["patient_truth_store"])
    assert set(truth_store_3b["subexperiment_id"].astype(str)) == {"3B-1"}
    assert set(truth_store_3b["condition_id"].astype(str)) == {
        "relation_null",
        "relation_weak",
        "relation_mid",
        "relation_strong",
    }
    assert set(truth_store_3b.groupby("rerun_id")["patient_id"].nunique().astype(int)) == {8}
    native_outputs_3b = pd.read_csv(result_3b.raw_artifact_paths["method_native_output_store"])
    assert set(native_outputs_3b["subexperiment_id"].astype(str)) == {"3B-1"}
    assert set(native_outputs_3b["method_name"].astype(str)) == {
        "stride_reference",
        "balanced_ot_baseline",
        "uot_baseline",
        "partial_ot_baseline",
        "diagonal_transport_baseline",
    }
    assert native_outputs_3b["A_json"].astype(str).str.startswith("[").all()

    raw_index_3b2 = pd.read_csv(result_3b2.raw_index_path, keep_default_na=False)
    assert set(raw_index_3b2["artifact_role"].astype(str)) == {
        "bundle_manifest",
        "raw_index",
        "generator_rerun_registry",
        "generator_split_registry",
        "patient_truth_store",
        "method_native_output_store",
        "3b2_patient_metrics",
        "3b2_condition_summary",
    }
    native_outputs_3b2 = pd.read_csv(result_3b2.raw_artifact_paths["method_native_output_store"])
    assert set(native_outputs_3b2["subexperiment_id"].astype(str)) == {"3B-2"}
    assert set(native_outputs_3b2["method_name"].astype(str)) == {
        "stride_reference",
        "uot_baseline",
        "partial_ot_baseline",
        "diagonal_transport_baseline",
    }
    assert "P_json" in native_outputs_3b2.columns
    plan_rows_3b2 = native_outputs_3b2.loc[
        native_outputs_3b2["method_name"].astype(str)
        .isin({"uot_baseline", "partial_ot_baseline", "diagonal_transport_baseline"})
    ].copy()
    assert not plan_rows_3b2.empty
    assert plan_rows_3b2["P_json"].astype(str).str.startswith("[").all()
    expected_open_mass_scales = {round(index / 10, 1) for index in range(11)}
    assert set(native_outputs_3b2["open_mass_scale"].astype(float).round(1)) == expected_open_mass_scales
    native_3b2_scales = native_outputs_3b2["open_mass_scale"].astype(float).to_numpy()
    assert not np.isclose(native_3b2_scales, 0.25).any()
    assert "balanced_ot_baseline" not in set(native_outputs_3b2["method_name"].astype(str))
    assert "abundance_only_baseline" not in set(native_outputs_3b2["method_name"].astype(str))
    zero_scale_native = native_outputs_3b2.loc[
        native_outputs_3b2["open_mass_scale"].astype(float).round(1) == 0.0
    ].copy()
    assert not zero_scale_native.empty
    uot_metadata = json.loads(
        str(zero_scale_native.loc[zero_scale_native["method_name"].astype(str) == "uot_baseline"].iloc[0]["metadata_json"])
    )
    assert "selected_lambda" in uot_metadata
    assert "target_overlap" in uot_metadata
    assert "boundary_hit" in uot_metadata
    assert "achieved_by_lambda" in uot_metadata
    assert "absolute_error_by_lambda" in uot_metadata
    assert "lambda" in uot_metadata
    assert "solver_status" in uot_metadata
    assert "matched_mass" in uot_metadata

    truth_store_3b2 = pd.read_csv(result_3b2.raw_artifact_paths["patient_truth_store"])
    assert set(truth_store_3b2["subexperiment_id"].astype(str)) == {"3B-2"}
    assert set(truth_store_3b2["condition_id"].astype(str)) == {"open_mass_scale_grid"}
    assert set(truth_store_3b2["open_mass_scale"].astype(float).round(1)) == expected_open_mass_scales
    truth_3b2_scales = truth_store_3b2["open_mass_scale"].astype(float).to_numpy()
    assert not np.isclose(truth_3b2_scales, 0.25).any()
    zero_scale_truth = truth_store_3b2.loc[
        truth_store_3b2["open_mass_scale"].astype(float).round(1) == 0.0
    ].copy()
    assert not zero_scale_truth.empty
    assert set(zero_scale_truth["open_mass"].astype(float).round(12)) == {0.0}

    patient_metrics_3b2 = pd.read_csv(result_3b2.raw_artifact_paths["3b2_patient_metrics"])
    assert set(patient_metrics_3b2["method_name"].astype(str)) == {
        "stride_reference",
        "uot_baseline",
        "partial_ot_baseline",
        "diagonal_transport_baseline",
    }
    assert "balanced_ot_baseline" not in set(patient_metrics_3b2["method_name"].astype(str))
    assert "abundance_only_baseline" not in set(patient_metrics_3b2["method_name"].astype(str))
    assert set(patient_metrics_3b2["open_mass_scale"].astype(float).round(1)) == expected_open_mass_scales
    metric_3b2_scales = patient_metrics_3b2["open_mass_scale"].astype(float).to_numpy()
    assert not np.isclose(metric_3b2_scales, 0.25).any()
    zero_scale_metrics = patient_metrics_3b2.loc[
        patient_metrics_3b2["open_mass_scale"].astype(float).round(1) == 0.0
    ].copy()
    assert not zero_scale_metrics.empty
    zero_statuses = {
        metric_name: set(group["metric_status"].astype(str))
        for metric_name, group in zero_scale_metrics.groupby("metric_name")
    }
    assert zero_statuses["open_support_F1"] == {"not_applicable"}
    assert zero_statuses["d_MAE"] == {"reported"}
    assert zero_statuses["e_MAE"] == {"reported"}
    assert zero_statuses["d_MSE"] == {"reported"}
    assert zero_statuses["e_MSE"] == {"reported"}

    raw_index_3c1 = pd.read_csv(result_3c1.raw_index_path, keep_default_na=False)
    assert set(raw_index_3c1["artifact_role"].astype(str)) == {
        "bundle_manifest",
        "raw_index",
        "generator_rerun_registry",
        "generator_split_registry",
        "patient_truth_store",
        "method_native_output_store",
        "3c1_patient_metrics",
        "3c1_condition_summary",
    }
    patient_metrics_3c1 = pd.read_csv(result_3c1.raw_artifact_paths["3c1_patient_metrics"])
    assert set(patient_metrics_3c1["condition_id"].astype(str)) == {"open_module_shared_realization_set"}
    assert not patient_metrics_3c1["condition_id"].astype(str).str.startswith("relation_").any()
    assert set(patient_metrics_3c1["method_name"].astype(str)) == {
        "stride_reference",
        "open_channel_ablation",
    }
    native_outputs_3c1 = pd.read_csv(result_3c1.raw_artifact_paths["method_native_output_store"])
    assert set(native_outputs_3c1["method_name"].astype(str)) == {
        "stride_reference",
        "open_channel_ablation",
    }
    ablated_rows_3c1 = native_outputs_3c1.loc[
        native_outputs_3c1["method_name"].astype(str) == "open_channel_ablation"
    ].copy()
    assert not ablated_rows_3c1.empty
    ablated_d_totals = ablated_rows_3c1["d_json"].map(json.loads).map(
        lambda values: float(np.sum(np.asarray(values, dtype=float), dtype=float))
    )
    ablated_e_totals = ablated_rows_3c1["e_json"].map(json.loads).map(
        lambda values: float(np.sum(np.asarray(values, dtype=float), dtype=float))
    )
    assert (ablated_d_totals > 0.0).any()
    assert (ablated_e_totals > 0.0).any()

    patient_metrics_3c2 = pd.read_csv(result_3c2.raw_artifact_paths["3c2_patient_metrics"])
    assert set(patient_metrics_3c2["condition_id"].astype(str)) == {"cohort_module_shared_realization_set"}
    assert not patient_metrics_3c2["condition_id"].astype(str).str.startswith("relation_").any()
    assert set(patient_metrics_3c2["method_name"].astype(str)) == {
        "stride_reference",
        "cohort_ablation",
    }
    assert set(patient_metrics_3c2["metric_name"].astype(str)) == {
        "A_MAE_active",
        "A_MSE_active",
        "open_support_F1",
        "d_MAE",
        "e_MAE",
        "d_MSE",
        "e_MSE",
    }

    review_index_3b = pd.read_csv(result_3b.review_index_path, keep_default_na=False)
    assert set(review_index_3b["artifact_role"].astype(str)) == {
        "review_manifest",
        "review_index",
        "3b1_review_surface",
    }
    review_index_3c1 = pd.read_csv(result_3c1.review_index_path, keep_default_na=False)
    assert set(review_index_3c1["artifact_role"].astype(str)) == {
        "review_manifest",
        "review_index",
        "3c1_review_surface",
    }
    review_index_3c2 = pd.read_csv(result_3c2.review_index_path, keep_default_na=False)
    assert set(review_index_3c2["artifact_role"].astype(str)) == {
        "review_manifest",
        "review_index",
        "3c2_review_surface",
    }
