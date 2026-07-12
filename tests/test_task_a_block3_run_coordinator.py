from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stride.errors import ContractError
from tasks.task_A.block3 import execution
from tasks.task_A.block3.run import (
    _load_generator_bundle,
    _write_generator_bundle,
    _write_generator_review_decision,
)


def _cohort(tmp_path: Path) -> execution.Block3CohortInputs:
    patient_ids = tuple(f"p{index:02d}" for index in range(26))
    profiles = {
        patient_id: np.array([0.6, 0.4], dtype=float) for patient_id in patient_ids
    }
    fovs = {
        patient_id: np.array([[0.7, 0.3], [0.5, 0.5]], dtype=float)
        for patient_id in patient_ids
    }
    return execution.Block3CohortInputs(
        stage0_h5ad=tmp_path / "stage0.h5ad",
        config_path=tmp_path / "config.yaml",
        output_dir=tmp_path,
        master_seed=17,
        state_ids=(0, 1),
        state_basis=None,
        identity_vectors=np.eye(2),
        cost_matrix=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        cost_scale=1.0,
        source_domain="TC",
        target_domain="IM",
        patient_source_fovs=fovs,
        patient_target_fovs=fovs,
        patient_source_profiles=profiles,
        patient_target_profiles=profiles,
    )


def _reruns() -> tuple[execution.Block3GeneratorRerun, ...]:
    truths = {}
    for index, patient_id in enumerate(("p24", "p25")):
        truths[patient_id] = execution.Block3PatientTruth(
            rerun_id="rerun_01",
            patient_id=patient_id,
            x=np.array([0.6, 0.4]),
            y=np.array([0.5, 0.5]),
            A=np.array([[0.8, 0.1], [0.1, 0.8]]),
            d=np.array([0.1, 0.1]),
            e=np.array([0.05, 0.05]),
            open_mass=0.1,
            y_endpoint=np.array([0.5, 0.5]),
            source_fovs=np.array([[0.7, 0.3], [0.5, 0.5]]),
            target_fovs=np.array([[0.4, 0.6], [0.6, 0.4], [0.5, 0.5]]),
            sampled_template_patient_id=f"p0{index}",
            medoid_template_patient_id="p00",
            row_imputed_mask=np.array([False, True]),
            endpoint_closure_l1=0.0,
            generator_diagnostics={"finite": True, "index": index},
        )
    return (
        execution.Block3GeneratorRerun(
            rerun_id="rerun_01",
            split_seed=17,
            train_patient_ids=tuple(f"p{index:02d}" for index in range(24)),
            test_patient_ids=("p24", "p25"),
            hidden_relation_condition_id="shared",
            open_mass_scale=1.0,
            generator_truths=truths,
            baseline_truths={
                execution._A_BENCHMARK_CONDITION_ID: dict(truths),
                execution._DE_BENCHMARK_CONDITION_ID: dict(truths),
            },
            template_medoid_patient_id="p00",
            generator_parameters={"tau": 2.0},
        ),
    )


def test_generator_bundle_roundtrip_preserves_axes_and_fovs(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "generator_bundle.npz"
    cohort = _cohort(tmp_path)
    expected = _reruns()

    _write_generator_bundle(path, reruns=expected, cohort=cohort)
    observed = _load_generator_bundle(path, cohort=cohort)

    assert observed[0].train_patient_ids == expected[0].train_patient_ids
    assert observed[0].test_patient_ids == expected[0].test_patient_ids
    for patient_id in expected[0].test_patient_ids:
        expected_truth = expected[0].generator_truths[patient_id]
        observed_truth = observed[0].generator_truths[patient_id]
        np.testing.assert_array_equal(observed_truth.A, expected_truth.A)
        np.testing.assert_array_equal(observed_truth.d, expected_truth.d)
        np.testing.assert_array_equal(observed_truth.e, expected_truth.e)
        np.testing.assert_array_equal(observed_truth.source_fovs, expected_truth.source_fovs)
        np.testing.assert_array_equal(observed_truth.target_fovs, expected_truth.target_fovs)
        np.testing.assert_array_equal(
            observed_truth.row_imputed_mask,
            expected_truth.row_imputed_mask,
        )


def test_generator_bundle_checksum_corruption_fails(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "generator_bundle.npz"
    cohort = _cohort(tmp_path)
    _write_generator_bundle(path, reruns=_reruns(), cohort=cohort)
    path.write_bytes(path.read_bytes() + b"corrupt")

    with pytest.raises(ContractError, match="checksum"):
        _load_generator_bundle(path, cohort=cohort)


def test_stride_cache_checksum_corruption_fails(tmp_path: Path) -> None:
    runtime = execution.Block3RuntimeControls(cache_dir=tmp_path / "cache")
    output = execution.Block3MethodOutput(
        patient_id="p24",
        fit_status="ok",
        A=np.eye(2),
        d=np.zeros(2),
        e=np.zeros(2),
        mu_minus=np.array([0.6, 0.4]),
        mu_plus=np.array([0.5, 0.5]),
        metadata={"fit_surface": "stride.tl.fit"},
    )
    execution._write_stride_method_cache(
        runtime=runtime,
        rerun_id="rerun_01",
        ablation_mode="none",
        outputs={"p24": output},
    )
    cache_path = tmp_path / "cache" / "reference" / "rerun_01.npz"
    cache_path.write_bytes(cache_path.read_bytes() + b"corrupt")
    truth = execution.Block3PatientTruth(
        rerun_id="rerun_01",
        patient_id="p24",
        x=np.array([0.6, 0.4]),
        y=np.array([0.5, 0.5]),
        A=np.eye(2),
        d=np.zeros(2),
        e=np.zeros(2),
        open_mass=0.0,
    )

    with pytest.raises(ContractError, match="checksum"):
        execution._load_stride_method_cache(
            runtime=runtime,
            rerun_id="rerun_01",
            ablation_mode="none",
            truths=[truth],
        )


def test_generator_review_decision_records_observed_ordering(tmp_path: Path) -> None:
    diagnostics_path = tmp_path / "raw" / "generator_diagnostics.csv"
    diagnostics_path.parent.mkdir(parents=True)
    diagnostics_path.write_text(
        "truth_finite,burden_ordering_status\n"
        "True,retained_gt_offdiag_gt_open\n"
        "True,outside_target_order\n",
        encoding="utf-8",
    )

    decision_path = _write_generator_review_decision(
        tmp_path,
        review_note="Accepted without generator retuning.",
    )

    payload = __import__("json").loads(decision_path.read_text(encoding="utf-8"))
    assert payload["decision"] == "accepted_for_downstream_execution"
    assert payload["n_truth_records"] == 2
    assert payload["n_truth_finite"] == 2
    assert payload["burden_ordering_counts"] == {
        "outside_target_order": 1,
        "retained_gt_offdiag_gt_open": 1,
    }
    assert payload["generator_retuned"] is False
