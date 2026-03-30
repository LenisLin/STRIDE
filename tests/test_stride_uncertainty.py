from __future__ import annotations

import numpy as np
import pytest

from stride.api.fit import fit_stride
from stride.basis import load_state_basis
from stride.errors import ContractError
from stride.observation import FovObservation
from stride.outputs.uncertainty import (
    BootstrapArraySummary,
    CohortBootstrapUncertaintySummary,
    PatientBootstrapConfig,
    PatientBootstrapUncertaintyResult,
    STRIDEBootstrapUncertaintyResult,
)
from stride.workflows.fit_stride import STRIDEFitConfig


def _make_state_basis() -> object:
    return load_state_basis(
        centroids=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        cost_matrix=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        cost_scale=1.0,
        state_ids=(0, 1),
    )


def _make_observation(
    *,
    patient_id: str,
    timepoint: str,
    fov_id: str,
    domain_label: str,
    composition: tuple[float, ...],
) -> FovObservation:
    return FovObservation(
        patient_id=patient_id,
        timepoint=timepoint,
        fov_id=fov_id,
        domain_label=domain_label,
        community_composition=np.asarray(composition, dtype=float),
        mass=1.0,
        mass_mode="uniform",
    )


def _make_bootstrap_ready_observations() -> tuple[FovObservation, ...]:
    return (
        _make_observation(
            patient_id="p_boot",
            timepoint="pre",
            fov_id="p_boot_pre_tc_1",
            domain_label="TC",
            composition=(0.90, 0.10),
        ),
        _make_observation(
            patient_id="p_boot",
            timepoint="pre",
            fov_id="p_boot_pre_tc_2",
            domain_label="TC",
            composition=(0.65, 0.35),
        ),
        _make_observation(
            patient_id="p_boot",
            timepoint="pre",
            fov_id="p_boot_pre_im_1",
            domain_label="IM",
            composition=(0.55, 0.45),
        ),
        _make_observation(
            patient_id="p_boot",
            timepoint="pre",
            fov_id="p_boot_pre_im_2",
            domain_label="IM",
            composition=(0.30, 0.70),
        ),
        _make_observation(
            patient_id="p_boot",
            timepoint="post",
            fov_id="p_boot_post_tc_1",
            domain_label="TC",
            composition=(0.85, 0.15),
        ),
        _make_observation(
            patient_id="p_boot",
            timepoint="post",
            fov_id="p_boot_post_tc_2",
            domain_label="TC",
            composition=(0.35, 0.65),
        ),
        _make_observation(
            patient_id="p_boot",
            timepoint="post",
            fov_id="p_boot_post_im_1",
            domain_label="IM",
            composition=(0.60, 0.40),
        ),
        _make_observation(
            patient_id="p_boot",
            timepoint="post",
            fov_id="p_boot_post_im_2",
            domain_label="IM",
            composition=(0.15, 0.85),
        ),
    )


def _make_unsupported_observations() -> tuple[FovObservation, ...]:
    return (
        _make_observation(
            patient_id="p_deferred",
            timepoint="pre",
            fov_id="p_deferred_pre_tc",
            domain_label="TC",
            composition=(0.7, 0.3),
        ),
        _make_observation(
            patient_id="p_deferred",
            timepoint="post",
            fov_id="p_deferred_post_tc",
            domain_label="TC",
            composition=(0.5, 0.5),
        ),
        _make_observation(
            patient_id="p_deferred",
            timepoint="followup",
            fov_id="p_deferred_followup_tc",
            domain_label="TC",
            composition=(0.3, 0.7),
        ),
    )


def _make_array_summary(shape: tuple[int, ...]) -> BootstrapArraySummary:
    return BootstrapArraySummary(
        mean=np.zeros(shape, dtype=float),
        std=np.zeros(shape, dtype=float),
        nonzero_frequency=np.zeros(shape, dtype=float),
        mean_abs_deviation=0.0,
        max_abs_deviation=0.0,
    )


def test_fit_stride_bootstrap_uncertainty_realizes_supported_patient() -> None:
    result = fit_stride(
        _make_bootstrap_ready_observations(),
        state_basis=_make_state_basis(),
        config=STRIDEFitConfig(
            timepoint_order=("pre", "post"),
            uncertainty_config=PatientBootstrapConfig(n_boot=8, random_state=17),
        ),
    )

    assert result.uncertainty is not None
    assert result.summaries["uncertainty_status"] == "ok"
    patient_uncertainty = result.uncertainty.patient_results[0]
    assert patient_uncertainty.patient_id == "p_boot"
    assert patient_uncertainty.eligible is True
    assert patient_uncertainty.uncertainty_status == "ok"
    assert patient_uncertainty.n_boot == 8
    assert patient_uncertainty.n_ok == 8
    assert patient_uncertainty.n_deferred == 0
    assert patient_uncertainty.n_failed == 0
    assert result.uncertainty.cohort_summary.n_eligible_patients == 1
    assert result.uncertainty.cohort_summary.n_realized_patients == 1
    assert result.uncertainty.cohort_summary.mean_patient_success_rate == 1.0

    realized_patient = result.patient_results[0]
    assert patient_uncertainty.A_summary is not None
    assert patient_uncertainty.d_summary is not None
    assert patient_uncertainty.e_summary is not None
    assert patient_uncertainty.A_summary.mean.shape == realized_patient.A.shape
    assert patient_uncertainty.d_summary.mean.shape == realized_patient.d.shape
    assert patient_uncertainty.e_summary.mean.shape == realized_patient.e.shape
    assert np.any(patient_uncertainty.A_summary.std > 0.0)
    assert np.any(patient_uncertainty.d_summary.std > 0.0)
    assert np.any(patient_uncertainty.e_summary.std > 0.0)
    assert patient_uncertainty.A_summary.mean_abs_deviation >= 0.0

    expected_group_counts = dict(result.patient_inputs[0].n_observations_by_group)
    expected_group_domain_counts = {
        group_label: dict(domain_counts)
        for group_label, domain_counts in result.patient_inputs[0].n_observations_by_group_and_domain.items()
    }
    for replicate_diagnostic in patient_uncertainty.replicate_diagnostics:
        assert replicate_diagnostic["status"] == "ok"
        assert replicate_diagnostic["n_observations_by_group"] == expected_group_counts
        assert replicate_diagnostic["n_observations_by_group_and_domain"] == expected_group_domain_counts


def test_fit_stride_bootstrap_uncertainty_is_reproducible_with_fixed_seed() -> None:
    observations = _make_bootstrap_ready_observations()
    state_basis = _make_state_basis()
    config = STRIDEFitConfig(
        timepoint_order=("pre", "post"),
        uncertainty_config=PatientBootstrapConfig(n_boot=6, random_state=29),
    )

    result_one = fit_stride(observations, state_basis=state_basis, config=config)
    result_two = fit_stride(observations, state_basis=state_basis, config=config)

    assert result_one.uncertainty is not None
    assert result_two.uncertainty is not None
    patient_one = result_one.uncertainty.patient_results[0]
    patient_two = result_two.uncertainty.patient_results[0]
    assert patient_one.bootstrap_seed == patient_two.bootstrap_seed
    assert patient_one.replicate_statuses == patient_two.replicate_statuses
    np.testing.assert_allclose(patient_one.A_summary.mean, patient_two.A_summary.mean)
    np.testing.assert_allclose(patient_one.A_summary.std, patient_two.A_summary.std)
    np.testing.assert_allclose(patient_one.d_summary.mean, patient_two.d_summary.mean)
    np.testing.assert_allclose(patient_one.e_summary.mean, patient_two.e_summary.mean)


def test_fit_stride_bootstrap_uncertainty_defers_unsupported_patient() -> None:
    result = fit_stride(
        _make_unsupported_observations(),
        state_basis=_make_state_basis(),
        config=STRIDEFitConfig(
            timepoint_order=("pre", "post", "followup"),
            uncertainty_config=PatientBootstrapConfig(n_boot=5, random_state=3),
        ),
    )

    assert result.uncertainty is not None
    assert result.patient_results[0].fit_status == "deferred"
    patient_uncertainty = result.uncertainty.patient_results[0]
    assert patient_uncertainty.eligible is False
    assert patient_uncertainty.uncertainty_status == "deferred"
    assert patient_uncertainty.A_summary is None
    assert patient_uncertainty.d_summary is None
    assert patient_uncertainty.e_summary is None
    assert patient_uncertainty.replicate_statuses == ("deferred",) * 5
    assert result.uncertainty.cohort_summary.uncertainty_status == "deferred"
    assert result.uncertainty.cohort_summary.n_eligible_patients == 0
    assert result.uncertainty.cohort_summary.n_realized_patients == 0


def test_patient_bootstrap_uncertainty_rejects_invalid_status_combinations() -> None:
    with pytest.raises(
        ContractError,
        match="Non-ok realized_fit_status requires uncertainty_status='deferred'",
    ):
        PatientBootstrapUncertaintyResult(
            patient_id="p1",
            realized_fit_status="deferred",
            uncertainty_status="failed",
            eligible=False,
            n_boot=2,
            replicate_statuses=("failed", "failed"),
            replicate_diagnostics=({}, {}),
        )

    with pytest.raises(
        ContractError,
        match="Non-ok PatientBootstrapUncertaintyResult objects must not carry bootstrap summaries",
    ):
        PatientBootstrapUncertaintyResult(
            patient_id="p1",
            realized_fit_status="ok",
            uncertainty_status="deferred",
            eligible=True,
            n_boot=2,
            replicate_statuses=("deferred", "deferred"),
            replicate_diagnostics=({}, {}),
            A_summary=_make_array_summary((2, 2)),
            d_summary=_make_array_summary((2,)),
            e_summary=_make_array_summary((2,)),
        )


def test_patient_bootstrap_uncertainty_rejects_invalid_summary_shapes() -> None:
    with pytest.raises(
        ContractError,
        match="must align to one shared K-state axis",
    ):
        PatientBootstrapUncertaintyResult(
            patient_id="p1",
            realized_fit_status="ok",
            uncertainty_status="ok",
            eligible=True,
            n_boot=2,
            replicate_statuses=("ok", "ok"),
            replicate_diagnostics=({}, {}),
            A_summary=_make_array_summary((2, 2)),
            d_summary=_make_array_summary((3,)),
            e_summary=_make_array_summary((2,)),
        )


def test_stride_bootstrap_uncertainty_result_rejects_incoherent_cohort_summary() -> None:
    patient_result = PatientBootstrapUncertaintyResult(
        patient_id="p1",
        realized_fit_status="deferred",
        uncertainty_status="deferred",
        eligible=False,
        n_boot=2,
        replicate_statuses=("deferred", "deferred"),
        replicate_diagnostics=({}, {}),
    )

    with pytest.raises(
        ContractError,
        match="patient_status_counts must match patient_results",
    ):
        STRIDEBootstrapUncertaintyResult(
            config=PatientBootstrapConfig(n_boot=2, random_state=0),
            patient_results=(patient_result,),
            cohort_summary=CohortBootstrapUncertaintySummary(
                uncertainty_status="deferred",
                n_patients=1,
                n_eligible_patients=0,
                n_realized_patients=0,
                patient_status_counts={"ok": 1, "deferred": 0, "failed": 0},
                mean_patient_success_rate=float("nan"),
                mean_patient_A_mean_element_std=float("nan"),
                mean_patient_d_mean_element_std=float("nan"),
                mean_patient_e_mean_element_std=float("nan"),
            ),
        )
