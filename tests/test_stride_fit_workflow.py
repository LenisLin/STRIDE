from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stride.api.basis import BasisSpec
from stride.api.dataset import DatasetHandle
from stride.api.fit import bridge_observation_matches, build_patient_relation, fit_stride
from stride.basis import load_state_basis
from stride.errors import ContractError
from stride.geometry import build_state_geometry
from stride.latent.operators import PatientRelationAudit
from stride.latent.recurrence import RecurrenceResult, estimate_recurrence
from stride.observation import FovObservation
from stride.outputs.fit_result import PatientBridgeResult, STRIDEFitResult
from stride.outputs.uncertainty import (
    CohortBootstrapUncertaintySummary,
    PatientBootstrapConfig,
    PatientBootstrapUncertaintyResult,
    STRIDEBootstrapUncertaintyResult,
)
from stride.workflows.fit_stride import STRIDEFitConfig, build_patient_bridge_inputs, run_stride_fit


def _make_state_basis() -> object:
    return load_state_basis(
        centroids=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        cost_matrix=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        cost_scale=1.0,
        state_ids=(0, 1),
    )


def _make_three_state_basis() -> object:
    return load_state_basis(
        centroids=np.asarray([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=float),
        cost_matrix=np.asarray(
            [
                [0.0, 1.0, 3.0],
                [1.0, 0.0, 1.0],
                [3.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        cost_scale=1.0,
        state_ids=(0, 1, 2),
    )


def _make_geometry(state_basis: object, *, n_neighbors: int = 5) -> object:
    return build_state_geometry(
        cost_matrix=state_basis.cost_matrix,
        cost_scale=state_basis.cost_scale,
        state_ids=state_basis.resolved_state_ids,
        n_neighbors=n_neighbors,
    )


def _make_observation(
    *,
    patient_id: str,
    timepoint: str,
    fov_id: str,
    domain_label: str | None,
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


def _make_grouped_observations() -> tuple[FovObservation, ...]:
    return (
        _make_observation(
            patient_id="p1",
            timepoint="post",
            fov_id="p1_post_tc",
            domain_label="TC",
            composition=(0.7, 0.3),
        ),
        _make_observation(
            patient_id="p1",
            timepoint="pre",
            fov_id="p1_pre_tc",
            domain_label="TC",
            composition=(0.6, 0.4),
        ),
        _make_observation(
            patient_id="p1",
            timepoint="post",
            fov_id="p1_post_im",
            domain_label="IM",
            composition=(0.2, 0.8),
        ),
        _make_observation(
            patient_id="p2",
            timepoint="pre",
            fov_id="p2_pre_tc",
            domain_label="TC",
            composition=(0.5, 0.5),
        ),
        _make_observation(
            patient_id="p2",
            timepoint="post",
            fov_id="p2_post_tc",
            domain_label="TC",
            composition=(0.4, 0.6),
        ),
    )


def _make_three_state_shift_observations() -> tuple[FovObservation, ...]:
    return (
        _make_observation(
            patient_id="p3",
            timepoint="pre",
            fov_id="p3_pre_tc",
            domain_label="TC",
            composition=(0.8, 0.2, 0.0),
        ),
        _make_observation(
            patient_id="p3",
            timepoint="post",
            fov_id="p3_post_tc",
            domain_label="TC",
            composition=(0.2, 0.0, 0.8),
        ),
    )


def test_build_patient_bridge_inputs_groups_by_patient_and_domain() -> None:
    state_basis = _make_state_basis()
    geometry = _make_geometry(state_basis)

    bundles = build_patient_bridge_inputs(
        _make_grouped_observations(),
        state_basis=state_basis,
        geometry=geometry,
        config=STRIDEFitConfig(timepoint_order=("pre", "post")),
    )

    assert [bundle.patient_id for bundle in bundles] == ["p1", "p2"]

    p1_bundle = bundles[0]
    assert p1_bundle.ordered_group_labels == ("pre", "post")
    assert dict(p1_bundle.n_observations_by_group) == {"pre": 1, "post": 2}
    assert dict(p1_bundle.n_observations_by_domain) == {"TC": 2, "IM": 1}
    assert {
        group_label: dict(domain_counts)
        for group_label, domain_counts in p1_bundle.n_observations_by_group_and_domain.items()
    } == {
        "pre": {"TC": 1},
        "post": {"TC": 1, "IM": 1},
    }
    assert [observation.fov_id for observation in p1_bundle.groups_by_label["post"].observations] == [
        "p1_post_tc",
        "p1_post_im",
    ]
    assert list(p1_bundle.groups_by_label["post"].observations_by_domain) == ["TC", "IM"]


def test_build_patient_bridge_inputs_rejects_missing_domain_label() -> None:
    state_basis = _make_state_basis()

    with pytest.raises(ContractError, match="must declare domain_label"):
        build_patient_bridge_inputs(
            (
                _make_observation(
                    patient_id="p1",
                    timepoint="pre",
                    fov_id="p1_pre",
                    domain_label=None,
                    composition=(0.6, 0.4),
                ),
            ),
            state_basis=state_basis,
            config=STRIDEFitConfig(timepoint_order=("pre",)),
        )


def test_patient_bridge_result_exposes_valid_relation() -> None:
    result = PatientBridgeResult(
        patient_id="p1",
        fit_status="ok",
        A=np.asarray([[0.6, 0.2], [0.1, 0.7]], dtype=float),
        d=np.asarray([0.2, 0.2], dtype=float),
        e=np.asarray([0.1, 0.2], dtype=float),
        state_ids=(0, 1),
        audit=PatientRelationAudit(patient_id="p1", bridge_status="ok"),
    )

    assert result.is_ok
    relation = result.relation
    assert relation is not None
    np.testing.assert_allclose(relation.A, np.asarray([[0.6, 0.2], [0.1, 0.7]], dtype=float))
    np.testing.assert_allclose(relation.d, np.asarray([0.2, 0.2], dtype=float))
    np.testing.assert_allclose(relation.e, np.asarray([0.1, 0.2], dtype=float))


def test_patient_bridge_result_rejects_invalid_arrays_and_status_mixing() -> None:
    with pytest.raises(ContractError, match="row-substochastic"):
        PatientBridgeResult(
            patient_id="p1",
            fit_status="ok",
            A=np.asarray([[0.9, 0.3], [0.1, 0.7]], dtype=float),
            d=np.asarray([0.1, 0.2], dtype=float),
            e=np.asarray([0.1, 0.2], dtype=float),
        )

    with pytest.raises(ContractError, match="must be non-negative"):
        PatientBridgeResult(
            patient_id="p1",
            fit_status="ok",
            A=np.asarray([[0.6, 0.2], [0.1, 0.7]], dtype=float),
            d=np.asarray([0.2, -0.1], dtype=float),
            e=np.asarray([0.1, 0.2], dtype=float),
        )

    with pytest.raises(ContractError, match="must not carry bridge arrays"):
        PatientBridgeResult(
            patient_id="p1",
            fit_status="deferred",
            A=np.eye(2, dtype=float),
            d=np.zeros(2, dtype=float),
            e=np.zeros(2, dtype=float),
        )


def test_patient_bridge_result_rejects_vector_shape_drift_and_duplicate_state_ids() -> None:
    with pytest.raises(ContractError, match="d must be a 1D"):
        PatientBridgeResult(
            patient_id="p1",
            fit_status="ok",
            A=np.asarray([[0.6, 0.2], [0.1, 0.7]], dtype=float),
            d=np.asarray([[0.2], [0.2]], dtype=float),
            e=np.asarray([0.1, 0.2], dtype=float),
        )

    with pytest.raises(ContractError, match="state_ids must be unique"):
        PatientBridgeResult(
            patient_id="p1",
            fit_status="ok",
            A=np.asarray([[0.6, 0.2], [0.1, 0.7]], dtype=float),
            d=np.asarray([0.2, 0.2], dtype=float),
            e=np.asarray([0.1, 0.2], dtype=float),
            state_ids=(0, 0),
        )


def test_patient_bridge_result_rejects_incoherent_auxiliary_burden_payloads() -> None:
    with pytest.raises(ContractError, match="realized auxiliary bridge arrays"):
        PatientBridgeResult(
            patient_id="p1",
            fit_status="deferred",
            auxiliary={"matched_transition_burden": np.eye(2, dtype=float)},
        )

    with pytest.raises(ContractError, match="must agree with A"):
        PatientBridgeResult(
            patient_id="p1",
            fit_status="ok",
            A=np.asarray([[0.6, 0.2], [0.1, 0.7]], dtype=float),
            d=np.asarray([0.2, 0.2], dtype=float),
            e=np.asarray([0.1, 0.2], dtype=float),
            mu_minus=np.asarray([0.6, 0.4], dtype=float),
            mu_plus=np.asarray([0.45, 0.55], dtype=float),
            auxiliary={
                "matched_transition_burden": np.asarray(
                    [[0.2, 0.1], [0.1, 0.1]],
                    dtype=float,
                ),
            },
        )


def test_bridge_observation_matches_remains_explicitly_deferred() -> None:
    with pytest.raises(NotImplementedError) as exc_info:
        bridge_observation_matches()

    message = str(exc_info.value)
    assert "remains deferred" in message
    assert "Use fit_stride" in message
    assert "build_patient_relation" in message


def test_estimate_recurrence_returns_explicit_deferred_result() -> None:
    relation = build_patient_relation(
        patient_id="p1",
        A=np.asarray([[0.7, 0.1], [0.2, 0.6]], dtype=float),
        d=np.asarray([0.2, 0.2], dtype=float),
        e=np.asarray([0.1, 0.2], dtype=float),
        state_ids=(0, 1),
    )

    result = estimate_recurrence((relation,))

    assert result.fit_status == "deferred"
    assert result.patient_ids == ("p1",)
    assert result.families == ()
    assert result.parameters is not None
    assert result.parameters.basis_dim == 2
    assert result.embeddings[0].patient_id == "p1"
    assert result.embeddings[0].fit_status == "deferred"
    assert np.isnan(result.embeddings[0].coordinates).all()
    assert "remains deferred" in str(result.metadata["message"])


def test_stride_fit_result_validates_patient_recurrence_alignment() -> None:
    state_basis = _make_state_basis()
    geometry = _make_geometry(state_basis)
    patient_input = build_patient_bridge_inputs(
        (
            _make_observation(
                patient_id="p1",
                timepoint="pre",
                fov_id="p1_pre_tc",
                domain_label="TC",
                composition=(0.6, 0.4),
            ),
            _make_observation(
                patient_id="p1",
                timepoint="post",
                fov_id="p1_post_tc",
                domain_label="TC",
                composition=(0.4, 0.6),
            ),
        ),
        state_basis=state_basis,
        geometry=geometry,
        config=STRIDEFitConfig(timepoint_order=("pre", "post")),
    )[0]
    patient_result = PatientBridgeResult(patient_id="p1", fit_status="deferred")

    with pytest.raises(ContractError, match="recurrence.patient_ids"):
        STRIDEFitResult(
            patient_inputs=(patient_input,),
            patient_results=(patient_result,),
            recurrence=RecurrenceResult(patient_ids=("p2",), families=(), fit_status="deferred"),
            fit_status="deferred",
        )


def test_stride_fit_result_validates_uncertainty_alignment() -> None:
    state_basis = _make_state_basis()
    geometry = _make_geometry(state_basis)
    patient_input = build_patient_bridge_inputs(
        (
            _make_observation(
                patient_id="p1",
                timepoint="pre",
                fov_id="p1_pre_tc",
                domain_label="TC",
                composition=(0.6, 0.4),
            ),
            _make_observation(
                patient_id="p1",
                timepoint="post",
                fov_id="p1_post_tc",
                domain_label="TC",
                composition=(0.4, 0.6),
            ),
        ),
        state_basis=state_basis,
        geometry=geometry,
        config=STRIDEFitConfig(timepoint_order=("pre", "post")),
    )[0]
    patient_result = PatientBridgeResult(patient_id="p1", fit_status="deferred")
    uncertainty = STRIDEBootstrapUncertaintyResult(
        config=PatientBootstrapConfig(n_boot=2, random_state=0),
        patient_results=(
            PatientBootstrapUncertaintyResult(
                patient_id="p2",
                realized_fit_status="deferred",
                uncertainty_status="deferred",
                eligible=False,
                n_boot=2,
                replicate_statuses=("deferred", "deferred"),
                replicate_diagnostics=({}, {}),
            ),
        ),
        cohort_summary=CohortBootstrapUncertaintySummary(
            uncertainty_status="deferred",
            n_patients=1,
            n_eligible_patients=0,
            n_realized_patients=0,
            patient_status_counts={"ok": 0, "deferred": 1, "failed": 0},
            mean_patient_success_rate=float("nan"),
            mean_patient_A_mean_element_std=float("nan"),
            mean_patient_d_mean_element_std=float("nan"),
            mean_patient_e_mean_element_std=float("nan"),
        ),
    )

    with pytest.raises(ContractError, match="uncertainty.patient_ids"):
        STRIDEFitResult(
            patient_inputs=(patient_input,),
            patient_results=(patient_result,),
            recurrence=RecurrenceResult(patient_ids=("p1",), families=(), fit_status="deferred"),
            fit_status="deferred",
            uncertainty=uncertainty,
        )


def test_stride_fit_result_validates_patient_status_count_summaries() -> None:
    state_basis = _make_state_basis()
    geometry = _make_geometry(state_basis)
    patient_input = build_patient_bridge_inputs(
        (
            _make_observation(
                patient_id="p1",
                timepoint="pre",
                fov_id="p1_pre_tc",
                domain_label="TC",
                composition=(0.6, 0.4),
            ),
            _make_observation(
                patient_id="p1",
                timepoint="post",
                fov_id="p1_post_tc",
                domain_label="TC",
                composition=(0.4, 0.6),
            ),
        ),
        state_basis=state_basis,
        geometry=geometry,
        config=STRIDEFitConfig(timepoint_order=("pre", "post")),
    )[0]
    patient_result = PatientBridgeResult(patient_id="p1", fit_status="deferred")

    with pytest.raises(ContractError, match="summaries\\['patient_status_counts'\\]"):
        STRIDEFitResult(
            patient_inputs=(patient_input,),
            patient_results=(patient_result,),
            recurrence=RecurrenceResult(patient_ids=("p1",), families=(), fit_status="deferred"),
            fit_status="deferred",
            summaries={"patient_status_counts": {"ok": 1}},
        )


def test_stride_fit_result_rejects_uncertainty_realized_fit_status_drift() -> None:
    state_basis = _make_state_basis()
    geometry = _make_geometry(state_basis)
    patient_input = build_patient_bridge_inputs(
        (
            _make_observation(
                patient_id="p1",
                timepoint="pre",
                fov_id="p1_pre_tc",
                domain_label="TC",
                composition=(0.6, 0.4),
            ),
            _make_observation(
                patient_id="p1",
                timepoint="post",
                fov_id="p1_post_tc",
                domain_label="TC",
                composition=(0.4, 0.6),
            ),
        ),
        state_basis=state_basis,
        geometry=geometry,
        config=STRIDEFitConfig(timepoint_order=("pre", "post")),
    )[0]
    patient_result = PatientBridgeResult(patient_id="p1", fit_status="deferred")
    uncertainty = STRIDEBootstrapUncertaintyResult(
        config=PatientBootstrapConfig(n_boot=2, random_state=0),
        patient_results=(
            PatientBootstrapUncertaintyResult(
                patient_id="p1",
                realized_fit_status="failed",
                uncertainty_status="deferred",
                eligible=False,
                n_boot=2,
                replicate_statuses=("deferred", "deferred"),
                replicate_diagnostics=({}, {}),
            ),
        ),
        cohort_summary=CohortBootstrapUncertaintySummary(
            uncertainty_status="deferred",
            n_patients=1,
            n_eligible_patients=0,
            n_realized_patients=0,
            patient_status_counts={"ok": 0, "deferred": 1, "failed": 0},
            mean_patient_success_rate=float("nan"),
            mean_patient_A_mean_element_std=float("nan"),
            mean_patient_d_mean_element_std=float("nan"),
            mean_patient_e_mean_element_std=float("nan"),
        ),
    )

    with pytest.raises(ContractError, match="preserve each realized patient fit_status"):
        STRIDEFitResult(
            patient_inputs=(patient_input,),
            patient_results=(patient_result,),
            recurrence=RecurrenceResult(patient_ids=("p1",), families=(), fit_status="deferred"),
            fit_status="deferred",
            uncertainty=uncertainty,
        )


def test_fit_stride_realizes_supported_two_group_patient_bridge() -> None:
    state_basis = _make_state_basis()
    observations = tuple(
        observation
        for observation in _make_grouped_observations()
        if observation.patient_id == "p1"
    )

    result = fit_stride(
        observations,
        state_basis=state_basis,
        config=STRIDEFitConfig(timepoint_order=("pre", "post")),
    )

    assert isinstance(result, STRIDEFitResult)
    assert result.fit_status == "deferred"
    assert result.uncertainty is None
    assert result.patient_ids == ("p1",)
    patient_result = result.patient_results[0]
    assert patient_result.fit_status == "ok"
    assert patient_result.relation is not None
    np.testing.assert_allclose(patient_result.mu_minus, np.asarray([0.6, 0.4], dtype=float))
    np.testing.assert_allclose(patient_result.mu_plus, np.asarray([0.45, 0.55], dtype=float))
    assert patient_result.diagnostics["supported_case"] == "two_group_uniform_patient_bridge"
    assert patient_result.diagnostics["estimator_mode"] == "observation_to_patient_bridge_v1"
    assert (
        patient_result.diagnostics["estimator_method"]
        == "domain_stratified_cartesian_observation_discrepancy"
    )
    assert "domain_coupling_mode" not in patient_result.diagnostics
    assert "geometry_transport_mode" not in patient_result.diagnostics
    assert patient_result.diagnostics["matched_burden_mass"] > 0.0
    assert len(patient_result.diagnostics["domain_pair_statuses"]) == 2
    assert patient_result.audit is not None
    assert patient_result.audit.bridge_status == "ok"
    assert patient_result.audit.observation_fit_status == "observation_discrepancy_bridge"
    assert patient_result.audit.n_pre_observations == 1
    assert patient_result.audit.n_post_observations == 2


def test_fit_stride_realized_bridge_preserves_legality_and_offdiagonal_mass() -> None:
    state_basis = _make_state_basis()
    result = fit_stride(
        tuple(
            observation
            for observation in _make_grouped_observations()
            if observation.patient_id == "p1"
        ),
        state_basis=state_basis,
        config=STRIDEFitConfig(timepoint_order=("pre", "post")),
    )
    patient_result = result.patient_results[0]

    assert patient_result.fit_status == "ok"
    assert result.uncertainty is None
    assert patient_result.A is not None
    assert patient_result.d is not None
    assert patient_result.e is not None
    assert patient_result.mu_minus is not None
    assert patient_result.mu_plus is not None
    assert patient_result.A.shape == (2, 2)
    assert patient_result.d.shape == (2,)
    assert patient_result.e.shape == (2,)
    assert np.all(patient_result.A >= 0.0)
    assert np.all(patient_result.d >= 0.0)
    assert np.all(patient_result.e >= 0.0)
    assert np.sum(patient_result.A, dtype=float) >= np.trace(patient_result.A)
    np.testing.assert_allclose(
        np.sum(patient_result.A, axis=1, dtype=float) + patient_result.d,
        np.ones_like(patient_result.d),
    )
    np.testing.assert_allclose(patient_result.mu_minus, np.asarray([0.6, 0.4], dtype=float))
    np.testing.assert_allclose(patient_result.mu_plus, np.asarray([0.45, 0.55], dtype=float))
    assert patient_result.auxiliary["matched_transition_burden"].shape == (2, 2)
    assert patient_result.auxiliary["source_unmatched_burden"].shape == (2,)
    assert patient_result.auxiliary["target_unmatched_burden"].shape == (2,)


def test_fit_stride_default_bridge_no_longer_uses_legacy_mean_transport_metadata() -> None:
    state_basis = _make_three_state_basis()
    patient_result = fit_stride(
        _make_three_state_shift_observations(),
        state_basis=state_basis,
        config=STRIDEFitConfig(timepoint_order=("pre", "post")),
    ).patient_results[0]

    assert patient_result.fit_status == "ok"
    assert patient_result.diagnostics["estimator_mode"] == "observation_to_patient_bridge_v1"
    assert (
        patient_result.diagnostics["estimator_method"]
        == "domain_stratified_cartesian_observation_discrepancy"
    )
    assert "same_label_domain_means_cost_ordered_greedy_transport" not in str(patient_result.diagnostics)
    assert patient_result.audit is not None
    assert patient_result.audit.observation_fit_status == "observation_discrepancy_bridge"


def test_run_stride_fit_without_geometry_defers_geometry_aware_bridge() -> None:
    state_basis = _make_state_basis()
    observations = tuple(
        observation
        for observation in _make_grouped_observations()
        if observation.patient_id == "p1"
    )

    result = run_stride_fit(
        observations,
        state_basis=state_basis,
        geometry=None,
        config=STRIDEFitConfig(timepoint_order=("pre", "post")),
    )

    assert result.fit_status == "deferred"
    assert result.uncertainty is None
    assert result.diagnostics["mode"] == "deferred"
    patient_result = result.patient_results[0]
    assert patient_result.fit_status == "deferred"
    assert patient_result.diagnostics["defer_reason"] == "requires_shared_state_geometry"
    assert "requires shared state geometry" in str(patient_result.diagnostics["message"])


def test_fit_stride_observation_sequence_mixes_realized_and_deferred_results() -> None:
    state_basis = _make_state_basis()
    observations = _make_grouped_observations() + (
        _make_observation(
            patient_id="p2",
            timepoint="followup",
            fov_id="p2_followup_im",
            domain_label="IM",
            composition=(0.1, 0.9),
        ),
    )

    result = fit_stride(
        observations,
        state_basis=state_basis,
        config=STRIDEFitConfig(timepoint_order=("pre", "post", "followup")),
    )

    assert isinstance(result, STRIDEFitResult)
    assert result.fit_status == "deferred"
    assert result.uncertainty is None
    assert result.patient_ids == ("p1", "p2")
    assert result.recurrence.fit_status == "deferred"
    assert [patient_result.fit_status for patient_result in result.patient_results] == [
        "ok",
        "deferred",
    ]
    assert dict(result.summaries["patient_status_counts"]) == {"ok": 1, "deferred": 1}
    assert result.diagnostics["mode"] == "patient_bridge_realized_recurrence_deferred"
    assert result.patient_results[0].diagnostics["supported_case"] == "two_group_uniform_patient_bridge"
    assert result.patient_results[1].diagnostics["defer_reason"] == "requires_exactly_two_ordered_groups"
    assert "exactly two ordered groups" in str(result.patient_results[1].diagnostics["message"])
    assert dict(result.patient_inputs[0].n_observations_by_group) == {"pre": 1, "post": 2}


def test_fit_stride_dataset_path_uses_deterministic_upstream_route() -> None:
    handle = DatasetHandle.from_tables(
        pd.DataFrame(
            {
                "patient_id": ["p1", "p1", "p1", "p1"],
                "timepoint": ["pre", "pre", "post", "post"],
                "fov_id": ["f1", "f1", "f2", "f2"],
                "domain_label": ["TC", "TC", "IM", "IM"],
                "cell_subtype_label": ["A", "B", "A", "B"],
                "x": [0.0, 1.0, 0.0, 1.0],
                "y": [0.0, 0.0, 1.0, 1.0],
            }
        )
    )
    spec = BasisSpec(K=2, k_neighbors=1, random_state=0, geometry_neighbors=1)

    result = fit_stride(
        handle,
        basis_spec=spec,
        config=STRIDEFitConfig(timepoint_order=("pre", "post")),
    )

    assert result.fit_status == "deferred"
    assert result.uncertainty is None
    assert result.patient_ids == ("p1",)
    assert result.recurrence.fit_status == "deferred"
    assert result.diagnostics["mode"] == "patient_bridge_realized_recurrence_deferred"
    assert dict(result.patient_inputs[0].n_observations_by_group) == {"pre": 1, "post": 1}
    assert "state_id" in handle.adata.obs.columns
    patient_result = result.patient_results[0]
    assert patient_result.fit_status == "ok"
    np.testing.assert_allclose(patient_result.mu_minus, np.asarray([0.5, 0.5], dtype=float))
    np.testing.assert_allclose(patient_result.mu_plus, np.asarray([0.5, 0.5], dtype=float))
    assert patient_result.diagnostics["supported_case"] == "two_group_uniform_patient_bridge"
    assert patient_result.diagnostics["estimator_mode"] == "observation_to_patient_bridge_v1"
    np.testing.assert_allclose(
        np.sum(patient_result.A, axis=1, dtype=float) + patient_result.d,
        np.ones_like(patient_result.d),
    )
