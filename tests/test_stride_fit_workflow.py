from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stride.api.basis import BasisSpec
from stride.api.dataset import DatasetHandle
from stride.api.fit import bridge_observation_matches, build_patient_relation, fit_stride, fit_stride_proxy
from stride.basis import load_state_basis
from stride.errors import ContractError
from stride.geometry import build_state_geometry
from stride.latent.operators import PatientRelationAudit
from stride.latent.recurrence import RecurrenceResult, estimate_recurrence
from stride.observation import FovObservation
from stride.objectives import LossWeights
from stride.outputs.fit_result import PatientBridgeResult, STRIDEFitResult
from stride.outputs.uncertainty import (
    CohortBootstrapUncertaintySummary,
    PatientBootstrapConfig,
    PatientBootstrapUncertaintyResult,
    STRIDEBootstrapUncertaintyResult,
)
from stride.settings import RuntimeSettings
import stride.workflows.fit_stride as fit_stride_module
from stride.workflows.fit_stride import STRIDEFitConfig, build_patient_bridge_inputs, run_stride_fit, run_stride_proxy_fit


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


def _make_chunked_bridge_observations() -> tuple[FovObservation, ...]:
    return (
        _make_observation(
            patient_id="p_chunk",
            timepoint="pre",
            fov_id="p_chunk_pre_tc_1",
            domain_label="TC",
            composition=(0.70, 0.30),
        ),
        _make_observation(
            patient_id="p_chunk",
            timepoint="pre",
            fov_id="p_chunk_pre_tc_2",
            domain_label="TC",
            composition=(0.60, 0.40),
        ),
        _make_observation(
            patient_id="p_chunk",
            timepoint="post",
            fov_id="p_chunk_post_tc_1",
            domain_label="TC",
            composition=(0.40, 0.60),
        ),
        _make_observation(
            patient_id="p_chunk",
            timepoint="post",
            fov_id="p_chunk_post_tc_2",
            domain_label="TC",
            composition=(0.35, 0.65),
        ),
        _make_observation(
            patient_id="p_chunk",
            timepoint="post",
            fov_id="p_chunk_post_tc_3",
            domain_label="TC",
            composition=(0.30, 0.70),
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
    assert "fit_stride_proxy" in message
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


def test_estimate_recurrence_realizes_one_family_template_for_two_relations() -> None:
    relation_one = build_patient_relation(
        patient_id="p1",
        A=np.asarray([[0.7, 0.1], [0.2, 0.6]], dtype=float),
        d=np.asarray([0.2, 0.2], dtype=float),
        e=np.asarray([0.1, 0.2], dtype=float),
        state_ids=(0, 1),
    )
    relation_two = build_patient_relation(
        patient_id="p2",
        A=np.asarray([[0.6, 0.2], [0.1, 0.7]], dtype=float),
        d=np.asarray([0.2, 0.2], dtype=float),
        e=np.asarray([0.2, 0.1], dtype=float),
        state_ids=(0, 1),
    )

    result = estimate_recurrence((relation_one, relation_two))

    assert result.fit_status == "ok"
    assert result.used_patient_ids == ("p1", "p2")
    assert len(result.families) == 1
    assert result.families[0].support_n_patients == 2
    assert result.families[0].member_patient_ids == ("p1", "p2")
    assert result.parameters is not None
    assert result.parameters.loadings is not None
    assert {embedding.patient_id for embedding in result.embeddings} == {"p1", "p2"}


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
    assert result.implementation_tier == "canonical_full"
    assert result.uncertainty is None
    assert result.patient_ids == ("p1",)
    assert result.recurrence.fit_status == "deferred"
    patient_result = result.patient_results[0]
    assert patient_result.fit_status == "ok"
    assert patient_result.is_canonical_full
    assert patient_result.relation is not None
    assert patient_result.objective is not None
    np.testing.assert_allclose(patient_result.mu_minus, np.asarray([0.6, 0.4], dtype=float))
    np.testing.assert_allclose(patient_result.mu_plus, np.asarray([0.45, 0.55], dtype=float))
    np.testing.assert_allclose(
        patient_result.auxiliary["model_implied_mu_plus"],
        np.sum(patient_result.A * patient_result.mu_minus[:, None], axis=0, dtype=float)
        + (patient_result.e * float(np.sum(patient_result.mu_minus, dtype=float))),
    )
    assert patient_result.objective.data_fit == patient_result.objective.observation_data_fit
    assert patient_result.objective.cohort_recurrence == pytest.approx(0.0)
    assert patient_result.objective.total >= 0.0
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
    assert patient_result.diagnostics["cohort_fit_status"] == "deferred"


def test_fit_stride_realizes_canonical_cohort_layer_when_two_patients_are_supported() -> None:
    result = fit_stride(
        _make_grouped_observations(),
        state_basis=_make_state_basis(),
        config=STRIDEFitConfig(timepoint_order=("pre", "post")),
    )

    assert result.fit_status == "ok"
    assert result.implementation_tier == "canonical_full"
    assert result.objective is not None
    assert result.recurrence.fit_status == "ok"
    assert result.recurrence.used_patient_ids == ("p1", "p2")
    assert len(result.recurrence.families) == 1
    assert len(result.recurrence.embeddings) == 2
    assert all(patient_result.is_canonical_full for patient_result in result.patient_results)
    assert all(patient_result.fit_status == "ok" for patient_result in result.patient_results)
    assert all(
        "proxy_initializer_matched_transition_burden" in patient_result.auxiliary
        for patient_result in result.patient_results
    )
    assert all(
        patient_result.objective is not None and patient_result.objective.cohort_recurrence >= 0.0
        for patient_result in result.patient_results
    )
    assert result.summaries["n_recurrence_used_patients"] == 2


def test_fit_stride_proxy_remains_explicitly_labeled_and_recurrence_deferred() -> None:
    result = fit_stride_proxy(
        _make_grouped_observations(),
        state_basis=_make_state_basis(),
        config=STRIDEFitConfig(timepoint_order=("pre", "post")),
    )

    assert result.fit_status == "deferred"
    assert result.implementation_tier == "approximate_proxy"
    assert result.recurrence.fit_status == "deferred"
    assert result.recurrence.used_patient_ids == ("p1", "p2")
    assert all(patient_result.is_proxy_path for patient_result in result.patient_results)
    assert all(patient_result.objective is None for patient_result in result.patient_results)


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
    assert patient_result.implementation_tier == "canonical_full"
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
    assert patient_result.auxiliary["local_initializer_A"].shape == (2, 2)
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


def test_relation_refinement_disabled_preserves_default_patient_arrays() -> None:
    state_basis = _make_three_state_basis()
    observations = _make_three_state_shift_observations()

    default_result = fit_stride(
        observations,
        state_basis=state_basis,
        config=STRIDEFitConfig(timepoint_order=("pre", "post")),
    )
    disabled_result = fit_stride(
        observations,
        state_basis=state_basis,
        config=STRIDEFitConfig(
            timepoint_order=("pre", "post"),
            enable_relation_refinement=False,
        ),
    )

    default_patient = default_result.patient_results[0]
    disabled_patient = disabled_result.patient_results[0]
    np.testing.assert_allclose(disabled_patient.A, default_patient.A)
    np.testing.assert_allclose(disabled_patient.d, default_patient.d)
    np.testing.assert_allclose(disabled_patient.e, default_patient.e)
    assert disabled_patient.diagnostics["relation_refinement_enabled"] is False
    assert disabled_patient.diagnostics["relation_refinement_status"] == "disabled"


def test_relation_refinement_enabled_high_geometry_weight_changes_A_and_lowers_locality_loss() -> None:
    state_basis = _make_three_state_basis()
    observations = _make_three_state_shift_observations()
    locality_weights = LossWeights(
        observation_data_fit=1.0,
        patient_consistency=0.0,
        open_relation=0.0,
        cohort_recurrence=0.0,
        geometry_structure=1_000.0,
    )

    disabled_result = fit_stride(
        observations,
        state_basis=state_basis,
        config=STRIDEFitConfig(
            timepoint_order=("pre", "post"),
            objective_weights=locality_weights,
            enable_relation_refinement=False,
        ),
    )
    refined_result = fit_stride(
        observations,
        state_basis=state_basis,
        config=STRIDEFitConfig(
            timepoint_order=("pre", "post"),
            objective_weights=locality_weights,
            enable_relation_refinement=True,
            relation_refinement_max_iter=200,
            relation_refinement_tol=1e-10,
        ),
    )

    disabled_patient = disabled_result.patient_results[0]
    refined_patient = refined_result.patient_results[0]
    assert refined_patient.diagnostics["relation_refinement_enabled"] is True
    assert refined_patient.diagnostics["relation_refinement_status"] == "ok"
    assert refined_patient.diagnostics["relation_refinement_changed_Ade"] is True
    assert not np.allclose(refined_patient.A, disabled_patient.A)
    assert refined_patient.objective.geometry_structure < disabled_patient.objective.geometry_structure
    assert refined_patient.objective.total < disabled_patient.objective.total
    np.testing.assert_allclose(
        np.sum(refined_patient.A, axis=1, dtype=float) + refined_patient.d,
        np.ones_like(refined_patient.d),
        atol=1e-8,
    )
    assert np.all(refined_patient.A >= 0.0)
    assert np.all(refined_patient.d >= 0.0)
    assert np.all(refined_patient.e >= 0.0)


def test_objective_weights_change_reporting_only_when_relation_refinement_disabled() -> None:
    state_basis = _make_three_state_basis()
    observations = _make_three_state_shift_observations()

    default_result = fit_stride(
        observations,
        state_basis=state_basis,
        config=STRIDEFitConfig(timepoint_order=("pre", "post")),
    )
    weighted_result = fit_stride(
        observations,
        state_basis=state_basis,
        config=STRIDEFitConfig(
            timepoint_order=("pre", "post"),
            objective_weights=LossWeights(
                observation_data_fit=1.0,
                patient_consistency=1.0,
                open_relation=0.25,
                cohort_recurrence=0.25,
                geometry_structure=250.0,
            ),
            enable_relation_refinement=False,
        ),
    )

    default_patient = default_result.patient_results[0]
    weighted_patient = weighted_result.patient_results[0]
    np.testing.assert_allclose(weighted_patient.A, default_patient.A)
    np.testing.assert_allclose(weighted_patient.d, default_patient.d)
    np.testing.assert_allclose(weighted_patient.e, default_patient.e)
    assert weighted_patient.objective.geometry_structure == pytest.approx(
        default_patient.objective.geometry_structure
    )
    assert weighted_patient.objective.total != pytest.approx(default_patient.objective.total)


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
    assert result.diagnostics["mode"] == "canonical_full_patient_or_cohort_deferred"
    patient_result = result.patient_results[0]
    assert patient_result.fit_status == "deferred"
    assert patient_result.implementation_tier == "canonical_full"
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
    assert result.diagnostics["mode"] == "canonical_full_patient_or_cohort_deferred"
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
    assert result.diagnostics["mode"] == "canonical_full_patient_or_cohort_deferred"
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


def test_run_stride_fit_uses_small_plan_chunks_without_changing_bridge_arrays() -> None:
    state_basis = _make_state_basis()
    geometry = _make_geometry(state_basis)
    observations = _make_chunked_bridge_observations()

    default_result = run_stride_fit(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        config=STRIDEFitConfig(timepoint_order=("pre", "post")),
    )
    chunked_result = run_stride_fit(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        config=STRIDEFitConfig(
            timepoint_order=("pre", "post"),
            runtime_settings=RuntimeSettings(
                max_calibration_workers=1,
                plan_chunk_elements=4,
            ),
        ),
    )

    default_patient = default_result.patient_results[0]
    chunked_patient = chunked_result.patient_results[0]
    assert default_patient.fit_status == "ok"
    assert chunked_patient.fit_status == "ok"
    np.testing.assert_allclose(chunked_patient.A, default_patient.A, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(chunked_patient.d, default_patient.d, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(chunked_patient.e, default_patient.e, rtol=1e-7, atol=1e-9)
    assert (
        chunked_patient.diagnostics["observation_discrepancy_config"]["plan_chunk_elements"]
        == 4
    )


def test_run_stride_fit_streams_plan_chunks_without_requesting_dense_matching_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_basis = _make_state_basis()
    geometry = _make_geometry(state_basis)
    observations = _make_chunked_bridge_observations()
    recorded_calls: list[tuple[bool, bool]] = []
    original_match = fit_stride_module.match_observation_clouds

    def _recording_match_observation_clouds(*args: object, **kwargs: object):
        recorded_calls.append(
            (
                bool(kwargs.get("return_plan", False)),
                kwargs.get("plan_consumer") is not None,
            )
        )
        return original_match(*args, **kwargs)

    monkeypatch.setattr(
        fit_stride_module,
        "match_observation_clouds",
        _recording_match_observation_clouds,
    )

    result = run_stride_fit(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        config=STRIDEFitConfig(
            timepoint_order=("pre", "post"),
            runtime_settings=RuntimeSettings(
                max_calibration_workers=1,
                plan_chunk_elements=4,
            ),
        ),
    )

    assert result.patient_results[0].fit_status == "ok"
    assert recorded_calls
    assert all(not return_plan for return_plan, _has_consumer in recorded_calls)
    assert all(has_consumer for _return_plan, has_consumer in recorded_calls)


def test_run_stride_fit_torch_backend_matches_numpy_or_falls_back_cleanly() -> None:
    state_basis = _make_state_basis()
    geometry = _make_geometry(state_basis)
    observations = _make_chunked_bridge_observations()
    torch_runtime = RuntimeSettings(
        uot_backend="torch",
        device="cuda",
        max_calibration_workers=1,
        plan_chunk_elements=4,
    )
    resolved_backend, resolved_device = torch_runtime.resolved_execution()

    numpy_result = run_stride_fit(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        config=STRIDEFitConfig(timepoint_order=("pre", "post")),
    )
    torch_result = run_stride_fit(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        config=STRIDEFitConfig(
            timepoint_order=("pre", "post"),
            runtime_settings=torch_runtime,
        ),
    )

    numpy_patient = numpy_result.patient_results[0]
    torch_patient = torch_result.patient_results[0]
    assert torch_patient.fit_status == "ok"
    np.testing.assert_allclose(torch_patient.A, numpy_patient.A, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(torch_patient.d, numpy_patient.d, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(torch_patient.e, numpy_patient.e, rtol=1e-7, atol=1e-9)
    assert (
        torch_patient.diagnostics["observation_discrepancy_config"]["requested_uot_backend"]
        == "torch"
    )
    assert (
        torch_patient.diagnostics["observation_discrepancy_config"]["uot_backend"]
        == resolved_backend
    )
    assert (
        torch_patient.diagnostics["observation_discrepancy_config"]["requested_device"]
        == "cuda"
    )
    assert (
        torch_patient.diagnostics["observation_discrepancy_config"]["device"]
        == resolved_device
    )
    assert (
        torch_patient.diagnostics["observation_discrepancy_config"]["execution_hardware"]
        == ("gpu" if str(resolved_device).startswith(("cuda", "mps")) else "cpu")
    )


def test_stride_fit_config_rejects_unknown_benchmark_mode() -> None:
    with pytest.raises(ContractError, match="benchmark_mode"):
        STRIDEFitConfig(benchmark_mode="unsupported_mode")


def test_run_stride_fit_open_channel_ablation_changes_patient_outputs_and_records_mode() -> None:
    state_basis = _make_three_state_basis()
    geometry = _make_geometry(state_basis)
    observations = _make_three_state_shift_observations()

    proxy_reference = run_stride_proxy_fit(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        config=STRIDEFitConfig(
            timepoint_order=("pre", "post"),
            benchmark_mode="reference",
        ),
    )
    proxy_ablated = run_stride_proxy_fit(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        config=STRIDEFitConfig(
            timepoint_order=("pre", "post"),
            benchmark_mode="open_channel_ablation",
        ),
    )
    reference = run_stride_fit(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        config=STRIDEFitConfig(
            timepoint_order=("pre", "post"),
            benchmark_mode="reference",
        ),
    )
    ablated = run_stride_fit(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        config=STRIDEFitConfig(
            timepoint_order=("pre", "post"),
            benchmark_mode="open_channel_ablation",
        ),
    )

    proxy_reference_patient = proxy_reference.patient_results[0]
    proxy_ablated_patient = proxy_ablated.patient_results[0]
    reference_patient = reference.patient_results[0]
    ablated_patient = ablated.patient_results[0]
    assert reference.metadata["benchmark_mode"] == "reference"
    assert ablated.metadata["benchmark_mode"] == "open_channel_ablation"
    assert ablated.diagnostics["benchmark_mode"] == "open_channel_ablation"
    assert proxy_reference.metadata["benchmark_mode"] == "reference"
    assert proxy_ablated.metadata["benchmark_mode"] == "open_channel_ablation"
    assert reference_patient.fit_status == "ok"
    assert ablated_patient.fit_status == "ok"
    assert proxy_reference_patient.fit_status == "ok"
    assert proxy_ablated_patient.fit_status == "ok"
    assert float(np.sum(reference_patient.d, dtype=float) + np.sum(reference_patient.e, dtype=float)) > 0.0
    assert proxy_ablated_patient.diagnostics["benchmark_mode"] == "open_channel_ablation"
    assert proxy_ablated_patient.diagnostics["effective_match_penalty"] > 1.0
    assert not np.allclose(proxy_ablated_patient.A, proxy_reference_patient.A)
    assert not np.allclose(proxy_ablated_patient.d, proxy_reference_patient.d)
    assert not np.allclose(proxy_ablated_patient.e, proxy_reference_patient.e)
    assert float(np.sum(proxy_ablated_patient.d, dtype=float)) < float(
        np.sum(proxy_reference_patient.d, dtype=float)
    )
    assert float(np.sum(proxy_ablated_patient.e, dtype=float)) < float(
        np.sum(proxy_reference_patient.e, dtype=float)
    )
    assert not np.allclose(
        ablated_patient.auxiliary["local_initializer_A"],
        reference_patient.auxiliary["local_initializer_A"],
    )
    assert not np.allclose(
        ablated_patient.auxiliary["local_initializer_d"],
        reference_patient.auxiliary["local_initializer_d"],
    )
    assert not np.allclose(
        ablated_patient.auxiliary["local_initializer_e"],
        reference_patient.auxiliary["local_initializer_e"],
    )
    assert float(np.sum(ablated_patient.d, dtype=float)) < float(np.sum(reference_patient.d, dtype=float))
    assert float(np.sum(ablated_patient.e, dtype=float)) < float(np.sum(reference_patient.e, dtype=float))
    assert not np.allclose(ablated_patient.A, reference_patient.A)


def test_run_stride_fit_open_channel_ablation_changes_recurrence_template() -> None:
    state_basis = _make_state_basis()
    geometry = _make_geometry(state_basis)
    observations = (
        _make_observation(
            patient_id="p1",
            timepoint="pre",
            fov_id="p1_pre_tc",
            domain_label="TC",
            composition=(0.85, 0.15),
        ),
        _make_observation(
            patient_id="p1",
            timepoint="post",
            fov_id="p1_post_im",
            domain_label="IM",
            composition=(0.20, 0.80),
        ),
        _make_observation(
            patient_id="p2",
            timepoint="pre",
            fov_id="p2_pre_tc",
            domain_label="TC",
            composition=(0.35, 0.65),
        ),
        _make_observation(
            patient_id="p2",
            timepoint="post",
            fov_id="p2_post_im",
            domain_label="IM",
            composition=(0.70, 0.30),
        ),
    )

    reference = run_stride_fit(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        config=STRIDEFitConfig(
            timepoint_order=("pre", "post"),
            benchmark_mode="reference",
        ),
    )
    ablated = run_stride_fit(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        config=STRIDEFitConfig(
            timepoint_order=("pre", "post"),
            benchmark_mode="open_channel_ablation",
        ),
    )

    assert reference.recurrence.fit_status == "ok"
    assert ablated.recurrence.fit_status == "ok"
    assert not np.allclose(
        reference.recurrence.families[0].template_A,
        ablated.recurrence.families[0].template_A,
    )
    assert not np.allclose(
        reference.recurrence.families[0].template_d,
        ablated.recurrence.families[0].template_d,
    )
    assert not np.allclose(
        reference.recurrence.families[0].template_e,
        ablated.recurrence.families[0].template_e,
    )


def test_run_stride_fit_cohort_ablation_disables_template_shrinkage_and_changes_outputs() -> None:
    state_basis = _make_state_basis()
    geometry = _make_geometry(state_basis)
    observations = (
        _make_observation(
            patient_id="p1",
            timepoint="pre",
            fov_id="p1_pre_tc",
            domain_label="TC",
            composition=(0.85, 0.15),
        ),
        _make_observation(
            patient_id="p1",
            timepoint="post",
            fov_id="p1_post_im",
            domain_label="IM",
            composition=(0.20, 0.80),
        ),
        _make_observation(
            patient_id="p2",
            timepoint="pre",
            fov_id="p2_pre_tc",
            domain_label="TC",
            composition=(0.35, 0.65),
        ),
        _make_observation(
            patient_id="p2",
            timepoint="post",
            fov_id="p2_post_im",
            domain_label="IM",
            composition=(0.70, 0.30),
        ),
    )

    reference = run_stride_fit(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        config=STRIDEFitConfig(
            timepoint_order=("pre", "post"),
            benchmark_mode="reference",
        ),
    )
    ablated = run_stride_fit(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        config=STRIDEFitConfig(
            timepoint_order=("pre", "post"),
            benchmark_mode="cohort_ablation",
        ),
    )

    reference_patient = reference.patient_results[0]
    ablated_patient = ablated.patient_results[0]
    assert reference.metadata["benchmark_mode"] == "reference"
    assert ablated.metadata["benchmark_mode"] == "cohort_ablation"
    assert ablated.diagnostics["benchmark_mode"] == "cohort_ablation"
    assert reference.recurrence.fit_status == "ok"
    assert ablated.recurrence.fit_status == "ok"
    assert reference_patient.fit_status == "ok"
    assert ablated_patient.fit_status == "ok"
    assert reference_patient.diagnostics["template_shrinkage_applied"] is True
    assert ablated_patient.diagnostics["template_shrinkage_applied"] is False
    assert ablated.metadata["effective_cohort_shrinkage_weight"] == pytest.approx(0.0)
    assert not np.allclose(ablated_patient.A, reference_patient.A)


def test_run_stride_fit_cohort_ablation_preserves_proxy_initializer_outputs() -> None:
    state_basis = _make_state_basis()
    geometry = _make_geometry(state_basis)
    observations = (
        _make_observation(
            patient_id="p1",
            timepoint="pre",
            fov_id="p1_pre_tc",
            domain_label="TC",
            composition=(0.85, 0.15),
        ),
        _make_observation(
            patient_id="p1",
            timepoint="post",
            fov_id="p1_post_im",
            domain_label="IM",
            composition=(0.20, 0.80),
        ),
        _make_observation(
            patient_id="p2",
            timepoint="pre",
            fov_id="p2_pre_tc",
            domain_label="TC",
            composition=(0.35, 0.65),
        ),
        _make_observation(
            patient_id="p2",
            timepoint="post",
            fov_id="p2_post_im",
            domain_label="IM",
            composition=(0.70, 0.30),
        ),
    )

    proxy_reference = run_stride_proxy_fit(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        config=STRIDEFitConfig(
            timepoint_order=("pre", "post"),
            benchmark_mode="reference",
        ),
    )
    proxy_ablated = run_stride_proxy_fit(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        config=STRIDEFitConfig(
            timepoint_order=("pre", "post"),
            benchmark_mode="cohort_ablation",
        ),
    )
    canonical_reference = run_stride_fit(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        config=STRIDEFitConfig(
            timepoint_order=("pre", "post"),
            benchmark_mode="reference",
        ),
    )
    canonical_ablated = run_stride_fit(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        config=STRIDEFitConfig(
            timepoint_order=("pre", "post"),
            benchmark_mode="cohort_ablation",
        ),
    )

    for proxy_reference_patient, proxy_ablated_patient in zip(
        proxy_reference.patient_results,
        proxy_ablated.patient_results,
        strict=True,
    ):
        np.testing.assert_allclose(proxy_ablated_patient.A, proxy_reference_patient.A)
        np.testing.assert_allclose(proxy_ablated_patient.d, proxy_reference_patient.d)
        np.testing.assert_allclose(proxy_ablated_patient.e, proxy_reference_patient.e)
        assert proxy_ablated_patient.diagnostics["effective_match_penalty"] == pytest.approx(1.0)

    for canonical_reference_patient, canonical_ablated_patient in zip(
        canonical_reference.patient_results,
        canonical_ablated.patient_results,
        strict=True,
    ):
        np.testing.assert_allclose(
            canonical_reference_patient.auxiliary["local_initializer_A"],
            canonical_ablated_patient.auxiliary["local_initializer_A"],
        )
        np.testing.assert_allclose(
            canonical_reference_patient.auxiliary["local_initializer_d"],
            canonical_ablated_patient.auxiliary["local_initializer_d"],
        )
        np.testing.assert_allclose(
            canonical_reference_patient.auxiliary["local_initializer_e"],
            canonical_ablated_patient.auxiliary["local_initializer_e"],
        )
