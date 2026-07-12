from __future__ import annotations

import numpy as np
import pytest
import torch

from stride.errors import ContractError
from stride.tl._losses import LossLedger
from stride.tl._output import (
    CohortResult,
    FitResult,
    RelationResult,
    _validate_relation_result,
    assemble_fit_result,
    assemble_relation_result,
)
from stride.tl._parameters import RelationParameters
from stride.tl._resolve import EvidenceBlock, RelationInput
from stride.tl._train import TrainingResult, TrainingRunInfo


def _relation() -> RelationInput:
    return RelationInput(
        relation_id="pre_TC__on_IM",
        source_timepoint="pre",
        target_timepoint="on",
        source_domain="TC",
        target_domain="IM",
        patient_ids=("p1", "p2"),
        support_counts={
            "p1": {"source": 2, "target": 1},
            "p2": {"source": 1, "target": 2},
            "p3": {"source": 1, "target": 0},
        },
        skipped_patient_ids=("p3",),
        blocks=(
            EvidenceBlock(
                patient_id="p1",
                source_bag=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
                target_bag=torch.tensor([[0.9, 0.1]], dtype=torch.float64),
                block_id="p1:subbag_0",
                metadata={"block_construction_policy": "partitioned_fov_subbag_v1"},
            ),
            EvidenceBlock(
                patient_id="p2",
                source_bag=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
                target_bag=torch.tensor([[0.2, 0.8]], dtype=torch.float64),
                block_id="p2:subbag_0",
                metadata={"block_construction_policy": "partitioned_fov_subbag_v1"},
            ),
        ),
        metadata={
            "block_construction_policy": "partitioned_fov_subbag_v1",
            "warning": {"code": "skipped_support", "patient_ids": ("p3",)},
            "operator_metadata": {"operator_version": "custom-ignored"},
        },
    )


def _training_result() -> TrainingResult:
    A = torch.tensor(
        [
            [[0.70, 0.20], [0.10, 0.80]],
            [[0.60, 0.25], [0.30, 0.55]],
        ],
        dtype=torch.float64,
    )
    d = torch.tensor([[0.10, 0.10], [0.15, 0.15]], dtype=torch.float64)
    e = torch.tensor([[0.05, 0.20], [0.15, 0.25]], dtype=torch.float64)
    total = torch.tensor(3.5, dtype=torch.float64)
    return TrainingResult(
        parameters=RelationParameters(patient_ids=("p1", "p2"), A=A, d=d, e=e),
        loss_ledger=LossLedger(
            total=total,
            fit=torch.tensor(1.0, dtype=torch.float64),
            prior=torch.tensor(2.0, dtype=torch.float64),
            cohort=torch.tensor(7.0, dtype=torch.float64),
            components={
                "obs_raw": torch.tensor(0.25, dtype=torch.float64),
                "obs_normalized": torch.tensor(0.5, dtype=torch.float64),
                "open_raw": torch.tensor(0.3, dtype=torch.float64),
                "geometry_raw": torch.tensor(0.4, dtype=torch.float64),
                "geometry_normalized": torch.tensor(0.8, dtype=torch.float64),
                "geometry_effective": torch.tensor(0.008, dtype=torch.float64),
                "consistency_raw": torch.tensor(0.0, dtype=torch.float64),
                "recurrence_raw": torch.tensor(0.04, dtype=torch.float64),
            },
            metadata={
                "objective_contract_version": "stride_full_estimator_three_block_v1",
                "objective_constants": {
                    "rho_subbag": 1.0,
                    "geometry_effective_weight": 0.01,
                    "s_cohort": 0.01,
                    "epsilon_norm": 0.01,
                },
                "loss_scales": {
                    "obs_scale": torch.tensor(2.0, dtype=torch.float64),
                    "obs_scale_floor_used": False,
                    "geometry_scale": torch.tensor(0.5, dtype=torch.float64),
                    "geometry_scale_floor_used": False,
                },
                "observation_discrepancy": {
                    "operator_version": "D_obs^BalancedSinkhornDivergence-v1",
                    "backend": "torch",
                    "dtype": "float64",
                    "inner_epsilon_schedule": [0.5, 0.2, 0.1],
                    "outer_epsilon_schedule": [0.5, 0.2, 0.1],
                    "max_iter": 100,
                    "tol": 1e-6,
                    "warning_tol": 1e-4,
                },
                "state_geometry": {
                    "normalization": "C_norm = C_raw / s_C",
                    "s_C": 1.5,
                },
            },
            warnings=({"code": "loss_warning"},),
        ),
        run_info=TrainingRunInfo(
            reason="max_steps",
            optimizer_exit_flag="max_steps_exhausted_finite",
            n_steps=3,
            warmup_steps_completed=1,
            main_steps_completed=2,
            initial_total=4.0,
            final_total=3.5,
            absolute_improvement=0.5,
            relative_improvement=0.125,
            random_seed=11,
        ),
        trace={"steps": ({"stage": "main", "step": 1, "total": 3.5},)},
    )


def test_assemble_relation_result_returns_numpy_arrays_aligned_to_patients() -> None:
    result = assemble_relation_result(_relation(), _training_result())

    assert result.relation_id == "pre_TC__on_IM"
    assert result.patient_ids == ("p1", "p2")
    assert isinstance(result.A, np.ndarray)
    assert isinstance(result.d, np.ndarray)
    assert isinstance(result.e, np.ndarray)
    assert result.A.shape == (2, 2, 2)
    assert result.d.shape == (2, 2)
    assert result.e.shape == (2, 2)
    np.testing.assert_allclose(result.A[0], [[0.70, 0.20], [0.10, 0.80]])
    assert result.support["support_counts"]["p1"] == {"source": 2, "target": 1}
    assert result.support["skipped_patient_ids"] == ("p3",)
    assert result.support["n_evidence_blocks"] == 2


def test_assemble_relation_result_scalarizes_loss_summary() -> None:
    result = assemble_relation_result(_relation(), _training_result())

    assert result.loss["total"] == 3.5
    assert result.loss["fit"] == 1.0
    assert result.loss["prior"] == 2.0
    assert result.loss["cohort"] == 7.0
    assert result.loss["components"]["obs_raw"] == 0.25
    assert result.loss["components"]["recurrence_raw"] == 0.04


def test_assemble_relation_result_provenance_contains_compact_contract_facts() -> None:
    result = assemble_relation_result(_relation(), _training_result())
    provenance = result.provenance

    assert set(provenance) == {
        "provenance_schema_version",
        "objective_contract_version",
        "random_seed",
        "objective_constants",
        "objective_scale_initialization",
        "optimizer_start_initialization",
        "loss",
        "e_bounds",
        "post_reconstruction_form",
        "observation_comparison_plan",
        "observation_discrepancy",
        "state_geometry",
        "optimizer",
        "recurrence",
        "detailed_optimizer_trace",
    }
    assert provenance["provenance_schema_version"] == "stride_fit_provenance.v1"
    assert provenance["objective_contract_version"] == "stride_full_estimator_three_block_v1"
    assert provenance["random_seed"] == 11
    assert provenance["objective_scale_initialization"]["policy"] == ("identity_plus_small_open")
    assert provenance["optimizer_start_initialization"]["policy"] == (
        "offdiag_seeded_identity_plus_small_open"
    )
    assert provenance["optimizer_start_initialization"]["offdiag_init_mass"] == 0.01
    assert provenance["loss"]["components"]["obs"] == {
        "raw": 0.25,
        "scale": 2.0,
        "normalized": 0.5,
        "floor_used": False,
    }
    assert provenance["loss"]["components"]["open"] == {
        "raw": 0.3,
        "normalized": 0.3,
    }
    assert provenance["loss"]["components"]["geometry"] == {
        "raw": 0.4,
        "scale": 0.5,
        "normalized": 0.8,
        "effective": 0.008,
        "floor_used": False,
    }
    assert provenance["loss"]["components"]["subbag_consistency"] == {
        "raw": 0.0,
        "effective": 0.0,
        "status": "insufficient_blocks",
    }
    assert provenance["loss"]["components"]["recurrence"] == {
        "raw": 0.04,
        "cohort_scaled": 7.0,
    }
    assert provenance["e_bounds"] == [0.0, 1.0]
    assert provenance["post_reconstruction_form"] == "normalize(q_minus @ A + e)"
    assert provenance["observation_discrepancy"]["operator_version"] == (
        "D_obs^BalancedSinkhornDivergence-v1"
    )
    optimizer = provenance["optimizer"]
    assert optimizer["exit_flag"] == "max_steps_exhausted_finite"
    assert optimizer["framework"] == "torch"
    assert optimizer["algorithm"] == "AdamW"
    assert optimizer["reason"] == "max_steps"
    assert optimizer["n_steps"] == 3
    assert optimizer["initial_total"] == 4.0
    assert optimizer["final_total"] == 3.5
    assert optimizer["absolute_improvement"] == 0.5
    assert optimizer["relative_improvement"] == 0.125
    assert optimizer["warmup"]["steps_completed"] == 1
    assert optimizer["main"]["steps_completed"] == 2
    assert "max_steps" in optimizer["main"]
    assert provenance["detailed_optimizer_trace"] is True
    assert provenance["state_geometry"]["s_C"] == 1.5
    assert provenance["observation_comparison_plan"]["block_construction_policy"] == (
        "partitioned_fov_subbag_v1"
    )
    assert provenance["observation_comparison_plan"]["domain_policy"] == ("observation_layer_only")
    assert provenance["observation_comparison_plan"]["n_blocks_by_patient"] == {
        "p1": 1,
        "p2": 1,
    }
    assert provenance["recurrence"] == {"support_n_patients": 2, "dispersion": 0.04}


def test_assemble_relation_result_builds_cohort_template_from_patient_mean() -> None:
    result = assemble_relation_result(_relation(), _training_result())

    assert isinstance(result.cohort, CohortResult)
    np.testing.assert_allclose(result.cohort.template_A, result.A.mean(axis=0))
    np.testing.assert_allclose(result.cohort.template_d, result.d.mean(axis=0))
    np.testing.assert_allclose(result.cohort.template_e, result.e.mean(axis=0))
    assert result.cohort.support_n_patients == 2
    assert result.cohort.dispersion == 0.04
    assert result.cohort.metadata["summary"] == "recurrence_regularized_mean_consensus"
    assert result.cohort.metadata["template_source"] == "mean_of_fitted_patient_relations"


def test_recurrence_ablation_marks_cohort_as_unregularized_diagnostic() -> None:
    fit = _training_result()
    metadata = dict(fit.loss_ledger.metadata)
    metadata["objective_policy"] = {
        "name": "recurrence_ablation",
        "consistency_weight": 1.0,
        "geometry_weight": 1.0,
        "recurrence_weight": 0.0,
        "fixed_block_denominators": True,
    }
    fit = TrainingResult(
        parameters=fit.parameters,
        loss_ledger=LossLedger(
            total=fit.loss_ledger.total,
            fit=fit.loss_ledger.fit,
            prior=fit.loss_ledger.prior,
            cohort=torch.tensor(0.0, dtype=torch.float64),
            components=fit.loss_ledger.components,
            metadata=metadata,
        ),
        run_info=fit.run_info,
    )

    result = assemble_relation_result(_relation(), fit)

    assert result.cohort.metadata == {
        "summary": "unregularized_cohort_diagnostic",
        "template_source": "mean_of_fitted_patient_relations",
        "dispersion_source": "loss.components.recurrence_raw",
        "recurrence_regularized": False,
    }


def test_assemble_relation_result_rejects_patient_axis_mismatch() -> None:
    fit = _training_result()
    bad_parameters = RelationParameters(
        patient_ids=("p2", "p1"),
        A=fit.parameters.A,
        d=fit.parameters.d,
        e=fit.parameters.e,
    )
    bad_fit = TrainingResult(
        parameters=bad_parameters,
        loss_ledger=fit.loss_ledger,
        run_info=fit.run_info,
    )

    with pytest.raises(ContractError, match="patient_ids"):
        assemble_relation_result(_relation(), bad_fit)


def test_assemble_relation_result_rejects_missing_recurrence_raw() -> None:
    fit = _training_result()
    components = dict(fit.loss_ledger.components)
    components.pop("recurrence_raw")
    bad_fit = TrainingResult(
        parameters=fit.parameters,
        loss_ledger=LossLedger(
            total=fit.loss_ledger.total,
            fit=fit.loss_ledger.fit,
            prior=fit.loss_ledger.prior,
            cohort=fit.loss_ledger.cohort,
            components=components,
            metadata=fit.loss_ledger.metadata,
            warnings=fit.loss_ledger.warnings,
        ),
        run_info=fit.run_info,
    )

    with pytest.raises(ContractError, match="recurrence_raw"):
        assemble_relation_result(_relation(), bad_fit)


def test_validate_relation_result_rejects_invalid_row_simplex() -> None:
    result = assemble_relation_result(_relation(), _training_result())
    invalid = RelationResult(
        relation_id=result.relation_id,
        patient_ids=result.patient_ids,
        A=result.A.copy(),
        d=result.d.copy(),
        e=result.e.copy(),
        support=result.support,
    )
    invalid.A[0, 0, 0] = 0.9

    with pytest.raises(ContractError, match="row simplex"):
        _validate_relation_result(invalid)


def test_validate_relation_result_rejects_invalid_cohort_template() -> None:
    result = assemble_relation_result(_relation(), _training_result())
    bad_cohort = CohortResult(
        relation_id=result.relation_id,
        patient_ids=result.patient_ids,
        template_A=result.cohort.template_A.copy(),
        template_d=result.cohort.template_d.copy(),
        template_e=result.cohort.template_e.copy(),
        support_n_patients=result.cohort.support_n_patients,
        dispersion=float("nan"),
    )
    invalid = RelationResult(
        relation_id=result.relation_id,
        patient_ids=result.patient_ids,
        A=result.A,
        d=result.d,
        e=result.e,
        support=result.support,
        cohort=bad_cohort,
    )

    with pytest.raises(ContractError, match="dispersion"):
        _validate_relation_result(invalid)

    bad_template = CohortResult(
        relation_id=result.relation_id,
        patient_ids=result.patient_ids,
        template_A=result.cohort.template_A.copy(),
        template_d=result.cohort.template_d.copy(),
        template_e=result.cohort.template_e.copy(),
        support_n_patients=result.cohort.support_n_patients,
        dispersion=result.cohort.dispersion,
    )
    bad_template.template_A[0, 0] = 0.95
    invalid_template = RelationResult(
        relation_id=result.relation_id,
        patient_ids=result.patient_ids,
        A=result.A,
        d=result.d,
        e=result.e,
        support=result.support,
        cohort=bad_template,
    )

    with pytest.raises(ContractError, match="row simplex"):
        _validate_relation_result(invalid_template)


def test_assemble_fit_result_preserves_order_and_rejects_duplicates() -> None:
    first = assemble_relation_result(_relation(), _training_result())
    second = RelationResult(
        relation_id="pre_PT__on_IM",
        patient_ids=first.patient_ids,
        A=first.A,
        d=first.d,
        e=first.e,
        support=first.support,
    )

    fit = assemble_fit_result(
        relations=(first, second),
        warnings=({"code": "fit_warning"},),
        source="pre",
        target="on",
        n_states=2,
    )

    assert isinstance(fit, FitResult)
    assert fit.relation_ids == ("pre_TC__on_IM", "pre_PT__on_IM")
    assert tuple(fit.relations) == fit.relation_ids
    assert fit.provenance["n_relations"] == 2
    assert fit.provenance["n_states"] == 2
    assert fit.warnings == ({"code": "fit_warning"},)

    with pytest.raises(ContractError, match="duplicate"):
        assemble_fit_result(
            relations=(first, first),
            warnings=(),
            source="pre",
            target="on",
            n_states=2,
        )


def test_assemble_fit_result_rejects_n_states_mismatch() -> None:
    result = assemble_relation_result(_relation(), _training_result())

    with pytest.raises(ContractError, match="n_states"):
        assemble_fit_result(
            relations=(result,),
            warnings=(),
            source="pre",
            target="on",
            n_states=3,
        )
