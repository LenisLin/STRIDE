from __future__ import annotations

import numpy as np
import pytest
import torch

from stride.errors import ContractError
from stride.geometry import build_state_geometry
from stride.objectives import (
    FullEstimatorEvidenceBlock,
    FullEstimatorParameters,
    assemble_full_estimator_totals,
    compute_consistency_raw_from_block_losses,
    compute_full_estimator_objective,
    compute_geometry_raw,
    compute_init_fov_cost_scale,
    compute_open_raw,
    compute_recurrence_raw,
    identity_plus_small_open_initialization,
    parameters_from_unconstrained,
    post_reconstruct,
    unconstrained_from_initialization,
)


def _geometry():
    return build_state_geometry(
        cost_matrix=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        cost_scale=1.0,
        state_ids=(0, 1),
    )


def test_identity_plus_small_open_initialization_exact_values() -> None:
    init = identity_plus_small_open_initialization(3)

    assert init.delta_init == pytest.approx(0.05)
    assert init.A.dtype == torch.float64
    torch.testing.assert_close(init.A, 0.95 * torch.eye(3, dtype=torch.float64))
    torch.testing.assert_close(init.d, torch.full((3,), 0.05, dtype=torch.float64))
    torch.testing.assert_close(init.e, torch.full((3,), 0.05 / 3.0, dtype=torch.float64))

    assert identity_plus_small_open_initialization(1).delta_init == pytest.approx(0.05)
    large = identity_plus_small_open_initialization(100)
    assert large.delta_init == pytest.approx(1.0 / 101.0)


def test_parameterization_enforces_row_simplex_and_e_bounds_with_gradients() -> None:
    unconstrained = unconstrained_from_initialization(("p1", "p2"), 3)
    unconstrained.row_logits.requires_grad_(True)
    unconstrained.e_logits.requires_grad_(True)

    params = parameters_from_unconstrained(unconstrained)
    row_sums = params.A.sum(dim=2) + params.d

    torch.testing.assert_close(row_sums, torch.ones_like(row_sums))
    assert bool(((params.e >= 0.0) & (params.e <= 1.0)).all())

    loss = params.A[..., 0].sum() + params.d.sum() + params.e.sum()
    loss.backward()
    assert unconstrained.row_logits.grad is not None
    assert unconstrained.e_logits.grad is not None
    assert torch.isfinite(unconstrained.row_logits.grad).all()
    assert torch.isfinite(unconstrained.e_logits.grad).all()


def test_optimizer_initialization_uses_finite_logits_that_can_move_off_diagonal_mass() -> None:
    unconstrained = unconstrained_from_initialization(("p1",), 3)
    assert torch.isfinite(unconstrained.row_logits).all()
    assert torch.isfinite(unconstrained.e_logits).all()

    unconstrained.row_logits.requires_grad_(True)
    unconstrained.e_logits.requires_grad_(True)
    before = parameters_from_unconstrained(unconstrained).A[0, 0, 1].detach().clone()
    loss = -parameters_from_unconstrained(unconstrained).A[0, 0, 1]
    loss.backward()
    with torch.no_grad():
        unconstrained.row_logits -= 0.5 * unconstrained.row_logits.grad
    after = parameters_from_unconstrained(unconstrained).A[0, 0, 1].detach()

    assert float(before) > 0.0
    assert float(after) > float(before)


def test_parameters_from_unconstrained_rejects_nonfinite_row_logits() -> None:
    unconstrained = unconstrained_from_initialization(("p1",), 2)
    unconstrained.row_logits = unconstrained.row_logits.clone()
    unconstrained.row_logits[0, 0, 1] = float("inf")

    with pytest.raises(ContractError, match="row_logits.*finite"):
        parameters_from_unconstrained(unconstrained)


def test_post_reconstruction_normalizes_q_minus_A_plus_e() -> None:
    q_minus = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
    A = torch.tensor([[0.80, 0.10], [0.20, 0.70]], dtype=torch.float64)
    e = torch.tensor([0.05, 0.15], dtype=torch.float64)

    predicted = post_reconstruct(q_minus, A, e)
    raw = q_minus @ A + e
    expected = raw / raw.sum(dim=1, keepdim=True)

    torch.testing.assert_close(predicted, expected)


def test_post_reconstruction_rejects_zero_raw_rows() -> None:
    with pytest.raises(ContractError, match="raw_post"):
        post_reconstruct(
            torch.tensor([[1.0, 0.0]], dtype=torch.float64),
            torch.zeros((2, 2), dtype=torch.float64),
            torch.zeros((2,), dtype=torch.float64),
        )


def test_open_raw_and_scale_policy_are_fixed() -> None:
    params = FullEstimatorParameters(
        patient_ids=("p1",),
        A=torch.tensor([[[0.70, 0.10], [0.20, 0.40]]], dtype=torch.float64),
        d=torch.tensor([[0.20, 0.40]], dtype=torch.float64),
        e=torch.tensor([[0.10, 0.30]], dtype=torch.float64),
    )

    raw = compute_open_raw(params)
    totals = assemble_full_estimator_totals(
        raw_components={"obs": 0.0, "open": raw, "geometry": 0.0, "consistency": 0.0, "recurrence": 0.0},
        baseline_components={
            "obs": 0.0,
            "geometry": 0.0,
            "consistency": 0.0,
            "recurrence": 0.0,
        },
    )

    assert float(raw) == pytest.approx(0.5)
    assert float(totals.components["open"].scale) == pytest.approx(1.0)
    assert totals.components["open"].floor_used is False
    assert float(totals.components["open"].normalized) == pytest.approx(0.5)


@pytest.mark.parametrize("helper_name", ["open", "geometry", "recurrence"])
def test_exported_component_helpers_reject_invalid_A_d_e_constraints(helper_name: str) -> None:
    params = FullEstimatorParameters(
        patient_ids=("p1",),
        A=torch.tensor([[[0.70, 0.20], [0.30, 0.40]]], dtype=torch.float64),
        d=torch.tensor([[0.20, 0.40]], dtype=torch.float64),
        e=torch.tensor([[0.10, 1.20]], dtype=torch.float64),
    )

    with pytest.raises(ContractError, match="simplex|bounded"):
        if helper_name == "open":
            compute_open_raw(params)
        elif helper_name == "geometry":
            compute_geometry_raw(params, _geometry())
        else:
            compute_recurrence_raw(params)


def test_geometry_raw_uses_raw_A_and_one_over_K_denominator() -> None:
    geometry = _geometry()
    params = FullEstimatorParameters(
        patient_ids=("p1",),
        A=torch.tensor([[[0.70, 0.20], [0.30, 0.40]]], dtype=torch.float64),
        d=torch.tensor([[0.10, 0.30]], dtype=torch.float64),
        e=torch.tensor([[0.20, 0.10]], dtype=torch.float64),
    )

    raw = compute_geometry_raw(params, geometry)

    assert float(raw) == pytest.approx((0.20 + 0.30) / 2.0)


def test_baseline_normalization_uses_epsilon_floor_without_failure() -> None:
    totals = assemble_full_estimator_totals(
        raw_components={"obs": 0.2, "open": 0.1, "geometry": 0.3, "consistency": 0.0, "recurrence": 0.0},
        baseline_components={
            "obs": 0.0,
            "geometry": 0.5,
            "consistency": 0.0,
            "recurrence": 0.0,
        },
    )

    assert float(totals.components["obs"].scale) == pytest.approx(1e-2)
    assert totals.components["obs"].floor_used is True
    assert float(totals.components["geometry"].scale) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("ablation_mode", "local", "regularization", "total"),
    [
        ("geometry", (0.1 + 0.2 + 0.0) / 3.0, (0.4 + 0.5) / 2.0, 0.275),
        ("recurrence", (0.1 + 0.2 + 0.3) / 3.0, (0.4 + 0.0) / 2.0, 0.2),
        ("consistency", (0.1 + 0.2 + 0.3) / 3.0, (0.0 + 0.5) / 2.0, 0.225),
    ],
)
def test_ablation_totals_keep_fixed_denominators(
    ablation_mode: str,
    local: float,
    regularization: float,
    total: float,
) -> None:
    totals = assemble_full_estimator_totals(
        raw_components={"obs": 0.1, "open": 0.2, "geometry": 0.3, "consistency": 0.4, "recurrence": 0.5},
        baseline_components={"obs": 1.0, "geometry": 1.0, "consistency": 1.0, "recurrence": 1.0},
        ablation_mode=ablation_mode,
    )

    assert float(totals.local) == pytest.approx(local)
    assert float(totals.regularization) == pytest.approx(regularization)
    assert float(totals.total) == pytest.approx(total)


def test_consistency_uses_normalized_block_losses_and_reports_insufficient_blocks() -> None:
    raw, records = compute_consistency_raw_from_block_losses(
        patient_ids=("p1", "p2"),
        block_patient_ids=("p1", "p1", "p2"),
        normalized_block_losses=torch.tensor([1.0, 3.0, 9.0], dtype=torch.float64),
    )

    assert float(raw) == pytest.approx(0.5)
    assert records["p1"].status == "ok"
    assert records["p2"].status == "insufficient_blocks"


def test_recurrence_consensus_formula_and_support_fields() -> None:
    params = FullEstimatorParameters(
        patient_ids=("p1", "p2"),
        A=torch.tensor(
            [
                [[0.8, 0.1], [0.1, 0.8]],
                [[0.6, 0.3], [0.3, 0.6]],
            ],
            dtype=torch.float64,
        ),
        d=torch.tensor([[0.1, 0.1], [0.1, 0.1]], dtype=torch.float64),
        e=torch.tensor([[0.2, 0.0], [0.0, 0.2]], dtype=torch.float64),
    )

    result = compute_recurrence_raw(params)

    assert result.support_n_patients == 2
    assert float(result.raw) == pytest.approx(0.02)
    assert float(result.dispersion) == pytest.approx(0.02)
    torch.testing.assert_close(result.A_bar, params.A.mean(dim=0))


def test_objective_is_differentiable_on_tiny_fixture() -> None:
    unconstrained = unconstrained_from_initialization(("p1",), 2)
    unconstrained.row_logits = torch.tensor(
        [[[2.0, -0.3, -1.0], [-0.1, 1.5, -0.8]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    unconstrained.e_logits = torch.tensor([[0.1, -0.2]], dtype=torch.float64, requires_grad=True)
    params = parameters_from_unconstrained(unconstrained)
    blocks = (
        FullEstimatorEvidenceBlock(
            patient_id="p1",
            source_bag=torch.tensor([[0.75, 0.25]], dtype=torch.float64),
            target_bag=torch.tensor([[0.20, 0.80]], dtype=torch.float64),
        ),
    )

    ledger = compute_full_estimator_objective(params, blocks, _geometry())
    ledger.total.backward()

    assert torch.isfinite(ledger.total)
    assert unconstrained.row_logits.grad is not None
    assert unconstrained.e_logits.grad is not None
    assert torch.isfinite(unconstrained.row_logits.grad).all()
    assert torch.isfinite(unconstrained.e_logits.grad).all()


def test_observation_raw_uses_patient_balanced_averaging_with_unequal_blocks() -> None:
    unconstrained = unconstrained_from_initialization(("p1", "p2"), 2)
    unconstrained.row_logits = torch.tensor(
        [
            [[1.5, -0.2, -1.0], [-0.3, 1.3, -0.8]],
            [[0.8, 0.4, -1.1], [0.6, 0.2, -0.9]],
        ],
        dtype=torch.float64,
    )
    params = parameters_from_unconstrained(unconstrained)
    blocks = (
        FullEstimatorEvidenceBlock(
            patient_id="p1",
            source_bag=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
            target_bag=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        ),
        FullEstimatorEvidenceBlock(
            patient_id="p1",
            source_bag=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
            target_bag=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        ),
        FullEstimatorEvidenceBlock(
            patient_id="p2",
            source_bag=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
            target_bag=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        ),
    )

    ledger = compute_full_estimator_objective(params, blocks, _geometry())
    p1_raw = torch.stack([record.raw for record in ledger.observation_blocks if record.patient_id == "p1"]).mean()
    p2_raw = torch.stack([record.raw for record in ledger.observation_blocks if record.patient_id == "p2"]).mean()
    patient_balanced = (p1_raw + p2_raw) / 2.0
    flat = torch.stack([record.raw for record in ledger.observation_blocks]).mean()

    torch.testing.assert_close(ledger.components["obs"].raw, patient_balanced)
    assert not torch.isclose(patient_balanced, flat)


def test_core_computes_init_fov_cost_scale_and_rejects_wrong_precomputed_scale() -> None:
    block = FullEstimatorEvidenceBlock(
        patient_id="p1",
        source_bag=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
        target_bag=torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64),
        fov_cost_scale=999.0,
    )

    with pytest.raises(ContractError, match="s_G_init"):
        compute_full_estimator_objective(
            parameters_from_unconstrained(unconstrained_from_initialization(("p1",), 2)),
            (block,),
            _geometry(),
        )

    scale = compute_init_fov_cost_scale(block, _geometry(), K=2)
    assert scale.floor_used is False
    assert float(scale.value) > 0.0
    ledger = compute_full_estimator_objective(
        parameters_from_unconstrained(unconstrained_from_initialization(("p1",), 2)),
        (
            FullEstimatorEvidenceBlock(
                patient_id="p1",
                source_bag=block.source_bag,
                target_bag=block.target_bag,
                fov_cost_scale=float(scale.value),
            ),
        ),
        _geometry(),
    )
    assert ledger.observation_blocks[0].fov_cost_scale == pytest.approx(float(scale.value))


def test_init_fov_cost_scale_floor_fallback_when_all_init_costs_are_zero() -> None:
    block = FullEstimatorEvidenceBlock(
        patient_id="p1",
        source_bag=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        target_bag=post_reconstruct(
            torch.tensor([[1.0, 0.0]], dtype=torch.float64),
            identity_plus_small_open_initialization(2).A,
            identity_plus_small_open_initialization(2).e,
        ),
    )

    scale = compute_init_fov_cost_scale(block, _geometry(), K=2)

    assert float(scale.value) == pytest.approx(1.0)
    assert scale.floor_used is True


def test_objective_ledger_carries_canonical_D_obs_metadata() -> None:
    ledger = compute_full_estimator_objective(
        parameters_from_unconstrained(unconstrained_from_initialization(("p1",), 2)),
        (
            FullEstimatorEvidenceBlock(
                patient_id="p1",
                source_bag=torch.tensor([[0.75, 0.25]], dtype=torch.float64),
                target_bag=torch.tensor([[0.20, 0.80]], dtype=torch.float64),
            ),
        ),
        _geometry(),
    )

    metadata = ledger.metadata["observation_discrepancy"]
    assert metadata["operator_version"] == "D_obs^BalancedSinkhornDivergence-v1"
    assert metadata["backend"] == "torch"
    assert metadata["dtype"] == "float64"
    assert metadata["inner_epsilon_schedule"] == [0.5, 0.2, 0.1]
    assert metadata["outer_epsilon_schedule"] == [0.5, 0.2, 0.1]
    assert ledger.metadata["state_geometry"]["normalization"] == "C_norm = C_raw / s_C"
    assert ledger.metadata["state_geometry"]["s_C"] == pytest.approx(1.0)
    assert ledger.observation_blocks[0].metadata["operator_version"] == metadata["operator_version"]
