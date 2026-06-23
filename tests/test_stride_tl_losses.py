from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pytest
import torch

import stride.tl._losses as losses_module
from stride.tl._losses import (
    GEOMETRY_EFFECTIVE_WEIGHT,
    RHO_SUBBAG,
    S_COHORT,
    LossContext,
    _compute_geometry_loss,
    _compute_observation_loss,
    _compute_open_loss,
    _compute_recurrence_loss,
    _compute_subbag_consistency,
    _ObservationLoss,
    compute_total_loss,
)
from stride.tl._parameters import RelationParameters
from stride.tl._resolve import EvidenceBlock


def _parameters() -> RelationParameters:
    A = torch.tensor(
        [
            [[0.9, 0.1], [0.2, 0.8]],
            [[0.7, 0.3], [0.4, 0.6]],
        ],
        dtype=torch.float64,
    )
    d = torch.tensor([[0.05, 0.15], [0.2, 0.1]], dtype=torch.float64)
    e = torch.tensor([[0.03, 0.07], [0.11, 0.13]], dtype=torch.float64)
    return RelationParameters(patient_ids=("p1", "p2"), A=A, d=d, e=e)


def _blocks() -> tuple[EvidenceBlock, ...]:
    return (
        EvidenceBlock(
            patient_id="p1",
            source_bag=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
            target_bag=torch.tensor([[0.9, 0.1]], dtype=torch.float64),
            block_id="b0",
        ),
        EvidenceBlock(
            patient_id="p1",
            source_bag=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
            target_bag=torch.tensor([[0.2, 0.8]], dtype=torch.float64),
            block_id="b1",
        ),
        EvidenceBlock(
            patient_id="p2",
            source_bag=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
            target_bag=torch.tensor([[0.7, 0.3]], dtype=torch.float64),
            block_id="b2",
        ),
    )


@dataclass(frozen=True)
class _FakeSinkhornResult:
    value: torch.Tensor
    warnings: tuple[dict[str, object], ...] = ()


def test_open_loss_matches_contract() -> None:
    parameters = _parameters()

    result = _compute_open_loss(parameters)

    torch.testing.assert_close(result, parameters.d.mean() + parameters.e.mean())


def test_geometry_loss_matches_raw_A_contract() -> None:
    parameters = _parameters()
    cost_matrix = torch.tensor([[0.0, 3.0], [3.0, 0.0]], dtype=torch.float64)
    cost_scale = 1.5

    result = _compute_geometry_loss(parameters, cost_matrix, cost_scale)

    C_norm = cost_matrix / cost_scale
    expected = ((parameters.A * C_norm.unsqueeze(0)).sum(dim=(1, 2)) / 2.0).mean()
    torch.testing.assert_close(result, expected)


def test_recurrence_loss_is_zero_for_identical_patients() -> None:
    base = _parameters()
    parameters = RelationParameters(
        patient_ids=("p1", "p2"),
        A=base.A[0].repeat(2, 1, 1),
        d=base.d[0].repeat(2, 1),
        e=base.e[0].repeat(2, 1),
    )

    result = _compute_recurrence_loss(parameters)

    torch.testing.assert_close(result, torch.zeros((), dtype=torch.float64))


def test_recurrence_loss_positive_for_different_patients() -> None:
    result = _compute_recurrence_loss(_parameters())

    assert bool(result > 0.0)


def test_subbag_consistency_uses_within_patient_variance() -> None:
    normalized_block_losses = torch.tensor([2.0, 4.0, 8.0], dtype=torch.float64)

    result = _compute_subbag_consistency(
        patient_ids=("p1", "p2"),
        block_patient_ids=("p1", "p1", "p2"),
        normalized_block_losses=normalized_block_losses,
    )

    expected = torch.tensor(0.5, dtype=torch.float64)
    torch.testing.assert_close(result, expected)


def test_compute_total_loss_assembles_three_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = _parameters()
    context = LossContext(
        obs_scale=torch.tensor(1.0, dtype=torch.float64),
        obs_scale_floor_used=True,
        geometry_scale=torch.tensor(2.0, dtype=torch.float64),
        geometry_scale_floor_used=False,
        fov_cost_scales={"b0": 1.0, "b1": 1.0, "b2": 1.0},
        fov_cost_scale_floor_used={"b1": True},
    )
    obs = _ObservationLoss(
        raw=torch.tensor(2.0, dtype=torch.float64),
        normalized=torch.tensor(5.0, dtype=torch.float64),
        block_values=torch.tensor([2.0, 4.0, 8.0], dtype=torch.float64),
        normalized_block_values=torch.tensor([2.0, 4.0, 8.0], dtype=torch.float64),
        block_patient_ids=("p1", "p1", "p2"),
    )

    monkeypatch.setattr(losses_module, "_compute_observation_loss", lambda *a, **k: obs)
    monkeypatch.setattr(
        losses_module,
        "_compute_open_loss",
        lambda parameters: torch.tensor(7.0, dtype=torch.float64),
    )
    monkeypatch.setattr(
        losses_module,
        "_compute_geometry_loss",
        lambda parameters, cost_matrix, cost_scale: torch.tensor(11.0, dtype=torch.float64),
    )
    monkeypatch.setattr(
        losses_module,
        "_compute_recurrence_loss",
        lambda parameters: torch.tensor(13.0, dtype=torch.float64),
    )

    ledger = compute_total_loss(
        parameters,
        _blocks(),
        torch.eye(2, dtype=torch.float64),
        1.0,
        context=context,
    )

    consistency = torch.tensor(0.5, dtype=torch.float64)
    fit = obs.normalized + RHO_SUBBAG * consistency
    geometry_effective = GEOMETRY_EFFECTIVE_WEIGHT * (torch.tensor(11.0, dtype=torch.float64) / 2.0)
    prior = (torch.tensor(7.0, dtype=torch.float64) + geometry_effective) / 2.0
    cohort = torch.tensor(13.0, dtype=torch.float64) / S_COHORT
    total = (fit + prior + cohort) / 3.0
    torch.testing.assert_close(ledger.fit, fit)
    torch.testing.assert_close(ledger.prior, prior)
    torch.testing.assert_close(ledger.cohort, cohort)
    torch.testing.assert_close(ledger.total, total)
    assert ledger.components["obs_raw"] is obs.raw
    torch.testing.assert_close(ledger.components["consistency_raw"], consistency)
    assert ledger.metadata["objective_contract_version"] == losses_module.OBJECTIVE_CONTRACT_VERSION
    loss_scales = ledger.metadata["loss_scales"]
    assert loss_scales["obs_scale"] is context.obs_scale
    assert loss_scales["obs_scale_floor_used"] is True
    assert loss_scales["geometry_scale"] is context.geometry_scale
    assert loss_scales["geometry_scale_floor_used"] is False
    assert loss_scales["fov_cost_scales"] == {"b0": 1.0, "b1": 1.0, "b2": 1.0}
    assert loss_scales["fov_cost_scale_floor_used"] == {"b1": True}
    observation_discrepancy = ledger.metadata["observation_discrepancy"]
    assert observation_discrepancy["operator_version"] == "D_obs^BalancedSinkhornDivergence-v1"
    assert observation_discrepancy["backend"] == "torch"
    assert observation_discrepancy["dtype"] == "float64"
    assert observation_discrepancy["inner_epsilon_schedule"] == [0.5, 0.2, 0.1]
    assert observation_discrepancy["outer_epsilon_schedule"] == [0.5, 0.2, 0.1]
    assert ledger.metadata["state_geometry"] == {
        "normalization": "C_norm = C_raw / s_C",
        "s_C": 1.0,
    }


def test_compute_total_loss_real_sinkhorn_smoke() -> None:
    parameters = RelationParameters(
        patient_ids=("p1",),
        A=torch.tensor([[[0.85, 0.10], [0.15, 0.75]]], dtype=torch.float64),
        d=torch.tensor([[0.05, 0.10]], dtype=torch.float64),
        e=torch.tensor([[0.02, 0.03]], dtype=torch.float64),
    )
    blocks = (
        EvidenceBlock(
            patient_id="p1",
            source_bag=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
            target_bag=torch.tensor([[0.86, 0.14], [0.20, 0.80]], dtype=torch.float64),
            block_id="b0",
        ),
        EvidenceBlock(
            patient_id="p1",
            source_bag=torch.tensor([[0.75, 0.25]], dtype=torch.float64),
            target_bag=torch.tensor([[0.70, 0.30]], dtype=torch.float64),
            block_id="b1",
        ),
    )
    context = LossContext(
        obs_scale=torch.tensor(1.0, dtype=torch.float64),
        geometry_scale=torch.tensor(1.0, dtype=torch.float64),
        fov_cost_scales={"b0": 1.0, "b1": 1.0},
    )

    ledger = compute_total_loss(
        parameters,
        blocks,
        torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64),
        1.0,
        context=context,
    )

    assert bool(torch.isfinite(ledger.total))
    assert {
        "obs_raw",
        "obs_normalized",
        "open_raw",
        "geometry_raw",
        "geometry_normalized",
        "geometry_effective",
        "consistency_raw",
        "recurrence_raw",
    }.issubset(ledger.components)


def test_loss_context_mapping_bridge_defaults_scale_floor_flags() -> None:
    context = losses_module._coerce_loss_context(
        {
            "obs_scale": torch.tensor(1.0, dtype=torch.float64),
            "geometry_scale": torch.tensor(2.0, dtype=torch.float64),
            "fov_cost_scales": {"b0": 1.0},
        }
    )

    assert context.obs_scale_floor_used is False
    assert context.geometry_scale_floor_used is False


def test_loss_context_has_no_prepacked_observation_cache() -> None:
    field_names = {field.name for field in dataclasses.fields(LossContext)}
    payload_field = "observation" + "_payload"
    payload_builder = "build_packed_observation" + "_payload"

    assert payload_field not in field_names
    assert not hasattr(losses_module, payload_builder)


def test_observation_loss_calls_sinkhorn_with_hot_path_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    values = {"b0": 1.0, "b1": 3.0, "b2": 8.0}

    def fake_sinkhorn(*args: object, **kwargs: object) -> _FakeSinkhornResult:
        calls.append(dict(kwargs))
        block_id = _blocks()[len(calls) - 1].block_id
        return _FakeSinkhornResult(torch.tensor(values[block_id], dtype=torch.float64))

    monkeypatch.setattr(losses_module, "compute_sinkhorn_divergence", fake_sinkhorn)
    context = LossContext(
        obs_scale=torch.tensor(2.0, dtype=torch.float64),
        geometry_scale=torch.tensor(1.0, dtype=torch.float64),
        fov_cost_scales={"b0": 1.5, "b1": 2.5, "b2": 3.5},
        fov_cost_scale_floor_used={"b1": True},
        observed_self_ground_costs={
            "b0": torch.tensor([[0.0]], dtype=torch.float64),
            "b1": torch.tensor([[0.1]], dtype=torch.float64),
            "b2": torch.tensor([[0.2]], dtype=torch.float64),
        },
        observed_self_clipped_negative={"b1": True},
    )

    result = losses_module._compute_observation_loss_single_block_path(
        _parameters(),
        _blocks(),
        torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64),
        1.0,
        context=context,
    )

    assert [call["validate_inputs"] for call in calls] == [False, False, False]
    assert [call["collect_warnings"] for call in calls] == [False, False, False]
    assert [call["fov_cost_scale"] for call in calls] == [1.5, 2.5, 3.5]
    assert [call["fov_cost_scale_floor_used"] for call in calls] == [False, True, False]
    torch.testing.assert_close(
        calls[0]["observed_self_ground_cost"],
        torch.tensor([[0.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        calls[1]["observed_self_ground_cost"],
        torch.tensor([[0.1]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        calls[2]["observed_self_ground_cost"],
        torch.tensor([[0.2]], dtype=torch.float64),
    )
    assert [call["observed_self_clipped_negative"] for call in calls] == [
        False,
        True,
        False,
    ]
    torch.testing.assert_close(
        result.block_values,
        torch.tensor([1.0, 3.0, 8.0], dtype=torch.float64),
    )


def test_observation_loss_balances_by_patient_not_block_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = iter(
        (
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(3.0, dtype=torch.float64),
            torch.tensor(8.0, dtype=torch.float64),
        )
    )

    def fake_sinkhorn(*args: object, **kwargs: object) -> _FakeSinkhornResult:
        return _FakeSinkhornResult(next(values))

    monkeypatch.setattr(losses_module, "compute_sinkhorn_divergence", fake_sinkhorn)
    context = LossContext(
        obs_scale=torch.tensor(2.0, dtype=torch.float64),
        geometry_scale=torch.tensor(1.0, dtype=torch.float64),
        fov_cost_scales={"b0": 1.0, "b1": 1.0, "b2": 1.0},
    )

    result = losses_module._compute_observation_loss_single_block_path(
        _parameters(),
        _blocks(),
        torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64),
        1.0,
        context=context,
    )

    expected_raw = torch.tensor(((1.0 + 3.0) / 2.0 + 8.0) / 2.0, dtype=torch.float64)
    torch.testing.assert_close(result.raw, expected_raw)
    torch.testing.assert_close(result.normalized, expected_raw / 2.0)
    torch.testing.assert_close(
        result.normalized_block_values,
        torch.tensor([0.5, 1.5, 4.0], dtype=torch.float64),
    )


def test_observation_loss_batch_path_matches_single_block_path_for_1x1_and_1x2() -> None:
    parameters = RelationParameters(
        patient_ids=("p1", "p2"),
        A=torch.tensor(
            [
                [[0.82, 0.10], [0.12, 0.78]],
                [[0.72, 0.18], [0.22, 0.68]],
            ],
            dtype=torch.float64,
        ),
        d=torch.tensor([[0.08, 0.10], [0.10, 0.10]], dtype=torch.float64),
        e=torch.tensor([[0.01, 0.04], [0.03, 0.02]], dtype=torch.float64),
    )
    blocks = (
        EvidenceBlock(
            patient_id="p1",
            source_bag=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
            target_bag=torch.tensor([[0.84, 0.16]], dtype=torch.float64),
            block_id="p1_1x1",
        ),
        EvidenceBlock(
            patient_id="p1",
            source_bag=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
            target_bag=torch.tensor(
                [[0.24, 0.76], [0.32, 0.68]],
                dtype=torch.float64,
            ),
            block_id="p1_1x2",
        ),
        EvidenceBlock(
            patient_id="p2",
            source_bag=torch.tensor([[0.70, 0.30]], dtype=torch.float64),
            target_bag=torch.tensor([[0.62, 0.38]], dtype=torch.float64),
            block_id="p2_1x1",
        ),
        EvidenceBlock(
            patient_id="p2",
            source_bag=torch.tensor([[0.20, 0.80]], dtype=torch.float64),
            target_bag=torch.tensor(
                [[0.35, 0.65], [0.58, 0.42]],
                dtype=torch.float64,
            ),
            block_id="p2_1x2",
        ),
    )
    context = LossContext(
        obs_scale=torch.tensor(1.7, dtype=torch.float64),
        geometry_scale=torch.tensor(1.0, dtype=torch.float64),
        fov_cost_scales={
            "p1_1x1": 1.0,
            "p1_1x2": 1.2,
            "p2_1x1": 0.9,
            "p2_1x2": 1.4,
        },
        observed_self_ground_costs={
            "p1_1x1": torch.zeros((1, 1), dtype=torch.float64),
            "p1_1x2": torch.tensor([[0.0, 0.2], [0.2, 0.0]], dtype=torch.float64),
            "p2_1x1": torch.zeros((1, 1), dtype=torch.float64),
            "p2_1x2": torch.tensor([[0.0, 0.3], [0.3, 0.0]], dtype=torch.float64),
        },
    )
    cost_matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)

    batched = _compute_observation_loss(
        parameters,
        blocks,
        cost_matrix,
        1.0,
        context=context,
    )
    single = losses_module._compute_observation_loss_single_block_path(
        parameters,
        blocks,
        cost_matrix,
        1.0,
        context=context,
    )

    torch.testing.assert_close(batched.raw, single.raw, atol=1e-8, rtol=0.0)
    torch.testing.assert_close(batched.normalized, single.normalized, atol=1e-8, rtol=0.0)
    torch.testing.assert_close(
        batched.block_values,
        single.block_values,
        atol=1e-8,
        rtol=0.0,
    )
    assert batched.block_patient_ids == single.block_patient_ids
