from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.optimize import minimize_scalar

from stride.errors import ContractError
from stride.geometry import StateGeometry, build_state_geometry
from stride.observation import (
    BalancedSinkhornDivergenceConfig,
    compute_balanced_sinkhorn_observation_discrepancy,
)
from stride.observation.balanced_sinkhorn import _apply_small_negative_rule, _sinkhorn_transport_cost
from stride.observation.balanced_sinkhorn import (
    _pairwise_composition_ground_cost_batched,
    _pairwise_composition_ground_cost_loop,
)


def _geometry() -> StateGeometry:
    return build_state_geometry(
        cost_matrix=np.asarray(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 1.5],
                [2.0, 1.5, 0.0],
            ],
            dtype=float,
        ),
        cost_scale=1.5,
        state_ids=(0, 1, 2),
    )


def _two_state_geometry() -> StateGeometry:
    return build_state_geometry(
        cost_matrix=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        cost_scale=1.0,
        state_ids=(0, 1),
    )


def _invalid_geometry(*, cost_matrix: np.ndarray | None = None, cost_scale: float = 1.0) -> StateGeometry:
    matrix = (
        np.asarray(cost_matrix, dtype=float)
        if cost_matrix is not None
        else np.asarray(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 1.5],
                [2.0, 1.5, 0.0],
            ],
            dtype=float,
        )
    )
    return StateGeometry(
        cost_matrix=matrix,
        cost_scale=cost_scale,
        adjacency_matrix=np.eye(matrix.shape[0], dtype=float),
        similarity_graph=np.eye(matrix.shape[0], dtype=float),
        state_ids=tuple(range(matrix.shape[0])),
    )


def _entropy_objective(plan: np.ndarray, cost: np.ndarray, epsilon: float) -> float:
    positive = plan > 0.0
    entropy = float(np.sum(plan[positive] * (np.log(plan[positive]) - 1.0)))
    return float(np.sum(plan * cost) + epsilon * entropy)


def _bruteforce_2state_entropic_ot(
    left: np.ndarray,
    right: np.ndarray,
    cost: np.ndarray,
    epsilon: float,
) -> float:
    lower = max(0.0, float(left[0] + right[0] - 1.0))
    upper = min(float(left[0]), float(right[0]))

    def objective(t: float) -> float:
        plan = np.asarray(
            [
                [t, float(left[0] - t)],
                [float(right[0] - t), float(left[1] - right[0] + t)],
            ],
            dtype=float,
        )
        return _entropy_objective(plan, cost, epsilon)

    result = minimize_scalar(objective, bounds=(lower, upper), method="bounded")
    assert result.success
    return float(result.fun)


def _bruteforce_2state_sinkhorn_divergence(
    left: np.ndarray,
    right: np.ndarray,
    cost: np.ndarray,
    epsilon: float,
) -> float:
    return (
        _bruteforce_2state_entropic_ot(left, right, cost, epsilon)
        - 0.5 * _bruteforce_2state_entropic_ot(left, left, cost, epsilon)
        - 0.5 * _bruteforce_2state_entropic_ot(right, right, cost, epsilon)
    )


def test_balanced_sinkhorn_divergence_is_nonnegative_symmetric_and_self_zero() -> None:
    geometry = _geometry()
    predicted = torch.tensor(
        [[0.70, 0.20, 0.10], [0.10, 0.80, 0.10]],
        dtype=torch.float64,
    )
    observed = torch.tensor(
        [[0.20, 0.20, 0.60], [0.60, 0.30, 0.10]],
        dtype=torch.float64,
    )

    xy = compute_balanced_sinkhorn_observation_discrepancy(
        predicted,
        observed,
        geometry,
        fov_cost_scale=1.25,
    )
    yx = compute_balanced_sinkhorn_observation_discrepancy(
        observed,
        predicted,
        geometry,
        fov_cost_scale=1.25,
    )
    xx = compute_balanced_sinkhorn_observation_discrepancy(
        predicted,
        predicted,
        geometry,
        fov_cost_scale=1.25,
    )

    assert xy.status in {"ok", "ok_with_warnings"}
    assert xy.value.dtype == torch.float64
    assert xy.value.item() >= 0.0
    torch.testing.assert_close(xy.value, yx.value, rtol=1e-6, atol=1e-8)
    assert abs(float(xx.value.detach().cpu())) <= 1e-8
    assert xy.clipped_negative is False


def test_balanced_sinkhorn_matches_independent_2state_entropic_reference() -> None:
    geometry = _two_state_geometry()
    predicted = torch.tensor([[0.72, 0.28]], dtype=torch.float64)
    observed = torch.tensor([[0.21, 0.79]], dtype=torch.float64)

    result = compute_balanced_sinkhorn_observation_discrepancy(
        predicted,
        observed,
        geometry,
        fov_cost_scale=1.0,
    )

    reference = _bruteforce_2state_sinkhorn_divergence(
        np.asarray([0.72, 0.28], dtype=float),
        np.asarray([0.21, 0.79], dtype=float),
        np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        epsilon=0.1,
    )
    assert float(result.value.detach().cpu()) == pytest.approx(reference, abs=5e-5)


def test_balanced_sinkhorn_gradient_backpropagates_through_predicted_distribution() -> None:
    geometry = _geometry()
    raw_prediction = torch.tensor(
        [[1.4, 0.2, -0.3], [-0.7, 1.1, 0.5]],
        dtype=torch.float64,
        requires_grad=True,
    )
    predicted = torch.softmax(raw_prediction, dim=1)
    observed = torch.tensor(
        [[0.15, 0.25, 0.60], [0.50, 0.35, 0.15]],
        dtype=torch.float64,
    )

    result = compute_balanced_sinkhorn_observation_discrepancy(
        predicted,
        observed,
        geometry,
        fov_cost_scale=1.25,
    )

    assert result.value.requires_grad
    result.value.backward()
    assert raw_prediction.grad is not None
    assert torch.isfinite(raw_prediction.grad).all()
    assert float(torch.linalg.vector_norm(raw_prediction.grad).detach().cpu()) > 0.0


def test_batched_pairwise_ground_cost_matches_loop_value_and_gradient() -> None:
    config = BalancedSinkhornDivergenceConfig()
    C_norm = torch.tensor(
        [
            [0.0, 0.5, 1.0],
            [0.5, 0.0, 0.75],
            [1.0, 0.75, 0.0],
        ],
        dtype=torch.float64,
    )
    left_base = torch.tensor(
        [[0.70, 0.20, 0.10], [0.15, 0.65, 0.20]],
        dtype=torch.float64,
    )
    right = torch.tensor(
        [[0.20, 0.30, 0.50], [0.50, 0.25, 0.25], [0.10, 0.80, 0.10]],
        dtype=torch.float64,
    )

    left_loop = left_base.clone().detach().requires_grad_(True)
    loop = _pairwise_composition_ground_cost_loop(
        left_loop,
        right,
        C_norm,
        config=config,
        label="equivalence.loop",
    )
    loop.value.sum().backward()

    left_batched = left_base.clone().detach().requires_grad_(True)
    batched = _pairwise_composition_ground_cost_batched(
        left_batched,
        right,
        C_norm,
        config=config,
        label="equivalence.batched",
    )
    batched.value.sum().backward()

    torch.testing.assert_close(batched.value, loop.value, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(left_batched.grad, left_loop.grad, rtol=1e-7, atol=1e-9)
    assert len(batched.final_updates) == len(loop.final_updates)
    assert len(batched.iterations) == len(loop.iterations)
    assert len(batched.max_iter_reached) == len(loop.max_iter_reached)


def test_batched_pairwise_ground_cost_handles_exact_zero_support() -> None:
    config = BalancedSinkhornDivergenceConfig()
    C_norm = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    left = torch.tensor([[1.0, 0.0], [0.25, 0.75]], dtype=torch.float64, requires_grad=True)
    right = torch.tensor([[0.0, 1.0], [0.80, 0.20]], dtype=torch.float64)

    batched = _pairwise_composition_ground_cost_batched(
        left,
        right,
        C_norm,
        config=config,
        label="zero_support.batched",
    )
    assert torch.isfinite(batched.value).all()
    batched.value.sum().backward()
    assert left.grad is not None
    assert torch.isfinite(left.grad).all()


def test_balanced_sinkhorn_backward_is_finite_with_exact_zero_support() -> None:
    geometry = _two_state_geometry()
    predicted = torch.tensor(
        [[1.0, 0.0], [0.45, 0.55]],
        dtype=torch.float64,
        requires_grad=True,
    )
    observed = torch.tensor(
        [[0.10, 0.90], [0.65, 0.35]],
        dtype=torch.float64,
    )

    result = compute_balanced_sinkhorn_observation_discrepancy(
        predicted,
        observed,
        geometry,
        fov_cost_scale=1.0,
    )
    result.value.backward()

    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_sinkhorn_transport_uses_bounded_envelope_autograd_graph() -> None:
    def count_autograd_nodes(value: torch.Tensor) -> int:
        seen: set[object] = set()
        stack = [value.grad_fn]
        while stack:
            node = stack.pop()
            if node is None or node in seen:
                continue
            seen.add(node)
            stack.extend(next_fn for next_fn, _index in node.next_functions)
        return len(seen)

    raw_left = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64, requires_grad=True)
    left = torch.softmax(raw_left, dim=0)
    right = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    cost = torch.tensor(
        [
            [0.0, 0.5, 1.0],
            [0.5, 0.0, 0.75],
            [1.0, 0.75, 0.0],
        ],
        dtype=torch.float64,
    )

    result = _sinkhorn_transport_cost(
        left,
        right,
        cost,
        epsilon_schedule=BalancedSinkhornDivergenceConfig().inner_epsilon_schedule,
        config=BalancedSinkhornDivergenceConfig(),
        label="bounded_graph_regression",
    )

    assert result.value.requires_grad
    assert count_autograd_nodes(result.value) < 40
    result.value.backward()
    assert raw_left.grad is not None
    assert torch.isfinite(raw_left.grad).all()
    assert float(torch.linalg.vector_norm(raw_left.grad).detach().cpu()) > 0.0


@pytest.mark.parametrize(
    ("predicted", "observed", "message"),
    [
        (
            torch.tensor([[0.8, 0.3, -0.1]], dtype=torch.float64),
            torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64),
            "nonnegative",
        ),
        (
            torch.tensor([[float("nan"), 0.5, 0.5]], dtype=torch.float64),
            torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64),
            "finite",
        ),
        (
            torch.tensor([[0.2, 0.2, 0.2]], dtype=torch.float64),
            torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64),
            "sum to 1.0",
        ),
        (
            torch.tensor([[0.5, 0.5]], dtype=torch.float64),
            torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64),
            "K-state axis",
        ),
        (
            torch.empty((0, 3), dtype=torch.float64),
            torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64),
            "non-empty",
        ),
    ],
)
def test_balanced_sinkhorn_rejects_invalid_distribution_inputs(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        compute_balanced_sinkhorn_observation_discrepancy(
            predicted,
            observed,
            _geometry(),
            fov_cost_scale=1.0,
        )


@pytest.mark.parametrize(
    ("geometry", "message"),
    [
        (_invalid_geometry(cost_scale=0.0), "cost_scale"),
        (
            _invalid_geometry(
                cost_matrix=np.asarray(
                    [
                        [0.0, 1.0, 2.0],
                        [1.2, 0.0, 1.5],
                        [2.0, 1.5, 0.0],
                    ],
                    dtype=float,
                ),
            ),
            "symmetric",
        ),
        (
            _invalid_geometry(cost_matrix=np.zeros((3, 3), dtype=float)),
            "positive off-diagonal",
        ),
    ],
)
def test_balanced_sinkhorn_rejects_invalid_geometry(
    geometry: StateGeometry,
    message: str,
) -> None:
    predicted = torch.tensor([[0.70, 0.20, 0.10]], dtype=torch.float64)
    observed = torch.tensor([[0.20, 0.30, 0.50]], dtype=torch.float64)

    with pytest.raises(ContractError, match=message):
        compute_balanced_sinkhorn_observation_discrepancy(
            predicted,
            observed,
            geometry,
            fov_cost_scale=1.0,
        )


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"inner_epsilon_schedule": (0.5, 0.2, 0.05)},
        {"outer_epsilon_schedule": (0.5, 0.2, 0.05)},
        {"max_iter": 999},
        {"max_iter": 1000.9},
        {"tol": 1e-5},
        {"warning_tol": 1e-3},
        {"backend": "numpy"},
        {"dtype": "float32"},
    ],
)
def test_balanced_sinkhorn_rejects_noncanonical_config_values(
    config_kwargs: dict[str, object],
) -> None:
    predicted = torch.tensor([[0.70, 0.20, 0.10]], dtype=torch.float64)
    observed = torch.tensor([[0.20, 0.30, 0.50]], dtype=torch.float64)

    with pytest.raises(ContractError, match="canonical"):
        config = BalancedSinkhornDivergenceConfig(**config_kwargs)
        compute_balanced_sinkhorn_observation_discrepancy(
            predicted,
            observed,
            _geometry(),
            fov_cost_scale=1.0,
            config=config,
        )


def test_balanced_sinkhorn_rejects_duck_config_that_bypasses_config_constructor() -> None:
    class DuckConfig:
        inner_epsilon_schedule = (0.5, 0.2, 0.05)
        outer_epsilon_schedule = (0.5, 0.2, 0.1)
        max_iter = 1000
        tol = 1e-6
        warning_tol = 1e-4
        backend = "torch"
        dtype = "float64"

    predicted = torch.tensor([[0.70, 0.20, 0.10]], dtype=torch.float64)
    observed = torch.tensor([[0.20, 0.30, 0.50]], dtype=torch.float64)

    with pytest.raises(ContractError, match="BalancedSinkhornDivergenceConfig"):
        compute_balanced_sinkhorn_observation_discrepancy(
            predicted,
            observed,
            _geometry(),
            fov_cost_scale=1.0,
            config=DuckConfig(),  # type: ignore[arg-type]
        )


def test_balanced_sinkhorn_requires_explicit_fov_cost_scale() -> None:
    predicted = torch.tensor([[0.70, 0.20, 0.10]], dtype=torch.float64)
    observed = torch.tensor([[0.20, 0.30, 0.50]], dtype=torch.float64)

    with pytest.raises(ContractError, match="fov_cost_scale.*explicit"):
        compute_balanced_sinkhorn_observation_discrepancy(predicted, observed, _geometry())

    with pytest.raises(ContractError, match="s_G_init"):
        compute_balanced_sinkhorn_observation_discrepancy(
            predicted,
            observed,
            _geometry(),
            fov_cost_scale=0.0,
        )


def test_balanced_sinkhorn_defaults_and_provenance_metadata() -> None:
    geometry = _geometry()
    predicted = torch.tensor([[0.70, 0.20, 0.10]], dtype=torch.float64)
    observed = torch.tensor([[0.20, 0.30, 0.50]], dtype=torch.float64)

    result = compute_balanced_sinkhorn_observation_discrepancy(
        predicted,
        observed,
        geometry,
        fov_cost_scale=1.25,
    )
    metadata = result.metadata
    config = BalancedSinkhornDivergenceConfig()

    assert metadata["operator_version"] == "D_obs^BalancedSinkhornDivergence-v1"
    assert metadata["backend"] == "torch"
    assert metadata["dtype"] == "float64"
    assert tuple(metadata["inner_epsilon_schedule"]) == config.inner_epsilon_schedule
    assert tuple(metadata["outer_epsilon_schedule"]) == config.outer_epsilon_schedule
    assert metadata["max_iter"] == 1000
    assert metadata["tol"] == pytest.approx(1e-6)
    assert metadata["warning_tol"] == pytest.approx(1e-4)
    assert metadata["state_geometry"]["normalization"] == "C_norm = C_raw / s_C"
    assert metadata["state_geometry"]["s_C"] == pytest.approx(1.5)
    np.testing.assert_allclose(
        np.asarray(metadata["state_geometry"]["C_norm"], dtype=float),
        geometry.cost_matrix / geometry.cost_scale,
    )
    assert metadata["fov_ground_cost"]["normalization"] == "G_norm = G / s_G_init"
    assert metadata["fov_ground_cost"]["s_G_init"] == pytest.approx(1.25)
    assert metadata["fov_ground_cost"]["s_G_init_floor_used"] is False
    assert metadata["warning_flags"]["clipped_negative"] is False
    assert metadata["status"] == result.status
    assert result.status in {"ok", "ok_with_warnings"}


def test_balanced_sinkhorn_records_explicit_fov_scale_floor_usage() -> None:
    predicted = torch.tensor([[0.70, 0.20, 0.10]], dtype=torch.float64)
    observed = torch.tensor([[0.20, 0.30, 0.50]], dtype=torch.float64)

    result = compute_balanced_sinkhorn_observation_discrepancy(
        predicted,
        observed,
        _geometry(),
        fov_cost_scale=1.0,
        fov_cost_scale_floor_used=True,
    )

    assert result.metadata["fov_ground_cost"]["s_G_init"] == pytest.approx(1.0)
    assert result.metadata["fov_ground_cost"]["s_G_init_floor_used"] is True


def test_balanced_sinkhorn_metadata_exposes_outer_balanced_marginals() -> None:
    geometry = _geometry()
    predicted = torch.tensor(
        [[0.70, 0.20, 0.10], [0.10, 0.80, 0.10]],
        dtype=torch.float64,
    )
    observed = torch.tensor(
        [[0.20, 0.20, 0.60], [0.60, 0.30, 0.10], [0.25, 0.65, 0.10]],
        dtype=torch.float64,
    )

    result = compute_balanced_sinkhorn_observation_discrepancy(
        predicted,
        observed,
        geometry,
        fov_cost_scale=1.25,
    )
    marginals = result.metadata["outer_transport_marginals"]

    np.testing.assert_allclose(
        np.asarray(marginals["row_sums"], dtype=float),
        np.asarray(marginals["left_mass"], dtype=float),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(marginals["column_sums"], dtype=float),
        np.asarray(marginals["right_mass"], dtype=float),
        atol=1e-4,
    )


def test_small_negative_rule_clips_and_fails_fast_below_threshold() -> None:
    values = torch.tensor([-5e-11, 0.25], dtype=torch.float64)

    clipped, clipped_negative, warnings = _apply_small_negative_rule(values, label="unit-test")

    torch.testing.assert_close(clipped, torch.tensor([0.0, 0.25], dtype=torch.float64))
    assert clipped_negative is True
    assert warnings

    with pytest.raises(ContractError, match="unit-test"):
        _apply_small_negative_rule(torch.tensor([-2e-10], dtype=torch.float64), label="unit-test")
