from __future__ import annotations

import pytest
import torch

import stride.tl._sinkhorn as sinkhorn_module
from stride.errors import ContractError
from stride.tl._sinkhorn import (
    BALANCED_SINKHORN_OPERATOR_VERSION,
    SinkhornConfig,
    compute_fov_ground_cost_matrix,
    compute_observed_self_ground_cost,
    compute_sinkhorn_divergence,
)


def _bags_and_cost() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted = torch.tensor(
        [[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]],
        dtype=torch.float64,
    )
    observed = torch.tensor(
        [[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]],
        dtype=torch.float64,
    )
    cost = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]],
        dtype=torch.float64,
    )
    return predicted, observed, cost


def test_sinkhorn_config_defaults_are_canonical() -> None:
    config = SinkhornConfig()

    assert config.inner_epsilon_schedule == (0.5, 0.2, 0.1)
    assert config.outer_epsilon_schedule == (0.5, 0.2, 0.1)
    assert config.max_iter == 100
    assert config.tol == 1e-6
    assert config.warning_tol == 1e-4
    assert config.backend == "torch"
    assert config.dtype == "float64"
    assert config.debiased is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"inner_epsilon_schedule": (0.1,)},
        {"outer_epsilon_schedule": (0.1,)},
        {"max_iter": 50},
        {"tol": 1e-5},
        {"warning_tol": 1e-3},
        {"backend": "numpy"},
        {"dtype": "float32"},
        {"debiased": False},
    ],
)
def test_sinkhorn_config_rejects_noncanonical(kwargs: dict[str, object]) -> None:
    with pytest.raises(ContractError):
        SinkhornConfig(**kwargs)


def test_noncanonical_config_helper_is_not_part_of_live_sinkhorn_surface() -> None:
    config_type_name = "_" + "Diagnostic" + "SinkhornConfig"
    helper_name = "_diagnostic" + "_sinkhorn_config"

    assert not hasattr(sinkhorn_module, config_type_name)
    assert not hasattr(sinkhorn_module, helper_name)


@pytest.mark.parametrize("n_left,n_right", [(2, 2), (2, 3)])
def test_batched_sinkhorn_divergence_matches_scalar_path(
    n_left: int,
    n_right: int,
) -> None:
    config = SinkhornConfig()
    batch_size = 2
    left_mass = torch.full((batch_size, n_left), 1.0 / float(n_left), dtype=torch.float64)
    right_mass = torch.full((batch_size, n_right), 1.0 / float(n_right), dtype=torch.float64)
    cross_cost = torch.arange(
        1,
        1 + batch_size * n_left * n_right,
        dtype=torch.float64,
    ).reshape(batch_size, n_left, n_right)
    cross_cost = cross_cost / cross_cost.max()
    left_self_base = torch.abs(
        torch.arange(n_left, dtype=torch.float64)[:, None]
        - torch.arange(n_left, dtype=torch.float64)[None, :]
    )
    right_self_base = torch.abs(
        torch.arange(n_right, dtype=torch.float64)[:, None]
        - torch.arange(n_right, dtype=torch.float64)[None, :]
    )
    left_self_cost = left_self_base.unsqueeze(0).repeat(batch_size, 1, 1)
    right_self_cost = right_self_base.unsqueeze(0).repeat(batch_size, 1, 1)

    batched = sinkhorn_module._batched_sinkhorn_divergence_value(
        left_mass,
        right_mass,
        cross_cost,
        left_self_cost,
        right_self_cost,
        epsilon_schedule=config.outer_epsilon_schedule,
        config=config,
        label="outer_fov_bag_divergence.test",
    )
    scalar_values = []
    for index in range(batch_size):
        scalar = sinkhorn_module._sinkhorn_divergence_value(
            left_mass[index],
            right_mass[index],
            cross_cost[index],
            left_self_cost[index],
            right_self_cost[index],
            epsilon_schedule=config.outer_epsilon_schedule,
            config=config,
            label="outer_fov_bag_divergence.test",
        )
        scalar_values.append(scalar.value)

    torch.testing.assert_close(
        batched.value,
        torch.stack(scalar_values),
        atol=1e-8,
        rtol=0.0,
    )


def test_equal_size_batched_sinkhorn_warning_labels_match_scalar_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SinkhornConfig()
    batch_size = 2
    n_fov = 2
    mass = torch.full((batch_size, n_fov), 1.0 / float(n_fov), dtype=torch.float64)
    cross_cost = torch.tensor(
        [
            [[0.0, 0.2], [0.3, 0.0]],
            [[0.0, 0.4], [0.1, 0.0]],
        ],
        dtype=torch.float64,
    )
    self_cost = torch.zeros((batch_size, n_fov, n_fov), dtype=torch.float64)
    labels: list[str] = []
    shapes: list[tuple[int, ...]] = []

    def fake_warning(
        transport: sinkhorn_module._BatchedSinkhornValue,
        *,
        label: str,
        epsilon_schedule: tuple[float, ...],
        config: SinkhornConfig,
    ) -> tuple[dict[str, object], ...]:
        labels.append(label)
        shapes.append(tuple(transport.value.shape))
        return ({"type": "diagnostic_called", "label": label},)

    monkeypatch.setattr(
        sinkhorn_module,
        "_compact_convergence_warnings",
        fake_warning,
    )

    result = sinkhorn_module._batched_sinkhorn_divergence_value(
        mass,
        mass,
        cross_cost,
        self_cost,
        self_cost,
        epsilon_schedule=config.outer_epsilon_schedule,
        config=config,
        label="outer_fov_bag_divergence.equal_size",
        collect_warnings=True,
    )

    assert result.warnings
    assert labels == [
        "outer_fov_bag_divergence.equal_size.cross_forward",
        "outer_fov_bag_divergence.equal_size.cross_reverse",
        "outer_fov_bag_divergence.equal_size.left_self",
        "outer_fov_bag_divergence.equal_size.right_self",
    ]
    assert shapes == [(batch_size,), (batch_size,), (batch_size,), (batch_size,)]


def test_compute_sinkhorn_divergence_returns_finite_scalar_with_scale_metadata() -> None:
    predicted, observed, cost = _bags_and_cost()

    result = compute_sinkhorn_divergence(
        predicted,
        observed,
        cost,
        1.5,
        fov_cost_scale=2.0,
    )

    assert result.value.shape == ()
    assert result.value.dtype == torch.float64
    assert result.value.device == predicted.device
    assert bool(torch.isfinite(result.value))
    assert bool(result.value >= 0.0)
    assert result.metadata["operator_version"] == BALANCED_SINKHORN_OPERATOR_VERSION
    assert result.metadata["state_geometry"]["s_C"] == 1.5
    assert result.metadata["fov_ground_cost"]["s_G_init"] == 2.0


def test_identical_bags_have_near_zero_divergence() -> None:
    predicted, _, cost = _bags_and_cost()

    result = compute_sinkhorn_divergence(
        predicted,
        predicted,
        cost,
        1.5,
        fov_cost_scale=1.0,
    )

    assert bool(result.value >= 0.0)
    assert bool(result.value < 1e-6)


def test_compute_sinkhorn_divergence_supports_unequal_fov_counts() -> None:
    _, observed, cost = _bags_and_cost()
    predicted = torch.tensor(
        [[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.25, 0.25, 0.5]],
        dtype=torch.float64,
    )

    result = compute_sinkhorn_divergence(
        predicted,
        observed,
        cost,
        1.5,
        fov_cost_scale=1.0,
    )

    assert result.value.shape == ()
    assert bool(torch.isfinite(result.value))
    assert bool(result.value >= 0.0)


def test_compute_fov_ground_cost_matrix_returns_finite_matrix() -> None:
    predicted, observed, cost = _bags_and_cost()

    result = compute_fov_ground_cost_matrix(predicted, observed, cost, 1.5)

    assert result.value.shape == (predicted.shape[0], observed.shape[0])
    assert result.value.dtype == torch.float64
    assert result.value.requires_grad is False
    assert bool(torch.isfinite(result.value).all())
    assert "outer_fov_bag_divergence" not in result.metadata
    assert result.metadata["fov_ground_cost"]["role"] == (
        "setup_time_inner_ground_cost_matrix"
    )


def test_fov_ground_cost_matrix_matches_observed_self_ground_cost() -> None:
    _, observed, cost = _bags_and_cost()

    matrix = compute_fov_ground_cost_matrix(observed, observed, cost, 1.5)
    observed_self = compute_observed_self_ground_cost(observed, cost, 1.5)

    torch.testing.assert_close(matrix.value, observed_self.value)
    assert matrix.metadata["clipped_negative"] == observed_self.metadata["clipped_negative"]


def test_fov_ground_cost_matrix_validates_simplex_inputs() -> None:
    _, observed, cost = _bags_and_cost()
    bad_left = torch.tensor([[0.2, 0.2, 0.2]], dtype=torch.float64)

    with pytest.raises(ContractError):
        compute_fov_ground_cost_matrix(bad_left, observed, cost, 1.5)


def test_fov_cost_scale_changes_outer_value_without_changing_state_scale() -> None:
    predicted, observed, cost = _bags_and_cost()

    unit_result = compute_sinkhorn_divergence(
        predicted,
        observed,
        cost,
        1.5,
        fov_cost_scale=1.0,
    )
    scaled_result = compute_sinkhorn_divergence(
        predicted,
        observed,
        cost,
        1.5,
        fov_cost_scale=2.0,
    )

    assert unit_result.metadata["state_geometry"]["s_C"] == 1.5
    assert scaled_result.metadata["state_geometry"]["s_C"] == 1.5
    assert unit_result.metadata["fov_ground_cost"]["s_G_init"] == 1.0
    assert scaled_result.metadata["fov_ground_cost"]["s_G_init"] == 2.0
    assert not bool(torch.isclose(unit_result.value, scaled_result.value))


def test_cached_observed_self_ground_cost_matches_uncached_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predicted, observed, cost = _bags_and_cost()

    uncached = compute_sinkhorn_divergence(
        predicted,
        observed,
        cost,
        1.5,
        fov_cost_scale=1.0,
    )
    observed_self = compute_observed_self_ground_cost(
        observed,
        cost,
        1.5,
    )

    original_pairwise = sinkhorn_module._pairwise_composition_ground_cost_value
    labels: list[str] = []

    def tracking_pairwise(*args: object, **kwargs: object) -> object:
        labels.append(str(kwargs["label"]))
        return original_pairwise(*args, **kwargs)

    monkeypatch.setattr(
        sinkhorn_module,
        "_pairwise_composition_ground_cost_value",
        tracking_pairwise,
    )
    cached = compute_sinkhorn_divergence(
        predicted,
        observed,
        cost,
        1.5,
        fov_cost_scale=1.0,
        observed_self_ground_cost=observed_self.value,
        observed_self_clipped_negative=bool(observed_self.metadata["clipped_negative"]),
    )

    torch.testing.assert_close(cached.value, uncached.value)
    assert "inner_composition_distance.cross" in labels
    assert "inner_composition_distance.predicted_self" in labels
    assert "inner_composition_distance.observed_self" not in labels


def test_observed_self_ground_cost_is_setup_constant() -> None:
    _, observed, cost = _bags_and_cost()
    observed = observed.clone().requires_grad_(True)
    cost = cost.clone().requires_grad_(True)

    result = compute_observed_self_ground_cost(
        observed,
        cost,
        1.5,
    )

    assert result.value.requires_grad is False


def test_predicted_bag_gradient_path_is_available() -> None:
    predicted, observed, cost = _bags_and_cost()
    predicted = predicted.clone().requires_grad_(True)

    result = compute_sinkhorn_divergence(
        predicted,
        observed,
        cost,
        1.5,
        fov_cost_scale=1.0,
    )
    result.value.backward()

    assert predicted.grad is not None
    assert predicted.grad.shape == predicted.shape
    assert bool(torch.isfinite(predicted.grad).all())


def test_validated_tensor_route_skips_repeated_distribution_audit() -> None:
    _, observed, cost = _bags_and_cost()
    prevalidated_predicted = torch.tensor(
        [[0.2, 0.2, 0.2], [0.1, 0.2, 0.1]],
        dtype=torch.float64,
    )

    result = compute_sinkhorn_divergence(
        prevalidated_predicted,
        observed,
        cost,
        1.5,
        fov_cost_scale=1.0,
        validate_inputs=False,
    )

    assert result.value.shape == ()
    assert bool(torch.isfinite(result.value))


def test_default_value_path_does_not_collect_convergence_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predicted, observed, cost = _bags_and_cost()

    def fail_if_called(*args: object, **kwargs: object) -> tuple[dict[str, object], ...]:
        raise AssertionError("convergence warning collection should be explicit")

    monkeypatch.setattr(
        sinkhorn_module,
        "_compact_convergence_warnings",
        fail_if_called,
    )
    result = compute_sinkhorn_divergence(
        predicted,
        observed,
        cost,
        1.5,
        fov_cost_scale=1.0,
    )

    assert result.warnings == ()


def test_validated_hot_path_avoids_cpu_sync_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predicted, observed, cost = _bags_and_cost()
    observed_self = compute_observed_self_ground_cost(observed, cost, 1.5)

    def fail_sync(*args: object, **kwargs: object) -> bool:
        raise AssertionError("hot path should not use CPU-sync checks")

    monkeypatch.setattr(sinkhorn_module, "_ensure_bool", fail_sync)
    result = compute_sinkhorn_divergence(
        predicted,
        observed,
        cost,
        1.5,
        fov_cost_scale=1.0,
        observed_self_ground_cost=observed_self.value,
        observed_self_clipped_negative=bool(observed_self.metadata["clipped_negative"]),
        validate_inputs=False,
        collect_warnings=False,
    )

    assert result.value.shape == ()


def test_hot_path_small_negative_rule_marks_large_negative_as_nonfinite() -> None:
    values = torch.tensor([-1e-12, -1e-3, 2.0], dtype=torch.float64)

    clipped, clipped_negative, warnings = sinkhorn_module._apply_small_negative_rule(
        values,
        label="hot_path_probe",
        runtime_checks=False,
    )

    assert clipped_negative is False
    assert warnings == ()
    assert clipped[0] == 0.0
    assert bool(torch.isnan(clipped[1]))
    assert clipped[2] == 2.0


def test_explicit_diagnostic_path_collects_convergence_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predicted, observed, cost = _bags_and_cost()

    def fake_warning(*args: object, **kwargs: object) -> tuple[dict[str, object], ...]:
        return ({"type": "diagnostic_called", "label": str(kwargs["label"])},)

    monkeypatch.setattr(
        sinkhorn_module,
        "_compact_convergence_warnings",
        fake_warning,
    )
    result = compute_sinkhorn_divergence(
        predicted,
        observed,
        cost,
        1.5,
        fov_cost_scale=1.0,
        collect_warnings=True,
    )

    assert result.warnings
    assert {warning["type"] for warning in result.warnings} == {"diagnostic_called"}


@pytest.mark.parametrize(
    "bad_predicted",
    [
        torch.empty((0, 3), dtype=torch.float64),
        torch.tensor([[float("nan"), 0.5, 0.5]], dtype=torch.float64),
        torch.tensor([[-0.1, 0.6, 0.5]], dtype=torch.float64),
        torch.tensor([[0.2, 0.2, 0.2]], dtype=torch.float64),
    ],
)
def test_invalid_predicted_bag_raises(bad_predicted: torch.Tensor) -> None:
    _, observed, cost = _bags_and_cost()

    with pytest.raises(ContractError):
        compute_sinkhorn_divergence(
            bad_predicted,
            observed,
            cost,
            1.5,
            fov_cost_scale=1.0,
        )


@pytest.mark.parametrize("bad_cost_scale", [0.0, -1.0, float("inf"), float("nan")])
def test_bad_cost_scale_raises(bad_cost_scale: float) -> None:
    predicted, observed, cost = _bags_and_cost()

    with pytest.raises(ContractError):
        compute_sinkhorn_divergence(
            predicted,
            observed,
            cost,
            bad_cost_scale,
            fov_cost_scale=1.0,
        )


def test_all_zero_cost_matrix_raises() -> None:
    predicted, observed, _ = _bags_and_cost()
    cost = torch.zeros((3, 3), dtype=torch.float64)

    with pytest.raises(ContractError):
        compute_sinkhorn_divergence(
            predicted,
            observed,
            cost,
            1.5,
            fov_cost_scale=1.0,
        )


def test_bad_fov_cost_scale_raises() -> None:
    predicted, observed, cost = _bags_and_cost()

    with pytest.raises(ContractError):
        compute_sinkhorn_divergence(
            predicted,
            observed,
            cost,
            1.5,
            fov_cost_scale=0.0,
        )


def test_floor_flag_requires_unit_fov_scale() -> None:
    predicted, observed, cost = _bags_and_cost()

    with pytest.raises(ContractError):
        compute_sinkhorn_divergence(
            predicted,
            observed,
            cost,
            1.5,
            fov_cost_scale=2.0,
            fov_cost_scale_floor_used=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_output_stays_on_predicted_device_cuda() -> None:
    predicted, observed, cost = _bags_and_cost()
    predicted = predicted.cuda()
    observed = observed.cuda()
    cost = cost.cuda()

    result = compute_sinkhorn_divergence(
        predicted,
        observed,
        cost,
        1.5,
        fov_cost_scale=1.0,
    )

    assert result.value.device == predicted.device
