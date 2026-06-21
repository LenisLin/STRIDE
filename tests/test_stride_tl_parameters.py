from __future__ import annotations

import pytest
import torch

from stride.errors import ContractError
from stride.tl._parameters import (
    ParameterLogits,
    _normalize_patient_ids,
    constrain_parameters,
    initialize_parameters,
    predict_target_composition,
)


def test_normalize_patient_ids_strips_and_preserves_order() -> None:
    assert _normalize_patient_ids([" p2 ", "p1", 3]) == ("p2", "p1", "3")


@pytest.mark.parametrize("patient_ids", [[], ["p1", " "], ["p1", "p1"], "p1", b"p1"])
def test_normalize_patient_ids_rejects_invalid_ids(patient_ids: object) -> None:
    with pytest.raises(ContractError):
        _normalize_patient_ids(patient_ids)  # type: ignore[arg-type]


def test_initialize_parameters_returns_constrained_feasible_shapes() -> None:
    logits = initialize_parameters(["p1", "p2"], 3)

    assert logits.patient_ids == ("p1", "p2")
    assert logits.row_logits.shape == (2, 3, 4)
    assert logits.e_logits.shape == (2, 3)
    assert logits.row_logits.dtype == torch.float64
    assert logits.e_logits.dtype == torch.float64

    parameters = constrain_parameters(logits)

    assert parameters.A.shape == (2, 3, 3)
    assert parameters.d.shape == (2, 3)
    assert parameters.e.shape == (2, 3)
    torch.testing.assert_close(
        parameters.A.sum(dim=2) + parameters.d,
        torch.ones((2, 3), dtype=torch.float64),
    )
    assert bool((parameters.A >= 0.0).all())
    assert bool((parameters.d >= 0.0).all())
    assert bool(((parameters.e >= 0.0) & (parameters.e <= 1.0)).all())
    torch.testing.assert_close(
        parameters.A,
        torch.tensor(
            [
                [[0.93, 0.01, 0.01], [0.01, 0.93, 0.01], [0.01, 0.01, 0.93]],
                [[0.93, 0.01, 0.01], [0.01, 0.93, 0.01], [0.01, 0.01, 0.93]],
            ],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(parameters.d, torch.full((2, 3), 0.05, dtype=torch.float64))
    torch.testing.assert_close(parameters.e, torch.full((2, 3), 0.05 / 3.0, dtype=torch.float64))


def test_initialize_parameters_supports_single_state() -> None:
    logits = initialize_parameters(["p1"], 1)
    parameters = constrain_parameters(logits)

    assert logits.row_logits.shape == (1, 1, 2)
    assert logits.e_logits.shape == (1, 1)
    torch.testing.assert_close(
        parameters.A.sum(dim=2) + parameters.d,
        torch.ones((1, 1), dtype=torch.float64),
    )
    torch.testing.assert_close(parameters.A, torch.tensor([[[0.95]]], dtype=torch.float64))
    torch.testing.assert_close(parameters.d, torch.tensor([[0.05]], dtype=torch.float64))
    torch.testing.assert_close(parameters.e, torch.tensor([[0.05]], dtype=torch.float64))


@pytest.mark.parametrize("n_states", [0, -1, True, 1.5, "3"])
def test_initialize_parameters_rejects_invalid_state_count(n_states: object) -> None:
    with pytest.raises(ContractError):
        initialize_parameters(["p1"], n_states)  # type: ignore[arg-type]


def test_constrain_parameters_rejects_bad_object() -> None:
    with pytest.raises(ContractError):
        constrain_parameters(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "logits",
    [
        ParameterLogits(("p1",), torch.zeros((1, 2), dtype=torch.float64), torch.zeros((1, 2))),
        ParameterLogits(("p1",), torch.zeros((1, 2, 2), dtype=torch.float64), torch.zeros((1, 2))),
        ParameterLogits(("p1", "p2"), torch.zeros((1, 2, 3), dtype=torch.float64), torch.zeros((1, 2))),
        ParameterLogits(("p1",), torch.zeros((1, 2, 3), dtype=torch.float64), torch.zeros((1, 3))),
    ],
)
def test_constrain_parameters_rejects_bad_logits_shape(logits: ParameterLogits) -> None:
    with pytest.raises(ContractError):
        constrain_parameters(logits)


def test_constrain_parameters_outputs_finite_nonnegative_bounded_parameters() -> None:
    logits = ParameterLogits(
        ("p1",),
        torch.tensor([[[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]]]),
        torch.tensor([[0.5, -0.5]]),
    )

    parameters = constrain_parameters(logits)

    assert bool(torch.isfinite(parameters.A).all())
    assert bool(torch.isfinite(parameters.d).all())
    assert bool(torch.isfinite(parameters.e).all())
    assert bool((parameters.A >= 0.0).all())
    assert bool((parameters.d >= 0.0).all())
    assert bool(((parameters.e >= 0.0) & (parameters.e <= 1.0)).all())


def test_predict_target_composition_normalizes_raw_prediction() -> None:
    source_bag = torch.tensor([[1.0, 0.0], [0.25, 0.75]])
    A = torch.tensor([[0.8, 0.1], [0.2, 0.7]])
    e = torch.tensor([0.05, 0.15])

    predicted = predict_target_composition(source_bag, A, e)

    assert predicted.shape == (2, 2)
    assert predicted.dtype == torch.float64
    torch.testing.assert_close(
        predicted.sum(dim=1),
        torch.ones(2, dtype=torch.float64),
    )
    assert predicted.device == A.device


def test_predict_target_composition_uses_parameter_device_on_cpu() -> None:
    source_bag = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    A = torch.eye(2, dtype=torch.float64, device=torch.device("cpu"))
    e = torch.zeros(2, dtype=torch.float64, device=A.device)

    predicted = predict_target_composition(source_bag, A, e)

    assert predicted.device == A.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_predict_target_composition_keeps_cuda_parameter_device() -> None:
    device = torch.device("cuda")
    source_bag = torch.tensor([[1.0, 0.0]], dtype=torch.float64, device=torch.device("cpu"))
    A = torch.eye(2, dtype=torch.float64, device=device)
    e = torch.zeros(2, dtype=torch.float64, device=device)

    predicted = predict_target_composition(source_bag, A, e)

    assert predicted.device == A.device


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS is not available",
)
def test_predict_target_composition_keeps_mps_parameter_device() -> None:
    device = torch.device("mps")
    source_bag = torch.tensor([[1.0, 0.0]], dtype=torch.float64, device=torch.device("cpu"))
    A = torch.eye(2, dtype=torch.float64, device=device)
    e = torch.zeros(2, dtype=torch.float64, device=device)

    predicted = predict_target_composition(source_bag, A, e)

    assert predicted.device == A.device


@pytest.mark.parametrize(
    "source_bag",
    [
        torch.tensor([1.0, 0.0]),
        torch.empty((0, 2)),
    ],
)
def test_predict_target_composition_rejects_bad_source(source_bag: torch.Tensor) -> None:
    with pytest.raises(ContractError):
        predict_target_composition(
            source_bag,
            torch.eye(2, dtype=torch.float64),
            torch.zeros(2, dtype=torch.float64),
        )


@pytest.mark.parametrize(
    ("A", "e"),
    [
        (torch.eye(3, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)),
        (torch.eye(2, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)),
    ],
)
def test_predict_target_composition_rejects_bad_A_or_e(A: torch.Tensor, e: torch.Tensor) -> None:
    with pytest.raises(ContractError):
        predict_target_composition(torch.tensor([[1.0, 0.0]]), A, e)


def test_predict_target_composition_does_not_audit_source_simplex() -> None:
    predicted = predict_target_composition(
        torch.tensor([[0.2, 0.2]], dtype=torch.float64),
        torch.eye(2, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
    )

    assert predicted.shape == (1, 2)
