from __future__ import annotations

import numpy as np
import pytest

from stride.errors import ContractError
from stride.geometry import build_similarity_graph, build_state_geometry


def test_build_state_geometry_uses_positive_off_diagonal_median_scale() -> None:
    geometry = build_state_geometry(
        cost_matrix=np.asarray(
            [
                [0.0, 2.0, 4.0],
                [2.0, 0.0, 6.0],
                [4.0, 6.0, 0.0],
            ],
            dtype=float,
        ),
        n_neighbors=1,
    )

    assert geometry.cost_scale == pytest.approx(4.0)
    np.testing.assert_allclose(geometry.cost_matrix, geometry.cost_matrix.T)
    np.testing.assert_allclose(np.diag(geometry.cost_matrix), 0.0)
    assert geometry.state_ids == (0, 1, 2)


@pytest.mark.parametrize(
    ("cost_matrix", "message"),
    [
        (
            np.asarray([[0.0, np.nan], [np.nan, 0.0]], dtype=float),
            "finite",
        ),
        (
            np.asarray([[0.0, -1.0], [-1.0, 0.0]], dtype=float),
            "nonnegative",
        ),
        (
            np.asarray([[0.0, 1.0], [2.0, 0.0]], dtype=float),
            "symmetric",
        ),
        (
            np.asarray([[1.0, 2.0], [2.0, 0.0]], dtype=float),
            "diagonal",
        ),
        (
            np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=float),
            "positive off-diagonal",
        ),
    ],
)
def test_build_state_geometry_rejects_invalid_geometry_cost_matrix(
    cost_matrix: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        build_state_geometry(cost_matrix=cost_matrix)


def test_build_state_geometry_rejects_all_zero_cost_matrix_even_with_explicit_scale() -> None:
    with pytest.raises(ContractError, match="positive off-diagonal"):
        build_state_geometry(
            cost_matrix=np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=float),
            cost_scale=1.0,
        )


def test_build_state_geometry_rejects_state_id_length_mismatch() -> None:
    with pytest.raises(ContractError, match="state_ids"):
        build_state_geometry(
            cost_matrix=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float),
            state_ids=(10,),
        )


def test_build_similarity_graph_rejects_structurally_invalid_cost_matrix() -> None:
    with pytest.raises(ContractError, match="symmetric"):
        build_similarity_graph(
            np.asarray([[0.0, 1.0], [3.0, 0.0]], dtype=float),
            n_neighbors=1,
        )
