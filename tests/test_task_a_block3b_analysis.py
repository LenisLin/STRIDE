from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tasks.task_A.block3.analysis import derive_A_d_e_from_plan


def test_derive_A_d_e_from_plan_uses_frozen_marginal_rule() -> None:
    x = np.array([2.0, 0.0, 4.0])
    y = np.array([1.0, 3.0, 5.0])
    P = np.array(
        [
            [0.5, 1.0, 0.0],
            [0.2, 0.3, 0.4],
            [0.0, 1.0, 2.0],
        ]
    )

    A, d, e = derive_A_d_e_from_plan(x=x, y=y, P=P)

    np.testing.assert_allclose(
        A,
        np.array(
            [
                [0.25, 0.5, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.25, 0.5],
            ]
        ),
    )
    np.testing.assert_allclose(d, np.array([0.25, 0.0, 0.25]))
    np.testing.assert_allclose(e, np.array([0.3, 0.7, 2.6]))


@pytest.mark.parametrize(
    "x,y,P,match",
    [
        (np.array([1.0, -0.1]), np.array([1.0, 0.0]), np.eye(2), "non-negative"),
        (np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([[1.0, np.nan], [0.0, 0.0]]), "finite"),
        (np.array([1.0]), np.array([1.0, 0.0]), np.eye(2), "shape"),
    ],
)
def test_derive_A_d_e_from_plan_rejects_invalid_inputs(
    x: np.ndarray,
    y: np.ndarray,
    P: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        derive_A_d_e_from_plan(x=x, y=y, P=P)
