from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stride.errors import ContractError
from stride.io import build_adata
from stride.pp import (
    build_fov_observations,
    build_local_features,
    build_state_basis,
    build_state_geometry,
    validate_ready,
)


def _build_ready_adata():
    cell = pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(1, 9)],
            "patient": ["p1"] * 8,
            "time": ["pre", "pre", "pre", "pre", "post", "post", "post", "post"],
            "fov": ["f1", "f1", "f1", "f1", "f2", "f2", "f2", "f2"],
            "domain": ["TC", "TC", "TC", "TC", "IM", "IM", "IM", "IM"],
            "cell_type": ["a", "a", "b", "c", "b", "b", "a", "c"],
            "x": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
            "y": [0.0] * 8,
        }
    )
    adata = build_adata(
        X=np.ones((cell.shape[0], 2), dtype=float),
        var=["g1", "g2"],
        cell=cell,
        cell_id="cell_id",
        patient="patient",
        time="time",
        fov_id="fov",
        domain="domain",
        cell_type="cell_type",
        x="x",
        y="y",
        source="pre",
        target="post",
        time_order=["pre", "post"],
        relations=[("TC", "IM")],
        community_mode="fraction",
        n_states=3,
        k_neighbors=2,
    )
    build_local_features(adata)
    build_state_basis(adata)
    build_state_geometry(adata)
    build_fov_observations(adata)
    return adata


def test_validate_ready_accepts_complete_pp_outputs() -> None:
    adata = _build_ready_adata()

    assert validate_ready(adata) is None


def test_validate_ready_rejects_missing_state_id() -> None:
    adata = _build_ready_adata()
    del adata.obs["state_id"]

    with pytest.raises(ContractError):
        validate_ready(adata)


def test_validate_ready_rejects_missing_state_centroids() -> None:
    adata = _build_ready_adata()
    del adata.uns["state_centroids"]

    with pytest.raises(ContractError):
        validate_ready(adata)


@pytest.mark.parametrize("slot", ["cost_matrix", "cost_scale"])
def test_validate_ready_rejects_missing_geometry_slot(slot: str) -> None:
    adata = _build_ready_adata()
    del adata.uns[slot]

    with pytest.raises(ContractError):
        validate_ready(adata)


def test_validate_ready_rejects_missing_fov_observations() -> None:
    adata = _build_ready_adata()
    del adata.uns["stride"]["fov_observations"]

    with pytest.raises(ContractError):
        validate_ready(adata)


@pytest.mark.parametrize("slot", ["cost_matrix", "community_composition"])
def test_validate_ready_rejects_k_axis_mismatch(slot: str) -> None:
    adata = _build_ready_adata()
    if slot == "cost_matrix":
        adata.uns["cost_matrix"] = np.zeros((2, 2), dtype=float)
    else:
        adata.uns["stride"]["fov_observations"]["community_composition"] = np.zeros(
            (2, 2),
            dtype=float,
        )

    with pytest.raises(ContractError):
        validate_ready(adata)


@pytest.mark.parametrize(
    ("cost_matrix", "message"),
    [
        (np.array([[0.0, np.nan, 1.0], [np.nan, 0.0, 1.0], [1.0, 1.0, 0.0]]), "NaN/Inf"),
        (np.array([[0.0, -1.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]), "nonnegative"),
        (np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 1.0], [2.0, 1.0, 0.0]]), "symmetric"),
        (np.array([[1.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]), "diagonal"),
    ],
)
def test_validate_ready_rejects_invalid_geometry_values(
    cost_matrix: np.ndarray,
    message: str,
) -> None:
    adata = _build_ready_adata()
    adata.uns["cost_matrix"] = cost_matrix

    with pytest.raises(ContractError, match=message):
        validate_ready(adata)


def test_validate_ready_rejects_invalid_cost_scale() -> None:
    adata = _build_ready_adata()
    adata.uns["cost_scale"] = 0.0

    with pytest.raises(ContractError, match="finite and strictly positive"):
        validate_ready(adata)


def test_validate_ready_rejects_relation_without_source_support() -> None:
    adata = _build_ready_adata()
    metadata = adata.uns["stride"]["fov_observations"]["metadata"].copy()
    source_rows = metadata["timepoint"] == "pre"
    metadata.loc[source_rows, "domain_label"] = "IM"
    adata.uns["stride"]["fov_observations"]["metadata"] = metadata

    with pytest.raises(ContractError, match="relations\\[0\\].*source"):
        validate_ready(adata)


def test_validate_ready_rejects_relation_without_target_support() -> None:
    adata = _build_ready_adata()
    metadata = adata.uns["stride"]["fov_observations"]["metadata"].copy()
    target_rows = metadata["timepoint"] == "post"
    metadata.loc[target_rows, "domain_label"] = "TC"
    adata.uns["stride"]["fov_observations"]["metadata"] = metadata

    with pytest.raises(ContractError, match="relations\\[0\\].*target"):
        validate_ready(adata)
