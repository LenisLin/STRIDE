from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stride.errors import ContractError
from stride.io import build_adata
from stride.pp import build_local_features, build_state_basis


def _build_pp_adata(*, n_states: int = 3):
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
        n_states=n_states,
        k_neighbors=2,
    )
    build_local_features(adata)
    return adata


def test_build_state_basis_writes_canonical_slots() -> None:
    adata = _build_pp_adata(n_states=3)

    returned = build_state_basis(adata)

    assert returned is adata
    assert "state_id" in adata.obs
    assert "state_centroids" in adata.uns
    assert adata.obs["state_id"].shape[0] == adata.n_obs
    assert adata.obs["state_id"].dtype.kind in {"i", "u"}
    assert adata.obs["state_id"].between(0, 2).all()
    assert adata.uns["state_centroids"].shape == (3, 3)


def test_build_state_basis_does_not_write_geometry_slots() -> None:
    adata = _build_pp_adata(n_states=3)

    build_state_basis(adata)

    assert "cost_matrix" not in adata.uns
    assert "cost_scale" not in adata.uns


def test_build_state_basis_reuses_existing_valid_basis() -> None:
    adata = _build_pp_adata(n_states=3)
    build_state_basis(adata)
    existing_state_id = adata.obs["state_id"].copy()
    existing_centroids = adata.uns["state_centroids"].copy()

    with pytest.warns(
        UserWarning,
        match=(
            "adata.obs\\['state_id'\\] and adata.uns\\['state_centroids'\\] "
            "already exist; reusing existing values"
        ),
    ):
        returned = build_state_basis(adata)

    assert returned is adata
    pd.testing.assert_series_equal(adata.obs["state_id"], existing_state_id)
    np.testing.assert_array_equal(adata.uns["state_centroids"], existing_centroids)


def test_build_state_basis_rejects_partial_existing_basis() -> None:
    adata = _build_pp_adata(n_states=3)
    adata.obs["state_id"] = np.zeros(adata.n_obs, dtype=int)

    with pytest.raises(ContractError, match="partial"):
        build_state_basis(adata)


def test_build_state_basis_rejects_too_few_feature_rows_for_n_states() -> None:
    adata = _build_pp_adata(n_states=9)

    with pytest.raises(ContractError, match="fewer than config\\['n_states'\\]"):
        build_state_basis(adata)
