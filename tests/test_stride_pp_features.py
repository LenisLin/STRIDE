from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stride.errors import ContractError
from stride.io import build_adata
from stride.pp import build_local_features


def _build_pp_adata(*, repeated_fov: bool = False, k_neighbors: int = 2):
    if repeated_fov:
        cell = pd.DataFrame(
            {
                "cell_id": [f"c{i}" for i in range(1, 7)],
                "patient": ["p1", "p1", "p1", "p2", "p2", "p2"],
                "time": ["pre", "pre", "pre", "pre2", "pre2", "pre2"],
                "fov": ["shared", "shared", "shared", "shared", "shared", "shared"],
                "domain": ["TC", "TC", "TC", "TC", "TC", "TC"],
                "cell_type": ["a", "a", "b", "b", "b", "a"],
                "x": [0.0, 10.0, 20.0, 0.0, 10.0, 20.0],
                "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
    else:
        cell = pd.DataFrame(
            {
                "cell_id": [f"c{i}" for i in range(1, 7)],
                "patient": ["p1", "p1", "p1", "p1", "p1", "p1"],
                "time": ["pre", "pre", "pre", "post", "post", "post"],
                "fov": ["f1", "f1", "f1", "f2", "f2", "f2"],
                "domain": ["TC", "TC", "TC", "IM", "IM", "IM"],
                "cell_type": ["a", "a", "b", "b", "b", "a"],
                "x": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )

    return build_adata(
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
        target="post" if not repeated_fov else "pre2",
        time_order=["pre", "post"] if not repeated_fov else ["pre", "pre2"],
        relations=[("TC", "IM")] if not repeated_fov else [("TC", "TC")],
        community_mode="fraction",
        n_states=3,
        k_neighbors=k_neighbors,
    )


def test_build_local_features_writes_canonical_slots() -> None:
    adata = _build_pp_adata()

    returned = build_local_features(adata)

    assert returned is adata
    assert adata.obsm["local_state_features"].shape == (adata.n_obs, 2)
    np.testing.assert_allclose(
        adata.obsm["local_state_features"],
        np.asarray(
            [
                [0.5, 0.5],
                [0.5, 0.5],
                [1.0, 0.0],
                [0.5, 0.5],
                [0.5, 0.5],
                [0.0, 1.0],
            ],
            dtype=float,
        ),
    )
    metadata = adata.uns["state_feature_metadata"]
    assert metadata["feature_names"] == ["subtype_fraction:a", "subtype_fraction:b"]
    assert metadata["subtype_labels"] == ["a", "b"]
    assert metadata["k_neighbors"] == 2


def test_build_local_features_uses_full_fov_identity() -> None:
    adata = _build_pp_adata(repeated_fov=True)

    build_local_features(adata)

    np.testing.assert_allclose(
        adata.obsm["local_state_features"],
        np.asarray(
            [
                [0.5, 0.5],
                [0.5, 0.5],
                [1.0, 0.0],
                [0.5, 0.5],
                [0.5, 0.5],
                [0.0, 1.0],
            ],
            dtype=float,
        ),
    )


def test_build_local_features_excludes_self_when_coordinates_are_duplicated() -> None:
    cell = pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(1, 8)],
            "patient": ["p1"] * 7,
            "time": ["pre", "pre", "pre", "pre", "post", "post", "post"],
            "fov": ["f1", "f1", "f1", "f1", "f2", "f2", "f2"],
            "domain": ["TC", "TC", "TC", "TC", "IM", "IM", "IM"],
            "cell_type": ["a", "b", "b", "b", "a", "a", "b"],
            "x": [0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "y": [0.0] * 7,
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

    np.testing.assert_allclose(
        adata.obsm["local_state_features"][:4],
        np.asarray(
            [
                [0.0, 1.0],
                [0.5, 0.5],
                [0.5, 0.5],
                [0.0, 1.0],
            ],
            dtype=float,
        ),
    )


def test_build_local_features_rejects_fov_smaller_than_declared_neighborhood() -> None:
    adata = _build_pp_adata(k_neighbors=3)

    with pytest.raises(ContractError, match="fewer than k_neighbors \\+ 1 cells"):
        build_local_features(adata)


def test_build_local_features_reuses_existing_valid_slot() -> None:
    adata = _build_pp_adata()
    build_local_features(adata)
    existing = adata.obsm["local_state_features"].copy()

    with pytest.warns(
        UserWarning,
        match="adata.obsm\\['local_state_features'\\] already exists; reusing existing values",
    ):
        returned = build_local_features(adata)

    assert returned is adata
    np.testing.assert_array_equal(adata.obsm["local_state_features"], existing)


def test_build_local_features_rejects_existing_invalid_slot() -> None:
    adata = _build_pp_adata()
    adata.obsm["local_state_features"] = np.full((adata.n_obs, 2), -1.0, dtype=float)
    adata.uns["state_feature_metadata"] = {
        "feature_names": ["subtype_fraction:a", "subtype_fraction:b"],
        "subtype_labels": ["a", "b"],
        "k_neighbors": 2,
    }

    with pytest.raises(ContractError, match="nonnegative"):
        build_local_features(adata)
