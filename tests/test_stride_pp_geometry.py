from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stride.errors import ContractError
from stride.io import build_adata
from stride.pp import build_local_features, build_state_basis, build_state_geometry


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
    build_state_basis(adata)
    return adata


def _canonical_centroid_adata():
    adata = _build_pp_adata(n_states=3)
    adata.uns["state_centroids"] = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    return adata


def _canonical_cost_matrix() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, np.sqrt(5.0)],
            [2.0, np.sqrt(5.0), 0.0],
        ],
        dtype=float,
    )


def _stale_cost_matrix() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ],
        dtype=float,
    )


def test_build_state_geometry_writes_canonical_slots() -> None:
    adata = _canonical_centroid_adata()

    returned = build_state_geometry(adata)

    assert returned is adata
    assert "cost_matrix" in adata.uns
    assert "cost_scale" in adata.uns
    assert adata.uns["cost_matrix"].shape == (3, 3)
    np.testing.assert_allclose(adata.uns["cost_matrix"], _canonical_cost_matrix())
    assert float(adata.uns["cost_scale"]) == 2.0
    np.testing.assert_allclose(adata.uns["cost_matrix"], adata.uns["cost_matrix"].T)
    np.testing.assert_allclose(np.diag(adata.uns["cost_matrix"]), 0.0)


def test_build_state_geometry_reuses_existing_valid_geometry() -> None:
    adata = _canonical_centroid_adata()
    build_state_geometry(adata)
    existing_matrix = adata.uns["cost_matrix"].copy()
    existing_scale = float(adata.uns["cost_scale"])

    with pytest.warns(
        UserWarning,
        match=(
            "adata.uns\\['cost_matrix'\\] and adata.uns\\['cost_scale'\\] already "
            "exist; using existing precomputed geometry"
        ),
    ):
        returned = build_state_geometry(adata)

    assert returned is adata
    np.testing.assert_array_equal(adata.uns["cost_matrix"], existing_matrix)
    assert float(adata.uns["cost_scale"]) == existing_scale


def test_build_state_geometry_completes_existing_cost_matrix() -> None:
    adata = _canonical_centroid_adata()
    adata.uns["cost_matrix"] = _canonical_cost_matrix()
    adata.uns.pop("cost_scale", None)

    with pytest.warns(
        UserWarning,
        match=(
            "adata.uns\\['cost_matrix'\\] exists without adata.uns\\['cost_scale'\\]; "
            "computed cost_scale from existing precomputed geometry"
        ),
    ):
        returned = build_state_geometry(adata)

    assert returned is adata
    assert float(adata.uns["cost_scale"]) == 2.0


def test_build_state_geometry_rejects_lone_cost_scale() -> None:
    adata = _canonical_centroid_adata()
    adata.uns["cost_scale"] = 1.0

    with pytest.raises(ContractError, match="shared-state geometry is partial"):
        build_state_geometry(adata)


def test_build_state_geometry_rejects_cost_scale_mismatch() -> None:
    adata = _canonical_centroid_adata()
    adata.uns["cost_matrix"] = _canonical_cost_matrix()
    adata.uns["cost_scale"] = 1.5

    with pytest.raises(
        ContractError,
        match="must match the median positive off-diagonal cost",
    ):
        build_state_geometry(adata)


def test_build_state_geometry_accepts_valid_precomputed_non_euclidean_matrix() -> None:
    adata = _canonical_centroid_adata()
    adata.uns["cost_matrix"] = _stale_cost_matrix()
    adata.uns["cost_scale"] = 2.0

    with pytest.warns(UserWarning, match="using existing precomputed geometry"):
        returned = build_state_geometry(adata)

    assert returned is adata
    np.testing.assert_array_equal(adata.uns["cost_matrix"], _stale_cost_matrix())
    assert float(adata.uns["cost_scale"]) == 2.0


def test_build_state_geometry_builds_from_named_metric() -> None:
    adata = _canonical_centroid_adata()

    build_state_geometry(adata, metric="cityblock")

    np.testing.assert_allclose(adata.uns["cost_matrix"], _stale_cost_matrix())
    assert float(adata.uns["cost_scale"]) == 2.0


def test_build_state_geometry_rejects_unknown_metric() -> None:
    adata = _canonical_centroid_adata()

    with pytest.raises(ContractError, match="not a supported scipy cdist metric"):
        build_state_geometry(adata, metric="not_a_metric")


def test_build_state_geometry_rejects_degenerate_no_positive_cost() -> None:
    adata = _canonical_centroid_adata()
    adata.uns["state_centroids"] = np.zeros((3, 3), dtype=float)
    adata.uns["cost_matrix"] = np.zeros((3, 3), dtype=float)

    with pytest.raises(ContractError, match="positive off-diagonal cost"):
        build_state_geometry(adata)


def test_build_state_geometry_warns_zero_offdiag_pairs() -> None:
    adata = _canonical_centroid_adata()
    adata.uns["state_centroids"] = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    with pytest.warns(UserWarning) as recorded:
        build_state_geometry(adata)

    messages = [str(item.message) for item in recorded]
    assert any(
        (
            "adata.uns['cost_matrix'] contains 1 zero off-diagonal state pairs; "
            "the corresponding states are geometry-equivalent"
        )
        in message
        for message in messages
    )
    assert float(adata.uns["cost_scale"]) == 1.0


def test_build_state_geometry_rejects_invalid_matrix_shape() -> None:
    adata = _canonical_centroid_adata()
    adata.uns["cost_matrix"] = np.zeros((2, 2), dtype=float)

    with pytest.raises(ContractError, match="shape must be \\[n_states, n_states\\]"):
        build_state_geometry(adata)


def test_build_state_geometry_requires_state_centroids() -> None:
    adata = _build_pp_adata(n_states=3)
    del adata.uns["state_centroids"]

    with pytest.raises(ContractError, match="adata.uns\\['state_centroids'\\] is required"):
        build_state_geometry(adata)
