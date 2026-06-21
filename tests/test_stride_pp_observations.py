from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stride.errors import ContractError
from stride.io import build_adata
from stride.pp import build_fov_observations


def _build_observation_adata():
    cell = pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(1, 11)],
            "patient": ["p1", "p1", "p1", "p1", "p1", "p2", "p2", "p2", "p2", "p2"],
            "time": ["t0", "t0", "t0", "t1", "t1", "t0", "t0", "t0", "t1", "t1"],
            "fov": ["f1", "f1", "f1", "f2", "f2", "f1", "f1", "f1", "f3", "f3"],
            "domain": ["TC", "TC", "TC", "IM", "IM", "TC", "TC", "TC", "IM", "IM"],
            "cell_type": ["a", "a", "b", "b", "c", "a", "b", "c", "b", "c"],
            "x": np.arange(10, dtype=float),
            "y": np.zeros(10, dtype=float),
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
        source="t0",
        target="t1",
        time_order=["t0", "t1"],
        relations=[("TC", "IM")],
        community_mode="fraction",
        n_states=3,
        k_neighbors=2,
    )
    adata.obs["state_id"] = np.asarray([0, 1, 1, 2, 2, 0, 0, 2, 1, 2], dtype=int)
    return adata


def _fov_observations(adata):
    return adata.uns["stride"]["fov_observations"]


def test_build_fov_observations_writes_canonical_slot() -> None:
    adata = _build_observation_adata()

    returned = build_fov_observations(adata)

    assert returned is adata
    payload = _fov_observations(adata)
    matrix = payload["community_composition"]
    metadata = payload["metadata"]
    assert matrix.shape == (4, 3)
    np.testing.assert_allclose(
        matrix,
        np.asarray(
            [
                [1.0 / 3.0, 2.0 / 3.0, 0.0],
                [0.0, 0.0, 1.0],
                [2.0 / 3.0, 0.0, 1.0 / 3.0],
                [0.0, 0.5, 0.5],
            ],
            dtype=float,
        ),
    )
    pd.testing.assert_frame_equal(
        metadata.reset_index(drop=True),
        pd.DataFrame(
            {
                "patient_id": ["p1", "p1", "p2", "p2"],
                "timepoint": ["t0", "t1", "t0", "t1"],
                "fov_id": ["f1", "f2", "f1", "f3"],
                "domain_label": ["TC", "IM", "TC", "IM"],
            }
        ),
    )


def test_build_fov_observations_uses_full_fov_identity() -> None:
    adata = _build_observation_adata()

    build_fov_observations(adata)

    payload = _fov_observations(adata)
    metadata = payload["metadata"].reset_index(drop=True)
    repeated_fov_rows = metadata.index[metadata["fov_id"] == "f1"].to_numpy()
    assert repeated_fov_rows.tolist() == [0, 2]
    np.testing.assert_allclose(
        payload["community_composition"][repeated_fov_rows],
        np.asarray(
            [
                [1.0 / 3.0, 2.0 / 3.0, 0.0],
                [2.0 / 3.0, 0.0, 1.0 / 3.0],
            ],
            dtype=float,
        ),
    )


def test_build_fov_observations_retains_domain_metadata() -> None:
    adata = _build_observation_adata()

    build_fov_observations(adata)

    payload = _fov_observations(adata)
    metadata = payload["metadata"].reset_index(drop=True)
    assert metadata["domain_label"].tolist() == ["TC", "IM", "TC", "IM"]
    assert payload["community_composition"].shape[1] == 3


def test_build_fov_observations_reuses_existing_valid_cache() -> None:
    adata = _build_observation_adata()
    build_fov_observations(adata)
    existing_matrix = _fov_observations(adata)["community_composition"].copy()
    existing_metadata = _fov_observations(adata)["metadata"].copy()

    with pytest.warns(
        UserWarning,
        match="adata.uns\\['stride'\\]\\['fov_observations'\\] already exists",
    ):
        returned = build_fov_observations(adata)

    assert returned is adata
    np.testing.assert_array_equal(
        _fov_observations(adata)["community_composition"],
        existing_matrix,
    )
    pd.testing.assert_frame_equal(
        _fov_observations(adata)["metadata"],
        existing_metadata,
    )


def test_build_fov_observations_rejects_stale_existing_cache() -> None:
    adata = _build_observation_adata()
    build_fov_observations(adata)
    adata.obs["state_id"] = np.asarray([0, 0, 1, 2, 2, 0, 0, 2, 1, 2], dtype=int)

    with pytest.raises(ContractError, match="does not match current state aggregation"):
        build_fov_observations(adata)


def test_build_fov_observations_rejects_invalid_state_id() -> None:
    adata = _build_observation_adata()
    adata.obs["state_id"] = np.asarray([0, 1, 1.5, 2, 2, 0, 0, 2, 1, 2])

    with pytest.raises(ContractError, match="integer-compatible"):
        build_fov_observations(adata)

    adata = _build_observation_adata()
    adata.obs["state_id"] = np.asarray([0, 1, 3, 2, 2, 0, 0, 2, 1, 2], dtype=int)

    with pytest.raises(ContractError, match="values must be in \\[0, n_states - 1\\]"):
        build_fov_observations(adata)


def test_build_fov_observations_rejects_multi_domain_fov() -> None:
    adata = _build_observation_adata()
    adata.obs.loc[adata.obs.index[0], "domain_label"] = "IM"

    with pytest.raises(ContractError, match="must map to exactly one domain_label"):
        build_fov_observations(adata)


@pytest.mark.parametrize("column", ["patient_id", "timepoint", "fov_id", "domain_label"])
def test_build_fov_observations_rejects_missing_metadata(column: str) -> None:
    adata = _build_observation_adata()
    adata.obs.loc[adata.obs.index[0], column] = np.nan

    with pytest.raises(ContractError, match="metadata.*missing values"):
        build_fov_observations(adata)


@pytest.mark.parametrize("column", ["patient_id", "timepoint", "fov_id", "domain_label"])
def test_build_fov_observations_rejects_empty_metadata(column: str) -> None:
    adata = _build_observation_adata()
    adata.obs.loc[adata.obs.index[0], column] = "   "

    with pytest.raises(ContractError, match="empty string values"):
        build_fov_observations(adata)


def test_build_fov_observations_requires_fraction_mode() -> None:
    adata = _build_observation_adata()
    adata.uns["stride"]["config"]["community_mode"] = "density"

    with pytest.raises(ContractError, match="community_mode"):
        build_fov_observations(adata)
