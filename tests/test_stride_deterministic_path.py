from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stride.api.basis import BasisSpec
from stride.api.dataset import DatasetHandle
from stride.basis import build_local_state_features, learn_shared_state_axis
from stride.data import assemble_longitudinal_adata, validate_longitudinal_adata
from stride.data.longitudinal import CANONICAL_TIMEPOINT_KEY, TIMEPOINT_KEY_ALIASES
from stride.errors import ContractError


def _alias_cell_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p1", "p1"],
            "timepoint_id": ["t0", "t0", "t1", "t1"],
            "roi_id": ["f1", "f1", "f2", "f2"],
            "compartment": ["TC", "TC", "IM", "IM"],
            "cell_type": ["A", "B", "A", "B"],
            "x": [0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.0, 1.0, 1.0],
        }
    )


def test_dataset_handle_from_tables_normalizes_alias_columns() -> None:
    handle = DatasetHandle.from_tables(_alias_cell_table())
    validate_longitudinal_adata(handle.adata)

    assert set(handle.adata.obs.columns) >= {
        "patient_id",
        "timepoint",
        "fov_id",
        "domain_label",
        "cell_subtype_label",
    }
    assert "timepoint_id" not in handle.adata.obs.columns
    assert "roi_id" not in handle.adata.obs.columns
    assert "compartment" not in handle.adata.obs.columns
    assert "cell_type" not in handle.adata.obs.columns
    assert "roi_areas" not in handle.adata.uns
    assert handle.timepoint_key == "timepoint"
    assert handle.fov_key == "fov_id"
    assert handle.domain_key == "domain_label"
    assert handle.cell_subtype_key == "cell_subtype_label"


def test_longitudinal_contract_keeps_timepoint_canonical() -> None:
    assert CANONICAL_TIMEPOINT_KEY == "timepoint"
    assert TIMEPOINT_KEY_ALIASES == ("timepoint", "timepoint_id")


def test_assemble_longitudinal_adata_accepts_domain_metadata_from_fov_table() -> None:
    cell_table = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p1", "p1"],
            "timepoint": ["t0", "t0", "t1", "t1"],
            "fov_id": ["f1", "f1", "f2", "f2"],
            "cell_subtype_label": ["A", "B", "A", "B"],
            "x": [0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.0, 1.0, 1.0],
        }
    )
    fov_table = pd.DataFrame(
        {
            "patient_id": ["p1", "p1"],
            "timepoint": ["t0", "t1"],
            "fov_id": ["f1", "f2"],
            "compartment": ["TC", "IM"],
        }
    )

    adata = assemble_longitudinal_adata(cell_table, fov_table=fov_table)
    assert set(adata.obs["domain_label"].astype(str)) == {"TC", "IM"}


def test_assemble_longitudinal_adata_rejects_invalid_coordinates() -> None:
    cell_table = _alias_cell_table()
    cell_table.loc[0, "x"] = np.nan

    with pytest.raises(ContractError, match="coordinates contain NaN/Inf"):
        assemble_longitudinal_adata(cell_table)


def test_assemble_longitudinal_adata_rejects_metadata_only_empty_fovs() -> None:
    cell_table = pd.DataFrame(
        {
            "patient_id": ["p1", "p1"],
            "timepoint": ["t0", "t0"],
            "fov_id": ["f1", "f1"],
            "cell_subtype_label": ["A", "B"],
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
        }
    )
    fov_table = pd.DataFrame(
        {
            "patient_id": ["p1", "p1"],
            "timepoint": ["t0", "t0"],
            "fov_id": ["f1", "f2"],
            "domain_label": ["TC", "IM"],
        }
    )

    with pytest.raises(ContractError, match="metadata-only empty FOVs"):
        assemble_longitudinal_adata(cell_table, fov_table=fov_table)


def test_build_local_state_features_uses_knn_subtype_proportions() -> None:
    handle = DatasetHandle.from_tables(_alias_cell_table())

    features = build_local_state_features(handle.adata, k=1)

    np.testing.assert_allclose(
        features,
        np.asarray(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert handle.adata.uns["state_feature_metadata"]["subtype_labels"] == ["A", "B"]


def test_build_local_state_features_rejects_single_cell_fov() -> None:
    adata = assemble_longitudinal_adata(
        pd.DataFrame(
            {
                "patient_id": ["p1"],
                "timepoint": ["t0"],
                "fov_id": ["f1"],
                "domain_label": ["TC"],
                "cell_subtype_label": ["A"],
                "x": [0.0],
                "y": [0.0],
            }
        )
    )

    with pytest.raises(ContractError, match="must contain at least 2 cells"):
        build_local_state_features(adata, k=20)


def test_learn_shared_state_axis_fits_all_cell_features() -> None:
    handle = DatasetHandle.from_tables(_alias_cell_table())
    build_local_state_features(handle.adata, k=1)

    basis = learn_shared_state_axis(handle.adata, K=2, random_state=0)

    observed_centroids = sorted(np.round(np.asarray(basis.centroids), 6).tolist())
    assert observed_centroids == [[0.0, 1.0], [1.0, 0.0]]
    assert set(handle.adata.obs["state_id"].astype(int)) == {0, 1}
    assert np.asarray(handle.adata.uns["cost_matrix"]).shape == (2, 2)
    assert float(handle.adata.uns["cost_scale"]) > 0.0


def test_state_construction_is_tissue_agnostic_with_respect_to_domain_metadata() -> None:
    base_table = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p1", "p1", "p2", "p2", "p2", "p2"],
            "timepoint": ["t0", "t0", "t1", "t1", "t0", "t0", "t1", "t1"],
            "fov_id": ["f1", "f1", "f2", "f2", "f3", "f3", "f4", "f4"],
            "domain_label": ["TC", "TC", "IM", "IM", "TC", "TC", "IM", "IM"],
            "cell_subtype_label": ["A", "B", "A", "B", "B", "A", "B", "A"],
            "x": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        }
    )
    relabeled_table = base_table.copy()
    relabeled_table["domain_label"] = ["PT", "PT", "TC", "TC", "IM", "IM", "PT", "PT"]

    base_handle = DatasetHandle.from_tables(base_table)
    relabeled_handle = DatasetHandle.from_tables(relabeled_table)

    base_features = build_local_state_features(base_handle.adata, k=1)
    relabeled_features = build_local_state_features(relabeled_handle.adata, k=1)
    np.testing.assert_allclose(base_features, relabeled_features)

    base_basis = learn_shared_state_axis(base_handle.adata, K=2, random_state=0)
    relabeled_basis = learn_shared_state_axis(relabeled_handle.adata, K=2, random_state=0)

    np.testing.assert_allclose(base_basis.centroids, relabeled_basis.centroids)
    np.testing.assert_allclose(base_basis.cost_matrix, relabeled_basis.cost_matrix)
    np.testing.assert_array_equal(
        base_handle.adata.obs["state_id"].to_numpy(),
        relabeled_handle.adata.obs["state_id"].to_numpy(),
    )


def test_basis_spec_builds_uniform_observations_and_geometry() -> None:
    handle = DatasetHandle.from_tables(_alias_cell_table())
    spec = BasisSpec(K=2, k_neighbors=1, random_state=0, geometry_neighbors=1)

    basis = spec.fit(handle.adata)
    observations = spec.build_observations(handle.adata)
    geometry = spec.build_geometry(state_basis=basis)

    assert len(observations) == 2
    for observation in observations:
        np.testing.assert_allclose(observation.community_composition, np.asarray([0.5, 0.5]))
        assert observation.mass == 1.0
        assert observation.mass_mode == "uniform"
        assert observation.domain_label in {"TC", "IM"}

    np.testing.assert_allclose(geometry.distance_matrix, geometry.cost_matrix)
    np.testing.assert_allclose(geometry.cost_matrix, geometry.cost_matrix.T)
    np.testing.assert_allclose(geometry.adjacency_matrix, geometry.adjacency_matrix.T)
    assert geometry.adjacency_matrix.shape == (2, 2)
