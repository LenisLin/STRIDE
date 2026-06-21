from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stride.errors import ContractError
from stride.io import build_adata, read_h5ad, write_h5ad
from stride.io._validation import validate_raw_adata


def _cell_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": ["c1", "c2", "c3", "c4"],
            "patient": ["p1", "p1", "p1", "p1"],
            "time": ["pre", "pre", "post", "post"],
            "fov": ["f1", "f2", "f3", "f4"],
            "domain": ["TC", "PT", "IM", "PT"],
            "cell_type": ["a", "b", "a", "b"],
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 2.0, 3.0],
        }
    )


def _build_with_relations(relations: object, **kwargs: object):
    build_kwargs = {
        "X": np.ones((4, 2), dtype=float),
        "var": ["g1", "g2"],
        "cell": _cell_table(),
        "cell_id": "cell_id",
        "patient": "patient",
        "time": "time",
        "fov_id": "fov",
        "domain": "domain",
        "cell_type": "cell_type",
        "x": "x",
        "y": "y",
        "source": "pre",
        "target": "post",
        "time_order": ["pre", "post"],
        "relations": relations,
        "community_mode": "fraction",
        "n_states": 3,
        "k_neighbors": 2,
    }
    build_kwargs.update(kwargs)
    return build_adata(
        **build_kwargs,
    )


def test_build_adata_stores_declared_relation_domain_pairs() -> None:
    adata = _build_with_relations([("TC", "IM"), ("TC", "PT")])

    config = adata.uns["stride"]["config"]
    assert config["relations"].shape == (2, 2)
    assert config["relations"].tolist() == [["TC", "IM"], ["TC", "PT"]]
    assert config["relation_ids"] == [
        "pre_TC_to_post_IM",
        "pre_TC_to_post_PT",
    ]
    assert config["community_mode"] == "fraction"


def test_build_adata_rejects_density_community_mode() -> None:
    with pytest.raises(ContractError):
        _build_with_relations([("TC", "IM")], community_mode="density")


def test_fraction_community_mode_does_not_require_area() -> None:
    fov = pd.DataFrame(
        {
            "patient": ["p1", "p1", "p1", "p1"],
            "time": ["pre", "pre", "post", "post"],
            "fov": ["f1", "f2", "f3", "f4"],
            "domain": ["TC", "PT", "IM", "PT"],
        }
    )

    adata = _build_with_relations([("TC", "IM")], fov=fov)

    config = adata.uns["stride"]["config"]
    assert config["community_mode"] == "fraction"
    assert "area" not in adata.uns["stride"]["fov_metadata"].columns


def test_declared_relations_round_trip_through_h5ad(tmp_path) -> None:
    adata = _build_with_relations([("TC", "IM"), ("TC", "PT")])

    output_path = write_h5ad(adata, tmp_path / "stride.h5ad")
    loaded = read_h5ad(output_path)

    validate_raw_adata(loaded)
    config = loaded.uns["stride"]["config"]
    assert config["community_mode"] == "fraction"
    assert config["relations"].tolist() == [["TC", "IM"], ["TC", "PT"]]
    assert list(config["relation_ids"]) == [
        "pre_TC_to_post_IM",
        "pre_TC_to_post_PT",
    ]


@pytest.mark.parametrize(
    "relations",
    [
        [],
        "TC_to_IM",
        [("TC",)],
        [("", "IM")],
        [("TC", "IM"), ("TC", "IM")],
    ],
)
def test_build_adata_rejects_invalid_declared_relations(relations: object) -> None:
    with pytest.raises(ContractError):
        _build_with_relations(relations)


def test_build_adata_rejects_source_domain_without_fov_support() -> None:
    with pytest.raises(ContractError, match="source domain_label"):
        _build_with_relations([("IM", "PT")])


def test_build_adata_rejects_target_domain_without_fov_support() -> None:
    with pytest.raises(ContractError, match="target domain_label"):
        _build_with_relations([("TC", "TC")])


def test_build_adata_rejects_relation_batch_with_unsupported_domain() -> None:
    with pytest.raises(ContractError, match="relations\\[1\\] source domain_label"):
        _build_with_relations([("TC", "IM"), ("IM", "PT")])
