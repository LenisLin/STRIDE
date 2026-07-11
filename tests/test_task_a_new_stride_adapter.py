from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from stride.errors import ContractError
from stride.pp import validate_ready
from stride.tl import FitResult, RelationResult
from tasks.task_A.block1.functions import stride_fit as block1_stride_fit
from tasks.task_A.config import TaskAOrderedPairFamilySpec
from tasks.task_A.workflows.stride_adapter import (
    normalize_task_a_stage0_aliases,
    prepare_task_a_pair_adata,
)


def _family() -> TaskAOrderedPairFamilySpec:
    return TaskAOrderedPairFamilySpec(
        name="TC-IM",
        source_domain="TC",
        target_domain="IM",
        claim_role="confirmatory",
        pair_types=("TC->IM",),
    )


def _stage0_alias_adata() -> AnnData:
    obs = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p1", "p1", "p2", "p2", "p2", "p2"],
            "roi_id": ["p1_tc", "p1_tc", "p1_im", "p1_im", "p2_tc", "p2_tc", "p2_im", "p2_im"],
            "compartment": ["TC", "TC", "IM", "IM", "TC", "TC", "IM", "IM"],
            "proto_id": [0, 1, 0, 1, 0, 0, 1, 1],
        },
        index=[f"c{i}" for i in range(8)],
    )
    adata = AnnData(X=np.ones((obs.shape[0], 1), dtype=float), obs=obs)
    adata.obsm["community_features"] = np.eye(2, dtype=float)[obs["proto_id"].to_numpy()]
    adata.uns["prototype_centroids"] = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    adata.uns["cost_matrix"] = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    adata.uns["s_C"] = 1.0
    return adata


def test_normalize_task_a_stage0_aliases_does_not_overwrite_canonical_keys() -> None:
    adata = _stage0_alias_adata()
    adata.obs["state_id"] = [1, 1, 1, 1, 0, 0, 0, 0]
    adata.uns["state_centroids"] = np.asarray([[2.0, 0.0], [3.0, 0.0]], dtype=float)
    adata.obsm["local_state_features"] = np.ones((adata.n_obs, 2), dtype=float)
    adata.uns["cost_scale"] = 2.0

    normalized = normalize_task_a_stage0_aliases(adata, copy_adata=True)

    assert normalized is not adata
    assert normalized.obs["state_id"].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    np.testing.assert_allclose(normalized.uns["state_centroids"], [[2.0, 0.0], [3.0, 0.0]])
    np.testing.assert_allclose(normalized.obsm["local_state_features"], np.ones((adata.n_obs, 2)))
    assert normalized.uns["cost_scale"] == 2.0


def test_prepare_task_a_pair_adata_writes_config_and_validates_ready() -> None:
    prepared = prepare_task_a_pair_adata(_stage0_alias_adata(), _family())

    config = prepared.uns["stride"]["config"]
    assert config["source"] == "TC"
    assert config["target"] == "IM"
    assert config["relations"].tolist() == [["TC", "IM"]]
    assert config["relation_ids"] == ["TC_TC_to_IM_IM"]
    validate_ready(prepared)


def test_prepare_task_a_pair_adata_can_subset_patients() -> None:
    prepared = prepare_task_a_pair_adata(
        _stage0_alias_adata(),
        _family(),
        patient_ids=("p2",),
    )

    metadata = prepared.uns["stride"]["fov_observations"]["metadata"]
    assert tuple(metadata["patient_id"].unique()) == ("p2",)
    validate_ready(prepared)


def test_prepare_task_a_pair_adata_preserves_requested_patient_order() -> None:
    prepared = prepare_task_a_pair_adata(
        _stage0_alias_adata(),
        _family(),
        patient_ids=("p2", "p1"),
    )

    assert tuple(dict.fromkeys(prepared.obs["patient_id"].tolist())) == ("p2", "p1")
    validate_ready(prepared)


@pytest.mark.parametrize("patient_ids", [(), ("missing",)])
def test_prepare_task_a_pair_adata_rejects_invalid_patient_selectors(patient_ids) -> None:
    with pytest.raises(ContractError):
        prepare_task_a_pair_adata(
            _stage0_alias_adata(),
            _family(),
            patient_ids=patient_ids,
        )


def test_fit_block1_family_calls_stride_tl_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = prepare_task_a_pair_adata(_stage0_alias_adata(), _family())
    calls: list[object] = []
    relation = RelationResult(
        relation_id="TC_TC_to_IM_IM",
        patient_ids=("p1",),
        A=np.asarray([[[0.9, 0.0], [0.0, 0.9]]], dtype=float),
        d=np.asarray([[0.1, 0.1]], dtype=float),
        e=np.asarray([[0.05, 0.05]], dtype=float),
        support={},
    )
    expected = FitResult(
        relations={relation.relation_id: relation},
        relation_ids=(relation.relation_id,),
        source="TC",
        target="IM",
        n_states=2,
    )

    def fake_fit(adata: AnnData, *, device: object) -> FitResult:
        calls.append(device)
        assert adata is prepared
        return expected

    monkeypatch.setattr(block1_stride_fit, "fit", fake_fit)

    observed = block1_stride_fit.fit_block1_family(
        prepared,
        family_spec=_family(),
        device="cpu",
    )

    assert observed is expected
    assert calls == ["cpu"]
