from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from stride.tl import FitResult, RelationResult
from tasks.task_A.block1.functions.stride_fit import (
    fit_block1_family,
    require_block1_fit_ok,
    summarize_fit_status_for_manifest,
)
from tasks.task_A.config import TaskAOrderedPairFamilySpec
from tasks.task_A.workflows.stride_adapter import prepare_task_a_pair_adata


def _family() -> TaskAOrderedPairFamilySpec:
    return TaskAOrderedPairFamilySpec(
        name="TC-IM",
        source_domain="TC",
        target_domain="IM",
        claim_role="confirmatory",
        pair_types=("TC->IM",),
    )


def _stage0_adata() -> AnnData:
    obs = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p1", "p1"],
            "fov_id": ["tc", "tc", "im", "im"],
            "domain_label": ["TC", "TC", "IM", "IM"],
            "state_id": [0, 1, 0, 1],
        },
        index=["c1", "c2", "c3", "c4"],
    )
    adata = AnnData(X=np.ones((4, 1), dtype=float), obs=obs)
    adata.uns["state_centroids"] = np.asarray([[0.0], [1.0]], dtype=float)
    adata.uns["cost_matrix"] = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    adata.uns["cost_scale"] = 1.0
    return adata


def _fit_result() -> FitResult:
    relation = RelationResult(
        relation_id="TC_TC_to_IM_IM",
        patient_ids=("p1", "p2"),
        A=np.asarray(
            [
                [[0.8, 0.1], [0.2, 0.7]],
                [[0.7, 0.2], [0.1, 0.8]],
            ],
            dtype=float,
        ),
        d=np.asarray([[0.1, 0.1], [0.1, 0.1]], dtype=float),
        e=np.asarray([[0.05, 0.1], [0.1, 0.05]], dtype=float),
        support={},
    )
    return FitResult(
        relations={relation.relation_id: relation},
        relation_ids=(relation.relation_id,),
        source="TC",
        target="IM",
        n_states=2,
    )


def test_block1_wrapper_accepts_ann_data_and_returns_fit_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare_task_a_pair_adata(_stage0_adata(), _family())
    expected = _fit_result()

    monkeypatch.setattr(
        "tasks.task_A.block1.functions.stride_fit.fit",
        lambda adata, *, device: expected,
    )

    observed = fit_block1_family(prepared, family_spec=_family(), device="cpu")

    assert isinstance(observed, FitResult)
    assert observed is expected


def test_require_block1_fit_ok_accepts_realized_patient_arrays() -> None:
    require_block1_fit_ok(
        _fit_result(),
        pair_family="TC-IM",
        run_scope="full_cohort",
    )


def test_summary_fit_status_for_manifest_uses_stride_tl_surface() -> None:
    summary = summarize_fit_status_for_manifest(_fit_result(), pair_family="TC-IM")

    assert summary == {
        "pair_family": "TC-IM",
        "fit_surface": "stride.tl.fit",
        "relation_count": 1,
        "relation_ids": ["TC_TC_to_IM_IM"],
        "patient_count": 2,
        "k_states": 2,
        "warning_count": 0,
        "warnings": [],
        "relation_support": {"TC_TC_to_IM_IM": {}},
        "relation_warnings": {"TC_TC_to_IM_IM": []},
    }
    assert summary["fit_surface"] == "stride.tl.fit"
