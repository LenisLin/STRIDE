from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from stride.errors import ContractError
from stride.tl import FitResult, RelationResult
from tasks.task_A.block0.functions.schemas import FIT_LABEL_NULL, FIT_LABEL_REAL
from tasks.task_A.block0.functions.stride_fit import extract_block0_fit_records
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


def _prepared_adata() -> AnnData:
    obs = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p1", "p1", "p2", "p2", "p2", "p2"],
            "fov_id": ["p1_tc", "p1_tc", "p1_im", "p1_im", "p2_tc", "p2_tc", "p2_im", "p2_im"],
            "domain_label": ["TC", "TC", "IM", "IM", "TC", "TC", "IM", "IM"],
            "state_id": [0, 1, 0, 1, 0, 0, 1, 1],
        },
        index=[f"c{i}" for i in range(8)],
    )
    adata = AnnData(X=np.ones((8, 1), dtype=float), obs=obs)
    adata.uns["state_centroids"] = np.asarray([[0.0], [1.0]], dtype=float)
    adata.uns["cost_matrix"] = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    adata.uns["cost_scale"] = 1.0
    return prepare_task_a_pair_adata(adata, _family())


def _fit_result() -> FitResult:
    relation = RelationResult(
        relation_id="TC_TC_to_IM_IM",
        patient_ids=("p2", "p1"),
        A=np.asarray(
            [
                [[0.7, 0.2], [0.1, 0.8]],
                [[0.8, 0.1], [0.2, 0.7]],
            ],
            dtype=float,
        ),
        d=np.asarray([[0.1, 0.1], [0.1, 0.1]], dtype=float),
        e=np.asarray([[0.05, 0.1], [0.1, 0.05]], dtype=float),
        support={"support_counts": {"p1": {}, "p2": {}}},
    )
    return FitResult(
        relations={relation.relation_id: relation},
        relation_ids=(relation.relation_id,),
        source="TC",
        target="IM",
        n_states=2,
    )


def test_block0_records_preserve_fit_patient_axis_and_composition_weights() -> None:
    records = extract_block0_fit_records(
        _fit_result(),
        source_adata=_prepared_adata(),
        fit_label=FIT_LABEL_REAL,
    )

    assert tuple(record.patient_id for record in records) == ("p2", "p1")
    np.testing.assert_allclose(records[0].A, _fit_result().relations["TC_TC_to_IM_IM"].A[0])
    np.testing.assert_allclose(records[0].source_burden, [1.0, 0.0])
    np.testing.assert_allclose(records[0].e_weights, [0.0, 1.0])
    np.testing.assert_allclose(records[1].source_burden, [0.5, 0.5])
    np.testing.assert_allclose(records[1].e_weights, [0.5, 0.5])


def test_block0_null_records_require_permutation_index() -> None:
    with pytest.raises(ContractError, match="require permutation_index"):
        extract_block0_fit_records(
            _fit_result(),
            source_adata=_prepared_adata(),
            fit_label=FIT_LABEL_NULL,
        )
