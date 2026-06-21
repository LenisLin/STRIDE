from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import stride.io as stride_io
from stride.errors import ContractError
from stride.tl import CohortResult, FitResult, RelationResult


def _adata() -> AnnData:
    obs = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p2"],
            "timepoint": ["pre", "post", "pre", "post"],
            "fov_id": ["f1", "f2", "f3", "f4"],
            "domain_label": ["colon", "colon", "liver", "liver"],
            "cell_subtype_label": ["T", "B", "T", "M"],
            "state_id": [0, 1, 0, 1],
        },
        index=["c1", "c2", "c3", "c4"],
    )
    adata = AnnData(X=np.ones((4, 1), dtype=float), obs=obs)
    adata.uns["stride"] = {
        "config": {"n_states": 2},
        "fov_observations": {
            "community_composition": np.asarray(
                [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8]],
                dtype=float,
            ),
            "metadata": pd.DataFrame(
                {
                    "patient_id": ["p1", "p1", "p2", "p2"],
                    "timepoint": ["pre", "post", "pre", "post"],
                    "fov_id": ["f1", "f2", "f3", "f4"],
                    "domain_label": ["colon", "colon", "liver", "liver"],
                }
            ),
        },
    }
    return adata


def _cohort(relation_id: str) -> CohortResult:
    return CohortResult(
        relation_id=relation_id,
        patient_ids=("p1", "p2"),
        template_A=np.asarray([[0.7, 0.2], [0.1, 0.8]], dtype=float),
        template_d=np.asarray([0.1, 0.1], dtype=float),
        template_e=np.asarray([0.2, 0.3], dtype=float),
        support_n_patients=2,
        dispersion=0.04,
    )


def _fit() -> FitResult:
    relations = {}
    for relation_id in ("r1", "r2"):
        cohort = _cohort(relation_id)
        relations[relation_id] = RelationResult(
            relation_id=relation_id,
            patient_ids=cohort.patient_ids,
            A=np.stack([cohort.template_A, cohort.template_A]),
            d=np.stack([cohort.template_d, cohort.template_d]),
            e=np.stack([cohort.template_e, cohort.template_e]),
            support={},
            cohort=cohort,
        )
    return FitResult(
        relations=relations,
        relation_ids=("r1", "r2"),
        source="pre",
        target="post",
        n_states=2,
    )


def test_write_r_handover_writes_csv_and_validates_primary_key(tmp_path) -> None:
    table = pd.DataFrame({"id": ["a"], "value": [1.0]})

    path = stride_io.write_r_handover(table, tmp_path, "handover.csv", primary_key="id")

    assert path.exists()
    pd.testing.assert_frame_equal(pd.read_csv(path), table)
    with pytest.raises(ContractError, match="primary_key"):
        stride_io.write_r_handover(table, tmp_path, "bad.csv", primary_key="missing")


def test_write_descriptive_tables_outputs_two_all_info_csvs(tmp_path) -> None:
    paths = stride_io.write_descriptive_tables(
        _adata(),
        tmp_path,
        state_labels=("C0", "C1"),
        patient_groups={"p1": "A", "p2": "B"},
    )

    assert [path.name for path in paths] == ["community_annotation.csv", "fov_composition.csv"]
    annotation = pd.read_csv(paths[0])
    composition = pd.read_csv(paths[1])
    assert set(annotation.columns) == {
        "community_id",
        "community_label",
        "cell_subtype",
        "cell_fraction",
        "dominant_domain",
        "dominant_timepoint",
        "patient_prevalence",
        "fov_prevalence",
    }
    assert set(composition.columns) == {
        "fov_index",
        "fov_id",
        "patient_id",
        "timepoint",
        "domain_label",
        "group",
        "community_id",
        "community_label",
        "fraction",
    }
    assert composition.shape[0] == 8


def test_write_fraction_table_keeps_all_communities(tmp_path) -> None:
    table = pd.DataFrame(
        {
            "scale": ["fov", "fov"],
            "relation_id": ["r1", "r1"],
            "group": ["A", "A"],
            "patient_id": ["p1", "p1"],
            "side": ["source", "source"],
            "community_id": [0, 1],
            "community_label": ["C0", "C1"],
            "fraction": [0.2, 0.8],
            "p_value": [0.5, 0.01],
            "q_value": [0.5, 0.02],
            "test_name": ["wilcoxon", "wilcoxon"],
            "correction_method": ["BH", "BH"],
        }
    )

    path = stride_io.write_fraction_table(table, tmp_path)

    exported = pd.read_csv(path)
    assert exported["community_id"].tolist() == [0, 1]


def test_write_cohort_table_exports_all_fit_relations(tmp_path) -> None:
    path = stride_io.write_cohort_table(_fit(), tmp_path, state_labels=("C0", "C1"))

    table = pd.read_csv(path)
    assert set(table["relation_id"]) == {"r1", "r2"}
    assert table.groupby("relation_id").size().to_dict() == {"r1": 9, "r2": 9}
    assert {"A", "d", "e", "masked"} == set(table["entry_type"])


def test_write_program_score_table_merges_supplied_stats(tmp_path) -> None:
    scores = pd.DataFrame(
        {
            "relation_id": ["r1", "r1"],
            "program_id": ["prog1", "prog1"],
            "patient_id": ["p1", "p2"],
            "group_id": ["A", "B"],
            "program_component_score": [0.2, 0.8],
        }
    )
    stats = pd.DataFrame(
        {
            "relation_id": ["r1"],
            "program_id": ["prog1"],
            "comparison_id": ["all_groups"],
            "comparison_type": ["multi_group"],
            "test_name": ["one_way_anova"],
            "effect_size": [0.7],
            "effect_size_type": ["eta_squared"],
            "p_value": [0.01],
            "q_value": [0.02],
        }
    )

    path = stride_io.write_program_score_table(scores, tmp_path, program_stats=stats)

    table = pd.read_csv(path)
    assert table["comparison_id"].tolist() == ["all_groups", "all_groups"]
    assert "program_component_score" in table
