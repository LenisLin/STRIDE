from __future__ import annotations

import numpy as np
import pytest

import stride
import stride.da as da
from stride.errors import ContractError
from stride.tl import FitResult, RelationResult


def _relation(
    relation_id: str = "pre_TC_to_post_IM",
    *,
    patient_ids: tuple[str, ...] = ("p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"),
) -> RelationResult:
    A = np.asarray(
        [
            [[0.80, 0.10], [0.10, 0.80]],
            [[0.70, 0.20], [0.20, 0.70]],
            [[0.75, 0.15], [0.15, 0.75]],
            [[0.20, 0.70], [0.70, 0.20]],
            [[0.10, 0.80], [0.80, 0.10]],
            [[0.15, 0.75], [0.75, 0.15]],
            [[0.45, 0.45], [0.45, 0.45]],
            [[0.40, 0.50], [0.50, 0.40]],
            [[0.50, 0.40], [0.40, 0.50]],
        ],
        dtype=float,
    )
    d = np.asarray(
        [
            [0.10, 0.10],
            [0.10, 0.10],
            [0.10, 0.10],
            [0.10, 0.10],
            [0.10, 0.10],
            [0.10, 0.10],
            [0.10, 0.10],
            [0.10, 0.10],
            [0.10, 0.10],
        ],
        dtype=float,
    )
    e = np.asarray(
        [
            [0.05, 0.20],
            [0.10, 0.25],
            [0.08, 0.22],
            [0.30, 0.40],
            [0.35, 0.45],
            [0.32, 0.42],
            [0.20, 0.30],
            [0.22, 0.32],
            [0.18, 0.28],
        ],
        dtype=float,
    )
    n_patients = len(patient_ids)
    return RelationResult(
        relation_id=relation_id,
        patient_ids=patient_ids,
        A=A[:n_patients],
        d=d[:n_patients],
        e=e[:n_patients],
        support={},
    )


def _fit() -> FitResult:
    first = _relation("pre_TC_to_post_IM")
    second = _relation("pre_PT_to_post_IM")
    return FitResult(
        relations={first.relation_id: first, second.relation_id: second},
        relation_ids=(first.relation_id, second.relation_id),
        source="pre",
        target="post",
        n_states=2,
    )


def test_da_functions_are_namespace_only() -> None:
    assert "patient_relation_arrays" in da.__all__
    assert "augmented_entry_group_association" in da.__all__
    assert "patient_relation_arrays" not in stride.__all__
    assert "augmented_entry_group_association" not in stride.__all__


def test_patient_relation_arrays_extracts_grouped_copies_from_relation_result() -> None:
    groups = {
        "p1": "control",
        "p2": "control",
        "p3": "control",
        "p4": "treated",
        "p5": "treated",
        "p6": "treated",
        "p7": "context",
        "p8": "context",
        "p9": "context",
    }

    extracted = da.patient_relation_arrays(_relation(), group_labels=groups)

    assert tuple(extracted) == ("pre_TC_to_post_IM",)
    assert tuple(extracted["pre_TC_to_post_IM"]) == ("control", "treated", "context")
    control = extracted["pre_TC_to_post_IM"]["control"]
    assert control["relation_id"] == "pre_TC_to_post_IM"
    assert control["group_id"] == "control"
    assert control["patient_ids"] == ("p1", "p2", "p3")
    np.testing.assert_allclose(control["A"], _relation().A[:3])
    np.testing.assert_allclose(control["d"], _relation().d[:3])
    np.testing.assert_allclose(control["e"], _relation().e[:3])
    assert control["A"] is not _relation().A


def test_patient_relation_arrays_extracts_fit_result_in_declared_order() -> None:
    extracted = da.patient_relation_arrays(_fit(), relation_ids=["pre_PT_to_post_IM"])

    assert tuple(extracted) == ("pre_PT_to_post_IM",)
    assert tuple(extracted["pre_PT_to_post_IM"]) == ("all",)
    entry = extracted["pre_PT_to_post_IM"]["all"]
    assert entry["patient_ids"] == ("p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9")
    assert entry["group_id"] == "all"


def test_patient_relation_arrays_rejects_missing_group_label() -> None:
    with pytest.raises(ContractError, match="group_labels"):
        da.patient_relation_arrays(_relation(), group_labels={"p1": "control"})


def test_augmented_entry_group_association_tests_native_entries_and_corrects_by_relation_comparison() -> None:
    groups = {
        "p1": "low",
        "p2": "low",
        "p3": "low",
        "p4": "high",
        "p5": "high",
        "p6": "high",
        "p7": "context",
        "p8": "context",
        "p9": "context",
    }
    patient_arrays = da.patient_relation_arrays(_relation(), group_labels=groups)

    stats = da.augmented_entry_group_association(
        patient_arrays,
        comparisons=(
            {
                "comparison_id": "low_vs_high",
                "groups": ("low", "high"),
            },
        ),
    )

    assert set(stats["entry_type"]) == {"A", "d", "e"}
    assert len(stats) == 8
    assert not ((stats["row_id"] == 2) & (stats["col_id"] == 2)).any()
    assert set(stats["correction_scope"]) == {"pre_TC_to_post_IM:low_vs_high"}
    assert set(stats["correction_method"]) == {"BH"}

    row = stats[
        (stats["entry_type"] == "A") & (stats["row_id"] == 0) & (stats["col_id"] == 0)
    ].iloc[0]
    assert row["comparison_type"] == "two_group"
    assert row["group_1"] == "low"
    assert row["group_2"] == "high"
    assert row["effect_size_type"] == "cliffs_delta"
    assert row["effect_size"] > 0
    assert np.isfinite(row["p_value"])
    assert np.isfinite(row["q_value"])


def test_augmented_entry_group_association_uses_anova_for_multiple_groups() -> None:
    groups = {
        "p1": "a",
        "p2": "a",
        "p3": "a",
        "p4": "b",
        "p5": "b",
        "p6": "b",
        "p7": "c",
        "p8": "c",
        "p9": "c",
    }
    patient_arrays = da.patient_relation_arrays(_relation(), group_labels=groups)

    stats = da.augmented_entry_group_association(
        patient_arrays,
        comparisons=(
            {
                "comparison_id": "three_groups",
                "groups": ("a", "b", "c"),
            },
        ),
    )

    assert set(stats["comparison_type"]) == {"multi_group"}
    assert set(stats["test_name"]) == {"one_way_anova"}
    assert set(stats["effect_size_type"]) == {"eta_squared"}


def test_augmented_entry_group_association_rejects_unknown_group() -> None:
    patient_arrays = da.patient_relation_arrays(
        _relation(),
        group_labels={
            "p1": "low",
            "p2": "low",
            "p3": "low",
            "p4": "high",
            "p5": "high",
            "p6": "high",
            "p7": "context",
            "p8": "context",
            "p9": "context",
        },
    )

    with pytest.raises(ContractError, match="unknown group"):
        da.augmented_entry_group_association(
            patient_arrays,
            comparisons=({"comparison_id": "bad", "groups": ("low", "missing")},),
        )


def test_augmented_entry_group_association_rejects_groups_with_fewer_than_three_patients() -> None:
    patient_arrays = da.patient_relation_arrays(
        _relation(patient_ids=("p1", "p2", "p3", "p4")),
        group_labels={"p1": "low", "p2": "low", "p3": "high", "p4": "high"},
    )

    with pytest.raises(ContractError, match="at least 3 patients"):
        da.augmented_entry_group_association(
            patient_arrays,
            comparisons=({"comparison_id": "low_vs_high", "groups": ("low", "high")},),
        )
