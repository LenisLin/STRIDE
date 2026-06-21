from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stride
import stride.da as da
from stride.da._stats import finite_p_value
from stride.errors import ContractError


def _patient_arrays() -> dict[str, dict[str, dict[str, object]]]:
    A_low = np.asarray(
        [
            [[0.82, 0.08], [0.12, 0.78]],
            [[0.76, 0.14], [0.18, 0.72]],
            [[0.79, 0.11], [0.15, 0.75]],
        ],
        dtype=float,
    )
    A_high = np.asarray(
        [
            [[0.24, 0.66], [0.68, 0.22]],
            [[0.18, 0.72], [0.75, 0.15]],
            [[0.21, 0.69], [0.71, 0.19]],
        ],
        dtype=float,
    )
    d_low = np.full((3, 2), 0.10, dtype=float)
    d_high = np.full((3, 2), 0.10, dtype=float)
    e_low = np.asarray([[0.05, 0.08], [0.06, 0.09], [0.07, 0.10]], dtype=float)
    e_high = np.asarray([[0.25, 0.28], [0.26, 0.29], [0.27, 0.30]], dtype=float)
    return {
        "pre_TC_to_post_IM": {
            "low": {
                "relation_id": "pre_TC_to_post_IM",
                "group_id": "low",
                "patient_ids": ("p1", "p2", "p3"),
                "A": A_low,
                "d": d_low,
                "e": e_low,
            },
            "high": {
                "relation_id": "pre_TC_to_post_IM",
                "group_id": "high",
                "patient_ids": ("p4", "p5", "p6"),
                "A": A_high,
                "d": d_high,
                "e": e_high,
            },
        }
    }


def test_relation_program_functions_are_da_namespace_only() -> None:
    assert "relation_program_rank_diagnostics" in da.__all__
    assert "relation_program_decomposition" in da.__all__
    assert "relation_program_group_association" in da.__all__
    assert "relation_program_rank_diagnostics" not in stride.__all__
    assert "relation_program_decomposition" not in stride.__all__
    assert "relation_program_group_association" not in stride.__all__


def test_relation_program_rank_diagnostics_reports_rank_restart_error_status_metadata() -> None:
    diagnostics = da.relation_program_rank_diagnostics(
        _patient_arrays(),
        rank_grid=[(1, 1, 1), (2, 2, 2)],
        n_restarts=2,
        random_state=7,
    )

    assert list(diagnostics.columns) == [
        "relation_id",
        "rank_patient",
        "rank_source",
        "rank_target_open",
        "restart_id",
        "random_seed",
        "reconstruction_error",
        "relative_error",
        "status",
        "error_message",
        "selected",
    ]
    assert len(diagnostics) == 4
    assert set(diagnostics["relation_id"]) == {"pre_TC_to_post_IM"}
    assert set(diagnostics["restart_id"]) == {0, 1}
    assert set(diagnostics["status"]) == {"ok"}
    assert diagnostics["random_seed"].nunique() == 4
    assert np.isfinite(diagnostics["reconstruction_error"]).all()
    assert np.isfinite(diagnostics["relative_error"]).all()
    assert (diagnostics["relative_error"] >= 0.0).all()
    assert set(diagnostics["selected"]) == {False}


def test_relation_program_rank_diagnostics_reports_absolute_and_relative_errors(monkeypatch) -> None:
    import tensorly
    import tensorly.decomposition

    class FakeTucker:
        def __init__(self) -> None:
            self.core = np.asarray([[[1.0]]], dtype=float)
            self.factors = (
                np.ones((6, 1), dtype=float),
                np.ones((2, 1), dtype=float),
                np.ones((3, 1), dtype=float),
            )

    tensor = np.asarray(
        [
            [[0.82, 0.08, 0.10], [0.12, 0.78, 0.10]],
            [[0.76, 0.14, 0.10], [0.18, 0.72, 0.10]],
            [[0.79, 0.11, 0.10], [0.15, 0.75, 0.10]],
            [[0.24, 0.66, 0.10], [0.68, 0.22, 0.10]],
            [[0.18, 0.72, 0.10], [0.75, 0.15, 0.10]],
            [[0.21, 0.69, 0.10], [0.71, 0.19, 0.10]],
        ],
        dtype=float,
    )
    tensor_norm = float(np.linalg.norm(tensor))

    def fake_non_negative_tucker_hals(*args, **kwargs):
        return FakeTucker(), [0.25]

    def fake_tucker_to_tensor(*args, **kwargs):
        reconstructed = tensor.copy()
        reconstructed[0, 0, 0] -= 0.25 * tensor_norm
        return reconstructed

    monkeypatch.setattr(tensorly.decomposition, "non_negative_tucker_hals", fake_non_negative_tucker_hals)
    monkeypatch.setattr(tensorly, "tucker_to_tensor", fake_tucker_to_tensor)

    diagnostics = da.relation_program_rank_diagnostics(
        _patient_arrays(),
        rank_grid=[(1, 1, 1)],
        n_restarts=1,
        random_state=7,
    )

    row = diagnostics.iloc[0]
    assert row["status"] == "ok"
    assert row["reconstruction_error"] == pytest.approx(0.25 * tensor_norm)
    assert row["relative_error"] == pytest.approx(0.25)


def test_relation_program_rank_diagnostics_rejects_invalid_rank_negative_input_and_unknown_relation() -> None:
    with pytest.raises(ContractError, match="rank"):
        da.relation_program_rank_diagnostics(_patient_arrays(), rank_grid=[(0, 1, 1)])

    with pytest.raises(ContractError, match="integer"):
        da.relation_program_rank_diagnostics(_patient_arrays(), rank_grid=[(1.9, 1, 1)])  # type: ignore[list-item]

    negative = _patient_arrays()
    negative["pre_TC_to_post_IM"]["low"]["A"] = -np.ones((3, 2, 2), dtype=float)
    with pytest.raises(ContractError, match="nonnegative"):
        da.relation_program_rank_diagnostics(negative, rank_grid=[(1, 1, 1)])

    with pytest.raises(ContractError, match="unknown relation_id"):
        da.relation_program_rank_diagnostics(
            _patient_arrays(),
            rank_grid=[(1, 1, 1)],
            relation_ids=["missing"],
        )

    duplicated = _patient_arrays()
    duplicated["pre_TC_to_post_IM"]["high"]["patient_ids"] = ("p1", "p5", "p6")
    with pytest.raises(ContractError, match="duplicate patient_ids"):
        da.relation_program_rank_diagnostics(duplicated, rank_grid=[(1, 1, 1)])


def test_relation_program_decomposition_returns_all_tables_and_aligned_program_ids() -> None:
    result = da.relation_program_decomposition(
        _patient_arrays(),
        ranks=(2, 2, 2),
        n_restarts=2,
        random_state=11,
    )

    assert tuple(result) == (
        "rank_diagnostics",
        "patient_factors",
        "source_factors",
        "target_open_factors",
        "core",
        "program_components",
        "program_entries",
        "patient_program_scores",
    )
    assert len(result["rank_diagnostics"]) == 2
    assert set(result["rank_diagnostics"]["status"]) == {"ok"}
    assert result["rank_diagnostics"]["selected"].sum() == 1
    assert len(result["patient_factors"]) == 12
    assert set(result["patient_factors"]["patient_id"]) == {"p1", "p2", "p3", "p4", "p5", "p6"}
    assert len(result["source_factors"]) == 4
    assert len(result["target_open_factors"]) == 6
    assert len(result["core"]) == 8
    assert len(result["program_components"]) == 8
    assert set(result["program_components"]["program_id"]) == set(result["program_entries"]["program_id"])
    assert set(result["program_components"]["program_id"]) == set(
        result["patient_program_scores"]["program_id"]
    )
    assert set(result["program_components"]["program_weight_rank"]) == set(range(1, 9))
    assert (result["program_components"]["core_weight"].diff().dropna() <= 0.0).all()


def test_relation_program_decomposition_normalizes_factor_scale(monkeypatch) -> None:
    import tensorly
    import tensorly.decomposition

    tensor = np.asarray(
        [
            [[0.82, 0.08, 0.10], [0.12, 0.78, 0.10]],
            [[0.76, 0.14, 0.10], [0.18, 0.72, 0.10]],
            [[0.79, 0.11, 0.10], [0.15, 0.75, 0.10]],
            [[0.24, 0.66, 0.10], [0.68, 0.22, 0.10]],
            [[0.18, 0.72, 0.10], [0.75, 0.15, 0.10]],
            [[0.21, 0.69, 0.10], [0.71, 0.19, 0.10]],
        ],
        dtype=float,
    )

    class FakeTucker:
        def __init__(self) -> None:
            self.core = np.asarray([[[2.0]]], dtype=float)
            self.factors = (
                np.asarray([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], dtype=float),
                np.asarray([[2.0], [6.0]], dtype=float),
                np.asarray([[3.0], [3.0], [6.0]], dtype=float),
            )

    def fake_non_negative_tucker_hals(*args, **kwargs):
        return FakeTucker(), [0.0]

    def fake_tucker_to_tensor(*args, **kwargs):
        return tensor

    monkeypatch.setattr(tensorly.decomposition, "non_negative_tucker_hals", fake_non_negative_tucker_hals)
    monkeypatch.setattr(tensorly, "tucker_to_tensor", fake_tucker_to_tensor)

    result = da.relation_program_decomposition(
        _patient_arrays(),
        ranks=(1, 1, 1),
        n_restarts=1,
        random_state=17,
    )

    assert result["program_components"].iloc[0]["core_weight"] == pytest.approx(
        2.0 * 21.0 * 8.0 * 12.0
    )
    assert result["source_factors"]["loading"].sum() == pytest.approx(1.0)
    assert result["target_open_factors"]["loading"].sum() == pytest.approx(1.0)
    assert result["patient_factors"]["loading"].sum() == pytest.approx(1.0)
    first_score = result["patient_program_scores"].iloc[0]["program_component_score"]
    assert first_score == pytest.approx((1.0 / 21.0) * 2.0 * 21.0 * 8.0 * 12.0)


def test_relation_program_decomposition_marks_target_open_axis_and_patient_groups() -> None:
    result = da.relation_program_decomposition(
        _patient_arrays(),
        ranks=(1, 2, 2),
        n_restarts=1,
        random_state=13,
    )

    target_open_factors = result["target_open_factors"]
    assert set(target_open_factors["target_open_axis_id"]) == {0, 1, 2}
    assert set(
        target_open_factors.loc[
            target_open_factors["target_open_axis_id"] < 2,
            "target_open_axis_type",
        ]
    ) == {"target_community"}
    assert set(
        target_open_factors.loc[
            target_open_factors["target_open_axis_id"] == 2,
            "target_open_axis_type",
        ]
    ) == {"source_open"}

    program_entries = result["program_entries"]
    assert {"target_community", "source_open"} <= set(program_entries["target_open_axis_type"])
    score_groups = result["patient_program_scores"].set_index("patient_id")["group_id"].to_dict()
    assert score_groups["p1"] == "low"
    assert score_groups["p6"] == "high"


def test_relation_program_group_association_uses_scores_and_corrects_by_relation_comparison() -> None:
    scores = pd.DataFrame(
        [
            {
                "relation_id": "r1",
                "patient_id": "p1",
                "group_id": "low",
                "program_id": "a",
                "program_component_score": 0.1,
            },
            {
                "relation_id": "r1",
                "patient_id": "p2",
                "group_id": "low",
                "program_id": "a",
                "program_component_score": 0.2,
            },
            {
                "relation_id": "r1",
                "patient_id": "p3",
                "group_id": "low",
                "program_id": "a",
                "program_component_score": 0.3,
            },
            {
                "relation_id": "r1",
                "patient_id": "p4",
                "group_id": "high",
                "program_id": "a",
                "program_component_score": 0.8,
            },
            {
                "relation_id": "r1",
                "patient_id": "p5",
                "group_id": "high",
                "program_id": "a",
                "program_component_score": 0.9,
            },
            {
                "relation_id": "r1",
                "patient_id": "p6",
                "group_id": "high",
                "program_id": "a",
                "program_component_score": 1.0,
            },
            {
                "relation_id": "r1",
                "patient_id": "p1",
                "group_id": "low",
                "program_id": "b",
                "program_component_score": 1.0,
            },
            {
                "relation_id": "r1",
                "patient_id": "p2",
                "group_id": "low",
                "program_id": "b",
                "program_component_score": 1.0,
            },
            {
                "relation_id": "r1",
                "patient_id": "p3",
                "group_id": "low",
                "program_id": "b",
                "program_component_score": 1.0,
            },
            {
                "relation_id": "r1",
                "patient_id": "p4",
                "group_id": "high",
                "program_id": "b",
                "program_component_score": 1.0,
            },
            {
                "relation_id": "r1",
                "patient_id": "p5",
                "group_id": "high",
                "program_id": "b",
                "program_component_score": 1.0,
            },
            {
                "relation_id": "r1",
                "patient_id": "p6",
                "group_id": "high",
                "program_id": "b",
                "program_component_score": 1.0,
            },
        ]
    )

    stats = da.relation_program_group_association(
        scores,
        comparisons=({"comparison_id": "low_vs_high", "groups": ("low", "high")},),
    )

    assert len(stats) == 2
    assert set(stats["program_id"]) == {"a", "b"}
    assert set(stats["correction_scope"]) == {"r1:low_vs_high"}
    assert set(stats["correction_method"]) == {"BH"}
    row = stats[stats["program_id"] == "a"].iloc[0]
    assert row["comparison_type"] == "two_group"
    assert row["group_1"] == "low"
    assert row["group_2"] == "high"
    assert row["test_name"] == "mannwhitneyu"
    assert row["effect_size_type"] == "cliffs_delta"
    assert row["effect_size"] < 0.0
    assert np.isfinite(row["q_value"])


def test_relation_program_group_association_rejects_groups_with_fewer_than_three_patients() -> None:
    scores = pd.DataFrame(
        [
            {"relation_id": "r1", "patient_id": "p1", "group_id": "low", "program_id": "a", "program_component_score": 0.1},
            {"relation_id": "r1", "patient_id": "p2", "group_id": "low", "program_id": "a", "program_component_score": 0.2},
            {"relation_id": "r1", "patient_id": "p3", "group_id": "high", "program_id": "a", "program_component_score": 0.8},
            {"relation_id": "r1", "patient_id": "p4", "group_id": "high", "program_id": "a", "program_component_score": 0.9},
        ]
    )

    with pytest.raises(ContractError, match="at least 3 patients"):
        da.relation_program_group_association(
            scores,
            comparisons=({"comparison_id": "low_vs_high", "groups": ("low", "high")},),
        )


def test_relation_program_group_association_requires_three_patients_per_group_per_program() -> None:
    scores = pd.DataFrame(
        [
            {"relation_id": "r1", "patient_id": "p1", "group_id": "low", "program_id": "a", "program_component_score": 0.1},
            {"relation_id": "r1", "patient_id": "p2", "group_id": "low", "program_id": "a", "program_component_score": 0.2},
            {"relation_id": "r1", "patient_id": "p3", "group_id": "low", "program_id": "b", "program_component_score": 0.3},
            {"relation_id": "r1", "patient_id": "p4", "group_id": "high", "program_id": "a", "program_component_score": 0.8},
            {"relation_id": "r1", "patient_id": "p5", "group_id": "high", "program_id": "a", "program_component_score": 0.9},
            {"relation_id": "r1", "patient_id": "p6", "group_id": "high", "program_id": "a", "program_component_score": 1.0},
            {"relation_id": "r1", "patient_id": "p1", "group_id": "low", "program_id": "b", "program_component_score": 0.4},
            {"relation_id": "r1", "patient_id": "p2", "group_id": "low", "program_id": "b", "program_component_score": 0.5},
            {"relation_id": "r1", "patient_id": "p4", "group_id": "high", "program_id": "b", "program_component_score": 0.6},
            {"relation_id": "r1", "patient_id": "p5", "group_id": "high", "program_id": "b", "program_component_score": 0.7},
            {"relation_id": "r1", "patient_id": "p6", "group_id": "high", "program_id": "b", "program_component_score": 0.8},
        ]
    )

    with pytest.raises(ContractError, match="program_id 'a'.*low n=2"):
        da.relation_program_group_association(
            scores,
            comparisons=({"comparison_id": "low_vs_high", "groups": ("low", "high")},),
        )


def test_relation_program_group_association_rejects_string_group_sequence() -> None:
    scores = pd.DataFrame(
        [
            {"relation_id": "r1", "patient_id": "p1", "group_id": "A", "program_id": "a", "program_component_score": 0.1},
            {"relation_id": "r1", "patient_id": "p2", "group_id": "A", "program_id": "a", "program_component_score": 0.2},
            {"relation_id": "r1", "patient_id": "p3", "group_id": "A", "program_id": "a", "program_component_score": 0.3},
            {"relation_id": "r1", "patient_id": "p4", "group_id": "B", "program_id": "a", "program_component_score": 0.8},
            {"relation_id": "r1", "patient_id": "p5", "group_id": "B", "program_id": "a", "program_component_score": 0.9},
            {"relation_id": "r1", "patient_id": "p6", "group_id": "B", "program_id": "a", "program_component_score": 1.0},
        ]
    )

    with pytest.raises(ContractError, match="sequence"):
        da.relation_program_group_association(
            scores,
            comparisons=({"comparison_id": "bad", "groups": "AB"},),
        )


def test_relation_program_stats_reject_nonfinite_p_values() -> None:
    with pytest.raises(ContractError, match="non-finite p_value"):
        finite_p_value(np.nan)
