from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import stride
import stride.pl as pl
from stride.errors import ContractError


def _rank_diagnostics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "relation_id": "r1",
                "rank_patient": 1,
                "rank_source": 1,
                "rank_target_open": 1,
                "restart_id": 0,
                "random_seed": 10,
                "reconstruction_error": 0.5,
                "relative_error": 0.50,
                "status": "ok",
                "error_message": None,
            },
            {
                "relation_id": "r1",
                "rank_patient": 2,
                "rank_source": 1,
                "rank_target_open": 1,
                "restart_id": 0,
                "random_seed": 11,
                "reconstruction_error": 0.3,
                "relative_error": 0.30,
                "status": "ok",
                "error_message": None,
            },
            {
                "relation_id": "r2",
                "rank_patient": 1,
                "rank_source": 1,
                "rank_target_open": 1,
                "restart_id": 0,
                "random_seed": 12,
                "reconstruction_error": 0.4,
                "relative_error": 0.40,
                "status": "ok",
                "error_message": None,
            },
        ]
    )


def _scores() -> pd.DataFrame:
    rows = []
    for program_id, values in {
        "program_pf0_sf0_tof0": (0.1, 0.2, 0.3, 0.8, 0.9, 1.0),
        "program_pf1_sf0_tof0": (1.0, 0.9, 0.8, 0.3, 0.2, 0.1),
    }.items():
        for patient_id, group_id, score in zip(
            ("p1", "p2", "p3", "p4", "p5", "p6"),
            ("low", "low", "low", "high", "high", "high"),
            values,
            strict=True,
        ):
            rows.append(
                {
                    "relation_id": "r1",
                    "patient_id": patient_id,
                    "group_id": group_id,
                    "program_id": program_id,
                    "program_component_score": score,
                }
            )
    return pd.DataFrame(rows)


def _three_group_scores() -> pd.DataFrame:
    scores = _scores()
    rows = []
    for score in (0.45, 0.50, 0.55):
        rows.append(
            {
                "relation_id": "r1",
                "patient_id": f"pm{len(rows) + 1}",
                "group_id": "medium",
                "program_id": "program_pf0_sf0_tof0",
                "program_component_score": score,
            }
        )
    return pd.concat([scores, pd.DataFrame(rows)], ignore_index=True)


def _association_stats() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "relation_id": "r1",
                "comparison_id": "low_vs_high",
                "program_id": "program_pf0_sf0_tof0",
                "comparison_type": "two_group",
                "group_1": "low",
                "group_2": "high",
                "groups": ("low", "high"),
                "n_total": 6,
                "n_by_group": {"low": 3, "high": 3},
                "mean_by_group": {"low": 0.2, "high": 0.9},
                "median_by_group": {"low": 0.2, "high": 0.9},
                "std_by_group": {"low": 0.1, "high": 0.1},
                "test_name": "mannwhitneyu",
                "effect_size": -1.0,
                "effect_size_type": "cliffs_delta",
                "effect_direction": "low<high",
                "p_value": 0.05,
                "q_value": 0.04,
                "correction_method": "BH",
                "correction_scope": "r1:low_vs_high",
            }
        ]
    )


def _multi_group_stats() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "relation_id": "r1",
                "comparison_id": "three_groups",
                "program_id": "program_pf0_sf0_tof0",
                "comparison_type": "multi_group",
                "group_1": np.nan,
                "group_2": np.nan,
                "groups": ("low", "medium", "high"),
                "n_total": 9,
                "n_by_group": {"low": 3, "medium": 3, "high": 3},
                "mean_by_group": {"low": 0.2, "medium": 0.5, "high": 0.9},
                "median_by_group": {"low": 0.2, "medium": 0.5, "high": 0.9},
                "std_by_group": {"low": 0.1, "medium": 0.05, "high": 0.1},
                "test_name": "one_way_anova",
                "effect_size": 0.7,
                "effect_size_type": "eta_squared",
                "effect_direction": "multi_group",
                "p_value": 0.014,
                "q_value": 0.012,
                "correction_method": "BH",
                "correction_scope": "r1:three_groups",
            }
        ]
    )


def _program_entries() -> pd.DataFrame:
    rows = []
    for source_id in (0, 1):
        for axis_id, axis_type, contribution in (
            (0, "target_community", 0.10 + source_id),
            (1, "target_community", 0.20 + source_id),
            (2, "source_open", 0.30 + source_id),
        ):
            rows.append(
                {
                    "relation_id": "r1",
                    "program_id": "program_pf0_sf0_tof0",
                    "source_community_id": source_id,
                    "target_open_axis_id": axis_id,
                    "target_open_axis_type": axis_type,
                    "program_component_contribution": contribution,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def _statannotations_available(monkeypatch):
    calls: list[tuple[str, object]] = []
    package = types.ModuleType("statannotations")
    module = types.ModuleType("statannotations.Annotator")

    class FakeAnnotator:
        def __init__(self, *args, **kwargs):
            calls.append(("init", {"args": args, "kwargs": kwargs}))

        def configure(self, **kwargs):
            calls.append(("configure", kwargs))
            assert kwargs["test"] is None
            return self

        def set_pvalues_and_annotate(self, pvalues):
            calls.append(("set_pvalues_and_annotate", list(pvalues)))
            return None

        def apply_test(self):
            raise AssertionError(".pl must not compute statistical tests")

    module.Annotator = FakeAnnotator
    monkeypatch.setitem(sys.modules, "statannotations", package)
    monkeypatch.setitem(sys.modules, "statannotations.Annotator", module)
    return calls


def test_relation_program_plot_functions_are_pl_only_public_api() -> None:
    assert "relation_program_rank_elbow_plot" in pl.__all__
    assert "relation_program_score_boxplot" in pl.__all__
    assert "relation_program_structure_heatmap" in pl.__all__
    assert "relation_program_rank_elbow_plot" not in stride.__all__
    assert "relation_program_score_boxplot" not in stride.__all__
    assert "relation_program_structure_heatmap" not in stride.__all__


def test_relation_program_rank_elbow_plot_consumes_diagnostics_only() -> None:
    fig = pl.relation_program_rank_elbow_plot(_rank_diagnostics())

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    assert fig.axes[0].get_ylabel() == "Relative reconstruction error"
    assert fig.axes[0].get_title() == ""
    assert [tick.get_text() for tick in fig.axes[0].get_xticklabels()] == [
        "1/1/1",
        "2/1/1",
    ]
    assert {tick.get_rotation() for tick in fig.axes[0].get_xticklabels()} == {90.0}


def test_relation_program_rank_elbow_plot_ignores_failed_diagnostic_rows() -> None:
    failed = pd.DataFrame(
        [
            {
                "relation_id": "r1",
                "rank_patient": 1,
                "rank_source": 1,
                "rank_target_open": 1,
                "restart_id": 1,
                "random_seed": 99,
                "reconstruction_error": np.nan,
                "relative_error": np.nan,
                "status": "failed",
                "error_message": "numerical failure",
            }
        ]
    )
    fig = pl.relation_program_rank_elbow_plot(pd.concat([_rank_diagnostics(), failed], ignore_index=True))

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2


def test_relation_program_score_boxplot_consumes_scores_and_optional_stats(
    _statannotations_available,
) -> None:
    fig = pl.relation_program_score_boxplot(
        _scores(),
        association_stats=_association_stats(),
        program_ids=["program_pf0_sf0_tof0"],
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_xlabel() == "group_id"
    assert ax.get_ylabel() == "Patient program score"
    assert ax.get_title() == "Patient program score by group"
    assert len(ax.collections) >= 1
    assert ("set_pvalues_and_annotate", [0.04]) in _statannotations_available


def test_relation_program_score_boxplot_puts_multi_group_stats_in_title(
    _statannotations_available,
) -> None:
    fig = pl.relation_program_score_boxplot(
        _three_group_scores(),
        association_stats=_multi_group_stats(),
        comparison_id="three_groups",
        program_ids=["program_pf0_sf0_tof0"],
    )

    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == "Patient program score by group\nANOVA, q=0.012 *"
    assert not any(call[0] == "set_pvalues_and_annotate" for call in _statannotations_available)


def test_relation_program_score_boxplot_renders_multiple_pairwise_rows(
    _statannotations_available,
) -> None:
    high_medium = _association_stats().copy()
    high_medium["comparison_id"] = "all_pairwise"
    high_medium["group_1"] = "medium"
    high_medium["group_2"] = "high"
    high_medium["q_value"] = 0.03
    low_high = _association_stats().copy()
    low_high["comparison_id"] = "all_pairwise"
    stats = pd.concat([low_high, high_medium], ignore_index=True)

    fig = pl.relation_program_score_boxplot(
        _three_group_scores(),
        association_stats=stats,
        comparison_id="all_pairwise",
        program_ids=["program_pf0_sf0_tof0"],
    )

    assert isinstance(fig, Figure)
    init_calls = [call for call in _statannotations_available if call[0] == "init"]
    assert len(init_calls) == 1
    assert init_calls[0][1]["args"][1] == [("low", "high"), ("medium", "high")]
    assert ("set_pvalues_and_annotate", [0.04, 0.03]) in _statannotations_available


def test_relation_program_score_boxplot_rejects_ambiguous_stats_annotation(
    _statannotations_available,
) -> None:
    stats = pd.concat([_association_stats(), _association_stats()], ignore_index=True)

    with pytest.raises(ContractError, match="multiple association"):
        pl.relation_program_score_boxplot(
            _scores(),
            association_stats=stats,
            program_ids=["program_pf0_sf0_tof0"],
        )


def test_relation_program_score_boxplot_rejects_mixed_stats_rows(
    _statannotations_available,
) -> None:
    stats = pd.concat([_association_stats(), _multi_group_stats()], ignore_index=True)

    with pytest.raises(ContractError, match="mix"):
        pl.relation_program_score_boxplot(
            _three_group_scores(),
            association_stats=stats,
            program_ids=["program_pf0_sf0_tof0"],
        )


def test_relation_program_score_boxplot_selects_stats_comparison_id(
    _statannotations_available,
) -> None:
    extra = _association_stats().copy()
    extra["comparison_id"] = "low_vs_context"
    extra["q_value"] = 0.25
    stats = pd.concat([_association_stats(), extra], ignore_index=True)

    fig = pl.relation_program_score_boxplot(
        _scores(),
        association_stats=stats,
        comparison_id="low_vs_context",
        program_ids=["program_pf0_sf0_tof0"],
    )

    assert isinstance(fig, Figure)
    assert ("set_pvalues_and_annotate", [0.25]) in _statannotations_available


def test_relation_program_structure_heatmap_preserves_target_open_axis_alignment() -> None:
    fig = pl.relation_program_structure_heatmap(
        _program_entries(),
        state_labels=("C0", "C1"),
    )

    assert isinstance(fig, Figure)
    heatmap_axes = [ax for ax in fig.axes if ax.images]
    assert len(heatmap_axes) == 1
    ax = heatmap_axes[0]
    assert ax.get_xlabel() == "Target community"
    assert ax.get_ylabel() == "Source community"
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["C0", "C1", "source open d"]
    assert {tick.get_rotation() for tick in ax.get_xticklabels()} == {90.0}
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["C0", "C1"]
    np.testing.assert_allclose(ax.images[0].get_array(), [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]])


def test_relation_program_structure_heatmap_rejects_duplicate_cells_with_contract_error() -> None:
    entries = pd.concat([_program_entries(), _program_entries().iloc[[0]]], ignore_index=True)

    with pytest.raises(ContractError, match="duplicate"):
        pl.relation_program_structure_heatmap(entries)


def test_relation_program_plots_save_pdf_and_reject_non_pdf(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, object]] = []
    original_savefig = Figure.savefig

    def spy_savefig(self, *args, **kwargs):
        calls.append(kwargs)
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", spy_savefig)

    returned = pl.relation_program_rank_elbow_plot(
        _rank_diagnostics(),
        relation_id="r1",
        save=tmp_path / "rank.pdf",
    )

    assert returned is None
    assert calls[0]["bbox_inches"] == "tight"
    assert calls[0]["pad_inches"] == 0.08
    with pytest.raises(ContractError, match="PDF"):
        pl.relation_program_structure_heatmap(_program_entries(), save=tmp_path / "heatmap.png")
