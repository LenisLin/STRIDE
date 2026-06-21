from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import stride
import stride.pl as pl
from stride.errors import ContractError


def _stats(effect_size_type: str = "cliffs_delta") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "relation_id": "pre_TC_to_post_IM",
                "comparison_id": "low_vs_high",
                "comparison_type": "two_group",
                "row_id": 0,
                "col_id": 0,
                "entry_type": "A",
                "group_1": "low",
                "group_2": "high",
                "groups": ("low", "high"),
                "n_total": 4,
                "n_by_group": {"low": 2, "high": 2},
                "mean_by_group": {"low": 0.75, "high": 0.15},
                "median_by_group": {"low": 0.75, "high": 0.15},
                "std_by_group": {"low": 0.05, "high": 0.05},
                "test_name": "mannwhitneyu",
                "effect_size": 1.0,
                "effect_size_type": effect_size_type,
                "effect_direction": "low>high",
                "p_value": 0.05,
                "q_value": 0.04,
                "correction_method": "BH",
                "correction_scope": "pre_TC_to_post_IM:low_vs_high",
            },
            {
                "relation_id": "pre_TC_to_post_IM",
                "comparison_id": "low_vs_high",
                "comparison_type": "two_group",
                "row_id": 0,
                "col_id": 2,
                "entry_type": "d",
                "group_1": "low",
                "group_2": "high",
                "groups": ("low", "high"),
                "n_total": 4,
                "n_by_group": {"low": 2, "high": 2},
                "mean_by_group": {"low": 0.10, "high": 0.10},
                "median_by_group": {"low": 0.10, "high": 0.10},
                "std_by_group": {"low": 0.0, "high": 0.0},
                "test_name": "mannwhitneyu",
                "effect_size": 0.0,
                "effect_size_type": effect_size_type,
                "effect_direction": "none",
                "p_value": 1.0,
                "q_value": 0.60,
                "correction_method": "BH",
                "correction_scope": "pre_TC_to_post_IM:low_vs_high",
            },
            {
                "relation_id": "pre_TC_to_post_IM",
                "comparison_id": "low_vs_high",
                "comparison_type": "two_group",
                "row_id": 2,
                "col_id": 0,
                "entry_type": "e",
                "group_1": "low",
                "group_2": "high",
                "groups": ("low", "high"),
                "n_total": 4,
                "n_by_group": {"low": 2, "high": 2},
                "mean_by_group": {"low": 0.08, "high": 0.33},
                "median_by_group": {"low": 0.08, "high": 0.33},
                "std_by_group": {"low": 0.03, "high": 0.03},
                "test_name": "mannwhitneyu",
                "effect_size": -1.0,
                "effect_size_type": effect_size_type,
                "effect_direction": "low<high",
                "p_value": 0.05,
                "q_value": 0.004,
                "correction_method": "BH",
                "correction_scope": "pre_TC_to_post_IM:low_vs_high",
            },
        ]
    )


def test_augmented_relation_association_bubble_plot_is_pl_only_public_api() -> None:
    assert "augmented_relation_association_bubble_plot" in pl.__all__
    assert "augmented_relation_association_bubble_plot" not in stride.__all__


def test_augmented_relation_association_bubble_plot_renders_supplied_stats() -> None:
    fig = pl.augmented_relation_association_bubble_plot(
        _stats(),
        state_labels=("C0", "C1"),
    )

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Target community"
    assert ax.get_ylabel() == "Source community"
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["C0", "C1", "source open d"]
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["C0", "C1", "target open e"]
    assert {tick.get_rotation() for tick in ax.get_xticklabels()} == {90.0}
    assert len(ax.collections) >= 1
    offsets = ax.collections[0].get_offsets()
    assert offsets.shape[0] == 3
    assert not np.any((offsets[:, 0] == 2) & (offsets[:, 1] == 2))
    assert len(fig.axes) >= 2


def test_augmented_relation_association_bubble_plot_filters_relation_and_comparison() -> None:
    stats = pd.concat(
        [
            _stats(),
            _stats().assign(relation_id="other", comparison_id="other_comparison"),
        ],
        ignore_index=True,
    )

    fig = pl.augmented_relation_association_bubble_plot(
        stats,
        relation_id="other",
        comparison_id="other_comparison",
    )

    assert isinstance(fig, Figure)
    assert fig.axes[0].collections[0].get_offsets().shape[0] == 3
    assert fig.axes[0].get_title() == ""


def test_augmented_relation_association_bubble_plot_rejects_mixed_effect_size_types() -> None:
    stats = _stats()
    stats.loc[0, "effect_size_type"] = "eta_squared"

    with pytest.raises(ContractError, match="effect_size_type"):
        pl.augmented_relation_association_bubble_plot(stats)


def test_augmented_relation_association_bubble_plot_rejects_bottom_right_cell() -> None:
    stats = _stats()
    stats.loc[0, ["row_id", "col_id"]] = [2, 2]

    with pytest.raises(ContractError, match="bottom-right"):
        pl.augmented_relation_association_bubble_plot(stats)
