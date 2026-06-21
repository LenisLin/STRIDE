"""Relation-program plotting surfaces for STRIDE `.pl`."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from stride.errors import ContractError

from ._utils import (
    DEFAULT_GROUP_PALETTE,
    _apply_clean_axes_style,
    _apply_figure_margins,
    _bio_continuous_cmap,
    _format_pq_label,
    _group_color_map,
    _ordered_unique,
    _save_or_return_figure,
    _set_right_angle_xticklabels,
)


def relation_program_rank_elbow_plot(
    rank_diagnostics: pd.DataFrame,
    *,
    relation_id: str | None = None,
    figsize: tuple[float, float] | None = None,
    save: str | Path | None = None,
) -> Figure | None:
    """Plot relation-program rank diagnostics as an elbow-style figure.

    Scientific question:
        How does reconstruction error vary across caller-supplied candidate
        ranks for exploratory T-only relation-program decomposition?

    Required input:
        `rank_diagnostics` is a `.da` diagnostics table with relation id,
        candidate ranks, restart id, random seed, reconstruction error,
        relative error, and status metadata.

    Visual role:
        The plot summarizes diagnostic evidence for rank discussion. It may
        display candidate ranks, restart variability, and reconstruction error
        trends, depending on implementation.

    Boundary:
        This is a rendering function. It does not run tensor decomposition,
        compute reconstruction loss, choose rank, mutate diagnostics, or write
        scientific result payloads.
    """
    data = _prepare_rank_diagnostics(rank_diagnostics, relation_id=relation_id)
    relation_ids = _ordered_unique(data["relation_id"])
    if figsize is None:
        figsize = (max(5.0, 4.2 * len(relation_ids)), 3.8)
    fig, axes = plt.subplots(1, len(relation_ids), figsize=figsize, squeeze=False)
    for ax, resolved_relation_id in zip(axes[0], relation_ids, strict=True):
        relation_data = data[data["relation_id"] == resolved_relation_id]
        candidate_order = _rank_candidate_order(relation_data)
        positions = np.arange(len(candidate_order), dtype=float)
        grouped = relation_data.groupby("rank_label", sort=False)
        for index, rank_label in enumerate(candidate_order):
            values = grouped.get_group(rank_label)["relative_error"].to_numpy(dtype=float)
            x_values = np.full(values.shape, positions[index], dtype=float)
            ax.scatter(
                x_values,
                values,
                s=34.0,
                color=DEFAULT_GROUP_PALETTE[1],
                edgecolors="#2F3437",
                linewidths=0.35,
                zorder=2,
            )
            ax.plot(
                [positions[index] - 0.18, positions[index] + 0.18],
                [float(np.median(values)), float(np.median(values))],
                color="#2F3437",
                linewidth=1.0,
                zorder=3,
            )
        medians = [float(np.median(grouped.get_group(label)["relative_error"])) for label in candidate_order]
        ax.plot(positions, medians, color=DEFAULT_GROUP_PALETTE[2], linewidth=1.0, zorder=1)
        _mark_selected_rank(ax, relation_data, candidate_order, positions)
        ax.set_xticks(positions, labels=candidate_order)
        _set_right_angle_xticklabels(ax)
        ax.set_ylabel("Relative reconstruction error")
        ax.set_xlabel("Candidate Tucker rank (patient/source/target-open)")
        _apply_clean_axes_style(ax)
    _apply_figure_margins(fig, left=0.16, right=0.98, bottom=0.34, top=0.94, wspace=0.35)
    return _save_or_return_figure(fig, save)


def relation_program_score_boxplot(
    patient_program_scores: pd.DataFrame,
    *,
    association_stats: pd.DataFrame | None = None,
    relation_id: str | None = None,
    comparison_id: str | None = None,
    program_ids: Sequence[str] | None = None,
    group_key: str = "group_id",
    figsize: tuple[float, float] | None = None,
    save: str | Path | None = None,
) -> Figure | None:
    """Plot patient relation-program scores by caller-provided group.

    Scientific question:
        Which exploratory relation-program scores differ across
        caller-provided patient groups according to externally supplied
        downstream analysis?

    Required input:
        `patient_program_scores` is a `.da` table with patient-level program
        component scores. `association_stats`, when supplied, is a `.da` association
        table containing externally computed effect sizes, p-values, q-values,
        and test metadata. `comparison_id` selects one supplied comparison
        when multiple association rows exist for the same relation/program.

    Default visual form:
        Boxplot grouped by `group_key`, optionally faceted or filtered by
        relation and program. Jittered points may be added by implementation,
        but the program component score distribution remains the plotted
        patient-level object.

    Boundary:
        This function does not compute program scores, group summaries,
        p-values, q-values, effect sizes, BH correction, tensor decomposition,
        or rank selection. It renders supplied statistics as annotations only.
    """
    data = _prepare_patient_program_scores(
        patient_program_scores,
        relation_id=relation_id,
        program_ids=program_ids,
        group_key=group_key,
    )
    resolved_program_ids = _ordered_unique(data["program_id"])
    if figsize is None:
        figsize = (max(4.8, 3.6 * len(resolved_program_ids)), 4.0)
    fig, axes = plt.subplots(1, len(resolved_program_ids), figsize=figsize, squeeze=False)
    for ax, program_id in zip(axes[0], resolved_program_ids, strict=True):
        program_data = data[data["program_id"] == program_id]
        group_ids = _ordered_unique(program_data[group_key])
        values = [
            program_data.loc[
                program_data[group_key].astype(str) == group_id,
                "program_component_score",
            ].to_numpy(dtype=float)
            for group_id in group_ids
        ]
        group_colors = _group_color_map(group_ids)
        boxes = ax.boxplot(
            values,
            tick_labels=group_ids,
            patch_artist=True,
            boxprops={"facecolor": "#F7FAFC", "edgecolor": "#2F3437", "linewidth": 0.8},
            medianprops={"color": DEFAULT_GROUP_PALETTE[0], "linewidth": 1.2},
            whiskerprops={"color": "#2F3437", "linewidth": 0.8},
            capprops={"color": "#2F3437", "linewidth": 0.8},
        )
        for box, group_id in zip(boxes["boxes"], group_ids, strict=True):
            box.set_facecolor(group_colors[group_id])
            box.set_alpha(0.35)
        for index, group_values in enumerate(values, start=1):
            jitter = np.linspace(-0.08, 0.08, num=len(group_values)) if len(group_values) else []
            ax.scatter(
                np.full(len(group_values), index, dtype=float) + jitter,
                group_values,
                s=28.0,
                color=group_colors[group_ids[index - 1]],
                edgecolors="#2F3437",
                linewidths=0.35,
                zorder=3,
            )
        ax.set_xlabel(group_key)
        ax.set_ylabel("Patient program score")
        ax.set_title("Patient program score by group", fontsize=10)
        _annotate_program_score_stats(
            ax,
            program_data,
            association_stats,
            relation_id=_single_value(program_data["relation_id"], "relation_id"),
            comparison_id=comparison_id,
            program_id=str(program_id),
            group_key=group_key,
            group_ids=group_ids,
        )
        _apply_clean_axes_style(ax)
    _apply_figure_margins(fig, left=0.18, right=0.98, bottom=0.18, top=0.82, wspace=0.35)
    return _save_or_return_figure(fig, save)


def relation_program_structure_heatmap(
    program_entries: pd.DataFrame,
    *,
    relation_id: str | None = None,
    program_id: str | None = None,
    state_labels: Sequence[str] | None = None,
    state_order: Sequence[int] | None = None,
    figsize: tuple[float, float] | None = None,
    save: str | Path | None = None,
) -> Figure | None:
    """Plot a relation-program source-to-target/open contribution heatmap.

    Scientific question:
        What source-community to target/open structure is represented by an
        exploratory relation program?

    Required input:
        `program_entries` is a `.da` table with one row per relation, program,
        source community, target/open axis entry, target/open axis type, and
        program component contribution value.

    Display object:
        Rows are source communities. Columns are target communities plus the
        source-open `d` column. Values are decomposition-derived program
        component contributions, not original fitted `A` or `d` entries.

    Boundary:
        This function does not decompose tensors, compute program
        contributions, test groups, compute p-values/q-values/effect sizes, or
        mutate input tables.
    """
    data = _prepare_program_entries(
        program_entries,
        relation_id=relation_id,
        program_id=program_id,
    )
    K = _n_states_from_program_entries(data)
    order = _resolve_state_order(K, state_order)
    labels = _resolve_state_labels(K, state_labels)
    ordered_labels = [labels[index] for index in order]
    matrix = _program_entry_matrix(data, order=order, n_states=K)
    if figsize is None:
        figsize = (max(5.5, 0.55 * (K + 1) + 3.0), max(3.2, 0.45 * K + 2.0))
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(matrix, aspect="auto", cmap=_bio_continuous_cmap(), vmin=0.0)
    ax.set_xticks(np.arange(K + 1), labels=[*ordered_labels, "source open d"])
    ax.set_yticks(np.arange(K), labels=ordered_labels)
    _set_right_angle_xticklabels(ax)
    ax.set_xlabel("Target community")
    ax.set_ylabel("Source community")
    _apply_clean_axes_style(ax)
    fig.colorbar(
        image,
        ax=ax,
        fraction=0.045,
        pad=0.02,
        label="Program component contribution",
    )
    _apply_figure_margins(fig, left=0.24, right=0.88, bottom=0.32, top=0.95)
    return _save_or_return_figure(fig, save)


_REQUIRED_RANK_DIAGNOSTIC_COLUMNS = {
    "relation_id",
    "rank_patient",
    "rank_source",
    "rank_target_open",
    "restart_id",
    "random_seed",
    "reconstruction_error",
    "relative_error",
    "status",
}

_REQUIRED_SCORE_COLUMNS = {
    "relation_id",
    "patient_id",
    "group_id",
    "program_id",
    "program_component_score",
}

_REQUIRED_PROGRAM_ENTRY_COLUMNS = {
    "relation_id",
    "program_id",
    "source_community_id",
    "target_open_axis_id",
    "target_open_axis_type",
    "program_component_contribution",
}


def _prepare_rank_diagnostics(
    rank_diagnostics: pd.DataFrame,
    *,
    relation_id: str | None,
) -> pd.DataFrame:
    """Validate and filter rank diagnostics for plotting."""
    if not isinstance(rank_diagnostics, pd.DataFrame):
        raise ContractError("rank_diagnostics must be a pandas DataFrame")
    missing = sorted(_REQUIRED_RANK_DIAGNOSTIC_COLUMNS.difference(rank_diagnostics.columns))
    if missing:
        raise ContractError("rank_diagnostics is missing required columns: " + ", ".join(missing))
    data = rank_diagnostics.copy()
    if relation_id is not None:
        data = data[data["relation_id"].astype(str) == str(relation_id)]
    if data.empty:
        raise ContractError("rank_diagnostics filter produced no rows")
    data = data[data["status"].astype(str) == "ok"].copy()
    if data.empty:
        raise ContractError("rank_diagnostics has no ok rows after filtering")
    for column in ["rank_patient", "rank_source", "rank_target_open", "restart_id", "random_seed"]:
        data[column] = _as_int_column(data[column], name=column)
    for column in ["reconstruction_error", "relative_error"]:
        data[column] = _as_finite_float_column(data[column], name=column)
    if (data["relative_error"] < 0.0).any():
        raise ContractError("rank_diagnostics relative_error must be nonnegative")
    data["relation_id"] = data["relation_id"].astype(str)
    data["rank_label"] = [
        f"{row.rank_patient}/{row.rank_source}/{row.rank_target_open}"
        for row in data.itertuples(index=False)
    ]
    return data.reset_index(drop=True)


def _mark_selected_rank(
    ax,
    relation_data: pd.DataFrame,
    candidate_order: Sequence[str],
    positions: np.ndarray,
) -> None:
    """Highlight caller-selected diagnostics when the table contains them."""
    if "selected" not in relation_data:
        return
    selected = relation_data[relation_data["selected"].astype(bool)]
    if selected.empty:
        return
    for rank_label, rows in selected.groupby("rank_label", sort=False):
        if rank_label not in candidate_order:
            continue
        position = positions[candidate_order.index(rank_label)]
        value = float(rows["relative_error"].min())
        ax.scatter(
            [position],
            [value],
            marker="*",
            s=110.0,
            color=DEFAULT_GROUP_PALETTE[0],
            edgecolors="#2F3437",
            linewidths=0.5,
            zorder=4,
            label="selected restart",
        )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:1], labels[:1], frameon=False, fontsize=8)


def _rank_candidate_order(data: pd.DataFrame) -> tuple[str, ...]:
    """Return rank labels ordered by first observed rank tuple."""
    return _ordered_unique(data["rank_label"])


def _prepare_patient_program_scores(
    patient_program_scores: pd.DataFrame,
    *,
    relation_id: str | None,
    program_ids: Sequence[str] | None,
    group_key: str,
) -> pd.DataFrame:
    """Validate and filter patient program scores for plotting."""
    if not isinstance(patient_program_scores, pd.DataFrame):
        raise ContractError("patient_program_scores must be a pandas DataFrame")
    required = set(_REQUIRED_SCORE_COLUMNS)
    required.add(group_key)
    missing = sorted(required.difference(patient_program_scores.columns))
    if missing:
        raise ContractError("patient_program_scores is missing required columns: " + ", ".join(missing))
    data = patient_program_scores.copy()
    if relation_id is not None:
        data = data[data["relation_id"].astype(str) == str(relation_id)]
    if program_ids is not None:
        allowed = {str(program_id) for program_id in program_ids}
        data = data[data["program_id"].astype(str).isin(allowed)]
    if data.empty:
        raise ContractError("patient_program_scores filter produced no rows")
    for column in ["relation_id", "patient_id", group_key, "program_id"]:
        data[column] = data[column].astype(str)
    data["program_component_score"] = _as_finite_float_column(
        data["program_component_score"],
        name="program_component_score",
    )
    return data.reset_index(drop=True)


def _annotate_program_score_stats(
    ax,
    program_data: pd.DataFrame,
    association_stats: pd.DataFrame | None,
    *,
    relation_id: str,
    comparison_id: str | None,
    program_id: str,
    group_key: str,
    group_ids: Sequence[str],
) -> None:
    """Render supplied program association statistics without computing tests."""
    rows = _program_stats_rows(
        association_stats,
        relation_id=relation_id,
        comparison_id=comparison_id,
        program_id=program_id,
    )
    if rows.empty:
        return
    comparison_types = set(rows["comparison_type"].astype(str))
    if comparison_types == {"multi_group"}:
        if len(rows) > 1:
            raise ContractError("multiple multi_group association_stats rows match one relation/program")
        _append_panel_subtitle(ax, _format_multi_group_stats_title(rows.iloc[0]))
        return
    if comparison_types != {"two_group"}:
        raise ContractError("association_stats must not mix multi_group and two_group rows")

    pairs: list[tuple[str, str]] = []
    pvalues: list[float] = []
    seen_pairs: set[tuple[str, str]] = set()
    for _, row in rows.iterrows():
        group_1 = row.get("group_1")
        group_2 = row.get("group_2")
        if pd.isna(group_1) or pd.isna(group_2):
            raise ContractError("two_group association_stats rows require group_1 and group_2")
        group_1 = str(group_1)
        group_2 = str(group_2)
        if group_1 not in group_ids or group_2 not in group_ids:
            continue
        pair = (group_1, group_2)
        pair_key = tuple(sorted(pair))
        if pair_key in seen_pairs:
            raise ContractError("multiple association_stats rows match one two_group pair")
        seen_pairs.add(pair_key)
        pairs.append(pair)
        pvalues.append(_annotation_value(row))
    if not pairs:
        return
    try:
        from statannotations.Annotator import Annotator
    except ModuleNotFoundError as exc:
        raise ContractError(
            "statannotations is required for program score significance annotations"
        ) from exc
    annotator = Annotator(
        ax,
        pairs,
        data=program_data,
        x=group_key,
        y="program_component_score",
        order=list(group_ids),
    )
    annotator.configure(
        test=None,
        text_format="star",
        loc="outside",
        line_height=0.03,
        line_width=1.0,
        show_test_name=False,
        verbose=0,
    )
    annotator.set_pvalues_and_annotate(pvalues)


def _append_panel_subtitle(ax, subtitle: str) -> None:
    """Append supplied evidence text under the current panel title."""
    title = ax.get_title()
    if title:
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    else:
        ax.set_title(subtitle, fontsize=10)


def _program_stats_rows(
    association_stats: pd.DataFrame | None,
    *,
    relation_id: str,
    comparison_id: str | None,
    program_id: str,
) -> pd.DataFrame:
    """Return supplied stats rows for one relation/program selection."""
    if association_stats is None:
        return pd.DataFrame()
    if not isinstance(association_stats, pd.DataFrame):
        raise ContractError("association_stats must be a pandas DataFrame")
    required = {
        "relation_id",
        "program_id",
        "comparison_type",
        "test_name",
        "p_value",
        "q_value",
    }
    if comparison_id is not None:
        required.add("comparison_id")
    missing = sorted(required.difference(association_stats.columns))
    if missing:
        raise ContractError("association_stats is missing required columns: " + ", ".join(missing))
    matches = association_stats[
        (association_stats["relation_id"].astype(str) == relation_id)
        & (association_stats["program_id"].astype(str) == program_id)
    ]
    if comparison_id is not None:
        matches = matches[matches["comparison_id"].astype(str) == str(comparison_id)]
    return matches.reset_index(drop=True)


def _format_multi_group_stats_title(row: pd.Series) -> str:
    """Format one supplied multi-group association row as panel evidence text."""
    test_name = str(row["test_name"])
    if "anova" in test_name.lower():
        test_name = "ANOVA"
    q_label = None
    if "q_value" in row and not pd.isna(row["q_value"]):
        q_label = _format_pq_label(float(row["q_value"]), "q", show_ns=True)
    p_label = None
    if "p_value" in row and not pd.isna(row["p_value"]):
        p_label = _format_pq_label(float(row["p_value"]), "p", show_ns=False)
    labels = [test_name]
    if q_label is not None:
        labels.append(q_label)
    elif p_label is not None:
        labels.append(p_label)
    return ", ".join(labels)


def _annotation_value(row: pd.Series) -> float:
    """Prefer supplied q-value for plotted multiplicity-aware significance."""
    if "q_value" in row and not pd.isna(row["q_value"]):
        return float(row["q_value"])
    return float(row["p_value"])


def _prepare_program_entries(
    program_entries: pd.DataFrame,
    *,
    relation_id: str | None,
    program_id: str | None,
) -> pd.DataFrame:
    """Validate and filter program entries for heatmap rendering."""
    if not isinstance(program_entries, pd.DataFrame):
        raise ContractError("program_entries must be a pandas DataFrame")
    missing = sorted(_REQUIRED_PROGRAM_ENTRY_COLUMNS.difference(program_entries.columns))
    if missing:
        raise ContractError("program_entries is missing required columns: " + ", ".join(missing))
    data = program_entries.copy()
    if relation_id is not None:
        data = data[data["relation_id"].astype(str) == str(relation_id)]
    if program_id is not None:
        data = data[data["program_id"].astype(str) == str(program_id)]
    if data.empty:
        raise ContractError("program_entries filter produced no rows")
    data["relation_id"] = data["relation_id"].astype(str)
    data["program_id"] = data["program_id"].astype(str)
    if data["relation_id"].nunique() != 1:
        raise ContractError("program_entries must contain one relation after filtering")
    if data["program_id"].nunique() != 1:
        raise ContractError("program_entries must contain one program after filtering")
    data["source_community_id"] = _as_int_column(
        data["source_community_id"],
        name="source_community_id",
    )
    data["target_open_axis_id"] = _as_int_column(
        data["target_open_axis_id"],
        name="target_open_axis_id",
    )
    data["program_component_contribution"] = _as_finite_float_column(
        data["program_component_contribution"],
        name="program_component_contribution",
    )
    if (data["program_component_contribution"] < 0.0).any():
        raise ContractError("program_entries program_component_contribution must be nonnegative")
    return data.reset_index(drop=True)


def _n_states_from_program_entries(data: pd.DataFrame) -> int:
    """Infer K from source and target/open coordinates."""
    source_ids = sorted(set(data["source_community_id"].astype(int)))
    if source_ids != list(range(len(source_ids))):
        raise ContractError("source_community_id must cover 0..K-1")
    K = len(source_ids)
    axis_ids = sorted(set(data["target_open_axis_id"].astype(int)))
    if axis_ids != list(range(K + 1)):
        raise ContractError("target_open_axis_id must cover 0..K")
    for _, row in data.iterrows():
        axis_id = int(row["target_open_axis_id"])
        axis_type = str(row["target_open_axis_type"])
        if axis_id < K and axis_type != "target_community":
            raise ContractError("target community columns must use target_community type")
        if axis_id == K and axis_type != "source_open":
            raise ContractError("source-open column must use source_open type")
    expected_n = K * (K + 1)
    coordinates = data[["source_community_id", "target_open_axis_id"]]
    if coordinates.duplicated().any():
        raise ContractError("program_entries contains duplicate source/target-open cells")
    if len(coordinates) != expected_n:
        raise ContractError("program_entries must contain one row for each source/target-open cell")
    return K


def _program_entry_matrix(
    data: pd.DataFrame,
    *,
    order: tuple[int, ...],
    n_states: int,
) -> np.ndarray:
    """Build source x target/open contribution matrix with display ordering."""
    pivot = data.pivot(
        index="source_community_id",
        columns="target_open_axis_id",
        values="program_component_contribution",
    )
    matrix = np.empty((n_states, n_states + 1), dtype=float)
    for new_row, old_row in enumerate(order):
        for new_col, old_col in enumerate(order):
            matrix[new_row, new_col] = float(pivot.loc[old_row, old_col])
        matrix[new_row, n_states] = float(pivot.loc[old_row, n_states])
    return matrix


def _resolve_state_order(n_states: int, state_order: Sequence[int] | None) -> tuple[int, ...]:
    """Resolve display state order as a full permutation."""
    if state_order is None:
        return tuple(range(n_states))
    order = tuple(int(value) for value in state_order)
    if sorted(order) != list(range(n_states)):
        raise ContractError("state_order must be a complete permutation of 0..K-1")
    return order


def _resolve_state_labels(n_states: int, state_labels: Sequence[str] | None) -> tuple[str, ...]:
    """Resolve display labels for the shared state basis."""
    if state_labels is None:
        return tuple(f"C{index}" for index in range(n_states))
    labels = tuple(str(label) for label in state_labels)
    if len(labels) != n_states:
        raise ContractError("state_labels length must match n_states")
    return labels


def _single_value(values: pd.Series, name: str) -> str:
    """Return one unique string value from a column."""
    unique = tuple(dict.fromkeys(values.astype(str)))
    if len(unique) != 1:
        raise ContractError(f"table must contain one {name} after filtering")
    return unique[0]


def _as_int_column(values: pd.Series, *, name: str) -> pd.Series:
    """Convert a column to integers."""
    try:
        return values.astype(int)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must contain integer values") from exc


def _as_finite_float_column(values: pd.Series, *, name: str) -> pd.Series:
    """Convert a column to finite floats."""
    try:
        result = values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be numeric") from exc
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ContractError(f"{name} must contain only finite values")
    return result
