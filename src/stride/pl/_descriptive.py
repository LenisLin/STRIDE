"""Descriptive plotting functions for STRIDE `.pl`."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from stride._array_contracts import resolve_axis_labels, resolve_full_axis_order
from stride._schema import (
    OBS_CELL_TYPE_KEY,
    OBS_DOMAIN_KEY,
    OBS_FOV_KEY,
    OBS_PATIENT_KEY,
    OBS_STATE_ID_KEY,
    OBS_TIMEPOINT_KEY,
    STRIDE_CONFIG_KEY,
    STRIDE_FOV_OBSERVATIONS_KEY,
    STRIDE_RELATION_IDS_KEY,
    STRIDE_RELATIONS_KEY,
    STRIDE_UNS_KEY,
)
from stride.errors import ContractError

from ._utils import (
    BIO_PASTEL_PALETTE,
    SOURCE_TARGET_PALETTE,
    _apply_clean_axes_style,
    _apply_figure_margins,
    _auto_grid,
    _bio_continuous_cmap,
    _categorical_color_map,
    _default_figsize,
    _format_pq_label,
    _ordered_unique,
    _save_or_return_figure,
)


def community_annotation_heatmap(
    adata: AnnData,
    *,
    state_key: str = OBS_STATE_ID_KEY,
    cell_type_key: str = OBS_CELL_TYPE_KEY,
    patient_key: str = OBS_PATIENT_KEY,
    fov_key: str = OBS_FOV_KEY,
    domain_key: str = OBS_DOMAIN_KEY,
    time_key: str | None = OBS_TIMEPOINT_KEY,
    state_labels: Sequence[str] | None = None,
    state_order: Sequence[int] | None = None,
    cell_type_order: Sequence[str] | None = None,
    include_timepoint_annotation: bool = True,
    figsize: tuple[float, float] | None = None,
    save: str | Path | None = None,
) -> Figure | None:
    """Describe shared communities/states using cell-level annotations.

    Scientific question:
        What cell-ecology structure defines each shared community/state, and
        how much patient, FOV, domain, and optional timepoint support is
        visible for that state?

    Required input:
        `adata.obs` must contain `state_key`, `cell_type_key`, `patient_key`,
        `fov_key`, and `domain_key`. `time_key` is optional; when requested but
        absent, the timepoint side annotation is omitted. The state axis size
        `K` is read from `adata.uns["stride"]["config"]["n_states"]` when
        available and otherwise inferred from observed nonnegative integer
        state ids.

    Derived display quantities:
        The main heatmap is `cell_fraction[K, C]`, the fraction of cell
        subtypes within each state. Side annotations show state-level domain
        composition, optional timepoint composition, patient prevalence, and
        FOV prevalence. Empty states are retained as zero rows. FOV prevalence
        uses `(patient_id, timepoint, fov_id)` when a timepoint field is
        available and `(patient_id, fov_id)` otherwise.

    Ordering and output:
        States default to `0..K-1`; `state_order`, if supplied, must be a full
        permutation rather than a subset. Cell subtypes default to alphabetical
        order. With `save=None`, the function returns a matplotlib `Figure`.
        With `save="*.pdf"`, it writes a local PDF, closes the figure, and
        returns `None`.

    Boundary:
        This is descriptive community annotation. It does not test
        differential abundance, estimate STRIDE relations, compute p-values, or
        mutate `adata`.
    """
    _require_anndata(adata)
    required = [state_key, cell_type_key, patient_key, fov_key, domain_key]
    if include_timepoint_annotation and time_key is not None and time_key not in adata.obs:
        include_timepoint_annotation = False
    _require_obs_columns(adata, required)

    obs = adata.obs.copy()
    state_ids = _validate_state_ids(obs[state_key], n_obs=adata.n_obs)
    n_states = _resolve_n_states(adata, state_ids=state_ids)
    state_ids = _validate_state_ids(obs[state_key], n_states=n_states, n_obs=adata.n_obs)
    state_order_tuple = _resolve_state_order(n_states, state_order)
    labels = _resolve_state_labels(n_states, state_labels)
    ordered_labels = [labels[state] for state in state_order_tuple]

    if cell_type_order is None:
        cell_types = tuple(sorted(_ordered_unique(obs[cell_type_key])))
    else:
        cell_types = tuple(str(value) for value in cell_type_order)
        observed = set(str(value) for value in obs[cell_type_key])
        missing_types = sorted(observed.difference(cell_types))
        if missing_types:
            raise ContractError(
                "cell_type_order is missing observed cell subtype labels: "
                + ", ".join(missing_types)
            )

    obs["_stride_plot_state_id"] = state_ids
    cell_counts = pd.crosstab(obs["_stride_plot_state_id"], obs[cell_type_key])
    cell_counts = cell_counts.reindex(index=range(n_states), columns=cell_types, fill_value=0)
    cell_fraction = _row_fraction(cell_counts.to_numpy(dtype=float))

    domain_fraction = _state_category_fraction(
        obs,
        n_states=n_states,
        state_column="_stride_plot_state_id",
        category_column=domain_key,
    )
    time_fraction: pd.DataFrame | None = None
    if include_timepoint_annotation and time_key is not None:
        time_fraction = _state_category_fraction(
            obs,
            n_states=n_states,
            state_column="_stride_plot_state_id",
            category_column=time_key,
        )

    patient_prevalence = _state_prevalence(
        obs,
        n_states=n_states,
        state_column="_stride_plot_state_id",
        identity_columns=[patient_key],
    )
    fov_identity_columns = [patient_key, fov_key]
    if time_key is not None and time_key in obs:
        fov_identity_columns = [patient_key, time_key, fov_key]
    fov_prevalence = _state_prevalence(
        obs,
        n_states=n_states,
        state_column="_stride_plot_state_id",
        identity_columns=fov_identity_columns,
    )

    cell_plot = cell_fraction[list(state_order_tuple), :]
    domain_plot = domain_fraction.iloc[list(state_order_tuple)]
    time_plot = time_fraction.iloc[list(state_order_tuple)] if time_fraction is not None else None
    patient_plot = patient_prevalence[list(state_order_tuple)]
    fov_plot = fov_prevalence[list(state_order_tuple)]

    if figsize is None:
        figsize = _default_figsize("community_annotation", len(state_order_tuple), len(cell_types))
    heatmap_df = pd.DataFrame(cell_plot, index=ordered_labels, columns=cell_types)
    categorical_annotations: dict[str, pd.Series] = {
        "Domain": _dominant_category(domain_plot),
    }
    if time_plot is not None:
        categorical_annotations["Timepoint"] = _dominant_category(time_plot)
    continuous_annotations = pd.DataFrame(
        {
            "Patient prevalence": patient_plot,
            "FOV prevalence": fov_plot,
        },
        index=ordered_labels,
    )

    fig = plt.figure(figsize=figsize)
    _draw_annotated_heatmap(
        fig,
        heatmap_df,
        show_rownames=True,
        show_colnames=True,
        xlabel="Cell subtype",
        ylabel="Community",
        colorbar_label="Cell subtype fraction",
        vmin=0.0,
        vmax=1.0,
        categorical_annotations=categorical_annotations,
        continuous_annotations=continuous_annotations,
    )
    return _save_or_return_figure(fig, save)


def fov_composition_heatmap(
    adata: AnnData,
    *,
    state_labels: Sequence[str] | None = None,
    state_order: Sequence[int] | None = None,
    sort_by: Sequence[str] | None = (
        OBS_TIMEPOINT_KEY,
        OBS_DOMAIN_KEY,
        OBS_PATIENT_KEY,
        OBS_FOV_KEY,
    ),
    patient_groups: Mapping[str, str] | pd.Series | pd.DataFrame | None = None,
    group_key: str = "group",
    show_patient_annotation: bool = False,
    show_fov_labels: bool = False,
    figsize: tuple[float, float] | None = None,
    save: str | Path | None = None,
) -> Figure | None:
    """Display the FOV-level community-composition handoff consumed by `.tl`.

    Scientific question:
        What FOV-level observation matrix is passed from `.pp` to `.tl.fit`,
        and are source/target, domain, group, or patient annotations visibly
        imbalanced at the observation layer?

    Required input:
        The function reads
        `adata.uns["stride"]["fov_observations"]["community_composition"]`
        with shape `[n_fov, K]` and aligned metadata from
        `adata.uns["stride"]["fov_observations"]["metadata"]`. Metadata must
        include `patient_id`, `timepoint`, `fov_id`, and `domain_label`.
        Matrix rows must be finite, nonnegative, and normalized to sum to one.

    Display behavior:
        The main heatmap is FOV by community fraction. Metadata sorting affects
        only display order and never rewrites the AnnData slot. Timepoint and
        domain strips are shown by default. Group strips are shown when
        `patient_groups` is supplied. Patient annotation and FOV labels are
        opt-in because they can become unreadable for many patients or FOVs.

    Ordering and output:
        States default to `0..K-1`; `state_order`, if supplied, must be a full
        permutation. With `save=None`, the function returns a matplotlib
        `Figure`. With `save="*.pdf"`, it writes a local PDF, closes the
        figure, and returns `None`.

    Boundary:
        This directly displays the observation slot. It does not evaluate fit
        quality, infer biological transition, cluster FOVs, compute statistics,
        or mutate `adata`.
    """
    _require_anndata(adata)
    matrix, metadata = _read_fov_observations(adata)
    n_states = matrix.shape[1]
    _check_config_n_states_if_present(adata, n_states)
    state_order_tuple = _resolve_state_order(n_states, state_order)
    labels = _resolve_state_labels(n_states, state_labels)
    ordered_labels = [labels[state] for state in state_order_tuple]

    metadata_plot = metadata.copy()
    group_series = _normalize_patient_groups(
        patient_groups,
        patient_ids=metadata_plot[OBS_PATIENT_KEY].astype(str).tolist(),
        group_key=group_key,
    )
    if group_series is not None:
        metadata_plot[group_key] = metadata_plot[OBS_PATIENT_KEY].astype(str).map(group_series)

    if sort_by is None:
        row_order = np.arange(matrix.shape[0])
    else:
        missing_sort = [column for column in sort_by if column not in metadata_plot]
        if missing_sort:
            raise ContractError("sort_by columns are missing from FOV metadata: " + ", ".join(missing_sort))
        row_order = metadata_plot.reset_index(drop=True).sort_values(list(sort_by), kind="mergesort").index.to_numpy()

    matrix_plot = matrix[row_order][:, list(state_order_tuple)]
    metadata_plot = metadata_plot.iloc[row_order].reset_index(drop=True)
    annotation_fields = [OBS_TIMEPOINT_KEY, OBS_DOMAIN_KEY]
    if group_series is not None:
        annotation_fields.append(group_key)
    if show_patient_annotation:
        annotation_fields.append(OBS_PATIENT_KEY)

    if figsize is None:
        figsize = _default_figsize("fov_composition", matrix.shape[0], n_states)
    if show_fov_labels:
        row_labels = [
            f"{row[OBS_PATIENT_KEY]}|{row[OBS_TIMEPOINT_KEY]}|{row[OBS_DOMAIN_KEY]}|{row[OBS_FOV_KEY]}"
            for _, row in metadata_plot.iterrows()
        ]
    else:
        row_labels = [f"FOV{index}" for index in range(matrix_plot.shape[0])]
    heatmap_df = pd.DataFrame(matrix_plot, index=row_labels, columns=ordered_labels)
    categorical_annotations = {
        _annotation_display_label(field, group_key=group_key): metadata_plot[field].astype(str)
        for field in annotation_fields
    }
    vmax = _composition_heatmap_vmax(matrix_plot)

    fig = plt.figure(figsize=figsize)
    _draw_annotated_heatmap(
        fig,
        heatmap_df,
        show_rownames=show_fov_labels,
        show_colnames=True,
        xlabel="Community",
        ylabel="FOV",
        colorbar_label="Community fraction",
        vmin=0.0,
        vmax=vmax,
        categorical_annotations=categorical_annotations,
        continuous_annotations=None,
    )

    return _save_or_return_figure(fig, save)


def community_fraction_comparison(
    adata: AnnData,
    *,
    scale: Literal[
        "fov_state_fraction_mean",
        "cell_state_fraction",
    ] = "fov_state_fraction_mean",
    relation_ids: Sequence[str] | None = None,
    state_key: str = OBS_STATE_ID_KEY,
    patient_key: str = OBS_PATIENT_KEY,
    time_key: str = OBS_TIMEPOINT_KEY,
    domain_key: str = OBS_DOMAIN_KEY,
    fov_key: str = OBS_FOV_KEY,
    group_labels: Mapping[str, str] | pd.Series | pd.DataFrame | None = None,
    group_key: str = "group",
    stats: pd.DataFrame | None = None,
    state_labels: Sequence[str] | None = None,
    state_order: Sequence[int] | None = None,
    plot_kind: Literal["box", "violin", "box_strip"] = "box_strip",
    paired: bool = True,
    facet_by_relation: bool = True,
    show_ns: bool = True,
    stats_x1_key: str = "x1",
    stats_x2_key: str = "x2",
    figsize: tuple[float, float] | None = None,
    save: str | Path | None = None,
) -> Figure | None:
    """Plot patient-level community-fraction baselines for relation sides.

    Scientific question:
        Are source/target or relation-domain differences already visible in
        direct community fractions before interpreting fitted STRIDE relation
        quantities? If caller-provided groups are supplied, are group patterns
        visible in this descriptive composition baseline?

    Default data route:
        With `scale="fov_state_fraction_mean"`, the function reads the FOV
        observation slot and averages FOV community-composition vectors within
        `patient_id x relation_id x side`. Relation sides are resolved from
        `adata.uns["stride"]["config"]["source"]`, `target`, `relations`, and
        `relation_ids`.

    Cell-level route:
        With `scale="cell_state_fraction"`, the function reads cell-level
        state ids from `adata.obs[state_key]` and computes patient-level
        state fractions after filtering cells to each relation side. The y-axis
        label records whether the plotted quantity is FOV-mean or cell-level.

    Display behavior:
        The x-axis is community, the y-axis is fraction, and hue is relation
        side (`source`/`target`). Panels default to relation id; when
        `group_labels` are supplied, panels are relation id by group. Patients
        missing one side can still contribute the available side. Paired lines
        connect only patient/community pairs with both sides present.

    External statistics:
        `stats` is an optional table of precomputed annotations. `.pl` only
        renders supplied labels, p-values, or q-values; it does not compute
        tests, effect sizes, correction procedures, or evidence. The default
        stats schema uses `community_id`, `x1`, `x2`, optional `relation_id`,
        optional `group`, and optional `y_position`. `stats_x1_key` and
        `stats_x2_key` expose endpoint mappings so external table schemas can
        be used without renaming columns. In the current side-hue layout,
        bracket endpoints must be `source` or `target`.

    Ordering and output:
        States default to `0..K-1`; `state_order`, if supplied, must be a full
        permutation. With `save=None`, the function returns a matplotlib
        `Figure`. With `save="*.pdf"`, it writes a local PDF, closes the
        figure, and returns `None`.

    Boundary:
        This is a descriptive composition baseline. It does not estimate or
        display `A`, `d`, or `e`; it does not replace `.da` statistical
        analysis; and it does not mutate `adata` or write result payloads.
    """
    _require_anndata(adata)
    if scale not in {"fov_state_fraction_mean", "cell_state_fraction"}:
        raise ContractError("scale must be 'fov_state_fraction_mean' or 'cell_state_fraction'")
    if plot_kind not in {"box", "violin", "box_strip"}:
        raise ContractError("plot_kind must be 'box', 'violin', or 'box_strip'")

    relation_records = _resolve_relation_records(adata, relation_ids=relation_ids)
    if scale == "fov_state_fraction_mean":
        fraction_table, n_states = _fraction_table_from_fov(adata, relation_records)
        y_label = "Community fraction (FOV mean)"
    else:
        fraction_table, n_states = _fraction_table_from_cells(
            adata,
            relation_records,
            state_key=state_key,
            patient_key=patient_key,
            time_key=time_key,
            domain_key=domain_key,
        )
        y_label = "Community fraction (cell-level)"
    if fraction_table.empty:
        raise ContractError("community fraction comparison has no plottable rows")

    state_order_tuple = _resolve_state_order(n_states, state_order)
    labels = _resolve_state_labels(n_states, state_labels)
    ordered_labels = [labels[state] for state in state_order_tuple]
    fraction_table = fraction_table[
        fraction_table["community_id"].isin(state_order_tuple)
    ].copy()
    fraction_table["community_label"] = pd.Categorical(
        fraction_table["community_id"].map(lambda value: labels[int(value)]),
        categories=ordered_labels,
        ordered=True,
    )

    group_series = _normalize_patient_groups(
        group_labels,
        patient_ids=fraction_table["patient_id"].astype(str).tolist(),
        group_key=group_key,
    )
    panel_columns = ["relation_id"] if facet_by_relation else []
    if group_series is not None:
        fraction_table[group_key] = fraction_table["patient_id"].astype(str).map(group_series)
        panel_columns.append(group_key)
    if not panel_columns:
        fraction_table["_panel"] = "community fraction"
        panel_columns = ["_panel"]

    panels = fraction_table[panel_columns].drop_duplicates().reset_index(drop=True)
    n_panels = panels.shape[0]
    if n_panels == 0:
        raise ContractError("community fraction comparison has no plottable rows")
    if figsize is None:
        figsize = _default_figsize("fraction_comparison", fraction_table.shape[0], n_states, n_panels)

    n_rows, n_cols = _auto_grid(n_panels, max_cols=3)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True, squeeze=False)
    axes_flat = list(axes.ravel())
    for ax, (_, panel_row) in zip(axes_flat, panels.iterrows(), strict=True):
        panel_data = fraction_table.copy()
        title_parts: list[str] = []
        for column in panel_columns:
            value = panel_row[column]
            panel_data = panel_data[panel_data[column] == value]
            if column != "_panel":
                title_parts.append(str(value))
        _draw_fraction_panel(
            ax,
            panel_data,
            ordered_labels=ordered_labels,
            plot_kind=plot_kind,
            paired=paired,
        )
        if len(panel_columns) > 1 and ax not in axes[-1, :]:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Community")
        ax.set_ylabel(y_label)
        if title_parts:
            ax.set_title(" | ".join(title_parts), fontsize=10)
        _apply_clean_axes_style(ax)
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    if stats is not None:
        _draw_stats_annotations(
            axes_flat[:n_panels],
            panels,
            panel_columns,
            fraction_table,
            stats,
            ordered_labels=ordered_labels,
            show_ns=show_ns,
            stats_x1_key=stats_x1_key,
            stats_x2_key=stats_x2_key,
        )

    _apply_figure_margins(fig, left=0.08, right=0.98, bottom=0.24, top=0.90, wspace=0.34, hspace=0.70)
    return _save_or_return_figure(fig, save)


def _require_anndata(adata: object) -> None:
    if not isinstance(adata, AnnData):
        raise ContractError("adata must be an AnnData object")


def _require_pycomplexheatmap() -> dict[str, Any]:
    try:
        from PyComplexHeatmap import (
            ClusterMapPlotter,
            HeatmapAnnotation,
            anno_barplot,
            anno_simple,
            composite,
        )
    except ModuleNotFoundError as exc:
        raise ContractError(
            "PyComplexHeatmap is required for STRIDE descriptive heatmap plots"
        ) from exc
    return {
        "ClusterMapPlotter": ClusterMapPlotter,
        "HeatmapAnnotation": HeatmapAnnotation,
        "anno_barplot": anno_barplot,
        "anno_simple": anno_simple,
        "composite": composite,
    }


def _annotation_display_label(field: str, *, group_key: str) -> str:
    """Return compact annotation strip labels for heatmap panels."""
    labels = {
        OBS_TIMEPOINT_KEY: "Time",
        OBS_DOMAIN_KEY: "Domain",
        OBS_PATIENT_KEY: "Patient",
    }
    if str(field) == str(group_key):
        return "Group"
    return labels.get(str(field), str(field))


def _draw_annotated_heatmap(
    fig: Figure,
    data: pd.DataFrame,
    *,
    show_rownames: bool,
    show_colnames: bool,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    vmin: float,
    vmax: float,
    categorical_annotations: Mapping[str, pd.Series],
    continuous_annotations: pd.DataFrame | None,
) -> None:
    n_categorical = len(categorical_annotations)
    n_continuous = 0 if continuous_annotations is None else continuous_annotations.shape[1]
    width_ratios = [
        max(4.0, data.shape[1] * 0.42),
        *([0.34] * n_categorical),
        *([0.62] * n_continuous),
        0.24,
        1.80,
    ]
    grid = fig.add_gridspec(
        1,
        len(width_ratios),
        width_ratios=width_ratios,
        left=0.08,
        right=0.96,
        bottom=0.22,
        top=0.90,
        wspace=0.18,
    )
    heatmap_ax = fig.add_subplot(grid[0, 0])
    cmap = _bio_continuous_cmap()
    image = heatmap_ax.imshow(data.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    if show_colnames:
        heatmap_ax.set_xticks(np.arange(data.shape[1]), labels=list(data.columns))
        for label in heatmap_ax.get_xticklabels():
            label.set_rotation(90)
            label.set_ha("right")
            label.set_va("center")
            label.set_rotation_mode("anchor")
    else:
        heatmap_ax.set_xticks([])
    if show_rownames:
        heatmap_ax.set_yticks(np.arange(data.shape[0]), labels=list(data.index))
    else:
        heatmap_ax.set_yticks([])
    heatmap_ax.set_xlabel(xlabel)
    heatmap_ax.set_ylabel(ylabel)
    _apply_clean_axes_style(heatmap_ax)

    legend_groups: list[tuple[str, list[Patch], list[str]]] = []
    col_index = 1
    for annotation_name, values in categorical_annotations.items():
        ax = fig.add_subplot(grid[0, col_index], sharey=heatmap_ax)
        color_map = _categorical_color_map(values.astype(str))
        _draw_categorical_strip(ax, values, color_map=color_map, label=annotation_name)
        handles: list[Patch] = []
        labels: list[str] = []
        for value, color in color_map.items():
            handles.append(Patch(facecolor=color, edgecolor="none"))
            labels.append(str(value))
        legend_groups.append((str(annotation_name), handles, labels))
        col_index += 1

    if continuous_annotations is not None:
        for column in continuous_annotations.columns:
            ax = fig.add_subplot(grid[0, col_index], sharey=heatmap_ax)
            _draw_continuous_bar_annotation(
                ax,
                continuous_annotations[column].to_numpy(dtype=float),
                label=str(column),
            )
            col_index += 1

    cbar_ax = fig.add_subplot(grid[0, col_index])
    colorbar = fig.colorbar(image, cax=cbar_ax)
    colorbar.set_label(colorbar_label, labelpad=10)

    legend_ax = fig.add_subplot(grid[0, col_index + 1])
    legend_ax.axis("off")
    _draw_grouped_legends(legend_ax, legend_groups)


def _draw_grouped_legends(
    ax,
    legend_groups: Sequence[tuple[str, list[Patch], list[str]]],
) -> None:
    """Draw categorical annotation legends as separated groups."""
    y_anchor = 1.0
    for title, handles, labels in legend_groups:
        if not handles:
            continue
        legend = ax.legend(
            handles,
            labels,
            title=title,
            loc="upper left",
            bbox_to_anchor=(0.0, y_anchor),
            frameon=False,
            fontsize=7,
            title_fontsize=8,
            handlelength=0.9,
            handletextpad=0.4,
            borderaxespad=0.0,
            labelspacing=0.28,
        )
        ax.add_artist(legend)
        y_anchor -= min(0.30, 0.08 + 0.045 * len(labels))

def _draw_categorical_strip(
    ax,
    values: pd.Series,
    *,
    color_map: Mapping[str, str],
    label: str,
) -> None:
    labels = values.astype(str).tolist()
    color_ids = {value: index for index, value in enumerate(color_map)}
    matrix = np.asarray([[color_ids[value]] for value in labels], dtype=float)
    cmap = ListedColormap([color_map[value] for value in color_map])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-0.5, vmax=max(len(color_map) - 0.5, 0.5))
    ax.set_xticks([0], labels=[label])
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
        tick.set_ha("right")
        tick.set_va("center")
        tick.set_rotation_mode("anchor")
    ax.tick_params(axis="y", left=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_continuous_bar_annotation(
    ax,
    values: np.ndarray,
    *,
    label: str,
) -> None:
    y = np.arange(values.shape[0])
    ax.barh(y, values, color=BIO_PASTEL_PALETTE["teal"], height=0.78)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 1.0])
    ax.set_xlabel(label, fontsize=8)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.invert_yaxis()
    _apply_clean_axes_style(ax)


def _dominant_category(fractions: pd.DataFrame) -> pd.Series:
    """Return dominant category per community from row-normalized fractions."""
    if fractions.empty:
        return pd.Series(dtype=str)
    return fractions.idxmax(axis=1).astype(str)


def _composition_heatmap_vmax(matrix: np.ndarray) -> float:
    """Return a contrast-preserving FOV composition color scale upper bound."""
    percentile = float(np.nanpercentile(matrix, 99))
    if not np.isfinite(percentile):
        return 1.0
    return min(1.0, max(0.25, percentile))


def _require_obs_columns(adata: AnnData, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in adata.obs]
    if missing:
        raise ContractError("adata.obs is missing required columns: " + ", ".join(missing))


def _read_config(adata: AnnData) -> Mapping[str, object]:
    stride_uns = adata.uns.get(STRIDE_UNS_KEY)
    if not isinstance(stride_uns, Mapping):
        raise ContractError("adata.uns['stride'] must be a mapping")
    config = stride_uns.get(STRIDE_CONFIG_KEY)
    if not isinstance(config, Mapping):
        raise ContractError("adata.uns['stride']['config'] must be a mapping")
    return config


def _resolve_n_states(adata: AnnData, *, state_ids: np.ndarray) -> int:
    stride_uns = adata.uns.get(STRIDE_UNS_KEY)
    if isinstance(stride_uns, Mapping):
        config = stride_uns.get(STRIDE_CONFIG_KEY)
        if isinstance(config, Mapping) and "n_states" in config:
            value = config["n_states"]
            if (
                not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_))
                or int(value) <= 0
            ):
                raise ContractError("config['n_states'] must be a positive integer")
            return int(value)
    if state_ids.size == 0:
        raise ContractError("cannot infer n_states from empty state ids")
    return int(state_ids.max()) + 1


def _validate_state_ids(
    values: object,
    *,
    n_states: int | None = None,
    n_obs: int | None = None,
) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ContractError("state ids must be numeric") from exc
    if arr.ndim != 1:
        raise ContractError("state ids must be a 1D array")
    if n_obs is not None and arr.shape[0] != n_obs:
        raise ContractError("state ids length must align to adata.n_obs")
    if not np.all(np.isfinite(arr)):
        raise ContractError("state ids must be finite")
    if not np.all(np.equal(arr, np.floor(arr))):
        raise ContractError("state ids must be integer-compatible")
    ids = arr.astype(int)
    if np.any(ids < 0):
        raise ContractError("state ids must be nonnegative")
    if n_states is not None and np.any(ids >= n_states):
        raise ContractError("state ids values must be in [0, n_states - 1]")
    return ids


def _resolve_state_order(
    n_states: int,
    state_order: Sequence[int] | None = None,
) -> tuple[int, ...]:
    try:
        return resolve_full_axis_order(n_states, state_order, name="state_order")
    except ContractError as exc:
        raise ContractError("state_order must be a complete permutation of 0..K-1") from exc


def _resolve_state_labels(
    n_states: int,
    state_labels: Sequence[str] | None = None,
) -> tuple[str, ...]:
    try:
        return resolve_axis_labels(
            n_states,
            state_labels,
            name="state_labels",
            prefix="C",
        )
    except ContractError as exc:
        raise ContractError("state_labels length must match n_states") from exc


def _row_fraction(matrix: np.ndarray) -> np.ndarray:
    result = np.zeros_like(matrix, dtype=float)
    row_sums = matrix.sum(axis=1)
    nonempty = row_sums > 0
    result[nonempty] = matrix[nonempty] / row_sums[nonempty, None]
    return result


def _state_category_fraction(
    obs: pd.DataFrame,
    *,
    n_states: int,
    state_column: str,
    category_column: str,
) -> pd.DataFrame:
    categories = _ordered_unique(obs[category_column])
    counts = pd.crosstab(obs[state_column], obs[category_column])
    counts = counts.reindex(index=range(n_states), columns=categories, fill_value=0)
    return pd.DataFrame(
        _row_fraction(counts.to_numpy(dtype=float)),
        index=counts.index,
        columns=counts.columns,
    )


def _state_prevalence(
    obs: pd.DataFrame,
    *,
    n_states: int,
    state_column: str,
    identity_columns: Sequence[str],
) -> np.ndarray:
    denominator = obs[list(identity_columns)].drop_duplicates().shape[0]
    result = np.zeros(n_states, dtype=float)
    if denominator == 0:
        return result
    for state in range(n_states):
        state_rows = obs[obs[state_column] == state]
        result[state] = state_rows[list(identity_columns)].drop_duplicates().shape[0] / denominator
    return result


def _draw_stacked_barh(ax, fractions: pd.DataFrame, label: str) -> None:
    colors = _categorical_color_map(fractions.columns)
    left = np.zeros(fractions.shape[0], dtype=float)
    y = np.arange(fractions.shape[0])
    for column in fractions.columns:
        values = fractions[column].to_numpy(dtype=float)
        ax.barh(y, values, left=left, color=colors[str(column)], height=0.8, linewidth=0)
        left += values
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 1.0])
    ax.set_yticks([])
    ax.set_xlabel(label)
    ax.invert_yaxis()
    handles = [Patch(facecolor=colors[str(column)], label=str(column)) for column in fractions.columns]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=7)
    _apply_clean_axes_style(ax)


def _draw_prevalence_barh(ax, values: np.ndarray, label: str) -> None:
    y = np.arange(values.shape[0])
    ax.barh(y, values, color=BIO_PASTEL_PALETTE["teal"], height=0.8)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 1.0])
    ax.set_yticks([])
    ax.set_xlabel(label)
    ax.invert_yaxis()
    _apply_clean_axes_style(ax)


def _draw_annotation_strip(ax, values: pd.Series, field: str) -> None:
    labels = values.astype(str).tolist()
    colors = _categorical_color_map(labels)
    color_ids = {label: index for index, label in enumerate(colors)}
    data = np.asarray([[color_ids[label] for label in labels]], dtype=float)
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap([colors[label] for label in color_ids])
    ax.imshow(data, aspect="auto", cmap=cmap, vmin=-0.5, vmax=max(len(color_ids) - 0.5, 0.5))
    ax.set_yticks([0], labels=[field])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _read_fov_observations(adata: AnnData) -> tuple[np.ndarray, pd.DataFrame]:
    stride_uns = adata.uns.get(STRIDE_UNS_KEY)
    if not isinstance(stride_uns, Mapping):
        raise ContractError("adata.uns['stride'] must be a mapping")
    fov_observations = stride_uns.get(STRIDE_FOV_OBSERVATIONS_KEY)
    if not isinstance(fov_observations, Mapping):
        raise ContractError("adata.uns['stride']['fov_observations'] must be a mapping")
    if "community_composition" not in fov_observations:
        raise ContractError("fov_observations['community_composition'] is required")
    if "metadata" not in fov_observations:
        raise ContractError("fov_observations['metadata'] is required")

    matrix = np.asarray(fov_observations["community_composition"], dtype=float)
    if matrix.ndim != 2:
        raise ContractError("fov_observations['community_composition'] must be a 2D matrix")
    if not np.all(np.isfinite(matrix)):
        raise ContractError("fov_observations['community_composition'] must be finite")
    if np.any(matrix < 0):
        raise ContractError("fov_observations['community_composition'] must be nonnegative")
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise ContractError("fov_observations['community_composition'] rows must sum to 1")

    metadata = fov_observations["metadata"]
    if not isinstance(metadata, pd.DataFrame):
        raise ContractError("fov_observations['metadata'] must be a pandas DataFrame")
    required = [OBS_PATIENT_KEY, OBS_TIMEPOINT_KEY, OBS_FOV_KEY, OBS_DOMAIN_KEY]
    missing = [column for column in required if column not in metadata]
    if missing:
        raise ContractError("fov_observations['metadata'] is missing required columns: " + ", ".join(missing))
    if metadata.shape[0] != matrix.shape[0]:
        raise ContractError(
            "fov_observations['metadata'] row count must match "
            "fov_observations['community_composition'] row count"
        )
    return matrix.copy(), metadata.copy()


def _check_config_n_states_if_present(adata: AnnData, n_states: int) -> None:
    stride_uns = adata.uns.get(STRIDE_UNS_KEY)
    if not isinstance(stride_uns, Mapping):
        return
    config = stride_uns.get(STRIDE_CONFIG_KEY)
    if not isinstance(config, Mapping) or "n_states" not in config:
        return
    value = config["n_states"]
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or int(value) <= 0
    ):
        raise ContractError("config['n_states'] must be a positive integer")
    if int(value) != n_states:
        raise ContractError("config['n_states'] must match FOV community_composition columns")


def _normalize_patient_groups(
    group_labels: Mapping[str, str] | pd.Series | pd.DataFrame | None,
    *,
    patient_ids: Sequence[str],
    group_key: str,
) -> pd.Series | None:
    if group_labels is None:
        return None
    if isinstance(group_labels, pd.DataFrame):
        if group_key not in group_labels:
            raise ContractError(f"group label DataFrame must contain {group_key!r}")
        if OBS_PATIENT_KEY in group_labels:
            series = group_labels.set_index(OBS_PATIENT_KEY)[group_key]
        else:
            series = group_labels[group_key]
    elif isinstance(group_labels, pd.Series):
        series = group_labels
    elif isinstance(group_labels, Mapping):
        series = pd.Series(dict(group_labels), dtype=object)
    else:
        raise ContractError("patient group labels must be a mapping, Series, or DataFrame")
    series = series.astype(str)
    needed = set(str(patient_id) for patient_id in patient_ids)
    missing = sorted(patient_id for patient_id in needed if patient_id not in series.index.astype(str))
    if missing:
        raise ContractError("patient group labels are missing plotted patients: " + ", ".join(missing))
    series.index = series.index.astype(str)
    return series


def _resolve_relation_records(
    adata: AnnData,
    *,
    relation_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    config = _read_config(adata)
    for key in ("source", "target", STRIDE_RELATIONS_KEY, STRIDE_RELATION_IDS_KEY):
        if key not in config:
            raise ContractError(f"config[{key!r}] is required")
    source = str(config["source"])
    target = str(config["target"])
    relations = np.asarray(config[STRIDE_RELATIONS_KEY], dtype=object)
    if relations.ndim != 2 or relations.shape[1] != 2:
        raise ContractError("config['relations'] shape must be [n_relations, 2]")
    raw_ids = config[STRIDE_RELATION_IDS_KEY]
    if isinstance(raw_ids, str):
        raise ContractError("config['relation_ids'] must be a sequence, not a string")
    ids = [str(value) for value in raw_ids]
    if len(ids) != relations.shape[0]:
        raise ContractError("config['relation_ids'] length must match config['relations'] row count")
    records = pd.DataFrame(
        {
            "relation_id": ids,
            "source_timepoint": source,
            "target_timepoint": target,
            "source_domain": relations[:, 0].astype(str),
            "target_domain": relations[:, 1].astype(str),
        }
    )
    if relation_ids is None:
        return records
    requested = [str(value) for value in relation_ids]
    missing = [value for value in requested if value not in set(records["relation_id"])]
    if missing:
        raise ContractError("unknown relation_id values: " + ", ".join(missing))
    return records.set_index("relation_id").loc[requested].reset_index()


def _fraction_table_from_fov(
    adata: AnnData,
    relation_records: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    matrix, metadata = _read_fov_observations(adata)
    n_states = matrix.shape[1]
    metadata = metadata.reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for record in relation_records.to_dict("records"):
        side_filters = {
            "source": (
                (metadata[OBS_TIMEPOINT_KEY].astype(str) == record["source_timepoint"])
                & (metadata[OBS_DOMAIN_KEY].astype(str) == record["source_domain"])
            ),
            "target": (
                (metadata[OBS_TIMEPOINT_KEY].astype(str) == record["target_timepoint"])
                & (metadata[OBS_DOMAIN_KEY].astype(str) == record["target_domain"])
            ),
        }
        for side, mask in side_filters.items():
            side_metadata = metadata.loc[mask]
            for patient_id, patient_rows in side_metadata.groupby(OBS_PATIENT_KEY, sort=False):
                indices = patient_rows.index.to_numpy()
                values = matrix[indices].mean(axis=0)
                for community_id, fraction in enumerate(values):
                    rows.append(
                        {
                            "relation_id": record["relation_id"],
                            "patient_id": str(patient_id),
                            "side": side,
                            "community_id": community_id,
                            "fraction": float(fraction),
                        }
                    )
    return pd.DataFrame(rows), n_states


def _fraction_table_from_cells(
    adata: AnnData,
    relation_records: pd.DataFrame,
    *,
    state_key: str,
    patient_key: str,
    time_key: str,
    domain_key: str,
) -> tuple[pd.DataFrame, int]:
    _require_obs_columns(adata, [state_key, patient_key, time_key, domain_key])
    obs = adata.obs.copy()
    state_ids = _validate_state_ids(obs[state_key], n_obs=adata.n_obs)
    n_states = _resolve_n_states(adata, state_ids=state_ids)
    state_ids = _validate_state_ids(obs[state_key], n_states=n_states, n_obs=adata.n_obs)
    obs["_stride_plot_state_id"] = state_ids
    rows: list[dict[str, object]] = []
    for record in relation_records.to_dict("records"):
        side_filters = {
            "source": (
                (obs[time_key].astype(str) == record["source_timepoint"])
                & (obs[domain_key].astype(str) == record["source_domain"])
            ),
            "target": (
                (obs[time_key].astype(str) == record["target_timepoint"])
                & (obs[domain_key].astype(str) == record["target_domain"])
            ),
        }
        for side, mask in side_filters.items():
            side_obs = obs.loc[mask]
            for patient_id, patient_rows in side_obs.groupby(patient_key, sort=False):
                counts = patient_rows["_stride_plot_state_id"].value_counts().reindex(range(n_states), fill_value=0)
                fractions = counts.to_numpy(dtype=float) / float(counts.sum())
                for community_id, fraction in enumerate(fractions):
                    rows.append(
                        {
                            "relation_id": record["relation_id"],
                            "patient_id": str(patient_id),
                            "side": side,
                            "community_id": community_id,
                            "fraction": float(fraction),
                        }
                    )
    return pd.DataFrame(rows), n_states


def _draw_fraction_panel(
    ax,
    panel_data: pd.DataFrame,
    *,
    ordered_labels: Sequence[str],
    plot_kind: str,
    paired: bool,
) -> None:
    try:
        import seaborn as sns
    except ModuleNotFoundError:
        sns = None

    if sns is None:
        _draw_fraction_panel_matplotlib(
            ax,
            panel_data,
            ordered_labels=ordered_labels,
            plot_kind=plot_kind,
            paired=paired,
        )
        return

    common = {
        "data": panel_data,
        "x": "community_label",
        "y": "fraction",
        "hue": "side",
        "order": ordered_labels,
        "hue_order": ["source", "target"],
        "palette": SOURCE_TARGET_PALETTE,
        "ax": ax,
    }
    if plot_kind == "violin":
        sns.violinplot(**common, cut=0, inner=None)
    else:
        sns.boxplot(**common, fliersize=0, width=0.65)
        if plot_kind == "box_strip":
            sns.stripplot(**common, dodge=True, jitter=0.12, size=3.0, linewidth=0.3, edgecolor="#4A4A4A")
    if paired:
        _draw_paired_lines(ax, panel_data, ordered_labels)
    handles, labels = ax.get_legend_handles_labels()
    dedup: dict[str, object] = {}
    for handle, label in zip(handles, labels, strict=False):
        if label in {"source", "target"} and label not in dedup:
            dedup[label] = handle
    if dedup:
        ax.legend(dedup.values(), dedup.keys(), frameon=False, fontsize=8, title=None)
    elif ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_ylim(bottom=0.0)
    _set_fraction_xticklabels(ax)


def _draw_fraction_panel_matplotlib(
    ax,
    panel_data: pd.DataFrame,
    *,
    ordered_labels: Sequence[str],
    plot_kind: str,
    paired: bool,
) -> None:
    offsets = {"source": -0.18, "target": 0.18}
    width = 0.28
    for base_index, label in enumerate(ordered_labels):
        for side in ("source", "target"):
            values = panel_data[
                (panel_data["community_label"] == label) & (panel_data["side"] == side)
            ]["fraction"].to_numpy(dtype=float)
            if values.size == 0:
                continue
            x = base_index + offsets[side]
            if plot_kind in {"box", "box_strip"}:
                ax.boxplot(
                    values,
                    positions=[x],
                    widths=width,
                    patch_artist=True,
                    showfliers=False,
                    boxprops={"facecolor": SOURCE_TARGET_PALETTE[side], "alpha": 0.65},
                    medianprops={"color": "#4A4A4A", "linewidth": 0.8},
                    whiskerprops={"color": "#6F7378", "linewidth": 0.7},
                    capprops={"color": "#6F7378", "linewidth": 0.7},
                )
            else:
                ax.scatter(
                    np.full(values.shape, x),
                    values,
                    s=24,
                    color=SOURCE_TARGET_PALETTE[side],
                    alpha=0.75,
                    edgecolor="#4A4A4A",
                    linewidth=0.3,
                )
            if plot_kind == "box_strip":
                jitter = np.linspace(-0.03, 0.03, values.size) if values.size > 1 else np.zeros(values.size)
                ax.scatter(
                    np.full(values.shape, x) + jitter,
                    values,
                    s=18,
                    color=SOURCE_TARGET_PALETTE[side],
                    alpha=0.85,
                    edgecolor="#4A4A4A",
                    linewidth=0.3,
                    zorder=3,
                )
    if paired:
        _draw_paired_lines(ax, panel_data, ordered_labels)
    ax.set_xticks(np.arange(len(ordered_labels)), labels=ordered_labels)
    ax.legend(
        handles=[
            Patch(facecolor=SOURCE_TARGET_PALETTE["source"], label="source", alpha=0.65),
            Patch(facecolor=SOURCE_TARGET_PALETTE["target"], label="target", alpha=0.65),
        ],
        frameon=False,
        fontsize=8,
        title=None,
    )
    ax.set_ylim(bottom=0.0)
    _set_fraction_xticklabels(ax)


def _set_fraction_xticklabels(ax) -> None:
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_ha("right")
        label.set_va("center")
        label.set_rotation_mode("anchor")


def _draw_paired_lines(ax, panel_data: pd.DataFrame, ordered_labels: Sequence[str]) -> None:
    offsets = {"source": -0.20, "target": 0.20}
    x_positions = {label: index for index, label in enumerate(ordered_labels)}
    for (_, community_label, _patient_id), rows in panel_data.groupby(
        ["relation_id", "community_label", "patient_id"],
        observed=True,
        sort=False,
    ):
        by_side = rows.set_index("side")
        if {"source", "target"}.issubset(by_side.index):
            base_x = x_positions[str(community_label)]
            ax.plot(
                [base_x + offsets["source"], base_x + offsets["target"]],
                [
                    float(by_side.loc["source", "fraction"]),
                    float(by_side.loc["target", "fraction"]),
                ],
                color="#6F7378",
                linewidth=0.5,
                alpha=0.55,
                zorder=1,
            )


def _draw_stats_annotations(
    axes: Sequence[object],
    panels: pd.DataFrame,
    panel_columns: Sequence[str],
    fraction_table: pd.DataFrame,
    stats: pd.DataFrame,
    *,
    ordered_labels: Sequence[str],
    show_ns: bool,
    stats_x1_key: str,
    stats_x2_key: str,
) -> None:
    required = {"community_id", stats_x1_key, stats_x2_key}
    missing = sorted(required.difference(stats.columns))
    if missing:
        raise ContractError("stats is missing required columns: " + ", ".join(missing))
    panel_lookup: dict[tuple[object, ...], object] = {}
    for ax, (_, row) in zip(axes, panels.iterrows(), strict=True):
        panel_lookup[tuple(row[column] for column in panel_columns)] = ax
    for _, stat in stats.iterrows():
        matches = panels.copy()
        for column in panel_columns:
            if column == "_panel":
                continue
            if column in stat.index and not pd.isna(stat[column]):
                matches = matches[matches[column] == stat[column]]
            elif column in {"relation_id", "group"} and panels.shape[0] > 1:
                raise ContractError(f"stats row must specify {column!r} for multi-panel plots")
        if matches.shape[0] != 1:
            raise ContractError("stats row must match exactly one plotted panel")
        panel_key = tuple(matches.iloc[0][column] for column in panel_columns)
        ax = panel_lookup[panel_key]
        community_id = int(stat["community_id"])
        community_label_rows = fraction_table[fraction_table["community_id"] == community_id]
        if community_label_rows.empty:
            raise ContractError("stats community_id does not match plotted data")
        community_label = str(community_label_rows["community_label"].iloc[0])
        if community_label not in ordered_labels:
            raise ContractError("stats community_id does not match plotted state_order")
        label = _stat_display_label(stat, show_ns=show_ns)
        if label is None:
            continue
        side1 = _resolve_stats_side(stat, side_key=stats_x1_key)
        side2 = _resolve_stats_side(stat, side_key=stats_x2_key)
        x1 = _side_x_position(community_label, side1, ordered_labels)
        x2 = _side_x_position(community_label, side2, ordered_labels)
        panel_mask = np.ones(fraction_table.shape[0], dtype=bool)
        for column in panel_columns:
            if column == "_panel":
                continue
            panel_mask &= fraction_table[column].to_numpy() == matches.iloc[0][column]
        panel_data = fraction_table.loc[panel_mask & (fraction_table["community_id"] == community_id)]
        if panel_data.empty:
            raise ContractError("stats row does not match plotted data")
        if "y_position" in stat.index and not pd.isna(stat["y_position"]):
            y = float(stat["y_position"])
        else:
            y = float(panel_data["fraction"].max()) + 0.08
        ax.plot([x1, x1, x2, x2], [y - 0.02, y, y, y - 0.02], color="#4A4A4A", linewidth=0.8)
        ax.text((x1 + x2) / 2.0, y + 0.01, label, ha="center", va="bottom", fontsize=8)
        current_top = ax.get_ylim()[1]
        if y + 0.08 > current_top:
            ax.set_ylim(top=y + 0.08)


def _stat_display_label(row: pd.Series, *, show_ns: bool) -> str | None:
    if "label" in row.index and not pd.isna(row["label"]):
        label = str(row["label"])
        if label == "ns" and not show_ns:
            return None
        return label
    if "q_value" in row.index and not pd.isna(row["q_value"]):
        return _format_pq_label(float(row["q_value"]), "q", show_ns=show_ns)
    if "p_value" in row.index and not pd.isna(row["p_value"]):
        return _format_pq_label(float(row["p_value"]), "p", show_ns=show_ns)
    raise ContractError("stats row must provide label, q_value, or p_value")


def _resolve_stats_side(
    row: pd.Series,
    *,
    side_key: str,
) -> str:
    side = str(row[side_key])
    if side not in {"source", "target"}:
        raise ContractError(f"stats {side_key} values must be 'source' or 'target'")
    return side


def _side_x_position(
    community_label: str,
    side: str,
    ordered_labels: Sequence[str],
) -> float:
    if side not in {"source", "target"}:
        raise ContractError("stats x1/x2 values must be 'source' or 'target'")
    base = ordered_labels.index(community_label)
    return base + (-0.20 if side == "source" else 0.20)
