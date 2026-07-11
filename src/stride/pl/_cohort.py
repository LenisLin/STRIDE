"""Cohort-level relation plotting functions for STRIDE `.pl`."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from stride._array_contracts import (
    as_finite_float_array,
    resolve_axis_labels,
    resolve_full_axis_order,
)
from stride.errors import ContractError
from stride.tl import CohortResult, FitResult, RelationResult

from ._utils import (
    _apply_clean_axes_style,
    _apply_figure_margins,
    _auto_grid,
    _bio_continuous_cmap,
    _save_or_return_figure,
    _set_right_angle_xticklabels,
)


def cohort_relation_heatmap(
    result: CohortResult | RelationResult | FitResult,
    *,
    relation_id: str | None = None,
    state_labels: Sequence[str] | None = None,
    state_order: Sequence[int] | None = None,
    figsize: tuple[float, float] | None = None,
    save: str | Path | None = None,
) -> Figure | None:
    """Plot cohort-supported fitted relations as augmented heatmaps.

    Scientific question:
        What cohort-supported relation structure is visible on the shared
        community axis for one or more declared relations?

    Required input:
        The function consumes `.tl` output only: a `CohortResult`, a
        `RelationResult` with an attached cohort, or a `FitResult`. For
        `FitResult`, `relation_id=None` displays all stored `fit.relation_ids`
        in declared order; an explicit `relation_id` displays only that
        relation.

    Display object:
        Each relation is shown as `M_aug = [[A, d], [e^T, masked]]`, where
        `A` is `[K, K]`, `d` is the source-open column, and `e` is the
        target-open row. The bottom-right cell is display-only and masked; it
        is not a fitted STRIDE variable.

    Ordering and output:
        States default to `0..K-1`; `state_order`, if supplied, must be a full
        permutation. With `save=None`, the function returns a matplotlib
        `Figure`. With `save="*.pdf"`, it writes a local PDF, closes the
        figure, and returns `None`.

    Boundary:
        This is a fitted cohort consensus visualization. It does not fit
        models, compute statistics, derive downstream tables, rank relations,
        cluster relations, or mutate inputs.
    """
    cohorts = _resolve_cohorts(result, relation_id=relation_id)
    cohort_arrays = tuple(_cohort_arrays(cohort) for cohort in cohorts)
    K = _common_n_states(cohort_arrays)
    order = _resolve_state_order(K, state_order)
    labels = _resolve_state_labels(K, state_labels)
    ordered_labels = [labels[index] for index in order]

    if figsize is None:
        figsize = _default_augmented_figsize(len(cohorts), K)

    n_rows, n_cols = _auto_grid(len(cohorts), max_cols=2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = list(axes.ravel())
    images = []
    for ax, cohort, arrays in zip(axes_flat, cohorts, cohort_arrays, strict=True):
        images.append(
            _plot_augmented_cohort(
                ax,
                cohort,
                arrays,
                order=order,
                ordered_labels=ordered_labels,
                show_panel_title=len(cohorts) > 1,
            )
        )
        if len(cohorts) == 1:
            ax.set_title(f"Cohort relation template: {cohort.relation_id}", fontsize=10)
    for ax in axes_flat[len(cohorts) :]:
        ax.set_visible(False)

    cbar_ax = fig.add_axes([0.91, 0.24, 0.018, 0.52])
    colorbar = fig.colorbar(images[0], cax=cbar_ax)
    colorbar.set_label("Template value", labelpad=10)
    _apply_figure_margins(fig, left=0.10, right=0.86, bottom=0.28, top=0.90, wspace=0.42, hspace=0.35)
    return _save_or_return_figure(fig, save)


def _resolve_cohorts(
    result: CohortResult | RelationResult | FitResult,
    *,
    relation_id: str | None,
) -> tuple[CohortResult, ...]:
    """Resolve public `.tl` result containers into cohort summaries.

    Purpose:
        Keep the public plotting function flexible while ensuring that
        multi-relation split views come only from the standard `FitResult`
        relation order.

    Key variables:
        result: public `.tl` result container.
        relation_id: optional selector for one relation.
        cohorts: cohort summaries to display, in plot order.
    """
    if isinstance(result, CohortResult):
        if relation_id is not None and str(relation_id) != result.relation_id:
            raise ContractError("relation_id does not match CohortResult")
        return (result,)

    if isinstance(result, RelationResult):
        if relation_id is not None and str(relation_id) != result.relation_id:
            raise ContractError("relation_id does not match RelationResult")
        if result.cohort is None:
            raise ContractError("RelationResult.cohort is required")
        return (result.cohort,)

    if isinstance(result, FitResult):
        cohorts = []
        for resolved_relation_id in _resolve_fit_relation_ids(result, relation_id):
            relation = result.relations[resolved_relation_id]
            if relation.cohort is None:
                raise ContractError("RelationResult.cohort is required")
            cohorts.append(relation.cohort)
        return tuple(cohorts)

    raise ContractError("result must be a CohortResult, RelationResult, or FitResult")


def _resolve_fit_relation_ids(result: FitResult, relation_id: str | None) -> tuple[str, ...]:
    """Resolve the `FitResult` relation traversal for cohort plotting.

    Purpose:
        Use `FitResult.relation_ids` as the only multi-relation ordering
        surface. A supplied `relation_id` narrows the plot to one relation.
    """
    if relation_id is None:
        relation_ids = tuple(str(value) for value in result.relation_ids)
        if not relation_ids:
            raise ContractError("FitResult.relation_ids must not be empty")
        return relation_ids
    resolved = str(relation_id)
    if resolved not in result.relations:
        raise ContractError("unknown relation_id values: " + resolved)
    return (resolved,)


def _cohort_arrays(cohort: CohortResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return validated cohort template arrays for plotting.

    Purpose:
        Apply lightweight plotting-safety checks at the `.pl` boundary without
        creating a separate audit surface. The checks mirror the public output
        contract enough to prevent misleading or malformed heatmaps.

    Key variables:
        A: cohort relation template `[K, K]`.
        d: source-open template `[K]`.
        e: target-open template `[K]`.
    """
    A = _as_array(cohort.template_A, name="template_A")
    d = _as_array(cohort.template_d, name="template_d")
    e = _as_array(cohort.template_e, name="template_e")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ContractError("CohortResult.template_A must have shape [K, K]")
    K = A.shape[0]
    if d.shape != (K,):
        raise ContractError("CohortResult.template_d must have shape [K]")
    if e.shape != (K,):
        raise ContractError("CohortResult.template_e must have shape [K]")
    if not np.isfinite(float(cohort.dispersion)):
        raise ContractError("CohortResult.dispersion must be finite")
    if cohort.fit_status == "ok" and int(cohort.support_n_patients) <= 0:
        raise ContractError("ok CohortResult must have positive patient support")
    if (A < 0.0).any():
        raise ContractError("CohortResult.template_A entries must be nonnegative")
    if (d < 0.0).any():
        raise ContractError("CohortResult.template_d entries must be nonnegative")
    if ((e < 0.0) | (e > 1.0)).any():
        raise ContractError("CohortResult.template_e entries must satisfy 0 <= e <= 1")
    if not np.allclose(A.sum(axis=1) + d, np.ones(K), rtol=1e-8, atol=1e-8):
        raise ContractError("CohortResult template A/d row simplex constraint failed")
    return A, d, e


def _common_n_states(
    cohort_arrays: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> int:
    """Return the shared state count for a cohort split figure.

    Purpose:
        Prevent one multi-relation figure from mixing incompatible state bases.
    """
    if not cohort_arrays:
        raise ContractError("cohort relation heatmap has no relations to plot")
    n_states = int(cohort_arrays[0][0].shape[0])
    for A, _, _ in cohort_arrays:
        if int(A.shape[0]) != n_states:
            raise ContractError("all plotted cohort relations must share n_states")
    return n_states


def _plot_augmented_cohort(
    ax,
    cohort: CohortResult,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    order: Sequence[int],
    ordered_labels: Sequence[str],
    show_panel_title: bool,
):
    """Draw one cohort relation as an augmented display matrix.

    Purpose:
        Place `A`, source-open `d`, and target-open `e` into one heatmap while
        keeping the bottom-right display-only cell masked.

    Key variables:
        matrix: masked augmented matrix `[K + 1, K + 1]`.
        open labels: explicit row/column labels that prevent interpreting
        `d/e` as ordinary community-to-community entries.
    """
    A, d, e = arrays
    A_plot = A[np.ix_(order, order)]
    d_plot = d[list(order)]
    e_plot = e[list(order)]
    matrix = _build_augmented_matrix(A_plot, d_plot, e_plot)
    cmap = _bio_continuous_cmap().copy()
    cmap.set_bad("#E8E8E8")
    image = ax.imshow(matrix, aspect="equal", vmin=0.0, vmax=1.0, cmap=cmap)
    K = len(ordered_labels)
    x_labels = [*ordered_labels, "source open d"]
    y_labels = [*ordered_labels, "target open e"]
    ax.set_xticks(np.arange(K + 1), labels=x_labels)
    ax.set_yticks(np.arange(K + 1), labels=y_labels)
    _set_right_angle_xticklabels(ax)
    ax.set_xlabel("Target community")
    ax.set_ylabel("Source community")
    ax.axhline(K - 0.5, color="#4A4A4A", linewidth=0.8)
    ax.axvline(K - 0.5, color="#4A4A4A", linewidth=0.8)
    if show_panel_title:
        ax.set_title(_cohort_relation_title(cohort), fontsize=10)
    _apply_clean_axes_style(ax)
    return image


def _build_augmented_matrix(A: np.ndarray, d: np.ndarray, e: np.ndarray) -> np.ma.MaskedArray:
    """Build the augmented display matrix `[[A, d], [e^T, masked]]`.

    Purpose:
        Create a compact visualization object without changing the fitted
        STRIDE variables. The bottom-right cell is masked because no fitted
        biological quantity lives there.
    """
    K = A.shape[0]
    matrix = np.zeros((K + 1, K + 1), dtype=float)
    mask = np.zeros((K + 1, K + 1), dtype=bool)
    matrix[:K, :K] = A
    matrix[:K, K] = d
    matrix[K, :K] = e
    mask[K, K] = True
    return np.ma.array(matrix, mask=mask)


def _cohort_relation_title(cohort: CohortResult) -> str:
    """Return the stable relation label used in cohort heatmaps."""
    return f"{cohort.relation_id} | n={cohort.support_n_patients} | disp={float(cohort.dispersion):.4g}"


def _default_augmented_figsize(n_relations: int, n_states: int) -> tuple[float, float]:
    """Estimate a deterministic figure size for augmented cohort heatmaps."""
    n_rows, n_cols = _auto_grid(n_relations, max_cols=2)
    return (
        max(6.5, n_cols * (0.62 * (n_states + 1) + 3.0)),
        max(4.8, n_rows * (0.62 * (n_states + 1) + 2.5)),
    )


def _as_array(value: Any, *, name: str) -> np.ndarray:
    """Convert a cohort field to a finite float numpy copy."""
    return as_finite_float_array(value, name=f"CohortResult.{name}").copy()


def _resolve_state_order(
    n_states: int,
    state_order: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Resolve the displayed state order without allowing state subsets."""
    try:
        return resolve_full_axis_order(n_states, state_order, name="state_order")
    except ContractError as exc:
        raise ContractError("state_order must be a complete permutation of 0..K-1") from exc


def _resolve_state_labels(
    n_states: int,
    state_labels: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Resolve display labels for the shared K-state basis."""
    try:
        return resolve_axis_labels(
            n_states,
            state_labels,
            name="state_labels",
            prefix="C",
        )
    except ContractError as exc:
        raise ContractError("state_labels length must match n_states") from exc
