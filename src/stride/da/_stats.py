"""Private statistical helpers shared by STRIDE `.da` analyses."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import SupportsFloat, SupportsIndex, cast

import numpy as np
from scipy import stats

from stride.errors import ContractError

MIN_PATIENTS_PER_GROUP = 3


def comparison_fields(comparison: Mapping[str, object]) -> tuple[str, tuple[str, ...]]:
    """Read comparison id and explicit group ids."""
    if "comparison_id" not in comparison:
        raise ContractError("comparison record is missing comparison_id")
    if "groups" not in comparison:
        raise ContractError("comparison record is missing groups")
    comparison_id = str(comparison["comparison_id"])
    groups = comparison["groups"]
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Iterable):
        raise ContractError("comparison groups must be a sequence of group ids")
    group_ids = tuple(str(group_id) for group_id in groups)
    if len(group_ids) < 2:
        raise ContractError("comparison groups must contain at least two groups")
    if len(set(group_ids)) != len(group_ids):
        raise ContractError("comparison groups must be unique")
    return comparison_id, group_ids


def two_group_stats(
    first: np.ndarray,
    second: np.ndarray,
    *,
    group_1: str,
    group_2: str,
) -> tuple[str, float, str, str, float]:
    """Return Mann-Whitney U p-value and Cliff's delta."""
    if first.size == 0 or second.size == 0:
        raise ContractError("two-group comparisons require patients in both groups")
    result = stats.mannwhitneyu(first, second, alternative="two-sided", method="auto")
    delta = cliffs_delta(first, second)
    return (
        "mannwhitneyu",
        delta,
        "cliffs_delta",
        signed_direction(delta, positive=f"{group_1}>{group_2}", negative=f"{group_1}<{group_2}"),
        finite_p_value(result.pvalue),
    )


def multi_group_stats(groups: tuple[np.ndarray, ...]) -> tuple[str, float, str, str, float]:
    """Return one-way ANOVA p-value and eta-squared."""
    if any(group.size == 0 for group in groups):
        raise ContractError("multi-group comparisons require patients in every group")
    if np.allclose(np.concatenate(groups), np.concatenate(groups)[0]):
        return "one_way_anova", 0.0, "eta_squared", "none", 1.0
    result = stats.f_oneway(*groups)
    eta_squared_value = eta_squared(groups)
    return "one_way_anova", eta_squared_value, "eta_squared", "unsigned", finite_p_value(
        result.pvalue
    )


def cliffs_delta(first: np.ndarray, second: np.ndarray) -> float:
    """Compute Cliff's delta for two independent groups."""
    comparisons = first[:, None] - second[None, :]
    numerator = float(np.sum(comparisons > 0.0) - np.sum(comparisons < 0.0))
    denominator = float(first.size * second.size)
    if denominator == 0.0:
        raise ContractError("Cliff's delta requires nonempty groups")
    return numerator / denominator


def eta_squared(groups: tuple[np.ndarray, ...]) -> float:
    """Compute one-way ANOVA eta-squared."""
    values = np.concatenate(groups)
    grand_mean = float(np.mean(values))
    ss_total = float(np.sum((values - grand_mean) ** 2))
    if ss_total == 0.0:
        return 0.0
    ss_between = 0.0
    for group in groups:
        ss_between += float(group.size) * float((np.mean(group) - grand_mean) ** 2)
    return ss_between / ss_total


def signed_direction(value: float, *, positive: str, negative: str) -> str:
    """Return a compact direction label for signed effects."""
    if value > 0.0:
        return positive
    if value < 0.0:
        return negative
    return "none"


def finite_p_value(value: object) -> float:
    """Normalize scipy p-values to finite floats."""
    try:
        p_value = float(cast(str | bytes | SupportsFloat | SupportsIndex, value))
    except (TypeError, ValueError) as exc:
        raise ContractError("statistical test returned a non-numeric p_value") from exc
    if not np.isfinite(p_value):
        raise ContractError("statistical test returned a non-finite p_value")
    return min(max(p_value, 0.0), 1.0)


def apply_bh_correction(rows: list[dict[str, object]]) -> None:
    """Apply Benjamini-Hochberg correction in-place to one correction scope."""
    if not rows:
        return
    p_values = np.array([finite_p_value(row["p_value"]) for row in rows], dtype=float)
    if not np.isfinite(p_values).all():
        raise ContractError("p_value must be finite before BH correction")
    q_values = bh_q_values(p_values)
    for row, q_value in zip(rows, q_values, strict=True):
        row["q_value"] = float(q_value)


def bh_q_values(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted q-values."""
    n_tests = int(p_values.size)
    if n_tests == 0:
        return p_values.copy()
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.empty(n_tests, dtype=float)
    cumulative = 1.0
    for index in range(n_tests - 1, -1, -1):
        rank = index + 1
        cumulative = min(cumulative, ranked[index] * n_tests / rank)
        adjusted[index] = cumulative
    q_values = np.empty(n_tests, dtype=float)
    q_values[order] = np.clip(adjusted, 0.0, 1.0)
    return q_values
