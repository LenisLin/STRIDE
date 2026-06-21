"""Augmented-entry association surfaces for STRIDE `.da`."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from stride.errors import ContractError

from ._stats import (
    MIN_PATIENTS_PER_GROUP,
    apply_bh_correction,
    bh_q_values,
    comparison_fields,
    finite_p_value,
    multi_group_stats,
    signed_direction,
    two_group_stats,
)


def augmented_entry_group_association(
    patient_arrays: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    comparisons: Sequence[Mapping[str, object]],
    correction: Literal["BH"] = "BH",
) -> pd.DataFrame:
    """Test group associations on native augmented STRIDE relation entries.

    Scientific question:
        Within each declared relation, are caller-provided patient groups
        associated with fitted augmented relation entries `A[i,j]`, `d[i]`, or
        `e[j]`?

    Input:
        `patient_arrays` is the output of `patient_relation_arrays`.
        `comparisons` contains explicit comparison records. A record with two
        groups uses the two-group route; a record with more than two groups
        uses the multi-group route. Selecting two groups from a multi-group
        dataset is represented by an explicit two-group comparison record.

    Statistical policy:
        Patient is the statistical unit. Two independent groups use Wilcoxon
        rank-sum / Mann-Whitney U with Cliff's delta. Multiple groups use
        one-way ANOVA with eta-squared. BH correction is applied over all
        non-masked augmented entries within each `relation_id + comparison_id`.

    Output:
        DataFrame with augmented-entry coordinates, group summaries,
        `effect_size`, `effect_size_type`, `effect_direction`, `p_value`,
        `q_value`, and correction metadata.

    Boundary:
        This is an association analysis of fitted model outputs. It does not
        imply causality, physical transport, true disappearance, true
        neogenesis, lineage, or mechanism. It does not fit/refit STRIDE, mutate
        inputs, write files, or prepare plot aesthetics.
    """
    if correction != "BH":
        raise ContractError("correction must be 'BH'")
    rows: list[dict[str, object]] = []
    for relation_id, relation_groups in patient_arrays.items():
        K = _n_states_from_relation_groups(relation_groups)
        for comparison in comparisons:
            comparison_id, group_ids = comparison_fields(comparison)
            for group_id in group_ids:
                if group_id not in relation_groups:
                    raise ContractError(f"unknown group '{group_id}' for relation_id '{relation_id}'")
            _require_minimum_group_sizes(
                relation_groups,
                group_ids=group_ids,
                relation_id=str(relation_id),
                comparison_id=comparison_id,
            )
            comparison_rows = _association_rows_for_comparison(
                relation_id=str(relation_id),
                relation_groups=relation_groups,
                comparison_id=comparison_id,
                group_ids=group_ids,
                n_states=K,
            )
            apply_bh_correction(comparison_rows)
            rows.extend(comparison_rows)
    return pd.DataFrame(rows, columns=_STATS_COLUMNS)


_STATS_COLUMNS = [
    "relation_id",
    "comparison_id",
    "comparison_type",
    "row_id",
    "col_id",
    "entry_type",
    "group_1",
    "group_2",
    "groups",
    "n_total",
    "n_by_group",
    "mean_by_group",
    "median_by_group",
    "std_by_group",
    "test_name",
    "effect_size",
    "effect_size_type",
    "effect_direction",
    "p_value",
    "q_value",
    "correction_method",
    "correction_scope",
]


def _association_rows_for_comparison(
    *,
    relation_id: str,
    relation_groups: Mapping[str, Mapping[str, object]],
    comparison_id: str,
    group_ids: tuple[str, ...],
    n_states: int,
) -> list[dict[str, object]]:
    """Build uncorrected augmented-entry association rows for one comparison."""
    rows: list[dict[str, object]] = []
    for entry_type, row_id, col_id in _augmented_entry_coordinates(n_states):
        values_by_group = {
            group_id: _entry_values(relation_groups[group_id], entry_type, row_id, col_id, n_states)
            for group_id in group_ids
        }
        if len(group_ids) == 2:
            test_name, effect_size, effect_size_type, effect_direction, p_value = two_group_stats(
                values_by_group[group_ids[0]],
                values_by_group[group_ids[1]],
                group_1=group_ids[0],
                group_2=group_ids[1],
            )
            comparison_type = "two_group"
            group_1 = group_ids[0]
            group_2 = group_ids[1]
        else:
            test_name, effect_size, effect_size_type, effect_direction, p_value = multi_group_stats(
                tuple(values_by_group[group_id] for group_id in group_ids)
            )
            comparison_type = "multi_group"
            group_1 = None
            group_2 = None

        rows.append(
            {
                "relation_id": relation_id,
                "comparison_id": comparison_id,
                "comparison_type": comparison_type,
                "row_id": row_id,
                "col_id": col_id,
                "entry_type": entry_type,
                "group_1": group_1,
                "group_2": group_2,
                "groups": group_ids,
                "n_total": int(sum(len(values) for values in values_by_group.values())),
                "n_by_group": {
                    group_id: int(len(values)) for group_id, values in values_by_group.items()
                },
                "mean_by_group": {
                    group_id: float(np.mean(values)) for group_id, values in values_by_group.items()
                },
                "median_by_group": {
                    group_id: float(np.median(values)) for group_id, values in values_by_group.items()
                },
                "std_by_group": {
                    group_id: float(np.std(values, ddof=0))
                    for group_id, values in values_by_group.items()
                },
                "test_name": test_name,
                "effect_size": effect_size,
                "effect_size_type": effect_size_type,
                "effect_direction": effect_direction,
                "p_value": p_value,
                "q_value": np.nan,
                "correction_method": "BH",
                "correction_scope": f"{relation_id}:{comparison_id}",
            }
        )
    return rows


def _require_minimum_group_sizes(
    relation_groups: Mapping[str, Mapping[str, object]],
    *,
    group_ids: tuple[str, ...],
    relation_id: str,
    comparison_id: str,
) -> None:
    """Require enough patients for direct group comparisons."""
    too_small = []
    for group_id in group_ids:
        group = relation_groups[group_id]
        if "patient_ids" in group:
            n_patients = len(tuple(group["patient_ids"]))  # type: ignore[arg-type]
        else:
            n_patients = int(_as_group_array(group, "A").shape[0])
        if n_patients < MIN_PATIENTS_PER_GROUP:
            too_small.append(f"{group_id} n={n_patients}")
    if too_small:
        raise ContractError(
            "comparison groups must contain at least 3 patients each "
            f"for relation_id '{relation_id}' and comparison_id '{comparison_id}': "
            + ", ".join(too_small)
        )


def _n_states_from_relation_groups(
    relation_groups: Mapping[str, Mapping[str, object]],
) -> int:
    """Resolve K from grouped fitted arrays."""
    if not relation_groups:
        raise ContractError("patient_arrays relation entry must contain at least one group")
    n_states: int | None = None
    for group in relation_groups.values():
        A = _as_group_array(group, "A")
        d = _as_group_array(group, "d")
        e = _as_group_array(group, "e")
        if A.ndim != 3 or A.shape[1] != A.shape[2]:
            raise ContractError("patient_arrays group A must have shape [P, K, K]")
        K = int(A.shape[1])
        if d.shape != (A.shape[0], K):
            raise ContractError("patient_arrays group d must have shape [P, K]")
        if e.shape != (A.shape[0], K):
            raise ContractError("patient_arrays group e must have shape [P, K]")
        if n_states is None:
            n_states = K
        elif n_states != K:
            raise ContractError("all groups within a relation must share n_states")
    if n_states is None:
        raise ContractError("patient_arrays relation entry must contain arrays")
    return n_states


def _as_group_array(group: Mapping[str, object], key: str) -> np.ndarray:
    """Return one grouped fitted array as finite float values."""
    if key not in group:
        raise ContractError(f"patient_arrays group is missing '{key}'")
    try:
        array = np.asarray(group[key], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"patient_arrays group '{key}' must be numeric") from exc
    if not np.isfinite(array).all():
        raise ContractError(f"patient_arrays group '{key}' must contain only finite values")
    return array


def _augmented_entry_coordinates(n_states: int) -> tuple[tuple[str, int, int], ...]:
    """Return native augmented-entry coordinates excluding the masked cell."""
    coords: list[tuple[str, int, int]] = []
    for row_id in range(n_states):
        for col_id in range(n_states):
            coords.append(("A", row_id, col_id))
    for row_id in range(n_states):
        coords.append(("d", row_id, n_states))
    for col_id in range(n_states):
        coords.append(("e", n_states, col_id))
    return tuple(coords)


def _entry_values(
    group: Mapping[str, object],
    entry_type: str,
    row_id: int,
    col_id: int,
    n_states: int,
) -> np.ndarray:
    """Extract one native fitted entry across patients in a group."""
    if entry_type == "A":
        values = _as_group_array(group, "A")[:, row_id, col_id]
    elif entry_type == "d":
        values = _as_group_array(group, "d")[:, row_id]
    elif entry_type == "e":
        values = _as_group_array(group, "e")[:, col_id]
    else:
        raise ContractError(f"unknown entry_type '{entry_type}'")
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ContractError("comparison groups must contain at least one patient")
    if row_id == n_states and col_id == n_states:
        raise ContractError("bottom-right augmented-matrix cell is masked")
    return values


def _two_group_stats(
    first: np.ndarray,
    second: np.ndarray,
    *,
    group_1: str,
    group_2: str,
) -> tuple[str, float, str, str, float]:
    """Return Mann-Whitney U p-value and Cliff's delta."""
    return two_group_stats(first, second, group_1=group_1, group_2=group_2)


def _multi_group_stats(groups: tuple[np.ndarray, ...]) -> tuple[str, float, str, str, float]:
    """Return one-way ANOVA p-value and eta-squared."""
    return multi_group_stats(groups)


def _cliffs_delta(first: np.ndarray, second: np.ndarray) -> float:
    """Compute Cliff's delta for two independent groups."""
    comparisons = first[:, None] - second[None, :]
    numerator = float(np.sum(comparisons > 0.0) - np.sum(comparisons < 0.0))
    denominator = float(first.size * second.size)
    if denominator == 0.0:
        raise ContractError("Cliff's delta requires nonempty groups")
    return numerator / denominator


def _eta_squared(groups: tuple[np.ndarray, ...]) -> float:
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


def _signed_direction(value: float, *, positive: str, negative: str) -> str:
    """Return a compact direction label for signed effects."""
    return signed_direction(value, positive=positive, negative=negative)


def _finite_p_value(value: object) -> float:
    """Normalize scipy p-values to finite floats."""
    return finite_p_value(value)


def _apply_bh_correction(rows: list[dict[str, object]]) -> None:
    """Apply Benjamini-Hochberg correction in-place to one correction scope."""
    apply_bh_correction(rows)


def _bh_q_values(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted q-values."""
    return bh_q_values(p_values)
