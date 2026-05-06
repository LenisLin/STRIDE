"""Analysis-layer calibration primitives for Task A Block 0 fit caches.

These helpers derive Block 1-facing family-summary calibration tables from
realized Block 0 real/null STRIDE fit records. Fit/permutation execution must
write reusable `A,d,e,mu` cache records first; calibration summaries are
analysis artifacts computed later, with no multiplicity correction, pass/fail
gate, biological interpretation, or downstream execution decision.
See `tasks/task_A/README.md`, `tasks/task_A/contracts/artifact_contracts.md`,
and `tasks/task_A/contracts/design_freeze.py`.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd

from stride.errors import ContractError

from .schemas import (
    EFFECT_RATIO_STATUS_ESTIMABLE,
    EFFECT_RATIO_STATUS_NOT_ESTIMABLE,
    FAMILY_SUMMARY_SCALES,
    FIT_LABEL_NULL,
    FIT_LABEL_REAL,
    METRIC_SUMMARY_COLUMNS,
    NULL_FAMILY,
    PATIENT_CALIBRATION_COLUMNS,
    REAL_FAMILY,
    REFERENCE_STATS,
    SUMMARY_EXPECTED_TAILS,
    SUMMARY_NAMES,
    SUMMARY_ROLES,
    Block0FitRecord,
)


def _as_float_array(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ContractError(f"{name} must contain only finite values")
    return array


def _normalized_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    total = float(np.sum(values, dtype=float))
    if total <= 0.0:
        raise ContractError(f"{name} weights must have positive total")
    return values / total


def _as_float_vector(values: object, *, name: str) -> np.ndarray:
    vector = _as_float_array(values, name=name)
    if vector.ndim != 1:
        raise ContractError(f"{name} must be a vector")
    return vector


def _as_square_matrix(values: object, *, name: str) -> np.ndarray:
    matrix = _as_float_array(values, name=name)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ContractError(f"{name} must be a square matrix")
    return matrix


def _validated_fit_arrays(record: Block0FitRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A = _as_square_matrix(record.A, name="A")
    d = _as_float_vector(record.d, name="d")
    e = _as_float_vector(record.e, name="e")
    source_burden = _as_float_vector(record.source_burden, name="source_burden")
    target_burden = _as_float_vector(record.e_weights, name="target_burden")
    expected_shape = (A.shape[0],)
    if d.shape != expected_shape or e.shape != expected_shape:
        raise ContractError("Block 0 family summaries require d/e vectors to match A")
    if source_burden.shape != expected_shape or target_burden.shape != expected_shape:
        raise ContractError("Block 0 family summaries require burden vectors to match A")
    if (
        (A < 0.0).any()
        or (d < 0.0).any()
        or (e < 0.0).any()
        or (source_burden < 0.0).any()
        or (target_burden < 0.0).any()
    ):
        raise ContractError("Block 0 family summary inputs must be non-negative")
    return A, d, e, source_burden, target_burden


def _summary_axis(summary_name: str) -> str:
    if summary_name == "emergence":
        return "target"
    if summary_name in {"self_retention", "depletion", "off_diagonal_remodeling"}:
        return "source"
    raise ContractError(f"Unsupported Block 0 summary_name: {summary_name!r}")


def _opposite_tail(tail: str) -> str:
    if tail == "right":
        return "left"
    if tail == "left":
        return "right"
    raise ContractError(f"Unsupported empirical tail: {tail!r}")


def _tail_mask(reference: np.ndarray, observed: float, *, tail: str) -> np.ndarray:
    if tail == "right":
        return reference >= float(observed)
    if tail == "left":
        return reference <= float(observed)
    raise ContractError(f"Unsupported empirical tail: {tail!r}")


def empirical_tail_p_value(
    observed: float,
    null_reference: Sequence[float] | np.ndarray,
    *,
    tail: str,
) -> float:
    """Return plus-one corrected empirical p-value for the declared tail."""
    observed_value = float(observed)
    if not np.isfinite(observed_value):
        raise ContractError("observed p-value statistic must be finite")
    reference = _as_float_array(null_reference, name="null_reference").reshape(-1)
    if reference.size == 0:
        raise ContractError("null_reference must not be empty")
    tail_count = int(np.sum(_tail_mask(reference, observed_value, tail=tail)))
    return float((1 + tail_count) / (1 + int(reference.size)))


def median_and_mean(values: Sequence[float] | np.ndarray) -> tuple[float, float]:
    """Return median and mean for a non-empty finite vector."""
    array = _as_float_array(values, name="summary values").reshape(-1)
    if array.size == 0:
        raise ContractError("summary values must not be empty")
    return float(np.median(array)), float(np.mean(array, dtype=float))


def tail_null_fraction(
    observed: float,
    null_reference: Sequence[float] | np.ndarray,
    *,
    tail: str,
) -> float:
    """Return the uncorrected empirical tail fraction for descriptive reporting."""
    observed_value = float(observed)
    if not np.isfinite(observed_value):
        raise ContractError("observed tail statistic must be finite")
    reference = _as_float_array(null_reference, name="null_reference").reshape(-1)
    if reference.size == 0:
        raise ContractError("null_reference must not be empty")
    return float(np.mean(_tail_mask(reference, observed_value, tail=tail), dtype=float))


def effect_delta(observed: float, reference: float) -> float:
    """Return the descriptive real-vs-null minus null-vs-null effect delta."""
    observed_value = float(observed)
    reference_value = float(reference)
    if not np.isfinite(observed_value) or not np.isfinite(reference_value):
        raise ContractError("effect_delta inputs must be finite")
    return float(observed_value - reference_value)


def effect_ratio(
    observed: float,
    reference: float,
    *,
    denominator_atol: float = 1e-12,
) -> tuple[float | None, str]:
    """Return a descriptive ratio without adding an epsilon to the denominator."""
    observed_value = float(observed)
    reference_value = float(reference)
    if float(denominator_atol) < 0.0:
        raise ContractError("effect_ratio denominator_atol must be non-negative")
    if not np.isfinite(observed_value) or not np.isfinite(reference_value):
        raise ContractError("effect_ratio inputs must be finite")
    if abs(reference_value) <= float(denominator_atol):
        return None, EFFECT_RATIO_STATUS_NOT_ESTIMABLE
    return float(observed_value / reference_value), EFFECT_RATIO_STATUS_ESTIMABLE


def family_summary_values(record: Block0FitRecord) -> dict[tuple[str, str], float]:
    """Return Block 1-facing family summaries for one Block 0 fit record."""
    A, d, e, source_burden, target_burden = _validated_fit_arrays(record)
    source_mask = np.asarray(source_burden > 0.0, dtype=bool)
    target_mask = np.asarray(target_burden > 0.0, dtype=bool)
    if not bool(source_mask.any()):
        raise ContractError("Block 0 source-axis family summaries require positive source burden")
    if not bool(target_mask.any()):
        raise ContractError("Block 0 target-axis family summaries require positive target burden")

    source_weights = np.zeros_like(source_burden, dtype=float)
    source_weights[source_mask] = _normalized_vector(
        source_burden[source_mask],
        name="source_burden",
    )
    target_weights = np.zeros_like(target_burden, dtype=float)
    target_weights[target_mask] = _normalized_vector(
        target_burden[target_mask],
        name="target_burden",
    )

    diagonal_operator = np.diag(A).astype(float, copy=False)
    off_diagonal_operator = np.sum(A, axis=1, dtype=float) - diagonal_operator
    source_vectors = {
        "self_retention": diagonal_operator,
        "depletion": d,
        "off_diagonal_remodeling": off_diagonal_operator,
    }

    values: dict[tuple[str, str], float] = {}
    for summary_name, vector in source_vectors.items():
        values[(summary_name, "burden_weighted")] = float(
            np.sum(source_weights * vector, dtype=float)
        )
        values[(summary_name, "community_mean")] = float(
            np.mean(vector[source_mask], dtype=float)
        )
    values[("emergence", "burden_weighted")] = float(
        np.sum(target_weights * e, dtype=float)
    )
    values[("emergence", "community_mean")] = float(
        np.mean(e[target_mask], dtype=float)
    )
    return values


def _aggregate(values: Sequence[float] | np.ndarray, *, stat_name: str) -> float:
    median, mean = median_and_mean(values)
    if stat_name == "median":
        return median
    if stat_name == "mean":
        return mean
    raise ContractError(f"Unsupported Block 0 reference stat: {stat_name!r}")


def _aggregate_aligned_reference(
    reference_vectors: Sequence[Sequence[float] | np.ndarray],
    *,
    stat_name: str,
) -> np.ndarray:
    """Aggregate permutation-aligned patient references into a cohort reference."""
    if not reference_vectors:
        raise ContractError("Block 0 cohort null reference requires patient references")
    arrays = [
        _as_float_array(reference_vector, name="cohort null reference").reshape(-1)
        for reference_vector in reference_vectors
    ]
    reference_size = int(arrays[0].size)
    if reference_size == 0:
        raise ContractError("Block 0 cohort null reference vectors must not be empty")
    if any(int(array.size) != reference_size for array in arrays):
        raise ContractError("Block 0 cohort null reference vectors must be aligned")

    matrix = np.vstack(arrays)
    if stat_name == "median":
        return np.median(matrix, axis=0)
    if stat_name == "mean":
        return np.mean(matrix, axis=0, dtype=float)
    raise ContractError(f"Unsupported Block 0 reference stat: {stat_name!r}")


def _patient_delta_counts(
    real_values: Sequence[float],
    null_vectors: Sequence[Sequence[float] | np.ndarray],
    *,
    stat_name: str,
    zero_atol: float = 1e-12,
) -> dict[str, int]:
    if len(real_values) != len(null_vectors):
        raise ContractError("Block 0 patient delta counts require aligned patient vectors")
    counts = {"positive": 0, "negative": 0, "zero": 0}
    for observed, null_vector in zip(real_values, null_vectors, strict=True):
        reference_value = _aggregate(null_vector, stat_name=stat_name)
        delta = effect_delta(float(observed), reference_value)
        if abs(delta) <= float(zero_atol):
            counts["zero"] += 1
        elif delta > 0.0:
            counts["positive"] += 1
        else:
            counts["negative"] += 1
    return counts


def _group_records_by_patient(
    records: Sequence[Block0FitRecord],
    *,
    fit_label: str,
) -> dict[str, list[Block0FitRecord]]:
    grouped: dict[str, list[Block0FitRecord]] = defaultdict(list)
    for record in records:
        if record.fit_label != fit_label:
            raise ContractError(
                f"Block 0 expected {fit_label!r} fit records, got {record.fit_label!r}"
            )
        grouped[str(record.patient_id)].append(record)
    if not grouped:
        raise ContractError(f"Block 0 requires at least one {fit_label} fit record")
    return dict(grouped)


def _validate_patient_record_sets(
    real_records: Sequence[Block0FitRecord],
    null_records: Sequence[Block0FitRecord],
    *,
    n_permutations: int,
) -> tuple[dict[str, Block0FitRecord], dict[str, tuple[Block0FitRecord, ...]]]:
    real_grouped = _group_records_by_patient(real_records, fit_label=FIT_LABEL_REAL)
    null_grouped = _group_records_by_patient(null_records, fit_label=FIT_LABEL_NULL)
    if set(real_grouped) != set(null_grouped):
        raise ContractError("Block 0 real and null fit records must cover the same patients")

    real_by_patient: dict[str, Block0FitRecord] = {}
    null_by_patient: dict[str, tuple[Block0FitRecord, ...]] = {}
    expected_indices = tuple(range(int(n_permutations)))
    for patient_id in sorted(real_grouped):
        if len(real_grouped[patient_id]) != 1:
            raise ContractError(f"Block 0 patient {patient_id!r} must have exactly one real fit")
        patient_nulls = tuple(
            sorted(
                null_grouped[patient_id],
                key=lambda record: int(record.permutation_index),
            )
        )
        permutation_indices = tuple(int(record.permutation_index) for record in patient_nulls)
        if len(patient_nulls) != int(n_permutations) or permutation_indices != expected_indices:
            raise ContractError(
                f"Block 0 patient {patient_id!r} must have null permutations {expected_indices}"
            )
        real_by_patient[patient_id] = real_grouped[patient_id][0]
        null_by_patient[patient_id] = patient_nulls
    return real_by_patient, null_by_patient


def build_block0_calibration_frames(
    real_records: Sequence[Block0FitRecord],
    null_records: Sequence[Block0FitRecord],
    *,
    run_scope: str,
    n_permutations: int,
    readiness_status: str,
    real_family: str = REAL_FAMILY,
    null_family: str = NULL_FAMILY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build strict patient and cohort family-summary calibration frames."""
    if int(n_permutations) < 2:
        raise ContractError("Block 0 calibration requires at least two null permutations")
    real_by_patient, null_by_patient = _validate_patient_record_sets(
        real_records,
        null_records,
        n_permutations=n_permutations,
    )

    patient_rows: list[dict[str, object]] = []
    patient_summary_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    patient_null_vectors: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)

    for patient_id in sorted(real_by_patient):
        real_record = real_by_patient[patient_id]
        patient_nulls = null_by_patient[patient_id]
        real_summaries = family_summary_values(real_record)
        null_summaries_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
        for null_record in patient_nulls:
            null_summaries = family_summary_values(null_record)
            for key, value in null_summaries.items():
                null_summaries_by_key[key].append(float(value))

        for summary_name in SUMMARY_NAMES:
            expected_tail = SUMMARY_EXPECTED_TAILS[summary_name]
            opposite_tail = _opposite_tail(expected_tail)
            for scale in FAMILY_SUMMARY_SCALES:
                key = (summary_name, scale)
                observed = float(real_summaries[key])
                reference_distribution = np.asarray(null_summaries_by_key[key], dtype=float)
                if int(reference_distribution.size) != int(n_permutations):
                    raise ContractError(
                        "Block 0 patient null family-summary distribution "
                        "does not match n_permutations"
                    )
                patient_summary_values[key].append(observed)
                patient_null_vectors[key].append(reference_distribution)

                for stat_name in REFERENCE_STATS:
                    reference_value = _aggregate(reference_distribution, stat_name=stat_name)
                    ratio, ratio_status = effect_ratio(observed, reference_value)
                    patient_rows.append(
                        {
                            "patient_id": patient_id,
                            "run_scope": run_scope,
                            "real_family": real_family,
                            "null_family": null_family,
                            "n_permutations": int(n_permutations),
                            "real_fit_status": real_record.fit_status,
                            "null_fit_status": "ok",
                            "summary_name": summary_name,
                            "summary_role": SUMMARY_ROLES[summary_name],
                            "eligible_entity_axis": _summary_axis(summary_name),
                            "scale": scale,
                            "reference_stat": stat_name,
                            "expected_tail": expected_tail,
                            "real_value": observed,
                            "null_reference": reference_value,
                            "empirical_p_value": empirical_tail_p_value(
                                observed,
                                reference_distribution,
                                tail=expected_tail,
                            ),
                            "primary_tail_fraction": tail_null_fraction(
                                observed,
                                reference_distribution,
                                tail=expected_tail,
                            ),
                            "opposite_tail_fraction": tail_null_fraction(
                                observed,
                                reference_distribution,
                                tail=opposite_tail,
                            ),
                            "effect_delta": effect_delta(observed, reference_value),
                            "effect_ratio": ratio,
                            "effect_ratio_status": ratio_status,
                            "readiness_status": readiness_status,
                        }
                    )

    metric_rows: list[dict[str, object]] = []
    for summary_name in SUMMARY_NAMES:
        expected_tail = SUMMARY_EXPECTED_TAILS[summary_name]
        opposite_tail = _opposite_tail(expected_tail)
        for scale in FAMILY_SUMMARY_SCALES:
            key = (summary_name, scale)
            for stat_name in REFERENCE_STATS:
                real_value = _aggregate(patient_summary_values[key], stat_name=stat_name)
                cohort_null_reference = _aggregate_aligned_reference(
                    patient_null_vectors[key],
                    stat_name=stat_name,
                )
                reference_value = _aggregate(cohort_null_reference, stat_name=stat_name)
                ratio, ratio_status = effect_ratio(real_value, reference_value)
                delta_counts = _patient_delta_counts(
                    patient_summary_values[key],
                    patient_null_vectors[key],
                    stat_name=stat_name,
                )
                metric_rows.append(
                    {
                        "summary_name": summary_name,
                        "summary_role": SUMMARY_ROLES[summary_name],
                        "eligible_entity_axis": _summary_axis(summary_name),
                        "scale": scale,
                        "cohort_stat": stat_name,
                        "expected_tail": expected_tail,
                        "real_value": real_value,
                        "null_reference": reference_value,
                        "empirical_p_value": empirical_tail_p_value(
                            real_value,
                            cohort_null_reference,
                            tail=expected_tail,
                        ),
                        "primary_tail_fraction": tail_null_fraction(
                            real_value,
                            cohort_null_reference,
                            tail=expected_tail,
                        ),
                        "opposite_tail_fraction": tail_null_fraction(
                            real_value,
                            cohort_null_reference,
                            tail=opposite_tail,
                        ),
                        "effect_delta": effect_delta(real_value, reference_value),
                        "effect_ratio": ratio,
                        "effect_ratio_status": ratio_status,
                        "n_patient_delta_positive": delta_counts["positive"],
                        "n_patient_delta_negative": delta_counts["negative"],
                        "n_patient_delta_zero": delta_counts["zero"],
                        "readiness_status": readiness_status,
                    }
                )

    return (
        pd.DataFrame(patient_rows, columns=PATIENT_CALIBRATION_COLUMNS),
        pd.DataFrame(metric_rows, columns=METRIC_SUMMARY_COLUMNS),
    )


__all__ = [
    "build_block0_calibration_frames",
    "effect_delta",
    "effect_ratio",
    "empirical_tail_p_value",
    "family_summary_values",
    "median_and_mean",
    "tail_null_fraction",
]
