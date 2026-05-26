"""Internal `3A` generator-validation row builder.

This module is intentionally internal. It does not expose a public Block 3
workflow; it only separates the generator-validation implementation from the
legacy monolithic execution module.
"""
from __future__ import annotations

import numpy as np

from stride.errors import ContractError

from . import execution as shared
from .contracts import (
    Block3GeneratorObjectScoreRow,
    Block3GeneratorReviewRow,
    Block3GeneratorStabilityRow,
    Block3MetricName,
    Block3SubexperimentId,
    Block3SubexperimentRawRows,
    Block3SubexperimentReviewRows,
    MetricRole,
    ValidationObjectId,
    make_metric_value,
)
from .registry import get_condition_spec, get_subexperiment_spec


def build_3a_rows(
    *,
    reruns: tuple[shared.Block3GeneratorRerun, ...],
    cohort_inputs: shared.Block3CohortInputs,
) -> tuple[Block3SubexperimentRawRows, Block3SubexperimentReviewRows]:
    """Build raw and review rows for `3A` generator validation."""

    subexperiment_id = Block3SubexperimentId.GENERATOR_VALIDATION.value
    subexperiment = get_subexperiment_spec(subexperiment_id)
    condition = get_condition_spec(shared._GENERATOR_VALIDATION_CONDITION_ID)
    if condition.evaluation_family != subexperiment.evaluation_family:
        raise ContractError("Block 3 generator-validation condition routing is misaligned")

    object_rows: list[Block3GeneratorObjectScoreRow] = []
    stability_rows: list[Block3GeneratorStabilityRow] = []
    review_rows: list[Block3GeneratorReviewRow] = []
    truth_records: list[dict[str, object]] = []
    object_vectors: dict[ValidationObjectId, list[np.ndarray]] = {
        ValidationObjectId.COMMUNITY_SPACE_TARGET: [],
        ValidationObjectId.IDENTITY_PROJECTED_TARGET: [],
    }
    metric_history: dict[ValidationObjectId, dict[Block3MetricName, list[float]]] = {
        ValidationObjectId.COMMUNITY_SPACE_TARGET: {},
        ValidationObjectId.IDENTITY_PROJECTED_TARGET: {},
    }

    for rerun in reruns:
        real_surface = shared._mean_profile(
            [cohort_inputs.patient_target_profiles[patient_id] for patient_id in rerun.test_patient_ids]
        )
        synthetic_surface = shared._mean_profile(
            [rerun.generator_truths[patient_id].y for patient_id in rerun.test_patient_ids]
        )
        truth_records.extend(
            shared._truth_store_records(
                subexperiment_id=subexperiment_id,
                condition_id=condition.condition_id,
                truths=[rerun.generator_truths[patient_id] for patient_id in rerun.test_patient_ids],
            )
        )
        object_pairs = {
            ValidationObjectId.COMMUNITY_SPACE_TARGET: (real_surface, synthetic_surface),
            ValidationObjectId.IDENTITY_PROJECTED_TARGET: (
                real_surface @ cohort_inputs.identity_vectors,
                synthetic_surface @ cohort_inputs.identity_vectors,
            ),
        }
        for object_id, (real_vector, synthetic_vector) in object_pairs.items():
            object_vectors[object_id].append(synthetic_vector)
            for metric_name, metric_number in shared._metric_bundle(real_vector, synthetic_vector).items():
                metric_value = make_metric_value(metric_name=metric_name, value=metric_number, status="reported")
                shared._validate_metric_value(
                    subexperiment_id=subexperiment_id,
                    metric_value=metric_value,
                    expected_role=MetricRole.GENERATOR_VALIDATION,
                )
                metric_history.setdefault(object_id, {}).setdefault(metric_name, []).append(metric_number)
                row = Block3GeneratorObjectScoreRow(
                    rerun_id=rerun.rerun_id,
                    subexperiment_id=subexperiment_id,
                    condition_id=condition.condition_id,
                    evaluation_family=subexperiment.evaluation_family,
                    validation_object_id=object_id,
                    metric_role=MetricRole.GENERATOR_VALIDATION,
                    metric_value=metric_value,
                )
                object_rows.append(row)
                review_rows.append(
                    Block3GeneratorReviewRow(
                        rerun_id=rerun.rerun_id,
                        subexperiment_id=subexperiment_id,
                        condition_id=condition.condition_id,
                        evaluation_family=subexperiment.evaluation_family,
                        validation_object_id=object_id,
                        metric_role=MetricRole.GENERATOR_VALIDATION,
                        metric_value=metric_value,
                        stability_summary_level="",
                        review_surface_role="generator_object_score",
                    )
                )

    for object_id, vectors in object_vectors.items():
        vector_matrix = np.vstack(vectors)
        object_variability = float(np.mean(np.std(vector_matrix, axis=0, dtype=float)))
        metric_variability = float(
            np.mean(
                [
                    np.std(np.asarray(metric_history[object_id][metric_name], dtype=float))
                    for metric_name in metric_history[object_id]
                ]
            )
        )
        metric_value = make_metric_value(
            metric_name=Block3MetricName.RERUN_VARIABILITY,
            value=object_variability + metric_variability,
            status="reported",
        )
        shared._validate_metric_value(
            subexperiment_id=subexperiment_id,
            metric_value=metric_value,
            expected_role=MetricRole.STABILITY_SUMMARY,
        )
        stability_rows.append(
            Block3GeneratorStabilityRow(
                rerun_id="all_reruns",
                subexperiment_id=subexperiment_id,
                condition_id=condition.condition_id,
                evaluation_family=subexperiment.evaluation_family,
                validation_object_id=object_id,
                metric_role=MetricRole.STABILITY_SUMMARY,
                metric_value=metric_value,
                stability_summary_level="between_rerun",
            )
        )
        review_rows.append(
            Block3GeneratorReviewRow(
                rerun_id="all_reruns",
                subexperiment_id=subexperiment_id,
                condition_id=condition.condition_id,
                evaluation_family=subexperiment.evaluation_family,
                validation_object_id=object_id,
                metric_role=MetricRole.STABILITY_SUMMARY,
                metric_value=metric_value,
                stability_summary_level="between_rerun",
                review_surface_role="generator_rerun_stability",
            )
        )

    shared_tables = shared._shared_registry_tables(reruns)
    shared_tables["patient_truth_store"] = tuple(truth_records)
    return (
        Block3SubexperimentRawRows(
            object_scores=tuple(object_rows),
            rerun_stability=tuple(stability_rows),
            shared_tables=shared_tables,
        ),
        Block3SubexperimentReviewRows(generator_rows=tuple(review_rows)),
    )


__all__ = ["build_3a_rows"]
