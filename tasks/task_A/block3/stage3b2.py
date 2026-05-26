"""Internal `3B-2` open-focused d/e benchmark row builder."""
from __future__ import annotations

from . import execution as shared
from .contracts import (
    Block3SubexperimentId,
    Block3SubexperimentRawRows,
    Block3SubexperimentReviewRows,
)
from .metrics_open import build_mass_level_metric_rows
from .registry import get_subexperiment_spec


def build_3b2_rows(
    *,
    reruns: tuple[shared.Block3GeneratorRerun, ...],
    cohort_inputs: shared.Block3CohortInputs,
    runtime: shared.Block3RuntimeControls | None = None,
) -> tuple[Block3SubexperimentRawRows, Block3SubexperimentReviewRows]:
    """Build raw and review rows for executable `3B-2` d/e benchmark."""

    resolved_runtime = runtime or shared.Block3RuntimeControls()
    subexperiment_id = Block3SubexperimentId.DE_BENCHMARK.value
    condition_id = shared._DE_BENCHMARK_CONDITION_ID
    evaluation_family = get_subexperiment_spec(subexperiment_id).evaluation_family
    patient_rows = []
    truth_records: list[dict[str, object]] = []
    native_records: list[dict[str, object]] = []

    for rerun in reruns:
        uot_calibration = shared._calibrated_uot_lambda_for_train(
            cohort_inputs=cohort_inputs,
            train_patient_ids=rerun.train_patient_ids,
            cost_matrix=cohort_inputs.cost_matrix,
            runtime=resolved_runtime,
        )
        matched_mass_budget = shared._matched_mass_budget_from_train(
            cohort_inputs=cohort_inputs,
            train_patient_ids=rerun.train_patient_ids,
        )
        calibration_metadata = shared._uot_calibration_metadata(
            uot_calibration,
            n_train_pairs=len(rerun.train_patient_ids),
        )
        partial_ot_calibration_metadata = shared._partial_ot_calibration_metadata(
            n_train_pairs=len(rerun.train_patient_ids),
        )
        truths = [rerun.baseline_truths[condition_id][patient_id] for patient_id in rerun.test_patient_ids]
        truth_records.extend(
            shared._truth_store_records(
                subexperiment_id=subexperiment_id,
                condition_id=condition_id,
                truths=truths,
                open_mass_scale=rerun.open_mass_scale,
            )
        )
        outputs_by_method = {
            "stride_reference": shared._run_stride_method(
                cohort_inputs=cohort_inputs,
                truths=truths,
                runtime=resolved_runtime,
            ),
            "uot_baseline": shared._run_uot_baseline(
                truths=truths,
                cost_matrix=cohort_inputs.cost_matrix,
                match_penalty=uot_calibration.selected_lambda,
                calibration_metadata=calibration_metadata,
                runtime=resolved_runtime,
            ),
            "partial_ot_baseline": shared._run_partial_ot_baseline(
                truths=truths,
                cost_matrix=cohort_inputs.cost_matrix,
                matched_mass_budget=matched_mass_budget,
                calibration_metadata=partial_ot_calibration_metadata,
            ),
            "diagonal_transport_baseline": shared._run_diagonal_transport_baseline(truths=truths),
        }
        for method_name, outputs in outputs_by_method.items():
            native_records.extend(
                shared._method_native_records(
                    subexperiment_id=subexperiment_id,
                    condition_id=condition_id,
                    rerun_id=rerun.rerun_id,
                    method_name=method_name,
                    outputs=outputs,
                    open_mass_scale=rerun.open_mass_scale,
                )
            )
            for truth in truths:
                patient_rows.extend(
                    build_mass_level_metric_rows(
                        rerun_id=rerun.rerun_id,
                        subexperiment_id=subexperiment_id,
                        condition_id=condition_id,
                        evaluation_family=evaluation_family,
                        method_name=method_name,
                        truth=truth,
                        output=outputs[truth.patient_id],
                        open_mass_scale=rerun.open_mass_scale,
                    )
                )

    summary_rows = shared._summarize_patient_rows(
        subexperiment_id=subexperiment_id,
        patient_rows=tuple(patient_rows),
    )
    shared_tables = shared._shared_registry_tables(reruns)
    shared_tables["patient_truth_store"] = tuple(truth_records)
    shared_tables["method_native_output_store"] = tuple(native_records)
    return (
        Block3SubexperimentRawRows(
            patient_metrics=tuple(patient_rows),
            condition_summaries=summary_rows,
            shared_tables=shared_tables,
        ),
        Block3SubexperimentReviewRows(
            section_rows=shared._build_section_review_rows(
                subexperiment_id=subexperiment_id,
                summary_rows=summary_rows,
            )
        ),
    )


__all__ = ["build_3b2_rows"]
