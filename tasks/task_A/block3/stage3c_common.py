"""Shared implementation for internal `3C-*` STRIDE refit ablations."""
from __future__ import annotations

from . import execution as shared
from .contracts import Block3SubexperimentRawRows, Block3SubexperimentReviewRows
from .metrics_open import build_relation_and_open_metric_rows
from .registry import get_subexperiment_spec


def build_core_ablation_rows(
    *,
    reruns: tuple[shared.Block3GeneratorRerun, ...],
    cohort_inputs: shared.Block3CohortInputs,
    subexperiment_id: str,
    condition_id: str,
    ablation_method_name: str,
    ablation_mode: str,
    runtime: shared.Block3RuntimeControls | None = None,
) -> tuple[Block3SubexperimentRawRows, Block3SubexperimentReviewRows]:
    """Build rows for one `3C` core STRIDE refit ablation request."""

    resolved_runtime = runtime or shared.Block3RuntimeControls()
    evaluation_family = get_subexperiment_spec(subexperiment_id).evaluation_family
    patient_rows = []
    truth_records: list[dict[str, object]] = []
    native_records: list[dict[str, object]] = []

    for rerun in reruns:
        truths = [rerun.generator_truths[patient_id] for patient_id in rerun.test_patient_ids]
        truth_records.extend(
            shared._truth_store_records(
                subexperiment_id=subexperiment_id,
                condition_id=condition_id,
                truths=truths,
            )
        )
        outputs_by_method = {
            "stride_reference": shared._run_stride_method(
                cohort_inputs=cohort_inputs,
                truths=truths,
                runtime=resolved_runtime,
                rerun_id=rerun.rerun_id,
            ),
            ablation_method_name: shared._run_stride_method(
                cohort_inputs=cohort_inputs,
                truths=truths,
                runtime=resolved_runtime,
                ablation_mode=ablation_mode,
                rerun_id=rerun.rerun_id,
            ),
        }
        for method_name, outputs in outputs_by_method.items():
            native_records.extend(
                shared._method_native_records(
                    subexperiment_id=subexperiment_id,
                    condition_id=condition_id,
                    rerun_id=rerun.rerun_id,
                    method_name=method_name,
                    outputs=outputs,
                )
            )
            for truth in truths:
                patient_rows.extend(
                    build_relation_and_open_metric_rows(
                        rerun_id=rerun.rerun_id,
                        subexperiment_id=subexperiment_id,
                        condition_id=condition_id,
                        evaluation_family=evaluation_family,
                        method_name=method_name,
                        truth=truth,
                        output=outputs[truth.patient_id],
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


__all__ = ["build_core_ablation_rows"]
