"""Internal `3C-3` recurrence-ablation row builder."""
from __future__ import annotations

from . import execution as shared
from .contracts import (
    Block3SubexperimentId,
    Block3SubexperimentRawRows,
    Block3SubexperimentReviewRows,
)
from .stage3c_common import build_core_ablation_rows

BLOCK3_SUBEXPERIMENT_ID = Block3SubexperimentId.RECURRENCE_ABLATION.value


def build_3c3_rows(
    *,
    reruns: tuple[shared.Block3GeneratorRerun, ...],
    cohort_inputs: shared.Block3CohortInputs,
    runtime: shared.Block3RuntimeControls | None = None,
) -> tuple[Block3SubexperimentRawRows, Block3SubexperimentReviewRows]:
    """Build raw and review rows for `3C-3` recurrence ablation."""

    return build_core_ablation_rows(
        reruns=reruns,
        cohort_inputs=cohort_inputs,
        subexperiment_id=BLOCK3_SUBEXPERIMENT_ID,
        condition_id="recurrence_ablation_shared_realization_set",
        ablation_method_name="recurrence_ablation",
        ablation_mode="recurrence",
        runtime=runtime,
    )


__all__ = ["BLOCK3_SUBEXPERIMENT_ID", "build_3c3_rows"]
