"""Grouped loss definitions and total-objective assembly for STRIDE fitting."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LossWeights:
    """Relative weights for grouped canonical STRIDE loss components.

    The legacy names ``data_fit``, ``structural_bias``, and
    ``open_channel_control`` remain accepted as compatibility aliases.
    """

    observation_data_fit: float = 1.0
    patient_consistency: float = 1.0
    open_relation: float = 0.25
    cohort_recurrence: float = 0.25
    geometry_structure: float = 0.10
    data_fit: float | None = None
    structural_bias: float | None = None
    open_channel_control: float | None = None

    @property
    def resolved_observation_data_fit(self) -> float:
        return float(self.data_fit if self.data_fit is not None else self.observation_data_fit)

    @property
    def resolved_open_relation(self) -> float:
        return float(
            self.open_channel_control
            if self.open_channel_control is not None
            else self.open_relation
        )

    @property
    def resolved_geometry_structure(self) -> float:
        return float(
            self.structural_bias
            if self.structural_bias is not None
            else self.geometry_structure
        )


@dataclass(frozen=True)
class LossBreakdown:
    """Grouped canonical STRIDE loss summary for one fitting step."""

    observation_data_fit: float
    patient_consistency: float
    open_relation: float
    cohort_recurrence: float
    geometry_structure: float
    total: float

    @property
    def data_fit(self) -> float:
        """Compatibility alias for older observation-fit naming."""
        return float(self.observation_data_fit)

    @property
    def structural_bias(self) -> float:
        """Compatibility alias for older structural-bias naming."""
        return float(self.geometry_structure)

    @property
    def open_channel_control(self) -> float:
        """Compatibility alias for older open-channel naming."""
        return float(self.open_relation)


def compute_data_fit_loss(observed: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute an L2 data-fit loss between observed and reconstructed payloads."""
    obs = np.asarray(observed, dtype=float)
    rec = np.asarray(reconstructed, dtype=float)
    return float(np.sum(np.square(obs - rec), dtype=float))


def compute_patient_consistency_loss(
    A: np.ndarray,
    d: np.ndarray,
    e: np.ndarray,
    *,
    reference_A: np.ndarray | None = None,
    reference_d: np.ndarray | None = None,
    reference_e: np.ndarray | None = None,
) -> float:
    """Measure how far one patient relation moves from its observation-grounded reference."""
    A_arr = np.asarray(A, dtype=float)
    d_arr = np.asarray(d, dtype=float)
    e_arr = np.asarray(e, dtype=float)
    if reference_A is None or reference_d is None or reference_e is None:
        row_residual = np.sum(A_arr, axis=1, dtype=float) + d_arr - 1.0
        return float(np.sum(np.square(row_residual), dtype=float))
    return float(
        np.sum(np.square(A_arr - np.asarray(reference_A, dtype=float)), dtype=float)
        + np.sum(np.square(d_arr - np.asarray(reference_d, dtype=float)), dtype=float)
        + np.sum(np.square(e_arr - np.asarray(reference_e, dtype=float)), dtype=float)
    )


def compute_open_relation_loss(
    d: np.ndarray,
    e: np.ndarray,
    *,
    target_total: float | None = None,
) -> float:
    """Compute a simple open-relation penalty on depletion/emergence mass."""
    depletion = float(np.sum(np.asarray(d, dtype=float), dtype=float))
    emergence = float(np.sum(np.asarray(e, dtype=float), dtype=float))
    total = depletion + emergence
    if target_total is None:
        return 0.0
    return float((total - float(target_total)) ** 2)


def compute_cohort_recurrence_loss(
    A: np.ndarray,
    d: np.ndarray,
    e: np.ndarray,
    *,
    template_A: np.ndarray | None = None,
    template_d: np.ndarray | None = None,
    template_e: np.ndarray | None = None,
) -> float:
    """Measure deviation from the current cohort-level recurrence template."""
    if template_A is None or template_d is None or template_e is None:
        return 0.0
    return float(
        np.sum(np.square(np.asarray(A, dtype=float) - np.asarray(template_A, dtype=float)), dtype=float)
        + np.sum(np.square(np.asarray(d, dtype=float) - np.asarray(template_d, dtype=float)), dtype=float)
        + np.sum(np.square(np.asarray(e, dtype=float) - np.asarray(template_e, dtype=float)), dtype=float)
    )


def compute_geometry_structure_loss(
    A: np.ndarray,
    *,
    geometry_cost_matrix: np.ndarray | None = None,
    target_diagonal_fraction: float | None = None,
) -> float:
    """Compute a geometry/locality penalty for the continuity operator."""
    matrix = np.asarray(A, dtype=float)
    total = float(np.sum(matrix, dtype=float))
    if total <= 0.0:
        return 0.0
    if geometry_cost_matrix is not None:
        geometry = np.asarray(geometry_cost_matrix, dtype=float)
        return float(np.sum(matrix * geometry, dtype=float) / total)
    diagonal_fraction = float(np.trace(matrix) / total)
    if target_diagonal_fraction is None:
        return 0.0
    return float((diagonal_fraction - float(target_diagonal_fraction)) ** 2)


def compute_structural_bias_loss(
    A: np.ndarray,
    *,
    target_diagonal_fraction: float | None = None,
) -> float:
    """Compatibility wrapper over the canonical geometry/locality penalty."""
    return compute_geometry_structure_loss(
        A,
        target_diagonal_fraction=target_diagonal_fraction,
    )


def compute_open_channel_control_loss(
    d: np.ndarray,
    e: np.ndarray,
    *,
    target_total: float | None = None,
) -> float:
    """Compatibility wrapper over the canonical open-relation penalty."""
    return compute_open_relation_loss(d, e, target_total=target_total)


def evaluate_loss_bundle(
    *,
    observed: np.ndarray,
    reconstructed: np.ndarray,
    A: np.ndarray,
    d: np.ndarray,
    e: np.ndarray,
    reference_A: np.ndarray | None = None,
    reference_d: np.ndarray | None = None,
    reference_e: np.ndarray | None = None,
    template_A: np.ndarray | None = None,
    template_d: np.ndarray | None = None,
    template_e: np.ndarray | None = None,
    geometry_cost_matrix: np.ndarray | None = None,
    weights: LossWeights | None = None,
    target_diagonal_fraction: float | None = None,
    target_open_channel_total: float | None = None,
) -> LossBreakdown:
    """Evaluate grouped canonical loss components and return the weighted total."""
    resolved_weights = weights or LossWeights()
    observation_data_fit = compute_data_fit_loss(observed, reconstructed)
    patient_consistency = compute_patient_consistency_loss(
        A,
        d,
        e,
        reference_A=reference_A,
        reference_d=reference_d,
        reference_e=reference_e,
    )
    open_relation = compute_open_relation_loss(
        d,
        e,
        target_total=target_open_channel_total,
    )
    cohort_recurrence = compute_cohort_recurrence_loss(
        A,
        d,
        e,
        template_A=template_A,
        template_d=template_d,
        template_e=template_e,
    )
    geometry_structure = compute_geometry_structure_loss(
        A,
        geometry_cost_matrix=geometry_cost_matrix,
        target_diagonal_fraction=target_diagonal_fraction,
    )
    total = (
        resolved_weights.resolved_observation_data_fit * observation_data_fit
        + float(resolved_weights.patient_consistency) * patient_consistency
        + resolved_weights.resolved_open_relation * open_relation
        + float(resolved_weights.cohort_recurrence) * cohort_recurrence
        + resolved_weights.resolved_geometry_structure * geometry_structure
    )
    return LossBreakdown(
        observation_data_fit=float(observation_data_fit),
        patient_consistency=float(patient_consistency),
        open_relation=float(open_relation),
        cohort_recurrence=float(cohort_recurrence),
        geometry_structure=float(geometry_structure),
        total=float(total),
    )


def aggregate_loss_breakdowns(
    breakdowns: Sequence[LossBreakdown],
) -> LossBreakdown:
    """Aggregate per-patient losses into one cohort-wide loss bundle."""
    if len(breakdowns) == 0:
        return LossBreakdown(
            observation_data_fit=0.0,
            patient_consistency=0.0,
            open_relation=0.0,
            cohort_recurrence=0.0,
            geometry_structure=0.0,
            total=0.0,
        )
    return LossBreakdown(
        observation_data_fit=float(sum(item.observation_data_fit for item in breakdowns)),
        patient_consistency=float(sum(item.patient_consistency for item in breakdowns)),
        open_relation=float(sum(item.open_relation for item in breakdowns)),
        cohort_recurrence=float(sum(item.cohort_recurrence for item in breakdowns)),
        geometry_structure=float(sum(item.geometry_structure for item in breakdowns)),
        total=float(sum(item.total for item in breakdowns)),
    )


def build_total_objective(*args: object, **kwargs: object) -> LossBreakdown:
    """Thin public alias for the grouped loss bundle."""
    return evaluate_loss_bundle(*args, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "LossBreakdown",
    "LossWeights",
    "aggregate_loss_breakdowns",
    "build_total_objective",
    "compute_cohort_recurrence_loss",
    "compute_data_fit_loss",
    "compute_geometry_structure_loss",
    "compute_open_channel_control_loss",
    "compute_open_relation_loss",
    "compute_patient_consistency_loss",
    "compute_structural_bias_loss",
    "evaluate_loss_bundle",
]
