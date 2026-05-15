"""Canonical STRIDE fit workflow."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..audit.results import assemble_stride_fit_result
from ..audit.status import FitRunStatus
from ..basis.contracts import StateBasis
from ..errors import ContractError
from ..geometry.state_geometry import StateGeometry
from ..losses import EvidenceBlock
from ..optimize import TrainConfig, run_training
from ..observation.contracts import FovObservation
from ..outputs.fit_result import STRIDEFitResult
from ._fit_inputs import _FitObservationGroup, _PatientFitInput, _build_patient_fit_inputs
from .config import TaskConfig


def _group_state_matrix(observations: Sequence[FovObservation]) -> np.ndarray:
    return np.vstack(
        [np.asarray(observation.community_composition, dtype=float) for observation in observations]
    ).astype(float, copy=False)


def _patient_inputs_support_full_optimizer(patient_inputs: Sequence[_PatientFitInput]) -> bool:
    if not patient_inputs:
        return False
    return all(len(patient_input.groups) == 2 for patient_input in patient_inputs)


def _subset_counts_by_domain(groups: Sequence[_FitObservationGroup]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for group in groups:
        for domain_label, observations in group.observations_by_domain.items():
            counts[str(domain_label)] = counts.get(str(domain_label), 0) + len(observations)
    return counts


def _subset_counts_by_group_and_domain(
    groups: Sequence[_FitObservationGroup],
) -> dict[str, dict[str, int]]:
    return {
        group.group_label: {
            str(domain_label): len(observations)
            for domain_label, observations in group.observations_by_domain.items()
        }
        for group in groups
    }


def _select_task_patient_inputs(
    patient_inputs: Sequence[_PatientFitInput],
    *,
    task_config: TaskConfig,
) -> tuple[tuple[_PatientFitInput, ...], tuple[str, ...]]:
    selected_inputs: list[_PatientFitInput] = []
    missing_patient_ids: list[str] = []
    for patient_input in patient_inputs:
        group_by_label = {group.group_label: group for group in patient_input.groups}
        source_group = group_by_label.get(task_config.source)
        target_group = group_by_label.get(task_config.target)
        if source_group is None or target_group is None:
            missing_patient_ids.append(patient_input.patient_id)
            selected_inputs.append(patient_input)
            continue
        selected_groups = (source_group, target_group)
        selected_inputs.append(
            _PatientFitInput(
                patient_id=patient_input.patient_id,
                ordered_group_labels=(task_config.source, task_config.target),
                groups=selected_groups,
                n_observations_by_group={
                    group.group_label: len(group.observations) for group in selected_groups
                },
                n_observations_by_domain=_subset_counts_by_domain(selected_groups),
                n_observations_by_group_and_domain=_subset_counts_by_group_and_domain(
                    selected_groups
                ),
                state_basis=patient_input.state_basis,
                geometry=patient_input.geometry,
                mass_mode=patient_input.mass_mode,
                metadata={
                    **dict(patient_input.metadata),
                    "observed_group_labels": tuple(patient_input.ordered_group_labels),
                    "task_source": task_config.source,
                    "task_target": task_config.target,
                },
            )
        )
    return tuple(selected_inputs), tuple(missing_patient_ids)


def _build_evidence_blocks(patient_inputs: Sequence[_PatientFitInput]) -> tuple[EvidenceBlock, ...]:
    blocks: list[EvidenceBlock] = []
    for patient_input in patient_inputs:
        source_group, target_group = patient_input.groups
        for source_domain, source_observations in sorted(source_group.observations_by_domain.items()):
            for target_domain, target_observations in sorted(target_group.observations_by_domain.items()):
                block_id = (
                    f"{patient_input.patient_id}:"
                    f"{source_group.group_label}:{source_domain}->"
                    f"{target_group.group_label}:{target_domain}"
                )
                blocks.append(
                    EvidenceBlock(
                        patient_id=patient_input.patient_id,
                        source_bag=_group_state_matrix(source_observations),
                        target_bag=_group_state_matrix(target_observations),
                        block_id=block_id,
                    )
                )
    if not blocks:
        raise ContractError("Canonical full estimator requires at least one evidence block")
    return tuple(blocks)


def run_stride_fit(
    observations: tuple[FovObservation, ...] | list[FovObservation],
    *,
    task_config: TaskConfig,
    train_config: TrainConfig | None = None,
    state_basis: StateBasis | None = None,
    geometry: StateGeometry | None = None,
) -> STRIDEFitResult:
    """Run the canonical end-to-end STRIDE fit path."""
    if geometry is None:
        raise ContractError("canonical fit_stride requires resolved geometry")
    K = int(np.asarray(geometry.cost_matrix, dtype=float).shape[0])
    if int(task_config.K) != K:
        raise ContractError("TaskConfig.K must match geometry dimension")
    if state_basis is not None and int(state_basis.n_states) != int(task_config.K):
        raise ContractError("TaskConfig.K must match state_basis.n_states")

    patient_inputs = _build_patient_fit_inputs(
        observations,
        state_basis=state_basis,
        geometry=geometry,
        timepoint_order=task_config.timepoint_order,
    )
    if task_config.patient_ids is not None:
        requested = tuple(task_config.patient_ids)
        patient_inputs = tuple(
            patient_input for patient_input in patient_inputs if patient_input.patient_id in requested
        )
        if tuple(patient_input.patient_id for patient_input in patient_inputs) != requested:
            raise ContractError("patient_ids must match observed patient order and identifiers")

    patient_inputs, missing_task_patient_ids = _select_task_patient_inputs(
        patient_inputs,
        task_config=task_config,
    )
    state_ids = tuple(geometry.state_ids) if geometry is not None else None
    if missing_task_patient_ids:
        train_result = None
        run_status = FitRunStatus(
            stage="task_feasibility",
            status="deferred",
            reason="missing_task_source_or_target",
            context={
                "missing_patient_ids": missing_task_patient_ids,
                "source": task_config.source,
                "target": task_config.target,
            },
        )
        evidence_blocks = ()
    elif not _patient_inputs_support_full_optimizer(patient_inputs):
        train_result = None
        run_status = FitRunStatus(
            stage="task_feasibility",
            status="deferred",
            reason="unsupported_patient_input_shape",
        )
        evidence_blocks: tuple[EvidenceBlock, ...] = ()
    else:
        evidence_blocks = _build_evidence_blocks(patient_inputs)
        train_result = run_training(
            patient_ids=tuple(patient_input.patient_id for patient_input in patient_inputs),
            K=task_config.K,
            evidence_blocks=evidence_blocks,
            geometry=geometry,
            config=train_config,
        )
        run_status = FitRunStatus(
            stage="optimizer",
            status=train_result.status,
            reason=train_result.run_info.reason,
        )

    return assemble_stride_fit_result(
        patient_inputs=patient_inputs,
        task_config=task_config,
        run_status=run_status,
        train_result=train_result,
        evidence_blocks=evidence_blocks,
        state_ids=state_ids,
    )


__all__ = ["run_stride_fit"]
