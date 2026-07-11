"""Stage 0 to Block 1 observation construction wrappers."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from stride.errors import ContractError

from ...config import TaskAConfigBundle, TaskAOrderedPairFamilySpec
from ...workflows.stride_adapter import (
    TaskAFovRecord,
    TaskAStage0Handle,
    TaskAStateAxis,
    build_task_a_family_observations,
    load_task_a_dataset_handle,
    resolve_task_a_state_basis,
)
from .schemas import validate_block1_family_contract


def resolve_block1_confirmatory_families(
    task_config: TaskAConfigBundle,
) -> tuple[TaskAOrderedPairFamilySpec, TaskAOrderedPairFamilySpec]:
    """Return TC-IM and TC-PT family specs in frozen order."""
    families = tuple(
        family_spec
        for family_spec in task_config.ordered_proxy.pair_families
        if family_spec.name in {"TC-IM", "TC-PT"}
    )
    if tuple(family.name for family in families) != ("TC-IM", "TC-PT"):
        raise ContractError("Block 1 requires confirmatory families TC-IM and TC-PT in frozen order")
    for family in families:
        validate_block1_family_contract(
            family.name,
            family.source_domain,
            family.target_domain,
            family.claim_role,
        )
    return families  # type: ignore[return-value]


def load_block1_stage0_inputs(stage0_h5ad_path: Path) -> TaskAStage0Handle:
    """Load and minimally validate the Stage 0 h5ad surface for Block 1."""
    handle = load_task_a_dataset_handle(stage0_h5ad_path)
    handle.validate(
        require_cell_type=True,
        require_state_axis=True,
        require_cost_scale=True,
        require_cost_matrix=True,
    )
    return handle


def build_block1_family_observations(
    handle: TaskAStage0Handle | Any | Path | str,
    family_spec: TaskAOrderedPairFamilySpec,
    *,
    state_basis: TaskAStateAxis,
    patient_ids: Sequence[str] | None = None,
    mass_mode: str = "uniform",
) -> tuple[TaskAFovRecord, ...]:
    """Build canonical observations for one Block 1 family."""
    dataset_handle = load_task_a_dataset_handle(handle)
    observations = build_task_a_family_observations(
        dataset_handle,
        family_spec,
        state_basis=state_basis,
        mass_mode=mass_mode,
        require_complete_patients=True,
    )
    if patient_ids is None:
        return observations
    allowed_patient_ids = {str(patient_id) for patient_id in patient_ids}
    return tuple(
        observation
        for observation in observations
        if str(observation.patient_id) in allowed_patient_ids
    )


def build_block1_observation_bundles(
    handle: TaskAStage0Handle | Any | Path | str,
    family_specs: Sequence[TaskAOrderedPairFamilySpec],
    *,
    state_basis: TaskAStateAxis,
    patient_ids: Sequence[str] | None = None,
    mass_mode: str = "uniform",
) -> dict[str, tuple[TaskAFovRecord, ...]]:
    """Build observation sequences keyed by pair_family."""
    dataset_handle = load_task_a_dataset_handle(handle)
    return {
        family_spec.name: build_block1_family_observations(
            dataset_handle,
            family_spec,
            state_basis=state_basis,
            patient_ids=patient_ids,
            mass_mode=mass_mode,
        )
        for family_spec in family_specs
    }


__all__ = [
    "build_block1_family_observations",
    "build_block1_observation_bundles",
    "load_block1_stage0_inputs",
    "resolve_block1_confirmatory_families",
    "resolve_task_a_state_basis",
]
