"""Block 1 preprocessing entrypoint for Stage 0 inputs and Task A families."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config import TaskAConfigBundle, TaskAOrderedPairFamilySpec, load_task_a_config_bundle
from ..workflows.stride_adapter import TaskAFovRecord, TaskAStage0Handle, TaskAStateAxis
from .functions.observations import (
    build_block1_observation_bundles,
    load_block1_stage0_inputs,
    resolve_block1_confirmatory_families,
    resolve_task_a_state_basis,
)


@dataclass(frozen=True)
class Block1PreprocessBundle:
    """Resolved Block 1 inputs before fitting."""

    task_config: TaskAConfigBundle
    dataset_handle: TaskAStage0Handle
    state_basis: TaskAStateAxis
    family_specs: tuple[TaskAOrderedPairFamilySpec, TaskAOrderedPairFamilySpec]
    observations_by_family: dict[str, tuple[TaskAFovRecord, ...]]


def prepare_block1_inputs(
    *,
    task_config_path: Path,
    stage0_h5ad_path: Path,
    patient_ids: Sequence[str] = (),
) -> Block1PreprocessBundle:
    """Resolve the frozen Block 1 families and their canonical observations."""
    task_config = load_task_a_config_bundle(task_config_path)
    family_specs = resolve_block1_confirmatory_families(task_config)
    dataset_handle = load_block1_stage0_inputs(stage0_h5ad_path)
    state_basis = resolve_task_a_state_basis(dataset_handle)
    observations_by_family = build_block1_observation_bundles(
        dataset_handle,
        family_specs,
        state_basis=state_basis,
        patient_ids=tuple(patient_ids) or None,
        mass_mode=task_config.data.mass_mode,
    )
    return Block1PreprocessBundle(
        task_config=task_config,
        dataset_handle=dataset_handle,
        state_basis=state_basis,
        family_specs=family_specs,
        observations_by_family=observations_by_family,
    )


__all__ = [
    "Block1PreprocessBundle",
    "prepare_block1_inputs",
]
