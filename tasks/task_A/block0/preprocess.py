"""Preprocessing entrypoints for Task A Block 0 calibration.

This module owns the root-level preparation surface for Block 0: resolving the
run config, building real `TC-IM` observations, and building patient-preserving
null observation bundles. Low-level dataclasses and helper routines live under
`tasks.task_A.block0.functions`.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from stride import DatasetHandle
from stride.observation.contracts import FovObservation

from ..config import TaskAConfigBundle
from .functions.observations import (
    Block0ObservationBundle,
    build_null_tc_im_observations,
    build_real_tc_im_observations,
    resolve_block0_run_config,
)
from .functions.permutation import build_domain_label_permutation_assignments
from .functions.schemas import Block0DomainLabelPermutationAssignment, Block0RunConfig


def resolve_run_config(
    *,
    config_path: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    n_permutations: int,
    master_seed: int,
    patient_ids: tuple[str, ...] | None = None,
    demo_subset_name: str | None = None,
) -> Block0RunConfig:
    """Resolve Block 0 path/scope controls before loading Stage 0 data."""
    return resolve_block0_run_config(
        config_path=config_path,
        data_path=data_path,
        output_dir=output_dir,
        n_permutations=n_permutations,
        master_seed=master_seed,
        patient_ids=patient_ids,
        demo_subset_name=demo_subset_name,
    )


def build_real_observations(
    handle: DatasetHandle,
    config_bundle: TaskAConfigBundle,
    *,
    patient_ids: Sequence[str] | None = None,
) -> Block0ObservationBundle:
    """Build real `TC-IM` observations from Stage 0 and Task A config."""
    return build_real_tc_im_observations(
        handle,
        config_bundle,
        patient_ids=patient_ids,
    )


def build_null_observations(
    real_observations: Block0ObservationBundle,
    *,
    permutation_index: int,
    master_seed: int,
) -> Block0ObservationBundle:
    """Build one within-patient count-preserving null observation bundle."""
    assignments = build_null_assignments(
        real_observations.observations,
        permutation_index=permutation_index,
        master_seed=master_seed,
    )
    return build_null_tc_im_observations(
        real_observations,
        assignments,
        permutation_index=permutation_index,
    )


def build_null_assignments(
    observations: Sequence[FovObservation],
    *,
    permutation_index: int,
    master_seed: int,
) -> tuple[Block0DomainLabelPermutationAssignment, ...]:
    """Build deterministic within-patient TC/IM label assignments."""
    return build_domain_label_permutation_assignments(
        observations,
        permutation_index=permutation_index,
        master_seed=master_seed,
    )


__all__ = [
    "Block0ObservationBundle",
    "build_null_assignments",
    "build_null_observations",
    "build_real_observations",
    "resolve_run_config",
]
