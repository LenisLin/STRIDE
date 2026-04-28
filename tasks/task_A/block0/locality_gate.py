"""Task-local Block 0 STRIDE-native gate helpers.

This module keeps the Block 0 implementation inside ``tasks/task_A`` while
reusing the stable STRIDE fit path to compare one real Task A family against a
seeded task-local null family. The live gate is defined on summaries derived
directly from realized ``A``, ``d``, and ``e`` outputs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
import pandas as pd

from stride.api.fit import STRIDEFitConfig, fit_stride
from stride.errors import ContractError
from stride.observation.contracts import FovObservation
from stride.outputs.fit_result import PatientBridgeResult, STRIDEFitResult

from ..config import TaskAConfigBundle, TaskAOrderedPairFamilySpec
from ..contracts import CONTRACT_PASSED_STATE, SCAFFOLD_ACTIVE_STATE
from ..workflows.stride_adapter import (
    _assert_timepoint_inert,
    build_task_a_family_observations,
    load_task_a_dataset_handle,
    resolve_task_a_state_basis,
)


BLOCK_NAME = "block0_locality_gate"
REAL_PAIR_FAMILY = "TC-IM"
NULL_PAIR_FAMILY = "TC-IM_randomized_target"
REAL_FAMILIES: tuple[str, ...] = (REAL_PAIR_FAMILY,)
NULL_FAMILIES: tuple[str, ...] = (NULL_PAIR_FAMILY,)
FULL_COHORT_SCOPE = "full_cohort"
PATIENT_SUBSET_SCOPE = "patient_subset"
DEMO_SUBSET_SCOPE = "demo_subset"
BLOCK0_SOURCE_DOMAIN = "TC"
BLOCK0_TARGET_DOMAIN = "IM"
REQUIRED_SUPPORT_FRACTION = 0.75
MIN_REQUIRED_SUPPORT = 6
MEDIAN_DELTA_THRESHOLD = 0.0
FRACTION_DELTA_THRESHOLD = 0.5
PAIR_METRICS_COLUMNS: tuple[str, ...] = (
    "comparison_id",
    "run_scope",
    "pair_family",
    "null_family",
    "anchor_patient_id",
    "null_target_donor_patient_id",
    "source_domain",
    "target_domain",
    "n_source_observations",
    "n_target_observations",
    "count_stratum_key",
    "selection_seed",
    "null_assignment_status",
    "null_assignment_reason",
    "real_fit_status",
    "null_fit_status",
    "real_defer_reason",
    "null_defer_reason",
    "real_total_continuity_mass",
    "null_total_continuity_mass",
    "delta_total_continuity_mass",
    "real_total_depletion_mass",
    "null_total_depletion_mass",
    "delta_total_depletion_mass",
    "real_total_emergence_mass",
    "null_total_emergence_mass",
    "delta_total_emergence_mass",
)
_FIT_STATUS_VALUES: frozenset[str] = frozenset({"ok", "deferred", "failed"})


@dataclass(frozen=True)
class Block0RuntimeConfig:
    random_seed: int
    real_pair_family_name: str
    source_domain: str
    target_domain: str

    @property
    def timepoint_order(self) -> tuple[str, str]:
        return (self.source_domain, self.target_domain)


@dataclass(frozen=True)
class Block0RealFamilyData:
    family_spec: TaskAOrderedPairFamilySpec
    state_basis: Any
    observations: tuple[FovObservation, ...]


@dataclass(frozen=True)
class Block0NullAssignment:
    anchor_patient_id: str
    donor_patient_id: str | None
    n_source_observations: int
    n_target_observations: int
    count_stratum_key: str
    selection_seed: int
    assignment_status: str
    assignment_reason: str | None = None


@dataclass(frozen=True)
class Block0GateResult:
    pair_metrics_df: pd.DataFrame
    n_eligible_patients: int
    required_support: int
    status: str
    artifact_state: str
    block0_passed: bool
    metrics_summary: dict[str, Any]
    gate_checks: dict[str, Any]
    failure_reasons: tuple[str, ...]


def _require_mapping(payload: object, *, where: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ContractError(f"{where} must be a mapping")
    return dict(payload)


def _require_non_negative_int(payload: object, *, where: str) -> int:
    try:
        value = int(payload)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{where} must be an integer") from exc
    if value < 0:
        raise ContractError(f"{where} must be non-negative")
    return value


def _median_or_none(values: pd.Series) -> float | None:
    if values.empty:
        return None
    value = float(values.median())
    return value if np.isfinite(value) else None


def _fraction_or_none(mask: pd.Series) -> float | None:
    if mask.empty:
        return None
    value = float(mask.mean())
    return value if np.isfinite(value) else None


def _json_scalar(value: object) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return _json_scalar(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sorted_observations(
    observations: tuple[FovObservation, ...],
) -> tuple[FovObservation, ...]:
    return tuple(
        sorted(
            observations,
            key=lambda observation: (
                str(observation.patient_id),
                str(observation.timepoint),
                str(observation.fov_id),
            ),
        )
    )


def resolve_block0_real_family_spec(config_bundle: TaskAConfigBundle) -> TaskAOrderedPairFamilySpec:
    for family in config_bundle.ordered_proxy.pair_families:
        if (
            family.name == REAL_PAIR_FAMILY
            and family.source_domain == BLOCK0_SOURCE_DOMAIN
            and family.target_domain == BLOCK0_TARGET_DOMAIN
        ):
            return family
    raise ContractError(
        "Task A Block 0 requires the frozen real family "
        f"{REAL_PAIR_FAMILY!r} with domains {BLOCK0_SOURCE_DOMAIN!r}->{BLOCK0_TARGET_DOMAIN!r}"
    )


def load_block0_runtime_config(config_bundle: TaskAConfigBundle) -> Block0RuntimeConfig:
    raw_config = dict(config_bundle.raw_config)
    block0 = _require_mapping(raw_config.get("block0"), where="Task A config key 'block0'")
    family_spec = resolve_block0_real_family_spec(config_bundle)
    return Block0RuntimeConfig(
        random_seed=_require_non_negative_int(
            block0.get("random_seed"),
            where="Task A config key 'block0.random_seed'",
        ),
        real_pair_family_name=family_spec.name,
        source_domain=family_spec.source_domain,
        target_domain=family_spec.target_domain,
    )


def empty_block0_pair_metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(PAIR_METRICS_COLUMNS))


def _prepare_block0_real_family(
    stage0_h5ad_or_adata: str | Any,
    *,
    config_bundle: TaskAConfigBundle,
) -> Block0RealFamilyData:
    handle = load_task_a_dataset_handle(stage0_h5ad_or_adata)
    handle.validate(
        require_cell_type=True,
        require_state_axis=True,
        require_cost_scale=True,
        require_cost_matrix=True,
    )
    _assert_timepoint_inert(handle)
    state_basis = resolve_task_a_state_basis(handle)
    family_spec = resolve_block0_real_family_spec(config_bundle)
    observations = build_task_a_family_observations(
        handle,
        family_spec,
        state_basis=state_basis,
        mass_mode=config_bundle.data.mass_mode,
        require_complete_patients=True,
    )
    return Block0RealFamilyData(
        family_spec=family_spec,
        state_basis=state_basis,
        observations=_sorted_observations(observations),
    )


def build_block0_real_family_observations(
    stage0_h5ad_or_adata: str | Any,
    *,
    config_bundle: TaskAConfigBundle,
) -> tuple[FovObservation, ...]:
    return _prepare_block0_real_family(
        stage0_h5ad_or_adata,
        config_bundle=config_bundle,
    ).observations


def _observations_by_patient_and_group(
    observations: tuple[FovObservation, ...],
    *,
    source_domain: str,
    target_domain: str,
) -> dict[str, dict[str, tuple[FovObservation, ...]]]:
    grouped: dict[str, dict[str, list[FovObservation]]] = defaultdict(lambda: defaultdict(list))
    for observation in observations:
        grouped[str(observation.patient_id)][str(observation.timepoint)].append(observation)

    resolved: dict[str, dict[str, tuple[FovObservation, ...]]] = {}
    for patient_id, groups in grouped.items():
        source_observations = tuple(sorted(groups.get(source_domain, []), key=lambda item: str(item.fov_id)))
        target_observations = tuple(sorted(groups.get(target_domain, []), key=lambda item: str(item.fov_id)))
        if not source_observations:
            continue
        resolved[patient_id] = {
            source_domain: source_observations,
            target_domain: target_observations,
        }
    return dict(sorted(resolved.items()))


def _count_stratum_key(
    *,
    source_domain: str,
    target_domain: str,
    n_source_observations: int,
    n_target_observations: int,
) -> str:
    return (
        f"{source_domain}:{int(n_source_observations)}|"
        f"{target_domain}:{int(n_target_observations)}"
    )


def _sample_derangement(
    patient_ids: tuple[str, ...],
    *,
    rng: np.random.Generator,
) -> tuple[str, ...]:
    if len(patient_ids) < 2:
        raise ContractError("Derangement sampling requires at least two patient ids")
    if len(patient_ids) == 2:
        return (patient_ids[1], patient_ids[0])

    indices = np.arange(len(patient_ids), dtype=int)
    for _ in range(2048):
        candidate = rng.permutation(len(patient_ids))
        if np.all(candidate != indices):
            return tuple(patient_ids[int(index)] for index in candidate.tolist())

    randomized_cycle = rng.permutation(len(patient_ids)).tolist()
    return tuple(
        patient_ids[int(randomized_cycle[(idx + 1) % len(randomized_cycle)])]
        for idx in range(len(randomized_cycle))
    )


def build_block0_null_assignments(
    real_observations: tuple[FovObservation, ...],
    runtime_config: Block0RuntimeConfig,
) -> tuple[Block0NullAssignment, ...]:
    if not real_observations:
        return ()

    by_patient = _observations_by_patient_and_group(
        real_observations,
        source_domain=runtime_config.source_domain,
        target_domain=runtime_config.target_domain,
    )
    strata: dict[tuple[int, int], list[str]] = defaultdict(list)
    for patient_id, groups in by_patient.items():
        n_source = len(groups.get(runtime_config.source_domain, ()))
        n_target = len(groups.get(runtime_config.target_domain, ()))
        strata[(n_source, n_target)].append(patient_id)

    rng = np.random.default_rng(runtime_config.random_seed)
    assignments: list[Block0NullAssignment] = []
    for (n_source, n_target), patient_ids in sorted(strata.items()):
        ordered_patients = tuple(sorted(patient_ids))
        stratum_key = _count_stratum_key(
            source_domain=runtime_config.source_domain,
            target_domain=runtime_config.target_domain,
            n_source_observations=n_source,
            n_target_observations=n_target,
        )
        if len(ordered_patients) < 2:
            for patient_id in ordered_patients:
                assignments.append(
                    Block0NullAssignment(
                        anchor_patient_id=patient_id,
                        donor_patient_id=None,
                        n_source_observations=n_source,
                        n_target_observations=n_target,
                        count_stratum_key=stratum_key,
                        selection_seed=runtime_config.random_seed,
                        assignment_status="deferred",
                        assignment_reason="count_stratum_has_fewer_than_two_patients",
                    )
                )
            continue

        donors = _sample_derangement(ordered_patients, rng=rng)
        for anchor_patient_id, donor_patient_id in zip(ordered_patients, donors, strict=True):
            assignments.append(
                Block0NullAssignment(
                    anchor_patient_id=anchor_patient_id,
                    donor_patient_id=donor_patient_id,
                    n_source_observations=n_source,
                    n_target_observations=n_target,
                    count_stratum_key=stratum_key,
                    selection_seed=runtime_config.random_seed,
                    assignment_status="assigned",
                    assignment_reason=None,
                )
            )
    return tuple(assignments)


def build_block0_null_family_observations(
    real_observations: tuple[FovObservation, ...],
    runtime_config: Block0RuntimeConfig,
    assignments: tuple[Block0NullAssignment, ...],
) -> tuple[FovObservation, ...]:
    by_patient = _observations_by_patient_and_group(
        real_observations,
        source_domain=runtime_config.source_domain,
        target_domain=runtime_config.target_domain,
    )
    null_observations: list[FovObservation] = []
    for assignment in assignments:
        anchor_groups = by_patient.get(assignment.anchor_patient_id)
        if anchor_groups is None:
            raise ContractError(
                "Task A Block 0 null construction lost the real-family anchor patient "
                f"{assignment.anchor_patient_id!r}"
            )
        for observation in anchor_groups.get(runtime_config.source_domain, ()):
            null_observations.append(
                FovObservation(
                    patient_id=str(assignment.anchor_patient_id),
                    timepoint=str(observation.timepoint),
                    fov_id=str(observation.fov_id),
                    community_composition=np.asarray(observation.community_composition, dtype=float).copy(),
                    mass=float(observation.mass),
                    mass_mode=str(observation.mass_mode),
                    domain_label=observation.domain_label,
                    metadata={
                        **dict(observation.metadata),
                        "block0_family": NULL_PAIR_FAMILY,
                        "block0_null_role": "source_anchor",
                        "block0_anchor_patient_id": str(assignment.anchor_patient_id),
                    },
                )
            )
        if assignment.donor_patient_id is None:
            continue

        donor_groups = by_patient.get(assignment.donor_patient_id)
        if donor_groups is None:
            raise ContractError(
                "Task A Block 0 null construction referenced an unknown donor patient "
                f"{assignment.donor_patient_id!r}"
            )
        for observation in donor_groups.get(runtime_config.target_domain, ()):
            null_observations.append(
                FovObservation(
                    patient_id=str(assignment.anchor_patient_id),
                    timepoint=str(observation.timepoint),
                    fov_id=(
                        f"{observation.fov_id}__{NULL_PAIR_FAMILY}__"
                        f"anchor_{assignment.anchor_patient_id}"
                    ),
                    community_composition=np.asarray(observation.community_composition, dtype=float).copy(),
                    mass=float(observation.mass),
                    mass_mode=str(observation.mass_mode),
                    domain_label=observation.domain_label,
                    metadata={
                        **dict(observation.metadata),
                        "block0_family": NULL_PAIR_FAMILY,
                        "block0_null_role": "target_donor",
                        "block0_anchor_patient_id": str(assignment.anchor_patient_id),
                        "block0_target_donor_patient_id": str(assignment.donor_patient_id),
                    },
                )
            )
    return _sorted_observations(tuple(null_observations))


def summarize_block0_bridge_result_totals(
    result: PatientBridgeResult,
) -> dict[str, float] | None:
    if result.fit_status != "ok" or result.A is None or result.d is None or result.e is None:
        return None
    matrix = np.asarray(result.A, dtype=float)
    depletion = np.asarray(result.d, dtype=float)
    emergence = np.asarray(result.e, dtype=float)
    return {
        "total_continuity_mass": float(np.sum(matrix, dtype=float)),
        "total_depletion_mass": float(np.sum(depletion, dtype=float)),
        "total_emergence_mass": float(np.sum(emergence, dtype=float)),
    }


def _defer_reason(result: PatientBridgeResult | None) -> str | None:
    if result is None:
        return "missing_patient_result"
    raw = result.diagnostics.get("defer_reason")
    if raw in (None, ""):
        return None
    return str(raw)


def _comparison_id(patient_id: str) -> str:
    return f"{BLOCK_NAME}::{REAL_PAIR_FAMILY}::{patient_id}"


def _validate_block0_pair_metrics_frame(frame: pd.DataFrame) -> None:
    missing = [column for column in PAIR_METRICS_COLUMNS if column not in frame.columns]
    if missing:
        raise ContractError(
            f"Task A Block 0 pair metrics are missing required columns: {missing}"
        )
    if frame["comparison_id"].isna().any():
        raise ContractError("Task A Block 0 pair metrics contain NA comparison_id values")
    if frame["comparison_id"].duplicated().any():
        raise ContractError("Task A Block 0 pair metrics contain duplicate comparison_id values")

    for column_name in ("real_fit_status", "null_fit_status"):
        invalid = ~frame[column_name].astype(str).isin(_FIT_STATUS_VALUES)
        if invalid.any():
            values = sorted(frame.loc[invalid, column_name].astype(str).unique().tolist())
            raise ContractError(
                f"Task A Block 0 pair metrics contain invalid {column_name} values: {values}"
            )

    real_metric_columns = (
        "real_total_continuity_mass",
        "real_total_depletion_mass",
        "real_total_emergence_mass",
    )
    null_metric_columns = (
        "null_total_continuity_mass",
        "null_total_depletion_mass",
        "null_total_emergence_mass",
    )
    delta_metric_columns = (
        "delta_total_continuity_mass",
        "delta_total_depletion_mass",
        "delta_total_emergence_mass",
    )
    real_non_ok = frame["real_fit_status"].astype(str) != "ok"
    null_non_ok = frame["null_fit_status"].astype(str) != "ok"
    if (~frame.loc[real_non_ok, list(real_metric_columns)].isna()).any().any():
        raise ContractError(
            "Task A Block 0 pair metrics require NaN real-family summaries when real_fit_status!=ok"
        )
    if (~frame.loc[null_non_ok, list(null_metric_columns)].isna()).any().any():
        raise ContractError(
            "Task A Block 0 pair metrics require NaN null-family summaries when null_fit_status!=ok"
        )
    either_non_ok = real_non_ok | null_non_ok
    if (~frame.loc[either_non_ok, list(delta_metric_columns)].isna()).any().any():
        raise ContractError(
            "Task A Block 0 pair metrics require NaN deltas unless both real and null fits are ok"
        )


def build_block0_pair_metrics_frame(
    *,
    real_result: STRIDEFitResult,
    null_result: STRIDEFitResult,
    assignments: tuple[Block0NullAssignment, ...],
    runtime_config: Block0RuntimeConfig,
    run_scope: str,
) -> pd.DataFrame:
    real_by_patient = {
        str(patient_result.patient_id): patient_result
        for patient_result in real_result.patient_results
    }
    null_by_patient = {
        str(patient_result.patient_id): patient_result
        for patient_result in null_result.patient_results
    }
    rows: list[dict[str, Any]] = []
    for assignment in assignments:
        real_patient_result = real_by_patient.get(assignment.anchor_patient_id)
        if real_patient_result is None:
            raise ContractError(
                "Task A Block 0 real family fit did not emit a patient result for "
                f"{assignment.anchor_patient_id!r}"
            )
        null_patient_result = null_by_patient.get(assignment.anchor_patient_id)
        if null_patient_result is None:
            raise ContractError(
                "Task A Block 0 null family fit did not emit a patient result for "
                f"{assignment.anchor_patient_id!r}"
            )

        real_summary = summarize_block0_bridge_result_totals(real_patient_result)
        null_summary = summarize_block0_bridge_result_totals(null_patient_result)
        continuity_delta = (
            float(real_summary["total_continuity_mass"] - null_summary["total_continuity_mass"])
            if real_summary is not None and null_summary is not None
            else np.nan
        )
        depletion_delta = (
            float(real_summary["total_depletion_mass"] - null_summary["total_depletion_mass"])
            if real_summary is not None and null_summary is not None
            else np.nan
        )
        emergence_delta = (
            float(real_summary["total_emergence_mass"] - null_summary["total_emergence_mass"])
            if real_summary is not None and null_summary is not None
            else np.nan
        )
        rows.append(
            {
                "comparison_id": _comparison_id(assignment.anchor_patient_id),
                "run_scope": run_scope,
                "pair_family": REAL_PAIR_FAMILY,
                "null_family": NULL_PAIR_FAMILY,
                "anchor_patient_id": assignment.anchor_patient_id,
                "null_target_donor_patient_id": (
                    ""
                    if assignment.donor_patient_id is None
                    else assignment.donor_patient_id
                ),
                "source_domain": runtime_config.source_domain,
                "target_domain": runtime_config.target_domain,
                "n_source_observations": int(assignment.n_source_observations),
                "n_target_observations": int(assignment.n_target_observations),
                "count_stratum_key": assignment.count_stratum_key,
                "selection_seed": int(assignment.selection_seed),
                "null_assignment_status": assignment.assignment_status,
                "null_assignment_reason": (
                    ""
                    if assignment.assignment_reason is None
                    else assignment.assignment_reason
                ),
                "real_fit_status": str(real_patient_result.fit_status),
                "null_fit_status": str(null_patient_result.fit_status),
                "real_defer_reason": "" if _defer_reason(real_patient_result) is None else _defer_reason(real_patient_result),
                "null_defer_reason": "" if _defer_reason(null_patient_result) is None else _defer_reason(null_patient_result),
                "real_total_continuity_mass": (
                    np.nan
                    if real_summary is None
                    else float(real_summary["total_continuity_mass"])
                ),
                "null_total_continuity_mass": (
                    np.nan
                    if null_summary is None
                    else float(null_summary["total_continuity_mass"])
                ),
                "delta_total_continuity_mass": continuity_delta,
                "real_total_depletion_mass": (
                    np.nan
                    if real_summary is None
                    else float(real_summary["total_depletion_mass"])
                ),
                "null_total_depletion_mass": (
                    np.nan
                    if null_summary is None
                    else float(null_summary["total_depletion_mass"])
                ),
                "delta_total_depletion_mass": depletion_delta,
                "real_total_emergence_mass": (
                    np.nan
                    if real_summary is None
                    else float(real_summary["total_emergence_mass"])
                ),
                "null_total_emergence_mass": (
                    np.nan
                    if null_summary is None
                    else float(null_summary["total_emergence_mass"])
                ),
                "delta_total_emergence_mass": emergence_delta,
            }
        )

    frame = pd.DataFrame.from_records(rows, columns=list(PAIR_METRICS_COLUMNS))
    _validate_block0_pair_metrics_frame(frame)
    return frame


def evaluate_block0_family_fits(
    source: str | Any,
    *,
    config_bundle: TaskAConfigBundle,
    runtime_config: Block0RuntimeConfig,
    run_scope: str,
) -> tuple[pd.DataFrame, int]:
    real_family = _prepare_block0_real_family(source, config_bundle=config_bundle)
    if not real_family.observations:
        return empty_block0_pair_metrics_frame(), 0

    assignments = build_block0_null_assignments(
        real_family.observations,
        runtime_config,
    )
    null_observations = build_block0_null_family_observations(
        real_family.observations,
        runtime_config,
        assignments,
    )
    fit_config = STRIDEFitConfig(timepoint_order=runtime_config.timepoint_order)
    real_result = fit_stride(
        real_family.observations,
        state_basis=real_family.state_basis,
        config=fit_config,
    )
    null_result = fit_stride(
        null_observations,
        state_basis=real_family.state_basis,
        config=fit_config,
    )
    frame = build_block0_pair_metrics_frame(
        real_result=real_result,
        null_result=null_result,
        assignments=assignments,
        runtime_config=runtime_config,
        run_scope=run_scope,
    )
    return frame, len(assignments)


def _build_family_summary(
    *,
    pair_metrics_df: pd.DataFrame,
    fit_status_column: str,
    continuity_column: str,
    depletion_column: str,
    emergence_column: str,
    family_name: str,
) -> dict[str, Any]:
    status_counts = Counter(pair_metrics_df[fit_status_column].astype(str))
    ok_rows = pair_metrics_df.loc[pair_metrics_df[fit_status_column].astype(str) == "ok"].copy()
    return {
        "family_name": family_name,
        "n_patients": int(pair_metrics_df.shape[0]),
        "fit_status_counts": {
            "ok": int(status_counts.get("ok", 0)),
            "deferred": int(status_counts.get("deferred", 0)),
            "failed": int(status_counts.get("failed", 0)),
        },
        "median_total_continuity_mass": _median_or_none(ok_rows[continuity_column]),
        "median_total_depletion_mass": _median_or_none(ok_rows[depletion_column]),
        "median_total_emergence_mass": _median_or_none(ok_rows[emergence_column]),
    }


def _build_paired_summary(
    *,
    pair_metrics_df: pd.DataFrame,
    required_support: int,
) -> dict[str, Any]:
    paired_ok_rows = pair_metrics_df.loc[
        (pair_metrics_df["real_fit_status"].astype(str) == "ok")
        & (pair_metrics_df["null_fit_status"].astype(str) == "ok")
    ].copy()
    return {
        "required_support": int(required_support),
        "paired_support": int(paired_ok_rows.shape[0]),
        "median_delta_total_continuity_mass": _median_or_none(
            paired_ok_rows["delta_total_continuity_mass"]
        ),
        "fraction_real_total_continuity_mass_gt_null": _fraction_or_none(
            paired_ok_rows["delta_total_continuity_mass"] > 0.0
        ),
        "median_delta_total_depletion_mass": _median_or_none(
            paired_ok_rows["delta_total_depletion_mass"]
        ),
        "fraction_real_total_depletion_mass_lt_null": _fraction_or_none(
            paired_ok_rows["delta_total_depletion_mass"] < 0.0
        ),
        "median_delta_total_emergence_mass": _median_or_none(
            paired_ok_rows["delta_total_emergence_mass"]
        ),
        "fraction_real_total_emergence_mass_lt_null": _fraction_or_none(
            paired_ok_rows["delta_total_emergence_mass"] < 0.0
        ),
    }


def aggregate_block0_gate(
    *,
    pair_metrics_df: pd.DataFrame,
    n_eligible_patients: int,
    run_scope: str,
    pre_block0_artifact_state: str,
) -> Block0GateResult:
    required_support = (
        max(MIN_REQUIRED_SUPPORT, int(ceil(REQUIRED_SUPPORT_FRACTION * n_eligible_patients)))
        if n_eligible_patients > 0
        else MIN_REQUIRED_SUPPORT
    )
    real_family_summary = _build_family_summary(
        pair_metrics_df=pair_metrics_df,
        fit_status_column="real_fit_status",
        continuity_column="real_total_continuity_mass",
        depletion_column="real_total_depletion_mass",
        emergence_column="real_total_emergence_mass",
        family_name=REAL_PAIR_FAMILY,
    )
    null_family_summary = _build_family_summary(
        pair_metrics_df=pair_metrics_df,
        fit_status_column="null_fit_status",
        continuity_column="null_total_continuity_mass",
        depletion_column="null_total_depletion_mass",
        emergence_column="null_total_emergence_mass",
        family_name=NULL_PAIR_FAMILY,
    )
    paired_summary = _build_paired_summary(
        pair_metrics_df=pair_metrics_df,
        required_support=required_support,
    )
    paired_support = int(paired_summary["paired_support"])
    median_continuity_delta = paired_summary["median_delta_total_continuity_mass"]
    fraction_continuity_delta = paired_summary["fraction_real_total_continuity_mass_gt_null"]
    median_emergence_delta = paired_summary["median_delta_total_emergence_mass"]
    fraction_emergence_delta = paired_summary["fraction_real_total_emergence_mass_lt_null"]

    gate_checks: dict[str, Any] = {
        "full_cohort_scope_required_for_pass": {
            "passed": run_scope == FULL_COHORT_SCOPE,
            "observed_run_scope": run_scope,
        },
        "pre_block0_data_suitability_contract_passed": {
            "passed": pre_block0_artifact_state == CONTRACT_PASSED_STATE,
            "observed_artifact_state": pre_block0_artifact_state,
        },
        "eligible_patients_positive": {
            "passed": n_eligible_patients > 0,
            "observed": int(n_eligible_patients),
        },
        "paired_support": {
            "passed": paired_support >= required_support,
            "observed": int(paired_support),
            "threshold": int(required_support),
        },
        "median_delta_total_continuity_mass_positive": {
            "passed": median_continuity_delta is not None and float(median_continuity_delta) > MEDIAN_DELTA_THRESHOLD,
            "observed": _json_scalar(median_continuity_delta),
            "threshold": MEDIAN_DELTA_THRESHOLD,
        },
        "fraction_real_total_continuity_mass_gt_null_above_half": {
            "passed": fraction_continuity_delta is not None and float(fraction_continuity_delta) > FRACTION_DELTA_THRESHOLD,
            "observed": _json_scalar(fraction_continuity_delta),
            "threshold": FRACTION_DELTA_THRESHOLD,
        },
        "median_delta_total_emergence_mass_negative": {
            "passed": median_emergence_delta is not None and float(median_emergence_delta) < MEDIAN_DELTA_THRESHOLD,
            "observed": _json_scalar(median_emergence_delta),
            "threshold": MEDIAN_DELTA_THRESHOLD,
        },
        "fraction_real_total_emergence_mass_lt_null_above_half": {
            "passed": fraction_emergence_delta is not None and float(fraction_emergence_delta) > FRACTION_DELTA_THRESHOLD,
            "observed": _json_scalar(fraction_emergence_delta),
            "threshold": FRACTION_DELTA_THRESHOLD,
        },
    }

    failure_reasons: list[str] = []
    if run_scope != FULL_COHORT_SCOPE:
        failure_reasons.append("run_scope_not_full_cohort")
    if pre_block0_artifact_state != CONTRACT_PASSED_STATE:
        failure_reasons.append("pre_block0_data_suitability_not_contract_passed")
    if n_eligible_patients <= 0:
        failure_reasons.append("no_eligible_patients")
    if paired_support < required_support:
        failure_reasons.append("paired_support_below_threshold")

    if (
        run_scope != FULL_COHORT_SCOPE
        or pre_block0_artifact_state != CONTRACT_PASSED_STATE
        or n_eligible_patients <= 0
        or paired_support < required_support
    ):
        status = "deferred"
        artifact_state = SCAFFOLD_ACTIVE_STATE
        block0_passed = False
    else:
        if not gate_checks["median_delta_total_continuity_mass_positive"]["passed"]:
            failure_reasons.append("median_delta_total_continuity_mass_not_positive")
        if not gate_checks["fraction_real_total_continuity_mass_gt_null_above_half"]["passed"]:
            failure_reasons.append("fraction_real_total_continuity_mass_gt_null_not_above_half")
        if not gate_checks["median_delta_total_emergence_mass_negative"]["passed"]:
            failure_reasons.append("median_delta_total_emergence_mass_not_negative")
        if not gate_checks["fraction_real_total_emergence_mass_lt_null_above_half"]["passed"]:
            failure_reasons.append("fraction_real_total_emergence_mass_lt_null_not_above_half")
        if failure_reasons:
            status = "failed"
            artifact_state = SCAFFOLD_ACTIVE_STATE
            block0_passed = False
        else:
            status = "passed"
            artifact_state = CONTRACT_PASSED_STATE
            block0_passed = True

    metrics_summary = {
        "eligible_patients": int(n_eligible_patients),
        "required_support": int(required_support),
        "gate_summary_quantities": [
            "delta_total_continuity_mass",
            "delta_total_emergence_mass",
        ],
        "real_family": real_family_summary,
        "null_family": null_family_summary,
        "paired_comparisons": paired_summary,
    }
    return Block0GateResult(
        pair_metrics_df=pair_metrics_df,
        n_eligible_patients=int(n_eligible_patients),
        required_support=int(required_support),
        status=status,
        artifact_state=artifact_state,
        block0_passed=block0_passed,
        metrics_summary=metrics_summary,
        gate_checks=gate_checks,
        failure_reasons=tuple(dict.fromkeys(failure_reasons)),
    )


__all__ = [
    "BLOCK_NAME",
    "BLOCK0_SOURCE_DOMAIN",
    "BLOCK0_TARGET_DOMAIN",
    "DEMO_SUBSET_SCOPE",
    "FULL_COHORT_SCOPE",
    "NULL_FAMILIES",
    "NULL_PAIR_FAMILY",
    "PAIR_METRICS_COLUMNS",
    "PATIENT_SUBSET_SCOPE",
    "REAL_FAMILIES",
    "REAL_PAIR_FAMILY",
    "Block0GateResult",
    "Block0NullAssignment",
    "Block0RuntimeConfig",
    "aggregate_block0_gate",
    "build_block0_null_assignments",
    "build_block0_null_family_observations",
    "build_block0_pair_metrics_frame",
    "build_block0_real_family_observations",
    "empty_block0_pair_metrics_frame",
    "evaluate_block0_family_fits",
    "load_block0_runtime_config",
    "resolve_block0_real_family_spec",
    "summarize_block0_bridge_result_totals",
]
