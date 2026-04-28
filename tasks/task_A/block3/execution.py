"""Contract-strict internal execution for Task A Block 3 Phase 3.

Role:
    Execute the internal, non-authority Phase 3 Block 3 benchmark stack from
    evidence-ready Block 1/2 inputs and produce proof-carrying raw/review row
    families for one subexperiment at a time.

Authority anchors:
    - docs/task_A_spec.md §4.5.2-§4.5.6, §5.1 Phase 3
    - docs/task_A_block3_redesign_v1_1.md §4.1-§4.4, §5.5, §5.6

Local boundary:
    - This module owns semisynthetic rerun generation, method execution, and
      metric-row construction for internal Phase 3 use.
    - It does not reopen the public Block 3 workflow, review CLI, or packet
      bridge.
    - It does not promote internal outputs into scientific authority.

Primary contents:
    - Upstream Block 1/2 input resolution.
    - The frozen 24 train / 8 test / 10 rerun generator flow.
    - Real STRIDE, baseline, and ablation runners over paired endpoints only.
    - Raw/review row builders for `3A`, `3B`, `3C-1`, and `3C-2`.

Core logic flow:
    1. Resolve evidence-ready Block 2 and Block 1 prerequisites.
    2. Load the carried patient cohort plus Block 1 identity vectors and the
       derived cost matrix.
    3. Generate rerun-specific semisynthetic truth objects under the frozen
       training-only calibration rules.
    4. Execute the section-specific methods without exposing hidden truth to
       ranked methods.
    5. Materialize raw/review row families that preserve truth, native method
       outputs, and metric status.

Why this module exists:
    Phase 3 needs a single implementation surface where scientific routing,
    generator semantics, and scorer semantics can be audited together. Keeping
    that logic in one place makes it possible to prove the implementation is a
    real internal benchmark path rather than a score-only scaffold.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from stride.api.fit import STRIDEFitConfig, fit_stride
from stride.errors import ContractError
from stride.geometry.state_geometry import build_state_geometry
from stride.observation import FovObservation

from ..config import load_task_a_config_bundle
from ..contracts import EVIDENCE_READY_STATE, SCAFFOLD_ACTIVE_STATE
from ..workflows.stride_adapter import (
    build_task_a_family_observations,
    load_task_a_dataset_handle,
    resolve_task_a_state_basis,
)
from .contracts import (
    Block3ConditionSummaryRow,
    Block3GeneratorObjectScoreRow,
    Block3GeneratorReviewRow,
    Block3GeneratorStabilityRow,
    Block3MetricName,
    Block3PatientMetricRow,
    Block3SectionReviewRow,
    Block3SubexperimentId,
    Block3SubexperimentRawRows,
    Block3SubexperimentReviewRows,
    MetricRole,
    MetricStatus,
    MetricValue,
    ValidationObjectId,
    make_metric_value,
)
from .analysis import derive_A_d_e_from_plan
from .baselines import (
    PlanBaselineResult,
    diagonal_transport_plan,
    estimate_uot_matched_mass,
    partial_ot_plan,
    solve_uot_plan,
)
from .calibration import UOTCalibrationResult, calibrate_uot_lambda
from .registry import (
    Block3Registry,
    get_condition_spec,
    get_live_block3_registry,
    get_metric_spec,
    get_method_spec,
    get_subexperiment_spec,
)


# Frozen boundary flag mirrored into all internal Block 3 manifests.
BLOCK3_PACKET_BRIDGE_POLICY = "deferred_non_authority_pending_clean_bridge_spec"
# The only executable units on the internal Phase 3 surface. `3C` stays a
# non-executable section container and is validated through `3C-1`/`3C-2`.
INTERNAL_EXECUTABLE_SUBEXPERIMENTS: tuple[str, ...] = ("3A", "3B-1", "3B-2", "3C-1", "3C-2")
_GENERATOR_VALIDATION_CONDITION_ID = "generator_validation"
_BLOCK3_PAIR_FAMILY = "TC-IM"
# Frozen outer-design constants from the Block 3 scientific contract.
_N_GENERATOR_RERUNS = 10
_N_TRAIN_PATIENTS = 24
_N_TEST_PATIENTS = 8
_EPSILON_FIXED = 0.01
_OPEN_SUPPORT_COVERAGE = 0.95
_TOL = 1e-12
# Public relation scenarios exist only inside `3B`; `3C-*` reuses opaque
# rerun-specific realizations instead of publishing a relation axis.
_RELATION_SPECS: dict[str, tuple[float, int]] = {
    "relation_null": (0.00, 0),
    "relation_weak": (0.05, 1),
    "relation_mid": (0.15, 1),
    "relation_strong": (0.30, 2),
}
_HIDDEN_RELATION_OPTIONS: tuple[str, ...] = (
    "relation_weak",
    "relation_mid",
    "relation_strong",
)
_SUPPORT_MODE_LEGACY_NEAREST_C = "legacy_nearest_c"
_SUPPORT_MODE_RELATION_MOTIF_PROBE = "relation_motif_probe_v1"
_ALLOWED_SUPPORT_MODES: tuple[str, ...] = (
    _SUPPORT_MODE_LEGACY_NEAREST_C,
    _SUPPORT_MODE_RELATION_MOTIF_PROBE,
)
_PUBLIC_OPEN_MASS_SCALE_GRID: tuple[float, ...] = tuple(round(index / 10, 1) for index in range(11))
_BLOCK3B_3B1_OPEN_MASS_SENSITIVITY_GRID: tuple[float, ...] = (0.1, 0.25, 0.5)
_ALLOWED_OPEN_MASS_SCALES: tuple[float, ...] = tuple(
    sorted({*_PUBLIC_OPEN_MASS_SCALE_GRID, *_BLOCK3B_3B1_OPEN_MASS_SENSITIVITY_GRID})
)


@dataclass(frozen=True)
class Block3ResolvedInputs:
    """Resolved upstream prerequisites for one internal Block 3 execution pass.

    Fields:
        block2_manifest_path: Evidence-ready Block 2 manifest that authorizes
            the internal Phase 3 run.
        block1_bundle_path: Evidence-ready Block 1 bundle referenced from the
            Block 2 manifest.
        output_dir: Internal output root where raw/review artifacts are written.
        block2_implementation_tier: Expected to remain `canonical_full`.
        block2_evidence_lineage: Expected to remain `canonical_rerun`.
    """

    block2_manifest_path: Path
    block1_bundle_path: Path
    output_dir: Path
    block2_implementation_tier: str
    block2_evidence_lineage: str


@dataclass(frozen=True)
class Block3ExecutionPlan:
    """Execution-plan metadata for one internal Phase 3 launch.

    The plan is the narrow routing contract shared by execution, raw writing,
    and review writing. It keeps the registry-derived routes together with the
    non-authority manifest flags that must remain fixed during Phase 3.
    """

    inputs: Block3ResolvedInputs
    registry: Block3Registry
    subexperiment_order: tuple[str, ...]
    method_routes: dict[str, tuple[str, ...]]
    condition_routes: dict[str, tuple[str, ...]]
    evaluation_families: dict[str, str]
    artifact_state: str
    scientific_interpretation_allowed: bool
    packet_bridge_enabled: bool
    packet_bridge_policy: str
    workflow_entrypoints: tuple[str, ...]


@dataclass(frozen=True)
class Block3InternalExecutionResult:
    """Filesystem outputs produced for one executed Block 3 subexperiment."""

    subexperiment_id: str
    raw_manifest_path: Path
    raw_index_path: Path
    review_manifest_path: Path
    review_index_path: Path
    raw_artifact_paths: dict[str, Path]
    review_artifact_paths: dict[str, Path]


@dataclass(frozen=True)
class Block3DiagnosticArmResult:
    """Filesystem outputs produced for one internal 3B diagnostic arm."""

    arm_id: str
    support_mode: str
    open_mass_scale: float
    output_dir: Path
    execution_result: Block3InternalExecutionResult


@dataclass(frozen=True)
class Block3DiagnosticMatrixResult:
    """Top-level outputs produced for the internal 3B diagnostic matrix."""

    output_dir: Path
    arm_results: tuple[Block3DiagnosticArmResult, ...]
    summary_artifact_paths: dict[str, Path]


@dataclass(frozen=True)
class Block3CohortInputs:
    """Resolved cohort-level inputs carried into the Block 3 generator.

    Fields:
        identity_vectors: `g_k`-style community identity vectors derived from
            Block 1 correspondence tables.
        cost_matrix: Normalized `C` matrix used by relation scenarios and the
            balanced OT baseline.
        patient_source_profiles: Held-out source-side `x_p` profiles on the
            shared K-state axis.
        patient_target_profiles: Real target-side profiles used only for
            training-side open calibration and `3A` validation objects.
    """

    stage0_h5ad: Path
    config_path: Path
    output_dir: Path
    master_seed: int
    state_ids: tuple[int, ...]
    state_basis: Any
    geometry: Any
    identity_vectors: np.ndarray
    cost_matrix: np.ndarray
    patient_source_profiles: dict[str, np.ndarray]
    patient_target_profiles: dict[str, np.ndarray]


@dataclass(frozen=True)
class Block3PatientTruth:
    """Hidden per-patient truth object for one rerun-specific realization.

    Fields:
        x: Source-side endpoint fractions shown to all ranked methods.
        y: Synthetic target endpoint fractions shown to all ranked methods.
        A: Hidden relation operator scored in `3B` and `3C-2`.
        d/e: Hidden depletion and emergence profiles scored in `3C-*`.
        open_mass: Total emergence burden used for traceability and audit.
    """

    rerun_id: str
    patient_id: str
    x: np.ndarray
    y: np.ndarray
    A: np.ndarray
    d: np.ndarray
    e: np.ndarray
    open_mass: float


@dataclass(frozen=True)
class Block3MethodOutput:
    """Recovered native method outputs for one patient and one method route.

    Inputs are arrays emitted by STRIDE or Block3b baseline runners. Outputs are
    serialized into `method_native_output_store`; non-`ok` `fit_status` means
    metric builders propagate non-estimable status instead of substituting
    fallback arrays. `P` and metadata are optional method-native diagnostics.
    """

    patient_id: str
    fit_status: str
    A: np.ndarray | None
    d: np.ndarray | None
    e: np.ndarray | None
    mu_minus: np.ndarray | None
    mu_plus: np.ndarray | None
    P: np.ndarray | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class Block3GeneratorRerun:
    """One complete rerun-specific Block 3 semisynthetic realization set.

    `generator_truths` holds the opaque relation condition used by `3A` and
    `3C-*`, while `baseline_truths` expands the public `relation_*` scenarios
    required only for `3B`.
    """

    rerun_id: str
    split_seed: int
    train_patient_ids: tuple[str, ...]
    test_patient_ids: tuple[str, ...]
    hidden_relation_condition_id: str
    support_mode: str
    open_mass_scale: float
    relation_motif: np.ndarray | None
    generator_truths: dict[str, Block3PatientTruth]
    baseline_truths: dict[str, dict[str, Block3PatientTruth]]


def _reruns_with_open_mass_scale(
    reruns: tuple[Block3GeneratorRerun, ...],
    *,
    open_mass_scale: float,
) -> tuple[Block3GeneratorRerun, ...]:
    """Return rerun views at the requested open-mass scale.

    Inputs are generator reruns from the same support-mode route. When the
    requested scale already matches the current reruns, the original tuple is
    returned. Otherwise, `generator_truths` and `baseline_truths` are rebuilt by
    scaling each truth's `d/e` burdens and recomputing endpoint `y`; relation
    operators, splits, and support-mode metadata are preserved.
    """

    target_scale = _validate_open_mass_scale(open_mass_scale)
    scaled_reruns: list[Block3GeneratorRerun] = []
    for rerun in reruns:
        if rerun.open_mass_scale == target_scale:
            scaled_reruns.append(rerun)
            continue
        if rerun.open_mass_scale <= _TOL:
            raise ContractError("Cannot rescale Block 3 reruns generated at zero open_mass_scale")
        scale_factor = target_scale / rerun.open_mass_scale

        def _scale_truth(truth: Block3PatientTruth) -> Block3PatientTruth:
            d = np.asarray(truth.d, dtype=float) * scale_factor
            e = np.asarray(truth.e, dtype=float) * scale_factor
            y = np.sum(truth.x[:, None] * truth.A, axis=0, dtype=float) + e
            return Block3PatientTruth(
                rerun_id=truth.rerun_id,
                patient_id=truth.patient_id,
                x=truth.x,
                y=_normalize_probabilities(y),
                A=truth.A,
                d=d,
                e=e,
                open_mass=float(np.sum(e, dtype=float)),
            )

        scaled_reruns.append(
            Block3GeneratorRerun(
                rerun_id=rerun.rerun_id,
                split_seed=rerun.split_seed,
                train_patient_ids=rerun.train_patient_ids,
                test_patient_ids=rerun.test_patient_ids,
                hidden_relation_condition_id=rerun.hidden_relation_condition_id,
                support_mode=rerun.support_mode,
                open_mass_scale=target_scale,
                relation_motif=rerun.relation_motif,
                generator_truths={patient_id: _scale_truth(truth) for patient_id, truth in rerun.generator_truths.items()},
                baseline_truths={
                    condition_id: {patient_id: _scale_truth(truth) for patient_id, truth in truths.items()}
                    for condition_id, truths in rerun.baseline_truths.items()
                },
            )
        )
    return tuple(scaled_reruns)


def _resolve_path(path: str | Path) -> Path:
    """Resolve a filesystem path into an absolute expanded `Path`."""

    return Path(path).expanduser().resolve()


def _load_json_dict(path: str | Path, *, label: str) -> dict[str, Any]:
    """Load one JSON object sidecar and require dictionary shape."""

    resolved = _resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} was not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must be a JSON object: {resolved}")
    return payload


def resolve_upstream_inputs(
    *,
    block2_manifest_path: str | Path,
    output_dir: str | Path,
) -> Block3ResolvedInputs:
    """Resolve and validate the upstream Block 1/2 prerequisites.

    Purpose:
        Confirm that Block 3 is reading from the evidence-ready, canonical-full
        Block 2 and Block 1 surfaces rather than from reduced or proxy inputs.

    Inputs:
        block2_manifest_path: Path to the evidence-ready Block 2 manifest.
        output_dir: Internal output directory for the current subexperiment run.

    Returns:
        A `Block3ResolvedInputs` object containing the validated upstream paths
        and the Block 2 evidence metadata required by the internal run.

    Raises:
        ContractError: The upstream manifests are missing required readiness
            flags or evidence lineage.

    Core flow:
        1. Load the Block 2 manifest and require `evidence_ready`.
        2. Require `implementation_tier=canonical_full` and
           `evidence_lineage=canonical_rerun`.
        3. Resolve the referenced Block 1 bundle and require it to be
           evidence-ready as well.
        4. Return the resolved path bundle used by downstream execution.

    Doc support:
        This function enforces the Phase 3 rule that Block 3 consumes
        evidence-ready Block 1/2 inputs and does not reopen reduced-data
        surrogates as live scientific prerequisites.
    """
    block2_manifest = _resolve_path(block2_manifest_path)
    block2_payload = _load_json_dict(block2_manifest, label="Task A Block 2 manifest")

    artifact_state = str(block2_payload.get("artifact_state", ""))
    if artifact_state != EVIDENCE_READY_STATE:
        raise ContractError("Block 3 internal execution requires an evidence_ready Block 2 manifest")

    implementation_tier = str(block2_payload.get("implementation_tier", ""))
    if implementation_tier != "canonical_full":
        raise ContractError("Block 3 internal execution requires block2_implementation_tier=canonical_full")

    evidence_lineage = str(block2_payload.get("evidence_lineage", ""))
    if evidence_lineage != "canonical_rerun":
        raise ContractError("Block 3 internal execution requires block2_evidence_lineage=canonical_rerun")

    block1_bundle_raw = block2_payload.get("block1_bundle_path")
    if block1_bundle_raw in (None, ""):
        raise ContractError("Task A Block 2 manifest must declare block1_bundle_path")
    block1_bundle = _resolve_path(block1_bundle_raw)
    block1_payload = _load_json_dict(block1_bundle, label="Task A Block 1 bundle")
    if str(block1_payload.get("artifact_state", "")) != EVIDENCE_READY_STATE:
        raise ContractError("Block 3 internal execution requires an evidence_ready Block 1 bundle")

    return Block3ResolvedInputs(
        block2_manifest_path=block2_manifest,
        block1_bundle_path=block1_bundle,
        output_dir=_resolve_path(output_dir),
        block2_implementation_tier=implementation_tier,
        block2_evidence_lineage=evidence_lineage,
    )


def build_phase2_execution_plan(
    inputs: Block3ResolvedInputs,
    registry: Block3Registry | None = None,
) -> Block3ExecutionPlan:
    """Build the internal non-authority execution plan for Block 3.

    Purpose:
        Materialize the registry-derived routing layer together with the fixed
        Phase 3 boundary flags that must stay scaffold-only and non-authority.

    Inputs:
        inputs: Validated upstream Block 1/2 prerequisites.
        registry: Optional registry override for tests; defaults to the live
            Block 3 routing registry.

    Returns:
        A `Block3ExecutionPlan` that can be shared by execution, raw writing,
        and review writing.

    Core flow:
        1. Resolve the live registry.
        2. Copy the per-subexperiment method, condition, and family routes.
        3. Attach the fixed non-authority manifest flags.
        4. Validate the resulting plan before returning it.
    """
    resolved_registry = get_live_block3_registry() if registry is None else registry
    plan = Block3ExecutionPlan(
        inputs=inputs,
        registry=resolved_registry,
        subexperiment_order=tuple(resolved_registry.subexperiments.keys()),
        method_routes={
            subexperiment_id: tuple(method.value for method in spec.methods)
            for subexperiment_id, spec in resolved_registry.subexperiments.items()
        },
        condition_routes={
            subexperiment_id: spec.condition_ids
            for subexperiment_id, spec in resolved_registry.subexperiments.items()
        },
        evaluation_families={
            subexperiment_id: spec.evaluation_family
            for subexperiment_id, spec in resolved_registry.subexperiments.items()
        },
        artifact_state=SCAFFOLD_ACTIVE_STATE,
        scientific_interpretation_allowed=False,
        packet_bridge_enabled=False,
        packet_bridge_policy=BLOCK3_PACKET_BRIDGE_POLICY,
        workflow_entrypoints=(),
    )
    validate_phase2_execution_plan(plan)
    return plan


def validate_phase2_execution_plan(plan: Block3ExecutionPlan) -> None:
    """Require the internal execution plan to preserve Phase 3 boundaries."""

    if plan.artifact_state != SCAFFOLD_ACTIVE_STATE:
        raise ContractError("Block 3 Phase 3 internal execution must remain scaffold_active")
    if plan.scientific_interpretation_allowed:
        raise ContractError("Block 3 Phase 3 internal execution must remain non-interpretive")
    if plan.packet_bridge_enabled:
        raise ContractError("Block 3 Phase 3 internal execution must keep packet bridge disabled")
    if plan.workflow_entrypoints:
        raise ContractError("Block 3 Phase 3 internal execution must not expose public workflow entrypoints")
    if Block3SubexperimentId.ABLATION_STUDY.value in plan.subexperiment_order:
        raise ContractError("Block 3 execution plan must not schedule 3C as an executable unit")


def _normalize_probabilities(values: np.ndarray) -> np.ndarray:
    """Normalize one non-negative vector to unit mass for Block 3 scoring."""

    vector = np.asarray(values, dtype=float).reshape(-1)
    total = float(np.sum(vector, dtype=float))
    if total <= 0.0:
        raise ContractError("Probability vectors must have positive total mass")
    return vector / total


def _pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Compute the normalized Pearson correlation for one validation surface."""

    left_vector = _normalize_probabilities(left)
    right_vector = _normalize_probabilities(right)
    if np.allclose(left_vector, left_vector.mean()) or np.allclose(right_vector, right_vector.mean()):
        return 1.0 if np.allclose(left_vector, right_vector) else 0.0
    return float(np.clip(np.corrcoef(left_vector, right_vector)[0, 1], -1.0, 1.0))


def _mae(left: np.ndarray, right: np.ndarray) -> float:
    """Compute mean absolute error after unit-mass normalization."""

    return float(np.mean(np.abs(_normalize_probabilities(left) - _normalize_probabilities(right))))


def _mse(left: np.ndarray, right: np.ndarray) -> float:
    """Compute mean squared error after unit-mass normalization."""

    delta = _normalize_probabilities(left) - _normalize_probabilities(right)
    return float(np.mean(delta * delta))


def _kl_divergence(left: np.ndarray, right: np.ndarray) -> float:
    """Compute KL divergence over normalized positive-support vectors."""

    left_vector = _normalize_probabilities(left)
    right_vector = _normalize_probabilities(right)
    mask = left_vector > 0.0
    return float(np.sum(left_vector[mask] * np.log(left_vector[mask] / right_vector[mask]), dtype=float))


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence for one pair of probability vectors."""

    left_vector = _normalize_probabilities(left)
    right_vector = _normalize_probabilities(right)
    midpoint = 0.5 * (left_vector + right_vector)
    return 0.5 * _kl_divergence(left_vector, midpoint) + 0.5 * _kl_divergence(right_vector, midpoint)


def _metric_bundle(left: np.ndarray, right: np.ndarray) -> dict[Block3MetricName, float]:
    """Collect the frozen `3A` object-comparison metrics for one surface pair."""

    return {
        Block3MetricName.PEARSON_CORRELATION: _pearson_correlation(left, right),
        Block3MetricName.MAE: _mae(left, right),
        Block3MetricName.MSE: _mse(left, right),
        Block3MetricName.JS_DIVERGENCE: _js_divergence(left, right),
    }


def _validate_metric_value(
    *,
    subexperiment_id: str,
    metric_value: MetricValue,
    expected_role: MetricRole,
) -> None:
    """Check that a non-method-bearing metric payload matches registry rules."""

    metric_spec = get_metric_spec(metric_value.metric_name.value)
    if subexperiment_id not in metric_spec.allowed_subexperiments:
        raise ContractError(
            f"Block 3 metric {metric_value.metric_name.value!r} is not allowed in {subexperiment_id!r}"
        )
    if metric_value.status not in metric_spec.allowed_statuses:
        raise ContractError(
            f"Block 3 metric {metric_value.metric_name.value!r} does not allow status {metric_value.status.value!r}"
        )
    if metric_spec.role is not expected_role:
        raise ContractError(
            f"Block 3 metric {metric_value.metric_name.value!r} requires role {metric_spec.role.value!r}"
        )


def _validate_method_metric_value(
    *,
    subexperiment_id: str,
    method_name: str,
    metric_value: MetricValue,
) -> tuple[MetricRole, str]:
    """Check a method-bearing metric payload against method and metric routes."""

    method_spec = get_method_spec(method_name)
    if subexperiment_id not in method_spec.allowed_subexperiments:
        raise ContractError(f"Block 3 method {method_name!r} is not allowed in {subexperiment_id!r}")
    metric_spec = get_metric_spec(metric_value.metric_name.value)
    if subexperiment_id not in metric_spec.allowed_subexperiments:
        raise ContractError(
            f"Block 3 metric {metric_value.metric_name.value!r} is not allowed in {subexperiment_id!r}"
        )
    if metric_value.status not in metric_spec.allowed_statuses:
        raise ContractError(
            f"Block 3 metric {metric_value.metric_name.value!r} does not allow status {metric_value.status.value!r}"
        )
    return metric_spec.role, method_spec.method_class.value


def _make_patient_metric_row(
    *,
    rerun_id: str,
    subexperiment_id: str,
    condition_id: str,
    evaluation_family: str,
    method_name: str,
    metric_name: Block3MetricName,
    value: float | None,
    status: str | MetricStatus,
    patient_id: str,
    open_mass_scale: float | None = None,
) -> Block3PatientMetricRow:
    """Construct one validated patient-level metric row for `3B` or `3C-*`."""

    metric_value = make_metric_value(metric_name=metric_name, value=value, status=status)
    metric_role, _method_class = _validate_method_metric_value(
        subexperiment_id=subexperiment_id,
        method_name=method_name,
        metric_value=metric_value,
    )
    method_spec = get_method_spec(method_name)
    return Block3PatientMetricRow(
        rerun_id=rerun_id,
        subexperiment_id=subexperiment_id,
        condition_id=condition_id,
        evaluation_family=evaluation_family,
        method_name=method_spec.name,
        method_class=method_spec.method_class,
        metric_role=metric_role,
        metric_value=metric_value,
        patient_id=patient_id,
        open_mass_scale=open_mass_scale,
    )


def _bootstrap_ci(values: list[float]) -> tuple[float, float]:
    """Estimate a simple rerun-level bootstrap interval for summary rows."""

    samples = np.asarray(values, dtype=float)
    if samples.size == 1:
        return float(samples[0]), float(samples[0])
    rng = np.random.default_rng(17)
    draws = []
    for _ in range(256):
        draws.append(float(np.mean(rng.choice(samples, size=samples.size, replace=True))))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _summarize_patient_rows(
    *,
    subexperiment_id: str,
    patient_rows: tuple[Block3PatientMetricRow, ...],
) -> tuple[Block3ConditionSummaryRow, ...]:
    """Aggregate patient-level metrics into rerun-aware condition summaries.

    Purpose:
        Collapse patient rows to condition-level summary rows while preserving
        the documented status contract and paired-difference semantics.

    Inputs:
        subexperiment_id: Method-bearing Block 3 section being summarized.
        patient_rows: Patient-level metric rows already validated against the
            live registry.

    Returns:
        Ordered condition-summary rows keyed by condition, metric, and method.

    Raises:
        ContractError: Metric statuses would be silently mixed inside one
            summary cell.

    Core flow:
        1. Group patient rows by condition, method, and metric.
        2. Propagate `not_estimable` / `not_applicable` according to the
           summary status contract.
        3. For reported rows, aggregate at rerun level first and then compute
           mean, bootstrap CI, and paired difference versus `stride_reference`.
        4. Emit sorted summary rows for raw and review surfaces.
    """
    if not patient_rows:
        return ()
    summary_rows: list[Block3ConditionSummaryRow] = []
    grouped: dict[tuple[str, float | None, str, str], list[Block3PatientMetricRow]] = {}
    for row in patient_rows:
        grouped.setdefault(
            (
                row.condition_id,
                row.open_mass_scale,
                row.method_name.value,
                row.metric_value.metric_name.value,
            ),
            [],
        ).append(row)

    for (condition_id, open_mass_scale, method_name, metric_name), rows in grouped.items():
        method_spec = get_method_spec(method_name)
        metric_spec = get_metric_spec(metric_name)
        condition = get_condition_spec(condition_id)
        statuses = {row.metric_value.status for row in rows}
        if MetricStatus.NOT_ESTIMABLE in statuses:
            metric_value = make_metric_value(
                metric_name=metric_spec.name,
                value=None,
                status=MetricStatus.NOT_ESTIMABLE,
            )
            mean_value = None
            ci_lower = None
            ci_upper = None
            paired_difference = None
        elif statuses == {MetricStatus.NOT_APPLICABLE}:
            metric_value = make_metric_value(
                metric_name=metric_spec.name,
                value=None,
                status=MetricStatus.NOT_APPLICABLE,
            )
            mean_value = None
            ci_lower = None
            ci_upper = None
            paired_difference = None
        elif statuses != {MetricStatus.REPORTED}:
            raise ContractError(
                f"Block 3 summary rows must not silently mix metric statuses for {(condition_id, method_name, metric_name)!r}"
            )
        else:
            rerun_means: dict[str, float] = {}
            for rerun_id in sorted({row.rerun_id for row in rows}):
                rerun_values = [float(row.metric_value.value) for row in rows if row.rerun_id == rerun_id]
                rerun_means[rerun_id] = float(np.mean(rerun_values))
            rerun_mean_values = list(rerun_means.values())
            mean_value = float(np.mean(rerun_mean_values))
            ci_lower, ci_upper = _bootstrap_ci(rerun_mean_values)
            metric_value = make_metric_value(
                metric_name=metric_spec.name,
                value=mean_value,
                status=MetricStatus.REPORTED,
            )
            if method_name == "stride_reference":
                paired_difference = 0.0
            else:
                reference_rows = grouped[(condition_id, open_mass_scale, "stride_reference", metric_name)]
                reference_statuses = {row.metric_value.status for row in reference_rows}
                if reference_statuses != {MetricStatus.REPORTED}:
                    metric_value = make_metric_value(
                        metric_name=metric_spec.name,
                        value=None,
                        status=MetricStatus.NOT_ESTIMABLE,
                    )
                    mean_value = None
                    ci_lower = None
                    ci_upper = None
                    paired_difference = None
                else:
                    reference_rerun_means: dict[str, float] = {}
                    for rerun_id in sorted({row.rerun_id for row in reference_rows}):
                        rerun_values = [
                            float(row.metric_value.value)
                            for row in reference_rows
                            if row.rerun_id == rerun_id
                        ]
                        reference_rerun_means[rerun_id] = float(np.mean(rerun_values))
                    paired_difference = float(
                        np.mean(
                            [
                                rerun_means[rerun_id] - reference_rerun_means[rerun_id]
                                for rerun_id in sorted(rerun_means)
                            ]
                        )
                    )
        summary_rows.append(
            Block3ConditionSummaryRow(
                rerun_id="all_reruns",
                subexperiment_id=subexperiment_id,
                condition_id=condition_id,
                evaluation_family=condition.evaluation_family,
                method_name=method_spec.name,
                method_class=method_spec.method_class,
                metric_role=metric_spec.role,
                metric_value=metric_value,
                summary_level="rerun_mean_bootstrap95",
                mean_value=mean_value,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                paired_difference_vs_stride_reference=paired_difference,
                open_mass_scale=open_mass_scale,
            )
        )
    summary_rows.sort(
        key=lambda row: (
            row.condition_id,
            -1.0 if row.open_mass_scale is None else row.open_mass_scale,
            row.metric_value.metric_name.value,
            row.method_name.value,
        )
    )
    return tuple(summary_rows)


def _build_section_review_rows(
    *,
    subexperiment_id: str,
    summary_rows: tuple[Block3ConditionSummaryRow, ...],
) -> tuple[Block3SectionReviewRow, ...]:
    """Translate condition summaries into review-facing section rows."""

    subexperiment = get_subexperiment_spec(subexperiment_id)
    review_rows: list[Block3SectionReviewRow] = []
    for row in summary_rows:
        condition = get_condition_spec(row.condition_id)
        review_rows.append(
            Block3SectionReviewRow(
                rerun_id=row.rerun_id,
                subexperiment_id=row.subexperiment_id,
                condition_id=row.condition_id,
                evaluation_family=row.evaluation_family,
                method_name=row.method_name,
                method_class=row.method_class,
                metric_role=row.metric_role,
                metric_value=row.metric_value,
                summary_level=row.summary_level,
                mean_value=row.mean_value,
                ci_lower=row.ci_lower,
                ci_upper=row.ci_upper,
                paired_difference_vs_stride_reference=row.paired_difference_vs_stride_reference,
                section_title=subexperiment.title,
                condition_title=condition.title,
                review_surface_role="section_condition_summary",
                open_mass_scale=row.open_mass_scale,
            )
        )
    return tuple(review_rows)


def _json_array(value: np.ndarray | None) -> str:
    """Serialize one optional array payload to a JSON list string."""

    if value is None:
        return "[]"
    return json.dumps(np.asarray(value, dtype=float).tolist(), sort_keys=False)


def _mean_profile(vectors: list[np.ndarray]) -> np.ndarray:
    """Average aligned profiles and renormalize them to unit mass."""

    return _normalize_probabilities(np.mean(np.vstack(vectors), axis=0, dtype=float))


def _median_total_variation_deviation(profiles: list[np.ndarray]) -> float:
    """Estimate the frozen robust median TV deviation around one channel centroid."""

    if not profiles:
        return 0.0
    matrix = np.vstack([_normalize_probabilities(profile) for profile in profiles])
    centroid = _normalize_probabilities(np.mean(matrix, axis=0, dtype=float))
    deviations = 0.5 * np.sum(np.abs(matrix - centroid[None, :]), axis=1, dtype=float)
    return float(np.median(deviations))


def _estimate_kappa(profiles: list[np.ndarray]) -> float:
    """Estimate the frozen Gamma-simplex concentration from robust TV dispersion."""

    if not profiles:
        return 20.0
    tv_deviation = _median_total_variation_deviation(profiles)
    return float(np.clip(1.0 / max(tv_deviation, 1e-8), 1.0, 200.0))


def _sample_gamma_simplex(
    *,
    base: np.ndarray,
    kappa: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one simplex draw from the frozen Gamma-normalize family."""

    resolved_base = _normalize_probabilities(base)
    shape = np.maximum(resolved_base * float(max(kappa, 1.0)), 1e-3)
    draw = rng.gamma(shape=shape, scale=1.0)
    if float(np.sum(draw, dtype=float)) <= 0.0:
        return resolved_base
    return _normalize_probabilities(draw)


def _sample_capped_depletion(
    *,
    x: np.ndarray,
    mass: float,
    base: np.ndarray,
    kappa: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample depletion burden while respecting the source-side capacity cap."""

    if mass <= 0.0:
        return np.zeros_like(x, dtype=float)
    distribution = _sample_gamma_simplex(base=base, kappa=kappa, rng=rng)
    capacity = np.asarray(x, dtype=float).copy()
    depletion = np.zeros_like(capacity, dtype=float)
    remaining = float(mass)
    while remaining > _TOL and np.any(capacity > _TOL):
        active = capacity > _TOL
        active_distribution = distribution * active.astype(float)
        if float(np.sum(active_distribution, dtype=float)) <= 0.0:
            # The preferred depletion scaffold can exhaust before the patient-
            # specific capacity cap is reached. Reallocate the remainder across
            # still-active source mass rather than failing or dropping mass.
            active_distribution = capacity * active.astype(float)
        proposal = remaining * _normalize_probabilities(active_distribution)
        assigned = np.minimum(proposal, capacity)
        depletion += assigned
        capacity -= assigned
        updated_remaining = float(mass - np.sum(depletion, dtype=float))
        if abs(updated_remaining - remaining) <= 1e-12:
            break
        remaining = updated_remaining
    return depletion


def _nearest_targets(cost_matrix: np.ndarray, source_index: int, max_targets: int) -> tuple[int, ...]:
    """Return the nearest off-diagonal targets under the frozen cost matrix."""

    if max_targets <= 0:
        return ()
    candidates = [
        (float(cost_matrix[source_index, target_index]), int(target_index))
        for target_index in range(cost_matrix.shape[0])
        if target_index != source_index
    ]
    candidates.sort(key=lambda item: (item[0], item[1]))
    return tuple(target_index for _cost, target_index in candidates[:max_targets])


def _validate_support_mode(
    support_mode: str,
) -> Literal["legacy_nearest_c", "relation_motif_probe_v1"]:
    """Require one supported internal Block 3 support-mode identifier."""

    if support_mode not in _ALLOWED_SUPPORT_MODES:
        raise ContractError(f"Unsupported Block 3 support_mode: {support_mode!r}")
    return support_mode  # type: ignore[return-value]


def _validate_open_mass_scale(open_mass_scale: float) -> float:
    """Require one supported internal Block 3 open-mass scale."""

    normalized = round(float(open_mass_scale), 10)
    for allowed_scale in _ALLOWED_OPEN_MASS_SCALES:
        if abs(normalized - float(allowed_scale)) <= _TOL:
            return float(allowed_scale)
    raise ContractError(
        "Unsupported Block 3 open_mass_scale: "
        f"{open_mass_scale!r}; expected one of {_ALLOWED_OPEN_MASS_SCALES!r}"
    )


def _resolve_legacy_relation_targets(
    *,
    cost_matrix: np.ndarray,
    source_index: int,
    max_targets: int,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Resolve support/weights under the frozen nearest-target legacy rule."""

    targets = _nearest_targets(cost_matrix, source_index, max_targets)
    if not targets:
        return (), np.asarray([], dtype=float)
    if len(targets) == 1:
        return targets, np.asarray([1.0], dtype=float)
    weights = np.asarray(
        [np.exp(-float(cost_matrix[source_index, target_index])) for target_index in targets],
        dtype=float,
    )
    return targets, _normalize_probabilities(weights)


def _resolve_motif_relation_targets(
    *,
    relation_motif: np.ndarray,
    cost_matrix: np.ndarray,
    source_index: int,
    max_targets: int,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Resolve support/weights from one source row of the rerun-shared motif."""

    if relation_motif.ndim != 2 or relation_motif.shape != cost_matrix.shape:
        raise ContractError("Block 3 relation motif must match the cost-matrix shape")
    if max_targets <= 0:
        return (), np.asarray([], dtype=float)
    row = np.asarray(relation_motif[source_index], dtype=float).reshape(-1)
    candidates = [
        (-float(row[target_index]), float(cost_matrix[source_index, target_index]), int(target_index))
        for target_index in range(row.shape[0])
        if target_index != source_index and float(row[target_index]) > _TOL
    ]
    if not candidates:
        return (), np.asarray([], dtype=float)
    candidates.sort()
    targets = tuple(target_index for _score, _cost, target_index in candidates[:max_targets])
    weights = np.asarray([float(row[target_index]) for target_index in targets], dtype=float)
    return targets, _normalize_probabilities(weights)


def _resolve_relation_targets(
    *,
    support_mode: str,
    relation_motif: np.ndarray | None,
    cost_matrix: np.ndarray,
    source_index: int,
    max_targets: int,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Resolve off-diagonal support/weights for one source row.

    The probe variant prefers the rerun-shared motif when a positive motif row
    exists, then falls back deterministically to the frozen nearest-target
    `exp(-C)` rule.
    """

    if (
        support_mode == _SUPPORT_MODE_RELATION_MOTIF_PROBE
        and relation_motif is not None
    ):
        motif_targets, motif_weights = _resolve_motif_relation_targets(
            relation_motif=relation_motif,
            cost_matrix=cost_matrix,
            source_index=source_index,
            max_targets=max_targets,
        )
        if motif_targets:
            return motif_targets, motif_weights
    return _resolve_legacy_relation_targets(
        cost_matrix=cost_matrix,
        source_index=source_index,
        max_targets=max_targets,
    )


def _build_truth_for_condition(
    *,
    rerun_id: str,
    patient_id: str,
    x: np.ndarray,
    delta_minus: np.ndarray,
    delta_plus: np.ndarray,
    condition_id: str,
    cost_matrix: np.ndarray,
    support_mode: str = _SUPPORT_MODE_LEGACY_NEAREST_C,
    relation_motif: np.ndarray | None = None,
) -> Block3PatientTruth:
    """Build one patient truth object for one declared relation condition.

    Purpose:
        Combine the fixed open burdens with a declared relation scenario to
        produce the hidden patient-level truth scored later by Block 3.

    Inputs:
        x: Source-side endpoint fractions for the held-out patient.
        delta_minus / delta_plus: Open depletion and emergence burdens sampled
            earlier from the training-side calibration model.
        condition_id: Public or hidden relation condition controlling the
            off-diagonal burden pattern on the matched mass.
        cost_matrix: Identity-aware cost matrix used to choose off-diagonal
            targets.

    Returns:
        A `Block3PatientTruth` bundle carrying `x`, synthetic `y`, and the
        hidden `A/d/e` objects for the requested condition.

    Core flow:
        1. Convert the source profile to unit mass.
        2. Use the open burden to determine row-wise depletion.
        3. Allocate the matched mass between diagonal and off-diagonal targets
           according to the declared relation scenario.
        4. Add emergence burden and renormalize the synthetic target endpoint.
    """
    support_mode = _validate_support_mode(support_mode)
    offdiag_fraction, max_targets = _RELATION_SPECS[condition_id]
    resolved_x = _normalize_probabilities(x)
    A = np.zeros((resolved_x.shape[0], resolved_x.shape[0]), dtype=float)
    d = np.zeros_like(resolved_x, dtype=float)
    for source_index, source_mass in enumerate(resolved_x):
        if source_mass <= _TOL:
            A[source_index, source_index] = 1.0
            continue
        d[source_index] = float(np.clip(delta_minus[source_index] / source_mass, 0.0, 1.0))
        matched_fraction = max(0.0, 1.0 - d[source_index])
        if matched_fraction <= _TOL or max_targets == 0 or offdiag_fraction <= 0.0:
            A[source_index, source_index] = matched_fraction
            continue
        targets, weights = _resolve_relation_targets(
            support_mode=support_mode,
            relation_motif=relation_motif,
            cost_matrix=cost_matrix,
            source_index=source_index,
            max_targets=max_targets,
        )
        if not targets:
            A[source_index, source_index] = matched_fraction
            continue
        offdiag_total = matched_fraction * float(offdiag_fraction)
        diagonal_total = matched_fraction - offdiag_total
        A[source_index, source_index] = diagonal_total
        if len(targets) == 1:
            A[source_index, targets[0]] = offdiag_total
            continue
        for target_index, weight in zip(targets, weights, strict=True):
            A[source_index, target_index] = offdiag_total * float(weight)
    e = np.asarray(delta_plus, dtype=float)
    y = np.sum(resolved_x[:, None] * A, axis=0, dtype=float) + e
    return Block3PatientTruth(
        rerun_id=rerun_id,
        patient_id=patient_id,
        x=resolved_x,
        y=_normalize_probabilities(y),
        A=A,
        d=d,
        e=e,
        open_mass=float(np.sum(e, dtype=float)),
    )


def _load_identity_vectors(
    *,
    block1_payload: dict[str, Any],
    state_ids: tuple[int, ...],
) -> np.ndarray:
    """Load the Block 1 community identity vectors aligned to the K-state axis.

    The live Block 3 contract requires a complete, non-degenerate `g_k` surface
    aligned to every shared state carried into the benchmark. Missing states or
    zero-total rows are contract violations and must fail fast rather than
    silently falling back to a cohort-average proxy row.
    """

    output_dir = _resolve_path(block1_payload["output_dir"])
    fractions_path = output_dir / "community_correspondence" / "tables" / "community_cell_subtype_row_fractions.csv"
    frame = pd.read_csv(fractions_path, index_col=0)
    frame.index = frame.index.astype(int)
    if frame.empty:
        raise ContractError("Block 3 requires non-empty community identity vectors from Block 1 correspondence")

    missing_state_ids = [int(state_id) for state_id in state_ids if int(state_id) not in frame.index]
    if missing_state_ids:
        raise ContractError(
            "Block 3 requires community identity vectors for every shared state; "
            f"missing community identity vectors for state_ids={missing_state_ids}"
        )

    vectors: list[np.ndarray] = []
    for state_id in state_ids:
        row = frame.loc[int(state_id)].to_numpy(dtype=float)
        if not np.all(np.isfinite(row)):
            raise ContractError(
                "Block 3 requires finite community identity vectors; "
                f"state_id={int(state_id)} contains non-finite values"
            )
        row_total = float(np.sum(row, dtype=float))
        if row_total <= 0.0:
            raise ContractError(
                "Block 3 requires positive-total community identity vectors; "
                f"state_id={int(state_id)} has zero total mass"
            )
        vectors.append(_normalize_probabilities(row))
    return np.vstack(vectors)


def _build_identity_cost_matrix(identity_vectors: np.ndarray) -> np.ndarray:
    """Build the normalized identity-aware Block 3 cost matrix `C`.

    The scientific contract freezes `C_raw[i,j] = sqrt(JS(g_i, g_j))` with the
    normalization scale `s_C` defined as the median of positive off-diagonal
    entries. That median-scaled `C` is reused both for relation-support
    geometry and the balanced OT baseline.
    """

    n_states = identity_vectors.shape[0]
    cost_raw = np.zeros((n_states, n_states), dtype=float)
    for left_index in range(n_states):
        for right_index in range(left_index + 1, n_states):
            value = float(np.sqrt(_js_divergence(identity_vectors[left_index], identity_vectors[right_index])))
            cost_raw[left_index, right_index] = value
            cost_raw[right_index, left_index] = value
    positive = cost_raw[cost_raw > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    return cost_raw / scale


def _load_block3_cohort_inputs(inputs: Block3ResolvedInputs) -> Block3CohortInputs:
    """Resolve the real cohort inputs needed by the Block 3 generator.

    Purpose:
        Pull the eligible TC/IM patient cohort, state basis, Block 1 identity
        vectors, and identity-aware cost matrix into one generator-ready
        object.

    Inputs:
        inputs: Validated upstream Block 1/2 prerequisites.

    Returns:
        A `Block3CohortInputs` bundle containing the carried patient profiles
        and the aligned state-space metadata required by methods and scorers.

    Raises:
        ContractError: The frozen `TC-IM` family is missing or fewer than
            32 eligible patients are available for the 24/8 split design.

    Core flow:
        1. Load Block 1 provenance and the Task A config bundle.
        2. Resolve the frozen `TC-IM` family and shared state basis.
        3. Aggregate patient-level source and target endpoint profiles.
        4. Enforce the frozen minimum cohort size for Phase 3 reruns.
        5. Resolve identity vectors and build the normalized cost matrix.

    Doc support:
        The returned source profiles become the held-out `x_p` carriers, while
        the real target profiles remain generator-side inputs for open
        calibration and `3A` validation only.
    """
    block1_payload = _load_json_dict(inputs.block1_bundle_path, label="Task A Block 1 bundle")
    stage0_h5ad = _resolve_path(block1_payload["stage0_h5ad"])
    config_path = _resolve_path(block1_payload["config_path"])
    config_bundle = load_task_a_config_bundle(config_path)
    family_spec = next(
        (family for family in config_bundle.ordered_proxy.pair_families if family.name == _BLOCK3_PAIR_FAMILY),
        None,
    )
    if family_spec is None:
        raise ContractError("Task A Block 3 requires the frozen TC-IM pair family in Task A config")

    handle = load_task_a_dataset_handle(stage0_h5ad)
    state_basis = resolve_task_a_state_basis(handle)
    observations = build_task_a_family_observations(
        handle,
        family_spec,
        state_basis=state_basis,
        mass_mode=config_bundle.data.mass_mode,
        require_complete_patients=True,
    )
    patient_vectors: dict[str, dict[str, list[np.ndarray]]] = {}
    for observation in observations:
        patient_vectors.setdefault(observation.patient_id, {}).setdefault(str(observation.timepoint), []).append(
            np.asarray(observation.community_composition, dtype=float)
        )
    patient_source_profiles: dict[str, np.ndarray] = {}
    patient_target_profiles: dict[str, np.ndarray] = {}
    for patient_id, domain_vectors in sorted(patient_vectors.items()):
        if family_spec.source_domain not in domain_vectors or family_spec.target_domain not in domain_vectors:
            continue
        patient_source_profiles[patient_id] = _mean_profile(domain_vectors[family_spec.source_domain])
        patient_target_profiles[patient_id] = _mean_profile(domain_vectors[family_spec.target_domain])
    if len(patient_source_profiles) < (_N_TRAIN_PATIENTS + _N_TEST_PATIENTS):
        raise ContractError("Block 3 internal execution requires at least 32 eligible patients for the frozen 24/8 split")
    identity_vectors = _load_identity_vectors(
        block1_payload=block1_payload,
        state_ids=tuple(int(state_id) for state_id in state_basis.resolved_state_ids),
    )
    cost_matrix = _build_identity_cost_matrix(identity_vectors)
    geometry = build_state_geometry(
        cost_matrix=cost_matrix,
        cost_scale=1.0,
        state_ids=tuple(int(state_id) for state_id in state_basis.resolved_state_ids),
    )
    return Block3CohortInputs(
        stage0_h5ad=stage0_h5ad,
        config_path=config_path,
        output_dir=_resolve_path(block1_payload["output_dir"]),
        master_seed=int(config_bundle.block2.master_seed),
        state_ids=tuple(int(state_id) for state_id in state_basis.resolved_state_ids),
        state_basis=state_basis,
        geometry=geometry,
        identity_vectors=identity_vectors,
        cost_matrix=cost_matrix,
        patient_source_profiles=patient_source_profiles,
        patient_target_profiles=patient_target_profiles,
    )


def _estimate_open_calibration(
    *,
    cohort_inputs: Block3CohortInputs,
    train_patient_ids: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Estimate the training-side open calibration quantities for one rerun.

    Returns:
        `(pi_d, pi_e, kappa_d, kappa_e, masses)` where `masses` is the
        empirical `P(m)` support sampled from real train-side TC/IM shifts.
    """
    source_fallback = _mean_profile([cohort_inputs.patient_source_profiles[patient_id] for patient_id in train_patient_ids])
    target_fallback = _mean_profile([cohort_inputs.patient_target_profiles[patient_id] for patient_id in train_patient_ids])
    masses: list[float] = []
    depletion_profiles: list[np.ndarray] = []
    emergence_profiles: list[np.ndarray] = []
    for patient_id in train_patient_ids:
        source = cohort_inputs.patient_source_profiles[patient_id]
        target = cohort_inputs.patient_target_profiles[patient_id]
        depletion = np.maximum(source - target, 0.0)
        emergence = np.maximum(target - source, 0.0)
        masses.append(float(0.5 * np.sum(np.abs(source - target), dtype=float)))
        if float(np.sum(depletion, dtype=float)) > 0.0:
            depletion_profiles.append(_normalize_probabilities(depletion))
        if float(np.sum(emergence, dtype=float)) > 0.0:
            emergence_profiles.append(_normalize_probabilities(emergence))
    pi_d = _mean_profile(depletion_profiles) if depletion_profiles else source_fallback
    pi_e = _mean_profile(emergence_profiles) if emergence_profiles else target_fallback
    return (
        pi_d,
        pi_e,
        _estimate_kappa(depletion_profiles or [pi_d]),
        _estimate_kappa(emergence_profiles or [pi_e]),
        np.asarray(masses, dtype=float) if masses else np.asarray([0.0], dtype=float),
    )


def _estimate_relation_motif(
    *,
    cohort_inputs: Block3CohortInputs,
    train_patient_ids: tuple[str, ...],
) -> np.ndarray:
    """Estimate one rerun-shared source-specific relation motif from train endpoints."""

    n_states = len(cohort_inputs.state_ids)
    relation_motif = np.zeros((n_states, n_states), dtype=float)
    for patient_id in train_patient_ids:
        source = _normalize_probabilities(cohort_inputs.patient_source_profiles[patient_id])
        target = _normalize_probabilities(cohort_inputs.patient_target_profiles[patient_id])
        positive_gain = np.maximum(target - source, 0.0)
        relation_motif += source[:, None] * positive_gain[None, :]
    np.fill_diagonal(relation_motif, 0.0)
    for source_index in range(n_states):
        row = np.asarray(relation_motif[source_index], dtype=float).reshape(-1)
        row[source_index] = 0.0
        row_total = float(np.sum(row, dtype=float))
        if row_total <= _TOL:
            relation_motif[source_index] = 0.0
            continue
        relation_motif[source_index] = row / row_total
    return relation_motif


def _build_generator_reruns(
    *,
    cohort_inputs: Block3CohortInputs,
    support_mode: str = _SUPPORT_MODE_LEGACY_NEAREST_C,
    open_mass_scale: float = 1.0,
) -> tuple[Block3GeneratorRerun, ...]:
    """Build the full frozen set of rerun-specific semisynthetic realizations.

    Purpose:
        Materialize the common generator flow shared by `3A`, `3B`, and `3C-*`
        so that downstream sections reuse one audit-traceable truth source.

    Inputs:
        cohort_inputs: Real cohort inputs and state-space metadata resolved from
            evidence-ready Block 1/2 artifacts.

    Returns:
        A tuple of `Block3GeneratorRerun` objects, each carrying the frozen
        24-train / 8-test split plus the truth objects required by each section.

    Core flow:
        1. Sample 10 rerun-specific train/test splits from the eligible cohort.
        2. Estimate train-only open calibration quantities.
        3. Sample rerun-specific open burdens for each held-out patient.
        4. Build one hidden shared realization for `3A` and `3C-*`.
        5. Expand the same open realization across the public `relation_*`
           conditions for `3B`.
    """
    support_mode = _validate_support_mode(support_mode)
    open_mass_scale = _validate_open_mass_scale(open_mass_scale)
    patient_ids = tuple(sorted(cohort_inputs.patient_source_profiles))
    reruns: list[Block3GeneratorRerun] = []
    for rerun_index in range(_N_GENERATOR_RERUNS):
        split_seed = int(cohort_inputs.master_seed + rerun_index)
        rng = np.random.default_rng(split_seed)
        selected = tuple(rng.permutation(patient_ids)[: (_N_TRAIN_PATIENTS + _N_TEST_PATIENTS)].tolist())
        train_patient_ids = tuple(sorted(selected[:_N_TRAIN_PATIENTS]))
        test_patient_ids = tuple(sorted(selected[_N_TRAIN_PATIENTS : _N_TRAIN_PATIENTS + _N_TEST_PATIENTS]))
        pi_d, pi_e, kappa_d, kappa_e, masses = _estimate_open_calibration(
            cohort_inputs=cohort_inputs,
            train_patient_ids=train_patient_ids,
        )
        relation_motif = (
            _estimate_relation_motif(
                cohort_inputs=cohort_inputs,
                train_patient_ids=train_patient_ids,
            )
            if support_mode == _SUPPORT_MODE_RELATION_MOTIF_PROBE
            else None
        )
        hidden_relation_condition_id = str(
            rng.choice(np.asarray(_HIDDEN_RELATION_OPTIONS, dtype=object))
        )
        generator_truths: dict[str, Block3PatientTruth] = {}
        baseline_truths = {condition_id: {} for condition_id in _RELATION_SPECS}
        rerun_id = f"rerun_{rerun_index + 1:02d}"
        for patient_id in test_patient_ids:
            x = cohort_inputs.patient_source_profiles[patient_id]
            sampled_mass = float(rng.choice(masses))
            base_minus = _normalize_probabilities(x * pi_d if float(np.sum(x * pi_d)) > 0.0 else x)
            base_plus = _normalize_probabilities((x + _EPSILON_FIXED) * pi_e)
            delta_minus = _sample_capped_depletion(
                x=x,
                mass=sampled_mass,
                base=base_minus,
                kappa=kappa_d,
                rng=rng,
            )
            delta_plus = sampled_mass * _sample_gamma_simplex(
                base=base_plus,
                kappa=kappa_e,
                rng=rng,
            )
            scaled_delta_minus = open_mass_scale * delta_minus
            scaled_delta_plus = open_mass_scale * delta_plus
            generator_truths[patient_id] = _build_truth_for_condition(
                rerun_id=rerun_id,
                patient_id=patient_id,
                x=x,
                delta_minus=scaled_delta_minus,
                delta_plus=scaled_delta_plus,
                condition_id=hidden_relation_condition_id,
                cost_matrix=cohort_inputs.cost_matrix,
                support_mode=support_mode,
                relation_motif=relation_motif,
            )
            for condition_id in _RELATION_SPECS:
                baseline_truths[condition_id][patient_id] = _build_truth_for_condition(
                    rerun_id=rerun_id,
                    patient_id=patient_id,
                    x=x,
                    delta_minus=scaled_delta_minus,
                    delta_plus=scaled_delta_plus,
                    condition_id=condition_id,
                    cost_matrix=cohort_inputs.cost_matrix,
                    support_mode=support_mode,
                    relation_motif=relation_motif,
                )
        reruns.append(
            Block3GeneratorRerun(
                rerun_id=rerun_id,
                split_seed=split_seed,
                train_patient_ids=train_patient_ids,
                test_patient_ids=test_patient_ids,
                hidden_relation_condition_id=hidden_relation_condition_id,
                support_mode=support_mode,
                open_mass_scale=open_mass_scale,
                relation_motif=relation_motif,
                generator_truths=generator_truths,
                baseline_truths=baseline_truths,
            )
        )
    return tuple(reruns)


def _make_observations_for_truths(truths: list[Block3PatientTruth]) -> tuple[FovObservation, ...]:
    """Expose one truth set to methods as paired endpoints only."""

    observations: list[FovObservation] = []
    for truth in truths:
        observations.append(
            FovObservation(
                patient_id=truth.patient_id,
                timepoint="pre",
                fov_id=f"{truth.rerun_id}_{truth.patient_id}_pre",
                domain_label="TC",
                community_composition=np.asarray(truth.x, dtype=float),
                mass=1.0,
                mass_mode="uniform",
            )
        )
        observations.append(
            FovObservation(
                patient_id=truth.patient_id,
                timepoint="post",
                fov_id=f"{truth.rerun_id}_{truth.patient_id}_post",
                domain_label="IM",
                community_composition=np.asarray(truth.y, dtype=float),
                mass=1.0,
                mass_mode="uniform",
            )
        )
    return tuple(observations)


def _run_stride_method(
    *,
    cohort_inputs: Block3CohortInputs,
    truths: list[Block3PatientTruth],
    benchmark_mode: str,
) -> dict[str, Block3MethodOutput]:
    """Run canonical STRIDE or one STRIDE ablation on paired endpoint inputs."""

    fit_result = fit_stride(
        _make_observations_for_truths(truths),
        state_basis=cohort_inputs.state_basis,
        geometry=cohort_inputs.geometry,
        config=STRIDEFitConfig(
            timepoint_order=("pre", "post"),
            benchmark_mode=benchmark_mode,
        ),
    )
    outputs: dict[str, Block3MethodOutput] = {}
    for patient_result in fit_result.patient_results:
        outputs[patient_result.patient_id] = Block3MethodOutput(
            patient_id=patient_result.patient_id,
            fit_status=patient_result.fit_status,
            A=(np.asarray(patient_result.A, dtype=float) if patient_result.A is not None else None),
            d=(np.asarray(patient_result.d, dtype=float) if patient_result.d is not None else None),
            e=(np.asarray(patient_result.e, dtype=float) if patient_result.e is not None else None),
            mu_minus=(
                np.asarray(patient_result.mu_minus, dtype=float)
                if patient_result.mu_minus is not None
                else None
            ),
            mu_plus=(
                np.asarray(patient_result.mu_plus, dtype=float)
                if patient_result.mu_plus is not None
                else None
            ),
        )
    return outputs


def _run_balanced_ot_baseline(
    *,
    cohort_inputs: Block3CohortInputs,
    truths: list[Block3PatientTruth],
) -> dict[str, Block3MethodOutput]:
    """Run the exact balanced-OT closed baseline over the fixed cost matrix.

    The transport equalities are encoded with one redundant marginal
    constraint removed. On simplex marginals the final column sum is implied by
    the row sums plus the preceding column sums, and keeping that dependent
    equality can trigger false HiGHS infeasibility on numerically delicate
    real-data marginals.
    """

    from scipy.optimize import linprog

    outputs: dict[str, Block3MethodOutput] = {}
    cost_matrix = np.asarray(cohort_inputs.cost_matrix, dtype=float)
    n_states = cost_matrix.shape[0]
    c = cost_matrix.reshape(-1)
    row_constraints = []
    col_constraints = []
    for row_index in range(n_states):
        row = np.zeros(n_states * n_states, dtype=float)
        row[row_index * n_states : (row_index + 1) * n_states] = 1.0
        row_constraints.append(row)
    for col_index in range(n_states):
        col = np.zeros(n_states * n_states, dtype=float)
        col[col_index::n_states] = 1.0
        col_constraints.append(col)
    A_eq = np.vstack(row_constraints + col_constraints[:-1])
    for truth in truths:
        x = np.asarray(truth.x, dtype=float)
        y = np.asarray(truth.y, dtype=float)
        b_eq = np.concatenate([x, y[:-1]]).astype(float, copy=False)
        result = linprog(
            c=c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=[(0.0, None)] * (n_states * n_states),
            method="highs",
        )
        if not result.success:
            raise ContractError(f"Balanced OT baseline failed for patient {truth.patient_id}: {result.message}")
        transport = np.asarray(result.x, dtype=float).reshape(n_states, n_states)
        operator = np.zeros_like(transport)
        positive_mask = x > _TOL
        operator[positive_mask] = transport[positive_mask] / x[positive_mask, None]
        outputs[truth.patient_id] = Block3MethodOutput(
            patient_id=truth.patient_id,
            fit_status="ok",
            A=operator,
            d=np.zeros(n_states, dtype=float),
            e=np.zeros(n_states, dtype=float),
            mu_minus=x,
            mu_plus=np.sum(x[:, None] * operator, axis=0, dtype=float),
            P=transport,
        )
    return outputs


def _run_plan_baseline(
    *,
    truths: list[Block3PatientTruth],
    plan_builder: Any,
) -> dict[str, Block3MethodOutput]:
    """Run one plan-based baseline through the shared `P -> A/d/e` layer.

    Inputs are test-side truth endpoint pairs and a callable returning either a
    native `P` array or `PlanBaselineResult`. Outputs are method records with
    derived `A/d/e` for `ok` plans and non-estimable status for failed solves.
    This internal helper does not compute metrics directly.
    """

    outputs: dict[str, Block3MethodOutput] = {}
    for truth in truths:
        x = np.asarray(truth.x, dtype=float)
        y = np.asarray(truth.y, dtype=float)
        result = plan_builder(x, y)
        if isinstance(result, PlanBaselineResult):
            plan = None if result.P is None else np.asarray(result.P, dtype=float)
            fit_status = result.status
            metadata = result.metadata
        else:
            plan = np.asarray(result, dtype=float)
            fit_status = "ok"
            metadata = None
        if fit_status != "ok" or plan is None:
            outputs[truth.patient_id] = Block3MethodOutput(
                patient_id=truth.patient_id,
                fit_status=fit_status,
                A=None,
                d=None,
                e=None,
                mu_minus=x,
                mu_plus=None,
                P=None,
                metadata=metadata,
            )
            continue
        A, d, e = derive_A_d_e_from_plan(x=x, y=y, P=plan, tol=_TOL)
        outputs[truth.patient_id] = Block3MethodOutput(
            patient_id=truth.patient_id,
            fit_status="ok",
            A=A,
            d=d,
            e=e,
            mu_minus=x,
            mu_plus=np.sum(plan, axis=0, dtype=float),
            P=plan,
            metadata=metadata,
        )
    return outputs


def _run_uot_baseline(
    *,
    truths: list[Block3PatientTruth],
    cost_matrix: np.ndarray,
    match_penalty: float = 1.0,
    calibration_metadata: dict[str, object] | None = None,
) -> dict[str, Block3MethodOutput]:
    """Run UOT baseline plans and derive native `A/d/e` outputs.

    Inputs are test-side truths, a shared cost matrix, and calibrated lambda.
    Outputs include native `P` and UOT metadata per patient. Solver failures are
    carried as non-ok status for metric propagation.
    """

    outputs = _run_plan_baseline(
        truths=truths,
        plan_builder=lambda x, y: solve_uot_plan(
            x=x,
            y=y,
            cost_matrix=cost_matrix,
            match_penalty=match_penalty,
        ),
    )
    if calibration_metadata is None:
        return outputs
    return {
        patient_id: Block3MethodOutput(
            patient_id=output.patient_id,
            fit_status=output.fit_status,
            A=output.A,
            d=output.d,
            e=output.e,
            mu_minus=output.mu_minus,
            mu_plus=output.mu_plus,
            P=output.P,
            metadata={**(output.metadata or {}), **calibration_metadata},
        )
        for patient_id, output in outputs.items()
    }


def _run_partial_ot_baseline(
    *,
    truths: list[Block3PatientTruth],
    cost_matrix: np.ndarray | None = None,
    matched_mass_budget: float | None = None,
) -> dict[str, Block3MethodOutput]:
    """Run hard-budget partial-OT plans and derive native `A/d/e` outputs.

    Inputs are test-side truths, optional costs, and an optional train-side
    matched-mass budget. Output metadata records requested/effective budget and
    clipping status for each patient.
    """

    return _run_plan_baseline(
        truths=truths,
        plan_builder=lambda x, y: partial_ot_plan(
            x,
            y,
            cost_matrix=cost_matrix,
            matched_mass_budget=matched_mass_budget,
        ),
    )


def _run_diagonal_transport_baseline(*, truths: list[Block3PatientTruth]) -> dict[str, Block3MethodOutput]:
    """Run diagonal native plans on test-side truths.

    Inputs are test-side truths. Outputs include native diagonal `P`; `A/d/e`
    are derived by the shared `P -> A/d/e` analysis layer. This comparator is
    not the historical abundance-only baseline.
    """

    return _run_plan_baseline(truths=truths, plan_builder=diagonal_transport_plan)


def _train_pairs_for_truths(truths: list[Block3PatientTruth]) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return endpoint pairs used by Block3b train-side calibration helpers.

    Inputs are train-side truth objects from one rerun/condition. Output is a
    tuple of `(x, y)` arrays consumed by UOT calibration or partial-OT budget
    estimation. Hidden `A/d/e` fields are not exposed to ranked methods.
    """

    return tuple((np.asarray(truth.x, dtype=float), np.asarray(truth.y, dtype=float)) for truth in truths)


def _matched_mass_budget_from_train(truths: list[Block3PatientTruth]) -> float:
    """Estimate deterministic train-side matched-mass budget for partial OT.

    Inputs are train-side endpoint-only truths. Output is the mean endpoint
    overlap used as the requested hard budget; callers still clip per patient
    to feasible source/target mass.
    """

    pairs = _train_pairs_for_truths(truths)
    if not pairs:
        raise ValueError("Partial OT budget estimation requires train truths")
    # matched_mass_budget is the train-side requested hard comparator budget.
    matched_mass_budget = float(np.mean([np.sum(np.minimum(x, y), dtype=float) for x, y in pairs]))
    return matched_mass_budget


def _calibrated_uot_lambda_for_train(
    *,
    train_truths: list[Block3PatientTruth],
    cost_matrix: np.ndarray,
) -> UOTCalibrationResult:
    """Select one train-side UOT lambda for a Block3b rerun/condition.

    Inputs are train endpoint truths and the shared cost matrix. Output is the
    calibrated lambda result; boundary-hit diagnostics are carried on patient
    UOT metadata rather than failing the run.
    """

    train_pairs = _train_pairs_for_truths(train_truths)
    calibration = calibrate_uot_lambda(
        train_pairs=train_pairs,
        achieved_mass_fn=lambda candidate, pairs: estimate_uot_matched_mass(
            train_pairs=tuple(pairs),
            cost_matrix=cost_matrix,
            match_penalty=candidate,
        ),
    )
    return calibration


def _uot_calibration_metadata(calibration: UOTCalibrationResult) -> dict[str, object]:
    """Serialize UOT train-side calibration diagnostics for native metadata."""

    return {
        "selected_lambda": calibration.selected_lambda,
        "target_overlap": calibration.target_overlap,
        "boundary_hit": calibration.boundary_hit,
        "achieved_by_lambda": calibration.achieved_by_lambda,
        "absolute_error_by_lambda": calibration.absolute_error_by_lambda,
    }


def _matrix_mae_active(x: np.ndarray, A_true: np.ndarray, A_hat: np.ndarray) -> float:
    """Compute truth-anchored active-row MAE on transported mass."""

    active_rows = np.asarray(x, dtype=float) > _TOL
    if not np.any(active_rows):
        return 0.0
    truth_mass = np.asarray(x, dtype=float)[:, None] * np.asarray(A_true, dtype=float)
    pred_mass = np.asarray(x, dtype=float)[:, None] * np.asarray(A_hat, dtype=float)
    return float(np.mean(np.abs(truth_mass[active_rows] - pred_mass[active_rows])))


def _matrix_mse_active(x: np.ndarray, A_true: np.ndarray, A_hat: np.ndarray) -> float:
    """Compute truth-anchored active-row MSE on transported mass."""

    active_rows = np.asarray(x, dtype=float) > _TOL
    if not np.any(active_rows):
        return 0.0
    truth_mass = np.asarray(x, dtype=float)[:, None] * np.asarray(A_true, dtype=float)
    pred_mass = np.asarray(x, dtype=float)[:, None] * np.asarray(A_hat, dtype=float)
    delta = truth_mass[active_rows] - pred_mass[active_rows]
    return float(np.mean(delta * delta))


def _target_recall_at_k(x: np.ndarray, A_true: np.ndarray, A_hat: np.ndarray) -> float | None:
    """Recover the spec-defined off-diagonal target recall at truth-derived `k`."""

    truth_mass = np.asarray(x, dtype=float)[:, None] * np.asarray(A_true, dtype=float)
    pred_mass = np.asarray(x, dtype=float)[:, None] * np.asarray(A_hat, dtype=float)
    true_target_scores = np.sum(truth_mass, axis=0, dtype=float) - np.diag(truth_mass)
    pred_target_scores = np.sum(pred_mass, axis=0, dtype=float) - np.diag(pred_mass)
    true_targets = {int(index) for index, value in enumerate(true_target_scores) if value > _TOL}
    if not true_targets:
        return None
    k = len(true_targets)
    predicted_rank = sorted(
        range(pred_target_scores.shape[0]),
        key=lambda index: (-float(pred_target_scores[index]), int(index)),
    )[:k]
    return float(len(true_targets.intersection(predicted_rank)) / len(true_targets))


def _coverage_support(values: np.ndarray) -> set[int]:
    """Return the smallest support set covering the required open-mass fraction."""

    vector = np.asarray(values, dtype=float)
    total = float(np.sum(vector, dtype=float))
    if total <= _TOL:
        return set()
    normalized = vector / total
    ordered = sorted(
        range(vector.shape[0]),
        key=lambda index: (-float(normalized[index]), int(index)),
    )
    support: set[int] = set()
    covered = 0.0
    for index in ordered:
        if normalized[index] <= 0.0 and covered >= _OPEN_SUPPORT_COVERAGE:
            break
        support.add(int(index))
        covered += float(normalized[index])
        if covered >= _OPEN_SUPPORT_COVERAGE - 1e-12:
            break
    return support


def _f1_score(true_support: set[int], predicted_support: set[int]) -> float:
    """Compute a simple F1 score for two finite support sets."""

    if not true_support and not predicted_support:
        return 1.0
    if not true_support or not predicted_support:
        return 0.0
    true_positive = len(true_support.intersection(predicted_support))
    if true_positive == 0:
        return 0.0
    precision = true_positive / len(predicted_support)
    recall = true_positive / len(true_support)
    return float(2.0 * precision * recall / (precision + recall))


def _open_support_f1(
    true_depletion: np.ndarray,
    pred_depletion: np.ndarray,
    true_emergence: np.ndarray,
    pred_emergence: np.ndarray,
) -> float:
    """Compute the burden-scale open support F1 required by `3C-*`."""

    depletion_f1 = _f1_score(
        _coverage_support(true_depletion),
        _coverage_support(pred_depletion),
    )
    emergence_f1 = _f1_score(
        _coverage_support(true_emergence),
        _coverage_support(pred_emergence),
    )
    return float(0.5 * (depletion_f1 + emergence_f1))


def _method_native_records(
    *,
    subexperiment_id: str,
    condition_id: str,
    rerun_id: str,
    method_name: str,
    outputs: dict[str, Block3MethodOutput],
    open_mass_scale: float | None = None,
) -> list[dict[str, object]]:
    """Serialize native method outputs into the proof-carrying raw table."""

    method_spec = get_method_spec(method_name)
    records: list[dict[str, object]] = []
    for patient_id, output in sorted(outputs.items()):
        records.append(
            {
                "rerun_id": rerun_id,
                "subexperiment_id": subexperiment_id,
                "condition_id": condition_id,
                "patient_id": patient_id,
                "method_name": method_name,
                "method_class": method_spec.method_class.value,
                "fit_status": output.fit_status,
                "A_json": _json_array(output.A),
                "d_json": _json_array(output.d),
                "e_json": _json_array(output.e),
                "mu_minus_json": _json_array(output.mu_minus),
                "mu_plus_json": _json_array(output.mu_plus),
                "P_json": _json_array(output.P),
                "metadata_json": json.dumps(output.metadata, sort_keys=True, separators=(",", ":"))
                if output.metadata is not None
                else None,
                "open_mass_scale": open_mass_scale,
            }
        )
    return records


def _truth_store_records(
    *,
    subexperiment_id: str,
    condition_id: str,
    truths: list[Block3PatientTruth],
    open_mass_scale: float | None = None,
) -> list[dict[str, object]]:
    """Serialize hidden truth objects into the proof-carrying raw table."""

    return [
        {
            "rerun_id": truth.rerun_id,
            "subexperiment_id": subexperiment_id,
            "condition_id": condition_id,
            "patient_id": truth.patient_id,
            "x_json": _json_array(truth.x),
            "y_json": _json_array(truth.y),
            "A_json": _json_array(truth.A),
            "d_json": _json_array(truth.d),
            "e_json": _json_array(truth.e),
            "open_mass": truth.open_mass,
            "open_mass_scale": open_mass_scale,
        }
        for truth in truths
    ]


def _shared_registry_tables(reruns: tuple[Block3GeneratorRerun, ...]) -> dict[str, tuple[dict[str, object], ...]]:
    """Build the shared rerun and split registries reused by every section."""

    rerun_records = tuple(
        {
            "rerun_id": rerun.rerun_id,
            "split_seed": rerun.split_seed,
            "n_train_patients": len(rerun.train_patient_ids),
            "n_test_patients": len(rerun.test_patient_ids),
            "hidden_relation_condition_id": rerun.hidden_relation_condition_id,
        }
        for rerun in reruns
    )
    split_records: list[dict[str, object]] = []
    for rerun in reruns:
        for patient_id in rerun.train_patient_ids:
            split_records.append(
                {
                    "rerun_id": rerun.rerun_id,
                    "split_seed": rerun.split_seed,
                    "patient_id": patient_id,
                    "split_role": "train",
                }
            )
        for patient_id in rerun.test_patient_ids:
            split_records.append(
                {
                    "rerun_id": rerun.rerun_id,
                    "split_seed": rerun.split_seed,
                    "patient_id": patient_id,
                    "split_role": "test",
                }
            )
    return {
        "generator_rerun_registry": rerun_records,
        "generator_split_registry": tuple(split_records),
    }


def _build_3a_rows(
    *,
    reruns: tuple[Block3GeneratorRerun, ...],
    cohort_inputs: Block3CohortInputs,
) -> tuple[Block3SubexperimentRawRows, Block3SubexperimentReviewRows]:
    """Build raw and review rows for `3A` generator validation.

    `3A` stays non-method-bearing. It compares rerun-level synthetic target
    surfaces against held-out real target surfaces and exports rerun-stability
    rows rather than method summaries.
    """
    subexperiment_id = Block3SubexperimentId.GENERATOR_VALIDATION.value
    subexperiment = get_subexperiment_spec(subexperiment_id)
    condition = get_condition_spec(_GENERATOR_VALIDATION_CONDITION_ID)
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
        real_surface = _mean_profile(
            [cohort_inputs.patient_target_profiles[patient_id] for patient_id in rerun.test_patient_ids]
        )
        synthetic_surface = _mean_profile(
            [rerun.generator_truths[patient_id].y for patient_id in rerun.test_patient_ids]
        )
        truth_records.extend(
            _truth_store_records(
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
            for metric_name, metric_number in _metric_bundle(real_vector, synthetic_vector).items():
                metric_value = make_metric_value(metric_name=metric_name, value=metric_number, status="reported")
                _validate_metric_value(
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
        _validate_metric_value(
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

    shared_tables = _shared_registry_tables(reruns)
    shared_tables["patient_truth_store"] = tuple(truth_records)
    return (
        Block3SubexperimentRawRows(
            object_scores=tuple(object_rows),
            rerun_stability=tuple(stability_rows),
            shared_tables=shared_tables,
        ),
        Block3SubexperimentReviewRows(generator_rows=tuple(review_rows)),
    )


def _build_3b1_rows(
    *,
    reruns: tuple[Block3GeneratorRerun, ...],
    cohort_inputs: Block3CohortInputs,
) -> tuple[Block3SubexperimentRawRows, Block3SubexperimentReviewRows]:
    """Build raw and review rows for executable `3B-1` A benchmark."""

    subexperiment_id = Block3SubexperimentId.A_BENCHMARK.value
    evaluation_family = get_subexperiment_spec(subexperiment_id).evaluation_family
    patient_rows: list[Block3PatientMetricRow] = []
    truth_records: list[dict[str, object]] = []
    native_records: list[dict[str, object]] = []

    for rerun in reruns:
        for condition_id in _RELATION_SPECS:
            truths = [rerun.baseline_truths[condition_id][patient_id] for patient_id in rerun.test_patient_ids]
            train_truths = list(rerun.baseline_truths[condition_id].values())
            uot_calibration = _calibrated_uot_lambda_for_train(
                train_truths=train_truths,
                cost_matrix=cohort_inputs.cost_matrix,
            )
            matched_mass_budget = _matched_mass_budget_from_train(train_truths)
            truth_records.extend(
                _truth_store_records(
                    subexperiment_id=subexperiment_id,
                    condition_id=condition_id,
                    truths=truths,
                    open_mass_scale=rerun.open_mass_scale,
                )
            )
            outputs_by_method = {
                "stride_reference": _run_stride_method(
                    cohort_inputs=cohort_inputs,
                    truths=truths,
                    benchmark_mode="reference",
                ),
                "balanced_ot_baseline": _run_balanced_ot_baseline(
                    cohort_inputs=cohort_inputs,
                    truths=truths,
                ),
                "uot_baseline": _run_uot_baseline(
                    truths=truths,
                    cost_matrix=cohort_inputs.cost_matrix,
                    match_penalty=uot_calibration.selected_lambda,
                    calibration_metadata=_uot_calibration_metadata(uot_calibration),
                ),
                "partial_ot_baseline": _run_partial_ot_baseline(
                    truths=truths,
                    cost_matrix=cohort_inputs.cost_matrix,
                    matched_mass_budget=matched_mass_budget,
                ),
                "diagonal_transport_baseline": _run_diagonal_transport_baseline(truths=truths),
            }
            for method_name, outputs in outputs_by_method.items():
                native_records.extend(
                    _method_native_records(
                        subexperiment_id=subexperiment_id,
                        condition_id=condition_id,
                        rerun_id=rerun.rerun_id,
                        method_name=method_name,
                        outputs=outputs,
                        open_mass_scale=rerun.open_mass_scale,
                    )
                )
                for truth in truths:
                    output = outputs[truth.patient_id]
                    if output.fit_status != "ok" or output.A is None:
                        for metric_name in (
                            Block3MetricName.A_MAE_ACTIVE,
                            Block3MetricName.A_MSE_ACTIVE,
                            Block3MetricName.TARGET_RECALL_AT_K,
                        ):
                            status = (
                                MetricStatus.NOT_APPLICABLE
                                if condition_id == "relation_null" and metric_name is Block3MetricName.TARGET_RECALL_AT_K
                                else MetricStatus.NOT_ESTIMABLE
                            )
                            patient_rows.append(
                                _make_patient_metric_row(
                                    rerun_id=rerun.rerun_id,
                                    subexperiment_id=subexperiment_id,
                                    condition_id=condition_id,
                                    evaluation_family=evaluation_family,
                                    method_name=method_name,
                                    metric_name=metric_name,
                                    value=None,
                                    status=status,
                                    patient_id=truth.patient_id,
                                    open_mass_scale=rerun.open_mass_scale,
                                )
                            )
                        continue
                    patient_rows.append(
                        _make_patient_metric_row(
                            rerun_id=rerun.rerun_id,
                            subexperiment_id=subexperiment_id,
                            condition_id=condition_id,
                            evaluation_family=evaluation_family,
                            method_name=method_name,
                            metric_name=Block3MetricName.A_MAE_ACTIVE,
                            value=_matrix_mae_active(truth.x, truth.A, output.A),
                            status=MetricStatus.REPORTED,
                            patient_id=truth.patient_id,
                            open_mass_scale=rerun.open_mass_scale,
                        )
                    )
                    patient_rows.append(
                        _make_patient_metric_row(
                            rerun_id=rerun.rerun_id,
                            subexperiment_id=subexperiment_id,
                            condition_id=condition_id,
                            evaluation_family=evaluation_family,
                            method_name=method_name,
                            metric_name=Block3MetricName.A_MSE_ACTIVE,
                            value=_matrix_mse_active(truth.x, truth.A, output.A),
                            status=MetricStatus.REPORTED,
                            patient_id=truth.patient_id,
                            open_mass_scale=rerun.open_mass_scale,
                        )
                    )
                    recall = _target_recall_at_k(truth.x, truth.A, output.A)
                    patient_rows.append(
                        _make_patient_metric_row(
                            rerun_id=rerun.rerun_id,
                            subexperiment_id=subexperiment_id,
                            condition_id=condition_id,
                            evaluation_family=evaluation_family,
                            method_name=method_name,
                            metric_name=Block3MetricName.TARGET_RECALL_AT_K,
                            value=recall,
                            status=(
                                MetricStatus.NOT_APPLICABLE
                                if recall is None
                                else MetricStatus.REPORTED
                            ),
                            patient_id=truth.patient_id,
                            open_mass_scale=rerun.open_mass_scale,
                        )
                    )

    summary_rows = _summarize_patient_rows(
        subexperiment_id=subexperiment_id,
        patient_rows=tuple(patient_rows),
    )
    shared_tables = _shared_registry_tables(reruns)
    shared_tables["patient_truth_store"] = tuple(truth_records)
    shared_tables["method_native_output_store"] = tuple(native_records)
    return (
        Block3SubexperimentRawRows(
            patient_metrics=tuple(patient_rows),
            condition_summaries=summary_rows,
            shared_tables=shared_tables,
        ),
        Block3SubexperimentReviewRows(
            section_rows=_build_section_review_rows(
                subexperiment_id=subexperiment_id,
                summary_rows=summary_rows,
            )
        ),
    )


def _build_3b2_rows(
    *,
    reruns: tuple[Block3GeneratorRerun, ...],
    cohort_inputs: Block3CohortInputs,
) -> tuple[Block3SubexperimentRawRows, Block3SubexperimentReviewRows]:
    """Build raw and review rows for executable `3B-2` d/e benchmark."""

    subexperiment_id = Block3SubexperimentId.DE_BENCHMARK.value
    condition_id = "open_mass_scale_grid"
    evaluation_family = get_subexperiment_spec(subexperiment_id).evaluation_family
    patient_rows: list[Block3PatientMetricRow] = []
    truth_records: list[dict[str, object]] = []
    native_records: list[dict[str, object]] = []

    for open_mass_scale in _PUBLIC_OPEN_MASS_SCALE_GRID:
        scaled_reruns = _reruns_with_open_mass_scale(reruns, open_mass_scale=open_mass_scale)
        for rerun in scaled_reruns:
            truths = [rerun.baseline_truths["relation_mid"][patient_id] for patient_id in rerun.test_patient_ids]
            train_truths = list(rerun.baseline_truths["relation_mid"].values())
            uot_calibration = _calibrated_uot_lambda_for_train(
                train_truths=train_truths,
                cost_matrix=cohort_inputs.cost_matrix,
            )
            matched_mass_budget = _matched_mass_budget_from_train(train_truths)
            truth_records.extend(
                _truth_store_records(
                    subexperiment_id=subexperiment_id,
                    condition_id=condition_id,
                    truths=truths,
                    open_mass_scale=open_mass_scale,
                )
            )
            outputs_by_method = {
                "stride_reference": _run_stride_method(
                    cohort_inputs=cohort_inputs,
                    truths=truths,
                    benchmark_mode="reference",
                ),
                "uot_baseline": _run_uot_baseline(
                    truths=truths,
                    cost_matrix=cohort_inputs.cost_matrix,
                    match_penalty=uot_calibration.selected_lambda,
                    calibration_metadata=_uot_calibration_metadata(uot_calibration),
                ),
                "partial_ot_baseline": _run_partial_ot_baseline(
                    truths=truths,
                    cost_matrix=cohort_inputs.cost_matrix,
                    matched_mass_budget=matched_mass_budget,
                ),
                "diagonal_transport_baseline": _run_diagonal_transport_baseline(truths=truths),
            }
            for method_name, outputs in outputs_by_method.items():
                native_records.extend(
                    _method_native_records(
                        subexperiment_id=subexperiment_id,
                        condition_id=condition_id,
                        rerun_id=rerun.rerun_id,
                        method_name=method_name,
                        outputs=outputs,
                        open_mass_scale=open_mass_scale,
                    )
                )
                for truth in truths:
                    patient_rows.extend(
                        _build_open_metric_rows(
                            rerun_id=rerun.rerun_id,
                            subexperiment_id=subexperiment_id,
                            condition_id=condition_id,
                            evaluation_family=evaluation_family,
                            method_name=method_name,
                            truth=truth,
                            output=outputs[truth.patient_id],
                            open_mass_scale=open_mass_scale,
                        )
                    )

    summary_rows = _summarize_patient_rows(
        subexperiment_id=subexperiment_id,
        patient_rows=tuple(patient_rows),
    )
    shared_tables = _shared_registry_tables(reruns)
    shared_tables["patient_truth_store"] = tuple(truth_records)
    shared_tables["method_native_output_store"] = tuple(native_records)
    return (
        Block3SubexperimentRawRows(
            patient_metrics=tuple(patient_rows),
            condition_summaries=summary_rows,
            shared_tables=shared_tables,
        ),
        Block3SubexperimentReviewRows(
            section_rows=_build_section_review_rows(
                subexperiment_id=subexperiment_id,
                summary_rows=summary_rows,
            )
        ),
    )

def _build_open_metric_rows(
    *,
    rerun_id: str,
    subexperiment_id: str,
    condition_id: str,
    evaluation_family: str,
    method_name: str,
    truth: Block3PatientTruth,
    output: Block3MethodOutput,
    open_mass_scale: float | None = None,
) -> list[Block3PatientMetricRow]:
    """Build the open-profile recovery metrics shared by `3C-1` and `3C-2`."""

    metric_rows: list[Block3PatientMetricRow] = []
    if output.fit_status != "ok" or output.A is None or output.d is None or output.e is None:
        for metric_name in (
            Block3MetricName.OPEN_SUPPORT_F1,
            Block3MetricName.D_MAE,
            Block3MetricName.E_MAE,
            Block3MetricName.D_MSE,
            Block3MetricName.E_MSE,
        ):
            metric_rows.append(
                _make_patient_metric_row(
                    rerun_id=rerun_id,
                    subexperiment_id=subexperiment_id,
                    condition_id=condition_id,
                    evaluation_family=evaluation_family,
                    method_name=method_name,
                    metric_name=metric_name,
                    value=None,
                    status=MetricStatus.NOT_ESTIMABLE,
                    patient_id=truth.patient_id,
                    open_mass_scale=open_mass_scale,
                )
            )
        return metric_rows
    pred_depletion = np.asarray(truth.x, dtype=float) * np.asarray(output.d, dtype=float)
    pred_emergence = np.asarray(output.e, dtype=float)
    true_depletion = np.asarray(truth.x, dtype=float) * np.asarray(truth.d, dtype=float)
    true_emergence = np.asarray(truth.e, dtype=float)
    metric_rows.append(
        _make_patient_metric_row(
            rerun_id=rerun_id,
            subexperiment_id=subexperiment_id,
            condition_id=condition_id,
            evaluation_family=evaluation_family,
            method_name=method_name,
            metric_name=Block3MetricName.OPEN_SUPPORT_F1,
            value=None
            if truth.open_mass <= _TOL
            else _open_support_f1(true_depletion, pred_depletion, true_emergence, pred_emergence),
            status=MetricStatus.NOT_APPLICABLE if truth.open_mass <= _TOL else MetricStatus.REPORTED,
            patient_id=truth.patient_id,
            open_mass_scale=open_mass_scale,
        )
    )
    metric_rows.append(
        _make_patient_metric_row(
            rerun_id=rerun_id,
            subexperiment_id=subexperiment_id,
            condition_id=condition_id,
            evaluation_family=evaluation_family,
            method_name=method_name,
            metric_name=Block3MetricName.D_MAE,
            value=float(np.mean(np.abs(true_depletion - pred_depletion))),
            status=MetricStatus.REPORTED,
            patient_id=truth.patient_id,
            open_mass_scale=open_mass_scale,
        )
    )
    metric_rows.append(
        _make_patient_metric_row(
            rerun_id=rerun_id,
            subexperiment_id=subexperiment_id,
            condition_id=condition_id,
            evaluation_family=evaluation_family,
            method_name=method_name,
            metric_name=Block3MetricName.E_MAE,
            value=float(np.mean(np.abs(true_emergence - pred_emergence))),
            status=MetricStatus.REPORTED,
            patient_id=truth.patient_id,
            open_mass_scale=open_mass_scale,
        )
    )
    metric_rows.append(
        _make_patient_metric_row(
            rerun_id=rerun_id,
            subexperiment_id=subexperiment_id,
            condition_id=condition_id,
            evaluation_family=evaluation_family,
            method_name=method_name,
            metric_name=Block3MetricName.D_MSE,
            value=float(np.mean(np.square(true_depletion - pred_depletion))),
            status=MetricStatus.REPORTED,
            patient_id=truth.patient_id,
            open_mass_scale=open_mass_scale,
        )
    )
    metric_rows.append(
        _make_patient_metric_row(
            rerun_id=rerun_id,
            subexperiment_id=subexperiment_id,
            condition_id=condition_id,
            evaluation_family=evaluation_family,
            method_name=method_name,
            metric_name=Block3MetricName.E_MSE,
            value=float(np.mean(np.square(true_emergence - pred_emergence))),
            status=MetricStatus.REPORTED,
            patient_id=truth.patient_id,
            open_mass_scale=open_mass_scale,
        )
    )
    return metric_rows


def _build_3c1_rows(
    *,
    reruns: tuple[Block3GeneratorRerun, ...],
    cohort_inputs: Block3CohortInputs,
) -> tuple[Block3SubexperimentRawRows, Block3SubexperimentReviewRows]:
    """Build raw and review rows for `3C-1` open-module ablation.

    This section consumes the opaque shared realization set from each rerun and
    does not publish any public `relation_*` condition surface.
    """
    subexperiment_id = Block3SubexperimentId.OPEN_MODULE_ABLATION.value
    condition_id = "open_module_shared_realization_set"
    evaluation_family = get_subexperiment_spec(subexperiment_id).evaluation_family
    patient_rows: list[Block3PatientMetricRow] = []
    truth_records: list[dict[str, object]] = []
    native_records: list[dict[str, object]] = []

    for rerun in reruns:
        truths = [rerun.generator_truths[patient_id] for patient_id in rerun.test_patient_ids]
        truth_records.extend(
            _truth_store_records(
                subexperiment_id=subexperiment_id,
                condition_id=condition_id,
                truths=truths,
            )
        )
        outputs_by_method = {
            "stride_reference": _run_stride_method(
                cohort_inputs=cohort_inputs,
                truths=truths,
                benchmark_mode="reference",
            ),
            "open_channel_ablation": _run_stride_method(
                cohort_inputs=cohort_inputs,
                truths=truths,
                benchmark_mode="open_channel_ablation",
            ),
        }
        for method_name, outputs in outputs_by_method.items():
            native_records.extend(
                _method_native_records(
                    subexperiment_id=subexperiment_id,
                    condition_id=condition_id,
                    rerun_id=rerun.rerun_id,
                    method_name=method_name,
                    outputs=outputs,
                )
            )
            for truth in truths:
                patient_rows.extend(
                    _build_open_metric_rows(
                        rerun_id=rerun.rerun_id,
                        subexperiment_id=subexperiment_id,
                        condition_id=condition_id,
                        evaluation_family=evaluation_family,
                        method_name=method_name,
                        truth=truth,
                        output=outputs[truth.patient_id],
                    )
                )

    summary_rows = _summarize_patient_rows(
        subexperiment_id=subexperiment_id,
        patient_rows=tuple(patient_rows),
    )
    shared_tables = _shared_registry_tables(reruns)
    shared_tables["patient_truth_store"] = tuple(truth_records)
    shared_tables["method_native_output_store"] = tuple(native_records)
    return (
        Block3SubexperimentRawRows(
            patient_metrics=tuple(patient_rows),
            condition_summaries=summary_rows,
            shared_tables=shared_tables,
        ),
        Block3SubexperimentReviewRows(
            section_rows=_build_section_review_rows(
                subexperiment_id=subexperiment_id,
                summary_rows=summary_rows,
            )
        ),
    )


def _build_3c2_rows(
    *,
    reruns: tuple[Block3GeneratorRerun, ...],
    cohort_inputs: Block3CohortInputs,
) -> tuple[Block3SubexperimentRawRows, Block3SubexperimentReviewRows]:
    """Build raw and review rows for `3C-2` cohort-module ablation.

    `3C-2` reuses the same opaque realization set as `3C-1` while restoring
    relation-operator recovery metrics on `A_p`.
    """
    subexperiment_id = Block3SubexperimentId.COHORT_MODULE_ABLATION.value
    condition_id = "cohort_module_shared_realization_set"
    evaluation_family = get_subexperiment_spec(subexperiment_id).evaluation_family
    patient_rows: list[Block3PatientMetricRow] = []
    truth_records: list[dict[str, object]] = []
    native_records: list[dict[str, object]] = []

    for rerun in reruns:
        truths = [rerun.generator_truths[patient_id] for patient_id in rerun.test_patient_ids]
        truth_records.extend(
            _truth_store_records(
                subexperiment_id=subexperiment_id,
                condition_id=condition_id,
                truths=truths,
            )
        )
        outputs_by_method = {
            "stride_reference": _run_stride_method(
                cohort_inputs=cohort_inputs,
                truths=truths,
                benchmark_mode="reference",
            ),
            "cohort_ablation": _run_stride_method(
                cohort_inputs=cohort_inputs,
                truths=truths,
                benchmark_mode="cohort_ablation",
            ),
        }
        for method_name, outputs in outputs_by_method.items():
            native_records.extend(
                _method_native_records(
                    subexperiment_id=subexperiment_id,
                    condition_id=condition_id,
                    rerun_id=rerun.rerun_id,
                    method_name=method_name,
                    outputs=outputs,
                )
            )
            for truth in truths:
                output = outputs[truth.patient_id]
                if output.fit_status != "ok" or output.A is None:
                    for metric_name in (Block3MetricName.A_MAE_ACTIVE, Block3MetricName.A_MSE_ACTIVE):
                        patient_rows.append(
                            _make_patient_metric_row(
                                rerun_id=rerun.rerun_id,
                                subexperiment_id=subexperiment_id,
                                condition_id=condition_id,
                                evaluation_family=evaluation_family,
                                method_name=method_name,
                                metric_name=metric_name,
                                value=None,
                                status=MetricStatus.NOT_ESTIMABLE,
                                patient_id=truth.patient_id,
                            )
                        )
                else:
                    patient_rows.append(
                        _make_patient_metric_row(
                            rerun_id=rerun.rerun_id,
                            subexperiment_id=subexperiment_id,
                            condition_id=condition_id,
                            evaluation_family=evaluation_family,
                            method_name=method_name,
                            metric_name=Block3MetricName.A_MAE_ACTIVE,
                            value=_matrix_mae_active(truth.x, truth.A, output.A),
                            status=MetricStatus.REPORTED,
                            patient_id=truth.patient_id,
                        )
                    )
                    patient_rows.append(
                        _make_patient_metric_row(
                            rerun_id=rerun.rerun_id,
                            subexperiment_id=subexperiment_id,
                            condition_id=condition_id,
                            evaluation_family=evaluation_family,
                            method_name=method_name,
                            metric_name=Block3MetricName.A_MSE_ACTIVE,
                            value=_matrix_mse_active(truth.x, truth.A, output.A),
                            status=MetricStatus.REPORTED,
                            patient_id=truth.patient_id,
                        )
                    )
                patient_rows.extend(
                    _build_open_metric_rows(
                        rerun_id=rerun.rerun_id,
                        subexperiment_id=subexperiment_id,
                        condition_id=condition_id,
                        evaluation_family=evaluation_family,
                        method_name=method_name,
                        truth=truth,
                        output=output,
                    )
                )

    summary_rows = _summarize_patient_rows(
        subexperiment_id=subexperiment_id,
        patient_rows=tuple(patient_rows),
    )
    shared_tables = _shared_registry_tables(reruns)
    shared_tables["patient_truth_store"] = tuple(truth_records)
    shared_tables["method_native_output_store"] = tuple(native_records)
    return (
        Block3SubexperimentRawRows(
            patient_metrics=tuple(patient_rows),
            condition_summaries=summary_rows,
            shared_tables=shared_tables,
        ),
        Block3SubexperimentReviewRows(
            section_rows=_build_section_review_rows(
                subexperiment_id=subexperiment_id,
                summary_rows=summary_rows,
            )
        ),
    )


def _write_internal_block3_outputs(
    *,
    output_dir: str | Path,
    plan: Block3ExecutionPlan,
    subexperiment_id: str,
    raw_rows: Block3SubexperimentRawRows,
    review_rows: Block3SubexperimentReviewRows,
) -> Block3InternalExecutionResult:
    """Write one subexperiment's raw and review outputs and return their paths."""

    from .bundle import build_block3_bundle_layout, write_block3_subexperiment_raw_bundle
    from .review import write_block3_subexperiment_review_surface

    raw_bundle = write_block3_subexperiment_raw_bundle(
        output_dir=output_dir,
        plan=plan,
        subexperiment_id=subexperiment_id,
        raw_rows=raw_rows,
    )
    review_surface = write_block3_subexperiment_review_surface(
        output_dir=output_dir,
        plan=plan,
        bundle_layout=build_block3_bundle_layout(plan, subexperiment_ids=(subexperiment_id,)),
        subexperiment_id=subexperiment_id,
        review_rows=review_rows,
    )
    return Block3InternalExecutionResult(
        subexperiment_id=subexperiment_id,
        raw_manifest_path=raw_bundle.manifest_path,
        raw_index_path=raw_bundle.raw_index_path,
        review_manifest_path=review_surface.manifest_path,
        review_index_path=review_surface.review_index_path,
        raw_artifact_paths=raw_bundle.artifact_paths,
        review_artifact_paths=review_surface.artifact_paths,
    )


def _transport_budget_components(
    *,
    x: np.ndarray,
    A: np.ndarray,
    d: np.ndarray | None,
    e: np.ndarray | None,
) -> dict[str, float]:
    """Compute one truth- or method-budget breakdown on the truth `x` surface."""

    transport = np.asarray(x, dtype=float)[:, None] * np.asarray(A, dtype=float)
    diag = float(np.trace(transport))
    total = float(np.sum(transport, dtype=float))
    offdiag = max(0.0, total - diag)
    depletion = (
        float(np.sum(np.asarray(x, dtype=float) * np.asarray(d, dtype=float), dtype=float))
        if d is not None
        else 0.0
    )
    emergence = float(np.sum(np.asarray(e, dtype=float), dtype=float)) if e is not None else 0.0
    matched = diag + offdiag
    return {
        "diag_budget": diag,
        "offdiag_budget": offdiag,
        "open_budget": emergence,
        "matched_budget": matched,
        "d_budget": depletion,
        "e_budget": emergence,
        "offdiag_over_open": (offdiag / emergence) if emergence > _TOL else np.nan,
        "offdiag_over_matched": (offdiag / matched) if matched > _TOL else 0.0,
    }


def _row_relation_distribution(row: np.ndarray, source_index: int) -> np.ndarray:
    """Map one `A` row to a proper distribution for row-level relation diagnostics."""

    distribution = np.asarray(row, dtype=float).reshape(-1)
    total = float(np.sum(distribution, dtype=float))
    if total <= _TOL:
        fallback = np.zeros_like(distribution)
        fallback[source_index] = 1.0
        return fallback
    return distribution / total


def _distribution_w1(cost_matrix: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    """Solve the balanced W1 distance between two same-axis distributions."""

    from scipy.optimize import linprog

    left_distribution = _normalize_probabilities(left)
    right_distribution = _normalize_probabilities(right)
    n_states = left_distribution.shape[0]
    c = np.asarray(cost_matrix, dtype=float).reshape(-1)
    row_constraints = []
    col_constraints = []
    for row_index in range(n_states):
        row = np.zeros(n_states * n_states, dtype=float)
        row[row_index * n_states : (row_index + 1) * n_states] = 1.0
        row_constraints.append(row)
    for col_index in range(n_states):
        col = np.zeros(n_states * n_states, dtype=float)
        col[col_index::n_states] = 1.0
        col_constraints.append(col)
    A_eq = np.vstack(row_constraints + col_constraints[:-1])
    b_eq = np.concatenate([left_distribution, right_distribution[:-1]]).astype(float, copy=False)
    result = linprog(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0.0, None)] * (n_states * n_states),
        method="highs",
    )
    if not result.success:
        raise ContractError(f"Block 3 row-level W1 solver failed: {result.message}")
    return float(np.dot(c, np.asarray(result.x, dtype=float)))


def _active_relation_rows(x: np.ndarray, A_true: np.ndarray) -> list[int]:
    """Return active truth rows that retain matched mass for relation diagnostics."""

    return [
        int(source_index)
        for source_index in range(np.asarray(A_true, dtype=float).shape[0])
        if float(x[source_index]) > _TOL and float(np.sum(A_true[source_index], dtype=float)) > _TOL
    ]


def _candidate_row_tv_active(x: np.ndarray, A_true: np.ndarray, A_hat: np.ndarray) -> float:
    """Compute the active-row mean TV over normalized relation rows."""

    active_rows = _active_relation_rows(np.asarray(x, dtype=float), np.asarray(A_true, dtype=float))
    if not active_rows:
        return 0.0
    row_scores = []
    for source_index in active_rows:
        truth_distribution = _row_relation_distribution(A_true[source_index], source_index)
        pred_distribution = _row_relation_distribution(A_hat[source_index], source_index)
        row_scores.append(0.5 * float(np.sum(np.abs(truth_distribution - pred_distribution), dtype=float)))
    return float(np.mean(row_scores))


def _candidate_row_w1_active(
    x: np.ndarray,
    A_true: np.ndarray,
    A_hat: np.ndarray,
    cost_matrix: np.ndarray,
) -> float:
    """Compute the active-row mean W1 over normalized relation rows."""

    active_rows = _active_relation_rows(np.asarray(x, dtype=float), np.asarray(A_true, dtype=float))
    if not active_rows:
        return 0.0
    row_scores = []
    for source_index in active_rows:
        truth_distribution = _row_relation_distribution(A_true[source_index], source_index)
        pred_distribution = _row_relation_distribution(A_hat[source_index], source_index)
        row_scores.append(
            _distribution_w1(
                np.asarray(cost_matrix, dtype=float),
                truth_distribution,
                pred_distribution,
            )
        )
    return float(np.mean(row_scores))


def _candidate_row_target_recall_at_k(
    x: np.ndarray,
    A_true: np.ndarray,
    A_hat: np.ndarray,
) -> float | None:
    """Compute row-level off-diagonal target recall at truth-derived `k`."""

    recall_values: list[float] = []
    for source_index in _active_relation_rows(np.asarray(x, dtype=float), np.asarray(A_true, dtype=float)):
        true_targets = {
            int(target_index)
            for target_index, value in enumerate(np.asarray(A_true[source_index], dtype=float))
            if target_index != source_index and float(value) > _TOL
        }
        if not true_targets:
            continue
        k = len(true_targets)
        predicted_rank = sorted(
            [
                int(target_index)
                for target_index in range(A_hat.shape[1])
                if target_index != source_index
            ],
            key=lambda target_index: (
                -float(A_hat[source_index, target_index]),
                int(target_index),
            ),
        )[:k]
        recall_values.append(float(len(true_targets.intersection(predicted_rank)) / len(true_targets)))
    if not recall_values:
        return None
    return float(np.mean(recall_values))


def _decode_truth_store(
    truth_records: tuple[dict[str, object], ...],
) -> dict[tuple[str, str, str], dict[str, np.ndarray | float]]:
    """Decode raw truth-store records into array-carrying lookup entries."""

    truth_map: dict[tuple[str, str, str], dict[str, np.ndarray | float]] = {}
    for record in truth_records:
        truth_map[
            (
                str(record["rerun_id"]),
                str(record["condition_id"]),
                str(record["patient_id"]),
            )
        ] = {
            "x": np.asarray(json.loads(str(record["x_json"])), dtype=float),
            "A": np.asarray(json.loads(str(record["A_json"])), dtype=float),
            "d": np.asarray(json.loads(str(record["d_json"])), dtype=float),
            "e": np.asarray(json.loads(str(record["e_json"])), dtype=float),
            "open_mass": float(record["open_mass"]),
        }
    return truth_map


def _summarize_rerun_histories(histories: dict[str, list[float]]) -> tuple[float | None, float | None, float | None]:
    """Collapse patient histories to rerun means and a bootstrap interval."""

    rerun_means = [
        float(np.mean(np.asarray(values, dtype=float)))
        for rerun_id, values in sorted(histories.items())
        if values
    ]
    if not rerun_means:
        return None, None, None
    mean_value = float(np.mean(np.asarray(rerun_means, dtype=float)))
    ci_lower, ci_upper = _bootstrap_ci(rerun_means)
    return mean_value, ci_lower, ci_upper


def _collect_truth_budget_summary_rows(
    *,
    support_mode: str,
    open_mass_scale: float,
    reruns: tuple[Block3GeneratorRerun, ...],
) -> list[dict[str, object]]:
    """Aggregate truth budgets for one diagnostic arm across reruns and conditions."""

    condition_histories: dict[str, dict[str, dict[str, list[float]]]] = {}
    for rerun in reruns:
        for condition_id in _RELATION_SPECS:
            histories = condition_histories.setdefault(
                condition_id,
                {
                    "diag_budget": {},
                    "offdiag_budget": {},
                    "open_budget": {},
                    "offdiag_over_open": {},
                    "offdiag_over_matched": {},
                },
            )
            for patient_id in rerun.test_patient_ids:
                truth = rerun.baseline_truths[condition_id][patient_id]
                budgets = _transport_budget_components(
                    x=truth.x,
                    A=truth.A,
                    d=truth.d,
                    e=truth.e,
                )
                for component_name in histories:
                    histories[component_name].setdefault(rerun.rerun_id, []).append(
                        float(budgets[component_name])
                    )
    rows: list[dict[str, object]] = []
    for condition_id, histories in sorted(condition_histories.items()):
        diag_mean, diag_ci_lower, diag_ci_upper = _summarize_rerun_histories(histories["diag_budget"])
        offdiag_mean, offdiag_ci_lower, offdiag_ci_upper = _summarize_rerun_histories(histories["offdiag_budget"])
        open_mean, open_ci_lower, open_ci_upper = _summarize_rerun_histories(histories["open_budget"])
        offdiag_open_mean, _, _ = _summarize_rerun_histories(histories["offdiag_over_open"])
        offdiag_matched_mean, _, _ = _summarize_rerun_histories(histories["offdiag_over_matched"])
        rows.append(
            {
                "support_mode": support_mode,
                "open_mass_scale": open_mass_scale,
                "condition_id": condition_id,
                "summary_level": "rerun_mean_bootstrap95",
                "diag_truth_budget_mean": diag_mean,
                "diag_truth_budget_ci_lower": diag_ci_lower,
                "diag_truth_budget_ci_upper": diag_ci_upper,
                "offdiag_truth_budget_mean": offdiag_mean,
                "offdiag_truth_budget_ci_lower": offdiag_ci_lower,
                "offdiag_truth_budget_ci_upper": offdiag_ci_upper,
                "open_truth_budget_mean": open_mean,
                "open_truth_budget_ci_lower": open_ci_lower,
                "open_truth_budget_ci_upper": open_ci_upper,
                "offdiag_over_open_mean": offdiag_open_mean,
                "offdiag_over_matched_mean": offdiag_matched_mean,
            }
        )
    return rows


def _collect_method_budget_summary_rows(
    *,
    support_mode: str,
    open_mass_scale: float,
    raw_rows: Block3SubexperimentRawRows,
) -> list[dict[str, object]]:
    """Aggregate truth-weighted method budgets for one diagnostic arm."""

    truth_map = _decode_truth_store(raw_rows.shared_tables["patient_truth_store"])
    native_records = raw_rows.shared_tables["method_native_output_store"]
    grouped: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = {}
    status_map: dict[tuple[str, str], str] = {}

    for record in native_records:
        key = (str(record["condition_id"]), str(record["method_name"]))
        grouped.setdefault(
            key,
            {
                "diag_budget": {},
                "offdiag_budget": {},
                "d_budget": {},
                "e_budget": {},
            },
        )
        fit_status = str(record["fit_status"])
        if fit_status != "ok":
            status_map[key] = "not_estimable"
            continue
        truth = truth_map[
            (
                str(record["rerun_id"]),
                str(record["condition_id"]),
                str(record["patient_id"]),
            )
        ]
        A = np.asarray(json.loads(str(record["A_json"])), dtype=float)
        d = np.asarray(json.loads(str(record["d_json"])), dtype=float)
        e = np.asarray(json.loads(str(record["e_json"])), dtype=float)
        budgets = _transport_budget_components(
            x=np.asarray(truth["x"], dtype=float),
            A=A,
            d=d,
            e=e,
        )
        for component_name in grouped[key]:
            grouped[key][component_name].setdefault(str(record["rerun_id"]), []).append(
                float(budgets[component_name])
            )
        status_map.setdefault(key, "reported")

    rows: list[dict[str, object]] = []
    for (condition_id, method_name), histories in sorted(grouped.items()):
        diag_mean, diag_ci_lower, diag_ci_upper = _summarize_rerun_histories(histories["diag_budget"])
        offdiag_mean, offdiag_ci_lower, offdiag_ci_upper = _summarize_rerun_histories(histories["offdiag_budget"])
        d_mean, d_ci_lower, d_ci_upper = _summarize_rerun_histories(histories["d_budget"])
        e_mean, e_ci_lower, e_ci_upper = _summarize_rerun_histories(histories["e_budget"])
        rows.append(
            {
                "support_mode": support_mode,
                "open_mass_scale": open_mass_scale,
                "condition_id": condition_id,
                "method_name": method_name,
                "summary_level": "rerun_mean_bootstrap95",
                "summary_status": status_map[(condition_id, method_name)],
                "diag_usage_mean": diag_mean,
                "diag_usage_ci_lower": diag_ci_lower,
                "diag_usage_ci_upper": diag_ci_upper,
                "offdiag_usage_mean": offdiag_mean,
                "offdiag_usage_ci_lower": offdiag_ci_lower,
                "offdiag_usage_ci_upper": offdiag_ci_upper,
                "d_usage_mean": d_mean,
                "d_usage_ci_lower": d_ci_lower,
                "d_usage_ci_upper": d_ci_upper,
                "e_usage_mean": e_mean,
                "e_usage_ci_lower": e_ci_lower,
                "e_usage_ci_upper": e_ci_upper,
            }
        )
    return rows


def _collect_candidate_metric_summary_rows(
    *,
    support_mode: str,
    open_mass_scale: float,
    raw_rows: Block3SubexperimentRawRows,
    cost_matrix: np.ndarray,
) -> list[dict[str, object]]:
    """Aggregate sidecar candidate relation metrics for one diagnostic arm."""

    truth_map = _decode_truth_store(raw_rows.shared_tables["patient_truth_store"])
    native_records = raw_rows.shared_tables["method_native_output_store"]
    histories: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    status_histories: dict[tuple[str, str, str], list[str]] = {}

    for record in native_records:
        fit_status = str(record["fit_status"])
        rerun_id = str(record["rerun_id"])
        condition_id = str(record["condition_id"])
        method_name = str(record["method_name"])
        truth = truth_map[
            (
                rerun_id,
                condition_id,
                str(record["patient_id"]),
            )
        ]
        metric_items: dict[str, tuple[float | None, str]] = {}
        if fit_status != "ok":
            metric_items = {
                "A_row_TV_active": (None, MetricStatus.NOT_ESTIMABLE.value),
                "A_row_W1_active": (None, MetricStatus.NOT_ESTIMABLE.value),
                "A_row_target_recall_at_k": (None, MetricStatus.NOT_ESTIMABLE.value),
            }
        else:
            A_hat = np.asarray(json.loads(str(record["A_json"])), dtype=float)
            x = np.asarray(truth["x"], dtype=float)
            A_true = np.asarray(truth["A"], dtype=float)
            metric_items = {
                "A_row_TV_active": (
                    _candidate_row_tv_active(x, A_true, A_hat),
                    MetricStatus.REPORTED.value,
                ),
                "A_row_W1_active": (
                    _candidate_row_w1_active(x, A_true, A_hat, np.asarray(cost_matrix, dtype=float)),
                    MetricStatus.REPORTED.value,
                ),
            }
            row_recall = _candidate_row_target_recall_at_k(x, A_true, A_hat)
            metric_items["A_row_target_recall_at_k"] = (
                row_recall,
                MetricStatus.NOT_APPLICABLE.value if row_recall is None else MetricStatus.REPORTED.value,
            )
        for metric_name, (value, status) in metric_items.items():
            key = (condition_id, method_name, metric_name)
            histories.setdefault(key, {})
            status_histories.setdefault(key, []).append(status)
            if value is not None:
                histories[key].setdefault(rerun_id, []).append(float(value))

    rows: list[dict[str, object]] = []
    for (condition_id, method_name, metric_name), rerun_histories in sorted(histories.items()):
        statuses = set(status_histories[(condition_id, method_name, metric_name)])
        if MetricStatus.NOT_ESTIMABLE.value in statuses:
            summary_status = MetricStatus.NOT_ESTIMABLE.value
            mean_value = None
            ci_lower = None
            ci_upper = None
        elif statuses == {MetricStatus.NOT_APPLICABLE.value}:
            summary_status = MetricStatus.NOT_APPLICABLE.value
            mean_value = None
            ci_lower = None
            ci_upper = None
        else:
            summary_status = MetricStatus.REPORTED.value
            mean_value, ci_lower, ci_upper = _summarize_rerun_histories(rerun_histories)
        rows.append(
            {
                "support_mode": support_mode,
                "open_mass_scale": open_mass_scale,
                "condition_id": condition_id,
                "method_name": method_name,
                "metric_name": metric_name,
                "summary_level": "rerun_mean_bootstrap95",
                "metric_status": summary_status,
                "mean_value": mean_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )
    return rows


def _metric_direction(metric_name: str) -> Literal["lower_better", "higher_better"]:
    """Return the comparison direction for one formal or candidate metric."""

    if metric_name in {"target_recall_at_k", "A_row_target_recall_at_k"}:
        return "higher_better"
    return "lower_better"


def _method_rank_signature(
    *,
    metric_rows: list[dict[str, object]],
    methods: tuple[str, ...],
    metric_names: tuple[str, ...],
) -> str:
    """Collapse one metric surface to an aggregated method-ranking signature."""

    average_ranks: dict[str, list[float]] = {method_name: [] for method_name in methods}
    for metric_name in metric_names:
        ranked_rows = [
            row
            for row in metric_rows
            if str(row["metric_name"]) == metric_name and str(row["metric_status"]) == MetricStatus.REPORTED.value
        ]
        ordered = sorted(
            ranked_rows,
            key=lambda row: (
                -float(row["mean_value"])
                if _metric_direction(metric_name) == "higher_better"
                else float(row["mean_value"]),
                str(row["method_name"]),
            ),
        )
        rank_map = {str(row["method_name"]): rank for rank, row in enumerate(ordered, start=1)}
        for method_name in methods:
            average_ranks[method_name].append(float(rank_map.get(method_name, len(methods) + 1)))
    ordered_methods = tuple(
        sorted(
            methods,
            key=lambda method_name: (
                float(np.mean(np.asarray(average_ranks[method_name], dtype=float))),
                str(method_name),
            ),
        )
    )
    return " > ".join(ordered_methods)


def _stride_vs_ot_candidate_wins(
    *,
    candidate_rows: list[dict[str, object]],
) -> int:
    """Count how many candidate relation metrics STRIDE wins versus exact OT."""

    wins = 0
    for metric_name in ("A_row_TV_active", "A_row_W1_active", "A_row_target_recall_at_k"):
        stride_row = next(
            (
                row
                for row in candidate_rows
                if str(row["method_name"]) == "stride_reference" and str(row["metric_name"]) == metric_name
            ),
            None,
        )
        ot_row = next(
            (
                row
                for row in candidate_rows
                if str(row["method_name"]) == "balanced_ot_baseline" and str(row["metric_name"]) == metric_name
            ),
            None,
        )
        if stride_row is None or ot_row is None:
            continue
        if (
            str(stride_row["metric_status"]) != MetricStatus.REPORTED.value
            or str(ot_row["metric_status"]) != MetricStatus.REPORTED.value
        ):
            continue
        stride_value = float(stride_row["mean_value"])
        ot_value = float(ot_row["mean_value"])
        if _metric_direction(metric_name) == "higher_better":
            if stride_value > ot_value + _TOL:
                wins += 1
        elif stride_value + _TOL < ot_value:
            wins += 1
    return wins


def _classify_diagnostic_verdict(
    *,
    condition_rows: list[dict[str, object]],
    method_budget_rows: list[dict[str, object]],
) -> tuple[str, dict[str, object]]:
    """Classify the matrix-wide internal diagnostic verdict from the sidecar summaries."""

    focus_conditions = {"relation_mid", "relation_strong"}

    def _relation_win_count(*, support_mode: str, open_mass_scale: float) -> int:
        return sum(
            1
            for row in condition_rows
            if (
                str(row["support_mode"]) == support_mode
                and float(row["open_mass_scale"]) == open_mass_scale
                and str(row["condition_id"]) in focus_conditions
                and bool(row["relation_win"])
            )
        )

    open_effect = max(
        _relation_win_count(support_mode=_SUPPORT_MODE_LEGACY_NEAREST_C, open_mass_scale=0.5)
        - _relation_win_count(support_mode=_SUPPORT_MODE_LEGACY_NEAREST_C, open_mass_scale=1.0),
        _relation_win_count(support_mode=_SUPPORT_MODE_LEGACY_NEAREST_C, open_mass_scale=0.25)
        - _relation_win_count(support_mode=_SUPPORT_MODE_LEGACY_NEAREST_C, open_mass_scale=1.0),
        _relation_win_count(support_mode=_SUPPORT_MODE_RELATION_MOTIF_PROBE, open_mass_scale=0.5)
        - _relation_win_count(support_mode=_SUPPORT_MODE_RELATION_MOTIF_PROBE, open_mass_scale=1.0),
        _relation_win_count(support_mode=_SUPPORT_MODE_RELATION_MOTIF_PROBE, open_mass_scale=0.25)
        - _relation_win_count(support_mode=_SUPPORT_MODE_RELATION_MOTIF_PROBE, open_mass_scale=1.0),
    )
    support_effect = max(
        _relation_win_count(support_mode=_SUPPORT_MODE_RELATION_MOTIF_PROBE, open_mass_scale=1.0)
        - _relation_win_count(support_mode=_SUPPORT_MODE_LEGACY_NEAREST_C, open_mass_scale=1.0),
        _relation_win_count(support_mode=_SUPPORT_MODE_RELATION_MOTIF_PROBE, open_mass_scale=0.5)
        - _relation_win_count(support_mode=_SUPPORT_MODE_LEGACY_NEAREST_C, open_mass_scale=0.5),
        _relation_win_count(support_mode=_SUPPORT_MODE_RELATION_MOTIF_PROBE, open_mass_scale=0.25)
        - _relation_win_count(support_mode=_SUPPORT_MODE_LEGACY_NEAREST_C, open_mass_scale=0.25),
    )
    formal_candidate_split_count = sum(
        1
        for row in condition_rows
        if not bool(row["formal_rank_consistent_with_candidate"])
    )
    stride_retains_nonzero_open = any(
        str(row["method_name"]) == "stride_reference"
        and (
            float(row["d_usage_mean"]) > _TOL
            or float(row["e_usage_mean"]) > _TOL
        )
        for row in method_budget_rows
        if row["d_usage_mean"] is not None and row["e_usage_mean"] is not None
    )

    if open_effect >= 1 and open_effect > support_effect:
        return "generator_open_dominance", {
            "open_effect_score": open_effect,
            "support_effect_score": support_effect,
            "formal_candidate_split_count": formal_candidate_split_count,
        }
    if support_effect >= 1 and support_effect > open_effect:
        return "generator_support_isomorphism", {
            "open_effect_score": open_effect,
            "support_effect_score": support_effect,
            "formal_candidate_split_count": formal_candidate_split_count,
        }
    if formal_candidate_split_count >= 2 and stride_retains_nonzero_open:
        return "input_surface_mismatch", {
            "open_effect_score": open_effect,
            "support_effect_score": support_effect,
            "formal_candidate_split_count": formal_candidate_split_count,
        }
    return "implementation_defect", {
        "open_effect_score": open_effect,
        "support_effect_score": support_effect,
        "formal_candidate_split_count": formal_candidate_split_count,
    }


def _collect_rank_concordance_rows(
    *,
    formal_summary_rows: list[dict[str, object]],
    candidate_summary_rows: list[dict[str, object]],
    method_budget_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Build condition-level rank-concordance rows plus one matrix verdict row."""

    methods = (
        "stride_reference",
        "balanced_ot_baseline",
        "uot_baseline",
        "partial_ot_baseline",
        "diagonal_transport_baseline",
    )
    condition_keys = sorted(
        {
            (str(row["support_mode"]), float(row["open_mass_scale"]), str(row["condition_id"]))
            for row in formal_summary_rows
        }
    )
    condition_rows: list[dict[str, object]] = []
    for support_mode, open_mass_scale, condition_id in condition_keys:
        formal_rows = [
            row
            for row in formal_summary_rows
            if (
                str(row["support_mode"]) == support_mode
                and float(row["open_mass_scale"]) == open_mass_scale
                and str(row["condition_id"]) == condition_id
            )
        ]
        candidate_rows = [
            row
            for row in candidate_summary_rows
            if (
                str(row["support_mode"]) == support_mode
                and float(row["open_mass_scale"]) == open_mass_scale
                and str(row["condition_id"]) == condition_id
            )
        ]
        formal_signature = _method_rank_signature(
            metric_rows=formal_rows,
            methods=methods,
            metric_names=("A_MAE_active", "A_MSE_active", "target_recall_at_k"),
        )
        candidate_signature = _method_rank_signature(
            metric_rows=candidate_rows,
            methods=methods,
            metric_names=("A_row_TV_active", "A_row_W1_active", "A_row_target_recall_at_k"),
        )
        condition_rows.append(
            {
                "row_type": "condition",
                "support_mode": support_mode,
                "open_mass_scale": open_mass_scale,
                "condition_id": condition_id,
                "formal_rank_signature": formal_signature,
                "candidate_rank_signature": candidate_signature,
                "formal_rank_consistent_with_candidate": formal_signature == candidate_signature,
                "stride_relation_metric_wins_vs_balanced_ot": _stride_vs_ot_candidate_wins(
                    candidate_rows=candidate_rows,
                ),
                "relation_win": (
                    _stride_vs_ot_candidate_wins(candidate_rows=candidate_rows) >= 2
                ),
                "verdict_label": "",
                "open_effect_score": None,
                "support_effect_score": None,
                "formal_candidate_split_count": None,
            }
        )
    verdict_label, verdict_extras = _classify_diagnostic_verdict(
        condition_rows=condition_rows,
        method_budget_rows=method_budget_rows,
    )
    condition_rows.append(
        {
            "row_type": "matrix_verdict",
            "support_mode": "all_support_modes",
            "open_mass_scale": np.nan,
            "condition_id": "all_conditions",
            "formal_rank_signature": "",
            "candidate_rank_signature": "",
            "formal_rank_consistent_with_candidate": False,
            "stride_relation_metric_wins_vs_balanced_ot": np.nan,
            "relation_win": False,
            "verdict_label": verdict_label,
            "open_effect_score": verdict_extras["open_effect_score"],
            "support_effect_score": verdict_extras["support_effect_score"],
            "formal_candidate_split_count": verdict_extras["formal_candidate_split_count"],
        }
    )
    return condition_rows


def _formal_condition_summary_records(
    *,
    support_mode: str,
    open_mass_scale: float,
    raw_rows: Block3SubexperimentRawRows,
) -> list[dict[str, object]]:
    """Serialize formal `3B` condition summaries into flat diagnostic records."""

    records: list[dict[str, object]] = []
    for row in raw_rows.condition_summaries:
        records.append(
            {
                "support_mode": support_mode,
                "open_mass_scale": open_mass_scale,
                "condition_id": row.condition_id,
                "method_name": row.method_name.value,
                "metric_name": row.metric_value.metric_name.value,
                "metric_status": row.metric_value.status.value,
                "mean_value": row.mean_value,
                "ci_lower": row.ci_lower,
                "ci_upper": row.ci_upper,
            }
        )
    return records


def _write_dataframe(path: Path, rows: list[dict[str, object]]) -> Path:
    """Write a flat row collection to CSV and return the written path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_csv(path, index=False)
    return path


def _diagnostic_arm_id(*, support_mode: str, open_mass_scale: float) -> str:
    """Build one filesystem-safe internal 3B diagnostic arm identifier."""

    return f"{support_mode}__open_{str(open_mass_scale).replace('.', 'p')}"


def execute_internal_block3_3b_diagnostic_matrix(
    *,
    block2_manifest_path: str | Path,
    output_dir: str | Path,
) -> Block3DiagnosticMatrixResult:
    """Execute the internal-only `3B` diagnostic matrix over support/open arms."""

    inputs = resolve_upstream_inputs(
        block2_manifest_path=block2_manifest_path,
        output_dir=output_dir,
    )
    plan = build_phase2_execution_plan(inputs)
    cohort_inputs = _load_block3_cohort_inputs(inputs)
    resolved_output_dir = _resolve_path(output_dir)

    arm_results: list[Block3DiagnosticArmResult] = []
    truth_budget_rows: list[dict[str, object]] = []
    method_budget_rows: list[dict[str, object]] = []
    candidate_summary_rows: list[dict[str, object]] = []
    formal_summary_rows: list[dict[str, object]] = []

    for support_mode in _ALLOWED_SUPPORT_MODES:
        for open_mass_scale in _BLOCK3B_3B1_OPEN_MASS_SENSITIVITY_GRID:
            reruns = _build_generator_reruns(
                cohort_inputs=cohort_inputs,
                support_mode=support_mode,
                open_mass_scale=open_mass_scale,
            )
            raw_rows, review_rows = _build_3b1_rows(
                reruns=reruns,
                cohort_inputs=cohort_inputs,
            )
            arm_id = _diagnostic_arm_id(
                support_mode=support_mode,
                open_mass_scale=open_mass_scale,
            )
            arm_output_dir = resolved_output_dir / arm_id
            execution_result = _write_internal_block3_outputs(
                output_dir=arm_output_dir,
                plan=plan,
                subexperiment_id=Block3SubexperimentId.A_BENCHMARK.value,
                raw_rows=raw_rows,
                review_rows=review_rows,
            )
            arm_results.append(
                Block3DiagnosticArmResult(
                    arm_id=arm_id,
                    support_mode=support_mode,
                    open_mass_scale=open_mass_scale,
                    output_dir=arm_output_dir,
                    execution_result=execution_result,
                )
            )
            truth_budget_rows.extend(
                _collect_truth_budget_summary_rows(
                    support_mode=support_mode,
                    open_mass_scale=open_mass_scale,
                    reruns=reruns,
                )
            )
            method_budget_rows.extend(
                _collect_method_budget_summary_rows(
                    support_mode=support_mode,
                    open_mass_scale=open_mass_scale,
                    raw_rows=raw_rows,
                )
            )
            candidate_summary_rows.extend(
                _collect_candidate_metric_summary_rows(
                    support_mode=support_mode,
                    open_mass_scale=open_mass_scale,
                    raw_rows=raw_rows,
                    cost_matrix=cohort_inputs.cost_matrix,
                )
            )
            formal_summary_rows.extend(
                _formal_condition_summary_records(
                    support_mode=support_mode,
                    open_mass_scale=open_mass_scale,
                    raw_rows=raw_rows,
                )
            )

    rank_concordance_rows = _collect_rank_concordance_rows(
        formal_summary_rows=formal_summary_rows,
        candidate_summary_rows=candidate_summary_rows,
        method_budget_rows=method_budget_rows,
    )
    summary_artifact_paths = {
        "3b_truth_budget_summary": _write_dataframe(
            resolved_output_dir / "3b_truth_budget_summary.csv",
            truth_budget_rows,
        ),
        "3b_method_budget_summary": _write_dataframe(
            resolved_output_dir / "3b_method_budget_summary.csv",
            method_budget_rows,
        ),
        "3b_rank_concordance_summary": _write_dataframe(
            resolved_output_dir / "3b_rank_concordance_summary.csv",
            rank_concordance_rows,
        ),
    }
    return Block3DiagnosticMatrixResult(
        output_dir=resolved_output_dir,
        arm_results=tuple(arm_results),
        summary_artifact_paths=summary_artifact_paths,
    )


def execute_internal_block3_subexperiment(
    *,
    block2_manifest_path: str | Path,
    output_dir: str | Path,
    subexperiment_id: str,
    support_mode: str = _SUPPORT_MODE_LEGACY_NEAREST_C,
    open_mass_scale: float = 1.0,
) -> Block3InternalExecutionResult:
    """Execute one internal Block 3 Phase 3 subexperiment end to end.

    Purpose:
        Run the real internal generator, method execution, and artifact-writing
        path for one executable Block 3 subexperiment while preserving the
        non-authority Phase 3 boundary.

    Inputs:
        block2_manifest_path: Evidence-ready Block 2 manifest anchoring the
            internal Block 3 run.
        output_dir: Output root for this subexperiment's raw and review files.
        subexperiment_id: One of `3A`, `3B-1`, `3B-2`, `3C-1`, or `3C-2`.
        support_mode: Internal-only generator support mode. Defaults to the
            legacy nearest-`C` support rule and optionally enables the
            motif-informed probe variant.
        open_mass_scale: Internal-only open-mass scaling factor applied after
            sampling the rerun-specific open realization and before rebuilding
            `d/e` plus matched mass.

    Returns:
        A `Block3InternalExecutionResult` with the written raw/review manifests,
        indexes, and artifact paths.

    Core flow:
        1. Resolve and validate upstream Block 1/2 inputs.
        2. Build the fixed non-authority execution plan.
        3. Load real cohort inputs and generate rerun-specific truths.
        4. Dispatch to the section-specific row builder.
        5. Write paired raw and review artifacts and return their paths.
    """
    inputs = resolve_upstream_inputs(
        block2_manifest_path=block2_manifest_path,
        output_dir=output_dir,
    )
    plan = build_phase2_execution_plan(inputs)
    executable_spec = get_subexperiment_spec(subexperiment_id)
    if subexperiment_id not in INTERNAL_EXECUTABLE_SUBEXPERIMENTS:
        raise ContractError(f"{subexperiment_id!r} is not enabled for internal Block 3 execution")

    cohort_inputs = _load_block3_cohort_inputs(inputs)
    reruns = _build_generator_reruns(
        cohort_inputs=cohort_inputs,
        support_mode=support_mode,
        open_mass_scale=open_mass_scale,
    )

    if executable_spec.subexperiment_id == Block3SubexperimentId.GENERATOR_VALIDATION.value:
        raw_rows, review_rows = _build_3a_rows(reruns=reruns, cohort_inputs=cohort_inputs)
    elif executable_spec.subexperiment_id == Block3SubexperimentId.A_BENCHMARK.value:
        raw_rows, review_rows = _build_3b1_rows(reruns=reruns, cohort_inputs=cohort_inputs)
    elif executable_spec.subexperiment_id == Block3SubexperimentId.DE_BENCHMARK.value:
        raw_rows, review_rows = _build_3b2_rows(reruns=reruns, cohort_inputs=cohort_inputs)
    elif executable_spec.subexperiment_id == Block3SubexperimentId.OPEN_MODULE_ABLATION.value:
        raw_rows, review_rows = _build_3c1_rows(reruns=reruns, cohort_inputs=cohort_inputs)
    elif executable_spec.subexperiment_id == Block3SubexperimentId.COHORT_MODULE_ABLATION.value:
        raw_rows, review_rows = _build_3c2_rows(reruns=reruns, cohort_inputs=cohort_inputs)
    else:  # pragma: no cover
        raise ContractError(f"Unsupported Block 3 subexperiment: {subexperiment_id!r}")

    return _write_internal_block3_outputs(
        output_dir=plan.inputs.output_dir,
        plan=plan,
        subexperiment_id=subexperiment_id,
        raw_rows=raw_rows,
        review_rows=review_rows,
    )


__all__ = [
    "BLOCK3_PACKET_BRIDGE_POLICY",
    "Block3DiagnosticArmResult",
    "Block3DiagnosticMatrixResult",
    "Block3ExecutionPlan",
    "Block3InternalExecutionResult",
    "Block3ResolvedInputs",
    "INTERNAL_EXECUTABLE_SUBEXPERIMENTS",
    "build_phase2_execution_plan",
    "execute_internal_block3_3b_diagnostic_matrix",
    "execute_internal_block3_subexperiment",
    "resolve_upstream_inputs",
    "validate_phase2_execution_plan",
]
