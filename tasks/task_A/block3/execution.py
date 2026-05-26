"""Contract-strict internal execution for Task A Block 3 Phase 3.

Role:
    Execute the internal, non-authority Phase 3 Block 3 benchmark stack from
    Stage 0 h5ad plus Task A config and produce proof-carrying raw/review row
    families for one semantic subexperiment at a time.

Authority anchors:
    - docs/task_A/spec.md §4.5.2-§4.5.6, §5.1 Phase 3
    - docs/task_A/block3/scientific_contract.md §4.1-§4.4, §5.5, §5.6

Local boundary:
    - This module owns semisynthetic rerun generation, method execution, and
      metric-row construction for internal Phase 3 use.
    - It does not reopen the public Block 3 workflow, review CLI, or packet
      bridge.
    - It does not promote internal outputs into scientific authority.

Primary contents:
    - Stage0-only cohort input resolution.
    - The formal 24 train / 8 test / 10 rerun multi-FOV generator flow,
      with optional smaller subset controls for engineering smoke tests.
    - Real STRIDE, baseline, and core-ablation runners over generated
      multi-FOV observations and endpoint projections.
    - Raw/review row builders for `3A`, `3B`, and `3C-*`.

Core logic flow:
    1. Resolve Stage 0 h5ad plus Task A config.
    2. Load the carried patient cohort plus Stage 0-derived identity vectors
       and the derived cost matrix.
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
from typing import Any

import numpy as np
import pandas as pd

from stride.errors import ContractError
from stride.geometry.state_geometry import build_state_geometry
from stride.observation import FovObservation
from stride.optimize import TrainConfig
from stride.settings import RuntimeSettings
from stride.workflows.config import TaskConfig
from stride.workflows.fit_stride import run_stride_fit

from ..config import load_task_a_config_bundle
from ..contracts import SCAFFOLD_ACTIVE_STATE
from ..workflows.stride_adapter import (
    build_task_a_family_observations,
    load_task_a_dataset_handle,
    resolve_task_a_state_basis,
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
from .multifov_generator import (
    build_template_bank,
    compute_patient_diagnostics,
    generate_patient_multifov,
    sample_split,
    select_template_medoid,
    synthesize_patient_truth,
)
from .registry import (
    Block3Registry,
    get_condition_spec,
    get_live_block3_registry,
    get_method_spec,
    get_metric_spec,
    get_subexperiment_spec,
    resolve_block3_experiment_name,
)

# Frozen boundary flag mirrored into all internal Block 3 manifests.
BLOCK3_PACKET_BRIDGE_POLICY = "deferred_non_authority_pending_clean_bridge_spec"
# The only executable units on the internal Phase 3 surface. `3C` stays a
# non-executable section container and is validated through `3C-*`.
INTERNAL_EXECUTABLE_SUBEXPERIMENTS: tuple[str, ...] = (
    "3A",
    "3B-1",
    "3B-2",
    "3C-1",
    "3C-2",
    "3C-3",
)
_GENERATOR_VALIDATION_CONDITION_ID = "generator_validation"
# Frozen outer-design constants from the Block 3 scientific contract.
_N_GENERATOR_RERUNS = 10
_N_TRAIN_PATIENTS = 24
_N_TEST_PATIENTS = 8
_OPEN_SUPPORT_COVERAGE = 0.95
_TOL = 1e-12
_GENERATOR_TAU = 2.0
_GENERATOR_LAMBDA_INDIVIDUAL = 0.10
_GENERATOR_ETA = 0.3
_GENERATOR_ALPHA_S = 1.0
_A_BENCHMARK_CONDITION_ID = "a_benchmark_shared_realization_set"
_DE_BENCHMARK_CONDITION_ID = "de_benchmark_shared_realization_set"
_SHARED_GENERATOR_TRUTH_CONDITION_ID = "multi_fov_shared_realization_set"


@dataclass(frozen=True)
class Block3Stage0ExecutionPlan:
    """Execution-plan metadata for the Stage0 + config internal CLI surface."""

    output_dir: Path
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
    execution_scope: str
    n_generator_reruns: int
    n_test_patients: int


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
class Block3RuntimeControls:
    """Optional runtime controls for internal Block 3 method execution."""

    device: object | None = None

    def requested_device_label(self) -> str | None:
        return None if self.device is None else str(self.device)

    def uot_runtime_settings(self) -> RuntimeSettings:
        label = self.requested_device_label()
        if label is not None and label.startswith(("cuda", "mps")):
            return RuntimeSettings(uot_backend="torch", device=label)
        return RuntimeSettings()


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
    source_domain: str
    target_domain: str
    patient_source_fovs: dict[str, np.ndarray]
    patient_target_fovs: dict[str, np.ndarray]
    patient_source_profiles: dict[str, np.ndarray]
    patient_target_profiles: dict[str, np.ndarray]


@dataclass(frozen=True)
class _Block3IdentityCostComponents:
    """Identity-derived state-geometry cost with raw and normalized forms."""

    C_raw: np.ndarray
    s_C: float
    C_norm: np.ndarray


@dataclass(frozen=True)
class Block3PatientTruth:
    """Hidden per-patient truth object for one rerun-specific realization.

    Fields:
        x: Source-side endpoint projection used by endpoint-only baselines and
            scoring.
        y: Synthetic target endpoint projection used by endpoint-only baselines
            and scoring.
        A: Hidden relation operator scored in `3B` and `3C-*`.
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
    y_endpoint: np.ndarray | None = None
    source_fovs: np.ndarray | None = None
    target_fovs: np.ndarray | None = None
    sampled_template_patient_id: str | None = None
    medoid_template_patient_id: str | None = None
    row_imputed_mask: np.ndarray | None = None
    endpoint_closure_l1: float | None = None
    generator_diagnostics: dict[str, object] | None = None


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

    `generator_truths` holds the single multi-FOV hidden truth shared by `3A`,
    `3B`, and `3C-*`. `baseline_truths` mirrors that same truth by condition
    only so older row builders can remain explicit about condition routing.
    """

    rerun_id: str
    split_seed: int
    train_patient_ids: tuple[str, ...]
    test_patient_ids: tuple[str, ...]
    hidden_relation_condition_id: str
    open_mass_scale: float
    generator_truths: dict[str, Block3PatientTruth]
    baseline_truths: dict[str, dict[str, Block3PatientTruth]]
    template_medoid_patient_id: str | None = None
    generator_parameters: dict[str, object] | None = None


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


def build_stage0_execution_plan(
    *,
    output_dir: str | Path,
    registry: Block3Registry | None = None,
    n_generator_reruns: int = _N_GENERATOR_RERUNS,
    n_test_patients: int = _N_TEST_PATIENTS,
) -> Block3Stage0ExecutionPlan:
    """Build the non-authority execution plan for Stage0 + config Block 3 runs."""

    resolved_registry = get_live_block3_registry() if registry is None else registry
    execution_scope = (
        "formal_full_data"
        if int(n_generator_reruns) == _N_GENERATOR_RERUNS and int(n_test_patients) == _N_TEST_PATIENTS
        else "subset_engineering_test"
    )
    plan = Block3Stage0ExecutionPlan(
        output_dir=_resolve_path(output_dir),
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
        execution_scope=execution_scope,
        n_generator_reruns=int(n_generator_reruns),
        n_test_patients=int(n_test_patients),
    )
    validate_phase2_execution_plan(plan)
    return plan


def validate_phase2_execution_plan(plan: Block3Stage0ExecutionPlan) -> None:
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
    if plan.n_generator_reruns <= 0 or plan.n_generator_reruns > _N_GENERATOR_RERUNS:
        raise ContractError(f"Block 3 n_generator_reruns must be in [1, {_N_GENERATOR_RERUNS}]")
    if plan.n_test_patients <= 0 or plan.n_test_patients > _N_TEST_PATIENTS:
        raise ContractError(f"Block 3 n_test_patients must be in [1, {_N_TEST_PATIENTS}]")


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
    grouped: dict[tuple[str, str, str], list[Block3PatientMetricRow]] = {}
    for row in patient_rows:
        grouped.setdefault(
            (
                row.condition_id,
                row.method_name.value,
                row.metric_value.metric_name.value,
            ),
            [],
        ).append(row)

    for (condition_id, method_name, metric_name), rows in grouped.items():
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
                reference_rows = grouped[(condition_id, "stride_reference", metric_name)]
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
                open_mass_scale=None,
            )
        )
    summary_rows.sort(
        key=lambda row: (
            row.condition_id,
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


def _build_identity_cost_components(identity_vectors: np.ndarray) -> _Block3IdentityCostComponents:
    """Build raw and normalized identity-aware Block 3 cost components.

    The scientific contract freezes `C_raw[i,j] = sqrt(JS(g_i, g_j))` with the
    normalization scale `s_C` defined as the median of positive off-diagonal
    entries. `C_raw` and `s_C` are carried into `StateGeometry` so provenance can
    report the original scale, while `C_norm` is reused by relation-support
    scenarios and the balanced OT baseline.
    """

    n_states = identity_vectors.shape[0]
    C_raw = np.zeros((n_states, n_states), dtype=float)
    for left_index in range(n_states):
        for right_index in range(left_index + 1, n_states):
            value = float(np.sqrt(_js_divergence(identity_vectors[left_index], identity_vectors[right_index])))
            C_raw[left_index, right_index] = value
            C_raw[right_index, left_index] = value
    positive = C_raw[C_raw > 0.0]
    s_C = float(np.median(positive)) if positive.size else 1.0
    return _Block3IdentityCostComponents(C_raw=C_raw, s_C=s_C, C_norm=C_raw / s_C)


def _build_identity_cost_matrix(identity_vectors: np.ndarray) -> np.ndarray:
    """Build the normalized identity-aware Block 3 cost matrix `C_norm`."""

    return _build_identity_cost_components(identity_vectors).C_norm


def _derive_identity_vectors_from_stage0(
    handle: Any,
    *,
    state_ids: tuple[int, ...],
) -> np.ndarray:
    """Derive `g_k` community identity vectors from Stage 0 state/subtype rows."""

    obs = handle.adata.obs.loc[:, [handle.state_id_key, handle.cell_subtype_key]].copy()
    obs[handle.state_id_key] = obs[handle.state_id_key].astype(int)
    obs[handle.cell_subtype_key] = obs[handle.cell_subtype_key].astype(str)
    subtype_labels = tuple(sorted(str(label) for label in obs[handle.cell_subtype_key].unique()))
    if not subtype_labels:
        raise ContractError("Block 3 requires non-empty Stage 0 cell-subtype labels to derive g_k")
    frame = pd.crosstab(obs[handle.state_id_key], obs[handle.cell_subtype_key])
    frame = frame.reindex(index=list(state_ids), columns=list(subtype_labels), fill_value=0.0)
    vectors: list[np.ndarray] = []
    for state_id in state_ids:
        row = frame.loc[int(state_id)].to_numpy(dtype=float)
        if not np.all(np.isfinite(row)):
            raise ContractError(
                "Block 3 requires finite Stage 0-derived community identity vectors; "
                f"state_id={int(state_id)} contains non-finite values"
            )
        row_total = float(np.sum(row, dtype=float))
        if row_total <= 0.0:
            raise ContractError(
                "Block 3 requires positive-total Stage 0-derived community identity vectors; "
                f"state_id={int(state_id)} has zero total mass"
            )
        vectors.append(row / row_total)
    return np.vstack(vectors)


def _build_block3_cohort_inputs_from_stage0(
    *,
    stage0_h5ad: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    n_test_patients: int = _N_TEST_PATIENTS,
) -> Block3CohortInputs:
    """Resolve Block 3 hard inputs from Stage 0 h5ad plus Task A config."""

    resolved_stage0_h5ad = _resolve_path(stage0_h5ad)
    resolved_config_path = _resolve_path(config_path)
    config_bundle = load_task_a_config_bundle(resolved_config_path)
    family_name = config_bundle.block3.benchmark_pair_family
    family_spec = next(
        (family for family in config_bundle.ordered_proxy.pair_families if family.name == family_name),
        None,
    )
    if family_spec is None:  # pragma: no cover - validated by config loader.
        raise ContractError(f"Task A Block 3 requires configured pair family {family_name!r}")

    handle = load_task_a_dataset_handle(resolved_stage0_h5ad)
    state_basis = resolve_task_a_state_basis(handle)
    state_ids = tuple(int(state_id) for state_id in state_basis.resolved_state_ids)
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
    patient_source_fovs: dict[str, np.ndarray] = {}
    patient_target_fovs: dict[str, np.ndarray] = {}
    patient_source_profiles: dict[str, np.ndarray] = {}
    patient_target_profiles: dict[str, np.ndarray] = {}
    for patient_id, domain_vectors in sorted(patient_vectors.items()):
        if family_spec.source_domain not in domain_vectors or family_spec.target_domain not in domain_vectors:
            continue
        source_matrix = np.vstack(domain_vectors[family_spec.source_domain]).astype(float, copy=False)
        target_matrix = np.vstack(domain_vectors[family_spec.target_domain]).astype(float, copy=False)
        patient_source_fovs[patient_id] = source_matrix
        patient_target_fovs[patient_id] = target_matrix
        patient_source_profiles[patient_id] = _mean_profile(source_matrix)
        patient_target_profiles[patient_id] = _mean_profile(target_matrix)
    n_required_patients = _N_TRAIN_PATIENTS + int(n_test_patients)
    if len(patient_source_profiles) < n_required_patients:
        raise ContractError(
            "Block 3 internal execution requires enough eligible patients for the "
            f"requested 24/{int(n_test_patients)} split; found {len(patient_source_profiles)}"
        )

    identity_vectors = _derive_identity_vectors_from_stage0(handle, state_ids=state_ids)
    identity_cost = _build_identity_cost_components(identity_vectors)
    geometry = build_state_geometry(
        cost_matrix=identity_cost.C_raw,
        cost_scale=identity_cost.s_C,
        state_ids=state_ids,
    )
    return Block3CohortInputs(
        stage0_h5ad=resolved_stage0_h5ad,
        config_path=resolved_config_path,
        output_dir=_resolve_path(output_dir),
        master_seed=int(config_bundle.block3.master_seed),
        state_ids=state_ids,
        state_basis=state_basis,
        geometry=geometry,
        identity_vectors=identity_vectors,
        cost_matrix=identity_cost.C_norm,
        source_domain=family_spec.source_domain,
        target_domain=family_spec.target_domain,
        patient_source_fovs=patient_source_fovs,
        patient_target_fovs=patient_target_fovs,
        patient_source_profiles=patient_source_profiles,
        patient_target_profiles=patient_target_profiles,
    )


def _build_generator_reruns(
    *,
    cohort_inputs: Block3CohortInputs,
    n_reruns: int = _N_GENERATOR_RERUNS,
    n_test_patients: int = _N_TEST_PATIENTS,
) -> tuple[Block3GeneratorRerun, ...]:
    """Build the shared multi-FOV train-template generator realizations."""

    if n_reruns <= 0 or n_reruns > _N_GENERATOR_RERUNS:
        raise ContractError(f"Block 3 n_reruns must be in [1, {_N_GENERATOR_RERUNS}]")
    if n_test_patients <= 0 or n_test_patients > _N_TEST_PATIENTS:
        raise ContractError(f"Block 3 n_test_patients must be in [1, {_N_TEST_PATIENTS}]")
    patient_ids = tuple(sorted(cohort_inputs.patient_source_profiles))
    n_selected_patients = _N_TRAIN_PATIENTS + int(n_test_patients)
    if len(patient_ids) < n_selected_patients:
        raise ContractError(
            "Block 3 generator requires enough eligible patients for the requested "
            f"24/{int(n_test_patients)} split; found {len(patient_ids)}"
        )
    reruns: list[Block3GeneratorRerun] = []
    for rerun_index in range(int(n_reruns)):
        split_seed = int(cohort_inputs.master_seed + rerun_index)
        selected_rng = np.random.default_rng(split_seed)
        selected = tuple(
            str(patient_id)
            for patient_id in selected_rng.permutation(patient_ids)[
                :n_selected_patients
            ].tolist()
        )
        train_patient_ids, test_patient_ids = sample_split(
            selected,
            n_test=int(n_test_patients),
            seed=split_seed,
        )
        template_bank = build_template_bank(
            train_patient_ids=train_patient_ids,
            source_endpoints=cohort_inputs.patient_source_profiles,
            target_endpoints=cohort_inputs.patient_target_profiles,
            C_norm=cohort_inputs.cost_matrix,
            tau=_GENERATOR_TAU,
        )
        medoid = select_template_medoid(template_bank, alpha_s=_GENERATOR_ALPHA_S)
        generator_truths: dict[str, Block3PatientTruth] = {}
        rerun_id = f"rerun_{rerun_index + 1:02d}"
        patient_rng = np.random.default_rng(split_seed + 1000)
        for patient_id in test_patient_ids:
            truth = synthesize_patient_truth(
                patient_id=patient_id,
                test_x=cohort_inputs.patient_source_profiles[patient_id],
                template_bank=template_bank,
                medoid=medoid,
                rng=patient_rng,
                lambda_individual=_GENERATOR_LAMBDA_INDIVIDUAL,
            )
            generated = generate_patient_multifov(
                truth=truth,
                real_source_fovs=cohort_inputs.patient_source_fovs[patient_id],
                n_target_fovs=int(cohort_inputs.patient_target_fovs[patient_id].shape[0]),
                eta=_GENERATOR_ETA,
            )
            diagnostics = compute_patient_diagnostics(
                generated=generated,
                C_norm=cohort_inputs.cost_matrix,
            )
            generator_truths[patient_id] = Block3PatientTruth(
                rerun_id=rerun_id,
                patient_id=patient_id,
                x=generated.endpoint_x_obs,
                y=generated.endpoint_y_obs,
                A=truth.A,
                d=truth.d,
                e=truth.e,
                open_mass=float(np.sum(truth.e, dtype=float)),
                y_endpoint=generated.endpoint_y_obs,
                source_fovs=generated.source_fovs,
                target_fovs=generated.target_fovs,
                sampled_template_patient_id=truth.sampled_template_patient_id,
                medoid_template_patient_id=truth.medoid_template_patient_id,
                row_imputed_mask=truth.row_imputed_mask,
                endpoint_closure_l1=generated.endpoint_closure_l1,
                generator_diagnostics=diagnostics,
            )
        baseline_truths = {
            _A_BENCHMARK_CONDITION_ID: dict(generator_truths),
            _DE_BENCHMARK_CONDITION_ID: dict(generator_truths),
        }
        reruns.append(
            Block3GeneratorRerun(
                rerun_id=rerun_id,
                split_seed=split_seed,
                train_patient_ids=train_patient_ids,
                test_patient_ids=test_patient_ids,
                hidden_relation_condition_id=_SHARED_GENERATOR_TRUTH_CONDITION_ID,
                open_mass_scale=1.0,
                generator_truths=generator_truths,
                baseline_truths=baseline_truths,
                template_medoid_patient_id=medoid.patient_id,
                generator_parameters={
                    "tau": _GENERATOR_TAU,
                    "lambda_individual": _GENERATOR_LAMBDA_INDIVIDUAL,
                    "eta": _GENERATOR_ETA,
                    "alpha_s": _GENERATOR_ALPHA_S,
                    "target_noise": "none",
                    "template_bank_source": "real_train_TC_IM_endpoints",
                },
            )
        )
    return tuple(reruns)


def _make_observations_for_truths(truths: list[Block3PatientTruth]) -> tuple[FovObservation, ...]:
    """Expose one truth set to STRIDE as generated multi-FOV observations."""

    observations: list[FovObservation] = []
    for truth in truths:
        if truth.source_fovs is None or truth.target_fovs is None:
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
            continue
        for index, row in enumerate(np.asarray(truth.source_fovs, dtype=float)):
            observations.append(
                FovObservation(
                    patient_id=truth.patient_id,
                    timepoint="pre",
                    fov_id=f"{truth.rerun_id}_{truth.patient_id}_pre_fov{index + 1:02d}",
                    domain_label="TC",
                    community_composition=row,
                    mass=1.0,
                    mass_mode="uniform",
                )
            )
        for index, row in enumerate(np.asarray(truth.target_fovs, dtype=float)):
            observations.append(
                FovObservation(
                    patient_id=truth.patient_id,
                    timepoint="post",
                    fov_id=f"{truth.rerun_id}_{truth.patient_id}_post_fov{index + 1:02d}",
                    domain_label="IM",
                    community_composition=row,
                    mass=1.0,
                    mass_mode="uniform",
                )
            )
    return tuple(observations)


def _fit_provenance_payload(provenance: object | None) -> dict[str, Any] | None:
    """Return a compact provenance mapping without retaining the source object."""

    if provenance is None:
        return None
    to_dict = getattr(provenance, "to_dict", None)
    payload = to_dict() if callable(to_dict) else provenance
    if not isinstance(payload, dict):
        return None
    return dict(payload)


def _compact_stride_fit_metadata(
    *,
    fit_result: object,
    patient_result: object,
    ablation_mode: str,
) -> dict[str, object]:
    """Build Task A's per-patient compact STRIDE fit status/provenance summary."""

    metadata = dict(getattr(fit_result, "metadata", {}) or {})
    diagnostics = dict(getattr(fit_result, "diagnostics", {}) or {})
    summaries = dict(getattr(fit_result, "summaries", {}) or {})
    patient_diagnostics = dict(getattr(patient_result, "diagnostics", {}) or {})
    patient_audit = getattr(patient_result, "audit", None)
    patient_audit_metadata = dict(getattr(patient_audit, "metadata", {}) or {})
    provenance = _fit_provenance_payload(getattr(fit_result, "provenance", None))

    summary: dict[str, object] = {
        "implementation_tier": str(
            getattr(
                patient_result,
                "implementation_tier",
                getattr(fit_result, "implementation_tier", "unknown"),
            )
        ),
        "fit_status": str(
            getattr(patient_result, "fit_status", getattr(fit_result, "fit_status", "unknown"))
        ),
    }
    optimizer_status = (
        metadata.get("optimizer_status")
        or diagnostics.get("optimizer_status")
        or summaries.get("optimizer_status")
        or patient_diagnostics.get("optimizer_status")
        or patient_audit_metadata.get("optimizer_status")
    )
    if optimizer_status is not None:
        summary["optimizer_status"] = str(optimizer_status)
    completion_reason = diagnostics.get("completion_reason")
    if completion_reason is not None:
        summary["optimizer_completion_reason"] = str(completion_reason)

    for reason_key in ("defer_reason", "failure_reason", "message"):
        reason = (
            patient_diagnostics.get(reason_key)
            or patient_audit_metadata.get(reason_key)
            or diagnostics.get(reason_key)
        )
        if reason is not None:
            summary[reason_key] = str(reason)

    if provenance is not None:
        observation_discrepancy = dict(provenance.get("observation_discrepancy", {}) or {})
        state_geometry = dict(provenance.get("state_geometry", {}) or {})
        comparison_plan = dict(provenance.get("observation_comparison_plan", {}) or {})
        summary.update(
            {
                "provenance_schema_version": str(provenance["provenance_schema_version"]),
                "observation_discrepancy_operator_version": str(
                    observation_discrepancy["operator_version"]
                ),
                "state_geometry_s_C": float(state_geometry["s_C"]),
                "n_evidence_blocks": int(comparison_plan["n_evidence_blocks"]),
                "post_reconstruction_form": str(provenance["post_reconstruction_form"]),
            }
        )
        if "ablation_mode" in provenance:
            summary["ablation_mode"] = str(provenance["ablation_mode"])
        if "ablation_term_handling" in provenance:
            summary["ablation_term_handling"] = str(provenance["ablation_term_handling"])
        if "ablation_denominator_policy" in provenance:
            summary["ablation_denominator_policy"] = str(
                provenance["ablation_denominator_policy"]
            )
    elif "n_evidence_blocks" in metadata:
        summary["n_evidence_blocks"] = int(metadata["n_evidence_blocks"])

    if ablation_mode != "none":
        summary.setdefault("ablation_mode", ablation_mode)
        summary.setdefault("ablation_term_handling", "zero_weight")
        summary.setdefault(
            "ablation_denominator_policy",
            "fixed_denominator_no_reweighting",
        )
        ablation_status = metadata.get("ablation_status") or diagnostics.get("ablation_status")
        if ablation_status is not None:
            summary["ablation_status"] = str(ablation_status)
    return summary


def _run_stride_method(
    *,
    cohort_inputs: Block3CohortInputs,
    truths: list[Block3PatientTruth],
    runtime: Block3RuntimeControls | None = None,
    ablation_mode: str = "none",
) -> dict[str, Block3MethodOutput]:
    """Run canonical STRIDE or one STRIDE ablation on generated FOV inputs."""

    resolved_runtime = runtime or Block3RuntimeControls()
    fit_result = run_stride_fit(
        _make_observations_for_truths(truths),
        task_config=TaskConfig(
            source="pre",
            target="post",
            K=len(cohort_inputs.state_ids),
            timepoint_order=("pre", "post"),
        ),
        train_config=TrainConfig(
            ablation_mode=ablation_mode,
            device=resolved_runtime.device,
        ),
        state_basis=cohort_inputs.state_basis,
        geometry=cohort_inputs.geometry,
    )
    outputs: dict[str, Block3MethodOutput] = {}
    for patient_result in fit_result.patient_results:
        metadata = _compact_stride_fit_metadata(
            fit_result=fit_result,
            patient_result=patient_result,
            ablation_mode=ablation_mode,
        )
        metadata["requested_device"] = resolved_runtime.requested_device_label()
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
            metadata=metadata,
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
    runtime: Block3RuntimeControls | None = None,
) -> dict[str, Block3MethodOutput]:
    """Run UOT baseline plans and derive native `A/d/e` outputs.

    Inputs are test-side truths, a shared cost matrix, and calibrated lambda.
    Outputs include native `P` and UOT metadata per patient. Solver failures are
    carried as non-ok status for metric propagation.
    """

    resolved_runtime = runtime or Block3RuntimeControls()
    uot_runtime = resolved_runtime.uot_runtime_settings()
    outputs = _run_plan_baseline(
        truths=truths,
        plan_builder=lambda x, y: solve_uot_plan(
            x=x,
            y=y,
            cost_matrix=cost_matrix,
            match_penalty=match_penalty,
            runtime_settings=uot_runtime,
        ),
    )
    runtime_metadata = {"requested_device": resolved_runtime.requested_device_label()}
    if calibration_metadata is None:
        calibration_metadata = {}
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
            metadata={**(output.metadata or {}), **calibration_metadata, **runtime_metadata},
        )
        for patient_id, output in outputs.items()
    }


def _run_partial_ot_baseline(
    *,
    truths: list[Block3PatientTruth],
    cost_matrix: np.ndarray | None = None,
    matched_mass_budget: float | None = None,
    calibration_metadata: dict[str, object] | None = None,
) -> dict[str, Block3MethodOutput]:
    """Run hard-budget partial-OT plans and derive native `A/d/e` outputs.

    Inputs are test-side truths, optional costs, and an optional train-side
    matched-mass budget. Output metadata records requested/effective budget and
    clipping status for each patient.
    """

    outputs = _run_plan_baseline(
        truths=truths,
        plan_builder=lambda x, y: partial_ot_plan(
            x,
            y,
            cost_matrix=cost_matrix,
            matched_mass_budget=matched_mass_budget,
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


def _run_diagonal_transport_baseline(*, truths: list[Block3PatientTruth]) -> dict[str, Block3MethodOutput]:
    """Run diagonal native plans on test-side truths.

    Inputs are test-side truths. Outputs include native diagonal `P`; `A/d/e`
    are derived by the shared `P -> A/d/e` analysis layer. This comparator is
    not the historical abundance-only baseline.
    """

    return _run_plan_baseline(truths=truths, plan_builder=diagonal_transport_plan)


def _train_endpoint_pairs_for_rerun(
    *,
    cohort_inputs: Block3CohortInputs,
    train_patient_ids: tuple[str, ...],
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return real train endpoint pairs for Block3b comparator calibration.

    Inputs are the rerun's train patient ids plus Stage0-derived real endpoint
    profiles. Output is consumed by UOT calibration and partial-OT budget
    estimation. Hidden semisynthetic `A/d/e` truth is not read.
    """

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for patient_id in train_patient_ids:
        source = np.asarray(
            cohort_inputs.patient_source_profiles[patient_id],
            dtype=float,
        )
        target = np.asarray(
            cohort_inputs.patient_target_profiles[patient_id],
            dtype=float,
        )
        pairs.append((_normalize_probabilities(source), _normalize_probabilities(target)))
    return tuple(pairs)


def _matched_mass_budget_from_train(
    *,
    cohort_inputs: Block3CohortInputs,
    train_patient_ids: tuple[str, ...],
) -> float:
    """Estimate deterministic train-side matched-mass budget for partial OT.

    Inputs are real train endpoint profiles. Output is the mean endpoint overlap
    used as the requested hard budget; callers still clip per patient to
    feasible source/target mass.
    """

    pairs = _train_endpoint_pairs_for_rerun(
        cohort_inputs=cohort_inputs,
        train_patient_ids=train_patient_ids,
    )
    if not pairs:
        raise ValueError("Partial OT budget estimation requires train endpoint pairs")
    # matched_mass_budget is the train-side requested hard comparator budget.
    matched_mass_budget = float(np.mean([np.sum(np.minimum(x, y), dtype=float) for x, y in pairs]))
    return matched_mass_budget


def _calibrated_uot_lambda_for_train(
    *,
    cohort_inputs: Block3CohortInputs,
    train_patient_ids: tuple[str, ...],
    cost_matrix: np.ndarray,
    runtime: Block3RuntimeControls | None = None,
) -> UOTCalibrationResult:
    """Select one train-side UOT lambda for a Block3b rerun.

    Inputs are real train endpoint profiles and the shared cost matrix. Output
    is the calibrated lambda result; boundary-hit diagnostics are carried on
    patient UOT metadata rather than failing the run.
    """

    train_pairs = _train_endpoint_pairs_for_rerun(
        cohort_inputs=cohort_inputs,
        train_patient_ids=train_patient_ids,
    )
    resolved_runtime = runtime or Block3RuntimeControls()
    uot_runtime = resolved_runtime.uot_runtime_settings()
    calibration = calibrate_uot_lambda(
        train_pairs=train_pairs,
        achieved_mass_fn=lambda candidate, pairs: estimate_uot_matched_mass(
            train_pairs=tuple(pairs),
            cost_matrix=cost_matrix,
            match_penalty=candidate,
            runtime_settings=uot_runtime,
        ),
    )
    return calibration


def _uot_calibration_metadata(
    calibration: UOTCalibrationResult,
    *,
    n_train_pairs: int,
) -> dict[str, object]:
    """Serialize UOT train-side calibration diagnostics for native metadata."""

    return {
        "calibration_source": "real_train_endpoints",
        "n_train_pairs": n_train_pairs,
        "selected_lambda": calibration.selected_lambda,
        "target_overlap": calibration.target_overlap,
        "boundary_hit": calibration.boundary_hit,
        "achieved_by_lambda": calibration.achieved_by_lambda,
        "absolute_error_by_lambda": calibration.absolute_error_by_lambda,
    }


def _partial_ot_calibration_metadata(*, n_train_pairs: int) -> dict[str, object]:
    """Serialize partial-OT train-side budget calibration provenance."""

    return {
        "calibration_source": "real_train_endpoints",
        "budget_calibration": "mean_endpoint_overlap",
        "n_train_pairs": n_train_pairs,
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
            "y_endpoint_json": _json_array(truth.y_endpoint),
            "sampled_template_patient_id": truth.sampled_template_patient_id,
            "medoid_template_patient_id": truth.medoid_template_patient_id,
            "row_imputed_mask_json": _json_array(truth.row_imputed_mask),
            "endpoint_closure_l1": truth.endpoint_closure_l1,
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
            "template_medoid_patient_id": rerun.template_medoid_patient_id,
            "generator_parameters_json": json.dumps(
                rerun.generator_parameters or {},
                sort_keys=True,
                separators=(",", ":"),
            ),
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
        "generator_diagnostics": tuple(
            {
                "rerun_id": rerun.rerun_id,
                **dict(truth.generator_diagnostics or {}),
            }
            for rerun in reruns
            for truth in rerun.generator_truths.values()
        ),
    }


def _build_3a_rows(
    *,
    reruns: tuple[Block3GeneratorRerun, ...],
    cohort_inputs: Block3CohortInputs,
) -> tuple[Block3SubexperimentRawRows, Block3SubexperimentReviewRows]:
    """Compatibility wrapper for the split `3A` implementation."""

    from .stage3a import build_3a_rows

    return build_3a_rows(reruns=reruns, cohort_inputs=cohort_inputs)


def _build_3b1_rows(
    *,
    reruns: tuple[Block3GeneratorRerun, ...],
    cohort_inputs: Block3CohortInputs,
    runtime: Block3RuntimeControls | None = None,
) -> tuple[Block3SubexperimentRawRows, Block3SubexperimentReviewRows]:
    """Compatibility wrapper for the split `3B-1` implementation."""

    from .stage3b1 import build_3b1_rows

    return build_3b1_rows(reruns=reruns, cohort_inputs=cohort_inputs, runtime=runtime)


def _build_3b2_rows(
    *,
    reruns: tuple[Block3GeneratorRerun, ...],
    cohort_inputs: Block3CohortInputs,
    runtime: Block3RuntimeControls | None = None,
) -> tuple[Block3SubexperimentRawRows, Block3SubexperimentReviewRows]:
    """Compatibility wrapper for the split `3B-2` implementation."""

    from .stage3b2 import build_3b2_rows

    return build_3b2_rows(reruns=reruns, cohort_inputs=cohort_inputs, runtime=runtime)


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
    """Compatibility wrapper for the split open-profile metric builder."""

    from .metrics_open import build_open_metric_rows

    return build_open_metric_rows(
        rerun_id=rerun_id,
        subexperiment_id=subexperiment_id,
        condition_id=condition_id,
        evaluation_family=evaluation_family,
        method_name=method_name,
        truth=truth,
        output=output,
        open_mass_scale=open_mass_scale,
    )


def _build_relation_and_open_metric_rows(
    *,
    rerun_id: str,
    subexperiment_id: str,
    condition_id: str,
    evaluation_family: str,
    method_name: str,
    truth: Block3PatientTruth,
    output: Block3MethodOutput,
) -> list[Block3PatientMetricRow]:
    """Compatibility wrapper for the split relation/open metric builder."""

    from .metrics_open import build_relation_and_open_metric_rows

    return build_relation_and_open_metric_rows(
        rerun_id=rerun_id,
        subexperiment_id=subexperiment_id,
        condition_id=condition_id,
        evaluation_family=evaluation_family,
        method_name=method_name,
        truth=truth,
        output=output,
    )


def _build_core_ablation_rows(
    *,
    reruns: tuple[Block3GeneratorRerun, ...],
    cohort_inputs: Block3CohortInputs,
    subexperiment_id: str,
    condition_id: str,
    ablation_method_name: str,
    ablation_mode: str,
    runtime: Block3RuntimeControls | None = None,
) -> tuple[Block3SubexperimentRawRows, Block3SubexperimentReviewRows]:
    """Compatibility wrapper for the split `3C` common implementation."""

    from .stage3c_common import build_core_ablation_rows

    return build_core_ablation_rows(
        reruns=reruns,
        cohort_inputs=cohort_inputs,
        subexperiment_id=subexperiment_id,
        condition_id=condition_id,
        ablation_method_name=ablation_method_name,
        ablation_mode=ablation_mode,
        runtime=runtime,
    )


def _build_3c1_rows(
    *,
    reruns: tuple[Block3GeneratorRerun, ...],
    cohort_inputs: Block3CohortInputs,
    runtime: Block3RuntimeControls | None = None,
) -> tuple[Block3SubexperimentRawRows, Block3SubexperimentReviewRows]:
    """Compatibility wrapper for the split `3C-1` consistency implementation."""

    from .stage3c_consistency import build_3c1_rows

    return build_3c1_rows(reruns=reruns, cohort_inputs=cohort_inputs, runtime=runtime)


def _build_3c2_rows(
    *,
    reruns: tuple[Block3GeneratorRerun, ...],
    cohort_inputs: Block3CohortInputs,
    runtime: Block3RuntimeControls | None = None,
) -> tuple[Block3SubexperimentRawRows, Block3SubexperimentReviewRows]:
    """Compatibility wrapper for the split `3C-2` geometry implementation."""

    from .stage3c_geometry import build_3c2_rows

    return build_3c2_rows(reruns=reruns, cohort_inputs=cohort_inputs, runtime=runtime)


def _build_3c3_rows(
    *,
    reruns: tuple[Block3GeneratorRerun, ...],
    cohort_inputs: Block3CohortInputs,
    runtime: Block3RuntimeControls | None = None,
) -> tuple[Block3SubexperimentRawRows, Block3SubexperimentReviewRows]:
    """Compatibility wrapper for the split `3C-3` recurrence implementation."""

    from .stage3c_recurrence import build_3c3_rows

    return build_3c3_rows(reruns=reruns, cohort_inputs=cohort_inputs, runtime=runtime)


def _write_internal_block3_outputs(
    *,
    output_dir: str | Path,
    plan: Block3Stage0ExecutionPlan,
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


def _build_rows_for_subexperiment(
    *,
    subexperiment_id: str,
    reruns: tuple[Block3GeneratorRerun, ...],
    cohort_inputs: Block3CohortInputs,
    runtime: Block3RuntimeControls,
) -> tuple[Block3SubexperimentRawRows, Block3SubexperimentReviewRows]:
    executable_spec = get_subexperiment_spec(subexperiment_id)
    if executable_spec.subexperiment_id == Block3SubexperimentId.GENERATOR_VALIDATION.value:
        from .stage3a import build_3a_rows

        return build_3a_rows(reruns=reruns, cohort_inputs=cohort_inputs)
    if executable_spec.subexperiment_id == Block3SubexperimentId.A_BENCHMARK.value:
        from .stage3b1 import build_3b1_rows

        return build_3b1_rows(reruns=reruns, cohort_inputs=cohort_inputs, runtime=runtime)
    if executable_spec.subexperiment_id == Block3SubexperimentId.DE_BENCHMARK.value:
        from .stage3b2 import build_3b2_rows

        return build_3b2_rows(reruns=reruns, cohort_inputs=cohort_inputs, runtime=runtime)
    if executable_spec.subexperiment_id == Block3SubexperimentId.CONSISTENCY_ABLATION.value:
        from .stage3c_consistency import build_3c1_rows

        return build_3c1_rows(reruns=reruns, cohort_inputs=cohort_inputs, runtime=runtime)
    if executable_spec.subexperiment_id == Block3SubexperimentId.GEOMETRY_ABLATION.value:
        from .stage3c_geometry import build_3c2_rows

        return build_3c2_rows(reruns=reruns, cohort_inputs=cohort_inputs, runtime=runtime)
    if executable_spec.subexperiment_id == Block3SubexperimentId.RECURRENCE_ABLATION.value:
        from .stage3c_recurrence import build_3c3_rows

        return build_3c3_rows(reruns=reruns, cohort_inputs=cohort_inputs, runtime=runtime)
    raise ContractError(f"Unsupported Block 3 subexperiment: {subexperiment_id!r}")


def execute_internal_block3_experiment(
    *,
    experiment_name: str,
    task_config_path: str | Path,
    stage0_h5ad: str | Path,
    output_dir: str | Path,
    device: object | None = None,
    max_reruns: int | None = None,
    n_test: int | None = None,
) -> Block3InternalExecutionResult:
    """Execute one semantic Block 3 experiment from Stage0 h5ad plus config."""

    executable_spec = resolve_block3_experiment_name(experiment_name)
    runtime = Block3RuntimeControls(device=device)
    n_generator_reruns = _N_GENERATOR_RERUNS if max_reruns is None else int(max_reruns)
    n_test_patients = _N_TEST_PATIENTS if n_test is None else int(n_test)
    plan = build_stage0_execution_plan(
        output_dir=output_dir,
        n_generator_reruns=n_generator_reruns,
        n_test_patients=n_test_patients,
    )
    cohort_inputs = _build_block3_cohort_inputs_from_stage0(
        stage0_h5ad=stage0_h5ad,
        config_path=task_config_path,
        output_dir=output_dir,
        n_test_patients=n_test_patients,
    )
    reruns = _build_generator_reruns(
        cohort_inputs=cohort_inputs,
        n_reruns=n_generator_reruns,
        n_test_patients=n_test_patients,
    )
    raw_rows, review_rows = _build_rows_for_subexperiment(
        subexperiment_id=executable_spec.subexperiment_id,
        reruns=reruns,
        cohort_inputs=cohort_inputs,
        runtime=runtime,
    )
    return _write_internal_block3_outputs(
        output_dir=plan.output_dir,
        plan=plan,
        subexperiment_id=executable_spec.subexperiment_id,
        raw_rows=raw_rows,
        review_rows=review_rows,
    )


__all__ = [
    "BLOCK3_PACKET_BRIDGE_POLICY",
    "Block3InternalExecutionResult",
    "Block3RuntimeControls",
    "Block3Stage0ExecutionPlan",
    "INTERNAL_EXECUTABLE_SUBEXPERIMENTS",
    "build_stage0_execution_plan",
    "execute_internal_block3_experiment",
    "validate_phase2_execution_plan",
]
