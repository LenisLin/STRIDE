"""Frozen Block 3 contract vocabulary and typed row shapes.

Role:
    Define the implementation-carried contract vocabulary for Block 3 together
    with the typed row objects shared by execution, raw bundle writing, and
    review-surface writing.

Authority anchors:
    - docs/task_A/spec.md §4.5.2, §4.5.5, §4.5.6, §5.1 Phase 3
    - docs/task_A/block3/scientific_contract.md §4.1-§4.4, §5.5, §5.6

Local boundary:
    - This module freezes identifiers, typed specs, and metric-status rules for
      the internal Phase 3 implementation surface.
    - It does not define new scientific authority or interpret Block 3 results.
    - It does not execute subexperiments, build registries, or write artifacts.

Primary contents:
    - Frozen enums for subexperiments, methods, metrics, statuses, and
      validation objects.
    - Dataclasses describing registry specs and flat artifact row families.
    - Helpers that normalize metric status and build validated metric values.

Why this module exists:
    Execution, registry, raw bundle, and review modules all need the same
    vocabulary and row types. Centralizing them here keeps the routing layer
    consistent and makes the split between non-method-bearing `3A` rows and the
    method-bearing `3B`/`3C-*` rows explicit in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from stride.errors import ContractError


class Block3SubexperimentId(str, Enum):
    """Frozen Block 3 section and subexperiment identifiers."""

    GENERATOR_VALIDATION = "3A"
    BASELINE_COMPARISON = "3B"
    A_BENCHMARK = "3B-1"
    DE_BENCHMARK = "3B-2"
    ABLATION_STUDY = "3C"
    CONSISTENCY_ABLATION = "3C-1"
    GEOMETRY_ABLATION = "3C-2"
    RECURRENCE_ABLATION = "3C-3"


class Block3MethodName(str, Enum):
    """Frozen method names used only by method-bearing Block 3 sections."""

    STRIDE_REFERENCE = "stride_reference"
    BALANCED_OT_BASELINE = "balanced_ot_baseline"
    UOT_BASELINE = "uot_baseline"
    PARTIAL_OT_BASELINE = "partial_ot_baseline"
    DIAGONAL_TRANSPORT_BASELINE = "diagonal_transport_baseline"
    RECURRENCE_ABLATION = "recurrence_ablation"
    GEOMETRY_ABLATION = "geometry_ablation"
    CONSISTENCY_ABLATION = "consistency_ablation"


class Block3MethodClass(str, Enum):
    """Coarse method families carried on method-bearing Block 3 rows."""

    REFERENCE = "reference"
    BASELINE = "baseline"
    ABLATION = "ablation"


class Block3MetricName(str, Enum):
    """Frozen metric names reported across the live Block 3 sections."""

    F_L1_TOTAL = "F_L1_total"
    G_L1_TOTAL = "g_L1_total"
    E_L1_TOTAL = "e_L1_total"
    OFFDIAG_MASS_ABS_ERROR = "offdiag_mass_abs_error"
    DEPLETION_MASS_ABS_ERROR = "depletion_mass_abs_error"
    EMERGENCE_MASS_ABS_ERROR = "emergence_mass_abs_error"
    OFFDIAG_RATIO = "offdiag_ratio"
    DEPLETION_CAPTURE = "depletion_capture"
    EMERGENCE_CAPTURE = "emergence_capture"
    ENDPOINT_Y_MAE = "endpoint_y_MAE"
    A_MAE_ACTIVE = "A_MAE_active"
    A_MSE_ACTIVE = "A_MSE_active"
    TARGET_RECALL_AT_K = "target_recall_at_k"
    OPEN_SUPPORT_F1 = "open_support_F1"
    D_MAE = "d_MAE"
    E_MAE = "e_MAE"
    D_MSE = "d_MSE"
    E_MSE = "e_MSE"
    PEARSON_CORRELATION = "Pearson correlation"
    MAE = "MAE"
    MSE = "MSE"
    JS_DIVERGENCE = "JS divergence"
    RERUN_VARIABILITY = "rerun variability"


class MetricStatus(str, Enum):
    """Allowed reporting statuses for Block 3 metric rows."""

    REPORTED = "reported"
    NOT_APPLICABLE = "not_applicable"
    NOT_ESTIMABLE = "not_estimable"


class MetricRole(str, Enum):
    """Routing roles that explain what each reported metric is measuring."""

    GENERATOR_VALIDATION = "generator_validation"
    STABILITY_SUMMARY = "stability_summary"
    PRIMARY_MASS = "primary_mass"
    PRIMARY_RATIO = "primary_ratio"
    SECONDARY_ENDPOINT = "secondary_endpoint"
    RELATION_RECOVERY = "relation_recovery"
    OPEN_PROFILE_RECOVERY = "open_profile_recovery"
    OPEN_SUPPORT_RECOVERY = "open_support_recovery"


class ValidationObjectId(str, Enum):
    """Frozen held-out validation objects used only by `3A`."""

    COMMUNITY_SPACE_TARGET = "community_space_target_fraction"
    IDENTITY_PROJECTED_TARGET = "identity_projected_target_fraction"


@dataclass(frozen=True)
class SectionGroupSpec:
    """Frozen grouping metadata for non-executable section containers such as `3C`."""

    section_id: str
    title: str
    child_subexperiments: tuple[str, ...]


@dataclass(frozen=True)
class MethodSpec:
    """Frozen routing spec for a live Block 3 method name."""

    name: Block3MethodName
    method_class: Block3MethodClass
    title: str
    allowed_subexperiments: tuple[str, ...]


@dataclass(frozen=True)
class MetricSpec:
    """Frozen routing and status contract for one Block 3 metric."""

    name: Block3MetricName
    role: MetricRole
    allowed_subexperiments: tuple[str, ...]
    allowed_statuses: tuple[MetricStatus, ...]


@dataclass(frozen=True)
class ConditionSpec:
    """Frozen condition-level routing record for one Block 3 evaluation condition."""

    condition_id: str
    subexperiment_id: str
    title: str
    evaluation_family: str
    is_public: bool


@dataclass(frozen=True)
class ValidationObjectSpec:
    """Frozen validation-object contract used by `3A` generator validation."""

    object_id: ValidationObjectId
    title: str
    metrics: tuple[Block3MetricName, ...]


@dataclass(frozen=True)
class SubexperimentSpec:
    """Frozen subexperiment contract for one executable Block 3 unit."""

    subexperiment_id: str
    title: str
    evaluation_family: str
    methods: tuple[Block3MethodName, ...]
    metrics: tuple[Block3MetricName, ...]
    condition_ids: tuple[str, ...]
    validation_objects: tuple[ValidationObjectId, ...] = ()


@dataclass(frozen=True)
class MetricValue:
    """Validated metric payload combining a metric name, value, and status."""

    metric_name: Block3MetricName
    value: float | None
    status: MetricStatus


@dataclass(frozen=True)
class Block3GeneratorObjectScoreRow:
    """Typed raw row for non-method-bearing `3A` validation-object scores."""

    rerun_id: str
    subexperiment_id: str
    condition_id: str
    evaluation_family: str
    validation_object_id: ValidationObjectId
    metric_role: MetricRole
    metric_value: MetricValue

    def to_record(self) -> dict[str, object]:
        """Convert a typed `3A` raw row into flat bundle columns.

        Purpose:
            Flatten the typed generator-object score row into the raw artifact
            schema used by `generator_validation_object_scores`.

        Inputs / Returns:
            Uses the current dataclass fields and returns a record that preserves
            rerun, subexperiment, condition, evaluation-family, validation
            object, and metric-routing columns.

        Core flow:
            1. Keep the shared routing fields unchanged.
            2. Serialize enum-valued fields to their public string values.
            3. Emit only raw bundle columns; validation is handled upstream.
        """
        return {
            "rerun_id": self.rerun_id,
            "subexperiment_id": self.subexperiment_id,
            "condition_id": self.condition_id,
            "evaluation_family": self.evaluation_family,
            "validation_object_id": self.validation_object_id.value,
            "metric_name": self.metric_value.metric_name.value,
            "metric_role": self.metric_role.value,
            "metric_status": self.metric_value.status.value,
            "reported_value": self.metric_value.value,
        }


@dataclass(frozen=True)
class Block3GeneratorStabilityRow:
    """Typed raw row for `3A` rerun-stability summaries."""

    rerun_id: str
    subexperiment_id: str
    condition_id: str
    evaluation_family: str
    validation_object_id: ValidationObjectId
    metric_role: MetricRole
    metric_value: MetricValue
    stability_summary_level: str

    def to_record(self) -> dict[str, object]:
        """Convert a typed `3A` stability row into flat bundle columns.

        Purpose:
            Flatten the rerun-stability row into the raw artifact schema used by
            `generator_validation_rerun_stability`.

        Inputs / Returns:
            Uses the current dataclass fields and returns a record that adds the
            stability summary level on top of the shared `3A` routing columns.

        Core flow:
            1. Keep the shared `3A` routing fields unchanged.
            2. Serialize enum-valued fields to artifact-safe strings.
            3. Append the stability-specific column and leave validation to the
               builder that created the row.
        """
        return {
            "rerun_id": self.rerun_id,
            "subexperiment_id": self.subexperiment_id,
            "condition_id": self.condition_id,
            "evaluation_family": self.evaluation_family,
            "validation_object_id": self.validation_object_id.value,
            "metric_name": self.metric_value.metric_name.value,
            "metric_role": self.metric_role.value,
            "metric_status": self.metric_value.status.value,
            "reported_value": self.metric_value.value,
            "stability_summary_level": self.stability_summary_level,
        }


@dataclass(frozen=True)
class Block3GeneratorTargetSurfaceProfileRow:
    """Typed raw row for `3A` per-patient target surface profiles."""

    rerun_id: str
    subexperiment_id: str
    condition_id: str
    evaluation_family: str
    patient_id: str
    split_role: str
    surface_source: str
    validation_object_id: ValidationObjectId
    state_id: int | str
    feature_index: int
    reported_value: float

    def to_record(self) -> dict[str, object]:
        """Convert a per-patient target surface profile row into flat columns."""

        return {
            "rerun_id": self.rerun_id,
            "subexperiment_id": self.subexperiment_id,
            "condition_id": self.condition_id,
            "evaluation_family": self.evaluation_family,
            "patient_id": self.patient_id,
            "split_role": self.split_role,
            "surface_source": self.surface_source,
            "validation_object_id": self.validation_object_id.value,
            "state_id": self.state_id,
            "feature_index": self.feature_index,
            "reported_value": self.reported_value,
        }


@dataclass(frozen=True)
class Block3PatientMetricRow:
    """Typed raw row for patient-level method-bearing metrics in `3B`/`3C-*`."""

    rerun_id: str
    subexperiment_id: str
    condition_id: str
    evaluation_family: str
    method_name: Block3MethodName
    method_class: Block3MethodClass
    metric_role: MetricRole
    metric_value: MetricValue
    patient_id: str
    open_mass_scale: float | None = None

    def to_record(self) -> dict[str, object]:
        """Convert a typed patient metric row into flat bundle columns.

        Purpose:
            Flatten one method-bearing patient-level row into the raw artifact
            schema used by `*_patient_metrics`.

        Inputs / Returns:
            Uses the current dataclass fields and returns a record that preserves
            rerun, condition, evaluation-family, method, metric, and patient
            routing columns.

        Core flow:
            1. Keep the shared method-bearing routing fields unchanged.
            2. Serialize enum-valued method and metric fields to strings.
            3. Emit only raw bundle columns; contract validation happens before
               row construction.
        """
        return {
            "rerun_id": self.rerun_id,
            "subexperiment_id": self.subexperiment_id,
            "condition_id": self.condition_id,
            "evaluation_family": self.evaluation_family,
            "method_name": self.method_name.value,
            "method_class": self.method_class.value,
            "metric_name": self.metric_value.metric_name.value,
            "metric_role": self.metric_role.value,
            "metric_status": self.metric_value.status.value,
            "reported_value": self.metric_value.value,
            "open_mass_scale": self.open_mass_scale,
            "patient_id": self.patient_id,
        }


@dataclass(frozen=True)
class Block3ConditionSummaryRow:
    """Typed raw row for condition-level summaries in method-bearing sections."""

    rerun_id: str
    subexperiment_id: str
    condition_id: str
    evaluation_family: str
    method_name: Block3MethodName
    method_class: Block3MethodClass
    metric_role: MetricRole
    metric_value: MetricValue
    summary_level: str
    mean_value: float | None
    ci_lower: float | None
    ci_upper: float | None
    paired_difference_vs_stride_reference: float | None
    open_mass_scale: float | None = None

    def to_record(self) -> dict[str, object]:
        """Convert a typed condition summary into flat bundle columns.

        Purpose:
            Flatten one method-bearing condition summary row into the raw
            artifact schema used by `*_condition_summary`.

        Inputs / Returns:
            Uses the current dataclass fields and returns a record that carries
            routing columns plus summary statistics and paired-difference fields.

        Core flow:
            1. Keep shared routing, method, and metric fields unchanged.
            2. Serialize enum-valued fields to artifact-safe strings.
            3. Append summary statistics without revalidating the already-built
               metric payload.
        """
        return {
            "rerun_id": self.rerun_id,
            "subexperiment_id": self.subexperiment_id,
            "condition_id": self.condition_id,
            "evaluation_family": self.evaluation_family,
            "method_name": self.method_name.value,
            "method_class": self.method_class.value,
            "metric_name": self.metric_value.metric_name.value,
            "metric_role": self.metric_role.value,
            "metric_status": self.metric_value.status.value,
            "reported_value": self.metric_value.value,
            "open_mass_scale": self.open_mass_scale,
            "summary_level": self.summary_level,
            "mean_value": self.mean_value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "paired_difference_vs_stride_reference": self.paired_difference_vs_stride_reference,
        }


@dataclass(frozen=True)
class Block3GeneratorReviewRow:
    """Typed review row for non-method-bearing `3A` review surfaces."""

    rerun_id: str
    subexperiment_id: str
    condition_id: str
    evaluation_family: str
    validation_object_id: ValidationObjectId
    metric_role: MetricRole
    metric_value: MetricValue
    stability_summary_level: str
    review_surface_role: str

    def to_record(self) -> dict[str, object]:
        """Convert a typed `3A` review row into flat review-surface columns.

        Purpose:
            Flatten a generator-validation review row into the schema used by
            `generator_validation_review_surface`.

        Inputs / Returns:
            Uses the current dataclass fields and returns a record that preserves
            `3A` routing columns, validation-object identity, and the review
            surface role.

        Core flow:
            1. Keep shared `3A` review-routing fields unchanged.
            2. Serialize enum-valued fields to strings.
            3. Append the review-surface role and stability-level fields without
               rechecking contract logic.
        """
        return {
            "rerun_id": self.rerun_id,
            "subexperiment_id": self.subexperiment_id,
            "condition_id": self.condition_id,
            "evaluation_family": self.evaluation_family,
            "validation_object_id": self.validation_object_id.value,
            "metric_name": self.metric_value.metric_name.value,
            "metric_role": self.metric_role.value,
            "metric_status": self.metric_value.status.value,
            "reported_value": self.metric_value.value,
            "stability_summary_level": self.stability_summary_level,
            "review_surface_role": self.review_surface_role,
        }


@dataclass(frozen=True)
class Block3SectionReviewRow:
    """Typed review row for method-bearing section summaries."""

    rerun_id: str
    subexperiment_id: str
    condition_id: str
    evaluation_family: str
    method_name: Block3MethodName
    method_class: Block3MethodClass
    metric_role: MetricRole
    metric_value: MetricValue
    summary_level: str
    mean_value: float | None
    ci_lower: float | None
    ci_upper: float | None
    paired_difference_vs_stride_reference: float | None
    section_title: str
    condition_title: str
    review_surface_role: str
    open_mass_scale: float | None = None

    def to_record(self) -> dict[str, object]:
        """Convert a typed section review row into flat review-surface columns.

        Purpose:
            Flatten one method-bearing summary row into the schema used by
            semantic Block 3 review-surface artifact
            surface.

        Inputs / Returns:
            Uses the current dataclass fields and returns a record that keeps the
            routing, summary-statistic, and review-label columns required by the
            review layer.

        Core flow:
            1. Keep the method-bearing routing and summary fields unchanged.
            2. Serialize enum-valued method and metric fields to strings.
            3. Emit review-facing flat columns only; this method does not decide
               section titles or paired-difference semantics.
        """
        return {
            "rerun_id": self.rerun_id,
            "subexperiment_id": self.subexperiment_id,
            "condition_id": self.condition_id,
            "evaluation_family": self.evaluation_family,
            "method_name": self.method_name.value,
            "method_class": self.method_class.value,
            "metric_name": self.metric_value.metric_name.value,
            "metric_role": self.metric_role.value,
            "metric_status": self.metric_value.status.value,
            "reported_value": self.metric_value.value,
            "open_mass_scale": self.open_mass_scale,
            "summary_level": self.summary_level,
            "mean_value": self.mean_value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "paired_difference_vs_stride_reference": self.paired_difference_vs_stride_reference,
            "section_title": self.section_title,
            "condition_title": self.condition_title,
            "review_surface_role": self.review_surface_role,
        }


@dataclass(frozen=True)
class Block3SubexperimentRawRows:
    """Container grouping the raw row families emitted by one subexperiment."""

    object_scores: tuple[Block3GeneratorObjectScoreRow, ...] = ()
    rerun_stability: tuple[Block3GeneratorStabilityRow, ...] = ()
    target_surface_profiles: tuple[Block3GeneratorTargetSurfaceProfileRow, ...] = ()
    patient_metrics: tuple[Block3PatientMetricRow, ...] = ()
    condition_summaries: tuple[Block3ConditionSummaryRow, ...] = ()
    shared_tables: dict[str, tuple[dict[str, object], ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Block3SubexperimentReviewRows:
    """Container grouping the review row families emitted by one subexperiment."""

    generator_rows: tuple[Block3GeneratorReviewRow, ...] = ()
    section_rows: tuple[Block3SectionReviewRow, ...] = ()


def ensure_metric_status(value: str | MetricStatus) -> MetricStatus:
    """Normalize a raw metric-status value to the frozen Block 3 enum.

    Purpose:
        Convert user- or artifact-provided metric-status values into the frozen
        `MetricStatus` enum used across the internal Block 3 stack.

    Inputs:
        value: A status value that should already belong to the live Block 3
            metric-status vocabulary.

    Returns:
        The normalized `MetricStatus` enum that downstream builders and writers
        can rely on without doing string checks again.

    Raises:
        ContractError: The provided value is not part of the frozen Block 3
            metric-status vocabulary.

    Core flow:
        1. Return the value unchanged when it is already a `MetricStatus`.
        2. Convert string-like inputs into the enum namespace.
        3. Fail fast if the input falls outside the frozen contract.

    Notes:
        This helper validates only the status token itself. It does not decide
        whether a given metric is allowed to use that status.
    """
    if isinstance(value, MetricStatus):
        return value
    try:
        return MetricStatus(str(value))
    except ValueError as exc:
        raise ContractError(f"Unknown Block 3 metric_status: {value!r}") from exc


def make_metric_value(
    *,
    metric_name: Block3MetricName,
    value: float | None,
    status: MetricStatus | str,
) -> MetricValue:
    """Build a validated metric payload for Block 3 row objects.

    Purpose:
        Construct a `MetricValue` object that already respects the invariant
        linking metric status to numeric nullability.

    Inputs:
        metric_name: Frozen metric identifier being reported.
        value: Numeric payload for reported metrics, or `None` for non-reported
            statuses.
        status: Metric-status token that should already belong to the live Block
            3 vocabulary.

    Returns:
        A `MetricValue` instance that row builders can embed directly in typed
        raw or review rows.

    Raises:
        ContractError: A reported metric is missing a numeric value, or a
            non-reported metric carries a numeric payload.

    Core flow:
        1. Normalize the incoming status with `ensure_metric_status()`.
        2. Require a numeric payload when the status is `reported`.
        3. Require `None` when the status is `not_applicable` or
           `not_estimable`.
        4. Return a validated `MetricValue` object for downstream row builders.

    Notes:
        This helper enforces status/nullability only. Subexperiment-specific
        metric applicability is checked elsewhere.
    """
    resolved_status = ensure_metric_status(status)
    if resolved_status is MetricStatus.REPORTED:
        if value is None:
            raise ContractError("Block 3 reported metrics require a numeric value")
        numeric_value = float(value)
        return MetricValue(metric_name=metric_name, value=numeric_value, status=resolved_status)
    if value is not None:
        raise ContractError("Block 3 non-reported metrics must keep value=None")
    return MetricValue(metric_name=metric_name, value=None, status=resolved_status)


__all__ = [
    "Block3ConditionSummaryRow",
    "Block3GeneratorObjectScoreRow",
    "Block3GeneratorReviewRow",
    "Block3GeneratorStabilityRow",
    "Block3GeneratorTargetSurfaceProfileRow",
    "Block3MethodClass",
    "Block3MethodName",
    "Block3MetricName",
    "Block3PatientMetricRow",
    "Block3SectionReviewRow",
    "Block3SubexperimentId",
    "Block3SubexperimentRawRows",
    "Block3SubexperimentReviewRows",
    "ConditionSpec",
    "MetricRole",
    "MetricSpec",
    "MetricStatus",
    "MetricValue",
    "MethodSpec",
    "SectionGroupSpec",
    "SubexperimentSpec",
    "ValidationObjectId",
    "ValidationObjectSpec",
    "ensure_metric_status",
    "make_metric_value",
]
