"""Live Block 3 registry for the internal Block 3 rebuild surface.

Role:
    Hold the frozen lookup inventory for executable Block 3 subexperiments,
    methods, metrics, conditions, and validation objects.

Authority anchors:
    - docs/task_A_spec.md §4.5.2, §4.5.3, §4.5.5, §4.5.6
    - docs/task_A_block3_redesign_v1_1.md §4.1-§4.4, §5.3, §5.5, §5.6

Local boundary:
    - This module provides live routing inventory only.
    - It does not execute subexperiments, write artifacts, or define review
      prose.
    - It does not provide alias compatibility or historical naming fallback.

Primary contents:
    - The frozen `Block3Registry` container.
    - The builder for the live Block 3 registry payload.
    - Getter helpers that fail fast when callers request unsupported routing
      names.

Why this module exists:
    The execution, bundle, and review layers all need the same route inventory
    but should not each re-specify scientific names and applicability rules.
    Centralizing those lookups here keeps the contract declarative and makes
    execution modules depend on frozen data rather than duplicated constants.
"""
from __future__ import annotations

from dataclasses import dataclass

from stride.errors import ContractError

from .contracts import (
    Block3MethodClass,
    Block3MethodName,
    Block3MetricName,
    Block3SubexperimentId,
    ConditionSpec,
    MetricRole,
    MetricSpec,
    MetricStatus,
    MethodSpec,
    SectionGroupSpec,
    SubexperimentSpec,
    ValidationObjectId,
    ValidationObjectSpec,
)


@dataclass(frozen=True)
class Block3Registry:
    """Frozen Block 3 lookup inventory.

    The registry keeps separate dictionaries for non-executable section groups,
    executable subexperiments, methods, metrics, conditions, and `3A`
    validation objects so that callers can ask for one routing layer without
    implicitly traversing the others.
    """

    section_groups: dict[str, SectionGroupSpec]
    subexperiments: dict[str, SubexperimentSpec]
    methods: dict[str, MethodSpec]
    metrics: dict[str, MetricSpec]
    conditions: dict[str, ConditionSpec]
    validation_objects: dict[str, ValidationObjectSpec]


def build_live_block3_registry() -> Block3Registry:
    """Build the frozen live Block 3 routing inventory.

    Purpose:
        Assemble the authoritative in-memory registry used by the internal Block
        3 implementation surface.

    Inputs:
        This function has no runtime inputs. It materializes the frozen Block 3
        routing inventory directly from the implementation-carried contract.

    Returns:
        A `Block3Registry` containing the live section groups, executable
        subexperiments, methods, metrics, conditions, and validation objects.

    Raises:
        ContractError: Duplicate names appear in the assembled registry.

    Core flow:
        1. Define the non-executable section-group inventory.
        2. Define the live method registry for method-bearing sections.
        3. Define the metric registry with role and status applicability.
        4. Define condition and validation-object routing records.
        5. Define executable subexperiments that reference those frozen names.
        6. Assemble the `Block3Registry` and run uniqueness checks.

    Notes:
        This builder defines only the live routing inventory. It does not carry
        compatibility aliases or historical naming fallbacks.
    """
    section_groups = {
        Block3SubexperimentId.BASELINE_COMPARISON.value: SectionGroupSpec(
            section_id=Block3SubexperimentId.BASELINE_COMPARISON.value,
            title="3B baseline benchmark umbrella",
            child_subexperiments=(
                Block3SubexperimentId.A_BENCHMARK.value,
                Block3SubexperimentId.DE_BENCHMARK.value,
            ),
        ),
        Block3SubexperimentId.ABLATION_STUDY.value: SectionGroupSpec(
            section_id=Block3SubexperimentId.ABLATION_STUDY.value,
            title="3C ablation study",
            child_subexperiments=(
                Block3SubexperimentId.OPEN_MODULE_ABLATION.value,
                Block3SubexperimentId.COHORT_MODULE_ABLATION.value,
            ),
        )
    }
    methods = {
        Block3MethodName.STRIDE_REFERENCE.value: MethodSpec(
            name=Block3MethodName.STRIDE_REFERENCE,
            method_class=Block3MethodClass.REFERENCE,
            title="stride_reference",
            allowed_subexperiments=(
                Block3SubexperimentId.A_BENCHMARK.value,
                Block3SubexperimentId.DE_BENCHMARK.value,
                Block3SubexperimentId.OPEN_MODULE_ABLATION.value,
                Block3SubexperimentId.COHORT_MODULE_ABLATION.value,
            ),
        ),
        Block3MethodName.BALANCED_OT_BASELINE.value: MethodSpec(
            name=Block3MethodName.BALANCED_OT_BASELINE,
            method_class=Block3MethodClass.BASELINE,
            title="balanced_ot_baseline",
            allowed_subexperiments=(Block3SubexperimentId.A_BENCHMARK.value,),
        ),
        Block3MethodName.UOT_BASELINE.value: MethodSpec(
            name=Block3MethodName.UOT_BASELINE,
            method_class=Block3MethodClass.BASELINE,
            title="uot_baseline",
            allowed_subexperiments=(
                Block3SubexperimentId.A_BENCHMARK.value,
                Block3SubexperimentId.DE_BENCHMARK.value,
            ),
        ),
        Block3MethodName.PARTIAL_OT_BASELINE.value: MethodSpec(
            name=Block3MethodName.PARTIAL_OT_BASELINE,
            method_class=Block3MethodClass.BASELINE,
            title="partial_ot_baseline",
            allowed_subexperiments=(
                Block3SubexperimentId.A_BENCHMARK.value,
                Block3SubexperimentId.DE_BENCHMARK.value,
            ),
        ),
        Block3MethodName.DIAGONAL_TRANSPORT_BASELINE.value: MethodSpec(
            name=Block3MethodName.DIAGONAL_TRANSPORT_BASELINE,
            method_class=Block3MethodClass.BASELINE,
            title="diagonal_transport_baseline",
            allowed_subexperiments=(
                Block3SubexperimentId.A_BENCHMARK.value,
                Block3SubexperimentId.DE_BENCHMARK.value,
            ),
        ),
        Block3MethodName.OPEN_CHANNEL_ABLATION.value: MethodSpec(
            name=Block3MethodName.OPEN_CHANNEL_ABLATION,
            method_class=Block3MethodClass.ABLATION,
            title="open_channel_ablation",
            allowed_subexperiments=(Block3SubexperimentId.OPEN_MODULE_ABLATION.value,),
        ),
        Block3MethodName.COHORT_ABLATION.value: MethodSpec(
            name=Block3MethodName.COHORT_ABLATION,
            method_class=Block3MethodClass.ABLATION,
            title="cohort_ablation",
            allowed_subexperiments=(Block3SubexperimentId.COHORT_MODULE_ABLATION.value,),
        ),
    }
    metrics = {
        Block3MetricName.A_MAE_ACTIVE.value: MetricSpec(
            name=Block3MetricName.A_MAE_ACTIVE,
            role=MetricRole.RELATION_RECOVERY,
            allowed_subexperiments=(
                Block3SubexperimentId.A_BENCHMARK.value,
                Block3SubexperimentId.COHORT_MODULE_ABLATION.value,
            ),
            allowed_statuses=(
                MetricStatus.REPORTED,
                MetricStatus.NOT_APPLICABLE,
                MetricStatus.NOT_ESTIMABLE,
            ),
        ),
        Block3MetricName.A_MSE_ACTIVE.value: MetricSpec(
            name=Block3MetricName.A_MSE_ACTIVE,
            role=MetricRole.RELATION_RECOVERY,
            allowed_subexperiments=(
                Block3SubexperimentId.A_BENCHMARK.value,
                Block3SubexperimentId.COHORT_MODULE_ABLATION.value,
            ),
            allowed_statuses=(
                MetricStatus.REPORTED,
                MetricStatus.NOT_APPLICABLE,
                MetricStatus.NOT_ESTIMABLE,
            ),
        ),
        Block3MetricName.TARGET_RECALL_AT_K.value: MetricSpec(
            name=Block3MetricName.TARGET_RECALL_AT_K,
            role=MetricRole.RELATION_RECOVERY,
            allowed_subexperiments=(Block3SubexperimentId.A_BENCHMARK.value,),
            allowed_statuses=(
                MetricStatus.REPORTED,
                MetricStatus.NOT_APPLICABLE,
                MetricStatus.NOT_ESTIMABLE,
            ),
        ),
        Block3MetricName.OPEN_SUPPORT_F1.value: MetricSpec(
            name=Block3MetricName.OPEN_SUPPORT_F1,
            role=MetricRole.OPEN_SUPPORT_RECOVERY,
            allowed_subexperiments=(
                Block3SubexperimentId.DE_BENCHMARK.value,
                Block3SubexperimentId.OPEN_MODULE_ABLATION.value,
                Block3SubexperimentId.COHORT_MODULE_ABLATION.value,
            ),
            allowed_statuses=(
                MetricStatus.REPORTED,
                MetricStatus.NOT_APPLICABLE,
                MetricStatus.NOT_ESTIMABLE,
            ),
        ),
        Block3MetricName.D_MAE.value: MetricSpec(
            name=Block3MetricName.D_MAE,
            role=MetricRole.OPEN_PROFILE_RECOVERY,
            allowed_subexperiments=(
                Block3SubexperimentId.DE_BENCHMARK.value,
                Block3SubexperimentId.OPEN_MODULE_ABLATION.value,
                Block3SubexperimentId.COHORT_MODULE_ABLATION.value,
            ),
            allowed_statuses=(MetricStatus.REPORTED, MetricStatus.NOT_ESTIMABLE),
        ),
        Block3MetricName.E_MAE.value: MetricSpec(
            name=Block3MetricName.E_MAE,
            role=MetricRole.OPEN_PROFILE_RECOVERY,
            allowed_subexperiments=(
                Block3SubexperimentId.DE_BENCHMARK.value,
                Block3SubexperimentId.OPEN_MODULE_ABLATION.value,
                Block3SubexperimentId.COHORT_MODULE_ABLATION.value,
            ),
            allowed_statuses=(MetricStatus.REPORTED, MetricStatus.NOT_ESTIMABLE),
        ),
        Block3MetricName.D_MSE.value: MetricSpec(
            name=Block3MetricName.D_MSE,
            role=MetricRole.OPEN_PROFILE_RECOVERY,
            allowed_subexperiments=(
                Block3SubexperimentId.DE_BENCHMARK.value,
                Block3SubexperimentId.OPEN_MODULE_ABLATION.value,
                Block3SubexperimentId.COHORT_MODULE_ABLATION.value,
            ),
            allowed_statuses=(MetricStatus.REPORTED, MetricStatus.NOT_ESTIMABLE),
        ),
        Block3MetricName.E_MSE.value: MetricSpec(
            name=Block3MetricName.E_MSE,
            role=MetricRole.OPEN_PROFILE_RECOVERY,
            allowed_subexperiments=(
                Block3SubexperimentId.DE_BENCHMARK.value,
                Block3SubexperimentId.OPEN_MODULE_ABLATION.value,
                Block3SubexperimentId.COHORT_MODULE_ABLATION.value,
            ),
            allowed_statuses=(MetricStatus.REPORTED, MetricStatus.NOT_ESTIMABLE),
        ),
        Block3MetricName.PEARSON_CORRELATION.value: MetricSpec(
            name=Block3MetricName.PEARSON_CORRELATION,
            role=MetricRole.GENERATOR_VALIDATION,
            allowed_subexperiments=(Block3SubexperimentId.GENERATOR_VALIDATION.value,),
            allowed_statuses=(MetricStatus.REPORTED, MetricStatus.NOT_ESTIMABLE),
        ),
        Block3MetricName.MAE.value: MetricSpec(
            name=Block3MetricName.MAE,
            role=MetricRole.GENERATOR_VALIDATION,
            allowed_subexperiments=(Block3SubexperimentId.GENERATOR_VALIDATION.value,),
            allowed_statuses=(MetricStatus.REPORTED, MetricStatus.NOT_ESTIMABLE),
        ),
        Block3MetricName.MSE.value: MetricSpec(
            name=Block3MetricName.MSE,
            role=MetricRole.GENERATOR_VALIDATION,
            allowed_subexperiments=(Block3SubexperimentId.GENERATOR_VALIDATION.value,),
            allowed_statuses=(MetricStatus.REPORTED, MetricStatus.NOT_ESTIMABLE),
        ),
        Block3MetricName.JS_DIVERGENCE.value: MetricSpec(
            name=Block3MetricName.JS_DIVERGENCE,
            role=MetricRole.GENERATOR_VALIDATION,
            allowed_subexperiments=(Block3SubexperimentId.GENERATOR_VALIDATION.value,),
            allowed_statuses=(MetricStatus.REPORTED, MetricStatus.NOT_ESTIMABLE),
        ),
        Block3MetricName.RERUN_VARIABILITY.value: MetricSpec(
            name=Block3MetricName.RERUN_VARIABILITY,
            role=MetricRole.STABILITY_SUMMARY,
            allowed_subexperiments=(Block3SubexperimentId.GENERATOR_VALIDATION.value,),
            allowed_statuses=(MetricStatus.REPORTED, MetricStatus.NOT_ESTIMABLE),
        ),
    }
    conditions = {
        "generator_validation": ConditionSpec(
            condition_id="generator_validation",
            subexperiment_id=Block3SubexperimentId.GENERATOR_VALIDATION.value,
            title="held_out_generator_validation",
            evaluation_family="generator_validation",
            is_public=False,
        ),
        "relation_null": ConditionSpec(
            condition_id="relation_null",
            subexperiment_id=Block3SubexperimentId.A_BENCHMARK.value,
            title="relation_null",
            evaluation_family="baseline_comparison",
            is_public=True,
        ),
        "relation_weak": ConditionSpec(
            condition_id="relation_weak",
            subexperiment_id=Block3SubexperimentId.A_BENCHMARK.value,
            title="relation_weak",
            evaluation_family="baseline_comparison",
            is_public=True,
        ),
        "relation_mid": ConditionSpec(
            condition_id="relation_mid",
            subexperiment_id=Block3SubexperimentId.A_BENCHMARK.value,
            title="relation_mid",
            evaluation_family="baseline_comparison",
            is_public=True,
        ),
        "relation_strong": ConditionSpec(
            condition_id="relation_strong",
            subexperiment_id=Block3SubexperimentId.A_BENCHMARK.value,
            title="relation_strong",
            evaluation_family="baseline_comparison",
            is_public=True,
        ),
        "open_mass_scale_grid": ConditionSpec(
            condition_id="open_mass_scale_grid",
            subexperiment_id=Block3SubexperimentId.DE_BENCHMARK.value,
            title="open_mass_scale_grid",
            evaluation_family="baseline_open_benchmark",
            is_public=True,
        ),
        "open_module_shared_realization_set": ConditionSpec(
            condition_id="open_module_shared_realization_set",
            subexperiment_id=Block3SubexperimentId.OPEN_MODULE_ABLATION.value,
            title="matched_open_module_realization_set",
            evaluation_family="open_module_ablation",
            is_public=False,
        ),
        "cohort_module_shared_realization_set": ConditionSpec(
            condition_id="cohort_module_shared_realization_set",
            subexperiment_id=Block3SubexperimentId.COHORT_MODULE_ABLATION.value,
            title="matched_cohort_module_realization_set",
            evaluation_family="cohort_module_ablation",
            is_public=False,
        ),
    }
    validation_objects = {
        ValidationObjectId.COMMUNITY_SPACE_TARGET.value: ValidationObjectSpec(
            object_id=ValidationObjectId.COMMUNITY_SPACE_TARGET,
            title="community_space_target_fraction",
            metrics=(
                Block3MetricName.PEARSON_CORRELATION,
                Block3MetricName.MAE,
                Block3MetricName.MSE,
                Block3MetricName.JS_DIVERGENCE,
            ),
        ),
        ValidationObjectId.IDENTITY_PROJECTED_TARGET.value: ValidationObjectSpec(
            object_id=ValidationObjectId.IDENTITY_PROJECTED_TARGET,
            title="identity_projected_target_fraction",
            metrics=(
                Block3MetricName.PEARSON_CORRELATION,
                Block3MetricName.MAE,
                Block3MetricName.MSE,
                Block3MetricName.JS_DIVERGENCE,
            ),
        ),
    }
    subexperiments = {
        Block3SubexperimentId.GENERATOR_VALIDATION.value: SubexperimentSpec(
            subexperiment_id=Block3SubexperimentId.GENERATOR_VALIDATION.value,
            title="3A generator validation",
            evaluation_family="generator_validation",
            methods=(),
            metrics=(
                Block3MetricName.PEARSON_CORRELATION,
                Block3MetricName.MAE,
                Block3MetricName.MSE,
                Block3MetricName.JS_DIVERGENCE,
                Block3MetricName.RERUN_VARIABILITY,
            ),
            condition_ids=("generator_validation",),
            validation_objects=(
                ValidationObjectId.COMMUNITY_SPACE_TARGET,
                ValidationObjectId.IDENTITY_PROJECTED_TARGET,
            ),
        ),
        Block3SubexperimentId.A_BENCHMARK.value: SubexperimentSpec(
            subexperiment_id=Block3SubexperimentId.A_BENCHMARK.value,
            title="3B-1 A benchmark",
            evaluation_family="baseline_comparison",
            methods=(
                Block3MethodName.STRIDE_REFERENCE,
                Block3MethodName.BALANCED_OT_BASELINE,
                Block3MethodName.UOT_BASELINE,
                Block3MethodName.PARTIAL_OT_BASELINE,
                Block3MethodName.DIAGONAL_TRANSPORT_BASELINE,
            ),
            metrics=(
                Block3MetricName.A_MAE_ACTIVE,
                Block3MetricName.A_MSE_ACTIVE,
                Block3MetricName.TARGET_RECALL_AT_K,
            ),
            condition_ids=("relation_null", "relation_weak", "relation_mid", "relation_strong"),
        ),
        Block3SubexperimentId.DE_BENCHMARK.value: SubexperimentSpec(
            subexperiment_id=Block3SubexperimentId.DE_BENCHMARK.value,
            title="3B-2 d/e benchmark",
            evaluation_family="baseline_open_benchmark",
            methods=(
                Block3MethodName.STRIDE_REFERENCE,
                Block3MethodName.UOT_BASELINE,
                Block3MethodName.PARTIAL_OT_BASELINE,
                Block3MethodName.DIAGONAL_TRANSPORT_BASELINE,
            ),
            metrics=(
                Block3MetricName.OPEN_SUPPORT_F1,
                Block3MetricName.D_MAE,
                Block3MetricName.E_MAE,
                Block3MetricName.D_MSE,
                Block3MetricName.E_MSE,
            ),
            condition_ids=("open_mass_scale_grid",),
        ),
        Block3SubexperimentId.OPEN_MODULE_ABLATION.value: SubexperimentSpec(
            subexperiment_id=Block3SubexperimentId.OPEN_MODULE_ABLATION.value,
            title="3C-1 open-module ablation",
            evaluation_family="open_module_ablation",
            methods=(
                Block3MethodName.STRIDE_REFERENCE,
                Block3MethodName.OPEN_CHANNEL_ABLATION,
            ),
            metrics=(
                Block3MetricName.OPEN_SUPPORT_F1,
                Block3MetricName.D_MAE,
                Block3MetricName.E_MAE,
                Block3MetricName.D_MSE,
                Block3MetricName.E_MSE,
            ),
            condition_ids=("open_module_shared_realization_set",),
        ),
        Block3SubexperimentId.COHORT_MODULE_ABLATION.value: SubexperimentSpec(
            subexperiment_id=Block3SubexperimentId.COHORT_MODULE_ABLATION.value,
            title="3C-2 cohort-module ablation",
            evaluation_family="cohort_module_ablation",
            methods=(
                Block3MethodName.STRIDE_REFERENCE,
                Block3MethodName.COHORT_ABLATION,
            ),
            metrics=(
                Block3MetricName.A_MAE_ACTIVE,
                Block3MetricName.A_MSE_ACTIVE,
                Block3MetricName.OPEN_SUPPORT_F1,
                Block3MetricName.D_MAE,
                Block3MetricName.E_MAE,
                Block3MetricName.D_MSE,
                Block3MetricName.E_MSE,
            ),
            condition_ids=("cohort_module_shared_realization_set",),
        ),
    }
    registry = Block3Registry(
        section_groups=section_groups,
        subexperiments=subexperiments,
        methods=methods,
        metrics=metrics,
        conditions=conditions,
        validation_objects=validation_objects,
    )
    _validate_uniqueness(registry)
    return registry


def _validate_uniqueness(registry: Block3Registry) -> None:
    """Check that the live registry does not reuse public identifiers.

    Purpose:
        Fail fast when the assembled registry accidentally reuses subexperiment,
        method, or metric names.

    Inputs / Returns:
        registry: Fully assembled `Block3Registry` to validate.
        Returns `None` after successful validation.

    Raises:
        ContractError: Any public routing namespace contains duplicate keys.

    Core flow:
        1. Compare the number of subexperiment keys to the corresponding set.
        2. Repeat the same uniqueness check for methods and metrics.
        3. Raise immediately on the first duplicate namespace.
    """
    if len(registry.subexperiments) != len(set(registry.subexperiments)):
        raise ContractError("Duplicate Block 3 subexperiment ids are not allowed")
    if len(registry.methods) != len(set(registry.methods)):
        raise ContractError("Duplicate Block 3 method names are not allowed")
    if len(registry.metrics) != len(set(registry.metrics)):
        raise ContractError("Duplicate Block 3 metric names are not allowed")


LIVE_BLOCK3_REGISTRY = build_live_block3_registry()


def get_live_block3_registry() -> Block3Registry:
    """Return the frozen live Block 3 registry.

    Purpose:
        Provide callers with the already-built registry singleton used by the
        internal Block 3 implementation stack.

    Inputs:
        This function has no runtime inputs.

    Returns:
        The frozen `LIVE_BLOCK3_REGISTRY` object.

    Core flow:
        1. Read the prebuilt module-level singleton.
        2. Return it unchanged to the caller.

    Notes:
        This accessor does not rebuild or mutate the registry.
    """
    return LIVE_BLOCK3_REGISTRY


def get_subexperiment_spec(subexperiment_id: str) -> SubexperimentSpec:
    """Resolve one executable Block 3 subexperiment spec.

    Purpose:
        Fetch the frozen routing spec for one executable subexperiment such as
        `3A`, `3B`, `3C-1`, or `3C-2`.

    Inputs:
        subexperiment_id: Public subexperiment identifier expected to belong to
            the live executable Block 3 surface.

    Returns:
        The frozen `SubexperimentSpec` requested by the caller.

    Raises:
        ContractError: The requested name is not part of the executable Block 3
            registry.

    Core flow:
        1. Load the live registry singleton.
        2. Look up the requested subexperiment id in the executable inventory.
        3. Raise immediately if the name is absent.
        4. Return the frozen spec unchanged.

    Notes:
        This getter does not provide alias compatibility for non-live names such
        as section-group labels or historical identifiers.
    """
    registry = get_live_block3_registry()
    if subexperiment_id not in registry.subexperiments:
        raise ContractError(
            f"{subexperiment_id!r} does not define an executable Block 3 subexperiment"
        )
    return registry.subexperiments[subexperiment_id]


def get_method_spec(method_name: str) -> MethodSpec:
    """Resolve one live Block 3 method spec.

    Purpose:
        Fetch the frozen routing spec for one live method-bearing Block 3 method
        name.

    Inputs:
        method_name: Public method identifier expected to belong to the live
            Block 3 method registry.

    Returns:
        The frozen `MethodSpec` requested by the caller.

    Raises:
        ContractError: The requested method name is not part of the live method
            inventory.

    Core flow:
        1. Load the live registry singleton.
        2. Look up the requested method in the method inventory.
        3. Raise immediately if the name is absent.
        4. Return the frozen spec unchanged.

    Notes:
        This getter does not perform alias compatibility or method-class
        inference outside the live registry.
    """
    registry = get_live_block3_registry()
    if method_name not in registry.methods:
        raise ContractError(f"Unknown Block 3 method {method_name!r}")
    return registry.methods[method_name]


def get_metric_spec(metric_name: str) -> MetricSpec:
    """Resolve one live Block 3 metric spec.

    Purpose:
        Fetch the frozen metric routing record that defines role and allowed
        statuses for one Block 3 metric name.

    Inputs:
        metric_name: Public metric identifier expected to belong to the live
            Block 3 metric registry.

    Returns:
        The frozen `MetricSpec` requested by the caller.

    Raises:
        ContractError: The metric name is not part of the live metric inventory.

    Core flow:
        1. Load the live registry singleton.
        2. Look up the requested metric in the metric inventory.
        3. Raise immediately if the metric is absent.
        4. Return the frozen spec unchanged.

    Notes:
        This getter validates name presence only. It does not decide whether the
        metric is appropriate for a specific row instance.
    """
    registry = get_live_block3_registry()
    if metric_name not in registry.metrics:
        raise ContractError(f"Unknown Block 3 metric {metric_name!r}")
    return registry.metrics[metric_name]


def get_condition_spec(condition_id: str) -> ConditionSpec:
    """Resolve one live Block 3 condition spec.

    Purpose:
        Fetch the frozen routing record for one declared Block 3 condition.

    Inputs:
        condition_id: Public condition identifier expected to belong to the live
            Block 3 condition registry.

    Returns:
        The frozen `ConditionSpec` requested by the caller.

    Raises:
        ContractError: The condition id is not part of the live condition
            inventory.

    Core flow:
        1. Load the live registry singleton.
        2. Look up the requested condition in the condition inventory.
        3. Raise immediately if the condition is absent.
        4. Return the frozen spec unchanged.

    Notes:
        This getter does not derive conditions from evaluation-family labels or
        historical names.
    """
    registry = get_live_block3_registry()
    if condition_id not in registry.conditions:
        raise ContractError(f"Unknown Block 3 condition {condition_id!r}")
    return registry.conditions[condition_id]


def get_validation_object_spec(object_id: str) -> ValidationObjectSpec:
    """Resolve one `3A` validation-object spec.

    Purpose:
        Fetch the frozen validation-object record used by the non-method-bearing
        `3A` generator-validation surface.

    Inputs:
        object_id: Public validation-object identifier expected to belong to the
            live `3A` validation inventory.

    Returns:
        The frozen `ValidationObjectSpec` requested by the caller.

    Raises:
        ContractError: The validation-object id is not part of the live `3A`
            validation inventory.

    Core flow:
        1. Load the live registry singleton.
        2. Look up the requested validation object.
        3. Raise immediately if the object is absent.
        4. Return the frozen spec unchanged.

    Notes:
        This getter is specific to `3A`; method-bearing sections do not consume
        this routing layer.
    """
    registry = get_live_block3_registry()
    if object_id not in registry.validation_objects:
        raise ContractError(f"Unknown Block 3 validation object {object_id!r}")
    return registry.validation_objects[object_id]


def get_metric_specs_for_subexperiment(subexperiment_id: str) -> tuple[MetricSpec, ...]:
    """Resolve the ordered metric specs for one executable subexperiment.

    Purpose:
        Convert a subexperiment-level metric list into the corresponding frozen
        `MetricSpec` objects used by downstream execution or writer logic.

    Inputs:
        subexperiment_id: Executable Block 3 subexperiment name whose metric
            surface should be expanded.

    Returns:
        A tuple of `MetricSpec` objects ordered exactly as the subexperiment
        declares them.

    Raises:
        ContractError: The subexperiment name is not part of the live executable
            registry.

    Core flow:
        1. Resolve the requested subexperiment spec.
        2. Load the live registry singleton.
        3. Expand the subexperiment's metric names into `MetricSpec` objects in
           declared order.
        4. Return the ordered tuple for downstream consumers.

    Notes:
        This helper preserves subexperiment ordering. It does not filter metrics
        by row status or per-method applicability.
    """
    subexperiment = get_subexperiment_spec(subexperiment_id)
    registry = get_live_block3_registry()
    return tuple(registry.metrics[metric_name.value] for metric_name in subexperiment.metrics)


__all__ = [
    "Block3Registry",
    "LIVE_BLOCK3_REGISTRY",
    "build_live_block3_registry",
    "get_condition_spec",
    "get_live_block3_registry",
    "get_method_spec",
    "get_metric_spec",
    "get_metric_specs_for_subexperiment",
    "get_subexperiment_spec",
    "get_validation_object_spec",
]
