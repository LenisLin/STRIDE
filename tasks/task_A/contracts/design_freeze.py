"""Frozen Task A engineering contracts for the current active Task A surface.

This module is the machine-readable source of truth for the *active* Task A
surface responsibilities, execution order, and artifact expectations. The
scientific Block 3 contract still lives in the docs hierarchy, but the polluted
local Block 3 engineering surface was removed from the active path and is not
enumerated here as an executable contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .artifact_states import (
    CONTRACT_PASSED_STATE,
    EVIDENCE_READY_STATE,
    SCAFFOLD_ACTIVE_STATE,
)


SURFACE_EXECUTION_STATUS: frozenset[str] = frozenset(
    {"executable", "scaffold_only", "helper_only", "pending_implementation"}
)
GRAPH_EXECUTION_STATUS: frozenset[str] = frozenset(
    {"executable", "scaffold_only", "blocked_by_deferred_surface", "pending_implementation"}
)
READINESS_CLASSIFICATIONS: frozenset[str] = frozenset(
    {
        SCAFFOLD_ACTIVE_STATE,
        CONTRACT_PASSED_STATE,
        EVIDENCE_READY_STATE,
        "descriptive_only",
        "calibration_ready_or_diagnostic",
        "pending_implementation",
    }
)
_BLOCK0_SUMMARY_NAMES: tuple[str, ...] = (
    "self_retention",
    "depletion",
    "off_diagonal_remodeling",
    "emergence",
)
_BLOCK0_REFERENCE_STATS: tuple[str, ...] = ("median", "mean")
_BLOCK0_PATIENT_CALIBRATION_FIELDS: tuple[str, ...] = (
    "patient_id",
    "run_scope",
    "real_family",
    "null_family",
    "n_permutations",
    "real_fit_status",
    "null_fit_status",
    "summary_name",
    "summary_role",
    "eligible_entity_axis",
    "scale",
    "reference_stat",
    "expected_tail",
    "real_value",
    "null_reference",
    "empirical_p_value",
    "primary_tail_fraction",
    "opposite_tail_fraction",
    "effect_delta",
    "effect_ratio",
    "effect_ratio_status",
    "readiness_status",
)
_BLOCK0_METRIC_SUMMARY_FIELDS: tuple[str, ...] = (
    "summary_name",
    "summary_role",
    "eligible_entity_axis",
    "scale",
    "cohort_stat",
    "expected_tail",
    "real_value",
    "null_reference",
    "empirical_p_value",
    "primary_tail_fraction",
    "opposite_tail_fraction",
    "effect_delta",
    "effect_ratio",
    "effect_ratio_status",
    "n_patient_delta_positive",
    "n_patient_delta_negative",
    "n_patient_delta_zero",
    "readiness_status",
)


@dataclass(frozen=True)
class TaskASurfaceSpec:
    name: str
    owner: str
    kind: str
    consumes: tuple[str, ...]
    produces: tuple[str, ...]
    emitted_artifact_states: tuple[str, ...]
    execution_status: str
    does_not_do: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "owner": self.owner,
            "kind": self.kind,
            "consumes": list(self.consumes),
            "produces": list(self.produces),
            "emitted_artifact_states": list(self.emitted_artifact_states),
            "execution_status": self.execution_status,
            "does_not_do": list(self.does_not_do),
        }


@dataclass(frozen=True)
class TaskAExecutionNode:
    name: str
    surface_name: str
    canonical_order: int
    canonical_predecessors: tuple[str, ...]
    hard_prerequisites: tuple[str, ...]
    produced_artifacts: tuple[str, ...]
    execution_status: str
    blocker_reason: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "surface_name": self.surface_name,
            "canonical_order": self.canonical_order,
            "canonical_predecessors": list(self.canonical_predecessors),
            "hard_prerequisites": list(self.hard_prerequisites),
            "produced_artifacts": list(self.produced_artifacts),
            "execution_status": self.execution_status,
        }
        if self.blocker_reason is not None:
            payload["blocker_reason"] = self.blocker_reason
        return payload


@dataclass(frozen=True)
class TaskAArtifactSpec:
    name: str
    filename: str
    producer: str
    purpose: str
    minimum_fields: tuple[str, ...]
    artifact_state_location: str
    allowed_artifact_states: tuple[str, ...]
    readiness_classification: str
    does_not_mean: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "filename": self.filename,
            "producer": self.producer,
            "purpose": self.purpose,
            "minimum_fields": list(self.minimum_fields),
            "artifact_state_location": self.artifact_state_location,
            "allowed_artifact_states": list(self.allowed_artifact_states),
            "readiness_classification": self.readiness_classification,
            "does_not_mean": list(self.does_not_mean),
        }


TASK_A_SURFACE_SPECS: tuple[TaskASurfaceSpec, ...] = (
    TaskASurfaceSpec(
        name="build_stage0_artifacts",
        owner="tasks.task_A.stage0.build_artifacts",
        kind="builder",
        consumes=("CRLM cohort RDS", "Stage 0 build parameters"),
        produces=("task_A_stage0_k{K}.h5ad", "task_A_stage0_validation.json"),
        emitted_artifact_states=(SCAFFOLD_ACTIVE_STATE, CONTRACT_PASSED_STATE),
        execution_status="executable",
        does_not_do=(
            "Run Step 1 prepare",
            "Run Block 0 or Block 1",
            "Emit scientific evidence claims",
        ),
    ),
    TaskASurfaceSpec(
        name="prepare_task_a_stage0_mapping",
        owner="tasks.task_A.workflows.prepare",
        kind="runner",
        consumes=(
            "task-config",
            "stage0-h5ad",
            "output-dir",
            "optional patient-id selectors",
            "optional demo-subset selector",
        ),
        produces=(
            "task_a_stride_mapping.json",
            "task_a_core_fit_dry_run.csv",
            "task_a_prepare_manifest.json",
        ),
        emitted_artifact_states=(SCAFFOLD_ACTIVE_STATE, CONTRACT_PASSED_STATE),
        execution_status="executable",
        does_not_do=(
            "Emit Block 0 calibration evidence",
            "Run block-local interpretation",
            "Create a shared task-global export index",
        ),
    ),
    TaskASurfaceSpec(
        name="write_task_a_descriptive_atlas",
        owner="tasks.task_A.descriptive",
        kind="runner",
        consumes=(
            "task-config",
            "stage0-h5ad",
            "output-dir",
            "optional patient-id selectors",
            "optional max-overlay-communities",
        ),
        produces=(
            "task_a_descriptive_atlas_manifest.json",
            "task_a_descriptive_atlas_output_index.csv",
            "descriptive atlas tables",
            "descriptive atlas figures",
        ),
        emitted_artifact_states=(),
        execution_status="executable",
        does_not_do=(
            "Consume Step 1 prepare artifacts as hard inputs",
            "Run Block 0 or Block 1",
            "Emit confirmatory or inferential claims",
        ),
    ),
    TaskASurfaceSpec(
        name="run_block0_execute",
        owner="tasks.task_A.block0",
        kind="runner",
        consumes=(
            "task-config",
            "stage0-h5ad",
            "output-dir",
            "n-permutations",
            "master-seed",
            "optional patient-id selectors",
            "optional demo-subset selector",
            "optional parallel controls",
            "optional resume flag",
        ),
        produces=(
            "block0_execution_manifest.json",
            "block0_fit_cache.npz",
            "block0_fit_cache_index.csv",
        ),
        emitted_artifact_states=(),
        execution_status="executable",
        does_not_do=(
            "Read Step 1 files as hard inputs",
            "Read descriptive-atlas files as hard inputs",
            "Read externally supplied state-basis artifacts as hard inputs",
            "Borrow TC/IM labels or observations across patients for the Block 0 null",
            "Alter patient_id, FOV composition, FOV count structure, mass, or mass_mode in the null",
            "Derive calibration metrics or p-values",
            "Emit interpretation prose",
            "Emit multiplicity-corrected significance decisions",
            "Emit downstream execution decisions",
            "Promote subset or diagnostic runs into full calibration evidence",
        ),
    ),
    TaskASurfaceSpec(
        name="analyze_block0_cache",
        owner="tasks.task_A.block0",
        kind="runner",
        consumes=(
            "block0_fit_cache.npz",
            "block0_fit_cache_index.csv",
            "optional block0_execution_manifest.json",
            "output-dir",
        ),
        produces=(
            "block0_calibration_manifest.json",
            "block0_patient_calibration.csv",
            "block0_metric_summary.csv",
        ),
        emitted_artifact_states=(),
        execution_status="executable",
        does_not_do=(
            "Call fit_stride",
            "Rerun real or null permutations",
            "Read Step 1 files as hard inputs",
            "Read descriptive-atlas files as hard inputs",
            "Emit interpretation prose",
            "Emit multiplicity-corrected significance decisions",
            "Emit downstream execution decisions",
        ),
    ),
    TaskASurfaceSpec(
        name="block1_real_data_discovery",
        owner="tasks.task_A.block1",
        kind="runner",
        consumes=("task-config", "stage0-h5ad", "output-dir"),
        produces=(
            "raw full STRIDE fit outputs",
            "raw recurrence/common-structure outputs",
            "block1_family_summary.csv",
            "block1_source_community_summary.csv",
            "block1_target_community_summary.csv",
            "block1_confirmatory_family_comparison.csv",
            "block1_source_community_comparison.csv",
            "block1_target_community_comparison.csv",
            "block1_cohort_relation_comparison.csv",
        ),
        emitted_artifact_states=(),
        execution_status="pending_implementation",
        does_not_do=(
            "Implement Block 0",
            "Read Block 0 outputs",
            "Read descriptive-atlas outputs",
            "Read result packets or proxy-history artifacts",
            "Emit figures, p-values, FDR calls, or significance labels",
            "Emit descriptive community annotation tables",
            "Emit interpretation prose",
            "Promote diagnostic subsets into formal evidence",
        ),
    ),
    TaskASurfaceSpec(
        name="write_semisynthetic_artifacts",
        owner="tasks.task_A.benchmarks.semisynthetic",
        kind="exporter",
        consumes=("output-root", "manifest filename", "n_patients", "seed"),
        produces=("semisynthetic manifest CSV", "task_a_semisynthetic_contract.json"),
        emitted_artifact_states=(CONTRACT_PASSED_STATE,),
        execution_status="executable",
        does_not_do=(
            "Run the real-data Task A pipeline",
            "Substitute for Block 0 or Block 1 evidence",
            "Define new STRIDE core semantics",
        ),
    ),
    TaskASurfaceSpec(
        name="TaskADemoSubset registry",
        owner="tasks.task_A.real_data.demo_subset",
        kind="registry",
        consumes=("demo subset name",),
        produces=("prepare manifest provenance fields",),
        emitted_artifact_states=(),
        execution_status="helper_only",
        does_not_do=(
            "Store derived data artifacts",
            "Filter data outside the prepare workflow",
            "Define a new real-data runner",
        ),
    ),
    TaskASurfaceSpec(
        name="stride_adapter helpers",
        owner="tasks.task_A.workflows.stride_adapter",
        kind="adapter",
        consumes=("Stage 0 h5ad or AnnData", "Task A config bundle"),
        produces=("mapping summaries", "family-sliced observations", "dry-run records"),
        emitted_artifact_states=(),
        execution_status="helper_only",
        does_not_do=(
            "Write task-global manifests on its own",
            "Move Task A semantics into src/stride/",
            "Redefine the STRIDE core method",
        ),
    ),
    TaskASurfaceSpec(
        name="Task A contract helpers",
        owner="tasks.task_A.contracts",
        kind="contract_helper",
        consumes=("bundle/manfiest payloads",),
        produces=("artifact-state validation", "mapping schemas", "design freeze registry"),
        emitted_artifact_states=(),
        execution_status="helper_only",
        does_not_do=(
            "Execute experiments",
            "Interpret scientific results",
            "Expose task-specific APIs from src/stride/",
        ),
    ),
)


TASK_A_EXECUTION_GRAPH: tuple[TaskAExecutionNode, ...] = (
    TaskAExecutionNode(
        name="stage0_artifact_builder",
        surface_name="build_stage0_artifacts",
        canonical_order=0,
        canonical_predecessors=(),
        hard_prerequisites=("CRLM cohort RDS",),
        produced_artifacts=("task_A_stage0_k{K}.h5ad", "task_A_stage0_validation.json"),
        execution_status="executable",
    ),
    TaskAExecutionNode(
        name="step1_prepare_full_cohort",
        surface_name="prepare_task_a_stage0_mapping",
        canonical_order=1,
        canonical_predecessors=("stage0_artifact_builder",),
        hard_prerequisites=("stage0-h5ad", "task-config"),
        produced_artifacts=(
            "task_a_stride_mapping.json",
            "task_a_core_fit_dry_run.csv",
            "task_a_prepare_manifest.json",
        ),
        execution_status="executable",
    ),
    TaskAExecutionNode(
        name="descriptive_atlas_context_layer",
        surface_name="write_task_a_descriptive_atlas",
        canonical_order=2,
        canonical_predecessors=("stage0_artifact_builder",),
        hard_prerequisites=("stage0-h5ad", "task-config"),
        produced_artifacts=(
            "task_a_descriptive_atlas_manifest.json",
            "task_a_descriptive_atlas_output_index.csv",
            "descriptive atlas tables",
            "descriptive atlas figures",
        ),
        execution_status="executable",
    ),
    TaskAExecutionNode(
        name="step1_prepare_subset_or_demo",
        surface_name="prepare_task_a_stage0_mapping",
        canonical_order=3,
        canonical_predecessors=("stage0_artifact_builder",),
        hard_prerequisites=("stage0-h5ad", "task-config", "patient-id or demo-subset"),
        produced_artifacts=(
            "task_a_stride_mapping.json",
            "task_a_core_fit_dry_run.csv",
            "task_a_prepare_manifest.json",
        ),
        execution_status="executable",
    ),
    TaskAExecutionNode(
        name="block0_execution_cache",
        surface_name="run_block0_execute",
        canonical_order=4,
        canonical_predecessors=("step1_prepare_full_cohort", "descriptive_atlas_context_layer"),
        hard_prerequisites=("stage0-h5ad", "task-config", "n-permutations", "master-seed"),
        produced_artifacts=(
            "block0_execution_manifest.json",
            "block0_fit_cache.npz",
            "block0_fit_cache_index.csv",
        ),
        execution_status="executable",
    ),
    TaskAExecutionNode(
        name="block0_calibration_analysis",
        surface_name="analyze_block0_cache",
        canonical_order=5,
        canonical_predecessors=("block0_execution_cache",),
        hard_prerequisites=("block0 fit cache",),
        produced_artifacts=(
            "block0_calibration_manifest.json",
            "block0_patient_calibration.csv",
            "block0_metric_summary.csv",
        ),
        execution_status="executable",
    ),
    TaskAExecutionNode(
        name="block1_real_data_discovery",
        surface_name="block1_real_data_discovery",
        canonical_order=6,
        canonical_predecessors=("block0_calibration_analysis",),
        hard_prerequisites=("stage0-h5ad", "task-config"),
        produced_artifacts=(
            "raw full STRIDE fit outputs",
            "raw recurrence/common-structure outputs",
            "block1_family_summary.csv",
            "block1_source_community_summary.csv",
            "block1_target_community_summary.csv",
            "block1_confirmatory_family_comparison.csv",
            "block1_source_community_comparison.csv",
            "block1_target_community_comparison.csv",
            "block1_cohort_relation_comparison.csv",
        ),
        execution_status="pending_implementation",
        blocker_reason="Block 1 execute/analyze implementation is deferred to the follow-up stage.",
    ),
    TaskAExecutionNode(
        name="semisynthetic_benchmark_export",
        surface_name="write_semisynthetic_artifacts",
        canonical_order=8,
        canonical_predecessors=(),
        hard_prerequisites=("benchmark settings",),
        produced_artifacts=("semisynthetic manifest CSV", "task_a_semisynthetic_contract.json"),
        execution_status="executable",
    ),
)


TASK_A_ARTIFACT_SPECS: tuple[TaskAArtifactSpec, ...] = (
    TaskAArtifactSpec(
        name="stage0_validation_report",
        filename="task_A_stage0_validation.json",
        producer="build_stage0_artifacts",
        purpose="Validate that a built Stage 0 h5ad satisfies the frozen Task A input contract.",
        minimum_fields=(
            "artifact_state",
            "taska_minimum_contract",
            "representation_completeness",
            "counts",
        ),
        artifact_state_location="embedded field",
        allowed_artifact_states=(SCAFFOLD_ACTIVE_STATE, CONTRACT_PASSED_STATE),
        readiness_classification=CONTRACT_PASSED_STATE,
        does_not_mean=(
            "Step 1 alignment already passed",
            "Block 0 calibration is complete",
        ),
    ),
    TaskAArtifactSpec(
        name="step1_stride_mapping",
        filename="task_a_stride_mapping.json",
        producer="prepare_task_a_stage0_mapping",
        purpose="Freeze the Stage 0 to STRIDE crosswalk and per-family eligibility summary.",
        minimum_fields=(
            "field_mapping",
            "patient_ids",
            "family_summaries",
            "real_data_crosswalk",
        ),
        artifact_state_location="task_a_prepare_manifest.json.artifact_state",
        allowed_artifact_states=(SCAFFOLD_ACTIVE_STATE, CONTRACT_PASSED_STATE),
        readiness_classification=CONTRACT_PASSED_STATE,
        does_not_mean=(
            "Block 0 calibration evidence exists",
            "Block 1 evidence is ready",
        ),
    ),
    TaskAArtifactSpec(
        name="step1_core_fit_dry_run",
        filename="task_a_core_fit_dry_run.csv",
        producer="prepare_task_a_stage0_mapping",
        purpose="Record task-local dry-run bridge statuses for confirmatory pair families.",
        minimum_fields=(
            "pair_family",
            "claim_role",
            "patient_id",
            "fit_status",
            "bridge_realized",
            "defer_reason",
            "uncertainty_status",
            "source_domain",
            "target_domain",
        ),
        artifact_state_location="task_a_prepare_manifest.json.artifact_state",
        allowed_artifact_states=(SCAFFOLD_ACTIVE_STATE, CONTRACT_PASSED_STATE),
        readiness_classification=CONTRACT_PASSED_STATE,
        does_not_mean=(
            "Scientific interpretation is allowed",
            "Block 0 calibration evidence exists",
        ),
    ),
    TaskAArtifactSpec(
        name="step1_prepare_manifest",
        filename="task_a_prepare_manifest.json",
        producer="prepare_task_a_stage0_mapping",
        purpose="Declare Step 1 provenance, run scope, and readiness for the prepare surface.",
        minimum_fields=(
            "task_name",
            "config_path",
            "stage0_h5ad",
            "mapping_manifest",
            "core_fit_dry_run",
            "pair_families",
            "confirmatory_pair_families",
            "run_scope",
            "artifact_state",
            "scientific_interpretation_allowed",
            "mass_mode",
        ),
        artifact_state_location="embedded field",
        allowed_artifact_states=(SCAFFOLD_ACTIVE_STATE, CONTRACT_PASSED_STATE),
        readiness_classification=CONTRACT_PASSED_STATE,
        does_not_mean=(
            "Block 0 calibration evidence",
            "Primary Block 1 evidence",
        ),
    ),
    TaskAArtifactSpec(
        name="descriptive_atlas_manifest",
        filename="task_a_descriptive_atlas_manifest.json",
        producer="write_task_a_descriptive_atlas",
        purpose=(
            "Declare descriptive-atlas labeling, Stage 0 field keys, counts, "
            "and the indexed output surface."
        ),
        minimum_fields=(
            "workflow_name",
            "atlas_role",
            "claim_scope",
            "scientific_interpretation_allowed",
            "config_path",
            "stage0_h5ad",
            "community_id_key",
            "cell_subtype_key",
            "domain_key",
            "fov_key",
            "spatial_key",
            "configured_community_ids",
            "observed_community_ids",
            "output_index",
        ),
        artifact_state_location="none",
        allowed_artifact_states=(),
        readiness_classification="descriptive_only",
        does_not_mean=(
            "Block 0 calibration evidence",
            "Any confirmatory claim is justified",
            "Block 1 evidence is ready",
        ),
    ),
    TaskAArtifactSpec(
        name="descriptive_atlas_output_index",
        filename="task_a_descriptive_atlas_output_index.csv",
        producer="write_task_a_descriptive_atlas",
        purpose="Index the descriptive-atlas tables and figures written to the output directory.",
        minimum_fields=(
            "relative_path",
            "artifact_kind",
            "category",
            "format",
            "description",
        ),
        artifact_state_location="none",
        allowed_artifact_states=(),
        readiness_classification="descriptive_only",
        does_not_mean=(
            "Every listed file is inferential",
            "The atlas may substitute for Block 0 calibration",
        ),
    ),
    TaskAArtifactSpec(
        name="block0_execution_manifest",
        filename="block0_execution_manifest.json",
        producer="run_block0_execute",
        purpose=(
            "Declare Block 0 fit-cache execution provenance, seed policy, "
            "readiness, and cache paths under the within-patient domain-label null."
        ),
        minimum_fields=(
            "task_name",
            "config_path",
            "stage0_h5ad",
            "run_scope",
            "n_permutations",
            "master_seed",
            "seed_derivation_policy",
            "real_family",
            "null_family",
            "permutation_policy",
            "fit_status",
            "readiness_status",
            "patient_count",
            "record_count",
            "k_states",
            "fit_cache_schema_version",
            "fit_cache_path",
            "fit_cache_index_path",
            "fit_cache_sha256",
            "fit_cache_index_sha256",
        ),
        artifact_state_location="none",
        allowed_artifact_states=(),
        readiness_classification="calibration_ready_or_diagnostic",
        does_not_mean=(
            "A calibration p-value was computed",
            "Biology was interpreted",
            "Multiplicity correction or a pass/fail gate was applied",
        ),
    ),
    TaskAArtifactSpec(
        name="block0_fit_cache",
        filename="block0_fit_cache.npz",
        producer="run_block0_execute",
        purpose=(
            "Persist per-patient real/null full-STRIDE `A`, `d`, `e`, "
            "`mu_minus`, and `mu_plus` arrays for cache-derived analyses."
        ),
        minimum_fields=("A", "d", "e", "source_burden", "target_burden"),
        artifact_state_location="block0_execution_manifest.json.readiness_status",
        allowed_artifact_states=(),
        readiness_classification="calibration_ready_or_diagnostic",
        does_not_mean=(
            "The arrays are a biological result surface",
            "A calibration p-value was computed",
        ),
    ),
    TaskAArtifactSpec(
        name="block0_fit_cache_index",
        filename="block0_fit_cache_index.csv",
        producer="run_block0_execute",
        purpose="Index the rows of `block0_fit_cache.npz` by fit label, permutation, and patient.",
        minimum_fields=("record_id", "fit_label", "permutation_index", "patient_id", "fit_status"),
        artifact_state_location="block0_execution_manifest.json.readiness_status",
        allowed_artifact_states=(),
        readiness_classification="calibration_ready_or_diagnostic",
        does_not_mean=(
            "The indexed records are biological interpretations",
            "A downstream execution decision was made",
        ),
    ),
    TaskAArtifactSpec(
        name="block0_calibration_manifest",
        filename="block0_calibration_manifest.json",
        producer="analyze_block0_cache",
        purpose=(
            "Declare Block 0 cache-derived calibration provenance, analysis "
            "specification, fit readiness, and output paths."
        ),
        minimum_fields=(
            "task_name",
            "config_path",
            "stage0_h5ad",
            "run_scope",
            "n_permutations",
            "master_seed",
            "seed_derivation_policy",
            "real_family",
            "null_family",
            "permutation_policy",
            "summary_roles",
            "fit_status",
            "readiness_status",
            "analysis_spec_version",
            "source_execution_manifest_path",
            "source_fit_cache_path",
            "source_fit_cache_index_path",
            "source_fit_cache_sha256",
            "source_fit_cache_index_sha256",
            "patient_calibration_path",
            "metric_summary_path",
        ),
        artifact_state_location="none",
        allowed_artifact_states=(),
        readiness_classification="calibration_ready_or_diagnostic",
        does_not_mean=(
            "Block 1 may or may not run",
            "A significant/not significant conclusion was made",
            "Biology was interpreted",
            "Multiplicity correction or a pass/fail gate was applied",
        ),
    ),
    TaskAArtifactSpec(
        name="block0_patient_calibration",
        filename="block0_patient_calibration.csv",
        producer="analyze_block0_cache",
        purpose=(
            "Record patient-level family-summary calibration context under the "
            "within-patient domain-label null without biological "
            "interpretation or downstream authorization."
        ),
        minimum_fields=_BLOCK0_PATIENT_CALIBRATION_FIELDS,
        artifact_state_location="block0_calibration_manifest.json.readiness_status",
        allowed_artifact_states=(),
        readiness_classification="calibration_ready_or_diagnostic",
        does_not_mean=(
            "The patient is biologically significant",
            "The calibration statistics authorize Task A advancement",
        ),
    ),
    TaskAArtifactSpec(
        name="block0_metric_summary",
        filename="block0_metric_summary.csv",
        producer="analyze_block0_cache",
        purpose=(
            "Summarize cohort-level family-summary calibration departures "
            "under the within-patient domain-label null without biological "
            "interpretation or execution decisions."
        ),
        minimum_fields=_BLOCK0_METRIC_SUMMARY_FIELDS,
        artifact_state_location="block0_calibration_manifest.json.readiness_status",
        allowed_artifact_states=(),
        readiness_classification="calibration_ready_or_diagnostic",
        does_not_mean=(
            "Totals are primary decision metrics",
            "Patient-level p-values were aggregated into cohort inference",
            "FDR or Bonferroni correction was applied",
            "Block 1 consumed a Block 0 execution-decision artifact",
        ),
    ),
    TaskAArtifactSpec(
        name="block1_family_summary",
        filename="block1_family_summary.csv",
        producer="block1_real_data_discovery",
        purpose="Freeze patient-level Block 1 family summaries on the frozen summary-name and scale axes.",
        minimum_fields=(
            "patient_id",
            "pair_family",
            "claim_role",
            "source_domain",
            "target_domain",
            "summary_name",
            "summary_role",
            "scale",
            "value",
            "eligible_entity_axis",
            "eligible_entity_count",
            "burden_total",
        ),
        artifact_state_location="pending Block 1 analyze carrier",
        allowed_artifact_states=(),
        readiness_classification="pending_implementation",
        does_not_mean=(
            "Target-side supportive summaries became proof-carrying",
        ),
    ),
    TaskAArtifactSpec(
        name="block1_source_community_summary",
        filename="block1_source_community_summary.csv",
        producer="block1_real_data_discovery",
        purpose="Freeze patient-level source-community Block 1 summaries including self-retention, depletion, remodeling, and top targets.",
        minimum_fields=(
            "patient_id",
            "pair_family",
            "claim_role",
            "source_domain",
            "target_domain",
            "source_community_id",
            "source_burden",
            "source_weight",
            "self_retention",
            "depletion",
            "off_diagonal_remodeling",
            "self_retention_burden",
            "depletion_burden",
            "off_diagonal_burden",
            "top_target_1_id",
            "top_target_1_value",
        ),
        artifact_state_location="pending Block 1 analyze carrier",
        allowed_artifact_states=(),
        readiness_classification="pending_implementation",
        does_not_mean=(
            "Neighborhood-based continuity was estimated",
            "Target-side emergence is confirmed",
        ),
    ),
    TaskAArtifactSpec(
        name="block1_target_community_summary",
        filename="block1_target_community_summary.csv",
        producer="block1_real_data_discovery",
        purpose="Freeze patient-level target-community Block 1 summaries including incoming matched burden and supportive emergence.",
        minimum_fields=(
            "patient_id",
            "pair_family",
            "claim_role",
            "source_domain",
            "target_domain",
            "target_community_id",
            "target_burden",
            "target_weight",
            "incoming_matched_operator",
            "incoming_matched_burden",
            "emergence_tendency",
            "emergence_burden",
        ),
        artifact_state_location="pending Block 1 analyze carrier",
        allowed_artifact_states=(),
        readiness_classification="pending_implementation",
        does_not_mean=(
            "Emergence is proof-carrying in this pass",
        ),
    ),
    TaskAArtifactSpec(
        name="block1_confirmatory_family_comparison",
        filename="block1_confirmatory_family_comparison.csv",
        producer="block1_real_data_discovery",
        purpose="Freeze the patient-paired confirmatory `TC-IM` versus `TC-PT` comparison surface on the frozen Block 1 family summary axes.",
        minimum_fields=(
            "patient_id",
            "pair_family_left",
            "pair_family_right",
            "summary_name",
            "summary_role",
            "scale",
            "eligible_entity_axis",
            "tc_im_value",
            "tc_pt_value",
            "delta_tc_im_minus_tc_pt",
            "contrast_direction",
            "comparison_status",
            "comparison_scope_role",
        ),
        artifact_state_location="pending Block 1 analyze carrier",
        allowed_artifact_states=(),
        readiness_classification="pending_implementation",
        does_not_mean=(
            "Community-level outputs became confirmatory",
        ),
    ),
    TaskAArtifactSpec(
        name="block1_source_community_comparison",
        filename="block1_source_community_comparison.csv",
        producer="block1_real_data_discovery",
        purpose="Export the patient-paired descriptive source-community `TC-IM` versus `TC-PT` comparison surface.",
        minimum_fields=(
            "patient_id",
            "pair_family_left",
            "pair_family_right",
            "source_community_id",
            "summary_name",
            "summary_role",
            "tc_im_value",
            "tc_pt_value",
            "delta_tc_im_minus_tc_pt",
            "comparison_status",
            "comparison_scope_role",
        ),
        artifact_state_location="pending Block 1 analyze carrier",
        allowed_artifact_states=(),
        readiness_classification="pending_implementation",
        does_not_mean=(
            "The source-community comparison is confirmatory",
            "Missing community rows imply fitting failed",
        ),
    ),
    TaskAArtifactSpec(
        name="block1_target_community_comparison",
        filename="block1_target_community_comparison.csv",
        producer="block1_real_data_discovery",
        purpose="Export the patient-paired descriptive target-community `TC-IM` versus `TC-PT` comparison surface.",
        minimum_fields=(
            "patient_id",
            "pair_family_left",
            "pair_family_right",
            "target_community_id",
            "summary_name",
            "summary_role",
            "tc_im_value",
            "tc_pt_value",
            "delta_tc_im_minus_tc_pt",
            "comparison_status",
            "comparison_scope_role",
        ),
        artifact_state_location="pending Block 1 analyze carrier",
        allowed_artifact_states=(),
        readiness_classification="pending_implementation",
        does_not_mean=(
            "The target-community comparison is confirmatory",
            "Emergence became proof-carrying in this pass",
        ),
    ),
    TaskAArtifactSpec(
        name="block1_cohort_relation_comparison",
        filename="block1_cohort_relation_comparison.csv",
        producer="block1_real_data_discovery",
        purpose=(
            "Export the descriptive cohort-level `TC-IM` versus `TC-PT` "
            "relation comparison over template_A/template_d/template_e on the "
            "shared community axis."
        ),
        minimum_fields=(
            "component",
            "relation_axis",
            "source_community_id",
            "target_community_id",
            "tc_im_value",
            "tc_pt_value",
            "delta_tc_im_minus_tc_pt",
            "contrast_direction",
            "tc_im_support_n_patients",
            "tc_pt_support_n_patients",
            "comparison_scope_role",
        ),
        artifact_state_location="pending Block 1 analyze carrier",
        allowed_artifact_states=(),
        readiness_classification="pending_implementation",
        does_not_mean=(
            "A specific community was selected for interpretation",
            "A p-value, FDR call, or robustness decision was emitted",
        ),
    ),
    TaskAArtifactSpec(
        name="semisynthetic_manifest",
        filename="user-specified manifest filename",
        producer="write_semisynthetic_artifacts",
        purpose="Export deterministic semisynthetic patient-level benchmark rows.",
        minimum_fields=(
            "patient_id",
            "pair_family",
            "continuity_score",
            "source_residual_mass",
            "target_residual_mass",
        ),
        artifact_state_location="task_a_semisynthetic_contract.json.artifact_state",
        allowed_artifact_states=(CONTRACT_PASSED_STATE,),
        readiness_classification=CONTRACT_PASSED_STATE,
        does_not_mean=(
            "Real-data Block 1 evidence is ready",
            "Real-data Block 0 calibration exists",
        ),
    ),
    TaskAArtifactSpec(
        name="semisynthetic_contract",
        filename="task_a_semisynthetic_contract.json",
        producer="write_semisynthetic_artifacts",
        purpose="Freeze the deterministic same-marginals benchmark contract for semisynthetic exports.",
        minimum_fields=(
            "artifact_state",
            "n_patients",
            "seed",
            "same_marginals_pair_family",
            "stronger_continuity_score",
            "weaker_continuity_score",
        ),
        artifact_state_location="embedded field",
        allowed_artifact_states=(CONTRACT_PASSED_STATE,),
        readiness_classification=CONTRACT_PASSED_STATE,
        does_not_mean=(
            "Any real-data block already ran",
        ),
    ),
)


TASK_A_SURFACE_SPEC_BY_NAME: dict[str, TaskASurfaceSpec] = {
    spec.name: spec for spec in TASK_A_SURFACE_SPECS
}
TASK_A_EXECUTION_NODE_BY_NAME: dict[str, TaskAExecutionNode] = {
    node.name: node for node in TASK_A_EXECUTION_GRAPH
}
TASK_A_ARTIFACT_SPEC_BY_NAME: dict[str, TaskAArtifactSpec] = {
    spec.name: spec for spec in TASK_A_ARTIFACT_SPECS
}


__all__ = [
    "GRAPH_EXECUTION_STATUS",
    "READINESS_CLASSIFICATIONS",
    "SURFACE_EXECUTION_STATUS",
    "TASK_A_ARTIFACT_SPEC_BY_NAME",
    "TASK_A_ARTIFACT_SPECS",
    "TASK_A_EXECUTION_GRAPH",
    "TASK_A_EXECUTION_NODE_BY_NAME",
    "TASK_A_SURFACE_SPEC_BY_NAME",
    "TASK_A_SURFACE_SPECS",
    "TaskAArtifactSpec",
    "TaskAExecutionNode",
    "TaskASurfaceSpec",
]
