# Task A Artifact Contracts

This file records the major Task A artifact schemas as a derived operational
mirror of the live Task A scientific documents. For Block 3 scientific
meaning, the authority chain is
[`docs/task_A_spec.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A_spec.md),
then
[`docs/task_A_block3_redesign_v1_1.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A_block3_redesign_v1_1.md),
then active derived docs such as this contract mirror. If consulted,
[`design_freeze.py`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/contracts/design_freeze.py)
is legacy/stale machine-readable inventory and implementation-lag evidence
only, not current Block 3 authority.

## Stage 0 and Step 1

| Artifact | Purpose | Minimum required fields | `artifact_state` expectation | Readiness |
|---|---|---|---|---|
| `task_A_stage0_validation.json` | Validate a built Stage 0 h5ad against the frozen Task A input contract. | `artifact_state`, `taska_minimum_contract`, `representation_completeness`, `counts` | Embedded field. `contract_passed` when both contract checks pass; otherwise `scaffold_active`. | contract-passed |
| `task_a_stride_mapping.json` | Freeze the Stage 0 to STRIDE field crosswalk and per-family eligibility summary. | `field_mapping`, `patient_ids`, `family_summaries`, `real_data_crosswalk` | Inherits from sibling `task_a_prepare_manifest.json`. Allowed: `scaffold_active`, `contract_passed`. | contract-passed |
| `task_a_core_fit_dry_run.csv` | Record confirmatory dry-run statuses on the task-local adapter surface. | `pair_family`, `claim_role`, `patient_id`, `implementation_tier`, `fit_surface`, `fit_status`, `bridge_realized`, `defer_reason`, `uncertainty_status`, `cohort_recurrence_fit_status`, `n_recurrence_families`, `n_recurrence_used_patients`, `source_domain`, `target_domain` | Inherits from sibling `task_a_prepare_manifest.json`. Allowed: `scaffold_active`, `contract_passed`. | contract-passed |
| `task_a_prepare_manifest.json` | Declare Step 1 provenance and readiness. | `task_name`, `config_path`, `stage0_h5ad`, `mapping_manifest`, `core_fit_dry_run`, `pair_families`, `confirmatory_pair_families`, `run_scope`, `artifact_state`, `block0_gate_status`, `scientific_interpretation_allowed`, `mass_mode`, `fit_surface`, `implementation_tier`, `evidence_lineage` | Embedded field. Full cohort may emit `contract_passed`; subset/demo emits `scaffold_active`. | contract-passed |
| `task_a_pre_block0_data_suitability.json` | Report pre-Block 0 suitability without claiming evidence. | `task_name`, `config_path`, `stage0_h5ad`, `report_scope`, `run_scope`, `artifact_state`, `block0_gate_status`, `scientific_interpretation_allowed`, `mass_mode`, `fit_surface`, `implementation_tier`, `evidence_lineage`, `confirmatory_pair_families`, `audit_pair_families`, `stage0_validation` | Embedded field. `contract_passed` only when the compatibility summary passes; otherwise `scaffold_active`. | contract-passed |

## Descriptive atlas

| Artifact | Purpose | Minimum required fields | `artifact_state` expectation | Readiness |
|---|---|---|---|---|
| `task_a_descriptive_atlas_manifest.json` | Declare descriptive-atlas provenance, descriptive-only labeling, and the indexed output surface. | `workflow_name`, `atlas_role`, `claim_scope`, `scientific_interpretation_allowed`, `artifact_state`, `block0_gate_status`, `prepare_manifest_path`, `mapping_manifest_path`, `config_path`, `stage0_h5ad`, `run_scope`, `community_id_key`, `cell_subtype_key`, `domain_key`, `fov_key`, `spatial_key`, `output_index` | Embedded field. Inherits the pre-inferential state from the Step 1 prepare manifest; allowed: `scaffold_active`, `contract_passed`. | contract-passed |
| `task_a_descriptive_atlas_output_index.csv` | Index all descriptive-atlas tables and figures in a machine-readable form. | `relative_path`, `artifact_kind`, `category`, `format`, `description` | Inherits from sibling `task_a_descriptive_atlas_manifest.json`. Allowed: `scaffold_active`, `contract_passed`. | contract-passed |

Expected atlas-side table and figure families:
- `tables/community_cell_subtype_*.csv`
- `tables/community_domain_*.csv`
- `tables/community_patient_occurrence_*.csv`
- `tables/representative_overlay_selection.csv`
- `figures/community_by_cell_subtype_heatmap.svg`
- `figures/community_domain_abundance_heatmap.svg`
- `figures/community_domain_roi_prevalence_heatmap.svg`
- `figures/patient_level_community_prevalence.svg`
- `figures/representative_spatial_overlays/community_*.svg`

Non-goal reminders for the atlas:
- The atlas is descriptive only and does not constitute hypothesis testing.
- The atlas does not satisfy or summarize Block 0.
- The atlas does not promote descriptive community context into confirmatory evidence.

## Block outputs

Block 0 freeze:
- Real family: `TC-IM`.
- Null family: `TC-IM_randomized_target`, built by keeping each anchor
  patient's `TC` observations fixed and reassigning the `IM` group from a
  different patient in the same exact `(n_TC, n_IM)` count stratum.
- Pass/fail uses paired summaries derived directly from realized `A`, `d`, and
  `e`, with live gate decisions driven by `delta_total_continuity_mass` and
  `delta_total_emergence_mass`.

| Artifact | Purpose | Minimum required fields | `artifact_state` expectation | Readiness |
|---|---|---|---|---|
| `block0_bundle.json` | Freeze Block 0 provenance, gate summaries, and pass/defer/fail state for the STRIDE-native real-vs-null gate. | `block`, `status`, `artifact_state`, `implementation_tier`, `evidence_lineage`, `run_scope`, `block0_passed`, `config_fingerprint`, `config_path`, `stage0_h5ad`, `output_dir`, `bundle_path`, `pair_metrics_path`, `real_families`, `null_families`, `pre_block0_data_suitability`, `gate_checks`, `metrics_summary`, `failure_reasons`, `inputs` | Embedded field. Full-cohort pass emits `contract_passed`; subset/demo or insufficient-support runs remain `scaffold_active`. Downstream acceptance requires both `status="passed"` and `artifact_state="contract_passed"`. | contract-passed |
| `block0_pair_metrics.csv` | Record one paired patient-level real/null Block 0 comparison row with STRIDE-native summaries. | `comparison_id`, `run_scope`, `pair_family`, `null_family`, `anchor_patient_id`, `null_target_donor_patient_id`, `source_domain`, `target_domain`, `n_source_observations`, `n_target_observations`, `count_stratum_key`, `selection_seed`, `null_assignment_status`, `null_assignment_reason`, `real_fit_status`, `null_fit_status`, `real_defer_reason`, `null_defer_reason`, `real_total_continuity_mass`, `null_total_continuity_mass`, `delta_total_continuity_mass`, `real_total_depletion_mass`, `null_total_depletion_mass`, `delta_total_depletion_mass`, `real_total_emergence_mass`, `null_total_emergence_mass`, `delta_total_emergence_mass` | Inherits from sibling `block0_bundle.json`. Allowed: `scaffold_active`, `contract_passed`. | contract-passed |
| `block1_family_summary.csv` | Freeze patient-level Block 1 family summaries on `summary_name x scale x pair_family`. | `patient_id`, `pair_family`, `claim_role`, `source_domain`, `target_domain`, `summary_name`, `summary_role`, `scale`, `value`, `eligible_entity_axis`, `eligible_entity_count`, `burden_total` | Inherits from sibling `block1_bundle.json`. Allowed: `evidence_ready`. | evidence-ready |
| `block1_source_community_summary.csv` | Freeze patient-level source-community summaries with `SR_i / D_i / R_i / TopTargets_i`. | `patient_id`, `pair_family`, `claim_role`, `source_domain`, `target_domain`, `source_community_id`, `source_burden`, `source_weight`, `self_retention`, `depletion`, `off_diagonal_remodeling`, `self_retention_burden`, `depletion_burden`, `off_diagonal_burden`, `top_target_1_id`, `top_target_1_value` | Inherits from sibling `block1_bundle.json`. Allowed: `evidence_ready`. | evidence-ready |
| `block1_target_community_summary.csv` | Freeze patient-level target-community summaries with `I_j / E_j` supportive context. | `patient_id`, `pair_family`, `claim_role`, `source_domain`, `target_domain`, `target_community_id`, `target_burden`, `target_weight`, `incoming_matched_operator`, `incoming_matched_burden`, `emergence_tendency`, `emergence_burden` | Inherits from sibling `block1_bundle.json`. Allowed: `evidence_ready`. | evidence-ready |
| `block1_stage0_mapping.json` | Block 1 mapping summary used inside the evidence bundle. | `field_mapping`, `patient_ids`, `family_summaries`, `real_data_crosswalk` | Inherits from sibling `block1_bundle.json`. Allowed: `evidence_ready`. | evidence-ready |
| `block1_core_fit_dry_run.csv` | Freeze confirmatory Block 1 fit realization context alongside the summary outputs. | `pair_family`, `claim_role`, `patient_id`, `implementation_tier`, `fit_surface`, `fit_status`, `bridge_realized`, `defer_reason`, `uncertainty_status`, `cohort_recurrence_fit_status`, `n_recurrence_families`, `n_recurrence_used_patients`, `source_domain`, `target_domain` | Inherits from sibling `block1_bundle.json`. Allowed: `evidence_ready`. | evidence-ready |
| `block1_recurrence_summary.json` | Declare canonical cohort-level recurrence/common-structure status across the confirmatory Block 1 family fits. | `fit_surface`, `implementation_tier`, `evidence_lineage`, `cohort_recurrence_fit_status`, `cohort_recurrence_fit_status_by_pair_family`, `cohort_recurrence_family_count`, `cohort_recurrence_family_count_by_pair_family`, `n_recurrence_used_patients`, `n_recurrence_used_patients_by_pair_family`, `pair_families` | Inherits from sibling `block1_bundle.json`. Allowed: `evidence_ready`. | evidence-ready |
| `block1_recurrence_families.json` | Export family-level canonical recurrence templates linked to Block 1 pair families. | `pair_family`, `family_id`, `fit_status`, `support_n_patients`, `member_patient_ids`, `template_A`, `template_d`, `template_e` | Inherits from sibling `block1_bundle.json`. Allowed: `evidence_ready`. | evidence-ready |
| `block1_recurrence_embeddings.csv` | Export patient-level canonical recurrence embeddings linked to Block 1 pair families. | `pair_family`, `patient_id`, `fit_status`, `used_for_recurrence`, `coord_1` | Inherits from sibling `block1_bundle.json`. Allowed: `evidence_ready`. | evidence-ready |
| `block1_confirmatory_family_comparison.csv` | Freeze the patient-paired confirmatory `TC-IM` versus `TC-PT` family comparison surface on the frozen Block 1 axes. | `patient_id`, `pair_family_left`, `pair_family_right`, `summary_name`, `summary_role`, `scale`, `eligible_entity_axis`, `tc_im_value`, `tc_pt_value`, `delta_tc_im_minus_tc_pt`, `contrast_direction`, `comparison_status`, `comparison_scope_role` | Inherits from sibling `block1_bundle.json`. Allowed: `evidence_ready`. | evidence-ready |
| `block1_exploratory_source_community_comparison.csv` | Export the patient-paired exploratory/supportive source-community comparison surface for `TC-IM` versus `TC-PT`. | `patient_id`, `pair_family_left`, `pair_family_right`, `source_community_id`, `summary_name`, `summary_role`, `tc_im_value`, `tc_pt_value`, `delta_tc_im_minus_tc_pt`, `comparison_status`, `comparison_scope_role` | Inherits from sibling `block1_bundle.json`. Allowed: `evidence_ready`. | evidence-ready |
| `block1_exploratory_target_community_comparison.csv` | Export the patient-paired exploratory/supportive target-community comparison surface for `TC-IM` versus `TC-PT`. | `patient_id`, `pair_family_left`, `pair_family_right`, `target_community_id`, `summary_name`, `summary_role`, `tc_im_value`, `tc_pt_value`, `delta_tc_im_minus_tc_pt`, `comparison_status`, `comparison_scope_role` | Inherits from sibling `block1_bundle.json`. Allowed: `evidence_ready`. | evidence-ready |
| `community_correspondence/block1_community_correspondence_manifest.json` | Declare the objective Block 1 community-correspondence packet and point to its machine-readable outputs. | `workflow_name`, `packet_role`, `scientific_interpretation_allowed`, `artifact_state`, `config_path`, `stage0_h5ad`, `community_id_key`, `cell_subtype_key`, `patient_id_key`, `configured_state_ids`, `observed_community_ids`, `output_index` | Inherits from sibling `block1_bundle.json`. Allowed: `evidence_ready`. | evidence-ready |
| `community_correspondence/block1_community_correspondence_index.csv` | Index the correspondence packet tables plus referenced Block 1 summary/comparison surfaces. | `relative_path`, `artifact_kind`, `category`, `format`, `description` | Inherits from sibling `block1_bundle.json`. Allowed: `evidence_ready`. | evidence-ready |
| `block1_bundle.json` | Freeze Block 1 provenance, compatibility block id, scientific role, and summary-contract output paths. | `block`, `scientific_role`, `status`, `artifact_state`, `implementation_tier`, `evidence_lineage`, `fit_surface`, `block0_bundle_path`, `block0_gate_status`, `config_fingerprint`, `config_path`, `stage0_h5ad`, `output_dir`, `mapping_manifest_path`, `core_fit_dry_run_path`, `recurrence_summary_path`, `recurrence_families_path`, `recurrence_embeddings_path`, `family_summary_path`, `source_community_summary_path`, `target_community_summary_path`, `confirmatory_family_comparison_path`, `exploratory_source_community_comparison_path`, `exploratory_target_community_comparison_path`, `community_correspondence_manifest_path`, `community_correspondence_index_path`, `bundle_path`, `pair_families`, `confirmatory_pair_families`, `summary_contract_version`, `paired_comparison_contract_version`, `proof_carrying_family_summaries`, `supportive_family_summaries`, `family_summary_scales`, `source_eligibility_rule`, `target_eligibility_rule`, `fit_result_counts`, `cohort_recurrence_fit_status`, `cohort_recurrence_fit_status_by_pair_family`, `cohort_recurrence_family_count`, `cohort_recurrence_family_count_by_pair_family`, `n_recurrence_used_patients`, `n_recurrence_used_patients_by_pair_family`, `stage0_mapping` | Embedded field. Allowed: `evidence_ready`. | evidence-ready |
| `block1_workflow_manifest.json` | Compact Block 1 pointer manifest for the frozen summary contract. | `block`, `scientific_role`, `status`, `artifact_state`, `implementation_tier`, `evidence_lineage`, `fit_surface`, `block0_bundle_path`, `block0_gate_status`, `config_fingerprint`, `bundle_path`, `core_fit_dry_run_path`, `mapping_manifest_path`, `recurrence_summary_path`, `recurrence_families_path`, `recurrence_embeddings_path`, `family_summary_path`, `source_community_summary_path`, `target_community_summary_path`, `confirmatory_family_comparison_path`, `exploratory_source_community_comparison_path`, `exploratory_target_community_comparison_path`, `community_correspondence_manifest_path`, `community_correspondence_index_path`, `summary_contract_version`, `paired_comparison_contract_version`, `proof_carrying_family_summaries`, `supportive_family_summaries`, `family_summary_scales`, `cohort_recurrence_fit_status`, `cohort_recurrence_fit_status_by_pair_family`, `cohort_recurrence_family_count`, `cohort_recurrence_family_count_by_pair_family`, `n_recurrence_used_patients`, `n_recurrence_used_patients_by_pair_family` | Embedded field. Allowed: `evidence_ready`. | evidence-ready |
| `block2_bounded_audit_summary.csv` | Compatibility-named top-level Block 2 finding-call table over the primary robustness routes. | `block`, `summary_scope`, `finding_priority`, `summary_name`, `scale`, `community_id`, `full_data_direction`, `primary_routes_executed`, `primary_routes_robust`, `overall_robustness_call` | Inherits from sibling `block2_bounded_audit_manifest.json`. Allowed: `evidence_ready`. | evidence-ready |
| `block2_contract_audit.csv` | Audit the provenance checks required for the compatibility-named Block 2 robustness-over-summaries surface. | `check`, `passed`, `detail` | Inherits from sibling `block2_bounded_audit_manifest.json`. Allowed: `evidence_ready`. | evidence-ready |
| `block2_replicate_manifest.csv` | Record attempted Block 2 perturbation replicates, retained cohort sizes, and replicate-level failures. | `route_name`, `route_group`, `replicate_index`, `replicate_label`, `selection_seed`, `patient_subset_json`, `dropped_roi_ids_json`, `route_status`, `failure_reason`, `n_patients_retained`, `n_rois_retained`, `n_cells_retained` | Inherits from sibling `block2_bounded_audit_manifest.json`. Allowed: `evidence_ready`. | evidence-ready |
| `block2_family_robustness.csv` | Carry route-level robustness summaries for the frozen family-level Block 1 findings. | `block`, `route_name`, `route_group`, `finding_id`, `summary_scope`, `finding_priority`, `summary_name`, `scale`, `full_data_direction`, `direction_recovery_rate`, `estimable_replicate_fraction`, `median_replicate_support_fraction`, `robustness_call` | Inherits from sibling `block2_bounded_audit_manifest.json`. Allowed: `evidence_ready`. | evidence-ready |
| `block2_source_community_robustness.csv` | Carry route-level robustness summaries for the scoped source-community Block 1 findings. | `block`, `route_name`, `route_group`, `finding_id`, `summary_scope`, `finding_priority`, `summary_name`, `community_id`, `full_data_direction`, `direction_recovery_rate`, `estimable_replicate_fraction`, `median_replicate_support_fraction`, `median_replicate_rank`, `robustness_call` | Inherits from sibling `block2_bounded_audit_manifest.json`. Allowed: `evidence_ready`. | evidence-ready |
| `block2_target_community_robustness.csv` | Carry route-level robustness summaries for the scoped target-community Block 1 findings. | `block`, `route_name`, `route_group`, `finding_id`, `summary_scope`, `finding_priority`, `summary_name`, `community_id`, `full_data_direction`, `direction_recovery_rate`, `estimable_replicate_fraction`, `median_replicate_support_fraction`, `median_replicate_rank`, `robustness_call` | Inherits from sibling `block2_bounded_audit_manifest.json`. Allowed: `evidence_ready`. | evidence-ready |
| `block2_bounded_audit_manifest.json` | Declare Block 2 provenance, scientific role, perturbation-route scope, and the linked Block 1 / Block 2 artifact paths it consumes and emits. | `block`, `scientific_role`, `config_path`, `config_fingerprint`, `output_dir`, `bundle_path`, `implementation_tier`, `evidence_lineage`, `fit_surface`, `block1_bundle_path`, `block0_bundle_path`, `block1_stage0_mapping_path`, `block1_core_fit_dry_run_path`, `block1_recurrence_summary_path`, `block1_recurrence_families_path`, `block1_recurrence_embeddings_path`, `block1_family_summary_path`, `block1_source_community_summary_path`, `block1_target_community_summary_path`, `block1_confirmatory_family_comparison_path`, `block1_exploratory_source_community_comparison_path`, `block1_exploratory_target_community_comparison_path`, `summary_path`, `contract_path`, `replicate_manifest_path`, `family_robustness_path`, `source_community_robustness_path`, `target_community_robustness_path`, `status`, `artifact_state`, `claim_scope`, `summary_rows`, `replicate_rows`, `block1_cohort_recurrence_fit_status`, `block1_cohort_recurrence_fit_status_by_pair_family`, `block1_cohort_recurrence_family_count`, `block1_cohort_recurrence_family_count_by_pair_family`, `block1_n_recurrence_used_patients`, `block1_n_recurrence_used_patients_by_pair_family`, `primary_routes` | Embedded field. Allowed: `evidence_ready`. | evidence-ready |

Frozen Block 1 summary naming and interpretation:

- `self_retention` is the machine-readable source-side `SR` summary.
- `depletion` is the machine-readable source-side `D` summary.
- `off_diagonal_remodeling` is the machine-readable source-side `R` summary,
  where `R = sum_{j != i} A_ij`.
- `emergence` is the machine-readable target-side supportive `E` summary.
- `burden_weighted` and `community_mean` are the only frozen family-level
  scales in this pass.
- `block1_confirmatory_family_comparison.csv` is the mandatory confirmatory
  patient-paired `TC-IM` versus `TC-PT` family comparison surface.
- `block1_exploratory_source_community_comparison.csv` and
  `block1_exploratory_target_community_comparison.csv` are exploratory/supportive
  community comparison surfaces and do not redefine confirmatory scope.
- The community-correspondence packet is objective only. It exists to connect
  community ids, cell-subtype context, and burden tables without adding
  interpretation prose.
- Block 2 reuses these same frozen Block 1 comparison objects under
  perturbation. It does not define a new discovery block or baseline-comparison
  language.

## Semi-synthetic and export-side artifacts

| Artifact | Purpose | Minimum required fields | `artifact_state` expectation | Readiness |
|---|---|---|---|---|
| semisynthetic manifest CSV | Export deterministic benchmark rows. | `patient_id`, `pair_family`, `continuity_score`, `source_residual_mass`, `target_residual_mass` | Inherits from sibling `task_a_semisynthetic_contract.json`. Allowed: `contract_passed`. | contract-passed |
| `task_a_semisynthetic_contract.json` | Freeze the benchmark contract for same-marginals teaching worlds and manifest exports. | `artifact_state`, `n_patients`, `seed`, `same_marginals_pair_family`, `stronger_continuity_score`, `weaker_continuity_score` | Embedded field. Allowed: `contract_passed`. | contract-passed |

## Step 3 result packet

| Artifact | Purpose | Minimum required fields | `artifact_state` expectation | Readiness |
|---|---|---|---|---|
| `task_a_result_packet_manifest.json` | Declare the Step 3 objective review packet over atlas plus canonical Block 0-2 surfaces. | `workflow_name`, `packet_role`, `packet_spec_version`, `packet_root`, `central_index_path`, `human_index_path`, `layer_manifest_paths`, `layer_review_index_paths`, `included_layers`, `deferred_layers`, `surface_lineage`, `input_sources`, `artifact_counts` | Packet-local manifest; no Task A evidence state is inferred from packet assembly alone. | packet-local |
| `task_a_result_packet_index.csv` | Record one row per mirrored or packet-local review artifact in the Step 3 packet. | `layer`, `artifact_name`, `expected_relative_path`, `packet_relative_path`, `artifact_status`, `contract_alignment`, `implementation_tier`, `evidence_lineage`, `artifact_kind`, `claim_scope`, `review_role`, `analysis_level`, `source_workflow`, `sha256` | Packet-local index; available rows must hash-match the mirrored packet files. | packet-local |
| `RESULTS_INDEX.md` | Provide a human-readable packet entrypoint summarizing included layers, deferred layers, and suggested first-inspection artifacts. | Human-readable markdown summary. | Packet-local only. | packet-local |

Packet-specific boundary notes:
- The Step 3 packet includes only `atlas`, `block0`, `block1`, and `block2`.
- `block3` is explicitly deferred from the Step 3 packet manifest rather than silently omitted.
- The packet index now carries per-artifact `implementation_tier` and `evidence_lineage` so canonical rerun files and preserved proxy-history files cannot be confused.
- Any Block 3-facing packet builder, packet-local review surface, or preserved
  historical packet is result-layer implementation context only. None of those
  packet surfaces is current live Block 3 scientific authority or canonical
  Block 3 result authority for this pass.

## Block 3 deferred public boundary

The Block 3 scientific contract remains frozen in the docs hierarchy. The Task
A repository still does **not** ship an active public Block 3 workflow,
review CLI, or packet bridge. The former demo-style `tasks/task_A/block3/`
implementation was removed after it was judged too polluted by stale
proxy-era contract topology.

The repository may carry an on-disk `tasks/task_A/block3/` package for
internal Phase 3 execution, registry construction, execution planning, raw
bundle writing, and review-surface writing. That package may execute real
generator/method/scoring logic against evidence-ready Block 1/2 inputs, but it
remains implementation context only: non-authority, non-public, and not a
proof-carrying workflow surface.

Current engineering policy:

- `run_block3` is a fail-fast deprecation wrapper, not an executable bundle
  writer.
- `review_block3` is a fail-fast deprecation wrapper.
- `package_results` must fail-fast on `--block3-manifest`.
- `result_packet.py` keeps `block3` explicitly deferred rather than proof-carrying.
- `design_freeze.py` no longer lists active Block 3 executable surfaces or
  Block 3 artifacts in the active engineering registry.

Use the docs hierarchy for Block 3 scientific meaning, but do not treat this
repository state as having a public executable Block 3 producer, review
surface, or packet bridge.

## Non-goal reminders

- No Step 1 artifact may be read as proof that Block 0 has passed.
- No Block 2 artifact may be read as proof of true disappearance or true emergence.
- No Block 2 artifact may be read as a baseline comparison, ablation, or semi-synthetic validation surface.
- Semi-synthetic exports are benchmark contracts, not real-data evidence surfaces.
- No internal Block 3 Phase 3 artifact may be read as public comparator evidence or final method-validation closure.
