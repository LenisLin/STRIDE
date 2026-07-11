# Task A Artifact Contracts

This file records the major Task A artifact schemas as a derived operational
mirror of the live Task A scientific documents. For Block 3 scientific
meaning, the authority chain is
[`docs/task_A/spec.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A/spec.md),
then
[`docs/task_A/block3/scientific_contract.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A/block3/scientific_contract.md),
then active derived docs such as this contract mirror. If consulted,
[`design_freeze.py`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/contracts/design_freeze.py)
is legacy/stale machine-readable inventory and implementation-lag evidence
only, not current Block 3 authority.

## Stage 0 and Step 1

| Artifact | Purpose | Minimum required fields | `artifact_state` expectation | Readiness |
|---|---|---|---|---|
| `task_A_stage0_validation.json` | Validate a built Stage 0 h5ad against the frozen Task A input contract. | `artifact_state`, `taska_minimum_contract`, `representation_completeness`, `counts` | Embedded field. `contract_passed` when both contract checks pass; otherwise `scaffold_active`. | contract-passed |
| `task_a_stride_mapping.json` | Freeze the Stage 0 to STRIDE field crosswalk and per-family eligibility summary. | `field_mapping`, `patient_ids`, `family_summaries`, `real_data_crosswalk` | Inherits from sibling `task_a_prepare_manifest.json`. Allowed: `scaffold_active`, `contract_passed`. | contract-passed |
| `task_a_core_fit_dry_run.csv` | Record confirmatory dry-run statuses on the task-local adapter surface. | `pair_family`, `claim_role`, `patient_id`, `implementation_tier`, `fit_surface`, `fit_status`, `bridge_realized`, `defer_reason`, `uncertainty_status`, `cohort_recurrence_fit_status`, `n_recurrence_families`, `n_recurrence_used_patients`, `source_domain`, `target_domain` | Inherits from sibling `task_a_prepare_manifest.json`. Allowed: `scaffold_active`, `contract_passed`. `fit_surface` should use `stride.tl.fit` for the live fitting route. | contract-passed |
| `task_a_prepare_manifest.json` | Declare Step 1 provenance and readiness. | `task_name`, `config_path`, `stage0_h5ad`, `mapping_manifest`, `core_fit_dry_run`, `pair_families`, `confirmatory_pair_families`, `run_scope`, `artifact_state`, `scientific_interpretation_allowed`, `mass_mode`, `fit_surface`, `implementation_tier`, `evidence_lineage` | Embedded field. Full cohort may emit `contract_passed`; subset/demo emits `scaffold_active`. `fit_surface` should use `stride.tl.fit` for new live manifests. | contract-passed |

## Descriptive atlas

| Artifact | Purpose | Minimum required fields | `artifact_state` expectation | Readiness |
|---|---|---|---|---|
| `task_a_descriptive_atlas_manifest.json` | Declare descriptive-atlas labeling, Stage 0 field keys, counts, and the indexed output surface. | `workflow_name`, `atlas_role`, `claim_scope`, `scientific_interpretation_allowed`, `config_path`, `stage0_h5ad`, `community_id_key`, `cell_subtype_key`, `domain_key`, `fov_key`, `spatial_key`, `configured_community_ids`, `observed_community_ids`, `output_index` | No readiness state. | descriptive-only |
| `task_a_descriptive_atlas_output_index.csv` | Index all descriptive-atlas tables and figures in a machine-readable form. | `relative_path`, `artifact_kind`, `category`, `format`, `description` | No readiness state. | descriptive-only |

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

Block 0 execution/cache and calibration freeze:
- Real family: `TC-IM`.
- Null family: within-patient count-preserving `TC-IM` FOV/ROI domain-label
  permutation null. It preserves `patient_id`, FOV composition, FOV count
  structure, and each patient's exact `n_TC`/`n_IM` counts while randomly
  permuting `TC`/`IM` labels inside the same patient. Identity permutations are
  allowed; cross-patient label borrowing and relaxed fallback are not allowed.
- Hard inputs: Stage 0 h5ad and Task A config only. Prepare and
  descriptive-atlas artifacts are not evidence inputs.
- Formal full calibration uses `B=199`; `B` is configurable for diagnostics.
- Execution writes a reusable per-fit cache over `A`, `d`, `e`,
  `source_burden`, and `target_burden`; it does not derive p-values or
  scientific metrics. The live route runs both real and null fits through
  `stride.tl.fit` and derives burden fields from the fitted AnnData context.
- Analysis consumes that cache and derives fixed Block 1-facing family-summary
  calibration tables without rerunning permutations.
- Block 0 does not perform FDR/Bonferroni correction, emit significance labels,
  or create a pass/fail gate.

| Artifact | Purpose | Minimum required fields | `artifact_state` expectation | Readiness |
|---|---|---|---|---|
| `block0_execution_manifest.json` | Declare Block 0 fit-cache execution provenance, run scope, seed policy, fit readiness, and cache paths. | `task_name`, `config_path`, `stage0_h5ad`, `run_scope`, `n_permutations`, `master_seed`, `seed_derivation_policy`, `real_family`, `null_family`, `permutation_policy`, `fit_status`, `readiness_status`, `patient_count`, `record_count`, `k_states`, `fit_cache_schema_version`, `fit_cache_path`, `fit_cache_index_path`, `fit_cache_sha256`, `fit_cache_index_sha256`, `progress_path` | Embedded execution/readiness fields. | calibration-ready or diagnostic |
| `block0_fit_cache.npz` | Persist per-patient real/null full-STRIDE `A`, `d`, `e`, `mu_minus`, and `mu_plus` arrays for cache-derived analyses. | Arrays: `A`, `d`, `e`, `source_burden`, `target_burden`; first dimension aligns to `block0_fit_cache_index.csv`. | Inherits readiness context from execution manifest. | calibration-ready or diagnostic |
| `block0_fit_cache_index.csv` | Index the fit-cache rows by fit label, permutation, and patient. | `record_id`, `fit_label`, `permutation_index`, `patient_id`, `fit_status` | Inherits readiness context from execution manifest. | calibration-ready or diagnostic |
| `block0_calibration_manifest.json` | Declare cache-derived Block 0 family-summary calibration provenance, fixed analysis spec, fit readiness, and output paths. | `task_name`, `config_path`, `stage0_h5ad`, `run_scope`, `n_permutations`, `master_seed`, `seed_derivation_policy`, `real_family`, `null_family`, `permutation_policy`, `summary_roles`, `fit_status`, `readiness_status`, `analysis_spec_version`, `source_execution_manifest_path`, `source_fit_cache_path`, `source_fit_cache_index_path`, `source_fit_cache_sha256`, `source_fit_cache_index_sha256`, `patient_calibration_path`, `metric_summary_path` | Embedded analysis/readiness fields. | calibration-ready or diagnostic |
| `block0_patient_calibration.csv` | Record cache-derived patient-level family-summary calibration context. | `patient_id`, `run_scope`, `real_family`, `null_family`, `n_permutations`, `real_fit_status`, `null_fit_status`, `summary_name`, `summary_role`, `eligible_entity_axis`, `scale`, `reference_stat`, `expected_tail`, `real_value`, `null_reference`, `empirical_p_value`, `primary_tail_fraction`, `opposite_tail_fraction`, `effect_delta`, `effect_ratio`, `effect_ratio_status`, `readiness_status` | Inherits calibration/readiness context from sibling analysis manifest. | calibration-ready or diagnostic |
| `block0_metric_summary.csv` | Summarize cache-derived cohort-level family-summary calibration departures without biological interpretation or execution decisions. | `summary_name`, `summary_role`, `eligible_entity_axis`, `scale`, `cohort_stat`, `expected_tail`, `real_value`, `null_reference`, `empirical_p_value`, `primary_tail_fraction`, `opposite_tail_fraction`, `effect_delta`, `effect_ratio`, `effect_ratio_status`, `n_patient_delta_positive`, `n_patient_delta_negative`, `n_patient_delta_zero`, `readiness_status` | Inherits calibration/readiness context from sibling analysis manifest. | calibration-ready or diagnostic |
| `block1_family_summary.csv` | Patient-level Block 1 family summaries on `summary_name x scale x pair_family`. | `patient_id`, `pair_family`, `claim_role`, `source_domain`, `target_domain`, `summary_name`, `summary_role`, `scale`, `value`, `eligible_entity_axis`, `eligible_entity_count`, `burden_total` | Derived from task-native `stride.tl.fit` payload plus pair AnnData burden reconstruction. | executable |
| `block1_source_community_summary.csv` | Patient-level source-community summaries with self-retention, depletion, and off-diagonal remodeling. | `patient_id`, `pair_family`, `claim_role`, `source_domain`, `target_domain`, `source_community_id`, `source_burden`, `source_weight`, `self_retention`, `self_retention_burden`, `depletion`, `depletion_burden`, `off_diagonal_remodeling`, `off_diagonal_burden` | Derived from task-native `stride.tl.fit` payload plus pair AnnData burden reconstruction. | executable |
| `block1_target_community_summary.csv` | Patient-level target-community summaries with burden-scale matched incoming and target-open accounting. | `patient_id`, `pair_family`, `claim_role`, `source_domain`, `target_domain`, `target_community_id`, `target_burden`, `target_weight`, `matched_incoming_burden`, `open_incoming_tendency`, `open_incoming_burden` | Derived from task-native `stride.tl.fit` payload plus pair AnnData burden reconstruction. | executable |
| `block1_confirmatory_family_comparison.csv` | Patient-paired `TC-IM` versus `TC-PT` descriptive family contrast on the frozen Block 1 axes. | `patient_id`, `pair_family_left`, `pair_family_right`, `summary_name`, `summary_role`, `scale`, `eligible_entity_axis`, `tc_im_value`, `tc_pt_value`, `delta_tc_im_minus_tc_pt`, `contrast_direction`, `comparison_status`, `comparison_scope_role` | Derived during Block 1 analyze from executable family summaries. | executable |
| `block1_source_community_comparison.csv` | Patient-paired source-community descriptive contrast for `TC-IM` versus `TC-PT`. | `patient_id`, `pair_family_left`, `pair_family_right`, `source_community_id`, `summary_name`, `summary_role`, `tc_im_value`, `tc_pt_value`, `delta_tc_im_minus_tc_pt`, `comparison_status`, `comparison_scope_role` | Derived during Block 1 analyze from executable source-community summaries. | executable |
| `block1_target_community_comparison.csv` | Patient-paired target-community descriptive contrast for `TC-IM` versus `TC-PT`. | `patient_id`, `pair_family_left`, `pair_family_right`, `target_community_id`, `summary_name`, `summary_role`, `tc_im_value`, `tc_pt_value`, `delta_tc_im_minus_tc_pt`, `comparison_status`, `comparison_scope_role` | Derived during Block 1 analyze from executable target-community summaries. | executable |
| `block1_cohort_relation_comparison.csv` | Cohort-level descriptive direct contrast for `TC-IM` versus `TC-PT` over `template_A`, `template_d`, and `template_e`. | `component`, `relation_axis`, `source_community_id`, `target_community_id`, `tc_im_value`, `tc_pt_value`, `delta_tc_im_minus_tc_pt`, `contrast_direction`, `tc_im_support_n_patients`, `tc_pt_support_n_patients`, `comparison_scope_role` | Derived from the task-native `stride.tl.fit` payload through the Block 1 adapter. | executable |

Block 1 execute manifests and native fitting payloads should record
`fit_surface="stride.tl.fit"`. This changes the implementation surface only;
it does not change the scientific meaning of `A`, `d`, `e`, or the Task A
ordered-pair proxy.

Block 1 composition-scale note:

- The historical artifact field name `burden` is retained for compatibility.
  When populated from Task A FOV observations, its value is the patient-side
  mean of composition-scale FOV vectors under equal FOV weighting, not an
  independently measured absolute burden. Formal fit artifacts fail when an
  expected patient lacks source or target FOV support rather than silently
  dropping that patient.
- In the current uniform-FOV Task A setting, `mu_minus` and `mu_plus` are
  side-level composition burdens: the source-side and target-side means of FOV
  community fraction vectors. They are not raw cell counts, tissue areas, or
  physical masses.
- Block 1 analyze derives relation-component composition burdens from native
  STRIDE outputs: matched transition burden as `A * mu_minus[:, None]`, source
  open burden as `d * mu_minus`, and target open burden as
  `e * sum(mu_minus)`.
- The legacy unweighted target incoming operator-column summary is retired.
- Legacy source top-target ranking shorthands are retired.
- `emergence` is retained only as a family-level supportive target-open
  rollup.
- Target-community fields do not use `emergence` naming; they use
  `open_incoming_tendency` and `open_incoming_burden`.

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
- `block1_source_community_comparison.csv` and
  `block1_target_community_comparison.csv` are descriptive community contrast
  surfaces and do not redefine confirmatory scope.
- `block1_cohort_relation_comparison.csv` is an analysis-only cohort template
  direct contrast over all shared-axis communities and does not select or
  annotate specific communities.
- Direct Block 1 summaries and descriptive contrasts are the current Block 1
  live artifact surface. Community annotation or crosswalk needs must be
  defined outside Block 1 engineering.

## Semi-synthetic and export-side artifacts

| Artifact | Purpose | Minimum required fields | `artifact_state` expectation | Readiness |
|---|---|---|---|---|
| semisynthetic manifest CSV | Export deterministic benchmark rows. | `patient_id`, `pair_family`, `continuity_score`, `source_residual_mass`, `target_residual_mass` | Inherits from sibling `task_a_semisynthetic_contract.json`. Allowed: `contract_passed`. | contract-passed |
| `task_a_semisynthetic_contract.json` | Freeze the benchmark contract for same-marginals teaching worlds and manifest exports. | `artifact_state`, `n_patients`, `seed`, `same_marginals_pair_family`, `stronger_continuity_score`, `weaker_continuity_score` | Embedded field. Allowed: `contract_passed`. | contract-passed |

## Step 3 result packet

| Artifact | Purpose | Minimum required fields | `artifact_state` expectation | Readiness |
|---|---|---|---|---|
| `task_a_result_packet_manifest.json` | Declare the Step 3 objective review packet over atlas plus canonical Block 0/1 surfaces. | `workflow_name`, `packet_role`, `packet_spec_version`, `packet_root`, `central_index_path`, `human_index_path`, `layer_manifest_paths`, `layer_review_index_paths`, `included_layers`, `deferred_layers`, `surface_lineage`, `input_sources`, `artifact_counts` | Packet-local manifest; no Task A evidence state is inferred from packet assembly alone. | packet-local |
| `task_a_result_packet_index.csv` | Record one row per mirrored or packet-local review artifact in the Step 3 packet. | `layer`, `artifact_name`, `expected_relative_path`, `packet_relative_path`, `artifact_status`, `contract_alignment`, `implementation_tier`, `evidence_lineage`, `artifact_kind`, `claim_scope`, `review_role`, `analysis_level`, `source_workflow`, `sha256` | Packet-local index; available rows must hash-match the mirrored packet files. Atlas rows are descriptive-only and need not carry lineage values. | packet-local |
| `RESULTS_INDEX.md` | Provide a human-readable packet entrypoint summarizing included layers, deferred layers, and suggested first-inspection artifacts. | Human-readable markdown summary. | Packet-local only. | packet-local |

Packet-specific boundary notes:
- The Step 3 packet includes only `atlas`, `block0`, and `block1`.
- `block3` is explicitly deferred from the Step 3 packet manifest rather than silently omitted.
- The packet index keeps `implementation_tier` and `evidence_lineage` columns for prepare/Block rows so canonical rerun files and preserved proxy-history files cannot be confused. Descriptive atlas rows remain descriptive-only.
- Any Block 3-facing packet builder, packet-local review surface, or preserved
  historical packet is result-layer implementation context only. None of those
  packet surfaces is current live Block 3 scientific authority or canonical
  Block 3 result authority for this pass.

## Block 3 deferred public boundary

The Block 3 scientific contract remains frozen in the docs hierarchy. The Task
A repository still does **not** ship an active public Block 3 workflow,
review CLI, or packet bridge. The former public/demo-style Block 3 route was
removed after it was judged too polluted by stale proxy-era contract topology.

The repository may carry an on-disk `tasks/task_A/block3/` package for
internal Phase 3 execution, registry construction, execution planning, raw
bundle writing, and review-surface writing. That package may execute real
generator/method/scoring logic against Stage 0 h5ad plus Task A config, but it
remains implementation context only: non-authority, non-public, and not a
proof-carrying workflow surface.

Current engineering policy:

- Block 3 internal execution uses `python -m tasks.task_A.block3` with semantic
  experiment names.
- `package_results` has no Block 3 manifest bridge parameter.
- `result_packet.py` keeps `block3` explicitly deferred rather than proof-carrying.
- `design_freeze.py` no longer lists active Block 3 executable surfaces or
  Block 3 artifacts in the active engineering registry.

Use the docs hierarchy for Block 3 scientific meaning, but do not treat this
repository state as having a public executable Block 3 producer, review
surface, or packet bridge.

## Non-goal reminders

- No Step 1 artifact may be read as Block 0 calibration evidence.
- Semi-synthetic exports are benchmark contracts, not real-data evidence surfaces.
- No internal Block 3 Phase 3 artifact may be read as public comparator evidence or final method-validation closure.
