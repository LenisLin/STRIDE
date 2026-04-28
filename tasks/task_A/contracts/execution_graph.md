# Task A Execution Graph

This file records the canonical Task A execution order for Step 2 as a derived
operational mirror of the live Task A scientific documents. For Block 3
scientific meaning, the authority chain is
[`docs/task_A_spec.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A_spec.md),
then
[`docs/task_A_block3_redesign_v1_1.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A_block3_redesign_v1_1.md),
then active derived docs such as this execution graph. If consulted,
[`design_freeze.py`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/contracts/design_freeze.py)
is legacy/stale machine-readable inventory and implementation-lag evidence
only, not current Block 3 authority.

This graph describes execution routing and output carriers only. It does not
override Block 3 scientific authority or promote packet/review surfaces into
result authority.

## Canonical order

```text
build_stage0_artifacts
  -> prepare(full cohort)
  -> descriptive_atlas
  -> run_block0
  -> run_block1
  -> run_block2

Sidecar paths:
- check_data_suitability branches from the same Stage 0 input surface
- prepare(subset/demo) branches from the same Stage 0 input surface
- write_semisynthetic_artifacts is an independent benchmark export path
```

Block 3 remains scientifically specified in docs, but its public engineering
surface is currently deferred from the active execution graph. The on-disk
`tasks/task_A/block3/` package may support internal non-authority Phase 3
execution from evidence-ready Block 1/2 inputs, but that does not reopen a
public `run_block3` node.

## Frozen nodes

| Node | Surface | Hard prerequisites | Produces | Status | Notes |
|---|---|---|---|---|---|
| `stage0_artifact_builder` | `build_stage0_artifacts` | CRLM cohort RDS | `task_A_stage0_k{K}.h5ad`, `task_A_stage0_validation.json` | executable | Builds the frozen Task A input surface only. |
| `step1_prepare_full_cohort` | `prepare_task_a_stage0_mapping` | `stage0-h5ad`, `task-config` | `task_a_stride_mapping.json`, `task_a_core_fit_dry_run.csv`, `task_a_prepare_manifest.json` | executable | Canonical Step 1 real-data entrypoint. |
| `descriptive_atlas_context_layer` | `write_task_a_descriptive_atlas` | `task_a_prepare_manifest.json` | `task_a_descriptive_atlas_manifest.json`, `task_a_descriptive_atlas_output_index.csv`, atlas tables, atlas figures | executable | Canonical biological context layer; descriptive only and upstream of Block 0. |
| `pre_block0_data_suitability_report` | `check_task_a_pre_block0_data_suitability` | `stage0-h5ad`, `task-config` | `task_a_pre_block0_data_suitability.json` | executable | Report wrapper over Step 1 semantics; never substitutes for Block 0. |
| `step1_prepare_subset_or_demo` | `prepare_task_a_stage0_mapping` | `stage0-h5ad`, `task-config`, subset selector | same as full prepare | executable | Cheap wiring path only; emits `scaffold_active`. |
| `block0_locality_gate` | `run_block0_workflow` | `stage0-h5ad`, `task-config` | `block0_bundle.json`, `block0_pair_metrics.csv` | executable | Runs the STRIDE-native `TC-IM` real-vs-null gate after the descriptive atlas context layer. Full cohort may emit `contract_passed`; subset/demo sidecars remain non-passing. |
| `block1_continuity_backbone` | `write_task_a_block1_bundle` | `stage0-h5ad`, `task-config`, passed `block0_bundle.json` | `block1_stage0_mapping.json`, `block1_core_fit_dry_run.csv`, `block1_family_summary.csv`, `block1_source_community_summary.csv`, `block1_target_community_summary.csv`, `block1_confirmatory_family_comparison.csv`, `block1_exploratory_source_community_comparison.csv`, `block1_exploratory_target_community_comparison.csv`, `community_correspondence/block1_community_correspondence_manifest.json`, `community_correspondence/block1_community_correspondence_index.csv`, `block1_bundle.json`, `block1_workflow_manifest.json` | executable | Legacy block id retained for compatibility; scientific role is real-data biological discovery via the frozen summary contract. |
| `block2_bounded_audit` | `write_block2_bundle` | `block1_bundle.json` | `block2_bounded_audit_summary.csv`, `block2_contract_audit.csv`, `block2_replicate_manifest.csv`, `block2_family_robustness.csv`, `block2_source_community_robustness.csv`, `block2_target_community_robustness.csv`, `block2_bounded_audit_manifest.json` | executable | Compatibility-named Block 2 surface that perturbs the frozen cohort, re-estimates the same Block 1 objects, and summarizes robustness from an evidence-ready Block 1 bundle. |
| `semisynthetic_benchmark_export` | `write_semisynthetic_artifacts` | benchmark settings | semisynthetic manifest CSV, `task_a_semisynthetic_contract.json` | executable | Independent benchmark/export path; not a real-data runner. |

## Dependency clarifications

- Block 1 recomputes its own mapping and dry-run surfaces from `stage0_h5ad`
  plus a passed Block 0 bundle. It does **not** read Step 1 artifacts as hard
  file inputs.
- Block 1 bundle/manifest provenance is now summary-contract based:
  `family_summary_path`, `source_community_summary_path`,
  `target_community_summary_path`, paired comparison paths,
  correspondence-packet paths, summary roles, summary scales, and eligibility
  rules replace the older `primary_evidence_lines` wording.
- Step 1 full-cohort prepare is still the canonical readiness check before the
  descriptive atlas and downstream block execution path.
- The descriptive atlas consumes the Step 1 prepare manifest so that run scope,
  patient subset provenance, and descriptive-only labeling stay anchored to the
  frozen Step 1 surface.
- Block 2 consumes the Block 1 bundle plus the frozen Stage 0 and Block 1
  output paths declared inside that bundle. It re-estimates the same Block 1
  summary/comparison objects under perturbation rather than introducing a new
  metric system.
- Block 3 remains scientifically downstream of the established Block 0-2
  evidence stack, but its engineering surface is currently removed/deferred.
- Any future Block 3 rebuild must start from the docs hierarchy rather than
  preserved packet-local or proxy-era carrier names.
- The compatibility filename `block2_bounded_audit_summary.csv` remains frozen
  even though the scientific role is robustness over frozen Block 1 findings
  rather than a dry-run-only bounded audit.
