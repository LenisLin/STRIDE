# Task A Execution Graph

This file records the canonical Task A execution order for Step 2 as a derived
operational mirror of the live Task A scientific documents. For Block 3
scientific meaning, the authority chain is
[`docs/task_A/spec.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A/spec.md),
then
[`docs/task_A/block3/scientific_contract.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A/block3/scientific_contract.md),
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
  -> task_a_pair_adata_adapter
  -> block0_execute
  -> block0_analyze
  -> block1_real_data_discovery

Sidecar paths:
- prepare(subset/demo) branches from the same Stage 0 input surface
- write_semisynthetic_artifacts is an independent benchmark export path
```

Block 3 remains scientifically specified in docs, but its public engineering
surface is not part of the active public execution graph. The on-disk
`tasks/task_A/block3/` package supports internal non-authority Phase 3
execution from Stage 0 h5ad plus Task A config using semantic experiment names,
but that does not reopen a public Block 3 workflow node.

Block 0, Block 1, and the Block 3 reference method share the task-local route
`prepare_task_a_pair_adata -> fit_task_a_pair -> stride.tl.fit ->
extract_task_a_relations`. Task-owned labels, permutation semantics, benchmark
truth, and artifact serialization remain outside `src/stride`. The three Block
3C semantic names stop before Stage 0 loading and write structured deferred
status while the public estimator lacks ablation hooks.

## Frozen nodes

| Node | Surface | Hard prerequisites | Produces | Status | Notes |
|---|---|---|---|---|---|
| `stage0_artifact_builder` | `build_stage0_artifacts` | CRLM cohort RDS | `task_A_stage0_k{K}.h5ad`, `task_A_stage0_validation.json` | executable | Builds the frozen Task A input surface only. |
| `step1_prepare_full_cohort` | `prepare_task_a_stage0_mapping` | `stage0-h5ad`, `task-config` | `task_a_stride_mapping.json`, `task_a_core_fit_dry_run.csv`, `task_a_prepare_manifest.json` | executable | Canonical Step 1 real-data entrypoint. |
| `descriptive_atlas_context_layer` | `write_task_a_descriptive_atlas` | `stage0-h5ad`, `task-config` | `task_a_descriptive_atlas_manifest.json`, `task_a_descriptive_atlas_output_index.csv`, atlas tables, atlas figures | executable | Canonical biological context layer; descriptive only and upstream of Block 0. |
| `step1_prepare_subset_or_demo` | `prepare_task_a_stage0_mapping` | `stage0-h5ad`, `task-config`, subset selector | same as full prepare | executable | Cheap wiring path only; emits `scaffold_active`. |
| `task_a_pair_adata_adapter` | `prepare_task_a_pair_adata` | `stage0-h5ad`, Task A ordered pair family, optional patient subset | pp-ready pair AnnData | executable adapter | Task-local bridge from Stage 0 aliases to `.pp-ready` AnnData for `stride.tl.fit`; does not redefine Task A science. |
| `block0_execution_cache` | `run_block0_execute` | `stage0-h5ad`, `task-config`, `n-permutations`, `master-seed` | `block0_execution_manifest.json`, `block0_fit_cache.npz`, `block0_fit_cache_index.csv` | executable | Real and null fits enter `stride.tl.fit` through the Task A pair/null AnnData route; cache extraction derives burden fields from the fitted AnnData context. |
| `block0_calibration_analysis` | `analyze_block0_cache` | `block0_fit_cache.npz`, `block0_fit_cache_index.csv`, optional execution manifest | `block0_calibration_manifest.json`, `block0_patient_calibration.csv`, `block0_metric_summary.csv` | executable | Derives fixed family-summary calibration tables from a cache without rerunning STRIDE or interpreting biology. |
| `block1_real_data_discovery` | `python -m tasks.task_A.block1 execute/analyze` | `stage0-h5ad`, `task-config` | `stride.tl.fit` native result manifests, family/source/target summaries, `TC-IM` versus `TC-PT` descriptive contrast tables, cohort relation comparison table | executable | Live Block 1 id for the descriptive real-data discovery layer; analyze reads the task-native `.tl.fit` payload through a compatibility adapter. |
| `semisynthetic_benchmark_export` | `write_semisynthetic_artifacts` | benchmark settings | semisynthetic manifest CSV, `task_a_semisynthetic_contract.json` | executable | Independent benchmark/export path; not a real-data runner. |

## Dependency clarifications

- Block 0 execution and Block 1 share `stage0_h5ad` and Task A config. Block 1
  reads those inputs directly; it does not consume the raw Block 0 fit cache as
  a biological result surface.
- Live fitting enters `stride.tl.fit` only after the task-local
  `task_a_pair_adata_adapter` resolves ordered pair metadata and validates
  `.pp-ready` AnnData.
- Direct Block 1 summaries and descriptive contrasts are the live Block 1
  analysis outputs for this phase.
- Step 1 full-cohort prepare is still the canonical readiness check for the
  STRIDE-consuming downstream block execution path.
- The descriptive atlas consumes Stage 0 plus the Task A config directly and
  stays descriptive-only; it does not carry Step 1 mapping or dry-run context.
- Block 3 remains scientifically downstream of the established Block 0/1
  evidence stack. Its internal `stride_reference` route enters
  `stride.tl.fit`; 3C ablation refit arms require a live core ablation hook
  before they can be claimed as executable refits.
