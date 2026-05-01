# Task A Operations

This is the live Task A engineering runbook for the current
canonical Task A Block 0-2 first-pass path and the deferred/internal Block 3
boundary. The canonical full STRIDE definition now lives in
[`docs/stride_design_freeze.md`](/home/lenislin/Experiment/projects/STRIDE/docs/stride_design_freeze.md),
and the frozen Task A migration target lives in
[`docs/task_A_rewiring_plan.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A_rewiring_plan.md).
Task A scientific boundaries and preserved proxy-history context remain in
[`docs/task_A_spec.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A_spec.md).
The canonical Task A results memo through Block 2, with explicit preserved
proxy-history context, lives in
[`docs/task_A_result.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A_result.md).
This README is an operational mirror. Current `fit_stride(...)` implementation
status and supported-input boundaries are owned by
[`docs/state.md`](/home/lenislin/Experiment/projects/STRIDE/docs/state.md).

## Start here

- If you do not yet have a Task A Stage 0 h5ad, start with
  [`stage0/build_artifacts.py`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/stage0/build_artifacts.py).
- If you already have a frozen Stage 0 h5ad, start with the descriptive atlas
  and Step 1 prepare as separate Stage 0 consumers.
- Run [`descriptive/`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/descriptive)
  before Block 0 to export the descriptive biological context layer.
- To gather the currently available descriptive atlas plus downstream Task A
  artifact surface into one objective human-review packet, use
  [`workflows/package_results.py`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/workflows/package_results.py).
  The packet builder covers atlas plus Block 0-2 surfaces for the current
  result packet layer.
- For Block 3 scientific meaning inside live Task A, use this authority chain:
  - [`docs/task_A_spec.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A_spec.md)
  - [`docs/task_A_block3_redesign_v1_1.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A_block3_redesign_v1_1.md)
  - this README, [`execution_graph.md`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/contracts/execution_graph.md), [`artifact_contracts.md`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/contracts/artifact_contracts.md), and [`block3_execution_runbook.md`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/block3_execution_runbook.md) as derived operational mirrors

## Agent-first Operations

- For agent routing and verification rules, start with
  [`AGENTS.md`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/AGENTS.md).
- For a focused Block 3 change, use
  [`docs/agent/playbooks/task-a-block3-change.md`](/home/lenislin/Experiment/projects/STRIDE/docs/agent/playbooks/task-a-block3-change.md).
- For doc-only routing updates or final verification discipline, use the
  playbooks indexed in
  [`docs/agent/README.md`](/home/lenislin/Experiment/projects/STRIDE/docs/agent/README.md).
- These agent docs are operational helpers only; they do not replace the Task A
  authority chain documented above.

## Step 3 boundary

- This runbook now documents the canonical Task A first-pass rerun path through
  Block 2.
- It uses the Task A authority chain and the core STRIDE source-of-truth order.
- The Step 3 packet layer covers atlas plus Block 0-2 evidence surfaces.

## What Step 1 already guarantees

- `prepare` is the canonical Step 1 real-data entrypoint.
- `data.mass_mode` is frozen to `uniform`.
- Semantic misalignment hard-fails instead of degrading into a warning report.
- Machine-readable readiness is frozen to:
  - `scaffold_active`
  - `contract_passed`
  - `evidence_ready`
- Full-cohort `prepare` emits `contract_passed` only when the Step 1 alignment
  contract holds.
- Subset or demo-subset `prepare` emits `scaffold_active` for cheap wiring
  checks only.
- No Step 1 artifact substitutes for Block 0 passage or Block 1 evidence.

## What Step 2 freezes

- Surface responsibilities stay task-local under `tasks/task_A/`.
- The block ids and Python entrypoints are:
  - `block0_locality_gate`
  - `block1_continuity_backbone`
  - `block2_bounded_audit`
- The canonical real-data order is:
  1. build Stage 0 artifacts if needed
  2. run full-cohort `prepare`
  3. run the descriptive atlas
  4. run Block 0 locality gate
  5. run Block 1 real-data biological discovery with a passed Block 0 bundle
  6. run Block 2 robustness over Block 1 summaries from an evidence-ready Block 1 bundle
  7. run Block 3 method validation from an evidence-ready Block 2 manifest
- The destructive-refactor stage labels mirror that scientific order as:
  `descriptive -> block0 -> block1 -> block2 -> block3a -> block3b-1 ->
  block3b-2 -> block3c-1 -> block3c-2 -> block3c-3`.
- Under the current frozen Block 3 contract, `block3c-1` maps to
  `recurrence_ablation`, `block3c-2` maps to `geometry_ablation`, and
  `block3c-3` maps to `consistency_ablation`. A consistency-first 3C ordering
  requires an explicit contract migration before it can be used in live
  mirrors, code, tests, schemas, or output names.
- The hard file prerequisites are narrower than the canonical order:
  - The descriptive atlas consumes `stage0_h5ad` plus `task-config` directly
    and remains the biological context layer above Block 0.
  - Block 1 consumes `stage0_h5ad` plus a passed Block 0 bundle and
    recomputes its own mapping/dry-run files plus frozen summary exports.
  - Block 2 consumes the Block 1 bundle surface plus the frozen Stage 0 and
    Block 1 summary/comparison paths declared inside that bundle.
- Semi-synthetic exports remain a sidecar benchmark path. There is no
  dedicated Task A semisynthetic CLI runner in this pass.
- There is still no task-global real-data runner or semisynthetic runner.
- The descriptive atlas now owns a task-local output index surface; this does
  not create a shared task-global export registry.

## Frozen Block 1 summary contract

- `block1_family_summary.csv` is the family-level summary surface on
  `patient_id x pair_family x scale x summary_name`.
- `block1_source_community_summary.csv` is the source-community surface on
  `patient_id x pair_family x source_community`.
- `block1_target_community_summary.csv` is the target-community surface on
  `patient_id x pair_family x target_community`.
- `block1_confirmatory_family_comparison.csv` is the patient-paired
  confirmatory `TC-IM` versus `TC-PT` family comparison surface on
  `patient_id x summary_name x scale`.
- `block1_exploratory_source_community_comparison.csv` and
  `block1_exploratory_target_community_comparison.csv` are exploratory/supportive
  community comparison surfaces and do not redefine confirmatory scope.
- Source-side frozen scientific summaries are `SR / D / R`, where
  `R = sum_{j != i} A_ij` and `SR` is strict self-retention.
- Target-side `E` remains exported as supportive context only.
- Family-level estimands are frozen to `burden_weighted` and `community_mean`.
- Source eligibility defaults to non-zero source burden.
- `community_correspondence/` exports an objective packet with
  community-by-cell-subtype tables, major-target tables, burden-component
  tables, and a community-id crosswalk for later human reading.
- `block1_bundle.json` and `block1_workflow_manifest.json` now point to these
  summary/comparison/correspondence files and carry the summary-contract
  metadata instead of `primary_evidence_lines`.
- Canonical Block 1 also exports cohort-level recurrence/context artifacts:
  `block1_recurrence_summary.json`,
  `block1_recurrence_families.json`, and
  `block1_recurrence_embeddings.csv`.

## Frozen Block 2 robustness contract

- `block2_bounded_audit` is the robustness layer for frozen Block 1 findings.
- Block 2 reuses the same Block 1 family/source/target comparison objects under
  perturbation.
- Primary Block 2 routes are patient perturbation routes:
  `patient_subsample` and `leave_some_out`.
- `seed_rerun` and `roi_drop_one_per_domain` remain secondary sensitivity
  routes.
- `block2_bounded_audit_summary.csv` is the top-level finding call table over
  primary routes.
- `block2_family_robustness.csv`,
  `block2_source_community_robustness.csv`, and
  `block2_target_community_robustness.csv` are the proof-carrying route-level
  robustness tables.
- `block2_replicate_manifest.csv` records which perturbations were attempted,
  retained patient/ROI/cell counts, and any failed replicates.
- Long-running Block 2 full-cohort runs can be resumed by rerunning
  `run_block2 --resume` against the same output directory.
- Block 2 reports robustness over Block 1 summaries.

## Current Block 3 boundary

- Block 3 scientific authority remains frozen in:
  - `docs/task_A_spec.md`
  - `docs/task_A_block3_redesign_v1_1.md`
- The live scientific section structure is:
  - `3A generator validation`
  - `3B baseline comparison`
    - `3B-1 A benchmark`
    - `3B-2 d/e benchmark`
  - `3C ablation study`
    - `3C-1 recurrence ablation`
    - `3C-2 geometry ablation`
    - `3C-3 consistency ablation`
- This section structure is exhaustive for the live Block 3 public naming
  surface.
- The task-local stage labels therefore stay aligned as `block3a`,
  `block3b-1`, `block3b-2`, `block3c-1`, `block3c-2`, and `block3c-3`;
  `block3b-1` and `block3b-2` are separate benchmark stages rather than one
  mixed `3B` implementation surface.
- `3C` is the core STRIDE refit-ablation surface. Each `3C` arm removes or
  zero-weights the corresponding objective term and refits `A_p`, `d_p`, and
  `e_p` on the same rerun-specific patient-level semi-synthetic realization,
  resolved evidence blocks, deterministic initialization, and optimizer
  protocol as the reference fit.
- `3C` uses fixed full-estimator group denominators. Retained objective terms
  keep the reference-fit objective coefficients.
- The live `3C` native patient-level readout surface is `A_MAE_active`,
  `A_MSE_active`, `d_MAE`, `e_MAE`, `d_MSE`, `e_MSE`, and
  `open_support_F1`, with metric activity reported through the
  `reported` / `not_applicable` / `not_estimable` status semantics in
  `docs/task_A_spec.md`.
- The active public Block 3 runner/review/packet bridge remains removed from
  the execution path. The on-disk `tasks/task_A/block3/` package may host
  internal single-subexperiment Phase 3 execution with real generator/method/
  scoring logic plus raw/review artifact writing, but it is still not a public
  workflow surface and is not proof-carrying authority.
- `run_block3` now exists only as a fail-fast deprecation wrapper so callers do
  not accidentally treat the internal Block 3 package as a live public runner.
- `review_block3` remains a fail-fast deprecation stub.
- `package_results` remains atlas/Block 0-2 only and fail-fast rejects
  `--block3-manifest`.
- The Block 3 runbook and contract mirrors in `tasks/task_A/contracts/` now
  describe deferred public workflow/packet state plus the internal
  non-authority execution boundary. They do not authorize public execution,
  packet proof-carrying claims, or review-layer resurrection.
- For a current implementation audit of the internal Phase 3 Block 3 stack,
  see
  [`block3_phase3_audit_memo.md`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/contracts/block3_phase3_audit_memo.md)
  and
  [`block3_phase3_traceability_matrix.md`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/contracts/block3_phase3_traceability_matrix.md).
- `tasks/task_A/block3/` is the internal Block 3 rebuild package. It carries
  generator validation, baseline comparison, ablation execution, and raw/review
  writers for internal Phase 3 runs.
- A public Block 3 workflow or packet bridge requires an explicit follow-up
  specification.

## Pre-coding review gates

Before the next Block 1/2 coding round, two review notes must be completed:

- [`descriptive_atlas_review_memo.md`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/contracts/descriptive_atlas_review_memo.md)
- [`contract_alignment_review_memo.md`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/contracts/contract_alignment_review_memo.md)

## Block 0 gate freeze

- Block 0 is now STRIDE-native and is defined on realized `TC-IM` fit outputs.
- The real family is the Task A near-proxy family `TC-IM`.
- The null family is `TC-IM_randomized_target`:
  keep each anchor patient's `TC` observations fixed and reassign the `IM`
  group from a different patient in the same exact `(n_TC, n_IM)` count
  stratum.
- Null-family donor assignment is seeded and reproducible. Singleton strata do
  not fall back to a looser random control; they emit a deferred null fit for
  that anchor patient instead.
- The live Block 0 gate summaries are derived directly from `A`, `d`, and `e`,
  with pass/fail driven by paired deltas on `sum(A)` and `sum(e)`.

## Surface map

| Surface | Consumes | Produces | Artifact state(s) | Does not do |
|---|---|---|---|---|
| `build_stage0_artifacts` | CRLM cohort RDS, build params | `task_A_stage0_k{K}.h5ad`, `task_A_stage0_validation.json` | `scaffold_active`, `contract_passed` | Does not run Step 1 or any block |
| `check_data_suitability` | `--task-config`, `--stage0-h5ad`, `--output-dir` | `task_a_pre_block0_data_suitability.json` | `scaffold_active`, `contract_passed` | Does not satisfy Block 0 |
| `prepare` | `--task-config`, `--stage0-h5ad`, `--output-dir`, optional subset selectors | `task_a_stride_mapping.json`, `task_a_core_fit_dry_run.csv`, `task_a_prepare_manifest.json` | `scaffold_active`, `contract_passed` | Does not satisfy Block 0 |
| `descriptive_atlas` | `--task-config`, `--stage0-h5ad`, `--output-dir`, optional `--patient-id`, optional `--max-overlay-communities` | `task_a_descriptive_atlas_manifest.json`, `task_a_descriptive_atlas_output_index.csv`, atlas tables/figures | descriptive only | Does not run Block 0/1/2 or emit confirmatory claims |
| `run_block0` | `--task-config`, `--stage0-h5ad`, `--output-dir`, optional subset selectors | `block0_bundle.json`, `block0_pair_metrics.csv`, `task_a_pre_block0_data_suitability.json` | `scaffold_active`, `contract_passed` | Does not read Step 1 files as hard inputs or redesign Block 1/2 |
| `run_block1` | `--task-config`, `--stage0-h5ad`, `--block0-bundle`, `--output-dir` | `block1_stage0_mapping.json`, `block1_core_fit_dry_run.csv`, `block1_recurrence_summary.json`, `block1_recurrence_families.json`, `block1_recurrence_embeddings.csv`, `block1_family_summary.csv`, `block1_source_community_summary.csv`, `block1_target_community_summary.csv`, `block1_confirmatory_family_comparison.csv`, `block1_exploratory_source_community_comparison.csv`, `block1_exploratory_target_community_comparison.csv`, `community_correspondence/`, `block1_bundle.json`, `block1_workflow_manifest.json` | `evidence_ready` | Requires a passed Block 0 bundle |
| `run_block2` | `--block1-bundle`, `--output-dir` | `block2_bounded_audit_summary.csv`, `block2_contract_audit.csv`, `block2_replicate_manifest.csv`, `block2_family_robustness.csv`, `block2_source_community_robustness.csv`, `block2_target_community_robustness.csv`, `block2_bounded_audit_manifest.json` | `evidence_ready` | Does not compare baselines, redesign Block 1, or prove true emergence/disappearance |
| `run_block3` | reserved | fail-fast message | reserved | Public workflow requires a follow-up specification |
| `review_block3` | reserved | fail-fast message | reserved | Public review workflow requires a follow-up specification |
| `package_results` | `--atlas-manifest`, `--prepare-manifest`, `--block0-bundle`, `--output-dir`, optional `--block0-suitability-report`, optional `--block1-bundle`, optional `--block2-manifest` | `task_a_result_packet_manifest.json`, `task_a_result_packet_index.csv`, `RESULTS_INDEX.md`, layer manifests/review indexes, mirrored atlas/Block 0/Block 1/Block 2 files, and packet-local review tables | packet-local only | Does not interpret biology, re-run large experiments, fabricate missing result surfaces, package Block 3 into the Step 3 canonical packet, or accept `--block3-manifest` before a clean non-authority bridge is approved |
| `write_semisynthetic_artifacts` | output root, manifest filename, patient count, seed | semisynthetic manifest CSV, `task_a_semisynthetic_contract.json` | `contract_passed` | Does not run the real-data graph |
| `stride_adapter` helpers | Stage 0 h5ad/AnnData, config bundle | mapping summaries, family-sliced observations, dry-run rows | helper only | Does not touch `src/stride/` semantics |

## Layout

| Directory | Purpose |
|---|---|
| `config/` | Typed config loading and frozen Task A config validation. |
| `contracts/` | Machine-readable freeze registry, execution graph, artifact contracts, mapping contract, and deferred-boundary notes. |
| `workflows/` | Canonical entry surfaces and the task-local STRIDE adapter. |
| `block0/` | Block 0 provenance contract helpers plus the STRIDE-native real-vs-null gate implementation. |
| `block1/` | Block 1 real-data biological discovery bundle plus frozen summary writers. |
| `block2/` | Block 2 perturb-and-reestimate robustness writer. |
| `block3/` | Internal Block 3 rebuild package for generator validation, baseline comparison, ablation execution, and raw/review writers. |
| `stage0/` | Stage 0 extraction, building, and validation helpers. |
| `real_data/` | Frozen demo-subset definitions for cheap subset wiring checks. |
| `semisynthetic/` | Deterministic benchmark world builders. |
| `benchmarks/` | Semisynthetic export helpers. |

## Deferred surfaces

- Task A Block 0-2 call canonical `fit_stride(...)`.
- Standalone bridge-expert APIs remain deferred.
- New public STRIDE core APIs require a separate contract update.

## Examples

Full-cohort Step 1:

```bash
PYTHONPATH=.:src python -m tasks.task_A.workflows.prepare \
  --task-config tasks/task_A/config.yaml \
  --stage0-h5ad /mnt/NAS_21T/ProjectData/STRIDE/task_A_stage0/task_A_stage0_k25.h5ad \
  --output-dir /tmp/task_a_prepare_full
```

Descriptive atlas from Stage 0:

```bash
PYTHONPATH=.:src python -m tasks.task_A.descriptive \
  --task-config tasks/task_A/config.yaml \
  --stage0-h5ad /mnt/NAS_21T/ProjectData/STRIDE/task_A_stage0/task_A_stage0_k25.h5ad \
  --output-dir /tmp/task_a_descriptive_atlas_full
```

Pre-Block 0 suitability report:

```bash
PYTHONPATH=.:src python -m tasks.task_A.workflows.check_data_suitability \
  --task-config tasks/task_A/config.yaml \
  --stage0-h5ad /mnt/NAS_21T/ProjectData/STRIDE/task_A_stage0/task_A_stage0_k25.h5ad \
  --output-dir /tmp/task_a_data_suitability
```

Block 0 smoke subset sidecar:

```bash
PYTHONPATH=.:src python -m tasks.task_A.workflows.run_block0 \
  --task-config tasks/task_A/config.yaml \
  --stage0-h5ad /mnt/NAS_21T/ProjectData/STRIDE/task_A_stage0/task_A_stage0_k25.h5ad \
  --patient-id B10 \
  --patient-id B11 \
  --patient-id B12 \
  --patient-id B16 \
  --output-dir /tmp/task_a_block0_smoke
```

Block 0 full cohort:

```bash
PYTHONPATH=.:src python -m tasks.task_A.workflows.run_block0 \
  --task-config tasks/task_A/config.yaml \
  --stage0-h5ad /mnt/NAS_21T/ProjectData/STRIDE/task_A_stage0/task_A_stage0_k25.h5ad \
  --output-dir /tmp/task_a_block0_full
```

Block 1 from a passed Block 0 bundle:

```bash
PYTHONPATH=.:src python -m tasks.task_A.workflows.run_block1 \
  --task-config tasks/task_A/config.yaml \
  --stage0-h5ad /mnt/NAS_21T/ProjectData/STRIDE/task_A_stage0/task_A_stage0_k25.h5ad \
  --block0-bundle /path/to/passed/block0_bundle.json \
  --output-dir /tmp/task_a_block1
```

Block 2 from an evidence-ready Block 1 bundle:

```bash
PYTHONPATH=.:src python -m tasks.task_A.workflows.run_block2 \
  --block1-bundle /tmp/task_a_block1/block1_bundle.json \
  --output-dir /tmp/task_a_block2
```

Resume an interrupted Block 2 long run:

```bash
PYTHONPATH=.:src python -m tasks.task_A.workflows.run_block2 \
  --block1-bundle /tmp/task_a_block1/block1_bundle.json \
  --output-dir /tmp/task_a_block2 \
  --resume
```

Objective Step 3 result packet from the canonical atlas plus Block 0-2
surfaces, with Block 3 explicitly deferred:

```bash
PYTHONPATH=.:src python -m tasks.task_A.workflows.package_results \
  --atlas-manifest /tmp/task_a_descriptive_atlas_full/task_a_descriptive_atlas_manifest.json \
  --prepare-manifest /tmp/task_a_prepare_full/task_a_prepare_manifest.json \
  --block0-bundle /path/to/block0_bundle.json \
  --block0-suitability-report /path/to/task_a_pre_block0_data_suitability.json \
  --block1-bundle /path/to/block1_bundle.json \
  --block2-manifest /path/to/block2_bounded_audit_manifest.json \
  --output-dir tasks/task_A/result_packets/DATE_task_a_objective_packet
```

Packet notes:
- `task_a_result_packet_manifest.json` now declares `included_layers=["atlas", "block0", "block1", "block2"]`.
- `task_a_result_packet_manifest.json` now declares `deferred_layers=["block3"]`.
- `--prepare-manifest` is an explicit packet input; the descriptive atlas
  manifest no longer carries Step 1 prepare or mapping pointers.
- `package_results` fail-fast rejects `--block3-manifest` with
  `Block 3 packet integration is deferred / non-authority / pending clean bridge spec`.
- `task_a_result_packet_index.csv` keeps lineage columns for prepare/Block rows; descriptive atlas rows remain descriptive-only.
