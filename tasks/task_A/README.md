# Task A Operations

This is the live Task A engineering runbook for the current
canonical Task A Block 0/1 first-pass path and the deferred/internal Block 3
boundary. The canonical full STRIDE definition now lives in
[`docs/stride_design_freeze.md`](/home/lenislin/Experiment/projects/STRIDE/docs/stride_design_freeze.md),
and the frozen Task A migration target lives in
[`docs/task_A/spec.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A/spec.md).
Task A scientific boundaries and preserved proxy-history context remain in
[`docs/task_A/spec.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A/spec.md).
The canonical Task A results memo through Block 1, with explicit preserved
proxy-history context, lives in
[`docs/task_A/result.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A/result.md).
This README is an operational mirror. Current `stride.tl.fit(...)`
implementation status and supported-input boundaries are owned by
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
  The packet builder covers atlas plus Block 0/1 surfaces for the current
  result packet layer.
- For Block 3 scientific meaning inside live Task A, use this authority chain:
  - [`docs/task_A/spec.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A/spec.md)
  - [`docs/task_A/block3/scientific_contract.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A/block3/scientific_contract.md)
  - stage docs under [`docs/task_A/block3/`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A/block3)
  - [`docs/task_A/block3/refactor_contract_map.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A/block3/refactor_contract_map.md) for migration mapping only
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
  Block 1.
- It uses the Task A authority chain and the core STRIDE source-of-truth order.
- The Step 3 packet layer covers atlas plus Block 0/1 evidence surfaces.

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
- No Step 1 artifact substitutes for Block 0 calibration evidence or Block 1
  evidence.

## What Step 2 freezes

- Surface responsibilities stay task-local under `tasks/task_A/`.
- Task A current migration route is
  `Stage0 h5ad -> TaskA adapter -> pp-ready pair AnnData -> stride.tl.fit`.
- The block ids and Python entrypoints are:
  - `block0_calibration`
  - `block1_real_data_discovery`
- The canonical real-data order is:
  1. build Stage 0 artifacts if needed
  2. run full-cohort `prepare`
  3. run the descriptive atlas
  4. run Block 0 `TC-IM` empirical null calibration
  5. run Block 1 descriptive real-data discovery from Stage 0 plus config
  6. run Block 3 from Stage0 h5ad plus Task A config through the internal
     semantic CLI
- The destructive-refactor stage labels mirror that scientific order as:
  `descriptive -> block0 -> block1 -> block3a -> block3b-1 ->
  block3b-2 -> block3c-1 -> block3c-2 -> block3c-3`.
- Under the current frozen Block 3 contract, `block3c-1` maps to
  `subbag_consistency_ablation`, `block3c-2` maps to `geometry_ablation`, and
  `block3c-3` maps to `recurrence_ablation`.
- The hard file prerequisites are narrower than the canonical order:
  - The descriptive atlas consumes `stage0_h5ad` plus `task-config` directly
    and remains the biological context layer above Block 0.
  - Block 0 consumes `stage0_h5ad` plus `task-config` directly and does not
    read prepare or descriptive-atlas artifacts as hard inputs.
  - Block 1 consumes `stage0_h5ad` plus `task-config` directly. It does not
    read Block 0 outputs, descriptive-atlas outputs, or result packets.
- Semi-synthetic exports remain a sidecar benchmark path. There is no
  dedicated Task A semisynthetic CLI runner in this pass.
- There is still no task-global real-data runner or semisynthetic runner.
- The descriptive atlas now owns a task-local output index surface; this does
  not create a shared task-global export registry.

## Frozen Block 1 summary contract

The rebuilt Block 1 core schema does not retain legacy community-level target
incoming or source top-target shorthands. Source-community outputs report
all-community self-retention, depletion, and off-diagonal remodeling with
burden counterparts. Target-community outputs report target burden, matched
incoming burden, and target-open tendency/burden. Family-level `emergence` is
retained only as a supportive target-open rollup.

- `block1_family_summary.csv` is the family-level summary surface on
  `patient_id x pair_family x scale x summary_name`.
- `block1_source_community_summary.csv` is the source-community surface on
  `patient_id x pair_family x source_community`.
- `block1_target_community_summary.csv` is the target-community surface on
  `patient_id x pair_family x target_community`.
- `block1_confirmatory_family_comparison.csv` is the patient-paired
  confirmatory `TC-IM` versus `TC-PT` family comparison surface on
  `patient_id x summary_name x scale`.
- `block1_source_community_comparison.csv` and
  `block1_target_community_comparison.csv` are descriptive community contrast
  surfaces and do not redefine confirmatory scope.
- Source-side frozen scientific summaries are `SR / D / R`, where
  `R = sum_{j != i} A_ij` and `SR` is strict self-retention.
- Target-side `E` remains exported as supportive context only.
- Family-level estimands are frozen to `burden_weighted` and `community_mean`.
- Source eligibility defaults to non-zero source burden.
- Direct Block 1 summaries and descriptive contrasts are the current live
  artifact surface. Community annotation or crosswalk needs must be defined
  outside Block 1 engineering.
- Canonical Block 1 records raw full STRIDE fit outputs and raw
  recurrence/common-structure outputs during execution.
- `python -m tasks.task_A.block1 execute` writes only native relation exports
  plus `block1_execute_manifest.json`.
- `python -m tasks.task_A.block1 analyze` reads only the execute manifest and
  referenced native exports, then writes the three summary CSVs, the three
  direct comparison CSVs, `block1_cohort_relation_comparison.csv`, and
  `block1_analysis_manifest.json`.

## Current Block 3 boundary

- Block 3 scientific authority remains frozen in:
  - `docs/task_A/spec.md`
  - `docs/task_A/block3/scientific_contract.md`
- The live scientific section structure is:
  - `3A generator validation`
  - `3B baseline comparison`
    - `3B-1 A benchmark`
    - `3B-2 d/e benchmark`
  - `3C ablation study`
    - `3C-1 subbag consistency ablation`
    - `3C-2 geometry ablation`
    - `3C-3 recurrence ablation`
- This section structure is exhaustive for the live Block 3 public naming
  surface.
- The internal CLI uses semantic experiment names:
  `generator_validation`, `a_benchmark`, `de_benchmark`,
  `subbag_consistency_ablation`, `geometry_ablation`, and
  `recurrence_ablation`. The registry maps these names to the numbered
  `subexperiment_id` values.
- The internal CLI takes Stage 0 h5ad plus Task A config directly. Example:
  `python -m tasks.task_A.block3 generator_validation --task-config tasks/task_A/config.yaml --stage0-h5ad <stage0.h5ad> --output-dir <out>`.
- Formal runs omit engineering subset flags and use the contract default of
  `10` reruns and `8` held-out test patients. `--max-reruns <n>` and
  `--n-test <n>` are internal engineering-smoke controls only; manifests from
  such runs must record `execution_scope=subset_engineering_test`.
- First export `generator_validation` for manual sanity review. After manual
  confirmation, `a_benchmark` and `de_benchmark` may run through the formal
  `stride.tl.fit` reference surface. The three `3C` semantic names are
  recognized but currently return structured `deferred` status because the
  public estimator exposes no corresponding ablation hook.
- Block 3 uses the train-derived multi-FOV generator described in
  `docs/task_A/block3/scientific_contract.md`: train TC-IM templates,
  geometry-gated residual coupling with `tau=2.0`, medoid plus individual
  template mixture with `lambda_individual=0.10`, and FOV generation with
  `eta=0.3`. The live execution route does not use Block1/2 manifests or
  packet-local proxy diagnostics.
- `3C` remains the frozen refit-ablation design. Current execution does not
  implement those arms: it writes a structured unsupported/deferred record
  with the experiment name, requested ablation, `fit_surface="stride.tl.fit"`,
  reason code, and frozen-contract reference. It never copies a reference fit
  or emits a post-hoc surrogate as an ablation result.
- `3C` uses fixed full-estimator group denominators. Retained objective terms
  keep the reference-fit objective coefficients.
- `de_benchmark` is open-focused but keeps the complete shared metric
  vocabulary used by the method-bearing Block 3 routes.
- The live method-bearing patient-level readout surface includes
  `F_L1_total`, `g_L1_total`, `e_L1_total`, mass absolute-error metrics,
  `offdiag_ratio`, `depletion_capture`, `emergence_capture`,
  `endpoint_y_MAE`, `A_MAE_active`, `A_MSE_active`, `target_recall_at_k`,
  `open_support_F1`, `d_MAE`, `d_MSE`, `e_MAE`, and `e_MSE`, with metric
  activity reported through the
  `reported` / `not_applicable` / `not_estimable` status semantics in
  `docs/task_A/spec.md`.
- The active public Block 3 runner/review/packet bridge remains removed from
  the execution path. The on-disk `tasks/task_A/block3/` package hosts internal
  single-subexperiment Phase 3 execution with real generator/method/scoring
  logic plus raw/review artifact writing, but it is still not a public workflow
  surface and is not proof-carrying authority.
- `package_results` remains atlas/Block 0/1 only and has no Block 3 manifest
  bridge parameter.
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
- Reference-fit reuse across method-bearing routes is a deferred full-data
  runtime optimization. It is not a subset engineering-smoke acceptance
  condition.
- A public Block 3 workflow or packet bridge requires an explicit follow-up
  specification.

## Pre-coding review gates

Before the next Block 1 coding round, two review notes must be completed:

- [`descriptive_atlas_review_memo.md`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/contracts/descriptive_atlas_review_memo.md)
- [`contract_alignment_review_memo.md`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/contracts/contract_alignment_review_memo.md)

## Block 0 execution/cache and calibration freeze

- Block 0 is a `TC-IM` empirical null calibration layer split into execution
  cache generation and cache-derived analysis.
- The real family is the Task A near-proxy family `TC-IM`.
- The null family is a within-patient count-preserving `TC-IM` FOV/ROI
  domain-label permutation null. It preserves `patient_id`, FOV composition,
  FOV count structure, and each patient's exact `n_TC`/`n_IM` counts while
  randomly permuting `TC`/`IM` labels inside the same patient. Identity
  permutations are allowed; cross-patient label borrowing and relaxed fallback
  are not allowed.
- Formal full calibration uses `B=199` permutations. `B` remains configurable
  for diagnostics; smaller checks use smaller `B` or subsets rather than a
  separate smoke-mode contract.
- Full-cohort Block 0 runs may parallelize independent null permutations.
  The recommended formal setting is 8 permutation workers with 4 CPU threads
  per worker.
- Block 0 execution writes `block0_execution_manifest.json`,
  `block0_fit_cache.npz`, `block0_fit_cache_index.csv`, and progress. It does
  not compute p-values or derived scientific metrics.
- Block 0 analysis consumes an existing fit cache and writes
  `block0_calibration_manifest.json`, `block0_patient_calibration.csv`, and
  `block0_metric_summary.csv`.
- Cache-derived analysis reports raw plus-one empirical p-values on the fixed
  family-summary surface only. It does not perform FDR/Bonferroni correction,
  emit significance labels, or create a pass/fail gate.
- Block 0 cache and analysis outputs do not emit biological interpretation or
  downstream execution authorization.

## Surface map

| Surface | Consumes | Produces | Artifact state(s) | Does not do |
|---|---|---|---|---|
| `build_stage0_artifacts` | CRLM cohort RDS, build params | `task_A_stage0_k{K}.h5ad`, `task_A_stage0_validation.json` | `scaffold_active`, `contract_passed` | Does not run Step 1 or any block |
| `prepare` | `--task-config`, `--stage0-h5ad`, `--output-dir`, optional subset selectors | `task_a_stride_mapping.json`, `task_a_core_fit_dry_run.csv`, `task_a_prepare_manifest.json` | `scaffold_active`, `contract_passed` | Does not emit Block 0 calibration evidence |
| `descriptive_atlas` | `--task-config`, `--stage0-h5ad`, `--output-dir`, optional `--patient-id`, optional `--max-overlay-communities` | `task_a_descriptive_atlas_manifest.json`, `task_a_descriptive_atlas_output_index.csv`, atlas tables/figures | descriptive only | Does not run Block 0/1 or emit confirmatory claims |
| `block0 execute` | `--task-config`, `--stage0-h5ad`, `--output-dir`, `--n-permutations`, `--master-seed`, optional subset selectors, optional parallel controls | `block0_execution_manifest.json`, `block0_fit_cache.npz`, `block0_fit_cache_index.csv`, `block0_execution_progress.jsonl` | calibration/readiness fields only | Writes reusable fit cache only; does not derive metrics or p-values |
| `block0 analyze` | `--fit-cache`, `--fit-cache-index`, `--output-dir`, optional `--execution-manifest` | `block0_calibration_manifest.json`, `block0_patient_calibration.csv`, `block0_metric_summary.csv` | calibration/readiness fields only | Reads an existing cache; derives fixed family-summary calibration tables without rerunning STRIDE or permutations |
| `block1 execute` / `block1 analyze` | `execute: --task-config, --stage0-h5ad, --output-dir, optional --patient-id, optional --device`; `analyze: --execute-manifest, --output-dir` | family-level `stride.tl.fit` native result exports, `block1_execute_manifest.json`, `block1_analysis_manifest.json`, family/source/target summaries, direct `TC-IM` versus `TC-PT` comparison CSVs, and `block1_cohort_relation_comparison.csv` | `diagnostic` for subset routing, `evidence_ready` for full-cohort ready analysis routing | Reads Stage 0 plus config directly through the Task A pair AnnData adapter during execute, then execute/native exports only during analyze; does not read Block 0, descriptive atlas, result packets, or proxy history |
| `python -m tasks.task_A.block3` | semantic experiment name, `--task-config`, `--stage0-h5ad`, `--output-dir`, optional `--device`, optional engineering-smoke `--max-reruns` and `--n-test` | internal Block 3 raw/review artifacts for one subexperiment | `scaffold_active`; subset runs mark `execution_scope=subset_engineering_test` | Does not expose a public workflow or packet bridge |
| `package_results` | canonical atlas manifest, prepare manifest, Block 0 calibration manifest, output root; optional Block 1 packet input | `task_a_result_packet_manifest.json`, `task_a_result_packet_index.csv`, `RESULTS_INDEX.md`, layer manifests/review indexes, and mirrored atlas/Block 0 calibration files | packet-local only | Does not interpret biology, re-run large experiments, fabricate missing result surfaces, copy raw Block 0 fit cache by default, or package Block 3 into the Step 3 canonical packet |
| `write_semisynthetic_artifacts` | output root, manifest filename, patient count, seed | semisynthetic manifest CSV, `task_a_semisynthetic_contract.json` | `contract_passed` | Does not run the real-data graph |
| `stride_adapter` helpers | Stage 0 h5ad/AnnData, config bundle | mapping summaries, family-sliced observations, pp-ready pair AnnData, dry-run rows | helper only | Does not touch `src/stride/` semantics |

## Layout

| Directory | Purpose |
|---|---|
| `config/` | Typed config loading and frozen Task A config validation. |
| `contracts/` | Machine-readable freeze registry, execution graph, artifact contracts, mapping contract, and deferred-boundary notes. |
| `workflows/` | Step 1, Block 1, packet, and task-local STRIDE adapter surfaces. Block 0 lives in `block0/`. |
| `block0/` | Block 0 calibration execute/analyze implementation, cache schemas, and cache-derived analysis writers. |
| `block1/` | Block 1 real-data discovery summary and contrast helpers; execute/analyze entrypoints are deferred. |
| `block3/` | Internal Block 3 rebuild package for generator validation, baseline comparison, ablation execution, and raw/review writers. |
| `stage0/` | Stage 0 extraction, building, and validation helpers. |
| `real_data/` | Frozen demo-subset definitions for cheap subset wiring checks. |
| `semisynthetic/` | Deterministic benchmark world builders. |
| `benchmarks/` | Semisynthetic export helpers. |

## Deferred surfaces

- Task A Block 0/1 real fitting migrates through the Task A adapter to
  canonical `stride.tl.fit(...)`.
- Standalone bridge-expert APIs remain deferred.
- New public STRIDE core APIs require a separate contract update.
- Block 3 reference/ablation migration remains deferred.

## Examples

Full-cohort Step 1:

```bash
PYTHONPATH=.:src python -m tasks.task_A.workflows.prepare \
  --task-config tasks/task_A/config.yaml \
  --stage0-h5ad /mnt/NAS_21T/ProjectData/STRIDE/task_A_stage0_k10/task_A_stage0_k10.h5ad \
  --output-dir /mnt/NAS_21T/ProjectResult/STRIDE/task_A/prepare
```

Descriptive atlas from Stage 0:

```bash
PYTHONPATH=.:src python -m tasks.task_A.descriptive \
  --task-config tasks/task_A/config.yaml \
  --stage0-h5ad /mnt/NAS_21T/ProjectData/STRIDE/task_A_stage0_k10/task_A_stage0_k10.h5ad \
  --output-dir /mnt/NAS_21T/ProjectResult/STRIDE/task_A/descriptive
```

Block 0 execution-cache diagnostic subset:

```bash
PYTHONPATH=.:src python -m tasks.task_A.block0 execute \
  --task-config tasks/task_A/config.yaml \
  --stage0-h5ad /mnt/NAS_21T/ProjectData/STRIDE/task_A_stage0_k10/task_A_stage0_k10.h5ad \
  --n-permutations 9 \
  --master-seed 20260503 \
  --patient-id B10 \
  --patient-id B12 \
  --patient-id B3 \
  --patient-id W18 \
  --output-dir /tmp/task_a_block0_smoke
```

Block 0 full execution cache:

```bash
PYTHONPATH=.:src python -m tasks.task_A.block0 execute \
  --task-config tasks/task_A/config.yaml \
  --stage0-h5ad /mnt/NAS_21T/ProjectData/STRIDE/task_A_stage0_k10/task_A_stage0_k10.h5ad \
  --n-permutations 199 \
  --master-seed 20260503 \
  --parallel-permutations 8 \
  --worker-cpu-threads 4 \
  --device cuda \
  --output-dir /mnt/NAS_21T/ProjectResult/STRIDE/task_A/block0
```

Block 0 cache-derived analysis:

```bash
PYTHONPATH=.:src python -m tasks.task_A.block0 analyze \
  --fit-cache /mnt/NAS_21T/ProjectResult/STRIDE/task_A/block0/block0_fit_cache.npz \
  --fit-cache-index /mnt/NAS_21T/ProjectResult/STRIDE/task_A/block0/block0_fit_cache_index.csv \
  --execution-manifest /mnt/NAS_21T/ProjectResult/STRIDE/task_A/block0/block0_execution_manifest.json \
  --output-dir /mnt/NAS_21T/ProjectResult/STRIDE/task_A/block0
```

Block 1 real-data discovery:

```bash
PYTHONPATH=.:src python -m tasks.task_A.block1 execute \
  --task-config tasks/task_A/config.yaml \
  --stage0-h5ad /mnt/NAS_21T/ProjectData/STRIDE/task_A_stage0_k10/task_A_stage0_k10.h5ad \
  --confirm-full-cohort \
  --device cuda \
  --output-dir /mnt/NAS_21T/ProjectResult/STRIDE/task_A/block1
```

```bash
PYTHONPATH=.:src python -m tasks.task_A.block1 analyze \
  --execute-manifest /mnt/NAS_21T/ProjectResult/STRIDE/task_A/block1/block1_execute_manifest.json \
  --output-dir /mnt/NAS_21T/ProjectResult/STRIDE/task_A/block1
```

Objective Step 3 result packet from the canonical atlas plus Block 0/1
surfaces, with Block 3 explicitly deferred:

```bash
PYTHONPATH=.:src python -m tasks.task_A.workflows.package_results \
  --atlas-manifest /mnt/NAS_21T/ProjectResult/STRIDE/task_A/descriptive/task_a_descriptive_atlas_manifest.json \
  --prepare-manifest /mnt/NAS_21T/ProjectResult/STRIDE/task_A/prepare/task_a_prepare_manifest.json \
  --block0-calibration-manifest /mnt/NAS_21T/ProjectResult/STRIDE/task_A/block0/block0_calibration_manifest.json \
  --output-dir /mnt/NAS_21T/ProjectResult/STRIDE/task_A/result_packet
```

Packet notes:
- `task_a_result_packet_manifest.json` now declares `included_layers=["atlas", "block0", "block1"]`.
- `task_a_result_packet_manifest.json` now declares `deferred_layers=["block3"]`.
- `--prepare-manifest` is an explicit packet input; the descriptive atlas
  manifest no longer carries Step 1 prepare or mapping pointers.
- `package_results` has no Block 3 manifest bridge parameter.
- `task_a_result_packet_index.csv` keeps lineage columns for prepare/Block rows; descriptive atlas rows remain descriptive-only.
