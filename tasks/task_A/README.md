# Task A Mainline

This README is the authoritative Task A engineering guide for the rewritten
mainline. It documents the active data path, the directory layout, runner
responsibilities, and acceptance gates. Scientific claims stay in
`docs/task_A_spec.md`.

## 1. Directory Layout

| Directory        | Purpose |
|------------------|---------|
| `config/`        | Typed config loading (`model.py`). Parses `config.yaml` into frozen dataclasses and rejects legacy config keys. |
| `contracts/`     | Stage 0 mapping contract (`stride_mapping.py`), deferred boundary notes, and the crosswalk specification. |
| `workflows/`     | Workflow entrypoints: `prepare.py`, `stride_adapter.py`, `run_block0.py`, `run_block1.py`, `run_block2.py`. |
| `block0/`        | Block 0 (locality gate) bundle writer. Produces intra-compartment same-patient pair schedules. |
| `block1/`        | Block 1 (continuity backbone) bundle writer. Produces ordered cross-compartment pair schedules. |
| `block2/`        | Block 2 (bounded audit) bundle writer. Reads a Block 1 bundle and produces audit-pair schedules. |
| `stage0/`        | Stage 0 artifact construction (`build_artifacts.py`) and the R extraction script for CRLM col-data. |
| `real_data/`     | Real-data demo subset definitions (e.g. `alignment_v1`). Used for cheap iteration dry-runs. |
| `semisynthetic/` | Semi-synthetic benchmark data generators for controlled validation. |
| `benchmarks/`    | Benchmark harness for timing and correctness checks against semi-synthetic data. |
| `plans/`         | Internal planning documents (execution checklist, implementation plan). Not part of the runtime surface. |

## 2. Runner Responsibility Table

| Runner | Status | What it does | Inputs | Outputs |
|--------|--------|-------------|--------|---------|
| `workflows/prepare.py` | Active | Maps Stage 0 artifact to STRIDE, runs confirmatory core-fit dry-run, writes manifests. | `--task-config`, `--stage0-h5ad`, `--output-dir`, optional `--patient-id` or `--demo-subset` | `task_a_stride_mapping.json`, `task_a_prepare_manifest.json`, `task_a_core_fit_dry_run.csv` |
| `workflows/run_block0.py` | Active | Runs Block 0 locality gate through the task-local workflow wrapper. | `--task-config`, `--stage0-h5ad`, `--output-dir` | Block 0 bundle (JSON manifest + pair schedule) |
| `workflows/run_block1.py` | Active | Runs Block 1 continuity backbone through the task-local workflow wrapper. | `--task-config`, `--stage0-h5ad`, `--output-dir` | Block 1 bundle (JSON manifest + pair schedule) |
| `workflows/run_block2.py` | Active | Runs Block 2 bounded audit over an existing Block 1 bundle. | `--block1-bundle`, `--output-dir` | Block 2 bundle (JSON manifest + audit-pair schedule) |
| `workflows/run_real_data.py` | Planned | End-to-end real-data pipeline: prepare + block sequence. | TBD | TBD |
| `workflows/summarize.py` | Planned | Post-run summary and evidence-surface collation. | TBD | TBD |

## 3. Step 1 Sign-Off (Real-Data Interface Alignment)

Step 1 is closed only when the Stage 0 artifact can attach to stable `stride`
through the task-local adapter without hidden compatibility wrappers.

### Acceptance criteria

- The Stage 0 crosswalk is frozen under `contracts/`.
- `prepare.py` is the sole accepted entrypoint for checking real-data alignment.
- Ordered-group semantics are derived from tissue domain labels, not the inert
  raw `timepoint` column.
- ROI/FOV observations attach to stable `stride` with uniform mass semantics
  (`mass_mode: "density"` is the single source of truth in `config.yaml`).
- All three prepare outputs (`task_a_stride_mapping.json`,
  `task_a_prepare_manifest.json`, `task_a_core_fit_dry_run.csv`) are produced
  without error for both full-data and demo-subset invocations.

### Acceptance commands

```bash
PYTHONPATH=.:src python -m tasks.task_A.workflows.prepare \
  --task-config tasks/task_A/config.yaml \
  --stage0-h5ad /mnt/NAS_21T/ProjectData/STRIDE/task_A_stage0/task_A_stage0_k25.h5ad \
  --output-dir /tmp/task_a_prepare

PYTHONPATH=.:src python -m tasks.task_A.workflows.prepare \
  --task-config tasks/task_A/config.yaml \
  --stage0-h5ad /mnt/NAS_21T/ProjectData/STRIDE/task_A_stage0/task_A_stage0_k25.h5ad \
  --output-dir /tmp/task_a_prepare_demo \
  --demo-subset alignment_v1
```

### Real-data subset

- `alignment_v1` is the frozen demo subset under `real_data/`.
- It exists only to make real-data dry-runs cheap enough for iteration.
- It is representative, but it is not a synthetic or toy validation surface.

## 4. Step 2 Sign-Off (Design Freeze)

Step 2 freezes responsibility boundaries before block implementation expands.

### Acceptance criteria

- Stage 0 / Block 0 / Block 1 / Block 2 boundaries are documented and do not
  overlap.
- Each runner's inputs, outputs, manifests, and honest-failure behavior are
  frozen.
- Surfaces that depend on stable `stride` are identified separately from
  task-local surfaces.
- Scientific interpretation is kept outside the implementation path.

Step 2 does not mean "run more experiments." It means the Task A execution
graph is stable enough that Block 0, Block 1, and Block 2 can be implemented
and tested one by one without re-opening the data contract.

## 5. Deferred Boundaries

The following items are explicitly deferred from the current implementation
pass. Runners that depend on deferred surfaces must fail honestly rather than
silently degrade.

- **Block 0**: May fail honestly until stable `stride` exposes the locality
  support it needs.
- **Block 1**: Continuity summaries are task-local evidence surfaces, not
  canonical recurrence outputs.
- **Block 2**: Remains a bounded audit and must fail honestly if it needs
  unsupported assignment surfaces from stable `stride`.
- **run_real_data.py**: Not yet implemented. Will orchestrate the full
  prepare + block sequence once all blocks pass their individual gates.
- **summarize.py**: Not yet implemented. Will collate evidence surfaces
  post-run once the output contract is frozen.
- Canonical core recurrence estimation remains deferred.
- Standalone bridge-expert APIs remain deferred.
- Task A does not request new public Block 0 locality or same-pair comparator
  APIs from `src/stride/` in this pass.
