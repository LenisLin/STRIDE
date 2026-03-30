# Task A Arm-3 Historical Summary

> Historical note: preserved Arm-3 planning/implementation artifact. Live methodology: `docs/task_A_spec.md`. Live API: `docs/api_specs.md`. Live current-output contracts: `docs/data_contracts.md`.

This file consolidates the prior Arm-3 skeleton, implementation, patch, and coverage-ladder notes into one archival summary. The exact frozen artifact path and schema details remain in `task_A_arm3_local_manifest.md`.

## Source notes consolidated here
- `task_A_arm3_script_skeleton_plan.md`
- `task_A_arm3_skeleton_generation_note.md`
- `task_A_arm3_phase0_3_impl_note.md`
- `task_A_arm3_phase4_impl_note.md`
- `task_A_arm3_phase5_6_impl_note.md`
- `task_A_arm3_phase7_impl_note.md`
- `task_A_arm3_phase7_8_patch_note.md`
- `task_A_arm3_coverage_ladder_update_note.md`

## Historical timeline
### 1. Skeleton and tranche planning
- Arm-3 was originally scoped as a task-layer reduced-coverage robustness workflow on the frozen Stage-0 artifact, not a library-level feature and not an Arm-IV extension.
- The planned module split was:
  - `arm3/block_partition.py` for frozen grid partition and block summaries
  - `arm3/pseudo_roi.py` for pseudo-ROI bootstrap sampling
  - `arm3/calibrate.py` for family-level `lambda_dens` and compartment-level `tau` freezing
  - `arm3/inference.py` for tensor assembly, frozen support masks, broadcast calibration, and Arm-3 density metrics
  - `arm3/retention.py` for degradation and floor-dominated summaries
  - `arm3/output.py` for parquet/csv/json/memo output
  - `arm3_uq_stress.py` as the task-layer runner
- The historical planning boundary explicitly left `src/slotar/uq.py::bootstrap_single_roi` out of scope and kept Arm-3 bootstrap logic in `tasks/task_A`.

### 2. Skeleton generation
- The first implementation pass generated the Arm-3 module skeletons and locked the early public function layout for block partition, pseudo-ROI bootstrap, calibration, inference, retention, output, and runner entrypoints.
- That tranche established the archival intent for:
  - frozen feature reuse from Stage 0
  - density-primary Arm-3 semantics
  - exact per-phase output file naming under the task-layer result root

### 3. Phase 0-3 landing
- Phase 0 validated the frozen `K=25` contract, resolved Arm-3 block-size settings, and wrote a manifest of run parameters.
- Phase 1 implemented direct Stage-0 HDF5 reads, frozen grid partitioning, and zero-cell-preserving ROI block summaries.
- Phase 2 built the full-coverage density reference by dividing block-summed counts by total geometric ROI block area.
- Phase 3 generated the full ordered pair universe from the frozen ROI metadata and retained `TC->IM` and `TC->PT` as the Arm-3 anchor directions.

### 4. Phase 4 landing
- Phase 4 fixed the grid-dimension integrity check so block-universe validation no longer assumed unsafe 0-based indexing.
- `lambda_dens` calibration was implemented jointly by unordered family across both ordered directions in that family.
- `tau_by_compartment` calibration was implemented on the frozen full-coverage reference surface.
- The historical Phase-4 note preserved an important early boundary: later runner integration, bootstrap, and output finalization were still pending at that point.

### 5. Phase 5-6 landing
- Phase 5 locked the pseudo-ROI bootstrap contract:
  - sample `max(1, floor(target_coverage * n_total_blocks))` blocks with replacement
  - keep zero-cell blocks eligible
  - derive pseudo-ROI density from total sampled counts divided by total sampled block area
- Phase 6 locked the frozen-inference contract:
  - support masks are computed once from the full-coverage reference and never recomputed on pseudo-ROI inputs
  - `tau` is broadcast by `compartment_a`
  - `lambda_dens` is broadcast by unordered pair family
  - Arm-3-specific density summaries include `U_abs_dens`, `Q_src_dens`, `Q_tgt_dens`, `S_src`, `S_tgt`, `Delta_scale`, and `scale_ratio`
- The forbidden transportability ratio `T / (T + B_pos + D_pos + eps)` remained intentionally absent from the Arm-3 output surface.

### 6. Phase 7 and Phase 8 patching
- Phase 7 introduced degradation and contrast summaries without adding scientific pass/fail verdicts.
- The historical zero-sign tie rule was locked as:
  - full-coverage zero sign means non-evaluable
  - reduced-coverage zero sign counts as failure when the full-coverage sign is non-zero
- The later Phase 7/8 patch note added:
  - exact prototype-event extraction from solver details
  - contrast-based sign-consistency logic
  - Phase-8 prototype `Delta_U_k` preparation and stability tables
- These notes preserved the boundary that Arm-3 output finalization remained descriptive and did not itself authorize scientific closure.

### 7. Coverage-ladder cleanup
- The active reduced-coverage ladder was later simplified to `75% / 50% / 25%`.
- `100%` remained the frozen full-coverage reference baseline outside the bootstrap loop.
- Historical `10%` wording was retained only where it referred to the floor-dominated threshold rule, not as an active reduced-coverage level.

## Historical facts still worth preserving
- Arm-3 was always intended as a bounded robustness continuation of Arm-2 on the frozen Stage-0 representation.
- The frozen artifact path/schema audit remains in `task_A_arm3_local_manifest.md`.
- The live Task-A methodology and boundaries moved out of the historical notes and now belong to the canonical docs:
  - `docs/task_A_spec.md`
  - `docs/task_A_results.md`
  - `docs/data_contracts.md`

## Cleanup result
- This summary supersedes the prior per-phase and per-patch Arm-3 history notes listed above.
- The old note files are removed to keep `docs/history/task_A_arm3/` focused on:
  - one archival summary
  - one frozen-artifact manifest
  - one README that points back to live docs
