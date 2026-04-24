# Task A Rewiring Plan

This document freezes how Task A should be rewired onto full STRIDE after the
Step 1 documentation reset. It is documentation only. It does not change live
execution logic, rewrite pipelines, or rerun experiments.

For the canonical full method definition, use
[`docs/stride_design_freeze.md`](/home/lenislin/Experiment/projects/STRIDE/docs/stride_design_freeze.md).

## 1. Boundary

Task A is currently a bounded proxy/approximate execution stack built on:

- a single-timepoint ordered tissue-domain proxy,
- the narrow live patient bridge,
- deferred cohort recurrence,
- task-local Block 0-3 summary layers.

That current stack is preserved as execution history. It is not the final
full-STRIDE evidence layer.

Previously generated Task A outputs, including the live Block 3 packet under
`tasks/task_A/result_packets/2026-04-05_block3_live_exec_01/`, must therefore
be preserved as approximate/proxy historical results rather than silently
treated as final full-STRIDE evidence.

## 2. Frozen Rewiring Targets

| Layer | Step 1 frozen decision |
|---|---|
| Descriptive atlas | Mostly reusable as the biological context layer. It remains descriptive and should be regenerated only as needed to stay aligned with the rerun full-STRIDE shared axis and provenance. |
| Block 0 | Must be rerun on summaries derived from the rewired full-STRIDE surface rather than on the current proxy-only evidence layer. |
| Block 1 | Must be rerun on full STRIDE patient plus cohort outputs, not only on the current proxy-stack summary exports. |
| Block 2 | Must be rerun as robustness over the full STRIDE evidence stack, including the rewired patient and cohort objects that support the scientific claim. |
| Block 3 | Must be rebuilt as a semisynthetic truth-recovery benchmark against full STRIDE, with `3A generator validation`, `3B baseline comparison`, `3C ablation study`, shared rerun-specific patient-level semi-synthetic realizations, `3A` frozen to held-out cohort `community-space` realism plus `g_k`-projected identity-aware biological plausibility and rerun stability, `3C-2 shared` interpreted as shared train-derived cohort-level generation across held-out patients within a rerun, `P(m)` treated as train-side generator calibration only, no dedicated null/random comparator design, and `3C-1`/`3C-2` defined as loss/regularization ablations rather than proxy-style comparator rewrites. |
| Historical outputs | Current packets and memos remain preserved as proxy history and implementation context. They must not be relabeled as final full-STRIDE evidence. |

## 3. Four-Step Migration Plan

| Step | Objective | Expected artifacts | In scope | Out of scope | Completion |
|---|---|---|---|---|---|
| 1. Documentation reset and design freeze | Freeze the intended full method, live-approx boundary, Task A rewiring, and source ordering. | `docs/stride_design_freeze.md`, `docs/task_A_rewiring_plan.md`, updated status/index/Task A boundary docs. | Documentation, specification, canon/proxy/history separation. | Code changes, reruns, result reinterpretation. | The repo has one unambiguous full-STRIDE canon and one unambiguous live-approx description. |
| 2. STRIDE script/workflow refactor | Rewire the live implementation so Task A can consume and emit the full-STRIDE patient and cohort objects. | Refactored script/workflow/core-task interfaces and aligned docs/contracts. | Script/workflow refactor, implementation alignment to the Step 1 canon. | Block 0-3 reruns and result relabeling. | The live stack can operate on the intended full STRIDE objects rather than the current proxy-only surface. |
| 3. Block 0-2 rerun and result reorganization | Regenerate the descriptive atlas and Block 0-2 evidence on the full-STRIDE surface and separate it from proxy history. | New full-STRIDE result packets, updated review packets, preserved old proxy packets. | Reruns, packet reorganization, provenance boundary updates. | Block 3 rebuild. | Full-STRIDE Block 0-2 evidence exists and old packets are clearly historical. |
| 4. Block 3 rebuild and rerun | Rebuild Block 3 around the full method as a semisynthetic truth-recovery benchmark with hidden `(A,d,e)` generators, paired observed baseline/target inputs, `3A generator validation`, `3B baseline comparison`, `3C ablation study`, rerun-specific patient-level semi-synthetic realizations reused across benchmark and ablation, `3A` limited to cohort-level realism / identity-aware plausibility / rerun stability on held-out cohort surfaces, and explicit non-estimable metric semantics. | New Block 3 manifests, full objective export surfaces, updated review packet. | Comparator/ablation/semi-synthetic execution against full STRIDE. | Reopening the Step 1 method definition. | Block 3 validates the full method rather than the proxy stack. |

The detailed rebuilt Block 3 design freeze now lives in
[`docs/task_A_spec.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A_spec.md),
which defines the frozen Step 4 scientific role, Task A Block 3 sections,
method registry, rerun-specific semi-synthetic generator semantics, metric hierarchy, and
execution phases.
The design-alignment note for the current transitional hardening lives in
[`docs/task_A_block3_redesign_v1_1.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A_block3_redesign_v1_1.md).

The current repository's proxy Block 3 ablation implementation remains
historical/proxy execution context only. It must not be treated as the live
scientific `3C` contract, which is now defined as loss/regularization
ablation on shared rerun-specific patient-level semi-synthetic realizations.

## 4. Task A Rewiring Interpretation Rules

- The descriptive atlas is context, not proof.
- Block 0 remains a gate and must stay downstream of the descriptive atlas.
- Block 1 remains the main real-data biological discovery layer, but after the
  refactor it must be grounded in full-STRIDE patient and cohort outputs.
- Block 2 remains robustness, but after the refactor it must validate the full
  evidence stack rather than only the current proxy summary surface.
- Block 3 remains method validation, but it must be rebuilt against full
  STRIDE, with `3A` explicitly limited to held-out cohort `community-space`
  realism, `g_k`-projected identity-aware biological plausibility, and rerun
  stability on those same objects. It must treat `P(m)` as train-side
  generator calibration rather than a public benchmark axis, must interpret
  `3C-2 shared` as shared train-derived cohort-level generation rather than a
  planted latent template, must not introduce a dedicated null/random
  comparator design, must not rewrite `3A` into an annotation benchmark, and
  must not inherit final authority from the current proxy packet.

## 5. Source Boundary For Future Work

Implementers should read in this order:

1. `docs/stride_design_freeze.md`
2. `docs/task_A_rewiring_plan.md`
3. `docs/state.md`
4. `docs/task_A_spec.md`
5. `tasks/task_A/README.md`
6. `docs/task_A_result.md` as the canonical Task A result memo through Block 2,
   with preserved proxy-history context
7. `history/docs/` and `tasks/task_A/result_packets/` only for historical
   context

## 6. Minimal Canonical Document Set

After this freeze, the minimal canonical document set for future STRIDE and
Task A work is:

- `docs/stride_design_freeze.md`
- `docs/task_A_rewiring_plan.md`
- `docs/state.md`
- `docs/task_A_spec.md`
- `docs/task_A_result.md`
- `tasks/task_A/README.md`
