# Task A Block 3 Refactor Contract Map

This live migration map is the only Task A Block 3 live document that preserves
old-to-new path and artifact naming correspondences. It is a migration aid, not
a scientific contract.

## Document Path Map

| Previous live path | New live path |
| --- | --- |
| `docs/task_A_spec.md` | `docs/task_A/spec.md` |
| `docs/task_A_result.md` | `docs/task_A/result.md` |
| `docs/task_A_rewiring_plan.md` | Integrated into `docs/task_A/spec.md`; no standalone live rewiring document exists |
| `docs/task_A_block3_redesign_v1_1.md` | `docs/task_A/block3/scientific_contract.md` |

## Block 3 Stage Docs

| Subexperiment id | Semantic live path |
| --- | --- |
| `3A` | `docs/task_A/block3/3A/generator_validation.md` |
| `3B-1` | `docs/task_A/block3/3B/a_benchmark.md` |
| `3B-2` | `docs/task_A/block3/3B/de_benchmark.md` |
| `3C-1` | `docs/task_A/block3/3C/subbag_consistency_ablation.md` |
| `3C-2` | `docs/task_A/block3/3C/geometry_ablation.md` |
| `3C-3` | `docs/task_A/block3/3C/recurrence_ablation.md` |

## Artifact Root Map

| Previous artifact root | New semantic root |
| --- | --- |
| `3a_*` | `generator_validation_*` |
| `3b1_*` | `a_benchmark_*` |
| `3b2_*` | `de_benchmark_*` |
| `3c1_*` | `subbag_consistency_ablation_*` |
| `3c2_*` | `geometry_ablation_*` |
| `3c3_*` | `recurrence_ablation_*` |

The `subexperiment_id` field continues to carry `3A`, `3B-1`, `3B-2`,
`3C-1`, `3C-2`, and `3C-3` as stage identifiers. Semantic experiment names and
artifact roots are the primary live routing surface.

## Generator Surface Map

The live generator surface is the train-derived multi-FOV template generator
defined in
`docs/task_A/block3/scientific_contract.md`; `3B` and `3C` reuse the same
generated hidden truth and FOV observations.
