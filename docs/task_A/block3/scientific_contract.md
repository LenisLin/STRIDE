# Task A Block 3 Scientific Contract

This file is the live scientific contract for Task A Block 3. It is subordinate
to `docs/task_A/spec.md` and defines the Stage0-only, cost-only semi-synthetic
benchmark surface used by the internal Task A validation route.

## Authority Boundary

- Full STRIDE scientific authority remains in `docs/stride_design_freeze.md`.
- Task A Block 3 is a bounded validation surface, not a redefinition of STRIDE.
- Block 3 execution hard inputs are Stage 0 h5ad and Task A config only.
- The active execution surface is Stage0-only. Block 1 manifest-driven
  execution is retired from the live Block3 contract.
- Block 3 derives community identity vectors `g_k` internally from the Stage 0
  shared-state/cell-subtype surface. The descriptive atlas may explain
  community meaning but is not a hard input.
- Block 3 shares Stage0, TC-IM, the K-state surface, and identity-derived
  geometry with Task A, but it does not validate Block 1 biological findings.

## Experiment Structure

- `3A = generator validation`
- `3B-1 = A benchmark`
- `3B-2 = d/e benchmark`
- `3C-1 = subbag consistency ablation`
- `3C-2 = geometry ablation`
- `3C-3 = recurrence ablation`

The semantic CLI and artifact roots are `generator_validation`, `a_benchmark`,
`de_benchmark`, `subbag_consistency_ablation`, `geometry_ablation`, and
`recurrence_ablation`. The data column `subexperiment_id` continues to carry
`3A`, `3B-1`, `3B-2`, `3C-1`, `3C-2`, and `3C-3`.

Formal Block 3 execution uses `10` reruns and `8` held-out test patients per
rerun. The internal CLI may expose smaller rerun or test-patient limits only
for engineering correctness smoke tests. Such subset outputs must be marked
`execution_scope=subset_engineering_test` in raw and review manifests and must
not be interpreted as formal full-data Block 3 results.

## Generator Contract

Block 3 constructs one shared semi-synthetic multi-FOV realization on the
Stage 0 `K`-state axis and reuses it across `3A`, `3B`, and `3C`. Hidden truth
is patient-level `(A_p, d_p, e_p)`. Hidden truth, sampled template identity, and
real held-out target endpoints are used only for scoring and generator audit.

The live generator uses a train-derived template bank from real train TC-IM
endpoint pairs. For each train patient, endpoint residuals are decomposed into
geometry-gated off-diagonal remodeling and open residuals using the normalized
state cost matrix with `tau=2.0`. Each train template carries row-simplex
`[A^(q) | d^(q)]` and an emergence shape `s^(q)`. Missing or unidentifiable
source rows are filled from train-row pooled templates, with diagonal
no-remodeling fallback only when no train patient identifies that row.
Held-out patient truth mixes the cohort medoid template with a sampled
individual template using `lambda_individual=0.10`; the emergence vector is
`e_p = (x_p * d_p).sum() * s_p`. Source FOVs are deterministic shrinkage
mixtures of the held-out source endpoint and the real held-out source FOVs
with `eta=0.3`. Generated target FOVs are computed from generated source-FOV
anchors through the hidden `(A_p, d_p, e_p)` with no target noise.

Endpoint-only baselines consume deterministic endpoint projections
`(\bar x_p^{obs}, \bar y_p^{obs})` of the generated multi-FOV observations.
STRIDE reference and all STRIDE ablations consume the generated source/target
FOV observations. All methods are scored against the same hidden
`(A_p, d_p, e_p)`.

Generator diagnostics include row accounting, burden ordering
(`retained diagonal > off-diagonal > open` as a diagnostic target only),
endpoint closure, expected evidence blocks, imputed-row source mass,
finite-truth checks, FOV simplex checks, geometry locality of true
off-diagonal mass, template provenance, and cohort dispersion around the
medoid template. The generator is documentation-first and does not claim that
train-derived hidden truth is the true clinical mechanism. The main generator
risks to report rather than hide are too-strong or too-weak off-diagonal truth,
row-pooling artifacts in rare source states, endpoint-closure error from
multi-FOV projection, and cohort-medoid mixing that makes ablations too easy
or too noisy.

## 3A Generator Validation

`3A` exports generator-validation raw and review surfaces for manual inspection
before running `3B/3C`; it is not a formal pass/fail gate. It reports
`object_scores`, `rerun_stability`, and `review_surface` for the held-out
community-space target surface and the identity-projected target surface.

Manual review checks whether metrics are finite, whether real and synthetic
target surfaces show no evident shape or scale anomaly, and whether rerun
variability is not visibly uncontrolled. These checks determine whether to
continue the first rapid run sequence; they do not create a thresholded
scientific conclusion.

## Benchmark Contract

`3B-1` is the `A` benchmark. It compares `stride_reference` with the transport
family baselines on the shared multi-FOV realization. `3B-2` is the open-focused
`d/e` benchmark and uses the same generated hidden truth and endpoint
projections. The split remains operational, but both sections report the
shared metric vocabulary so relation, open-channel, mass, and endpoint
behavior can be read together.

Primary mass metrics are `F_L1_total`, `g_L1_total`, `e_L1_total`,
`offdiag_mass_abs_error`, `depletion_mass_abs_error`, and
`emergence_mass_abs_error`, where `F = x[:, None] * A` and `g = x * d`.
Primary ratio metrics are `offdiag_ratio`, `depletion_capture`, and
`emergence_capture`. `endpoint_y_MAE` is a secondary endpoint-closure metric.
Relation and open-channel metrics retain `A_MAE_active`, `A_MSE_active`,
`target_recall_at_k`, `open_support_F1`, `d_MAE`, `d_MSE`, `e_MAE`, and
`e_MSE`. The `d_*` metrics are computed on the burden-scale depletion carrier
`g = x * d`; the `e_*` metrics are computed on the emergence carrier `e`.

`balanced_ot_baseline` is not a `3B-2` method because its closed marginal
constraints imply zero derived depletion and emergence under the shared
analysis layer. It remains a `3B-1` closed relation comparator.

The benchmark pair family is controlled by the explicit Task A config field
`block3.benchmark_pair_family: "TC-IM"`. Current validation accepts only
`TC-IM`.

## 3C Ablation Contract

`3C` is a method-bearing refit ablation family. Each ablation compares
`stride_reference` with one ablated STRIDE objective variant on the same
generated multi-FOV realization set used by `3B`.

3C-1 display name is subbag consistency ablation; method key remains
`consistency_ablation`; core ablation_mode is consistency.

3C-2 method key is `geometry_ablation`; core ablation_mode is geometry.

3C-3 method key is `recurrence_ablation`; core ablation_mode is recurrence.

`3C-1`, `3C-2`, and `3C-3` all use the shared hidden
`(A_p, d_p, e_p)` truth and must refit `A_p`, `d_p`, and `e_p`. They must not
be implemented by masking `stride_reference` output, and they
must not be post-hoc rescoring only.
Retained objective terms are not reweighted. The fixed denominator policy
remains unchanged.
The compact native provenance must record the active `ablation_mode`, zero-term
handling, and fixed-denominator policy for ablation arms.

`consistency_ablation` removes only the subbag consistency term and retains
observation discrepancy, explicit open-channel terms, geometry/locality, cohort
recurrence/common structure, and audit/plausibility handling.

`geometry_ablation` removes only the geometry/locality term and retains
observation discrepancy, explicit open-channel terms, subbag consistency,
cohort recurrence/common structure, and audit/plausibility handling.

`recurrence_ablation` removes only the cohort recurrence/common-structure term
and retains observation discrepancy, explicit open-channel terms,
geometry/locality, subbag consistency, and audit/plausibility handling.

`consistency_ablation` remains a reported ablation arm in this migration. If
formal full-data Block 3 reproduces the trial-level null trend, interpretation
should treat it as neutral or dispensability evidence rather than tune the
generator to manufacture a consistency advantage.

## Artifact Contract

Raw and review artifact roles and paths use semantic roots only:

- `generator_validation`
- `a_benchmark`
- `de_benchmark`
- `subbag_consistency_ablation`
- `geometry_ablation`
- `recurrence_ablation`

Review surfaces are internal review carriers, not scientific authority by
themselves. Result interpretation is deferred until exported metrics are
reviewed against explicit questions. Block 3 execution inputs are Stage 0 h5ad
plus Task A config only.
