# Task A Result Context Through Block 1 Statistical Supplement

This document records the Task A result layer supported by the checksummed
K=10 Stage 0, Block 0, and Block 1 artifacts. It does not redefine STRIDE or
the Task A scientific contract. The governing authorities remain
[`docs/stride_design_freeze.md`](/home/lenislin/Experiment/projects/STRIDE/docs/stride_design_freeze.md)
and [`docs/task_A/spec.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A/spec.md).

The result surfaces used here are:

- `/mnt/NAS_21T/ProjectData/STRIDE/task_A_stage0_k10/task_A_stage0_k10.h5ad`
- `/mnt/NAS_21T/ProjectResult/STRIDE/task_A/block0`
- `/mnt/NAS_21T/ProjectResult/STRIDE/task_A/block1`

A read-only compatibility reconstruction is stored at
`/mnt/NAS_21T/ProjectResult/STRIDE/task_A/block1_compatibility_review_20260712`.
It is verification-only and does not replace the Block 1 source artifacts.

## 1. Evidence Status And Provenance

Task A remains a bounded proxy-validation task. Its ordered surface
`TC -> IM -> PT` is a single-timepoint tissue-domain proxy, not literal time,
lineage, or a causal transition process.

The K=10 Stage 0 artifact passes the current Task A pair adapter and
`stride.pp.validate_ready` for `TC-IM`, `TC-PT`, and `IM-PT`. The state basis,
cost matrix, cost scale, and patient/FOV observations are therefore reusable
for the current analysis boundary.

The May 2026 Block 0 run is retained as the formal calibration result. It
contains 32 patients, one observed fit and 199 patient-preserving null fits per
patient, 6,400 cache records, and `readiness_status="calibration_ready"`.
Its cache and index SHA-256 values match the execution manifest. A June 2026
`stride.tl.fit` diagnostic run showed that the observed-fit arrays were
numerically stable across the API migration: maximum absolute differences were
`4.11e-05` for `A`, `1.37e-04` for `d`, and `7.44e-06` for `e`; source and
target composition vectors were identical. The Block 0 provenance is therefore
classified as `formal pre-PR17 Block 0 result, numerically bridged to
stride.tl.fit`. No new 199-permutation run was required.

The May 2026 Block 1 native exports remain the analysis source. Both family
manifests and their patient/cohort indexes and arrays pass their recorded
SHA-256 checks. The current Block 1 analysis code read those legacy native
arrays without fitting and regenerated all 11 summary, comparison, and
statistical-supplement CSV files byte-for-byte identically. Patient axes,
K=10 state axes, fit statuses, cohort support, directions, and review-candidate
sets were unchanged. Block 1 therefore does not require refitting.

## 2. Descriptive K=10 Context

The descriptive atlas is context rather than inferential evidence. On the K=10
axis, communities `0`, `2`, `4`, and `8` are predominantly TC-associated;
communities `1`, `3`, and `9` are predominantly PT-associated; and communities
`5`, `6`, and `7` have broader IM or mixed-domain representation. These labels
summarize domain distribution only and do not establish biological identity,
lineage, or direction of change.

## 3. Block 0 Patient-Preserving Exchangeability Result

Block 0 tests whether the observed `TC-IM` relation surface differs from a
within-patient TC/IM FOV-label permutation reference while preserving patient
identity, FOV compositions, FOV counts, and each patient's TC/IM count
structure.

Burden-weighted `self_retention` was lower in the observed data than in the
null reference: median real value `0.4833`, null reference `0.9646`, median
delta `-0.4813`, and empirical p-value `0.005`. On the community-mean scale,
the median delta was `-0.2894`, also with empirical p-value `0.005`.

Burden-weighted `depletion` was higher in the observed data: median real value
`0.0460`, null reference `0.0031`, median delta `+0.0429`, and empirical
p-value `0.005`. The community-mean median delta was `+0.0589`, with empirical
p-value `0.005`.

`off_diagonal_remodeling` was higher in the observed data but remains a
diagnostic-supportive endpoint. On the burden-weighted scale, the median real
value was `0.4271`, the null reference was `0.0322`, the median delta was
`+0.3948`, and the empirical p-value was `0.005`.

`emergence` did not show a stable positive departure from the null. The
empirical p-values were `0.945` and `0.965` for the burden-weighted median and
mean, and `0.970` and `0.975` for the community-mean median and mean.

These results support a departure of the observed `TC-IM` relation surface from
the patient-preserving label-permutation reference, concentrated in source-side
allocation. They do not identify a biological mechanism and do not test the
`TC-IM` versus `TC-PT` paired contrast.

## 4. Block 1 Real-Data Paired Contrast

Block 1 compares the prespecified `TC-IM` and `TC-PT` families in the same 32
patients. The statistical supplement uses paired patient-level
`TC-IM - TC-PT` deltas, two-sided Wilcoxon signed-rank tests, two-sided sign
tests, BH adjustment within declared surfaces, and an absolute median-delta
effect floor of `0.05`.

At the family level, `self_retention` was higher for `TC-IM`. The
burden-weighted median delta was `+0.3375`, with `32/32` patients positive and
BH q-value `6.21e-10`; the community-mean median delta was `+0.2138`, also with
`32/32` positive and BH q-value `6.21e-10`.

`off_diagonal_remodeling` was lower for `TC-IM`. The burden-weighted median
delta was `-0.3152`, with `32/32` patients negative and BH q-value `6.21e-10`;
the community-mean median delta was `-0.2042`, also with `32/32` negative and
BH q-value `6.21e-10`.

Family-level `depletion` was lower for `TC-IM`, but its median absolute effects
(`0.0084` burden-weighted and `0.0103` community-mean) did not pass the `0.05`
effect floor and were not review candidates. Family-level `emergence` was lower
for `TC-IM`; the burden-weighted median delta was `-0.0535`, with `32/32`
patients negative and BH q-value `6.21e-10`. This remains supportive target-open
context rather than a Block 0-calibrated primary claim.

At the source-community level, review-candidate self-retention deltas were
positive for K=10 communities `0`, `2`, `4`, `5`, `6`, and `8`. The largest
were community `8` (`+0.4879`, `31/31` positive), community `4` (`+0.4269`,
`31/31` positive), and community `0` (`+0.3187`, `32/32` positive). Their
corresponding off-diagonal deltas were negative. This is consistent with
stronger within-community retention for `TC-IM` on several TC-associated or
TC/IM-associated states.

At the target-community level, community `1`, which is predominantly
PT-associated in the descriptive atlas, had lower matched incoming burden for
`TC-IM` than `TC-PT` (median delta `-0.3253`, `32/32` negative, BH q-value
`2.10e-09`). Its open incoming burden/tendency also had median delta `-0.1295`
with `32/32` negative. Community `3` also had lower matched incoming burden
(`-0.0561`, `32/32` negative), whereas communities `6` and `8` had positive
matched incoming-burden deltas.

The cohort relation comparison is an effect map rather than a cohort-level
hypothesis test. Patient-level relation-element statistics support its largest
edges. Examples include `A 8->8` with median delta `+0.4878` and cohort delta
`+0.4896`, `A 8->1` with median delta `-0.5731` and cohort delta `-0.5737`,
`A 4->4` with median delta `+0.4269`, and `A 0->1` with median delta `-0.3548`;
each direction was shared by all 32 patients with estimable support.

The Block 1 evidence supports a proxy-scoped contrast in which `TC-IM` has
stronger within-state retention and lower off-diagonal allocation than `TC-PT`,
with substantial `TC-PT`-associated allocation toward PT-associated community
`1`. This does not establish literal temporal transition or causal remodeling.

## 5. Integrated Evidence Status

Block 0 shows that observed `TC-IM` structure departs from the patient-preserving
label-permutation reference through lower self-retention, higher depletion, and
higher off-diagonal remodeling, without a stable positive emergence departure.
Block 1 separately shows that `TC-IM` has higher self-retention and lower
off-diagonal allocation than `TC-PT` in the paired real-data comparison.

The two blocks answer different questions and should not be collapsed into one
causal statement. The combined evidence is limited to relation structure on the
ordered tissue-domain proxy and does not establish lineage, literal time,
mechanism, or method superiority.

## 6. Block 3 Boundary

Historical Block 3 outputs were removed from the active Task A result root on
2026-07-12. There is currently no formal Block 3 result available for analysis
or plotting. A future full rerun must start from generator construction and
`generator_validation`, then execute `a_benchmark`, `de_benchmark`, and the
three true refit ablations under the frozen Block 3 contract. Objective-level
private refit implementations now exist, but no Block 3 result becomes
available until the new formal run completes. A generator-only or partially
resumed run is not ablation evidence.

This memo does not claim literal time ordering, direct temporal transition,
lineage tracing, direct proof that a state disappears or emerges in vivo,
confirmatory status for `IM-PT`, method superiority over baselines, or Block 3
validation of Block 1 biology.
