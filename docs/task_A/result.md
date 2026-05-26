# Task A Result Context Through Block 1 Statistical Supplement

This document records the current Task A result layer after the rebuilt
Block 0 patient-preserving exchangeability analysis and the refreshed
Block 1 full-cohort real-data paired contrast with statistical supplement. It
does not redefine full STRIDE, which remains frozen in
[`docs/stride_design_freeze.md`](/home/lenislin/Experiment/projects/STRIDE/docs/stride_design_freeze.md),
and it does not replace the Task A migration boundary frozen in
[`docs/task_A/spec.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A/spec.md).
The governing live task specification remains
[`docs/task_A/spec.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A/spec.md).

The current formal result surfaces are archived at
`/mnt/NAS_21T/ProjectResult/STRIDE/task_A/block0` and
`/mnt/NAS_21T/ProjectResult/STRIDE/task_A/block1`.

## 1. Purpose and Current Evidence Status

Task A remains a bounded proxy-validation task inside STRIDE. Its question is
not whether the repository has proved a literal longitudinal transition
process, but whether, under the ordered tissue-domain proxy `TC -> IM -> PT`,
STRIDE recovers non-random relation structure and biologically interpretable
organization on a shared tissue-agnostic community axis. The ordered surface is
still a single-timepoint tissue-domain proxy rather than literal time.

The current live evidence stack contains the descriptive atlas, the rebuilt
Block 0 empirical calibration for `TC-IM`, and the refreshed Block 1
full-cohort paired contrast between `TC-IM` and `TC-PT` with statistical
supplement. Block 3 remains pending as method-level validation.

## 2. Evidence Authority And Historical Scope

The previous Task A result layer included packet-local mirrors that predate the
current Block 0 calibration contract and the current Block 1 statistical
supplement. Those preserved materials remain implementation and proxy-history
context only. They are not live evidence for the rebuilt Block 0 or refreshed
Block 1 result surfaces.

This document therefore treats the NAS Block 0 and Block 1 outputs listed above
as the current result authority.

## 3. Descriptive Atlas Context

The descriptive atlas remains the biological context layer rather than an
inferential proof layer. Its role is to make the shared tissue-agnostic
community axis legible before relation-derived summaries are interpreted. On
the canonical rerun surface, that axis remains biologically structured rather
than arbitrary. Tumor-dominant communities such as `0`, `1`, `3`, `6`, `10`,
`11`, `12`, `16`, and `17` are strongly TC-enriched, usually with secondary IM
representation and little or no PT burden. Myeloid-rich communities `2`, `13`,
and `14` sit predominantly on the IM-to-PT side of the surface, and PT-heavy
immune or interface communities `20`, `21`, `22`, and `23` remain clearly
represented. Mixed or interface-like communities such as `4`, `5`, `15`, `19`,
and `24` remain useful context but should stay supportive rather than
headline evidence.

The atlas also continues to show that the shared axis is cohort-recurrent
rather than a collection of single-patient artifacts. Several major
communities, including `0`, `2`, `3`, and `4`, are present in all 32 patients,
and multiple communities retain broad ROI prevalence across the cohort. The
appropriate atlas conclusion remains restrained: the canonical shared state and
community axis is biologically legible and tissue-context-aware enough to
support downstream interpretation. The atlas does not by itself establish the
main Task A finding.

## 4. Block 0 Patient-Preserving Exchangeability Result

Block 0 asks whether the observed `TC-IM` STRIDE relation surface differs from a
within-patient label-exchangeability reference. For each of 32 patients, the
analysis fitted the observed `TC-IM` relation and 199 within-patient `TC/IM`
domain-label permutations while preserving patient identity and domain-label
count structure. The run produced 6,400 full STRIDE fit records and the analysis
was derived from the saved `A`, `d`, `e`, `mu_minus`, and `mu_plus` cache.

The main cohort-level departures are concentrated in source-side relation
allocation summaries. Burden-weighted `self_retention` was lower in the
observed data than in the permutation reference: median real value `0.4488`
versus null reference `0.9192`, median delta `-0.4704`, empirical p-value
`0.005`. The community-mean scale showed the same direction, with median delta
`-0.2223` and empirical p-value `0.005`.

Burden-weighted `depletion` was higher in the observed data than in the
permutation reference: median real value `0.1582` versus null reference
`0.0033`, median delta `+0.1549`, empirical p-value `0.005`. The
community-mean scale again showed the same direction, with median delta
`+0.1009` and empirical p-value `0.005`.

`off_diagonal_remodeling` was also higher in the observed data than in the
permutation reference, but it remains a diagnostic-supportive summary rather
than a proof-carrying endpoint. On the burden-weighted scale, the median real
value was `0.3486` versus null reference `0.0760`, median delta `+0.2726`, and
empirical p-value `0.005`.

`emergence` did not show a stable cohort-level departure from the permutation
reference. The burden-weighted median p-value was `0.445`, the burden-weighted
mean p-value was `0.845`, the community-mean median p-value was `0.91`, and
the community-mean mean p-value was `0.95`.

The Block 0 result therefore supports the conclusion that the observed
`TC-IM` relation surface is not reducible to the patient-preserving label
permutation reference. It does not by itself identify a biological mechanism,
does not test the `TC-IM` versus `TC-PT` contrast, and does not support a stable
emergence claim.

## 5. Block 1 Real-Data Paired Contrast With Statistical Supplement

Block 1 records the observed full STRIDE relation structure for the
prespecified confirmatory families `TC-IM` and `TC-PT`. The current formal
surface uses the full 32-patient cohort, the canonical `fit_stride(...)` API,
and the Block 1 statistical supplement contract
`task_a_block1_statistical_supplement_v1`. The supplement uses paired
patient-level `TC-IM - TC-PT` deltas, two-sided Wilcoxon signed-rank tests,
two-sided sign tests, BH adjustment within declared surfaces, and an absolute
median-delta effect floor of `0.05`.

At the family level, the review-candidate results support two primary
directions. `self_retention` was higher for `TC-IM` than for `TC-PT`: on the
burden-weighted scale, median delta was `+0.2951`, with `31/32` patients in the
positive direction and BH q-value `2.48e-09`; on the community-mean scale,
median delta was `+0.1539`, with `32/32` patients in the positive direction and
BH q-value `1.86e-09`. `off_diagonal_remodeling` was lower for `TC-IM` than
for `TC-PT`: on the burden-weighted scale, median delta was `-0.2852`, with
`30/32` patients in the negative direction and BH q-value `9.31e-09`; on the
community-mean scale, median delta was `-0.1404`, with `32/32` patients in the
negative direction and BH q-value `1.86e-09`.

Family-level `depletion` is not promoted into the current Block 1 main line
because it did not enter the review-candidate set under the current q-value and
effect-size criteria. `emergence` remains supportive/audit context only,
because Block 0 did not show a stable emergence departure from the
patient-preserving null.

At the source-community level, the full screening surface supports a
tumor-dominant preservation pattern in `TC-IM`. Review-candidate
`self_retention` deltas were positive for communities `0`, `1`, `3`, `6`,
`10`, `12`, `16`, and `17`. The largest examples were community `6`
(median delta `+0.7275`, `31/31` positive, BH q-value `4.99e-09`), community
`10` (`+0.6842`, `30/30` positive, BH q-value `5.59e-09`), community `17`
(`+0.4575`, `32/32` positive, BH q-value `4.99e-09`), and community `12`
(`+0.4391`, `30/30` positive, BH q-value `5.59e-09`). The descriptive atlas
supports the interpretation that these are tumor-dominant communities, enriched
for tumor epithelial, proliferative, VEGF-associated, or CAIX-associated cell
subtypes and concentrated on the `TC`/`IM` side of the tissue-domain surface.

At the target-community level, the review-candidate results were restricted to
`matched_incoming_burden` and were negative for communities `2`, `14`, and
`23`, indicating higher matched incoming burden in `TC-PT` than in `TC-IM`.
Community `2` had median delta `-0.0893`, `32/32` negative direction, and BH
q-value `5.87e-09`; community `14` had median delta `-0.1000`, `28/32`
negative direction, and BH q-value `9.01e-07`; community `23` had median delta
`-0.0844`, `30/30` negative direction, and BH q-value `1.68e-08`. The
descriptive atlas supports a PT-heavy myeloid/interface interpretation for
these targets, with enrichment for macrophage/monocyte-associated subtypes and
high PT-domain fractions.

The cohort relation comparison remains a cohort-level effect map and does not
carry cohort-level p-values. Its key edges are supported by patient-level
relation-element statistics. For example, `A 6->6` had median patient-level
delta `+0.7274`, `32/32` positive direction, and cohort delta `+0.7277`;
`A 6->23` had median patient-level delta `-0.4539`, `32/32` negative
direction, and cohort delta `-0.4543`; `A 10->10` had median patient-level
delta `+0.6842`, `32/32` positive direction, and cohort delta `+0.6844`;
`A 10->22` had median patient-level delta `-0.2337`, `32/32` negative
direction, and cohort delta `-0.2339`.

The Block 1 result therefore supports a proxy-scoped real-data contrast:
`TC-IM` is associated with stronger retention of tumor-dominant source
communities, whereas `TC-PT` is associated with stronger off-diagonal
allocation and matched incoming burden toward PT-heavy myeloid/interface
targets. This is not evidence of literal temporal transition or causal
remodeling mechanism.

## 6. Integrated Evidence Status

The current live result stack supports a proxy-scoped relation-structure
finding. Block 0 shows that the observed `TC-IM` relation surface departs from
a within-patient label-permutation null mainly through reduced self-retention,
increased depletion, and increased off-diagonal remodeling, without a stable
emergence departure. Block 1 shows that, in the real-data confirmatory paired
contrast, `TC-IM` retains tumor-dominant source communities more strongly than
`TC-PT`, while `TC-PT` shows stronger off-diagonal allocation and matched
incoming burden toward PT-heavy myeloid/interface target communities.

This integrated result remains bounded by the ordered tissue-domain proxy
`TC -> IM -> PT`. It does not establish literal temporal transition, causal
remodeling mechanism, lineage tracing, or method superiority.

## 7. Preserved Pre-Migration Context

Earlier Task A outputs remain preserved on disk as proxy-history artifacts and
implementation context. They are not live evidence for the current Block 0
calibration contract and must not be relabeled as completed formal full-STRIDE
evidence.

## 8. Pending Block 3 and Non-Claim Boundary

Block3 first exports rapid semi-synthetic raw and review surfaces; scientific
interpretation is deferred until metrics are reviewed against explicit
questions. The live execution surface is Stage 0 h5ad plus Task A config only.
It uses identity-derived cost geometry for relation support and does not use a
Block 1 manifest path as a live input.

The first run order is export-first. `3A generator_validation` is run first for
manual sanity review of `object_scores`, `rerun_stability`, and the
`review_surface`. Full `3B/3C` execution follows only after that review.

The current Block 3 design uses one train-template multi-FOV semi-synthetic
realization shared across `3A`, `3B`, and `3C`. The generator builds
train-derived templates from real train TC-IM endpoint residuals, mixes a
cohort medoid template with a sampled individual template for held-out patient
truth, and exposes generated endpoint projections to endpoint-only baselines
while exposing generated source/target FOV observations to `stride_reference`
and STRIDE ablation arms. `3B-1` is the shared `A` benchmark, `3B-2` is the
open-focused `d/e` benchmark on the same realization, and `3C-1`, `3C-2`, and
`3C-3` refit `A_p`, `d_p`, and `e_p` on the same generated realization for
the reference and ablation arms.

Block3 does not support or negate Block1 biological conclusions in this
result memo. Preserved Block 3 packet-local materials remain historical/proxy
context and are not integrated here as final full-method closure.

This memo does not claim literal time ordering, direct temporal transition,
true lineage tracing, direct proof that a community disappears or emerges in
vivo, confirmatory status for `IM-PT`, method superiority over baselines, or
Block3 validation of Block1 biology. It does not treat preserved proxy-era
packets as canonical authority. The observation-layer tissue ordering remains
a proxy surface, and the shared community axis remains tissue-agnostic.
