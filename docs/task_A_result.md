# Task A Canonical Results Memo Through Block 2

This document records the canonical Task A result layer through the completed
Step 3 full-STRIDE rerun of the descriptive atlas, Block 0, Block 1, and Block
2. It does not redefine full STRIDE, which remains frozen in
[`docs/stride_design_freeze.md`](/home/lenislin/Experiment/projects/STRIDE/docs/stride_design_freeze.md),
and it does not replace the Task A migration boundary frozen in
[`docs/task_A_rewiring_plan.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A_rewiring_plan.md).
The governing live task specification remains
[`docs/task_A_spec.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A_spec.md).

Earlier versions of Task A result framing were proxy-history only. This memo
supersedes that older result-layer wording by making the canonical Step 3
Block 0-2 rerun explicit while still preserving the proxy-era outputs as
historical context rather than silently relabeling them as canonical evidence.

## 1. Purpose and Current Evidence Status

Task A remains a bounded proxy-validation task inside STRIDE. Its question is
not whether the repository has proved a literal longitudinal transition
process, but whether, under the ordered tissue-domain proxy `TC -> IM -> PT`,
STRIDE recovers non-random relation structure, biologically interpretable
organization, and robust repeatability on a shared tissue-agnostic community
axis. The ordered surface is still a single-timepoint tissue-domain proxy
rather than literal time, so the strongest supported Task A statement remains
proxy-scoped.

The canonical full-STRIDE Step 3 evidence stack now covers the descriptive
atlas, Block 0, Block 1, and Block 2 on the canonical rerun path. The
packet-local review mirror currently recorded in-repo is
`tasks/task_A/result_packets/2026-04-06_canonical_step3_objective_packet/`,
and the packet-local execution and boundary audit is
`tasks/task_A/result_packets/2026-04-06_canonical_step3_audit/`. The audit
records that the canonical Step 3 packet was generated on April 6, 2026, that
the canonical Block 2 manifest is valid, and that the atlas plus Block 0-2
surface is ready for human biological interpretation. Those packet-local
surfaces remain Block 0-2 result mirrors only, not current live Block 3
scientific authority or canonical Block 3 result authority for this pass.
Block 3 remains pending in the canonical scientific stack and is not written
here as completed scientific closure.

## 2. Canonical Provenance and Scope

This memo is grounded in the canonical execution root
`/mnt/NAS_21T/ProjectResult/STRIDE/task_A/2026-04-05_block0_2_canonical_exec_01`
and in the packet-local review surfaces mirrored into the repository on April
6, 2026. The audit states that the canonical prepare, atlas, Block 0, Block 1,
and Block 2 artifacts all report `implementation_tier = canonical_full` and
`evidence_lineage = canonical_rerun`. It also records that canonical Block 1
recurrence outputs are present with `cohort_recurrence_fit_status = ok`, and
that canonical Block 2 completed `284/284` replicates with zero failures.

This provenance matters because the earlier Task A result layer was written
against proxy-history packets. Those earlier packets remain preserved, but the
main result-facing Task A document should now reflect the canonical rerun
surface rather than the older proxy-only stack. The canonical Step 3 rerun is
therefore the primary evidence layer for Task A through Block 2, while the
proxy-era packets remain historical reference only.

## 3. Descriptive Atlas

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

## 4. Canonical Block 0: STRIDE-Native Null Gate

Canonical Block 0 remains the entry gate for discovery. It compares the real
near-proxy family `TC-IM` against the count-preserving randomized target null
family `TC-IM_randomized_target` and asks whether the real near-proxy surface
behaves differently from a broken-proxy null on STRIDE-native summary
quantities. Its scientific role remains narrow and it should not be promoted
into the main biological claim.

The canonical Block 0 gate passes. In
`block0/review/block0_gate_summary.csv`, the proof-carrying continuity
quantity shows a positive median real-minus-null delta of `+0.11819842366428279`
with `27/32` patients in the expected direction. The proof-carrying emergence
quantity shows a negative median real-minus-null delta of
`-0.014049196821615387` with `21/32` patients in the expected direction. The
depletion quantity is directionally aligned supportive context, with median
delta `-0.1181984236642819` and `27/32` patients in the expected direction.
The patient-level review table shows that this separation is distributed across
the cohort rather than being driven by a tiny outlier subset.

The correct Block 0 interpretation therefore remains restrained. The canonical
`TC-IM` near family is not equivalent to the broken-proxy null family on
STRIDE-native summaries, and the near-proxy surface carries non-random
relation structure strong enough to justify advancement to Block 1. Block 0
still does not establish the main biology claim and should not be read as
doing so.

## 5. Canonical Block 1: Real-Data Biological Discovery

Canonical Block 1 remains the primary real-data biological discovery layer. Its
confirmatory family-level comparison is the patient-paired contrast between
`TC-IM` and `TC-PT`. On the canonical rerun surface, the proof-carrying family
summaries remain `self_retention` and `depletion`, while
`off_diagonal_remodeling` and `emergence` remain secondary or supportive
context. The family-level contrast is directionally coherent on both frozen
scales. On the burden-weighted scale, `self_retention` shows a median
`TC-IM - TC-PT` delta of `+0.23959666965320678` with support in `32/32`
patients, while `depletion` shows a median delta of `-0.06844004518662505`
with support in `32/32` patients. On the community-mean scale,
`self_retention` shows median delta `+0.18107037663652514` with support in
`32/32`, and `depletion` shows median delta `-0.0334360262343079` with support
in `32/32`. Secondary summaries reproduce the same direction, but they remain
secondary in interpretive weight.

The community-level reading should remain tiered. On the source side, the main
primary pattern is preferential preservation of tumor-dominant, TC-enriched
communities in the nearer family `TC-IM`. The clearest primary communities
remain `1`, `10`, `11`, and `12`, each of which is carried by both
source-side self-retention and depletion evidence and is reinforced again in
Block 2. Additional tumor-dominant communities `0`, `3`, `6`, `16`, and `17`
remain biologically consistent supportive context rather than a new
confirmatory tier. Mixed or interface-heavy communities, including `15`,
should remain supportive only.

On the target side, the most defensible reading remains relative concentration
of target-side matched structure in PT-heavy immune, myeloid, and interface
communities in the farther family `TC-PT`. The strongest primary target
communities remain `2`, `20`, `22`, and `23` on the incoming-matched surface.
Communities `14` and `21` remain supportive context, while community `13`
remains weaker and should not be upgraded beyond a partial or supportive role.
Emergence-specific target language should stay especially cautious. It can
support the same broad pattern for some communities, but it does not carry the
same status as the source-side proof-carrying summaries.

Canonical Block 1 also now includes the cohort recurrence and common-structure
layer that was absent from the earlier proxy-era Block 1 surface. The
canonical recurrence summary reports `cohort_recurrence_fit_status = ok`, one
conservative first-pass recurrence family for `TC-IM`, one for `TC-PT`, and
`32/32` used patients for each pair family. These recurrence outputs should be
read as supportive cohort-level common structure on the shared state axis,
rather than as a new independent biology claim beyond the carried Block 1
contrast.

The appropriate Block 1 conclusion therefore remains bounded but now better
grounded than before. The canonical rerun supports a confirmatory family-level
contrast between `TC-IM` and `TC-PT`, a biologically interpretable
community-level pattern in which tumor-dominant source communities are
preferentially preserved in the nearer family, and a first-pass cohort
common-structure layer defined on patient relations rather than added only by
post hoc narration. These interpretations are still proxy-scoped and not
literal temporal transition claims.

## 6. Canonical Block 2: Robustness of the Block 1 Findings

Canonical Block 2 remains the robustness layer over the frozen Block 1 summary
surfaces. It does not test baselines, ablations, or method superiority. Its
role is to determine whether the main Block 1 pattern survives reasonable
perturbation of the observed cohort. The primary routes are `patient_subsample`
with `200` executed replicates and `leave_some_out` with `64` executed
replicates, both with zero failures. The secondary `seed_rerun` and
`roi_drop_one_per_domain` routes also executed fully and remain supportive
audit context, but they do not redefine the primary robustness call.

At the family level, the canonical Block 1 core remains robust. The
proof-carrying summaries `self_retention` and `depletion` are robust on both
the burden-weighted and community-mean scales. The secondary summaries
`off_diagonal_remodeling` and `emergence` are also called robust at the family
level, but they remain secondary and should not be promoted above the
proof-carrying source-side contrast.

The source-community robustness layer keeps the claim boundary disciplined
rather than broadening it. Communities `1`, `10`, `11`, and `12` remain the
primary robust source communities on both self-retention and depletion. The
additional tumor-dominant communities `0`, `3`, `6`, `16`, and `17` are also
robust on the canonical review surface, but they remain supportive extensions
of the same source-side preservation pattern rather than a new headline tier.

The target-community robustness layer supports the same bounded reading. The
primary incoming-matched communities `2`, `20`, `22`, and `23` are robust
across the primary routes and continue to define the main target-side PT-heavy
concentration pattern. Communities `14` and `21` remain robust supportive
context on the incoming-matched surface, while community `13` remains partial
there and should stay partial in the memo. Target-side emergence tendencies
remain secondary even when robustness-qualified. Communities `2`, `20`, `21`,
`22`, and `23` are directionally aligned and robustness-qualified on that
secondary surface, community `14` remains partial, and community `13` should
still not be promoted into the main biology layer because its stronger
incoming-matched reading is only partial.

The correct Block 2 conclusion is therefore that the main canonical Block 1
pattern survives the reviewed perturbation routes, but not every
community-level detail carries the same status. The robust core lies in the
family-level `self_retention` and `depletion` contrast, the primary
tumor-dominant source-community preservation pattern, the primary PT-heavy
target-side concentration pattern, and the fact that those findings remain
stable under the primary perturbation routes. Partial findings should remain
partial, and secondary findings should remain secondary.

## 7. Integrated Canonical Step 3 Evidence Statement

Taken together, the descriptive atlas, canonical Block 0, canonical Block 1,
and canonical Block 2 now define the Task A canonical Step 3 evidence stack
through Block 2. The atlas establishes that the shared community axis is
biologically legible. Block 0 shows that the near family `TC-IM` does not
behave like a broken-proxy null on STRIDE-native summaries. Block 1 shows that
the `TC-IM` versus `TC-PT` contrast is organized and biologically
interpretable, while also adding a first-pass cohort recurrence and
common-structure layer on patient relations. Block 2 shows that the main
family-level and community-level components of that pattern remain stable under
the primary perturbation routes.

This canonical evidence stack is stronger than the earlier proxy-history stack
for two concrete reasons. First, the current prepare, atlas, Block 0, Block 1,
and Block 2 surfaces now carry explicit `canonical_full` and
`canonical_rerun` lineage rather than only proxy-era execution history.
Second, the canonical Block 1 bundle now includes cohort recurrence and
common-structure outputs that were absent from the earlier proxy-era Block 1
surface. Even so, the strongest Task A statement remains bounded. Under the
ordered tissue-domain proxy `TC -> IM -> PT`, STRIDE recovers non-random
relation structure, biologically interpretable organization, a first-pass
cohort common-structure layer, and robust repeatability of the main Block 1
pattern through Block 2. This is still not literal temporal proof.

## 8. Preserved Proxy-History Context

The earlier Task A Block 0-2 and Block 3 outputs remain preserved as
proxy-history artifacts and implementation context. This includes the older
review packets under
`tasks/task_A/result_packets/2026-04-05_block2_objective_review_packet/`,
`tasks/task_A/result_packets/2026-04-05_block3_live_exec_01/`, and the earlier
April 3, 2026 packet roots. Those outputs remain useful for historical
reference, audit trail, and implementation context, but they do not supersede
the canonical Step 3 rerun and must not be relabeled as canonical full-STRIDE
evidence.

Preserving that proxy history matters scientifically. The proxy-era outputs
show how Task A evolved, but they were generated before the canonical full
Step 3 Block 0-2 rerun path was recorded explicitly and before the canonical
cohort recurrence layer was attached to the Block 1 surface. They should
therefore remain historical context only.

## 9. Pending Block 3 and Non-Claim Boundary

Canonical Block 3 remains pending as a result-layer conclusion. The preserved
live Block 3 objective review packet under
`tasks/task_A/result_packets/2026-04-05_block3_live_exec_01/` remains proxy
history and is not integrated here as final full-STRIDE method closure. The
current repository's proxy Block 3 ablation implementation likewise remains
historical/proxy execution context only and is not the live `3C` scientific
contract. The
rebuilt Block 3 v2 design is now frozen in
[`docs/task_A_spec.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A_spec.md),
with `3A generator validation`, `3B baseline comparison`, and
`3C ablation study`, where `3B` is fixed to `3B-1 A benchmark` plus
`3B-2 d/e benchmark`, and `3C` is fixed to `3C-1 open-module ablation` plus
`3C-2 cohort-module ablation`. Within that frozen section structure, the
formal section intent is fixed before later Block 3 result interpretation.
`3A` remains the generator-validation background layer for generator realism,
biological plausibility, and rerun-level stability that is sufficient for
downstream evaluation. `3B-1` is the shared-`A` benchmark and `3B-2` is the
shared analysis-layer `d/e` benchmark; later Block 3 interpretation must
follow those frozen intents rather than redefining them from local readouts.
Inside that contract, `3A` validates two held-out cohort-level objects reused
by `3B/3C`: the shared-community `community-space` target fraction surface
(`q_IM` versus synthetic `y`) and the corresponding
`g_k`-projected identity-aware / subtype surface. On each object pair, `3A`
reports `Pearson correlation`, `MAE`, `MSE`, and `JS divergence`, while
`rerun variability` summarizes between-rerun stability of those realism /
plausibility validations. The remaining Block 3 section-level metric sets are
frozen as `3B-1`: `A_MAE_active`, `A_MSE_active`, `target_recall_at_k`;
`3B-2`: `open_support_F1`, `d_MAE`, `e_MAE`, `d_MSE`, `e_MSE`; `3C-1`:
`open_support_F1`, `d_MAE`, `e_MAE`, `d_MSE`, `e_MSE`; and `3C-2`: native
patient-level recovery metrics `A_MAE_active`, `A_MSE_active`,
`open_support_F1`, `d_MAE`, `e_MAE`, `d_MSE`, and `e_MSE` on matched
rerun-specific patient-level semi-synthetic realizations.
Within that frozen contract, `P(m)` is a `train -> generator` quantity only:
it is estimated from `m_proxy = 0.5 * || q_TC - q_IM ||_1` on the `train`
split, used to generate held-out patient open burden within each rerun, and is
not a public benchmark axis or section-facing comparison target.
Within that frozen contract, the frozen `3B` comparator surface is
`balanced_ot_baseline`, `uot_baseline`, `partial_ot_baseline`, and
`diagonal_transport_baseline`, with the fixed baseline IDs interpreted through
the shared `3B` analysis layer. `balanced_ot_baseline` remains the closed
exact OT comparator on the fixed shared cost matrix `C`; `uot_baseline` and
`partial_ot_baseline` use rerun-shared train-side calibration; and
`diagonal_transport_baseline` is strict diagonal matched transport plus
residual open mass. In that framing, `3B-1` remains the relation-layer
benchmark with `A_MAE_active`, `A_MSE_active`, and `target_recall_at_k`,
while `3B-2` remains the open-surface benchmark with `open_support_F1`,
`d_MAE`, `e_MAE`, `d_MSE`, and `e_MSE`. This memo records that frozen intent
only and does not claim that the current internal carrier has already
implemented the full `3B-1/3B-2` split.
In `3C-1`, `open_support_F1` is the support-level recovery metric and the
`d/e` MAE/MSE metrics carry the quantitative profile-recovery role, `3C-1`
does not add `A` headline metrics, and `3C-1` is now a within-STRIDE
loss/regularization ablation with the live readout surface
`open_support_F1`, `d_MAE`, `e_MAE`, `d_MSE`, and `e_MSE`. In `3C-2`, the
live evidence target is a within-STRIDE cohort/common-structure
module-necessity benchmark on matched rerun-specific patient-level
semi-synthetic realizations, with patient-level `A_p`, open-support, and
`d/e` recovery as the core readouts rather than an external baseline
comparison or a biology interpretation section. Within one rerun,
the held-out patients share only the
same train-derived cohort-level generator quantities
(`P(m)`, `pi_d`, `pi_e`, `kappa_d`, `kappa_e`), while each patient's realized
hidden truth still depends on that patient's own `x_p` plus patient-level
sampling. The phrase `shared hidden cohort effect` refers only to that
rerun-specific shared generation context, and `3C-2` does not by itself
establish robustness across multiple cohort/common-structure generation
settings. The live `3C-2` evidence path is fully determined by native
patient-level `A_p`, open-support, and `d/e` recovery metrics; here `A` means
patient-level relation operator `A_p` on the shared `K`-state axis.
The transitional alignment note is recorded in
[`docs/task_A_block3_redesign_v1_1.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A_block3_redesign_v1_1.md).
That note now serves as an explanatory mirror of the live contract rather than
an independent source of scientific authority.
Its canonical scientific execution is still downstream of the evidence stack
recorded in this memo.

This memo does not claim literal time ordering, direct temporal transition,
true lineage tracing, direct proof that a community disappears or emerges in
vivo, confirmatory status for `IM-PT`, or method superiority over baselines. It
does not treat the preserved proxy-era Block 3 packet as canonical authority.
The observation-layer tissue ordering remains a proxy surface, the shared
community axis remains tissue-agnostic, and the strongest supported Task A
conclusion remains explicitly bounded to the canonical evidence established
through Block 2.
