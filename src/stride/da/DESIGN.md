# STRIDE `.da` Planning Note

Date: 2026-06-13

Status: planning note. This file records post-fit downstream analysis design.
It is not a final public API contract and does not replace higher-priority
STRIDE method documents.

## Source-Of-Truth Boundary

- `docs/stride_design_freeze.md`
- `docs/decisions.md`
- `docs/api_specs.md`
- `docs/data_contracts.md`
- `docs/constraints.md`

`.da` is a proposed downstream analysis namespace. It consumes fitted `.tl`
outputs and produces analysis tables or arrays for review and plotting. It does
not redefine the STRIDE objective, patient relation object, cohort recurrence
object, or plotting boundary.

Current higher-priority package namespace decisions do not yet list `.da` as a
target user namespace. This note records a temporary downstream-analysis design
surface until the package namespace contract is reviewed.

## Confirmed Decisions

- `.da` is post-fit downstream analysis.
- `.da` consumes `FitResult`, `RelationResult`, `CohortResult`, or derived
  patient-level arrays from `.tl`.
- `.da` is not exported from `stride.__init__`.
- `.da` does not fit or refit STRIDE.
- `.da` does not mutate `AnnData`, `FitResult`, `RelationResult`, or
  `CohortResult`.
- `.da` does not write h5ad or result files.
- `.da` does not hard-code clinical labels such as `NR`, `R`, `response`,
  survival, cancer type, or treatment class.
- `.da` outputs should be serialization-friendly `pandas.DataFrame` and
  `numpy.ndarray` objects. If result containers are later added, they should be
  thin wrappers around these tables and arrays.
- Tensor decomposition may use `tensorly` in a later implementation pass.
- v1 tensor decomposition discussion prioritizes T-only non-negative Tucker
  decomposition on `X_T[p, source_community, target_or_source_open]`, where
  `T_p = [A_p | d_p]`.
- The target-open `e` vector is not included in the v1 tensor-decomposition
  input. It remains available for augmented-entry association analysis.
- Augmented `M_aug` is a display/statistical coordinate object, not the
  primary tensor-decomposition input.
- `.da` statistical outputs may be consumed by `.pl` as external annotation
  inputs. `.pl` renders brackets, text, and significance symbols from supplied
  tables, but `.pl` does not compute tests, effect sizes, p-values, q-values,
  or tensor decompositions.

## Scientific Questions And Analysis Responsibilities

The `.da` discussion is organized around downstream scientific questions
before implementation details. Its outputs are fitted-output-derived analysis
surfaces. They are not new estimators, causal models, clinical label
interpreters, or persistence/export layers.

### Q1. How Are Patient-Level STRIDE Arrays Exposed For Downstream Analysis?

`.da` may consume the public `.tl` result surface:

- `RelationResult.relation_id`
- `RelationResult.patient_ids`, aligned to axis `0` of `A`, `d`, and `e`
- `RelationResult.A`, shape `[P, K, K]`
- `RelationResult.d`, shape `[P, K]`
- `RelationResult.e`, shape `[P, K]`
- `FitResult.relations`, `FitResult.relation_ids`, and `FitResult.n_states`
  for multi-relation traversal and shared-state alignment

The main scientific question is:

- For each declared relation and optional caller-defined patient group, which
  fitted patient-level STRIDE arrays are available as direct downstream
  analysis inputs?

The Q1 extraction object is `patient_relation_arrays`, organized by
`relation_id`, optional `group_id`, and patient axis. Each relation/group entry
contains:

- `relation_id`
- optional `group_id`
- `patient_ids`
- `A`
- `d`
- `e`

Inputs are `.tl` `RelationResult` or `FitResult` objects. Q1 consumes only the
relation id, patient ids, raw fitted `A/d/e`, and `FitResult.relation_ids` /
`relations` traversal fields. It does not inspect support, loss, provenance,
warnings, or cohort fields.

Boundary:

- no `.tl` contract revalidation;
- no feature derivation;
- no statistical testing;
- no filtering or biological interpretation of caller-provided groups.

### Q2. How Are Native Augmented Relation Entries Tested Across Groups?

Each declared relation remains a distinct analysis stratum. The default
statistical unit is the patient within a realized relation.

The main scientific question is:

- Are caller-provided patient groups associated with fitted
  augmented relation entries within each declared relation?

For each patient, Q2 uses the conceptual augmented display matrix:

```text
M_aug[p] = [[A[p], d[p][:, None]], [e[p][None, :], masked]]
```

The bottom-right masked cell is display-only and is excluded from all
statistics. Tested native entries are:

- `A[i,j]`
- `d[i]`
- `e[j]`

Statistical policy:

- two independent groups use Wilcoxon rank-sum / Mann-Whitney U;
- multiple groups use one-way ANOVA;
- selecting two groups from a multi-group dataset is represented by an
  explicit two-group comparison record and follows the two-group route;
- BH correction is applied over all non-masked entries within the same
  `relation_id + comparison_id`;
- two-group effect size is Cliff's delta;
- multi-group effect size is eta-squared.

Default analysis should not pool patients across relations, treat relation
rows as independent repeated patients, or auto-generate pairwise contrasts
unless a separate design adds that surface.

### Q3. How Should `.da` Hand Augmented-Entry Statistics To `.pl`?

`.da` owns analysis tables and arrays. `.pl` owns rendering.

The main scientific question is:

- How can downstream fitted-relation summaries be displayed without moving
  statistical testing, effect-size calculation, correction procedures, or
  tensor decomposition into `.pl`?

The intended handoff is:

- `.da` returns an augmented-entry statistics table.
- `.pl` consumes that table directly.
- `.pl` does not calculate tests, p-values, q-values, effect sizes, or BH
  correction.

The Q3 primary plot is the augmented relation association bubble plot using the
same `(K + 1) x (K + 1)` display geometry as cohort relation heatmaps.

### Q4. How Should Tensor Decomposition Be Used For Exploratory Relation Programs?

Tensor decomposition is an exploratory downstream surface. The primary tensor
is T-only, using the fitted source-row relation object `T_p = [A_p | d_p]`:

- `X_T[p, i, j] = A_p[i, j]` for `j < K`
- `X_T[p, i, K] = d_p[i]`
- shape `[P, K, K + 1]`

The main scientific question is:

- Do fitted patient-level `T = [A|d]` matrices contain low-dimensional
  exploratory source-to-target/open relation patterns?

This boundary keeps the row-simplex source-side relation object intact while
excluding the target-open `e` vector from the tensor input. The `e` vector is
still covered by Q2 augmented-entry association. Full `M_aug` tensor
decomposition remains outside the main route.

Default fit scope:

- one shared decomposition per declared relation across all selected patients;
- caller-provided groups are not used to fit separate decompositions;
- group labels are used later for patient-level association of program scores.

Rank policy:

- rank or rank grid is caller-provided;
- rank diagnostics and selected-rank metadata must be explicit;
- `.da` should not make an automatic rank-selection claim without visible
  diagnostics;
- multiple random starts and stability metadata should be recorded when the
  decomposition is implemented.
- selected restart metadata should identify which successful restart produced
  the reported factor, core, program-component, and
  program-component-score tables.

### Q5. How Should Tensor-Decomposition Outputs Be Statistically Tested?

The patient remains the statistical unit. Statistical testing should act on
explicitly defined patient-level program scores by default.

The main scientific question is:

- Are caller-provided patient groups associated with exploratory relation
  program scores?

Program definition:

- a relation program is a selected core factor triple
  `(patient_factor_id, source_factor_id, target_open_factor_id)`;
- the target/open axis contains target communities plus the source-open `d`
  column;
- target/open entries should carry both `target_open_axis_id` and
  `target_open_axis_type`, where the type is `target_community` or
  `source_open`.

Program component contribution:

```text
program_component_contribution[source_i, target_open_j] =
    core_weight[pf, sf, tof]
    * source_factor[source_i, sf]
    * target_open_factor[target_open_j, tof]
```

Patient program component score:

```text
program_component_score[p, program] =
    patient_factor_loading[p, pf] * core_weight[program]
```

Source-community and target/open factor loadings are not patient-level
observations and should not be directly group-tested against patient labels.
Rank choice, initialization, factor selection, and program naming must remain
visible in metadata because they affect interpretation.

Scale convention:

- Tucker factor scale is not identifiable by default.
- `.da` should use a fixed factor-scale convention before reporting
  `core_weight`, `program_component_contribution`, or
  `program_component_score`.
- The v1 convention is L1-normalized columns for patient, source-community, and
  target/open factors, with absorbed scale stored in the core tensor.

Testing policy mirrors Q2:

- patient is the statistical unit;
- two independent groups use Wilcoxon rank-sum / Mann-Whitney U with Cliff's
  delta;
- multiple groups use one-way ANOVA with eta-squared;
- BH correction is applied within `relation_id + comparison_id` over tested
  `program_id` values.

### Q6. How Should Tensor-Decomposition Outputs Be Passed To `.pl`?

The tensor visualization handoff should mirror the downstream-analysis
handoff:

- `.da` computes rank diagnostics, decomposition tables, patient program
  scores, and optional association tables.
- `.pl` renders rank diagnostics, patient program-component-score
  distributions,
  source-to-target/open program structures, and optional program/core
  summaries.

The main scientific question is:

- How can exploratory relation programs be visualized without treating them as
  biological pathways, recurrence families, or causal mechanisms?

Primary `.pl` views:

- rank elbow plot from the rank diagnostics table;
- patient program-component-score boxplot by caller-provided group;
- program structure heatmap with rows as source communities and columns as
  target communities plus the source-open `d` column.

## Discussion And Implementation Phases

### Phase 1: Patient Relation Arrays And Augmented-Entry Association

Phase 1 covers Q1-Q3:

- extract `patient_relation_arrays` from `FitResult` / `RelationResult`,
- organize raw `A/d/e` by `relation_id`, optional `group_id`, and
  `patient_id`,
- test native augmented entries `A[i,j]`, `d[i]`, and `e[j]`,
- use patients as the statistical unit,
- apply BH correction,
- hand the augmented-entry stats table to `.pl`.

Phase 1 does not require `tensorly` and does not address tensor rank,
decomposition stability, or relation-program interpretation.

### Phase 2: Exploratory T-Only Relation Programs

Phase 2 covers Q4-Q6:

- construct the T-only tensor `X_T[p, i, j]` from `T_p = [A_p | d_p]`,
- optionally run one shared non-negative Tucker decomposition per declared
  relation,
- summarize patient, source-community, target/open, core, program-component,
  program-entry, and patient-program-component-score tables,
- record rank diagnostics and selected-rank metadata,
- test patient program scores against caller-provided patient groups,
- provide `.pl` with rank diagnostics, program-component-score,
  program-entry, and optional association tables.

Phase 2 requires a separate dependency, rank, initialization, stability, and
metadata discussion before implementation.

## Input Objects

Primary input:

- `RelationResult.patient_ids`: `[P]`
- `RelationResult.A`: `[P, K, K]`
- `RelationResult.d`: `[P, K]`
- `RelationResult.e`: `[P, K]`

Optional input:

- caller-provided patient group labels aligned to `patient_ids`
- caller-provided state labels
- caller-provided tensor decomposition rank or rank grid
- caller-provided statistical test policy

## Analysis Surface Under Discussion

### 1. Patient Relation Arrays (Q1 Basis)

Extract raw fitted patient-level `A/d/e` arrays without deriving secondary
metrics.

Output:

- nested mapping
  `relation_id -> group_id -> {"patient_ids": tuple[str, ...], "A": ndarray, "d": ndarray, "e": ndarray}`

Scientific question:

- For each declared relation and optional caller-defined group, which fitted
  patient-level arrays are available for downstream analysis?

Boundary:

- this surface exposes fitted model outputs for downstream analysis;
- group ids are organization keys supplied by the caller;
- `.da` does not assign clinical or biological meaning to groups;
- `.da` does not inspect support, loss, provenance, warnings, or cohort fields;
- `.da` does not validate `.tl` contracts, derive features, compute
  statistics, or mutate inputs.

### 2. Augmented Relation Entry Group Association (Q2/Q3)

Compare native augmented relation entries across caller-provided groups.

Required stats table columns:

```text
relation_id
comparison_id
comparison_type
row_id
col_id
entry_type
group_1
group_2
groups
n_total
n_by_group
mean_by_group
median_by_group
std_by_group
test_name
effect_size
effect_size_type
effect_direction
p_value
q_value
correction_method
correction_scope
```

Scientific question:

- Are caller-provided patient groups associated with fitted augmented relation
  entries within each declared relation?

Boundary:

- patient is the statistical unit.
- bottom-right augmented-matrix cell is masked and excluded.
- BH correction should be applied within `relation_id + comparison_id` over
  non-masked entries.
- `.da` must not assign task-specific biological meaning to group labels.
- default discussion assumes relation-stratified analysis rather than pooled
  cross-relation testing.
- association of a fitted `A/d/e` entry does not prove literal physical
  transport, true disappearance, true neogenesis, lineage, or causal
  mechanism.

### 3. T-Only Relation Program Decomposition (Q4)

Use the fitted relation tensor:

- `X_T[p,i,j] = A_p[i,j]` for `j < K`
- `X_T[p,i,K] = d_p[i]`
- shape `[P, K, K + 1]`

Target method:

- non-negative Tucker decomposition.
- rank is caller-provided or chosen by an explicit downstream rank-selection
  procedure.
- output patient factors, source-community factors, target/open factors, core
  tensor, program component definitions, program-entry component
  contributions, and patient program component scores as arrays plus tidy
  tables.

Output tables:

- rank diagnostics table: `relation_id`, `rank_patient`, `rank_source`,
  `rank_target_open`, `restart_id`, `random_seed`, `reconstruction_error`,
  `relative_error`, `status`, `selected`
- patient factor table: `relation_id`, `patient_id`, `group_id`,
  `patient_factor_id`, `loading`
- source factor table: `relation_id`, `source_factor_id`,
  `source_community_id`, `loading`
- target/open factor table: `relation_id`, `target_open_factor_id`,
  `target_open_axis_id`, `target_open_axis_type`, `loading`
- core tensor table: `relation_id`, `patient_factor_id`,
  `source_factor_id`, `target_open_factor_id`, `weight`
- program components table: `relation_id`, `program_id`, `patient_factor_id`,
  `source_factor_id`, `target_open_factor_id`, `core_weight`,
  `program_weight_rank`
- program entry table: `relation_id`, `program_id`, `source_community_id`,
  `target_open_axis_id`, `target_open_axis_type`,
  `program_component_contribution`
- patient program score table: `relation_id`, `patient_id`, `group_id`,
  `program_id`, `program_component_score`

Scientific question:

- Do fitted patient-level `T = [A|d]` matrices contain exploratory
  low-dimensional source-to-target/open relation patterns?

Boundary:

- relation programs are exploratory latent relation factors;
- they are not biological pathways, recurrence families, or causal mechanisms
  without external validation.

### 4. Relation Program Group Association (Q5/Q6)

Compare patient program scores across caller-provided groups.

Output:

- `DataFrame` with `relation_id`, `comparison_id`, `program_id`,
  `comparison_type`, group labels, group summaries, `test_name`,
  `effect_size`, `effect_size_type`, `effect_direction`, `p_value`,
  `q_value`, `correction_method`, and `correction_scope`.

Interpretation:

- relation programs are exploratory latent relation factors.
- they are not biological pathways, recurrence families, or causal mechanisms
  without external validation.
- statistical testing should act on patient-level program scores, not on
  source or target/open factor loadings directly.

## Claim Boundaries And Non-Answers

`.da` analyses do not answer:

- whether a patient group causes remodeling;
- whether a fitted `A/d/e` entry or augmented relation entry is a true physical
  transport, disappearance, neogenesis, or lineage event;
- whether a latent relation program is a biological pathway, recurrence
  family, or causal mechanism;
- whether direct community-fraction or state-abundance baselines can be
  omitted;
- whether task-specific clinical labels such as response, survival, treatment,
  or cancer type have a built-in STRIDE meaning.

Baseline abundance or fraction analysis remains a separate context surface.
When STRIDE-derived augmented-entry associations and baseline composition
associations are reported together, their distinct evidential roles should
remain explicit.

## Not Adopted As Main v1 Analyses

- patient-minus-pooled-cohort-template as the main clinical group comparison.
- group-specific relation robustness as a main analysis.
- within-group relation heterogeneity as a main analysis.
- augmented `M_aug` tensor decomposition as the primary tensor route.
- group-specific tensor decomposition as the main comparison route.
- automatic tensor rank selection without explicit rank diagnostics and
  selected-rank metadata.

These exclusions define the main `.da` discussion surface. They do not prevent
task-level exploratory analyses from being developed separately with explicit
scope and interpretation boundaries.

## Testing Targets

Future implementation tests should verify:

- `patient_relation_arrays` preserves relation, group, and patient axis
  alignment.
- augmented-entry stats exclude the bottom-right masked cell.
- two-group route uses Mann-Whitney / Wilcoxon rank-sum and Cliff's delta.
- multi-group route uses ANOVA and eta-squared.
- BH correction applies within `relation_id + comparison_id` non-masked
  entries.
- T-only tensor construction has shape `[P, K, K + 1]` and maps the final
  column to source-open `d`.
- tensor rank diagnostics preserve rank, restart, seed, reconstruction error,
  relative error, status, and selected-restart metadata.
- tensor decomposition outputs table/array objects with aligned patient,
  source-community, target/open-axis, core, program, and program-entry ids.
- relation program group association uses patient program scores as the
  default tested object.
- `.da` is not exported from the package root.
