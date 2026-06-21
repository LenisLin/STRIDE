# STRIDE `.pl` Planning Note

Date: 2026-06-13

Status: planning note. This file records current plotting-surface decisions and
implementation constraints. It is not a final API contract and does not replace
the higher-priority STRIDE method documents.

## Source-Of-Truth Boundary

Use the repository source-of-truth order when this note conflicts with method
documents. The relevant live authorities are:

- `docs/stride_design_freeze.md`
- `docs/decisions.md`
- `docs/api_specs.md`
- `docs/data_contracts.md`
- `docs/constraints.md`

The `.pl` namespace must remain a visualization layer. It must not redefine
the fitted STRIDE method object, recurrence semantics, AnnData persistence, or
runtime training policy.

## Confirmed Decisions

- `.pl` is for visualization and plotting only.
- `.pl` does not fit, refit, orchestrate training, call optimizer code, or use
  runtime device policy.
- `.pl` does not write h5ad files, result files, R handover files, or
  scientific result payloads.
- `.pl` does not mutate `AnnData`, `FitResult`, `RelationResult`, or
  `CohortResult` inputs.
- `.pl` is not exported from the package root. Users should call it through
  `import stride.pl as pl` or `from stride import pl`.
- Plotting uses `matplotlib`, `seaborn`, and `PyComplexHeatmap` as package
  dependencies. Descriptive heatmap functions use PyComplexHeatmap for complex
  heatmap layout; matplotlib remains the Figure/save boundary.
- Program-score boxplots may use `statannotations` only to render
  caller-supplied `.da` p/q values and significance symbols. `.pl` must not
  ask `statannotations` to run a statistical test.
- If `save` is not supplied, public plotting functions return
  `matplotlib.figure.Figure`.
- If `save` is supplied, public plotting functions write a PDF figure, close
  it, and return `None`.
- PDF export is figure export only; `.pl` does not write h5ad, CSV, stats
  tables, or scientific result payloads.
- Plot data preparation helpers are private by default.
- Reusable `.pl` utilities are plotting utilities only. Input checks remain
  local and lightweight; `_utils.py` is not a validation or audit module.
- Default categorical colors derive from `bio_pastel_pal` and are represented
  in code as `BIO_PASTEL_PALETTE`.
- Plot colors should use stable semantic mappings where possible, so the same
  group, side, or effect-size meaning is represented consistently across
  figures.
- Vertical tick labels should use a consistent right-rotated 90 degree
  orientation when rotation is needed for readability.
- Default community/state order is the existing shared-state index order
  `0..K-1`; `.pl` must not auto-cluster or auto-reorder states.
- Default community/state labels are `C0`, `C1`, ..., `C{K-1}`.
- Descriptive v1 public functions are `community_annotation_heatmap`,
  `fov_composition_heatmap`, and `community_fraction_comparison`.
- Cohort v1 public function is `cohort_relation_heatmap`.
- `community_fraction_comparison` defaults to the `.pp` FOV
  `community_composition` slot and patient-level FOV-mean aggregation.
- External statistics may be rendered as brackets, text, or significance
  symbols, but `.pl` does not compute tests, effect sizes, p-values, q-values,
  tensor decompositions, or correction procedures.
- QC, loss, provenance, warnings, and optimizer details are not public `.pl`
  plots. They may appear only as minimal text annotations when directly needed
  to identify a plotted result.
- Community fraction plots are baseline/context visualizations and must be
  visually and textually distinguished from STRIDE fitted relation plots.
- `.pl` may consume downstream analysis tables or arrays produced by `.da`, but
  `.pl` must not compute group statistics, tensor decompositions, or analysis
  result tables itself.

## Scientific Interpretation Rules

`.pl` must distinguish descriptive input structure from fitted STRIDE relation
outputs.

Descriptive plots show representation and observation context. They do not
show remodeling relations and must not be presented as evidence of fitted
patient remodeling.

Fitted relation plots consume `.tl` result objects. The canonical fitted
patient-level quantities are `A`, `d`, and `e`:

- `A[p]` has shape `[K, K]` and is the patient-level continuity/remodeling
  operator.
- `d[p]` has shape `[K]` and is source-side outgoing open tendency.
- `e[p]` has shape `[K]` and is target-side incoming open-entry tendency.
- Each source row `[A[p, i, :], d[p, i]]` is on a simplex.
- `e[p, j]` is bounded in `[0, 1]`.

For exposition, relation plots may use the augmented display matrix
`M_aug = [[A, d], [e^T, 0]]` with shape `[K + 1, K + 1]`. This is a display
object only. It does not replace the fitted variables `A`, `d`, and `e`.

Plots must not claim:

- literal physical transport,
- true disappearance,
- true neogenesis,
- lineage tracing,
- automatic discovery of multiple remodeling families,
- causal proof from descriptive composition.

## Descriptive v1 Surface

The first `.pl` implementation surface is descriptive and input-aligned. Its
functions are skeletons until plotting logic is implemented.

### 1. Community Annotation Heatmap

Function: `community_annotation_heatmap`

Purpose: explain the shared community/state axis using cell-level AnnData
metadata.

Input:

- AnnData with cell-level community/state assignments.
- Cell subtype, patient, FOV, domain, and optional timepoint metadata.

Default behavior:

- Plot a community by cell-type heatmap.
- Include domain, timepoint, patient/FOV prevalence context where implemented.
- Use cell type alphabetical order unless `cell_type_order` is supplied.
- Use the full AnnData object as the prevalence denominator.
- Treat FOV identity as `(patient_id, timepoint, fov_id)` when timepoint is
  available and `(patient_id, fov_id)` otherwise.
- Retain empty states rather than dropping them.

Scientific question:

- What cell subtype, domain, timepoint, and patient/FOV context explains each
  shared community/state?

Interpretation:

- this is descriptive annotation of the shared state axis,
- it is not a fitted relation plot,
- it does not test differential abundance or remodeling.

### 2. FOV Composition Heatmap

Function: `fov_composition_heatmap`

Purpose: show the `.pp -> .tl` FOV-level community-composition handoff.

Input:

- `adata.uns["stride"]["fov_observations"]["community_composition"]`
- aligned FOV metadata including patient, timepoint, domain, and FOV id.

Default behavior:

- Plot FOV by community composition.
- Sort by metadata for display only; do not modify the slot.
- Show timepoint/domain annotation by default.
- Hide patient annotation by default because patient labels become unreadable
  with many patients.
- Raise `ContractError` in a later implementation when caller-provided patient
  group labels are missing for plotted patients.

Scientific question:

- What FOV-level community composition is consumed by `.tl.fit`?

Interpretation:

- this is an observation-handoff plot,
- it is not a fitted patient relation,
- display sorting does not imply clustering or state redefinition.

### 3. Community Fraction Comparison

Function: `community_fraction_comparison`

Purpose: show the traditional patient-level community fraction baseline that
STRIDE should be interpreted against.

Default input route:

- `adata.uns["stride"]["fov_observations"]["community_composition"]`
- aligned FOV metadata
- source, target, `relations`, and `relation_ids` from
  `adata.uns["stride"]["config"]`

Default scale:

- `fov_state_fraction_mean`, a patient-level mean of FOV state fractions.

Optional scale:

- `cell_state_fraction`, a traditional cell-level abundance baseline from
  `adata.obs`.

Default visual form:

- x-axis: community.
- y-axis: community fraction.
- hue: relation side, source or target.
- panel: `relation_id`.
- with group labels: panel by `relation_id` and group.
- supported plot kinds: box, violin, or box plus strip.

The y-axis label must distinguish:

- `Community fraction (FOV mean)`
- `Community fraction (cell-level)`

Relation handling:

- v1 does not expose `source` or `target` parameters.
- source, target, relation records, and relation ids are read from
  `adata.uns["stride"]["config"]`.
- Patients missing one side of a relation may still contribute the available
  side.
- Paired lines connect only patients with both sides available.

External statistics:

`.pl` only renders supplied statistics. It does not compute p-values,
q-values, effect sizes, BH correction, or statistical tests.

Stats bracket schema:

```text
relation_id optional if only one panel
group optional if group facet exists
community_id required
x1 required
x2 required
p_value optional
q_value optional
label optional
y_position optional
```

Star rendering uses fixed thresholds:

- `<=0.001`: `***`
- `<=0.01`: `**`
- `<=0.05`: `*`
- otherwise: `ns`

Stats rows that cannot be matched to plotted panels, groups, or communities
raise `ContractError`.

`x1` and `x2` identify the bracket endpoints on the rendered side positions.
In the default descriptive v1 layout those endpoint values must resolve to
`source` or `target`. The public mapping parameters `stats_x1_key` and
`stats_x2_key` let callers adapt external stats tables without renaming
columns.

Scientific question:

- Which communities show direct source/target fraction differences?
- Are those differences already visible without STRIDE relation modeling?
- Do caller-provided groups differ at this descriptive baseline level?

Interpretation:

- this is a baseline/context plot,
- it does not show remodeling relation,
- it helps prevent overstating fitted relation findings that are already
  explained by direct fraction differences.

## Cohort v1 Surface

### Cohort Relation Heatmap

Function: `cohort_relation_heatmap`

Purpose: show fitted cohort-supported relation templates on the shared
community axis.

Input:

- `CohortResult`, or
- `RelationResult` with `relation.cohort`, or
- `FitResult`; when `relation_id` is absent, all stored `fit.relation_ids` are
  displayed in declared order.

Required data:

- `template_A`, shape `[K, K]`
- `template_d`, shape `[K]`
- `template_e`, shape `[K]`
- `support_n_patients`, scalar
- `dispersion`, scalar

Default visual form:

- augmented heatmap `M_aug = [[A, d], [e^T, masked]]`,
- upper-left block: `A`, source community by target community,
- right column: `d`, source-side open tendency,
- bottom row: `e`, target-side open-entry tendency,
- bottom-right cell: masked display-only cell, not a fitted value,
- fixed value range `[0, 1]`,
- state order `C0..C{K-1}` unless a full permutation is supplied,
- `FitResult` inputs are split by relation, one augmented heatmap per declared
  relation, without clustering, sorting, or statistical ranking,
- multi-relation figures use compact grid layout; two-relation figures are
  displayed horizontally by default.

Interpretation:

- this is a fitted cohort consensus relation plot,
- the augmented matrix is a visualization object and does not replace the
  fitted `A`, `d`, and `e` variables,
- source-open `d` and target-open `e` are open-channel summaries and should not
  be interpreted as ordinary community-to-community edges,
- it is not a statistical group-comparison plot,
- it does not compute recurrence, dispersion, p-values, q-values, or tensor
  decomposition.

## Fitted Downstream Plotting Surface Under Discussion

The following surfaces remain under discussion and are not part of descriptive
or cohort v1 implementation.

### Patient-Level Downstream Visualization

Purpose: visualize patient-level downstream analyses computed outside `.pl`,
especially `.da` tables derived from fitted `A/d/e`.

Candidate visualization: augmented relation association bubble plot.

- Input:
  - `.da` augmented-entry association statistics table.
- Geometry:
  - same `M_aug = [[A, d], [e^T, masked]]` display layout as
    `cohort_relation_heatmap`,
  - upper-left block is `A`,
  - right column is `d`,
  - bottom row is `e`,
  - bottom-right cell is masked.
- Bubble size:
  - `score = -log10(q_value)`,
  - `<1`: smallest grey point,
  - `[1,2)`: level 1,
  - `[2,3)`: level 2,
  - `[3,4)`: level 3,
  - `>=4`: level 4.
- Color:
  - `cliffs_delta`: diverging, signed, centered at 0,
  - `eta_squared`: sequential, unsigned.
- Labels:
  - axes are labeled as source and target community axes,
  - the `d` column and `e` row remain open-channel display entries rather than
    ordinary community-to-community edges,
  - legend elements explain both effect-size color and `-log10(q_value)` point
    size.
- Boundary:
  - `.pl` does not compute tests, p-values, q-values, effect sizes, or BH
    correction,
  - `.pl` does not read raw patient arrays for this plot.

Candidate visualization: relation-program decomposition outputs.

- Input:
  - `.da` T-only relation-program decomposition tables,
  - rank diagnostics,
  - patient program scores with `program_component_score`,
  - optional program-level group-association statistics,
  - program-entry contribution tables with
    `program_component_contribution`.
- Expected tensor:
  - `X_T[p, i, j] = A_p[i, j]` for `j < K`,
  - `X_T[p, i, K] = d_p[i]`,
  - shape `[P, K, K + 1]`.
- Default visual forms:
  - rank elbow plot from `.da` rank diagnostics,
  - patient program component score boxplot grouped by caller-provided labels,
  - source-community x target/open-axis program structure heatmap,
  - optional core/program weight summary.
- Plot wording:
  - rank elbow y-axis is relative reconstruction error,
  - selected ranks are highlighted only when `.da` marks `selected=True`,
  - program-score y-axis is patient program score,
  - program-structure columns represent target communities plus the
    source-open column.
- Rank diagnostics:
  - rank elbow plots render `status == "ok"` rows,
  - failed restart rows may remain in the `.da` diagnostics table but do not
    block plotting when at least one ok row is available.
- Association annotations:
  - program component score boxplots may select one supplied association
    comparison by `comparison_id`,
  - when multiple stats rows match one relation/program and no comparison is
    selected, `.pl` raises `ContractError` rather than choosing implicitly.
- Boundary:
  - `.pl` does not compute tensor rank, decomposition factors, program
    scores, group statistics, p-values, q-values, or effect sizes,
  - `.pl` renders supplied association annotations without changing them.

## Community-Centric Integration

Purpose: connect community fraction baselines with fitted STRIDE relation
context without making community-role summaries the Q1-Q3 downstream main
route.

Candidate plot: `community_relation_summary`

- Input:
  - community fraction comparison data,
  - cohort `A/d/e`,
  - optionally patient-level variability or group summaries.
- Internal summaries per community:
  - marginal source/target fraction difference,
  - retention: `A[k, k]`,
  - outgoing remodeling: `sum_{j != k} A[k, j]`,
  - incoming remodeling: `sum_{i != k} A[i, k]`,
  - source-open tendency: `d[k]`,
  - target-open tendency: `e[k]`,
  - optional patient variability for these summaries.

This surface is future context. It is not the Q1-Q3 downstream workflow and
does not depend on a `.da` community-role association table. `.pl` still must
not compute association statistics itself.

## Implementation Constraints

Validation should be lightweight and local:

- Use `ContractError` for missing required slots, invalid object types, invalid
  shapes, unknown `relation_id`, unknown `patient_id`, absent `cohort`, missing
  plotted-patient group labels, or mismatched external statistics.
- Use simple finite/nonnegative checks only where needed for plotting safety.
- Do not create audit modules or add persistent audit fields.
- Do not mutate inputs to cache plot data.
- `state_order` v1 requires a complete permutation and does not support
  subsets.
- Empty states are retained and not automatically dropped.

Matplotlib and seaborn behavior:

- Import matplotlib in `.pl` implementation modules.
- Return `Figure` objects when `save` is not supplied.
- When `save` is supplied, require a `.pdf` suffix, create parent directories,
  save with `bbox_inches="tight"`, close the figure, and return `None`.
- Saving writes only the local PDF file. It does not return image bytes,
  display objects, or result payloads.
- Seaborn may be used for categorical distribution plots.
- PyComplexHeatmap is a required descriptive heatmap dependency. Heatmap
  functions fail fast with `ContractError` if it is not importable and use
  PyComplexHeatmap for complex heatmap layout.
- `statannotations` is a required plotting dependency for program-score
  significance rendering. Its role is restricted to drawing supplied
  annotations.
- Prefer deterministic, stable plot layout over automatic clustering.
- `.pl` uses `BIO_PASTEL_PALETTE` as the default categorical palette.

Namespace behavior:

- Keep public exports limited to confirmed plotting functions.
- Do not add `.pl` functions to `src/stride/__init__.py`.

## Testing Targets

Initial tests should be narrow and deterministic:

- `import stride.pl` succeeds.
- `.pl` functions are not present in the package root public `__all__`.
- `BIO_PASTEL_PALETTE` exists and includes the required color keys.
- `community_fraction_comparison` default scale is
  `fov_state_fraction_mean`.
- stats annotation is external-only.
- `community_annotation_heatmap` rejects missing patient, FOV, domain, cell
  type, or community/state fields once implemented.
- `community_annotation_heatmap` preserves default state order and retains
  empty states once implemented.
- `fov_composition_heatmap` consumes the `.pp` FOV observation slot without
  mutating AnnData once implemented.
- `fov_composition_heatmap` display sorting does not modify stored FOV
  observations once implemented.
- `community_fraction_comparison` reads relation source/target sides from
  `adata.uns["stride"]["config"]` once implemented.
- `community_fraction_comparison` distinguishes `fov_state_fraction_mean` from
  `cell_state_fraction` in plot labels once implemented.
- augmented-entry bubble plot consumes the `.da` stats table without mutating
  it once implemented.
- augmented-entry bubble plot masks the bottom-right augmented-matrix cell once
  implemented.
- augmented-entry bubble plot applies the documented `q_value` size bins and
  grey low-signal point policy once implemented.
- augmented-entry bubble plot selects signed diverging color mode for
  `cliffs_delta` and unsigned sequential color mode for `eta_squared` once
  implemented.
- relation-program rank elbow plot consumes `.da` rank diagnostics without
  choosing rank and ignores failed restart rows once implemented.
- relation-program score boxplot consumes `.da` patient
  `program_component_score` and optional association tables without computing
  scores or statistics, with `comparison_id` selecting among multiple
  supplied comparisons once implemented.
- relation-program structure heatmap preserves program, source-community, and
  target/open-axis label alignment, including the source-open `d` column and
  `program_component_contribution`, once implemented.
- `.pl` does not compute group statistics or tensor decompositions.
- plotting functions return `matplotlib.figure.Figure` when `save=None`.
- plotting functions write PDF and return `None` when `save="*.pdf"`.
- non-PDF save paths raise `ContractError`.
- heatmap functions require PyComplexHeatmap to be importable and call the
  PyComplexHeatmap backend.
- `cohort_relation_heatmap` accepts `CohortResult`, `RelationResult`, and
  `FitResult` inputs. `FitResult` inputs default to all stored
  `fit.relation_ids`; `relation_id` narrows the display to one relation.
- community fraction comparison raises `ContractError` when selected relation
  records produce no plottable rows.
- stats endpoint field mappings can be customized through `stats_x1_key` and
  `stats_x2_key`.

Use targeted tests under `tests/`. Set up test execution with the same import
path convention used by the project environment.

## Future Targets

- Patient-level `.tl` relation plots.
- `.da` downstream plots.
- Public plot-data functions.
- Multi-relation grid figures.
- Chord/circle plots.
- Optional file-saving helpers. Current default is no file I/O.
