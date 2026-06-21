# STRIDE Package API Design

This document defines the target user-facing package architecture for STRIDE.
It describes package organization, user workflows, and namespace
responsibilities. The current implemented API remains recorded in
`docs/api_specs.md` and `docs/state.md`.

## Package Goal

STRIDE is organized as an AnnData-oriented spatial remodeling analysis package.
The user API should let users prepare data, fit patient-level open remodeling
relations, summarize relation outputs, and visualize results through a compact
set of package namespaces.

## Target User Namespaces

| Namespace | Responsibility | Target examples |
|---|---|---|
| `stride.io` | Read/write STRIDE-ready data and relation outputs | `read_h5ad`, `read_relation`, `write_relation` |
| `stride.pp` | Prepare AnnData, validate fields, construct shared states and geometry | `build_local_features`, `build_state_basis`, `build_state_geometry`, `build_fov_observations`, `validate_ready` |
| `stride.tl` | Fit STRIDE relations and expose fitted result containers | `fit`, `FitResult`, `RelationResult`, `CohortResult` |
| `stride.pl` | Plot fitted relations and derived summaries | `community_annotation_heatmap`, `fov_composition_heatmap`, `community_fraction_comparison`, `cohort_relation_heatmap` |
| `stride.da` | Analyze fitted relation outputs downstream | `patient_relation_arrays`, `relation_program_decomposition`, `relation_program_group_association` |

## Current stride.io v1 Implementation

`stride.io` v1 is implemented with AnnData/h5ad helpers and explicit R
handover CSV helpers:

- `build_adata`
- `read_h5ad`
- `write_h5ad`
- `write_r_handover`
- `write_descriptive_tables`
- `write_fraction_table`
- `write_cohort_table`
- `write_program_score_table`

`build_adata()` assembles a raw STRIDE AnnData object from caller-loaded inputs:
`X`, feature-name sequence `var`, `cell`, optional `fov`, explicit field
mappings, and explicit analysis declarations.

The assembled object contains:

- `adata.X`: dense finite numeric cell-by-feature matrix.
- `adata.var`: feature names from `var`; duplicate names use AnnData
  `var_names_make_unique()`.
- `adata.obs`: canonical cell fields plus unmapped cell metadata.
- `adata.obsm["spatial"]`: x/y coordinates from the cell table.
- `adata.uns["stride"]["config"]`: `source`, `target`, `time_order`,
  `community_mode`, `n_states`, `k_neighbors`, explicit `relations`, and
  generated `relation_ids`. Current v1 stores `community_mode = "fraction"`.
- `adata.uns["stride"]["fov_metadata"]`: used FOV metadata rows as a DataFrame.

`relations` records caller-declared domain pairs for the configured source and
target timepoints. Each row stores `source_domain_label` and
`target_domain_label`; `relation_ids` stores stable identifiers generated from
the source timepoint, source domain, target timepoint, and target domain. The
library stores these declarations without assigning biological meaning to the
labels.

FOV metadata is selected by used patient/time/FOV keys. Unused FOV rows are
accepted and omitted from the stored STRIDE metadata. Area may be retained as
optional FOV metadata when supplied, but density community observations are not
part of the current implemented `.io -> .pp -> .tl` fitting path.

`read_h5ad()` and `write_h5ad()` are local h5ad persistence wrappers.
`write_h5ad()` creates parent directories.

`write_r_handover()` writes one caller-supplied CSV handover table with
explicit output path, filename, and primary-key columns. The higher-level
handover helpers write complete plotting tables for descriptive heatmaps,
community-fraction comparisons, cohort relation templates, and relation-program
scores. They do not run statistics, plot figures, discover YAML/config output
paths, or create audit-matrix bundles.

## Current stride.pp v1 Implementation

`stride.pp` v1 prepares a raw STRIDE AnnData object for downstream fitting.
Its input is the AnnData object assembled or loaded through `stride.io`.

The implemented public surface is:

| Stage | Function | Role | Main input | Output / AnnData slot |
|---|---|---|---|---|
| 1 | `build_local_features` | Build cell-level neighborhood subtype proportion features | AnnData with cell subtype labels, FOV IDs, and spatial coordinates | `adata.obsm["local_state_features"]` |
| 2 | `build_state_basis` | Learn the shared community-state basis | AnnData with local state features and declared `n_states` | `adata.obs["state_id"]`, `adata.uns["state_centroids"]` |
| 3 | `build_state_geometry` | Build or validate shared-state geometry | AnnData with state centroids, optional metric, or precomputed cost matrix | `adata.uns["cost_matrix"]`, `adata.uns["cost_scale"]` |
| 4 | `build_fov_observations` | Aggregate cell states into FOV-level community-composition observations | AnnData with state IDs and patient/time/FOV/domain fields | observation-layer payload for `stride.tl` |
| 5 | `validate_ready` | Check the prepared handoff contract before fitting | AnnData with state geometry and FOV observations | no mutation; raises `ContractError` on contract failure |

The intended preparation sequence is:

```text
stride.io build/read AnnData
        |
        v
stride.pp.build_local_features
        |
        v
stride.pp.build_state_basis
        |
        v
stride.pp.build_state_geometry
        |
        v
stride.pp.build_fov_observations
        |
        v
stride.pp.validate_ready
        |
        v
stride.tl fit
```

The v1 design uses explicit stepwise functions. A combined `prepare_adata`
entrypoint is not part of the v1 surface.

During state-feature and community-state construction, cell subtype and spatial
neighborhood information define the shared state basis. Domain labels remain
observation-layer metadata and are attached after the shared basis is defined.
FOV-level observations are community-composition vectors on the shared
community-state axis. In v1, `.pp` builds fraction-scale community
composition for `.tl`; density community observations require a later
observation-schema extension.

`build_state_geometry()` defaults to Euclidean distances between shared-state
centroids and also accepts other built-in `scipy.spatial.distance.cdist` metric
names. If `adata.uns["cost_matrix"]` is already present, `.pp` treats it as
precomputed shared-state geometry: it validates the finite, nonnegative,
symmetric zero-diagonal `[K, K]` contract and computes or checks
`adata.uns["cost_scale"]` from the positive off-diagonal median.

## User Workflow Template

1. Load user data with `stride.io`.
2. Prepare AnnData with `stride.pp`.
3. Fit STRIDE relations with `stride.tl.fit`.
4. Analyze fitted relation arrays and programs with `stride.da` when needed.
5. Visualize fitted relations or derived summaries with `stride.pl`.

`stride.io` is the read/write layer used by these steps. It handles persistence
and exchange formats rather than fitting.

## Current Implementation Path

The current implemented beta estimator is `stride.tl.fit(...)`, exported at the
package root as `stride.fit(...)`. This path owns the formal objective-driven fit
of patient-level `A_p`, `d_p`, and `e_p`; downstream `.da` and `.pl` functions
consume fitted outputs and do not redefine the scientific objective.

## API And Cleanup Review Workflow

Package cleanup and user API definition proceed together. Each existing
function or class is reviewed before it is exposed through `stride.io`,
`stride.pp`, `stride.tl`, `stride.pl`, or `stride.da`.

1. Map the current object to one workflow step.
2. Check contract correctness against the active STRIDE method and data
   contracts.
3. Check implementation correctness for shape, dtype, identifiers, status
   handling, reproducibility, and dependency behavior.
4. Check user API suitability: parameter names, defaults, return type, error
   messages, and compatibility with the next workflow step.
5. Check redundancy against existing functions, compatibility aliases, and
   task-local helpers.
6. Assign one action: `reuse`, `wrap`, `refactor`, `split`, `internalize`,
   `task-localize`, `retire`, or `new`.
7. Add the narrow validation needed before making an implementation claim.

## Review Worksheet

| Field | Meaning |
|---|---|
| Current object | `path::symbol` |
| Current role | Actual behavior in the current code |
| Current callers | Root API, task workflow, tests, docs, or internal use |
| Target workflow step | `pp`, `tl.fit`, `da`, `pl`, `io`, `internal`, or `task-local` |
| Contract correctness | `pass`, `needs evidence`, or `conflict` |
| Implementation correctness | `pass`, `needs tests`, `bug risk`, or `unsupported input boundary` |
| API suitability | `user-ready`, `wrap`, `refactor before public`, or `internal only` |
| Flexibility | `general`, `first-pass bounded`, `task-specific`, or `hard-coded` |
| Redundancy | `unique`, `overlap`, `compat alias`, or `historical residue` |
| Action | `reuse`, `wrap`, `refactor`, `split`, `internalize`, `task-localize`, `retire`, or `new` |
| Required validation | Exact test, doc check, or workflow check needed |

## Step Review Order

1. `pp`: AnnData validation, field normalization, shared state basis, geometry,
   and FOV observation construction.
2. `tl.fit`: estimator wrapper, fit orchestration, optimizer/loss internals,
   status handling, and result construction.
3. `da`: relation-array extraction, relation-program decomposition, association
   summaries, and downstream analysis tables.
4. `pl`: plotting functions that read fitted results or derived summaries.
5. `io`: read/write functions used across the workflow.

## Implementation Confirmation Checklist

Before a reviewed object becomes part of the target user API, confirm:

1. Its target namespace and workflow step are recorded.
2. Its current callers are known.
3. Its contract and implementation correctness have been reviewed.
4. Its user-facing inputs, outputs, and failure behavior are specified.
5. Its redundancy status and migration action are recorded.
6. Its narrow validation requirement is defined.
