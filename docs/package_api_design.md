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
| `stride.pp` | Prepare AnnData, validate fields, construct shared states and geometry | `validate_adata`, `build_state_features`, `build_community_state_basis`, `build_geometry`, `build_observations` |
| `stride.tl` | Fit and analyze STRIDE relations | `fit`, `summarize`, `compare`, `bootstrap` |
| `stride.pl` | Plot fitted relations and derived summaries | `relation_heatmap`, `open_channels`, `cohort_relation` |
| `stride.ds` | Provide small tutorial datasets | `toy_spatial`, `task_a_demo` |

## Current stride.io v1 Implementation

`stride.io` v1 is implemented with three public functions:

- `build_adata`
- `read_h5ad`
- `write_h5ad`

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
  `mass_mode`, `n_states`, and `k_neighbors`.
- `adata.uns["stride"]["fov_metadata"]`: used FOV metadata rows as a DataFrame.

FOV metadata is selected by used patient/time/FOV keys. Unused FOV rows are
accepted and omitted from the stored STRIDE metadata. Density input requires
finite positive area for used FOV rows.

`read_h5ad()` and `write_h5ad()` are local h5ad persistence wrappers.
`write_h5ad()` creates parent directories.

## Proposed stride.pp v1 Design

`stride.pp` v1 prepares a raw STRIDE AnnData object for downstream fitting.
Its input is the AnnData object assembled or loaded through `stride.io`.

The proposed public surface is:

| Stage | Function | Role | Main input | Output / AnnData slot |
|---|---|---|---|---|
| 1 | `validate_adata` | Check the raw or prepared AnnData contract | AnnData | no mutation; raises `ContractError` on contract failure |
| 2 | `build_state_features` | Build cell-level neighborhood subtype proportion features | AnnData with cell subtype labels, FOV IDs, and spatial coordinates | `adata.obsm["local_state_features"]` |
| 3 | `build_community_state_basis` | Learn the shared community-state basis | AnnData with local state features and declared `n_states` | `adata.obs["state_id"]`, `adata.uns["state_centroids"]`, `adata.uns["cost_matrix"]`, `adata.uns["cost_scale"]` |
| 4 | `build_geometry` | Build shared-state geometry from the community-state basis | AnnData with state centroids or cost matrix | state-geometry metadata for downstream fitting |
| 5 | `build_observations` | Aggregate cell states into FOV-level community-composition observations | AnnData with state IDs and patient/time/FOV/domain fields | observation-layer payload for `stride.tl` |

The intended preparation sequence is:

```text
stride.io build/read AnnData
        |
        v
stride.pp.validate_adata
        |
        v
stride.pp.build_state_features
        |
        v
stride.pp.build_community_state_basis
        |
        v
stride.pp.build_geometry
        |
        v
stride.pp.build_observations
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
community-state axis.

## User Workflow Template

1. Load a tutorial dataset with `stride.ds` or load user data with `stride.io`.
2. Prepare AnnData with `stride.pp`.
3. Fit STRIDE relations with `stride.tl.fit`.
4. Summarize fitted relations with `stride.tl.summarize`.
5. Visualize summaries or fitted relations with `stride.pl`.

`stride.io` is the read/write layer used by these steps. It handles persistence
and exchange formats rather than fitting.

## Current Implementation Path

The current implemented beta estimator remains `stride.fit_stride(...)`.
Future `stride.tl.fit(...)` should wrap or delegate to the validated core fit
path rather than redefine the scientific objective.

## API And Cleanup Review Workflow

Package cleanup and user API definition proceed together. Each existing
function or class is reviewed before it is exposed through `stride.io`,
`stride.pp`, `stride.tl`, `stride.pl`, or `stride.ds`.

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
| Target workflow step | `ds`, `pp`, `tl.fit`, `tl.summarize`, `pl`, `io`, `internal`, or `task-local` |
| Contract correctness | `pass`, `needs evidence`, or `conflict` |
| Implementation correctness | `pass`, `needs tests`, `bug risk`, or `unsupported input boundary` |
| API suitability | `user-ready`, `wrap`, `refactor before public`, or `internal only` |
| Flexibility | `general`, `first-pass bounded`, `task-specific`, or `hard-coded` |
| Redundancy | `unique`, `overlap`, `compat alias`, or `historical residue` |
| Action | `reuse`, `wrap`, `refactor`, `split`, `internalize`, `task-localize`, `retire`, or `new` |
| Required validation | Exact test, doc check, or workflow check needed |

## Step Review Order

1. `ds`: tutorial data and demo result fixtures.
2. `pp`: AnnData validation, field normalization, shared state basis, geometry,
   and FOV observation construction.
3. `tl.fit`: estimator wrapper, fit orchestration, optimizer/loss internals,
   status handling, and result construction.
4. `tl.summarize`: relation summaries, cohort summaries, uncertainty summaries,
   and summary tables.
5. `pl`: plotting functions that read fitted results or summaries.
6. `io`: read/write functions used across the workflow.

## Implementation Confirmation Checklist

Before a reviewed object becomes part of the target user API, confirm:

1. Its target namespace and workflow step are recorded.
2. Its current callers are known.
3. Its contract and implementation correctness have been reviewed.
4. Its user-facing inputs, outputs, and failure behavior are specified.
5. Its redundancy status and migration action are recorded.
6. Its narrow validation requirement is defined.
