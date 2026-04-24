# Task A Stage 0 to STRIDE Mapping

## Scope

This contract maps the frozen Task A Stage 0 artifact into the stable STRIDE input tier.

## Real-Data Field Crosswalk

| Raw Stage 0 key | Canonical STRIDE key | Mapping | Layer |
|---|---|---|---|
| `patient_id` | `patient_id` | direct | obs |
| `roi_id` | `fov_id` | alias | obs |
| `compartment` | `domain_label` | alias | obs |
| `compartment` | `FovObservation.timepoint` | derived | obs |
| `timepoint` | _(inert metadata)_ | retained | obs |
| `cell_type` | `cell_subtype_label` | alias | obs |
| `proto_id` | `state_id` | alias | obs |
| `spatial` | `spatial` | direct | obsm |
| `community_features` | `local_state_features` | alias | obsm |
| `prototype_centroids` | `state_centroids` | alias | uns |
| `s_C` | `cost_scale` | alias | uns |
| `cost_matrix` | `cost_matrix` | direct | uns |

### Derived observation-layer semantics

| Field | Value | Note |
|---|---|---|
| `mass` | `1.0` | uniform at adapter time |
| `mass_mode` | `"uniform"` | observation-layer mass contract |

### Timepoint inertness

Raw `timepoint` carries only a single observed value (`"0"`) in the current
cohort.  It is retained as Stage 0 metadata but is **not** used for
ordered-group derivation.  The adapter derives `FovObservation.timepoint`
from the `compartment`/`domain_label` field for two-group family slicing
(`TC-IM`, `TC-PT`, `IM-PT`).

If raw `timepoint` ever carries more than one distinct value, the adapter
raises instead of silently mixing semantics.

### Explicitly unmapped / deferred

- **Unmapped obs fields**: `block_id`, `cell_area`
- **Deferred downstream-only fields** (task-local outputs, not STRIDE input):
  `comparison_id`, `count_stratum_key`, `real_fit_status`, `null_fit_status`

## Ordered tissue-domain family slicing

- Confirmatory families:
  - `TC-IM`
  - `TC-PT`
- Audit-only family:
  - `IM-PT`
- Task A rewrites compartment labels into temporary ordered-group labels only inside the task-local STRIDE adapter.
- Tissue-domain labels remain `domain_label` metadata on each `FovObservation`; they do not become shared-state identity.

## Stable STRIDE attachment

- Task A may wrap frozen Stage 0 artifacts in `DatasetHandle`.
- Task A may load or rebuild the shared basis with `BasisSpec`.
- Task A calls canonical `fit_stride(...)` on these two-group family slices.
- The preserved `fit_stride_proxy(...)` surface remains compatibility code only
  and is not the authoritative Task A rerun path.
- Task A consumes `STRIDEFitResult`, `PatientBridgeResult`, cohort-level
  recurrence exports, and optional bootstrap uncertainty only through
  task-local adapters and summaries.

## Demo subset: alignment_v1

Patients: `B10`, `B12`, `B3`, `W18`

| Patient | TC | IM | PT | ROIs |
|---|---|---|---|---|
| B10 | 3 | 3 | 3 | 9 |
| B12 | 2 | 4 | 3 | 9 |
| B3 | 3 | 3 | 4 | 10 |
| W18 | 3 | 3 | 2 | 8 |

Total: 215,605 cells, 36 ROIs, 4 patients.  Covers every real ROI-per-domain
pattern present in the full cohort while remaining cheap to run.
