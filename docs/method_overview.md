# Method Overview

STRIDE is a longitudinal spatial remodeling analysis framework for cohorts with
partial, multi-FOV spatial sampling. Its central target is a patient-level open
remodeling relation, not an exact physical transport map.

## Core Scientific Object

For patient `p`, STRIDE summarizes remodeling as `(T_p, e_p)` where
`T_p = [A_p | d_p]`.

### `A_p`: the canonical continuity/remodeling operator

`A_p` is the canonical patient-level relation on the shared `K`-state basis.
It is not documented as a pure conditional kernel.

- `A_p` is row-substochastic, with `sum_j A_{p,ij} + d_{p,i} = 1`.
- `diag(A_p)` captures retention-like structure.
- `offdiag(A_p)` captures remodeling-like structure across states.
- If a normalized conditional kernel is needed for exposition, use the derived
  auxiliary object `R_{p,ij} = A_{p,ij} / (1 - d_{p,i})` when
  `1 - d_{p,i} > 0`.

This keeps `A_p`, not `R_p`, as the canonical STRIDE object.

### `d_p` and `e_p`: explicit open channels

`d_p` captures pre-side depletion tendency. `e_p` captures post-side emergence.
Both remain bounded open-channel summaries inferred under partial observation.
They are not direct proof of true disappearance or true neogenesis.

## Burden versus Composition

STRIDE keeps burden-scale and composition-scale language separate.

- `mu_p^-` and `mu_p^+` are patient-level pseudo-mass / burden vectors on the
  shared `K`-state basis.
- `m_p^(d)` and `m_p^(e)` live on that same pseudo-mass / burden scale.
- `q_p^- = mu_p^- / ||mu_p^-||_1` and
  `q_p^+ = mu_p^+ / ||mu_p^+||_1` are derived normalized compositions.
- Conservation is treated as a soft burden-consistency anchor rather than
  literal physical conservation.

Composition-level structure is usually the most robust under partial coverage.
Burden-level claims require reasonably comparable coverage or platform regimes
and should be weakened or disabled when that comparability is poor.

## Shared State Basis and Domain Boundary

STRIDE is built around a shared `K`-state basis.

- Users should ideally provide patient IDs, ordered analysis sides, FOV/ROI
  IDs, cell/state labels, and domain/compartment labels when available.
- In the current first-pass route, shared community states are built
  tissue-agnostically before any tissue/domain stratification:
  per-cell subtype labels -> within-ROI kNN neighborhood subtype proportion
  vectors -> k-means shared community states.
- The neighborhood size `k` is user-configurable; the documented first-pass
  default is `20`.
- Domain is not part of canonical state identity.
- Domain or compartment labels act only as observation-layer stratification,
  grouped comparison, bridge input grouping, covariates, or analysis surfaces.
- Analyses should not encode domain into the state basis and then condition on
  domain again, because that double counts the same structure.
- Domain labels do not define state geometry or the axes of `A_p`, `d_p`, and
  `e_p`.

If domain labels are unavailable, the observation surface may fall back to one
declared domain class.

## Observation Layer

STRIDE builds evidence through an observation hierarchy:

1. Cells or spots are the local measurement units.
2. FOVs and ROIs are partial spatial observations.
3. Patients are the primary biological interpretation level.
4. Cohorts are where recurrence and consistency are evaluated.

Each observed FOV/ROI is represented in the current first pass as a normalized
community-composition vector `v` on the shared `K`-state basis.

- `v[k]` is formed by counting the cells assigned to shared community/state `k`
  within the ROI/FOV and dividing by the ROI/FOV total cell count.
- This ROI/FOV-level community composition is distinct from the cell-level
  neighborhood subtype composition used during shared community-state
  construction.
- Current first-pass observation mass is uniform, with `mass = 1` and
  `mass_mode = "uniform"` for each ROI/FOV within a study.
- The canonical observation object is a domain-stratified bag-of-FOV empirical
  measure in community-composition space with equal ROI/FOV mass:
  `nu_obs = sum_f w_f delta_{c(v_f)}`.

In the current first pass, `c(v_f) = v_f` and `w_f = 1`.

This observation layer is intentionally comparison-based. The canonical fit
surface is discrepancy or measure comparison over those empirical measures,
not a multinomial, Dirichlet-multinomial, or logistic-normal generative story.

## Role of OT / Sinkhorn

OT / Sinkhorn is used only as an observation-layer cloud comparison tool.

- It compares bag-of-FOV empirical measures defined on a shared state basis.
- It helps support continuity, depletion, and emergence summaries.
- It does not define the primary biological object.
- It should not be described as proving exact physical transport truth.
- Domain-stratified bag-of-FOV comparison is preferred over collapsing the data
  into one raw histogram before fitting.

## Current Uncertainty Surface

When uncertainty is requested in the current first pass, STRIDE reports
bootstrap/sampling-variance uncertainty over realized patient-level bridge
outputs (`A_p`, `d_p`, `e_p`).

- This uncertainty surface is attached to realized bridge outputs rather than
  to a separate hurdle or measurement-error model.
- Deferred or failed bridge fits remain explicit instead of being treated as
  realized uncertainty summaries.

## Why the Relation Is Open

Longitudinal spatial data rarely support a closed, exact matching story:

- pre and post coverage differ,
- FOV sampling is partial and uneven,
- some states may fade, persist, or appear without a one-to-one counterpart.

STRIDE therefore keeps depletion (`d_p`) and emergence (`e_p`) explicit instead
of forcing everything into a closed match. Continuity through `A_p` remains the
main backbone-facing interpretation; open-channel assignment is secondary.

## Cohort Recurrence and Task A

After patient-level relations are built, cohort-level analysis asks whether
similar remodeling relations recur across patients, subgroups, or outcome
strata.

- The cohort target is recurrence or consistency of patient-level relations.
- It is not a claim that one pooled transport plan represents the whole cohort.
- Task A is a bounded validation task under a single-timepoint ordered
  tissue-domain proxy and does not redefine the global STRIDE object.

## Interpretation Boundaries

STRIDE aims for biological interpretability without overclaiming. It does not
claim:

- lineage tracing,
- exact physical transport truth,
- literal FOV-to-FOV material transfer,
- that emergence or depletion is free of sampling or coverage effects.

Instead, it provides a structured patient-level language for describing
longitudinal spatial remodeling under partial observation. The open remodeling
relation is a model-based summarization object rather than a direct causal
object.

## Related Docs

- [Documentation Index](index.md)
- [Repository README](../README.md)
- Internal implementation and contract details remain in the rest of `docs/`
