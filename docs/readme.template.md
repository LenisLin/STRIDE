{{ one_liner }}

STRIDE is the live project and scientific name for the repository. It is a
longitudinal spatial remodeling analysis framework/package centered on
patient-level open remodeling relations rather than a transport-first ontology.

## What STRIDE Represents

For patient `p`, the canonical scientific object is `(T_p, e_p)` with
`T_p = [A_p | d_p]`.

- `A_p` is the canonical patient-level continuity/remodeling operator on the
  shared `K`-state basis.
- `A_p` is row-substochastic, with `sum_j A_{p,ij} + d_{p,i} = 1`.
- If exposition needs a normalized conditional kernel, use the derived
  auxiliary object `R_{p,ij} = A_{p,ij} / (1 - d_{p,i})` when
  `1 - d_{p,i} > 0`.
- `d_p` is pre-side depletion tendency.
- `e_p` is post-side emergence.

Continuity through `A_p` is the primary structural interpretation. `d_p` and
`e_p` remain bounded open-channel summaries rather than direct proof of true
biological disappearance or neogenesis.

## Burden and Composition

STRIDE keeps burden and composition distinct.

- `mu_p^-` and `mu_p^+` are patient-level pseudo-mass / burden vectors on the
  shared `K`-state basis.
- `m_p^(d)` and `m_p^(e)` live on that same pseudo-mass / burden scale.
- Normalized compositions are derived objects only:
  `q_p^- = mu_p^- / ||mu_p^-||_1` and `q_p^+ = mu_p^+ / ||mu_p^+||_1`.
- Conservation is a soft burden-consistency anchor rather than literal
  physical conservation.

Composition-level structure is usually the most robust under partial coverage.
Burden-level claims require reasonably comparable coverage or platform regimes
and should be weakened when that comparability is poor.

## Observation Layer

Each observed FOV/ROI is represented in the current first pass on the shared
`K`-state basis by a normalized community-composition vector `v`.

- `v[k]` is formed by counting the cells assigned to shared community/state `k`
  within the ROI/FOV and dividing by the ROI/FOV total cell count.
- Current first-pass observation mass is uniform, with `mass = 1` and
  `mass_mode = "uniform"` for each ROI/FOV within a study.
- The canonical observation object is a domain-stratified bag-of-FOV empirical
  measure in community-composition space with equal ROI/FOV mass:
  `nu_obs = sum_f w_f delta_{c(v_f)}`.
- In the current first pass, `c(v_f) = v_f` and `w_f = 1`.
- OT / Sinkhorn compares these empirical measures; it is an observation-layer
  comparison tool, not the primary biological object.
- Domain-stratified bag-of-FOV comparison is preferred over collapsing the
  data into one raw histogram.

## State Basis and Domain Boundary

STRIDE uses a shared `K`-state basis built before any tissue/domain
stratification.

- The current first-pass route is tissue-agnostic:
  per-cell subtype labels -> within-ROI kNN neighborhood subtype proportion
  vectors -> k-means shared community states.
- The neighborhood size `k` is user-configurable; the documented first-pass
  default is `20`.
- Domain is not part of canonical state identity.
- Domain or compartment labels act only as observation-layer stratification,
  grouped comparison, bridge input grouping, covariates, or analysis surfaces.
- Docs and analyses must not encode domain into the state basis and then
  condition on domain again, because that double counts the same structure.
- Domain labels do not define state geometry or the axes of `A_p`, `d_p`, and
  `e_p`.
- If domain labels are unavailable, analyses may fall back to one declared
  domain class.

## Task A Boundary

Task A remains the current bounded validation task. It uses a single-timepoint
ordered tissue-domain proxy, not full longitudinal proof, and it does not
redefine the global STRIDE object.

## Current Architecture Status

- `src/stride/` is the canonical future task-insensitive core package
  skeleton.
- `src/slotar/` is the current transitional compatibility and implementation
  namespace.
- `tasks/` owns task-specific workflows, benchmark code, and operational
  documentation.
- `history/` is archival only and is not part of the live installable surface.

## Documentation Map

### Live entry surfaces

- `README.md`
- `docs/index.md`
- `docs/method_overview.md`
- `docs/architecture.md`
- `docs/package_layout.md`

### Canonical project specifications

- `docs/state.md`
- `docs/decisions.md`
- `docs/api_specs.md`
- `docs/data_contracts.md`
- `docs/overall_validation_plan.md`
- `docs/constraints.md`

### Task and archive surfaces

- `docs/task_A_spec.md` is the sole live scientific Task A document.
- `tasks/task_A/README.md` is the Task A operational companion.
- `docs/task_B_spec.md`, `docs/task_C_spec.md`, and `docs/task_D_spec.md`
  remain bounded task/background notes only.
- `history/docs/index.md` is the in-tree archive-docs entrypoint.
- Historical code is archived outside the repo working tree.

### Repository layout

```text
docs/                    # active canonical and maintenance docs
history/docs/            # archived documentation
src/stride/              # canonical future core skeleton
src/slotar/              # transitional compatibility namespace
tasks/                   # task-specific workflows, docs, and benchmarks
scripts/dev/             # maintenance tooling
tests/                   # verification suite
```

## Minimal Setup

```bash
python -m pip install -e ".[dev]"
pytest -q
python scripts/dev/generate_readme.py --check
```

## Project At A Glance

- **Project:** `{{ title }}`
- **Python distribution:** `{{ name }}` (retained during migration)
- **Core package direction:** `stride` target core + `slotar` transitional compatibility layer
- **Domain:** {{ domain }}
- **Stage:** {{ stage }}
- **Owner:** {{ owner }}
- **License:** {{ license }}

{% if datasets %}
## Datasets

{% for dataset in datasets %}- `{{ dataset }}`
{% endfor %}
{% endif %}
{% if outputs %}
## Outputs

{% for output in outputs %}- `{{ output }}`
{% endfor %}
{% endif %}
## README Maintenance

`README.md` is generated from `project.yaml` and `docs/readme.template.md`.
