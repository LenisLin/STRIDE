{{ one_liner }}

STRIDE is the live project and scientific name for the repository. It is a
longitudinal spatial remodeling analysis framework/package centered on
patient-level open remodeling relations rather than a transport-first ontology.

The canonical full-method definition is frozen in
`docs/stride_design_freeze.md`. The live implementation exposes the canonical
`stride` first-pass fit path and Task A operational workflows for Block 0/1 and
the internal Block 3 rebuild surface.

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
- Canonical state identity is defined by the shared `K`-state basis.
- Domain or compartment labels act as observation-layer stratification, grouped
  comparison, bridge input grouping, covariates, or analysis surfaces.
- Docs and analyses keep state construction and domain stratification as
  separate modeling layers.
- State geometry and the axes of `A_p`, `d_p`, and `e_p` are defined on the
  shared `K`-state basis.
- If domain labels are unavailable, analyses may fall back to one declared
  domain class.

## Task A Boundary

Task A is the current bounded validation task. It uses a single-timepoint
ordered tissue-domain surface for Task A validation. The Task A rewiring freeze
lives in `docs/task_A/spec.md`; Block 0/1 and the internal Block 3
rebuild are documented in the Task A operational surfaces.

## Current Architecture Status

- `src/stride/` is the task-insensitive core package and live first-pass
  implementation surface.
- `tasks/` owns task-specific workflows, benchmark code, and operational
  documentation.
- `tasks/task_A/` owns Task A Block 0/1 workflows and the internal Block 3
  rebuild package.

## Documentation Map

### Live entry surfaces

- `README.md`
- `docs/index.md`
- `docs/method_overview.md`
- `docs/architecture.md`
- `docs/package_layout.md`

### Canonical project specifications

- `docs/stride_design_freeze.md`
- `docs/state.md`
- `docs/decisions.md`
- `docs/api_specs.md`
- `docs/data_contracts.md`
- `docs/overall_validation_plan.md`
- `docs/constraints.md`

### Task surfaces

- `docs/task_A/spec.md` is the top-level live Task A design document.
- `docs/task_A/block3/scientific_contract.md` records the live Block 3
  generator, benchmark, and ablation contract.
- `docs/task_A/block3/refactor_contract_map.md` is the Block 3 migration map.
- `docs/task_A/result.md` is the Task A results memo through Block 1.
- `tasks/task_A/README.md` is the Task A operational companion for Block 0/1
  workflows and the internal Block 3 rebuild surface.
- `docs/task_B_spec.md`, `docs/task_C_spec.md`, and `docs/task_D_spec.md`
  are bounded task/background notes.

### Source-of-truth order

1. `docs/stride_design_freeze.md`
2. `docs/decisions.md`, `docs/api_specs.md`, `docs/data_contracts.md`,
   `docs/overall_validation_plan.md`, `docs/constraints.md`
3. `docs/state.md`
4. `docs/task_A/spec.md`
5. `docs/task_A/block3/scientific_contract.md` and stage docs under
   `docs/task_A/block3/` for live Task A Block 3 contracts
6. `docs/task_A/block3/refactor_contract_map.md` for migration mapping only
7. `docs/task_A/result.md` and `tasks/task_A/README.md` as derived Task A
   result/operational docs
8. Historical/proxy references only

### Repository layout

```text
docs/                    # active canonical and maintenance docs
src/stride/              # canonical future core skeleton
tasks/                   # task-specific workflows, docs, and benchmarks
scripts/dev/             # maintenance tooling
tests/                   # verification suite
```

## Agent Collaboration

- `AGENTS.md` is the repository-level collaboration protocol for coding agents.
- `docs/agent/README.md` indexes repo-local playbooks for doc sync,
  verification, and Task A Block 3 work.
- `tasks/task_A/AGENTS.md` is the first detailed task-level trial surface.
- These files route work to the existing scientific authority chain; they do
  not replace `docs/*.md`.

## Minimal Setup

```bash
python -m pip install -e ".[dev]"
pytest -q
python scripts/dev/generate_readme.py --check
```

## Project At A Glance

- **Project:** `{{ title }}`
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
