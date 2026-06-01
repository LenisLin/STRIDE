# Documentation Index

STRIDE is the live project and scientific name for the repository. This docs
root separates canonical method docs, task-specific notes, and maintenance
memory.

`src/stride/` now hosts the live first-pass reusable-core implementation
surface. Broader standalone bridge and recurrence estimators remain explicitly
deferred.

## Canonical/Core Docs

- [Repository README](../README.md)
- [STRIDE Design Freeze](stride_design_freeze.md)
- [Method Overview](method_overview.md)
- [Architecture](architecture.md)
- [Package Layout](package_layout.md)
- [Migration Status](state.md)
- [Decisions](decisions.md)
- [API Specifications](api_specs.md)
- [Package API Design](package_api_design.md)
- [Data Contracts](data_contracts.md)
- [Overall Validation Plan](overall_validation_plan.md)
- [Constraints](constraints.md)

Use these files for the active repository story and normative STRIDE contracts:

- the canonical patient-level object `(T_p, e_p)` with `T_p = [A_p | d_p]`,
- row-substochastic `A_p` semantics and the derived-only role of `R_p`,
- burden versus composition semantics on the shared `K`-state basis,
- the domain-stratified bag-of-FOV observation layer and OT/Sinkhorn boundary,
- the rule that domain is an observation-layer surface rather than state
  identity,
- the target user-facing namespace layer and cleanup review workflow recorded
  in the package API design,
- the boundary between reusable `src/stride/` code and task-owned `tasks/` workflows.

For ambiguities, use this source-of-truth order:

1. [STRIDE Design Freeze](stride_design_freeze.md)
2. [Decisions](decisions.md), [API Specifications](api_specs.md),
   [Data Contracts](data_contracts.md), [Overall Validation Plan](overall_validation_plan.md),
   and [Constraints](constraints.md)
3. [Migration Status](state.md)
4. [Task A Spec](task_A/spec.md) for the live Task A specification
5. [Task A Block 3 Scientific Contract](task_A/block3/scientific_contract.md)
   and stage docs under `task_A/block3/`
6. [Task A Block 3 Refactor Contract Map](task_A/block3/refactor_contract_map.md)
   for migration mapping only
7. [Task A Results Memo](task_A/result.md) and
   [Task A Operations README](../tasks/task_A/README.md) as derived Task A
   result/operational docs, including the Block 0/1 results memo and the
   current rerun runbook
8. Historical/proxy references only

## Task Docs

- [Task A Spec](task_A/spec.md) is the top-level live Task A design document.
- [Task A Block 3 Scientific Contract](task_A/block3/scientific_contract.md)
  records the live Block 3 generator, benchmark, and ablation contract.
- [Task A Block 3 Refactor Contract Map](task_A/block3/refactor_contract_map.md)
  is the migration map for retired live paths and artifact roots.
- [Task A Results Memo](task_A/result.md) records the Task A result layer
  through Block 1.
- [Task A Operations README](../tasks/task_A/README.md) is the operational
  companion for Block 0/1 workflows and the internal Block 3 rebuild surface.
- [Task B Background Note](task_B_spec.md) and
  [Task B TONIC Audit](task_B_tonic_data_ecosystem_audit.md) remain
  task-specific background only.
- [Task C Background Note](task_C_spec.md) and
  [Task D Background Note](task_D_spec.md) are bounded task notes.

Task docs narrow or operationalize the method for one task. They do not
override the canonical full-STRIDE design freeze.

## Agent Collaboration

- [Repository Agent Protocol](../AGENTS.md)
- [Agent Playbooks](agent/README.md)
- [Task A Agent Protocol](../tasks/task_A/AGENTS.md)

These files define how coding agents should navigate the repository, verify
changes, and route Task A work without changing the scientific source-of-truth
order.

## Maintenance Docs

- [AVCP Guidelines](avcp_guidelines.md)
- [Dev Log](dev_log.md)
- `docs/readme.template.md` is the source template for the derived repository
  README.
