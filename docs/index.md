# Documentation Index

STRIDE is the live project and scientific name for the repository. This docs
root separates the canonical live docs from task-specific notes, maintenance
memory, and archival history.

`src/stride/` now hosts the live first-pass reusable-core implementation
surface. Broader standalone bridge and recurrence estimators remain explicitly
deferred.

## Canonical/Core Docs

- [Repository README](../README.md)
- [Method Overview](method_overview.md)
- [Architecture](architecture.md)
- [Package Layout](package_layout.md)
- [Migration Status](state.md)
- [Decisions](decisions.md)
- [API Specifications](api_specs.md)
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
- the boundary between `src/stride/`, `src/slotar/`, `tasks/`, and `history/`.

## Task Docs

- [Task A Spec](task_A_spec.md) is the sole live scientific Task A document.
- [Task A Operations README](../tasks/task_A/README.md) is the operational
  companion for Task A.
- [Task B Background Note](task_B_spec.md) and
  [Task B TONIC Audit](task_B_tonic_data_ecosystem_audit.md) remain
  task-specific background only.
- [Task C Background Note](task_C_spec.md) and
  [Task D Background Note](task_D_spec.md) remain bounded historical/task
  context only.

Task docs narrow or operationalize the method for one task. They are not the
canonical core-package specification.

## Archive/History

- [History Index](../history/docs/index.md)
- `history/docs/` preserves archived legacy specs, pre-refactor proposals, and
  task-history notes.
- Historical code has been moved out of the working tree and is no longer part
  of the in-tree archive.

Archived material is retained for interpretation of earlier repository states.
It is not part of the live installable surface.

## Maintenance Docs

- [AVCP Guidelines](avcp_guidelines.md)
- [Dev Log](dev_log.md)
- `docs/readme.template.md` is the source template for the derived repository
  README.
