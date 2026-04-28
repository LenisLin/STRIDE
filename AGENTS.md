# STRIDE Agent Protocol

This repository uses a lightweight agent-native collaboration protocol for
Codex, Claude Code, and similar coding agents. These instructions route work to
the existing scientific contracts; they do not replace them.

## Repository Goal

STRIDE is the live repository for longitudinal spatial remodeling analysis in
multi-ROI spatial omics. Agent work must preserve the scientific boundary that
STRIDE is documentation-first, with Task A acting as a bounded validation
surface rather than a redefinition of the full method.

## Source-of-truth order

Use the same source-of-truth order already established in the repository:

1. `docs/stride_design_freeze.md`
2. `docs/decisions.md`, `docs/api_specs.md`, `docs/data_contracts.md`,
   `docs/overall_validation_plan.md`, and `docs/constraints.md`
3. `docs/state.md`
4. `docs/task_A_rewiring_plan.md`
5. `docs/task_A_spec.md`
6. `docs/task_A_block3_redesign_v1_1.md`
7. `docs/task_A_result.md` and `tasks/task_A/README.md`
8. `history/docs/` and `tasks/task_A/result_packets/` as historical/proxy
   reference only

If a lower-priority surface conflicts with a higher-priority one, stop and
align with the higher-priority document before making code or doc changes.

## Writing And Constraint Discipline

- Use objective, evidence-bounded language in technical analysis, planning,
  audits, and review responses. Do not use praise, flattery, reassurance, or
  motivational filler as a substitute for evidence or reasoning.
- When a historical name is designated for removal from the live contract, do
  not preserve it through exclusionary wording. Define the live surface
  positively, and remove the retired name from live docs, active code
  registries, and tests unless a higher-priority authority explicitly requires
  that retention.
- When reading, writing, or revising docs, identify the constraint level of
  each surface before treating it as design authority. Distinguish at minimum
  between:
  - frozen scientific authority
  - derived operational mirror
  - implementation boundary or scaffold
  - historical or proxy reference only
- Lower-constraint surfaces may mirror the contract, but they must not be used
  to redefine higher-constraint ones.

## Standard Lifecycle

Every non-trivial change should follow this lifecycle:

`Explore -> Plan -> Execute -> Verify -> Sync docs/manifests`

- `Explore`: read the highest-priority contracts and the nearest task runbook
- `Plan`: identify the exact files, tests, and manifest/doc updates required
- `Execute`: make the smallest safe change set
- `Verify`: run the narrowest command that proves the change
- `Sync docs/manifests`: update affected runbooks, packet docs, or README entry
  surfaces before claiming completion

## Allowed commands

Default safe commands in this repo are:

- `rg`, `rg --files`, `sed`, `cat`, `ls`, `find`, `git status --short`
- `pytest -q <target>`
- `python scripts/dev/generate_readme.py --check`
- task-local CLI help or dry-run style reads under `tasks/task_A/workflows/`

Prefer targeted checks over full reruns. Do not trigger large scientific reruns
unless the user explicitly asks for them.

## Default verification commands

Pick the narrowest proof command that matches the change. Common defaults:

- `pytest -q tests/test_agent_protocol.py`
- `python scripts/dev/generate_readme.py --check` after touching
  `docs/readme.template.md` or `README.md`
- task-local targeted tests when a change touches `tasks/task_A/`

## Stop And Ask The User

Stop and request user confirmation before doing any of the following:

- a large rerun of Task A or any other long-running scientific workflow
- any external download, dependency installation, or network fetch not already
  requested
- destructive git operations such as reset, checkout-overwrite, or mass delete
- modifying frozen scientific contracts in `docs/` rather than routing through
  existing contract authority
- changing preserved proxy-history artifacts in
  `tasks/task_A/result_packets/`

## Task A Trial Surface

Detailed Task A guidance lives in [tasks/task_A/AGENTS.md](tasks/task_A/AGENTS.md).
Use it for any Task A workflow, Block 3, review-surface, or result-packet
change.

## Repo-local Playbooks

Reusable playbooks live in [docs/agent/README.md](docs/agent/README.md).
Start there for:

- Task A Block 3 changes
- doc/contract synchronization
- verification and review handoff
