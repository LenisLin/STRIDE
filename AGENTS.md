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
4. `docs/task_A/spec.md`
5. `docs/task_A/block3/scientific_contract.md` and stage docs under
   `docs/task_A/block3/`
6. `docs/task_A/block3/refactor_contract_map.md` for migration mapping only
7. `docs/task_A/result.md` and `tasks/task_A/README.md`
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

## Engineering Trials And Formal Development

- For non-trivial workflow expansion, first classify the requested work as an
  engineering trial or formal development before planning execution.
- Treat unclear early-stage work as an engineering trial only when it does not
  require new public workflow commands, runtime guarantees, stable APIs, or
  changes to frozen or derived scientific contracts.
- During engineering trials, prefer documentation-first reading, minimum
  necessary integration points, and small staged experiments. Record
  assumptions, observed behavior, and unresolved risks. Treat unknowns as
  blockers only when they affect the stated trial goal or cross a repository
  stop-and-ask boundary.
- During formal development, define preconditions, environment probes,
  validation gates, acceptance checks, and required docs/manifests sync before
  claiming executable support.
- Keep trial findings, prototype behavior, runtime support, production
  readiness, stable architecture claims, and stable API claims clearly separate.
- A successful trial may inform formal development, but does not by itself
  establish stable APIs, supported runtime behavior, or production readiness.

## Allowed commands

Default safe commands in this repo are:

- `rg`, `rg --files`, `sed`, `cat`, `ls`, `find`, `git status --short`
- `pytest -q <target>`
- `python scripts/dev/generate_readme.py --check`
- task-local CLI help or dry-run style reads under `tasks/task_A/workflows/`

Prefer targeted checks over full reruns. Do not trigger large scientific reruns
unless the user explicitly asks for them.

## Task A Block 3 Execution Environment

- Use `/home/lenislin/miniconda3/envs/slotar/bin/python` for all Task A Block 3
  Python execution and `/home/lenislin/miniconda3/envs/slotar/bin/Rscript` for
  Task A figure generation.
- Set `PYTHONPATH=src:.` for repository-local Task A imports.
- Prepend
  `/home/lenislin/miniconda3/envs/slotar/lib/python3.10/site-packages/nvidia/nvjitlink/lib`
  to `LD_LIBRARY_PATH` before importing Torch so the slotar CUDA 12.4 wheel does
  not resolve the host CUDA 12.3 `libnvJitLink` first.
- Formal Block 3 execution requires `torch.cuda.is_available() == True` and an
  available `cuda:0` device. STRIDE reference/refit and the task-local UOT
  comparator must fail before execution rather than silently fall back to CPU.
- Exact balanced OT, partial OT, and diagonal transport comparators remain on
  their contract-defined CPU routes.

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
