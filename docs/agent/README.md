# Agent Collaboration

This directory contains repo-local playbooks for agents working inside STRIDE.
These playbooks are operational guides only; they do not supersede the
scientific source-of-truth chain in [`../../README.md`](../../README.md) and
[`../index.md`](../index.md).

## Start Here

- Read [`../../AGENTS.md`](../../AGENTS.md) for repository-wide rules.
- If the task touches Task A, then read
  [`../../tasks/task_A/AGENTS.md`](../../tasks/task_A/AGENTS.md).
- Choose the narrowest playbook that matches the requested change.

## Playbooks

- [`playbooks/task-a-block3-change.md`](playbooks/task-a-block3-change.md)
  for Task A Block 3 execution, review-surface, or result-packet changes.
- [`playbooks/doc-contract-sync.md`](playbooks/doc-contract-sync.md) for
  doc-only or contract-routing updates that must stay aligned with higher-order
  docs.
- [`playbooks/verification-and-review.md`](playbooks/verification-and-review.md)
  for final verification, README sync, and handoff discipline.

## Operating Rule

Agents should prefer targeted tests, explicit contract alignment, and manifest
sync over broad reruns or speculative refactors.
