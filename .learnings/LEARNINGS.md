# Learnings

Corrections, insights, and knowledge gaps captured during development.

**Categories**: correction | insight | knowledge_gap | best_practice

---

## [LRN-20260516-001] correction

**Logged**: 2026-05-16T15:50:59+08:00
**Priority**: high
**Status**: pending
**Area**: docs

### Summary
Classify TaskA-Block3 review findings by current experiment impact before labeling them as blockers or residual risks.

### Details
During TaskA-Block3 review, I incorrectly treated two non-blocking observations as residual risks: a `3C` compact metadata fallback that is only an optional provenance-integration test enhancement, and a `partial_ot_plan` rectangular/default-budget concern that only matters if the helper is later reused as a general-purpose OT utility. The user clarified that Block3 benchmark and ablation methods are internal to TaskA-Block3 and do not need future public reuse assumptions.

Correct classification rule: before proposing a blocker, important finding, or residual risk for TaskA-Block3, first ask whether it affects the current internal experiment path, scientific contract, hidden-truth isolation, train/test separation, metric computation, or artifact reproducibility. If it only concerns future helper generalization, optional audit hardening, wording preference, or unrelated local artifacts accepted by the user, do not present it as a current readiness risk.

### Suggested Action
For future TaskA-Block3 audits, explicitly separate:
- current execution/scientific blockers;
- important current-contract issues;
- optional audit/test hardening;
- future reuse or generalization notes that are out of scope for the internal experiment.

### Metadata
- Source: user_feedback
- Related Files: tasks/task_A/block3/execution.py, tasks/task_A/block3/baselines.py, tests/test_task_a_block3_contract_migration.py
- Tags: task-a, block3, scientific-review, risk-classification
- Pattern-Key: task_a_block3.current_scope_risk_classification
- Recurrence-Count: 1
- First-Seen: 2026-05-16
- Last-Seen: 2026-05-16

---
