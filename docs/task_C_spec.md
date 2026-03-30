# Task C Background Note

Task C is not a current executed STRIDE task stack.

This file preserves only the reusable context from the older Task C surface:

- the historical cohort labels and the existence of a main cohort plus E1/E2
  replication cohorts,
- the fact that metadata such as `seg_version`, `mask_version`, and `panel_id`
  were treated as important audit fields,
- the fact that ROI-level QC thresholds and explicit structural-zero handling
  mattered in the historical workflow.

The full pre-refactor Task C design has been archived at
`history/docs/task_C_spec_legacy.md`.

Current boundary:

- do not treat Task C as an active scientific or engineering task stack,
- do not reuse the historical tiering, engine settings, or micro-UOT protocol
  as live STRIDE canon,
- if Task C is revisited, it must inherit the current canonical STRIDE
  contracts instead of reviving transport-first or domain-encoded state
  semantics,
- use the archived file only for historical interpretation of legacy outputs.
