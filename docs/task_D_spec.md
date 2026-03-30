# Task D Background Note

Task D is not a current executed STRIDE task stack.

This file keeps only the reusable background needed to remember the historical
external-dataset idea:

- the dataset context was a public PDAC primary-versus-liver-metastasis Visium
  setting,
- the historical design treated each section/sample as one ROI and relied on
  spot-level coordinates plus effective area metadata,
- paired Pri/LiM cohort construction, spot/ROI table semantics, and minimal
  ST representation assumptions remain useful context if this dataset is ever
  revisited.

The full pre-refactor protocol has been archived at
`history/docs/task_D_spec_legacy.md`.

Current boundary:

- do not read the historical protocol as a live task design,
- do not promote its `V1.6`, UOT-grid, transport-plan, or deliverable language
  into the current canonical STRIDE live story,
- if Task D is revisited, it must inherit the current canonical STRIDE
  contracts instead of reviving transport-first or domain-encoded state
  semantics,
- use the archived file only as historical background.
