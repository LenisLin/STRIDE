# Task B Background Note (TONIC)

Task B is not a current executed STRIDE task stack. This file keeps the
reusable TONIC background and boundary decisions that still matter if the
dataset is revisited later.

## 1. Reusable Cohort Background
- The only formal Task B cohort retained in the current repo memory is the
  curated 40-patient `pre_nivo__on_nivo` subset in the processed TONIC data.
- The broader observed 48-patient `pre_nivo -> on_nivo` co-presence slice is
  not promoted to a second formal cohort.
- Pairing is patient-level sample pairing using `Patient_ID`, `Timepoint`, and
  sample-level `Tissue_ID` or equivalent processed sample metadata.
- Same-lesion, same-organ-site, and FOV-to-FOV matching cannot be assumed from
  the current public/local metadata.

## 2. Reusable Data And Mass-Construction Context
- The task-scoped processed compartment axis is:
  `cancer_core`, `cancer_border`, `stroma_core`, `stroma_border`, and
  `immune_agg`.
- Canonical Task B masses would need to be reconstructed from processed
  cell-level state annotations, compartment assignments, sample metadata, and
  area metadata.
- Summary feature tables containing ratios, log-ratios, differences,
  normalized values, or negative-valued summaries are not valid direct mass
  inputs.
- When a sample contains multiple FOVs, those FOVs should be aggregated to the
  declared sample-level mass table while preserving FOV membership for
  uncertainty or sensitivity analysis.

## 3. Preserved Boundary Conditions
- Density is the intended default Task B mass scale when area metadata remain
  available after preprocessing.
- Same-scale comparison is mandatory: any baseline abundance comparator must
  use the same declared mass scale as STRIDE.
- Multi-FOV support matters operationally, but the strongest available claim
  would remain external concordance under public metadata constraints, not a
  primary method-validation surface.

## 4. Current Repo Boundary
- Keep this file as reusable background only.
- Keep [docs/task_B_tonic_data_ecosystem_audit.md](task_B_tonic_data_ecosystem_audit.md)
  as the detailed audited dataset inventory.
- If Task B is revisited, inherit the canonical STRIDE contracts from
  `docs/decisions.md`, `docs/api_specs.md`, and `docs/data_contracts.md`,
  including row-substochastic `A_p`, burden/composition separation,
  domain-stratified bag-of-FOV observation comparison, and the no
  state-domain-double-counting rule.
- Do not read this file as a near-term execution protocol or as an active Task
  B task stack.
