# Changelog

This file records short repository-history notes only. It is not a live design
surface. For the current scientific framing use `README.md`,
`docs/index.md`, and `docs/method_overview.md`; for live contracts use
`docs/state.md`, `docs/decisions.md`, `docs/api_specs.md`,
`docs/data_contracts.md`, and `docs/overall_validation_plan.md`; for Task A
use `docs/task_A/spec.md`.

## Unreleased
- feat(io): add `stride.io` v1 raw AnnData assembly and h5ad persistence surface
- feat(task-a-block0): release cache-backed Block 0 exchangeability calibration surface and result-packet intake
- docs: realign active repo docs around `src/stride/` as the target core architecture, `src/slotar/` as transitional compatibility, and `history/` as archive-only
- docs: rename the live project/scientific framing to STRIDE while retaining the `slotar` package namespace
- docs: keep Task A as the sole live scientific task surface and narrow Task B/C/D to bounded background notes
- docs: resolve the four hard spec ambiguities for `A_p`, burden/composition scale, the observation layer, and the state/domain boundary
- docs: align `src/slotar/io/longitudinal.py` wording with the canonical patient / timepoint / FOV contract
- build: stop tracking generated `*.egg-info` package metadata in the repository
