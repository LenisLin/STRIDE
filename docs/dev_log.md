# Changelog

## Unreleased
- chore: initialize repository skeleton
- docs(architecture): upgrade SLOTAR core algorithm architecture to V1.6 and define output data contracts
- docs(task): lock Task D (ST vignette) proposal and append ST modality adaptation contracts
- docs(api): enforce structural zero bypass and fail-fast solver constraints (C3)
- refactor(architecture): enforce strict physical boundary between library engine and domain-specific tasks (D004)
- 2026-03-15 feat(uot): canonicalize `src/slotar/uot.py::batched_uot_solve(...)` as the single solver entrypoint with fixed `(metrics, details, status)` return structure
- 2026-03-15 feat(task_A/arm3): land the exact-event Arm-3 inference path through solver `details`; proportional prototype allocation remains compatibility fallback only
- 2026-03-15 feat(task_A/arm3): land frozen COUNT-based support-mask enforcement through solver `external_support_mask`
- 2026-03-16 feat(task_A/arm3): refine tau calibration to a pooled Pi-weighted scaled-cost quantile with diagonal self-transport excluded from quantile support
- 2026-03-16 chore(smoke): confirm healthy exercised-subset smoke after the tau refinement, with unchanged `lambda_dens=10.0`, strictly positive `tau_by_compartment`, and healthy exact-event Phase-6 behavior
