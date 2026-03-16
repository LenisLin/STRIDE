# Task A Arm-3 Coverage Ladder Update Note

**Date:** 2026-03-16

## Files edited

- `docs/task_A_spec.md`
- `docs/task_A_arm3_script_skeleton_plan.md`
- `docs/task_A_arm3_phase0_3_impl_note.md`
- `docs/task_A_arm3_phase4_impl_note.md`
- `docs/task_A_arm3_phase5_6_impl_note.md`
- `tasks/task_A/arm3/constants.py`
- `tasks/task_A/arm3_uq_stress.py`
- `tasks/task_A/arm3/pseudo_roi.py`
- `tasks/task_A/arm3/retention.py`
- `tasks/task_A/arm3/output.py`

## Ladder change

- Old reduced-coverage ladder replaced: `80% / 40% / 20% / 10%`
- New approved scheme now in use:
  - full-coverage reference baseline: `100%`
  - reduced bootstrap levels: `75% / 50% / 25%`
  - bootstrap constant: `(0.75, 0.50, 0.25)`

## Confirmations

- `100%` remains the full-coverage reference baseline built outside the bootstrap loop.
- Active bootstrap levels are only `75% / 50% / 25%`.
- `20%` and `10%` were removed from active Arm-3 coverage-ladder usage.
- No scientific logic other than the coverage ladder wording/constant treatment was changed.
- `src/slotar/` was untouched.
- `pipeline.py` was untouched.

## Intentional remaining `0.10` / `10%` references

- `tasks/task_A/arm3/inference.py` retains `0.10` only for the floor-dominated rule threshold.
- `docs/task_A_arm3_phase5_6_impl_note.md` retains `0.10` / `10%` only to document that same floor-dominated rule threshold.

These remaining references are not Arm-3 coverage levels.
