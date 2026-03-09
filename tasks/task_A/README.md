# Task A Pipeline (Skeleton)

This task directory contains benchmark-specific orchestration code.
It must implement Plan B:
- structural zeros are bypassed at task level
- `src/slotar` fails fast on programmer-level contract violations and preserves batch-isolated per-item solver failures via `uot_status`

Edit `pipeline.py` to load your Task A dataset and produce:
- metrics table (with uot_status/bypass_reason)
- events table (may be empty initially)

## Baseline Comparators

Minimum required baselines for Task A:
- paired state-abundance comparison on the same physical mass scale used by SLOTAR in this task
- simple compositional summary on that same scale

These baselines answer whether states increase, decrease, disappear, or persist across the paired Task A samples.

For Task A, SLOTAR adds shared transport geometry, transport cost/work, decomposition into retention / remodeling / creation / destruction, uncertainty semantics, and explicit audit / failure semantics beyond those simpler baselines.
