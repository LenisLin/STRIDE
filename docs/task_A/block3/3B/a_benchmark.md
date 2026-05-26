# A Benchmark

Subexperiment id: `3B-1`

Semantic experiment name: `a_benchmark`

`3B-1` compares `stride_reference` with transport-family baselines on the
shared multi-FOV generator realization. Endpoint-only baselines consume
deterministic endpoint projections of the generated observations; STRIDE
consumes generated source/target FOVs.

`3B-1` reports the shared 18-metric vocabulary used by `3B-2` and `3C-*`.
The primary mass metrics are `F_L1_total`, `g_L1_total`, `e_L1_total`,
`offdiag_mass_abs_error`, `depletion_mass_abs_error`, and
`emergence_mass_abs_error`. Ratio metrics are `offdiag_ratio`,
`depletion_capture`, and `emergence_capture`; `endpoint_y_MAE` is secondary.
Relation and open-channel metrics retain `A_MAE_active`, `A_MSE_active`,
`target_recall_at_k`, `open_support_F1`, `d_MAE`, `d_MSE`, `e_MAE`, and
`e_MSE` for cross-section comparison.

`3B-1` uses one shared condition, `a_benchmark_shared_realization_set`, on the
same train-template multi-FOV generated realization used by `3B-2` and `3C`.

Interpretation reads the raw patient-level detail and the condition summary
columns `mean_value`, `ci_lower`, `ci_upper`, and
`paired_difference_vs_stride_reference`. No pass/fail label is defined here.

The benchmark family is configured by `block3.benchmark_pair_family: "TC-IM"`.
Current validation accepts only `TC-IM`.
