# d/e Benchmark

Subexperiment id: `3B-2`

Semantic experiment name: `de_benchmark`

`3B-2` is the open-focused `d/e` benchmark. It compares `stride_reference`
with open-profile comparators on the same shared multi-FOV generator
realization used by `3B-1` and `3C`. Endpoint-only comparators consume
generated endpoint projections; STRIDE consumes generated FOV observations.

The metric vocabulary is the same 18-metric set used by `3B-1` and `3C-*` so
open-channel behavior can be read together with relation, total mass, and
endpoint behavior. It includes `F_L1_total`, `g_L1_total`, `e_L1_total`,
mass absolute-error metrics, `offdiag_ratio`, `depletion_capture`,
`emergence_capture`, `endpoint_y_MAE`, `A_MAE_active`, `A_MSE_active`,
`target_recall_at_k`, `open_support_F1`, `d_MAE`, `d_MSE`, `e_MAE`, and
`e_MSE`.

`balanced_ot_baseline` is intentionally absent from this route. Its closed
marginal constraints force zero derived depletion and emergence under the
shared `P -> A/d/e` analysis layer, so it remains only in `3B-1` as a closed
relation comparator.

`3B-2` uses one shared condition, `de_benchmark_shared_realization_set`.

Interpretation reads raw patient-level detail and condition summaries. This
document does not define a pass/fail gate.

The benchmark family is configured by `block3.benchmark_pair_family: "TC-IM"`.
Current validation accepts only `TC-IM`.
