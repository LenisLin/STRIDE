# Subbag Consistency Ablation

Subexperiment id: `3C-1`

Semantic experiment name: `subbag_consistency_ablation`

Display name: `subbag consistency ablation`

Method key: `consistency_ablation`

Core ablation mode: `consistency`

This refit ablation removes only the subbag consistency term. It must refit
`A_p, d_p, e_p`; it must not mask `stride_reference` output; it must not be
post-hoc rescoring only. Retained objective terms are not reweighted, and the
fixed denominator policy is unchanged.

`stride_reference` and `consistency_ablation` both run on the same generated
multi-FOV realization used by `3B`. Both arms fully refit `A_p`, `d_p`, and
`e_p`.

Review reads the full metric direction set for `F_L1_total`, `g_L1_total`,
`e_L1_total`, mass absolute-error metrics, ratio metrics, `endpoint_y_MAE`,
`A_MAE_active`, `A_MSE_active`, `target_recall_at_k`, `open_support_F1`,
`d_MAE`, `d_MSE`, `e_MAE`, and `e_MSE`.
No single primary metric is prespecified in this stage document.
