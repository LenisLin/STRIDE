# Manual Generator Sanity Check

Subexperiment id: `3A`

Semantic experiment name: `generator_validation`

`3A` exports generator-validation raw and review surfaces for manual inspection
before running `3B/3C`. It is not a formal pass/fail gate and does not define a
thresholded scientific decision.

Hard inputs are Stage 0 h5ad and Task A config. The generator builds a
train-derived template bank from real train TC-IM endpoints, uses the
geometry-gated residual coupling with `tau=2.0`, mixes the cohort medoid with a
sampled individual template using `lambda_individual=0.10`, and generates
multi-FOV source/target observations using `eta=0.3`. Community identity
vectors `g_k` are derived internally from the Stage 0 shared-state/cell-subtype
surface.

Generated target FOVs are patient-level truth projections from generated
source-FOV anchors and use no target noise. Endpoint projections are
deterministic views of these generated multi-FOV observations; hidden
`(A_p, d_p, e_p)`, sampled template identity, and real held-out target
endpoints remain scoring or audit-only objects.

Required outputs:

- `object_scores`
- `rerun_stability`
- `review_surface`
- `generator_diagnostics`

Manual inspection checks:

- reported metrics are finite where they are expected to be reported
- real and synthetic target surfaces show no evident shape or scale anomaly
- rerun variability shows no evident uncontrolled instability
- row accounting, endpoint closure, expected evidence blocks, imputed-row mass,
  template provenance, burden ordering, and geometry-locality diagnostics are
  finite and reviewable

These checks decide whether to proceed with the first rapid run sequence. They
do not support a standalone generator-validity conclusion. Generator risks such
as row-pooling artifacts, endpoint-closure error, or too-easy cohort-medoid
structure must be reported as review context rather than hidden.
