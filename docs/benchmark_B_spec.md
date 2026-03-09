# Task B Specification: TONIC External Longitudinal Spatial-Validity Task

## 1. Role of Task B
- **Task A defines incremental value**: Task A establishes what SLOTAR adds beyond same-scale state-abundance baselines.
- **Task B defines external spatial validity**: Task B asks whether that added value remains biologically meaningful in an independent public longitudinal spatial cohort (TONIC).
- **Response-blind by design**: The main Task B analysis does not require publicly available patient-level response labels and must not be framed as the primary response-stratified clinical validation task.
- **Interpretation focus**: The strongest TONIC-linked validation axis is tumor-border / compartment concordance together with longitudinal on-treatment spatial reorganization under public metadata constraints.

## 2. Formal Cohort Definition
- **Sole formal Task B cohort**: Curated 40-patient `pre_nivo__on_nivo` patient-level subset in the processed TONIC data.
- **Interpretation**: This is the cleanest available non-primary, treatment-proximal longitudinal slice in the current local/public processed TONIC subset.
- **Observed expansion is not formalized**: The broader observed 48-patient `pre_nivo -> on_nivo` co-presence set is not promoted to a second formal cohort. It may be mentioned only as an exploratory robustness expansion if later needed.
- **Why the curated 40 is primary**: Audit evidence indicates the curated 40 is systematically richer/cleaner in retained processed content, cell support, feature coverage, and compartment breadth, while the excluded 8 are heterogeneous, partly routed into other curated comparison tracks, and partly weaker or borderline.

## 3. Pairing and Interpretation Boundary
- **Pairing unit**: Task B uses patient-level sample pairing, keyed by `Patient_ID`, `Timepoint`, and sample-level `Tissue_ID` / equivalent processed sample metadata.
- **Not lesion-matched**: Same-lesion matching cannot be assumed from current metadata.
- **Not FOV-to-FOV**: Multiple FOVs may exist within a sample; they must be aggregated at the sample level and carried forward through FOV-aware uncertainty and sensitivity logic rather than treated as matched longitudinal transport units.
- **Organ-site limitation**: Public/local TONIC metadata do not currently provide directly usable organ-site / biopsy-site / lesion-location fields.
- **Operational consequence**: Same-organ-site matching cannot be operationalized from current metadata and must appear only as an interpretation limitation, not as a matching rule.

## 4. Spatial Compartment Axis
- **Task-scoped processed TONIC compartments**:
  - `cancer_core`
  - `cancer_border`
  - `stroma_core`
  - `stroma_border`
  - `immune_agg`
- **Scope note**: These are processed TONIC annotations available in the local/public dataset for this task. They are not universal `src/slotar` core-library constructs.
- **Macro reporting folds are allowed**:
  - `tumor = cancer_core + cancer_border`
  - `stroma = stroma_core + stroma_border`
  - `border = cancer_border + stroma_border`
- **Reporting boundary**: Macro folds are acceptable for reporting and simple comparators, but the task may retain the native processed compartment axis where that yields better structural fidelity.

## 5. Canonical Mass Construction
- **Canonical Task B masses must be reconstructed** from processed cell-level data, cluster-level annotations, compartment assignments, and sample metadata.
- **Official operating level remains sample-level / ROI-level nonnegative mass tables** on the shared state axis, consistent with the canonical SLOTAR-ready contract defined elsewhere.
- **Forbidden as direct transport masses**: Summary feature tables containing ratios, log-ratios, differences, normalized values, or any negative-valued summaries must not be used directly as transport masses.
- **FOV handling**: When a sample contains multiple FOVs, their processed cell/compartment information must be aggregated into the declared sample-level mass table while preserving FOV membership for UQ and sensitivity analysis.

## 6. Same-Scale Principle
- **Primary route**: Use density-scale analysis as the default Task B route because area metadata are available in TONIC.
- **Same-scale rule is mandatory**: Baseline abundance comparators and SLOTAR must use the same physical scale.
- **Preferred physical interpretation**: Where area metadata survive preprocessing, density should remain the declared primary mass mode.
- **Restricted fallback**: If accurate area metadata become irrecoverable for a subset after preprocessing, fallback to `count` or `proportion` is allowed only if both baseline and SLOTAR identically adopt that fallback scale and the fallback is explicitly declared in task outputs and reporting.
- **Fallback is not the default**: Do not silently downgrade Task B to counts/proportions when density-scale analysis is still supportable.

## 7. Baseline Comparator Framework
- **Minimum baseline set**:
  - Paired state-abundance comparison
  - Simple compositional summary
  - Compartment-aware border-vs-core abundance contrast
- **Interpretation boundary**: These baselines answer which states and compartments change on the declared physical scale.
- **What they do not answer**: They do not recover transport cost/work, event decomposition, or uncertainty-aware transport structure. That incremental-value claim remains anchored by Task A and is only externally stress-tested here.

## 8. Calibration, Assembly, and UQ
- **Task-driven order is fixed**:
  1. Task defines cohort, pairing, grouping, and declared mass scale.
  2. Library calibration uses that task-compatible context.
  3. Task assembles sample-level batches and metadata.
  4. Library runs batched UOT last.
- **Multi-FOV samples**: Use FOV-aware resampling / bootstrap logic.
- **Single effective FOV / ROI settings**: Use adaptive grid block bootstrap or equivalent single-sample spatial resampling logic with frozen local feature construction where required by the official route.
- **Estimand**: These variances quantify within-sample spatial resampling uncertainty.
- **Interpretation limit**: They do not quantify inter-lesion uncertainty and must not be interpreted as same-lesion replication.
- **Drift handling**: Batch/drift diagnostics may be used for risk flagging and sensitivity analysis, but not for silent correction of matched mass tables.

## 9. External Spatial-Validity Targets
- **Primary validation framing**: Task B is an external concordance task, not proof of SLOTAR event semantics.
- **Core targets**:
  - **Border and compartment concordance**: Assess whether high-signal remodeling aligns with processed TONIC compartment structure, especially tumor-border vs stromal organization.
  - **Longitudinal on-treatment spatial reorganization**: Assess whether SLOTAR detects treatment-associated spatial restructuring in the curated `pre_nivo__on_nivo` cohort.
  - **Robustness to spatial undersampling**: Assess whether uncertainty-aware analysis remains interpretable under multi-FOV vs single-effective-FOV settings and spatial resampling.
- **Published TONIC anchor**: Tumor-border immune structure and on-treatment spatial signals are central published TONIC findings and therefore the strongest external concordance targets for Task B.
- **Claim boundary**: Agreement with those patterns is external biological concordance, not standalone proof that SLOTAR event semantics are fully validated.

## 10. Optional Future Extension
- **Optional extension only**: If response metadata are later obtained under approved access, Task B may be extended to test whether SLOTAR event / geometry / UQ semantics improve responder-vs-non-responder stratification beyond baseline abundance summaries.
- **Current scope boundary**: That response-based extension is not part of the present Task B definition.

## 11. Non-Goals and Interpretation Boundaries
- **Task B does not currently claim**:
  - Same-lesion matched transport
  - Same-organ-site matching
  - Response-stratified main analysis
  - Direct clinical prediction from public metadata alone
- **Cohort boundary**: The curated 40-patient `pre_nivo__on_nivo` subset is the only formal Task B cohort under current metadata reality.
- **Exploratory-only expansion**: The observed 48-patient `pre_nivo -> on_nivo` co-presence set may be used only as an exploratory robustness expansion if explicitly declared later.
