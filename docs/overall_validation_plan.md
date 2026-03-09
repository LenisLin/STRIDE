# Overall Validation Plan (Evidence Chain)

## 1. Task A: Single-timepoint Controlled Stress Test
- **Data**: Public Single-timepoint IMC (35 markers, ~9 ROIs/patient).
- **Purpose**: Validate selection-awareness (unmatched mass absorbs bias), UQ calibration (CI width vs coverage), and synthetic drift flagging under a strict zero-effect counterfactual constraint.

## 2. Task B: Public Longitudinal Spatial Proteomics
- **Data**: TONIC TNBC spatiotemporal dataset.
- **Purpose**: Demonstrate external longitudinal spatial validity in an independent public cohort, centered on compartment/border concordance, on-treatment spatial reorganization, and robustness of single-ROI UQ under public metadata constraints.

## 3. Task C & C': Private Clinical Cohort (Ultimate Application)
- **Data**: Main multi-ROI Gastric Cancer IMC (pCR vs NR) + External site replication.
- **Purpose**: Provide mechanism-level attribution (RTB/AB remapping) and demonstrate the tool's indispensability in handling severe asymmetric sampling and compartmental structural zeros.

## 4. Task D: Public External Validation / Extensibility Task
- **Data**: Public PDAC Visium spatial transcriptomics (Primary -> Liver Metastasis).
- **Purpose**: Provide a bounded public-dataset external validation and extensibility test for SLOTAR outside the main IMC setting, supporting spatial transcriptomics applicability, metastasis-scenario transfer, and cross-scenario / cross-modality framework extension without overstating biological discovery claims.
