# Task A Specification (v1.1 Locked)
**Title:** Single-timepoint IMC Selection-aware Stress Test

## 1. Global Constraints
- **Data Source**: Single-timepoint IMC, $1\text{ mm}^2$ nominal area, ~9 ROIs/patient (3 CT, 3 IM, 3 PT).
- **Model Params**: $K=25$, $kNN=20$.
- **Calibration Rule**: Cross-compartment contrasts (Arm II) MUST use pair-specific joint calibration (e.g., $\lambda_{CT,PT}$ calibrated on CT $\cup$ PT pool).

## 2. Experimental Arms
### Arm I: Random vs Random (Null/Specificity)
- **Action**: Sample $m=1$ (or 2) ROIs for A, and independently for B, from the *same* compartment.
- **Goal**: Define null distribution of $M$ and $U$. 

### Arm II: Cross-compartment Biased Selection
- **Action**: CT $\leftrightarrow$ PT and IM $\leftrightarrow$ PT. Sample $k \in \{1, 2, 3\}$ ROIs for A (CT) and B (PT).
- **Baseline Requirement**: MUST run Balanced OT (shape-only) on the exact same A/B pairs to show $M_{Balanced} \gg M_{UOT}$.

### Arm III: Coverage Reduction (UQ Calibration)
- **Action**: Grid block sampling at 80%, 40%, 20%, 10% coverage. 
- **Constraint 1 (Sampling)**: A and B independently sample blocks *with replacement* from the full block set (overlap allowed).
- **Constraint 2 (Active Set)**: Active set $\mathcal{K}^{full}$ is frozen based on 100% coverage; it must NOT shrink as coverage drops.

### Arm IV: Synthetic Drift
- **Action**: Inject known drift $\delta$ (offset/gain) into B. 
- **Constraint**: The drift vector passed to the alignment module must be the injected ground truth $\delta$ mapped through the robust scaler, NOT estimated from sparse anchors.

## 3. Required Outputs
- `task_A_metrics.parquet`: Contains `arm`, `k_bias`/`coverage`/`drift`, $U$, $M$, $R$, $\log\_scale$, CI width, drift_aligned_ratio.
