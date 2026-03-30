# Task C Specification (Gastric Cancer Main & E1/E2 Replication)
**Version:** v3.0 (Final Locked)

## 1. Evidence Tiers & Cohorts
- **Main Cohort**: Primary discovery and application. Runs full SLOTAR pipeline.
- **E1/E2 Cohorts**: Biological replication ONLY. Do NOT run UOT. Evaluated on directional consistency of A1 endpoints.
- **Tier 1 (Primary)**: A1 Macro structural endpoints (Two-part models) and within-compartment longitudinal micro-metrics (restricted to PT).
- **Tier 2 (Supportive)**: Micro-metrics in non-PT compartments, remapping, drift flagging.
- **Tier 3 (Exploratory)**: Cross-compartment alignment (`pre-CT` $\to$ `post-RTB`).

## 2. Hard Data Constraints & Audit Fields
- **Mandatory Metadata**: `seg_version`, `mask_version`, `panel_id` MUST be logged to ensure A1 comparability.
- **QC Threshold**: `n_min_roi = 200`. ROIs below this are flagged (`roi_qc_flag=1`), not silently dropped.

## 3. SLOTAR Parameters (Main Cohort Only)
- **Engine**: $k=20$, $K=25$, $\alpha=0.15$, $\tau_q=0.25$.
- **Structural Zeros**: $I_{p,t,g} = \mathbf{1}\{n\_cells > 0\}$. If $I=0$, micro UOT metrics are `NaN` (STRICTLY NO ZERO-PADDING).
- **UQ**: $\ge 2$ ROIs uses ROI bootstrap. $=1$ ROI uses adaptive grid block frozen bootstrap.

## 4. Endpoints Definition
- **A1 Macro Scale**: $S^{macro}_{p,t,g} = \frac{n\_cells(p,t,g)}{Area(p,t,g)}$.
- **A1-CT (Clearance)**: Evaluated via the zero-component (Hurdle) on $I^{CT}_{post}$, and positive-component measurement error model on $\log S^{macro}_{post, CT}$.
- **A1-RTB (Bed Formation)**: Evaluated via the zero-component (Hurdle) on $I^{RTB}_{post}$, and positive-component measurement error model on $\log S^{macro}_{post, RTB}$.
- **TBCI (Tumor-to-Bed Clearance Index)**: $S^{macro}_{post,CT} / (S^{macro}_{post,CT} + S^{macro}_{post,RTB})$.

## 5. TBD Isolations (Guardrails)
- **`lambda_cross`**: Restricted to Tier 3 exploratory remapping.
- **IVW**: Strictly deprecated. All spatial uncertainty must be handled via the Hurdle + Measurement Error framework (D006).

### Structural Zero Engineering Constraint
- **Implementation Bypass**: The pipeline MUST pre-calculate existence $I_{p,t,g} = \mathbf{1}\{n\_cells > 0\}$. The implementation MUST NOT call `solve_uot` when $I_{pre}=0$ or $I_{post}=0$. 
- It must explicitly catch these cases, write a structural-zero fit-status code, specify `bypass_reason`, and fill micro-metrics with `NaN` to avoid silent failures and math domain errors.
