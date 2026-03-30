# Task D Protocol v1.1

## Public Visium External Validation: Primary → Liver Metastasis (PDAC)

### Role in the evidence chain (pre-registered)

Task D is a **public-dataset external validation and extensibility task**. Its goal is to demonstrate that SLOTAR’s **measure alignment abstraction** can be instantiated on spatial transcriptomics (Visium) and applied to a **public metastasis use case** (primary→liver metastasis) outside the main IMC setting, without claiming causal treatment effects or making strong cross-organ calibration assumptions.
This task supports **framework-level cross-scenario and cross-modality transfer** with bounded cohort-level interpretation. Primary outputs emphasize **runability, auditability, remapping, and sensitivity to regularization**, not standalone biological discovery claims.

---

## D0. Data source and scope

### D0.1 Primary dataset

* GEO SuperSeries **GSE277783**, Visium subset **GSE274557** (PDAC primary + matched metastases), with samples designed such that **matched primary and metastasis are placed on the same Visium slide** where possible to reduce batch risk.
* CosMx subset **GSE277782** exists but is **out of scope** for the core vignette (optional extension only).

### D0.2 Cohort definition for this vignette

* Patients: 13 PDAC patients (as per GEO description).
* **Primary analysis cohort**: first-pass paired samples only, with `A = Pri` and `B = LiM` as the default operational mapping for the main cohort.
* Sites: **Primary (Pri)** and **Liver metastasis (LiM)** only for the default primary analysis. Other metastatic sites are excluded from that main cohort.
* ROI: each Visium section/sample (GSM) is treated as **one ROI**.
* Additional samples (`Pri2/Pri3`, `C/D/E`, `PDX`) are retained only for audit, replicate sanity checks, or optional sensitivity analyses; they are not part of the default primary analysis cohort.

---

## D1. Frozen decisions (must be implemented)

### D1.1 Data pairing (Primary analysis)

* **Paired-only, first-pass default**: include only patients with a first-pass `A/B` pair, where `A = Pri` and `B = LiM`.
* If either the `A` or `B` sample is missing or technically unusable, exclude that patient from the default primary analysis cohort and retain the sample only in audit outputs or optional sensitivity work.
* **Slide matching**:

  * If `slide_id` is identifiable: retain only Pri–LiM pairs that are **same-slide matched**.
  * If `slide_id` is not identifiable: retain paired samples but set `slide_match="unknown"` and use it only as an audit field (no correction).

### D1.2 ROI definition

* `roi_id := GSM`
* `patient_id := Pt-*` parsed from metadata/sample naming
* `site := Pri vs LiM` (LiM restricted to liver)

### D1.3 Inclusion/exclusion

* Include only first-pass `A/B` samples in the **default primary analysis cohort**.
* Treat **Pri2/Pri3** as auxiliary primary replicates for audit, replicate sanity checks, or optional sensitivity analyses only; they are not part of the default primary analysis cohort.
* Exclude **PDX** samples, non-liver metastases, and non-`A/B` labels from the default primary analysis cohort.

### D1.4 Grouping g

* **Primary**: `g = all` (no tumor/stroma splitting).
* **Optional sensitivity** (Tier=descriptive only): proxy grouping via module scoring (epithelial vs stromal), clearly labeled as a proxy.

### D1.5 Representation and prototypes

* kNN = 20 (fixed)
* K = 25 (fixed)
* Cost scaling (s_C) fixed per dictionary.
* The default primary path uses the minimal contract-compatible Visium representation in Section D4; any more selective representation choice remains task-scoped and sensitivity-only unless separately frozen.

---

## D2. Input data contracts (tables and audit fields)

### D2.1 `cells` table (Visium spots as “cells”)

Each row is a Visium spot:

* `patient_id`, `roi_id`, `site` (Pri/LiM), `slide_id` (if available), `cohort_id="GSE274557"`
* spot coordinates: `x, y` (in Visium coordinate system)
* expression: raw counts (gene × spot) stored separately; derived embeddings stored as fields:

  * `pc[1..P]` (PCA coordinates), with P fixed (default 50)

### D2.2 `rois` table

* `roi_id`, `patient_id`, `site`, `slide_id` (optional)
* `n_spots_total`, `n_spots_tissue`
* `area_effective` and `area_mode`:

  * `area_effective = n_spots_tissue × spot_area` (spot_area constant for Visium; tissue spots determined by Space Ranger tissue positions)
  * `area_mode="spots×spot_area"` (primary)
* Optional: `image_path`/`scalefactors` pointers for remapping.

### D2.3 `patients` table

* `patient_id`
* `has_pri`, `has_lim`, `n_pri_roi`, `n_lim_roi`, `has_pri_replicates`
* `slide_match` status per included Pri–LiM pair: `{yes,no,unknown}`

### D2.4 Mandatory audit fields

* `slide_match_rate`, `excluded_reason` (PDX/other sites)
* Expression preprocessing parameters (Section D3): `norm_target`, `hvg_n`, `pca_n`, `scale_max`, `regress_vars`
* Modality adaptation flags (Section D4): `delta_mode`, `p_mode`

---

## D3. Expression preprocessing and embedding (fixed main pipeline)

### D3.1 Preprocessing (standard, auditable)

* Library-size normalize: scale to `norm_target` (default 1e4)
* `log1p`
* HVG selection: `hvg_n` (default 2000)
* Scale: `scale_max` (default 10)
* PCA: compute `pca_n` PCs (default 50), retain first P for downstream (default P=50)

**Notes**

* No mandatory regression of mito% by default; record if applied (`regress_vars` audit).
* PCA fitting domain: **pooled Pri+LiM** (primary), with optional sensitivity Pri-only PCA.

### D3.2 Representation used by SLOTAR in ST

* (\mathbf{m}_c := \text{PC coordinates}\in\mathbb{R}^P)

---

## D4. Modality Adaptation: community features on ST (explicit contract alignment)

SLOTAR V1.6 defines community features as
(u_c=[\mathbf{p}_c,\bar{\mathbf{m}}_c,\mathbf{m}_c,\delta_c]).
For Visium, we instantiate a **modality-specific adaptor**:

### D4.1 Default (minimal; Tier 1)

* (\mathbf{m}_c): PCA vector of the spot
* (\bar{\mathbf{m}}_c): kNN mean PCA vector (k=20)
* (\delta_c := \text{const}) (record `delta_mode="const"`)
* (\mathbf{p}_c := \mathbf{0}) (record `p_mode="zero"`)

This preserves the API contract (field exists) while avoiding unsupported biological claims.

### D4.2 Optional sensitivities (Tier 2; not required)

* `delta_mode="spot_density"`: derive local spot density from tissue-spot kNN radius to capture missing-spot boundary effects.
* `p_mode="soft_comp"`: use deconvolution/module score as a soft composition proxy (explicitly labeled as proxy; not primary).
* Deconvolution-aware, purity-aware, or tumor-restricted representation variants are acceptable as task-scoped sensitivity analyses if they still terminate in the canonical SLOTAR-ready contract; they are not the default primary representation path.

---

## D5. Prototype dictionary and cost scaling

### D5.1 Dictionary training (cohort-level pooled)

* Train k-means on pooled (\tilde u_c) across all included ROIs (Pri+LiM), balanced sampling by patient×site.
* K=25 fixed.
* Compute global static cost scaling:
  [
  s_C=\mathrm{median}{|z_i-z_j|*2^2: i\neq j}
  ]
  and
  [
  C*{ij}=|z_i-z_j|_2^2/s_C
  ]

### D5.2 Assignment and ROI-level counts

* Assign each spot to a prototype `proto_id`.
* ROI prototype counts: (n_{p,site,roi,k})
* ROI density vector:
  [
  \rho_{roi,k}=\frac{n_{roi,k}}{area_{roi}}
  ]

---

## D6. Aggregation: density / shape / scale (area-weighted)

For each patient (p), site (t\in{\text{Pri},\text{LiM}}), group (g=\text{all}):

* Aggregate across ROIs (area-weighted):
  [
  a_{p,t,k} = \frac{\sum_{roi\in(p,t)} n_{roi,k}}{\sum_{roi\in(p,t)} area_{roi}}
  ]
* Scale:
  [
  S_{p,t}=\sum_k a_{p,t,k}
  ]
* Shape:
  [
  \bar a_{p,t}=a_{p,t}/S_{p,t} \quad(S_{p,t}>0)
  ]

Audit: `n_roi_{p,t}`, `n_spots_{p,t}`, `area_{p,t}`.

---

## D7. UOT inference and metrics (paired Pri→LiM)

### D7.1 Solver interface (contract)

* UOT solver receives **only (\lambda)** (plus (\varepsilon), cost (C), etc.).
* **(\alpha=0.15)** is a **calibration target** only (not a solver argument).

### D7.2 λ selection strategy (C4-compliant; primary = sensitivity grid)

Because Pri→LiM is cross-organ, we do **not** claim a unique biologically grounded λ. The primary analysis uses a **pre-registered λ sensitivity grid**:
[
\lambda \in {0.1, 0.3, 1, 3, 10}
]
For each λ, compute UOT on both density-level and shape-level:

* density: (a_{p,Pri}\to a_{p,LiM})
* shape: (\bar a_{p,Pri}\to \bar a_{p,LiM}) (if both scales >0)

We summarize stability of:

* unmatched ratio (U)
* remodeling intensity (M)
* retention (R) (with (\tau) fixed as in D7.3)
* top events (rank correlation across λ)

**Optional reference λ (audit only, not “truth”)**
If auxiliary Pri replicates (Pri2/3) are retained for sensitivity work, we may compute a within-patient Pri–Pri calibration curve and report a “reference λ” for audit (`calibration_mode="within_patient_pri_reference"`), without using it to make cross-organ biological claims or redefining the default `A/B` main cohort.

### D7.3 τ definition (retention labeling; fixed per λ or fixed globally)

* Retention threshold (\tau) is used for labeling only (does not affect (\Pi) solve).
* Primary: (\tau) defined as a fixed quantile of the (\Pi)-weighted cost distribution under each λ (e.g., (q=0.25)); report `tau_mode="pi_weighted_q25"`.
* Alternative (optional): fixed numeric τ in cost units; not required.

### D7.4 Metrics (V1.6-aligned)

Given transport plan (\Pi):

* transported mass (T=\sum_{ij}\Pi_{ij})
* positive-part destruction/creation:
  [
  D^+=\sum_i (a_i-(\Pi\mathbf{1})*i)*+, \quad
  B^+=\sum_j (b_j-(\Pi^\top\mathbf{1})*j)*+
  ]
* unmatched ratio:
  [
  U=\frac{B^+ + D^+}{T + B^+ + D^+}
  ]
* remodeling intensity:
  [
  M=\frac{\langle C,\Pi\rangle}{T+\epsilon}
  ]
* retention:
  [
  R=\frac{\sum_{C_{ij}\le\tau}\Pi_{ij}}{T+\epsilon}
  ]

---

## D8. Uncertainty quantification (UQ)

### D8.1 ROI bootstrap (when ≥2 ROIs per site)

For ((p,t)) with ≥2 ROIs:

* resample ROIs with replacement, recompute (a_{p,t}), (\bar a_{p,t}), and downstream UOT metrics across λ grid.
* output CI and event reproducibility.
* Whether this is estimable in practice depends on the final audited paired cohort and any explicitly enabled auxiliary-replicate sensitivity design.

### D8.2 Single-ROI (within-section) frozen block bootstrap (spot-block)

If ((p,t)) has only 1 ROI:

* partition tissue spots into an adaptive (G\times G) grid, filter invalid blocks, and resample blocks to create pseudo-ROIs.
* **freeze** PCA embedding and kNN neighborhoods (do not recompute neighbor features after resampling).
* output `UQ_mode="spot_block_frozen"`, `G_used`, `n_blocks_valid`.

---

## D9. Drift (risk flagging only; contract compliant)

* No domain classifier, no AUC metric.
* drift-aligned requires a drift vector (\Delta_{batch}) and cosine similarity in the same scaled subspace.
* For this Visium vignette, if slide_id confirms same-slide matching, drift risk is expected lower; however, if (\Delta_{batch}) cannot be reliably estimated, we set:

  * `drift_aligned = null`
  * `drift_mode="unavailable"`
    and only report slide-match audit fields.

No correction is performed.

---

## D10. Baselines (minimal set)

**Required**

1. Balanced OT (shape-only)
2. UOT without density/shape/scale decoupling (ablation)

**Recommended sanity baseline (very low cost)**
3) L1 / Jensen–Shannon distance between endpoint prototype frequency vectors (no OT)

All baselines share the same dictionary, cost scaling, and active set rules.

---

## D11. Outputs and deliverables

### D11.1 Per-patient outputs (for each λ)

* dens/shape: (U,M,R)
* (B^+,D^+,T)
* top events: edges and created/destroyed prototypes (with weights)
* UQ: CI, event reproducibility
* audits: `slide_match`, `n_roi`, `n_spots`, `area`, `UQ_mode`

### D11.2 Minimal figures (Box/Supplementary)

* λ-sensitivity heatmap: (U) and (M) across λ (paired Pri→LiM)
* example remapping: highlight top events on Visium spatial coordinates
* baseline comparison: Balanced OT vs UOT (forced matching vs unmatched absorption)

---

## D12. Failure modes and downgrade rules (pre-registered)

* If patient has no same-slide matched pair and slide_id is missing: keep but mark `slide_match="unknown"` and treat as descriptive.
* If UOT unstable for some λ: record and exclude that λ from stability summary; report solver diagnostics.
* If too few paired patients: report only descriptive outputs; no cohort-level testing.

---

## D13. Limitations (required)

* Visium spots are not single cells; interpretation is spot-level.
* Pri→LiM is cross-organ; λ is treated as a regularization sensitivity parameter, not a biologically calibrated “truth.”
* Drift detection may be unavailable; we do not correct for drift.
* g=all is used to minimize proxy assumptions; optional proxy grouping is labeled as such.

Adheres strictly to the structural zero bypass mechanism and explicit fit-status data contract defined in V1.6 for missing compartments or shape-level dropouts.
