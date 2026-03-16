"""
Module: tasks.task_A.arm3.retention

Phase 7 of the Arm-3 pipeline: continuous retention summary statistics.

Responsibilities:
- Compute per-patient continuous degradation statistics comparing reduced-coverage
  bootstrap estimates to frozen full-coverage reference estimates.
- Flag pseudo-ROI replicates where eta_floor numerical padding dominates meaningful
  mass (floor-dominated flag).

Design constraints (all locked):
- Outputs are continuous statistics only: absolute degradation, sign consistency
  rate, and floor-dominated replicate rate.
- No pass/fail thresholding inside the pipeline. No boolean retention flags.
  Threshold constants (D_MAX_m, PI_MIN_m, PHI_MAX) are applied externally.
- No separate full-reference tolerance-calibration resampling layer in Arm-3 v1.
  The 100% full-coverage reference remains outside the bootstrap loop.
- Both df_full_cov and df_reduced must already be filtered to anchor directions
  (TC->IM and TC->PT only) before calling compute_degradation_summary.
- Reverse directions and exploratory directions must not enter the primary
  degradation summary.

Zero-Sign Tie-Breaking Rule (locked):
- Let sign_100 = np.sign(m_100), where m_100 is the full-coverage reference
  value for a given patient/pair_type/quantity.
- Let sign_c = np.sign(median(m_reduced_replicates)).
- If sign_100 == 0: this patient is EXCLUDED from the sign consistency
  calculation (denominator drops by 1).
- If sign_100 != 0 and sign_c == 0: FAILURE to retain sign (numerator does
  not increment).
- sign_consistency_rate = (Count of exact sign matches) /
  (Count of patients with non-zero 100% reference sign).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_degradation_summary(
    df_full_cov: pd.DataFrame,
    df_reduced: pd.DataFrame,
    monitored_quantities: list[str],
    pair_id_col: str = "pair_id",
    patient_id_col: str = "patient_id",
) -> pd.DataFrame:
    """
    Compute per-patient continuous degradation statistics at one or more
    coverage levels against the frozen full-coverage reference baseline.

    For each patient p, pair_type, coverage c, and monitored quantity m:

        d(p, c, m) = |median_replicate(m, p, c) - m_full_cov(p)|

    Sign consistency rate pi_c(m) — Zero-Sign Tie-Breaking Rule (locked):
        - sign_100 = np.sign(m_100)
        - sign_c   = np.sign(median(m_reduced_replicates))
        - If sign_100 == 0: patient excluded from denominator and numerator.
        - If sign_100 != 0 and sign_c == 0: failure (numerator does not increment).
        - pi_c(m) = matches / patients_with_nonzero_reference_sign.
        This is a population-level statistic over (pair_type, coverage, quantity)
        and is broadcast to all patient rows in that group.

    Floor-dominated replicate rate phi_c:
        Mean of the boolean floor_dominated flag across all replicates for a
        given (patient_id, pair_type, coverage). Independent of quantity.

    Both df_full_cov and df_reduced MUST already be filtered to anchor directions
    (TC->IM and TC->PT only) before this function is called. This function does
    not perform direction filtering internally.

    Parameters
    ----------
    df_full_cov : pd.DataFrame
        Full-coverage reference results. One row per pair (anchor directions only).
        Must contain pair_id_col, patient_id_col, "pair_type", and all
        monitored_quantities.
    df_reduced : pd.DataFrame
        Bootstrap results at one or more coverage levels (anchor directions only).
        Must contain pair_id_col, patient_id_col, "pair_type", "replicate_id",
        "coverage", "floor_dominated", and all monitored_quantities.
    monitored_quantities : list[str]
        Minimum required: ['U_abs_dens', 'Q_src_dens'].
        May also include secondary quantities: ['Q_tgt_dens', 'T'].
    pair_id_col : str
        Column name for pair identity (validated for presence; not used for
        patient-level aggregation).
    patient_id_col : str
        Column name for patient identity.

    Returns
    -------
    pd.DataFrame with columns:
        patient_id, pair_type, coverage, quantity,
        median_abs_degradation, sign_consistency_rate, floor_dominated_rate,
        mean_replicate_value, std_replicate_value
    No pass/fail columns. No boolean retention flags. No threshold comparisons.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if "U_abs_dens" not in monitored_quantities or "Q_src_dens" not in monitored_quantities:
        raise ValueError(
            "compute_degradation_summary: monitored_quantities must include "
            "at least 'U_abs_dens' and 'Q_src_dens'"
        )

    required_full = {pair_id_col, patient_id_col, "pair_type"} | set(monitored_quantities)
    missing_full = required_full - set(df_full_cov.columns)
    if missing_full:
        raise ValueError(
            f"compute_degradation_summary: df_full_cov is missing columns: "
            f"{sorted(missing_full)}"
        )

    required_reduced = (
        {pair_id_col, patient_id_col, "pair_type", "replicate_id", "coverage", "floor_dominated"}
        | set(monitored_quantities)
    )
    missing_reduced = required_reduced - set(df_reduced.columns)
    if missing_reduced:
        raise ValueError(
            f"compute_degradation_summary: df_reduced is missing columns: "
            f"{sorted(missing_reduced)}"
        )

    # ------------------------------------------------------------------
    # Prepare string keys for safe grouping (avoid dtype mismatch)
    # ------------------------------------------------------------------
    df_full = df_full_cov.copy()
    df_boot = df_reduced.copy()
    df_full["_pid"] = df_full[patient_id_col].astype(str)
    df_full["_pt"] = df_full["pair_type"].astype(str)
    df_boot["_pid"] = df_boot[patient_id_col].astype(str)
    df_boot["_pt"] = df_boot["pair_type"].astype(str)

    records: list[dict] = []

    for pair_type in sorted(df_boot["_pt"].unique()):
        full_pt = df_full[df_full["_pt"] == pair_type]
        boot_pt = df_boot[df_boot["_pt"] == pair_type]

        for coverage in sorted(boot_pt["coverage"].unique()):
            boot_cov = boot_pt[boot_pt["coverage"] == coverage]

            # ------------------------------------------------------------------
            # Floor-dominated rate: per-(patient, coverage), independent of quantity.
            # Rule: mean of the boolean floor_dominated flag across all replicates.
            # ------------------------------------------------------------------
            floor_by_patient: dict[str, float] = {}
            for pid, grp in boot_cov.groupby("_pid", sort=False):
                floor_vals = grp["floor_dominated"].to_numpy(dtype=float)
                floor_by_patient[str(pid)] = (
                    float(np.mean(floor_vals)) if floor_vals.size > 0 else float("nan")
                )

            for quantity in monitored_quantities:
                # ----------------------------------------------------------
                # Per-patient intermediate statistics
                # ----------------------------------------------------------
                patient_stats: list[dict] = []

                for pid in sorted(boot_cov["_pid"].unique()):
                    # Full-coverage reference for this patient/pair_type
                    full_pat = full_pt[full_pt["_pid"] == pid]
                    if full_pat.empty:
                        continue
                    ref_vals = full_pat[quantity].dropna().to_numpy(dtype=float)
                    if ref_vals.size == 0:
                        continue
                    # Aggregate across multiple pairs of the same patient/pair_type
                    # via median (single pair is the common case; median is identity).
                    m_100 = float(np.median(ref_vals))

                    # Reduced-coverage replicate values for this patient
                    boot_pat = boot_cov[boot_cov["_pid"] == pid]
                    rep_vals = boot_pat[quantity].dropna().to_numpy(dtype=float)
                    if rep_vals.size == 0:
                        continue

                    median_rep = float(np.median(rep_vals))

                    # Absolute degradation: d(p, c, m) = |median_rep - m_100|
                    median_abs_deg = float(abs(median_rep - m_100))

                    # Descriptive replicate stats
                    mean_rep = float(np.mean(rep_vals))
                    std_rep = (
                        float(np.std(rep_vals, ddof=1))
                        if rep_vals.size > 1
                        else float("nan")
                    )

                    # Zero-Sign Tie-Breaking Rule (locked)
                    sign_100 = float(np.sign(m_100))
                    sign_c = float(np.sign(median_rep))

                    patient_stats.append(
                        {
                            "_pid": pid,
                            # Preserve the original (non-string) patient_id value
                            patient_id_col: full_pat.iloc[0][patient_id_col],
                            "median_abs_degradation": median_abs_deg,
                            "mean_replicate_value": mean_rep,
                            "std_replicate_value": std_rep,
                            "floor_dominated_rate": floor_by_patient.get(
                                pid, float("nan")
                            ),
                            "sign_100": sign_100,
                            "sign_c": sign_c,
                        }
                    )

                # ----------------------------------------------------------
                # Population-level sign consistency rate pi_c(m)
                #
                # Zero-Sign Tie-Breaking Rule (locked):
                #   - If sign_100 == 0: patient excluded from denominator.
                #   - If sign_100 != 0 and sign_c == 0: failure (no increment).
                #   - pi_c = n_match / n_denom  (nan if n_denom == 0)
                # ----------------------------------------------------------
                n_denom = sum(1 for r in patient_stats if r["sign_100"] != 0)
                n_match = sum(
                    1
                    for r in patient_stats
                    if r["sign_100"] != 0 and r["sign_c"] == r["sign_100"]
                )
                sign_consistency_rate = (
                    float(n_match / n_denom) if n_denom > 0 else float("nan")
                )

                # ----------------------------------------------------------
                # Emit one output row per patient for this group
                # ----------------------------------------------------------
                for r in patient_stats:
                    records.append(
                        {
                            patient_id_col: r[patient_id_col],
                            "pair_type": pair_type,
                            "coverage": float(coverage),
                            "quantity": quantity,
                            "median_abs_degradation": r["median_abs_degradation"],
                            "sign_consistency_rate": sign_consistency_rate,
                            "floor_dominated_rate": r["floor_dominated_rate"],
                            "mean_replicate_value": r["mean_replicate_value"],
                            "std_replicate_value": r["std_replicate_value"],
                        }
                    )

    if not records:
        return pd.DataFrame(
            columns=[
                patient_id_col,
                "pair_type",
                "coverage",
                "quantity",
                "median_abs_degradation",
                "sign_consistency_rate",
                "floor_dominated_rate",
                "mean_replicate_value",
                "std_replicate_value",
            ]
        )

    return pd.DataFrame.from_records(records).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Patch 2A — Contrast-based sign consistency for Phase 7
# ---------------------------------------------------------------------------


def compute_contrast_degradation_summary(
    df_full_cov: pd.DataFrame,
    df_reduced: pd.DataFrame,
    pair_id_col: str = "pair_id",
    patient_id_col: str = "patient_id",
) -> pd.DataFrame:
    """
    Compute contrast-based sign consistency for Phase 7 anchor summaries.

    Computes two signed patient-level contrasts that are biologically
    meaningful for the TC->IM / TC->PT ordered-anchor comparison:

        Delta_U_abs  = U_abs_dens(TC->PT) - U_abs_dens(TC->IM)
        Delta_Q_src  = Q_src_dens(TC->IM) - Q_src_dens(TC->PT)

    Sign consistency is evaluated on these contrasts rather than on the
    raw always-positive quantities U_abs_dens and Q_src_dens directly.

    ALIGNMENT SAFETY: contrasts are computed after grouping and pivoting to
    wide format on ['patient_id', 'coverage', 'replicate_id'], ensuring
    TC->PT and TC->IM values are strictly aligned in the SAME ROW before
    subtraction.  Patients missing either TC->IM or TC->PT at any
    (coverage, replicate_id) level are silently dropped for that group only.

    Zero-Sign Tie-Breaking Rule (locked — same as compute_degradation_summary):
        sign_100 = np.sign(reference_contrast)
        sign_c   = np.sign(median(replicate_contrasts))
        If sign_100 == 0: patient is non-evaluable (excluded from denominator).
        If sign_100 != 0 and sign_c == 0: failure (numerator does not increment).
        pi_c = matches / n_evaluable   (nan if n_evaluable == 0)

    Parameters
    ----------
    df_full_cov : pd.DataFrame
        Full-coverage reference results (anchor directions only).
        Required columns: patient_id_col, pair_type, U_abs_dens, Q_src_dens.
    df_reduced : pd.DataFrame
        Bootstrap results (anchor directions only).
        Required columns: patient_id_col, pair_type, replicate_id, coverage,
        U_abs_dens, Q_src_dens.
    pair_id_col : str
        Column validated for presence; not used for patient-level aggregation.
    patient_id_col : str
        Patient identity column.

    Returns
    -------
    pd.DataFrame with columns:
        patient_id, coverage, contrast_name,
        reference_contrast, median_replicate_contrast, abs_degradation,
        sign_consistency_rate, n_evaluable, n_zero_reference_sign
    """
    _TC_IM = "TC->IM"
    _TC_PT = "TC->PT"

    # --- Input validation ---
    required_full = {patient_id_col, "pair_type", "U_abs_dens", "Q_src_dens"}
    missing_full = required_full - set(df_full_cov.columns)
    if missing_full:
        raise ValueError(
            "compute_contrast_degradation_summary: df_full_cov missing columns: "
            f"{sorted(missing_full)}"
        )
    required_reduced = {
        patient_id_col, "pair_type", "replicate_id", "coverage",
        "U_abs_dens", "Q_src_dens",
    }
    missing_reduced = required_reduced - set(df_reduced.columns)
    if missing_reduced:
        raise ValueError(
            "compute_contrast_degradation_summary: df_reduced missing columns: "
            f"{sorted(missing_reduced)}"
        )

    # --- Contrast completeness check for full-coverage input ---
    full_types = set(df_full_cov["pair_type"].astype(str).unique())
    for req in (_TC_IM, _TC_PT):
        if req not in full_types:
            raise ValueError(
                "compute_contrast_degradation_summary: df_full_cov missing "
                f"pair_type '{req}' — cannot compute anchor contrasts"
            )

    _empty = pd.DataFrame(columns=[
        patient_id_col, "coverage", "contrast_name",
        "reference_contrast", "median_replicate_contrast", "abs_degradation",
        "sign_consistency_rate", "n_evaluable", "n_zero_reference_sign",
    ])

    # --- Full-coverage reference contrasts ---
    # Aggregate multiple ROI pairs per (patient, pair_type) via median first,
    # then pivot to wide, then subtract.
    df_full = df_full_cov.copy()
    df_full["_pid"] = df_full[patient_id_col].astype(str)
    df_full["_pt"] = df_full["pair_type"].astype(str)

    full_med = (
        df_full.groupby(["_pid", "_pt"])[["U_abs_dens", "Q_src_dens"]]
        .median()
        .reset_index()
    )
    full_im = (
        full_med[full_med["_pt"] == _TC_IM]
        .set_index("_pid")[["U_abs_dens", "Q_src_dens"]]
    )
    full_pt = (
        full_med[full_med["_pt"] == _TC_PT]
        .set_index("_pid")[["U_abs_dens", "Q_src_dens"]]
    )

    common_patients = sorted(full_im.index.intersection(full_pt.index))
    if not common_patients:
        return _empty

    ref_contrasts: dict[str, dict[str, float]] = {}
    for pid in common_patients:
        u_im = float(full_im.loc[pid, "U_abs_dens"])
        u_pt = float(full_pt.loc[pid, "U_abs_dens"])
        q_im = float(full_im.loc[pid, "Q_src_dens"])
        q_pt = float(full_pt.loc[pid, "Q_src_dens"])
        if not (np.isfinite(u_im) and np.isfinite(u_pt)):
            continue
        if not (np.isfinite(q_im) and np.isfinite(q_pt)):
            continue
        ref_contrasts[pid] = {
            "Delta_U_abs": u_pt - u_im,   # TC->PT minus TC->IM
            "Delta_Q_src": q_im - q_pt,   # TC->IM minus TC->PT
        }

    if not ref_contrasts:
        return _empty

    # --- Reduced-coverage replicate contrasts ---
    df_red = df_reduced.copy()
    df_red["_pid"] = df_red[patient_id_col].astype(str)
    df_red["_pt"] = df_red["pair_type"].astype(str)

    # Per-(patient, pair_type, coverage, replicate_id) median
    boot_med = (
        df_red.groupby(["_pid", "_pt", "coverage", "replicate_id"])[
            ["U_abs_dens", "Q_src_dens"]
        ]
        .median()
        .reset_index()
    )

    records: list[dict] = []

    for coverage in sorted(boot_med["coverage"].unique()):
        boot_cov = boot_med[boot_med["coverage"] == coverage]

        # Pivot to wide: TC->IM and TC->PT in the same (pid, replicate_id) row.
        im_reps = (
            boot_cov[boot_cov["_pt"] == _TC_IM]
            .set_index(["_pid", "replicate_id"])[["U_abs_dens", "Q_src_dens"]]
        )
        pt_reps = (
            boot_cov[boot_cov["_pt"] == _TC_PT]
            .set_index(["_pid", "replicate_id"])[["U_abs_dens", "Q_src_dens"]]
        )

        # Inner join on (pid, replicate_id) — drops any row missing either side.
        common_idx = im_reps.index.intersection(pt_reps.index)
        if len(common_idx) == 0:
            continue

        im_aligned = im_reps.loc[common_idx]
        pt_aligned = pt_reps.loc[common_idx]

        # Per-replicate contrasts (aligned by construction)
        delta_u = (
            pt_aligned["U_abs_dens"].to_numpy(dtype=float)
            - im_aligned["U_abs_dens"].to_numpy(dtype=float)
        )
        delta_q = (
            im_aligned["Q_src_dens"].to_numpy(dtype=float)
            - pt_aligned["Q_src_dens"].to_numpy(dtype=float)
        )

        rep_df = pd.DataFrame({
            "_pid": [idx[0] for idx in common_idx],
            "replicate_id": [idx[1] for idx in common_idx],
            "Delta_U_abs": delta_u,
            "Delta_Q_src": delta_q,
        })

        for contrast_name in ("Delta_U_abs", "Delta_Q_src"):
            patient_stats: list[dict] = []

            for pid in sorted(rep_df["_pid"].unique()):
                if pid not in ref_contrasts:
                    continue
                rep_vals = (
                    rep_df[rep_df["_pid"] == pid][contrast_name]
                    .dropna()
                    .to_numpy(dtype=float)
                )
                if rep_vals.size == 0:
                    continue

                ref_val = ref_contrasts[pid][contrast_name]
                median_rep = float(np.median(rep_vals))
                abs_deg = float(abs(median_rep - ref_val))

                # Locked sign rule
                sign_100 = float(np.sign(ref_val))
                sign_c = float(np.sign(median_rep))

                patient_stats.append({
                    "_pid": pid,
                    "reference_contrast": ref_val,
                    "median_replicate_contrast": median_rep,
                    "abs_degradation": abs_deg,
                    "sign_100": sign_100,
                    "sign_c": sign_c,
                })

            # Population-level sign-consistency with zero-sign tie-breaking (locked).
            n_evaluable = sum(1 for r in patient_stats if r["sign_100"] != 0)
            n_zero_ref = sum(1 for r in patient_stats if r["sign_100"] == 0)
            n_match = sum(
                1 for r in patient_stats
                if r["sign_100"] != 0 and r["sign_c"] == r["sign_100"]
            )
            sign_rate = (
                float(n_match / n_evaluable) if n_evaluable > 0 else float("nan")
            )

            # Self-check 4: Sign-consistency audit — report zero-reference cases
            _pi_str = f"{sign_rate:.3f}" if n_evaluable > 0 else "nan"
            print(
                f"[Phase 7 contrast] contrast={contrast_name}, coverage={coverage:.0%}: "
                f"n_total={len(patient_stats)}, "
                f"n_evaluable={n_evaluable}, "
                f"n_zero_reference_sign={n_zero_ref} (non-evaluable, excluded from pi_c), "
                f"n_sign_retained={n_match}, "
                f"sign_consistency_rate={_pi_str}"
            )

            for r in patient_stats:
                records.append({
                    patient_id_col: r["_pid"],
                    "coverage": float(coverage),
                    "contrast_name": contrast_name,
                    "reference_contrast": r["reference_contrast"],
                    "median_replicate_contrast": r["median_replicate_contrast"],
                    "abs_degradation": r["abs_degradation"],
                    "sign_consistency_rate": sign_rate,
                    "n_evaluable": n_evaluable,
                    "n_zero_reference_sign": n_zero_ref,
                })

    if not records:
        return _empty

    return pd.DataFrame.from_records(records).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Patch 2B — Prototype-level contrast table helper (Phase 8 prep)
# ---------------------------------------------------------------------------


def build_prototype_contrast_table(
    df_proto_events_full: pd.DataFrame,
    df_proto_events_bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build an intermediate prototype-level signed contrast table for Phase 8.

    Computes per-prototype unmatched-burden contrast:

        U_k = B_mass + D_mass   (per row and prototype)

        Delta_U_k = U_k(TC->PT) - U_k(TC->IM)

    This is the locked Phase 8 sign-consistency object.  Contrasts are
    computed after self-merging on ['patient_id', 'coverage', 'replicate_id',
    'prototype_k'] so TC->PT and TC->IM values are strictly aligned before
    subtraction.  Rows missing either side are dropped.

    Parameters
    ----------
    df_proto_events_full : pd.DataFrame
        Prototype event masses for full-coverage reference (Phase 6 output).
        Required columns: patient_id, pair_type, prototype_k, prototype_label,
        coverage, T_mass, B_mass, D_mass.
    df_proto_events_bootstrap : pd.DataFrame
        Prototype event masses for bootstrap replicates (Phase 6 output).
        Required columns: patient_id, pair_type, prototype_k, prototype_label,
        coverage, replicate_id, T_mass, B_mass, D_mass.

    Returns
    -------
    pd.DataFrame with columns:
        patient_id, coverage, replicate_id, prototype_k, prototype_label,
        U_k_TC_IM, U_k_TC_PT, Delta_U_k
    Phase 8 can then compute:
        - recurrence proportion based on Delta_U_k != 0
        - sign consistency based on sign(Delta_U_k) vs full-coverage reference
        - correlation of Delta_U_k to full-coverage reference
    """
    _TC_IM = "TC->IM"
    _TC_PT = "TC->PT"

    _empty = pd.DataFrame(columns=[
        "patient_id", "coverage", "replicate_id", "prototype_k",
        "prototype_label", "U_k_TC_IM", "U_k_TC_PT", "Delta_U_k",
    ])

    def _add_u_k(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["U_k"] = df["B_mass"] + df["D_mass"]
        return df

    records: list[dict] = []

    # --- Full-coverage prototype contrasts (coverage=1.0, replicate_id=None) ---
    df_full = _add_u_k(df_proto_events_full)
    df_full["_pid"] = df_full["patient_id"].astype(str)

    # Median per (patient, pair_type, prototype_k) for multi-pair aggregation
    full_agg = (
        df_full.groupby(["_pid", "pair_type", "prototype_k"])[["U_k"]]
        .median()
        .reset_index()
    )
    full_im = (
        full_agg[full_agg["pair_type"] == _TC_IM]
        .set_index(["_pid", "prototype_k"])["U_k"]
    )
    full_pt = (
        full_agg[full_agg["pair_type"] == _TC_PT]
        .set_index(["_pid", "prototype_k"])["U_k"]
    )
    common_full = full_im.index.intersection(full_pt.index)

    # Build proto_label lookup from full-coverage data
    label_lookup: dict[int, str] = {}
    for _, row in df_full[["prototype_k", "prototype_label"]].drop_duplicates().iterrows():
        label_lookup[int(row["prototype_k"])] = str(row["prototype_label"])

    for pid, proto_k in common_full:
        u_im = float(full_im.loc[(pid, proto_k)])
        u_pt = float(full_pt.loc[(pid, proto_k)])
        if not (np.isfinite(u_im) and np.isfinite(u_pt)):
            continue
        records.append({
            "patient_id": pid,
            "coverage": 1.0,
            "replicate_id": None,
            "prototype_k": int(proto_k),
            "prototype_label": label_lookup.get(int(proto_k), f"proto_{proto_k}"),
            "U_k_TC_IM": u_im,
            "U_k_TC_PT": u_pt,
            "Delta_U_k": u_pt - u_im,
        })

    # --- Bootstrap prototype contrasts ---
    df_boot = _add_u_k(df_proto_events_bootstrap)
    df_boot["_pid"] = df_boot["patient_id"].astype(str)

    for coverage in sorted(df_boot["coverage"].unique()):
        boot_cov = df_boot[df_boot["coverage"] == coverage]

        # Median per (patient, pair_type, coverage, replicate_id, prototype_k)
        boot_agg = (
            boot_cov.groupby(
                ["_pid", "pair_type", "coverage", "replicate_id", "prototype_k"]
            )[["U_k"]]
            .median()
            .reset_index()
        )

        im_reps = (
            boot_agg[boot_agg["pair_type"] == _TC_IM]
            .set_index(["_pid", "replicate_id", "prototype_k"])["U_k"]
        )
        pt_reps = (
            boot_agg[boot_agg["pair_type"] == _TC_PT]
            .set_index(["_pid", "replicate_id", "prototype_k"])["U_k"]
        )
        common_boot = im_reps.index.intersection(pt_reps.index)

        for pid, rep_id, proto_k in common_boot:
            u_im = float(im_reps.loc[(pid, rep_id, proto_k)])
            u_pt = float(pt_reps.loc[(pid, rep_id, proto_k)])
            if not (np.isfinite(u_im) and np.isfinite(u_pt)):
                continue
            records.append({
                "patient_id": pid,
                "coverage": float(coverage),
                "replicate_id": rep_id,
                "prototype_k": int(proto_k),
                "prototype_label": label_lookup.get(int(proto_k), f"proto_{proto_k}"),
                "U_k_TC_IM": u_im,
                "U_k_TC_PT": u_pt,
                "Delta_U_k": u_pt - u_im,
            })

    if not records:
        return _empty

    return pd.DataFrame.from_records(records).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Unimplemented skeleton (retained from original)
# ---------------------------------------------------------------------------


def compute_floor_dominated_flags(
    A_dens: np.ndarray,
    B_dens: np.ndarray,
    eta_floor: float,
    support_masks: np.ndarray,
) -> np.ndarray:
    """
    Flag pseudo-ROI replicates where eta_floor padding dominates meaningful mass.

    The exact criterion that maps the eta_floor mass fraction to a boolean
    per-replicate flag MUST be supplied as a task-fixed rule before this function
    is implemented. The rule must be recorded in Arm-3 run metadata before
    reduced-coverage interpretation begins.

    This function body will remain NotImplementedError until the floor-dominated
    criterion is confirmed and recorded in the Arm-3 audit metadata.

    Parameters
    ----------
    A_dens : np.ndarray, shape (N, K)
        Source-side density tensors (after eta_floor padding inside solver).
    B_dens : np.ndarray, shape (N, K)
        Target-side density tensors (after eta_floor padding inside solver).
    eta_floor : float
        Numerical floor from UOTSolveConfig.eta_floor. Used to assess what
        fraction of active-support mass originates from padding rather than
        genuine prototype counts.
    support_masks : np.ndarray, shape (N, K), dtype bool
        Frozen semantic support mask (K_r^100). Only prototypes within the
        frozen support are examined for floor dominance.

    Returns
    -------
    np.ndarray, shape (N,), dtype bool
        True where replicate is floor-dominated under the task-fixed criterion.
    """
    raise NotImplementedError(
        "Arm-3 skeleton only; floor-dominated criterion is an unresolved "
        "implementation fact — must be task-fixed before this function is implemented"
    )
