"""
Module: tasks.task_A.arm2.analysis_contract

Centralized non-compute contract surface for the post-hoc Arm-II focused
analysis rewrite.

This module owns:
- Locked startup-slice constants.
- Confirmatory versus audit-only family scope.
- Exact public focused-output filenames.
- Memo-boundary declarations.
- Shared dataclasses used across the rewrite modules.

This module does not own:
- Input loading.
- Solver reruns.
- Baseline or transport aggregation logic.
- Output writing.

Important internal-table note:
- `all_prototype_baseline_patient_family` is an internal all-prototype analysis
  table with row unit `(patient, family, prototype)`.
- It is a stable baseline-side summary-over-pairs layer between the raw
  `pair_prototype_baseline_long` table and downstream recurrence/view
  extraction.
- It summarizes raw ordered-pair x prototype values within each
  `(patient, family, prototype)` group and must not be interpreted as a
  sum-collapsed single-delta reconstruction.
- It is not a required public CSV output in the focused package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Locked Arm-II startup-slice constants
# ---------------------------------------------------------------------------

ARM_NAME = "A2_cross_compartment"

CONFIRMATORY_FAMILIES: tuple[str, ...] = ("TC-IM", "TC-PT")
AUDIT_ONLY_FAMILIES: tuple[str, ...] = ("IM-PT",)
PAIR_FAMILY_ORDER: tuple[str, ...] = ("TC-IM", "IM-PT", "TC-PT")
PAIR_TYPE_TO_FAMILY: dict[str, str] = {
    "TC->IM": "TC-IM",
    "IM->TC": "TC-IM",
    "IM->PT": "IM-PT",
    "PT->IM": "IM-PT",
    "TC->PT": "TC-PT",
    "PT->TC": "TC-PT",
}
PAIR_FAMILY_ROLE: dict[str, str] = {
    "TC-IM": "confirmatory",
    "TC-PT": "confirmatory",
    "IM-PT": "audit_only",
}
PAIR_TYPE_ORDER: tuple[str, ...] = (
    "TC->IM",
    "IM->TC",
    "IM->PT",
    "PT->IM",
    "TC->PT",
    "PT->TC",
)

PROTOTYPE_ANNOTATION_VALUE_COLUMNS: tuple[str, ...] = (
    "dominant_cell_type",
    "dominant_cell_type_fraction",
    "top_cell_type_mix",
    "total_cells",
    "top1_cell_type",
    "top1_fraction",
    "top2_cell_type",
    "top2_fraction",
    "top3_cell_type",
    "top3_fraction",
    "top12_fraction_sum",
    "prototype_label_top3",
)
PROTOTYPE_ANNOTATION_COLUMNS: tuple[str, ...] = (
    "proto_id",
    *PROTOTYPE_ANNOTATION_VALUE_COLUMNS,
)


# ---------------------------------------------------------------------------
# Public focused-output contract
# ---------------------------------------------------------------------------

FOCUSED_OUTPUT_FILENAMES: tuple[str, ...] = (
    "00_arm2_focused_results_memo.md",
    "01_prototype_biological_meaning_table.csv",
    "02_baseline_pair_audit.csv",
    "03_baseline_prototype_confirmatory_summary.csv",
    "04_baseline_patient_family_confirmatory_summary.csv",
    "05_global_transport_summary.csv",
    "06_uot_shared_transport_anchors.csv",
    "07_balanced_ot_forced_transport_prototypes.csv",
    "08_uot_unmatched_contributors.csv",
    "09_prototype_overlap_conflict_audit.csv",
    "10_prototype_family_specific_summary.csv",
    "11_prototype_patient_recurrence_summary.csv",
    "12_auxiliary_legacy_prototype_comparison.csv",
    "13_auxiliary_legacy_prototype_patient_recurrence.csv",
    "14_minimal_appendix_audit.csv",
)

# ---------------------------------------------------------------------------
# Memo-boundary declarations
# ---------------------------------------------------------------------------

MEMO_SUPPORTED_CLAIMS: tuple[str, ...] = (
    "Arm-II is interpreted as a biologically ordered benchmark ladder on the current startup slice.",
    "Baseline tissue differences are shown before transport and kept separate from transport claims.",
    "Confirmatory transport claims are restricted to TC-IM and TC-PT.",
    "Prototype-level interpretation is built from all active prototypes first.",
    "Focused prototype outputs are downstream views, not upstream analysis constraints.",
)

MEMO_NON_CLAIMS: tuple[str, ...] = (
    "No generic UOT superiority claim over Balanced OT.",
    "No claim that Arm-II is fully passed or closed.",
    "No claim that Balanced OT has unmatched semantics.",
    "No confirmatory claim based on IM-PT.",
    "No mechanistic or causal proof from prototype annotations.",
)

MEMO_OUT_OF_SCOPE_ITEMS: tuple[str, ...] = (
    "Stage-0 redesign or rebuilding.",
    "Solver redesign, lambda redesign, tau redesign, or R recovery.",
    "Arm-I, Arm-III, Arm-IV, uncertainty, drift, or bridge/export redesign.",
    "Broad exploratory reporting, plotting, or diffuse appendix generation.",
)


# ---------------------------------------------------------------------------
# Shared dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Arm2FocusedPaths:
    """Resolved file-system contract for one post-hoc Arm-II focused run."""

    arm2_metrics_parquet: Path
    stage0_h5ad: Path
    task_config: Path
    output_dir: Path
    prototype_view_ids: tuple[int, ...] | None


@dataclass(slots=True)
class Stage0AnalysisBundle:
    """Frozen Stage-0 analysis inputs needed by the focused Arm-II rewrite."""

    roi_vectors: dict[str, np.ndarray]
    roi_density_vectors: dict[str, np.ndarray]
    cost_matrix: np.ndarray
    cost_scale: float
    active_prototype_ids: np.ndarray
    patient_roi_audit_table: pd.DataFrame
    deviation_pattern_counts: pd.DataFrame
    prototype_cell_type_table: pd.DataFrame


@dataclass(slots=True)
class PairTensorBundle:
    """Ordered Arm-II pair tensors on the shared prototype axis."""

    A: np.ndarray
    B: np.ndarray
    A_density: np.ndarray
    B_density: np.ndarray
    k_full: int
    pair_metadata: pd.DataFrame


@dataclass(slots=True)
class LoadedArm2Inputs:
    """Fully loaded frozen inputs for the post-hoc Arm-II focused analysis."""

    paths: Arm2FocusedPaths
    task_config: dict[str, Any]
    metrics_df: pd.DataFrame
    stage0: Stage0AnalysisBundle
    pair_tensors: PairTensorBundle


@dataclass(slots=True)
class UotPlanBundle:
    """Raw UOT rerun products for the fixed Arm-II pair set."""

    edge_shares: np.ndarray
    transport_mass: np.ndarray
    status_columns: pd.DataFrame


@dataclass(slots=True)
class BalancedPlanBundle:
    """Raw same-pair Balanced OT rerun products for the fixed Arm-II pair set."""

    edge_shares: np.ndarray
    source_total_mass: np.ndarray
    target_total_mass: np.ndarray
    status_columns: pd.DataFrame


@dataclass(slots=True)
class PairPrototypeTransportSurface:
    """
    Pair-by-prototype transport surface for all active prototypes.

    The row axis must align to the ordered Arm-II pair table. The column axis
    must align to the active prototype IDs in `proto_ids`.
    """

    proto_ids: np.ndarray
    source_abs: np.ndarray
    source_share: np.ndarray
    target_abs: np.ndarray
    target_share: np.ndarray


@dataclass(slots=True)
class PairPrototypeUnmatchedSurface:
    """
    Pair-by-prototype unmatched surface for all active prototypes.

    This surface is UOT-only and must not be synthesized for Balanced OT.
    """

    proto_ids: np.ndarray
    destroy_abs: np.ndarray
    destroy_share: np.ndarray
    birth_abs: np.ndarray
    birth_share: np.ndarray


@dataclass(slots=True)
class ComputedArm2Surfaces:
    """
    One-time compute outputs reused by all downstream analysis modules.

    This bundle must expose enough pair-level and pair-by-prototype structure to
    support:
    - all-prototype transport summaries,
    - all-prototype unmatched summaries,
    - all-prototype recurrence,
    - downstream prototype extraction without rerunning solver internals.
    """

    pair_level_transport: pd.DataFrame
    uot_plan: UotPlanBundle
    balanced_plan: BalancedPlanBundle
    uot_transport_surface: PairPrototypeTransportSurface
    uot_unmatched_surface: PairPrototypeUnmatchedSurface
    balanced_transport_surface: PairPrototypeTransportSurface


@dataclass(slots=True)
class BaselineAnalysisTables:
    """
    Baseline-side all-prototype analysis tables.

    Relationship of the internal baseline tables:
    - `pair_prototype_baseline_long` is the raw pair-by-prototype baseline layer.
    - `all_prototype_baseline_patient_family` is the stable patient-by-family-by-
      prototype summary-over-pairs table derived from that raw layer.
    - `baseline_prototype_confirmatory` is a confirmatory collapsed prototype
      summary derived from the all-prototype baseline tables.
    - Recurrence and downstream extracted prototype views should consume
      `all_prototype_baseline_patient_family`, not re-aggregate the raw long
      table directly.
    """

    prototype_meaning: pd.DataFrame
    baseline_pair_audit: pd.DataFrame
    pair_prototype_baseline_long: pd.DataFrame
    all_prototype_baseline_patient_family: pd.DataFrame
    baseline_patient_family_confirmatory: pd.DataFrame
    baseline_prototype_confirmatory: pd.DataFrame
    baseline_validation: pd.DataFrame


@dataclass(slots=True)
class TransportAnalysisTables:
    """All-prototype transport and unmatched analysis tables."""

    global_transport_summary: pd.DataFrame
    all_prototype_uot_transport_patient_family: pd.DataFrame
    all_prototype_uot_unmatched_patient_family: pd.DataFrame
    all_prototype_balanced_transport_patient_family: pd.DataFrame
    all_prototype_ot_vs_uot_patient_family_delta: pd.DataFrame
    transport_validation: pd.DataFrame


@dataclass(slots=True)
class RecurrenceAnalysisTables:
    """
    All-prototype recurrence tables built before any downstream extraction.

    This bundle should remain reusable even if outputs `06` and `07` are later
    regenerated with a different selected-prototype list.
    """

    all_prototype_patient_recurrence: pd.DataFrame
    recurrence_validation: pd.DataFrame


@dataclass(slots=True)
class FocusedPrototypeViews:
    """
    Downstream extracted prototype views used for public outputs `06` and `07`.

    These are derived products only. They must never constrain upstream input,
    compute, baseline, transport, or recurrence layers.
    """

    prototype_comparison_view: pd.DataFrame
    prototype_recurrence_view: pd.DataFrame
    view_validation: pd.DataFrame


@dataclass(slots=True)
class FocusedOutputPackage:
    """
    Final write-ready focused-output package.

    `memo_text` stores the public memo content for output `00`.
    `tables_by_filename` stores the public CSV outputs. Together they define the
    exact focused-output package named in `FOCUSED_OUTPUT_FILENAMES`.
    """

    memo_text: str
    tables_by_filename: dict[str, pd.DataFrame]
    package_validation: pd.DataFrame
