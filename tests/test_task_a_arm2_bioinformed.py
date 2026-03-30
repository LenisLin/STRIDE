from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tasks.task_A.arm2.analysis_bioinformed import (
    PANEL_IMMUNE_STROMAL,
    PANEL_MIXED_INTERFACE,
    PANEL_TC_DOMINANT,
    build_directional_unmatched_patient_proto_table,
    build_panel_assignment_table,
    summarize_directional_unmatched_by_proto,
)


def test_build_panel_assignment_table_applies_default_thresholds() -> None:
    prototype_meaning = pd.DataFrame(
        [
            {
                "proto_id": 0,
                "dominant_cell_type": "TC_EpCAM",
                "dominant_cell_type_fraction": 0.6,
                "top_cell_type_mix": "TC_EpCAM:0.60;TC_Ki67:0.20;TC_VEGF:0.10",
                "total_cells": 100,
                "top1_cell_type": "TC_EpCAM",
                "top1_fraction": 0.55,
                "top2_cell_type": "TC_Ki67",
                "top2_fraction": 0.20,
                "top3_cell_type": "TC_VEGF",
                "top3_fraction": 0.10,
                "top12_fraction_sum": 0.75,
                "prototype_label_top3": "p0 | TC_EpCAM(0.55) / TC_Ki67(0.20) / TC_VEGF(0.10)",
            },
            {
                "proto_id": 1,
                "dominant_cell_type": "TC_EpCAM",
                "dominant_cell_type_fraction": 0.4,
                "top_cell_type_mix": "TC_EpCAM:0.40;TC_Ki67:0.25;B:0.10",
                "total_cells": 90,
                "top1_cell_type": "TC_EpCAM",
                "top1_fraction": 0.38,
                "top2_cell_type": "TC_Ki67",
                "top2_fraction": 0.25,
                "top3_cell_type": "B",
                "top3_fraction": 0.10,
                "top12_fraction_sum": 0.63,
                "prototype_label_top3": "p1 | TC_EpCAM(0.38) / TC_Ki67(0.25) / B(0.10)",
            },
            {
                "proto_id": 2,
                "dominant_cell_type": "SC_COLLAGEN",
                "dominant_cell_type_fraction": 0.45,
                "top_cell_type_mix": "SC_COLLAGEN:0.45;SC_aSMA:0.12;CD8T:0.08",
                "total_cells": 80,
                "top1_cell_type": "SC_COLLAGEN",
                "top1_fraction": 0.41,
                "top2_cell_type": "SC_aSMA",
                "top2_fraction": 0.12,
                "top3_cell_type": "CD8T",
                "top3_fraction": 0.08,
                "top12_fraction_sum": 0.53,
                "prototype_label_top3": "p2 | SC_COLLAGEN(0.41) / SC_aSMA(0.12) / CD8T(0.08)",
            },
            {
                "proto_id": 3,
                "dominant_cell_type": "Mono_CD11c",
                "dominant_cell_type_fraction": 0.2,
                "top_cell_type_mix": "Mono_CD11c:0.20;Macro_CD163:0.15;B:0.11",
                "total_cells": 70,
                "top1_cell_type": "Mono_CD11c",
                "top1_fraction": 0.20,
                "top2_cell_type": "Macro_CD163",
                "top2_fraction": 0.15,
                "top3_cell_type": "B",
                "top3_fraction": 0.11,
                "top12_fraction_sum": 0.35,
                "prototype_label_top3": "p3 | Mono_CD11c(0.20) / Macro_CD163(0.15) / B(0.11)",
            },
        ]
    )

    panel = build_panel_assignment_table(prototype_meaning).set_index("proto_id")

    assert panel.loc[0, "panel_name"] == PANEL_TC_DOMINANT
    assert panel.loc[1, "panel_name"] == PANEL_MIXED_INTERFACE
    assert bool(panel.loc[1, "is_borderline_tc_like"])
    assert panel.loc[2, "panel_name"] == PANEL_IMMUNE_STROMAL
    assert panel.loc[3, "panel_name"] == PANEL_MIXED_INTERFACE


def test_directional_unmatched_summary_preserves_direction_roles() -> None:
    prototype_meaning = pd.DataFrame(
        [
            {
                "proto_id": 0,
                "dominant_cell_type": "TC_EpCAM",
                "dominant_cell_type_fraction": 0.6,
                "top_cell_type_mix": "TC_EpCAM:0.60;TC_Ki67:0.20;TC_VEGF:0.10",
                "total_cells": 100,
                "top1_cell_type": "TC_EpCAM",
                "top1_fraction": 0.55,
                "top2_cell_type": "TC_Ki67",
                "top2_fraction": 0.20,
                "top3_cell_type": "TC_VEGF",
                "top3_fraction": 0.10,
                "top12_fraction_sum": 0.75,
                "prototype_label_top3": "p0 | TC_EpCAM(0.55) / TC_Ki67(0.20) / TC_VEGF(0.10)",
            },
            {
                "proto_id": 1,
                "dominant_cell_type": "Mono_CD11c",
                "dominant_cell_type_fraction": 0.2,
                "top_cell_type_mix": "Mono_CD11c:0.20;Macro_CD163:0.15;B:0.11",
                "total_cells": 70,
                "top1_cell_type": "Mono_CD11c",
                "top1_fraction": 0.20,
                "top2_cell_type": "Macro_CD163",
                "top2_fraction": 0.15,
                "top3_cell_type": "B",
                "top3_fraction": 0.11,
                "top12_fraction_sum": 0.35,
                "prototype_label_top3": "p1 | Mono_CD11c(0.20) / Macro_CD163(0.15) / B(0.11)",
            },
        ]
    )
    panel_assignment = build_panel_assignment_table(prototype_meaning)
    pair_level_transport = pd.DataFrame(
        {
            "patient_id": ["P1", "P1", "P2"],
            "ordered_direction": ["TC->PT", "PT->TC", "TC->IM"],
            "pair_family": ["TC-PT", "TC-PT", "TC-IM"],
        }
    )
    destroy_abs_surface = np.array(
        [
            [3.0, 1.0],
            [1.0, 2.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )
    birth_abs_surface = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )

    patient_proto = build_directional_unmatched_patient_proto_table(
        pair_level_transport,
        destroy_abs_surface=destroy_abs_surface,
        birth_abs_surface=birth_abs_surface,
        prototype_meaning=prototype_meaning,
    )
    summary = summarize_directional_unmatched_by_proto(patient_proto, panel_assignment)

    tc_pt_proto0 = summary.loc[
        (summary["pair_type"].astype(str) == "TC->PT")
        & (summary["proto_id"].astype(int) == 0)
    ].iloc[0]
    assert tc_pt_proto0["direction_role"] == "primary_anchor"
    assert np.isclose(tc_pt_proto0["source_depletion_prone_share"], 0.75)
    assert np.isclose(tc_pt_proto0["target_emergence_prone_share"], 1.0 / 3.0)
    assert np.isclose(tc_pt_proto0["depletion_minus_emergence"], 0.75 - (1.0 / 3.0))

    pt_tc_proto1 = summary.loc[
        (summary["pair_type"].astype(str) == "PT->TC")
        & (summary["proto_id"].astype(int) == 1)
    ].iloc[0]
    assert pt_tc_proto1["direction_role"] == "audit_only"
    assert np.isclose(pt_tc_proto1["source_depletion_gt_emergence_patient_prop"], 1.0)
