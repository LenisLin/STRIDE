from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

K_FULL = 25
FIXTURE_VARIANT_DEFAULT = "default"
FIXTURE_VARIANT_BLOCK3_STRENGTHENED = "block3_strengthened"
FIXTURE_VARIANT_BLOCK3_PHASE3 = "block3_phase3"
BLOCK1_ORDERED_PAIR_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("TC->IM", "TC-IM", "TC", "IM"),
    ("IM->TC", "TC-IM", "IM", "TC"),
    ("TC->PT", "TC-PT", "TC", "PT"),
    ("PT->TC", "TC-PT", "PT", "TC"),
)
ROI_SPECS: tuple[tuple[str, str, str, dict[int, int]], ...] = (
    ("P01", "TC", "P01_TC_01", {0: 3, 1: 1}),
    ("P01", "TC", "P01_TC_02", {0: 1, 2: 3}),
    ("P01", "IM", "P01_IM_01", {3: 2, 4: 2}),
    ("P01", "IM", "P01_IM_02", {3: 1, 4: 3}),
    ("P01", "PT", "P01_PT_01", {5: 4}),
    ("P01", "PT", "P01_PT_02", {6: 4}),
    ("P02", "TC", "P02_TC_01", {0: 2, 2: 2}),
    ("P02", "TC", "P02_TC_02", {1: 2, 2: 2}),
    ("P02", "IM", "P02_IM_01", {4: 1, 5: 3}),
    ("P02", "IM", "P02_IM_02", {4: 2, 5: 2}),
    ("P02", "PT", "P02_PT_01", {6: 3, 7: 1}),
    ("P02", "PT", "P02_PT_02", {7: 4}),
)
BLOCK3_STRENGTHENED_EXTRA_ROI_SPECS: tuple[tuple[str, str, str, dict[int, int]], ...] = (
    ("P03", "TC", "P03_TC_01", {0: 3, 1: 1}),
    ("P03", "TC", "P03_TC_02", {0: 1, 2: 3}),
    ("P03", "IM", "P03_IM_01", {3: 2, 4: 2}),
    ("P03", "IM", "P03_IM_02", {3: 1, 4: 3}),
    ("P03", "PT", "P03_PT_01", {5: 4}),
    ("P03", "PT", "P03_PT_02", {6: 4}),
    ("P04", "TC", "P04_TC_01", {0: 2, 2: 2}),
    ("P04", "TC", "P04_TC_02", {1: 2, 2: 2}),
    ("P04", "IM", "P04_IM_01", {4: 1, 5: 3}),
    ("P04", "IM", "P04_IM_02", {4: 2, 5: 2}),
    ("P04", "PT", "P04_PT_01", {6: 3, 7: 1}),
    ("P04", "PT", "P04_PT_02", {7: 4}),
)


def _roi_specs_for_variant(
    variant: str,
) -> tuple[tuple[str, str, str, dict[int, int]], ...]:
    if variant == FIXTURE_VARIANT_DEFAULT:
        return ROI_SPECS
    if variant == FIXTURE_VARIANT_BLOCK3_STRENGTHENED:
        return ROI_SPECS + BLOCK3_STRENGTHENED_EXTRA_ROI_SPECS
    if variant == FIXTURE_VARIANT_BLOCK3_PHASE3:
        return _build_block3_phase3_roi_specs()
    raise ValueError(f"Unknown Task A fixture variant: {variant!r}")


def _build_block3_phase3_roi_specs(
    *,
    n_patients: int = 32,
    n_states: int = K_FULL,
) -> tuple[tuple[str, str, str, dict[int, int]], ...]:
    specs: list[tuple[str, str, str, dict[int, int]]] = []
    for patient_index in range(n_patients):
        patient_id = f"P{patient_index + 1:02d}"
        anchor = int(patient_index % n_states)

        def _state(offset: int) -> int:
            return int((anchor + offset) % n_states)

        roi_payloads = (
            ("TC", f"{patient_id}_TC_01", {_state(0): 4, _state(1): 2}),
            ("TC", f"{patient_id}_TC_02", {_state(1): 3, _state(2): 3}),
            ("IM", f"{patient_id}_IM_01", {_state(2): 2, _state(3): 3, _state(4): 1}),
            ("IM", f"{patient_id}_IM_02", {_state(3): 2, _state(4): 3, _state(5): 1}),
            ("PT", f"{patient_id}_PT_01", {_state(4): 3, _state(5): 2, _state(6): 1}),
            ("PT", f"{patient_id}_PT_02", {_state(5): 2, _state(6): 3, _state(7): 1}),
        )
        specs.extend((patient_id, domain, roi_id, counts) for domain, roi_id, counts in roi_payloads)
    return tuple(specs)


def expected_roi_vectors(k_full: int = K_FULL) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    for _, _, roi_id, proto_counts in ROI_SPECS:
        vector = np.zeros(k_full, dtype=float)
        for proto_id, count in proto_counts.items():
            vector[proto_id] = float(count)
        vectors[roi_id] = vector
    return vectors


def expected_block0_locality_pair_records(
    *,
    n_draws: int,
    random_seed: int,
) -> list[dict[str, object]]:
    roi_groups: dict[tuple[str, str], list[str]] = {}
    for patient_id, compartment, roi_id, _ in ROI_SPECS:
        roi_groups.setdefault((patient_id, compartment), []).append(roi_id)

    rng = np.random.default_rng(random_seed)
    records: list[dict[str, object]] = []
    for draw_idx in range(1, n_draws + 1):
        draw_label = f"draw_{draw_idx:04d}"
        for slot_index, (patient_id, compartment) in enumerate(sorted(roi_groups), start=1):
            roi_ids = sorted(roi_groups[(patient_id, compartment)])
            roi_a, roi_b = rng.choice(roi_ids, size=2, replace=False).tolist()
            pair_id = f"block0_locality_gate::{draw_label}::{patient_id}::{compartment}::{roi_a}::{roi_b}"
            records.append(
                {
                    "pair_id": pair_id,
                    "patient_group_id": pair_id,
                    "draw_number": draw_idx,
                    "draw_label": draw_label,
                    "slot_index": slot_index,
                    "patient_id": patient_id,
                    "compartment": compartment,
                    "patient_id_a": patient_id,
                    "patient_id_b": patient_id,
                    "compartment_a": compartment,
                    "compartment_b": compartment,
                    "same_patient": True,
                    "same_compartment": True,
                    "pair_type": f"{compartment}-{compartment}",
                    "roi_a": roi_a,
                    "roi_b": roi_b,
                }
            )
    return records


def expected_block0_broken_pair_records(
    *,
    n_draws: int,
    random_seed: int,
) -> list[dict[str, object]]:
    roi_groups: dict[tuple[str, str], list[str]] = {}
    roi_pool: list[tuple[str, str, str]] = []
    for patient_id, compartment, roi_id, _ in ROI_SPECS:
        roi_groups.setdefault((patient_id, compartment), []).append(roi_id)
        roi_pool.append((patient_id, compartment, roi_id))
    roi_pool = sorted(roi_pool)

    anchored_rng = np.random.default_rng(random_seed)
    broken_rng = np.random.default_rng(random_seed + 1)
    records: list[dict[str, object]] = []
    for draw_idx in range(1, n_draws + 1):
        draw_label = f"draw_{draw_idx:04d}"
        for slot_index, (patient_id_a, compartment_a) in enumerate(sorted(roi_groups), start=1):
            roi_ids = sorted(roi_groups[(patient_id_a, compartment_a)])
            roi_a, _ = anchored_rng.choice(roi_ids, size=2, replace=False).tolist()
            candidates = [
                (patient_id_b, compartment_b, roi_b)
                for patient_id_b, compartment_b, roi_b in roi_pool
                if roi_b != roi_a and patient_id_b != patient_id_a and compartment_b != compartment_a
            ]
            patient_id_b, compartment_b, roi_b = candidates[int(broken_rng.integers(0, len(candidates)))]
            pair_id = f"block0_locality_gate::{draw_label}::{patient_id_a}::{compartment_a}::{roi_a}::{roi_b}"
            records.append(
                {
                    "pair_id": pair_id,
                    "patient_group_id": pair_id,
                    "draw_number": draw_idx,
                    "draw_label": draw_label,
                    "slot_index": slot_index,
                    "patient_id": patient_id_a,
                    "compartment": compartment_a,
                    "patient_id_a": patient_id_a,
                    "patient_id_b": patient_id_b,
                    "compartment_a": compartment_a,
                    "compartment_b": compartment_b,
                    "same_patient": False,
                    "same_compartment": False,
                    "pair_type": f"{compartment_a}-{compartment_b}",
                    "roi_a": roi_a,
                    "roi_b": roi_b,
                }
            )
    return records


def expected_block1_pair_records() -> list[dict[str, object]]:
    roi_groups: dict[tuple[str, str], list[str]] = {}
    for patient_id, compartment, roi_id, _ in ROI_SPECS:
        roi_groups.setdefault((patient_id, compartment), []).append(roi_id)

    records: list[dict[str, object]] = []
    for patient_id in sorted({patient_id for patient_id, _, _, _ in ROI_SPECS}):
        for pair_type, pair_family, source_compartment, target_compartment in BLOCK1_ORDERED_PAIR_SPECS:
            source_rois = sorted(roi_groups[(patient_id, source_compartment)])
            target_rois = sorted(roi_groups[(patient_id, target_compartment)])
            for roi_a in source_rois:
                for roi_b in target_rois:
                    pair_id = f"block1_continuity_backbone::{pair_type}::{patient_id}::{roi_a}::{roi_b}"
                    records.append(
                        {
                            "pair_id": pair_id,
                            "patient_group_id": pair_id,
                            "patient_id": patient_id,
                            "compartment": source_compartment,
                            "patient_id_a": patient_id,
                            "patient_id_b": patient_id,
                            "compartment_a": source_compartment,
                            "compartment_b": target_compartment,
                            "same_patient": True,
                            "same_compartment": False,
                            "pair_type": pair_type,
                            "pair_family": pair_family,
                            "roi_a": roi_a,
                            "roi_b": roi_b,
                        }
                    )
    return records


def build_task_a_fixture(
    k_full: int = K_FULL,
    *,
    variant: str = FIXTURE_VARIANT_DEFAULT,
) -> ad.AnnData:
    roi_specs = _roi_specs_for_variant(variant)
    records: list[dict[str, object]] = []
    coords: list[tuple[float, float]] = []
    cell_index = 0
    for patient_id, compartment, roi_id, proto_counts in roi_specs:
        for proto_id, count in proto_counts.items():
            for local_idx in range(count):
                records.append(
                    {
                        "patient_id": patient_id,
                        "timepoint": 0,
                        "roi_id": roi_id,
                        "compartment": compartment,
                        "cell_type": f"{compartment}_cell",
                        "proto_id": proto_id,
                    }
                )
                coords.append((float(cell_index), float(local_idx)))
                cell_index += 1

    obs = pd.DataFrame.from_records(records)
    obs.index = [f"cell_{idx}" for idx in range(len(obs))]
    adata = ad.AnnData(X=np.zeros((len(obs), 1), dtype=float), obs=obs)
    adata.obsm["spatial"] = np.asarray(coords, dtype=float)
    # community_features mirrors the real Stage-0 layout (n_cells x k_full)
    adata.obsm["community_features"] = np.zeros((len(obs), k_full), dtype=float)
    adata.uns["roi_areas"] = {roi_id: 1.0 for _, _, roi_id, _ in roi_specs}
    axis = np.arange(k_full, dtype=float)
    adata.uns["cost_matrix"] = np.abs(axis[:, None] - axis[None, :])
    adata.uns["s_C"] = 1.0
    # prototype_centroids: 2D spatial centroids for each prototype (k_full x 2)
    rng = np.random.default_rng(0)
    adata.uns["prototype_centroids"] = rng.uniform(0.0, 100.0, size=(k_full, 2))
    return adata


def write_task_a_fixture(
    path: str | Path,
    k_full: int = K_FULL,
    *,
    variant: str = FIXTURE_VARIANT_DEFAULT,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    build_task_a_fixture(k_full=k_full, variant=variant).write_h5ad(out_path)
    return out_path


def write_block3_phase3_task_a_fixture(
    path: str | Path,
    k_full: int = K_FULL,
) -> Path:
    return write_task_a_fixture(
        path,
        k_full=k_full,
        variant=FIXTURE_VARIANT_BLOCK3_PHASE3,
    )


def write_passed_block0_bundle(
    path: str | Path,
    *,
    config_path: str | Path,
    data_path: str | Path,
) -> Path:
    from tasks.task_A.config import load_task_a_config_bundle
    from tasks.task_A.block0.bundle import PAIR_METRICS_FILENAME
    from tasks.task_A.block0.locality_gate import NULL_FAMILIES, REAL_FAMILIES

    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    config_bundle = load_task_a_config_bundle(config_path)
    pair_metrics_path = out_path.parent / PAIR_METRICS_FILENAME
    pd.DataFrame(
        [
            {
                "comparison_id": "fixture-passed-block0",
                "run_scope": "full_cohort",
                "pair_family": "TC-IM",
                "null_family": "TC-IM_randomized_target",
                "anchor_patient_id": "P01",
                "null_target_donor_patient_id": "P02",
                "source_domain": "TC",
                "target_domain": "IM",
                "n_source_observations": 2,
                "n_target_observations": 2,
                "count_stratum_key": "TC:2|IM:2",
                "selection_seed": 0,
                "null_assignment_status": "assigned",
                "null_assignment_reason": "",
                "real_fit_status": "ok",
                "null_fit_status": "ok",
                "real_defer_reason": "",
                "null_defer_reason": "",
                "real_total_continuity_mass": 19.0,
                "null_total_continuity_mass": 17.5,
                "delta_total_continuity_mass": 1.5,
                "real_total_depletion_mass": 6.0,
                "null_total_depletion_mass": 7.5,
                "delta_total_depletion_mass": -1.5,
                "real_total_emergence_mass": 0.2,
                "null_total_emergence_mass": 0.5,
                "delta_total_emergence_mass": -0.3,
            }
        ]
    ).to_csv(pair_metrics_path, index=False)
    payload = {
        "block": "block0_locality_gate",
        "status": "passed",
        "artifact_state": "contract_passed",
        "implementation_tier": "canonical_full",
        "evidence_lineage": "canonical_rerun",
        "run_scope": "full_cohort",
        "block0_passed": True,
        "config_fingerprint": config_bundle.config_fingerprint,
        "config_path": str(config_bundle.config_path),
        "stage0_h5ad": str(Path(data_path).expanduser().resolve()),
        "output_dir": str(out_path.parent),
        "bundle_path": str(out_path),
        "pair_metrics_path": str(pair_metrics_path),
        "real_families": list(REAL_FAMILIES),
        "null_families": list(NULL_FAMILIES),
        "pre_block0_data_suitability": {
            "artifact_state": "contract_passed",
            "report_scope": "pre_block0_data_suitability",
            "run_scope": "full_cohort_alignment_check",
        },
        "gate_checks": {
            "paired_support": {"passed": True, "observed": 1, "threshold": 1},
        },
        "metrics_summary": {
            "eligible_patients": 1,
            "required_support": 1,
            "gate_summary_quantities": [
                "delta_total_continuity_mass",
                "delta_total_emergence_mass",
            ],
        },
        "failure_reasons": [],
        "inputs": {
            "task_config": str(config_bundle.config_path),
            "stage0_h5ad": str(Path(data_path).expanduser().resolve()),
            "run_scope": "full_cohort",
            "random_seed": 0,
            "real_family_definition": {
                "pair_family": "TC-IM",
                "source_domain": "TC",
                "target_domain": "IM",
                "construction": "task_a_stride_adapter_family_slice",
                "fit_surface": "fit_stride",
            },
            "null_family_definition": {
                "pair_family": "TC-IM_randomized_target",
                "source_domain": "TC",
                "target_domain": "IM",
                "construction": (
                    "same_anchor_source_with_target_group_reassigned_from_different_patient_"
                    "in_same_exact_count_stratum"
                ),
                "stratification_fields": [
                    "n_source_observations",
                    "n_target_observations",
                ],
                "donor_policy": "seeded_derangement_within_exact_count_strata",
                "singleton_stratum_policy": "emit_null_fit_status_deferred_for_anchor_patient",
            },
            "gate_summary_quantities": {
                "delta_total_continuity_mass": {
                    "definition": "sum(A_real) - sum(A_null)",
                    "decision_rule": "median > 0 and fraction_real_total_continuity_mass_gt_null > 0.5",
                },
                "delta_total_emergence_mass": {
                    "definition": "sum(e_real) - sum(e_null)",
                    "decision_rule": "median < 0 and fraction_real_total_emergence_mass_lt_null > 0.5",
                },
            },
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path
