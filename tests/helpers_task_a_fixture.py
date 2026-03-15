from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

K_FULL = 25
ARM2_ORDERED_PAIR_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("TC->IM", "TC-IM", "TC", "IM"),
    ("IM->TC", "TC-IM", "IM", "TC"),
    ("IM->PT", "IM-PT", "IM", "PT"),
    ("PT->IM", "IM-PT", "PT", "IM"),
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


def expected_roi_vectors(k_full: int = K_FULL) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    for _, _, roi_id, proto_counts in ROI_SPECS:
        vector = np.zeros(k_full, dtype=float)
        for proto_id, count in proto_counts.items():
            vector[proto_id] = float(count)
        vectors[roi_id] = vector
    return vectors


def expected_arm1_pair_records(
    *,
    n_draws: int,
    random_seed: int,
) -> list[dict[str, str]]:
    roi_groups: dict[tuple[str, str], list[str]] = {}
    for patient_id, compartment, roi_id, _ in ROI_SPECS:
        roi_groups.setdefault((patient_id, compartment), []).append(roi_id)

    rng = np.random.default_rng(random_seed)
    records: list[dict[str, str]] = []
    for draw_idx in range(1, n_draws + 1):
        draw_label = f"draw_{draw_idx:04d}"
        for slot_index, (patient_id, compartment) in enumerate(sorted(roi_groups), start=1):
            roi_ids = sorted(roi_groups[(patient_id, compartment)])
            roi_a, roi_b = rng.choice(roi_ids, size=2, replace=False).tolist()
            pair_id = f"A1_baseline::{draw_label}::{patient_id}::{compartment}::{roi_a}::{roi_b}"
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


def expected_broken_reference_pair_records(
    *,
    n_draws: int,
    random_seed: int,
) -> list[dict[str, str]]:
    roi_groups: dict[tuple[str, str], list[str]] = {}
    roi_pool: list[tuple[str, str, str]] = []
    for patient_id, compartment, roi_id, _ in ROI_SPECS:
        roi_groups.setdefault((patient_id, compartment), []).append(roi_id)
        roi_pool.append((patient_id, compartment, roi_id))
    roi_pool = sorted(roi_pool)

    anchored_rng = np.random.default_rng(random_seed)
    broken_rng = np.random.default_rng(random_seed + 1)
    records: list[dict[str, str]] = []
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
            choice_idx = int(broken_rng.integers(0, len(candidates)))
            patient_id_b, compartment_b, roi_b = candidates[choice_idx]
            pair_id = f"A1_broken_reference::{draw_label}::{patient_id_a}::{compartment_a}::{roi_a}::{roi_b}"
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


def expected_arm2_pair_records() -> list[dict[str, str]]:
    roi_groups: dict[tuple[str, str], list[str]] = {}
    for patient_id, compartment, roi_id, _ in ROI_SPECS:
        roi_groups.setdefault((patient_id, compartment), []).append(roi_id)

    records: list[dict[str, str]] = []
    for patient_id in sorted({patient_id for patient_id, _, _, _ in ROI_SPECS}):
        for pair_type, pair_family, source_compartment, target_compartment in ARM2_ORDERED_PAIR_SPECS:
            source_rois = sorted(roi_groups[(patient_id, source_compartment)])
            target_rois = sorted(roi_groups[(patient_id, target_compartment)])
            for roi_a in source_rois:
                for roi_b in target_rois:
                    pair_id = f"A2_cross_compartment::{pair_type}::{patient_id}::{roi_a}::{roi_b}"
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


def build_task_a_fixture(k_full: int = K_FULL) -> ad.AnnData:
    records: list[dict[str, object]] = []
    coords: list[tuple[float, float]] = []
    cell_index = 0
    for patient_id, compartment, roi_id, proto_counts in ROI_SPECS:
        for proto_id, count in proto_counts.items():
            for local_idx in range(count):
                records.append(
                    {
                        "patient_id": patient_id,
                        "timepoint": 0,
                        "roi_id": roi_id,
                        "compartment": compartment,
                        "proto_id": proto_id,
                    }
                )
                coords.append((float(cell_index), float(local_idx)))
                cell_index += 1

    obs = pd.DataFrame.from_records(records)
    obs.index = [f"cell_{idx}" for idx in range(len(obs))]
    adata = ad.AnnData(X=np.zeros((len(obs), 1), dtype=float), obs=obs)
    adata.obsm["spatial"] = np.asarray(coords, dtype=float)
    adata.uns["roi_areas"] = {roi_id: 1.0 for _, _, roi_id, _ in ROI_SPECS}
    axis = np.arange(k_full, dtype=float)
    adata.uns["cost_matrix"] = np.abs(axis[:, None] - axis[None, :])
    adata.uns["s_C"] = 1.0
    return adata


def write_task_a_fixture(path: str | Path, k_full: int = K_FULL) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    build_task_a_fixture(k_full=k_full).write_h5ad(out_path)
    return out_path
