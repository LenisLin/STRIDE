"""Deterministic semisynthetic Task A benchmark helpers.

This module defines toy relation worlds and manifest builders for the frozen
semi-synthetic benchmark surface. It does not run the real-data Task A block
graph.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SemisyntheticWorld:
    pair_family: str
    source_marginals: np.ndarray
    target_marginals: np.ndarray
    relation_matrix: np.ndarray
    source_residual: np.ndarray
    target_residual: np.ndarray


BLOCK3_SEMISYNTHETIC_SCENARIOS: tuple[str, ...] = (
    "same_marginals_relation",
    "bounded_open_channel",
    "family_identifiability",
)


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    total = float(vector.sum())
    if total <= 0.0:
        raise ValueError("Semisynthetic worlds require positive total mass")
    return vector / total


def _normalize_world(world: SemisyntheticWorld) -> SemisyntheticWorld:
    relation = np.asarray(world.relation_matrix, dtype=float)
    source_residual = np.asarray(world.source_residual, dtype=float).reshape(-1)
    if relation.ndim != 2 or relation.shape[0] != relation.shape[1]:
        raise ValueError("Semisynthetic relation_matrix must be square")
    n_states = relation.shape[0]
    if source_residual.shape != (n_states,):
        raise ValueError("Semisynthetic relation_matrix must be square and align to source_marginals")
    target_residual = np.asarray(world.target_residual, dtype=float).reshape(-1)
    if target_residual.shape != (n_states,):
        raise ValueError("Semisynthetic target_residual must align to target_marginals")
    if not np.isfinite(relation).all() or (relation < 0.0).any():
        raise ValueError("Semisynthetic relation_matrix must be finite and non-negative")
    if not np.isfinite(source_residual).all() or (source_residual < 0.0).any():
        raise ValueError("Semisynthetic source_residual must be finite and non-negative")
    if not np.isfinite(target_residual).all() or (target_residual < 0.0).any():
        raise ValueError("Semisynthetic target_residual must be finite and non-negative")
    source = np.sum(relation, axis=1, dtype=float) + source_residual
    target = np.sum(relation, axis=0, dtype=float) + target_residual
    return SemisyntheticWorld(
        pair_family=world.pair_family,
        source_marginals=source,
        target_marginals=target,
        relation_matrix=relation,
        source_residual=source_residual,
        target_residual=target_residual,
    )


def continuity_score(relation_matrix: np.ndarray) -> float:
    matrix = np.asarray(relation_matrix, dtype=float)
    total = float(matrix.sum())
    if total <= 0.0:
        return 0.0
    diag = float(np.trace(matrix))
    return diag / total


def build_same_marginals_world_pair() -> tuple[SemisyntheticWorld, SemisyntheticWorld]:
    source = np.asarray([40.0, 30.0, 20.0, 10.0], dtype=float)
    stronger = np.asarray(
        [
            [30.0, 5.0, 3.0, 2.0],
            [4.0, 18.0, 6.0, 2.0],
            [1.0, 2.0, 15.0, 2.0],
            [0.0, 0.0, 1.0, 9.0],
        ],
        dtype=float,
    )
    weaker = np.asarray(
        [
            [10.0, 18.0, 8.0, 4.0],
            [20.0, 4.0, 4.0, 2.0],
            [5.0, 3.0, 8.0, 4.0],
            [0.0, 0.0, 5.0, 5.0],
        ],
        dtype=float,
    )
    return (
        _normalize_world(SemisyntheticWorld("TC-IM", source, source, stronger, np.zeros(4), np.zeros(4))),
        _normalize_world(SemisyntheticWorld("TC-PT", source, source, weaker, np.zeros(4), np.zeros(4))),
    )


def build_bounded_residual_world(
    *,
    pair_family: str,
    seed: int = 0,
) -> SemisyntheticWorld:
    rng = np.random.default_rng(seed)
    source = np.asarray([24.0, 20.0, 18.0, 14.0], dtype=float)
    target = np.asarray([20.0, 18.0, 22.0, 16.0], dtype=float)
    matched = np.asarray(
        [
            [12.0, 2.0, 1.0, 0.0],
            [1.0, 10.0, 2.0, 1.0],
            [0.0, 2.0, 11.0, 1.0],
            [0.0, 0.0, 1.0, 8.0],
        ],
        dtype=float,
    )
    source_residual = np.asarray([4.0, 3.0, 2.0, 1.0], dtype=float)
    target_residual = np.asarray([1.0, 1.0, 4.0, 4.0], dtype=float)
    jitter = rng.normal(loc=0.0, scale=0.05, size=matched.shape)
    relation = np.maximum(matched + jitter, 0.0)
    relation *= matched.sum() / max(relation.sum(), 1e-12)
    return _normalize_world(
        SemisyntheticWorld(
            pair_family=pair_family,
            source_marginals=source,
            target_marginals=target,
            relation_matrix=relation,
            source_residual=source_residual,
            target_residual=target_residual,
        )
    )


def build_bounded_open_channel_world_pair(
    *,
    seed: int = 0,
) -> tuple[SemisyntheticWorld, SemisyntheticWorld]:
    tc_im = build_bounded_residual_world(pair_family="TC-IM", seed=seed)
    tc_pt = _normalize_world(
        SemisyntheticWorld(
            pair_family="TC-PT",
            source_marginals=tc_im.source_marginals,
            target_marginals=tc_im.target_marginals,
            relation_matrix=0.8 * tc_im.relation_matrix,
            source_residual=np.asarray([0.08, 0.06, 0.04, 0.02], dtype=float),
            target_residual=np.asarray([0.02, 0.03, 0.10, 0.05], dtype=float),
        )
    )
    return tc_im, tc_pt


def build_family_identifiability_world_pair() -> tuple[SemisyntheticWorld, SemisyntheticWorld]:
    tc_im = _normalize_world(
        SemisyntheticWorld(
            pair_family="TC-IM",
            source_marginals=np.asarray([36.0, 28.0, 20.0, 16.0], dtype=float),
            target_marginals=np.asarray([32.0, 26.0, 22.0, 20.0], dtype=float),
            relation_matrix=np.asarray(
                [
                    [26.0, 4.0, 2.0, 0.0],
                    [5.0, 18.0, 3.0, 0.0],
                    [1.0, 2.0, 9.0, 2.0],
                    [0.0, 1.0, 2.0, 11.0],
                ],
                dtype=float,
            ),
            source_residual=np.asarray([2.0, 2.0, 1.0, 1.0], dtype=float),
            target_residual=np.asarray([1.0, 1.0, 2.0, 2.0], dtype=float),
        )
    )
    tc_pt = _normalize_world(
        SemisyntheticWorld(
            pair_family="TC-PT",
            source_marginals=np.asarray([36.0, 28.0, 20.0, 16.0], dtype=float),
            target_marginals=np.asarray([24.0, 20.0, 28.0, 28.0], dtype=float),
            relation_matrix=np.asarray(
                [
                    [10.0, 6.0, 10.0, 6.0],
                    [7.0, 5.0, 9.0, 5.0],
                    [2.0, 1.0, 7.0, 5.0],
                    [0.0, 0.0, 5.0, 7.0],
                ],
                dtype=float,
            ),
            source_residual=np.asarray([5.0, 4.0, 2.0, 1.0], dtype=float),
            target_residual=np.asarray([1.0, 1.0, 4.0, 6.0], dtype=float),
        )
    )
    return tc_im, tc_pt


def build_block3_semisynthetic_worlds(
    n_patients: int = 6,
    *,
    seed: int = 0,
) -> dict[str, dict[str, tuple[SemisyntheticWorld, SemisyntheticWorld]]]:
    scenarios: dict[str, dict[str, tuple[SemisyntheticWorld, SemisyntheticWorld]]] = {
        scenario_name: {}
        for scenario_name in BLOCK3_SEMISYNTHETIC_SCENARIOS
    }
    for patient_idx in range(n_patients):
        patient_id = f"P{patient_idx + 1:02d}"
        stronger, weaker = build_same_marginals_world_pair()
        scenarios["same_marginals_relation"][patient_id] = (
            stronger,
            _normalize_world(
                SemisyntheticWorld(
                    pair_family="TC-PT",
                    source_marginals=weaker.source_marginals,
                    target_marginals=weaker.target_marginals,
                    relation_matrix=np.maximum(
                        weaker.relation_matrix
                        + np.random.default_rng(seed + patient_idx).normal(
                            0.0,
                            0.002,
                            size=weaker.relation_matrix.shape,
                        ),
                        0.0,
                    ),
                    source_residual=weaker.source_residual,
                    target_residual=weaker.target_residual,
                )
            ),
        )
        scenarios["bounded_open_channel"][patient_id] = build_bounded_open_channel_world_pair(seed=seed + patient_idx)
        scenarios["family_identifiability"][patient_id] = build_family_identifiability_world_pair()
    return scenarios


def build_semisynthetic_manifest(n_patients: int = 6, *, seed: int = 0) -> pd.DataFrame:
    worlds: list[dict[str, object]] = []
    for patient_idx in range(n_patients):
        pair_family = "TC-IM" if patient_idx % 2 == 0 else "TC-PT"
        world = build_bounded_residual_world(pair_family=pair_family, seed=seed + patient_idx)
        worlds.append(
            {
                "patient_id": f"P{patient_idx + 1:02d}",
                "pair_family": world.pair_family,
                "continuity_score": continuity_score(world.relation_matrix),
                "source_residual_mass": float(world.source_residual.sum()),
                "target_residual_mass": float(world.target_residual.sum()),
            }
        )
    return pd.DataFrame.from_records(worlds)
