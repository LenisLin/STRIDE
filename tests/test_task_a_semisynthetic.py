from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tasks.task_A.semisynthetic.benchmark import (
    BLOCK3_SEMISYNTHETIC_SCENARIOS,
    build_block3_semisynthetic_worlds,
    build_bounded_residual_world,
    build_family_identifiability_world_pair,
    build_same_marginals_world_pair,
    build_semisynthetic_manifest,
    continuity_score,
)


def test_same_marginals_worlds_have_different_relation_scores() -> None:
    stronger, weaker = build_same_marginals_world_pair()
    np.testing.assert_allclose(stronger.source_marginals, weaker.source_marginals)
    np.testing.assert_allclose(stronger.target_marginals, weaker.target_marginals)
    assert continuity_score(stronger.relation_matrix) > continuity_score(weaker.relation_matrix)


def test_bounded_residual_world_keeps_nonzero_residual_mass() -> None:
    world = build_bounded_residual_world(pair_family="TC-PT", seed=7)
    assert float(world.source_residual.sum()) > 0.0
    assert float(world.target_residual.sum()) > 0.0
    assert continuity_score(world.relation_matrix) > 0.0


def test_semisynthetic_manifest_covers_confirmatory_families() -> None:
    manifest = build_semisynthetic_manifest(n_patients=6, seed=1)
    assert set(manifest["pair_family"].astype(str)) == {"TC-IM", "TC-PT"}
    assert (manifest["source_residual_mass"] > 0.0).all()
    assert (manifest["target_residual_mass"] > 0.0).all()


def test_block3_semisynthetic_worlds_cover_all_frozen_scenarios() -> None:
    worlds = build_block3_semisynthetic_worlds(n_patients=3, seed=2)
    assert set(worlds) == set(BLOCK3_SEMISYNTHETIC_SCENARIOS)
    for scenario_worlds in worlds.values():
        assert set(scenario_worlds) == {"P01", "P02", "P03"}
        for tc_im_world, tc_pt_world in scenario_worlds.values():
            assert tc_im_world.pair_family == "TC-IM"
            assert tc_pt_world.pair_family == "TC-PT"
            np.testing.assert_allclose(
                tc_im_world.relation_matrix.sum(axis=1) + tc_im_world.source_residual,
                tc_im_world.source_marginals,
                atol=1e-8,
            )
            np.testing.assert_allclose(
                tc_pt_world.relation_matrix.sum(axis=0) + tc_pt_world.target_residual,
                tc_pt_world.target_marginals,
                atol=1e-8,
            )


def test_family_identifiability_world_pair_preserves_target_side_shift() -> None:
    tc_im, tc_pt = build_family_identifiability_world_pair()
    assert continuity_score(tc_im.relation_matrix) > continuity_score(tc_pt.relation_matrix)
    assert float(tc_pt.target_marginals[2:].sum()) > float(tc_im.target_marginals[2:].sum())
