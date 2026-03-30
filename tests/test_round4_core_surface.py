from __future__ import annotations

import numpy as np
import pandas as pd
import slotar
import slotar.io as slotar_io
import slotar.observation as observation_module
import slotar.uncertainty as uncertainty_module
from anndata import AnnData

from slotar.observation import (
    ObservationMatchConfig,
    build_observation_kernels,
    match_observation_clouds,
)
from slotar.representation import build_community_features, learn_global_prototypes
from slotar.state_space import build_local_state_features, learn_shared_state_axis
from slotar.uot import UOTSolveConfig, batched_uot_solve, precompute_logKernels


def _make_state_space_adata() -> AnnData:
    obs = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p1", "p1"],
            "timepoint": ["pre", "pre", "post", "post"],
            "roi_id": ["r1", "r1", "r2", "r2"],
            "compartment": ["c", "c", "c", "c"],
            "cell_type": ["A", "B", "A", "B"],
        }
    )
    adata = AnnData(X=np.zeros((4, 1), dtype=float), obs=obs)
    adata.obsm["spatial"] = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    adata.uns["roi_areas"] = {"r1": 1.0, "r2": 1.0}
    return adata


def test_canonical_state_space_keeps_alias_writes_out_of_live_surface() -> None:
    adata = _make_state_space_adata()

    features = build_local_state_features(adata, k=1)
    axis = learn_shared_state_axis(adata, n_bal=2, K=2, random_state=0)

    assert features.shape == (4, 2)
    assert axis.state_key == "state_id"
    assert "local_state_features" in adata.obsm
    assert "community_features" not in adata.obsm
    assert "state_id" in adata.obs.columns
    assert "proto_id" not in adata.obs.columns
    assert "state_centroids" in adata.uns
    assert "prototype_centroids" not in adata.uns
    assert "cost_scale" in adata.uns
    assert "s_C" not in adata.uns
    np.testing.assert_allclose(
        features,
        np.asarray(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    legacy_adata = _make_state_space_adata()
    build_community_features(legacy_adata, k=1)
    learn_global_prototypes(legacy_adata, n_bal=2, K=2, random_state=0)
    assert "community_features" in legacy_adata.obsm
    assert "proto_id" in legacy_adata.obs.columns


def test_canonical_observation_surface_matches_legacy_solver() -> None:
    A = np.asarray([[0.6, 0.4], [0.5, 0.5]], dtype=float)
    B = np.asarray([[0.55, 0.45], [0.4, 0.6]], dtype=float)
    penalty = np.asarray([5.0, 5.0], dtype=float)
    cost = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)

    cfg = ObservationMatchConfig(eps_schedule=[1.0, 0.2], max_iter=4000, tol=1e-8)
    kernels = build_observation_kernels(cost, cfg.eps_schedule)
    result = match_observation_clouds(
        state_mass_pre=A,
        state_mass_post=B,
        match_penalty=penalty,
        kernels=kernels,
        cfg=cfg,
        return_plan=True,
    )

    legacy_cfg = UOTSolveConfig(eps_schedule=[1.0, 0.2], max_iter=4000, tol=1e-8)
    legacy_kernels = precompute_logKernels(cost, legacy_cfg.eps_schedule)
    legacy_metrics, legacy_details, legacy_status = batched_uot_solve(
        A=A,
        B=B,
        lambda_pl=penalty,
        kernels=legacy_kernels,
        cfg=legacy_cfg,
        return_plan=True,
    )

    assert np.allclose(result.matched_mass, legacy_metrics["T"], equal_nan=True)
    assert np.allclose(result.source_unmatched_mass, legacy_metrics["D_pos"], equal_nan=True)
    assert np.allclose(result.target_unmatched_mass, legacy_metrics["B_pos"], equal_nan=True)
    assert np.array_equal(result.status, legacy_status)
    assert np.allclose(
        result.details["source_matching_mass_by_state"],
        legacy_details["source_matching_mass_by_state"],
        equal_nan=True,
    )
    assert result.matching_plan is not None
    assert "ObservationMatchConfig" in slotar.__all__
    assert "UOTSolveConfig" not in slotar.__all__
    assert not hasattr(slotar, "UOTSolveConfig")
    assert not hasattr(slotar, "save_for_r")


def test_canonical_submodule_all_lists_hide_compatibility_aliases() -> None:
    assert "write_r_handover" in slotar_io.__all__
    assert "save_for_r" not in slotar_io.__all__
    assert "DataContractError" not in slotar_io.__all__
    assert not hasattr(slotar_io, "save_for_r")
    assert not hasattr(slotar_io, "DataContractError")

    assert "STATUS_OK" not in observation_module.__all__
    assert not hasattr(observation_module, "STATUS_OK")

    assert "bootstrap_observation_unit" in uncertainty_module.__all__
    assert "estimate_log_measurement_error" in uncertainty_module.__all__
    assert "bootstrap_single_roi" not in uncertainty_module.__all__
    assert "compute_log_measurement_error" not in uncertainty_module.__all__
    assert not hasattr(uncertainty_module, "bootstrap_single_roi")
    assert not hasattr(uncertainty_module, "compute_log_measurement_error")
