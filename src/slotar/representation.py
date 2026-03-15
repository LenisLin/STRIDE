"""
Module: src.slotar.representation
Architecture: Library Level (Prerequisite Inputs)
Constraints:
- STRICTLY NO references to `tasks`, `config.yaml`, or clinical metadata.
- These are upstream utilities. They must mutate the AnnData object in-place to comply 
  with SLOTAR data contracts.
"""
from __future__ import annotations

import numpy as np
from anndata import AnnData
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def build_community_features(adata: AnnData, k: int = 30) -> None:
    """
    Computes spatial community features based on exact kNN and local density.

    Args:
        adata: AnnData object containing `.obsm['spatial']`.
        k: Number of nearest neighbors.

    Side Effects:
        - Stores robustly scaled vectors in `adata.obsm['community_features']`.
        - Stores scaling parameters in `adata.uns['scaler_params']`.

    Constraints for Codex:
        1. FAIL-FAST: Raise ValueError if NaNs are found in `adata.obsm['spatial']`.
        2. DENSITY & EXACT KNN: You MUST calculate the local density delta_c = k / (pi * r_k^2) 
           using exact kNN distances, and include it in the feature matrix before scaling.
        3. SCALER PARAMS: You MUST compute robust scaling (e.g., median/IQR) and save the 
           parameters to `adata.uns['scaler_params']` for auditability.
    """
    if "cell_type" not in adata.obs.columns:
        raise ValueError("adata.obs must contain 'cell_type' for representation construction")
    if "roi_id" not in adata.obs.columns:
        raise ValueError("adata.obs must contain 'roi_id' for within-ROI representation construction")

    spatial = np.asarray(adata.obsm["spatial"], dtype=float)
    if spatial.ndim != 2 or spatial.shape[1] != 2:
        raise ValueError("adata.obsm['spatial'] must have shape [n_cells, 2]")
    if not np.isfinite(spatial).all():
        raise ValueError("adata.obsm['spatial'] contains NaN/Inf")
    if k <= 0:
        raise ValueError("k must be strictly positive")

    cell_types = adata.obs["cell_type"].astype(str).to_numpy()
    roi_ids = adata.obs["roi_id"].astype(str).to_numpy()
    type_names = np.unique(cell_types)
    type_to_idx = {cell_type: idx for idx, cell_type in enumerate(type_names.tolist())}
    type_index = np.asarray([type_to_idx[cell_type] for cell_type in cell_types], dtype=int)

    features = np.zeros((spatial.shape[0], len(type_names) + 1), dtype=np.float32)
    for roi_id in np.unique(roi_ids):
        roi_mask = roi_ids == roi_id
        roi_indices = np.flatnonzero(roi_mask)
        roi_coords = spatial[roi_indices]
        if roi_indices.size < 2:
            raise ValueError(f"ROI {roi_id!r} must contain at least 2 cells for kNN features")

        effective_k = min(k, roi_indices.size - 1)
        model = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
        model.fit(roi_coords)
        distances, neighbors = model.kneighbors(roi_coords, return_distance=True)
        distances = distances[:, 1:]
        neighbors = neighbors[:, 1:]

        radii = distances[:, -1]
        if np.any(radii <= 0.0):
            raise ValueError(f"ROI {roi_id!r} has duplicate coordinates at the kNN radius boundary")

        neighbor_types = type_index[roi_indices][neighbors]
        composition = np.zeros((roi_indices.size, len(type_names)), dtype=np.float32)
        row_index = np.repeat(np.arange(roi_indices.size), effective_k)
        np.add.at(composition, (row_index, neighbor_types.reshape(-1)), 1.0)
        composition /= float(effective_k)

        density = (effective_k / (np.pi * np.square(radii))).astype(np.float32, copy=False)
        features[roi_indices] = np.column_stack([composition, density]).astype(np.float32, copy=False)

    center = np.median(features, axis=0).astype(np.float32, copy=False)
    q75 = np.percentile(features, 75, axis=0).astype(np.float32, copy=False)
    q25 = np.percentile(features, 25, axis=0).astype(np.float32, copy=False)
    scale = q75 - q25
    scale[scale <= 0.0] = 1.0

    adata.obsm["community_features"] = ((features - center) / scale).astype(np.float32, copy=False)
    adata.uns["scaler_params"] = {
        "feature_names": [f"cell_type_freq::{name}" for name in type_names.tolist()] + ["local_density"],
        "center": center,
        "scale": scale.astype(np.float32, copy=False),
    }


def learn_global_prototypes(adata: AnnData, n_bal: int, K: int, random_state: int = 42) -> None:
    """
    Learns global prototypes via balanced sub-sampling and KMeans clustering.

    Args:
        adata: AnnData object containing `.obsm['community_features']`.
        n_bal: Number of cells to sample per ROI/category for balanced clustering.
        K: Number of prototypes (clusters) to discover.
        random_state: Seed for reproducibility.

    Side Effects:
        - Assigns cluster IDs to `adata.obs['proto_id']`.
        - Computes the global cost scale and stores it in `adata.uns['s_C']`.
        - Stores the KMeans centroids in `adata.uns['prototype_centroids']`.

    Constraints for Codex:
        1. DETERMINISM: You MUST use the `random_state` for both subsampling and KMeans.
        2. AUDIT TRAIL: Save the learned cluster centers to `adata.uns['prototype_centroids']`.
        3. CANONICAL NAMES: You MUST write the cost scale exactly to `adata.uns['s_C']`.
    """
    if n_bal <= 0:
        raise ValueError("n_bal must be strictly positive")
    if K <= 0:
        raise ValueError("K must be strictly positive")
    if "community_features" not in adata.obsm:
        raise ValueError("adata.obsm must contain 'community_features' before prototype learning")
    if "roi_id" not in adata.obs.columns:
        raise ValueError("adata.obs must contain 'roi_id' for balanced prototype sampling")

    features = np.asarray(adata.obsm["community_features"], dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != adata.obs.shape[0]:
        raise ValueError("adata.obsm['community_features'] must have shape [n_cells, d]")
    if not np.isfinite(features).all():
        raise ValueError("adata.obsm['community_features'] contains NaN/Inf")

    roi_ids = adata.obs["roi_id"].astype(str).to_numpy()
    rng = np.random.default_rng(random_state)
    sampled_indices: list[np.ndarray] = []
    for roi_id in np.unique(roi_ids):
        roi_indices = np.flatnonzero(roi_ids == roi_id)
        n_take = min(int(n_bal), roi_indices.size)
        sampled_indices.append(np.sort(rng.choice(roi_indices, size=n_take, replace=False)))

    balanced_idx = np.concatenate(sampled_indices)
    if balanced_idx.size < K:
        raise ValueError(
            f"Balanced sample must contain at least K={K} cells, got {balanced_idx.size}"
        )

    model = KMeans(n_clusters=K, random_state=random_state, n_init=20)
    model.fit(features[balanced_idx])
    proto_id = model.predict(features).astype(int, copy=False)
    centroids = np.asarray(model.cluster_centers_, dtype=np.float32)
    cost_matrix = cdist(centroids, centroids, metric="euclidean").astype(np.float32, copy=False)
    positive_costs = cost_matrix[cost_matrix > 0.0]
    s_C = float(np.median(positive_costs)) if positive_costs.size > 0 else 1.0

    adata.obs["proto_id"] = proto_id
    adata.uns["prototype_centroids"] = centroids
    adata.uns["cost_matrix"] = cost_matrix
    adata.uns["s_C"] = s_C
