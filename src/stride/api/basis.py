"""Stable first-pass facades for tissue-agnostic STRIDE state preparation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..basis import StateBasis, build_local_state_features, learn_shared_state_axis
from ..data.longitudinal import AnnData, load_state_basis_from_adata
from ..geometry import StateGeometry, build_state_geometry
from ..observation import FovObservation, build_fov_observations


@dataclass(frozen=True)
class BasisSpec:
    """Configuration for the deterministic shared community-state basis route.

    Domain labels remain observation metadata only: this route constructs the
    shared basis from cell subtype neighborhoods before any domain
    stratification.
    """

    K: int
    k_neighbors: int = 20
    random_state: int = 42
    geometry_neighbors: int = 5

    def build_features(self, adata: AnnData, *, write_compat_aliases: bool = False) -> np.ndarray:
        """Build cell-level neighborhood subtype-composition features."""
        return build_local_state_features(
            adata,
            k=self.k_neighbors,
            write_compat_aliases=write_compat_aliases,
        )

    def learn_axis(
        self,
        adata: AnnData,
        *,
        feature_key: str | None = None,
        write_compat_aliases: bool = False,
    ) -> StateBasis:
        """Learn and persist the shared community-state axis for one dataset."""
        return learn_shared_state_axis(
            adata,
            K=self.K,
            random_state=self.random_state,
            feature_key=feature_key,
            write_compat_aliases=write_compat_aliases,
        )

    def fit(self, adata: AnnData, *, write_compat_aliases: bool = False) -> StateBasis:
        """Run the canonical end-to-end basis preparation route on one dataset."""
        self.build_features(adata, write_compat_aliases=write_compat_aliases)
        return self.learn_axis(
            adata,
            write_compat_aliases=write_compat_aliases,
        )

    def build_geometry(
        self,
        *,
        state_basis: StateBasis | None = None,
        adata: AnnData | None = None,
        n_neighbors: int | None = None,
    ) -> StateGeometry:
        """Construct state geometry from an explicit basis or dataset-attached artifacts."""
        resolved_basis = state_basis
        if resolved_basis is None:
            if adata is None:
                raise ValueError("build_geometry requires state_basis or adata")
            resolved_basis = load_state_basis_from_adata(adata)
        return build_state_geometry(
            centroids=resolved_basis.centroids,
            cost_matrix=resolved_basis.cost_matrix,
            cost_scale=resolved_basis.cost_scale,
            state_ids=resolved_basis.resolved_state_ids,
            n_neighbors=self.geometry_neighbors if n_neighbors is None else n_neighbors,
        )

    def build_observations(
        self,
        adata: AnnData,
        *,
        state_key: str | None = None,
        n_states: int | None = None,
    ) -> tuple[FovObservation, ...]:
        """Construct uniform-mass ROI/FOV observations from community assignments."""
        resolved_n_states = self.K if n_states is None else n_states
        return build_fov_observations(
            adata,
            state_key=state_key,
            n_states=resolved_n_states,
        )


__all__ = ["BasisSpec"]
