"""Stable dataset facade for the deterministic upstream STRIDE ingestion route.

This surface normalizes the live canonical observation keys (`timepoint`,
`fov_id`, `domain_label`) while still accepting migration-era aliases.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..basis import StateBasis
from ..data.longitudinal import (
    AnnData,
    assemble_longitudinal_adata,
    load_state_basis_from_adata,
    resolve_cell_subtype_key,
    resolve_domain_key,
    resolve_feature_key,
    resolve_fov_key,
    resolve_state_id_key,
    resolve_timepoint_key,
    validate_longitudinal_adata,
)


@dataclass(frozen=True)
class DatasetHandle:
    """Thin wrapper around one AnnData object on the canonical STRIDE data layer.

    The current first-pass route keeps `timepoint` as the canonical ordered-side
    field and does not require ROI area metadata under the uniform-mass
    observation contract.
    """

    adata: AnnData

    @classmethod
    def from_tables(
        cls,
        cell_table: pd.DataFrame,
        *,
        fov_table: pd.DataFrame | None = None,
    ) -> "DatasetHandle":
        """Assemble a canonical STRIDE dataset from cell and optional FOV tables."""
        return cls(adata=assemble_longitudinal_adata(cell_table, fov_table=fov_table))

    def validate(self, **kwargs: object) -> None:
        """Validate the wrapped dataset against the STRIDE longitudinal contract."""
        validate_longitudinal_adata(self.adata, **kwargs)

    def load_state_basis(self) -> StateBasis:
        """Load the stored shared state basis associated with this dataset."""
        return load_state_basis_from_adata(self.adata)

    @property
    def timepoint_key(self) -> str:
        """Return the resolved timepoint column name used by this dataset."""
        return resolve_timepoint_key(self.adata)

    @property
    def fov_key(self) -> str:
        """Return the resolved observation-unit column name."""
        return resolve_fov_key(self.adata)

    @property
    def domain_key(self) -> str:
        """Return the resolved domain-label column name."""
        return resolve_domain_key(self.adata)

    @property
    def cell_subtype_key(self) -> str:
        """Return the resolved cell-subtype label column name."""
        return resolve_cell_subtype_key(self.adata)

    @property
    def feature_key(self) -> str:
        """Return the resolved local feature matrix key in `adata.obsm`."""
        return resolve_feature_key(self.adata)

    @property
    def state_id_key(self) -> str:
        """Return the resolved shared-state identifier column name."""
        return resolve_state_id_key(self.adata)


__all__ = ["DatasetHandle"]
