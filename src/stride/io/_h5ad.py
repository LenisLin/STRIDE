"""Thin h5ad I/O wrappers for STRIDE .io."""
from __future__ import annotations

from pathlib import Path

import anndata as ad
from anndata import AnnData


def read_h5ad(path: str | Path, **kwargs: object) -> AnnData:
    """Read an h5ad file through anndata without STRIDE task semantics."""
    return ad.read_h5ad(path, **kwargs)


def write_h5ad(adata: AnnData, path: str | Path, **kwargs: object) -> Path:
    """Write an AnnData object as h5ad and return the written path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path, **kwargs)
    return output_path
