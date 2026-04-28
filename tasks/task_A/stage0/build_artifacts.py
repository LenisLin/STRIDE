"""Build and validate frozen Task A Stage 0 artifacts.

This module consumes the raw CRLM cohort extraction surface and writes the
task-local Stage 0 h5ad plus validation JSON. It does not run Step 1 prepare
or any Task A evidence block.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from stride.basis.aggregation import build_local_state_features, learn_shared_state_axis
from stride.data.longitudinal import (
    CANONICAL_STATE_FEATURE_METADATA_KEY,
    assemble_longitudinal_adata,
    resolve_cell_subtype_key,
    resolve_domain_key,
    resolve_feature_key,
    resolve_fov_key,
    resolve_state_id_key,
    validate_longitudinal_adata,
)
from stride.errors import ContractError, DataContractError

from ..contracts import CONTRACT_PASSED_STATE, SCAFFOLD_ACTIVE_STATE

DEFAULT_RDS_PATH: str | None = None
DEFAULT_OUTPUT_DIR: str | None = None
DEFAULT_K = 25
DEFAULT_KNN = 20
DEFAULT_N_BAL = 200
DEFAULT_RANDOM_STATE = 42

H5AD_FILENAME_TEMPLATE = "task_A_stage0_k{K}.h5ad"
VALIDATION_FILENAME = "task_A_stage0_validation.json"

REQUIRED_CELL_TABLE_COLUMNS: tuple[str, ...] = ("ID", "PID", "Tissue", "SubType", "Area", "x", "y")
REPRESENTATION_REQUIRED_SCALER_KEYS: tuple[str, ...] = ("feature_names", "center", "scale")


def run_r_extraction(
    rds_path: str | Path,
    *,
    cells_out: str | Path,
    roi_clinical_out: str | Path,
    expr_out: str | Path,
    markers_out: str | Path,
    script_path: str | Path | None = None,
) -> None:
    extract_script = Path(script_path) if script_path is not None else Path(__file__).with_name("extract_crlm_coldata.R")
    command = [
        "Rscript",
        str(extract_script),
        "--rds",
        str(rds_path),
        "--cells_out",
        str(cells_out),
        "--roi_clinical_out",
        str(roi_clinical_out),
        "--expr_out",
        str(expr_out),
        "--markers_out",
        str(markers_out),
    ]
    subprocess.run(command, check=True)


def load_extracted_tables(
    *,
    cells_path: str | Path,
    roi_clinical_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_table = pd.read_csv(cells_path)
    roi_clinical_table = pd.read_csv(roi_clinical_path)
    return cell_table, roi_clinical_table


def load_marker_names(markers_path: str | Path) -> list[str]:
    marker_names = [line.strip() for line in Path(markers_path).read_text(encoding="utf-8").splitlines()]
    marker_names = [marker for marker in marker_names if marker]
    if not marker_names:
        raise ValueError("Stage 0 extraction produced an empty marker list")
    marker_index = pd.Index(marker_names, dtype="object")
    if marker_index.has_duplicates:
        duplicates = sorted(marker_index[marker_index.duplicated()].unique().tolist())
        raise ValueError(f"Stage 0 extraction produced duplicate marker names: {duplicates}")
    return marker_names


def load_expression_matrix(
    expr_path: str | Path,
    *,
    n_cells: int,
    marker_names: Sequence[str],
) -> np.ndarray:
    n_markers = len(marker_names)
    if n_cells <= 0:
        raise ValueError("Stage 0 expression load requires at least one cell")
    if n_markers <= 0:
        raise ValueError("Stage 0 expression load requires at least one marker")

    flat = np.fromfile(expr_path, dtype=np.float32)
    expected_size = n_cells * n_markers
    if flat.size != expected_size:
        raise ValueError(
            f"Stage 0 expression binary has {flat.size} values; expected {expected_size} "
            f"for {n_cells} cells x {n_markers} markers"
        )

    features_by_cells = flat.reshape((n_markers, n_cells), order="F")
    expression_matrix = np.asarray(features_by_cells.T, dtype=np.float32, order="C")
    if not np.isfinite(expression_matrix).all():
        raise ValueError("Stage 0 expression matrix contains NaN/Inf")
    return expression_matrix


def _build_obs_index(cell_table: pd.DataFrame) -> pd.Index:
    if "CellID" in cell_table.columns:
        raw = cell_table["CellID"].astype(str)
        if not raw.duplicated().any() and raw.notna().all():
            return pd.Index(raw, dtype="object")
    return pd.Index([f"cell_{idx}" for idx in range(cell_table.shape[0])], dtype="object")


def build_stage0_adata_from_cell_table(
    cell_table: pd.DataFrame,
    expression_matrix: np.ndarray,
    marker_names: Sequence[str],
    *,
    k: int = DEFAULT_K,
    knn: int = DEFAULT_KNN,
    n_bal: int = DEFAULT_N_BAL,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> ad.AnnData:
    missing = [column for column in REQUIRED_CELL_TABLE_COLUMNS if column not in cell_table.columns]
    if missing:
        raise ValueError(f"Stage 0 cell table is missing required columns: {missing}")
    if expression_matrix.shape[0] != cell_table.shape[0]:
        raise ValueError(
            f"Stage 0 expression matrix row count {expression_matrix.shape[0]} does not match "
            f"cell table row count {cell_table.shape[0]}"
        )
    if expression_matrix.shape[1] != len(marker_names):
        raise ValueError(
            f"Stage 0 expression matrix column count {expression_matrix.shape[1]} does not match "
            f"marker count {len(marker_names)}"
        )

    canonical_cell_table = pd.DataFrame(
        {
            "patient_id": cell_table["PID"].astype(str).to_numpy(),
            "timepoint": np.repeat("0", cell_table.shape[0]),
            "fov_id": cell_table["ID"].astype(str).to_numpy(),
            "domain_label": cell_table["Tissue"].astype(str).to_numpy(),
            "cell_subtype_label": cell_table["SubType"].astype(str).to_numpy(),
            "x": pd.to_numeric(cell_table["x"], errors="raise").to_numpy(dtype=float),
            "y": pd.to_numeric(cell_table["y"], errors="raise").to_numpy(dtype=float),
        },
        index=_build_obs_index(cell_table),
    )
    assembled = assemble_longitudinal_adata(canonical_cell_table)
    obs = assembled.obs.copy()
    obs.index = canonical_cell_table.index.copy()
    adata = ad.AnnData(
        X=np.asarray(expression_matrix, dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(
            index=pd.Index([str(marker) for marker in marker_names], dtype="object", name="marker_name")
        ),
    )
    adata.obsm["spatial"] = np.asarray(assembled.obsm["spatial"], dtype=float)
    adata.obs["roi_id"] = adata.obs["fov_id"].astype(str)
    adata.obs["compartment"] = adata.obs["domain_label"].astype(str)
    adata.obs["cell_type"] = adata.obs["cell_subtype_label"].astype(str)
    adata.obs["cell_area"] = pd.to_numeric(cell_table["Area"], errors="raise").to_numpy(dtype=float)
    adata.uns["roi_areas"] = {
        roi_id: 1.0 for roi_id in sorted(adata.obs["roi_id"].astype(str).unique())
    }

    build_local_state_features(adata, k=knn, write_compat_aliases=True)
    learn_shared_state_axis(
        adata,
        n_bal=n_bal,
        K=k,
        random_state=random_state,
        write_compat_aliases=True,
    )
    return adata


def validate_stage0_minimum_contract(adata: ad.AnnData) -> None:
    validate_longitudinal_adata(
        adata,
        require_cell_type=True,
        require_representation=True,
        require_state_axis=True,
        require_cost_scale=True,
        require_cost_matrix=True,
    )


def validate_representation_completeness(adata: ad.AnnData) -> None:
    try:
        subtype_key = resolve_cell_subtype_key(adata)
    except ContractError as exc:
        raise DataContractError(f"representation completeness: {exc}") from exc
    if subtype_key not in adata.obs.columns:
        raise DataContractError(
            "representation completeness: missing canonical-or-alias cell subtype column"
        )

    try:
        feature_key = resolve_feature_key(adata)
    except ContractError as exc:
        raise DataContractError(f"representation completeness: {exc}") from exc
    features = np.asarray(adata.obsm[feature_key], dtype=float)
    if features.ndim != 2 or features.shape[0] != adata.obs.shape[0]:
        raise DataContractError(
            f"representation completeness: adata.obsm[{feature_key!r}] must have shape [n_cells, d]"
        )

    feature_metadata = adata.uns.get(CANONICAL_STATE_FEATURE_METADATA_KEY)
    if isinstance(feature_metadata, Mapping):
        if "feature_names" not in feature_metadata:
            raise DataContractError(
                "representation completeness: state_feature_metadata missing key 'feature_names'"
            )
    else:
        scaler_params = adata.uns.get("scaler_params")
        if not isinstance(scaler_params, Mapping):
            raise DataContractError(
                "representation completeness: missing uns mapping "
                "'state_feature_metadata' or compatibility 'scaler_params'"
            )
        missing_scaler = [key for key in REPRESENTATION_REQUIRED_SCALER_KEYS if key not in scaler_params]
        if missing_scaler:
            raise DataContractError(
                f"representation completeness: scaler_params missing keys {missing_scaler}"
            )

    if "state_centroids" in adata.uns:
        centroids_key = "state_centroids"
    elif "prototype_centroids" in adata.uns:
        centroids_key = "prototype_centroids"
    else:
        raise DataContractError(
            "representation completeness: missing uns key 'state_centroids' or 'prototype_centroids'"
        )
    centroids = np.asarray(adata.uns.get(centroids_key), dtype=float)
    if centroids.ndim != 2:
        raise DataContractError(
            f"representation completeness: {centroids_key} must be 2D"
        )

    if "cost_matrix" not in adata.uns:
        raise DataContractError("representation completeness: missing uns key 'cost_matrix'")
    cost_matrix = np.asarray(adata.uns["cost_matrix"], dtype=float)
    if centroids.shape[0] != cost_matrix.shape[0]:
        raise DataContractError(
            "representation completeness: prototype_centroids and cost_matrix disagree on K"
        )


def build_stage0_h5ad_validation_report(
    adata: ad.AnnData,
    *,
    require_all_proto_ids: bool,
    roi_clinical_rows: int | None = None,
) -> dict[str, Any]:
    def _safe_nunique(column: str) -> int | None:
        if column not in adata.obs.columns:
            return None
        return int(adata.obs[column].astype(str).nunique())

    def _safe_resolved_nunique(resolve_key: object) -> int | None:
        try:
            column = resolve_key(adata)
        except Exception:
            return None
        return _safe_nunique(str(column))

    level_a: dict[str, Any] = {"ok": True}
    try:
        validate_stage0_minimum_contract(adata)
    except Exception as exc:  # pragma: no cover - exercised through integration failures
        level_a = {"ok": False, "error": str(exc)}

    level_b: dict[str, Any] = {"ok": True}
    try:
        validate_representation_completeness(adata)
    except Exception as exc:  # pragma: no cover - exercised through integration failures
        level_b = {"ok": False, "error": str(exc)}

    proto_ids: pd.Series | None = None
    try:
        state_id_key = resolve_state_id_key(adata)
    except Exception:
        state_id_key = None
    if state_id_key is not None and state_id_key in adata.obs.columns:
        try:
            proto_ids = pd.Series(adata.obs[state_id_key]).astype(int)
        except Exception:  # pragma: no cover - defensive summary path
            proto_ids = None

    cost_matrix_shape0: int | None = None
    if "cost_matrix" in adata.uns:
        try:
            cost_matrix = np.asarray(adata.uns["cost_matrix"], dtype=float)
        except Exception:  # pragma: no cover - defensive summary path
            cost_matrix = None
        if cost_matrix is not None and cost_matrix.ndim == 2:
            cost_matrix_shape0 = int(cost_matrix.shape[0])

    all_proto_ids_present = bool(
        proto_ids is not None
        and cost_matrix_shape0 is not None
        and set(proto_ids.unique()) == set(range(cost_matrix_shape0))
    )
    if require_all_proto_ids and not all_proto_ids_present:
        level_b = {
            "ok": False,
            "error": "representation completeness: not all prototype IDs on the shared axis are present",
        }

    artifact_state = (
        CONTRACT_PASSED_STATE
        if bool(level_a["ok"]) and bool(level_b["ok"])
        else SCAFFOLD_ACTIVE_STATE
    )

    return {
        "artifact_state": artifact_state,
        "taska_minimum_contract": level_a,
        "representation_completeness": level_b,
        "counts": {
            "n_cells": int(adata.obs.shape[0]),
            "n_patients": _safe_nunique("patient_id"),
            "n_rois": _safe_resolved_nunique(resolve_fov_key),
            "n_compartments": _safe_resolved_nunique(resolve_domain_key),
            "n_unique_proto_ids": None if proto_ids is None else int(proto_ids.nunique()),
            "all_proto_ids_present": bool(all_proto_ids_present),
            "roi_clinical_rows": None if roi_clinical_rows is None else int(roi_clinical_rows),
        },
    }


def build_stage0_validation_report(
    adata: ad.AnnData,
    roi_clinical_table: pd.DataFrame,
    *,
    require_all_proto_ids: bool,
) -> dict[str, Any]:
    return build_stage0_h5ad_validation_report(
        adata,
        require_all_proto_ids=require_all_proto_ids,
        roi_clinical_rows=int(roi_clinical_table.shape[0]),
    )


def build_stage0_artifacts(
    *,
    rds_path: str | Path | None = DEFAULT_RDS_PATH,
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    k: int = DEFAULT_K,
    knn: int = DEFAULT_KNN,
    n_bal: int = DEFAULT_N_BAL,
    random_state: int = DEFAULT_RANDOM_STATE,
    require_all_proto_ids: bool = True,
) -> dict[str, Path]:
    if rds_path is None:
        raise ValueError("Stage 0 build requires an explicit rds_path")
    if output_dir is None:
        raise ValueError("Stage 0 build requires an explicit output_dir")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="stride_stage0_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        cells_csv = tmp_dir / "cells.csv"
        roi_clinical_csv = tmp_dir / "roi_clinical.csv"
        expression_bin = tmp_dir / "expression.bin"
        markers_txt = tmp_dir / "markers.txt"
        run_r_extraction(
            rds_path,
            cells_out=cells_csv,
            roi_clinical_out=roi_clinical_csv,
            expr_out=expression_bin,
            markers_out=markers_txt,
        )
        cell_table, roi_clinical_table = load_extracted_tables(
            cells_path=cells_csv,
            roi_clinical_path=roi_clinical_csv,
        )
        marker_names = load_marker_names(markers_txt)
        expression_matrix = load_expression_matrix(
            expression_bin,
            n_cells=cell_table.shape[0],
            marker_names=marker_names,
        )

    adata = build_stage0_adata_from_cell_table(
        cell_table,
        expression_matrix=expression_matrix,
        marker_names=marker_names,
        k=k,
        knn=knn,
        n_bal=n_bal,
        random_state=random_state,
    )

    h5ad_path = out_dir / H5AD_FILENAME_TEMPLATE.format(K=k)
    validation_path = out_dir / VALIDATION_FILENAME

    adata.write_h5ad(h5ad_path)
    report = build_stage0_validation_report(
        adata,
        roi_clinical_table,
        require_all_proto_ids=require_all_proto_ids,
    )
    validation_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "h5ad_path": h5ad_path,
        "validation_path": validation_path,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 0 Task-A artifacts from the CRLM cohort.")
    parser.add_argument("--rds-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--knn", type=int, default=DEFAULT_KNN)
    parser.add_argument("--n-bal", type=int, default=DEFAULT_N_BAL)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument(
        "--allow-missing-prototype-ids",
        action="store_true",
        help="Relax the real-artifact all-prototype integration check.",
    )
    return parser.parse_args(argv)


def main(
    output_dir: str | None = DEFAULT_OUTPUT_DIR,
    *,
    rds_path: str | None = DEFAULT_RDS_PATH,
    k: int = DEFAULT_K,
    knn: int = DEFAULT_KNN,
    n_bal: int = DEFAULT_N_BAL,
    random_state: int = DEFAULT_RANDOM_STATE,
    require_all_proto_ids: bool = True,
) -> dict[str, Path]:
    return build_stage0_artifacts(
        rds_path=rds_path,
        output_dir=output_dir,
        k=k,
        knn=knn,
        n_bal=n_bal,
        random_state=random_state,
        require_all_proto_ids=require_all_proto_ids,
    )


if __name__ == "__main__":  # pragma: no cover
    args = parse_args()
    outputs = main(
        output_dir=args.output_dir,
        rds_path=args.rds_path,
        k=args.k,
        knn=args.knn,
        n_bal=args.n_bal,
        random_state=args.random_state,
        require_all_proto_ids=not args.allow_missing_prototype_ids,
    )
    reloaded = ad.read_h5ad(outputs["h5ad_path"])
    validate_stage0_minimum_contract(reloaded)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
