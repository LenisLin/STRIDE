"""
Module: tasks.task_A.build_stage0_artifacts
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

from slotar.contracts import DataContractError, validate_adata_inputs
from slotar.representation import build_community_features, learn_global_prototypes

DEFAULT_RDS_PATH = "/mnt/NAS_21T/ProjectData/SLOTAR/CRLM_Cohort.rds"
DEFAULT_OUTPUT_DIR = "/mnt/NAS_21T/ProjectData/SLOTAR/task_A_stage0"
DEFAULT_K = 25
DEFAULT_KNN = 20
DEFAULT_N_BAL = 200
DEFAULT_RANDOM_STATE = 42

H5AD_FILENAME_TEMPLATE = "task_A_stage0_k{K}.h5ad"
ROI_CLINICAL_FILENAME = "task_A_stage0_roi_clinical.parquet"
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

    obs = pd.DataFrame(
        {
            "patient_id": cell_table["PID"].astype(str).to_numpy(),
            "timepoint": np.zeros(cell_table.shape[0], dtype=int),
            "roi_id": cell_table["ID"].astype(str).to_numpy(),
            "compartment": cell_table["Tissue"].astype(str).to_numpy(),
            "cell_type": cell_table["SubType"].astype(str).to_numpy(),
            "cell_area": pd.to_numeric(cell_table["Area"], errors="raise").to_numpy(),
        },
        index=_build_obs_index(cell_table),
    )
    var = pd.DataFrame(index=pd.Index([str(marker) for marker in marker_names], dtype="object", name="marker_name"))

    spatial = cell_table.loc[:, ["x", "y"]].to_numpy(dtype=float)
    adata = ad.AnnData(X=np.asarray(expression_matrix, dtype=np.float32), obs=obs, var=var)
    adata.obsm["spatial"] = spatial
    adata.uns["roi_areas"] = {roi_id: 1.0 for roi_id in sorted(obs["roi_id"].astype(str).unique())}

    build_community_features(adata, k=knn)
    learn_global_prototypes(adata, n_bal=n_bal, K=k, random_state=random_state)
    return adata


def validate_stage0_minimum_contract(adata: ad.AnnData) -> None:
    validate_adata_inputs(
        adata,
        require_prototypes=True,
        require_cost_scale=True,
        require_cost_matrix=True,
    )


def validate_representation_completeness(adata: ad.AnnData) -> None:
    if "cell_type" not in adata.obs.columns:
        raise DataContractError("representation completeness: missing obs column 'cell_type'")

    if "community_features" not in adata.obsm:
        raise DataContractError("representation completeness: missing obsm key 'community_features'")
    features = np.asarray(adata.obsm["community_features"], dtype=float)
    if features.ndim != 2 or features.shape[0] != adata.obs.shape[0]:
        raise DataContractError(
            "representation completeness: adata.obsm['community_features'] must have shape [n_cells, d]"
        )

    scaler_params = adata.uns.get("scaler_params")
    if not isinstance(scaler_params, Mapping):
        raise DataContractError("representation completeness: missing uns mapping 'scaler_params'")
    missing_scaler = [key for key in REPRESENTATION_REQUIRED_SCALER_KEYS if key not in scaler_params]
    if missing_scaler:
        raise DataContractError(
            f"representation completeness: scaler_params missing keys {missing_scaler}"
        )

    centroids = np.asarray(adata.uns.get("prototype_centroids"), dtype=float)
    if centroids.ndim != 2:
        raise DataContractError("representation completeness: prototype_centroids must be 2D")

    if "cost_matrix" not in adata.uns:
        raise DataContractError("representation completeness: missing uns key 'cost_matrix'")
    cost_matrix = np.asarray(adata.uns["cost_matrix"], dtype=float)
    if centroids.shape[0] != cost_matrix.shape[0]:
        raise DataContractError(
            "representation completeness: prototype_centroids and cost_matrix disagree on K"
        )


def build_stage0_validation_report(
    adata: ad.AnnData,
    roi_clinical_table: pd.DataFrame,
    *,
    require_all_proto_ids: bool,
) -> dict[str, Any]:
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

    proto_ids = pd.Series(adata.obs["proto_id"]).astype(int)
    all_proto_ids_present = set(proto_ids.unique()) == set(range(int(np.asarray(adata.uns["cost_matrix"]).shape[0])))
    if require_all_proto_ids and not all_proto_ids_present:
        level_b = {
            "ok": False,
            "error": "representation completeness: not all prototype IDs on the shared axis are present",
        }

    report = {
        "taska_minimum_contract": level_a,
        "representation_completeness": level_b,
        "counts": {
            "n_cells": int(adata.obs.shape[0]),
            "n_patients": int(adata.obs["patient_id"].astype(str).nunique()),
            "n_rois": int(adata.obs["roi_id"].astype(str).nunique()),
            "n_compartments": int(adata.obs["compartment"].astype(str).nunique()),
            "n_unique_proto_ids": int(proto_ids.nunique()),
            "all_proto_ids_present": bool(all_proto_ids_present),
            "roi_clinical_rows": int(roi_clinical_table.shape[0]),
        },
    }
    return report


def build_stage0_artifacts(
    *,
    rds_path: str | Path = DEFAULT_RDS_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    k: int = DEFAULT_K,
    knn: int = DEFAULT_KNN,
    n_bal: int = DEFAULT_N_BAL,
    random_state: int = DEFAULT_RANDOM_STATE,
    require_all_proto_ids: bool = True,
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="slotar_stage0_") as tmp_dir_name:
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
    roi_clinical_path = out_dir / ROI_CLINICAL_FILENAME
    validation_path = out_dir / VALIDATION_FILENAME

    adata.write_h5ad(h5ad_path)
    roi_clinical_table.to_parquet(roi_clinical_path, index=False)
    report = build_stage0_validation_report(
        adata,
        roi_clinical_table,
        require_all_proto_ids=require_all_proto_ids,
    )
    validation_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "h5ad_path": h5ad_path,
        "roi_clinical_path": roi_clinical_path,
        "validation_path": validation_path,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 0 Task-A artifacts from the CRLM cohort.")
    parser.add_argument("--rds-path", default=DEFAULT_RDS_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
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
    output_dir: str = DEFAULT_OUTPUT_DIR,
    *,
    rds_path: str = DEFAULT_RDS_PATH,
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
