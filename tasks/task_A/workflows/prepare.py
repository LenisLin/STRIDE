"""Canonical Task A Step 1 prepare workflow.

This module consumes the frozen Task A config and Stage 0 h5ad, optionally
filters to subset/demo surfaces, and writes mapping plus dry-run manifests. It
does not satisfy Block 0 or emit scientific interpretation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from stride.errors import ContractError

from ..config import load_task_a_config_bundle
from ..contracts import CONTRACT_PASSED_STATE, SCAFFOLD_ACTIVE_STATE
from ..real_data.demo_subset import TaskADemoSubset, resolve_demo_subset
from ..stage0.build_artifacts import build_stage0_h5ad_validation_report
from .stride_adapter import (
    SEMANTIC_ALIGNMENT_ERROR_PREFIX,
    describe_task_a_stage0_stride_mapping,
    load_task_a_dataset_handle,
    run_task_a_family_core_fit_dry_run,
)

try:
    import anndata as ad
except ModuleNotFoundError:  # pragma: no cover
    ad = None  # type: ignore[assignment]


PRE_BLOCK0_DATA_SUITABILITY_FILENAME = "task_a_pre_block0_data_suitability.json"
FULL_COHORT_ALIGNMENT_CHECK_SCOPE = "full_cohort_alignment_check"
PATIENT_SUBSET_SCOPE = "patient_subset"
DEMO_SUBSET_SCOPE = "demo_subset"


def _filter_adata_by_patients(
    data_path: str | Path,
    patient_ids: tuple[str, ...],
) -> Any:
    """Load an h5ad and subset to the given patient_ids."""
    if ad is None:
        raise ModuleNotFoundError("anndata is required to filter Stage 0 h5ad by patient_id")
    path = Path(data_path).expanduser().resolve()
    adata = ad.read_h5ad(path)
    mask = adata.obs["patient_id"].astype(str).isin(set(patient_ids))
    return adata[mask].copy()


def _load_all_patient_ids(data_path: str | Path) -> tuple[str, ...]:
    handle = load_task_a_dataset_handle(data_path)
    return tuple(sorted(handle.adata.obs["patient_id"].astype(str).unique().tolist()))


def _resolve_optional_demo_subset(name: str | None) -> TaskADemoSubset | None:
    if name is None:
        return None
    try:
        return resolve_demo_subset(name)
    except KeyError as exc:
        raise ContractError(str(exc.args[0])) from exc


def _resolve_selected_patient_ids(
    *,
    patient_ids: tuple[str, ...] | None,
    demo_subset: TaskADemoSubset | None,
) -> tuple[str, ...] | None:
    if patient_ids is None and demo_subset is not None:
        patient_ids = demo_subset.patient_ids
    if patient_ids is None:
        return None

    selected_patient_ids = tuple(dict.fromkeys(str(patient_id) for patient_id in patient_ids))
    if not selected_patient_ids:
        raise ContractError("Task A prepare requires at least one patient_id in pre-Block 0 mode")

    return selected_patient_ids


def _resolve_prepare_scope_metadata(
    *,
    selected_patient_ids: tuple[str, ...],
    demo_subset: TaskADemoSubset | None,
) -> tuple[str, str | None, str | None]:
    if demo_subset is None:
        return PATIENT_SUBSET_SCOPE, None, None
    if tuple(selected_patient_ids) != tuple(demo_subset.patient_ids):
        raise ContractError(
            "Task A prepare may label a run as demo_subset only when patient_ids "
            "exactly match the resolved named subset"
        )
    return DEMO_SUBSET_SCOPE, demo_subset.name, demo_subset.rationale


def _is_semantic_alignment_failure(exc: Exception) -> bool:
    return str(exc).startswith(SEMANTIC_ALIGNMENT_ERROR_PREFIX)


def _resolve_prepare_source(
    *,
    data_path: str | Path,
    patient_ids: tuple[str, ...] | None,
    demo_subset_name: str | None,
) -> tuple[Any, tuple[str, ...] | None, str, str, str | None, str | None]:
    resolved_demo_subset = _resolve_optional_demo_subset(demo_subset_name)
    selected_patient_ids = _resolve_selected_patient_ids(
        patient_ids=patient_ids,
        demo_subset=resolved_demo_subset,
    )
    if selected_patient_ids is None:
        return (
            load_task_a_dataset_handle(data_path),
            None,
            FULL_COHORT_ALIGNMENT_CHECK_SCOPE,
            CONTRACT_PASSED_STATE,
            None,
            None,
        )

    all_patient_ids = _load_all_patient_ids(data_path)
    matched_patient_ids = set(selected_patient_ids) & set(all_patient_ids)
    if matched_patient_ids == set(all_patient_ids):
        raise ContractError(
            "Task A prepare full-cohort alignment checks must omit subset selectors. "
            "Remove --patient-id/--demo-subset to run the canonical full-cohort Step 1 entrypoint."
        )

    run_scope, manifest_demo_subset_name, manifest_demo_subset_rationale = _resolve_prepare_scope_metadata(
        selected_patient_ids=selected_patient_ids,
        demo_subset=resolved_demo_subset,
    )
    return (
        _filter_adata_by_patients(data_path, selected_patient_ids),
        selected_patient_ids,
        run_scope,
        SCAFFOLD_ACTIVE_STATE,
        manifest_demo_subset_name,
        manifest_demo_subset_rationale,
    )


def _build_task_a_step1_alignment(
    *,
    source: Any,
    config_bundle: Any,
) -> tuple[Any, pd.DataFrame]:
    mapping_summary = describe_task_a_stage0_stride_mapping(
        source,
        config_bundle=config_bundle,
    )
    dry_run_df, _results = run_task_a_family_core_fit_dry_run(
        source,
        config_bundle=config_bundle,
        pair_families=config_bundle.ordered_proxy.confirmatory_pair_families,
    )
    return mapping_summary, dry_run_df


def build_task_a_pre_block0_data_suitability_report(
    *,
    config_path: str | Path,
    data_path: str | Path,
) -> dict[str, Any]:
    config_bundle = load_task_a_config_bundle(config_path)
    resolved_data_path = Path(data_path).expanduser().resolve()
    handle = load_task_a_dataset_handle(resolved_data_path)
    validation_report = build_stage0_h5ad_validation_report(
        handle.adata,
        require_all_proto_ids=False,
    )
    report: dict[str, Any] = {
        "task_name": config_bundle.raw_config.get("task_name", "Task A"),
        "config_path": str(config_bundle.config_path),
        "stage0_h5ad": str(resolved_data_path),
        "report_scope": "pre_block0_data_suitability",
        "run_scope": FULL_COHORT_ALIGNMENT_CHECK_SCOPE,
        "artifact_state": SCAFFOLD_ACTIVE_STATE,
        "block0_gate_status": "not_passed",
        "scientific_interpretation_allowed": False,
        "mass_mode": config_bundle.data.mass_mode,
        "fit_surface": "fit_stride",
        "implementation_tier": "canonical_full",
        "evidence_lineage": "canonical_rerun",
        "confirmatory_pair_families": [
            family.name for family in config_bundle.ordered_proxy.confirmatory_pair_families
        ],
        "audit_pair_families": [
            family.name for family in config_bundle.ordered_proxy.audit_pair_families
        ],
        "stage0_validation": validation_report,
    }
    try:
        mapping_summary, _dry_run_df = _build_task_a_step1_alignment(
            source=handle,
            config_bundle=config_bundle,
        )
    except (ContractError, ValueError) as exc:
        if _is_semantic_alignment_failure(exc):
            raise
        report["mapping_summary"] = None
        report["mapping_summary_error"] = str(exc)
    else:
        report["mapping_summary"] = mapping_summary.to_json_dict()
        if bool(report["stage0_validation"]["taska_minimum_contract"]["ok"]):
            report["artifact_state"] = CONTRACT_PASSED_STATE
    return report


def write_task_a_pre_block0_data_suitability_report(
    *,
    config_path: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
) -> Path:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report = build_task_a_pre_block0_data_suitability_report(
        config_path=config_path,
        data_path=data_path,
    )
    report_path = output_root / PRE_BLOCK0_DATA_SUITABILITY_FILENAME
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def prepare_task_a_stage0_mapping(
    *,
    config_path: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    patient_ids: tuple[str, ...] | None = None,
    demo_subset_name: str | None = None,
    demo_subset_rationale: str | None = None,
) -> dict[str, Any]:
    config_bundle = load_task_a_config_bundle(config_path)
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source, selected_patient_ids, run_scope, artifact_state, manifest_demo_subset_name, manifest_demo_subset_rationale = _resolve_prepare_source(
        data_path=data_path,
        patient_ids=patient_ids,
        demo_subset_name=demo_subset_name,
    )
    if demo_subset_rationale is not None and manifest_demo_subset_rationale is None:
        manifest_demo_subset_rationale = demo_subset_rationale

    mapping_summary, dry_run_df = _build_task_a_step1_alignment(
        source=source,
        config_bundle=config_bundle,
    )

    mapping_path = output_root / config_bundle.exports.mapping_manifest_filename
    mapping_path.write_text(
        json.dumps(mapping_summary.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    core_fit_path = output_root / config_bundle.exports.core_fit_dry_run_filename
    dry_run_df.to_csv(core_fit_path, index=False)

    manifest: dict[str, Any] = {
        "task_name": config_bundle.raw_config.get("task_name", "Task A"),
        "config_path": str(config_bundle.config_path),
        "stage0_h5ad": str(Path(data_path).expanduser().resolve()),
        "mapping_manifest": str(mapping_path),
        "core_fit_dry_run": str(core_fit_path),
        "pair_families": list(config_bundle.ordered_pair_family_names),
        "confirmatory_pair_families": [
            family.name for family in config_bundle.ordered_proxy.confirmatory_pair_families
        ],
        "run_scope": run_scope,
        "artifact_state": artifact_state,
        "block0_gate_status": "not_passed",
        "scientific_interpretation_allowed": False,
        "mass_mode": config_bundle.data.mass_mode,
        "fit_surface": "fit_stride",
        "implementation_tier": "canonical_full",
        "evidence_lineage": "canonical_rerun",
    }
    if selected_patient_ids is not None:
        manifest["patient_subset"] = list(selected_patient_ids)
    if manifest_demo_subset_name is not None:
        manifest["demo_subset_name"] = manifest_demo_subset_name
    if manifest_demo_subset_rationale is not None:
        manifest["demo_subset_rationale"] = manifest_demo_subset_rationale
    manifest_path = output_root / config_bundle.exports.prepare_manifest_filename
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Task-A stage0 mapping and STRIDE dry-run artifacts.",
    )
    parser.add_argument("--task-config", required=True, help="Path to task_A config.yaml")
    parser.add_argument("--stage0-h5ad", required=True, help="Path to Stage 0 h5ad")
    parser.add_argument("--output-dir", required=True, help="Output directory for artifacts")
    parser.add_argument(
        "--patient-id",
        action="append",
        default=None,
        help="Filter to specific patient_id(s); may be repeated",
    )
    parser.add_argument(
        "--demo-subset",
        default=None,
        help="Use a named demo subset (e.g. 'alignment_v1') instead of --patient-id",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        patient_ids: tuple[str, ...] | None = None
        demo_subset_name: str | None = None
        if args.demo_subset is not None:
            if args.patient_id:
                raise SystemExit("Cannot combine --demo-subset with --patient-id")
            demo_subset_name = args.demo_subset
        elif args.patient_id:
            patient_ids = tuple(args.patient_id)

        manifest = prepare_task_a_stage0_mapping(
            config_path=args.task_config,
            data_path=args.stage0_h5ad,
            output_dir=args.output_dir,
            patient_ids=patient_ids,
            demo_subset_name=demo_subset_name,
        )
        print(f"Wrote {len(manifest)} manifest keys to {manifest.get('mapping_manifest', args.output_dir)}")
    except (ContractError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()


__all__ = [
    "PRE_BLOCK0_DATA_SUITABILITY_FILENAME",
    "build_task_a_pre_block0_data_suitability_report",
    "prepare_task_a_stage0_mapping",
    "write_task_a_pre_block0_data_suitability_report",
]
