"""Task-local wrapper for the Block 0 locality gate surface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stride.errors import ContractError

from ..block0.bundle import (
    BLOCK_NAME,
    BUNDLE_FILENAME,
    PAIR_METRICS_FILENAME,
    TaskABlock0BundleContract,
)
from ..block0.locality_gate import (
    BLOCK0_SOURCE_DOMAIN,
    BLOCK0_TARGET_DOMAIN,
    DEMO_SUBSET_SCOPE,
    FULL_COHORT_SCOPE,
    NULL_FAMILIES,
    NULL_PAIR_FAMILY,
    PATIENT_SUBSET_SCOPE,
    REAL_FAMILIES,
    REAL_PAIR_FAMILY,
    aggregate_block0_gate,
    empty_block0_pair_metrics_frame,
    evaluate_block0_family_fits,
    load_block0_runtime_config,
)
from ..config import load_task_a_config_bundle
from ..real_data.demo_subset import TaskADemoSubset, resolve_demo_subset
from .prepare import write_task_a_pre_block0_data_suitability_report
from .stride_adapter import load_task_a_dataset_handle


def _resolve_demo_subset(name: str | None) -> TaskADemoSubset | None:
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
    active_patient_ids = patient_ids
    if active_patient_ids is None and demo_subset is not None:
        active_patient_ids = demo_subset.patient_ids
    if active_patient_ids is None:
        return None
    resolved = tuple(dict.fromkeys(str(patient_id) for patient_id in active_patient_ids))
    if not resolved:
        raise ContractError("Task A Block 0 requires at least one patient_id when subset selectors are used")
    return resolved


def _resolve_block0_scope(
    *,
    data_path: str | Path,
    patient_ids: tuple[str, ...] | None,
    demo_subset_name: str | None,
) -> tuple[Any, str, tuple[str, ...] | None, str | None, str | None]:
    handle = load_task_a_dataset_handle(data_path)
    resolved_demo_subset = _resolve_demo_subset(demo_subset_name)
    selected_patient_ids = _resolve_selected_patient_ids(
        patient_ids=patient_ids,
        demo_subset=resolved_demo_subset,
    )
    if selected_patient_ids is None:
        return handle, FULL_COHORT_SCOPE, None, None, None

    all_patient_ids = tuple(sorted(handle.adata.obs["patient_id"].astype(str).unique().tolist()))
    matched_patient_ids = set(selected_patient_ids) & set(all_patient_ids)
    if matched_patient_ids == set(all_patient_ids):
        raise ContractError(
            "Task A Block 0 full-cohort runs must omit subset selectors. "
            "Remove --patient-id/--demo-subset to run the canonical full-cohort locality gate."
        )

    if resolved_demo_subset is not None and tuple(selected_patient_ids) != tuple(resolved_demo_subset.patient_ids):
        raise ContractError(
            "Task A Block 0 may label a run as demo_subset only when patient_ids "
            "exactly match the resolved named subset"
        )

    mask = handle.adata.obs["patient_id"].astype(str).isin(set(selected_patient_ids))
    filtered = handle.adata[mask].copy()
    if resolved_demo_subset is not None:
        return (
            filtered,
            DEMO_SUBSET_SCOPE,
            selected_patient_ids,
            resolved_demo_subset.name,
            resolved_demo_subset.rationale,
        )
    return filtered, PATIENT_SUBSET_SCOPE, selected_patient_ids, None, None


def _n_obs(source: Any) -> int:
    if hasattr(source, "adata"):
        return int(getattr(source.adata, "n_obs", 0))
    return int(getattr(source, "n_obs", 0))


def _build_inputs_payload(
    *,
    config_path: Path,
    data_path: Path,
    run_scope: str,
    selected_patient_ids: tuple[str, ...] | None,
    demo_subset_name: str | None,
    demo_subset_rationale: str | None,
    runtime_config: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task_config": str(config_path),
        "stage0_h5ad": str(data_path),
        "run_scope": run_scope,
        "random_seed": int(runtime_config.random_seed),
        "real_family_definition": {
            "pair_family": REAL_PAIR_FAMILY,
            "source_domain": BLOCK0_SOURCE_DOMAIN,
            "target_domain": BLOCK0_TARGET_DOMAIN,
            "construction": "task_a_stride_adapter_family_slice",
            "fit_surface": "fit_stride",
        },
        "null_family_definition": {
            "pair_family": NULL_PAIR_FAMILY,
            "source_domain": BLOCK0_SOURCE_DOMAIN,
            "target_domain": BLOCK0_TARGET_DOMAIN,
            "construction": (
                "same_anchor_source_with_target_group_reassigned_from_different_patient_"
                "in_same_exact_count_stratum"
            ),
            "stratification_fields": [
                "n_source_observations",
                "n_target_observations",
            ],
            "donor_policy": "seeded_derangement_within_exact_count_strata",
            "singleton_stratum_policy": "emit_null_fit_status_deferred_for_anchor_patient",
        },
        "gate_summary_quantities": {
            "delta_total_continuity_mass": {
                "definition": "sum(A_real) - sum(A_null)",
                "decision_rule": "median > 0 and fraction_real_total_continuity_mass_gt_null > 0.5",
            },
            "delta_total_emergence_mass": {
                "definition": "sum(e_real) - sum(e_null)",
                "decision_rule": "median < 0 and fraction_real_total_emergence_mass_lt_null > 0.5",
            },
        },
    }
    if selected_patient_ids is not None:
        payload["patient_subset"] = list(selected_patient_ids)
    if demo_subset_name is not None:
        payload["demo_subset_name"] = demo_subset_name
    if demo_subset_rationale is not None:
        payload["demo_subset_rationale"] = demo_subset_rationale
    return payload


def run_block0_workflow(
    *,
    config_path: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    patient_ids: tuple[str, ...] | None = None,
    demo_subset_name: str | None = None,
) -> Path:
    config_bundle = load_task_a_config_bundle(config_path)
    if BLOCK_NAME not in config_bundle.enabled_blocks:
        raise ContractError(
            f"Task A config does not enable {BLOCK_NAME}; enabled_blocks={list(config_bundle.enabled_blocks)}"
        )
    runtime_config = load_block0_runtime_config(config_bundle)

    resolved_data_path = Path(data_path).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    pre_block0_path = write_task_a_pre_block0_data_suitability_report(
        config_path=config_bundle.config_path,
        data_path=resolved_data_path,
        output_dir=output_root,
    )
    pre_block0_data_suitability = json.loads(pre_block0_path.read_text(encoding="utf-8"))

    source, run_scope, selected_patient_ids, resolved_demo_subset_name, resolved_demo_subset_rationale = _resolve_block0_scope(
        data_path=resolved_data_path,
        patient_ids=patient_ids,
        demo_subset_name=demo_subset_name,
    )

    pair_metrics_df = empty_block0_pair_metrics_frame()
    n_eligible_patients = 0
    if (
        str(pre_block0_data_suitability.get("artifact_state", "")) == "contract_passed"
        and _n_obs(source) > 0
    ):
        pair_metrics_df, n_eligible_patients = evaluate_block0_family_fits(
            source,
            config_bundle=config_bundle,
            runtime_config=runtime_config,
            run_scope=run_scope,
        )

    gate_result = aggregate_block0_gate(
        pair_metrics_df=pair_metrics_df,
        n_eligible_patients=n_eligible_patients,
        run_scope=run_scope,
        pre_block0_artifact_state=str(pre_block0_data_suitability.get("artifact_state", "")),
    )

    pair_metrics_path = output_root / PAIR_METRICS_FILENAME
    pair_metrics_df.to_csv(pair_metrics_path, index=False)

    bundle = TaskABlock0BundleContract(
        block=BLOCK_NAME,
        status=gate_result.status,
        artifact_state=gate_result.artifact_state,
        implementation_tier="canonical_full",
        evidence_lineage="canonical_rerun",
        run_scope=run_scope,
        block0_passed=gate_result.block0_passed,
        config_fingerprint=config_bundle.config_fingerprint,
        config_path=config_bundle.config_path,
        stage0_h5ad=resolved_data_path,
        output_dir=output_root,
        bundle_path=output_root / BUNDLE_FILENAME,
        pair_metrics_path=pair_metrics_path,
        real_families=REAL_FAMILIES,
        null_families=NULL_FAMILIES,
        pre_block0_data_suitability=pre_block0_data_suitability,
        gate_checks=gate_result.gate_checks,
        metrics_summary=gate_result.metrics_summary,
        failure_reasons=gate_result.failure_reasons,
        inputs=_build_inputs_payload(
            config_path=config_bundle.config_path,
            data_path=resolved_data_path,
            run_scope=run_scope,
            selected_patient_ids=selected_patient_ids,
            demo_subset_name=resolved_demo_subset_name,
            demo_subset_rationale=resolved_demo_subset_rationale,
            runtime_config=runtime_config,
        ),
    )
    bundle.bundle_path.write_text(
        json.dumps(bundle.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return bundle.bundle_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task-A Block 0 through the task-local workflow wrapper.")
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--stage0-h5ad", required=True)
    parser.add_argument("--output-dir", required=True)
    scope_group = parser.add_mutually_exclusive_group()
    scope_group.add_argument(
        "--patient-id",
        action="append",
        default=None,
        help="Filter to specific patient_id(s); may be repeated for non-passing sidecar runs",
    )
    scope_group.add_argument(
        "--demo-subset",
        default=None,
        help="Use a named demo subset instead of --patient-id for a non-passing sidecar run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        run_block0_workflow(
            config_path=args.task_config,
            data_path=args.stage0_h5ad,
            output_dir=args.output_dir,
            patient_ids=None if args.patient_id is None else tuple(args.patient_id),
            demo_subset_name=args.demo_subset,
        )
    except (ContractError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()


__all__ = ["BLOCK_NAME", "run_block0_workflow"]
