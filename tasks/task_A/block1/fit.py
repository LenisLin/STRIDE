"""Block 1 fitting and CLI entrypoint using the formal STRIDE API."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from stride.errors import ContractError

from ..config import (
    TaskAConfigBundle,
    TaskAOrderedPairFamilySpec,
    load_task_a_config_bundle,
)
from ..workflows.stride_adapter import load_task_a_adata, prepare_task_a_pair_adata
from .analyze import run_block1_analyze
from .functions.observations import resolve_block1_confirmatory_families
from .functions.result_adapter import (
    BLOCK1_NATIVE_RESULT_ARRAYS_FILENAME,
    BLOCK1_NATIVE_RESULT_MANIFEST_FILENAME,
    write_block1_native_fit_result,
)
from .functions.schemas import (
    BLOCK1_EXECUTE_MANIFEST_FILENAME,
    BLOCK1_LIVE_ID,
    EXECUTE_MANIFEST_REQUIRED_FIELDS,
    READINESS_STATUS_DIAGNOSTIC,
    READINESS_STATUS_EVIDENCE_READY,
    RUN_SCOPE_FULL_COHORT,
    RUN_SCOPE_PATIENT_SUBSET,
    Block1FamilyExportRecord,
    Block1RunRequest,
)
from .functions.stride_fit import (
    fit_block1_family,
    require_block1_fit_ok,
    summarize_fit_status_for_manifest,
)
from .functions.writers import write_block1_json

NATIVE_RELATION_EXPORT_FILENAMES: tuple[str, ...] = (
    BLOCK1_NATIVE_RESULT_MANIFEST_FILENAME,
    BLOCK1_NATIVE_RESULT_ARRAYS_FILENAME,
)


def build_block1_run_request(
    *,
    task_config_path: Path,
    stage0_h5ad_path: Path,
    output_dir: Path,
    patient_ids: Sequence[str] = (),
    confirm_full_cohort: bool = False,
    device: object | None = None,
) -> Block1RunRequest:
    """Normalize execute inputs and assign full_cohort/patient_subset scope."""
    normalized_patient_ids = tuple(dict.fromkeys(str(patient_id) for patient_id in patient_ids))
    run_scope = RUN_SCOPE_FULL_COHORT if not normalized_patient_ids else RUN_SCOPE_PATIENT_SUBSET
    if run_scope == RUN_SCOPE_FULL_COHORT and not confirm_full_cohort:
        raise ContractError("Block 1 full-cohort execute requires --confirm-full-cohort")
    return Block1RunRequest(
        task_config_path=Path(task_config_path).resolve(),
        stage0_h5ad_path=Path(stage0_h5ad_path).resolve(),
        output_dir=Path(output_dir).resolve(),
        patient_ids=normalized_patient_ids,
        run_scope=run_scope,
        device=device,
    )


def build_block1_execute_manifest_payload(
    *,
    request: Block1RunRequest,
    task_config: TaskAConfigBundle,
    family_exports: Sequence[Block1FamilyExportRecord],
) -> dict[str, object]:
    """Build the thin Block 1 execute routing manifest."""
    return {
        "task_name": BLOCK1_LIVE_ID,
        "phase": "execute",
        "config_path": str(task_config.config_path),
        "config_fingerprint": task_config.config_fingerprint,
        "stage0_h5ad": str(request.stage0_h5ad_path),
        "run_scope": request.run_scope,
        "patient_ids": list(request.patient_ids),
        "requested_device": None if request.device is None else str(request.device),
        "confirmatory_pair_families": [record.pair_family for record in family_exports],
        "family_exports": [record.to_json_dict() for record in family_exports],
        "readiness_status": (
            READINESS_STATUS_EVIDENCE_READY
            if request.run_scope == RUN_SCOPE_FULL_COHORT
            else READINESS_STATUS_DIAGNOSTIC
        ),
        "scientific_interpretation_allowed": False,
        "prohibited_outputs": ["p_values", "fdr", "figures", "annotations"],
    }


def write_block1_execute_manifest(
    payload: Mapping[str, object],
    output_dir: Path,
) -> Path:
    """Validate and atomically write block1_execute_manifest.json."""
    return write_block1_json(
        payload,
        Path(output_dir) / BLOCK1_EXECUTE_MANIFEST_FILENAME,
        required_keys=EXECUTE_MANIFEST_REQUIRED_FIELDS,
    )


def _reject_existing_block1_execute_outputs(
    output_dir: Path,
    family_specs: Sequence[TaskAOrderedPairFamilySpec],
) -> None:
    output_path = Path(output_dir)
    expected_paths = [output_path / BLOCK1_EXECUTE_MANIFEST_FILENAME]
    for family_spec in family_specs:
        family_dir = output_path / str(family_spec.name)
        expected_paths.extend(family_dir / filename for filename in NATIVE_RELATION_EXPORT_FILENAMES)
    existing = tuple(path for path in expected_paths if path.exists())
    if existing:
        raise ContractError(
            "Block 1 execute output already exists and will not be overwritten: "
            + ", ".join(str(path) for path in existing)
        )


def _validate_requested_patients_available(
    request: Block1RunRequest,
) -> None:
    if not request.patient_ids:
        return
    adata = load_task_a_adata(request.stage0_h5ad_path, backed=True)
    available = set(adata.obs["patient_id"].astype(str).unique().tolist())
    missing = tuple(patient_id for patient_id in request.patient_ids if patient_id not in available)
    if missing:
        raise ContractError(
            "Block 1 execute has unmatched requested patient_ids: "
            + ", ".join(missing)
        )


def run_block1_execute(request: Block1RunRequest) -> Path:
    """Run TC-IM and TC-PT fits, export native relation files, and write manifest."""
    task_config = load_task_a_config_bundle(request.task_config_path)
    family_specs = resolve_block1_confirmatory_families(task_config)
    _reject_existing_block1_execute_outputs(request.output_dir, family_specs)
    _validate_requested_patients_available(request)

    family_exports: list[Block1FamilyExportRecord] = []
    observed_patient_ids: set[str] = set()
    for family_spec in family_specs:
        pair_adata = prepare_task_a_pair_adata(
            request.stage0_h5ad_path,
            family_spec,
            patient_ids=request.patient_ids or None,
            backed=False,
            copy_adata=True,
        )
        fit_result = fit_block1_family(
            pair_adata,
            family_spec=family_spec,
            device=request.device,
        )
        require_block1_fit_ok(
            fit_result,
            pair_family=family_spec.name,
            run_scope=request.run_scope,
        )
        summary = summarize_fit_status_for_manifest(
            fit_result,
            pair_family=family_spec.name,
        )
        export_manifest = write_block1_native_fit_result(
            fit_result,
            request.output_dir / family_spec.name,
        )
        for relation in fit_result.relations.values():
            observed_patient_ids.update(str(patient_id) for patient_id in relation.patient_ids)
        family_exports.append(
            Block1FamilyExportRecord(
                pair_family=family_spec.name,
                source_domain=family_spec.source_domain,
                target_domain=family_spec.target_domain,
                claim_role=family_spec.claim_role,
                native_export_manifest_path=export_manifest["manifest_path"],
                native_export_manifest_sha256=str(export_manifest["manifest_sha256"]),
                fit_status="ok",
                cohort_fit_status="not_exported_by_stride_tl_fit_v1",
                patient_count=int(export_manifest["patient_count"]),
                patient_record_count=int(export_manifest["patient_record_count"]),
                cohort_record_count=int(export_manifest["cohort_record_count"]),
                k_states=int(export_manifest["k_states"]),
                fit_surface=str(summary["fit_surface"]),
            )
        )

    payload = build_block1_execute_manifest_payload(
        request=Block1RunRequest(
            task_config_path=request.task_config_path,
            stage0_h5ad_path=request.stage0_h5ad_path,
            output_dir=request.output_dir,
            patient_ids=tuple(sorted(observed_patient_ids)),
            run_scope=request.run_scope,
            device=request.device,
        ),
        task_config=task_config,
        family_exports=tuple(family_exports),
    )
    return write_block1_execute_manifest(payload, request.output_dir)


def _add_execute_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("execute")
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--stage0-h5ad", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patient-id", action="append", default=[])
    parser.add_argument("--confirm-full-cohort", action="store_true")
    parser.add_argument("--device", default=None, help="Optional torch device forwarded to stride.tl.fit.")
    parser.set_defaults(command="execute")


def _add_analyze_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("analyze")
    parser.add_argument("--execute-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.set_defaults(command="analyze")


def build_parser() -> argparse.ArgumentParser:
    """Create the Block 1 parser with only execute and analyze subcommands."""
    parser = argparse.ArgumentParser(prog="python -m tasks.task_A.block1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_execute_parser(subparsers)
    _add_analyze_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse Block 1 command arguments and dispatch to execute/analyze."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "execute":
        request = build_block1_run_request(
            task_config_path=Path(args.task_config),
            stage0_h5ad_path=Path(args.stage0_h5ad),
            output_dir=Path(args.output_dir),
            patient_ids=tuple(args.patient_id),
            confirm_full_cohort=bool(args.confirm_full_cohort),
            device=args.device,
        )
        run_block1_execute(request)
        return 0
    run_block1_analyze(
        execute_manifest_path=Path(args.execute_manifest),
        output_dir=Path(args.output_dir),
    )
    return 0


__all__ = [
    "build_block1_execute_manifest_payload",
    "build_block1_run_request",
    "build_parser",
    "main",
    "run_block1_execute",
    "write_block1_execute_manifest",
]
