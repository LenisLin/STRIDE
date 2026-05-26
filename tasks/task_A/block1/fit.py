"""Block 1 fitting and CLI entrypoint using the formal STRIDE API."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from stride.errors import ContractError
from stride.observation import FovObservation
from stride.outputs.fit_export import (
    COHORT_ARRAYS_FILENAME,
    COHORT_INDEX_FILENAME,
    MANIFEST_FILENAME,
    PATIENT_ARRAYS_FILENAME,
    PATIENT_INDEX_FILENAME,
    write_stride_native_relation_export,
)

from .analyze import run_block1_analyze
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
from .functions.stride_fit import fit_block1_family, require_block1_fit_ok
from .functions.writers import write_block1_json
from .preprocess import Block1PreprocessBundle, prepare_block1_inputs
from ..config import TaskAConfigBundle, TaskAOrderedPairFamilySpec


NATIVE_RELATION_EXPORT_FILENAMES: tuple[str, ...] = (
    MANIFEST_FILENAME,
    PATIENT_INDEX_FILENAME,
    PATIENT_ARRAYS_FILENAME,
    COHORT_INDEX_FILENAME,
    COHORT_ARRAYS_FILENAME,
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


def _build_family_observations(
    request: Block1RunRequest,
    preprocess_bundle: Block1PreprocessBundle,
) -> tuple[dict[str, tuple[FovObservation, ...]], tuple[str, ...]]:
    family_observations_by_name: dict[str, tuple[FovObservation, ...]] = {}
    observed_patient_ids_from_observations: set[str] = set()
    for family_spec in preprocess_bundle.family_specs:
        observations = preprocess_bundle.observations_by_family.get(family_spec.name, ())
        if not observations:
            raise ContractError(f"Block 1 execute found no observations for family {family_spec.name!r}")
        family_observations_by_name[family_spec.name] = observations
        observed_patient_ids_from_observations.update(
            str(observation.patient_id) for observation in observations
        )
    unmatched_requested_patient_ids = tuple(
        sorted(set(request.patient_ids) - observed_patient_ids_from_observations)
    )
    return family_observations_by_name, unmatched_requested_patient_ids


def run_block1_execute(request: Block1RunRequest) -> Path:
    """Run TC-IM and TC-PT fits, export native relation files, and write manifest."""
    preprocess_bundle = prepare_block1_inputs(
        task_config_path=request.task_config_path,
        stage0_h5ad_path=request.stage0_h5ad_path,
        patient_ids=request.patient_ids,
    )
    _reject_existing_block1_execute_outputs(request.output_dir, preprocess_bundle.family_specs)
    family_observations_by_name, unmatched_requested_patient_ids = _build_family_observations(
        request,
        preprocess_bundle,
    )
    if unmatched_requested_patient_ids:
        raise ContractError(
            "Block 1 execute has unmatched requested patient_ids: "
            + ", ".join(unmatched_requested_patient_ids)
        )

    family_exports: list[Block1FamilyExportRecord] = []
    observed_patient_ids: set[str] = set()
    for family_spec in preprocess_bundle.family_specs:
        fit_result = fit_block1_family(
            family_observations_by_name[family_spec.name],
            family_spec=family_spec,
            state_basis=preprocess_bundle.state_basis,
            device=request.device,
        )
        require_block1_fit_ok(
            fit_result,
            pair_family=family_spec.name,
            run_scope=request.run_scope,
        )
        export_manifest = write_stride_native_relation_export(
            fit_result,
            request.output_dir / family_spec.name,
        )
        observed_patient_ids.update(fit_result.patient_ids)
        family_exports.append(
            Block1FamilyExportRecord(
                pair_family=family_spec.name,
                source_domain=family_spec.source_domain,
                target_domain=family_spec.target_domain,
                claim_role=family_spec.claim_role,
                native_export_manifest_path=export_manifest.manifest_path,
                native_export_manifest_sha256=str(export_manifest.manifest_sha256),
                fit_status=str(fit_result.fit_status),
                cohort_fit_status=str(export_manifest.cohort_fit_status),
                patient_count=int(len(fit_result.patient_ids)),
                patient_record_count=int(export_manifest.patient_record_count),
                cohort_record_count=int(export_manifest.cohort_record_count),
                k_states=int(export_manifest.k_states),
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
        task_config=preprocess_bundle.task_config,
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
    parser.add_argument("--device", default=None, help="Optional torch device forwarded to fit_stride.")
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
