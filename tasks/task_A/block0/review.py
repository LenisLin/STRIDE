"""Objective Block 0 review-surface writers for Task A packets."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from stride.errors import ContractError


BLOCK0_OBJECTIVE_REVIEW_MANIFEST_FILENAME = "block0_objective_review_manifest.json"
BLOCK0_PATIENT_REVIEW_FILENAME = "block0_patient_review_table.csv"
BLOCK0_FAMILY_SUMMARY_FILENAME = "block0_family_summary.csv"
BLOCK0_GATE_SUMMARY_FILENAME = "block0_gate_summary.csv"
BLOCK0_NULL_PROVENANCE_FILENAME = "block0_null_provenance.csv"
BLOCK0_REPRODUCIBILITY_METADATA_FILENAME = "block0_reproducibility_metadata.json"
BLOCK0_HUMAN_INDEX_FILENAME = "BLOCK0_RESULTS_INDEX.md"
REVIEW_DIRNAME = "review"

CURRENT_BUNDLE_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "block",
        "status",
        "artifact_state",
        "implementation_tier",
        "evidence_lineage",
        "run_scope",
        "block0_passed",
        "config_path",
        "stage0_h5ad",
        "output_dir",
        "bundle_path",
        "pair_metrics_path",
        "real_families",
        "null_families",
        "pre_block0_data_suitability",
        "gate_checks",
        "metrics_summary",
        "failure_reasons",
        "inputs",
    }
)
CURRENT_PAIR_METRICS_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "comparison_id",
        "run_scope",
        "pair_family",
        "null_family",
        "anchor_patient_id",
        "null_target_donor_patient_id",
        "selection_seed",
        "null_assignment_status",
        "real_fit_status",
        "null_fit_status",
        "delta_total_continuity_mass",
        "delta_total_depletion_mass",
        "delta_total_emergence_mass",
    }
)


@dataclass(frozen=True)
class Block0ReviewArtifact:
    artifact_name: str
    packet_relative_path: str
    artifact_kind: str
    format: str
    claim_scope: str
    review_role: str
    proof_carrying_status: str
    analysis_level: str
    family_surface_role: str
    rows_represent: str
    columns_represent: str
    source_workflow: str
    source_manifest_or_bundle: str
    notes: str
    review_rank: int


@dataclass(frozen=True)
class Block0ReviewSurface:
    schema_variant: str
    manifest_path: Path
    human_index_path: Path
    artifacts: tuple[Block0ReviewArtifact, ...]
    missing_artifacts: tuple[dict[str, str], ...]


def is_current_block0_bundle_payload(payload: dict[str, Any]) -> bool:
    return CURRENT_BUNDLE_REQUIRED_FIELDS.issubset(set(payload))


def is_current_block0_pair_metrics_columns(columns: list[str] | tuple[str, ...]) -> bool:
    return CURRENT_PAIR_METRICS_REQUIRED_COLUMNS.issubset({str(column) for column in columns})


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _load_json_dict(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = _resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} was not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must be a JSON object: {resolved}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _stringify_optional(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text == "nan" else text


def _int_mapping(payload: object) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {"ok": 0, "deferred": 0, "failed": 0}
    return {
        "ok": int(payload.get("ok", 0) or 0),
        "deferred": int(payload.get("deferred", 0) or 0),
        "failed": int(payload.get("failed", 0) or 0),
    }


def _build_patient_review_table(pair_metrics_df: pd.DataFrame) -> pd.DataFrame:
    patient_df = pair_metrics_df.copy()
    patient_df["paired_gate_eligible"] = (
        patient_df["null_assignment_status"].astype(str).eq("assigned")
        & patient_df["real_fit_status"].astype(str).eq("ok")
        & patient_df["null_fit_status"].astype(str).eq("ok")
    )
    patient_df["real_total_continuity_mass_gt_null"] = patient_df["delta_total_continuity_mass"] > 0
    patient_df["real_total_depletion_mass_lt_null"] = patient_df["delta_total_depletion_mass"] < 0
    patient_df["real_total_emergence_mass_lt_null"] = patient_df["delta_total_emergence_mass"] < 0
    ordered_columns = [
        "comparison_id",
        "anchor_patient_id",
        "pair_family",
        "null_family",
        "source_domain",
        "target_domain",
        "n_source_observations",
        "n_target_observations",
        "count_stratum_key",
        "selection_seed",
        "null_target_donor_patient_id",
        "null_assignment_status",
        "null_assignment_reason",
        "real_fit_status",
        "null_fit_status",
        "real_defer_reason",
        "null_defer_reason",
        "paired_gate_eligible",
        "real_total_continuity_mass",
        "null_total_continuity_mass",
        "delta_total_continuity_mass",
        "real_total_continuity_mass_gt_null",
        "real_total_depletion_mass",
        "null_total_depletion_mass",
        "delta_total_depletion_mass",
        "real_total_depletion_mass_lt_null",
        "real_total_emergence_mass",
        "null_total_emergence_mass",
        "delta_total_emergence_mass",
        "real_total_emergence_mass_lt_null",
    ]
    return patient_df.loc[:, ordered_columns].sort_values(
        ["anchor_patient_id", "comparison_id"], kind="mergesort"
    ).reset_index(drop=True)


def _build_null_provenance_table(pair_metrics_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "anchor_patient_id",
        "null_target_donor_patient_id",
        "pair_family",
        "null_family",
        "source_domain",
        "target_domain",
        "n_source_observations",
        "n_target_observations",
        "count_stratum_key",
        "selection_seed",
        "null_assignment_status",
        "null_assignment_reason",
        "real_fit_status",
        "null_fit_status",
    ]
    return pair_metrics_df.loc[:, columns].sort_values(
        ["count_stratum_key", "anchor_patient_id"], kind="mergesort"
    ).reset_index(drop=True)


def _build_family_summary_table(bundle_payload: dict[str, Any]) -> pd.DataFrame:
    metrics_summary = dict(bundle_payload.get("metrics_summary", {}))
    inputs = dict(bundle_payload.get("inputs", {}))
    real_definition = dict(inputs.get("real_family_definition", {}))
    null_definition = dict(inputs.get("null_family_definition", {}))
    real_summary = dict(metrics_summary.get("real_family", {}))
    null_summary = dict(metrics_summary.get("null_family", {}))

    rows: list[dict[str, Any]] = []
    for surface_role, family_summary, family_definition in (
        ("real_family", real_summary, real_definition),
        ("null_family", null_summary, null_definition),
    ):
        fit_status_counts = _int_mapping(family_summary.get("fit_status_counts", {}))
        rows.append(
            {
                "surface_role": surface_role,
                "family_name": _stringify_optional(
                    family_summary.get("family_name", family_definition.get("pair_family"))
                ),
                "source_domain": _stringify_optional(family_definition.get("source_domain")),
                "target_domain": _stringify_optional(family_definition.get("target_domain")),
                "n_patients": int(family_summary.get("n_patients", 0) or 0),
                "fit_status_ok": fit_status_counts["ok"],
                "fit_status_deferred": fit_status_counts["deferred"],
                "fit_status_failed": fit_status_counts["failed"],
                "median_total_continuity_mass": family_summary.get("median_total_continuity_mass"),
                "median_total_depletion_mass": family_summary.get("median_total_depletion_mass"),
                "median_total_emergence_mass": family_summary.get("median_total_emergence_mass"),
            }
        )
    return pd.DataFrame.from_records(rows)


def _build_gate_summary_table(bundle_payload: dict[str, Any]) -> pd.DataFrame:
    metrics_summary = dict(bundle_payload.get("metrics_summary", {}))
    paired_comparisons = dict(metrics_summary.get("paired_comparisons", {}))
    gate_checks = dict(bundle_payload.get("gate_checks", {}))
    real_family = _stringify_optional(
        dict(bundle_payload.get("inputs", {})).get("real_family_definition", {}).get("pair_family")
    )
    null_family = _stringify_optional(
        dict(bundle_payload.get("inputs", {})).get("null_family_definition", {}).get("pair_family")
    )
    shared_support = {
        "paired_support": int(paired_comparisons.get("paired_support", 0) or 0),
        "required_support": int(paired_comparisons.get("required_support", 0) or 0),
    }
    specs = (
        {
            "quantity_name": "delta_total_continuity_mass",
            "review_role": "proof_carrying",
            "expected_direction": "positive",
            "participates_in_pass_decision": True,
            "fraction_metric_name": "fraction_real_total_continuity_mass_gt_null",
            "median_metric_name": "median_delta_total_continuity_mass",
            "median_check_name": "median_delta_total_continuity_mass_positive",
            "fraction_check_name": "fraction_real_total_continuity_mass_gt_null_above_half",
        },
        {
            "quantity_name": "delta_total_depletion_mass",
            "review_role": "supportive",
            "expected_direction": "negative",
            "participates_in_pass_decision": False,
            "fraction_metric_name": "fraction_real_total_depletion_mass_lt_null",
            "median_metric_name": "median_delta_total_depletion_mass",
            "median_check_name": "",
            "fraction_check_name": "",
        },
        {
            "quantity_name": "delta_total_emergence_mass",
            "review_role": "proof_carrying",
            "expected_direction": "negative",
            "participates_in_pass_decision": True,
            "fraction_metric_name": "fraction_real_total_emergence_mass_lt_null",
            "median_metric_name": "median_delta_total_emergence_mass",
            "median_check_name": "median_delta_total_emergence_mass_negative",
            "fraction_check_name": "fraction_real_total_emergence_mass_lt_null_above_half",
        },
    )

    rows: list[dict[str, Any]] = []
    for spec in specs:
        median_check = dict(gate_checks.get(spec["median_check_name"], {})) if spec["median_check_name"] else {}
        fraction_check = dict(gate_checks.get(spec["fraction_check_name"], {})) if spec["fraction_check_name"] else {}
        rows.append(
            {
                "quantity_name": spec["quantity_name"],
                "review_role": spec["review_role"],
                "real_family": real_family,
                "null_family": null_family,
                "comparison_definition": f"{spec['quantity_name']} = real - null",
                "expected_direction": spec["expected_direction"],
                "median_delta_value": paired_comparisons.get(spec["median_metric_name"]),
                "fraction_in_expected_direction": paired_comparisons.get(spec["fraction_metric_name"]),
                "fraction_metric_name": spec["fraction_metric_name"],
                "paired_support": shared_support["paired_support"],
                "required_support": shared_support["required_support"],
                "participates_in_pass_decision": spec["participates_in_pass_decision"],
                "median_check_name": spec["median_check_name"],
                "median_check_passed": median_check.get("passed"),
                "median_threshold": median_check.get("threshold"),
                "fraction_check_name": spec["fraction_check_name"],
                "fraction_check_passed": fraction_check.get("passed"),
                "fraction_threshold": fraction_check.get("threshold"),
            }
        )
    return pd.DataFrame.from_records(rows)


def _build_reproducibility_payload(
    *,
    bundle_payload: dict[str, Any],
    pair_metrics_df: pd.DataFrame,
    schema_variant: str,
) -> dict[str, Any]:
    inputs = dict(bundle_payload.get("inputs", {}))
    if "count_stratum_key" in pair_metrics_df.columns:
        count_strata = (
            pair_metrics_df.groupby("count_stratum_key", dropna=False)
            .size()
            .reset_index(name="n_anchor_patients")
            .sort_values(["count_stratum_key"], kind="mergesort")
        )
        count_strata_records = count_strata.to_dict(orient="records")
    else:
        count_strata_records = []
    selection_seeds = sorted(
        {
            int(seed)
            for seed in pair_metrics_df.get("selection_seed", pd.Series(dtype="int64")).dropna().tolist()
        }
    )
    return {
        "workflow_name": "write_block0_review_surface",
        "schema_variant": schema_variant,
        "block": bundle_payload.get("block"),
        "status": bundle_payload.get("status"),
        "artifact_state": bundle_payload.get("artifact_state"),
        "block0_passed": bool(bundle_payload.get("block0_passed", False)),
        "run_scope": _stringify_optional(bundle_payload.get("run_scope")),
        "bundle_path": _stringify_optional(bundle_payload.get("bundle_path")),
        "pair_metrics_path": _stringify_optional(bundle_payload.get("pair_metrics_path")),
        "config_path": _stringify_optional(bundle_payload.get("config_path")),
        "config_fingerprint": _stringify_optional(bundle_payload.get("config_fingerprint")),
        "stage0_h5ad": _stringify_optional(bundle_payload.get("stage0_h5ad")),
        "random_seed": inputs.get("random_seed"),
        "real_family_definition": dict(inputs.get("real_family_definition", {})),
        "null_family_definition": dict(inputs.get("null_family_definition", {})),
        "gate_summary_quantities": dict(inputs.get("gate_summary_quantities", {})),
        "selection_seeds": selection_seeds,
        "n_unique_selection_seeds": len(selection_seeds),
        "count_strata_distribution": count_strata_records,
        "pair_metrics_row_count": int(len(pair_metrics_df)),
        "pair_metrics_columns": [str(column) for column in pair_metrics_df.columns.tolist()],
    }


def _write_human_index(
    *,
    block0_dir: Path,
    schema_variant: str,
    bundle_payload: dict[str, Any],
    generated_artifacts: list[Block0ReviewArtifact],
    missing_artifacts: list[dict[str, str]],
) -> Path:
    available_paths = {artifact.packet_relative_path for artifact in generated_artifacts}
    inspect_first = [
        "block0/bundle/block0_bundle.json",
        "block0/review/block0_gate_summary.csv",
        "block0/review/block0_patient_review_table.csv",
        "block0/review/block0_null_provenance.csv",
        "block0/review/block0_reproducibility_metadata.json",
        "block0/block0_review_index.csv",
    ]
    lines = [
        "# Block 0 Objective Review Surface",
        "",
        "This surface repackages the live Block 0 outputs without biological interpretation.",
        "",
        "## Current Run",
        f"- Status: `{bundle_payload.get('status', 'unknown')}`",
        f"- Artifact state: `{bundle_payload.get('artifact_state', 'unknown')}`",
        f"- `block0_passed`: `{bool(bundle_payload.get('block0_passed', False))}`",
        f"- Schema variant: `{schema_variant}`",
        "",
        "## Inspect First",
    ]
    for relative_path in inspect_first:
        candidate = block0_dir.parent / relative_path
        if candidate.exists() or relative_path in available_paths or relative_path == "block0/block0_review_index.csv":
            lines.append(f"- `{relative_path}`")
    lines.extend(
        [
            "",
            "## Generated Review Artifacts",
        ]
    )
    if not generated_artifacts:
        lines.append("- None.")
    else:
        for artifact in sorted(generated_artifacts, key=lambda item: (item.review_rank, item.artifact_name)):
            lines.append(f"- `{artifact.packet_relative_path}`: {artifact.notes}")
    lines.extend(
        [
            "",
            "## Missing Review Artifacts",
        ]
    )
    if not missing_artifacts:
        lines.append("- None.")
    else:
        for artifact in missing_artifacts:
            lines.append(f"- `{artifact['artifact_name']}`: {artifact['reason']}")
    human_index_path = block0_dir / BLOCK0_HUMAN_INDEX_FILENAME
    human_index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return human_index_path


def write_block0_review_surface(
    *,
    block0_bundle_path: str | Path,
    output_dir: str | Path,
) -> Block0ReviewSurface:
    bundle_path = _resolve_path(block0_bundle_path)
    bundle_payload = _load_json_dict(bundle_path, label="Task A Block 0 bundle")
    pair_metrics_path = _resolve_path(
        bundle_payload.get("pair_metrics_path", bundle_path.parent / "block0_pair_metrics.csv")
    )
    pair_metrics_df = pd.read_csv(pair_metrics_path)

    schema_variant = (
        "current_contract_available"
        if is_current_block0_bundle_payload(bundle_payload)
        and is_current_block0_pair_metrics_columns(pair_metrics_df.columns.tolist())
        else "legacy_live_run"
    )

    block0_dir = _resolve_path(output_dir)
    review_dir = block0_dir / REVIEW_DIRNAME
    review_dir.mkdir(parents=True, exist_ok=True)

    generated_artifacts: list[Block0ReviewArtifact] = []
    missing_artifacts: list[dict[str, str]] = []

    reproducibility_path = review_dir / BLOCK0_REPRODUCIBILITY_METADATA_FILENAME
    _write_json(
        reproducibility_path,
        _build_reproducibility_payload(
            bundle_payload=bundle_payload,
            pair_metrics_df=pair_metrics_df,
            schema_variant=schema_variant,
        ),
    )
    generated_artifacts.append(
        Block0ReviewArtifact(
            artifact_name=BLOCK0_REPRODUCIBILITY_METADATA_FILENAME,
            packet_relative_path=f"block0/{REVIEW_DIRNAME}/{BLOCK0_REPRODUCIBILITY_METADATA_FILENAME}",
            artifact_kind="manifest",
            format="json",
            claim_scope="provenance",
            review_role="provenance",
            proof_carrying_status="none",
            analysis_level="run_level",
            family_surface_role="mixed",
            rows_represent="Single JSON object for one Block 0 reproducibility/provenance summary.",
            columns_represent=(
                "Top-level keys record the canonical bundle paths, config fingerprint, seed metadata, "
                "family definitions, gate quantity definitions, and observed count strata."
            ),
            source_workflow="write_block0_review_surface",
            source_manifest_or_bundle=str(bundle_path),
            notes="Seed and provenance metadata extracted from the live Block 0 bundle and pair-metrics surface.",
            review_rank=24,
        )
    )

    if schema_variant == "current_contract_available":
        patient_review_path = review_dir / BLOCK0_PATIENT_REVIEW_FILENAME
        family_summary_path = review_dir / BLOCK0_FAMILY_SUMMARY_FILENAME
        gate_summary_path = review_dir / BLOCK0_GATE_SUMMARY_FILENAME
        null_provenance_path = review_dir / BLOCK0_NULL_PROVENANCE_FILENAME

        _write_csv(patient_review_path, _build_patient_review_table(pair_metrics_df))
        _write_csv(family_summary_path, _build_family_summary_table(bundle_payload))
        _write_csv(gate_summary_path, _build_gate_summary_table(bundle_payload))
        _write_csv(null_provenance_path, _build_null_provenance_table(pair_metrics_df))

        generated_artifacts.extend(
            [
                Block0ReviewArtifact(
                    artifact_name=BLOCK0_PATIENT_REVIEW_FILENAME,
                    packet_relative_path=f"block0/{REVIEW_DIRNAME}/{BLOCK0_PATIENT_REVIEW_FILENAME}",
                    artifact_kind="table",
                    format="csv",
                    claim_scope="supportive",
                    review_role="proof_carrying",
                    proof_carrying_status="all",
                    analysis_level="patient_level",
                    family_surface_role="comparison_surface",
                    rows_represent="Rows represent anchor-patient Block 0 real-versus-null comparison records.",
                    columns_represent=(
                        "Columns record real/null summary totals, deltas, donor provenance, fit statuses, "
                        "count strata, and contract-direction indicator flags."
                    ),
                    source_workflow="write_block0_review_surface",
                    source_manifest_or_bundle=str(bundle_path),
                    notes="Patient-level review table reorganized from the current-contract block0_pair_metrics.csv.",
                    review_rank=20,
                ),
                Block0ReviewArtifact(
                    artifact_name=BLOCK0_FAMILY_SUMMARY_FILENAME,
                    packet_relative_path=f"block0/{REVIEW_DIRNAME}/{BLOCK0_FAMILY_SUMMARY_FILENAME}",
                    artifact_kind="table",
                    format="csv",
                    claim_scope="supportive",
                    review_role="supportive",
                    proof_carrying_status="none",
                    analysis_level="family_level",
                    family_surface_role="mixed",
                    rows_represent="Rows represent the real-family and null-family Block 0 summary surfaces.",
                    columns_represent=(
                        "Columns record family identity, source/target domains, patient counts, fit-status counts, "
                        "and family-level median continuity/depletion/emergence totals."
                    ),
                    source_workflow="write_block0_review_surface",
                    source_manifest_or_bundle=str(bundle_path),
                    notes="Family-level summary table extracted from block0_bundle.json metrics_summary.",
                    review_rank=21,
                ),
                Block0ReviewArtifact(
                    artifact_name=BLOCK0_GATE_SUMMARY_FILENAME,
                    packet_relative_path=f"block0/{REVIEW_DIRNAME}/{BLOCK0_GATE_SUMMARY_FILENAME}",
                    artifact_kind="table",
                    format="csv",
                    claim_scope="supportive",
                    review_role="proof_carrying",
                    proof_carrying_status="all",
                    analysis_level="cohort_level",
                    family_surface_role="comparison_surface",
                    rows_represent="Rows represent cohort-level paired real-versus-null Block 0 summary quantities.",
                    columns_represent=(
                        "Columns record expected direction, median deltas, fraction-in-direction values, "
                        "support counts, and contract check names/thresholds/passed flags."
                    ),
                    source_workflow="write_block0_review_surface",
                    source_manifest_or_bundle=str(bundle_path),
                    notes="Cohort-level gate summary table faithful to the live Block 0 contract quantities.",
                    review_rank=22,
                ),
                Block0ReviewArtifact(
                    artifact_name=BLOCK0_NULL_PROVENANCE_FILENAME,
                    packet_relative_path=f"block0/{REVIEW_DIRNAME}/{BLOCK0_NULL_PROVENANCE_FILENAME}",
                    artifact_kind="table",
                    format="csv",
                    claim_scope="supportive",
                    review_role="supportive",
                    proof_carrying_status="none",
                    analysis_level="patient_level",
                    family_surface_role="null_family",
                    rows_represent="Rows represent anchor-patient null-family donor assignments for the Block 0 run.",
                    columns_represent=(
                        "Columns record donor patient ids, count strata, source/target observation counts, "
                        "assignment seed, and fit-status fields."
                    ),
                    source_workflow="write_block0_review_surface",
                    source_manifest_or_bundle=str(bundle_path),
                    notes="Null-family provenance table extracted from the current-contract block0_pair_metrics.csv.",
                    review_rank=23,
                ),
            ]
        )
    else:
        missing_artifacts.extend(
            [
                {
                    "artifact_name": BLOCK0_PATIENT_REVIEW_FILENAME,
                    "reason": "Current-contract patient-level Block 0 review rows could not be derived from the legacy live schema.",
                },
                {
                    "artifact_name": BLOCK0_FAMILY_SUMMARY_FILENAME,
                    "reason": "Current-contract family-level summary rows could not be derived from the legacy live schema.",
                },
                {
                    "artifact_name": BLOCK0_GATE_SUMMARY_FILENAME,
                    "reason": "Current-contract cohort-level gate rows could not be derived from the legacy live schema.",
                },
                {
                    "artifact_name": BLOCK0_NULL_PROVENANCE_FILENAME,
                    "reason": "Current-contract null-family provenance rows could not be derived from the legacy live schema.",
                },
            ]
        )

    manifest_payload = {
        "workflow_name": "write_block0_review_surface",
        "scientific_interpretation_allowed": False,
        "schema_variant": schema_variant,
        "bundle_path": str(bundle_path),
        "pair_metrics_path": str(pair_metrics_path),
        "block0_status": _stringify_optional(bundle_payload.get("status")),
        "artifact_state": _stringify_optional(bundle_payload.get("artifact_state")),
        "block0_passed": bool(bundle_payload.get("block0_passed", False)),
        "review_index_path": str(block0_dir / "block0_review_index.csv"),
        "human_index_path": str(block0_dir / BLOCK0_HUMAN_INDEX_FILENAME),
        "generated_artifacts": [
            {
                "artifact_name": artifact.artifact_name,
                "packet_relative_path": artifact.packet_relative_path,
                "artifact_kind": artifact.artifact_kind,
                "review_role": artifact.review_role,
                "proof_carrying_status": artifact.proof_carrying_status,
                "analysis_level": artifact.analysis_level,
                "family_surface_role": artifact.family_surface_role,
            }
            for artifact in sorted(generated_artifacts, key=lambda item: (item.review_rank, item.artifact_name))
        ],
        "missing_artifacts": missing_artifacts,
    }
    manifest_path = review_dir / BLOCK0_OBJECTIVE_REVIEW_MANIFEST_FILENAME
    _write_json(manifest_path, manifest_payload)
    generated_artifacts.append(
        Block0ReviewArtifact(
            artifact_name=BLOCK0_OBJECTIVE_REVIEW_MANIFEST_FILENAME,
            packet_relative_path=f"block0/{REVIEW_DIRNAME}/{BLOCK0_OBJECTIVE_REVIEW_MANIFEST_FILENAME}",
            artifact_kind="manifest",
            format="json",
            claim_scope="provenance",
            review_role="provenance",
            proof_carrying_status="none",
            analysis_level="run_level",
            family_surface_role="mixed",
            rows_represent="Single JSON object for the generated Block 0 objective review surface.",
            columns_represent=(
                "Top-level keys record schema status, canonical raw inputs, generated review artifacts, "
                "and any review artifacts that were not generated."
            ),
            source_workflow="write_block0_review_surface",
            source_manifest_or_bundle=str(bundle_path),
            notes="Manifest for the packet-local Block 0 objective review surface.",
            review_rank=25,
        )
    )

    human_index_path = _write_human_index(
        block0_dir=block0_dir,
        schema_variant=schema_variant,
        bundle_payload=bundle_payload,
        generated_artifacts=generated_artifacts,
        missing_artifacts=missing_artifacts,
    )
    generated_artifacts.append(
        Block0ReviewArtifact(
            artifact_name=BLOCK0_HUMAN_INDEX_FILENAME,
            packet_relative_path=f"block0/{BLOCK0_HUMAN_INDEX_FILENAME}",
            artifact_kind="report",
            format="md",
            claim_scope="provenance",
            review_role="provenance",
            proof_carrying_status="none",
            analysis_level="mixed",
            family_surface_role="mixed",
            rows_represent="Markdown review guide for the Block 0 packet subsection.",
            columns_represent="Markdown prose headings and bullet lists only.",
            source_workflow="write_block0_review_surface",
            source_manifest_or_bundle=str(bundle_path),
            notes="Human-readable Block 0 review guide for the packet subsection.",
            review_rank=26,
        )
    )

    return Block0ReviewSurface(
        schema_variant=schema_variant,
        manifest_path=manifest_path,
        human_index_path=human_index_path,
        artifacts=tuple(sorted(generated_artifacts, key=lambda item: (item.review_rank, item.artifact_name))),
        missing_artifacts=tuple(missing_artifacts),
    )


__all__ = [
    "BLOCK0_FAMILY_SUMMARY_FILENAME",
    "BLOCK0_GATE_SUMMARY_FILENAME",
    "BLOCK0_HUMAN_INDEX_FILENAME",
    "BLOCK0_NULL_PROVENANCE_FILENAME",
    "BLOCK0_OBJECTIVE_REVIEW_MANIFEST_FILENAME",
    "BLOCK0_PATIENT_REVIEW_FILENAME",
    "BLOCK0_REPRODUCIBILITY_METADATA_FILENAME",
    "Block0ReviewArtifact",
    "Block0ReviewSurface",
    "is_current_block0_bundle_payload",
    "is_current_block0_pair_metrics_columns",
    "write_block0_review_surface",
]
