"""Block 0 bundle contract helpers for Task A."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stride.errors import ContractError

from ..config import (
    compute_task_a_config_fingerprint,
    load_raw_task_a_config,
    load_task_a_config_bundle,
)
from ..contracts import CONTRACT_PASSED_STATE, coerce_task_a_artifact_state
from .locality_gate import NULL_FAMILIES, REAL_FAMILIES


BLOCK_NAME = "block0_locality_gate"
PASSED_STATUS = "passed"
PAIR_METRICS_FILENAME = "block0_pair_metrics.csv"
BUNDLE_FILENAME = "block0_bundle.json"


@dataclass(frozen=True)
class TaskABlock0BundleContract:
    block: str
    status: str
    artifact_state: str
    implementation_tier: str
    evidence_lineage: str
    run_scope: str
    block0_passed: bool
    config_fingerprint: str
    config_path: Path
    stage0_h5ad: Path
    output_dir: Path
    bundle_path: Path
    pair_metrics_path: Path
    real_families: tuple[str, ...]
    null_families: tuple[str, ...]
    pre_block0_data_suitability: dict[str, Any]
    gate_checks: dict[str, Any]
    metrics_summary: dict[str, Any]
    failure_reasons: tuple[str, ...]
    inputs: dict[str, Any]

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "TaskABlock0BundleContract":
        block = str(payload["block"])
        status = str(payload.get("status", "unknown"))
        return cls(
            block=block,
            status=status,
            artifact_state=coerce_task_a_artifact_state(
                artifact_state=None if payload.get("artifact_state") in (None, "") else str(payload["artifact_state"]),
                legacy_status=status,
                block=block,
            ),
            implementation_tier=str(payload.get("implementation_tier", "")),
            evidence_lineage=str(payload.get("evidence_lineage", "")),
            run_scope=str(payload.get("run_scope", "")),
            block0_passed=bool(payload.get("block0_passed", False)),
            config_fingerprint=str(payload.get("config_fingerprint", "")),
            config_path=Path(str(payload["config_path"])).expanduser().resolve(),
            stage0_h5ad=Path(str(payload["stage0_h5ad"])).expanduser().resolve(),
            output_dir=Path(str(payload["output_dir"])).expanduser().resolve(),
            bundle_path=Path(str(payload["bundle_path"])).expanduser().resolve(),
            pair_metrics_path=Path(str(payload["pair_metrics_path"])).expanduser().resolve(),
            real_families=tuple(str(name) for name in payload.get("real_families", [])),
            null_families=tuple(str(name) for name in payload.get("null_families", [])),
            pre_block0_data_suitability=dict(payload.get("pre_block0_data_suitability", {})),
            gate_checks=dict(payload.get("gate_checks", {})),
            metrics_summary=dict(payload.get("metrics_summary", {})),
            failure_reasons=tuple(str(reason) for reason in payload.get("failure_reasons", [])),
            inputs=dict(payload["inputs"]),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "block": self.block,
            "status": self.status,
            "artifact_state": self.artifact_state,
            "implementation_tier": self.implementation_tier,
            "evidence_lineage": self.evidence_lineage,
            "run_scope": self.run_scope,
            "block0_passed": self.block0_passed,
            "config_fingerprint": self.config_fingerprint,
            "config_path": str(self.config_path),
            "stage0_h5ad": str(self.stage0_h5ad),
            "output_dir": str(self.output_dir),
            "bundle_path": str(self.bundle_path),
            "pair_metrics_path": str(self.pair_metrics_path),
            "real_families": list(self.real_families),
            "null_families": list(self.null_families),
            "pre_block0_data_suitability": self.pre_block0_data_suitability,
            "gate_checks": self.gate_checks,
            "metrics_summary": self.metrics_summary,
            "failure_reasons": list(self.failure_reasons),
            "inputs": self.inputs,
        }


def load_block0_bundle_contract(bundle_path: str | Path) -> TaskABlock0BundleContract:
    path = Path(bundle_path).expanduser().resolve()
    if not path.exists():
        raise ContractError(f"Block 0 bundle does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = TaskABlock0BundleContract.from_json_dict(payload)
    if contract.bundle_path != path:
        contract = TaskABlock0BundleContract(
            block=contract.block,
            status=contract.status,
            artifact_state=contract.artifact_state,
            implementation_tier=contract.implementation_tier,
            evidence_lineage=contract.evidence_lineage,
            run_scope=contract.run_scope,
            block0_passed=contract.block0_passed,
            config_fingerprint=contract.config_fingerprint,
            config_path=contract.config_path,
            stage0_h5ad=contract.stage0_h5ad,
            output_dir=contract.output_dir,
            bundle_path=path,
            pair_metrics_path=contract.pair_metrics_path,
            real_families=contract.real_families,
            null_families=contract.null_families,
            pre_block0_data_suitability=contract.pre_block0_data_suitability,
            gate_checks=contract.gate_checks,
            metrics_summary=contract.metrics_summary,
            failure_reasons=contract.failure_reasons,
            inputs=contract.inputs,
        )
    return contract


def _config_matches_block0_provenance(
    contract: TaskABlock0BundleContract,
    *,
    current_fingerprint: str,
) -> bool:
    if contract.config_fingerprint == current_fingerprint:
        return True
    current_raw_config = load_raw_task_a_config(contract.config_path)
    block0_compat_config = dict(current_raw_config)
    for downstream_section in ("block2", "block3"):
        block0_compat_config.pop(downstream_section, None)
    block0_compat_config["enabled_blocks"] = [
        str(block_name)
        for block_name in current_raw_config.get("enabled_blocks", [])
        if str(block_name) not in {"block2_bounded_audit", "block3_method_validation"}
    ]
    block0_compat_fingerprint = compute_task_a_config_fingerprint(block0_compat_config)
    return contract.config_fingerprint == block0_compat_fingerprint


def require_block0_passed_contract(
    bundle_path: str | Path,
    *,
    config_path: str | Path,
    data_path: str | Path,
) -> TaskABlock0BundleContract:
    contract = load_block0_bundle_contract(bundle_path)
    expected_config_path = Path(config_path).expanduser().resolve()
    expected_data_path = Path(data_path).expanduser().resolve()
    config_bundle = load_task_a_config_bundle(expected_config_path)
    if contract.block != BLOCK_NAME:
        raise ContractError(
            f"Task A Block 1/2 requires a {BLOCK_NAME} bundle, got {contract.block!r}"
        )
    if contract.implementation_tier != "canonical_full":
        raise ContractError(
            "Task A Block 1/2 requires a canonical-full Block 0 bundle, "
            f"got implementation_tier={contract.implementation_tier!r}"
        )
    if contract.evidence_lineage != "canonical_rerun":
        raise ContractError(
            "Task A Block 1/2 requires a canonical-rerun Block 0 bundle, "
            f"got evidence_lineage={contract.evidence_lineage!r}"
        )
    if contract.status != PASSED_STATUS or contract.artifact_state != CONTRACT_PASSED_STATE:
        raise ContractError(
            "Task A Block 1/2 requires a passed, contract-passed Block 0 locality gate bundle; "
            f"got status={contract.status!r}, artifact_state={contract.artifact_state!r}"
        )
    if contract.config_path != expected_config_path:
        raise ContractError(
            "Block 0 bundle provenance mismatch: config_path does not match the "
            "requested Task A config"
        )
    if not _config_matches_block0_provenance(
        contract,
        current_fingerprint=config_bundle.config_fingerprint,
    ):
        raise ContractError(
            "Block 0 bundle provenance mismatch: config content changed after the "
            "bundle was produced outside the downstream Block 2/3 config surface"
        )
    if contract.stage0_h5ad != expected_data_path:
        raise ContractError(
            "Block 0 bundle provenance mismatch: stage0_h5ad does not match the "
            "requested Task A data surface"
        )
    if not contract.block0_passed:
        raise ContractError("Task A Block 1/2 requires block0_passed=true in the Block 0 bundle")
    if contract.real_families != REAL_FAMILIES:
        raise ContractError(
            "Task A Block 1/2 requires the frozen Block 0 real family set to remain "
            f"{REAL_FAMILIES!r}"
        )
    if contract.null_families != NULL_FAMILIES:
        raise ContractError(
            "Task A Block 1/2 requires the frozen Block 0 null family set to remain "
            f"{NULL_FAMILIES!r}"
        )
    return contract


__all__ = [
    "BLOCK_NAME",
    "BUNDLE_FILENAME",
    "PAIR_METRICS_FILENAME",
    "PASSED_STATUS",
    "TaskABlock0BundleContract",
    "load_block0_bundle_contract",
    "require_block0_passed_contract",
]
