"""Task-local artifact-state helpers for frozen Task A outputs.

These helpers validate and coerce the machine-readable readiness states used by
Task A manifests and bundles. They do not interpret scientific evidence.
"""
from __future__ import annotations

from stride.errors import ContractError


SCAFFOLD_ACTIVE_STATE = "scaffold_active"
CONTRACT_PASSED_STATE = "contract_passed"
EVIDENCE_READY_STATE = "evidence_ready"

TASK_A_ARTIFACT_STATES: frozenset[str] = frozenset(
    {
        SCAFFOLD_ACTIVE_STATE,
        CONTRACT_PASSED_STATE,
        EVIDENCE_READY_STATE,
    }
)


def validate_task_a_artifact_state(artifact_state: str) -> str:
    resolved_state = str(artifact_state).strip()
    if resolved_state not in TASK_A_ARTIFACT_STATES:
        raise ContractError(
            "Task A artifact_state must be one of "
            f"{sorted(TASK_A_ARTIFACT_STATES)}, got {resolved_state!r}"
        )
    return resolved_state


def coerce_task_a_artifact_state(
    *,
    artifact_state: str | None,
    legacy_status: str | None,
    block: str | None = None,
) -> str:
    if artifact_state not in (None, ""):
        return validate_task_a_artifact_state(str(artifact_state))

    normalized_status = None if legacy_status in (None, "") else str(legacy_status).strip()
    normalized_block = None if block in (None, "") else str(block).strip()

    if normalized_block == "block0_locality_gate":
        if normalized_status == "passed":
            return CONTRACT_PASSED_STATE
        return SCAFFOLD_ACTIVE_STATE

    if normalized_block == "block1_real_data_discovery":
        if normalized_status == "active":
            return EVIDENCE_READY_STATE
        if normalized_status == "passed":
            return CONTRACT_PASSED_STATE
        return SCAFFOLD_ACTIVE_STATE

    if normalized_status == "passed":
        return CONTRACT_PASSED_STATE
    if normalized_status == "active":
        return SCAFFOLD_ACTIVE_STATE

    raise ContractError(
        "Task A artifact_state is required because the payload does not carry a "
        "known legacy status compatibility value"
    )


__all__ = [
    "CONTRACT_PASSED_STATE",
    "EVIDENCE_READY_STATE",
    "SCAFFOLD_ACTIVE_STATE",
    "TASK_A_ARTIFACT_STATES",
    "coerce_task_a_artifact_state",
    "validate_task_a_artifact_state",
]
