"""Progress payload helpers for Task A Block 0 runs."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping


def fit_warning_summary(result: object, *, max_examples: int = 5) -> dict[str, object]:
    """Summarize fit warnings without materializing the full warning payload."""
    warnings = _collect_fit_warnings(result)
    categories = Counter(_warning_category(warning) for warning in warnings)
    return {
        "warning_count": len(warnings),
        "warning_categories": dict(sorted(categories.items())),
        "warning_examples": list(warnings[: int(max_examples)]),
    }


def fit_runtime_summary(result: object) -> dict[str, object]:
    """Summarize compact `.tl.fit` runtime/provenance facts for progress logs."""
    relation_ids = tuple(str(item) for item in tuple(getattr(result, "relation_ids", ()) or ()))
    relations = getattr(result, "relations", {}) or {}
    relation_summaries: dict[str, object] = {}
    total_blocks = 0
    total_patients = 0
    for relation_id in relation_ids:
        relation = relations.get(relation_id) if isinstance(relations, Mapping) else None
        if relation is None:
            continue
        support = getattr(relation, "support", {}) or {}
        provenance = getattr(relation, "provenance", {}) or {}
        optimizer = provenance.get("optimizer", {}) if isinstance(provenance, Mapping) else {}
        n_blocks = _optional_int(support.get("n_evidence_blocks"))
        patient_ids = tuple(str(item) for item in tuple(getattr(relation, "patient_ids", ()) or ()))
        if n_blocks is not None:
            total_blocks += n_blocks
        total_patients += len(patient_ids)
        relation_summaries[relation_id] = {
            "n_patients": len(patient_ids),
            "n_evidence_blocks": n_blocks,
            "optimizer": _optimizer_progress_summary(optimizer),
        }

    return {
        "fit_surface": "stride.tl.fit",
        "n_relations": len(relation_ids),
        "relation_ids": list(relation_ids),
        "n_patients_total": total_patients,
        "n_evidence_blocks_total": total_blocks,
        "relations": relation_summaries,
    }


def _warning_category(warning: str) -> str:
    normalized = warning.lower()
    if "sinkhorn" in normalized:
        return "sinkhorn"
    if "warning_flags" in normalized:
        return "warning_flags"
    if "finite" in normalized or "nan" in normalized or "inf" in normalized:
        return "finite_numeric"
    if "conver" in normalized or "residual" in normalized or "tol" in normalized:
        return "convergence"
    return "other"


def _collect_fit_warnings(result: object) -> tuple[str, ...]:
    collected: list[str] = []
    _extend_warning_payload(collected, getattr(result, "warnings", None))
    for mapping_name in ("diagnostics", "summaries", "metadata"):
        _extend_warning_mapping(collected, getattr(result, mapping_name, None))
    _extend_warning_object(collected, getattr(result, "objective", None))
    for ledger_name in ("final_ledger", "objective_ledger"):
        _extend_warning_object(collected, getattr(result, ledger_name, None))
    for patient_result in tuple(getattr(result, "patient_results", ()) or ()):
        _extend_warning_payload(collected, getattr(patient_result, "warnings", None))
        for mapping_name in ("diagnostics", "auxiliary", "metadata"):
            _extend_warning_mapping(collected, getattr(patient_result, mapping_name, None))
        _extend_warning_object(collected, getattr(patient_result, "objective", None))

    unique_warnings: list[str] = []
    seen: set[str] = set()
    for warning in collected:
        if warning in seen:
            continue
        seen.add(warning)
        unique_warnings.append(warning)
    return tuple(unique_warnings)


def _optimizer_progress_summary(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    warmup = value.get("warmup", {})
    main = value.get("main", {})
    return {
        "exit_flag": value.get("exit_flag"),
        "reason": value.get("reason"),
        "n_steps": value.get("n_steps"),
        "initial_total": value.get("initial_total"),
        "final_total": value.get("final_total"),
        "absolute_improvement": value.get("absolute_improvement"),
        "relative_improvement": value.get("relative_improvement"),
        "warmup_steps_completed": (
            warmup.get("steps_completed") if isinstance(warmup, Mapping) else None
        ),
        "main_steps_completed": (
            main.get("steps_completed") if isinstance(main, Mapping) else None
        ),
        "main_max_steps": main.get("max_steps") if isinstance(main, Mapping) else None,
    }


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extend_warning_object(collected: list[str], value: object) -> None:
    if value is None:
        return
    _extend_warning_payload(collected, getattr(value, "warnings", None))
    for block_name in ("observation_blocks", "block_records"):
        for record in tuple(getattr(value, block_name, ()) or ()):
            _extend_warning_payload(collected, getattr(record, "warnings", None))
            _extend_warning_mapping(collected, getattr(record, "metadata", None))


def _extend_warning_mapping(collected: list[str], mapping: object) -> None:
    if not isinstance(mapping, Mapping):
        return
    for key, value in mapping.items():
        normalized_key = str(key).lower()
        if normalized_key in {"warning", "warnings", "warning_message", "warning_messages"}:
            _extend_warning_payload(collected, value)
        elif normalized_key == "warning_flags" and isinstance(value, Mapping):
            active_flags = tuple(
                str(flag_name)
                for flag_name, flag_value in value.items()
                if bool(flag_value) and str(flag_name) != "has_warnings"
            )
            if active_flags:
                collected.append(f"warning_flags={active_flags}")


def _extend_warning_payload(collected: list[str], value: object) -> None:
    if value is None:
        return
    if isinstance(value, str):
        warning = value.strip()
        if warning:
            collected.append(warning)
        return
    if isinstance(value, Mapping):
        for nested_value in value.values():
            _extend_warning_payload(collected, nested_value)
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            _extend_warning_payload(collected, item)
        return
    warning = str(value).strip()
    if warning:
        collected.append(warning)


__all__ = ["fit_runtime_summary", "fit_warning_summary"]
