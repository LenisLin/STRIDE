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


__all__ = ["fit_warning_summary"]
