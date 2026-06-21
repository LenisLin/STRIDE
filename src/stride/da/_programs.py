"""Exploratory relation-program analysis surfaces for STRIDE `.da`."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, TypedDict

import numpy as np
import pandas as pd

from stride.errors import ContractError

from ._stats import (
    MIN_PATIENTS_PER_GROUP,
    apply_bh_correction,
    comparison_fields,
    multi_group_stats,
    two_group_stats,
)

_DIAGNOSTIC_COLUMNS = [
    "relation_id",
    "rank_patient",
    "rank_source",
    "rank_target_open",
    "restart_id",
    "random_seed",
    "reconstruction_error",
    "relative_error",
    "status",
    "error_message",
    "selected",
]

_PATIENT_FACTOR_COLUMNS = [
    "relation_id",
    "patient_id",
    "group_id",
    "patient_factor_id",
    "loading",
]

_SOURCE_FACTOR_COLUMNS = [
    "relation_id",
    "source_factor_id",
    "source_community_id",
    "loading",
]

_TARGET_OPEN_FACTOR_COLUMNS = [
    "relation_id",
    "target_open_factor_id",
    "target_open_axis_id",
    "target_open_axis_type",
    "loading",
]

_CORE_COLUMNS = [
    "relation_id",
    "patient_factor_id",
    "source_factor_id",
    "target_open_factor_id",
    "weight",
]

_PROGRAM_COLUMNS = [
    "relation_id",
    "program_id",
    "patient_factor_id",
    "source_factor_id",
    "target_open_factor_id",
    "core_weight",
    "program_weight_rank",
]

_PROGRAM_ENTRY_COLUMNS = [
    "relation_id",
    "program_id",
    "source_community_id",
    "target_open_axis_id",
    "target_open_axis_type",
    "program_component_contribution",
]

_PATIENT_PROGRAM_SCORE_COLUMNS = [
    "relation_id",
    "patient_id",
    "group_id",
    "program_id",
    "program_component_score",
]

_PROGRAM_STATS_COLUMNS = [
    "relation_id",
    "comparison_id",
    "program_id",
    "comparison_type",
    "group_1",
    "group_2",
    "groups",
    "n_total",
    "n_by_group",
    "mean_by_group",
    "median_by_group",
    "std_by_group",
    "test_name",
    "effect_size",
    "effect_size_type",
    "effect_direction",
    "p_value",
    "q_value",
    "correction_method",
    "correction_scope",
]


class _ProgramRow(TypedDict):
    relation_id: str
    program_id: str
    patient_factor_id: int
    source_factor_id: int
    target_open_factor_id: int
    core_weight: float
    program_weight_rank: int


def relation_program_rank_diagnostics(
    patient_arrays: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    rank_grid: Sequence[tuple[int, int, int]]
    | Mapping[str, Sequence[tuple[int, int, int]]],
    relation_ids: Sequence[str] | None = None,
    n_restarts: int = 10,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Evaluate candidate ranks for T-only relation-program decomposition.

    Scientific question:
        For each declared relation, which caller-provided non-negative Tucker
        ranks produce plausible exploratory decompositions of the fitted
        patient-level source-row relation tensor `T = [A|d]`?

    Input:
        `patient_arrays` is the output of `patient_relation_arrays`. The
        function consumes `A`, `d`, `patient_ids`, and relation ids. For each
        relation, it constructs `X_T[p, i, j] = A[p, i, j]` for `j < K` and
        `X_T[p, i, K] = d[p, i]`, with shape `[P, K, K + 1]`.

    Rank grid:
        `rank_grid` contains candidate `(rank_patient, rank_source,
        rank_target_open)` tuples. It may be shared across relations or keyed
        by `relation_id`. The selected rank is not inferred by this interface;
        downstream callers must make rank choice explicit.

    Output:
        DataFrame with relation id, candidate ranks, restart id, random seed,
        reconstruction error, relative error, convergence/status metadata, and
        implementation-specific diagnostics when available.

    Boundary:
        This is an exploratory diagnostic surface. It does not fit/refit
        STRIDE, mutate inputs, write files, assign clinical meaning, test
        patient groups, or make an automatic rank-selection claim.
    """
    if n_restarts < 1:
        raise ContractError("n_restarts must be at least 1")
    relation_items = _resolve_relation_items(patient_arrays, relation_ids)
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    for relation_id, relation_groups in relation_items:
        tensor_data = _relation_tensor(relation_id, relation_groups)
        for rank in _rank_grid_for_relation(rank_grid, relation_id=relation_id):
            _validate_rank(rank, shape=tensor_data.tensor.shape)
            for restart_id in range(n_restarts):
                seed = int(rng.integers(0, np.iinfo(np.int32).max))
                rows.append(
                    _diagnostic_row(
                        relation_id=relation_id,
                        tensor=tensor_data.tensor,
                        rank=rank,
                        restart_id=restart_id,
                        random_seed=seed,
                    )
                )
    return pd.DataFrame(rows, columns=_DIAGNOSTIC_COLUMNS)


def relation_program_decomposition(
    patient_arrays: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    ranks: tuple[int, int, int] | Mapping[str, tuple[int, int, int]],
    relation_ids: Sequence[str] | None = None,
    n_restarts: int = 10,
    random_state: int | None = None,
) -> Mapping[str, pd.DataFrame]:
    """Decompose fitted T-only relation tensors into exploratory programs.

    Scientific question:
        Do fitted patient-level `T = [A|d]` matrices contain low-dimensional
        source-to-target/open relation patterns that can be summarized as
        exploratory relation programs?

    Input:
        `patient_arrays` is the output of `patient_relation_arrays`. For each
        selected relation, the analysis constructs `X_T[p, i, j] = A[p, i, j]`
        for target-community columns and `X_T[p, i, K] = d[p, i]` for the
        source-open column. The target-open `e` vector is not part of this
        tensor input.

    Decomposition:
        The intended method is one shared non-negative Tucker decomposition per
        declared relation across selected patients. `ranks` supplies explicit
        `(rank_patient, rank_source, rank_target_open)` values. Multiple
        restarts and random-state metadata should be recorded by the
        implementation.

    Output:
        Mapping of tidy DataFrames, expected to include rank diagnostics,
        patient factors, source factors, target/open factors, core tensor,
        program component definitions, program-entry component contributions,
        and patient program component scores.

    Boundary:
        Relation programs are exploratory latent summaries. They are not
        biological pathways, recurrence families, or causal mechanisms without
        external validation. This function does not fit/refit STRIDE, run group
        tests, choose ranks automatically, mutate inputs, or write files.
    """
    if n_restarts < 1:
        raise ContractError("n_restarts must be at least 1")
    relation_items = _resolve_relation_items(patient_arrays, relation_ids)
    rng = np.random.default_rng(random_state)

    diagnostics: list[dict[str, object]] = []
    patient_factor_rows: list[dict[str, object]] = []
    source_factor_rows: list[dict[str, object]] = []
    target_open_factor_rows: list[dict[str, object]] = []
    core_rows: list[dict[str, object]] = []
    program_rows: list[dict[str, object]] = []
    program_entry_rows: list[dict[str, object]] = []
    patient_program_score_rows: list[dict[str, object]] = []

    for relation_id, relation_groups in relation_items:
        tensor_data = _relation_tensor(relation_id, relation_groups)
        rank = _rank_for_relation(ranks, relation_id=relation_id)
        _validate_rank(rank, shape=tensor_data.tensor.shape)
        relation_diagnostics: list[dict[str, object]] = []
        restart_results = []
        for restart_id in range(n_restarts):
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            result = _run_tucker(
                tensor_data.tensor,
                rank=rank,
                random_seed=seed,
            )
            relation_diagnostics.append(
                _diagnostic_row_from_result(
                    relation_id=relation_id,
                    rank=rank,
                    restart_id=restart_id,
                    random_seed=seed,
                    result=result,
                )
            )
            if result.status == "ok":
                restart_results.append((restart_id, seed, result))
        if not restart_results:
            diagnostics.extend(relation_diagnostics)
            raise ContractError(f"all decomposition restarts failed for relation_id '{relation_id}'")
        best_restart_id, _, best_result = min(
            restart_results,
            key=lambda item: item[2].relative_error,
        )
        for row in relation_diagnostics:
            row["selected"] = row["status"] == "ok" and row["restart_id"] == best_restart_id
        diagnostics.extend(relation_diagnostics)

        relation_tables = _tables_from_decomposition(
            relation_id=relation_id,
            tensor_data=tensor_data,
            core=best_result.core,
            factors=best_result.factors,
        )
        patient_factor_rows.extend(relation_tables["patient_factors"])
        source_factor_rows.extend(relation_tables["source_factors"])
        target_open_factor_rows.extend(relation_tables["target_open_factors"])
        core_rows.extend(relation_tables["core"])
        program_rows.extend(relation_tables["program_components"])
        program_entry_rows.extend(relation_tables["program_entries"])
        patient_program_score_rows.extend(relation_tables["patient_program_scores"])

    return {
        "rank_diagnostics": pd.DataFrame(diagnostics, columns=_DIAGNOSTIC_COLUMNS),
        "patient_factors": pd.DataFrame(patient_factor_rows, columns=_PATIENT_FACTOR_COLUMNS),
        "source_factors": pd.DataFrame(source_factor_rows, columns=_SOURCE_FACTOR_COLUMNS),
        "target_open_factors": pd.DataFrame(
            target_open_factor_rows,
            columns=_TARGET_OPEN_FACTOR_COLUMNS,
        ),
        "core": pd.DataFrame(core_rows, columns=_CORE_COLUMNS),
        "program_components": pd.DataFrame(program_rows, columns=_PROGRAM_COLUMNS),
        "program_entries": pd.DataFrame(program_entry_rows, columns=_PROGRAM_ENTRY_COLUMNS),
        "patient_program_scores": pd.DataFrame(
            patient_program_score_rows,
            columns=_PATIENT_PROGRAM_SCORE_COLUMNS,
        ),
    }


def relation_program_group_association(
    patient_program_scores: pd.DataFrame,
    *,
    comparisons: Sequence[Mapping[str, object]],
    correction: Literal["BH"] = "BH",
) -> pd.DataFrame:
    """Test group associations on patient-level relation-program scores.

    Scientific question:
        Are caller-provided patient groups associated with exploratory
        relation-program scores within a declared relation?

    Input:
        `patient_program_scores` is a `.da` table with one row per
        `relation_id`, `patient_id`, `group_id`, and `program_id`.
        The tested value is the patient-level `program_component_score`, conventionally
        `patient_factor_loading[p, pf] * core_weight[program]`.

    Statistical policy:
        Patient is the statistical unit. Two independent groups use Wilcoxon
        rank-sum / Mann-Whitney U with Cliff's delta. Multiple groups use
        one-way ANOVA with eta-squared. BH correction is applied within each
        `relation_id + comparison_id` over tested `program_id` values.

    Output:
        DataFrame with relation id, comparison id, program id, group summaries,
        test name, effect size, effect-size type and direction, p-value,
        q-value, correction method, and correction scope.

    Boundary:
        This function tests patient-level program scores only. It does not
        group-test source factors, target/open factors, or core weights; it
        does not compute tensor decompositions, select ranks, mutate inputs, or
        write files.
    """
    if correction != "BH":
        raise ContractError("correction must be 'BH'")
    scores = _validate_patient_program_scores(patient_program_scores)
    rows: list[dict[str, object]] = []
    for relation_id in _ordered_unique(scores["relation_id"]):
        relation_scores = scores[scores["relation_id"] == relation_id]
        for comparison in comparisons:
            comparison_id, group_ids = comparison_fields(comparison)
            for group_id in group_ids:
                if group_id not in set(relation_scores["group_id"]):
                    raise ContractError(f"unknown group '{group_id}' for relation_id '{relation_id}'")
            _require_score_group_sizes(
                relation_scores,
                group_ids=group_ids,
                relation_id=relation_id,
                comparison_id=comparison_id,
            )
            comparison_rows = _program_association_rows(
                relation_scores,
                relation_id=relation_id,
                comparison_id=comparison_id,
                group_ids=group_ids,
            )
            apply_bh_correction(comparison_rows)
            rows.extend(comparison_rows)
    return pd.DataFrame(rows, columns=_PROGRAM_STATS_COLUMNS)


class _TensorData:
    """Validated relation tensor and patient metadata."""

    def __init__(
        self,
        *,
        tensor: np.ndarray,
        patient_ids: tuple[str, ...],
        group_ids: tuple[str, ...],
    ) -> None:
        self.tensor = tensor
        self.patient_ids = patient_ids
        self.group_ids = group_ids


class _TuckerResult:
    """One non-negative Tucker restart result."""

    def __init__(
        self,
        *,
        status: str,
        reconstruction_error: float,
        relative_error: float,
        core: np.ndarray | None,
        factors: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        error_message: str | None,
    ) -> None:
        self.status = status
        self.reconstruction_error = reconstruction_error
        self.relative_error = relative_error
        self.core = core
        self.factors = factors
        self.error_message = error_message


def _resolve_relation_items(
    patient_arrays: Mapping[str, Mapping[str, Mapping[str, object]]],
    relation_ids: Sequence[str] | None,
) -> tuple[tuple[str, Mapping[str, Mapping[str, object]]], ...]:
    """Resolve relation traversal and explicit selectors."""
    if not patient_arrays:
        raise ContractError("patient_arrays must contain at least one relation")
    if relation_ids is None:
        requested = tuple(str(relation_id) for relation_id in patient_arrays)
    else:
        requested = tuple(str(relation_id) for relation_id in relation_ids)
        missing = [relation_id for relation_id in requested if relation_id not in patient_arrays]
        if missing:
            raise ContractError("unknown relation_id values: " + ", ".join(missing))
    return tuple((relation_id, patient_arrays[relation_id]) for relation_id in requested)


def _relation_tensor(
    relation_id: str,
    relation_groups: Mapping[str, Mapping[str, object]],
) -> _TensorData:
    """Construct one validated T-only tensor from grouped patient arrays."""
    if not relation_groups:
        raise ContractError("patient_arrays relation entry must contain at least one group")
    arrays: list[np.ndarray] = []
    d_arrays: list[np.ndarray] = []
    patient_ids: list[str] = []
    group_ids: list[str] = []
    n_states: int | None = None
    for group_id, group in relation_groups.items():
        A = _as_group_array(group, "A")
        d = _as_group_array(group, "d")
        if A.ndim != 3 or A.shape[1] != A.shape[2]:
            raise ContractError("patient_arrays group A must have shape [P, K, K]")
        K = int(A.shape[1])
        if d.shape != (A.shape[0], K):
            raise ContractError("patient_arrays group d must have shape [P, K]")
        if n_states is None:
            n_states = K
        elif n_states != K:
            raise ContractError("all groups within a relation must share n_states")
        if (A < 0.0).any() or (d < 0.0).any():
            raise ContractError("patient_arrays A and d entries must be nonnegative")
        patients = _patient_ids_for_group(group, n_patients=A.shape[0])
        patient_ids.extend(patients)
        group_ids.extend(str(group_id) for _ in patients)
        arrays.append(A)
        d_arrays.append(d)
    if n_states is None:
        raise ContractError(f"relation_id '{relation_id}' has no arrays")
    A_all = np.concatenate(arrays, axis=0)
    d_all = np.concatenate(d_arrays, axis=0)
    tensor = np.empty((A_all.shape[0], n_states, n_states + 1), dtype=float)
    tensor[:, :, :n_states] = A_all
    tensor[:, :, n_states] = d_all
    if tensor.shape[0] == 0:
        raise ContractError(f"relation_id '{relation_id}' must contain at least one patient")
    if len(set(patient_ids)) != len(patient_ids):
        raise ContractError(f"relation_id '{relation_id}' contains duplicate patient_ids across groups")
    return _TensorData(tensor=tensor, patient_ids=tuple(patient_ids), group_ids=tuple(group_ids))


def _as_group_array(group: Mapping[str, object], key: str) -> np.ndarray:
    """Return one grouped fitted array as finite float values."""
    if key not in group:
        raise ContractError(f"patient_arrays group is missing '{key}'")
    try:
        array = np.asarray(group[key], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"patient_arrays group '{key}' must be numeric") from exc
    if not np.isfinite(array).all():
        raise ContractError(f"patient_arrays group '{key}' must contain only finite values")
    return array


def _patient_ids_for_group(group: Mapping[str, object], *, n_patients: int) -> tuple[str, ...]:
    """Return patient ids aligned to one group array."""
    if "patient_ids" not in group:
        return tuple(str(index) for index in range(n_patients))
    raw_patient_ids = group["patient_ids"]
    if isinstance(raw_patient_ids, (str, bytes)) or not isinstance(raw_patient_ids, Iterable):
        raise ContractError("patient_ids must be a sequence")
    patient_ids = tuple(str(value) for value in raw_patient_ids)
    if len(patient_ids) != n_patients:
        raise ContractError("patient_ids length must match grouped arrays")
    if len(set(patient_ids)) != len(patient_ids):
        raise ContractError("patient_ids must be unique within a group")
    return patient_ids


def _rank_grid_for_relation(
    rank_grid: Sequence[tuple[int, int, int]] | Mapping[str, Sequence[tuple[int, int, int]]],
    *,
    relation_id: str,
) -> tuple[tuple[int, int, int], ...]:
    """Resolve candidate ranks for one relation."""
    if isinstance(rank_grid, Mapping):
        if relation_id not in rank_grid:
            raise ContractError(f"rank_grid is missing relation_id '{relation_id}'")
        values = rank_grid[relation_id]
    else:
        values = rank_grid
    ranks = tuple(_rank_tuple(value) for value in values)
    if not ranks:
        raise ContractError("rank_grid must contain at least one rank")
    return ranks


def _rank_for_relation(
    ranks: tuple[int, int, int] | Mapping[str, tuple[int, int, int]],
    *,
    relation_id: str,
) -> tuple[int, int, int]:
    """Resolve selected rank for one relation."""
    if isinstance(ranks, Mapping):
        if relation_id not in ranks:
            raise ContractError(f"ranks is missing relation_id '{relation_id}'")
        return _rank_tuple(ranks[relation_id])
    return _rank_tuple(ranks)


def _rank_tuple(value: object) -> tuple[int, int, int]:
    """Normalize one rank tuple."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ContractError("rank must be a tuple of three positive integers")
    raw_rank: tuple[object, ...] = tuple(value)
    rank: list[int] = []
    for item in raw_rank:
        if not isinstance(item, (int, np.integer)):
            raise ContractError("rank values must be integers")
        rank.append(int(item))
    if len(rank) != 3:
        raise ContractError("rank must be a tuple of three positive integers")
    return rank[0], rank[1], rank[2]


def _validate_rank(rank: tuple[int, int, int], *, shape: tuple[int, int, int]) -> None:
    """Check rank against the tensor shape."""
    if any(value < 1 for value in rank):
        raise ContractError("rank values must be positive")
    if any(value > dim for value, dim in zip(rank, shape, strict=True)):
        raise ContractError("rank values must not exceed tensor dimensions")


def _diagnostic_row(
    *,
    relation_id: str,
    tensor: np.ndarray,
    rank: tuple[int, int, int],
    restart_id: int,
    random_seed: int,
) -> dict[str, object]:
    """Run one restart and return its diagnostics row."""
    result = _run_tucker(tensor, rank=rank, random_seed=random_seed)
    return _diagnostic_row_from_result(
        relation_id=relation_id,
        rank=rank,
        restart_id=restart_id,
        random_seed=random_seed,
        result=result,
    )


def _diagnostic_row_from_result(
    *,
    relation_id: str,
    rank: tuple[int, int, int],
    restart_id: int,
    random_seed: int,
    result: _TuckerResult,
    selected: bool = False,
) -> dict[str, object]:
    """Format a restart result as a rank-diagnostic row."""
    return {
        "relation_id": relation_id,
        "rank_patient": int(rank[0]),
        "rank_source": int(rank[1]),
        "rank_target_open": int(rank[2]),
        "restart_id": int(restart_id),
        "random_seed": int(random_seed),
        "reconstruction_error": float(result.reconstruction_error),
        "relative_error": float(result.relative_error),
        "status": result.status,
        "error_message": result.error_message,
        "selected": bool(selected),
    }


def _run_tucker(
    tensor: np.ndarray,
    *,
    rank: tuple[int, int, int],
    random_seed: int,
) -> _TuckerResult:
    """Run one non-negative Tucker restart with fixed STRIDE settings."""
    try:
        from tensorly import tucker_to_tensor
        from tensorly.decomposition import non_negative_tucker_hals
    except ImportError as exc:
        raise ContractError("tensorly>=0.9.0 is required for relation program decomposition") from exc

    try:
        decomposition = non_negative_tucker_hals(
            tensor,
            rank=rank,
            n_iter_max=100,
            tol=1e-8,
            normalize_factors=False,
            random_state=random_seed,
            return_errors=True,
        )
        tucker_tensor, _errors = decomposition
        core = np.asarray(tucker_tensor.core, dtype=float)
        raw_factors = tuple(np.asarray(factor, dtype=float) for factor in tucker_tensor.factors)
        if len(raw_factors) != 3:
            raise ContractError("tensorly returned an invalid number of factors")
        factors = (raw_factors[0], raw_factors[1], raw_factors[2])
        reconstruction = np.asarray(tucker_to_tensor((core, list(factors))), dtype=float)
        reconstruction_error = float(np.linalg.norm(tensor - reconstruction))
        tensor_norm = float(np.linalg.norm(tensor))
        relative_error = reconstruction_error / tensor_norm if tensor_norm > 0.0 else reconstruction_error
        if not np.isfinite(reconstruction_error) or not np.isfinite(relative_error):
            raise ContractError("non-negative Tucker returned non-finite error")
        if (core < -1e-12).any() or any((factor < -1e-12).any() for factor in factors):
            raise ContractError("non-negative Tucker returned negative factors")
        core = np.maximum(core, 0.0)
        factors = tuple(np.maximum(factor, 0.0) for factor in factors)
        core, factors = _normalize_tucker_factors(core, factors)
        return _TuckerResult(
            status="ok",
            reconstruction_error=reconstruction_error,
            relative_error=relative_error,
            core=core,
            factors=factors,
            error_message=None,
        )
    except Exception as exc:
        message = str(exc) if isinstance(exc, ContractError) else f"{type(exc).__name__}: {exc}"
        return _TuckerResult(
            status="failed",
            reconstruction_error=np.nan,
            relative_error=np.nan,
            core=None,
            factors=None,
            error_message=message,
        )


def _normalize_tucker_factors(
    core: np.ndarray,
    factors: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Fix Tucker scale ambiguity by L1-normalizing factor columns."""
    normalized_core = core.copy()
    normalized_factors: list[np.ndarray] = []
    for mode, factor in enumerate(factors):
        normalized = factor.copy()
        for component_id in range(normalized.shape[1]):
            scale = float(np.sum(normalized[:, component_id]))
            core_slice: list[slice | int] = [slice(None)] * normalized_core.ndim
            core_slice[mode] = component_id
            if scale > 0.0:
                normalized[:, component_id] = normalized[:, component_id] / scale
                normalized_core[tuple(core_slice)] *= scale
            else:
                normalized_core[tuple(core_slice)] = 0.0
        normalized_factors.append(normalized)
    if not np.isfinite(normalized_core).all() or any(
        not np.isfinite(factor).all() for factor in normalized_factors
    ):
        raise ContractError("non-negative Tucker normalization produced non-finite values")
    if len(normalized_factors) != 3:
        raise ContractError("non-negative Tucker normalization produced an invalid factor count")
    return normalized_core, (normalized_factors[0], normalized_factors[1], normalized_factors[2])


def _tables_from_decomposition(
    *,
    relation_id: str,
    tensor_data: _TensorData,
    core: np.ndarray | None,
    factors: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> dict[str, list[dict[str, object]]]:
    """Convert one selected decomposition to tidy output rows."""
    if core is None or factors is None:
        raise ContractError("successful decomposition is missing core or factors")
    patient_factor, source_factor, target_open_factor = factors
    K = int(tensor_data.tensor.shape[1])

    patient_rows = [
        {
            "relation_id": relation_id,
            "patient_id": patient_id,
            "group_id": group_id,
            "patient_factor_id": int(patient_factor_id),
            "loading": float(patient_factor[patient_index, patient_factor_id]),
        }
        for patient_index, (patient_id, group_id) in enumerate(
            zip(tensor_data.patient_ids, tensor_data.group_ids, strict=True)
        )
        for patient_factor_id in range(patient_factor.shape[1])
    ]
    source_rows = [
        {
            "relation_id": relation_id,
            "source_factor_id": int(source_factor_id),
            "source_community_id": int(source_community_id),
            "loading": float(source_factor[source_community_id, source_factor_id]),
        }
        for source_community_id in range(source_factor.shape[0])
        for source_factor_id in range(source_factor.shape[1])
    ]
    target_open_rows = [
        {
            "relation_id": relation_id,
            "target_open_factor_id": int(target_open_factor_id),
            "target_open_axis_id": int(axis_id),
            "target_open_axis_type": _target_open_axis_type(axis_id, n_states=K),
            "loading": float(target_open_factor[axis_id, target_open_factor_id]),
        }
        for axis_id in range(target_open_factor.shape[0])
        for target_open_factor_id in range(target_open_factor.shape[1])
    ]

    core_rows: list[dict[str, object]] = []
    program_rows: list[_ProgramRow] = []
    for patient_factor_id in range(core.shape[0]):
        for source_factor_id in range(core.shape[1]):
            for target_open_factor_id in range(core.shape[2]):
                weight = float(core[patient_factor_id, source_factor_id, target_open_factor_id])
                program_id = _program_id(patient_factor_id, source_factor_id, target_open_factor_id)
                core_rows.append(
                    {
                        "relation_id": relation_id,
                        "patient_factor_id": int(patient_factor_id),
                        "source_factor_id": int(source_factor_id),
                        "target_open_factor_id": int(target_open_factor_id),
                        "weight": weight,
                    }
                )
                program_rows.append(
                    {
                        "relation_id": relation_id,
                        "program_id": program_id,
                        "patient_factor_id": int(patient_factor_id),
                        "source_factor_id": int(source_factor_id),
                        "target_open_factor_id": int(target_open_factor_id),
                        "core_weight": weight,
                        "program_weight_rank": 0,
                    }
                )
    program_rows.sort(key=lambda row: (-float(row["core_weight"]), str(row["program_id"])))
    for rank, row in enumerate(program_rows, start=1):
        row["program_weight_rank"] = rank

    program_entry_rows: list[dict[str, object]] = []
    patient_program_score_rows: list[dict[str, object]] = []
    for program in program_rows:
        program_id = str(program["program_id"])
        patient_factor_id = int(program["patient_factor_id"])
        source_factor_id = int(program["source_factor_id"])
        target_open_factor_id = int(program["target_open_factor_id"])
        core_weight = float(program["core_weight"])
        for source_community_id in range(source_factor.shape[0]):
            for axis_id in range(target_open_factor.shape[0]):
                program_entry_rows.append(
                    {
                        "relation_id": relation_id,
                        "program_id": program_id,
                        "source_community_id": int(source_community_id),
                        "target_open_axis_id": int(axis_id),
                        "target_open_axis_type": _target_open_axis_type(axis_id, n_states=K),
                        "program_component_contribution": float(
                            core_weight
                            * source_factor[source_community_id, source_factor_id]
                            * target_open_factor[axis_id, target_open_factor_id]
                        ),
                    }
                )
        for patient_index, (patient_id, group_id) in enumerate(
            zip(tensor_data.patient_ids, tensor_data.group_ids, strict=True)
        ):
            patient_program_score_rows.append(
                {
                    "relation_id": relation_id,
                    "patient_id": patient_id,
                    "group_id": group_id,
                    "program_id": program_id,
                    "program_component_score": float(
                        patient_factor[patient_index, patient_factor_id] * core_weight
                    ),
                }
            )
    return {
        "patient_factors": patient_rows,
        "source_factors": source_rows,
        "target_open_factors": target_open_rows,
        "core": core_rows,
        "program_components": [dict(row) for row in program_rows],
        "program_entries": program_entry_rows,
        "patient_program_scores": patient_program_score_rows,
    }


def _target_open_axis_type(axis_id: int, *, n_states: int) -> str:
    """Return target/open axis type for a tensor column."""
    if axis_id < n_states:
        return "target_community"
    if axis_id == n_states:
        return "source_open"
    raise ContractError("target_open_axis_id is outside 0..K")


def _program_id(patient_factor_id: int, source_factor_id: int, target_open_factor_id: int) -> str:
    """Return stable program id for one core factor triple."""
    return f"program_pf{patient_factor_id}_sf{source_factor_id}_tof{target_open_factor_id}"


def _validate_patient_program_scores(patient_program_scores: pd.DataFrame) -> pd.DataFrame:
    """Validate the patient-program score table for association testing."""
    if not isinstance(patient_program_scores, pd.DataFrame):
        raise ContractError("patient_program_scores must be a pandas DataFrame")
    required = {
        "relation_id",
        "patient_id",
        "group_id",
        "program_id",
        "program_component_score",
    }
    missing = sorted(required.difference(patient_program_scores.columns))
    if missing:
        raise ContractError(
            "patient_program_scores is missing required columns: " + ", ".join(missing)
        )
    scores = patient_program_scores.copy()
    for column in ["relation_id", "patient_id", "group_id", "program_id"]:
        scores[column] = scores[column].astype(str)
    try:
        scores["program_component_score"] = scores["program_component_score"].astype(float)
    except (TypeError, ValueError) as exc:
        raise ContractError("patient_program_scores program_component_score must be numeric") from exc
    if not np.isfinite(scores["program_component_score"].to_numpy(dtype=float)).all():
        raise ContractError(
            "patient_program_scores program_component_score must contain only finite values"
        )
    if scores.empty:
        raise ContractError("patient_program_scores must contain at least one row")
    duplicated = scores.duplicated(["relation_id", "patient_id", "program_id"])
    if duplicated.any():
        raise ContractError(
            "patient_program_scores must contain one program_component_score per patient/program"
        )
    return scores


def _require_score_group_sizes(
    relation_scores: pd.DataFrame,
    *,
    group_ids: tuple[str, ...],
    relation_id: str,
    comparison_id: str,
) -> None:
    """Require at least three unique patients per compared group."""
    too_small = []
    for group_id in group_ids:
        group_scores = relation_scores[relation_scores["group_id"] == group_id]
        n_patients = int(group_scores["patient_id"].nunique())
        if n_patients < MIN_PATIENTS_PER_GROUP:
            too_small.append(f"{group_id} n={n_patients}")
    if too_small:
        raise ContractError(
            "comparison groups must contain at least 3 patients each "
            f"for relation_id '{relation_id}' and comparison_id '{comparison_id}': "
            + ", ".join(too_small)
        )


def _program_association_rows(
    relation_scores: pd.DataFrame,
    *,
    relation_id: str,
    comparison_id: str,
    group_ids: tuple[str, ...],
) -> list[dict[str, object]]:
    """Build uncorrected program association rows for one comparison."""
    rows: list[dict[str, object]] = []
    for program_id in _ordered_unique(relation_scores["program_id"]):
        program_scores = relation_scores[relation_scores["program_id"] == program_id]
        values_by_group = {
            group_id: program_scores.loc[
                program_scores["group_id"] == group_id,
                "program_component_score",
            ].to_numpy(dtype=float)
            for group_id in group_ids
        }
        too_small = [
            f"{group_id} n={len(values_by_group[group_id])}"
            for group_id in group_ids
            if len(values_by_group[group_id]) < MIN_PATIENTS_PER_GROUP
        ]
        if too_small:
            raise ContractError(
                "comparison groups must contain at least 3 patients each "
                f"for relation_id '{relation_id}', comparison_id '{comparison_id}', "
                f"and program_id '{program_id}': "
                + ", ".join(too_small)
            )
        if len(group_ids) == 2:
            test_name, effect_size, effect_size_type, effect_direction, p_value = two_group_stats(
                values_by_group[group_ids[0]],
                values_by_group[group_ids[1]],
                group_1=group_ids[0],
                group_2=group_ids[1],
            )
            comparison_type = "two_group"
            group_1 = group_ids[0]
            group_2 = group_ids[1]
        else:
            test_name, effect_size, effect_size_type, effect_direction, p_value = multi_group_stats(
                tuple(values_by_group[group_id] for group_id in group_ids)
            )
            comparison_type = "multi_group"
            group_1 = None
            group_2 = None
        rows.append(
            {
                "relation_id": relation_id,
                "comparison_id": comparison_id,
                "program_id": program_id,
                "comparison_type": comparison_type,
                "group_1": group_1,
                "group_2": group_2,
                "groups": group_ids,
                "n_total": int(sum(len(values) for values in values_by_group.values())),
                "n_by_group": {
                    group_id: int(len(values)) for group_id, values in values_by_group.items()
                },
                "mean_by_group": {
                    group_id: float(np.mean(values)) for group_id, values in values_by_group.items()
                },
                "median_by_group": {
                    group_id: float(np.median(values)) for group_id, values in values_by_group.items()
                },
                "std_by_group": {
                    group_id: float(np.std(values, ddof=0))
                    for group_id, values in values_by_group.items()
                },
                "test_name": test_name,
                "effect_size": effect_size,
                "effect_size_type": effect_size_type,
                "effect_direction": effect_direction,
                "p_value": p_value,
                "q_value": np.nan,
                "correction_method": "BH",
                "correction_scope": f"{relation_id}:{comparison_id}",
            }
        )
    return rows


def _ordered_unique(values: Iterable[object]) -> tuple[str, ...]:
    """Return values in first-observed string order."""
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        label = str(value)
        if label not in seen:
            seen.add(label)
            unique.append(label)
    return tuple(unique)
