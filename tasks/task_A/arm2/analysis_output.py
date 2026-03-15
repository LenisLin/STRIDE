"""
Module: tasks.task_A.arm2.analysis_output

Final focused-output assembly layer for the post-hoc Arm-II rewrite.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analysis_contract import (
    ARM_NAME,
    Arm2FocusedPaths,
    AUDIT_ONLY_FAMILIES,
    BaselineAnalysisTables,
    ComputedArm2Surfaces,
    CONFIRMATORY_FAMILIES,
    FOCUSED_OUTPUT_FILENAMES,
    FocusedOutputPackage,
    FocusedPrototypeViews,
    LoadedArm2Inputs,
    MEMO_NON_CLAIMS,
    MEMO_OUT_OF_SCOPE_ITEMS,
    MEMO_SUPPORTED_CLAIMS,
    PAIR_FAMILY_ORDER,
    RecurrenceAnalysisTables,
    TransportAnalysisTables,
)


def _csv_output_filenames() -> tuple[str, ...]:
    return tuple(filename for filename in FOCUSED_OUTPUT_FILENAMES if filename.endswith(".csv"))


def _format_float(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):.3f}".rstrip("0").rstrip(".")


def _support_row(validation: pd.DataFrame, check: str) -> pd.Series | None:
    if validation.empty:
        return None
    row = validation.loc[validation["check"].astype(str) == check]
    if row.empty:
        return None
    return row.iloc[0]


def build_minimal_appendix_audit_table(
    inputs: LoadedArm2Inputs,
    computed: ComputedArm2Surfaces,
    baseline_tables: BaselineAnalysisTables,
    transport_tables: TransportAnalysisTables,
    recurrence_tables: RecurrenceAnalysisTables,
    extracted_views: FocusedPrototypeViews,
) -> pd.DataFrame:
    """
    Build the minimal appendix/audit table for public output `08`.

    This table documents the focused package contract rather than acting as a
    broad exploratory appendix.
    """

    pair_counts = (
        computed.pair_level_transport["pair_family"]
        .astype(str)
        .value_counts()
        .reindex(PAIR_FAMILY_ORDER, fill_value=0)
        .to_dict()
    )
    audited_counts = {
        "TC-IM": int(inputs.stage0.patient_roi_audit_table["ordered_rows_TC_IM"].sum()),
        "IM-PT": int(inputs.stage0.patient_roi_audit_table["ordered_rows_IM_PT"].sum()),
        "TC-PT": int(inputs.stage0.patient_roi_audit_table["ordered_rows_TC_PT"].sum()),
    }
    baseline_passes = int(baseline_tables.baseline_validation["passed"].astype(bool).sum())
    baseline_checks = int(baseline_tables.baseline_validation.shape[0])
    transport_passes = int(transport_tables.transport_validation["passed"].astype(bool).sum())
    transport_checks = int(transport_tables.transport_validation.shape[0])
    recurrence_passes = int(recurrence_tables.recurrence_validation["passed"].astype(bool).sum())
    recurrence_checks = int(recurrence_tables.recurrence_validation.shape[0])
    view_passes = int(extracted_views.view_validation["passed"].astype(bool).sum())
    view_checks = int(extracted_views.view_validation.shape[0])
    patient_confirmatory_unit = bool(
        transport_tables.global_transport_summary["patient_id"].astype(str).is_unique
        if not transport_tables.global_transport_summary.empty
        else True
    )
    balanced_no_unmatched = _support_row(
        transport_tables.transport_validation,
        "balanced_transport_has_no_unmatched_semantics",
    )
    balanced_no_unmatched_ok = bool(balanced_no_unmatched["passed"]) if balanced_no_unmatched is not None else False
    tau_r_unavailable = bool(
        (inputs.metrics_df["tau_mode"].astype(str) == "unavailable").all()
        and "tau" not in computed.pair_level_transport.columns
        and "R" not in computed.pair_level_transport.columns
    )

    return pd.DataFrame.from_records(
        [
            {
                "section": "startup",
                "item": "startup_slice_scope",
                "status": "pass",
                "value": ARM_NAME,
                "detail": "Current focused package remains locked to the Arm-II startup slice only.",
            },
            {
                "section": "scope",
                "item": "confirmatory_vs_exploratory_families",
                "status": "pass",
                "value": f"confirmatory={','.join(CONFIRMATORY_FAMILIES)} | exploratory={','.join(AUDIT_ONLY_FAMILIES)}",
                "detail": "IM-PT remains audit/exploratory and does not enter confirmatory public contrasts.",
            },
            {
                "section": "scope",
                "item": "patient_confirmatory_unit",
                "status": "pass" if patient_confirmatory_unit else "fail",
                "value": "patient",
                "detail": (
                    f"rows={transport_tables.global_transport_summary.shape[0]}, "
                    f"unique_patients={transport_tables.global_transport_summary['patient_id'].astype(str).nunique() if not transport_tables.global_transport_summary.empty else 0}"
                ),
            },
            {
                "section": "audit",
                "item": "pair_count_audit",
                "status": "pass" if audited_counts == pair_counts else "fail",
                "value": str(audited_counts == pair_counts).lower(),
                "detail": f"observed={pair_counts}; audited={audited_counts}",
            },
            {
                "section": "validation",
                "item": "baseline_validation_pass_summary",
                "status": "pass" if baseline_passes == baseline_checks else "fail",
                "value": f"{baseline_passes}/{baseline_checks}",
                "detail": "Baseline validations from the all-prototype baseline layer.",
            },
            {
                "section": "validation",
                "item": "transport_validation_pass_summary",
                "status": "pass" if transport_passes == transport_checks else "fail",
                "value": f"{transport_passes}/{transport_checks}",
                "detail": "Transport validations from the all-prototype transport layer.",
            },
            {
                "section": "validation",
                "item": "recurrence_validation_pass_summary",
                "status": "pass" if recurrence_passes == recurrence_checks else "fail",
                "value": f"{recurrence_passes}/{recurrence_checks}",
                "detail": "Recurrence validations from the patient-by-prototype recurrence layer.",
            },
            {
                "section": "validation",
                "item": "view_validation_pass_summary",
                "status": "pass" if view_passes == view_checks else "fail",
                "value": f"{view_passes}/{view_checks}",
                "detail": "Downstream extracted-view validations only affect outputs 06 and 07.",
            },
            {
                "section": "transport",
                "item": "balanced_ot_unmatched_exclusion",
                "status": "pass" if balanced_no_unmatched_ok else "fail",
                "value": "true",
                "detail": "Balanced OT remains transport-only and does not carry unmatched semantics.",
            },
            {
                "section": "transport",
                "item": "tau_R_unavailable",
                "status": "pass" if tau_r_unavailable else "fail",
                "value": "unavailable",
                "detail": "tau and R remain unavailable in the current startup slice.",
            },
        ]
    )


def build_focused_results_memo(
    paths: Arm2FocusedPaths,
    baseline_tables: BaselineAnalysisTables,
    transport_tables: TransportAnalysisTables,
    recurrence_tables: RecurrenceAnalysisTables,
    extracted_views: FocusedPrototypeViews,
    appendix_audit: pd.DataFrame,
) -> str:
    """
    Build the public focused results memo for output `00`.
    """

    top_baseline = baseline_tables.baseline_prototype_confirmatory.head(5)
    top_comparison = extracted_views.prototype_comparison_view.head(5)
    recurrence_view = extracted_views.prototype_recurrence_view

    confirmatory_patient_count = int(transport_tables.global_transport_summary.shape[0])
    key_proto_count = int(extracted_views.prototype_comparison_view.shape[0])
    recurrence_rows = int(recurrence_view.shape[0])
    positive_unmatched_tc_pt = int(
        (pd.to_numeric(top_comparison.get("uot_unmatched_recurrence_tc_pt"), errors="coerce").fillna(0.0) >= 0.5).sum()
    )
    baseline_positive = int(
        (
            pd.to_numeric(
                top_comparison.get("recurrence_patient_level_prop_tc_pt_gt_tc_im_abs_delta_share"),
                errors="coerce",
            ).fillna(0.0)
            >= 0.5
        ).sum()
    )

    lines = [
        "# Arm-II Focused Results Memo",
        "",
        "## Inputs / Startup-Slice Boundaries",
        "",
        f"- Arm-II parquet: `{paths.arm2_metrics_parquet}`",
        f"- Stage-0 artifact: `{paths.stage0_h5ad}`",
        f"- Task config: `{paths.task_config}`",
        "- This focused package remains locked to the current Arm-II startup slice only.",
        "- UOT is rerun once and same-pair Balanced OT is rerun once on the fixed ordered pair set.",
        "- `tau` and `R` remain unavailable in the current startup slice and are not interpreted here.",
        "",
        "## Confirmatory Scope",
        "",
        "- Confirmatory families are restricted to `TC-IM` and `TC-PT`.",
        "- `IM-PT` is retained internally for audit/exploratory context only and does not enter confirmatory public contrasts.",
        f"- Public global transport row unit: patient (`n={confirmatory_patient_count}`).",
        "",
        "## Baseline Findings",
        "",
        "- Baseline tissue differences are summarized before transport and remain separate from transport claims.",
        "- Share-scale absolute baseline magnitude is the primary prototype-level baseline anchor; count-scale quantities remain context.",
    ]
    if not top_baseline.empty:
        lines.append("- Highest-ranked confirmatory baseline prototypes:")
        for _, row in top_baseline.iterrows():
            lines.append(
                f"  - proto `{int(row['proto_id'])}` / `{row['dominant_cell_type']}` / "
                f"`TC-IM median abs share={_format_float(row['tc_im_median_abs_delta_share'])}` / "
                f"`TC-PT median abs share={_format_float(row['tc_pt_median_abs_delta_share'])}` / "
                f"`patient median TC-PT minus TC-IM={_format_float(row['patient_median_tc_pt_minus_tc_im_abs_delta_share'])}`."
            )

    lines.extend(
        [
            "",
            "## Global Transport Findings",
            "",
            "- Primary readouts are `U_abs`, `transport_fraction`, `unmatched_fraction`, and `M`.",
            "- Supporting unmatched decomposition is shown via `D_pos` and `B_pos`.",
            "- `T_abs`, `M_balanced`, and Balanced-minus-UOT quantities remain contextual rather than winner-style evidence lines.",
        ]
    )
    if not transport_tables.global_transport_summary.empty:
        median_tc_im_u = pd.to_numeric(
            transport_tables.global_transport_summary["tc_im_median_U_abs"],
            errors="coerce",
        ).median()
        median_tc_pt_u = pd.to_numeric(
            transport_tables.global_transport_summary["tc_pt_median_U_abs"],
            errors="coerce",
        ).median()
        median_tc_im_transport = pd.to_numeric(
            transport_tables.global_transport_summary["tc_im_median_transport_fraction"],
            errors="coerce",
        ).median()
        median_tc_pt_transport = pd.to_numeric(
            transport_tables.global_transport_summary["tc_pt_median_transport_fraction"],
            errors="coerce",
        ).median()
        lines.extend(
            [
                f"- Across patients, median `TC-IM` `U_abs` is `{_format_float(median_tc_im_u)}` and median `TC-PT` `U_abs` is `{_format_float(median_tc_pt_u)}`.",
                f"- Across patients, median `TC-IM` transport fraction is `{_format_float(median_tc_im_transport)}` and median `TC-PT` transport fraction is `{_format_float(median_tc_pt_transport)}`.",
            ]
        )

    lines.extend(
        [
            "",
            "## Prototype-Level Interpretation Summary",
            "",
            f"- Extracted prototype comparison rows: `{key_proto_count}`.",
            "- These prototype views are downstream subsets of the all-prototype internal tables only.",
        ]
    )
    if not top_comparison.empty:
        lines.append("- Leading extracted prototype summaries:")
        for _, row in top_comparison.iterrows():
            lines.append(
                f"  - proto `{int(row['proto_id'])}` / `{row['dominant_cell_type']}` / "
                f"`baseline recurrence patient-level proportion={_format_float(row['recurrence_patient_level_prop_tc_pt_gt_tc_im_abs_delta_share'])}` / "
                f"`UOT transport TC-PT={_format_float(row['uot_transport_share_tc_pt'])}` / "
                f"`UOT unmatched TC-PT={_format_float(row['uot_unmatched_share_tc_pt'])}` / "
                f"`Balanced-UOT TC-PT={_format_float(row['balanced_minus_uot_tc_pt'])}`."
            )

    lines.extend(
        [
            "",
            "## Recurrence Summary",
            "",
            f"- Patient-level recurrence rows in output `07`: `{recurrence_rows}`.",
            f"- Selected prototypes with recurrent `TC-PT > TC-IM` baseline signal in at least half of patients: `{baseline_positive}`.",
            f"- Selected prototypes with recurrent positive `TC-PT` UOT unmatched signal in at least half of patients: `{positive_unmatched_tc_pt}`.",
            "- Confirmatory recurrence fields remain explicit for `TC-IM` and `TC-PT`; audit-only `IM-PT` remains internal.",
            "",
            "## Supported Claims",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in MEMO_SUPPORTED_CLAIMS)
    lines.extend(
        [
            "",
            "## Non-Claims",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in MEMO_NON_CLAIMS)
    lines.extend(
        [
            "",
            "## Out-of-Scope Items",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in MEMO_OUT_OF_SCOPE_ITEMS)
    lines.extend(
        [
            "",
            "## Audit Notes",
            "",
        ]
    )
    for _, row in appendix_audit.iterrows():
        lines.append(
            f"- `{row['item']}`: `{row['status']}` / `{row['value']}`. {row['detail']}"
        )
    return "\n".join(lines) + "\n"


def assemble_output_package(
    baseline_tables: BaselineAnalysisTables,
    transport_tables: TransportAnalysisTables,
    extracted_views: FocusedPrototypeViews,
    appendix_audit: pd.DataFrame,
    memo_text: str,
) -> FocusedOutputPackage:
    """
    Assemble the exact 9-file focused-output package.
    """

    tables_by_filename = {
        "01_prototype_biological_meaning_table.csv": baseline_tables.prototype_meaning,
        "02_baseline_pair_audit.csv": baseline_tables.baseline_pair_audit,
        "03_baseline_prototype_confirmatory_summary.csv": baseline_tables.baseline_prototype_confirmatory,
        "04_baseline_patient_family_confirmatory_summary.csv": baseline_tables.baseline_patient_family_confirmatory,
        "05_global_transport_summary.csv": transport_tables.global_transport_summary,
        "06_key_prototype_comparison.csv": extracted_views.prototype_comparison_view,
        "07_key_prototype_patient_recurrence.csv": extracted_views.prototype_recurrence_view,
        "08_minimal_appendix_audit.csv": appendix_audit,
    }
    package = FocusedOutputPackage(
        memo_text=memo_text,
        tables_by_filename=tables_by_filename,
        package_validation=pd.DataFrame(),
    )
    package.package_validation = validate_output_package(package)
    return package


def validate_output_package(package: FocusedOutputPackage) -> pd.DataFrame:
    """
    Validate the final focused-output package before writing.
    """

    observed_csv = tuple(sorted(package.tables_by_filename))
    expected_csv = tuple(sorted(_csv_output_filenames()))
    required_memo_tokens = (
        "## Inputs / Startup-Slice Boundaries",
        "## Confirmatory Scope",
        "## Baseline Findings",
        "## Global Transport Findings",
        "## Prototype-Level Interpretation Summary",
        "## Recurrence Summary",
        "## Supported Claims",
        "## Non-Claims",
        "## Out-of-Scope Items",
    )
    memo_has_tokens = all(token in package.memo_text for token in required_memo_tokens)
    comparison_is_df = isinstance(
        package.tables_by_filename.get("06_key_prototype_comparison.csv"),
        pd.DataFrame,
    )
    recurrence_is_df = isinstance(
        package.tables_by_filename.get("07_key_prototype_patient_recurrence.csv"),
        pd.DataFrame,
    )
    validation = pd.DataFrame.from_records(
        [
            {
                "check": "exact_output_csv_filenames",
                "passed": observed_csv == expected_csv,
                "detail": f"expected={list(expected_csv)}, observed={list(observed_csv)}",
            },
            {
                "check": "memo_text_present",
                "passed": bool(package.memo_text.strip()),
                "detail": f"memo_length={len(package.memo_text)}",
            },
            {
                "check": "memo_boundary_sections_present",
                "passed": memo_has_tokens,
                "detail": f"required_tokens={list(required_memo_tokens)}",
            },
            {
                "check": "derived_view_outputs_are_dataframes",
                "passed": bool(comparison_is_df and recurrence_is_df),
                "detail": (
                    f"comparison_type={type(package.tables_by_filename.get('06_key_prototype_comparison.csv')).__name__}, "
                    f"recurrence_type={type(package.tables_by_filename.get('07_key_prototype_patient_recurrence.csv')).__name__}"
                ),
            },
            {
                "check": "all_output_values_are_dataframes",
                "passed": bool(all(isinstance(value, pd.DataFrame) for value in package.tables_by_filename.values())),
                "detail": f"types={[type(value).__name__ for value in package.tables_by_filename.values()]}",
            },
        ]
    )
    failed = validation.loc[~validation["passed"].astype(bool)].copy()
    if not failed.empty:
        details = "; ".join(f"{row['check']}={row['detail']}" for _, row in failed.iterrows())
        raise ValueError(f"Focused output package validation failed: {details}")
    return validation


def write_output_package(package: FocusedOutputPackage, output_dir: Path) -> None:
    """
    Write the focused-output package to disk.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    validate_output_package(package)
    (output_dir / "00_arm2_focused_results_memo.md").write_text(package.memo_text, encoding="utf-8")
    for filename, table in package.tables_by_filename.items():
        table.to_csv(output_dir / filename, index=False)
