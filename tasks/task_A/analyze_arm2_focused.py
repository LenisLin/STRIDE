"""
Module: tasks.task_A.analyze_arm2_focused

Operative end-to-end Arm-II focused analysis workflow.

This entrypoint is limited to the current Arm-II startup slice only.

Prototype-selection boundary:
- Any selected-prototype list applies only to downstream extracted views used
  for outputs `06` and `07`.
- Upstream compute, baseline, transport, and recurrence layers remain
  all-prototype-first regardless of the selected-prototype list.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    SRC_ROOT = REPO_ROOT / "src"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

from tasks.task_A.arm2.analysis_baseline import build_baseline_tables, build_prototype_meaning_table
from tasks.task_A.arm2.analysis_compute import compute_all_surfaces
from tasks.task_A.arm2.analysis_contract import Arm2FocusedPaths, FocusedOutputPackage
from tasks.task_A.arm2.analysis_io import load_inputs
from tasks.task_A.arm2.analysis_output import (
    assemble_output_package,
    build_focused_results_memo,
    build_minimal_appendix_audit_table,
    write_output_package,
)
from tasks.task_A.arm2.analysis_recurrence import build_recurrence_tables
from tasks.task_A.arm2.analysis_transport import build_transport_tables
from tasks.task_A.arm2.analysis_views import build_focused_prototype_views

# ---------------------------------------------------------------------------
# CLI and path handling
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse CLI arguments for the operative Arm-II focused analysis workflow.
    """

    parser = argparse.ArgumentParser(
        description="Run the operative Arm-II focused end-to-end analysis workflow.",
    )
    parser.add_argument(
        "--arm2-metrics-parquet",
        required=True,
        help="Path to the Arm-II metrics parquet.",
    )
    parser.add_argument(
        "--stage0-h5ad",
        required=True,
        help="Path to the frozen Stage-0 .h5ad artifact.",
    )
    parser.add_argument(
        "--task-config",
        required=True,
        help="Path to the Task-A config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the focused 9-file package will be written.",
    )
    parser.add_argument(
        "--prototype-view-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional downstream subset for outputs 06 and 07 only.",
    )
    return parser.parse_args(argv)


def build_paths_from_args(args: argparse.Namespace) -> Arm2FocusedPaths:
    """Build the resolved path bundle for one focused Arm-II run."""

    prototype_view_ids = None
    if args.prototype_view_ids:
        prototype_view_ids = tuple(sorted({int(proto_id) for proto_id in args.prototype_view_ids}))
    return Arm2FocusedPaths(
        arm2_metrics_parquet=Path(args.arm2_metrics_parquet).expanduser().resolve(),
        stage0_h5ad=Path(args.stage0_h5ad).expanduser().resolve(),
        task_config=Path(args.task_config).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        prototype_view_ids=prototype_view_ids,
    )


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def run_focused_analysis(paths: Arm2FocusedPaths) -> FocusedOutputPackage:
    """
    Run the operative Arm-II focused-analysis block.

    End-to-end data flow:
    1. Load frozen inputs.
    2. Build all-prototype biological meaning.
    3. Build all-prototype baseline tables.
    4. Compute all surfaces on the fixed ordered pair set.
    5. Build all-prototype transport/unmatched tables.
    6. Build all-prototype recurrence tables.
    7. Build focused prototype views.
    8. Build the minimal appendix audit.
    9. Build the focused memo.
    10. Assemble the full focused 9-file output package.
    """

    inputs = load_inputs(paths)
    prototype_meaning = build_prototype_meaning_table(inputs.stage0)
    baseline_tables = build_baseline_tables(
        inputs=inputs,
        prototype_meaning=prototype_meaning,
    )
    computed = compute_all_surfaces(inputs)
    transport_tables = build_transport_tables(
        inputs=inputs,
        computed=computed,
        prototype_meaning=prototype_meaning,
    )
    recurrence_tables = build_recurrence_tables(
        baseline_tables=baseline_tables,
        transport_tables=transport_tables,
    )
    extracted_views = build_focused_prototype_views(
        selected_proto_ids=paths.prototype_view_ids,
        baseline_tables=baseline_tables,
        transport_tables=transport_tables,
        recurrence_tables=recurrence_tables,
    )
    appendix_audit = build_minimal_appendix_audit_table(
        inputs=inputs,
        computed=computed,
        baseline_tables=baseline_tables,
        transport_tables=transport_tables,
        recurrence_tables=recurrence_tables,
        extracted_views=extracted_views,
    )
    memo_text = build_focused_results_memo(
        paths=paths,
        baseline_tables=baseline_tables,
        transport_tables=transport_tables,
        recurrence_tables=recurrence_tables,
        extracted_views=extracted_views,
        appendix_audit=appendix_audit,
    )
    return assemble_output_package(
        baseline_tables=baseline_tables,
        transport_tables=transport_tables,
        extracted_views=extracted_views,
        appendix_audit=appendix_audit,
        memo_text=memo_text,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for the full Arm-II focused rewrite package."""

    args = parse_args(argv)
    paths = build_paths_from_args(args)
    package = run_focused_analysis(paths)
    write_output_package(package, paths.output_dir)


if __name__ == "__main__":
    main()
