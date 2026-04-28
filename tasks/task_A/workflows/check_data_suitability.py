"""Pre-Block 0 Task A data suitability report wrapper.

This surface reuses Step 1 semantics to emit a compatibility report over a
Stage 0 h5ad. It must hard-fail semantic misalignment and never substitutes
for Block 0 passage.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stride.errors import ContractError

from .prepare import write_task_a_pre_block0_data_suitability_report


def check_task_a_pre_block0_data_suitability(
    *,
    config_path: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
) -> Path:
    return write_task_a_pre_block0_data_suitability_report(
        config_path=config_path,
        data_path=data_path,
        output_dir=output_dir,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Task-A data suitability before Block 0.",
    )
    parser.add_argument("--task-config", required=True, help="Path to task_A config.yaml")
    parser.add_argument("--stage0-h5ad", required=True, help="Path to Stage 0 h5ad")
    parser.add_argument("--output-dir", required=True, help="Output directory for the suitability report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        report_path = check_task_a_pre_block0_data_suitability(
            config_path=args.task_config,
            data_path=args.stage0_h5ad,
            output_dir=args.output_dir,
        )
        print(f"Wrote pre-Block 0 Task A data suitability report to {report_path}")
    except (ContractError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()


__all__ = ["check_task_a_pre_block0_data_suitability"]
