"""CLI for the Task A descriptive atlas."""
from __future__ import annotations

import argparse

from .atlas import write_task_a_descriptive_atlas
from .contracts import DEFAULT_MAX_OVERLAY_COMMUNITIES, DescriptiveAtlasContractError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the Task A descriptive atlas from a Stage0 h5ad.",
    )
    parser.add_argument("--task-config", required=True, help="Path to tasks/task_A/config.yaml")
    parser.add_argument("--stage0-h5ad", required=True, help="Path to the Task A Stage0 h5ad")
    parser.add_argument("--output-dir", required=True, help="Output directory for atlas artifacts")
    parser.add_argument(
        "--patient-id",
        action="append",
        dest="patient_ids",
        default=None,
        help="Optional patient id selector; repeat for multiple patients",
    )
    parser.add_argument(
        "--max-overlay-communities",
        type=int,
        default=DEFAULT_MAX_OVERLAY_COMMUNITIES,
        help="Maximum number of top communities to render as representative overlays",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        manifest = write_task_a_descriptive_atlas(
            config_path=args.task_config,
            stage0_h5ad=args.stage0_h5ad,
            output_dir=args.output_dir,
            patient_ids=args.patient_ids,
            max_overlay_communities=args.max_overlay_communities,
        )
        print(f"Wrote descriptive atlas index {manifest['output_index']}")
    except (DescriptiveAtlasContractError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


__all__ = ["main", "parse_args"]
