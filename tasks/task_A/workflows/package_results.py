"""Workflow wrapper for the Task A objective result packet surface.

This wrapper packages the descriptive atlas plus canonical Block 0/1 outputs.
Block 3 packet integration remains outside this active wrapper.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..result_packet import validate_task_a_result_packet, write_task_a_result_packet as _write_task_a_result_packet


def write_task_a_result_packet(
    *,
    atlas_manifest_path: str | Path,
    prepare_manifest_path: str | Path,
    block0_calibration_manifest_path: str | Path,
    output_dir: str | Path,
    block1_bundle_path: str | Path | None = None,
) -> Path:
    """Build the task-local objective result packet."""

    packet = _write_task_a_result_packet(
        atlas_manifest_path=atlas_manifest_path,
        prepare_manifest_path=prepare_manifest_path,
        block0_calibration_manifest_path=block0_calibration_manifest_path,
        output_dir=output_dir,
        block1_bundle_path=block1_bundle_path,
    )
    return packet.manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package the canonical Task A atlas plus Block 0/1 surfaces into "
            "one objective packet."
        )
    )
    parser.add_argument("--atlas-manifest", required=True)
    parser.add_argument("--prepare-manifest", required=True)
    parser.add_argument("--block0-calibration-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--block1-bundle")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        manifest_path = write_task_a_result_packet(
            atlas_manifest_path=args.atlas_manifest,
            prepare_manifest_path=args.prepare_manifest,
            block0_calibration_manifest_path=args.block0_calibration_manifest,
            output_dir=args.output_dir,
            block1_bundle_path=args.block1_bundle,
        )
        validate_task_a_result_packet(manifest_path)
        print(f"Wrote Task A result packet manifest to {manifest_path}")
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


__all__ = ["validate_task_a_result_packet", "write_task_a_result_packet"]


if __name__ == "__main__":  # pragma: no cover
    main()
