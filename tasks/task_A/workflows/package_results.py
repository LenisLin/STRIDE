"""Workflow wrapper for the Task A objective result packet surface.

This wrapper packages the descriptive atlas plus canonical Block 0-2 outputs.
Block 3 packet integration remains deferred and is intentionally rejected from
this active wrapper until a clean non-authority bridge is approved.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stride.errors import ContractError

from ..result_packet import validate_task_a_result_packet, write_task_a_result_packet as _write_task_a_result_packet

BLOCK3_PACKET_DEFERRED_MESSAGE = (
    "Block 3 packet integration is deferred / non-authority / pending clean bridge spec"
)


def write_task_a_result_packet(
    *,
    atlas_manifest_path: str | Path,
    prepare_manifest_path: str | Path,
    block0_bundle_path: str | Path,
    output_dir: str | Path,
    block0_suitability_report_path: str | Path | None = None,
    block1_bundle_path: str | Path | None = None,
    block2_manifest_path: str | Path | None = None,
    block3_manifest_path: str | Path | None = None,
) -> Path:
    """Build the task-local objective result packet.

    Block 3 integration is fail-fast from this active wrapper until a clean
    bridge spec is explicitly approved.
    """

    if block3_manifest_path is not None:
        raise ContractError(BLOCK3_PACKET_DEFERRED_MESSAGE)

    packet = _write_task_a_result_packet(
        atlas_manifest_path=atlas_manifest_path,
        prepare_manifest_path=prepare_manifest_path,
        block0_bundle_path=block0_bundle_path,
        output_dir=output_dir,
        block0_suitability_report_path=block0_suitability_report_path,
        block1_bundle_path=block1_bundle_path,
        block2_manifest_path=block2_manifest_path,
        block3_manifest_path=block3_manifest_path,
    )
    return packet.manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package the canonical Task A atlas plus Block 0-2 surfaces into "
            "one objective packet. Block 3 integration is deferred."
        )
    )
    parser.add_argument("--atlas-manifest", required=True)
    parser.add_argument("--prepare-manifest", required=True)
    parser.add_argument("--block0-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--block0-suitability-report")
    parser.add_argument("--block1-bundle")
    parser.add_argument("--block2-manifest")
    parser.add_argument("--block3-manifest")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        manifest_path = write_task_a_result_packet(
            atlas_manifest_path=args.atlas_manifest,
            prepare_manifest_path=args.prepare_manifest,
            block0_bundle_path=args.block0_bundle,
            output_dir=args.output_dir,
            block0_suitability_report_path=args.block0_suitability_report,
            block1_bundle_path=args.block1_bundle,
            block2_manifest_path=args.block2_manifest,
            block3_manifest_path=args.block3_manifest,
        )
        validate_task_a_result_packet(manifest_path)
        print(f"Wrote Task A result packet manifest to {manifest_path}")
    except (ContractError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


__all__ = ["validate_task_a_result_packet", "write_task_a_result_packet"]


if __name__ == "__main__":  # pragma: no cover
    main()
