"""Run the bounded Task A Block 0 execution/analyze CLI.

Block 0 `execute` writes a reusable real/null full-STRIDE fit cache from Task A
config, Stage 0 h5ad, run controls, and optional selectors. Block 0 `analyze`
derives calibration tables from that cache. Neither layer consumes prepare,
descriptive-atlas, old suitability, or passed-bundle artifacts, and neither
emits interpretation or execution decisions. See
`tasks/task_A/README.md`, `tasks/task_A/contracts/artifact_contracts.md`, and
`tasks/task_A/contracts/design_freeze.py`.
"""
from __future__ import annotations

from .cli import main


if __name__ == "__main__":
    main()
