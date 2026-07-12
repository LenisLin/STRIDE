from __future__ import annotations

from pathlib import Path


def test_figure2_script_requires_complete_block3_and_validates_staging_pdfs() -> None:
    script = Path("tasks/task_A/visualization/figure2_taskA_panels.R").read_text(
        encoding="utf-8"
    )

    assert 'Sys.getenv(\n  "TASK_A_RESULT_ROOT"' in script
    assert 'Sys.getenv(\n  "TASK_A_FIGURE_DIR"' in script
    assert 'block3_manifest$status, "complete"' in script
    assert 'block3_manifest$execution_scope, "formal_full_data"' in script
    assert 'length(rendered_pdfs) != 20L' in script
    assert 'system2("pdfinfo"' in script
    assert 'file.rename(FIG_DIR, FINAL_FIG_DIR)' in script
