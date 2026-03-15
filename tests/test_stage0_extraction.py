from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXTRACT_SCRIPT = ROOT / "tasks" / "task_A" / "extract_crlm_coldata.R"


def _run_r(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["Rscript", "-e", code],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_extract_position_parsing(tmp_path: Path) -> None:
    out_path = tmp_path / "parsed_positions.csv"
    code = textwrap.dedent(
        f"""
        source("{EXTRACT_SCRIPT.as_posix()}")
        parsed <- parse_position(c("(1.0, 2.5)", "(3,4)", "(-5.25, 6.75)"))
        utils::write.csv(parsed, "{out_path.as_posix()}", row.names = FALSE)
        """
    )

    _run_r(code)
    parsed = pd.read_csv(out_path)

    assert parsed.columns.tolist() == ["x", "y"]
    assert parsed.shape == (3, 2)
    assert parsed["x"].tolist() == [1.0, 3.0, -5.25]
    assert parsed["y"].tolist() == [2.5, 4.0, 6.75]


def test_extract_filters_tls_and_patient_subset(tmp_path: Path) -> None:
    out_path = tmp_path / "filtered_rois.csv"
    code = textwrap.dedent(
        f"""
        source("{EXTRACT_SCRIPT.as_posix()}")
        df <- data.frame(
          ID = c(
            "P1_TC_1", "P1_TC_1", "P1_TC_2", "P1_IM_1", "P1_IM_2", "P1_PT_1", "P1_PT_2", "P1_TLS_1",
            "P2_TC_1", "P2_TC_2", "P2_IM_1", "P2_IM_2", "P2_PT_1"
          ),
          PID = c(
            rep("P1", 8),
            rep("P2", 5)
          ),
          Tissue = c(
            "TC", "TC", "TC", "IM", "IM", "PT", "PT", "TLS",
            "TC", "TC", "IM", "IM", "PT"
          ),
          Position = c(
            "(0,0)", "(1,1)", "(2,2)", "(3,3)", "(4,4)", "(5,5)", "(6,6)", "(7,7)",
            "(8,8)", "(9,9)", "(10,10)", "(11,11)", "(12,12)"
          ),
          SubType = rep("SubtypeA", 13),
          Area = rep(10, 13),
          stringsAsFactors = FALSE
        )
        filtered <- filter_stage0_cells(df)
        roi_df <- unique(filtered[c("ID", "PID", "Tissue")])
        utils::write.csv(roi_df, "{out_path.as_posix()}", row.names = FALSE)
        """
    )

    _run_r(code)
    filtered = pd.read_csv(out_path)

    assert set(filtered["PID"]) == {"P1"}
    assert set(filtered["Tissue"]) == {"TC", "IM", "PT"}
    assert "TLS" not in set(filtered["Tissue"])
    assert set(filtered["ID"]) == {
        "P1_TC_1",
        "P1_TC_2",
        "P1_IM_1",
        "P1_IM_2",
        "P1_PT_1",
        "P1_PT_2",
    }


def test_extract_roi_clinical_stability(tmp_path: Path) -> None:
    roi_out = tmp_path / "roi_clinical.csv"
    err_out = tmp_path / "stability_error.txt"
    code = textwrap.dedent(
        f"""
        source("{EXTRACT_SCRIPT.as_posix()}")
        filtered <- data.frame(
          ID = c("R1", "R1", "R2", "R2"),
          PID = c("P1", "P1", "P2", "P2"),
          Tissue = c("TC", "TC", "IM", "IM"),
          Position = c("(0,0)", "(1,1)", "(2,2)", "(3,3)"),
          SubType = c("A", "B", "A", "C"),
          Area = c(10, 11, 12, 13),
          x = c(0, 1, 2, 3),
          y = c(0, 1, 2, 3),
          Prognosis = c("good", "good", "bad", "bad"),
          Batch = c("B1", "B1", "B2", "B2"),
          stringsAsFactors = FALSE
        )
        roi_table <- build_roi_clinical_table(filtered)
        utils::write.csv(roi_table, "{roi_out.as_posix()}", row.names = FALSE)

        unstable <- filtered
        unstable$Prognosis[2] <- "changed"
        msg <- tryCatch({{
          build_roi_clinical_table(unstable)
          ""
        }}, error = function(err) {{
          err$message
        }})
        writeLines(msg, "{err_out.as_posix()}")
        """
    )

    _run_r(code)
    roi_table = pd.read_csv(roi_out)
    err_msg = err_out.read_text(encoding="utf-8").strip()

    assert roi_table.shape[0] == 2
    assert set(roi_table.columns) >= {"ID", "PID", "Tissue", "Prognosis", "Batch"}
    assert "Position" not in roi_table.columns
    assert "SubType" not in roi_table.columns
    assert "Area" not in roi_table.columns
    assert "x" not in roi_table.columns
    assert "y" not in roi_table.columns
    assert "Prognosis" in err_msg
