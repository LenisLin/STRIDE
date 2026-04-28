from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


OUTPUT_DIR = Path("tasks/task_A/result_packets/2026-04-03_weekly_group_meeting")
DRAWIO_PATH = OUTPUT_DIR / "stride-weekly-group-meeting.drawio"
NOTES_PATH = OUTPUT_DIR / "stride-weekly-group-meeting-notes.md"

BLUE = "#0170C1"
BLUE_DARK = "#0B3C68"
BLUE_SOFT = "#E8F3FC"
BLUE_BORDER = "#C7DFF3"
TEXT = "#0F172A"
MUTED = "#475569"
BG = "#F8FBFF"
WHITE = "#FFFFFF"
GOLD = "#C9A84C"
GREEN = "#1E9B63"
GREEN_SOFT = "#DDF5E8"
AMBER = "#B45309"
AMBER_SOFT = "#FFF3DA"
GRAY_SOFT = "#E2E8F0"


def xml_escape(value: str) -> str:
    return escape(value, {'"': "&quot;"})


@dataclass
class IdFactory:
    prefix: str
    counter: int = 1

    def next(self) -> str:
        self.counter += 1
        return f"{self.prefix}_{self.counter}"


def rect(
    ids: IdFactory,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    fill: str,
    stroke: str = "none",
    rounded: bool = False,
    extra: str = "",
) -> str:
    cell_id = ids.next()
    rounded_value = "1" if rounded else "0"
    style = (
        f"rounded={rounded_value};whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};"
    )
    if rounded:
        style += "arcSize=10;"
    if extra:
        style += extra
    return (
        f'<mxCell id="{cell_id}" value="" style="{style}" vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
        f"</mxCell>"
    )


def text(
    ids: IdFactory,
    value: str,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    size: int,
    color: str = TEXT,
    bold: bool = False,
    align: str = "left",
    valign: str = "top",
    font: str = "微软雅黑",
    extra: str = "",
) -> str:
    cell_id = ids.next()
    font_style = "1" if bold else "0"
    style = (
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"align={align};verticalAlign={valign};fontSize={size};"
        f"fontColor={color};fontFamily={font};fontStyle={font_style};"
    )
    if extra:
        style += extra
    return (
        f'<mxCell id="{cell_id}" value="{xml_escape(value)}" '
        f'style="{style}" vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
        f"</mxCell>"
    )


def diagram(name: str, diagram_id: str, cells: list[str]) -> str:
    body = "\n        ".join(cells)
    return f"""  <diagram id="{diagram_id}" name="{xml_escape(name)}">
    <mxGraphModel dx="1920" dy="1080" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1920" pageHeight="1080" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {body}
      </root>
    </mxGraphModel>
  </diagram>"""


def make_common_frame(ids: IdFactory, title_text: str) -> list[str]:
    return [
        rect(ids, 0, 0, 1920, 1080, fill=BG),
        rect(ids, 0, 0, 1920, 116, fill=BLUE, stroke="none"),
        rect(ids, 0, 116, 1920, 6, fill=GOLD, stroke="none"),
        rect(ids, 42, 24, 20, 68, fill=WHITE, stroke="none"),
        text(ids, title_text, 92, 18, 1180, 76, size=30, color=WHITE, bold=True, valign="middle"),
        text(ids, "STRIDE weekly group update", 1370, 34, 450, 34, size=17, color=WHITE, align="right"),
        text(ids, "2026-04-03", 1640, 1038, 220, 28, size=14, color=MUTED, align="right", valign="middle"),
    ]


def slide_background_and_question() -> tuple[str, int]:
    ids = IdFactory("s1")
    cells = make_common_frame(ids, "故事背景与生物学问题")
    cells.append(
        text(
            ids,
            "Why Task A needs a bounded proxy design instead of direct longitudinal claims",
            120,
            156,
            1200,
            34,
            size=18,
            color=MUTED,
        )
    )

    cells.extend(
        [
            rect(ids, 120, 212, 820, 720, fill=WHITE, stroke=BLUE_BORDER, rounded=True, extra="strokeWidth=2;"),
            rect(ids, 146, 238, 196, 36, fill=BLUE_SOFT, stroke="none", rounded=True),
            text(ids, "Biological setting", 168, 244, 220, 24, size=18, color=BLUE_DARK, bold=True),
            text(
                ids,
                "Task A 使用 single-timepoint IMC cohort，当前保留 32 patients，\n观测层组织域为 TC / IM / PT，观测单元是 ROI / FOV。",
                150,
                300,
                760,
                82,
                size=18,
                color=TEXT,
            ),
            rect(ids, 146, 416, 176, 36, fill=GREEN_SOFT, stroke="none", rounded=True),
            text(ids, "Why this is hard", 168, 422, 220, 24, size=18, color=GREEN, bold=True),
            text(
                ids,
                "我们没有真实 before-after 纵向配对；如果直接把 TC -> IM -> PT 当作时间，\n会产生过强因果/时序宣称；如果只做静态丰度比较，又抓不住 patient-level relation structure。",
                150,
                478,
                760,
                108,
                size=18,
                color=TEXT,
            ),
            rect(ids, 146, 622, 198, 36, fill=AMBER_SOFT, stroke="none", rounded=True),
            text(ids, "Bounded question", 168, 628, 240, 24, size=18, color=AMBER, bold=True),
            rect(ids, 150, 686, 760, 182, fill=BLUE_DARK, stroke="none", rounded=True),
            text(
                ids,
                "Under the ordered tissue-domain proxy TC -> IM -> PT,\ncan STRIDE recover non-random and biologically interpretable\npatient-level remodeling relations on a shared tissue-agnostic state axis?",
                192,
                734,
                680,
                96,
                size=24,
                color=WHITE,
                bold=True,
                align="center",
                valign="middle",
            ),
        ]
    )

    cells.extend(
        [
            rect(ids, 980, 212, 820, 720, fill=WHITE, stroke=BLUE_BORDER, rounded=True, extra="strokeWidth=2;"),
            rect(ids, 1010, 248, 208, 144, fill=BLUE_SOFT, stroke="none", rounded=True),
            rect(ids, 1286, 248, 208, 144, fill=GREEN_SOFT, stroke="none", rounded=True),
            rect(ids, 1562, 248, 208, 144, fill=AMBER_SOFT, stroke="none", rounded=True),
            text(ids, "TC", 1086, 276, 56, 38, size=28, color=BLUE_DARK, bold=True, align="center"),
            text(ids, "tumor core", 1034, 322, 160, 28, size=18, color=TEXT, align="center"),
            text(ids, "IM", 1362, 276, 56, 38, size=28, color=GREEN, bold=True, align="center"),
            text(ids, "invasive margin", 1308, 322, 164, 28, size=18, color=TEXT, align="center"),
            text(ids, "PT", 1638, 276, 56, 38, size=28, color=AMBER, bold=True, align="center"),
            text(ids, "peritumoral tissue", 1580, 322, 176, 28, size=18, color=TEXT, align="center"),
            text(ids, "→", 1230, 294, 26, 40, size=32, color=BLUE_DARK, bold=True),
            text(ids, "→", 1506, 294, 26, 40, size=32, color=BLUE_DARK, bold=True),
            text(
                ids,
                "Observation-layer tissue strata only.\nThey define the proxy ordering, not canonical state identity.",
                1070,
                430,
                640,
                56,
                size=17,
                color=MUTED,
                align="center",
            ),
            rect(ids, 1030, 524, 720, 144, fill=BG, stroke=BLUE_BORDER, rounded=True, extra="strokeWidth=2;"),
            text(ids, "Shared tissue-agnostic K-state basis", 1070, 548, 640, 30, size=22, color=BLUE_DARK, bold=True, align="center"),
            rect(ids, 1080, 602, 120, 34, fill=BLUE_SOFT, stroke="none", rounded=True),
            rect(ids, 1218, 602, 120, 34, fill=GREEN_SOFT, stroke="none", rounded=True),
            rect(ids, 1356, 602, 140, 34, fill=AMBER_SOFT, stroke="none", rounded=True),
            rect(ids, 1514, 602, 120, 34, fill=BLUE_SOFT, stroke="none", rounded=True),
            rect(ids, 1652, 602, 78, 34, fill=GREEN_SOFT, stroke="none", rounded=True),
            text(ids, "Tumor", 1110, 608, 60, 22, size=16, color=BLUE_DARK, bold=True, align="center"),
            text(ids, "Myeloid", 1242, 608, 72, 22, size=16, color=GREEN, bold=True, align="center"),
            text(ids, "Lymphoid", 1390, 608, 72, 22, size=16, color=AMBER, bold=True, align="center"),
            text(ids, "Stromal", 1538, 608, 72, 22, size=16, color=BLUE_DARK, bold=True, align="center"),
            text(ids, "IF", 1678, 608, 30, 22, size=16, color=GREEN, bold=True, align="center"),
            rect(ids, 1030, 718, 720, 150, fill=BLUE_DARK, stroke="none", rounded=True),
            text(ids, "STRIDE output", 1080, 744, 200, 28, size=22, color=WHITE, bold=True),
            text(
                ids,
                "A = retention / remodeling relation\n"
                "d = source-side depletion tendency\n"
                "e = target-side emergence tendency",
                1080,
                790,
                620,
                74,
                size=19,
                color=WHITE,
            ),
        ]
    )

    return diagram("Background And Question", "background_and_question", cells), len(cells)


def slide_task_a_design() -> tuple[str, int]:
    ids = IdFactory("s2")
    cells = make_common_frame(ids, "Task A 设计流程与当前结论")
    cells.append(
        text(
            ids,
            "Atlas -> Null gate -> Discovery -> Robustness -> Comparator validation",
            120,
            156,
            1200,
            34,
            size=18,
            color=MUTED,
        )
    )

    steps = [
        ("Atlas", "state-axis context", "done", BLUE_SOFT, BLUE_DARK),
        ("Block 0", "TC-IM vs randomized null", "pass", GREEN_SOFT, GREEN),
        ("Block 1", "TC-IM vs TC-PT", "evidence-ready", BLUE_SOFT, BLUE_DARK),
        ("Block 2", "robustness over summaries", "next", AMBER_SOFT, AMBER),
        ("Block 3", "baseline / ablation / semisynthetic", "later", GRAY_SOFT, MUTED),
    ]
    start_x = 120
    step_w = 260
    gap = 55
    step_y = 220
    for index, (title_value, subtitle, status, pill_fill, pill_text) in enumerate(steps):
        x = start_x + index * (step_w + gap)
        cells.extend(
            [
                rect(ids, x, step_y, step_w, 170, fill=WHITE, stroke=BLUE_BORDER, rounded=True, extra="strokeWidth=2;"),
                text(ids, title_value, x + 26, step_y + 24, step_w - 52, 34, size=24, color=TEXT, bold=True, align="center"),
                text(ids, subtitle, x + 24, step_y + 76, step_w - 48, 54, size=17, color=MUTED, align="center"),
                rect(ids, x + 66, step_y + 132, 128, 26, fill=pill_fill, stroke="none", rounded=True),
                text(ids, status, x + 86, step_y + 136, 88, 20, size=14, color=pill_text, bold=True, align="center"),
            ]
        )
        if index < len(steps) - 1:
            cells.append(
                text(
                    ids,
                    "→",
                    x + step_w + 18,
                    step_y + 56,
                    24,
                    44,
                    size=34,
                    color=BLUE_DARK,
                    bold=True,
                )
            )

    cells.extend(
        [
            rect(ids, 120, 470, 820, 430, fill=WHITE, stroke=BLUE_BORDER, rounded=True, extra="strokeWidth=2;"),
            rect(ids, 980, 470, 820, 430, fill=WHITE, stroke=BLUE_BORDER, rounded=True, extra="strokeWidth=2;"),
            rect(ids, 146, 494, 176, 36, fill=BLUE_SOFT, stroke="none", rounded=True),
            rect(ids, 1006, 494, 240, 36, fill=AMBER_SOFT, stroke="none", rounded=True),
            text(ids, "当前最强证据", 168, 500, 220, 24, size=18, color=BLUE_DARK, bold=True),
            text(ids, "现在怎么讲最稳", 1028, 500, 280, 24, size=18, color=AMBER, bold=True),
            rect(ids, 150, 560, 760, 84, fill=BLUE_SOFT, stroke="none", rounded=True),
            rect(ids, 150, 664, 760, 112, fill=GREEN_SOFT, stroke="none", rounded=True),
            rect(ids, 150, 792, 760, 86, fill=BG, stroke=BLUE_BORDER, rounded=True, extra="strokeWidth=2;"),
            text(
                ids,
                "Atlas：shared state axis 已有生物学可解释性，且 major communities 在 cohort 中可复现。",
                176,
                588,
                710,
                30,
                size=18,
                color=TEXT,
            ),
            text(
                ids,
                "Block 0：真实 TC-IM 不像 randomized null。\ncontinuity median +0.099，expected-direction support 24/32。",
                176,
                690,
                710,
                58,
                size=18,
                color=TEXT,
            ),
            text(
                ids,
                "Block 1：TC-IM 比 TC-PT 更 retention-like。\nself-retention 32/32；depletion 32/32；emergence supportive 28/32。\nTumor-dominant source communities 更保留在 TC-IM；TC-PT 更偏向 PT-heavy immune / interface targets。",
                176,
                810,
                710,
                56,
                size=17,
                color=TEXT,
            ),
            text(
                ids,
                "1. 这仍是 ordered tissue-domain proxy，不是 literal time。\n"
                "2. 当前最稳的是 family-level confirmatory comparison；target-side emergence 只宜作 supportive reading。\n"
                "3. Task A 还不能宣称 overall pass；更准确的阶段表述是 ready to enter Block 2 robustness validation。\n"
                "4. 下一步先做 Block 2 robustness，再进入 Block 3 baseline / semisynthetic validation。",
                1010,
                566,
                748,
                244,
                size=18,
                color=TEXT,
            ),
            rect(ids, 1006, 818, 748, 70, fill=BLUE_DARK, stroke="none", rounded=True),
            text(
                ids,
                "Best current statement: Task A has completed atlas + null gate + primary discovery and is ready to enter robustness validation.",
                1038,
                832,
                684,
                40,
                size=19,
                color=WHITE,
                bold=True,
                align="center",
                valign="middle",
            ),
        ]
    )

    return diagram("TaskA Design And Evidence", "taska_design_and_evidence", cells), len(cells)


def build_drawio() -> tuple[str, list[tuple[str, int]]]:
    slides = [slide_background_and_question(), slide_task_a_design()]
    counts = [
        ("Background And Question", slides[0][1]),
        ("TaskA Design And Evidence", slides[1][1]),
    ]
    xml = [
        '<mxfile host="app.diagrams.net" modified="2026-04-03T00:00:00.000Z" agent="Codex" version="24.7.17">',
        slides[0][0],
        slides[1][0],
        "</mxfile>",
        "",
    ]
    return "\n".join(xml), counts


def build_notes() -> str:
    return """# STRIDE Weekly Group Meeting Notes

## Slide 1

先把故事背景讲清楚。Task A 面对的不是标准的纵向 before-after 设计，而是一个 single-timepoint、multi-ROI、within-patient 的 IMC 队列，组织域是 TC、IM、PT。真正要回答的问题不是“我们能不能直接证明时间上的转变”，而是“在这样一个有 proxy 性质的设计里，STRIDE 能不能恢复 patient-level remodeling relation”。右侧这张图的重点是：TC、IM、PT 只是 observation-layer strata；真正的分析对象是共享 tissue-agnostic state basis 上的 A、d、e。

## Slide 2

第二页就按流程图讲。先用 atlas 建立状态轴的生物学可读性，再用 Block 0 排除 broken-null 解释，再让 Block 1 做 real-data discovery。现在最稳的结论是 family-level confirmatory comparison：TC-IM 比 TC-PT 更 retention-like，而且肿瘤主导 source communities 在 TC-IM 里更保留。这里要主动加一句边界说明：这不是 literal time，不是 direct lineage tracing，也不是 Task A overall pass。最准确的阶段判断是：Task A 已完成 descriptive layer、null gate 和 primary discovery，下一步进入 Block 2 robustness，再进入 Block 3 comparator / semisynthetic validation。
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    drawio_xml, counts = build_drawio()
    DRAWIO_PATH.write_text(drawio_xml, encoding="utf-8")
    NOTES_PATH.write_text(build_notes(), encoding="utf-8")

    if drawio_xml.count("<diagram ") != 2:
        raise RuntimeError("Expected exactly 2 draw.io diagrams.")
    for slide_name, count in counts:
        if count < 15:
            raise RuntimeError(f"{slide_name} has too few cells: {count}")

    print(f"Wrote {DRAWIO_PATH}")
    print(f"Wrote {NOTES_PATH}")
    print("Slide cell counts:")
    for slide_name, count in counts:
        print(f"- {slide_name}: {count}")


if __name__ == "__main__":
    main()
