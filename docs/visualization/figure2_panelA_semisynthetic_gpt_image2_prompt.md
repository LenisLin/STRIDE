# Main Figure 2A Semi-Synthetic Benchmark Prompt

```text
Create only the lower half of STRIDE Main Figure 2A: a semi-synthetic validation benchmark schematic.

Canvas and use:
Wide horizontal scientific figure panel, approximately 180 mm main-figure width, aspect ratio 3:1 to 3.5:1. White background. This is a single lower-panel schematic, not the clinical atlas section.

Overall visual concept:
Design the workflow as an elegant "semi-synthetic validation console" or "assay strip" rather than a plain flowchart. Use one large translucent rounded container with soft lavender-blue fill, clean black outlines, subtle shadows, and a refined Nature Methods review-figure style. Use layered cards, small icons, FOV tiles, vector bars, locks, beads, matrix blocks, and branching pipes to replace long text.

Tone:
Polished, compact, biomedical, visually rich, but not cluttered. Give the layout some creative freedom: curved arrows, gradient connector ribbons, overlapping cards, depth shadows, and small inset diagrams are allowed if they improve clarity.

Required scientific story:
This panel shows a bounded semi-synthetic benchmark. It uses real held-out source FOVs and train-derived TC-IM templates to generate target FOVs with hidden benchmark truth, then compares STRIDE/ablations and transport baselines against that hidden truth.

Left-to-right workflow, with five visually distinct modules:

1. Repeated split
Show layered dataset cards or patient-sheet icons.
One card is divided into blue "24 train" and peach "8 held-out".
Show stacked shadows for repetition and a small label "10 reruns".
Keep this mostly visual.

2. Train-derived templates
Show several template cards from training data.
Each card contains a red stacked vector or bar labelled x_TC and a teal stacked vector or bar labelled y_IM.
Use colored community blocks or bead colors inside the bars.
Minimal label: "TC-IM templates".

3. Hidden benchmark truth
Show a locked central box, safe, or shield.
Inside it place three compact symbols: A_p, d_p, e_p.
Add the formula: y = xA + e.
Make it visually clear that these quantities are hidden from methods and used only for scoring.

4. Semi-synthetic FOV generation
Show held-out source FOV tiles on the left: square spatial fields with colored cells or community dots.
Show generated target FOV tiles on the right: similar FOV tiles with dashed outlines, synthetic glow, or translucent overlay.
Use arrows or a transformation ribbon from source to generated target.
Brief labels only: "held-out source FOVs" and "generated target FOVs".

5. Method input and scoring
Create two clean visual branches:
Top branch: multi-FOV observation stack -> blue box "STRIDE / ablations".
Bottom branch: endpoint projection bars -> orange box "transport baselines".
Merge both branches into a final box labelled "score vs hidden benchmark truth".
Inside final box show two metric icons:
A error, represented by a small matrix difference icon.
open-mass error, represented by a diamond or mass-balance icon.

Shared visual anchor:
Run a subtle colored bead ribbon along the bottom of the panel, connecting all modules.
Label it once: "same 10 shared communities".
Show beads C0, C1, ..., C9 visually, but do not write every community name.

Important scientific constraints:
Do not imply real biological causality, temporal disease progression, or a real TC-to-IM clinical transition.
Do not show patient/tissue atlas content; that belongs to the upper panel.
Do not show performance rankings, winners, p-values, statistical significance, or result plots.
Do not overstate hidden benchmark truth as biological truth.

Text policy:
Use very little text. Prefer icons over words.
Use crisp sans-serif labels, publication-quality legibility.
Avoid paragraphs.
Keep only essential labels:
"semi-synthetic benchmark"
"24 train"
"8 held-out"
"10 reruns"
"TC-IM templates"
"A_p"
"d_p"
"e_p"
"y = xA + e"
"held-out source FOVs"
"generated target FOVs"
"STRIDE / ablations"
"transport baselines"
"score vs hidden benchmark truth"
"A error"
"open-mass error"
"same 10 shared communities"

Rendering quality:
High-resolution, vector-like biomedical infographic. Balanced spacing, soft pastel scientific palette with red/teal/green/orange community accents, subtle shadows, no watermark, no decorative clutter. The panel should look suitable for a Nature Methods main figure.
```
