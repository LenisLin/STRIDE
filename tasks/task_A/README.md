# Task A Operations

This README is the task-local operational companion for Task A.

Use `docs/task_A_spec.md` for the live scientific definition and claim
boundary. This README covers entry points, outputs, and compatibility notes
only. It does not serve as a Task A results narrative.

## 1. Purpose and Document Split
- `docs/task_A_spec.md` is the sole live scientific Task A document.
- `tasks/task_A/README.md` is the local operational guide.
- No standalone live Task A results document is maintained.
- Current task outputs are task-layer operational artifacts, not canonical
  patient-level exports.
- The ordered `TC` / `IM` / `PT` tissue domains are observation-layer strata
  on top of a shared state basis; they are not part of canonical state
  identity.

## 2. Runtime Entry Points
- Shared formal run entrypoint: `tasks/task_A/pipeline.py`.
- Shared evaluator: `tasks/task_A/evaluator.py`.
- Runtime contract and artifact layout: `tasks/task_A/runtime_contract.py`.
- Default shared config surface: `tasks/task_A/config.yaml`.

## 3. Analysis and Audit Utilities
- Task A operational reporting follows the live sequence
  `proxy validity -> semi-synthetic gain -> real-data mirror -> bounded audit`.
- `tasks/task_A/analyze_arm2_results.py` refreshes or rebuilds the
  compatibility-labelled focused real-data mirror package from an existing
  Task A run.
- `tasks/task_A/analyze_arm2_focused.py` produces the current
  compatibility-labelled focused real-data mirror package on the confirmatory
  ordered tissue-domain surface.
- `tasks/task_A/analyze_arm2_bioinformed.py` produces the current bounded
  residual audit package for the same compatibility-labelled real-data surface.
- `tasks/task_A/extract_arm2_arm3_neutral.py` prepares neutral cross-surface
  summaries used by current review workflows.
- `tasks/task_A/arm3_uq_stress.py` runs the reduced-coverage continuation
  surface used in current task-local robustness workflows.

## 4. Output Surfaces
- A formal Task A run writes `task_A_metrics.parquet` and
  `task_a_run_manifest.json` at the run root.
- Multi-run layouts may also create per-runner artifact and analysis
  subdirectories beneath that root.
- The compatibility-labelled focused real-data mirror package is written
  beneath the configured `analysis/focused/` root, but its public outputs now
  use Block-1/Block-2 naming:
  `00_task_a_real_data_mirror_memo.md`,
  `05_patient_continuity_backbone_summary.csv`,
  `06_trusted_continuity_anchors.csv`,
  `07_closed_comparator_forced_closure.csv`,
  `08_bounded_residual_contributors.csv`,
  `09_anchor_residual_overlap_audit.csv`,
  `10_confirmatory_family_backbone_summary.csv`,
  `11_trusted_anchor_patient_recurrence.csv`,
  `12_auxiliary_legacy_comparator_view.csv`,
  `13_auxiliary_legacy_anchor_view.csv`,
  and `14_output_contract_audit.csv`.
- The bounded audit package is written beneath the configured
  `analysis/bioinformed/` root, but its public outputs now use Block-2 naming:
  `20_tc_dominant_backbone_context.csv`,
  `21_interface_residual_context.csv`,
  `22_bio_annotated_anchor_residual_overlap_audit.csv`,
  `23_closed_vs_open_prototype_contrast.csv`,
  `24_directional_residual_assignment_audit.csv`,
  and `25_block2_biointegrated_audit_table.csv`.
- Reduced-coverage continuation outputs are written beneath the configured
  stress-test result root.

## 5. Compatibility Notes
- Current code, configs, persisted paths, and some utilities still use legacy
  `A1` / `A2` / `A3` or arm-number naming for compatibility.
- Several filenames such as `analyze_arm2_*` and `arm3_uq_stress.py` are kept
  as operational compatibility residue rather than live scientific labels.
- Additional override configs such as `tasks/task_A/config_arm1.yaml` and
  `tasks/task_A/config_arm3.yaml` survive for runner compatibility only.
- Those runtime names are compatibility-only operational labels; they do not
  define the live scientific Task A framing.
- If a task-local relation matrix is discussed in benchmark or analysis notes,
  treat it as a Task A proxy-surface object unless it is explicitly documented
  as a canonical patient-level export.
- Some current task outputs still expose observation-layer compatibility fields
  and may leave `tau` / `R` unavailable on otherwise-ok rows in
  cross-compartment utilities.
- Treat those fields as compatibility or diagnostic surfaces, not as canonical
  patient-level exports.
