# Task A Descriptive Atlas Review Memo

Status: reviewed against latest full-cohort atlas bundle

Purpose:
- Record whether the latest descriptive atlas output supports the current Task A
  biological reading before any new Block 1/2 coding begins.

Reviewed input bundle:
- Manifest:
  `/tmp/task_a_descriptive_atlas_full_20260402_review/task_a_descriptive_atlas_manifest.json`
- Output index:
  `/tmp/task_a_descriptive_atlas_full_20260402_review/task_a_descriptive_atlas_output_index.csv`
- Upstream prepare manifest:
  `/tmp/task_a_prepare_full_20260402_review/task_a_prepare_manifest.json`

Bundle-level checks:
- Full-cohort atlas bundle was missing in-repo at review start, so a fresh
  `prepare -> descriptive_atlas` run was generated on 2026-04-02.
- The written atlas manifest reports `artifact_state=contract_passed`,
  `run_scope=full_cohort_alignment_check`, `n_patients=32`, and
  `n_observed_communities=25`.
- The output index contains all expected atlas-side families from
  `artifact_contracts.md`: 5 tables, 4 top-level figures, and 8 representative
  overlay figures.
- Overlay coverage is concentrated on the 8 most cell-abundant communities
  (`14, 13, 0, 2, 4, 23, 8, 3`), which is sufficient for the main descriptive
  context layer.

## Community and overlay risk register

| community_id | concise biological reading | domain / occurrence note | overlay | verdict |
|---|---|---|---|---|
| 0 | `TC_CAIX`-high tumor-like community | TC-enriched; 32/32 patients | yes | `acceptable_for_coding` |
| 1 | `TC_Ki67`-high proliferative tumor-like community | TC/IM only; 27/32 patients | no | `acceptable_for_coding` |
| 2 | `Macro_CD163` / monocyte-rich myeloid community | PT-dominant; 32/32 patients | yes | `acceptable_for_coding` |
| 3 | `TC_EpCAM` / `TC_VEGF` tumor-like community | TC-dominant; 32/32 patients | yes | `acceptable_for_coding` |
| 4 | collagen-plus-myeloid interface community | IM>PT with mixed lineage signal; 32/32 patients | yes | `weaken_wording` |
| 5 | `Macro_HLADR`-led immune/myeloid mix | IM-dominant; 32/32 patients | no | `need_more_figures` |
| 6 | epithelial proliferative tumor-like community | TC/IM only; 32/32 patients | no | `acceptable_for_coding` |
| 7 | `SC_aSMA` stromal / immune admixture | IM>PT>TC; 32/32 patients | no | `weaken_wording` |
| 8 | lymphoid-rich interface community | IM-dominant; 32/32 patients | yes | `acceptable_for_coding` |
| 9 | `Macro_CD163`-rich myeloid community | IM/PT split; 32/32 patients | no | `acceptable_for_coding` |
| 10 | `TC_Ki67` / `TC_VEGF` tumor-like community | TC/IM only; 31/32 patients | no | `acceptable_for_coding` |
| 11 | `TC_VEGF`-high tumor-like community | TC-dominant; 31/32 patients | no | `acceptable_for_coding` |
| 12 | `TC_Ki67`-high tumor-like community | TC/IM only; 31/32 patients | no | `acceptable_for_coding` |
| 13 | monocytic PT-skewed community | PT>IM; 32/32 patients | yes | `acceptable_for_coding` |
| 14 | monocyte / DC-like PT-skewed community | PT>IM; 32/32 patients | yes | `acceptable_for_coding` |
| 15 | `Macro_CD11b` myeloid-tumor interface | IM>TC with mixed signal; 32/32 patients | no | `weaken_wording` |
| 16 | `TC_EpCAM`-high tumor-like community | TC-dominant; 32/32 patients | no | `acceptable_for_coding` |
| 17 | proliferative tumor-like mix | TC/IM only; 32/32 patients | no | `acceptable_for_coding` |
| 18 | `SC_aSMA` stromal community | TC>IM; 32/32 patients | no | `acceptable_for_coding` |
| 19 | `Macro_HLADR` myeloid community | IM-dominant; 30/32 patients | no | `acceptable_for_coding` |
| 20 | NK / myeloid mixed PT community | PT-dominant; 32/32 patients | no | `weaken_wording` |
| 21 | NK-rich PT community | PT-dominant; 30/32 patients | no | `acceptable_for_coding` |
| 22 | Treg-plus-myeloid PT community | PT-dominant; 32/32 patients | no | `need_more_figures` |
| 23 | `UNKNOWN` / monocytic PT community | PT-dominant; 32/32 patients | yes | `weaken_wording` |
| 24 | collagen-rich stromal community | IM>TC; 32/32 patients | no | `acceptable_for_coding` |

Atlas-level interpretation:
- Dominant-community naming is adequate at the lineage level for most major
  communities. The strongest naming surface is the tumor-like set
  (`0, 1, 3, 6, 10, 11, 12, 16, 17`), plus coherent myeloid / stromal groups
  such as `2, 9, 13, 14, 18, 19, 21, 24`.
- The main wording-sensitive communities are `4, 5, 7, 15, 20, 22, 23`, where
  the top subtype fraction is either modest, mixed across lineages, or led by
  `UNKNOWN`.
- `TC / IM / PT` distribution broadly matches the current biological intuition:
  tumor-like communities are mostly TC-enriched, while immune / myeloid /
  interface communities shift toward IM and PT.
- Patient occurrence is strong enough for the next coding round. No community is
  rarer than 27/32 patients, so the current atlas does not reveal an obvious
  recurrence blocker for paired confirmatory or later robustness analyses.
- Overlay coverage is sufficient for the coding gate because it covers the
  highest-burden communities, but morphology-heavy prose should not lean on
  communities `5` or `22` until additional overlays exist.

## Required conclusion block

- Atlas bundle reviewed:
  `/tmp/task_a_descriptive_atlas_full_20260402_review/task_a_descriptive_atlas_manifest.json`
- Reviewer:
  `Codex`
- Review date:
  `2026-04-02`
- Dominant-community naming verdict:
  majority nameable at lineage level; use broad labels only for
  `4, 5, 7, 15, 20, 22, 23`
- Domain-distribution verdict:
  descriptive TC-versus-IM/PT structure is consistent with the current Task A
  biological framing
- Patient-occurrence verdict:
  sufficient for the next Block 1/2 coding round; no obvious rarity blocker
- Overlay sufficiency verdict:
  sufficient for coding because the 8 most cell-abundant communities are
  covered, but communities `5` and `22` should not receive strong morphology
  prose without more figures
- Wording changes required before coding:
  keep community labels lineage-level rather than subtype-precise for
  `4, 5, 7, 15, 20, 22, 23`; do not hard-name community `23` while `UNKNOWN`
  remains the top subtype; avoid morphology-heavy claims for communities `5`
  and `22`
- Decision:
  `ready_for_block1_block2_coding`
