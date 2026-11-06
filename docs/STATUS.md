# Project Status: MammoRisk+
_Last updated: 2026-11-06_

App release: end of January 2027 · Papers + thesis: Feb–May 2027

| Module / Objective | Owner | Planned | Status | Latest result | Next step |
|---|---|---|---|---|---|
| Data access | Faiqa | Oct | _fill in_ | — | PhysioNet (VinDr), EMBED registration; downloads |
| M2 · PRS (O1, core) | Pukhraj | Oct–Nov | chr1 PRSice-2 pilot only (now frozen in `module2_prs/legacy_prsice/`); plink2 pipeline not started | — (pilot scores not a result) | Pukhraj reviews legacy pilot; decide GRCh37 vs GRCh38 harmonised weights; build plink2 `--score` pipeline over all 313 SNPs |
| M1 · Mirai on EMBED (O4) | Faiqa | Nov–Dec | _fill in_ | — | — |
| M3 · Classifier (O2) | Faiqa, Manaal | Nov–Jan | _fill in_ | — | — |
| M3 · Explanations (O3) | Faiqa, Manaal | Nov–Jan | _fill in_ | — | — |
| Platform (O5) | Manaal | Dec–Jan | _fill in_ | — | — |

## Done (as of synopsis defence, 5 Oct 2026)
- Literature review revised (45 references verified)
- Datasets identified; licences checked
- PRS-313 weights located (PGS000004)
- Pakistani genomes located: 1000G PJL (96) + 8 HGDP groups
- Mirai availability confirmed (OncoServe container)
- EMBED open 20% release located on AWS

## Repo (2026-11-06)
- Restructured to the CLAUDE.md layout. `src/fusion` removed. Environments split into imaging (Windows +
  CUDA) and genomics (macOS). PDFs are kept locally only. See D-13.

## Open questions
See "Open items to confirm" in DECISIONS.md.

## Results log
Every number: module · metric · value (95% CI) · dataset/split · config · run · date

| Date | Module | Metric | Value (95% CI) | Dataset/split | Config / run |
|---|---|---|---|---|---|
