# Project Status: MammoRisk+
_Last updated: 2026-11-11_

App release: end of January 2027 · Papers + thesis: Feb–May 2027

| Module / Objective | Owner | Planned | Status | Latest result | Next step |
|---|---|---|---|---|---|
| Data access | Faiqa | Oct | Not started. Storage plan: raw data on USB drive, training on Kaggle/Colab (D-17) | — | PhysioNet (VinDr), EMBED registration; extract 313 PRS positions with bcftools (needs Mac/WSL) |
| M2 · PRS (O1, core) | Pukhraj | Oct–Nov | **Ready to run on Colab.** GRCh38 + gnomAD HGDP+1KG callset (D-19); samples step run on real metadata: 394 people (CEU 121, Pakistani pooled 273) | — (no PRS scores yet) | Run `notebooks/m2_prs_colab.ipynb`; record coverage + % reclassified here; Pukhraj reviews D-14/D-19 |
| M1 · Mirai on EMBED (O4) | Faiqa | Nov–Dec | **Code done, no data yet.** Cohort builder, OncoServe client, evaluation (6 tests pass) | — | Get EMBED tables; verify `COLUMNS` + OncoServe API (D-15); install Docker |
| M3 · Classifier (O2) | Faiqa, Manaal | Nov–Jan | **Split done; cache ready to build.** `splits/m3_cbis_split.csv` (3,032 images / 1,566 patients; D-20). Streaming fetcher tested on 3 real images | — | Run `notebooks/m3_cbis_cache_kaggle.ipynb` → private Kaggle dataset → `notebooks/m3_train_kaggle.ipynb` (written 2026-11-11, not run) once per backbone |
| M3 · Explanations (O3) | Faiqa, Manaal | Nov–Jan | **Code written, not run.** Grad-CAM/Score-CAM + scores (`evaluation/xai.py` tested) | — | Run after first trained checkpoint; VinDr after PhysioNet access |
| Platform (O5) | Manaal | Dec–Jan | **Skeleton done.** Flask app, enforced validation labels, login/CSRF, PDF, Docker (19 tests pass; Docker untested) | — | Manaal reviews D-16; wire adapters once models exist |
| Evaluation (shared) | Faiqa | — | **Done.** DeLong, bootstrap, IPCW C-index, td-AUC, calibration, XAI scores (14 tests pass) | — | — |

## Done (as of synopsis defence, 5 Oct 2026)
- Literature review revised (45 references verified)
- Datasets identified; licences checked
- PRS-313 weights located (PGS000004)
- Pakistani genomes located: 1000G PJL (96) + 8 HGDP groups
- Mirai availability confirmed (OncoServe container)
- EMBED open 20% release located on AWS

## Repo (2026-11-06)
- Code for all modules written while Pukhraj/Manaal were away; their decisions marked PROPOSED
  (D-14 to D-18). Local tests: `.venv/Scripts/python -m pytest` → 63 passed, 1 skipped (torch).
- Restructured to the CLAUDE.md layout. `src/fusion` removed. Environments split into imaging (Windows +
  CUDA) and genomics (macOS). PDFs are kept locally only. See D-13.

## Open questions
See "Open items to confirm" in DECISIONS.md.

## Results log
Every number: module · metric · value (95% CI) · dataset/split · config · run · date

| Date | Module | Metric | Value (95% CI) | Dataset/split | Config / run |
|---|---|---|---|---|---|
