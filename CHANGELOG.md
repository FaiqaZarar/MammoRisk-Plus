# Changelog

Newest first. One entry per meaningful change: date · module · what changed · files touched.

## 2026-11-12
- **Docs:** `docs/progress_update_2026-11-12.md`, a one-page update for the supervisor (status, blockers,
  requests, findings from real data, plan to the January release).
- **Platform:** `app/demo.py`: `python -m app.demo` starts the app locally with a throwaway random login
  (printed once, never stored).

## 2026-11-11
- **Data:** VinDr-Mammo DUA signed by Faiqa, files visible. `data/README.md` corrected: access is DUA only,
  not credentialed.
- **M3:** Added `module3_classifier/fetch_vindr.py` and `tests/test_m3_fetch_vindr.py` (7 tests). It selects
  the VinDr images for explanation scoring (788 images / 405 exams, ~13 GB; `--all-views` 1,620; `--sample N
  --seed S` gives a balanced seeded sample), writes the wget URL list, and verifies SHA256. wget (WSL) does
  the download because PhysioNet returned 403 to curl/Python (D-21). 3-image trial: login OK, hash check OK
  on the first real image. 200-image sample list generated (100 mass + 100 calc).
- **Data:** VinDr CSVs checked on the real files (columns, categories, subset size). Subset to download:
  788 images / 405 exams (~13 GB). Details in DECISIONS (D-18 open item).
- **Docs:** DECISIONS open item D-18 updated with the VinDr column check and access rule (DUA only).
- **M3:** Added `notebooks/m3_train_kaggle.ipynb`: Kaggle GPU notebook that trains one backbone per run
  (`MODEL = efficientnet_b3 | resnet50`) from the private `mammorisk-cbis-png-1024x512` dataset, checks the
  cache against the committed split (counts, no patient leakage), runs `module3_classifier.train`, then
  `evaluate run` once (threshold on val) and the paired DeLong `compare`. Only the paths and worker count are
  overridden; the merged YAML is saved with the results. Not run yet.

## 2026-11-09
- **M3:** Created the patient-level split `splits/m3_cbis_split.csv` from the official TCIA CSVs (D-20).
- **M3:** Added `module3_classifier/fetch_cbis.py` (streams CBIS-DDSM from the TCIA API into the PNG cache,
  deleting DICOMs at once; resumable; robust mask lookup) and `tests/test_m3_fetch.py`.
- **M3:** Added `notebooks/m3_cbis_cache_kaggle.ipynb` to build the cache on Kaggle.
- **Docs:** D-20, M3 row in STATUS, CLAUDE.md command, data/README storage note.

## 2026-11-07
- **M2:** Switched to GRCh38: added `data/prs/PGS000004_hmPOS_GRCh38.txt.gz` (harmonised, 313/313 mapped) and
  genotypes from the gnomAD v3.1.2 HGDP+1KG callset (D-19). New pipeline steps `samples`, `extract`,
  `dosage`; driver SNPs and an ancestry-outlier sensitivity analysis in `analyse`; `--data-root` /
  `--results-dir` overrides. Files: `module2_prs/{data_prep,run,ancestry}.py`, `configs/m2_prs.yaml`,
  `tests/test_m2_prs.py`.
- **M2:** Added `notebooks/m2_prs_colab.ipynb` (bcftools + plink2 Beta 1.3 on Colab, end-to-end run).
- **Docs:** D-19, deviation row 12, D-13 open item closed, M2 row in STATUS, data/README.

## 2026-11-06 (code build-out)
- **Evaluation:** `evaluation/` with DeLong AUC CI + paired test, Youden threshold, threshold metrics,
  stratified/clustered bootstrap, IPCW C-index, td-AUC, KM calibration, subgroup table, XAI scores, seeds.
  Tests: `tests/test_evaluation.py`. Also `pyproject.toml` (pytest config).
- **M2:** plink2 PRS-313 pipeline `module2_prs/{weights,harmonise,score,standardise,reclassify,drivers,
  ancestry,data_prep,run}.py`, `configs/m2_prs.yaml`, `tests/test_m2_prs.py`. Writes
  `reference_stats.json` for the app. Proposed methods in D-14.
- **M1:** `module1_mirai/{embed_cohort,oncoserve,evaluate}.py`, `gradcam_notes.md`,
  `configs/m1_mirai_embed.yaml`, `tests/test_m1_cohort.py` (D-15).
- **M3:** `module3_classifier/{config,cbis,splits,preprocess,datasets,models,train,evaluate,explain}.py`,
  `configs/m3_{effnetb3,resnet50}.yaml`, `tests/test_m3_data.py` (D-18).
- **Platform:** Flask app in `app/`, `Dockerfile`, `docker-compose.yml`, `.dockerignore`,
  `configs/app.yaml`, `tests/test_app_routes.py` (D-16).
- **Docs:** DECISIONS D-14 to D-18, STATUS rows, CLAUDE.md commands.

## 2026-11-06
- **Docs:** README.md rewritten to match CLAUDE.md and D-07 to D-11 (drops RSNA training, the C-index
  target, suspicious vs normal / AUC > 0.85, PRSice-2 and "from scratch"). Added D-13 and new open items to
  DECISIONS.md. Updated the M2 row in STATUS.md. CLAUDE.md commands now point to the two env files.
  Files: `README.md`, `docs/DECISIONS.md`, `docs/STATUS.md`, `CLAUDE.md`.
- **Repo / env:** `requirements.txt` replaced by `environment-imaging.yml` and `environment-genomics.yml`.
  `data/README.md` now lists the datasets, flags PGS000004 as GRCh37, and explains plink2 on Windows.
- **M2:** `genomics/scripts/` and `genomics/docs/PRS_Pipeline.md` moved unchanged to
  `module2_prs/legacy_prsice/`, with a README asking Pukhraj to review. Removed the duplicate unpacked
  `data/prs/PGS000004.txt/`. The `.gz` file is kept.
- **Repo:** Restructured to the CLAUDE.md layout (`module1_mirai/`, `module2_prs/`, `module3_classifier/`,
  `evaluation/`, `app/`, `splits/`). Deleted `src/fusion/` (D-02, D-13).
- **Repo:** `.gitignore` also ignores `docs/*.pdf`.
- **Repo:** Rewrote `.gitignore` per CLAUDE.md hard constraints. It now ignores all of `data/` (except
  README, `.gitkeep` and the public PGS000004 weights file), DICOM/NIfTI/raw images, VCF/BCF/PLINK
  genotype files, model weights/checkpoints and `models/`, `results/`, W&B/logs, secrets (`.env`, keys,
  `kaggle.json`), downloaded tools, Python/R caches and Flask uploads. Every rule from the old file is kept.
  Files: `.gitignore`.
- **Docs:** Added CLAUDE.md, docs/DECISIONS.md, docs/STATUS.md and CHANGELOG.md, aligned with the revised
  literature review and the supervisor-revised synopsis defence deck. Recorded decisions D-01 to D-12 and all
  deviations from the July review.
