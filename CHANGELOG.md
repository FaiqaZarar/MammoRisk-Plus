# Changelog

Newest first. One entry per meaningful change: date · module · what changed · files touched.

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
