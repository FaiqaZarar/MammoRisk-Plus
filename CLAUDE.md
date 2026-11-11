# MammoRisk+ — Project Instructions for Claude Code

## Read these first (every new session)
1. `docs/STATUS.md`: where each module stands right now and what's next
2. `docs/DECISIONS.md`: every design/methodology decision and why it was made, including what changed since the
   first literature review
3. `docs/literature_review_revised.pdf`: the revised literature review (synopsis version). **This is the
   source of truth for scope, objectives, datasets and methods.**
4. `docs/synopsis_defence_deck.pdf`: the supervisor-revised deck presented at the synopsis defence (5 Oct 2026)
5. `CHANGELOG.md`: what changed, by date

`docs/literature_review_v1_july.pdf` is the first submission. It is **outdated**. Never use it for methods.
If the code, these docs and the revised review disagree, stop and ask. Don't guess.

## Documentation rules (mandatory)
After any meaningful change, **before reporting the task as done**:
- **CHANGELOG.md**: add a dated entry under the right module: what changed and which files.
- **docs/DECISIONS.md**: if a choice was made (model, dataset, split, metric, threshold, library, scope),
  add a numbered entry: date, decision, alternatives, reason, who decided. Never delete entries. If one is
  reversed, mark it `SUPERSEDED by D-xx`.
- **docs/STATUS.md**: update the module's status, latest results (with config/W&B run), next steps.
- **Results**: every number must say which dataset, split, checkpoint and config produced it.
- Anything that departs from the revised literature review goes in DECISIONS.md under "Deviations".
  This feeds the thesis methodology chapter and the final defence.

## What this project is
**Research question:** Breast cancer risk tools built on Western and European women are used without
evidence that they classify **Pakistani / South Asian** women correctly. MammoRisk+ measures how far they
can be trusted and delivers them through a platform that **states its own reliability**: each result is
labelled with the populations it was validated on.

Title: *MammoRisk+: Development and Validation of a Clinical Platform for Breast Cancer Risk Assessment and
AI-Based Detection Using Deep Learning Mammography Analysis and Polygenic Risk Scoring*

**Three independent modules.** They are not fused because no public dataset has mammograms and genomes from
the same women. Fusion is future work, so don't build it.

| Module | What | Objective |
|---|---|---|
| **Module 1** | Pre-trained Mirai: 1–5 yr risk from 4-view mammograms, external validation on EMBED | O4 |
| **Module 2** | PRS-313: how risk category changes with the reference population across 9 Pakistani groups | **O1 (core contribution)** |
| **Module 3** | Benign vs malignant classifier + explanation reliability across datasets | O2 + O3 |
| Platform | Flask + Docker web app, imaging + genetic risk with validation labels, PDF report | O5 |

### Objectives (as presented at synopsis defence)
- **O1 (core):** Quantify how PRS-313 risk categories change when standardised against a European (CEU) vs
  Pakistani reference, across 9 Pakistani groups (1000G PJL + 8 HGDP groups). Output: % reclassified with
  bootstrap CIs, plus the SNPs driving the shift.
- **O2:** Train a benign vs malignant classifier on CBIS-DDSM, then test transfer to East and South Asian
  data (CMMD, DMID).
- **O3:** Test whether explanations localise lesions equally well across CBIS-DDSM, VinDr-Mammo and DMID,
  using the Pointing Game, energy-in-box, IoU and a sanity check.
- **O4:** Validate pre-trained Mirai on EMBED, broken down by breast density and race (Asian subgroup).
  Output: C-index + 1–5 yr time-dependent AUC with 95% CI, plus calibration.
- **O5:** An open platform that shows imaging and genetic risk with validation labels, plus papers.

⚠️ The revised review and the presented deck number O2–O4 slightly differently. See DECISIONS D-11.

## Team and ownership
- **Faiqa Zarar Noor:** team lead. Module 1 (Mirai), Module 3 (classifier + XAI), coordination
- **Pukhraj Tahir:** Module 2 (PRS, O1)
- **Manaal Tufail:** evaluation, platform (Flask + Docker), literature/references; supports Module 3
- Supervisor: Dr. Mehwish Noureen. Co-supervisor: Dr. Mian Ilyas Ahmad (AI focus; wants a working product)

Repo: github.com/FaiqaZarar/MammoRisk-Plus (public). **Never commit data, genomes, patient files, model
weights, API keys, or VinDr-Mammo/EMBED images** (VinDr is under a PhysioNet licence: report metrics, never
show its images).

## Timeline (revised, Table 7)
| Period | Work | Owner |
|---|---|---|
| Oct 2026 | Data access: PhysioNet credentialing, EMBED registration, downloads | Faiqa |
| Oct–Nov 2026 | Module 2: genomic risk (O1) | Pukhraj |
| Nov–Dec 2026 | Module 1: Mirai on EMBED (O4) | Faiqa |
| Nov 2026–Jan 2027 | Module 3: classifier + explanations (O2, O3) | Faiqa, Manaal |
| Dec 2026–Jan 2027 | Platform built in parallel; **app released end of January 2027** | Manaal |
| Feb–May 2027 | Papers (O1 first, then O3), thesis, final defence | All |
Dec–Jan includes exams, so treat it as buffer.

## Hard constraints
- **Compute:** SINES gives only ~20 GB. CBIS-DDSM alone is ~164 GB as DICOM. Train on PC GPU / Colab / Kaggle,
  run genomics + platform on the laptop. Cache downsampled PNGs and delete raw DICOMs once cached. Use
  mixed precision (AMP) and small batches. Warn before any large download or write. Paths come from config,
  never hard-coded.
- **Public data only.** Pakistani hospital mammograms are future work and not a dependency.
- **Reproducibility:** fixed seeds (Python/NumPy/PyTorch/CUDA), one YAML config per experiment, W&B
  logging, split CSVs saved and reused, Docker image for the platform.
- **No leakage:** patient-level splits only. Augmentation on training data only. Thresholds are chosen on
  validation and applied once to test.

## Datasets
| Dataset | Population | Use | Key limitation |
|---|---|---|---|
| CBIS-DDSM | USA, scanned film | Train/val/test classifier (M3) | No normal images, so benign vs malignant only |
| CMMD | China | External test of classifier (M3, O2) | Different machines/population |
| DMID | India (510 images) | External test + explanations (M3) | Labels from reports, not biopsy. Exclude normals from the 2-class AUC |
| VinDr-Mammo | Vietnam (5,000 exams) | Explanations only (lesion boxes) | No biopsy labels, so not used for AUC. PhysioNet licence |
| EMBED (open 20% subset) | USA, diverse | Mirai validation (M1) | Few future cancers or Asian patients → report as exploratory |
| 1000 Genomes | PJL (96) + CEU reference | PRS (M2) | No cancer outcomes |
| HGDP | 8 Pakistani groups (~20–25 each) | PRS (M2) | Small groups, so per-group results are descriptive only |
RSNA 2022 is **not** used (it has no follow-up times, so no C-index).

## Module specs

### Module 1: Mirai (O4)
- **No retraining.** This is external validation with pre-trained weights.
- Input: four 2D FFDM views per exam (L-CC, L-MLO, R-CC, R-MLO). Exclude 3D/synthetic views from EMBED.
  Mirai's aggregator combines **the four views of one exam**. It does NOT pool across prior exams.
- Outcomes: link each screening exam to later pathology-confirmed cancer. Women without cancer are censored
  at their last follow-up.
- Metrics: C-index (IPCW) + 1–5 yr time-dependent AUC with 95% CI, calibration, all broken down by density
  and race. Also check the machine vendor: Mirai was trained on Hologic, so stratify by vendor.
- Deployment: OncoServe Docker (`learn2cure/oncoserve_mirai:0.5.0`) for inference only. **Grad-CAM needs the
  Mirai repo + weights loaded directly in PyTorch** (OncoServe is a black box), applied to the image encoder.
- Not feasible / dropped: the Gail / Tyrer-Cuzick comparison on our data (we lack the questionnaire inputs).

### Module 2: PRS-313 (O1, the core contribution)
- Weights: PGS Catalog **PGS000004**, GRCh38 harmonised file.
- Tools: **PLINK 2.0 (`--score`) + bcftools**. We apply published weights, so PRSice-2 isn't needed.
  pgsc_calc is optional for allele matching. Confirm which one is used and log it in DECISIONS.
- Handle strand flips and ambiguous A/T and C/G SNPs. Always report coverage (n of 313 found). Missing SNPs
  are filled with the reference allele frequency.
- Standardise: z = (score − ref mean) / ref SD under **European (CEU) vs Pakistani** references.
  Use this wording everywhere. The "Pakistani reference" = PJL + 8 HGDP groups pooled (~280).
  Per-group results are a sensitivity analysis.
- Ancestry check: PCA against the reference panel.
- Output: per-group distributions, % changing risk category, bootstrap 95% CI (1,000 resamples, stratified
  by group), driver SNPs.
- Top-1% cut-offs come from the reference's fitted normal distribution, not from counting people, and are
  reported only for the pooled sample.
- **We do NOT claim** which label is clinically correct (no outcomes) or absolute risk.

### Module 3: Classifier + explanations (O2, O3)
- Task: **benign vs malignant** (CBIS-DDSM has no normals). Use **full mammograms**, not ROI crops, so
  heatmaps can be scored against masks. Resize to ~1024×512. An image is malignant if any lesion on it is
  malignant.
- Split: 70/10/20 **patient-level** (merge the mass + calc subsets by patient first). Save to `splits/`.
- Backbones: ResNet-50 baseline vs EfficientNet-B3 (ConvNeXt-Tiny was on the defence deck; see D-11). Use
  ImageNet pretraining, fine-tune, compare with the DeLong test on the same test set.
- Training: weighted cross-entropy, augmentation (flips, rotation ≤15°, brightness/contrast) on train only,
  weight decay, ≤50 epochs, early stopping on val AUC (patience 10).
- Metrics: AUC with bootstrap/DeLong 95% CI, sensitivity, specificity, F1, PPV. Target **AUC ≥ 0.80**.
  Threshold set on validation (Youden), applied once to test.
- External: AUC on CMMD and DMID. Report the drop. Note that machine type and population are confounded.
- Explanations: Grad-CAM + Score-CAM on the last conv block, upsampled. Scored against masks/boxes on
  CBIS-DDSM, VinDr-Mammo and DMID:
  - **Pointing Game:** heatmap max inside the lesion
  - **energy-in-box**
  - **IoU:** heatmap thresholded at the top 50%; report the threshold
  - **Sanity check:** randomise weights layer by layer; the heatmap must change
  Score malignant cases, reported both for all cases and for correctly classified ones.

### Platform (O5)
Flask + Docker. Upload a mammogram → M3 probability + heatmap (and M1 risk when 4 views are given).
Upload a VCF → M2 score under both references. **Every output carries a validation label** (datasets,
populations, metric) printed in the PDF report. Diagnostic features sit behind a hospital login. The public
Urdu awareness app is future work.

## Tech stack
Python 3.11 · PyTorch 2.x · timm · pytorch-grad-cam / captum · OpenCV · pydicom · PLINK 2.0 · bcftools ·
pandas/numpy · scikit-learn · lifelines / scikit-survival (C-index, IPCW, time-dependent AUC) · DeLong ·
Flask · Docker · Weights & Biases.

## Suggested repo layout
```
MammoRisk-Plus/
├── CLAUDE.md  CHANGELOG.md  README.md  .gitignore
├── environment-imaging.yml  environment-genomics.yml
├── docs/          # STATUS, DECISIONS, literature review PDFs, defence deck
├── configs/       # one YAML per experiment
├── data/          # gitignored; data/README.md = how to get each dataset
├── splits/        # patient-level split CSVs (committed, no images)
├── module1_mirai/
├── module2_prs/
├── module3_classifier/   # dataset, models, train, evaluate, explain
├── evaluation/    # shared metrics: AUC+CI, DeLong, C-index/IPCW, td-AUC, calibration, bootstrap, XAI scores
├── app/           # Flask app, templates, PDF report, Dockerfile
├── notebooks/     # exploration only
├── tests/
└── results/       # gitignored
```

## How to work with me (Faiqa)
- Short directive instructions → act directly, no long preambles.
- Explain concepts theory/definitions first, then examples.
- Show full step-by-step working for calculations and statistics.
- If I correct you, adopt it. If you think the correction is wrong, say so once, briefly, with the reason.
- Before big changes (new deps, restructuring, deleting files, long training runs): give a short plan and wait.
- Small focused commits with clear messages. Don't push unless asked.
- Comment any non-obvious ML/stats choice. AI-domain committee members will read this code.

## Commands (fill in as they work)
- Env (imaging, Windows + CUDA): `conda env create -f environment-imaging.yml && conda activate mammorisk-imaging`
- Env (genomics, macOS): `conda env create -f environment-genomics.yml && conda activate mammorisk-genomics`
  (Windows: standalone plink2 binary, see data/README.md)
- Tests: `pytest tests/` (on Faiqa's laptop: `.venv/Scripts/python -m pytest`, see D-17)
- PRS: `python -m module2_prs.run --config configs/m2_prs.yaml --step all|regions|harmonise|score|analyse`
- M1 cohort: `python -m module1_mirai.embed_cohort --config configs/m1_mirai_embed.yaml`
- M1 evaluate: `python -m module1_mirai.evaluate --config configs/m1_mirai_embed.yaml`
- M3 split: `python -m module3_classifier.splits --config configs/m3_effnetb3.yaml`
- M3 cache (stream from TCIA): `python -m module3_classifier.fetch_cbis --config configs/m3_effnetb3.yaml --workers 8` (Kaggle: `notebooks/m3_cbis_cache_kaggle.ipynb`)
- M3 VinDr subset (explanations only, local drive only): `python -m module3_classifier.fetch_vindr --root D:/mammorisk_data/vindr_mammo --user <physionet-user> [--all-views] [--dry-run]`
- M3 VinDr cache (local only, after the download): `python -m module3_classifier.cache_vindr --root D:/mammorisk_data/vindr_mammo --sample 200 --seed 42`
- Train M3: `python -m module3_classifier.train --config configs/m3_effnetb3.yaml`
- Evaluate M3: `python -m module3_classifier.evaluate run|compare ...` (see `--help`)
- Explain M3: `python -m module3_classifier.explain --config ... --dataset cbis|vindr|dmid`
- App: `docker compose up` or `flask --app app run --debug`
