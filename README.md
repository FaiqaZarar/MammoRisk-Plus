# MammoRisk+

### Development and Validation of a Clinical Platform for Breast Cancer Risk Assessment and AI-Based Detection Using Deep Learning Mammography Analysis and Polygenic Risk Scoring

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-academic--research-lightgrey)

BS Bioinformatics Final Year Project · NUST SINES · 2026–2027
**Not intended for clinical or diagnostic use.**

---

## Research question

Breast cancer risk tools built on Western and European women are used **without evidence that they
classify Pakistani / South Asian women correctly**. MammoRisk+ measures how far these tools can be
trusted and delivers them through a platform that **states its own reliability**: every result is
labelled with the datasets and populations it was validated on.

## Three independent modules

The modules are **not fused**. No public dataset has mammograms and genomes from the same women, so
fusion is future work (DECISIONS D-02).

| Module | What | Objective |
|---|---|---|
| **Module 1: Mirai** | Pre-trained Mirai, 1–5 yr risk from 4-view mammograms, **external validation on EMBED** | O4 |
| **Module 2: PRS-313** | How risk category changes with the reference population across 9 Pakistani groups | **O1 (core)** |
| **Module 3: Classifier + explanations** | Benign vs malignant classifier, transfer to Asian data, explanation reliability across datasets | O2 + O3 |
| **Platform** | Flask + Docker web app: imaging + genetic risk with validation labels, PDF report | O5 |

### Objectives
- **O1 (core):** Quantify how PRS-313 risk categories change when standardised against a **European
  (CEU) vs Pakistani** reference, across 9 Pakistani groups (1000G PJL + 8 HGDP groups). Output: %
  reclassified with bootstrap 95% CIs, plus the SNPs driving the shift.
- **O2:** Train a benign vs malignant classifier on CBIS-DDSM, then test transfer to East and South
  Asian data (CMMD, DMID).
- **O3:** Test whether explanations localise lesions equally well across CBIS-DDSM, VinDr-Mammo and
  DMID (Pointing Game, energy-in-box, IoU, sanity check).
- **O4:** Validate pre-trained Mirai on EMBED, broken down by breast density, race (Asian subgroup) and
  vendor. Output: C-index + 1–5 yr time-dependent AUC with 95% CI, plus calibration.
- **O5:** An open platform that shows imaging and genetic risk with validation labels, plus papers.

> Objective numbering follows the synopsis defence deck. The revised literature review numbers O2–O4
> slightly differently. This is still open, see [DECISIONS D-11](docs/DECISIONS.md).

---

## Module details

### Module 1: Mirai (O4)
- **No retraining.** Pre-trained Mirai (Yala et al.) is validated externally on the **EMBED open 20%
  subset**. There is no train/test split and no performance target: we measure, by subgroup (D-07).
- Input: the four 2D FFDM views of one exam (L-CC, L-MLO, R-CC, R-MLO). 3D/synthetic views are
  excluded. Mirai's aggregator combines **the four views of a single exam**. It does not pool prior
  exams (D-08).
- Outcomes: each screening exam is linked to later pathology-confirmed cancer. Women without cancer are
  censored at their last follow-up.
- Metrics: IPCW C-index, 1–5 yr time-dependent AUC with 95% CI, and calibration, broken down by
  density, race and vendor (Mirai was trained on Hologic). If EMBED has few events, results are
  exploratory.
- Inference runs through OncoServe Docker (`learn2cure/oncoserve_mirai:0.5.0`). Grad-CAM loads the Mirai
  repo + weights directly in PyTorch.
- The Gail / Tyrer-Cuzick comparison is dropped, because we lack the questionnaire inputs (D-08).

### Module 2: PRS-313 (O1, core contribution)
- Weights: PGS Catalog **PGS000004** (313 SNPs, Mavaddat et al. 2019). No training.
- Tools: **PLINK 2.0 (`--score`) + bcftools** (D-10). PRSice-2 is not used: it is for deriving or
  tuning PRS, and we apply published weights.
- Strand flips and ambiguous A/T and C/G SNPs are handled. Coverage (n of 313) is always reported.
  Missing SNPs are filled with the reference allele frequency.
- Standardisation: z = (score − ref mean) / ref SD under a **European (CEU) vs Pakistani** reference.
  The Pakistani reference is PJL + 8 HGDP groups pooled (~280). Per-group results are a sensitivity
  analysis (D-09).
- Ancestry check: PCA against the reference panel.
- Output: per-group distributions, % changing risk category, stratified bootstrap 95% CI (1,000
  resamples) and driver SNPs. Top-1% cut-offs come from the reference's fitted normal distribution and
  are reported for the pooled sample only.
- **We do not claim** which label is clinically correct, and we make no absolute-risk claims: there
  are no outcomes.

### Module 3: Classifier + explanations (O2, O3)
- Task: **benign vs malignant**, because CBIS-DDSM has no normal images (D-11). The model uses **full
  mammograms** resized to ~1024×512, not ROI crops. An image is malignant if any lesion on it is.
- Split: 70/10/20 **patient-level** (mass + calc subsets merged by patient), saved in `splits/`.
- Backbones: ResNet-50 baseline vs EfficientNet-B3, **ImageNet-pretrained and fine-tuned**, compared with
  the DeLong test on the same test set.
- Training: weighted cross-entropy, augmentation on train only, early stopping on val AUC.
- Metrics: AUC with 95% CI, sensitivity, specificity, F1, PPV. **Target AUC ≥ 0.80.** The threshold is
  set on validation (Youden) and applied once to test.
- External test: AUC on CMMD and DMID, reporting the drop. Machine type and population are confounded.
- Explanations: Grad-CAM + Score-CAM, scored against lesion masks/boxes on CBIS-DDSM, VinDr-Mammo and
  DMID with the **Pointing Game, energy-in-box, IoU (top-50% threshold) and a weight-randomisation
  sanity check**.

### Platform (O5)
Flask + Docker. Upload a mammogram to get the M3 probability + heatmap (and M1 risk when 4 views are
given). Upload a VCF to get the M2 score under both references. **Every output carries a validation
label** (datasets, populations, metric) that is printed in the PDF report. Diagnostic features sit
behind a hospital login.

---

## Datasets (public only)

| Dataset | Population | Use | Key limitation |
|---|---|---|---|
| CBIS-DDSM | USA, scanned film | Train/val/test classifier (M3) | No normals, so benign vs malignant only |
| CMMD | China | External test (M3, O2) | Different machines/population |
| DMID | India | External test + explanations (M3) | Labels from reports, not biopsy |
| VinDr-Mammo | Vietnam | Explanations only (lesion boxes) | No biopsy labels. PhysioNet licence |
| EMBED (open 20%) | USA, diverse | Mirai validation (M1) | Few future cancers / Asian patients, so exploratory |
| 1000 Genomes | PJL (96) + CEU | PRS (M2) | No cancer outcomes |
| HGDP | 8 Pakistani groups | PRS (M2) | Small groups, so per-group results are descriptive |

RSNA 2022 is **not** used, because it has no follow-up times and so no C-index (D-07). See
[data/README.md](data/README.md) for access. No data, images, genomes or weights are committed.

---

## Repository structure

```
MammoRisk-Plus/
├── CLAUDE.md  CHANGELOG.md  README.md  .gitignore
├── environment-imaging.yml    # Windows + CUDA: M1, M3, evaluation, app
├── environment-genomics.yml   # macOS: M2 (plink2, bcftools)
├── docs/            # STATUS.md, DECISIONS.md (PDFs kept locally, not committed)
├── configs/         # one YAML per experiment
├── data/            # gitignored; data/README.md = how to get each dataset
├── splits/          # patient-level split CSVs
├── module1_mirai/
├── module2_prs/     # legacy_prsice/ = frozen chr1 PRSice-2 pilot, pending review
├── module3_classifier/
├── evaluation/      # AUC+CI, DeLong, C-index/IPCW, td-AUC, calibration, bootstrap, XAI scores
├── app/             # Flask app, templates, static, PDF report, Dockerfile
├── models/          # local weights only (gitignored)
├── notebooks/  tests/  scripts/  paper/
└── results/         # gitignored
```

## Getting started

```bash
git clone https://github.com/FaiqaZarar/MammoRisk-Plus.git
cd MammoRisk-Plus

# Imaging / platform (Windows + CUDA)
conda env create -f environment-imaging.yml
conda activate mammorisk-imaging

# Genomics (macOS; on Windows use the standalone plink2 binary, see data/README.md)
conda env create -f environment-genomics.yml
conda activate mammorisk-genomics

# Mirai inference container (Module 1)
docker pull learn2cure/oncoserve_mirai:0.5.0
```

## Status

See [docs/STATUS.md](docs/STATUS.md) for per-module progress and results, and
[docs/DECISIONS.md](docs/DECISIONS.md) for every design decision. In short: data access is under way,
and the new PLINK 2.0 PRS-313 pipeline (M2) is next. A chromosome 1 PRSice-2 pilot exists in
`module2_prs/legacy_prsice/` for reference.

**Timeline:** M2 Oct–Nov 2026 · M1 Nov–Dec 2026 · M3 Nov 2026–Jan 2027 · **app release end of January
2027** · papers + thesis Feb–May 2027.

## Team

| Member | Responsibilities |
|---|---|
| **Faiqa Zarar Noor** (team lead) | Module 1 (Mirai), Module 3 (classifier + XAI), coordination |
| **Pukhraj Tahir** | Module 2 (PRS-313, O1) |
| **Manaal Tufail** | Evaluation, platform (Flask + Docker), literature; supports Module 3 |

**Supervisor:** Dr. Mehwish Noureen · **Co-supervisor:** Dr. Mian Ilyas Ahmad · NUST SINES

## Future work
- Fusion of imaging and genetic risk (needs a paired mammography + genotype dataset)
- Validation on Pakistani hospital mammograms
- Public Urdu awareness app
- Extension ideas from the literature check (reclassification versus genetic distance, a case-level reliability flag, explanation reliability under dataset shift, calibration and threshold transfer): see `docs/FUTURE_WORK.md`. Not decided or in scope.

## Key references
- Yala et al., Mirai, *Sci Transl Med* 2021; multi-institutional validation, *JCO* 2022
- Mavaddat et al., PRS-313, *AJHG* 2019 (PGS Catalog PGS000004)
- Selvaraju et al., Grad-CAM, *IJCV* 2020; Wang et al., Score-CAM, *CVPR-W* 2020
- Adebayo et al., Sanity Checks for Saliency Maps, *NeurIPS* 2018

## License
Developed for academic research as a Final Year Project at NUST SINES. Not for clinical or diagnostic use.
