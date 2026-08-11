# MammoRisk+

## Development and Validation of a Clinical Platform for Breast Cancer Risk Assessment and AI-Based Detection Using Deep Learning Mammography Analysis and Polygenic Risk Scoring

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-academic--research-lightgrey)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Modules](#modules)
4. [Repository Structure](#repository-structure)
5. [Datasets](#datasets)
6. [Team](#team)
7. [Module Progress](#module-progress)
8. [Tech Stack](#tech-stack)
9. [Getting Started](#getting-started)
10. [Roadmap](#roadmap)
11. [References](#references)
12. [License](#license)

---

## Project Overview

Breast cancer is the most frequently diagnosed malignancy among women worldwide, with 2.3 million new cases annually. Early and accurate risk assessment and detection are the most effective levers for reducing mortality.

**MammoRisk+** is a Final Year Project (BS Bioinformatics, NUST SINES, 2026) that delivers three independent, validated clinical modules for breast cancer risk assessment and AI-based detection.

The three modules operate completely independently — there is no fusion of outputs in the current scope. Each module addresses a distinct clinical question:

| Module | Clinical Question | Method |
|--------|------------------|--------|
| Module 1 — Mammography Risk Prediction | Will this patient develop breast cancer in the next 1 to 5 years? | Mirai deep learning model on DICOM mammograms |
| Module 2 — Genomic Risk Scoring | What is this patient's inherited genetic predisposition to breast cancer? | PRS-313 polygenic risk score from VCF genomic data |
| Module 3 — AI-Based Clinical Detection | Does this mammogram show a suspicious finding right now? | EfficientNet CNN trained on CBIS-DDSM with Grad-CAM XAI |

> **Note on scope:** Multimodal fusion of Module 1 and Module 2 outputs is technically motivated by the literature but is not in the current project scope due to the absence of publicly available paired mammography-genomic datasets. Fusion is proposed as future work.

---

## System Architecture

```
MODULE 1                    MODULE 2                    MODULE 3
─────────────────           ─────────────────           ─────────────────
Input: DICOM                Input: VCF File             Input: DICOM
Mammogram                   (Genomic SNP data)          Mammogram
     |                           |                           |
     v                           v                           v
Mirai Deep                  PRS-313 Formula             EfficientNet-B3
Learning Model              (PGS Catalog                CNN Classifier
(Pre-trained,               PGS000004)                  (Trained on
MIT licence)                PRSice-2 +                  CBIS-DDSM)
     |                      PLINK 2.0                        |
     v                           |                           v
1-5 Year Risk               Genomic Risk            Suspicious / Normal
Probability                 Score +                 Classification
+                           Population              +
Grad-CAM                    Percentile              Grad-CAM Heatmap
Heatmap                          |                  (Gradient-weighted
     |                           |                  Class Activation
     v                           v                  Mapping)
PDF Report                  PDF Report                  PDF Report
```

---

## Modules

### Module 1 — Mammography Risk Prediction

- **Model:** Mirai (Yala et al., MIT/MGH, 2021) — pre-trained, MIT licence
- **Input:** DICOM mammogram — all 4 standard views (L-CC, L-MLO, R-CC, R-MLO)
- **Output:** Calibrated 1-year to 5-year breast cancer risk probability
- **XAI:** Grad-CAM heatmap highlighting regions driving the prediction
- **Dataset:** RSNA 2022 (training/validation), EMBED (temporal validation)
- **Target:** C-index 0.75 to 0.84 on RSNA 2022 test set

### Module 2 — Genomic Risk Scoring

- **Model:** PRS-313 formula (Mavaddat et al., 2019) — no training required
- **Input:** VCF file (patient genotype data)
- **Output:** Polygenic risk score + population percentile + risk category
- **Formula:** PRS = Σ (βi × Gi) for i = 1 to 313
- **Tools:** PLINK 2.0, PRSice-2
- **Dataset:** PGS Catalog PGS000004 (formula), 1000 Genomes PJL (validation)

### Module 3 — AI-Based Clinical Detection + XAI

- **Model:** EfficientNet-B3 or ResNet-50 CNN (trained from scratch)
- **Input:** DICOM mammogram
- **Output:** Suspicious / Normal classification + Grad-CAM heatmap
- **Dataset:** CBIS-DDSM (10,239 pathology-verified mammograms)
- **Split:** 70% train / 10% validation / 20% test at patient level
- **Target:** AUC above 0.85 on held-out CBIS-DDSM test set

---

## Repository Structure

```
MammoRisk-Plus/
│
├── src/
│   ├── imaging/            # Module 1 — Mirai inference + Grad-CAM
│   ├── genomics/           # Module 2 — PRS-313 pipeline
│   ├── evaluation/         # Metrics: AUC, C-index, DeLong's test
│   ├── preprocessing/      # DICOM to PNG, metadata CSV generation
│   ├── api/                # Flask backend — endpoints for all 3 modules
│   └── fusion/             # Future work — multimodal fusion
│
├── data/
│   ├── raw/                # Raw datasets (gitignored)
│   │   ├── cbis_ddsm/
│   │   ├── rsna_2022/
│   │   ├── embed/
│   │   └── vindr_mammo/
│   ├── genomics/           # VCF files + PRS outputs (gitignored)
│   └── processed/          # Mirai-compatible metadata CSVs
│
├── models/                 # Trained models + snapshots (gitignored)
│   ├── mirai/
│   └── module3/
│
├── frontend/               # Web UI
│   ├── templates/
│   └── static/
│
├── genomics/               # Module 2 scripts (Pukhraj)
│   ├── scripts/
│   ├── docs/
│   └── examples/
│
├── notebooks/              # Exploratory analysis
├── tests/                  # Unit and integration tests
├── configs/                # Model and pipeline configuration files
├── docs/
│   └── weekly_reports/     # Supervisor logbook entries
├── paper/
│   ├── draft/
│   └── figures/
│
├── README.md
└── .gitignore
```

---

## Datasets

| Dataset | Module | Purpose | Size | Access |
|---------|--------|---------|------|--------|
| RSNA 2022 | Module 1 | Training and risk prediction validation | ~100 GB | AWS Open Data, free |
| EMBED (Emory) | Module 1 | Temporal validation (longitudinal, 5yr follow-up) | ~300 GB | AWS Open Data, free |
| CBIS-DDSM | Module 3 | CNN training and detection validation | ~6 GB | TCIA, free |
| VinDr-Mammo | Module 3 | Asian generalisation test | ~200 GB | PhysioNet, free |
| 1000 Genomes PJL | Module 2 | PRS pipeline validation | ~5 GB | internationalgenome.org, free |
| PGS Catalog PGS000004 | Module 2 | PRS-313 formula (313 SNPs + effect sizes) | <1 MB | pgscatalog.org, free |

> All datasets are publicly available for non-commercial research use. No ethics board approval required.

---

## Team

| Member | Role | Responsibilities |
|--------|------|-----------------|
| **Faiqa Zarar Noor (471543)** | ML Lead + Group Coordinator | Module 1 (Mirai), Module 3 (CNN + Grad-CAM), Flask backend, system integration |
| **Pukhraj Tahir (467407)** | Genomics Lead | Module 2 (PRS-313 pipeline), VCF processing, 1000 Genomes analysis |
| **Manaal Tufail (462781)** | Evaluation Lead | Dataset preparation, evaluation pipeline, paper writing |

**Supervisor:** Dr. Mehwish Noureen — Assistant Professor, NUST SINES  
**Co-Supervisor:** Dr. Mian Ilyas Ahmad — Professor, NUST SINES

---

## Module Progress

### Module 1 — Mammography Risk Prediction

**Status:** 🚧 In Progress

- [ ] Docker Desktop installed
- [ ] Mirai container pulled (`docker pull learn2cure/oncoserve_mirai:0.5.0`)
- [ ] RSNA 2022 dataset downloaded
- [ ] DICOM to PNG preprocessing pipeline built
- [ ] Mirai-compatible metadata CSV generated
- [ ] First Mirai inference on test mammogram
- [ ] Grad-CAM heatmap implementation
- [ ] Validation on RSNA 2022 (target C-index 0.75 to 0.84)
- [ ] Temporal validation on EMBED

---

### Module 2 — Genomic Risk Scoring

**Status:** ✅ Pipeline functional — chromosome 1 pilot complete

- [x] PRS-313 downloaded from PGS Catalog (PGS000004)
- [x] PLINK 2.0 installed
- [x] PRSice-2 installed
- [x] Chromosome 1 genotype data downloaded from 1000 Genomes Project
- [x] PRS variants matched to genotype data
- [x] PRSice base file generated
- [x] Initial PRS scores calculated (chromosome 1 pilot)
- [ ] Extend to genome-wide scoring (all chromosomes)
- [ ] Validate score distributions against published PRS-313 benchmarks
- [ ] Build VCF upload Flask endpoint

**Key scripts** (`genomics/scripts/`):

| Script | Purpose |
|--------|---------|
| `extract_chr1_prs.py` | Extracts chromosome 1 variants relevant to PRS-313 |
| `match_prs_to_bim.py` | Aligns PRS variants to genotype `.bim` files |
| `prepare_prsice_base.py` | Builds the PRSice-2 base (summary statistics) file |
| `run_prsice.bat` | Runs the PRSice-2 scoring pipeline |

---

### Module 3 — AI-Based Clinical Detection + XAI

**Status:** ⏳ Not started

- [ ] Download CBIS-DDSM from TCIA (cancerimagingarchive.net)
- [ ] DICOM to PNG preprocessing
- [ ] Build 70/10/20 train/validation/test split at patient level
- [ ] Train EfficientNet-B3 CNN on benign/malignant labels
- [ ] Implement Grad-CAM heatmap generation (captum library)
- [ ] Evaluate on held-out test set (target AUC above 0.85)
- [ ] Verify Grad-CAM regions against CBIS-DDSM ROI segmentation masks

---

### Web Platform

**Status:** ⏳ Not started

- [ ] Flask backend skeleton with three module endpoints
- [ ] DICOM upload interface
- [ ] VCF upload interface
- [ ] Risk score display + heatmap visualisation
- [ ] Downloadable PDF report generation

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| Deep Learning | PyTorch 2.x, torchvision |
| Image Processing | OpenCV, pydicom, albumentations |
| Genomics | PLINK 2.0, PRSice-2, bcftools |
| Explainability | captum (Grad-CAM), OpenCV |
| Evaluation | scikit-learn, lifelines (AUC, C-index, DeLong's test) |
| Web Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Containerisation | Docker |
| Experiment Tracking | Weights and Biases |
| Version Control | Git, GitHub |
| Compute | Google Colab Pro / NUST GPU cluster |

---

## Getting Started

> The project is under active development. Setup instructions expand as each module stabilises.

```bash
# Clone the repository
git clone https://github.com/FaiqaZarar/MammoRisk-Plus.git
cd MammoRisk-Plus

# Install Python dependencies
pip install torch torchvision pandas numpy scikit-learn matplotlib \
            opencv-python flask pydicom shap captum lifelines wandb

# Pull the Mirai Docker container (Module 1)
docker pull learn2cure/oncoserve_mirai:0.5.0

# Genomics pipeline setup — see genomics/docs/ for
# PLINK 2.0 and PRSice-2 installation instructions
```

---

## Roadmap

- [x] Repository setup and folder structure
- [x] PRS-313 computation pipeline — chromosome 1 pilot (Module 2)
- [ ] Genome-wide PRS scoring (Module 2)
- [ ] Mirai inference pipeline (Module 1)
- [ ] Grad-CAM explainability — Module 1
- [ ] CBIS-DDSM CNN training (Module 3)
- [ ] Grad-CAM explainability — Module 3
- [ ] Flask web platform with three module endpoints
- [ ] Validation — RSNA 2022 (Module 1), CBIS-DDSM (Module 3)
- [ ] Temporal validation — EMBED (Module 1)
- [ ] Downloadable PDF clinical risk report
- [ ] Paper submission — npj Breast Cancer / Cancers MDPI

**Future Work:**
- Multimodal fusion of Module 1 and Module 2 (requires paired mammography-genomic dataset)
- Molecular subtype classification extension of Module 3
- Population-specific PRS recalibration for South Asian genomes

---

## References

- Yala et al., "Multi-Institutional Validation of a Mammography-Based Breast Cancer Risk Model," JCO, 2022
- Mavaddat et al., "Polygenic Risk Scores for Prediction of Breast Cancer," AJHG, 2019
- Padrik et al., "Guidance for the Clinical Use of Breast Cancer Polygenic Risk Scores," Cancers, 2025
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks," IJCV, 2020
- PGS Catalog: pgscatalog.org/score/PGS000004
- 1000 Genomes Project: internationalgenome.org
- CBIS-DDSM: cancerimagingarchive.net/collection/cbis-ddsm
- RSNA 2022: registry.opendata.aws/rsna-screening-mammography-breast-cancer-detection
- EMBED: registry.opendata.aws/emory-breast-imaging-dataset-embed

---

## License

This project is developed for academic and research purposes as part of a Final Year Project at NUST SINES. It is **not intended for clinical or diagnostic use**.

---

**BS Bioinformatics · Final Year Project · NUST SINES · 2026–2027**
