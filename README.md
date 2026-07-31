# 🩺 MammoRisk-Plus

**A multimodal breast cancer risk prediction system integrating deep learning on mammograms, polygenic risk scores, and clinical risk factors.**

[![Status](https://img.shields.io/badge/status-in%20development-yellow)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-academic--research-lightgrey)]()

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [System Architecture](#system-architecture)
4. [Repository Structure](#repository-structure)
5. [Team](#team)
6. [Module Progress](#module-progress)
   - [Genomics](#-genomics-module)
   - [Imaging](#-imaging-module)
   - [Evaluation](#-evaluation)
7. [Tech Stack](#tech-stack)
8. [Getting Started](#getting-started)
9. [Roadmap](#roadmap)
10. [References](#references)
11. [License](#license)
12. [Contact](#contact)

---

## Project Overview

Breast cancer remains one of the leading causes of cancer-related mortality among women worldwide. Identifying high-risk individuals early enables tailored screening schedules, timely preventive intervention, and improved long-term outcomes.

**MammoRisk-Plus** is a Final Year Project (Department of Bioinformatics, 2026) that unifies three independent, complementary sources of risk information into a single predictive framework:

| Modality | Method | Signal Captured |
|---|---|---|
| 🧠 Imaging | Mirai deep learning model | Radiological texture and density patterns |
| 🧬 Genetic | Polygenic Risk Score (PRS-313) | Inherited genetic susceptibility |
| 📋 Clinical | Structured risk factors | Age, family history, reproductive history |

Rather than relying on any single modality, MammoRisk-Plus fuses all three to produce a more comprehensive, personalized breast cancer risk profile than traditional single-source models.

---

## Objectives

- [x] Calculate Polygenic Risk Scores using PRS-313 from the PGS Catalog
- [ ] Predict imaging-based risk from mammograms using the Mirai model
- [ ] Incorporate structured clinical risk factors into the prediction pipeline
- [ ] Fuse imaging, genomic, and clinical scores into a unified risk model
- [ ] Develop a web-based application for integrated risk assessment
- [ ] Produce an explainable, clinician-readable risk report

---

## System Architecture

```
                         Mammogram
                             │
                             ▼
                       Mirai Model
                             │
                     Imaging Risk Score
                             │
                             │
  Clinical Data ─────────────┼───────────── PRS-313 Score
   (age, history)            │           (genotype data)
                             ▼
                Multimodal Risk Integration
                             │
                             ▼
             Personalized Breast Cancer Risk Report
```

Each modality is processed independently by its own module, producing a standardized risk score. These scores are then combined in the integration layer to generate a single, weighted risk estimate.

---

## Repository Structure

```
MammoRisk-Plus/
├── app/                    # Web application (Flask)
├── data/                   # Dataset references and metadata
├── docs/                   # Project documentation
├── evaluation/              # Model evaluation and metrics
├── genomics/                # PRS computation pipeline
│   ├── docs/
│   ├── examples/
│   └── scripts/
├── imaging/                 # Mammography deep learning pipeline
├── models/                  # Trained/serialized models
├── notebooks/                # Exploratory analysis notebooks
├── tests/                    # Unit and integration tests
├── README.md
└── .gitignore
```

---

## Team

| Member | Role |
|---|---|
| **Faiqa Zarar** | Machine Learning Lead |
| **Pukhraj Tahir** | Genomics Lead |
| **Manaal** | Evaluation Lead |

---

## Module Progress

### 🧬 Genomics Module

**Status:** ✅ Pipeline functional (chromosome 1 pilot complete)

- [x] Downloaded PRS-313 from the Polygenic Score (PGS) Catalog
- [x] Installed PLINK 2.0
- [x] Installed PRSice-2
- [x] Downloaded chromosome 1 genotype data from the 1000 Genomes Project
- [x] Matched PRS variants against genotype data
- [x] Generated PRSice base file
- [x] Calculated initial Polygenic Risk Scores

**Key scripts** (`genomics/scripts/`):

| Script | Purpose |
|---|---|
| `extract_chr1_prs.py` | Extracts chromosome 1 variants relevant to PRS-313 |
| `match_prs_to_bim.py` | Aligns PRS variants to genotype `.bim` files |
| `prepare_prsice_base.py` | Builds the PRSice-2 base (summary statistics) file |
| `run_prsice.bat` | Runs the PRSice-2 scoring pipeline |

**Next steps:** extend the pipeline from chromosome 1 to genome-wide scoring, and validate scores against published PRS-313 benchmarks.

### 🧠 Imaging Module

**Status:** 🚧 In Progress

Planned tasks:
- [ ] CBIS-DDSM dataset preprocessing
- [ ] Mirai model environment setup
- [ ] Batch inference on mammogram images
- [ ] Imaging-based risk score generation

### 📊 Evaluation

**Status:** ⏳ Planned

The integrated system will be evaluated using standard classification metrics:

- AUROC
- Accuracy
- Precision / Recall
- F1-score

Evaluation will benchmark the multimodal model against each single-modality baseline (imaging-only, PRS-only, clinical-only) to quantify the benefit of fusion.

---

## Tech Stack

| Category | Tools |
|---|---|
| Languages | Python, R |
| Genomics | PLINK 2, PRSice-2 |
| Deep Learning | PyTorch |
| Web Application | Flask |
| Infrastructure | Docker |
| Version Control | Git, GitHub |

---

## Getting Started

> ⚠️ The project is under active development — setup instructions below will expand as each module stabilizes.

```bash
# Clone the repository
git clone https://github.com/<org>/MammoRisk-Plus.git
cd MammoRisk-Plus

# (Genomics pipeline) install dependencies
# See genomics/docs/ for PLINK 2 and PRSice-2 setup instructions
```

Module-specific setup and usage instructions are documented separately in `genomics/docs/`, with `imaging/` and `app/` documentation to follow as those modules mature.

---

## Roadmap

- [x] Repository setup and structure
- [x] PRS computation pipeline (chromosome 1 pilot)
- [ ] Genome-wide PRS scoring
- [ ] Mirai imaging pipeline integration
- [ ] Multimodal risk fusion model
- [ ] Flask web application
- [ ] Downloadable clinical risk reports
- [ ] Validation on external/held-out datasets

---

## References

- Polygenic Score (PGS) Catalog — [pgscatalog.org](https://www.pgscatalog.org)
- 1000 Genomes Project — [internationalgenome.org](https://www.internationalgenome.org)
- Mirai: deep learning model for mammography-based breast cancer risk prediction

---

## License

This project is developed for academic and research purposes as part of a Final Year Project. It is **not intended for clinical or diagnostic use**.

---

## Contact

For questions about this project, please reach out via the team members listed above or open an issue in this repository.

---

**Department of Bioinformatics · Final Year Project · 2026**
