# MammoRisk-Plus

**A multimodal breast cancer risk prediction system integrating deep learning, genomics, and clinical data.**

MammoRisk-Plus combines three complementary signals to improve individualized breast cancer risk stratification:

- 🧠 **Deep Learning on Mammography** — image-based risk modeling using Mirai
- 🧬 **Polygenic Risk Scores (PRS-313)** — genomic risk contribution derived from the PGS Catalog
- 📋 **Clinical Risk Factors** — established epidemiological and clinical predictors

By fusing imaging, genomic, and clinical modalities, the project aims to produce a more accurate and personalized risk profile than any single modality alone.

---

## Table of Contents

- [Team](#team)
- [Project Structure](#project-structure)
- [Progress Log](#progress-log)
- [Getting Started](#getting-started)
- [License](#license)

---

## Team

| Member | Role |
|---|---|
| Faiqa Zarar | Machine Learning Lead |
| Pukhraj Khan | Genomics Lead |
| Manaal | Evaluation Lead |

---

## Project Structure

```
MammoRisk-Plus/
├── app/            # Application / integration layer
├── data/           # Raw and processed data
├── docs/           # Documentation and reports
├── evaluation/      # Model evaluation and metrics
├── genomics/        # PRS computation pipeline
├── imaging/         # Mammography deep learning pipeline
├── scripts/         # Utility and preprocessing scripts
├── .gitignore
└── README.md
```

---

## Progress Log

### Week 1

**Genomics — Pukhraj** ✅
- Downloaded PRS-313 from the PGS Catalog
- Installed PLINK 2.0 and PRSice-2
- Processed chromosome 1 genotype data
- Matched PRS variants against the reference panel
- Generated initial polygenic risk scores

**Imaging — Faiqa** 🔄 *In Progress*

**Evaluation — Manaal** 🔄 *In Progress*

---

## Getting Started

> Setup instructions will be added as each module reaches a runnable state.

```bash
git clone https://github.com/<org>/MammoRisk-Plus.git
cd MammoRisk-Plus
```

---

## License

This repository is intended for academic and research purposes only. Not for clinical or diagnostic use.