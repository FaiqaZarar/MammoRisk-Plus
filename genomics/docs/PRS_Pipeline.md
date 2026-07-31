# Polygenic Risk Score (PRS) Pipeline

## Overview

This document describes the genomics workflow used in the MammoRisk-Plus project to calculate Polygenic Risk Scores (PRS) for breast cancer risk prediction.

The pipeline scores individuals using the **PRS313** model from the Polygenic Score (PGS) Catalog, applied to genotype data from the **1000 Genomes Project**, with final scores computed via **PRSice-2**. The current implementation is validated on chromosome 1 as a pilot before scaling to the full genome.

---

## Table of Contents

1. [Pipeline Summary](#pipeline-summary)
2. [Prerequisites](#prerequisites)
3. [Step 1 — Download the PRS Model](#step-1--download-the-prs-model)
4. [Step 2 — Download Genotype Data](#step-2--download-genotype-data)
5. [Step 3 — Convert Genotype Data to PLINK Format](#step-3--convert-genotype-data-to-plink-format)
6. [Step 4 — Extract PRS Variants](#step-4--extract-prs-variants)
7. [Step 5 — Match Variants to Genotype Data](#step-5--match-variants-to-genotype-data)
8. [Step 6 — Prepare the PRSice Base File](#step-6--prepare-the-prsice-base-file)
9. [Step 7 — Calculate Polygenic Risk Scores](#step-7--calculate-polygenic-risk-scores)
10. [Output Files](#output-files)
11. [Example Result](#example-result)
12. [Scripts Reference](#scripts-reference)
13. [Software Used](#software-used)
14. [Current Status](#current-status)
15. [Future Work](#future-work)
16. [References](#references)

---

## Pipeline Summary

```
Download PRS313
        │
        ▼
Download genotype data
        │
        ▼
Convert genotype data to PLINK format
        │
        ▼
Extract PRS variants
        │
        ▼
Match PRS variants to genotype data
        │
        ▼
Generate PRSice base file
        │
        ▼
Run PRSice-2
        │
        ▼
Calculate Polygenic Risk Scores
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| PLINK 2.0 | Genotype format conversion and variant handling |
| PRSice-2 | Requires R (≥ 3.5 recommended) |
| Python 3.x | With `pandas` installed |
| Genotype data | 1000 Genomes Project, GRCh37 build |
| PRS scoring file | PGS000004 (PRS313), GRCh37 build |

> ⚠️ Genome build consistency matters: both the PRS scoring file and the genotype data must be on **GRCh37**, since matching is done by chromosome and base-pair position rather than rsID.

---

## Step 1 — Download the PRS Model

The breast cancer Polygenic Risk Score used in this project:

| Field | Value |
|---|---|
| PGS ID | PGS000004 |
| Model | PRS313 |
| Trait | Breast Cancer |
| Genome Build | GRCh37 |

The scoring file provides, per variant:

- Chromosome number
- Base-pair position
- Effect allele
- Other allele
- Effect size (beta)

**Note:** This model does not include rsIDs. As a result, all downstream variant matching is performed using **chromosome + base-pair position** rather than rsID lookup.

---

## Step 2 — Download Genotype Data

Genotype data for pipeline testing was sourced from the **1000 Genomes Project**.

For initial validation, only **chromosome 1** was downloaded. Restricting the pilot to a single chromosome allows the full workflow — extraction, matching, scoring — to be validated end-to-end before committing compute time to a genome-wide run.

---

## Step 3 — Convert Genotype Data to PLINK Format

Genotype data was converted into PLINK binary format, producing:

- `.bed` — genotype calls
- `.bim` — variant information
- `.fam` — sample information

These three files are the required input format for all downstream PRSice-2 operations.

---

## Step 4 — Extract PRS Variants

**Script:** `extract_chr1_prs.py`

Extracts only the variants located on chromosome 1 from the full PRS313 scoring file, reducing it to the subset relevant to the available genotype data.

**Output:** `chr1_prs_variants.csv`

---

## Step 5 — Match Variants to Genotype Data

**Script:** `match_prs_to_bim.py`

Compares PRS variants against PLINK `.bim` variants. Since the scoring file lacks rsIDs, matching is performed using **chromosome + base-pair position**.

**Outputs:**

| File | Description |
|---|---|
| `matched_variants.csv` | PRS variants present in the genotype dataset |
| `missing_variants.csv` | PRS variants absent from the genotype dataset |

The proportion of matched vs. missing variants is a useful sanity check — a low match rate typically signals a genome-build mismatch or a genotyping coverage gap.

---

## Step 6 — Prepare the PRSice Base File

**Script:** `prepare_prsice_base.py`

Builds a PRSice-2–compatible scoring (base) file with the required columns:

| Column | Description |
|---|---|
| `SNP` | Variant identifier |
| `CHR` | Chromosome |
| `BP` | Base-pair position |
| `A1` | Effect allele |
| `A2` | Other allele |
| `BETA` | Effect size |
| `P` | Association p-value |

**Output:** `prsice_base.txt`

---

## Step 7 — Calculate Polygenic Risk Scores

PRSice-2 combines genotype calls, effect alleles, and effect sizes to compute an overall genetic risk score per individual.

**Script:** `run_prsice.bat`

```bash
Rscript PRSice.R \
  --base prsice_base.txt \
  --target genotype_dataset \
  --binary-target F \
  --out prs_output
```

| Flag | Meaning |
|---|---|
| `--base` | Path to the prepared PRSice base file |
| `--target` | PLINK-format genotype dataset (prefix, without extension) |
| `--binary-target F` | Target phenotype is continuous, not case/control |
| `--out` | Output file prefix |

---

## Output Files

| File | Description |
|---|---|
| `prs_output.all_score` | Final PRS scores for each individual |
| `prs_output.log` | Execution log |
| `prs_output.prsice` | Summary statistics for the scoring run |

---

## Example Result

| Individual | PRS Score |
|---|---|
| HG00096 | 0.0846 |
| HG00239 | 0.6786 |
| ... | ... |

Each value is an individual's Polygenic Risk Score, computed from the variants available in the current (chromosome 1) genotype set. Scores are expected to shift once scoring is extended genome-wide, since more of the 313 variants will be captured.

---

## Scripts Reference

| Script | Stage | Purpose |
|---|---|---|
| `extract_chr1_prs.py` | Step 4 | Extract chromosome 1 variants from the PRS313 file |
| `match_prs_to_bim.py` | Step 5 | Match PRS variants to genotype `.bim` variants |
| `prepare_prsice_base.py` | Step 6 | Build the PRSice-2 base scoring file |
| `run_prsice.bat` | Step 7 | Run the PRSice-2 scoring command |

---

## Software Used

| Tool | Role |
|---|---|
| Python | Variant extraction and matching scripts |
| pandas | Data manipulation within Python scripts |
| PLINK 2 | Genotype format conversion |
| PRSice-2 | Polygenic score calculation |
| R | Required runtime for PRSice-2 |

---

## Current Status

- [x] PRS313 scoring file downloaded
- [x] Genotype data (chromosome 1) processed
- [x] Variant matching completed
- [x] PRSice base file generated
- [x] Polygenic Risk Scores successfully calculated for the chromosome 1 pilot

---

## Future Work

- Extend variant extraction and matching to all autosomal chromosomes
- Re-validate match rate and score distribution on the genome-wide run
- Integrate PRS output with the mammography-based (Mirai) imaging risk score
- Incorporate structured clinical risk factors
- Produce a unified multimodal breast cancer risk score

---

## References

1. Polygenic Score (PGS) Catalog — [pgscatalog.org](https://www.pgscatalog.org) (PGS000004)
2. PRSice-2 Documentation — [choishingwan.github.io/PRSice](https://choishingwan.github.io/PRSice/)
3. PLINK 2 Documentation — [www.cog-genomics.org/plink/2.0](https://www.cog-genomics.org/plink/2.0/)
4. 1000 Genomes Project — [internationalgenome.org](https://www.internationalgenome.org)