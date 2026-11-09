# Data

**No dataset is stored in this repository.** Everything in `data/` is gitignored except this README,
`.gitkeep` placeholders and the public PRS-313 weights file. Paths are set in `configs/*.yaml`, never
hard-coded. Never commit images, genomes, patient files or model weights. VinDr-Mammo (PhysioNet
licence) and EMBED images must never be shown or committed.

## Datasets

| Dataset | Module | Use | Where to get it |
|---|---|---|---|
| PGS000004 (PRS-313) | M2 | Weights: 313 SNPs + effect sizes | pgscatalog.org/score/PGS000004 |
| 1000 Genomes: PJL (96) + CEU | M2 | Pakistani group + European reference | internationalgenome.org |
| HGDP: 8 Pakistani groups | M2 | Pakistani groups (~20–25 each) | HGDP / gnomAD HGDP+1kG release |
| EMBED (open 20% subset) | M1 | Mirai external validation | AWS Open Data (registration required) |
| CBIS-DDSM | M3 | Train / val / test classifier | TCIA |
| CMMD | M3 | External test | TCIA |
| DMID | M3 | External test + explanations | Published dataset release |
| VinDr-Mammo | M3 | Explanations only (lesion boxes) | PhysioNet (credentialed) |

RSNA 2022 is **not** used, because it has no follow-up times (see DECISIONS D-07).

### PRS-313 weights: genome build
- **In use:** `data/prs/PGS000004_hmPOS_GRCh38.txt.gz`, the PGS Catalog harmonised GRCh38 file (all 313
  variants mapped), which matches the gnomAD GRCh38 genotypes (DECISIONS D-19).
- Kept for reference: `data/prs/PGS000004.txt.gz`, the author-reported GRCh37 file. Don't score
  GRCh38 genotypes with it, because the positions differ (`check_build` refuses).

### Module 2 genotypes: gnomAD v3.1.2 HGDP+1KG (GRCh38)
CEU, PJL and the 8 HGDP Pakistani groups were jointly called in one public, tabix-indexed callset
(3.63 TB in total). We **never download it whole**: `python -m module2_prs.run --step extract` fetches only
the 313 positions with `bcftools view -R` (a few MB). The easiest place to run it is Colab:
`notebooks/m2_prs_colab.ipynb`. The sample metadata (3.4 MB) is downloaded by `--step samples`.

## Storage
CBIS-DDSM is **streamed** from the TCIA API straight into the PNG cache, so raw DICOMs are never kept
(`module3_classifier/fetch_cbis.py`, D-20). Build the cache on Kaggle. The rest of this section applies to
VinDr-Mammo, CMMD and EMBED.

CBIS-DDSM alone is ~164 GB as DICOM, and SINES only gives ~20 GB. Cache downsampled PNGs under
`data/processed/` and delete raw DICOMs once they are cached. Warn the team before any large download.

## Tools: plink2 and bcftools
- **macOS / Linux:** install both from bioconda with `conda env create -f environment-genomics.yml`.
- **Windows:** bioconda doesn't build for Windows. **plink2 is a standalone binary.** Download the
  Windows build from https://www.cog-genomics.org/plink/2.0/, unzip `plink2.exe`, and either add its
  folder to `PATH` or set its path in the config. Don't commit it (`tools/` and `*.exe` are gitignored).
  For bcftools on Windows, use WSL or the genomics Docker/Mac setup.
