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
`data/prs/PGS000004.txt.gz` is the **author-reported GRCh37** scoring file (`#genome_build=GRCh37`),
**not** the GRCh38 harmonised file specified in CLAUDE.md. It has not been replaced yet. Before
scoring, the weights' build must match the genotypes. 1000G phase 3 is GRCh37, while the 1000G
high-coverage and HGDP releases are GRCh38.

## Storage
CBIS-DDSM alone is ~164 GB as DICOM, and SINES only gives ~20 GB. Cache downsampled PNGs under
`data/processed/` and delete raw DICOMs once they are cached. Warn the team before any large download.

## Tools: plink2 and bcftools
- **macOS / Linux:** install both from bioconda with `conda env create -f environment-genomics.yml`.
- **Windows:** bioconda doesn't build for Windows. **plink2 is a standalone binary.** Download the
  Windows build from https://www.cog-genomics.org/plink/2.0/, unzip `plink2.exe`, and either add its
  folder to `PATH` or set its path in the config. Don't commit it (`tools/` and `*.exe` are gitignored).
  For bcftools on Windows, use WSL or the genomics Docker/Mac setup.
