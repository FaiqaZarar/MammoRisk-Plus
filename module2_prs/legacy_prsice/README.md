# Legacy PRSice-2 pilot (chromosome 1): needs review by Pukhraj

**Status: frozen. Kept for reference only. Do not use for O1 results.**

The scripts in `scripts/` and `PRS_Pipeline.md` were moved here unchanged on 2026-11-06 from
`genomics/scripts/` and `genomics/docs/` (see DECISIONS D-13). They are Pukhraj Tahir's
chromosome 1 pilot, built with PRSice-2.

> **Action for Pukhraj:** please review this folder. Module 2 is switching to **PLINK 2.0
> (`--score`) + bcftools** (DECISIONS D-10). Confirm what, if anything, should be carried over
> into the new `module2_prs/` pipeline. Nobody else should rewrite these files.

Why these scripts won't be used as they are:
- **PRSice-2** is replaced by `plink2 --score`, because we apply published PGS000004 weights and
  don't tune thresholds (D-10).
- **Chromosome 1 only.** PRS-313 needs all 313 SNPs, which sit on several chromosomes.
- **Hard-coded paths** (`C:\MammoRiskPlus\...`). CLAUDE.md requires paths to come from config.
- **Genome build:** `data/prs/PGS000004.txt.gz` is the author-reported **GRCh37** file, not the
  GRCh38 harmonised file that CLAUDE.md specifies. Match the build to the genotype data
  (1000G phase 3 = GRCh37; 1000G high-coverage / HGDP releases = GRCh38).
- `extract_chr1_prs.py` reads `data/prs/PGS000004.txt/PGS000004.txt`, an unpacked duplicate that
  has since been removed. Use the `.gz` instead.
