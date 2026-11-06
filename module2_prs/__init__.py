"""Module 2: PRS-313 under a European (CEU) vs Pakistani reference (Objective O1, core contribution).

Pipeline (``python -m module2_prs.run --config configs/m2_prs.yaml``):

1. ``weights``      load PGS000004 (author-reported GRCh37 or harmonised GRCh38) → tidy table
2. ``harmonise``    match the 313 weights to the target variants (.pvar/.bim): allele swaps, strand
                    flips, palindromic A/T C/G SNPs, indels → plink2 score file + coverage report
3. ``score``        ``plink2 --score`` (or a NumPy scorer on a dosage matrix) → raw PRS per person
4. ``standardise``  z = (score − ref mean) / ref SD under each reference; percentile from the
                    fitted normal; risk category
5. ``reclassify``   % changing category between the European (CEU) and Pakistani reference, with a
                    stratified bootstrap CI that also re-estimates the reference mean/SD
6. ``drivers``      per-SNP contribution to the mean shift between populations
7. ``ancestry``     plink2 PCA commands + outlier flags against the reference panel

We do NOT claim which category is clinically correct, and we make no absolute-risk claims: there
are no outcomes (CLAUDE.md).
"""
