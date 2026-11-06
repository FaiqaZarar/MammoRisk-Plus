"""Ancestry check: PCA of target + reference genotypes (plink2), and outlier flags.

Why: a person labelled "Pathan" whose genotypes cluster with another group (sample swap,
admixture, relatedness) would distort that group's PRS distribution. We run PCA on LD-pruned
common SNPs and flag people far from their own group's centroid.

Commands (genome-wide genotypes are needed here, not just the 313 PRS SNPs):
    plink2 --pfile merged --maf 0.05 --indep-pairwise 200 50 0.2 --out prune
    plink2 --pfile merged --extract prune.prune.in --pca 10 approx --out pca
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pca_commands(genotype_prefix: str, out_prefix: str, plink2: str = "plink2",
                 n_pcs: int = 10) -> list[list[str]]:
    return [
        [plink2, "--pfile", genotype_prefix, "--maf", "0.05", "--indep-pairwise", "200", "50",
         "0.2", "--out", f"{out_prefix}_prune"],
        [plink2, "--pfile", genotype_prefix, "--extract", f"{out_prefix}_prune.prune.in",
         "--pca", str(n_pcs), "approx", "--out", f"{out_prefix}_pca"],
    ]


def read_eigenvec(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.lstrip("#") for c in df.columns]
    df["IID"] = df["IID"].astype(str)
    return df


def flag_outliers(pcs: pd.DataFrame, groups: pd.Series, n_pcs: int = 2, k_sd: float = 4.0) -> pd.Series:
    """True if any of the first ``n_pcs`` PCs is > ``k_sd`` SDs from the person's group mean.

    Flagged people are reported (and excluded in a sensitivity analysis), not silently dropped.
    """
    cols = [f"PC{i}" for i in range(1, n_pcs + 1)]
    flags = pd.Series(False, index=pcs.index)
    for g, idx in groups.groupby(groups).groups.items():
        sub = pcs.loc[idx, cols]
        z = (sub - sub.mean()) / sub.std(ddof=1).replace(0, np.nan)
        flags.loc[idx] = (z.abs() > k_sd).any(axis=1)
    return flags
