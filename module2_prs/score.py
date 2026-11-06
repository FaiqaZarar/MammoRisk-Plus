"""Compute raw PRS: ``plink2 --score`` for real data, NumPy for tests and small extracts.

PRS_i = Σ_j β_j · G_ij, with G_ij ∈ {0, 1, 2} the number of effect alleles person i carries at SNP j.

plink2 notes:
* ``cols=+scoresums`` writes the plain SUM. By default plink2 reports the *average* per allele
  (sum / 2·n_non-missing), which changes scale when coverage differs. We always use the sum.
* A missing genotype in a scored SNP is mean-imputed by plink2 with that variant's allele frequency.
  ``--read-freq`` lets us use the reference frequency instead of the target's. We pass the pooled
  frequency file when provided, so every person is imputed the same way.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def plink2_score_command(genotype_prefix: str, score_file: str | Path, out_prefix: str | Path,
                         fileset: str = "pfile", plink2: str = "plink2",
                         read_freq: str | Path | None = None, keep: str | Path | None = None,
                         threads: int | None = None) -> list[str]:
    """Build the plink2 --score command line (list form, no shell quoting issues on Windows)."""
    flag = {"pfile": "--pfile", "bfile": "--bfile", "vcf": "--vcf"}[fileset]
    cmd = [plink2, flag, str(genotype_prefix), "--score", str(score_file), "1", "2", "3",
           "header-read", "cols=+scoresums", "list-variants", "--out", str(out_prefix)]
    if read_freq:
        cmd += ["--read-freq", str(read_freq)]
    if keep:
        cmd += ["--keep", str(keep)]
    if threads:
        cmd += ["--threads", str(threads)]
    return cmd


def run(cmd: list[str]) -> None:
    exe = cmd[0]
    if shutil.which(exe) is None and not Path(exe).exists():
        raise FileNotFoundError(
            f"'{exe}' not found. macOS: conda env create -f environment-genomics.yml. Windows: "
            "download the standalone plink2 binary and set plink2_path in configs/m2_prs.yaml "
            "(see data/README.md).")
    subprocess.run(cmd, check=True)


def read_sscore(path: str | Path) -> pd.DataFrame:
    """Read plink2 .sscore → columns IID, prs (the *_SUM column), n_alleles."""
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.lstrip("#") for c in df.columns]
    sum_cols = [c for c in df.columns if c.endswith("_SUM") and c != "NAMED_ALLELE_DOSAGE_SUM"]
    if len(sum_cols) != 1:
        raise ValueError(f"Expected one *_SUM score column, found {sum_cols}")
    out = pd.DataFrame({"IID": df["IID"].astype(str), "prs": df[sum_cols[0]].astype(float)})
    if "ALLELE_CT" in df:
        out["n_alleles"] = df["ALLELE_CT"]
    return out


def numpy_score(dosage: np.ndarray, weights: np.ndarray, eaf: np.ndarray | None = None) -> np.ndarray:
    """PRS from an (n_people × n_snps) effect-allele dosage matrix; NaN = missing genotype.

    Missing genotypes are mean-imputed with 2·EAF (reference allele frequency) when ``eaf`` is
    given, otherwise with the column mean of the sample (plink2's default behaviour).
    """
    d = np.asarray(dosage, float).copy()
    w = np.asarray(weights, float)
    fill = 2 * np.asarray(eaf, float) if eaf is not None else np.nanmean(d, axis=0)
    r, c = np.where(np.isnan(d))
    d[r, c] = fill[c]
    return d @ w
