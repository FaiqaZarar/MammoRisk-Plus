"""Which SNPs drive the shift between the European (CEU) and Pakistani PRS distributions?

Theory
------
The expected PRS in population k is E[PRS_k] = Σ_j β_j · 2·p_jk, where p_jk is the effect allele
frequency of SNP j in population k. So the difference in mean PRS between the Pakistani sample and
CEU decomposes *exactly* into per-SNP terms:

    Δ = Σ_j  c_j,   c_j = 2 · β_j · (p_j,PAK − p_j,CEU)

Dividing by σ_CEU expresses each term in CEU z-units, i.e. how far that SNP alone moves the
average Pakistani person along the European reference scale. SNPs are ranked by |c_j|. ``share``
is c_j / Δ. Shares can exceed 1 or be negative when SNPs push in opposite directions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def allele_freqs_from_dosage(dosage: np.ndarray, groups: np.ndarray, snp_ids: list[str]) -> pd.DataFrame:
    """Effect allele frequency per group from an (n × m) dosage matrix (NaN = missing)."""
    rows = {}
    for g in np.unique(groups):
        rows[g] = np.nanmean(np.asarray(dosage, float)[groups == g], axis=0) / 2
    return pd.DataFrame(rows, index=snp_ids)


def driver_snps(weights: pd.Series, freq_target: pd.Series, freq_ref: pd.Series,
                sd_ref: float, top: int = 20) -> pd.DataFrame:
    """Rank SNPs by their contribution to the mean PRS difference (target − reference).

    All Series are indexed by the same SNP id; ``freq_*`` are effect allele frequencies.
    """
    df = pd.DataFrame({"weight": weights, "eaf_target": freq_target, "eaf_reference": freq_ref}).dropna()
    df["contribution"] = 2 * df.weight * (df.eaf_target - df.eaf_reference)
    df["contribution_ref_sd"] = df.contribution / sd_ref
    total = df.contribution.sum()
    df["share_of_shift"] = df.contribution / total if total else np.nan
    df = df.reindex(df.contribution.abs().sort_values(ascending=False).index)
    df.attrs["total_shift"] = float(total)
    df.attrs["total_shift_ref_sd"] = float(total / sd_ref)
    return df.head(top) if top else df
