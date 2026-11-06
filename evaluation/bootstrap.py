"""Percentile bootstrap confidence intervals.

Theory
------
The bootstrap estimates the sampling distribution of a statistic by resampling the observed data
*with replacement* B times and recomputing the statistic each time. The 95% percentile CI is the
2.5th and 97.5th percentile of the B recomputed values.

Two variants matter for this project:

* **Stratified** (``strata``): resample *within* each stratum, keeping stratum sizes fixed. Module 2
  stratifies by population group so that a small HGDP group (~20 people) is never missing from a
  resample (CLAUDE.md: "bootstrap 95% CI, 1,000 resamples, stratified by group").
* **Clustered** (``clusters``): resample whole clusters (e.g. patients), not rows. Module 3 has
  several images per patient; resampling images would treat correlated images as independent and
  make the CI too narrow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class CI:
    estimate: float
    lower: float
    upper: float
    level: float = 0.95
    n_boot: int = 0
    n_failed: int = 0  # resamples where the statistic was undefined (e.g. one class only)

    def __str__(self) -> str:  # "0.812 (0.775–0.846)"
        return f"{self.estimate:.3f} ({self.lower:.3f}–{self.upper:.3f})"

    def as_dict(self, prefix: str = "") -> dict:
        return {f"{prefix}estimate": self.estimate, f"{prefix}ci_lower": self.lower,
                f"{prefix}ci_upper": self.upper, f"{prefix}ci_level": self.level}


def _resample_indices(n: int, rng: np.random.Generator, strata: np.ndarray | None,
                      clusters: np.ndarray | None) -> np.ndarray:
    if strata is not None and clusters is not None:
        raise ValueError("Use either strata or clusters, not both.")
    if strata is not None:
        parts = []
        for s in np.unique(strata):
            idx = np.flatnonzero(strata == s)
            parts.append(rng.choice(idx, size=idx.size, replace=True))
        return np.concatenate(parts)
    if clusters is not None:
        uniq, inverse = np.unique(clusters, return_inverse=True)
        members = [np.flatnonzero(inverse == k) for k in range(uniq.size)]
        picked = rng.integers(0, uniq.size, size=uniq.size)
        return np.concatenate([members[k] for k in picked])
    return rng.integers(0, n, size=n)


def bootstrap_ci(statistic: Callable[..., float], *arrays: Sequence, n_boot: int = 1000,
                 level: float = 0.95, seed: int = 42, strata: Sequence | None = None,
                 clusters: Sequence | None = None) -> CI:
    """Percentile bootstrap CI for ``statistic(*arrays)``.

    All ``arrays`` are resampled with the same row indices, so paired data stay paired.
    Resamples where the statistic raises ``ValueError`` or returns NaN (e.g. AUC on a resample
    containing only one class) are skipped and counted in ``n_failed``.
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("All arrays must have the same length.")
    strata_arr = None if strata is None else np.asarray(strata)
    cluster_arr = None if clusters is None else np.asarray(clusters)

    estimate = float(statistic(*arrays))
    rng = np.random.default_rng(seed)
    stats, failed = [], 0
    for _ in range(n_boot):
        idx = _resample_indices(n, rng, strata_arr, cluster_arr)
        try:
            v = float(statistic(*(a[idx] for a in arrays)))
        except ValueError:
            failed += 1
            continue
        if np.isnan(v):
            failed += 1
            continue
        stats.append(v)
    if not stats:
        raise ValueError("Statistic undefined in every bootstrap resample.")
    alpha = (1 - level) / 2
    lo, hi = np.quantile(stats, [alpha, 1 - alpha])
    return CI(estimate, float(lo), float(hi), level, n_boot, failed)
