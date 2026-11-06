"""Standardise raw PRS against a reference population and assign risk categories.

Theory
------
A raw PRS has no meaning by itself. It is interpreted relative to a *reference distribution*:

    z_i = (PRS_i − μ_ref) / σ_ref

and its percentile is read off the **fitted normal** distribution, Φ(z), not by counting people in
the reference (CLAUDE.md). With ~100 CEU or ~280 Pakistani people, an empirical 99th percentile
rests on 1–3 people. The normal fit uses all of them through μ and σ. PRS-313 is a sum of 313
small independent effects, so it is close to normal (central limit theorem). ``normality_check``
reports a Shapiro–Wilk test so that this assumption is visible.

The same person can land in different categories under the European (CEU) and the Pakistani
reference. That shift is O1.

Risk categories (``categories`` in the config) are defined on the reference percentile scale.
PROPOSED default, pending Pukhraj / supervisor (DECISIONS D-14):
    low < 20th · average 20th–80th · elevated 80th–99th · high ≥ 99th (top 1%)
These follow the percentile bands Mavaddat et al. (2019) report risks for. We make no absolute-risk
claim.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_CATEGORIES = {"low": 0.0, "average": 0.20, "elevated": 0.80, "high": 0.99}


@dataclass(frozen=True)
class Reference:
    name: str
    mean: float
    sd: float
    n: int

    def z(self, prs) -> np.ndarray:
        return (np.asarray(prs, float) - self.mean) / self.sd

    def percentile(self, prs) -> np.ndarray:
        """Percentile (0–1) under the fitted normal N(mean, sd²)."""
        return stats.norm.cdf(self.z(prs))


def fit_reference(name: str, prs) -> Reference:
    x = np.asarray(prs, float)
    if x.size < 2:
        raise ValueError(f"reference '{name}' needs ≥2 people")
    return Reference(name, float(x.mean()), float(x.std(ddof=1)), int(x.size))


def categorise(percentile, categories: dict[str, float] = DEFAULT_CATEGORIES) -> np.ndarray:
    """Map percentiles (0–1) to category labels. ``categories``: label → lower bound."""
    items = sorted(categories.items(), key=lambda kv: kv[1])
    labels = np.array([k for k, _ in items])
    bounds = np.array([v for _, v in items])
    idx = np.searchsorted(bounds, np.asarray(percentile, float), side="right") - 1
    return labels[np.clip(idx, 0, len(labels) - 1)]


def category_order(categories: dict[str, float] = DEFAULT_CATEGORIES) -> list[str]:
    return [k for k, _ in sorted(categories.items(), key=lambda kv: kv[1])]


def normality_check(prs) -> dict:
    x = np.asarray(prs, float)
    w, p = stats.shapiro(x) if 3 <= x.size <= 5000 else (np.nan, np.nan)
    return {"n": int(x.size), "shapiro_W": float(w), "shapiro_p": float(p),
            "skew": float(stats.skew(x)), "excess_kurtosis": float(stats.kurtosis(x))}


def standardise_all(df: pd.DataFrame, refs: dict[str, Reference],
                    categories: dict[str, float] = DEFAULT_CATEGORIES) -> pd.DataFrame:
    """Add z_<ref>, pct_<ref>, cat_<ref> columns for every reference."""
    out = df.copy()
    for name, ref in refs.items():
        out[f"z_{name}"] = ref.z(out["prs"])
        out[f"pct_{name}"] = ref.percentile(out["prs"])
        out[f"cat_{name}"] = categorise(out[f"pct_{name}"], categories)
    return out
