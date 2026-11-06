"""O1 core result: how many Pakistani individuals change PRS risk category when the reference
switches from European (CEU) to Pakistani.

Design
------
* Input: one row per person with ``IID``, ``prs``, ``group``. Groups are CEU plus the 9
  Pakistani groups: 1000G PJL + 8 HGDP groups (Balochi, Brahui, Burusho, Hazara, Kalash, Makrani,
  Pathan, Sindhi).
* References: ``european`` = CEU. ``pakistani`` = all 9 groups pooled (~280). Sensitivity
  analysis: each group against its **own** group mean/SD (``own_group``).
* Targets: the Pakistani individuals. Each person gets a category under each reference.
* Note: the pooled Pakistani reference contains the targets themselves, so under it the pooled
  sample has mean z = 0 by construction. The informative quantity is how far the *CEU* reference
  moves them, i.e. the reclassification.

Uncertainty
-----------
The reference mean and SD are themselves estimated from ~99 CEU and ~280 Pakistani people. That
uncertainty matters as much as the target sampling. So each bootstrap resample re-draws **everyone**
(CEU and Pakistani, stratified by group so no group vanishes), **re-fits both references**, and
recomputes all categories. 1,000 resamples, percentile 95% CI (CLAUDE.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.bootstrap import CI, _resample_indices
from module2_prs.standardise import (DEFAULT_CATEGORIES, categorise, category_order,
                                     fit_reference)

EUR, PAK = "european", "pakistani"


def _categories_under_refs(prs: np.ndarray, groups: np.ndarray, ceu_label: str,
                           pak_groups: list[str], categories: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    is_ceu = groups == ceu_label
    is_pak = np.isin(groups, pak_groups)
    ref_eur = fit_reference(EUR, prs[is_ceu])
    ref_pak = fit_reference(PAK, prs[is_pak])
    cat_eur = categorise(ref_eur.percentile(prs[is_pak]), categories)
    cat_pak = categorise(ref_pak.percentile(prs[is_pak]), categories)
    return cat_eur, cat_pak, groups[is_pak]


def _summary(cat_eur, cat_pak, grp, order) -> dict:
    rank = {c: i for i, c in enumerate(order)}
    r_eur = np.vectorize(rank.get)(cat_eur)
    r_pak = np.vectorize(rank.get)(cat_pak)
    out = {"pooled": float(np.mean(cat_eur != cat_pak)),
           # "up": higher category under the European reference than under the Pakistani one
           "pooled_up_under_european": float(np.mean(r_eur > r_pak)),
           "pooled_down_under_european": float(np.mean(r_eur < r_pak)),
           "pooled_top1_european": float(np.mean(cat_eur == order[-1])),
           "pooled_top1_pakistani": float(np.mean(cat_pak == order[-1]))}
    for g in np.unique(grp):
        m = grp == g
        out[f"group:{g}"] = float(np.mean(cat_eur[m] != cat_pak[m]))
    return out


def reclassification(df: pd.DataFrame, pak_groups: list[str], ceu_label: str = "CEU",
                     categories: dict[str, float] = DEFAULT_CATEGORIES, n_boot: int = 1000,
                     seed: int = 42) -> dict:
    """Point estimates + stratified bootstrap 95% CIs (references re-fitted in each resample)."""
    keep = df["group"].isin([ceu_label, *pak_groups])
    prs = df.loc[keep, "prs"].to_numpy(float)
    groups = df.loc[keep, "group"].astype(str).to_numpy()
    missing = set(pak_groups) - set(groups)
    if ceu_label not in groups or missing:
        raise ValueError(f"missing groups: {sorted(missing | ({ceu_label} - set(groups)))}")
    order = category_order(categories)

    cat_eur, cat_pak, grp = _categories_under_refs(prs, groups, ceu_label, pak_groups, categories)
    point = _summary(cat_eur, cat_pak, grp, order)
    crosstab = pd.crosstab(pd.Categorical(cat_eur, order), pd.Categorical(cat_pak, order),
                           rownames=["european_ref"], colnames=["pakistani_ref"], dropna=False)

    rng = np.random.default_rng(seed)
    boots = {k: [] for k in point}
    for _ in range(n_boot):
        idx = _resample_indices(len(prs), rng, groups, None)
        s = _summary(*_categories_under_refs(prs[idx], groups[idx], ceu_label, pak_groups,
                                             categories), order)
        for k in boots:
            boots[k].append(s[k])
    cis = {k: CI(point[k], float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)), 0.95, n_boot)
           for k, v in boots.items()}

    group_n = pd.Series(grp).value_counts().to_dict()
    rows = [{"metric": k, "n": len(grp) if not k.startswith("group:") else group_n[k[6:]],
             **cis[k].as_dict()} for k in point]
    table = pd.DataFrame(rows)
    # Top-1% shares are reported for the pooled sample only (CLAUDE.md).
    return {"table": table, "crosstab": crosstab, "categories": categories, "n_boot": n_boot,
            "seed": seed}


def own_group_sensitivity(df: pd.DataFrame, pak_groups: list[str], ceu_label: str = "CEU",
                          categories: dict[str, float] = DEFAULT_CATEGORIES) -> pd.DataFrame:
    """Sensitivity: each group standardised against itself vs against CEU (descriptive, no CI;
    groups of ~20–25 are too small for stable SDs)."""
    ceu = fit_reference(EUR, df.loc[df.group == ceu_label, "prs"])
    rows = []
    for g in pak_groups:
        x = df.loc[df.group == g, "prs"].to_numpy(float)
        own = fit_reference(g, x)
        c_eur, c_own = categorise(ceu.percentile(x), categories), categorise(own.percentile(x), categories)
        rows.append({"group": g, "n": len(x), "mean_z_vs_ceu": float(ceu.z(x).mean()),
                     "sd_ratio_vs_ceu": own.sd / ceu.sd,
                     "pct_reclassified_own_vs_ceu": float(np.mean(c_eur != c_own))})
    return pd.DataFrame(rows)


def group_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """Per-group raw PRS summary (mean, SD, quartiles) for the distribution figure."""
    return (df.groupby("group")["prs"]
              .describe()[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
              .reset_index())
