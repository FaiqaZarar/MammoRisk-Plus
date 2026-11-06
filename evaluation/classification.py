"""Binary classification metrics for Module 3 (benign vs malignant).

Theory
------
**AUC** is the probability that a randomly chosen malignant image gets a higher score than a randomly
chosen benign one (the Mann–Whitney U statistic divided by n1*n0).

**DeLong's method** (DeLong et al., 1988) gives the variance of that empirical AUC without
resampling. Each case gets a *structural component*:
  V10_i = fraction of negatives that positive i out-scores (ties count 1/2)
  V01_j = fraction of positives that out-score negative j
  Var(AUC) = S10/n1 + S01/n0, with S10, S01 the sample (co)variances of those components.
For two models scored on the **same** test cases, the components are correlated. The covariance
term makes the paired test much more powerful than comparing two independent CIs. This is the
test CLAUDE.md asks for (ResNet-50 vs EfficientNet-B3 on the same test set).

We use the fast O(n log n) midrank algorithm of Sun & Xu (2014), *IEEE Signal Process. Lett.*
21(11):1389–1393.

**Threshold:** chosen once on *validation* with Youden's J = sensitivity + specificity − 1, then
applied unchanged to test (CLAUDE.md "no leakage").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve

from evaluation.bootstrap import CI


# ----------------------------------------------------------------------------------------------
# DeLong
# ----------------------------------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, with ties given the average of their ranks."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks_sorted = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks_sorted[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def _delong_components(y_true: np.ndarray, scores: np.ndarray):
    """scores: (k_models, n_cases). Returns aucs (k,), V10 (k, n1), V01 (k, n0)."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.atleast_2d(np.asarray(scores, dtype=float))
    pos, neg = scores[:, y_true == 1], scores[:, y_true == 0]
    m, n = pos.shape[1], neg.shape[1]
    if m == 0 or n == 0:
        raise ValueError("AUC needs at least one positive and one negative case.")
    k = scores.shape[0]
    aucs, v10, v01 = np.empty(k), np.empty((k, m)), np.empty((k, n))
    for r in range(k):
        tx, ty = _midrank(pos[r]), _midrank(neg[r])
        tz = _midrank(np.concatenate([pos[r], neg[r]]))
        aucs[r] = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        v10[r] = (tz[:m] - tx) / n
        v01[r] = 1.0 - (tz[m:] - ty) / m
    return aucs, v10, v01


def _delong_cov(v10: np.ndarray, v01: np.ndarray) -> np.ndarray:
    m, n = v10.shape[1], v01.shape[1]
    s10 = np.atleast_2d(np.cov(v10))
    s01 = np.atleast_2d(np.cov(v01))
    return s10 / m + s01 / n


def auc_delong(y_true, y_score, level: float = 0.95) -> CI:
    """Empirical AUC with a DeLong confidence interval.

    The CI is computed on the logit scale and back-transformed, so it stays inside (0, 1) even
    when the AUC is close to 1. With AUC = A and SE = s, the logit SE is s / (A(1−A)) (delta method).
    """
    aucs, v10, v01 = _delong_components(y_true, y_score)
    auc = float(aucs[0])
    var = float(_delong_cov(v10, v01)[0, 0])
    z = stats.norm.ppf(1 - (1 - level) / 2)
    if auc in (0.0, 1.0) or var <= 0:
        return CI(auc, auc, auc, level)
    logit = np.log(auc / (1 - auc))
    se_logit = np.sqrt(var) / (auc * (1 - auc))
    lo, hi = logit - z * se_logit, logit + z * se_logit
    return CI(auc, float(1 / (1 + np.exp(-lo))), float(1 / (1 + np.exp(-hi))), level)


@dataclass(frozen=True)
class DeLongTest:
    auc_a: float
    auc_b: float
    difference: float  # auc_a − auc_b
    se: float
    z: float
    p_value: float     # two-sided

    def __str__(self) -> str:
        return (f"AUC A={self.auc_a:.3f}, B={self.auc_b:.3f}, Δ={self.difference:+.3f} "
                f"(SE {self.se:.3f}), z={self.z:.2f}, p={self.p_value:.4f}")


def delong_test(y_true, score_a, score_b) -> DeLongTest:
    """Paired two-sided DeLong test of H0: AUC_A = AUC_B on the same cases.

    z = (A − B) / sqrt(Var A + Var B − 2 Cov(A, B))
    """
    aucs, v10, v01 = _delong_components(y_true, np.vstack([score_a, score_b]))
    cov = _delong_cov(v10, v01)
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var <= 0:
        return DeLongTest(float(aucs[0]), float(aucs[1]), diff, 0.0, 0.0, 1.0)
    se = float(np.sqrt(var))
    z = diff / se
    p = float(2 * stats.norm.sf(abs(z)))
    return DeLongTest(float(aucs[0]), float(aucs[1]), diff, se, float(z), p)


# ----------------------------------------------------------------------------------------------
# Thresholds
# ----------------------------------------------------------------------------------------------
def youden_threshold(y_true, y_score) -> float:
    """Threshold maximising Youden's J = TPR − FPR. Call this on VALIDATION data only."""
    fpr, tpr, thr = roc_curve(np.asarray(y_true), np.asarray(y_score))
    j = tpr - fpr
    best = int(np.argmax(j))
    # roc_curve's first threshold is +inf (predict nothing positive); never return that.
    return float(thr[best]) if np.isfinite(thr[best]) else float(thr[1])


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int

    @staticmethod
    def _div(a, b):
        return float(a / b) if b else float("nan")

    @property
    def sensitivity(self): return self._div(self.tp, self.tp + self.fn)
    @property
    def specificity(self): return self._div(self.tn, self.tn + self.fp)
    @property
    def ppv(self): return self._div(self.tp, self.tp + self.fp)
    @property
    def npv(self): return self._div(self.tn, self.tn + self.fn)

    @property
    def f1(self):
        return self._div(2 * self.tp, 2 * self.tp + self.fp + self.fn)

    def as_dict(self) -> dict:
        return {"threshold": self.threshold, "tp": self.tp, "fp": self.fp, "tn": self.tn,
                "fn": self.fn, "sensitivity": self.sensitivity, "specificity": self.specificity,
                "ppv": self.ppv, "npv": self.npv, "f1": self.f1}


def threshold_metrics(y_true, y_score, threshold: float) -> ThresholdMetrics:
    """Confusion-matrix metrics with ``score >= threshold`` predicted malignant."""
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_score) >= threshold).astype(int)
    return ThresholdMetrics(float(threshold), int(((pred == 1) & (y == 1)).sum()),
                            int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 0)).sum()),
                            int(((pred == 0) & (y == 1)).sum()))
