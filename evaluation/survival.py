"""Time-to-event metrics for Module 1 (Mirai external validation on EMBED, O4).

Theory
------
Each screening exam has a follow-up time T (years from exam to cancer diagnosis, or to the last
follow-up if no cancer) and an event flag E (1 = pathology-confirmed cancer, 0 = censored).
Censoring means we only know that cancer did not happen *before* T.

**C-index (Harrell):** among comparable pairs (the one with the shorter time had the event), the
fraction where the exam with the earlier cancer got the higher risk. With heavy censoring, Harrell's C
depends on the censoring pattern.

**IPCW C-index (Uno et al., 2011):** weights each comparable pair by 1/Ĝ(T_i)², where Ĝ is the
Kaplan–Meier estimate of the *censoring* survival function. This removes the dependence on
censoring. It is truncated at a horizon τ (we use τ = 5 years, Mirai's longest horizon). We use the
scikit-survival implementation (``concordance_index_ipcw``).

**Time-dependent AUC at t (cumulative/dynamic):** cases are exams with cancer by t (T ≤ t, E = 1),
controls are exams still cancer-free after t (T > t). It answers: "does the t-year risk rank women
who get cancer within t years above those who don't?" It is also IPCW-weighted
(``cumulative_dynamic_auc``). For Mirai we score the t-year AUC with Mirai's own t-year prediction.

**Calibration at t:** group exams by predicted t-year risk (deciles), and compare the mean predicted
risk with the observed risk 1 − KM(t) in each group. The overall observed/expected (O/E) ratio
summarises calibration-in-the-large.

**Clustering:** EMBED women have several screening exams. Exams from the same woman are not
independent, so bootstrap CIs resample *patients* (``clusters=patient_id``), not exams.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.bootstrap import CI, bootstrap_ci


def _surv_array(event, time):
    from sksurv.util import Surv
    return Surv.from_arrays(event=np.asarray(event).astype(bool), time=np.asarray(time, float))


# ----------------------------------------------------------------------------------------------
# Kaplan–Meier (small own implementation: transparent, no extra dependency)
# ----------------------------------------------------------------------------------------------
def kaplan_meier(time, event, at: float) -> float:
    """KM estimate of S(at) = P(T > at).

    S(t) = Π_{t_i ≤ t} (1 − d_i / n_i), with d_i events and n_i at risk at each distinct event time.
    """
    time, event = np.asarray(time, float), np.asarray(event).astype(bool)
    s = 1.0
    for t in np.unique(time[event]):
        if t > at:
            break
        n_at_risk = np.sum(time >= t)
        d = np.sum((time == t) & event)
        if n_at_risk > 0:
            s *= 1.0 - d / n_at_risk
    return float(s)


# ----------------------------------------------------------------------------------------------
# Discrimination
# ----------------------------------------------------------------------------------------------
def c_index_ipcw(time, event, risk, tau: float = 5.0) -> float:
    """Uno's IPCW C-index truncated at ``tau`` years.

    The censoring distribution is estimated on the same (external) cohort, because we do not use
    Mirai's training data.
    """
    from sksurv.metrics import concordance_index_ipcw
    y = _surv_array(event, time)
    return float(concordance_index_ipcw(y, y, np.asarray(risk, float), tau=tau)[0])


def c_index_harrell(time, event, risk) -> float:
    """Harrell's C, reported alongside Uno's for comparison with older papers."""
    from sksurv.metrics import concordance_index_censored
    return float(concordance_index_censored(np.asarray(event).astype(bool),
                                            np.asarray(time, float), np.asarray(risk, float))[0])


def td_auc(time, event, risk_at_t: float | np.ndarray, t: float) -> float:
    """Cumulative/dynamic IPCW AUC at horizon ``t`` years for one risk vector."""
    from sksurv.metrics import cumulative_dynamic_auc
    y = _surv_array(event, time)
    auc, _ = cumulative_dynamic_auc(y, y, np.asarray(risk_at_t, float), [t])
    return float(auc[0])


def c_index_ipcw_ci(time, event, risk, tau=5.0, clusters=None, n_boot=1000, seed=42) -> CI:
    return bootstrap_ci(lambda t, e, r: c_index_ipcw(t, e, r, tau), time, event, risk,
                        n_boot=n_boot, seed=seed, clusters=clusters)


def td_auc_ci(time, event, risk_at_t, t, clusters=None, n_boot=1000, seed=42) -> CI:
    return bootstrap_ci(lambda tt, e, r: td_auc(tt, e, r, t), time, event, risk_at_t,
                        n_boot=n_boot, seed=seed, clusters=clusters)


# ----------------------------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Calibration:
    horizon: float
    table: pd.DataFrame      # bin, n, events, mean_predicted, observed (1 − KM)
    expected: float          # mean predicted risk
    observed: float          # 1 − KM(horizon) in the whole cohort
    oe_ratio: float          # observed / expected; 1 = calibrated in the large


def calibration_at(time, event, pred_risk_t, horizon: float, n_bins: int = 10) -> Calibration:
    """Decile calibration of t-year predicted risk against Kaplan–Meier observed risk."""
    time, event = np.asarray(time, float), np.asarray(event).astype(int)
    pred = np.asarray(pred_risk_t, float)
    # Quantile bins; duplicates="drop" handles many identical predictions.
    bins = pd.qcut(pred, q=n_bins, labels=False, duplicates="drop")
    rows = []
    for b in np.unique(bins):
        m = bins == b
        rows.append({"bin": int(b), "n": int(m.sum()), "events": int(event[m].sum()),
                     "mean_predicted": float(pred[m].mean()),
                     "observed": 1.0 - kaplan_meier(time[m], event[m], horizon)})
    expected = float(pred.mean())
    observed = 1.0 - kaplan_meier(time, event, horizon)
    oe = observed / expected if expected > 0 else float("nan")
    return Calibration(horizon, pd.DataFrame(rows), expected, observed, oe)


# ----------------------------------------------------------------------------------------------
# Subgroups (density, race, vendor)
# ----------------------------------------------------------------------------------------------
def subgroup_table(df: pd.DataFrame, by: str, time_col: str, event_col: str,
                   risk_cols: dict[float, str], tau: float = 5.0, cluster_col: str | None = None,
                   n_boot: int = 1000, seed: int = 42, min_events: int = 5) -> pd.DataFrame:
    """C-index and t-year AUCs with 95% CI for each level of ``by``.

    ``risk_cols`` maps horizon (years) → column holding Mirai's risk for that horizon; the C-index
    uses the risk at ``tau``. Subgroups with fewer than ``min_events`` cancers are reported with
    counts only and flagged, because a CI from 2–3 events is meaningless (CLAUDE.md: EMBED results
    may be exploratory).
    """
    rows = []
    for level, g in df.groupby(by, dropna=False):
        n_events = int(g[event_col].sum())
        row = {by: level, "n_exams": len(g), "n_events": n_events,
               "n_patients": g[cluster_col].nunique() if cluster_col else np.nan,
               "exploratory_only": n_events < min_events}
        if n_events >= min_events:
            clusters = g[cluster_col].to_numpy() if cluster_col else None
            for name, fn in [("c_index_ipcw", None)] + [(f"auc_{t:g}y", t) for t in risk_cols]:
                try:
                    if fn is None:
                        ci = c_index_ipcw_ci(g[time_col], g[event_col], g[risk_cols[tau]], tau,
                                             clusters, n_boot, seed)
                    else:
                        ci = td_auc_ci(g[time_col], g[event_col], g[risk_cols[fn]], fn, clusters,
                                       n_boot, seed)
                    row.update(ci.as_dict(prefix=f"{name}_"))
                except ValueError as err:  # e.g. no case reaches horizon t in this subgroup
                    row[f"{name}_error"] = str(err)
        rows.append(row)
    return pd.DataFrame(rows)
