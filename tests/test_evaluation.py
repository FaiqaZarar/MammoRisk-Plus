"""Tests for the shared evaluation library. Each metric is checked against an independent reference
(a slow textbook formula, scikit-learn, or lifelines) on small synthetic data."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from evaluation import bootstrap, classification as C, survival as S, xai as X


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    a = y * 1.0 + rng.normal(0, 1.0, 300)
    b = y * 0.5 + rng.normal(0, 1.0, 300)
    return y, a, b


# ---------------- DeLong -------------------------------------------------------------------
def _slow_delong_var(y, s):
    """Textbook O(n1·n0) DeLong variance."""
    pos, neg = s[y == 1], s[y == 0]
    psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    v10, v01 = psi.mean(1), psi.mean(0)
    return v10.var(ddof=1) / len(pos) + v01.var(ddof=1) / len(neg), psi.mean()


def test_auc_matches_sklearn_and_slow_variance(binary_data):
    y, a, _ = binary_data
    a = np.round(a, 1)  # force ties
    ci = C.auc_delong(y, a)
    assert ci.estimate == pytest.approx(roc_auc_score(y, a))
    var_slow, auc_slow = _slow_delong_var(y, a)
    _, v10, v01 = C._delong_components(y, a)
    assert C._delong_cov(v10, v01)[0, 0] == pytest.approx(var_slow)
    assert ci.lower < ci.estimate < ci.upper


def test_delong_identical_models_p_is_one(binary_data):
    y, a, _ = binary_data
    assert C.delong_test(y, a, a).p_value == 1.0


def test_delong_detects_better_model(binary_data):
    y, a, b = binary_data
    t = C.delong_test(y, a, b)
    assert t.difference > 0 and t.p_value < 0.05


def test_youden_and_threshold_metrics():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = C.youden_threshold(y, s)
    m = C.threshold_metrics(y, s, thr)
    assert (m.sensitivity, m.specificity, m.f1) == (1.0, 1.0, 1.0)
    m2 = C.threshold_metrics(y, s, 0.75)
    assert m2.tp == 2 and m2.fn == 1 and m2.ppv == 1.0


# ---------------- bootstrap ----------------------------------------------------------------
def test_bootstrap_stratified_keeps_group_sizes():
    groups = np.array(["a"] * 5 + ["b"] * 95)
    rng = np.random.default_rng(1)
    idx = bootstrap._resample_indices(100, rng, groups, None)
    assert (groups[idx] == "a").sum() == 5


def test_bootstrap_clustered_resamples_whole_clusters():
    clusters = np.repeat(np.arange(10), 3)
    idx = bootstrap._resample_indices(30, np.random.default_rng(2), None, clusters)
    counts = np.bincount(clusters[idx], minlength=10)
    assert set(np.unique(counts)) <= {0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30}


def test_bootstrap_ci_covers_mean():
    x = np.random.default_rng(3).normal(5, 1, 500)
    ci = bootstrap.bootstrap_ci(np.mean, x, n_boot=300)
    assert ci.lower < 5 < ci.upper


# ---------------- survival -----------------------------------------------------------------
@pytest.fixture
def surv_data():
    rng = np.random.default_rng(4)
    n = 400
    risk = rng.normal(size=n)
    t_event = rng.exponential(8 / np.exp(risk))
    t_cens = rng.uniform(1, 7, n)
    time = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(int)
    return time, event, risk


def test_km_matches_lifelines(surv_data):
    from lifelines import KaplanMeierFitter
    time, event, _ = surv_data
    kmf = KaplanMeierFitter().fit(time, event)
    for t in (1, 2.5, 5):
        assert S.kaplan_meier(time, event, t) == pytest.approx(float(kmf.predict(t)), abs=1e-9)


def test_harrell_matches_lifelines(surv_data):
    from lifelines.utils import concordance_index
    time, event, risk = surv_data
    # lifelines expects higher score = longer survival, so pass −risk
    assert S.c_index_harrell(time, event, risk) == pytest.approx(
        concordance_index(time, -risk, event), abs=1e-6)


def test_uno_and_td_auc_informative(surv_data):
    time, event, risk = surv_data
    assert S.c_index_ipcw(time, event, risk, tau=5) > 0.6
    assert S.td_auc(time, event, risk, 3) > 0.6
    assert 0.4 < S.c_index_ipcw(time, event, np.random.default_rng(9).normal(size=len(risk))) < 0.6


def test_calibration_perfect_in_the_large(surv_data):
    time, event, _ = surv_data
    obs = 1 - S.kaplan_meier(time, event, 3)
    cal = S.calibration_at(time, event, np.full(len(time), obs) + np.linspace(-1e-6, 1e-6, len(time)), 3)
    assert cal.oe_ratio == pytest.approx(1.0, rel=1e-3)


def test_subgroup_table_flags_small_groups(surv_data):
    import pandas as pd
    time, event, risk = surv_data
    df = pd.DataFrame({"t": time, "e": event, "r": risk, "pid": np.arange(len(time)) // 2,
                       "grp": ["big"] * (len(time) - 10) + ["tiny"] * 10})
    df.loc[df.grp == "tiny", "e"] = 0
    tab = S.subgroup_table(df, "grp", "t", "e", {1.0: "r", 5.0: "r"}, cluster_col="pid", n_boot=30)
    assert bool(tab.set_index("grp").loc["tiny", "exploratory_only"])
    assert tab.set_index("grp").loc["big", "c_index_ipcw_estimate"] > 0.6


# ---------------- XAI ----------------------------------------------------------------------
def test_xai_scores():
    mask = X.box_to_mask((100, 100), [(40, 40, 60, 60)])
    h = np.zeros((100, 100)); h[50, 50] = 1.0
    assert X.pointing_game(h, mask)
    e, base = X.energy_in_box(h, mask)
    assert e == 1.0 and base == pytest.approx(0.04)
    h_off = np.zeros((100, 100)); h_off[5, 5] = 1.0
    assert not X.pointing_game(h_off, mask)
    assert X.pointing_game(h_off, X.box_to_mask((100, 100), [(8, 8, 20, 20)]), tol=5)
    hm = mask.astype(float) + 0.01 * np.random.default_rng(0).random((100, 100))
    iou, thr = X.iou_top_fraction(hm, mask, top_fraction=0.04)
    assert iou > 0.9
    assert X.heatmap_similarity(hm, hm) == pytest.approx(1.0)


def test_flat_heatmap_is_not_a_hit():
    mask = X.box_to_mask((10, 10), [(0, 0, 5, 5)])
    assert not X.pointing_game(np.ones((10, 10)), mask)


def test_iou_with_a_mostly_zero_heatmap_does_not_select_the_whole_image():
    # Regression test (found in the CPU smoke test, 2026-11-12). A ReLU'd Grad-CAM is mostly exact zeros. The
    # old rule "h >= median" then had threshold 0 and selected EVERY pixel, so IoU collapsed to the lesion's
    # area fraction (here 4%) however well the heatmap localised the lesion.
    mask = X.box_to_mask((100, 100), [(40, 40, 60, 60)])           # lesion = 4% of the image
    h = np.zeros((100, 100)); h[42:58, 42:58] = 1.0                 # heatmap fires inside the lesion only; 97.4% zeros
    iou, thr = X.iou_top_fraction(h, mask, top_fraction=0.5)
    assert iou == pytest.approx(256 / 400)                          # 16x16 hit inside the 20x20 box
    assert iou > 0.5 > mask.mean()                                  # clearly better than the area baseline
    assert thr == 0.0
    # a heatmap that fires entirely OUTSIDE the lesion must score 0, not the baseline
    h_off = np.zeros((100, 100)); h_off[5:20, 5:20] = 1.0
    assert X.iou_top_fraction(h_off, mask, top_fraction=0.5)[0] == 0.0


def test_iou_of_an_all_zero_heatmap_is_zero():
    mask = X.box_to_mask((50, 50), [(10, 10, 20, 20)])
    assert X.iou_top_fraction(np.zeros((50, 50)), mask)[0] == 0.0


def test_iou_continuous_heatmap_still_uses_the_top_half_of_pixels():
    rng = np.random.default_rng(1)
    h = rng.random((100, 100))                                      # no ties
    mask = X.box_to_mask((100, 100), [(0, 0, 50, 100)])
    iou, thr = X.iou_top_fraction(h, mask, top_fraction=0.5)
    sel = (h >= thr).mean()
    assert 0.49 < sel < 0.51 and 0.3 < iou < 0.37                   # chance level for a half-image mask (~1/3)
    # the selected fraction is reported so it can be stated alongside the threshold
    assert X.score_all(h, mask)["iou_selected_fraction"] == pytest.approx(sel, abs=0.01)
