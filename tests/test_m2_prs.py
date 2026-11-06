"""Module 2 tests on the real PGS000004 file (committed) and synthetic genotypes."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from module2_prs import drivers, harmonise as H, reclassify as R, score, standardise as S, weights

WEIGHTS = Path("data/prs/PGS000004.txt.gz")


# ---------------- weights -------------------------------------------------------------------
def test_load_pgs000004():
    sf = weights.load_scoring_file(WEIGHTS)
    assert sf.pgs_id == "PGS000004" and sf.n_variants == 313
    assert sf.build == "GRCh37" and not sf.harmonised
    assert set(sf.variants.chr) == {str(i) for i in range(1, 23)}
    with pytest.raises(ValueError, match="GRCh37"):
        weights.check_build(sf, "GRCh38")
    weights.check_build(sf, "hg19")


# ---------------- harmonisation -------------------------------------------------------------
def _w(rows):
    return pd.DataFrame(rows, columns=["chr", "pos", "effect_allele", "other_allele", "weight", "eaf", "rsid"])


def test_harmonise_all_cases():
    w = _w([["1", 100, "A", "G", 0.1, 0.3, None],    # same, effect = ALT
            ["1", 200, "A", "G", 0.2, 0.3, None],    # same, effect = REF (swap)
            ["1", 300, "A", "G", 0.3, 0.3, None],    # strand flip: data T/C
            ["1", 400, "A", "T", 0.4, 0.3, None],    # palindromic
            ["1", 500, "C", "CAAA", 0.5, 0.3, None],  # indel
            ["1", 600, "A", "G", 0.6, 0.5, None],    # missing
            ["1", 700, "A", "G", 0.7, 0.3, None]])   # allele mismatch (data A/C)
    t = pd.DataFrame({"chr": ["1"] * 6, "pos": [100, 200, 300, 400, 500, 700],
                      "id": list("abcdef"), "ref": ["G", "A", "C", "A", "C", "A"],
                      "alt": ["A", "G", "T", "T", "CAAA", "C"]})
    h = H.harmonise(w, t, "keep").table.set_index("pos")
    assert h.loc[100, "status"] == "match" and h.loc[100, "scored_allele"] == "A"
    assert h.loc[200, "status"] == "match" and h.loc[200, "scored_allele"] == "A"
    assert h.loc[300, "status"] == "flip" and h.loc[300, "scored_allele"] == "T"
    assert h.loc[400, "status"] == "ambiguous_kept"
    assert h.loc[500, "status"] == "match"
    assert h.loc[600, "status"] == "missing"
    assert h.loc[700, "status"] == "allele_mismatch"
    hd = H.harmonise(w, t, "drop")
    assert (hd.table.set_index("pos").loc[400, "status"]) == "ambiguous_dropped"
    cov = hd.coverage()
    assert cov["n_scored"] == 4 and cov["n_weights"] == 7
    # missing offset = 2*eaf*beta for the 3 unscored (dropped, missing, mismatch)
    assert hd.missing_offset() == pytest.approx(2 * (0.3 * 0.4 + 0.5 * 0.6 + 0.3 * 0.7))


def test_write_plink_score(tmp_path):
    w = _w([["1", 100, "A", "G", 0.1, 0.3, None]])
    t = pd.DataFrame({"chr": ["1"], "pos": [100], "id": ["rs1"], "ref": ["G"], "alt": ["A"]})
    p = H.harmonise(w, t).write_plink_score(tmp_path / "s.score")
    assert p.read_text().splitlines() == ["ID\tA1\tBETA", "rs1\tA\t0.1"]


def test_plink2_command_uses_sums():
    cmd = score.plink2_score_command("geno", "s.score", "out")
    assert "cols=+scoresums" in cmd and cmd[:3] == ["plink2", "--pfile", "geno"]


def test_read_sscore(tmp_path):
    p = tmp_path / "x.sscore"
    p.write_text("#FID\tIID\tALLELE_CT\tNAMED_ALLELE_DOSAGE_SUM\tSCORE1_AVG\tSCORE1_SUM\n"
                 "f\ti1\t20\t5\t0.01\t0.2\n")
    df = score.read_sscore(p)
    assert df.loc[0, "prs"] == 0.2 and df.loc[0, "IID"] == "i1"


def test_numpy_score_imputes_with_eaf():
    d = np.array([[0, 2], [np.nan, 1]])
    s = score.numpy_score(d, np.array([1.0, 0.5]), eaf=np.array([0.25, 0.5]))
    assert s.tolist() == [1.0, 0.5 + 0.5]


# ---------------- standardise / categorise ---------------------------------------------------
def test_categorise_bounds():
    cats = S.categorise([0.0, 0.199, 0.2, 0.5, 0.8, 0.989, 0.99, 1.0])
    assert cats.tolist() == ["low", "low", "average", "average", "elevated", "elevated", "high", "high"]


def test_reference_percentile_from_normal():
    ref = S.Reference("r", 0.0, 1.0, 100)
    assert ref.percentile([2.326348])[0] == pytest.approx(0.99, abs=1e-6)


# ---------------- reclassification ----------------------------------------------------------
PAK = ["PJL", "Balochi", "Brahui", "Burusho", "Hazara", "Kalash", "Makrani", "Pathan", "Sindhi"]


def _synthetic_cohort(shift: float, seed=0):
    rng = np.random.default_rng(seed)
    rows = [("CEU", x) for x in rng.normal(0, 1, 99)]
    sizes = {"PJL": 96, **{g: 22 for g in PAK[1:]}}
    for g, n in sizes.items():
        rows += [(g, x) for x in rng.normal(shift, 1, n)]
    df = pd.DataFrame(rows, columns=["group", "prs"])
    df["IID"] = [f"s{i}" for i in range(len(df))]
    return df


def test_no_shift_means_little_reclassification():
    res = R.reclassification(_synthetic_cohort(0.0), PAK, n_boot=100)
    pooled = res["table"].set_index("metric").loc["pooled"]
    assert pooled.estimate < 0.15
    assert pooled.ci_lower <= pooled.estimate <= pooled.ci_upper


def test_shift_causes_upward_reclassification_under_european_ref():
    res = R.reclassification(_synthetic_cohort(0.8), PAK, n_boot=100)
    t = res["table"].set_index("metric")
    assert t.loc["pooled", "estimate"] > 0.25
    assert t.loc["pooled_up_under_european", "estimate"] > t.loc["pooled_down_under_european", "estimate"]
    assert t.loc["pooled_top1_european", "estimate"] > t.loc["pooled_top1_pakistani", "estimate"]
    assert {f"group:{g}" for g in PAK} <= set(t.index)
    assert res["crosstab"].to_numpy().sum() == 96 + 8 * 22


def test_missing_group_raises():
    df = _synthetic_cohort(0.0)
    with pytest.raises(ValueError, match="Kalash"):
        R.reclassification(df[df.group != "Kalash"], PAK, n_boot=10)


def test_own_group_sensitivity_runs():
    t = R.own_group_sensitivity(_synthetic_cohort(0.5), PAK)
    assert len(t) == 9 and (t.mean_z_vs_ceu > 0).all()


# ---------------- drivers -------------------------------------------------------------------
def test_driver_decomposition_is_exact():
    rng = np.random.default_rng(1)
    m = 20
    w = pd.Series(rng.normal(0, 0.05, m), index=[f"v{i}" for i in range(m)])
    p_ref = pd.Series(rng.uniform(0.1, 0.9, m), index=w.index)
    p_tgt = (p_ref + rng.normal(0, 0.05, m)).clip(0.01, 0.99)
    d = drivers.driver_snps(w, p_tgt, p_ref, sd_ref=0.5, top=0)
    expected = float((2 * w * (p_tgt - p_ref)).sum())
    assert d.attrs["total_shift"] == pytest.approx(expected)
    assert d.share_of_shift.sum() == pytest.approx(1.0)
    assert d.contribution.abs().is_monotonic_decreasing
