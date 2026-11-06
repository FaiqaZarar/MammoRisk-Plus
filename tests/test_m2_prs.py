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


# ---------------- end-to-end analyse step ---------------------------------------------------
def test_run_analyse_end_to_end(tmp_path):
    import json
    import yaml
    from module2_prs import run
    df = _synthetic_cohort(0.5)
    sscore = tmp_path / "x.sscore"
    pd.DataFrame({"#FID": df.IID, "IID": df.IID, "ALLELE_CT": 626, "SCORE1_SUM": df.prs}).to_csv(
        sscore, sep="\t", index=False)
    df[["IID", "group"]].to_csv(tmp_path / "sheet.csv", index=False)
    cfg = {"data_root": str(tmp_path), "results_dir": str(tmp_path / "out"), "n_boot": 20,
           "paths": {"weights": str(WEIGHTS.resolve()), "sscore": "x.sscore",
                     "sample_sheet": "sheet.csv"},
           "pakistani_groups": PAK, "genotype_build": "GRCh37"}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    run.main(["--config", str(tmp_path / "c.yaml"), "--step", "analyse"])
    out = tmp_path / "out"
    for f in ("reclassification.csv", "crosstab_european_vs_pakistani.csv", "per_person.csv",
              "own_group_sensitivity.csv", "reference_stats.json", "run_info_analyse.json"):
        assert (out / f).exists(), f
    stats = json.loads((out / "reference_stats.json").read_text())
    assert set(stats["references"]) == {"european", "pakistani"}


# ---------------- gnomAD sample sheet, extraction commands, drivers in analyse ----------------
def _fake_gnomad_meta(path):
    import gzip, json
    rows = []
    for i, (pop, project) in enumerate([("CEU", "1000 Genomes")] * 4 + [("PJL", "1000 Genomes")] * 3
                                       + [("Kalash", "HGDP")] * 3 + [("Yoruba", "HGDP")]):
        rows.append({
            "s": f"S{i}", "high_quality": i != 1,
            "relatedness_inference": json.dumps({"related": i == 2}),
            "gnomad_sex_imputation": json.dumps({"sex_karyotype": "XX" if i % 2 else "XY"}),
            "hgdp_tgp_meta": json.dumps({"population": pop, "project": project,
                                         "global_pca_scores": [0.1 * i, -0.1 * i],
                                         "subcontinental_pca": {"outlier": i == 9}})})
    with gzip.open(path, "wt") as fh:
        pd.DataFrame(rows).to_csv(fh, sep="\t", index=False)


def test_sample_sheet_from_gnomad(tmp_path):
    from module2_prs import data_prep
    p = tmp_path / "meta.tsv.bgz"
    _fake_gnomad_meta(p)
    kept, full = data_prep.sample_sheet_from_gnomad(p, ["CEU", "PJL", "Kalash"])
    assert "Yoruba" not in set(full.group)
    assert set(full.loc[full.IID == "S1", "drop_reason"]) == {"low_quality;"}
    assert set(full.loc[full.IID == "S2", "drop_reason"]) == {"related;"}
    assert len(kept) == 8 and {"global_PC1", "global_PC2"} <= set(kept.columns)
    assert kept.set_index("IID").loc["S9", "gnomad_subcont_outlier"]
    assert set(kept.source) == {"1000G", "HGDP"}


def test_extract_and_merge_commands():
    from module2_prs import data_prep
    ex = data_prep.bcftools_extract_commands("https://x/{chr}.vcf.bgz", "r.txt", "s.txt", "w", ["chr1"])
    assert ex[0][-1] == "https://x/chr1.vcf.bgz" and "PASS" in ex[0]
    mc = data_prep.merge_commands("w", ["chr1", "chr2"], "w/merged")
    assert mc[0][-2:] == [str(Path("w/prs313_chr1.vcf.gz")), str(Path("w/prs313_chr2.vcf.gz"))]
    assert "@:#:$r:$a" in mc[1]


def test_read_export_a(tmp_path):
    from module2_prs import data_prep
    p = tmp_path / "d.raw"
    p.write_text("FID\tIID\tPAT\tMAT\tSEX\tPHENOTYPE\tchr1:100:C:CAAA_C\tchr2:5:A:G_G\n"
                 "0\ta\t0\t0\t0\t-9\t2\t0\n0\tb\t0\t0\t0\t-9\tNA\t1\n")
    df, ids = data_prep.read_export_a(p)
    assert ids == ["chr1:100:C:CAAA", "chr2:5:A:G"] and np.isnan(df.loc[1, "chr1:100:C:CAAA"])


def test_analyse_with_drivers(tmp_path):
    """Simulate plink outputs for 2 SNPs where SNP v1 differs in frequency between populations."""
    import yaml
    from module2_prs import run
    rng = np.random.default_rng(5)
    sizes = {"CEU": 120, "PJL": 97, **{g: 22 for g in PAK[1:]}}
    rows = []
    for g, n in sizes.items():
        p1 = 0.2 if g == "CEU" else 0.5        # v1: big frequency difference
        for _ in range(n):
            rows.append((g, rng.binomial(2, p1), rng.binomial(2, 0.3)))
    df = pd.DataFrame(rows, columns=["group", "v1", "v2"])
    df["IID"] = [f"s{i}" for i in range(len(df))]
    w = {"v1": 0.3, "v2": 0.1}
    df["prs"] = df.v1 * w["v1"] + df.v2 * w["v2"] + rng.normal(0, 0.2, len(df))
    out = tmp_path / "out"; out.mkdir()
    pd.DataFrame({"#FID": df.IID, "IID": df.IID, "SCORE1_SUM": df.prs}).to_csv(out / "prs313.sscore", sep="\t", index=False)
    raw = pd.DataFrame({"FID": df.IID, "IID": df.IID, "PAT": 0, "MAT": 0, "SEX": 0, "PHENOTYPE": -9,
                        "v1_A": df.v1, "v2_G": df.v2})
    raw.to_csv(out / "prs313_dosage.raw", sep="\t", index=False)
    pd.DataFrame({"target_id": ["v1", "v2"], "weight": [0.3, 0.1], "chr": ["1", "2"], "pos": [1, 2],
                  "effect_allele": ["A", "G"], "other_allele": ["C", "T"]}).to_csv(out / "harmonised_variants.csv", index=False)
    sheet = df[["IID", "group"]].assign(ancestry_outlier=False, gnomad_subcont_outlier=False)
    sheet.loc[0, "ancestry_outlier"] = True
    sheet.to_csv(tmp_path / "sheet.csv", index=False)
    cfg = {"data_root": str(tmp_path), "results_dir": str(out), "n_boot": 20,
           "paths": {"weights": str(WEIGHTS.resolve()), "sample_sheet": "sheet.csv", "work_dir": "w"},
           "pakistani_groups": PAK}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    run.main(["--config", str(tmp_path / "c.yaml"), "--step", "analyse"])
    d = pd.read_csv(out / "driver_snps.csv")
    assert d.variant_id.iloc[0] == "v1"            # v1 drives the shift
    assert (out / "reclassification_excl_ancestry_outliers.csv").exists()
    assert (out / "effect_allele_freq_by_group.csv").exists()
