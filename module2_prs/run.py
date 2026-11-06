"""Module 2 pipeline runner.

    python -m module2_prs.run --config configs/m2_prs.yaml --step all
    python -m module2_prs.run --config configs/m2_prs.yaml --step analyse --data-root /content/data

Steps (in order):
  samples   gnomAD HGDP+1KG metadata -> sample sheet (QC + relatedness filters) + samples.txt
  regions   313 positions -> bcftools regions file
  extract   bcftools remote extraction per chromosome -> concat -> plink2 pfile   [needs bcftools, plink2]
  harmonise weights vs extracted variants -> score file + coverage report
  score     plink2 --score -> .sscore                                            [needs plink2]
  dosage    plink2 --export A of effect-allele dosages (for driver SNPs)          [needs plink2]
  analyse   reclassification + CIs, sensitivity analyses, distributions, drivers, ancestry flags

Every output goes to ``<results_dir>/`` with ``run_info_<step>.json`` (config, weights file + build,
coverage, seed, date), so each number in the thesis traces back to its inputs (CLAUDE.md "Results").
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluation.seed import set_seed
from module2_prs import ancestry, data_prep, drivers, harmonise, reclassify, score, standardise, weights

STEPS = ["samples", "regions", "extract", "harmonise", "score", "dosage", "analyse"]


def load_config(path: str | Path, data_root: str | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if data_root:
        cfg["data_root"] = data_root
    root = Path(cfg.get("data_root", "data"))
    # Every path is relative to data_root unless absolute, so moving the data to a USB drive or to
    # Colab means changing one line (or passing --data-root).
    for k, v in list(cfg.get("paths", {}).items()):
        if v and not Path(v).is_absolute():
            cfg["paths"][k] = str(root / v)
    return cfg


def _results_dir(cfg) -> Path:
    d = Path(cfg.get("results_dir", "results/m2_prs"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _work(cfg) -> Path:
    w = Path(cfg["paths"]["work_dir"])
    w.mkdir(parents=True, exist_ok=True)
    return w


def _chromosomes(sf) -> list[str]:
    return sorted(sf.variants.chr.unique(), key=int)


def _tools(cfg) -> tuple[str, str]:
    return cfg.get("bcftools_path", "bcftools"), cfg.get("plink2_path", "plink2")


# ---------------------------------------------------------------------------------------------
def step_samples(cfg, out: Path) -> dict:
    meta = Path(cfg["paths"]["gnomad_meta"])
    if not meta.exists():
        meta.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading gnomAD sample metadata (~3.4 MB) -> {meta}")
        urllib.request.urlretrieve(cfg["gnomad_meta_url"], meta)
    groups = [cfg.get("ceu_label", "CEU"), *cfg["pakistani_groups"]]
    kept, full = data_prep.sample_sheet_from_gnomad(meta, groups, cfg.get("require_high_quality", True),
                                                    cfg.get("drop_related", True))
    kept["ancestry_outlier"] = ancestry.flag_outliers(
        kept, kept["group"], n_pcs=cfg.get("ancestry_n_pcs", 2), k_sd=cfg.get("ancestry_k_sd", 4.0),
        prefix="global_PC")
    Path(cfg["paths"]["sample_sheet"]).parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(cfg["paths"]["sample_sheet"], index=False)
    kept["IID"].to_csv(cfg["paths"]["samples_list"], index=False, header=False)
    counts = pd.DataFrame({"all": full.groupby("group").size(),
                           "kept": kept.groupby("group").size(),
                           "female_kept": kept[kept.sex == "XX"].groupby("group").size(),
                           "ancestry_outliers": kept.groupby("group")["ancestry_outlier"].sum(),
                           "gnomad_subcont_outliers": kept.groupby("group")["gnomad_subcont_outlier"].sum()}
                          ).reindex(groups).fillna(0).astype(int)
    counts.to_csv(out / "sample_counts.csv")
    print(counts)
    return {"n_kept": len(kept), "per_group": counts.to_dict(orient="index")}


def step_regions(cfg, sf) -> Path:
    p = data_prep.write_regions(sf.variants, _work(cfg) / "prs313_regions.txt", cfg.get("chr_prefix", ""))
    print(f"regions file: {p} ({len(sf.variants)} positions)")
    return p


def step_extract(cfg, sf) -> None:
    bcftools, plink2 = _tools(cfg)
    work = _work(cfg)
    chroms = [cfg.get("chr_prefix", "") + c for c in _chromosomes(sf)]
    extract = []
    for c, raw_c in zip(chroms, _chromosomes(sf)):
        # One regions file per chromosome: each remote VCF holds a single chromosome.
        reg = data_prep.write_regions(sf.variants[sf.variants.chr == raw_c],
                                      work / f"prs313_regions_{c}.txt", cfg.get("chr_prefix", ""))
        extract += data_prep.bcftools_extract_commands(cfg["vcf_url_template"], reg,
                                                       cfg["paths"]["samples_list"], work, [c],
                                                       bcftools, cfg.get("pass_only", True))
    for cmd in extract:
        if Path(cmd[cmd.index("-o") + 1]).exists():
            print(f"skip (exists): {cmd[cmd.index('-o') + 1]}")
            continue
        print(" ".join(cmd))
        score.run(cmd)
    for cmd in data_prep.merge_commands(work, chroms, cfg["paths"]["genotype_prefix"], bcftools, plink2):
        print(" ".join(cmd))
        score.run(cmd)


def step_harmonise(cfg, sf, out: Path) -> harmonise.Harmonised:
    pvar = cfg["paths"].get("target_pvar") or cfg["paths"]["genotype_prefix"] + ".pvar"
    h = harmonise.harmonise(sf.variants, harmonise.read_pvar(pvar), cfg.get("ambiguous", "keep"))
    h.table.to_csv(out / "harmonised_variants.csv", index=False)
    h.write_plink_score(out / "prs313.score")
    cov = h.coverage()
    print(f"coverage: {cov['n_scored']}/{cov['n_weights']} ({cov['coverage_pct']:.1f}%)", cov)
    return h


def step_score(cfg, out: Path) -> None:
    cmd = score.plink2_score_command(cfg["paths"]["genotype_prefix"], out / "prs313.score",
                                     out / "prs313", cfg.get("fileset", "pfile"), _tools(cfg)[1],
                                     cfg["paths"].get("read_freq"))
    print(" ".join(map(str, cmd)))
    score.run(cmd)


def step_dosage(cfg, out: Path) -> None:
    data_prep.write_export_allele(out / "prs313.score", out / "prs313_dosage")
    for cmd in data_prep.dosage_export_command(cfg["paths"]["genotype_prefix"], out / "prs313.score",
                                               out / "prs313_dosage", _tools(cfg)[1]):
        print(" ".join(cmd))
        score.run(cmd)


def _driver_analysis(cfg, out: Path, sheet: pd.DataFrame, ceu_sd: float) -> dict | None:
    raw_path = out / "prs313_dosage.raw"
    harm_path = out / "harmonised_variants.csv"
    if not raw_path.exists() or not harm_path.exists():
        print("driver SNPs skipped: run --step dosage first")
        return None
    dos, ids = data_prep.read_export_a(raw_path)
    dos = dos.merge(sheet[["IID", "group"]], on="IID")
    ceu, pak = cfg.get("ceu_label", "CEU"), cfg["pakistani_groups"]
    pooled = np.where(dos.group.isin(pak), "PAKISTANI_POOLED", dos.group)
    freqs = drivers.allele_freqs_from_dosage(dos[ids].to_numpy(float), pooled, ids)
    per_group = drivers.allele_freqs_from_dosage(dos[ids].to_numpy(float), dos.group.to_numpy(), ids)
    harm = pd.read_csv(harm_path).dropna(subset=["target_id"]).set_index("target_id")
    w = harm.loc[harm.index.intersection(ids), "weight"]
    d = drivers.driver_snps(w, freqs["PAKISTANI_POOLED"], freqs[ceu], ceu_sd, top=0)
    totals = {"total_shift": d.attrs["total_shift"], "total_shift_ceu_sd": d.attrs["total_shift_ref_sd"]}
    d = d.join(harm[["chr", "pos", "effect_allele", "other_allele"]])  # join drops .attrs
    d.to_csv(out / "driver_snps.csv", index_label="variant_id")
    per_group.to_csv(out / "effect_allele_freq_by_group.csv", index_label="variant_id")
    return {**totals, "top5": d.head(5).index.tolist()}


def step_analyse(cfg, out: Path) -> dict:
    scores = score.read_sscore(cfg["paths"].get("sscore") or out / "prs313.sscore")
    sheet = pd.read_csv(cfg["paths"]["sample_sheet"], dtype={"IID": str})
    df = scores.merge(sheet, on="IID", how="inner")
    ceu, groups = cfg.get("ceu_label", "CEU"), cfg["pakistani_groups"]
    cats = cfg.get("categories", standardise.DEFAULT_CATEGORIES)
    n_boot, seed = cfg.get("n_boot", 1000), cfg.get("seed", 42)

    # Main analysis -------------------------------------------------------------------------
    res = reclassify.reclassification(df, groups, ceu, cats, n_boot, seed)
    res["table"].to_csv(out / "reclassification.csv", index=False)
    res["crosstab"].to_csv(out / "crosstab_european_vs_pakistani.csv")
    reclassify.own_group_sensitivity(df, groups, ceu, cats).to_csv(out / "own_group_sensitivity.csv", index=False)
    reclassify.group_distributions(df).to_csv(out / "group_distributions.csv", index=False)

    refs = {"european": standardise.fit_reference("european", df.loc[df.group == ceu, "prs"]),
            "pakistani": standardise.fit_reference("pakistani", df.loc[df.group.isin(groups), "prs"])}
    standardise.standardise_all(df[["IID", "group", "prs"]], refs, cats).to_csv(out / "per_person.csv", index=False)
    (out / "reference_stats.json").write_text(json.dumps(
        {"references": {k: vars(v) for k, v in refs.items()}, "categories": cats,
         "weights": cfg["paths"]["weights"]}, indent=2))
    normality = {"european": standardise.normality_check(df.loc[df.group == ceu, "prs"]),
                 "pakistani": standardise.normality_check(df.loc[df.group.isin(groups), "prs"])}

    # Sensitivity: exclude ancestry outliers (gnomAD subcontinental flag or our PC flag) ------
    info = {}
    if {"ancestry_outlier", "gnomad_subcont_outlier"} <= set(df.columns):
        flagged = df.ancestry_outlier.astype(bool) | df.gnomad_subcont_outlier.astype(bool)
        info["n_ancestry_flagged"] = int(flagged.sum())
        if flagged.any():
            sens = reclassify.reclassification(df[~flagged], groups, ceu, cats, n_boot, seed)
            sens["table"].to_csv(out / "reclassification_excl_ancestry_outliers.csv", index=False)

    info["drivers"] = _driver_analysis(cfg, out, sheet, refs["european"].sd)
    return {"n_people": len(df), "n_per_group": df.group.value_counts().to_dict(),
            "references": {k: vars(v) for k, v in refs.items()}, "normality": normality, **info}


def main(argv=None):
    # Plain-ASCII help: the Windows console (cp1252) cannot print the arrows in __doc__.
    ap = argparse.ArgumentParser(description="Module 2 PRS-313 pipeline. Steps: " + ", ".join(STEPS))
    ap.add_argument("--config", required=True)
    ap.add_argument("--step", default="all", choices=STEPS + ["all"])
    ap.add_argument("--data-root", default=None, help="override data_root (e.g. USB drive or /content/data)")
    ap.add_argument("--results-dir", default=None, help="override results_dir")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.data_root)
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    set_seed(cfg.get("seed", 42))
    out = _results_dir(cfg)
    sf = weights.load_scoring_file(cfg["paths"]["weights"])
    if cfg.get("genotype_build"):
        weights.check_build(sf, cfg["genotype_build"])
    info = {"date": dt.datetime.now().isoformat(timespec="seconds"), "config": cfg,
            "weights": {"pgs_id": sf.pgs_id, "build": sf.build, "harmonised": sf.harmonised,
                        "n_variants": sf.n_variants}}
    run = STEPS if args.step == "all" else [args.step]
    if "samples" in run:
        info["samples"] = step_samples(cfg, out)
    if "regions" in run:
        step_regions(cfg, sf)
    if "extract" in run:
        step_extract(cfg, sf)
    if "harmonise" in run:
        info["coverage"] = step_harmonise(cfg, sf, out).coverage()
    if "score" in run:
        step_score(cfg, out)
    if "dosage" in run:
        step_dosage(cfg, out)
    if "analyse" in run:
        info["analysis"] = step_analyse(cfg, out)
    (out / f"run_info_{args.step}.json").write_text(json.dumps(info, indent=2, default=str))
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
