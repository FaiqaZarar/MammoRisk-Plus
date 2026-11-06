"""Module 2 pipeline runner.

    python -m module2_prs.run --config configs/m2_prs.yaml --step all
    python -m module2_prs.run --config configs/m2_prs.yaml --step analyse   # needs only scores + sheet

Steps: regions → harmonise → score → analyse. Every output goes to ``<results_dir>/`` together
with ``run_info.json`` (config, weights file + build, coverage, seed, date), so each number in the
thesis can be traced back to its inputs (CLAUDE.md "Results" rule).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd
import yaml

from evaluation.seed import set_seed
from module2_prs import data_prep, drivers, harmonise, reclassify, score, standardise, weights


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    root = Path(cfg.get("data_root", "data"))
    # Every *_path is relative to data_root unless absolute, so moving the data to a USB drive
    # means changing one line.
    for k, v in list(cfg.get("paths", {}).items()):
        if v and not Path(v).is_absolute():
            cfg["paths"][k] = str(root / v)
    return cfg


def _results_dir(cfg) -> Path:
    d = Path(cfg.get("results_dir", "results/m2_prs"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def step_regions(cfg, sf):
    p = data_prep.write_regions(sf.variants, Path(cfg["paths"]["work_dir"]) / "prs313_regions.txt",
                                cfg.get("chr_prefix", ""))
    print(f"regions file: {p}")
    url = cfg.get("vcf_url_template")
    if url:
        for c in data_prep.bcftools_extract_commands(url, p, cfg["paths"]["samples_list"],
                                                     cfg["paths"]["work_dir"],
                                                     sorted(sf.variants.chr.unique(), key=int)):
            print(" ".join(c))


def step_harmonise(cfg, sf, out: Path) -> harmonise.Harmonised:
    target = harmonise.read_pvar(cfg["paths"]["target_pvar"])
    h = harmonise.harmonise(sf.variants, target, cfg.get("ambiguous", "keep"))
    h.table.to_csv(out / "harmonised_variants.csv", index=False)
    h.write_plink_score(out / "prs313.score")
    cov = h.coverage()
    print(f"coverage: {cov['n_scored']}/{cov['n_weights']} ({cov['coverage_pct']:.1f}%)", cov)
    return h


def step_score(cfg, out: Path):
    cmd = score.plink2_score_command(cfg["paths"]["genotype_prefix"], out / "prs313.score",
                                     out / "prs313", cfg.get("fileset", "pfile"),
                                     cfg.get("plink2_path", "plink2"), cfg["paths"].get("read_freq"))
    print(" ".join(map(str, cmd)))
    score.run(cmd)


def step_analyse(cfg, out: Path, sf=None) -> dict:
    scores = score.read_sscore(cfg["paths"].get("sscore") or out / "prs313.sscore")
    sheet = pd.read_csv(cfg["paths"]["sample_sheet"], dtype={"IID": str})
    df = scores.merge(sheet[["IID", "group"]], on="IID", how="inner")
    groups = cfg["pakistani_groups"]
    cats = cfg.get("categories", standardise.DEFAULT_CATEGORIES)

    res = reclassify.reclassification(df, groups, cfg.get("ceu_label", "CEU"), cats,
                                      cfg.get("n_boot", 1000), cfg.get("seed", 42))
    res["table"].to_csv(out / "reclassification.csv", index=False)
    res["crosstab"].to_csv(out / "crosstab_european_vs_pakistani.csv")
    reclassify.own_group_sensitivity(df, groups, cfg.get("ceu_label", "CEU"), cats).to_csv(
        out / "own_group_sensitivity.csv", index=False)
    reclassify.group_distributions(df).to_csv(out / "group_distributions.csv", index=False)

    refs = {"european": standardise.fit_reference("european", df.loc[df.group == cfg.get("ceu_label", "CEU"), "prs"]),
            "pakistani": standardise.fit_reference("pakistani", df.loc[df.group.isin(groups), "prs"])}
    standardise.standardise_all(df, refs, cats).to_csv(out / "per_person.csv", index=False)
    normality = {k: standardise.normality_check(df.loc[(df.group == cfg.get("ceu_label", "CEU")) if k == "european"
                                                       else df.group.isin(groups), "prs"]) for k in refs}
    return {"n_people": len(df), "n_per_group": df.group.value_counts().to_dict(),
            "references": {k: vars(v) for k, v in refs.items()}, "normality": normality}


def main(argv=None):
    # Plain-ASCII help: the Windows console (cp1252) cannot print the arrows in __doc__.
    ap = argparse.ArgumentParser(description="Module 2 PRS-313 pipeline: regions, harmonise, score, analyse")
    ap.add_argument("--config", required=True)
    ap.add_argument("--step", default="all", choices=["regions", "harmonise", "score", "analyse", "all"])
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    out = _results_dir(cfg)
    sf = weights.load_scoring_file(cfg["paths"]["weights"])
    if cfg.get("genotype_build"):
        weights.check_build(sf, cfg["genotype_build"])
    info = {"date": dt.datetime.now().isoformat(timespec="seconds"), "config": cfg,
            "weights": {"pgs_id": sf.pgs_id, "build": sf.build, "harmonised": sf.harmonised,
                        "n_variants": sf.n_variants}}
    if args.step in ("regions", "all"):
        step_regions(cfg, sf)
    if args.step in ("harmonise", "all"):
        info["coverage"] = step_harmonise(cfg, sf, out).coverage()
    if args.step in ("score", "all"):
        step_score(cfg, out)
    if args.step in ("analyse", "all"):
        info["analysis"] = step_analyse(cfg, out, sf)
    (out / f"run_info_{args.step}.json").write_text(json.dumps(info, indent=2, default=str))
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
