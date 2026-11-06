"""Get only what PRS-313 needs: the 313 positions, for the CEU + 9 Pakistani groups.

Saving storage: we never download whole-genome VCFs for scoring. ``bcftools view -R regions``
reads remote, tabix-indexed VCFs over HTTPS and downloads only the index plus the blocks that
contain our 313 positions (a few MB per chromosome instead of GBs). The ancestry PCA does need
genome-wide common SNPs; do it on the merged reference panel once, on the USB drive / data_root.

Sample sheet: one row per person, ``IID``, ``group``, ``source`` (1000G or HGDP), ``sex``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HGDP_PAKISTANI = ["Balochi", "Brahui", "Burusho", "Hazara", "Kalash", "Makrani", "Pathan", "Sindhi"]


def write_regions(weights: pd.DataFrame, path: str | Path, chr_prefix: str = "") -> Path:
    """bcftools regions file: CHROM<TAB>POS<TAB>POS for each weight (1-based, inclusive)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reg = pd.DataFrame({"chr": chr_prefix + weights["chr"].astype(str), "start": weights["pos"],
                        "end": weights["pos"]}).sort_values(["chr", "start"])
    reg.to_csv(path, sep="\t", header=False, index=False)
    return path


def bcftools_extract_commands(vcf_url_template: str, regions: str | Path, samples: str | Path,
                              out_dir: str | Path, chromosomes: list[str],
                              bcftools: str = "bcftools", pass_only: bool = True) -> list[list[str]]:
    """One command per chromosome: pull only the regions, for only the listed samples.

    ``vcf_url_template`` contains ``{chr}`` (e.g. the gnomAD HGDP+1KG per-chromosome URL).
    bcftools reads the remote ``.tbi`` index and fetches only the compressed blocks covering our
    positions. ``-m2`` drops monomorphic records. ``-f PASS`` (``pass_only``) keeps only sites passing
    gnomAD's variant QC. A PRS SNP that fails QC is then *missing* (imputed 2·EAF, counted in coverage)
    rather than scored from unreliable genotypes.
    Output: ``<out_dir>/prs313_<chrom>.vcf.gz``.
    """
    out_dir = Path(out_dir)
    cmds = []
    for c in chromosomes:
        cmd = [bcftools, "view", "-R", str(regions), "-S", str(samples), "--force-samples", "-m2"]
        if pass_only:
            cmd += ["-f", "PASS"]
        cmd += ["-Oz", "-o", str(out_dir / f"prs313_{c}.vcf.gz"), vcf_url_template.format(chr=c)]
        cmds.append(cmd)
    return cmds


def merge_commands(out_dir: str | Path, chromosomes: list[str], merged_prefix: str | Path,
                   bcftools: str = "bcftools", plink2: str = "plink2") -> list[list[str]]:
    """Concatenate per-chromosome extracts and convert to a plink2 fileset.

    ``--set-all-var-ids @:#:$r:$a`` gives every variant a unique chr:pos:ref:alt ID (gnomAD's ID
    column is often '.'), which the harmonisation step then uses in the score file.
    """
    out_dir = Path(out_dir)
    vcfs = [str(out_dir / f"prs313_{c}.vcf.gz") for c in chromosomes]
    merged_vcf = str(out_dir / "prs313_all.vcf.gz")
    return [
        [bcftools, "concat", "-Oz", "-o", merged_vcf, *vcfs],
        [plink2, "--vcf", merged_vcf, "--set-all-var-ids", "@:#:$r:$a",
         "--new-id-max-allele-len", "1000", "truncate", "--make-pgen", "--out", str(merged_prefix)],
    ]


def dosage_export_command(genotype_prefix: str | Path, score_file: str | Path,
                          out_prefix: str | Path, plink2: str = "plink2") -> list[list[str]]:
    """Export effect-allele dosages (0/1/2) of the scored variants, for the driver-SNP analysis.

    ``--export-allele`` makes plink count the *effect* allele (not ALT); it needs a headerless
    ID<TAB>allele file, which ``write_export_allele`` creates from the score file.
    """
    allele_file = Path(str(out_prefix) + "_export_allele.txt")
    ids_file = Path(str(out_prefix) + "_ids.txt")
    return [[plink2, "--pfile", str(genotype_prefix), "--extract", str(ids_file),
             "--export", "A", "--export-allele", str(allele_file), "--out", str(out_prefix)]]


def write_export_allele(score_file: str | Path, out_prefix: str | Path) -> Path:
    """Write ``<prefix>_export_allele.txt`` (ID, allele; no header) and ``<prefix>_ids.txt``."""
    s = pd.read_csv(score_file, sep="\t")
    p = Path(str(out_prefix) + "_export_allele.txt")
    s[["ID", "A1"]].to_csv(p, sep="\t", header=False, index=False)
    s[["ID"]].to_csv(Path(str(out_prefix) + "_ids.txt"), header=False, index=False)
    return p


def read_export_a(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Read plink2 ``--export A`` (.raw). Returns (IID + dosage columns renamed to variant ID, IDs)."""
    raw = pd.read_csv(path, sep=r"\s+")
    snp_cols = list(raw.columns[6:])
    # Column names are "<ID>_<counted allele>"; IDs contain ':' but the allele suffix is after the last '_'.
    ids = [c.rsplit("_", 1)[0] for c in snp_cols]
    out = raw[["IID"] + snp_cols].copy()
    out.columns = ["IID"] + ids
    out["IID"] = out["IID"].astype(str)
    return out, ids


def sample_sheet_from_gnomad(meta_path: str | Path, groups: list[str],
                             require_high_quality: bool = True,
                             drop_related: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample sheet from the gnomAD v3.1.2 HGDP+1KG metadata (``*_sample_meta.tsv.bgz``).

    Why this callset (DECISIONS D-19): CEU, PJL and the 8 HGDP Pakistani groups were jointly called
    on GRCh38 with one pipeline, so PRS differences between groups cannot come from different
    sequencing / calling batches.

    Filters: gnomAD ``high_quality`` (sample QC) and ``relatedness_inference.related`` (gnomAD's
    relatedness pruning). 1000G families would otherwise count each trio three times and shrink
    the reference SD. Returns (kept sheet, full table with the reason each sample was dropped).

    Also returns gnomAD's genome-wide PCA scores (``global_PC*``) and its subcontinental outlier flag,
    computed on genome-wide data, which we use for the ancestry check (``ancestry.flag_outliers``).
    """
    import json
    raw = pd.read_csv(meta_path, sep="\t", compression="gzip")
    meta = raw["hgdp_tgp_meta"].map(json.loads)
    rel = raw["relatedness_inference"].map(lambda s: json.loads(s) if isinstance(s, str) else {})
    sex = raw["gnomad_sex_imputation"].map(
        lambda s: json.loads(s).get("sex_karyotype") if isinstance(s, str) else None)
    df = pd.DataFrame({
        "IID": raw["s"].astype(str),
        "group": meta.map(lambda m: m.get("population")),
        "source": meta.map(lambda m: {"HGDP": "HGDP", "1000 Genomes": "1000G"}.get(m.get("project"), m.get("project"))),
        "sex": sex,
        "high_quality": raw["high_quality"].astype(bool),
        "related": rel.map(lambda r: bool(r.get("related", False))),
        "gnomad_subcont_outlier": meta.map(lambda m: bool((m.get("subcontinental_pca") or {}).get("outlier"))),
    })
    pcs = meta.map(lambda m: m.get("global_pca_scores") or [])
    n_pc = int(pcs.map(len).max() or 0)
    for k in range(n_pc):
        df[f"global_PC{k + 1}"] = pcs.map(lambda v, k=k: v[k] if len(v) > k else float("nan"))

    df = df[df["group"].isin(groups)].copy()
    df["drop_reason"] = ""
    if require_high_quality:
        df.loc[~df.high_quality, "drop_reason"] += "low_quality;"
    if drop_related:
        df.loc[df.related, "drop_reason"] += "related;"
    kept = df[df.drop_reason == ""].drop(columns="drop_reason").reset_index(drop=True)
    missing = set(groups) - set(kept.group)
    if missing:
        raise ValueError(f"no samples left for groups {sorted(missing)}")
    return kept, df


def sample_sheet(kg_panel: str | Path | None, hgdp_meta: str | Path | None,
                 kg_groups: tuple[str, ...] = ("CEU", "PJL"),
                 hgdp_groups: list[str] = HGDP_PAKISTANI,
                 hgdp_cols: dict | None = None) -> pd.DataFrame:
    """Combine the 1000G panel file and HGDP metadata into one sheet.

    1000G panel (``integrated_call_samples_v3.20130502.ALL.panel``): sample, pop, super_pop, gender.
    HGDP metadata columns vary by release, so ``hgdp_cols`` maps {"IID": ..., "group": ..., "sex": ...}
    (default: sample / population / sex). TO VERIFY against the release that is downloaded.
    """
    parts = []
    if kg_panel:
        kg = pd.read_csv(kg_panel, sep=r"\s+")
        kg = kg[kg["pop"].isin(kg_groups)]
        parts.append(pd.DataFrame({"IID": kg["sample"].astype(str), "group": kg["pop"],
                                   "source": "1000G", "sex": kg.get("gender")}))
    if hgdp_meta:
        cols = {"IID": "sample", "group": "population", "sex": "sex", **(hgdp_cols or {})}
        sep = "\t" if str(hgdp_meta).endswith((".tsv", ".txt")) else ","
        h = pd.read_csv(hgdp_meta, sep=sep)
        h = h[h[cols["group"]].isin(hgdp_groups)]
        parts.append(pd.DataFrame({"IID": h[cols["IID"]].astype(str), "group": h[cols["group"]],
                                   "source": "HGDP", "sex": h.get(cols["sex"])}))
    sheet = pd.concat(parts, ignore_index=True)
    dup = sheet.IID[sheet.IID.duplicated()]
    if len(dup):
        # HGDP samples also appear in some 1000G-based merges; keep one row, report it.
        sheet = sheet.drop_duplicates("IID")
    return sheet
