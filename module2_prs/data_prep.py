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
                              bcftools: str = "bcftools") -> list[list[str]]:
    """One command per chromosome: pull the regions for the listed samples, normalise multiallelics.

    ``vcf_url_template`` contains ``{chr}``, e.g. the 1000G or HGDP per-chromosome VCF URL.
    Output: ``<out_dir>/prs313_chr{chr}.vcf.gz`` (then ``bcftools concat`` + ``plink2 --make-pgen``).
    """
    out_dir = Path(out_dir)
    cmds = []
    for c in chromosomes:
        cmds.append([bcftools, "view", "-R", str(regions), "-S", str(samples),
                     "--force-samples", "-m2", "-Oz", "-o", str(out_dir / f"prs313_chr{c}.vcf.gz"),
                     vcf_url_template.format(chr=c)])
    return cmds


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
