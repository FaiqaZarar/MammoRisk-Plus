"""Load a PGS Catalog scoring file (format 2.0) into a tidy table.

Two flavours exist for PGS000004:
* **author-reported** (``PGS000004.txt.gz``, the file now in ``data/prs/``): ``chr_name``,
  ``chr_position`` on GRCh37, ``effect_allele``, ``other_allele``, ``effect_weight``,
  ``allelefrequency_effect``. There are no rsIDs.
* **harmonised** (``PGS000004_hmPOS_GRCh38.txt.gz``): adds ``hm_chr``, ``hm_pos``, ``hm_rsID`` mapped
  to GRCh38. CLAUDE.md specifies this one. When present, the ``hm_*`` columns are used.

The genome build of the weights must match the genotypes (positions differ between builds), so
``load_scoring_file`` returns the build and ``check_build`` refuses a mismatch.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

TIDY_COLUMNS = ["chr", "pos", "effect_allele", "other_allele", "weight", "eaf", "rsid"]


@dataclass
class ScoringFile:
    pgs_id: str
    build: str                       # "GRCh37" or "GRCh38"
    harmonised: bool
    variants: pd.DataFrame           # TIDY_COLUMNS
    header: dict = field(default_factory=dict)

    @property
    def n_variants(self) -> int:
        return len(self.variants)


def _read_header(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    header = {}
    with opener(path, "rt") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            line = line.lstrip("#").strip()
            if "=" in line:
                k, v = line.split("=", 1)
                header[k.strip()] = v.strip()
    return header


def _norm_chr(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("^chr", "", regex=True).str.upper()


def load_scoring_file(path: str | Path) -> ScoringFile:
    path = Path(path)
    header = _read_header(path)
    raw = pd.read_csv(path, sep="\t", comment="#", dtype={"chr_name": str, "hm_chr": str},
                      low_memory=False)
    harmonised = {"hm_chr", "hm_pos"}.issubset(raw.columns)
    if harmonised:
        build = header.get("HmPOS_build", "GRCh38")
        chr_col, pos_col = "hm_chr", "hm_pos"
        raw = raw.dropna(subset=["hm_chr", "hm_pos"])  # variants that failed to map
    else:
        build = header.get("genome_build", "NR")
        chr_col, pos_col = "chr_name", "chr_position"

    out = pd.DataFrame({
        "chr": _norm_chr(raw[chr_col]),
        "pos": raw[pos_col].astype("int64"),
        "effect_allele": raw["effect_allele"].astype(str).str.upper(),
        "other_allele": raw.get("other_allele", raw.get("hm_inferOtherAllele")).astype(str).str.upper(),
        "weight": raw["effect_weight"].astype(float),
        "eaf": raw["allelefrequency_effect"].astype(float) if "allelefrequency_effect" in raw else float("nan"),
        "rsid": raw["hm_rsID"] if "hm_rsID" in raw else raw.get("rsID", pd.Series([None] * len(raw))),
    })
    return ScoringFile(header.get("pgs_id", path.stem), build, harmonised,
                       out.reset_index(drop=True), header)


def check_build(scoring: ScoringFile, genotype_build: str) -> None:
    """Raise if the weights and genotypes are on different builds (e.g. GRCh37 vs GRCh38)."""
    norm = {"hg19": "GRCh37", "grch37": "GRCh37", "hg38": "GRCh38", "grch38": "GRCh38"}
    w = norm.get(scoring.build.lower(), scoring.build)
    g = norm.get(genotype_build.lower(), genotype_build)
    if w != g:
        raise ValueError(
            f"{scoring.pgs_id} weights are {w} but genotypes are {g}. Positions will not match. "
            "Use the harmonised file for the genotype build (see DECISIONS D-13 open item).")
