"""Match PRS weights to the target genotype variants (allele and strand harmonisation).

Theory
------
A weight says "each copy of allele E adds β to the score". In the genotype file the same SNP is
stored as REF/ALT, possibly:

1. **same alleles** (E = ALT or E = REF). plink2 counts the allele *named* in the score file, so
   either orientation works if we pass the effect allele as it appears in the target. No sign
   change is needed.
2. **opposite strand** (E is the complement of REF or ALT, e.g. weights say A/G, data says T/C).
   We take the complement of the effect allele.
3. **palindromic** (A/T or C/G). The complement of the pair is the same pair, so a strand flip
   cannot be detected from the alleles. Options (``ambiguous`` in the config):
   * ``keep``: assume both files are on the forward (+) strand. This holds for PGS Catalog files
     and for 1000G / HGDP VCFs, which are all reported on the reference forward strand.
     **Proposed default.**
   * ``drop``: exclude them. This is conservative, but loses up to 31 of the 313 SNPs.
   We do *not* resolve them by allele frequency: PGS000004 frequencies are European, and frequency
   differences between European and Pakistani groups are exactly what O1 studies.
4. **indels** are matched by exact allele strings (no complementing). Normalise the target VCF first
   (``bcftools norm -f ref.fa -m -any``) so the representation agrees.
5. **missing**: SNP absent from the target. It contributes the expected value 2·EAF·β to every
   person (mean imputation with the reference allele frequency, CLAUDE.md). Because this adds the
   same constant to every person, it cancels out of z = (score − ref mean)/ref SD when the
   reference panel is scored with the same variant set. The variance it removes is not recovered,
   which is why coverage (n of 313) is always reported.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COMPLEMENT = str.maketrans("ACGT", "TGCA")
SNV = set("ACGT")


def complement(allele: str) -> str:
    return allele.translate(COMPLEMENT)


def is_palindromic(a1: str, a2: str) -> bool:
    return len(a1) == 1 and len(a2) == 1 and complement(a1) == a2


def read_pvar(path: str | Path) -> pd.DataFrame:
    """Read a plink2 .pvar or plink1 .bim into columns chr, pos, id, ref, alt."""
    path = Path(path)
    if path.suffix == ".bim":
        df = pd.read_csv(path, sep=r"\s+", header=None,
                         names=["chr", "id", "cm", "pos", "alt", "ref"], dtype={"chr": str})
    else:
        with open(path) as fh:
            skip = 0
            for line in fh:
                if line.startswith("##"):
                    skip += 1
                else:
                    break
        df = pd.read_csv(path, sep="\t", skiprows=skip, dtype={"#CHROM": str}, usecols=range(5))
        df.columns = ["chr", "pos", "id", "ref", "alt"]
    df["chr"] = df["chr"].astype(str).str.replace("^chr", "", regex=True).str.upper()
    for c in ("ref", "alt"):
        df[c] = df[c].astype(str).str.upper()
    return df[["chr", "pos", "id", "ref", "alt"]]


@dataclass
class Harmonised:
    table: pd.DataFrame     # one row per weight variant with status + target id + scored allele
    policy: str

    @property
    def scored(self) -> pd.DataFrame:
        return self.table[self.table.status.isin(["match", "flip", "ambiguous_kept"])]

    def coverage(self) -> dict:
        counts = self.table.status.value_counts().to_dict()
        n = len(self.table)
        return {"n_weights": n, "n_scored": len(self.scored),
                "coverage_pct": 100 * len(self.scored) / n if n else float("nan"),
                "ambiguous_policy": self.policy, **{f"n_{k}": int(v) for k, v in counts.items()}}

    def missing_offset(self) -> float:
        """Σ 2·EAF·β over weights not scored (constant added to every person)."""
        miss = self.table[~self.table.index.isin(self.scored.index)]
        return float((2 * miss.eaf.fillna(0) * miss.weight).sum())

    def write_plink_score(self, path: str | Path) -> Path:
        """Score file for ``plink2 --score <file> 1 2 3 header-read``: ID, A1, BETA."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        s = self.scored[["target_id", "scored_allele", "weight"]]
        s.columns = ["ID", "A1", "BETA"]
        s.to_csv(path, sep="\t", index=False)
        return path


def harmonise(weights: pd.DataFrame, target: pd.DataFrame, ambiguous: str = "keep") -> Harmonised:
    """Match each weight (chr, pos, effect/other allele) to a target variant at the same position."""
    if ambiguous not in ("keep", "drop"):
        raise ValueError("ambiguous must be 'keep' or 'drop'")
    by_pos = {k: g for k, g in target.groupby(["chr", "pos"])}
    rows = []
    for _, w in weights.iterrows():
        ea, oa = w.effect_allele, w.other_allele
        rec = {**w.to_dict(), "target_id": None, "scored_allele": None, "status": "missing"}
        cands = by_pos.get((w.chr, w.pos))
        if cands is not None:
            pal = is_palindromic(ea, oa)
            for _, t in cands.iterrows():
                pair = {t.ref, t.alt}
                if {ea, oa} == pair:
                    if pal and ambiguous == "drop":
                        rec.update(target_id=t.id, status="ambiguous_dropped")
                    else:
                        rec.update(target_id=t.id, scored_allele=ea,
                                   status="ambiguous_kept" if pal else "match")
                    break
                is_snv = set(ea) <= SNV and set(oa) <= SNV and len(ea) == len(oa) == 1
                if is_snv and {complement(ea), complement(oa)} == pair:
                    rec.update(target_id=t.id, scored_allele=complement(ea), status="flip")
                    break
            else:
                rec["status"] = "allele_mismatch"  # position found, alleles disagree (e.g. tri-allelic)
        rows.append(rec)
    return Harmonised(pd.DataFrame(rows), ambiguous)
