"""Module 2 adapter: PRS-313 under European (CEU) and Pakistani references."""
from __future__ import annotations

import shutil
from pathlib import Path

from app.labels import LabelledResult
from app.services.base import ModelAdapter, repo_path


class PRSAdapter(ModelAdapter):
    name = "Module 2 PRS-313"
    label_keys = ("m2_prs_ceu", "m2_prs_pak")

    def availability(self) -> tuple[bool, str]:
        if shutil.which("plink2") is None:
            return False, "plink2 is not installed on this server (see data/README.md)."
        weights = repo_path(self.config.get("m2_weights"))
        if weights is None or not weights.exists():
            return False, "PGS000004 weights file not found."
        stats = repo_path(self.config.get("m2_reference_stats"))
        if stats is None or not stats.exists():
            return False, ("reference distributions (CEU and Pakistani mean/SD) have not been "
                           "computed yet by Module 2.")
        return True, ""

    def _predict(self, files: dict[str, Path]) -> list[LabelledResult]:  # pragma: no cover
        # TODO(Module 2): use module2_prs to (1) harmonise the uploaded VCF to the PGS000004
        # weights (strand flips, ambiguous A/T C/G SNPs, report coverage n/313), (2) score with
        # plink2 --score, (3) standardise z = (score - ref mean) / ref SD under BOTH references,
        # (4) map to percentile / category with the same cut-offs as O1, and return
        #   make_result(self.labels, "m2_prs_ceu", values={"z": .., "percentile": ..,
        #               "category": .., "coverage": "n/313"}, ...)
        #   make_result(self.labels, "m2_prs_pak", values={...}, ...)
        # Wording: "European (CEU) vs Pakistani" reference. No absolute-risk claims.
        raise NotImplementedError("PRS scoring is not wired in yet")
