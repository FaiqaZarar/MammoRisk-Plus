"""Module 1 adapter: Mirai 1–5 year risk from four FFDM views via the OncoServe container."""
from __future__ import annotations

from pathlib import Path

from app.labels import LabelledResult
from app.services.base import ModelAdapter

REQUIRED_VIEWS = ("L-CC", "L-MLO", "R-CC", "R-MLO")


class MiraiAdapter(ModelAdapter):
    name = "Module 1 Mirai"
    label_keys = ("m1_mirai_risk",)

    def availability(self) -> tuple[bool, str]:
        if not self.config.get("m1_oncoserve_url"):
            return False, ("the Mirai OncoServe container (learn2cure/oncoserve_mirai:0.5.0) is not "
                           "configured (paths.m1_oncoserve_url).")
        return True, ""

    def predict(self, files: dict[str, Path]) -> list[LabelledResult]:
        missing = [v for v in REQUIRED_VIEWS if v not in files]
        if missing:
            return self.unavailable("Mirai needs all four views; missing " + ", ".join(missing) + ".")
        return super().predict(files)

    def _predict(self, files: dict[str, Path]) -> list[LabelledResult]:  # pragma: no cover
        # TODO(Module 1): POST the four DICOMs to OncoServe (see module1_mirai for the request
        # format and output parsing) and return
        #   make_result(self.labels, "m1_mirai_risk",
        #               values={"1-year": r1, ..., "5-year": r5}, ...)
        # Mirai combines the four views of ONE exam; no prior exams are used.
        raise NotImplementedError("Mirai inference is not wired in yet")
