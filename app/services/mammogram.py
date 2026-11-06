"""Module 3 adapter: benign vs malignant probability + explanation heatmap."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from app.labels import LabelledResult
from app.services.base import ModelAdapter, repo_path


class MammogramClassifierAdapter(ModelAdapter):
    name = "Module 3 classifier"
    label_keys = ("m3_probability", "m3_heatmap")

    def availability(self) -> tuple[bool, str]:
        if importlib.util.find_spec("torch") is None:
            return False, "PyTorch is not installed in this environment (use environment-imaging.yml)."
        ckpt = repo_path(self.config.get("m3_checkpoint"))
        if ckpt is None or not ckpt.exists():
            return False, "no trained Module 3 checkpoint found (models/module3/). Training has not been run yet."
        return True, ""

    def _predict(self, files: dict[str, Path]) -> list[LabelledResult]:  # pragma: no cover
        # TODO(Module 3): load the checkpoint via module3_classifier (model factory + preprocessing
        # identical to training: same resize ~1024x512, same normalisation), return
        #   make_result(self.labels, "m3_probability", values={"probability": p, "threshold": t,
        #               "prediction": "malignant"/"benign"}, ...)
        # with t = the Youden threshold chosen on VALIDATION and stored with the checkpoint, and
        #   make_result(self.labels, "m3_heatmap", image_data_uri=<Grad-CAM overlay PNG>, ...)
        # using module3_classifier's explain code.
        raise NotImplementedError("Module 3 inference is not wired in yet")
