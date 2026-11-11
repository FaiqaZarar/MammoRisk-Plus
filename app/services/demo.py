"""Demo mode: clearly stamped ILLUSTRATIVE sample output so the result pages can be shown before any
model exists.

Rules (so this can never be mistaken for a result):
* off by default; enabled only by ``MAMMORISK_DEMO=1`` or ``python -m app.demo``; never in Docker
* every result is flagged ``illustrative=True`` -> stamped on the page, in the PDF, and by a page banner
* the numbers are fixed samples, NOT computed from the uploaded file, and the validation label still
  says "Not yet validated"
* PRS categories use the real cut-offs (D-14) on the real normal CDF, so the figure shows the actual
  idea of O1: the same person can fall in a different category under the two references.
"""
from __future__ import annotations

import math

from app.labels import LabelledResult, ValidationLabel, make_result
from app.viz import Z20, Z80, Z99, heatmap_placeholder_uri

# Fixed illustrative z-scores for one imaginary person under the two references.
ILLUSTRATIVE_Z = {"m2_prs_ceu": 1.12, "m2_prs_pak": 0.31}


def percentile(z: float) -> float:
    return 100 * 0.5 * (1 + math.erf(z / math.sqrt(2)))


def category(z: float) -> str:
    if z < Z20:
        return "Low (below 20th percentile)"
    if z < Z80:
        return "Average (20th-80th percentile)"
    if z < Z99:
        return "Elevated (80th-99th percentile)"
    return "High (top 1%)"


def demo_results(labels: dict[str, ValidationLabel], keys: tuple[str, ...]) -> list[LabelledResult]:
    out = []
    for key in keys:
        title = labels[key].output
        if key == "m3_probability":
            kw = {"values": {"probability": 0.64, "threshold": 0.47, "prediction": "malignant (sample)"}}
        elif key == "m3_heatmap":
            kw = {"values": {"method": "Grad-CAM (sample)", "note": "Synthetic picture, not computed from your file"},
                  "image_data_uri": heatmap_placeholder_uri()}
        elif key == "m1_mirai_risk":
            kw = {"values": {"1-year": 0.008, "2-year": 0.016, "3-year": 0.026,
                             "4-year": 0.036, "5-year": 0.047}}
        elif key in ILLUSTRATIVE_Z:
            z = ILLUSTRATIVE_Z[key]
            kw = {"values": {"z": z, "percentile": round(percentile(z), 1), "category": category(z),
                             "coverage": "301 / 313 SNPs found (sample)"}}
        else:  # unknown key: refuse rather than invent
            raise KeyError(f"no demo output defined for {key!r}")
        out.append(make_result(labels, key, title=title, available=True, illustrative=True, **kw))
    return out
