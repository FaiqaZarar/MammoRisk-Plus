"""Explanation (saliency map) localisation scores for Module 3 (O3).

All functions take a heatmap ``H`` (2D float array, any non-negative scale) and a binary lesion mask
``M`` of the same shape. Bounding boxes (VinDr-Mammo, DMID) are first rasterised with
``box_to_mask``. If an image has several lesions, pass the union of their masks.

Theory
------
* **Pointing Game** (Zhang et al., 2018): a hit if the single highest-value pixel of H lies inside
  the lesion. Optionally a tolerance of ``tol`` pixels around the lesion is allowed (the original
  paper used 15 px). Reported as the hit rate over images.
* **Energy-in-box** (the energy-based pointing game from Score-CAM, Wang et al., 2020): the share
  of total heatmap mass that falls inside the lesion, Σ_{M} H / Σ H. If the heatmap were spread
  uniformly, the expected value would be the lesion's area fraction, so we also return that
  baseline. A lesion covering 2% of the image with 2% energy has learned nothing.
* **IoU:** binarise H by keeping its top ``top_fraction`` of pixels (CLAUDE.md: top 50%, and
  report the threshold), then compute |B ∩ M| / |B ∪ M|.
* **Sanity check** (Adebayo et al., 2018): randomise the model's weights layer by layer, from the
  top down. A faithful explanation must change. We measure similarity between the original and
  randomised heatmap with Spearman rank correlation of pixel values (``heatmap_similarity``). High
  similarity after randomisation means the map reflects the image (edges), not the model.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, stats


def _check(h: np.ndarray, m: np.ndarray):
    h, m = np.asarray(h, float), np.asarray(m).astype(bool)
    if h.shape != m.shape:
        raise ValueError(f"heatmap {h.shape} and mask {m.shape} differ; upsample the heatmap first")
    if not m.any():
        raise ValueError("empty lesion mask")
    return h, m


def box_to_mask(shape: tuple[int, int], boxes) -> np.ndarray:
    """Rasterise boxes ``(x_min, y_min, x_max, y_max)`` (pixel coords, inclusive-exclusive)."""
    mask = np.zeros(shape, dtype=bool)
    for x0, y0, x1, y1 in boxes:
        x0, y0 = max(int(np.floor(x0)), 0), max(int(np.floor(y0)), 0)
        x1, y1 = min(int(np.ceil(x1)), shape[1]), min(int(np.ceil(y1)), shape[0])
        mask[y0:y1, x0:x1] = True
    return mask


def pointing_game(heatmap, mask, tol: int = 0) -> bool:
    """True if the heatmap's maximum is inside the lesion (dilated by ``tol`` pixels).

    With several equal maxima, every one must be inside for a hit (a flat map is not a hit).
    """
    h, m = _check(heatmap, mask)
    if tol > 0:
        # Euclidean tolerance: any pixel within `tol` px of the lesion counts as inside.
        m = ndimage.distance_transform_edt(~m) <= tol
    peaks = h == h.max()
    return bool(m[peaks].all())


def energy_in_box(heatmap, mask) -> tuple[float, float]:
    """(energy fraction inside lesion, uniform-heatmap baseline = lesion area fraction)."""
    h, m = _check(heatmap, mask)
    h = np.clip(h, 0, None)
    total = h.sum()
    energy = float(h[m].sum() / total) if total > 0 else float("nan")
    return energy, float(m.mean())


def iou_top_fraction(heatmap, mask, top_fraction: float = 0.5) -> tuple[float, float]:
    """(IoU of the top-``top_fraction`` heatmap pixels with the mask, threshold value used)."""
    h, m = _check(heatmap, mask)
    thr = float(np.quantile(h, 1 - top_fraction))
    b = h >= thr
    inter, union = np.logical_and(b, m).sum(), np.logical_or(b, m).sum()
    return float(inter / union), thr


def heatmap_similarity(h_a, h_b) -> float:
    """Spearman rank correlation between two heatmaps of the same shape (sanity check)."""
    a, b = np.asarray(h_a, float).ravel(), np.asarray(h_b, float).ravel()
    if a.shape != b.shape:
        raise ValueError("heatmaps differ in shape")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def score_all(heatmap, mask, top_fraction: float = 0.5, tol: int = 0) -> dict:
    """All localisation scores for one image, as a flat dict (one row of the results CSV)."""
    energy, baseline = energy_in_box(heatmap, mask)
    iou, thr = iou_top_fraction(heatmap, mask, top_fraction)
    return {"pointing_hit": pointing_game(heatmap, mask, tol), "energy_in_box": energy,
            "energy_baseline_area": baseline, "iou": iou, "iou_top_fraction": top_fraction,
            "iou_threshold": thr}
