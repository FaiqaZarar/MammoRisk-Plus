"""DICOM → downsampled PNG cache, with lesion masks transformed identically.

Why a cache: CBIS-DDSM is ~164 GB as DICOM, and SINES gives ~20 GB (CLAUDE.md compute constraint).
A 1024×512 16-bit PNG is ~1 MB, so the whole set fits in a few GB. Training then reads PNGs, not
DICOMs.

Steps per image
1. Read the pixel array, apply the DICOM VOI LUT / windowing if present, and invert MONOCHROME1 so
   bright = dense tissue everywhere.
2. Intensity normalisation: clip at the 0.5th / 99.5th percentile, then min–max to [0, 1]. Scanned
   film (CBIS-DDSM) and digital (CMMD, VinDr) have very different raw ranges. This puts them on one
   scale without per-dataset tuning.
3. Canonical orientation (optional, on by default): flip horizontally so the breast (the brighter
   half) is on the LEFT. The CNN then always sees the chest wall on the same side. The same flip is
   applied to the masks and boxes.
4. Resize keeping the aspect ratio, and pad (bottom/right, with 0) to exactly H×W = 1024×512.
   Plain resizing to 1024×512 would distort lesion shape differently per scanner.
5. Save as 16-bit PNG (keeps more grey levels than 8-bit; mammographic detail is subtle).

Every image's geometric transform (flip, scale, pad) is saved in the manifest, so boxes
(VinDr / DMID) and heatmaps can be mapped between original and cached coordinates.

**Raw DICOM deletion is NOT automated.** Delete raw DICOMs only after checking that the manifest has
no failed rows and spot-checking some PNGs (see the message printed at the end).

Usage:  python -m module3_classifier.preprocess --config configs/m3_effnetb3.yaml
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from module3_classifier.config import load_config, resolve


@dataclass
class Transform:
    orig_h: int
    orig_w: int
    flipped: bool
    scale: float
    out_h: int
    out_w: int


# ------------------------------------------------------------------------------------------------
# Pure-numpy geometry (testable without OpenCV)
# ------------------------------------------------------------------------------------------------
def breast_on_right(img: np.ndarray) -> bool:
    """True if the right half carries more intensity than the left (breast faces right)."""
    w = img.shape[1]
    return float(img[:, w // 2:].sum()) > float(img[:, : w // 2].sum())


def normalise_intensity(img: np.ndarray, lo_pct=0.5, hi_pct=99.5) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0, 1)


def fit_scale(h: int, w: int, out_h: int, out_w: int) -> float:
    return min(out_h / h, out_w / w)


def transform_boxes(boxes, t: Transform) -> list[tuple[float, float, float, float]]:
    """Map (x0, y0, x1, y1) boxes from original to cached image coordinates."""
    out = []
    for x0, y0, x1, y1 in boxes:
        if t.flipped:
            x0, x1 = t.orig_w - x1, t.orig_w - x0
        out.append((x0 * t.scale, y0 * t.scale, x1 * t.scale, y1 * t.scale))
    return out


# ------------------------------------------------------------------------------------------------
# I/O (lazy imports)
# ------------------------------------------------------------------------------------------------
def read_dicom(path: str | Path) -> np.ndarray:
    import pydicom
    try:                                   # pydicom >= 3.0 (the old module is removed in 4.0)
        from pydicom.pixels import apply_voi_lut
    except ImportError:
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:  # no usable VOI LUT / window → keep raw values
        pass
    arr = arr.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


def _resize_pad(img: np.ndarray, scale: float, out_h: int, out_w: int, nearest: bool) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA  # AREA = best for downsampling
    small = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.zeros((out_h, out_w), dtype=small.dtype)
    canvas[: min(nh, out_h), : min(nw, out_w)] = small[:out_h, :out_w]
    return canvas


def preprocess_array(img: np.ndarray, out_h=1024, out_w=512, canonical_flip=True):
    """Normalise, orient and resize one image. Returns (float32 image in [0, 1], Transform)."""
    img = normalise_intensity(img)
    flipped = bool(canonical_flip and breast_on_right(img))
    if flipped:
        img = img[:, ::-1]
    scale = fit_scale(*img.shape, out_h, out_w)
    out = _resize_pad(np.ascontiguousarray(img), scale, out_h, out_w, nearest=False)
    return out, Transform(img.shape[0], img.shape[1], flipped, scale, out_h, out_w)


def apply_transform_to_mask(mask: np.ndarray, t: Transform) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if m.shape != (t.orig_h, t.orig_w):
        raise ValueError(f"mask {m.shape} ≠ image {(t.orig_h, t.orig_w)}")
    if t.flipped:
        m = m[:, ::-1]
    return _resize_pad(np.ascontiguousarray(m), t.scale, t.out_h, t.out_w, nearest=True)


def pick_mask_file(candidates: list[Path], full_shape: tuple[int, int]) -> Path | None:
    """Return the real binary ROI mask among candidate DICOMs.

    Known CBIS-DDSM quirk: in some rows the 'ROI mask file path' points to the *cropped lesion
    image* instead of the mask (the two files share a folder). The mask is the file with the
    full-image shape and only two distinct values.
    """
    for c in candidates:
        try:
            a = read_dicom(c)
        except Exception:
            continue
        if a.shape == tuple(full_shape) and np.unique(a).size <= 2:
            return c
    return None


def save_png16(img01: np.ndarray, path: Path) -> None:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.clip(img01, 0, 1) * 65535).astype(np.uint16))


def save_mask(mask: np.ndarray, path: Path) -> None:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask > 0).astype(np.uint8) * 255)


# ------------------------------------------------------------------------------------------------
# CBIS-DDSM driver
# ------------------------------------------------------------------------------------------------
def _locate(root: Path, rel: str) -> Path:
    """CSV paths are relative to the TCIA download root; the downloaded folder layout differs
    between NBIA versions, so fall back to searching for the series folder."""
    p = root / rel
    if p.exists():
        return p
    series = Path(rel).parts[0] if rel else ""
    hits = list((root).glob(f"**/{series}/**/*.dcm")) if series else []
    if not hits:
        raise FileNotFoundError(rel)
    return sorted(hits)[0]


def preprocess_cbis(cfg: dict) -> pd.DataFrame:
    split = pd.read_csv(resolve(cfg["paths"]["split_csv"]))
    root = resolve(cfg["paths"]["cbis_dicom_root"])
    cache = resolve(cfg["paths"]["cache_dir"])
    out_h, out_w = cfg["image"]["height"], cfg["image"]["width"]
    rows = []
    for r in split.itertuples(index=False):
        rec = {"image_id": r.image_id, "status": "ok", "error": ""}
        try:
            arr = read_dicom(_locate(root, r.image_file_path))
            img, t = preprocess_array(arr, out_h, out_w, cfg["image"].get("canonical_flip", True))
            png = cache / "images" / f"{r.image_id}.png"
            save_png16(img, png)
            rec.update(cache_path=str(png.relative_to(cache)), **asdict(t))
            union = np.zeros((out_h, out_w), np.uint8)
            n_masks = 0
            for rel in [m for m in str(r.roi_mask_paths).split(";") if m and m != "nan"]:
                # The lesion's top folder (e.g. "Mass-Training_P_00001_LEFT_CC_1") holds both the
                # ROI mask and the cropped image, possibly in different series sub-folders.
                top = root / Path(rel).parts[0]
                cands = sorted(top.rglob("*.dcm")) if top.exists() else \
                    sorted(_locate(root, rel).parent.glob("*.dcm"))
                mfile = pick_mask_file(cands, arr.shape)
                if mfile is None:
                    continue
                union |= apply_transform_to_mask(read_dicom(mfile), t)
                n_masks += 1
            if n_masks:
                mpath = cache / "masks" / f"{r.image_id}.png"
                save_mask(union, mpath)
                rec["mask_path"] = str(mpath.relative_to(cache))
            rec["n_masks"] = n_masks
        except Exception as err:  # keep going; failures are listed in the manifest
            rec.update(status="failed", error=repr(err))
        rows.append(rec)
    man = split.merge(pd.DataFrame(rows), on="image_id", how="left")
    man_path = cache / "manifest.csv"
    man.to_csv(man_path, index=False)
    n_fail = int((man["status"] != "ok").sum())
    print(f"Cached {len(man) - n_fail}/{len(man)} images → {cache}; manifest {man_path}")
    print("Raw DICOM deletion is manual. Before deleting anything:\n"
          f"  1. confirm 0 failed rows in {man_path} (now: {n_fail})\n"
          "  2. open ~20 random PNGs and their masks and check orientation and mask overlap\n"
          "  3. keep the CBIS description CSVs (needed to rebuild splits)\n"
          "Only then delete the raw DICOM folder.")
    return man


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cache CBIS-DDSM DICOMs as PNG")
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    preprocess_cbis(load_config(args.config))


if __name__ == "__main__":
    main()
