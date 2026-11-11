"""VinDr-Mammo DICOM -> 1024x512 PNG cache + index with lesion boxes in cached coordinates (O3).

Input : the downloaded VinDr files (``fetch_vindr.py``): ``<root>/images/<study>/<image>.dicom`` plus
        ``finding_annotations.csv``.
Output: ``<out>/images/<image_id>.png`` (16-bit, same preprocessing as CBIS-DDSM) and ``<out>/index.csv``
        with the columns the explanation code reads (``cache_path``, ``label``, ``boxes_cached``, ...).

Which images: those that meet the D-18 rule (mass or suspicious calcification, BI-RADS >= 4), via
``datasets.load_vindr_boxes`` (the same function the downloader uses). With ``--sample N --seed S`` only the
seeded sample chosen by ``fetch_vindr.py``. Images not downloaded yet are skipped and counted, so the
script can be re-run as the download progresses (finished PNGs are kept).

Why the preprocessing is identical to CBIS-DDSM: the classifier must see the same intensity scale and
orientation on every dataset, otherwise a score difference could come from preprocessing, not population.

**Licence (PhysioNet Restricted Health Data License):** images stay on the local machine. The output folder
must be outside the repository or inside its gitignored ``data/`` folder; anything else is refused. Never
upload the cache (Kaggle/Colab included), never export overlays. Only metrics leave this machine.

Usage:
    python -m module3_classifier.cache_vindr --root D:/mammorisk_data/vindr_mammo --sample 200 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from module3_classifier import preprocess as P
from module3_classifier.config import REPO_ROOT
from module3_classifier.datasets import load_vindr_boxes
from module3_classifier.fetch_vindr import rel_path, sample_images, select_images

DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "vindr_png_1024x512"
H, W = 1024, 512


def _check_out_dir(out: Path) -> None:
    """Refuse an output folder that git could pick up (licence: VinDr images are never committed)."""
    out = out.resolve()
    try:
        rel = out.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return                      # outside the repo: fine
    if rel.parts and rel.parts[0] == "data":
        return                      # data/ is gitignored
    raise SystemExit(f"Refusing to write VinDr images into the repo at {out}. "
                     "Use a folder outside the repository or under data/ (gitignored).")


def build_index(root: Path, out: Path, sample: int | None = None, seed: int = 42,
                limit: int | None = None) -> pd.DataFrame:
    boxes = load_vindr_boxes(root / "finding_annotations.csv")          # one row per kept image
    if sample:
        keep = sample_images(root, select_images(root), sample, seed)[["image_id"]]
        boxes = keep.merge(boxes, on="image_id", how="left")            # keeps the sample's order
    if limit:
        boxes = boxes.head(limit)

    rows = []
    for r in boxes.itertuples():
        dcm = root / rel_path(r.study_id, r.image_id)
        rec = {"image_id": r.image_id, "study_id": r.study_id, "patient_id": r.patient_id,
               "laterality": r.laterality, "view_position": r.view_position, "label": 1,
               "dataset": "VinDr-Mammo", "boxes": r.boxes}
        if not dcm.exists():
            rows.append({**rec, "status": "not_downloaded"})
            continue
        try:
            img = P.read_dicom(dcm)
            if img.shape != (int(r.height), int(r.width)):
                # annotation coordinates refer to the annotated image size; a different DICOM size
                # would put the boxes in the wrong place, so refuse rather than guess
                rows.append({**rec, "status": f"size_mismatch dicom={img.shape} csv={(int(r.height), int(r.width))}"})
                continue
            small, t = P.preprocess_array(img, H, W, canonical_flip=True)
            png = out / "images" / f"{r.image_id}.png"
            P.save_png16(small, png)
            cached = P.transform_boxes(r.boxes, t)
            rows.append({**rec, "status": "ok", "cache_path": f"images/{r.image_id}.png",
                         "boxes_cached": [tuple(round(v, 2) for v in b) for b in cached],
                         "orig_h": t.orig_h, "orig_w": t.orig_w, "flipped": t.flipped, "scale": t.scale})
        except Exception as err:  # keep going; the failure is recorded in the index
            rows.append({**rec, "status": f"failed: {type(err).__name__}: {err}"})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cache the VinDr-Mammo images used for explanation scoring (O3)")
    ap.add_argument("--root", required=True, help="folder with finding_annotations.csv and images/")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="cache folder (default data/processed/vindr_png_1024x512)")
    ap.add_argument("--sample", type=int, help="only the seeded sample of N images (same --sample/--seed as fetch_vindr)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="first N images only (trial)")
    args = ap.parse_args(argv)

    root, out = Path(args.root), Path(args.out)
    _check_out_dir(out)
    out.mkdir(parents=True, exist_ok=True)
    idx = build_index(root, out, args.sample, args.seed, args.limit)
    idx.to_csv(out / "index.csv", index=False)

    status = idx["status"].str.split(":").str[0].str.split(" ").str[0].value_counts().to_dict()
    print(f"{len(idx)} images selected -> {status}")
    print(f"index: {out / 'index.csv'}  (ok rows are the ones the explanation code can use)")
    bad = idx[~idx["status"].isin(["ok", "not_downloaded"])]
    for r in bad.head(5).itertuples():
        print("PROBLEM", r.image_id, r.status)
    print("Licence reminder: keep this folder local. Do not upload it or commit it.")
    return 1 if len(bad) else 0


if __name__ == "__main__":
    sys.exit(main())
