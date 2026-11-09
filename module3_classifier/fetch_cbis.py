"""Stream CBIS-DDSM from the TCIA API straight into the PNG cache, never storing the 164 GB of DICOM.

For each full mammogram in ``splits/m3_cbis_split.csv``:
  1. download its DICOM series (zip) from the public TCIA NBIA API into a temporary folder
  2. preprocess exactly as ``preprocess.py`` does (normalise, canonical flip, resize + pad, 16-bit PNG)
  3. download every ROI series of each lesion on that image, pick the real binary mask (same shape
     as the full image, two values), and save the union of the masks after the same transform
  4. delete the temporary DICOMs

Peak disk use is a few series at a time (~100–300 MB), so it runs on the laptop, the USB drive,
Colab or Kaggle. It is **resumable**: images already in the cache with status ``ok`` are skipped,
and every finished image is appended to ``manifest_progress.jsonl`` immediately.

Why look masks up by lesion folder rather than by the CSV's mask path: in ~107 series the mask sits
in a series TCIA labels "cropped images" (a known CBIS-DDSM quirk). So we fetch all ROI series of
the lesion (TCIA ``PatientID`` = lesion folder, e.g. ``Mass-Training_P_00001_LEFT_CC_1``) and pick
the mask by shape (``preprocess.pick_mask_file``).

Usage:
    python -m module3_classifier.fetch_cbis --config configs/m3_effnetb3.yaml --workers 4
    python -m module3_classifier.fetch_cbis --config ... --limit 20        # quick trial
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from module3_classifier import preprocess as P
from module3_classifier.config import load_config, resolve

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
ROI_DESCRIPTIONS = {"ROI mask images", "cropped images"}
USER_AGENT = "MammoRisk-Plus research pipeline (academic)"


# ---------------------------------------------------------------------------------------------
# TCIA API
# ---------------------------------------------------------------------------------------------
def _get(url: str, timeout: int = 300, retries: int = 4) -> bytes:
    """HTTP GET with exponential back-off (TCIA occasionally drops long connections)."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 5)
    raise RuntimeError("unreachable")


def series_index(path: Path, collection: str = "CBIS-DDSM") -> pd.DataFrame:
    """All series of the collection (UID, PatientID, description, size), cached as CSV."""
    if path.exists():
        return pd.read_csv(path, dtype=str)
    q = urllib.parse.urlencode({"Collection": collection, "format": "json"})
    rows = json.loads(_get(f"{API}/getSeries?{q}"))
    df = pd.DataFrame(rows)[["SeriesInstanceUID", "PatientID", "SeriesDescription", "ImageCount", "FileSize"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df.astype(str)


def download_series(uid: str, dest: Path) -> list[Path]:
    """Download one series as a zip and extract its .dcm files into ``dest``."""
    data = _get(f"{API}/getImage?{urllib.parse.urlencode({'SeriesInstanceUID': uid})}")
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            if info.filename.lower().endswith(".dcm"):
                target = dest / f"{uid[-12:]}_{Path(info.filename).name}"
                target.write_bytes(z.read(info))
                out.append(target)
    if not out:
        raise RuntimeError(f"series {uid}: zip had no DICOM files")
    return out


# ---------------------------------------------------------------------------------------------
# Path helpers (CSV path = "<TCIA PatientID>/<StudyUID>/<SeriesUID>/<file>.dcm")
# ---------------------------------------------------------------------------------------------
def series_uid_from_path(p: str) -> str:
    parts = [x for x in str(p).strip().replace("\\", "/").split("/") if x]
    if len(parts) < 3:
        raise ValueError(f"unexpected CBIS path {p!r}")
    return parts[2]


def lesion_folder_from_path(p: str) -> str:
    return str(p).strip().replace("\\", "/").split("/")[0]


def roi_series_for_lesions(index: pd.DataFrame, roi_mask_paths: str) -> dict[str, list[str]]:
    """lesion folder → all ROI/cropped series UIDs TCIA lists for it."""
    roi = index[index.SeriesDescription.isin(ROI_DESCRIPTIONS)]
    by_patient = roi.groupby("PatientID")["SeriesInstanceUID"].apply(list).to_dict()
    out = {}
    for p in [x for x in str(roi_mask_paths).split(";") if x and x != "nan"]:
        folder = lesion_folder_from_path(p)
        out[folder] = by_patient.get(folder, [series_uid_from_path(p)])
    return out


# ---------------------------------------------------------------------------------------------
# One image
# ---------------------------------------------------------------------------------------------
def process_image(row: pd.Series, index: pd.DataFrame, cache: Path, tmp_root: Path,
                  out_h: int, out_w: int, canonical_flip: bool) -> dict:
    rec = {"image_id": row.image_id, "status": "ok", "error": ""}
    tmp = Path(tempfile.mkdtemp(prefix=f"{row.image_id}_", dir=tmp_root))
    try:
        full = download_series(series_uid_from_path(row.image_file_path), tmp / "full")
        arr = P.read_dicom(full[0])
        img, t = P.preprocess_array(arr, out_h, out_w, canonical_flip)
        png = cache / "images" / f"{row.image_id}.png"
        P.save_png16(img, png)
        rec.update(cache_path=str(png.relative_to(cache)).replace("\\", "/"), **asdict(t))

        union, n_masks, missing = np.zeros((out_h, out_w), np.uint8), 0, []
        for folder, uids in roi_series_for_lesions(index, row.roi_mask_paths).items():
            cands = []
            for uid in uids:
                cands += download_series(uid, tmp / folder)
            mfile = P.pick_mask_file(sorted(cands), arr.shape)
            if mfile is None:
                missing.append(folder)
                continue
            union |= P.apply_transform_to_mask(P.read_dicom(mfile), t)
            n_masks += 1
        if n_masks:
            mpath = cache / "masks" / f"{row.image_id}.png"
            P.save_mask(union, mpath)
            rec["mask_path"] = str(mpath.relative_to(cache)).replace("\\", "/")
        rec["n_masks"] = n_masks
        rec["masks_not_found"] = ";".join(missing)
    except Exception as err:  # keep going; failures are listed in the manifest and retried next run
        rec.update(status="failed", error=repr(err)[:500])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # never keep raw DICOMs
    return rec


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------
def _done_ids(progress: Path, cache: Path) -> dict[str, dict]:
    done = {}
    if progress.exists():
        for line in progress.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("status") == "ok" and (cache / r["cache_path"]).exists():
                done[r["image_id"]] = r  # last ok record wins
    return done


def write_manifest(split: pd.DataFrame, progress: Path, cache: Path) -> pd.DataFrame:
    recs = {}
    for line in progress.read_text(encoding="utf-8").splitlines() if progress.exists() else []:
        r = json.loads(line)
        if r["image_id"] not in recs or r["status"] == "ok":
            recs[r["image_id"]] = r
    man = split.merge(pd.DataFrame(list(recs.values())) if recs else pd.DataFrame({"image_id": []}),
                      on="image_id", how="left")
    man["status"] = man["status"].fillna("not_fetched")
    man.to_csv(cache / "manifest.csv", index=False)
    return man


def fetch(cfg: dict, workers: int = 4, limit: int | None = None, splits: list[str] | None = None,
          tmp_dir: str | None = None) -> pd.DataFrame:
    split = pd.read_csv(resolve(cfg["paths"]["split_csv"]))
    cache = resolve(cfg["paths"]["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    index = series_index(cache / "tcia_series_index.csv")
    progress = cache / "manifest_progress.jsonl"
    done = _done_ids(progress, cache)

    todo = split if not splits else split[split.split.isin(splits)]
    todo = todo[~todo.image_id.isin(done)]
    if limit:
        todo = todo.head(limit)
    h, w = cfg["image"]["height"], cfg["image"]["width"]
    flip = cfg["image"].get("canonical_flip", True)
    tmp_root = Path(tmp_dir) if tmp_dir else Path(tempfile.gettempdir())
    tmp_root.mkdir(parents=True, exist_ok=True)
    print(f"{len(done)} already cached; fetching {len(todo)} images with {workers} workers -> {cache}")

    t0, n_ok, n_fail = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=workers) as ex, open(progress, "a", encoding="utf-8") as fh:
        futs = {ex.submit(process_image, r, index, cache, tmp_root, h, w, flip): r.image_id
                for r in todo.itertuples(index=False)}
        for i, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_ok += rec["status"] == "ok"
            n_fail += rec["status"] != "ok"
            if i % 25 == 0 or i == len(futs):
                rate = i / max(time.time() - t0, 1e-9)
                print(f"  {i}/{len(futs)} done ({n_ok} ok, {n_fail} failed), "
                      f"{rate * 60:.1f} img/min, ~{(len(futs) - i) / max(rate, 1e-9) / 60:.0f} min left")
    man = write_manifest(split, progress, cache)
    counts = man["status"].value_counts().to_dict()
    print(f"manifest: {cache / 'manifest.csv'} {counts}")
    if counts.get("failed"):
        print("Re-run the same command to retry failed images.")
    return man


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stream CBIS-DDSM from TCIA into the PNG cache")
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads (be polite to TCIA: <=8)")
    ap.add_argument("--limit", type=int, default=None, help="only N images (trial run)")
    ap.add_argument("--splits", nargs="*", default=None, help="subset, e.g. --splits train val")
    ap.add_argument("--cache-dir", default=None, help="override paths.cache_dir (e.g. USB / Kaggle)")
    ap.add_argument("--tmp-dir", default=None, help="where temporary DICOMs go (deleted per image)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.cache_dir:
        cfg["paths"]["cache_dir"] = args.cache_dir
    fetch(cfg, args.workers, args.limit, args.splits, args.tmp_dir)


if __name__ == "__main__":
    main()
