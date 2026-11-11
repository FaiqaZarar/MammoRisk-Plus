"""Resumable downloader for the VinDr-Mammo SUBSET needed for explanation scoring (O3).

Why a subset: the full dataset is 337.8 GB. Explanations are scored only on lesion boxes that count as
"malignant-suspicious" (Mass or Suspicious Calcification, BI-RADS >= 4; D-18). That is 788 images in 405
exams, about 13 GB (27 GB with all four views of those exams). The selection reuses
``datasets.load_vindr_boxes`` so the download and the scoring can never disagree.

Licence (PhysioNet Restricted Health Data License): files stay on the local drive. Never commit them,
never upload them (Kaggle/Colab included), never export images or overlays. Report metrics only.

Credentials: you must have signed the VinDr-Mammo DUA on PhysioNet. The password is read with
``getpass`` (or the PHYSIONET_PASSWORD env var) and is never written to disk or logged.

Resumable and verified: each file goes to ``<name>.part``, is checked against PhysioNet's
``SHA256SUMS.txt``, then renamed. Files already present with a matching hash are skipped.

Usage (the password is prompted):
    python -m module3_classifier.fetch_vindr --root D:/mammorisk_data/vindr_mammo --user <physionet-username>
    python -m module3_classifier.fetch_vindr --root ... --user ... --all-views     # 4 views of those exams
    python -m module3_classifier.fetch_vindr --root ... --user ... --limit 3       # quick trial
    python -m module3_classifier.fetch_vindr --root ... --dry-run                  # list only, no password
"""
from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from module3_classifier.datasets import load_vindr_boxes

BASE = "https://physionet.org/files/vindr-mammo/1.0.0/"
AVG_MB_PER_IMAGE = 337.8 * 1000 / 20000   # dataset total / 20,000 images (for the size estimate only)


def select_images(root: Path, all_views: bool = False) -> pd.DataFrame:
    """Images to fetch: ``study_id, image_id``. Default = images that carry a kept box (D-18 rule)."""
    boxes = load_vindr_boxes(root / "finding_annotations.csv")
    if not all_views:
        return boxes[["study_id", "image_id"]].drop_duplicates().reset_index(drop=True)
    studies = set(boxes["study_id"])
    breast = pd.read_csv(root / "breast-level_annotations.csv")
    return (breast[breast["study_id"].isin(studies)][["study_id", "image_id"]]
            .drop_duplicates().reset_index(drop=True))


def rel_path(study_id: str, image_id: str) -> str:
    return f"images/{study_id}/{image_id}.dicom"


def parse_sha256sums(text: str) -> dict[str, str]:
    """``SHA256SUMS.txt`` lines look like ``<hash>  <relative path>`` (a leading ``*`` or ``./`` is tolerated)."""
    out = {}
    for line in text.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            out[parts[1].lstrip("*").removeprefix("./")] = parts[0].lower()
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _request(url: str, auth: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"Authorization": auth, "User-Agent": "MammoRisk-fetch"})


def fetch_file(url: str, dest: Path, auth: str, expected_sha: str | None, retries: int = 4) -> str:
    """Download ``url`` to ``dest`` (via .part). Returns 'skipped' | 'ok'. Raises on auth/hash failure."""
    if dest.exists() and (expected_sha is None or sha256_file(dest) == expected_sha):
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(_request(url, auth), timeout=120) as r, open(part, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if expected_sha is not None and sha256_file(part) != expected_sha:
                raise IOError("sha256 mismatch")
            part.replace(dest)
            return "ok"
        except urllib.error.HTTPError as err:
            if err.code in (401, 403):  # not a transient problem: stop instead of retrying
                raise PermissionError(f"HTTP {err.code}: wrong username/password, or the VinDr DUA is not signed") from err
            last = err
        except (urllib.error.URLError, IOError, TimeoutError) as err:
            last = err
        time.sleep(2 ** attempt)
    part.unlink(missing_ok=True)
    raise RuntimeError(f"failed after {retries} attempts: {last}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download the VinDr-Mammo subset needed for O3")
    ap.add_argument("--root", required=True, help="folder holding the three CSVs; images go to <root>/images")
    ap.add_argument("--user", help="PhysioNet username")
    ap.add_argument("--all-views", action="store_true", help="all four views of the selected exams")
    ap.add_argument("--workers", type=int, default=3, help="parallel downloads (be polite; default 3)")
    ap.add_argument("--limit", type=int, help="only the first N images (trial run)")
    ap.add_argument("--dry-run", action="store_true", help="show what would be downloaded, then stop")
    args = ap.parse_args(argv)

    root = Path(args.root)
    sel = select_images(root, args.all_views)
    if args.limit:
        sel = sel.head(args.limit)
    total_gb = len(sel) * AVG_MB_PER_IMAGE / 1000
    print(f"{len(sel)} images in {sel.study_id.nunique()} exams, about {total_gb:.1f} GB "
          f"(estimate at {AVG_MB_PER_IMAGE:.1f} MB/image) -> {root / 'images'}")
    if args.dry_run:
        return 0
    if not args.user:
        ap.error("--user is required (unless --dry-run)")

    password = os.environ.get("PHYSIONET_PASSWORD") or getpass.getpass(f"PhysioNet password for {args.user}: ")
    auth = "Basic " + base64.b64encode(f"{args.user}:{password}".encode()).decode()
    del password

    sums_path = root / "SHA256SUMS.txt"
    fetch_file(BASE + "SHA256SUMS.txt", sums_path, auth, None)   # small; re-fetched only if missing
    sums = parse_sha256sums(sums_path.read_text(encoding="utf-8"))

    jobs = []
    for r in sel.itertuples():
        rel = rel_path(r.study_id, r.image_id)
        if rel not in sums:
            print(f"WARNING: {rel} not in SHA256SUMS.txt, skipped")
            continue
        jobs.append((rel, BASE + rel, root / rel, sums[rel]))

    done = fail = skipped = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_file, url, dest, auth, sha): rel for rel, url, dest, sha in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                status = fut.result()
                skipped += status == "skipped"
                done += status == "ok"
            except PermissionError as err:
                print(f"STOP: {err}")
                for f in futs:
                    f.cancel()
                return 2
            except Exception as err:  # keep going; the failed ones are listed at the end
                fail += 1
                print(f"FAILED {futs[fut]}: {err}")
            if i % 25 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] downloaded {done}, already had {skipped}, failed {fail} "
                      f"({(time.time() - t0) / 60:.1f} min)")
    print("Done." if not fail else f"{fail} failed: run the same command again to retry only those.")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
