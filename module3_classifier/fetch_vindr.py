"""Select and verify the VinDr-Mammo SUBSET needed for explanation scoring (O3). Downloading is done by wget.

Why a subset: the full dataset is 337.8 GB. Explanations are scored only on lesion boxes that count as
"malignant-suspicious" (Mass or Suspicious Calcification, BI-RADS >= 4; D-18). That is 788 images in 405
exams, about 13 GB (27 GB with all four views of those exams). The selection reuses
``datasets.load_vindr_boxes`` so the download and the scoring can never disagree.

Why wget and not Python: PhysioNet's file server answered 403 to curl and to Python's urllib even with a
valid login and signed DUA (2026-11-11), but accepts the real GNU wget (its documented client). So this
script never touches credentials or the network: it writes the URL list, prints the wget command (wget
prompts for the password, which is never stored), and afterwards verifies every file against PhysioNet's
SHA256SUMS.txt.

Licence (PhysioNet Restricted Health Data License): files stay on the local drive. Never commit them,
never upload them (Kaggle/Colab included), never export images or overlays. Report metrics only.

Usage:
    1. wget SHA256SUMS.txt once (command printed by step 2 shows how), then
    2. python -m module3_classifier.fetch_vindr --root D:/mammorisk_data/vindr_mammo [--all-views] [--limit 3]
       -> writes <root>/subset_urls.txt and prints the wget command to run (in WSL)
    3. run that wget command (re-run it to resume; finished files are skipped)
    4. python -m module3_classifier.fetch_vindr --root ... --verify [--all-views]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

from module3_classifier.datasets import load_vindr_boxes

BASE = "https://physionet.org/files/vindr-mammo/1.0.0/"
AVG_MB_PER_IMAGE = 337.8 * 1000 / 20000   # dataset total / 20,000 images (size estimate only)


def select_images(root: Path, all_views: bool = False) -> pd.DataFrame:
    """Images to fetch: ``study_id, image_id``. Default = images that carry a kept box (D-18 rule)."""
    boxes = load_vindr_boxes(root / "finding_annotations.csv")
    if not all_views:
        return boxes[["study_id", "image_id"]].drop_duplicates().reset_index(drop=True)
    studies = set(boxes["study_id"])
    breast = pd.read_csv(root / "breast-level_annotations.csv")
    return (breast[breast["study_id"].isin(studies)][["study_id", "image_id"]]
            .drop_duplicates().reset_index(drop=True))


def sample_images(root: Path, sel: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Seeded sample of ``n`` images, balanced between lesion types, ordered so EVERY PREFIX is balanced.

    Why: the full subset (788 images) takes about a day to download at PhysioNet's speed (~140 KB/s,
    2026-11-11). Explanation scores need a sample, not all of them. Balanced between mass and calcification
    because the two are reported separately and calcifications are rarer. An image with both lesion types
    counts as "calcification" (the rarer group). Images are shuffled with ``seed`` within each group and
    then interleaved mass, calc, mass, calc, ... so stopping the download early still leaves a balanced
    sample, and rerunning with the same seed/N reproduces the same list (log both in DECISIONS).
    """
    import ast

    import numpy as np

    f = pd.read_csv(root / "finding_annotations.csv")
    f = f[f["image_id"].isin(set(sel["image_id"]))].dropna(subset=["xmin"])
    has_calc = f["finding_categories"].map(
        lambda s: "Suspicious Calcification" in (ast.literal_eval(s) if isinstance(s, str) and s.startswith("[") else [s]))
    group = has_calc.groupby(f["image_id"]).any().map({True: "calcification", False: "mass"})
    meta = sel.drop_duplicates("image_id").set_index("image_id").join(group.rename("group")).reset_index()
    rng = np.random.default_rng(seed)
    parts = {g: d.iloc[rng.permutation(len(d))].reset_index(drop=True) for g, d in meta.groupby("group")}
    order = []
    for i in range(max(len(d) for d in parts.values())):
        for g in ("mass", "calcification"):
            if i < len(parts.get(g, [])):
                order.append(parts[g].iloc[i])
    return pd.DataFrame(order).head(n).reset_index(drop=True)


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


def wsl_path(p: Path) -> str:
    """``D:/data`` or ``D:\\data`` -> ``/mnt/d/data`` (for wget running inside WSL)."""
    s = str(p).replace("\\", "/")
    if len(s) > 1 and s[1] == ":":
        return f"/mnt/{s[0].lower()}{s[2:]}"
    return s


def write_url_list(root: Path, sel: pd.DataFrame) -> Path:
    out = root / "subset_urls.txt"
    out.write_text("\n".join(BASE + rel_path(r.study_id, r.image_id) for r in sel.itertuples()) + "\n",
                   encoding="utf-8")
    return out


def wget_command(root: Path, url_list: Path, user: str = "<physionet-username>") -> str:
    # -x -nH --cut-dirs=3 keeps images/<study>/<image>.dicom under root; -c resumes; -N skips unchanged files.
    return (f"wsl -d Ubuntu -- wget -c -N -nv -x -nH --cut-dirs=3 --user {user} --ask-password "
            f"-P {wsl_path(root)} -i {wsl_path(url_list)}")


def verify(root: Path, sel: pd.DataFrame) -> dict:
    sums_path = root / "SHA256SUMS.txt"
    if not sums_path.exists():
        raise FileNotFoundError(f"{sums_path} missing: download it with wget first (see --help)")
    sums = parse_sha256sums(sums_path.read_text(encoding="utf-8"))
    res = {"ok": 0, "missing": [], "bad_hash": [], "not_in_sums": []}
    for r in sel.itertuples():
        rel = rel_path(r.study_id, r.image_id)
        f = root / rel
        if rel not in sums:
            res["not_in_sums"].append(rel)
        elif not f.exists():
            res["missing"].append(rel)
        elif sha256_file(f) != sums[rel]:
            res["bad_hash"].append(rel)
        else:
            res["ok"] += 1
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Select / verify the VinDr-Mammo subset needed for O3")
    ap.add_argument("--root", required=True, help="folder holding the three CSVs; images go to <root>/images")
    ap.add_argument("--user", default="<physionet-username>", help="shown in the printed wget command")
    ap.add_argument("--all-views", action="store_true", help="all four views of the selected exams")
    ap.add_argument("--limit", type=int, help="only the first N images (trial run)")
    ap.add_argument("--sample", type=int, help="seeded sample of N images, balanced mass/calcification "
                    "(any prefix is balanced); use the SAME --sample/--seed for download and --verify")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (default 42, as everywhere in the project)")
    ap.add_argument("--verify", action="store_true", help="check downloaded files against SHA256SUMS.txt")
    args = ap.parse_args(argv)

    root = Path(args.root)
    sel = select_images(root, args.all_views)
    if args.sample:
        sel = sample_images(root, sel, args.sample, args.seed)
        sample_csv = root / f"vindr_sample_n{args.sample}_seed{args.seed}.csv"
        sel.to_csv(sample_csv, index=False)   # kept next to the data (not in the repo): lists the exact images
        print(f"sample: {sel['group'].value_counts().to_dict()} -> {sample_csv.name}")
    if args.limit:
        sel = sel.head(args.limit)
    print(f"{len(sel)} images in {sel.study_id.nunique()} exams, about {len(sel) * AVG_MB_PER_IMAGE / 1000:.1f} GB "
          f"(estimate at {AVG_MB_PER_IMAGE:.1f} MB/image) -> {root / 'images'}")

    if args.verify:
        res = verify(root, sel)
        print(f"verified OK: {res['ok']} / {len(sel)}")
        for key in ("missing", "bad_hash", "not_in_sums"):
            if res[key]:
                print(f"{key}: {len(res[key])} (first: {res[key][0]})")
        bad = res["missing"] or res["bad_hash"] or res["not_in_sums"]
        if bad:
            print("Re-run the wget command to fetch/repair these (delete a bad_hash file first).")
        return 1 if bad else 0

    lst = write_url_list(root, sel)
    print(f"wrote {lst}\n\nRun this in PowerShell (wget asks for your PhysioNet password; it is not stored):\n")
    print(wget_command(root, lst, args.user))
    return 0


if __name__ == "__main__":
    sys.exit(main())
