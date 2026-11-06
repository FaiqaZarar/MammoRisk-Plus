"""Parse the official CBIS-DDSM description CSVs into one row per FULL mammogram.

CBIS-DDSM (TCIA) ships four lesion-level CSVs:
``mass_case_description_train_set.csv``, ``mass_case_description_test_set.csv``,
``calc_case_description_train_set.csv``, ``calc_case_description_test_set.csv``.
Each row is one *lesion* (abnormality). A full mammogram can carry several lesions, and the same
(patient, side, view) can appear in both the mass and the calc files.

Labelling rule (CLAUDE.md): an image is **malignant if any lesion on it is malignant**.
``BENIGN_WITHOUT_CALLBACK`` counts as benign.

We ignore the official CBIS train/test split. CLAUDE.md asks for our own 70/10/20 *patient-level*
split after merging mass + calc by patient (``splits.py``), because the official split is per
abnormality type, and a patient can appear in the mass-train and calc-test files.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PATIENT_RE = re.compile(r"(P_\d{5})")

# Column-name variants seen across CBIS-DDSM CSV versions → canonical name.
_ALIASES = {
    "breast_density": "breast_density",
    "breast density": "breast_density",
    "left or right breast": "laterality",
    "left_or_right_breast": "laterality",
    "image view": "view",
    "image_view": "view",
    "abnormality id": "abnormality_id",
    "abnormality_id": "abnormality_id",
    "abnormality type": "abnormality_type",
    "abnormality_type": "abnormality_type",
    "image file path": "image_file_path",
    "image_file_path": "image_file_path",
    "cropped image file path": "cropped_image_file_path",
    "roi mask file path": "roi_mask_file_path",
    "roi_mask_file_path": "roi_mask_file_path",
    "patient_id": "patient_id",
    "pathology": "pathology",
    "assessment": "assessment",
    "subtlety": "subtlety",
}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in df.columns:
        key = c.strip().lower()
        out[c] = _ALIASES.get(key, re.sub(r"\s+", "_", key))
    return df.rename(columns=out)


def normalise_patient_id(value: str) -> str:
    """'Mass-Training_P_00001_LEFT_CC' / 'P_00001' → 'P_00001'."""
    m = PATIENT_RE.search(str(value))
    if not m:
        raise ValueError(f"No CBIS patient id (P_xxxxx) in {value!r}")
    return m.group(1)


def _clean_path(p) -> str:
    # The CSV paths sometimes carry stray newlines / whitespace.
    return "" if pd.isna(p) else str(p).strip().replace("\\", "/")


def load_lesions(csv_paths: list[str | Path]) -> pd.DataFrame:
    """Concatenate lesion-level CSVs into one normalised table."""
    frames = []
    for p in csv_paths:
        df = normalise_columns(pd.read_csv(p))
        name = Path(p).name.lower()
        if "abnormality_type" not in df.columns:
            df["abnormality_type"] = "mass" if name.startswith("mass") else "calcification"
        df["official_split"] = "test" if "test" in name else "train"
        df["source_csv"] = Path(p).name
        frames.append(df)
    les = pd.concat(frames, ignore_index=True)
    required = {"patient_id", "laterality", "view", "pathology", "image_file_path"}
    missing = required - set(les.columns)
    if missing:
        raise ValueError(f"CBIS CSVs lack columns {missing}; got {list(les.columns)}")
    if "breast_density" not in les.columns:
        les["breast_density"] = float("nan")
    les["breast_density"] = pd.to_numeric(les["breast_density"], errors="coerce")
    les["patient_id"] = les["patient_id"].map(normalise_patient_id)
    les["laterality"] = les["laterality"].str.strip().str.upper()
    les["view"] = les["view"].str.strip().str.upper()
    les["pathology"] = les["pathology"].str.strip().str.upper()
    les["image_file_path"] = les["image_file_path"].map(_clean_path)
    if "roi_mask_file_path" in les.columns:
        les["roi_mask_file_path"] = les["roi_mask_file_path"].map(_clean_path)
    else:
        les["roi_mask_file_path"] = ""
    unknown = set(les["pathology"]) - {"MALIGNANT", "BENIGN", "BENIGN_WITHOUT_CALLBACK"}
    if unknown:
        raise ValueError(f"Unexpected pathology values: {unknown}")
    les["malignant"] = (les["pathology"] == "MALIGNANT").astype(int)
    return les


def build_image_table(lesions: pd.DataFrame) -> pd.DataFrame:
    """Collapse lesions to one row per full mammogram (patient, laterality, view).

    The mass and calc subsets store the same physical view as separate DICOM series. We keep one
    representative image path (the first, sorted for determinism) and record all of them.
    The ROI masks from every lesion are kept, so the explanation scores use the union of all lesions.
    """
    def _join(s):
        return ";".join(sorted({x for x in s if x}))

    g = lesions.groupby(["patient_id", "laterality", "view"], sort=True)
    img = g.agg(
        label=("malignant", "max"),
        n_lesions=("malignant", "size"),
        n_malignant=("malignant", "sum"),
        abnormality_types=("abnormality_type", _join),
        all_image_paths=("image_file_path", _join),
        roi_mask_paths=("roi_mask_file_path", _join),
        official_splits=("official_split", _join),
        breast_density=("breast_density", "max"),
    ).reset_index()
    img["image_file_path"] = img["all_image_paths"].str.split(";").str[0]
    img.insert(0, "image_id", img["patient_id"] + "_" + img["laterality"] + "_" + img["view"])
    img["label"] = img["label"].astype(int)
    return img


def build_from_config(cfg: dict) -> pd.DataFrame:
    from module3_classifier.config import resolve
    csvs = [resolve(p) for p in cfg["data"]["cbis_csvs"]]
    return build_image_table(load_lesions(csvs))
