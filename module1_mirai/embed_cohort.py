"""Build the exam-level survival cohort for Mirai external validation on EMBED (O4).

One row per **screening exam** that has all four 2D FFDM views (L-CC, L-MLO, R-CC, R-MLO), with
time to the woman's first later pathology-confirmed cancer (event = 1) or to her last follow-up
(event = 0, censored).

Why these rules (methodology notes for the thesis)
---------------------------------------------------
* **2D only.** Mirai was trained on 2D FFDM. EMBED also holds synthetic 2D views reconstructed from
  tomosynthesis ("C-View") and 3D slices. These look different and would be out of distribution, so
  they are dropped (CLAUDE.md Module 1).
* **Four views of one exam.** Mirai's aggregator combines the four standard views of a single exam.
  It does not use prior exams, so each exam is an independent prediction (DECISIONS D-08).
* **Outcome = first pathology-confirmed cancer** (invasive or DCIS). Benign or high-risk pathology
  is not an event.
* **Exams after the first cancer are excluded.** After diagnosis a woman is no longer a screening
  patient, and her later images show treatment changes.
* **Cancers diagnosed on/after the exam date count**, including a same-day diagnosis (a cancer
  visible at this screen is within Mirai's 1-year horizon). ``min_days_to_event`` can exclude
  cancers diagnosed within N days of the index exam, as a sensitivity analysis: those cancers were
  probably already visible, so this tests risk prediction rather than detection.
* **Censoring at last follow-up** = the woman's last exam of any type in EMBED. Exams with zero
  follow-up and no event carry no information and are dropped (``drop_zero_followup``).
* **Multiple exams per woman** are all kept. CIs are clustered by patient in ``evaluate.py``.

All column names live in ``COLUMNS``. ⚠️ **Verify them against the EMBED open-data dictionary**
before the first real run; they are written from the EMBED paper and README, not from the files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ⚠️ TO VERIFY against the EMBED open-data data dictionary (EMBED_OpenData_*.csv).
COLUMNS = {
    # shared keys
    "patient": "empi_anon",
    "exam": "acc_anon",
    "study_date": "study_date_anon",
    # metadata (one row per image)
    "image_type": "FinalImageType",        # '2D', 'cview' (synthetic), '3D' ...
    "view": "ViewPosition",                # 'CC', 'MLO', (also 'ML', 'XCCL', ... → dropped)
    "laterality": "ImageLateralityFinal",  # 'L', 'R'
    "manufacturer": "Manufacturer",
    "path": "anon_dicom_path",
    # clinical (one row per exam finding/side)
    "exam_desc": "desc",                   # free text; screening exams contain 'screen'
    "path_severity": "path_severity",      # 0 invasive, 1 DCIS, 2 high-risk, 3 borderline, 4 benign, 5 non-breast
    "path_date": "procdate_anon",          # date of the pathology procedure; falls back to study date
    "density": "tissueden",                # BI-RADS density 1–4
    "race": "ETHNICITY_DESC",
}

MALIGNANT_SEVERITY = {0, 1}  # invasive + DCIS = cancer event
VIEWS = ["L_CC", "L_MLO", "R_CC", "R_MLO"]
DENSITY_MAP = {1: "A", 2: "B", 3: "C", 4: "D"}  # A fatty … D extremely dense; anything else → Unknown


def race_group(value) -> str:
    """Collapse EMBED race/ethnicity text into reporting groups.

    'Asian' in EMBED is any Asian background (mostly East Asian in a US cohort), **not** South Asian,
    so it is only a proxy for our population. Say so whenever this subgroup is reported.
    """
    if not isinstance(value, str) or not value.strip():
        return "Unknown"
    v = value.lower()
    if "unknown" in v or "unreported" in v or "unavailable" in v or "declined" in v:
        return "Unknown"
    if "asian" in v:
        return "Asian"
    if "black" in v or "african" in v:
        return "Black"
    if "white" in v or "caucasian" in v:
        return "White"
    return "Other"


def vendor_group(manufacturers: pd.Series) -> str:
    """Hologic vs other. Mirai was trained on Hologic images, so vendor shift is a key subgroup."""
    vals = {str(m).upper() for m in manufacturers.dropna()}
    if not vals:
        return "Unknown"
    hologic = {("HOLOGIC" in v) for v in vals}
    if hologic == {True}:
        return "Hologic"
    if hologic == {False}:
        return "Other"
    return "Mixed"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def select_four_view_exams(meta: pd.DataFrame, flow: dict) -> pd.DataFrame:
    """Keep 2D CC/MLO images and exams that have exactly the four standard views.

    Duplicate images for the same view (re-takes) → keep the first by sorted file path. This is
    deterministic. Mirai has no rule for choosing between repeats, and this keeps one per view.
    """
    c = COLUMNS
    m = meta.copy()
    flow["images_total"] = len(m)
    m = m[m[c["image_type"]].astype(str).str.upper() == "2D"]
    flow["images_2d"] = len(m)
    m = m[m[c["view"]].isin(["CC", "MLO"]) & m[c["laterality"]].isin(["L", "R"])]
    flow["images_2d_cc_mlo"] = len(m)
    m["view_key"] = m[c["laterality"]] + "_" + m[c["view"]]
    m = m.sort_values([c["exam"], "view_key", c["path"]])
    flow["duplicate_view_images_dropped"] = int(m.duplicated([c["exam"], "view_key"]).sum())
    m = m.drop_duplicates([c["exam"], "view_key"], keep="first")

    wide = m.pivot(index=c["exam"], columns="view_key", values=c["path"])
    wide = wide.reindex(columns=VIEWS)
    complete = wide.dropna()
    flow["exams_with_any_2d_view"] = len(wide)
    flow["exams_with_all_4_views"] = len(complete)
    complete.columns = [f"path_{v}" for v in complete.columns]
    vendor = m.groupby(c["exam"])[c["manufacturer"]].agg(vendor_group).rename("vendor")
    manuf = m.groupby(c["exam"])[c["manufacturer"]].agg(
        lambda s: ";".join(sorted({str(x) for x in s.dropna()}))).rename("manufacturer")
    return complete.join(vendor).join(manuf).reset_index().rename(columns={c["exam"]: "exam_id"})


def build_cohort(meta: pd.DataFrame, clinical: pd.DataFrame, min_days_to_event: int = 0,
                 drop_zero_followup: bool = True) -> tuple[pd.DataFrame, dict]:
    """Return (cohort, flow). ``flow`` counts every exclusion step for the CONSORT-style diagram."""
    c = COLUMNS
    flow: dict = {}
    clin = clinical.copy()
    clin["_date"] = _to_date(clin[c["study_date"]])
    path_date = _to_date(clin[c["path_date"]]) if c["path_date"] in clin else pd.Series(pd.NaT, index=clin.index)
    clin["_path_date"] = path_date.fillna(clin["_date"])

    # --- first cancer date per woman --------------------------------------------------------
    sev = pd.to_numeric(clin[c["path_severity"]], errors="coerce")
    malignant = clin[sev.isin(MALIGNANT_SEVERITY)]
    first_cancer = malignant.groupby(c["patient"])["_path_date"].min().rename("first_cancer_date")
    # --- last follow-up per woman (any exam type) -------------------------------------------
    last_followup = clin.groupby(c["patient"])["_date"].max().rename("last_followup_date")

    # --- exam-level clinical attributes ------------------------------------------------------
    exams = clin.groupby(c["exam"]).agg(
        patient_id=(c["patient"], "first"),
        exam_date=("_date", "min"),
        exam_desc=(c["exam_desc"], "first"),
        density_raw=(c["density"], lambda s: pd.to_numeric(s, errors="coerce").dropna().iloc[0]
                     if pd.to_numeric(s, errors="coerce").notna().any() else np.nan),
        race_raw=(c["race"], "first"),
    ).reset_index().rename(columns={c["exam"]: "exam_id"})
    exams["is_screening"] = exams["exam_desc"].astype(str).str.contains("screen", case=False)
    flow["clinical_exams"] = len(exams)
    exams = exams[exams["is_screening"]]
    flow["screening_exams"] = len(exams)

    views = select_four_view_exams(meta, flow)
    df = exams.merge(views, on="exam_id", how="inner")
    flow["screening_exams_with_4_views"] = len(df)

    df = df.merge(first_cancer, left_on="patient_id", right_index=True, how="left")
    df = df.merge(last_followup, left_on="patient_id", right_index=True, how="left")

    # Exclude exams after the woman's first cancer (no longer screening).
    after = df["first_cancer_date"].notna() & (df["exam_date"] > df["first_cancer_date"])
    flow["excluded_after_first_cancer"] = int(after.sum())
    df = df[~after].copy()

    days_to_cancer = (df["first_cancer_date"] - df["exam_date"]).dt.days
    # Sensitivity analysis: drop exams whose cancer came within `min_days_to_event` days.
    too_soon = days_to_cancer.notna() & (days_to_cancer < min_days_to_event)
    flow["excluded_cancer_within_min_days"] = int(too_soon.sum())
    df = df[~too_soon].copy()
    days_to_cancer = days_to_cancer[~too_soon]

    df["event"] = days_to_cancer.notna().astype(int)
    followup_days = (df["last_followup_date"] - df["exam_date"]).dt.days
    days = np.where(df["event"] == 1, days_to_cancer, followup_days)
    # A same-day diagnosis gives time 0; use half a day so survival software keeps it as an event
    # occurring right after the exam (sksurv/lifelines treat t=0 inconsistently).
    days = np.where((df["event"] == 1) & (days <= 0), 0.5, days)
    df["time_years"] = days / 365.25

    if drop_zero_followup:
        zero = (df["event"] == 0) & (df["time_years"] <= 0)
        flow["excluded_zero_followup"] = int(zero.sum())
        df = df[~zero]

    df["density"] = df["density_raw"].map(DENSITY_MAP).fillna("Unknown")
    df["race"] = df["race_raw"].map(race_group)
    out_cols = ["patient_id", "exam_id", "exam_date"] + [f"path_{v}" for v in VIEWS] + [
        "time_years", "event", "density", "race", "vendor", "manufacturer"]
    df = df[out_cols].sort_values(["patient_id", "exam_date"]).reset_index(drop=True)
    flow.update(final_exams=len(df), final_patients=int(df["patient_id"].nunique()),
                final_events=int(df["event"].sum()),
                final_patients_with_event=int(df.loc[df.event == 1, "patient_id"].nunique()))
    return df, flow


def main(argv=None):
    import yaml
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    d, co = cfg["data"], cfg["cohort"]
    meta = pd.read_csv(d["embed_metadata_csv"], low_memory=False)
    clinical = pd.read_csv(d["embed_clinical_csv"], low_memory=False)
    cohort, flow = build_cohort(meta, clinical, co.get("min_days_to_event", 0),
                                co.get("drop_zero_followup", True))
    out = Path(d["cohort_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(out, index=False)
    out.with_suffix(".flow.json").write_text(json.dumps(flow, indent=2))
    print(json.dumps(flow, indent=2))


if __name__ == "__main__":
    main()
