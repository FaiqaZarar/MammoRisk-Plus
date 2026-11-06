"""70/10/20 patient-level split of CBIS-DDSM full mammograms.

Why patient-level: the left and right breasts and the CC/MLO views of one woman look alike. If
images from one patient fall in both train and test, the test AUC measures memorisation of that
patient, not generalisation (data leakage).

Why stratified by *patient* label: a patient is malignant if any of her images is. Stratifying
keeps the malignant share about equal in train / val / test, so val AUC (used for early stopping and
the Youden threshold) is not computed on an unrepresentative sample.

Usage:  python -m module3_classifier.splits --config configs/m3_effnetb3.yaml
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from module3_classifier.config import load_config, resolve


def patient_split(images: pd.DataFrame, fractions=(0.7, 0.1, 0.2), seed: int = 42) -> pd.DataFrame:
    """Return ``images`` with a ``split`` column in {train, val, test}."""
    f_train, f_val, f_test = fractions
    if abs(f_train + f_val + f_test - 1) > 1e-9:
        raise ValueError("fractions must sum to 1")
    patients = images.groupby("patient_id")["label"].max().reset_index()
    trval, test = train_test_split(patients, test_size=f_test, stratify=patients["label"],
                                   random_state=seed)
    # val share of the remaining (train+val) pool: 0.1 / 0.8 = 0.125
    train, val = train_test_split(trval, test_size=f_val / (f_train + f_val),
                                  stratify=trval["label"], random_state=seed)
    split_of = {**{p: "train" for p in train.patient_id}, **{p: "val" for p in val.patient_id},
                **{p: "test" for p in test.patient_id}}
    out = images.copy()
    out["split"] = out["patient_id"].map(split_of)
    assert_no_leakage(out)
    return out


def assert_no_leakage(df: pd.DataFrame) -> None:
    per_patient = df.groupby("patient_id")["split"].nunique()
    leaked = per_patient[per_patient > 1]
    if len(leaked):
        raise AssertionError(f"{len(leaked)} patients appear in more than one split: "
                             f"{list(leaked.index[:5])}")
    if df["split"].isna().any():
        raise AssertionError("Some images have no split")


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("split").agg(patients=("patient_id", "nunique"), images=("image_id", "size"),
                                   malignant_images=("label", "sum"),
                                   malignant_share=("label", "mean"))


def main(argv=None):
    from module3_classifier.cbis import build_from_config
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out_path = resolve(cfg["paths"]["split_csv"])
    if out_path.exists() and not cfg.get("overwrite_split", False):
        # Splits are saved once and reused by every experiment (CLAUDE.md).
        print(f"{out_path} exists; reusing it. Set overwrite_split: true to regenerate.")
        return
    images = build_from_config(cfg)
    df = patient_split(images, tuple(cfg["split"]["fractions"]), cfg["seed"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(summarise(df))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
