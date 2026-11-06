"""Evaluate a trained Module 3 checkpoint (O2).

Usage
  python -m module3_classifier.evaluate run --config configs/m3_effnetb3.yaml --checkpoint results/m3/<run>/best.pt
  python -m module3_classifier.evaluate compare --pred-a A/predictions_test.csv --pred-b B/predictions_test.csv

Protocol (CLAUDE.md no-leakage rule)
1. Predict on val and test.
2. Choose the operating threshold on **val** with Youden's J, and freeze it.
3. Apply it **once** to test: sensitivity, specificity, PPV, NPV, F1.
4. Test AUC with a DeLong 95% CI, and also a patient-clustered bootstrap CI. Several images per
   woman are correlated, so DeLong (which assumes independent images) can be slightly too narrow.
   We report both.
5. External sets (CMMD, DMID without normals): AUC + CI and the drop vs CBIS-DDSM test. Machine and
   population are confounded, so the drop cannot be attributed to population alone.
6. ``compare``: paired DeLong test between two models on the same test images.

Every output JSON records the dataset, split, checkpoint and config (CLAUDE.md results rule).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.bootstrap import bootstrap_ci
from evaluation.classification import auc_delong, delong_test, threshold_metrics, youden_threshold
from module3_classifier.config import load_config, resolve


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        raise ValueError("one class only")
    return roc_auc_score(y, p)


def summarise_predictions(pred: pd.DataFrame, threshold: float | None, n_boot: int = 1000,
                          seed: int = 42) -> dict:
    """AUC (DeLong + patient-clustered bootstrap) and, if given, metrics at a frozen threshold."""
    y, p = pred["label"].to_numpy(), pred["prob"].to_numpy()
    res = {"n_images": int(len(pred)), "n_patients": int(pred["patient_id"].nunique()),
           "n_malignant": int(y.sum())}
    res.update(auc_delong(y, p).as_dict("auc_delong_"))
    res.update(bootstrap_ci(_auc, y, p, n_boot=n_boot, seed=seed,
                            clusters=pred["patient_id"].to_numpy()).as_dict("auc_bootstrap_"))
    if threshold is not None:
        res["at_threshold"] = threshold_metrics(y, p, threshold).as_dict()
    return res


def _predict_df(model, df: pd.DataFrame, root, cfg, device, path_col="cache_path") -> pd.DataFrame:
    from torch.utils.data import DataLoader
    from module3_classifier.datasets import MammoDataset, build_transform
    from module3_classifier.train import predict
    ds = MammoDataset(df, root, build_transform(False), path_col=path_col)
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                    num_workers=cfg["train"].get("num_workers", 2))
    prob, idx = predict(model, dl, device, cfg["train"].get("amp", True))
    out = ds.df.iloc[idx][["image_id", "patient_id", "label"]].copy()
    out["prob"] = prob
    return out.reset_index(drop=True)


def load_checkpoint(path, device):
    import torch
    from module3_classifier.models import build_model
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["model_name"], pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def run(cfg: dict, checkpoint: str) -> dict:
    import torch
    from module3_classifier.train import load_manifest
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(checkpoint, device)
    out_dir = Path(checkpoint).parent
    man = load_manifest(cfg)
    cache = resolve(cfg["paths"]["cache_dir"])
    n_boot, seed = cfg["eval"]["n_boot"], cfg["seed"]

    preds = {s: _predict_df(model, man[man.split == s], cache, cfg, device) for s in ("val", "test")}
    for s, d in preds.items():
        d.to_csv(out_dir / f"predictions_{s}.csv", index=False)

    thr = youden_threshold(preds["val"]["label"], preds["val"]["prob"])  # VAL only
    report = {
        "checkpoint": str(checkpoint), "model": ck["model_name"], "best_epoch": ck["epoch"],
        "config": cfg["_config_path"], "config_hash": cfg["_config_hash"],
        "split_csv": cfg["paths"]["split_csv"],
        "threshold": {"value": thr, "chosen_on": "CBIS-DDSM val", "method": "Youden J"},
        "CBIS-DDSM/val": summarise_predictions(preds["val"], thr, n_boot, seed),
        "CBIS-DDSM/test": summarise_predictions(preds["test"], thr, n_boot, seed),
    }
    test_auc = report["CBIS-DDSM/test"]["auc_delong_estimate"]

    for name, ext in (cfg.get("external") or {}).items():
        if ext.get("explain_only"):  # VinDr: no biopsy labels, so no AUC (CLAUDE.md)
            continue
        idx = resolve(ext.get("index_csv"))
        if not idx or not idx.exists():
            report[f"{name}/external"] = {"skipped": f"index not found: {idx}"}
            continue
        df = pd.read_csv(idx)
        if "is_normal" in df.columns and not ext.get("include_normals", False):
            df = df[~df["is_normal"].astype(bool)]   # DMID: 2-class AUC excludes normals
        pe = _predict_df(model, df.reset_index(drop=True), resolve(ext.get("root")), cfg, device,
                         path_col=ext.get("path_col", "cache_path"))
        pe.to_csv(out_dir / f"predictions_{name}.csv", index=False)
        r = summarise_predictions(pe, thr, n_boot, seed)
        r["auc_drop_vs_cbis_test"] = test_auc - r["auc_delong_estimate"]
        r["note"] = "machine type and population are confounded"
        report[f"{name}/external"] = r

    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def compare(pred_a: str, pred_b: str) -> dict:
    """Paired DeLong test on the images both prediction files share (must be the same test set)."""
    a, b = pd.read_csv(pred_a), pd.read_csv(pred_b)
    m = a.merge(b, on=["image_id", "label"], suffixes=("_a", "_b"))
    if len(m) != len(a) or len(m) != len(b):
        raise ValueError(f"Prediction files cover different images ({len(a)}, {len(b)}, "
                         f"shared {len(m)}). DeLong needs the same test set.")
    t = delong_test(m["label"], m["prob_a"], m["prob_b"])
    return {"pred_a": pred_a, "pred_b": pred_b, "n": len(m), **t.__dict__}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate Module 3")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--checkpoint", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--pred-a", required=True)
    c.add_argument("--pred-b", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        res = run(load_config(args.config), args.checkpoint)
    else:
        res = compare(args.pred_a, args.pred_b)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
