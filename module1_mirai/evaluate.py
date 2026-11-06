"""Evaluate pre-trained Mirai on the EMBED cohort, overall and by density, race and vendor (O4).

    python -m module1_mirai.evaluate --config configs/m1_mirai_embed.yaml

Metrics (definitions in ``evaluation/survival.py``):
* Uno's IPCW C-index truncated at τ = 5 years, computed with Mirai's **5-year** risk. This follows
  Yala et al. (2021/2022): one risk score is needed for ranking, and the 5-year risk spans the whole
  follow-up window.
* Time-dependent (cumulative/dynamic) AUC at 1–5 years, each with Mirai's risk for that horizon.
* Calibration at each horizon: decile table + observed/expected ratio.
* 95% CIs: bootstrap with 1,000 resamples **clustered by patient**, because one woman contributes
  several exams.

There is no train/test split. The whole EMBED open subset is an external test set, because Mirai is
not retrained (DECISIONS D-07). Subgroups with fewer than ``min_events`` cancers are flagged
``exploratory_only`` and get counts but no CI.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import survival as S

HORIZONS = [1, 2, 3, 4, 5]
SUBGROUPS = ["density", "race", "vendor"]


def merge_predictions(cohort: pd.DataFrame, preds: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cohort = cohort.astype({"exam_id": str})
    preds = preds.astype({"exam_id": str})
    df = cohort.merge(preds, on="exam_id", how="inner")
    info = {"cohort_exams": len(cohort), "predicted_exams": len(preds), "merged_exams": len(df),
            "cohort_exams_without_prediction": int((~cohort.exam_id.isin(preds.exam_id)).sum())}
    return df, info


def _safe(fn, *a, **k):
    """Return a CI dict, or an error note when the metric is undefined (e.g. no event before t)."""
    try:
        return fn(*a, **k).as_dict()
    except ValueError as err:
        return {"error": str(err)}


def overall_metrics(df: pd.DataFrame, tau: float, n_boot: int, seed: int, n_bins: int) -> tuple[dict, dict]:
    t, e, pid = df["time_years"].to_numpy(), df["event"].to_numpy(), df["patient_id"].to_numpy()
    res = {"n_exams": len(df), "n_patients": int(df.patient_id.nunique()), "n_events": int(e.sum()),
           "c_index_ipcw": _safe(S.c_index_ipcw_ci, t, e, df["risk_5y"], tau, pid, n_boot, seed),
           "c_index_harrell": S.c_index_harrell(t, e, df["risk_5y"])}
    calib = {}
    for h in HORIZONS:
        res[f"auc_{h}y"] = _safe(S.td_auc_ci, t, e, df[f"risk_{h}y"], h, pid, n_boot, seed)
        res[f"events_within_{h}y"] = int(((t <= h) & (e == 1)).sum())
        if (t >= h).any():
            cal = S.calibration_at(t, e, df[f"risk_{h}y"], h, n_bins)
            res[f"calibration_{h}y"] = {"expected": cal.expected, "observed": cal.observed,
                                        "oe_ratio": cal.oe_ratio}
            calib[h] = cal.table
    return res, calib


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(cfg: dict, config_path: str = "<dict>") -> Path:
    d, ev = cfg["data"], cfg["evaluation"]
    cohort = pd.read_csv(d["cohort_csv"], dtype={"exam_id": str, "patient_id": str})
    preds = pd.read_csv(d["predictions_csv"], dtype={"exam_id": str})
    df, merge_info = merge_predictions(cohort, preds)
    if df.empty:
        raise SystemExit("No exams with predictions; run module1_mirai.oncoserve first.")

    tau, n_boot, seed = ev.get("tau", 5.0), ev.get("n_boot", 1000), ev.get("seed", 42)
    out = Path(cfg["output"]["results_dir"]) / cfg["output"].get("run_name", "m1_mirai_embed")
    out.mkdir(parents=True, exist_ok=True)

    overall, calib = overall_metrics(df, tau, n_boot, seed, ev.get("n_bins", 10))
    for h, tab in calib.items():
        tab.to_csv(out / f"calibration_{h}y.csv", index=False)

    risk_cols = {float(h): f"risk_{h}y" for h in HORIZONS}
    for by in ev.get("subgroups", SUBGROUPS):
        tab = S.subgroup_table(df, by, "time_years", "event", risk_cols, tau=tau,
                               cluster_col="patient_id", n_boot=n_boot, seed=seed,
                               min_events=ev.get("min_events", 5))
        tab.to_csv(out / f"subgroups_{by}.csv", index=False)

    meta = {"module": "M1 Mirai (pre-trained, OncoServe 0.5.0)", "objective": "O4",
            "dataset": cfg.get("dataset", "EMBED open 20% subset"),
            "split": "external validation, whole cohort (no split; Mirai not retrained, D-07)",
            "config_path": config_path, "config": cfg, "git_commit": git_commit(),
            "date": dt.datetime.now().isoformat(timespec="seconds"), "merge": merge_info,
            "ci_method": f"percentile bootstrap, {n_boot} resamples clustered by patient, seed {seed}",
            "overall": overall}
    (out / "metrics.json").write_text(json.dumps(meta, indent=2, default=_json_default))
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def main(argv=None):
    import yaml
    ap = argparse.ArgumentParser(description="Evaluate Mirai on EMBED")
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = run(cfg, args.config)
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
