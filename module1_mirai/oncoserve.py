"""Run pre-trained Mirai through the OncoServe container (inference only, no retraining).

Start the container (CPU is fine for inference; one exam takes a few seconds):

    docker pull learn2cure/oncoserve_mirai:0.5.0        # ~several GB, warn the team before pulling
    docker run -d --name oncoserve_mirai -p 5000:5000 learn2cure/oncoserve_mirai:0.5.0

Then:

    python -m module1_mirai.oncoserve --config configs/m1_mirai_embed.yaml

The run can be resumed: exams already in the predictions CSV are skipped, so an interrupted run
continues where it stopped.

⚠️ TO VERIFY against the OncoServe README (github.com/yala/OncoServe_Public) before the first real
run: the endpoint path (``/serve``), the multipart field names (``dicom`` files + ``data`` JSON with
``mrn`` / ``accession``), and the response layout. ``parse_response`` accepts the shapes we expect,
and fails loudly on anything else.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd

from module1_mirai.embed_cohort import VIEWS

HORIZONS = [1, 2, 3, 4, 5]
OUT_COLS = ["exam_id"] + [f"risk_{t}y" for t in HORIZONS]

# Keys we expect for each horizon in the response (TO VERIFY). Tried in order.
_HORIZON_KEYS = {
    1: ["Year 1", "1", "pred_one_year", "one_year"],
    2: ["Year 2", "2", "pred_two_year", "two_year"],
    3: ["Year 3", "3", "pred_three_year", "three_year"],
    4: ["Year 4", "4", "pred_four_year", "four_year"],
    5: ["Year 5", "5", "pred_five_year", "five_year"],
}


def parse_response(payload: dict) -> dict[int, float]:
    """Extract {horizon_years: risk} from an OncoServe JSON response.

    Looks for the prediction dict at payload['prediction'], payload['data']['predictions'] or the
    top level. Risks must be in [0, 1] and non-decreasing with horizon (cumulative risk); anything
    else raises, because a silent parsing error would corrupt every metric downstream.
    """
    candidates = [payload.get("prediction"), (payload.get("data") or {}).get("predictions"),
                  payload.get("predictions"), payload]
    for pred in candidates:
        if not isinstance(pred, dict):
            continue
        risks = {}
        for t, keys in _HORIZON_KEYS.items():
            for k in keys:
                if k in pred:
                    risks[t] = float(pred[k])
                    break
        if len(risks) == len(HORIZONS):
            vals = [risks[t] for t in HORIZONS]
            if not all(0.0 <= v <= 1.0 for v in vals):
                raise ValueError(f"risk outside [0,1]: {vals}")
            if any(b < a - 1e-9 for a, b in zip(vals, vals[1:])):
                raise ValueError(f"cumulative risk decreases with horizon: {vals}")
            return risks
    raise ValueError(f"unrecognised OncoServe response: {json.dumps(payload)[:300]}")


def predict_exam(url: str, exam_id: str, dicom_paths: list[str], timeout: float = 300,
                 retries: int = 3, backoff: float = 5.0) -> dict[int, float]:
    """POST the four DICOMs of one exam and return {horizon: risk}. Retries on network errors."""
    import requests  # imported lazily: only needed when the container is running

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        files = []
        try:
            files = [("dicom", (Path(p).name, open(p, "rb"), "application/dicom")) for p in dicom_paths]
            # TO VERIFY: metadata field name and keys.
            data = {"data": json.dumps({"mrn": "anon", "accession": str(exam_id)})}
            r = requests.post(url, files=files, data=data, timeout=timeout)
            r.raise_for_status()
            return parse_response(r.json())
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as err:
            last_err = err
            time.sleep(backoff * attempt)
        finally:
            for _, (_, fh, _) in files:
                fh.close()
    raise RuntimeError(f"exam {exam_id}: OncoServe failed after {retries} tries: {last_err}")


def done_exams(pred_csv: Path) -> set[str]:
    if not pred_csv.exists():
        return set()
    return set(pd.read_csv(pred_csv, usecols=["exam_id"], dtype=str)["exam_id"])


def run(cohort: pd.DataFrame, pred_csv: Path, url: str, dicom_root: str | Path = "",
        **kw) -> tuple[int, int]:
    """Score every exam not yet in ``pred_csv``. Returns (n_scored, n_failed)."""
    pred_csv.parent.mkdir(parents=True, exist_ok=True)
    done = done_exams(pred_csv)
    new_file = not pred_csv.exists()
    failed_log = pred_csv.with_suffix(".failed.csv")
    n_ok = n_fail = 0
    with open(pred_csv, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(OUT_COLS)
        for row in cohort.itertuples(index=False):
            exam_id = str(row.exam_id)
            if exam_id in done:
                continue
            paths = [str(Path(dicom_root) / getattr(row, f"path_{v}")) for v in VIEWS]
            try:
                risks = predict_exam(url, exam_id, paths, **kw)
            except (RuntimeError, ValueError, OSError) as err:
                n_fail += 1
                with open(failed_log, "a") as fl:
                    fl.write(f"{exam_id},{str(err).replace(',', ';')}\n")
                continue
            w.writerow([exam_id] + [risks[t] for t in HORIZONS])
            fh.flush()  # so an interrupted run loses at most one exam
            n_ok += 1
    return n_ok, n_fail


def main(argv=None):
    import yaml
    ap = argparse.ArgumentParser(description="Score EMBED exams with OncoServe Mirai")
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N exams (smoke test)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    cohort = pd.read_csv(cfg["data"]["cohort_csv"], dtype={"exam_id": str})
    if args.limit:
        cohort = cohort.head(args.limit)
    o = cfg["oncoserve"]
    n_ok, n_fail = run(cohort, Path(cfg["data"]["predictions_csv"]), o["url"],
                       cfg["data"].get("dicom_root", ""), timeout=o.get("timeout_s", 300),
                       retries=o.get("retries", 3))
    print(f"scored {n_ok} exams, {n_fail} failed")


if __name__ == "__main__":
    main()
