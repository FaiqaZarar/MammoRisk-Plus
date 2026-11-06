"""Module 1 tests on synthetic EMBED-like tables (no real data needed)."""
import json

import numpy as np
import pandas as pd
import pytest

from module1_mirai import embed_cohort as EC
from module1_mirai import evaluate as EV
from module1_mirai import oncoserve as OS

C = EC.COLUMNS


def _images(exam, views=("L_CC", "L_MLO", "R_CC", "R_MLO"), itype="2D", manuf="HOLOGIC, Inc."):
    rows = []
    for v in views:
        lat, view = v.split("_")
        rows.append({C["exam"]: exam, C["image_type"]: itype, C["view"]: view,
                     C["laterality"]: lat, C["manufacturer"]: manuf,
                     C["path"]: f"{exam}/{v}.dcm"})
    return rows


def _clin(patient, exam, date, desc="MG Screening Bilateral", sev=np.nan, dens=2,
          race="Asian", path_date=None):
    return {C["patient"]: patient, C["exam"]: exam, C["study_date"]: date, C["exam_desc"]: desc,
            C["path_severity"]: sev, C["density"]: dens, C["race"]: race,
            C["path_date"]: path_date}


@pytest.fixture
def tables():
    meta = (
        _images("E1") + _images("E2") + _images("E3")                  # patient P1
        + _images("E4", views=("L_CC", "L_MLO", "R_CC"))                # P2: incomplete
        + _images("E5", itype="cview")                                 # P2: synthetic only
        + _images("E6") + _images("E6", views=("L_CC",))               # P3: duplicate L_CC
        + _images("E7", manuf="GE MEDICAL SYSTEMS")                    # P4
        + _images("E9")                                                 # P5 diagnostic exam
    )
    clin = pd.DataFrame([
        # P1: screens 2015, 2016; cancer diagnosed 2016-06-01 via diagnostic exam; screen 2017 after cancer
        _clin("P1", "E1", "2015-01-01"),
        _clin("P1", "E2", "2016-01-01"),
        _clin("P1", "D1", "2016-05-20", desc="MG Diagnostic", sev=0, path_date="2016-06-01"),
        _clin("P1", "E3", "2017-01-01"),
        _clin("P2", "E4", "2015-01-01", race="White"),
        _clin("P2", "E5", "2016-01-01", race="White"),
        # P3: screen 2015, benign pathology (not an event), last follow-up 2019
        _clin("P3", "E6", "2015-03-01", race="African American  or Black", dens=4),
        _clin("P3", "B1", "2015-04-01", desc="MG Diagnostic", sev=4),
        _clin("P3", "F1", "2019-03-01", desc="MG Screening Bilateral"),
        # P4: single screen, no follow-up → dropped (zero follow-up)
        _clin("P4", "E7", "2018-01-01", race="Unknown, Unavailable or Unreported", dens=np.nan),
        # P5: four views but diagnostic, not screening → dropped
        _clin("P5", "E9", "2018-01-01", desc="MG Diagnostic Left"),
    ])
    return pd.DataFrame(meta), clin


def test_cohort_rules(tables):
    meta, clin = tables
    df, flow = EC.build_cohort(meta, clin)
    got = df.set_index("exam_id")
    # kept: E1, E2 (P1 before cancer), E6 (P3). Dropped: E3 after cancer, E4 incomplete,
    # E5 synthetic, E7 zero follow-up, E9 diagnostic.
    assert sorted(got.index) == ["E1", "E2", "E6"]
    assert flow["excluded_after_first_cancer"] == 1
    assert flow["excluded_zero_followup"] == 1
    assert flow["duplicate_view_images_dropped"] == 1
    # event times from exam to pathology date 2016-06-01
    assert got.loc["E1", "event"] == 1
    assert got.loc["E1", "time_years"] == pytest.approx((pd.Timestamp("2016-06-01") - pd.Timestamp("2015-01-01")).days / 365.25)
    assert got.loc["E2", "time_years"] == pytest.approx(152 / 365.25)
    # benign pathology is censored at last follow-up (2019-03-01)
    assert got.loc["E6", "event"] == 0
    assert got.loc["E6", "time_years"] == pytest.approx(1461 / 365.25)
    assert got.loc["E6", "density"] == "D" and got.loc["E6", "race"] == "Black"
    assert got.loc["E1", "race"] == "Asian" and got.loc["E1", "vendor"] == "Hologic"
    assert got.loc["E6", "path_L_CC"] == "E6/L_CC.dcm"


def test_min_days_sensitivity(tables):
    meta, clin = tables
    df, flow = EC.build_cohort(meta, clin, min_days_to_event=180)
    assert "E2" not in set(df.exam_id)          # cancer 152 days after E2
    assert flow["excluded_cancer_within_min_days"] == 1


def test_vendor_and_race_groups():
    assert EC.vendor_group(pd.Series(["HOLOGIC, Inc.", "GE"])) == "Mixed"
    assert EC.vendor_group(pd.Series(["GE MEDICAL SYSTEMS"])) == "Other"
    assert EC.race_group("Native Hawaiian or Other Pacific Islander") == "Other"
    assert EC.race_group(None) == "Unknown"


def test_parse_response_shapes():
    r = OS.parse_response({"prediction": {"Year 1": 0.01, "Year 2": 0.02, "Year 3": 0.03,
                                          "Year 4": 0.04, "Year 5": 0.05}})
    assert r[5] == 0.05
    r = OS.parse_response({"data": {"predictions": {str(t): t / 100 for t in range(1, 6)}}})
    assert r[1] == 0.01
    with pytest.raises(ValueError):
        OS.parse_response({"prediction": {"Year 1": 0.05, "Year 2": 0.01, "Year 3": 0.03,
                                          "Year 4": 0.04, "Year 5": 0.05}})
    with pytest.raises(ValueError):
        OS.parse_response({"foo": 1})


def _synthetic_cohort(n_pat=300, seed=0):
    rng = np.random.default_rng(seed)
    rows, preds = [], []
    for p in range(n_pat):
        latent = rng.normal()
        race = rng.choice(["Asian", "White", "Black"], p=[0.2, 0.5, 0.3])
        density = rng.choice(list("ABCD"))
        vendor = "Hologic" if rng.random() < 0.8 else "Other"
        for k in range(2):
            ex = f"P{p}_E{k}"
            t_event = rng.exponential(10 / np.exp(1.2 * latent))
            t_cens = rng.uniform(2, 8)
            rows.append({"patient_id": f"P{p}", "exam_id": ex, "time_years": min(t_event, t_cens),
                         "event": int(t_event <= t_cens), "density": density, "race": race,
                         "vendor": vendor})
            base = 1 / (1 + np.exp(-(latent - 4 + rng.normal(0, 0.3))))
            preds.append({"exam_id": ex, **{f"risk_{t}y": min(base * t, 0.99) for t in range(1, 6)}})
    return pd.DataFrame(rows), pd.DataFrame(preds)


def test_evaluate_end_to_end(tmp_path):
    cohort, preds = _synthetic_cohort()
    cohort.to_csv(tmp_path / "cohort.csv", index=False)
    preds.iloc[:-3].to_csv(tmp_path / "preds.csv", index=False)  # 3 exams without predictions
    cfg = {"data": {"cohort_csv": str(tmp_path / "cohort.csv"),
                    "predictions_csv": str(tmp_path / "preds.csv")},
           "evaluation": {"tau": 5.0, "n_boot": 25, "seed": 1, "n_bins": 5, "min_events": 5,
                          "subgroups": ["density", "race", "vendor"]},
           "output": {"results_dir": str(tmp_path / "results"), "run_name": "test"}}
    out = EV.run(cfg)
    m = json.loads((out / "metrics.json").read_text())
    assert m["merge"]["cohort_exams_without_prediction"] == 3
    assert m["overall"]["c_index_ipcw"]["estimate"] > 0.6
    assert m["overall"]["auc_3y"]["ci_lower"] < m["overall"]["auc_3y"]["estimate"]
    assert "oe_ratio" in m["overall"]["calibration_1y"]
    race = pd.read_csv(out / "subgroups_race.csv")
    assert set(race["race"]) == {"Asian", "White", "Black"}
    assert "c_index_ipcw_estimate" in race.columns
    assert (out / "calibration_5y.csv").exists() or "calibration_5y" not in m["overall"]


def test_oncoserve_resume(tmp_path, monkeypatch):
    cohort, _ = _synthetic_cohort(n_pat=3)
    pred_csv = tmp_path / "p.csv"
    pd.DataFrame([{"exam_id": "P0_E0", **{f"risk_{t}y": 0.1 for t in range(1, 6)}}]).to_csv(pred_csv, index=False)
    for v in EC.VIEWS:
        cohort[f"path_{v}"] = "x.dcm"
    calls = []
    monkeypatch.setattr(OS, "predict_exam",
                        lambda url, ex, paths, **kw: calls.append(ex) or {t: t / 100 for t in range(1, 6)})
    n_ok, n_fail = OS.run(cohort, pred_csv, "http://x")
    assert n_ok == len(cohort) - 1 and "P0_E0" not in calls
    assert len(pd.read_csv(pred_csv)) == len(cohort)
