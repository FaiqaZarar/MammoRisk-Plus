"""Module 3 data tests on synthetic CBIS-DDSM-style CSVs. They run without torch; torch-dependent
tests are skipped when torch is missing."""
import numpy as np
import pandas as pd
import pytest

from module3_classifier import cbis, preprocess, splits
from module3_classifier.train import class_weights


def _write_cbis(tmp_path, n_patients=200, seed=0):
    """Mass and calc CSVs in the official column style (spaces, mixed case)."""
    rng = np.random.default_rng(seed)
    mass, calc = [], []
    for k in range(n_patients):
        pid = f"P_{k:05d}"
        for side in ("LEFT", "RIGHT"):
            for view in ("CC", "MLO"):
                if rng.random() < 0.5:
                    continue
                path = rng.choice(["MALIGNANT", "BENIGN", "BENIGN_WITHOUT_CALLBACK"])
                row = {"patient_id": pid, "breast_density": 2, "left or right breast": side,
                       "image view": view, "abnormality id": 1, "pathology": path,
                       "image file path": f"Mass-Training_{pid}_{side}_{view}/1.2/1.3/000000.dcm\n",
                       "ROI mask file path": f"Mass-Training_{pid}_{side}_{view}_1/1.2/1.4/000001.dcm"}
                mass.append(row)
                if rng.random() < 0.2:  # same view also in calc file, with a second lesion
                    calc.append({**row, "breast density": 2, "pathology": "BENIGN",
                                 "image file path": f"Calc-Test_{pid}_{side}_{view}/9/9/000000.dcm"})
    m = pd.DataFrame(mass)
    c = pd.DataFrame(calc).drop(columns=["breast_density"]) if calc else pd.DataFrame()
    mp, cp = tmp_path / "mass_case_description_train_set.csv", tmp_path / "calc_case_description_test_set.csv"
    m.to_csv(mp, index=False)
    c.to_csv(cp, index=False)
    return [mp, cp]


def test_normalise_patient_id():
    assert cbis.normalise_patient_id("Mass-Training_P_00001_LEFT_CC") == "P_00001"
    assert cbis.normalise_patient_id("P_01234") == "P_01234"
    with pytest.raises(ValueError):
        cbis.normalise_patient_id("patient7")


def test_image_label_any_malignant():
    les = pd.DataFrame({
        "patient_id": ["P_00001"] * 3 + ["P_00002"],
        "laterality": ["LEFT", "LEFT", "RIGHT", "LEFT"], "view": ["CC", "CC", "CC", "MLO"],
        "pathology": ["BENIGN", "MALIGNANT", "BENIGN_WITHOUT_CALLBACK", "BENIGN"],
        "malignant": [0, 1, 0, 0], "abnormality_type": ["mass", "calcification", "mass", "mass"],
        "image_file_path": ["a", "b", "c", "d"], "roi_mask_file_path": ["m1", "m2", "m3", "m4"],
        "official_split": ["train"] * 4, "breast_density": [2, 2, 3, 1]})
    img = cbis.build_image_table(les).set_index("image_id")
    assert img.loc["P_00001_LEFT_CC", "label"] == 1
    assert img.loc["P_00001_LEFT_CC", "n_lesions"] == 2
    assert img.loc["P_00001_LEFT_CC", "roi_mask_paths"] == "m1;m2"
    assert img.loc["P_00001_RIGHT_CC", "label"] == 0     # BENIGN_WITHOUT_CALLBACK = benign
    assert img.loc["P_00002_LEFT_MLO", "label"] == 0


def test_load_lesions_normalises_and_merges(tmp_path):
    les = cbis.load_lesions(_write_cbis(tmp_path))
    assert {"laterality", "view", "image_file_path", "breast_density"} <= set(les.columns)
    assert not les["image_file_path"].str.contains("\n").any()
    img = cbis.build_image_table(les)
    assert img["image_id"].is_unique
    # images that appear in both mass and calc files are merged into one row
    assert (img["all_image_paths"].str.count(";") >= 1).any()


def test_unknown_pathology_rejected(tmp_path):
    p = tmp_path / "mass_case_description_train_set.csv"
    pd.DataFrame({"patient_id": ["P_00001"], "left or right breast": ["LEFT"], "image view": ["CC"],
                  "pathology": ["UNKNOWN"], "image file path": ["x"]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        cbis.load_lesions([p])


def test_patient_split_proportions_leakage_and_stratification(tmp_path):
    img = cbis.build_image_table(cbis.load_lesions(_write_cbis(tmp_path, n_patients=400)))
    df = splits.patient_split(img, seed=42)
    splits.assert_no_leakage(df)
    pat = df.groupby("patient_id").agg(split=("split", "first"), label=("label", "max"))
    share = pat["split"].value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.7, abs=0.02)
    assert share["val"] == pytest.approx(0.1, abs=0.02)
    assert share["test"] == pytest.approx(0.2, abs=0.02)
    rate = pat.groupby("split")["label"].mean()
    assert rate.max() - rate.min() < 0.05          # stratified by patient label
    again = splits.patient_split(img, seed=42)
    assert (again["split"] == df["split"]).all()    # deterministic with fixed seed


def test_leakage_assertion_fires():
    bad = pd.DataFrame({"patient_id": ["P_1", "P_1"], "split": ["train", "test"], "image_id": ["a", "b"]})
    with pytest.raises(AssertionError):
        splits.assert_no_leakage(bad)


def test_class_weights_balanced():
    w = class_weights(np.array([0] * 75 + [1] * 25))
    assert w[1] / w[0] == pytest.approx(3.0)
    assert (w * np.array([75, 25])).sum() == pytest.approx(100)


def test_orientation_and_box_transform():
    img = np.zeros((40, 20)); img[:, 15:] = 1.0       # breast on the right
    assert preprocess.breast_on_right(img)
    t = preprocess.Transform(orig_h=40, orig_w=20, flipped=True, scale=0.5, out_h=20, out_w=10)
    (x0, y0, x1, y1), = preprocess.transform_boxes([(15, 4, 20, 8)], t)
    assert (x0, y0, x1, y1) == (0.0, 2.0, 2.5, 4.0)


def test_dmid_parser(tmp_path):
    from module3_classifier.datasets import load_dmid
    info = tmp_path / "info.txt"
    info.write_text("IMG001 F CIRC M 100 200 30\nIMG001 F CALC B 50 60 10\n"
                    "IMG002 G NORM\nIMG003 D SPIC B 10 10 5\n")
    d = load_dmid(info, tmp_path)
    assert set(d.image_id) == {"IMG001", "IMG003"}           # normal excluded
    row = d.set_index("image_id").loc["IMG001"]
    assert row.label == 1 and len(row.boxes) == 2
    assert row.boxes[0] == (70.0, 170.0, 130.0, 230.0)


def test_vindr_boxes(tmp_path):
    from module3_classifier.datasets import load_vindr_boxes
    p = tmp_path / "finding_annotations.csv"
    pd.DataFrame({"study_id": ["s1", "s1", "s2"], "image_id": ["i1", "i1", "i2"],
                  "laterality": ["L", "L", "R"], "view_position": ["CC", "CC", "MLO"],
                  "height": [100] * 3, "width": [80] * 3,
                  "finding_categories": ["['Mass']", "['Suspicious Calcification']", "['Mass']"],
                  "finding_birads": ["BI-RADS 4", "BI-RADS 5", "BI-RADS 3"],
                  "xmin": [1, 2, 3], "ymin": [1, 2, 3], "xmax": [5, 6, 7], "ymax": [5, 6, 7]}
                 ).to_csv(p, index=False)
    v = load_vindr_boxes(p)
    assert list(v.image_id) == ["i1"] and len(v.boxes[0]) == 2   # BI-RADS 3 dropped


# ---------------- torch-dependent ------------------------------------------------------------
def test_models_build_and_target_layer():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from module3_classifier.models import build_model, cam_target_layer, randomisation_order
    for name in ("resnet50", "efficientnet_b3"):
        m = build_model(name, pretrained=False)
        assert m(torch.zeros(1, 3, 64, 32)).shape == (1, 2)
        assert cam_target_layer(m, name) is not None
        assert len(randomisation_order(m)) > 3
