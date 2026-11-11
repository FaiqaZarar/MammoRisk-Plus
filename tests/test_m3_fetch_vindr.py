import hashlib
from pathlib import Path

import pandas as pd

from module3_classifier import fetch_vindr as V


def _write_csvs(root):
    # image a: kept mass BI-RADS 4; b: calcification BI-RADS 3 (too low); c: kept calc BI-RADS 5 (other exam);
    # d: no finding; e: asymmetry BI-RADS 4 (wrong category). Exam S1 also has an unboxed view "f".
    rows = [
        ("S1", "a", "L", "CC", "['Mass']", "BI-RADS 4", 1, 2, 3, 4),
        ("S1", "b", "L", "MLO", "['Suspicious Calcification']", "BI-RADS 3", 1, 2, 3, 4),
        ("S2", "c", "R", "CC", "['Suspicious Calcification']", "BI-RADS 5", 1, 2, 3, 4),
        ("S3", "d", "R", "MLO", "['No Finding']", None, None, None, None, None),
        ("S4", "e", "L", "CC", "['Focal Asymmetry']", "BI-RADS 4", 1, 2, 3, 4),
    ]
    f = pd.DataFrame(rows, columns=["study_id", "image_id", "laterality", "view_position",
                                    "finding_categories", "finding_birads", "xmin", "ymin", "xmax", "ymax"])
    f["height"], f["width"] = 100, 50
    f.to_csv(root / "finding_annotations.csv", index=False)
    b = pd.DataFrame({"study_id": ["S1", "S1", "S1", "S2", "S3", "S4"],
                      "image_id": ["a", "b", "f", "c", "d", "e"]})
    b.to_csv(root / "breast-level_annotations.csv", index=False)


def test_select_only_boxed_images(tmp_path):
    _write_csvs(tmp_path)
    assert sorted(V.select_images(tmp_path).image_id) == ["a", "c"]


def test_select_all_views_keeps_whole_exam(tmp_path):
    _write_csvs(tmp_path)
    assert sorted(V.select_images(tmp_path, all_views=True).image_id) == ["a", "b", "c", "f"]


def test_parse_sha256sums():
    text = "AAA  images/s/x.dicom\nbbb *./images/s/y.dicom\n\nbad\n"
    assert V.parse_sha256sums(text) == {"images/s/x.dicom": "aaa", "images/s/y.dicom": "bbb"}


def test_wsl_path():
    assert V.wsl_path(Path("D:/mammorisk_data/vindr_mammo")) == "/mnt/d/mammorisk_data/vindr_mammo"
    assert V.wsl_path(Path("D:\\x\\y")) == "/mnt/d/x/y"


def test_url_list_and_wget_command(tmp_path, capsys):
    _write_csvs(tmp_path)
    assert V.main(["--root", str(tmp_path), "--user", "me"]) == 0
    urls = (tmp_path / "subset_urls.txt").read_text().splitlines()
    assert urls == [V.BASE + "images/S1/a.dicom", V.BASE + "images/S2/c.dicom"]
    out = capsys.readouterr().out
    assert "2 images in 2 exams" in out and "--user me --ask-password" in out
    assert "--password" not in out.replace("--ask-password", "")   # the script never takes a password


def test_verify_reports_ok_missing_and_bad(tmp_path):
    _write_csvs(tmp_path)
    good = tmp_path / "images/S1/a.dicom"
    good.parent.mkdir(parents=True)
    good.write_bytes(b"ok-bytes")
    sums = f"{hashlib.sha256(b'ok-bytes').hexdigest()}  images/S1/a.dicom\n{'0' * 64}  images/S2/c.dicom\n"
    (tmp_path / "SHA256SUMS.txt").write_text(sums)
    res = V.verify(tmp_path, V.select_images(tmp_path))
    assert res["ok"] == 1 and res["missing"] == ["images/S2/c.dicom"]

    bad = tmp_path / "images/S2/c.dicom"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"corrupt")
    res = V.verify(tmp_path, V.select_images(tmp_path))
    assert res["ok"] == 1 and res["bad_hash"] == ["images/S2/c.dicom"] and not res["missing"]


def test_sample_is_balanced_seeded_and_prefix_balanced(tmp_path):
    rows = []
    for i in range(10):   # 10 mass-only images, 6 calcification images, plus one with both (counts as calc)
        rows.append((f"M{i}", f"m{i}", "['Mass']", "BI-RADS 4", 1, 2, 3, 4))
    for i in range(6):
        rows.append((f"C{i}", f"c{i}", "['Suspicious Calcification']", "BI-RADS 5", 1, 2, 3, 4))
    rows.append(("B0", "b0", "['Mass', 'Suspicious Calcification']", "BI-RADS 4", 1, 2, 3, 4))
    f = pd.DataFrame(rows, columns=["study_id", "image_id", "finding_categories", "finding_birads",
                                    "xmin", "ymin", "xmax", "ymax"])
    f["laterality"], f["view_position"], f["height"], f["width"] = "L", "CC", 100, 50
    f.to_csv(tmp_path / "finding_annotations.csv", index=False)
    sel = V.select_images(tmp_path)
    assert len(sel) == 17

    a = V.sample_images(tmp_path, sel, 8, seed=42)
    b = V.sample_images(tmp_path, sel, 8, seed=42)
    c = V.sample_images(tmp_path, sel, 8, seed=7)
    assert list(a.image_id) == list(b.image_id)            # reproducible
    assert list(a.image_id) != list(c.image_id)            # seed matters
    assert a["group"].value_counts().to_dict() == {"mass": 4, "calcification": 4}   # balanced
    assert a.head(4)["group"].value_counts().to_dict() == {"mass": 2, "calcification": 2}  # prefix balanced
    assert V.sample_images(tmp_path, sel, 8, seed=42).image_id.is_unique
