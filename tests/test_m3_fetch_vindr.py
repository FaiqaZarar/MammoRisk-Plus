import hashlib

import pandas as pd
import pytest

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
    sel = V.select_images(tmp_path)
    assert sorted(sel.image_id) == ["a", "c"]


def test_select_all_views_keeps_whole_exam(tmp_path):
    _write_csvs(tmp_path)
    sel = V.select_images(tmp_path, all_views=True)
    assert sorted(sel.image_id) == ["a", "b", "c", "f"]   # exam S1 (4 images listed) + S2


def test_parse_sha256sums():
    text = "AAA  images/s/x.dicom\nbbb *./images/s/y.dicom\n\nbad\n"
    assert V.parse_sha256sums(text) == {"images/s/x.dicom": "aaa", "images/s/y.dicom": "bbb"}


def test_fetch_file_ok_skip_and_mismatch(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"dicom-bytes")
    sha = hashlib.sha256(b"dicom-bytes").hexdigest()
    url = src.as_uri()
    dest = tmp_path / "out" / "x.dicom"

    assert V.fetch_file(url, dest, "Basic x", sha) == "ok"
    assert dest.read_bytes() == b"dicom-bytes"
    assert not dest.with_name("x.dicom.part").exists()
    assert V.fetch_file(url, dest, "Basic x", sha) == "skipped"          # resumable

    bad = tmp_path / "out" / "y.dicom"
    with pytest.raises(RuntimeError):
        V.fetch_file(url, bad, "Basic x", "0" * 64, retries=1)           # wrong hash is never kept
    assert not bad.exists() and not bad.with_name("y.dicom.part").exists()


def test_dry_run_needs_no_password(tmp_path, capsys):
    _write_csvs(tmp_path)
    assert V.main(["--root", str(tmp_path), "--dry-run"]) == 0
    assert "2 images in 2 exams" in capsys.readouterr().out
