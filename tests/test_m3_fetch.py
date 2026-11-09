"""Offline tests for the streaming CBIS-DDSM fetcher (TCIA calls are mocked)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cv2")
from module3_classifier import fetch_cbis as F

FULL = "Mass-Training_P_00001_LEFT_CC/1.2.study/1.2.full/000000.dcm"
ROI = "Mass-Training_P_00001_LEFT_CC_1/1.2.study2/1.2.roi_a/000000.dcm"


def test_path_helpers():
    assert F.series_uid_from_path(FULL) == "1.2.full"
    assert F.series_uid_from_path(FULL.replace("/", "\\") + "\n") == "1.2.full"
    assert F.lesion_folder_from_path(ROI) == "Mass-Training_P_00001_LEFT_CC_1"
    with pytest.raises(ValueError):
        F.series_uid_from_path("only/two")


def test_roi_series_includes_cropped_quirk():
    index = pd.DataFrame({
        "SeriesInstanceUID": ["1.2.roi_a", "1.2.roi_b", "1.2.full"],
        "PatientID": ["Mass-Training_P_00001_LEFT_CC_1"] * 2 + ["Mass-Training_P_00001_LEFT_CC"],
        "SeriesDescription": ["ROI mask images", "cropped images", "full mammogram images"]})
    out = F.roi_series_for_lesions(index, ROI + ";")
    assert out == {"Mass-Training_P_00001_LEFT_CC_1": ["1.2.roi_a", "1.2.roi_b"]}


def _fake_env(monkeypatch, tmp_path, fail=False):
    """Fake TCIA: the full image is 400x200 with the breast on the RIGHT; the ROI series holds a crop
    (wrong shape) and the real mask."""
    full = np.zeros((400, 200), np.float32); full[:, 120:] = 1000; full[100:140, 150:180] = 3000
    mask = np.zeros((400, 200), np.float32); mask[100:140, 150:180] = 255
    crop = np.ones((40, 30), np.float32)
    arrays = {}

    def fake_download(uid, dest):
        if fail:
            raise RuntimeError("network down")
        dest.mkdir(parents=True, exist_ok=True)
        files = {"1.2.full": [("f.dcm", full)], "1.2.roi_a": [("c.dcm", crop), ("m.dcm", mask)]}[uid]
        out = []
        for name, arr in files:
            p = dest / f"{uid}_{name}"; p.write_bytes(b"x"); arrays[str(p)] = arr; out.append(p)
        return out
    monkeypatch.setattr(F, "download_series", fake_download)
    monkeypatch.setattr(F.P, "read_dicom", lambda p: arrays[str(p)])
    index = pd.DataFrame({"SeriesInstanceUID": ["1.2.roi_a"], "PatientID": ["Mass-Training_P_00001_LEFT_CC_1"],
                          "SeriesDescription": ["ROI mask images"]})
    row = pd.Series({"image_id": "P_00001_LEFT_CC", "image_file_path": FULL, "roi_mask_paths": ROI})
    return row, index


def test_process_image_caches_and_cleans_up(monkeypatch, tmp_path):
    import cv2
    row, index = _fake_env(monkeypatch, tmp_path)
    tmp_root = tmp_path / "tmp"; tmp_root.mkdir()
    rec = F.process_image(row, index, tmp_path / "cache", tmp_root, 64, 32, True)
    assert rec["status"] == "ok", rec
    assert rec["flipped"] and rec["n_masks"] == 1 and rec["masks_not_found"] == ""
    m = cv2.imread(str(tmp_path / "cache" / rec["mask_path"]), 0) > 0
    img = cv2.imread(str(tmp_path / "cache" / rec["cache_path"]), cv2.IMREAD_UNCHANGED)
    assert img.dtype == np.uint16 and img.shape == (64, 32)
    # after the canonical flip the lesion (and its mask) is on the LEFT half
    assert m.any() and np.argwhere(m)[:, 1].max() < 16
    assert list(tmp_root.iterdir()) == []          # temporary DICOMs deleted


def test_process_image_failure_is_recorded(monkeypatch, tmp_path):
    row, index = _fake_env(monkeypatch, tmp_path, fail=True)
    tmp_root = tmp_path / "tmp"; tmp_root.mkdir()
    rec = F.process_image(row, index, tmp_path / "cache", tmp_root, 64, 32, True)
    assert rec["status"] == "failed" and "network down" in rec["error"]
    assert list(tmp_root.iterdir()) == []


def test_resume_and_manifest(tmp_path):
    cache = tmp_path / "cache"; (cache / "images").mkdir(parents=True)
    (cache / "images" / "a.png").write_bytes(b"x")
    prog = cache / "manifest_progress.jsonl"
    prog.write_text("\n".join(json.dumps(r) for r in [
        {"image_id": "a", "status": "failed", "error": "e"},
        {"image_id": "a", "status": "ok", "cache_path": "images/a.png"},
        {"image_id": "b", "status": "ok", "cache_path": "images/b.png"},   # png missing -> redo
        {"image_id": "c", "status": "failed", "error": "e"}]) + "\n")
    assert set(F._done_ids(prog, cache)) == {"a"}
    split = pd.DataFrame({"image_id": ["a", "b", "c", "d"], "split": ["train"] * 4})
    man = F.write_manifest(split, prog, cache).set_index("image_id")
    assert man.loc["a", "status"] == "ok" and man.loc["c", "status"] == "failed"
    assert man.loc["d", "status"] == "not_fetched"
    assert (cache / "manifest.csv").exists()
