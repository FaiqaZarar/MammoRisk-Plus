"""VinDr cache tests on synthetic DICOMs (no VinDr data needed)."""
import ast

import numpy as np
import pandas as pd
import pytest

pydicom = pytest.importorskip("pydicom")
cv2 = pytest.importorskip("cv2")

from pydicom.dataset import FileDataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, generate_uid  # noqa: E402

from module3_classifier import cache_vindr as C  # noqa: E402
from module3_classifier.fetch_vindr import rel_path  # noqa: E402

H, W = 200, 100


def _write_dicom(path, arr):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.2"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelData = arr.astype(np.uint16).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)


def _breast(side: str) -> np.ndarray:
    """Dark background; a bright 'breast' on the given side with a brighter lesion at x 70-95, y 80-120 (right)
    or mirrored (left)."""
    a = np.full((H, W), 200, np.uint16)
    a[:, W // 2:] = 4000 if side == "right" else 200
    a[:, : W // 2] = 4000 if side == "left" else 200
    x0, x1 = (70, 95) if side == "right" else (5, 30)
    a[80:120, x0:x1] = 9000
    return a


@pytest.fixture
def root(tmp_path):
    rows = []
    for image_id, study, lat, side, (x0, x1) in [("imgR", "S1", "R", "right", (70, 95)),
                                                  ("imgL", "S2", "L", "left", (5, 30)),
                                                  ("imgGone", "S3", "L", "left", (5, 30)),
                                                  ("imgBadSize", "S4", "L", "left", (5, 30))]:
        rows.append({"study_id": study, "image_id": image_id, "laterality": lat, "view_position": "CC",
                     "height": H if image_id != "imgBadSize" else H + 10, "width": W,
                     "finding_categories": "['Mass']", "finding_birads": "BI-RADS 4",
                     "xmin": x0, "ymin": 80, "xmax": x1, "ymax": 120})
    pd.DataFrame(rows).to_csv(tmp_path / "finding_annotations.csv", index=False)
    _write_dicom(tmp_path / rel_path("S1", "imgR"), _breast("right"))
    _write_dicom(tmp_path / rel_path("S2", "imgL"), _breast("left"))
    _write_dicom(tmp_path / rel_path("S4", "imgBadSize"), _breast("left"))   # 200x100, CSV says 210x100
    return tmp_path                                                         # imgGone has no DICOM


def test_cache_statuses_and_files(root, tmp_path_factory):
    out = tmp_path_factory.mktemp("cache")
    idx = C.build_index(root, out).set_index("image_id")
    assert idx.loc["imgR", "status"] == "ok" and idx.loc["imgL", "status"] == "ok"
    assert idx.loc["imgGone", "status"] == "not_downloaded"
    assert idx.loc["imgBadSize", "status"].startswith("size_mismatch")   # boxes would be misplaced: refuse
    png = cv2.imread(str(out / idx.loc["imgR", "cache_path"]), cv2.IMREAD_UNCHANGED)
    assert png.dtype == np.uint16 and png.shape == (1024, 512)
    assert not (out / "images" / "imgBadSize.png").exists()


def test_right_breast_is_flipped_and_box_follows(root, tmp_path_factory):
    out = tmp_path_factory.mktemp("cache")
    idx = C.build_index(root, out).set_index("image_id")
    assert bool(idx.loc["imgR", "flipped"]) and not bool(idx.loc["imgL", "flipped"])
    scale = 5.12   # min(1024/200, 512/100)
    (x0, y0, x1, y1), = ast.literal_eval(str(idx.loc["imgR", "boxes_cached"]))
    # original x 70..95 mirrors to 5..30 (width 100), then x scale
    assert (x0, y0, x1, y1) == pytest.approx((5 * scale, 80 * scale, 30 * scale, 120 * scale), abs=0.05)
    # the unflipped left breast keeps its box
    (l0, _, l1, _), = ast.literal_eval(str(idx.loc["imgL", "boxes_cached"]))
    assert (l0, l1) == pytest.approx((5 * scale, 30 * scale), abs=0.05)


def test_box_lands_on_the_bright_lesion_after_preprocessing(root, tmp_path_factory):
    out = tmp_path_factory.mktemp("cache")
    idx = C.build_index(root, out).set_index("image_id")
    for image_id in ("imgR", "imgL"):
        im = cv2.imread(str(out / idx.loc[image_id, "cache_path"]), cv2.IMREAD_UNCHANGED).astype(float)
        (x0, y0, x1, y1), = ast.literal_eval(str(idx.loc[image_id, "boxes_cached"]))
        inside = im[int(y0):int(y1), int(x0):int(x1)].mean()
        assert inside > 1.2 * im[:, :256][im[:, :256] > 0].mean()       # lesion box is brighter than the breast


def test_index_is_usable_by_the_explanation_loader(root, tmp_path_factory):
    out = tmp_path_factory.mktemp("cache")
    C.build_index(root, out).to_csv(out / "index.csv", index=False)
    from module3_classifier.explain import load_eval_table
    cfg = {"external": {"vindr": {"index_csv": str(out / "index.csv"), "root": str(out)}}}
    df, got_root = load_eval_table(cfg, "vindr")
    assert sorted(df.image_id) == ["imgL", "imgR"] and got_root == out    # only cached images, all with boxes_cached


def test_output_folder_inside_the_repo_is_refused():
    with pytest.raises(SystemExit):
        C._check_out_dir(C.REPO_ROOT / "docs" / "vindr_images")
    C._check_out_dir(C.REPO_ROOT / "data" / "processed" / "x")          # data/ is gitignored: allowed


def test_rerun_after_more_downloads_fills_in_missing_images(root, tmp_path_factory):
    out = tmp_path_factory.mktemp("cache")
    first = C.build_index(root, out).set_index("image_id")
    assert first.loc["imgGone", "status"] == "not_downloaded"
    _write_dicom(root / rel_path("S3", "imgGone"), _breast("left"))     # the download finished meanwhile
    second = C.build_index(root, out).set_index("image_id")
    assert second.loc["imgGone", "status"] == "ok"


def test_box_overshooting_the_image_edge_is_clipped(tmp_path, tmp_path_factory):
    # real VinDr annotations sometimes extend past the image edge (x0 < 0 or x1 > width)
    rows = [{"study_id": "S9", "image_id": "imgEdge", "laterality": "L", "view_position": "CC",
             "height": H, "width": W, "finding_categories": "['Mass']", "finding_birads": "BI-RADS 4",
             "xmin": -10, "ymin": 80, "xmax": 30, "ymax": 120}]
    pd.DataFrame(rows).to_csv(tmp_path / "finding_annotations.csv", index=False)
    _write_dicom(tmp_path / rel_path("S9", "imgEdge"), _breast("left"))
    idx = C.build_index(tmp_path, tmp_path_factory.mktemp("cache")).set_index("image_id")
    (x0, y0, x1, y1), = ast.literal_eval(str(idx.loc["imgEdge", "boxes_cached"]))
    assert x0 == 0.0 and 0 < x1 <= 512 and 0 <= y0 < y1 <= 1024      # clipped at the left edge, still valid
