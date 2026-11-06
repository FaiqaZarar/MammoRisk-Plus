"""Datasets for Module 3.

``MammoDataset`` is a plain map-style dataset (``__len__`` + ``__getitem__``). PyTorch's
DataLoader accepts any such object, so this file imports without torch. Torch and torchvision are
imported only when an item is loaded or a transform is built.

External test sets (CMMD, DMID, VinDr-Mammo) are handled by *metadata adapters*. These return a
DataFrame with ``image_id, patient_id, path, label`` (and ``boxes`` where available) and feed the
same dataset class. Each external image must first go through ``preprocess.preprocess_array`` so it
gets the same normalisation and orientation as CBIS-DDSM.

⚠️ The column names below follow each dataset's public documentation. **Verify them against the
real files on first download** (they are marked ASSUMPTION).
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ------------------------------------------------------------------------------------------------
# Transforms
# ------------------------------------------------------------------------------------------------
def build_transform(train: bool, aug_cfg: dict | None = None):
    """Augmentation on TRAIN ONLY (CLAUDE.md no-leakage rule). Val/test get normalisation only.

    - flips: a mirrored or upside-down lesion is still the same lesion. Note that a horizontal flip
      undoes the canonical orientation for that sample. We keep it as specified, as a regulariser.
    - rotation ≤ 15°: positioning varies between technologists
    - brightness/contrast: exposure and film/scanner differences
    The single grey channel is repeated ×3 and ImageNet-normalised, because the backbones were
    pretrained on RGB ImageNet.
    """
    from torchvision.transforms import v2 as T
    import torch
    aug_cfg = aug_cfg or {}
    ops = []
    if train:
        ops += [
            T.RandomHorizontalFlip(p=aug_cfg.get("hflip", 0.5)),
            T.RandomVerticalFlip(p=aug_cfg.get("vflip", 0.5)),
            T.RandomRotation(degrees=aug_cfg.get("rotation_deg", 15)),
            T.ColorJitter(brightness=aug_cfg.get("brightness", 0.2),
                          contrast=aug_cfg.get("contrast", 0.2)),
        ]
    # A named callable, not a lambda: on Windows DataLoader workers are spawned, and the transform
    # must be picklable.
    ops += [GreyToRGB(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose([T.ToDtype(torch.float32, scale=False)] + ops)


class GreyToRGB:
    def __call__(self, x):
        return x.expand(3, -1, -1).contiguous() if x.shape[0] == 1 else x


def read_png01(path: str | Path) -> np.ndarray:
    """Read an 8- or 16-bit grey PNG into float32 [0, 1]."""
    from PIL import Image
    im = np.array(Image.open(path))
    if im.ndim == 3:
        im = im[..., 0]
    maxv = 65535.0 if im.dtype == np.uint16 or im.max() > 255 else 255.0
    return im.astype(np.float32) / maxv


class MammoDataset:
    """Items are ``(image_tensor[3,H,W], label, index)``. ``index`` lets callers join predictions
    back to the metadata row (image_id, patient_id)."""

    def __init__(self, df: pd.DataFrame, root: str | Path | None, transform=None,
                 path_col: str = "cache_path", label_col: str = "label"):
        self.df = df.reset_index(drop=True)
        self.root = Path(root) if root else None
        self.transform = transform
        self.path_col, self.label_col = path_col, label_col

    def __len__(self):
        return len(self.df)

    def path(self, i: int) -> Path:
        p = Path(self.df.at[i, self.path_col])
        return self.root / p if self.root and not p.is_absolute() else p

    def __getitem__(self, i: int):
        import torch
        x = torch.from_numpy(read_png01(self.path(i)))[None]  # [1, H, W]
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.df.at[i, self.label_col]), i

    def labels(self) -> np.ndarray:
        return self.df[self.label_col].to_numpy().astype(int)


# ------------------------------------------------------------------------------------------------
# External metadata adapters
# ------------------------------------------------------------------------------------------------
def load_cmmd(clinical_csv: str | Path, image_index_csv: str | Path) -> pd.DataFrame:
    """CMMD (China; TCIA). ASSUMPTION: the clinical sheet ``CMMD_clinicaldata_revision`` has
    columns ``ID1`` (patient), ``LeftRight`` (L/R), ``classification`` (Benign/Malignant).

    The images are DICOMs without the label, so ``image_index_csv`` (built when caching CMMD
    with ``preprocess``) must list ``patient_id, laterality (L/R), view, cache_path``.
    The label is per breast, so all views of a breast get that breast's label.
    """
    clin = pd.read_csv(clinical_csv) if str(clinical_csv).endswith(".csv") else \
        pd.read_excel(clinical_csv)
    clin = clin.rename(columns={"ID1": "patient_id", "LeftRight": "laterality"})
    clin["label"] = (clin["classification"].str.strip().str.lower() == "malignant").astype(int)
    idx = pd.read_csv(image_index_csv)
    df = idx.merge(clin[["patient_id", "laterality", "label"]], on=["patient_id", "laterality"],
                   how="inner")
    df["image_id"] = df["patient_id"] + "_" + df["laterality"] + "_" + df["view"].astype(str)
    df["dataset"] = "CMMD"
    return df


def load_dmid(info_txt: str | Path, image_dir_or_index: str | Path, y_origin: str = "top",
              include_normals: bool = False) -> pd.DataFrame:
    """DMID (India, 510 images). ASSUMPTION: the metadata follows the mini-MIAS layout,
    whitespace-separated: ``ref  tissue  class  severity  x  y  radius``, with severity B/M,
    class NORM for normal images, and lesion centre (x, y) + radius in pixels.

    ``y_origin``: mini-MIAS measures y from the BOTTOM-left corner. Check this for DMID on first
    use and set ``bottom`` if so. Boxes are returned in ORIGINAL pixel coordinates (top-left
    origin). Map them to the cache with ``preprocess.transform_boxes``.

    Normals are excluded by default: the 2-class AUC is benign vs malignant (CLAUDE.md).
    DMID labels come from radiology reports, not biopsy, which is a stated limitation.
    """
    rows = []
    for line in Path(info_txt).read_text().splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[0].startswith("#"):
            continue
        ref, tissue, cls = parts[0], parts[1], parts[2].upper()
        sev = parts[3].upper() if len(parts) > 3 else ""
        box = None
        if len(parts) >= 7:
            x, y, r = float(parts[4]), float(parts[5]), float(parts[6])
            box = (x - r, y - r, x + r, y + r)
        rows.append({"image_id": ref, "patient_id": ref, "tissue": tissue, "abnormality": cls,
                     "severity": sev, "box": box})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Several lesions on one image → one row with a list of boxes; malignant if any is M.
    agg = df.groupby("image_id").agg(
        patient_id=("patient_id", "first"), tissue=("tissue", "first"),
        abnormality=("abnormality", lambda s: ";".join(sorted(set(s)))),
        severities=("severity", lambda s: "".join(sorted(set(s)))),
        boxes=("box", lambda s: [b for b in s if b is not None])).reset_index()
    agg["is_normal"] = agg["abnormality"].eq("NORM")
    agg["label"] = agg["severities"].str.contains("M").astype(int)
    if not include_normals:
        agg = agg[~agg["is_normal"]].reset_index(drop=True)
    agg["y_origin"] = y_origin
    agg["dataset"] = "DMID"
    agg["path"] = [str(Path(image_dir_or_index) / f"{i}") for i in agg["image_id"]]
    return agg


def flip_y_if_bottom_origin(boxes, image_height: int, y_origin: str):
    if y_origin != "bottom":
        return boxes
    return [(x0, image_height - y1, x1, image_height - y0) for x0, y0, x1, y1 in boxes]


def load_vindr_boxes(finding_csv: str | Path, categories=("Mass", "Suspicious Calcification"),
                     min_birads: int = 4) -> pd.DataFrame:
    """VinDr-Mammo (Vietnam; PhysioNet licence). Used ONLY for explanation localisation (O3).

    ASSUMPTION: ``finding_annotations.csv`` has ``study_id, image_id, laterality, view_position,
    height, width, finding_categories, finding_birads, xmin, ymin, xmax, ymax``.
    ``finding_categories`` is a stringified list, e.g. "['Mass']"; ``finding_birads`` e.g.
    "BI-RADS 4".

    VinDr has no biopsy labels, so "malignant-suspicious" = findings of the chosen categories with
    BI-RADS ≥ ``min_birads`` (4/5). This proxy should be logged in DECISIONS.
    Never save VinDr images or overlays, only metrics.
    """
    f = pd.read_csv(finding_csv)
    f = f.dropna(subset=["xmin", "ymin", "xmax", "ymax"])

    def _cats(s):
        try:
            return set(ast.literal_eval(s)) if isinstance(s, str) and s.startswith("[") else {s}
        except (ValueError, SyntaxError):
            return {s}

    def _birads(s):
        digits = "".join(ch for ch in str(s) if ch.isdigit())
        return int(digits[0]) if digits else -1

    keep = f["finding_categories"].map(lambda s: bool(_cats(s) & set(categories))) & \
        (f["finding_birads"].map(_birads) >= min_birads)
    f = f[keep]
    boxes = f.groupby("image_id").apply(
        lambda g: [tuple(map(float, r)) for r in g[["xmin", "ymin", "xmax", "ymax"]].to_numpy()],
        include_groups=False)
    meta = f.groupby("image_id")[["study_id", "laterality", "view_position", "height",
                                  "width"]].first()
    out = meta.assign(boxes=boxes).reset_index()
    out["patient_id"] = out["study_id"]
    out["label"] = 1
    out["dataset"] = "VinDr-Mammo"
    return out
