"""Explanation reliability (O3): do Grad-CAM / Score-CAM point at the lesion, equally well across
CBIS-DDSM (USA, film), VinDr-Mammo (Vietnam) and DMID (India)?

Usage
  python -m module3_classifier.explain --config configs/m3_effnetb3.yaml \
      --checkpoint results/m3/<run>/best.pt --dataset cbis [--sanity]

Methods
- **Grad-CAM** (Selvaraju et al., 2017): weights each feature map of the target layer by the
  spatially averaged gradient of the malignant logit. It is fast, but it depends on gradients, which
  can be noisy or saturated.
- **Score-CAM** (Wang et al., 2020): weights each feature map by how much the malignant score
  rises when the input is masked by that (upsampled) map. It is gradient-free, but needs one
  forward pass per channel (2048 for ResNet-50), so it is slow. ``scorecam_batch_size`` controls
  memory.
Both explain the MALIGNANT class (target = class 1) on the last conv block (models.cam_target_layer).
pytorch-grad-cam upsamples the map to the input size, which equals the cached image and mask size.

Scoring (evaluation.xai): Pointing Game, energy-in-box (with its area baseline), IoU at the top 50%
of pixels (the threshold value is recorded), reported for **all malignant cases** and for the
**correctly classified** ones (prob ≥ the val-chosen threshold).

Sanity check (Adebayo et al., 2018): cascading randomisation from the top block down, recording the
Spearman similarity to the original heatmap after each block. Similarity should fall toward 0. If
it stays high, the method is acting like an edge detector.

**Licence rule:** VinDr-Mammo images and heatmap overlays are never written to disk. Only metrics are.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.xai import box_to_mask, heatmap_similarity, score_all
from module3_classifier.config import load_config, resolve

NO_IMAGE_EXPORT = ("vindr",)  # dataset keys whose images / overlays may never be saved


def _lesion_mask(row, root: Path, shape) -> np.ndarray | None:
    if isinstance(getattr(row, "mask_path", None), str) and row.mask_path:
        from module3_classifier.datasets import read_png01
        return read_png01(root / row.mask_path) > 0.5
    boxes = getattr(row, "boxes_cached", None)
    if isinstance(boxes, str) and boxes:
        boxes = ast.literal_eval(boxes)
    if boxes:
        return box_to_mask(shape, boxes)
    return None


def _threshold(checkpoint: Path) -> tuple[float, str]:
    ev = checkpoint.parent / "evaluation.json"
    if ev.exists():
        return float(json.loads(ev.read_text())["threshold"]["value"]), "val Youden (evaluation.json)"
    print("WARNING: evaluation.json not found; 'correct' uses 0.5. Run evaluate first.")
    return 0.5, "0.5 fallback"


def _cam_objects(model, model_name, methods, cfg):
    from pytorch_grad_cam import GradCAM, ScoreCAM
    from module3_classifier.models import cam_target_layer
    layer = [cam_target_layer(model, model_name)]
    cams = {}
    if "gradcam" in methods:
        cams["gradcam"] = GradCAM(model=model, target_layers=layer)
    if "scorecam" in methods:
        sc = ScoreCAM(model=model, target_layers=layer)
        sc.batch_size = cfg["xai"].get("scorecam_batch_size", 32)
        cams["scorecam"] = sc
    return cams


def _heatmap(cam, x):
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    return cam(input_tensor=x, targets=[ClassifierOutputTarget(1)] * len(x))


def load_eval_table(cfg: dict, dataset: str) -> tuple[pd.DataFrame, Path]:
    """Malignant cases with a lesion annotation, for the requested dataset."""
    if dataset == "cbis":
        from module3_classifier.train import load_manifest
        df = load_manifest(cfg)
        df = df[(df.split == "test") & (df.label == 1) & df["mask_path"].notna()]
        return df.reset_index(drop=True), resolve(cfg["paths"]["cache_dir"])
    ext = cfg["external"][dataset]
    df = pd.read_csv(resolve(ext["index_csv"]))
    df = df[df.label == 1]
    return df.reset_index(drop=True), resolve(ext["root"])


def explain(cfg: dict, checkpoint: str, dataset: str, sanity: bool = False) -> dict:
    import torch
    from module3_classifier.datasets import MammoDataset, build_transform
    from module3_classifier.evaluate import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(checkpoint)
    model, ck = load_checkpoint(checkpoint, device)
    name = ck["model_name"]
    thr, thr_src = _threshold(checkpoint)
    df, root = load_eval_table(cfg, dataset)
    xc = cfg["xai"]
    path_col = "cache_path" if dataset == "cbis" else cfg["external"][dataset].get("path_col", "cache_path")
    ds = MammoDataset(df, root, build_transform(False), path_col=path_col)
    cams = _cam_objects(model, name, xc["methods"], cfg)
    save_overlays = xc.get("save_overlays", False) and not any(k in dataset.lower() for k in NO_IMAGE_EXPORT)

    rows = []
    for i in range(len(ds)):
        x, y, _ = ds[i]
        x = x[None].to(device)
        with torch.no_grad():
            prob = float(torch.softmax(model(x).float(), 1)[0, 1])
        row = df.iloc[i]
        mask = _lesion_mask(row, root, tuple(x.shape[-2:]))
        if mask is None or not mask.any():
            continue
        for mname, cam in cams.items():
            h = _heatmap(cam, x)[0]
            rec = {"dataset": dataset, "image_id": row.image_id, "patient_id": row.patient_id,
                   "method": mname, "prob": prob, "correct": prob >= thr,
                   **score_all(h, mask, xc.get("iou_top_fraction", 0.5), xc.get("pointing_tol", 0))}
            rows.append(rec)
            if save_overlays:
                out = resolve(cfg["paths"]["results_dir"]) / cfg["run_name"] / "overlays" / mname
                out.mkdir(parents=True, exist_ok=True)
                np.save(out / f"{row.image_id}.npy", h.astype(np.float16))
    res = pd.DataFrame(rows)
    out_dir = checkpoint.parent
    res.to_csv(out_dir / f"xai_{dataset}.csv", index=False)

    summary = {"dataset": dataset, "checkpoint": str(checkpoint), "config": cfg["_config_path"],
               "config_hash": cfg["_config_hash"], "threshold": thr, "threshold_source": thr_src,
               "iou_top_fraction": xc.get("iou_top_fraction", 0.5), "by_method": {}}
    for mname, g in res.groupby("method") if len(res) else []:
        block = {}
        for subset, s in (("all_malignant", g), ("correctly_classified", g[g.correct])):
            block[subset] = {"n": int(len(s)), "pointing_game": float(s.pointing_hit.mean()) if len(s) else None,
                             "energy_in_box": float(s.energy_in_box.mean()) if len(s) else None,
                             "energy_area_baseline": float(s.energy_baseline_area.mean()) if len(s) else None,
                             "iou": float(s.iou.mean()) if len(s) else None}
        summary["by_method"][mname] = block

    if sanity:
        summary["sanity_check"] = sanity_check(model, name, ds, device, xc.get("n_sanity", 20))
        pd.DataFrame(summary["sanity_check"]).to_csv(out_dir / f"xai_sanity_{dataset}.csv", index=False)
    (out_dir / f"xai_{dataset}.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def sanity_check(model, model_name, ds, device, n_images: int = 20) -> list[dict]:
    """Cascading randomisation (Grad-CAM). Returns mean/SD Spearman similarity per randomised block."""
    import torch
    from pytorch_grad_cam import GradCAM
    from module3_classifier.models import cam_target_layer, randomisation_order, reinitialise

    xs = [ds[i][0][None].to(device) for i in range(min(n_images, len(ds)))]
    base_cam = GradCAM(model=model, target_layers=[cam_target_layer(model, model_name)])
    originals = [_heatmap(base_cam, x)[0] for x in xs]
    rand = copy.deepcopy(model)
    torch.manual_seed(0)
    results = []
    for block_name, block in randomisation_order(rand):
        reinitialise(block)  # cumulative: earlier (higher) blocks stay randomised
        cam = GradCAM(model=rand, target_layers=[cam_target_layer(rand, model_name)])
        sims = [heatmap_similarity(o, _heatmap(cam, x)[0]) for o, x in zip(originals, xs)]
        sims = [s for s in sims if not np.isnan(s)]
        results.append({"randomised_up_to": block_name, "n": len(sims),
                        "spearman_mean": float(np.mean(sims)) if sims else float("nan"),
                        "spearman_sd": float(np.std(sims)) if sims else float("nan")})
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score Grad-CAM / Score-CAM localisation")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default="cbis", help="cbis or a key under `external` (vindr, dmid)")
    ap.add_argument("--sanity", action="store_true", help="also run the cascading randomisation check")
    args = ap.parse_args(argv)
    print(json.dumps(explain(load_config(args.config), args.checkpoint, args.dataset, args.sanity),
                     indent=2, default=float))


if __name__ == "__main__":
    main()
