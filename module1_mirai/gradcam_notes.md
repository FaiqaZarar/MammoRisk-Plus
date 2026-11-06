# Grad-CAM for Mirai: plan

OncoServe is a black box: it returns risks, not gradients. Grad-CAM therefore needs Mirai loaded
**directly in PyTorch** (CLAUDE.md Module 1, DECISIONS D-08).

## Steps
1. Clone `github.com/yala/Mirai` and download its released snapshots: the image encoder, the
   transformer aggregator and the risk factor / calibrator files. Weights stay in `models/mirai/`
   (gitignored). Mirai pins an older stack (Python 3.6 era, torch 1.x), so run it in its own
   conda env or Docker image, separate from `environment-imaging.yml`.
2. Rebuild the model as Mirai's `main.py` does: load the snapshot, `model.eval()`, and apply the same
   preprocessing (DICOM → 16-bit PNG, Mirai's resize and normalisation).
3. Target layer: the **last convolutional block of the image encoder** (a ResNet-18 backbone). The
   four views go through the encoder separately, so each view gets its own heatmap.
4. Target scalar: the logit (or calibrated probability) of the **5-year risk** (and optionally each
   horizon). Gradients flow from the aggregator back into each view's encoder features.
5. Use `pytorch-grad-cam` with a wrapper `nn.Module` that takes the 4-view tensor and returns the
   chosen horizon's score. Upsample each map to the view's input size.
6. Sanity check (Adebayo et al., 2018): randomise the encoder top-down and confirm the maps change
   (`evaluation.xai.heatmap_similarity`).
7. Check that the PyTorch model reproduces OncoServe's risks on ~20 exams (max abs difference
   < 1e-3) before trusting its heatmaps.

## Limits
- EMBED has no lesion masks for future cancers, so Mirai heatmaps are **qualitative only** (the
  quantitative localisation scores in O3 are for the Module 3 classifier).
- Never show or commit EMBED images or heatmaps overlaid on them (licence).
