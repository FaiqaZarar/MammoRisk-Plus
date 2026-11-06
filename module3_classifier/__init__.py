"""Module 3: benign vs malignant classifier on full mammograms + explanation reliability (O2, O3).

Pipeline (each step is a CLI, all paths come from ``configs/m3_*.yaml``):

1. ``cbis``        parse CBIS-DDSM description CSVs → one row per full mammogram, image label
2. ``splits``      70/10/20 patient-level split, stratified, saved to ``splits/``
3. ``preprocess``  DICOM → downsampled 16-bit PNG cache (+ lesion masks), canonical orientation
4. ``train``       fine-tune an ImageNet backbone (ResNet-50 / EfficientNet-B3), early stop on val AUC
5. ``evaluate``    Youden threshold on val, applied once to test; external CMMD / DMID; DeLong
6. ``explain``     Grad-CAM / Score-CAM scored with Pointing Game, energy-in-box, IoU, sanity check

Torch, timm, pydicom, OpenCV and pytorch-grad-cam are imported lazily inside functions so the data
code (steps 1–2) and its tests run in any environment.
"""
