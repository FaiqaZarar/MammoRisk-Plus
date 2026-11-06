"""Shared evaluation metrics for MammoRisk+.

Every module reports through these functions so that the same definition of a metric (and of its
confidence interval) is used everywhere in the thesis.

- ``evaluation.classification``: AUC with DeLong CI, paired DeLong test, Youden threshold,
  threshold metrics (sensitivity, specificity, PPV, NPV, F1)                    [Module 3]
- ``evaluation.bootstrap``: percentile bootstrap CI, optionally stratified / clustered      [all]
- ``evaluation.survival``: IPCW C-index, time-dependent AUC, calibration at a horizon    [Module 1]
- ``evaluation.xai``: Pointing Game, energy-in-box, IoU, sanity-check similarity         [Module 3]
- ``evaluation.seed``: fixed seeds for Python / NumPy / PyTorch / CUDA                   [all]
"""
