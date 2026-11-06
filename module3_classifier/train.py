"""Fine-tune a backbone on the CBIS-DDSM training split, early-stopping on validation AUC.

Usage:  python -m module3_classifier.train --config configs/m3_effnetb3.yaml

Choices (each one is in the config):
- **Weighted cross-entropy:** class weight w_c = N / (2 · N_c) from the TRAIN split only. In
  CBIS-DDSM full images, benign outnumbers malignant. Without weights, the loss is dominated by the
  majority class. Weights are computed on train so no val/test information leaks in.
- **AdamW + weight decay:** decoupled L2 regularisation, the standard for fine-tuning.
- **Cosine LR schedule** over the max epochs, with a short linear warm-up, so a randomly initialised
  head doesn't wreck the pretrained features in the first steps.
- **AMP (mixed precision):** fp16 activations roughly halve GPU memory, which is what lets 1024×512
  inputs fit on a consumer / Colab GPU (CLAUDE.md compute constraint). A GradScaler prevents
  fp16 gradient underflow.
- **Gradient accumulation:** an effective batch = batch_size × accum_steps, for when only a batch
  of 4–8 fits in memory.
- **Early stopping on val AUC** (patience 10, ≤ 50 epochs). AUC, not loss, because AUC is the
  primary endpoint and is threshold-free.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from module3_classifier.config import load_config, resolve, run_dir


def class_weights(labels: np.ndarray) -> np.ndarray:
    """w_c = N / (K · N_c): balanced weights (as sklearn's 'balanced'), from training labels."""
    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels, minlength=2).astype(float)
    if (counts == 0).any():
        raise ValueError(f"A class is missing from the training split: counts={counts}")
    return len(labels) / (len(counts) * counts)


def load_manifest(cfg: dict) -> pd.DataFrame:
    man = pd.read_csv(resolve(cfg["paths"]["cache_dir"]) / "manifest.csv")
    bad = man["status"].ne("ok")
    if bad.any():
        print(f"Skipping {int(bad.sum())} images that failed preprocessing")
    return man[~bad].reset_index(drop=True)


def make_loader(df, cfg, train: bool):
    from torch.utils.data import DataLoader
    from module3_classifier.datasets import MammoDataset, build_transform
    ds = MammoDataset(df, resolve(cfg["paths"]["cache_dir"]),
                      build_transform(train, cfg.get("augmentation")))
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=train,
                      num_workers=cfg["train"].get("num_workers", 2), pin_memory=True,
                      drop_last=False, persistent_workers=cfg["train"].get("num_workers", 2) > 0)


def predict(model, loader, device, amp: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (P(malignant), dataset row index) for every item in ``loader``."""
    import torch
    model.eval()
    probs, idx = [], []
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        for x, _, i in loader:
            logits = model(x.to(device, non_blocking=True))
            probs.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
            idx.append(i.numpy())
    return np.concatenate(probs), np.concatenate(idx)


def train(cfg: dict) -> dict:
    import torch
    from sklearn.metrics import roc_auc_score
    from evaluation.seed import set_seed
    from module3_classifier.models import build_model

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tc = cfg["train"]
    out = run_dir(cfg)

    man = load_manifest(cfg)
    tr, va = man[man.split == "train"], man[man.split == "val"]
    dl_tr, dl_va = make_loader(tr, cfg, True), make_loader(va, cfg, False)
    y_va = va["label"].to_numpy()

    model = build_model(cfg["model"]["name"], cfg["model"].get("pretrained", True),
                        drop_rate=cfg["model"].get("drop_rate", 0.2)).to(device)
    w = torch.tensor(class_weights(tr["label"].to_numpy()), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    epochs, warm = tc["max_epochs"], tc.get("warmup_epochs", 1)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [
        torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warm),
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warm))],
        milestones=[warm])
    amp = bool(tc.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    accum = tc.get("accum_steps", 1)

    wb = None
    if cfg.get("wandb", {}).get("enabled"):
        try:
            import wandb
            wb = wandb.init(project=cfg["wandb"]["project"], name=cfg["run_name"],
                            config={k: v for k, v in cfg.items() if not k.startswith("_")})
        except Exception as err:  # never fail a run because logging is unavailable
            print(f"W&B disabled: {err}")

    best_auc, best_epoch, wait, history = -1.0, -1, 0, []
    for epoch in range(epochs):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, (x, y, _) in enumerate(dl_tr):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = loss_fn(model(x), y) / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0 or step + 1 == len(dl_tr):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            running += loss.item() * accum * len(y)
            n += len(y)
        sched.step()
        p_va, i_va = predict(model, dl_va, device, amp)
        val_auc = float(roc_auc_score(y_va[i_va], p_va))
        rec = {"epoch": epoch, "train_loss": running / max(n, 1), "val_auc": val_auc,
               "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0}
        history.append(rec)
        print(json.dumps(rec))
        if wb:
            wb.log(rec)
        if val_auc > best_auc:
            best_auc, best_epoch, wait = val_auc, epoch, 0
            torch.save({"model": model.state_dict(), "model_name": cfg["model"]["name"],
                        "epoch": epoch, "val_auc": val_auc, "config": cfg}, out / "best.pt")
        else:
            wait += 1
            if wait >= tc["patience"]:
                print(f"Early stop at epoch {epoch}: no val AUC gain for {wait} epochs")
                break

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    summary = {"run_name": cfg["run_name"], "config": cfg["_config_path"],
               "config_hash": cfg["_config_hash"], "best_epoch": best_epoch,
               "best_val_auc": best_auc, "checkpoint": str(out / "best.pt"),
               "split_csv": cfg["paths"]["split_csv"], "device": str(device)}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    if wb:
        wb.summary.update(summary)
        wb.finish()
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the Module 3 classifier")
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    print(json.dumps(train(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
