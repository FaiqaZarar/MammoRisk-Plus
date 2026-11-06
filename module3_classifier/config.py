"""Config loading. One YAML per experiment (CLAUDE.md reproducibility rule)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    # A short hash of the config content is stored with every result, so a number can always be
    # traced back to the exact settings that produced it.
    cfg["_config_hash"] = hashlib.sha1(json.dumps(
        {k: v for k, v in cfg.items() if not k.startswith("_")}, sort_keys=True, default=str
    ).encode()).hexdigest()[:10]
    return cfg


def resolve(p: str | Path | None) -> Path | None:
    """Relative paths in configs are relative to the repo root."""
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def run_dir(cfg: dict) -> Path:
    d = resolve(cfg["paths"]["results_dir"]) / cfg["run_name"]
    d.mkdir(parents=True, exist_ok=True)
    return d
