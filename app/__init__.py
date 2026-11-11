"""MammoRisk+ platform (O5): Flask app factory.

Run locally:  flask --app app run --debug
Production:   gunicorn "app:create_app()"   (see Dockerfile)
"""
from __future__ import annotations

import os
import secrets
import tempfile
from datetime import timedelta
from pathlib import Path

import yaml
from flask import Flask

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "app.yaml"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _nice_key(key) -> str:
    """'z' -> 'z-score', 'probability' -> 'Probability', '1-year' -> '1-year risk' (display only)."""
    k = str(key)
    special = {"z": "z-score", "percentile": "Percentile", "category": "Risk category"}
    if k in special:
        return special[k]
    if k.endswith("-year"):
        return k + " risk"
    return k.replace("_", " ").capitalize()


def _nice_value(value, key) -> str:
    """Percent for probabilities and risks, two decimals for other floats (display only)."""
    k = str(key)
    if isinstance(value, float):
        if k in ("probability", "threshold") or k.endswith("-year"):
            return f"{100 * value:.1f}%"
        if k == "percentile":
            from app.viz import ordinal
            return f"{ordinal(value)} percentile"
        return f"{value:.2f}"
    return str(value)


def create_app(config: dict | None = None) -> Flask:
    from app.labels import load_labels
    from app.routes import bp
    from app.security import csrf_token, load_users
    from app.services import MammogramClassifierAdapter, MiraiAdapter, PRSAdapter

    config = dict(config or {})
    cfg = _load_yaml(Path(config.pop("CONFIG_FILE", DEFAULT_CONFIG)))
    app_cfg, paths = cfg.get("app", {}), dict(cfg.get("paths", {}))
    paths.update(config.pop("PATHS", {}))

    app = Flask(__name__)
    secret = os.environ.get("MAMMORISK_SECRET_KEY")
    if not secret:
        # Dev only: random key per process (sessions reset on restart). Docker sets the env var.
        secret = secrets.token_hex(32)
    app.config.update(
        SECRET_KEY=secret,
        MAX_CONTENT_LENGTH=int(app_cfg.get("max_upload_mb", 200)) * 1024 * 1024,
        PERMANENT_SESSION_LIFETIME=timedelta(minutes=int(app_cfg.get("session_lifetime_minutes", 30))),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        UPLOAD_TMP_DIR=app_cfg.get("upload_tmp_dir") or tempfile.gettempdir(),
        ALLOWED_EXTENSIONS=cfg.get("allowed_extensions", {}),
        APP_TITLE=app_cfg.get("title", "MammoRisk+"),
        # Demo mode shows stamped ILLUSTRATIVE sample output when a model is missing. Off by default
        # (and never set in Docker): enable with MAMMORISK_DEMO=1 or `python -m app.demo`.
        DEMO_MODE=os.environ.get("MAMMORISK_DEMO") == "1",
    )
    app.config.update(config)  # test overrides last
    app.config["USERS"] = load_users(app.config)

    labels_path = Path(paths.get("validation_labels", "app/validation_labels.yaml"))
    labels = load_labels(labels_path if labels_path.is_absolute() else REPO_ROOT / labels_path)
    app.extensions["labels"] = labels
    app.extensions["adapters"] = {
        "m3": MammogramClassifierAdapter(labels, paths),
        "m1": MiraiAdapter(labels, paths),
        "m2": PRSAdapter(labels, paths),
    }

    for adapter in app.extensions["adapters"].values():
        adapter.demo_mode = bool(app.config["DEMO_MODE"])

    from app.labels import REGIONS
    from app.viz import result_viz

    app.jinja_env.globals.update(csrf_token=csrf_token, result_viz=result_viz, REGIONS=REGIONS)
    app.jinja_env.filters["nice_key"] = _nice_key
    app.jinja_env.filters["nice_value"] = _nice_value
    app.register_blueprint(bp)
    return app
