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

    app.jinja_env.globals["csrf_token"] = csrf_token
    app.register_blueprint(bp)
    return app
