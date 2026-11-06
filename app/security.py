"""Hospital login and CSRF protection (minimal, no extra dependencies).

* Users come from config (tests) or the ``MAMMORISK_USERS`` env var as
  ``username:werkzeug_hash,...``. Passwords are never stored in plain text.
* CSRF: a random per-session token is put in every POST form and compared in constant time.
"""
from __future__ import annotations

import hmac
import os
import secrets
from functools import wraps

from flask import abort, current_app, redirect, request, session, url_for
from werkzeug.security import check_password_hash


def parse_users_env(value: str | None) -> dict[str, str]:
    users = {}
    for pair in (value or "").split(","):
        pair = pair.strip()
        if not pair:
            continue
        # werkzeug hashes contain ':' (e.g. "scrypt:32768:8:1$..."), so split once on the first ':'
        name, _, pw_hash = pair.partition(":")
        if name and pw_hash:
            users[name] = pw_hash
    return users


def load_users(config: dict) -> dict[str, str]:
    return config.get("USERS") or parse_users_env(os.environ.get("MAMMORISK_USERS"))


def verify_user(username: str, password: str) -> bool:
    users = current_app.config["USERS"]
    pw_hash = users.get(username)
    if pw_hash is None:
        # Spend comparable time for unknown users to avoid revealing which usernames exist.
        check_password_hash("scrypt:32768:8:1$x$" + "0" * 128, password)
        return False
    return check_password_hash(pw_hash, password)


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("main.login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def csrf_token() -> str:
    if "csrf" not in session:
        session["csrf"] = secrets.token_urlsafe(32)
    return session["csrf"]


def check_csrf() -> None:
    """Abort with 400 unless the posted token matches the session token."""
    if request.method != "POST":
        return
    sent = request.form.get("csrf_token", "")
    expected = session.get("csrf", "")
    if not expected or not hmac.compare_digest(sent, expected):
        abort(400, description="Invalid or missing CSRF token. Reload the form and try again.")
