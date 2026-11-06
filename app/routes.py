"""Web routes. Diagnostic pages (uploads, results, report) require the hospital login."""
from __future__ import annotations

import tempfile
from pathlib import Path

from flask import (Blueprint, Response, abort, current_app, flash, redirect, render_template,
                   request, session, url_for)
from werkzeug.utils import secure_filename

from app.labels import LabelledResult
from app.report import DISCLAIMER, build_pdf
from app.security import check_csrf, login_required, verify_user
from app.services.mirai import REQUIRED_VIEWS

bp = Blueprint("main", __name__)


@bp.before_request
def _csrf():
    check_csrf()


# ----------------------------------------------------------------------------------------------
# Public pages
# ----------------------------------------------------------------------------------------------
@bp.get("/")
def index():
    return render_template("index.html", disclaimer=DISCLAIMER)


@bp.get("/about")
def about():
    return render_template("about.html", labels=current_app.extensions["labels"].values(),
                           disclaimer=DISCLAIMER)


@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("username", "").strip()
        if verify_user(user, request.form.get("password", "")):
            session.clear()  # new session on login (prevents session fixation)
            session["user"] = user
            session.permanent = True
            nxt = request.args.get("next", "")
            # Only allow local redirects.
            return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//")
                            else url_for("main.dashboard"))
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@bp.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("main.index"))


# ----------------------------------------------------------------------------------------------
# Diagnostic pages (login required)
# ----------------------------------------------------------------------------------------------
@bp.get("/dashboard")
@login_required
def dashboard():
    adapters = current_app.extensions["adapters"]
    status = {k: a.availability() for k, a in adapters.items()}
    return render_template("dashboard.html", status=status, adapters=adapters)


def _allowed(filename: str, kind: str) -> bool:
    exts = current_app.config["ALLOWED_EXTENSIONS"][kind]
    return any(filename.lower().endswith(e) for e in exts)


def _run(adapter_key: str, fields: dict[str, str], kind: str) -> list[LabelledResult] | None:
    """Save uploads to a temp dir outside the repo, run the adapter, delete the files."""
    uploads = {name: request.files.get(field) for name, field in fields.items()}
    uploads = {k: f for k, f in uploads.items() if f and f.filename}
    if not uploads:
        flash("Please choose a file.", "error")
        return None
    for f in uploads.values():
        if not _allowed(f.filename, kind):
            flash(f"Unsupported file type: {f.filename}", "error")
            return None
    with tempfile.TemporaryDirectory(dir=current_app.config["UPLOAD_TMP_DIR"]) as tmp:
        paths = {}
        for name, f in uploads.items():
            p = Path(tmp) / (secure_filename(f"{name}_{f.filename}") or name)
            f.save(p)
            paths[name] = p
        results = current_app.extensions["adapters"][adapter_key].predict(paths)
    # Temp dir and uploaded files are deleted here.
    session["results"] = [r.to_dict() for r in results]
    return results


@bp.route("/mammogram", methods=["GET", "POST"])
@login_required
def mammogram():
    results = None
    if request.method == "POST":
        results = _run("m3", {"image": "image"}, "mammogram")
    return render_template("upload.html", title="Mammogram: benign vs malignant",
                           fields=[("image", "Mammogram (PNG, JPG or DICOM)")],
                           accept=".png,.jpg,.jpeg,.dcm,.dicom", results=results)


@bp.route("/mirai", methods=["GET", "POST"])
@login_required
def mirai():
    results = None
    if request.method == "POST":
        results = _run("m1", {v: v for v in REQUIRED_VIEWS}, "mammogram")
    return render_template("upload.html", title="Mirai 1-5 year risk (four views)",
                           fields=[(v, f"{v} view (DICOM)") for v in REQUIRED_VIEWS],
                           accept=".dcm,.dicom,.png", results=results)


@bp.route("/prs", methods=["GET", "POST"])
@login_required
def prs():
    results = None
    if request.method == "POST":
        results = _run("m2", {"vcf": "vcf"}, "vcf")
    return render_template("upload.html", title="PRS-313 under European (CEU) vs Pakistani reference",
                           fields=[("vcf", "Genotype file (VCF or VCF.gz)")],
                           accept=".vcf,.gz", results=results)


@bp.get("/report.pdf")
@login_required
def report():
    labels = current_app.extensions["labels"]
    stored = session.get("results")
    if not stored:
        abort(404, description="No results yet. Run an analysis first.")
    results = [LabelledResult.from_dict(d, labels) for d in stored]
    pdf = build_pdf(results, user=session.get("user"))
    return Response(pdf, mimetype="application/pdf",
                    headers={"Content-Disposition": "attachment; filename=mammorisk_report.pdf"})
