"""Platform (O5) tests with the Flask test client. No models, data or plink2 needed."""
import io
import re

import pytest
from werkzeug.security import generate_password_hash

from app import create_app
from app.labels import LabelledResult, MissingLabelError, ValidationLabel, make_result
from app.report import build_pdf

PASSWORD = "test-password-123"


@pytest.fixture
def app(tmp_path):
    return create_app({
        "TESTING": True,
        "USERS": {"doctor": generate_password_hash(PASSWORD)},
        "UPLOAD_TMP_DIR": str(tmp_path),
        # Point every back-end at something that doesn't exist -> "model not available".
        "PATHS": {"m3_checkpoint": str(tmp_path / "none.pt"), "m1_oncoserve_url": None,
                  "m2_reference_stats": str(tmp_path / "none.json")},
    })


@pytest.fixture
def client(app):
    return app.test_client()


def _csrf(client, url):
    html = client.get(url).get_data(as_text=True)
    return re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)


def _login(client):
    token = _csrf(client, "/login")
    return client.post("/login", data={"username": "doctor", "password": PASSWORD,
                                       "csrf_token": token})


# ---------------- public pages -----------------------------------------------------------
def test_public_pages_render(client):
    home = client.get("/")
    assert home.status_code == 200 and b"not for clinical" in home.data.lower()
    about = client.get("/about").get_data(as_text=True)
    for output in ("Malignancy probability", "Mirai", "European (CEU)", "Pakistani reference"):
        assert output in about
    assert "Not yet validated" in about


# ---------------- auth / CSRF ------------------------------------------------------------
@pytest.mark.parametrize("url", ["/dashboard", "/mammogram", "/mirai", "/prs", "/report.pdf"])
def test_diagnostic_routes_require_login(client, url):
    r = client.get(url)
    assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_wrong_password_rejected(client):
    token = _csrf(client, "/login")
    r = client.post("/login", data={"username": "doctor", "password": "nope", "csrf_token": token})
    assert r.status_code == 200 and b"Invalid username or password" in r.data
    assert client.get("/dashboard").status_code == 302


def test_login_then_dashboard(client):
    assert _login(client).status_code == 302
    page = client.get("/dashboard").get_data(as_text=True)
    assert "Not available" in page


def test_post_without_csrf_rejected(client):
    r = client.post("/login", data={"username": "doctor", "password": PASSWORD})
    assert r.status_code == 400
    _login(client)
    r = client.post("/prs", data={"vcf": (io.BytesIO(b"x"), "a.vcf")},
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_open_redirect_blocked(client):
    token = _csrf(client, "/login")
    r = client.post("/login?next=//evil.example", data={"username": "doctor",
                    "password": PASSWORD, "csrf_token": token})
    assert r.headers["Location"].endswith("/dashboard")


# ---------------- uploads: model-unavailable path ----------------------------------------
def test_mammogram_upload_model_unavailable_shows_labels(client, tmp_path):
    _login(client)
    token = _csrf(client, "/mammogram")
    r = client.post("/mammogram", data={"csrf_token": token,
                                        "image": (io.BytesIO(b"\x89PNG fake"), "scan.png")},
                    content_type="multipart/form-data")
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "Model not available" in html
    assert "Validation label" in html and "Not yet validated" in html
    assert "CBIS-DDSM" in html
    assert list(tmp_path.iterdir()) == []  # uploaded file deleted after processing


def test_mirai_requires_four_views(client):
    _login(client)
    token = _csrf(client, "/mirai")
    r = client.post("/mirai", data={"csrf_token": token,
                                    "L-CC": (io.BytesIO(b"d"), "lcc.dcm")},
                    content_type="multipart/form-data")
    assert "missing L-MLO, R-CC, R-MLO" in r.get_data(as_text=True)


def test_prs_shows_both_references(client):
    _login(client)
    token = _csrf(client, "/prs")
    html = client.post("/prs", data={"csrf_token": token,
                                     "vcf": (io.BytesIO(b"##fileformat=VCFv4.2\n"), "me.vcf.gz")},
                       content_type="multipart/form-data").get_data(as_text=True)
    assert "European (CEU) reference" in html and "Pakistani reference" in html
    assert "Model not available" in html


def test_bad_extension_rejected(client):
    _login(client)
    token = _csrf(client, "/prs")
    html = client.post("/prs", data={"csrf_token": token, "vcf": (io.BytesIO(b"x"), "evil.exe")},
                       content_type="multipart/form-data").get_data(as_text=True)
    assert "Unsupported file type" in html


# ---------------- PDF report -------------------------------------------------------------
def test_pdf_report_contains_labels(client):
    _login(client)
    token = _csrf(client, "/prs")
    client.post("/prs", data={"csrf_token": token, "vcf": (io.BytesIO(b"x"), "me.vcf")},
                content_type="multipart/form-data")
    r = client.get("/report.pdf")
    assert r.status_code == 200 and r.mimetype == "application/pdf"
    assert r.data.startswith(b"%PDF")


def test_pdf_text_has_label_and_disclaimer(app):
    labels = app.extensions["labels"]
    res = make_result(labels, "m2_prs_pak", title="PRS", available=False, message="Model not available: x")
    pdf = build_pdf([res])
    # fpdf2 compresses page streams by default; check the text via an uncompressed build.
    from fpdf import FPDF
    import app.report as report_mod
    orig = FPDF.__init__

    def no_compress(self, *a, **k):
        orig(self, *a, **k)
        self.set_compression(False)

    FPDF.__init__ = no_compress
    try:
        raw = report_mod.build_pdf([res])
    finally:
        FPDF.__init__ = orig
    assert pdf.startswith(b"%PDF")
    assert b"Not yet validated" in raw
    assert b"NOT FOR CLINICAL" in raw
    assert b"HGDP" in raw


# ---------------- labels are mandatory ---------------------------------------------------
def test_result_without_label_is_rejected(app):
    with pytest.raises(MissingLabelError):
        LabelledResult(label=None, title="x", available=False)
    with pytest.raises(MissingLabelError):
        make_result(app.extensions["labels"], "unknown_output", title="x", available=False)
    with pytest.raises(MissingLabelError):
        build_pdf([{"title": "raw dict without label"}])


def test_validated_label_needs_value_and_source():
    with pytest.raises(ValueError):
        ValidationLabel(key="k", output="o", module="m", datasets=[], populations=[],
                        not_validated_on=[], metric="AUC", status="validated")


def test_unavailable_result_cannot_carry_numbers(app):
    with pytest.raises(ValueError):
        make_result(app.extensions["labels"], "m3_probability", title="x", available=False,
                    values={"probability": 0.9})
