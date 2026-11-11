"""Frontend tests: demo-mode stamping, figures, reliability-matrix logic, project status, error pages."""
import io
import re
from datetime import date

import pytest
from werkzeug.security import generate_password_hash

from app import create_app, project_status, viz
from app.labels import REGIONS, ValidationLabel
from app.report import build_pdf
from app.services.demo import category, demo_results, percentile

PASSWORD = "test-password-123"


def _make(tmp_path, **extra):
    cfg = {"TESTING": True, "USERS": {"doctor": generate_password_hash(PASSWORD)},
           "UPLOAD_TMP_DIR": str(tmp_path),
           "PATHS": {"m3_checkpoint": str(tmp_path / "none.pt"), "m1_oncoserve_url": None,
                     "m2_reference_stats": str(tmp_path / "none.json")}}
    cfg.update(extra)
    return create_app(cfg)


def _csrf(client, url):
    html = client.get(url).get_data(as_text=True)
    return re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)


def _login(client):
    client.post("/login", data={"username": "doctor", "password": PASSWORD,
                                "csrf_token": _csrf(client, "/login")})


def _post_prs(client):
    token = _csrf(client, "/prs")
    return client.post("/prs", data={"csrf_token": token, "vcf": (io.BytesIO(b"x"), "me.vcf")},
                       content_type="multipart/form-data").get_data(as_text=True)


# ---------------- demo mode ----------------------------------------------------------------
def test_demo_mode_is_off_by_default(tmp_path):
    client = _make(tmp_path).test_client()
    _login(client)
    html = _post_prs(client)
    assert "Model not available" in html
    assert "Illustrative" not in html and "Demo mode" not in html


def test_demo_mode_results_are_stamped_and_still_unvalidated(tmp_path):
    client = _make(tmp_path, DEMO_MODE=True).test_client()
    _login(client)
    html = _post_prs(client)
    assert "Illustrative, not real data" in html
    assert "Demo mode" in html                           # page banner
    assert "Not yet validated" in html                   # the label is never upgraded by demo mode
    assert "The category changes with the reference" in html
    assert "<svg" in html and "Reference distribution" in html


def test_demo_results_cover_every_label_and_are_flagged(tmp_path):
    app = _make(tmp_path)
    labels = app.extensions["labels"]
    keys = tuple(labels)
    results = demo_results(labels, keys)
    assert len(results) == len(keys) and all(r.illustrative and r.available for r in results)
    assert all(r.label.status == "not_yet_validated" for r in results)


def test_demo_pdf_carries_illustrative_stamp(tmp_path):
    app = _make(tmp_path, DEMO_MODE=True)
    results = demo_results(app.extensions["labels"], ("m2_prs_ceu",))
    from fpdf import FPDF
    import app.report as report_mod
    orig = FPDF.__init__

    def no_compress(self, *a, **k):
        orig(self, *a, **k)
        self.set_compression(False)

    FPDF.__init__ = no_compress
    try:
        raw = report_mod.build_pdf(results)
    finally:
        FPDF.__init__ = orig
    assert b"ILLUSTRATIVE" in raw and b"NOT REAL DATA" in raw
    assert build_pdf(results).startswith(b"%PDF")


def test_illustrative_flag_survives_the_session_roundtrip(tmp_path):
    client = _make(tmp_path, DEMO_MODE=True).test_client()
    _login(client)
    _post_prs(client)
    assert client.get("/report.pdf").status_code == 200
    assert "Illustrative" in client.get("/dashboard").get_data(as_text=True)


def test_prs_demo_shows_a_real_reclassification():
    # the sample person must land in different categories under the two references (the point of O1)
    assert category(1.12) != category(0.31)
    assert percentile(0.0) == pytest.approx(50.0)
    assert category(2.4).startswith("High") and category(-1.0).startswith("Low")


# ---------------- figures --------------------------------------------------------------------
def test_figures_draw_only_the_numbers_they_are_given():
    assert "64%" in viz.probability_bar(0.64, 0.47)
    assert "decision threshold 47%" in viz.probability_bar(0.64, 0.47)
    bars = viz.risk_bars({"1-year": 0.008, "5-year": 0.047})
    assert "4.7%" in bars and "0.8%" in bars and "5 yr" in bars
    bell = viz.bell_curve(1.12, 86.9)
    assert "87th percentile" in bell and "z = +1.12" in bell


def test_result_viz_is_empty_for_unavailable_or_malformed(tmp_path):
    labels = _make(tmp_path).extensions["labels"]
    from app.labels import make_result
    unavailable = make_result(labels, "m3_probability", title="x", available=False, message="m")
    assert str(viz.result_viz(unavailable)) == ""
    broken = make_result(labels, "m3_probability", title="x", available=True, values={"oops": 1})
    assert str(viz.result_viz(broken)) == ""   # a malformed adapter result gets no figure, never a wrong one


def test_ordinal_suffixes():
    assert [viz.ordinal(n) for n in (1, 2, 3, 11, 12, 13, 21, 22, 62, 87, 100)] == \
        ["1st", "2nd", "3rd", "11th", "12th", "13th", "21st", "22nd", "62nd", "87th", "100th"]


# ---------------- reliability matrix ---------------------------------------------------------
def _label(status="not_yet_validated", regions=None, **kw):
    base = dict(key="k", output="o", module="m", datasets=[], populations=[], not_validated_on=[],
                metric="AUC", status=status, regions=regions or {})
    base.update(kw)
    return ValidationLabel(**base)


def test_region_state_cannot_overstate_validation():
    planned = _label(regions={"US": "planned", "EU": "reference"})
    assert planned.region_state("US") == "planned"        # a dataset is identified, nothing is validated
    assert planned.region_state("EU") == "reference"
    assert planned.region_state("PK") == "none"
    validated = _label("validated", {"US": "planned"}, value=0.8, source="docs/STATUS.md run x")
    assert validated.region_state("US") == "validated"
    assert _label("exploratory", {"US": "planned"}).region_state("US") == "exploratory"


def test_unknown_region_is_rejected():
    with pytest.raises(ValueError):
        _label(regions={"MARS": "planned"})


def test_landing_page_matrix_says_pakistan_is_not_validated(tmp_path):
    html = _make(tmp_path).test_client().get("/").get_data(as_text=True)
    for name in REGIONS.values():
        assert name in html
    assert "No output has been validated on Pakistani women yet" in html


# ---------------- project status --------------------------------------------------------------
def test_status_table_parses_the_real_status_file():
    rows = project_status.load_modules()
    assert rows, "docs/STATUS.md module table was not found"
    assert {"module", "owner", "status", "next"} <= set(rows[0])
    assert all("**" not in v and "`" not in v for r in rows for v in r.values())


def test_status_missing_file_is_empty(tmp_path):
    assert project_status.load_modules(tmp_path / "nope.md") == []


def test_timeline_positions():
    t = project_status.timeline(date(2026, 11, 11))
    assert 0 < t["today_pct"] < 25 and t["months"][0] == "Oct"
    assert all(r["start"] < r["end"] for r in t["rows"])
    assert project_status.timeline(date(2030, 1, 1))["today_pct"] is None


def test_dashboard_in_demo_mode_says_demo_output(tmp_path):
    client = _make(tmp_path, DEMO_MODE=True).test_client()
    _login(client)
    page = client.get("/dashboard").get_data(as_text=True)
    assert "Demo output only" in page and "Project schedule" in page


# ---------------- error pages ------------------------------------------------------------------
def test_404_page_is_styled_and_helpful(tmp_path):
    r = _make(tmp_path).test_client().get("/no-such-page")
    html = r.get_data(as_text=True)
    assert r.status_code == 404 and "Page not found" in html and "Back to start" in html


def test_missing_csrf_page_explains_what_to_do(tmp_path):
    r = _make(tmp_path).test_client().post("/login", data={"username": "a", "password": "b"})
    assert r.status_code == 400 and "Reload the page" in r.get_data(as_text=True)
