"""PDF report: every result printed with its validation label and a not-for-clinical-use notice."""
from __future__ import annotations

from datetime import datetime, timezone

from fpdf import FPDF

from app.labels import LabelledResult, MissingLabelError

DISCLAIMER = ("RESEARCH PROTOTYPE - NOT FOR CLINICAL OR DIAGNOSTIC USE. Results must not be used "
              "to make decisions about any patient. Each result states the datasets and "
              "populations it was validated on; it has not been validated on Pakistani women "
              "unless stated.")


def _latin1(text: str) -> str:
    """Core PDF fonts are Latin-1 only; replace common Unicode punctuation."""
    table = {"–": "-", "—": "-", "‘": "'", "’": "'", "“": '"',
             "”": '"', "≤": "<=", "≥": ">=", "×": "x", "…": "..."}
    for k, v in table.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(results: list[LabelledResult], user: str | None = None) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    w = pdf.epw

    def para(text, size=10, style=""):
        pdf.set_font("Helvetica", style, size)
        pdf.multi_cell(w, 5, _latin1(text), new_x="LMARGIN", new_y="NEXT")

    para("MammoRisk+ report", 16, "B")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    para(f"Generated {stamp}" + (f" by {user}" if user else ""), 9)
    pdf.ln(2)
    pdf.set_text_color(160, 0, 0)
    para(DISCLAIMER, 10, "B")
    if any(getattr(r, "illustrative", False) for r in results):
        pdf.set_text_color(180, 71, 10)
        para("DEMO MODE: results marked ILLUSTRATIVE are sample output, NOT computed from any file "
             "and NOT real data.", 10, "B")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    if not results:
        para("No results in this session.")
    for r in results:
        if not isinstance(r, LabelledResult):
            raise MissingLabelError("refusing to print a result without a validation label")
        if r.illustrative:
            pdf.set_text_color(180, 71, 10)
            para("ILLUSTRATIVE - NOT REAL DATA", 10, "B")
            pdf.set_text_color(0, 0, 0)
        para(r.title, 12, "B")
        para(r.label.module, 9, "I")
        if r.available:
            for k, v in r.values.items():
                para(f"{k}: {v}")
        else:
            para(r.message)
        pdf.ln(1)
        for line in r.label.lines():
            para(line, 9)
        pdf.ln(4)
    return bytes(pdf.output())
