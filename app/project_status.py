"""Project progress panel for the dashboard.

Reads the module table from ``docs/STATUS.md`` (the project's single source of truth) so the dashboard
can never drift from it. If the file is not there (the Docker image only ships ``app/``) the panel is
simply hidden. The schedule is the revised timeline (Table 7 / CLAUDE.md), kept here as data.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS_MD = REPO_ROOT / "docs" / "STATUS.md"

# (label, owner, first month index, last month index) with month 0 = Oct 2026 ... 7 = May 2027.
MONTHS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May"]
MONTH_STARTS = [date(2026, 10, 1), date(2026, 11, 1), date(2026, 12, 1), date(2027, 1, 1),
                date(2027, 2, 1), date(2027, 3, 1), date(2027, 4, 1), date(2027, 5, 1), date(2027, 6, 1)]
TIMELINE = [
    ("Data access", "Faiqa", 0, 0),
    ("Module 2: PRS, O1 (core)", "Pukhraj", 0, 1),
    ("Module 1: Mirai on EMBED, O4", "Faiqa", 1, 2),
    ("Module 3: classifier and explanations, O2 and O3", "Faiqa, Manaal", 1, 3),
    ("Platform, released end of January", "Manaal", 2, 3),
    ("Papers, thesis, final defence", "All", 4, 7),
]


def _clean(cell: str) -> str:
    return re.sub(r"\s+", " ", cell.replace("**", "").replace("`", "")).strip()


def load_modules(path: Path = STATUS_MD) -> list[dict]:
    """Rows of the module table in STATUS.md; ``[]`` if the file or table is missing."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows, in_table = [], False
    for line in lines:
        if line.startswith("| Module"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            break
        cells = [_clean(c) for c in line.strip().strip("|").split("|")]
        if len(cells) < 6 or set(cells[0]) <= set("-: "):
            continue
        rows.append(dict(zip(("module", "owner", "planned", "status", "latest", "next"), cells[:6])))
    return rows


def timeline(today: date | None = None) -> dict:
    """Grid positions for the schedule bars and the position of today's marker (percent of the span)."""
    today = today or date.today()
    start, end = MONTH_STARTS[0], MONTH_STARTS[-1]
    span = (end - start).days
    pos = (today - start).days / span * 100
    return {
        "months": MONTHS,
        "rows": [{"label": a, "owner": b, "start": c + 1, "end": d + 2} for a, b, c, d in TIMELINE],
        "today_pct": pos if 0 <= pos <= 100 else None,
        "today": today.strftime("%d %b %Y").lstrip("0"),
    }
