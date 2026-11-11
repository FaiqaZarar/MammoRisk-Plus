"""Result visualisations as inline SVG (no JavaScript, no chart library, works offline).

Each function turns the ``values`` of one result into a small accessible figure. Colours come from CSS
variables (see static/css/style.css) so figures follow the light/dark theme. Figures never invent data:
they only draw numbers the adapter returned, and ``result_viz`` returns an empty string when a result is
unavailable or its label has no figure.
"""
from __future__ import annotations

import math
from html import escape

from markupsafe import Markup

# Risk-category cut-offs on the reference percentile scale (D-14): low < 20th, average 20th-80th,
# elevated 80th-99th, high >= 99th. Expressed as z of the fitted normal reference.
Z20, Z80, Z99 = -0.8416212, 0.8416212, 2.3263479
BANDS = [(-4.0, Z20, "Low", "b1"), (Z20, Z80, "Average", "b2"),
         (Z80, Z99, "Elevated", "b3"), (Z99, 4.0, "High", "b4")]


def ordinal(n: float) -> str:
    """87 -> '87th', 62 -> '62nd', 21 -> '21st', 11 -> '11th'."""
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _pct(x: float, digits: int = 0) -> str:
    return f"{100 * x:.{digits}f}%"


def _svg(vb_w: int, vb_h: int, label: str, inner: str) -> Markup:
    return Markup(
        f'<svg class="fig" viewBox="0 0 {vb_w} {vb_h}" role="img" aria-label="{escape(label)}" '
        f'xmlns="http://www.w3.org/2000/svg">{inner}</svg>')


def probability_bar(p: float, threshold: float | None = None) -> Markup:
    """Horizontal 0-100% scale with the decision threshold and the predicted probability."""
    x0, x1, y = 24, 536, 62
    w = x1 - x0
    px = x0 + w * max(0.0, min(1.0, p))
    parts = [f'<rect class="f-track" x="{x0}" y="{y}" width="{w}" height="14" rx="7"/>',
             f'<rect class="f-fill" x="{x0}" y="{y}" width="{px - x0:.1f}" height="14" rx="7"/>']
    if threshold is not None:
        tx = x0 + w * threshold
        parts += [f'<line class="f-thr" x1="{tx:.1f}" y1="{y - 12}" x2="{tx:.1f}" y2="{y + 26}"/>',
                  f'<text class="f-small" x="{tx:.1f}" y="{y + 42}" text-anchor="middle">'
                  f'decision threshold {_pct(threshold)}</text>']
    parts += [f'<circle class="f-dot" cx="{px:.1f}" cy="{y + 7}" r="9"/>',
              f'<text class="f-big" x="{px:.1f}" y="{y - 20}" text-anchor="middle">{_pct(p)}</text>',
              f'<text class="f-axis" x="{x0}" y="{y + 42}" text-anchor="start">0%</text>',
              f'<text class="f-axis" x="{x1}" y="{y + 42}" text-anchor="end">100%</text>']
    return _svg(560, 120, f"Predicted malignancy probability {_pct(p)}", "".join(parts))


def risk_bars(risks: dict[str, float]) -> Markup:
    """Cumulative risk by year (Mirai gives 1-5 year risk). Bars share one scale starting at zero."""
    items = list(risks.items())
    top = max(v for _, v in items) if items else 0.0
    ymax = max(0.01, math.ceil(top * 1.3 * 100) / 100)
    x0, x1, base, height = 56, 540, 176, 130
    slot = (x1 - x0) / max(1, len(items))
    parts = []
    for frac in (0.0, 0.5, 1.0):
        gy = base - height * frac
        parts.append(f'<line class="f-grid" x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}"/>')
        parts.append(f'<text class="f-axis" x="{x0 - 8}" y="{gy + 4:.1f}" text-anchor="end">'
                     f'{_pct(ymax * frac, 1 if ymax < 0.1 else 0)}</text>')
    for i, (name, v) in enumerate(items):
        bw = slot * 0.46
        bx = x0 + slot * i + (slot - bw) / 2
        bh = height * (v / ymax)
        parts.append(f'<rect class="f-bar" x="{bx:.1f}" y="{base - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3"/>')
        parts.append(f'<text class="f-val" x="{bx + bw / 2:.1f}" y="{base - bh - 7:.1f}" text-anchor="middle">{_pct(v, 1)}</text>')
        parts.append(f'<text class="f-axis" x="{bx + bw / 2:.1f}" y="{base + 20}" text-anchor="middle">'
                     f'{escape(name.replace("-year", " yr"))}</text>')
    label = "Risk by year: " + ", ".join(f"{k} {_pct(v, 1)}" for k, v in items)
    return _svg(560, 210, label, "".join(parts))


def _pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def bell_curve(z: float, percentile: float | None = None) -> Markup:
    """Standard-normal reference with the four risk bands (D-14) and the person's z-score marked.

    In z units both references look identical; what differs is where the same person falls, which is
    exactly the reclassification question of O1.
    """
    x0, x1, base, top = 24, 536, 150, 28
    zmin, zmax = -3.5, 3.5
    sx = lambda v: x0 + (max(zmin, min(zmax, v)) - zmin) / (zmax - zmin) * (x1 - x0)
    sy = lambda v: base - (v / _pdf(0)) * (base - top)
    parts = []
    for lo, hi, name, cls in BANDS:
        lo_c, hi_c = max(lo, zmin), min(hi, zmax)
        n = 40
        pts = [(lo_c + (hi_c - lo_c) * k / n) for k in range(n + 1)]
        d = f"M{sx(lo_c):.1f},{base} " + " ".join(f"L{sx(p):.1f},{sy(_pdf(p)):.1f}" for p in pts) + f" L{sx(hi_c):.1f},{base} Z"
        parts.append(f'<path class="f-{cls}" d="{d}"/>')
        mid = (lo_c + hi_c) / 2
        anchor_x = sx(min(mid, 3.05))
        parts.append(f'<text class="f-axis" x="{anchor_x:.1f}" y="{base + 20}" text-anchor="middle">{name}</text>')
    curve = " ".join(f"{'M' if k == 0 else 'L'}{sx(zmin + (zmax - zmin) * k / 120):.1f},"
                     f"{sy(_pdf(zmin + (zmax - zmin) * k / 120)):.1f}" for k in range(121))
    parts.append(f'<path class="f-curve" d="{curve}"/>')
    parts.append(f'<line class="f-base" x1="{x0}" y1="{base}" x2="{x1}" y2="{base}"/>')
    mx = sx(z)
    parts.append(f'<line class="f-mark" x1="{mx:.1f}" y1="{top - 6}" x2="{mx:.1f}" y2="{base}"/>')
    parts.append(f'<circle class="f-dot" cx="{mx:.1f}" cy="{sy(_pdf(max(zmin, min(zmax, z)))):.1f}" r="7"/>')
    text = f"z = {z:+.2f}" + (f", {ordinal(percentile)} percentile" if percentile is not None else "")
    anchor = "end" if mx > (x0 + x1) / 2 + 80 else "start"
    off = -12 if anchor == "end" else 12
    parts.append(f'<text class="f-val" x="{mx + off:.1f}" y="{top + 6}" text-anchor="{anchor}">{escape(text)}</text>')
    return _svg(560, 184, f"Reference distribution with your score at {text}", "".join(parts))


def result_viz(result) -> Markup:
    """Pick the figure for a LabelledResult by its label key. Empty when there is nothing to draw."""
    if not getattr(result, "available", False):
        return Markup("")
    v, key = result.values, result.label.key
    try:
        if key == "m3_probability":
            return probability_bar(float(v["probability"]), v.get("threshold"))
        if key == "m1_mirai_risk":
            return risk_bars({k: float(x) for k, x in v.items() if str(k).endswith("-year")})
        if key in ("m2_prs_ceu", "m2_prs_pak"):
            return bell_curve(float(v["z"]), v.get("percentile"))
    except (KeyError, TypeError, ValueError):
        return Markup("")   # a malformed adapter result simply gets no figure, never a wrong one
    return Markup("")


def heatmap_placeholder_uri() -> str:
    """Synthetic, clearly-stamped stand-in for an explanation heatmap (demo mode only; not a mammogram)."""
    from urllib.parse import quote
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 512">'
        '<defs><radialGradient id="h" cx="62%" cy="38%" r="30%"><stop offset="0" stop-color="#ffd23f" stop-opacity=".95"/>'
        '<stop offset=".5" stop-color="#ee6c4d" stop-opacity=".55"/><stop offset="1" stop-color="#ee6c4d" stop-opacity="0"/>'
        '</radialGradient></defs><rect width="256" height="512" fill="#1b2430"/>'
        '<path d="M0 0H70C150 40 226 120 232 250C236 380 160 470 70 512H0Z" fill="#46566a"/>'
        '<rect width="256" height="512" fill="url(#h)"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" font-weight="700" font-size="22" fill="#ffffff" fill-opacity=".9">'
        '<text transform="rotate(-62 128 256)" x="12" y="256">ILLUSTRATIVE</text>'
        '<text transform="rotate(-62 128 256)" x="12" y="300" font-size="15">not a real mammogram</text></g></svg>')
    return "data:image/svg+xml;utf8," + quote(svg)
