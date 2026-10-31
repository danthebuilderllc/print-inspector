"""Images for the saved results and the PDF report (BGR uint8, drawn with OpenCV).

* ``annotated_comparison`` - artwork and registered print side by side with
  every finding boxed and labelled (#n defects, Tn text, Bn barcodes, Cn
  colour), critical and ignored regions, and a header with the verdict and
  the file names;
* ``difference_heatmap`` - the comparison's combined difference score over
  the print, with a scale (1 = detection threshold).

OpenCV draws ASCII only; names with other characters are transliterated here
(the JSON and PDF reports keep them exactly).
"""
from __future__ import annotations

import unicodedata

import cv2
import numpy as np

from print_inspector.core.decision import OperatorMark
from print_inspector.core.inspection_record import (
    BARCODE,
    COLOUR,
    DISMISSED,
    STRUCTURAL,
    TEXT,
    InspectionRecord,
    collect_findings,
)
from print_inspector.core.models import InspectionStatus
from print_inspector.core.regions import RegionKind

MAX_PANEL_SIDE = 1800  # long side of each image panel
HEATMAP_MAX_SCORE = 4.0  # score shown at the top of the colour scale


def _bgr(hex_colour: str) -> tuple[int, int, int]:
    value = hex_colour.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return b, g, r


# The same colours as the boxes in the application.
FINDING_COLOURS = {
    STRUCTURAL: _bgr("#ef4444"), TEXT: _bgr("#f97316"), BARCODE: _bgr("#a855f7"), COLOUR: _bgr("#14b8a6"),
}
DISMISSED_COLOUR = _bgr("#9ca3af")
CRITICAL_REGION_COLOUR = _bgr("#dc2626")
IGNORED_REGION_COLOUR = (128, 128, 128)
STATUS_COLOURS = {
    InspectionStatus.PASS: _bgr("#15803d"), InspectionStatus.FAIL: _bgr("#b91c1c"),
    InspectionStatus.REVIEW: _bgr("#b45309"), InspectionStatus.NOT_INSPECTED: _bgr("#4b5563"),
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def ascii_text(text: str) -> str:
    """``text`` with accents removed and other non-ASCII characters replaced by '?'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c if ord(c) < 128 else ("" if unicodedata.combining(c) else "?") for c in decomposed)


def _fit(image: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    factor = min(1.0, MAX_PANEL_SIDE / max(h, w))
    if factor < 1.0:
        image = cv2.resize(image, (round(w * factor), round(h * factor)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image), factor


def _bgr_image(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()


def _text(image: np.ndarray, text: str, origin: tuple[int, int], scale: float, colour, thickness: int = 1,
          background=None) -> None:
    text = ascii_text(text)
    if background is not None:
        (tw, th), base = cv2.getTextSize(text, _FONT, scale, thickness)
        x, y = origin
        cv2.rectangle(image, (x - 2, y - th - 3), (x + tw + 2, y + base), background, -1)
    cv2.putText(image, text, origin, _FONT, scale, colour, thickness, cv2.LINE_AA)


def _hatch(image: np.ndarray, box: tuple[int, int, int, int], colour, step: int) -> None:
    x, y, w, h = box
    roi = image[y : y + h, x : x + w]
    if roi.size == 0:
        return
    shade = roi.copy()
    shade[:] = colour
    cv2.addWeighted(roi, 0.55, shade, 0.45, 0, dst=roi)
    for offset in range(-h, w, step):
        cv2.line(roi, (offset, h), (offset + h, 0), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), colour, 1)


def _dashed_rect(image: np.ndarray, box: tuple[int, int, int, int], colour, thickness: int, dash: int) -> None:
    x, y, w, h = box
    for x0, y0, x1, y1 in ((x, y, x + w, y), (x, y + h, x + w, y + h), (x, y, x, y + h), (x + w, y, x + w, y + h)):
        length = max(abs(x1 - x0), abs(y1 - y0))
        for start in range(0, length, dash * 2):
            end = min(start + dash, length)
            t0, t1 = start / max(length, 1), end / max(length, 1)
            p0 = (round(x0 + (x1 - x0) * t0), round(y0 + (y1 - y0) * t0))
            p1 = (round(x0 + (x1 - x0) * t1), round(y0 + (y1 - y0) * t1))
            cv2.line(image, p0, p1, colour, thickness, cv2.LINE_AA)


def _draw_regions(image: np.ndarray, record: InspectionRecord, factor: float, scale: float) -> None:
    h, w = image.shape[:2]
    for region in record.regions:
        box = region.rect(w, h)
        if region.kind is RegionKind.IGNORE:
            _hatch(image, box, IGNORED_REGION_COLOUR, max(6, round(10 * scale)))
            label = f"IGNORED: {region.name}"
            colour = IGNORED_REGION_COLOUR
        else:
            _dashed_rect(image, box, CRITICAL_REGION_COLOUR, max(1, round(2 * scale)), max(4, round(8 * scale)))
            label = f"CRITICAL: {region.name}"
            colour = CRITICAL_REGION_COLOUR
        _text(image, label, (box[0] + 3, min(h - 3, box[1] + box[3] - 4)), 0.45 * scale, (255, 255, 255),
              1, background=colour)


def _draw_findings(image: np.ndarray, record: InspectionRecord, factor: float, scale: float) -> None:
    pad = max(3, round(4 * scale))
    thickness = max(1, round(2 * scale))
    for row in collect_findings(record):
        x, y, w, h = (round(v * factor) for v in row.box)
        colour = DISMISSED_COLOUR if row.verdict == DISMISSED else FINDING_COLOURS[row.inspection]
        cv2.rectangle(image, (x - pad, y - pad), (x + w + pad, y + h + pad), colour, thickness)
        label = row.label + (" ok?" if row.operator is OperatorMark.NEEDS_REVIEW else "")
        _text(image, label, (x - pad, max(int(16 * scale), y - pad - 4)), 0.55 * scale, (255, 255, 255),
              max(1, thickness - 1), background=colour)


def _panel_title(panel: np.ndarray, title: str, scale: float) -> np.ndarray:
    band = np.full((round(30 * scale), panel.shape[1], 3), 245, np.uint8)
    _text(band, title, (8, round(21 * scale)), 0.6 * scale, (40, 40, 40), max(1, round(scale)))
    return np.vstack([band, panel])


def _header(width: int, record: InspectionRecord, scale: float) -> np.ndarray:
    status = record.decision.status
    rows = collect_findings(record)
    height = round(84 * scale)
    band = np.full((height, width, 3), 255, np.uint8)
    badge_w = round(150 * scale)
    cv2.rectangle(band, (0, 0), (badge_w, height - 1), STATUS_COLOURS[status], -1)
    _text(band, status.name, (round(12 * scale), round(52 * scale)), 1.1 * scale, (255, 255, 255),
          max(2, round(2 * scale)))
    original = record.original.path.name if record.original is not None else "-"
    printed = record.printed.path.name if record.printed is not None else "-"
    counts = {k: sum(1 for r in rows if r.inspection == k) for k in (STRUCTURAL, TEXT, BARCODE, COLOUR)}
    lines = [
        f"Original: {original}    Printed: {printed}",
        f"Inspected {record.created:%Y-%m-%d %H:%M:%S}    Sensitivity {record.sensitivity}"
        + (f"    Profile: {record.profile}" if record.profile else ""),
        f"Findings: {counts[STRUCTURAL]} defects (#), {counts[TEXT]} text (T), {counts[BARCODE]} barcode (B), "
        f"{counts[COLOUR]} colour (C)",
    ]
    for i, line in enumerate(lines):
        _text(band, line, (badge_w + round(12 * scale), round((22 + 24 * i) * scale)), 0.52 * scale, (30, 30, 30),
              max(1, round(scale)))
    return band


def _legend(width: int, scale: float, regions: bool) -> np.ndarray:
    band = np.full((round(30 * scale), width, 3), 255, np.uint8)
    items = [("Defect #", FINDING_COLOURS[STRUCTURAL]), ("Text T", FINDING_COLOURS[TEXT]),
             ("Barcode B", FINDING_COLOURS[BARCODE]), ("Colour C", FINDING_COLOURS[COLOUR]),
             ("Dismissed", DISMISSED_COLOUR)]
    if regions:
        items += [("Critical region", CRITICAL_REGION_COLOUR), ("Ignored region", IGNORED_REGION_COLOUR)]
    x = 8
    for label, colour in items:
        cv2.rectangle(band, (x, round(8 * scale)), (x + round(14 * scale), round(22 * scale)), colour, -1)
        _text(band, label, (x + round(20 * scale), round(20 * scale)), 0.45 * scale, (40, 40, 40))
        x += round((40 + 9 * len(label)) * scale)
    return band


def annotated_comparison(record: InspectionRecord) -> np.ndarray | None:
    """Artwork | registered print with all findings and regions; None before registration."""
    r = record.registration
    if r is None or r.registered is None or r.reference is None:
        return None
    artwork, factor = _fit(_bgr_image(r.reference))
    printed, _ = _fit(_bgr_image(r.registered))
    scale = max(1.0, max(artwork.shape[:2]) / 650)  # legible when both panels share one page width
    for panel in (artwork, printed):
        _draw_regions(panel, record, factor, scale)
        _draw_findings(panel, record, factor, scale)
    gap = np.full((artwork.shape[0], round(12 * scale), 3), 255, np.uint8)
    body = np.hstack([
        _panel_title(artwork, "Original artwork", scale),
        np.vstack([np.full((round(30 * scale), gap.shape[1], 3), 245, np.uint8), gap]),
        _panel_title(printed, "Printed label (registered to the artwork)", scale),
    ])
    return np.vstack([_header(body.shape[1], record, scale), body, _legend(body.shape[1], scale, bool(record.regions))])


def difference_heatmap(record: InspectionRecord) -> np.ndarray | None:
    """The combined difference score over the print with a scale; None before the comparison."""
    c, r = record.comparison, record.registration
    if c is None or r is None or r.registered is None:
        return None
    score = np.clip(c.combined_score.astype(np.float32) / HEATMAP_MAX_SCORE, 0, 1)
    colour = cv2.applyColorMap((score * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    gray = cv2.cvtColor(cv2.cvtColor(_bgr_image(r.registered), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    # The print, lightened, where nothing differs; the score colour where it does.
    light = 255 - (255 - gray.astype(np.float32)) * 0.45
    weight = np.clip(score * 3, 0.0, 0.9)[..., None]
    out = (light * (1 - weight) + colour.astype(np.float32) * weight).astype(np.uint8)
    if r.mask is not None:
        outside = r.mask == 0
        out[outside] = (out[outside] * 0.35).astype(np.uint8)
    contours, _ = cv2.findContours(c.candidate_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 1)
    out, factor = _fit(out)
    scale = max(0.8, max(out.shape[:2]) / 1100)
    for region in record.regions:
        if region.kind is RegionKind.IGNORE:
            _hatch(out, region.rect(out.shape[1], out.shape[0]), IGNORED_REGION_COLOUR, max(6, round(10 * scale)))

    # Colour scale on the right: 0 at the bottom, HEATMAP_MAX_SCORE at the top, 1 = detection threshold.
    bar_w, margin = round(26 * scale), round(70 * scale)
    h = out.shape[0]
    side = np.full((h, bar_w + margin, 3), 255, np.uint8)
    top, bottom = round(20 * scale), h - round(20 * scale)
    ramp = np.linspace(1, 0, max(2, bottom - top))[:, None]
    side[top:bottom, 6 : 6 + bar_w] = cv2.applyColorMap((ramp * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    for value in (0, 1, 2, 3, HEATMAP_MAX_SCORE):
        y = round(bottom - (bottom - top) * value / HEATMAP_MAX_SCORE)
        cv2.line(side, (6 + bar_w, y), (12 + bar_w, y), (0, 0, 0), 1)
        label = f"{value:g}" + (" = limit" if value == 1 else "+" if value == HEATMAP_MAX_SCORE else "")
        _text(side, label, (14 + bar_w, y + 5), 0.4 * scale, (0, 0, 0))
    body = np.hstack([out, side])
    title = np.full((round(34 * scale), body.shape[1], 3), 255, np.uint8)
    _text(title, f"Difference score (1 = detection threshold, sensitivity {record.sensitivity}); "
                 "white outlines = difference regions", (8, round(23 * scale)), 0.5 * scale, (30, 30, 30))
    return np.vstack([title, body])
