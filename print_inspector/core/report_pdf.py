"""The PDF inspection report.

The report is built from the same summary as ``inspection.json`` (so the two
never disagree) plus the report images, laid out as rich text by Qt and
written with ``QPdfWriter`` - offline, with no extra dependency. Every page
has a footer with the printed file name, the verdict and the page number.

Contents: verdict and reasons; input files; date and time; profile;
capture quality, label alignment and registration results; the findings of
every inspection with their coordinates, severity and operator decision;
the annotated comparison image and the difference heatmap; defect evidence;
the regions and zones; the inspection settings.

Qt needs a (Q)GuiApplication for fonts. In the application one exists; for
use from scripts one is created on first use (main thread only).
"""
from __future__ import annotations

import html
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QMarginsF, QPointF, QRectF, QSizeF, Qt, QUrl
from PySide6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QImage,
    QPageLayout,
    QPageSize,
    QPainter,
    QPdfWriter,
    QTextDocument,
)

from print_inspector.core.inspection_record import (
    REJECTS,
    NEEDS_REVIEW,
    InspectionRecord,
    collect_findings,
    grid_scale,
)
from print_inspector.core.decision import OperatorMark

RESOLUTION_DPI = 200
FONT_PT = 8.5
MARGIN_MM = 14.0
FOOTER_MM = 8.0
CONTENT_WIDTH_PX = 680  # usable A4 width in the document's 96-dpi pixels (for image widths)
_STATUS_HEX = {"PASS": "#15803d", "FAIL": "#b91c1c", "REVIEW": "#b45309", "NOT_INSPECTED": "#4b5563"}
_VERDICT_HEX = {REJECTS: "#b91c1c", NEEDS_REVIEW: "#b45309"}
_app = None  # a QApplication created here (kept alive for the process)


class ReportError(RuntimeError):
    """The PDF report could not be written."""


def _ensure_app() -> None:
    global _app
    if QGuiApplication.instance() is not None:
        return
    if threading.current_thread() is not threading.main_thread():
        raise ReportError("The PDF report needs the Qt application (start it from the main thread).")
    from PySide6.QtWidgets import QApplication  # a QApplication also serves widgets created later

    _app = QApplication([])


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}g}" if abs(value) < 1e5 else f"{value:.0f}"
    if isinstance(value, (list, tuple)):
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
        return (" × " if numeric else ", ").join(_number(v, digits) for v in value)
    return str(value)


def _table(headers: list[str], rows: list[list[str]], widths: list[int] | None = None, raw: bool = False) -> str:
    """An HTML table; cells are escaped unless ``raw``."""
    head = "".join(
        f"<th align='left' bgcolor='#e5e7eb'{f' width={widths[i]}%' if widths else ''}>{_e(h)}</th>"
        for i, h in enumerate(headers)
    )
    body = "".join(
        "<tr>" + "".join(f"<td>{c if raw else _e(c)}</td>" for c in row) + "</tr>" for row in rows
    )
    return (
        "<table width='100%' border='1' cellspacing='0' cellpadding='3' style='border-collapse: collapse; "
        f"border-color: #d1d5db;'><thead><tr>{head}</tr></thead>{body}</table>"
    )


def _key_values(pairs: list[tuple[str, str]], columns: int = 2) -> str:
    """Label/value pairs in ``columns`` column pairs."""
    rows = []
    per_row = [pairs[i : i + columns] for i in range(0, len(pairs), columns)]
    for chunk in per_row:
        cells = "".join(f"<td bgcolor='#f3f4f6'>{_e(k)}</td><td>{_e(v)}</td>" for k, v in chunk)
        cells += "<td></td><td></td>" * (columns - len(chunk))
        rows.append(f"<tr>{cells}</tr>")
    return (
        "<table width='100%' border='1' cellspacing='0' cellpadding='2' style='border-collapse: collapse; "
        f"border-color: #d1d5db;'>{''.join(rows)}</table>"
    )


def _check_block(title: str, check: dict[str, Any] | None, metrics: list[tuple[str, str, str]]) -> str:
    if check is None:
        return f"<h3>{_e(title)}</h3><p>Not performed.</p>"
    level = check["level"]
    colour = {"good": "#15803d", "marginal": "#b45309", "poor": "#b91c1c"}.get(level, "#374151")
    out = [f"<h3>{_e(title)} – <span style='color:{colour}'>{_e(level.upper())}</span></h3>"]
    if check["issues"]:
        out.append("<ul>" + "".join(
            f"<li><b>{_e(i['severity'])}</b>: {_e(i['message'])}</li>" for i in check["issues"]
        ) + "</ul>")
    else:
        out.append("<p>No problems found.</p>")
    values = check.get("metrics") or {}
    pairs = []
    for key, label, kind in metrics:
        value = values.get(key)
        if value is None:
            continue
        if kind == "percent" and isinstance(value, (int, float)):
            text = f"{value * 100:.1f} %"
        elif kind == "enum" and isinstance(value, str):
            text = value.replace("_", " ")
        else:
            text = _number(value) + (f" {kind}" if kind not in ("", "percent", "enum") else "")
        pairs.append((label, text))
    if pairs:
        out.append(_key_values(pairs))
    return "".join(out)


QUALITY_METRICS = [
    ("megapixels", "Megapixels", ""), ("blur_sigma_px", "Blur (σ)", "px"),
    ("clipped_fraction", "Overexposed", "percent"), ("largest_glare_fraction", "Largest glare", "percent"),
    ("dark_fraction", "Dark area", "percent"), ("lighting_unevenness", "Lighting unevenness", ""),
    ("effective_pixels_per_mm", "Resolution on the label", "px/mm"),
    ("label_long_side_mm", "Label long side", "mm"), ("label_size_assumed", "Label size assumed", ""),
]
ALIGNMENT_METRICS = [
    ("method", "Method", "enum"), ("edge_fit_rms_px", "Edge fit (RMS)", "px"),
    ("corner_angle_deviation_deg", "Corner angle deviation", "°"), ("aspect_error", "Aspect ratio error", "percent"),
    ("similarity", "Similarity to artwork", ""), ("residual_fraction", "Residual misalignment", "percent"),
    ("feature_inliers", "Feature matches (inliers)", ""), ("quarter_turns", "Quarter turns", ""),
]
REGISTRATION_METRICS = [
    ("method", "Method", "enum"), ("residual_median_px", "Residual (median)", "px"),
    ("residual_p95_px", "Residual (95th percentile)", "px"), ("residual_max_px", "Residual (max)", "px"),
    ("verified_tiles", "Verified tiles", ""), ("structured_tiles", "Structured tiles", ""),
    ("coverage", "Coverage", "percent"), ("structure_similarity", "Structure similarity", ""),
    ("mask_fraction", "Label area compared", "percent"), ("local_correction_applied", "Local correction", ""),
]
COMPARISON_METRICS = [
    ("sensitivity", "Sensitivity", "enum"), ("threshold_factor", "Threshold factor", ""),
    ("candidate_count", "Difference regions", ""), ("candidate_area_fraction", "Area with differences", "percent"),
    ("compared_fraction", "Label area compared", "percent"), ("model_similarity", "Similarity to artwork", ""),
    ("alignment_tolerance_px", "Alignment tolerance", "px"), ("seconds", "Time", "s"),
]


def _position(record: InspectionRecord, x: float, y: float) -> tuple[str, str]:
    """(mm from the artwork's top-left, artwork pixel) of a comparison-grid point."""
    px_per_mm = record.defects.px_per_mm if record.defects is not None else None
    scale = grid_scale(record)
    approx = "~" if record.defects is not None and record.defects.scale_estimated else ""
    mm = f"{approx}{x / px_per_mm:.1f}, {y / px_per_mm:.1f}" if px_per_mm else "–"
    px = f"{x / scale[0]:.0f}, {y / scale[1]:.0f}" if scale else "–"
    return mm, px


def _size_mm(record: InspectionRecord, w: float, h: float) -> str:
    px_per_mm = record.defects.px_per_mm if record.defects is not None else None
    return f"{w / px_per_mm:.1f} × {h / px_per_mm:.1f}" if px_per_mm else "–"


def _image_tag(name: str, width: int = CONTENT_WIDTH_PX) -> str:
    return f"<p align='center'><img src='img://{_e(name)}' width='{width}'></p>"


def _resource_width(image: np.ndarray, max_height_ratio: float = 1.25) -> int:
    """Display width that keeps tall images on one page."""
    h, w = image.shape[:2]
    return int(min(CONTENT_WIDTH_PX, CONTENT_WIDTH_PX * max_height_ratio * w / max(h, 1)))


def report_html(record: InspectionRecord, summary: dict[str, Any], images: dict[str, np.ndarray]) -> str:
    """The report as Qt rich text; ``images`` (name -> BGR) are referenced as ``img://name``."""
    cfg = record.config.report
    verdict = summary["verdict"]
    status = verdict["status"]
    info = summary["report"]
    parts: list[str] = []
    org = f"<p style='color:#6b7280'>{_e(cfg.organisation)}</p>" if cfg.organisation else ""
    parts.append(
        f"{org}<h1>{_e(cfg.title)}</h1>"
        f"<table width='100%' cellpadding='6'><tr><td width='22%' bgcolor='{_STATUS_HEX.get(status, '#4b5563')}' "
        f"align='center'><span style='color:white; font-size:20pt; font-weight:bold'>{_e(status)}</span></td>"
        "<td>" + "<br>".join(f"{_e(r)}" for r in verdict["reasons"]) + "</td></tr></table>"
    )
    files = summary["files"]
    pairs = [
        ("Original artwork", info["original_file"] or "–"),
        ("Printed label photo", info["printed_file"] or "–"),
        ("Inspection date and time", info["inspected_at"]),
        ("Application", f"{summary['application']['name']} {summary['application']['version']}"),
        ("Inspection profile", info["profile"] or "none (settings of this computer)"),
        ("Sensitivity", summary["settings"]["sensitivity"]),
        ("Inspector", cfg.inspector or "–"),
        ("Colour", summary["color"]["mode"] if summary.get("color") else "not analysed"),
    ]
    parts.append("<h2>Inspection</h2>" + _key_values(pairs))
    for role, label in (("original", "Original"), ("printed", "Printed")):
        f = files.get(role)
        if f:
            dpi = f" · {f['dpi'][0]:.0f} dpi" if f.get("dpi") else ""
            parts.append(f"<p style='color:#4b5563'>{label}: {_e(f['path'])} · {f['width']} × {f['height']} px · "
                         f"{_e(f['format'])}{dpi} · {f['size_bytes'] / 1024:.0f} KB</p>")

    overview = summary["findings_overview"]
    parts.append(
        "<h2>Result</h2>" + _key_values([
            ("Detected defects (structural)", str(overview["structural"])),
            ("Text findings", str(overview["text"]) if summary.get("text") else "not inspected"),
            ("Barcode findings", str(overview["barcode"]) if summary.get("barcodes") else "not inspected"),
            ("Colour findings", str(overview["colour"]) if summary.get("color") else "not analysed"),
            ("Rejecting the label", str(overview["rejects"])),
            ("Needing review", str(overview["needs_review"])),
            ("Reviewed by the operator", f"{overview['reviewed']} of {overview['total']}"),
            ("Dismissed by the operator", str(overview["dismissed"])),
        ])
    )
    if verdict["check_errors"]:
        parts.append("<p style='color:#b91c1c'>" + "<br>".join(_e(e) for e in verdict["check_errors"]) + "</p>")

    checks = summary["checks"]
    parts.append("<h2>Capture and alignment quality</h2>")
    parts.append(_check_block("Photo capture quality", checks["capture_quality"], QUALITY_METRICS))
    parts.append(_check_block("Label alignment (perspective correction)", checks["alignment"], ALIGNMENT_METRICS))
    parts.append(_check_block("Registration to the artwork", checks["registration"], REGISTRATION_METRICS))
    parts.append(_check_block("Comparison", checks["comparison"], COMPARISON_METRICS))

    rows = collect_findings(record)
    parts.append("<h2 style='page-break-before: always'>Findings and operator decisions</h2>")
    if record.defects is not None:
        unit = "estimated mm" if record.defects.scale_estimated else "mm"
        parts.append(
            f"<p>Positions are the centre of each finding, in {unit} from the top-left corner of the artwork "
            "and in artwork pixels. Sizes are the bounding box. "
            + (f"Operator: {_e(cfg.inspector)}." if cfg.inspector else "") + "</p>"
        )
    if rows:
        table_rows = []
        for r in rows:
            mm, px = _position(record, *r.centre)
            colour = _VERDICT_HEX.get(r.verdict)
            verdict_cell = f"<span style='color:{colour}'><b>{_e(r.verdict)}</b></span>" if colour else _e(r.verdict)
            area = f"<br><i>{_e(r.area)}</i>" if r.area else ""
            table_rows.append([
                f"<b>{_e(r.label)}</b>", _e(r.inspection), _e(r.description) + area, _e(r.severity),
                "reliable" if r.reliable else "uncertain", _e(mm), _e(px), _e(_size_mm(record, *r.box[2:])),
                _e(r.operator.label), verdict_cell,
            ])
        parts.append(_table(
            ["ID", "Check", "Finding", "Severity", "Detection", "Position (mm)", "Artwork px", "Size (mm)",
             "Operator decision", "Verdict"],
            table_rows, [5, 8, 23, 8, 9, 10, 9, 8, 11, 9], raw=True,
        ))
    else:
        parts.append("<p>No findings.</p>")
    review = summary["operator_review"]
    if review["decisions"]:
        parts.append(f"<p><b>Operator decisions:</b> {len(review['decisions'])} finding(s) reviewed"
                     + (f" by {_e(cfg.inspector)}" if cfg.inspector else "") + ". "
                     + "; ".join(f"{_e(d['finding'])}: {_e(d['decision_label'])}" for d in review["decisions"])
                     + ".</p>")
    else:
        parts.append("<p><b>Operator decisions:</b> none – every finding is assessed automatically.</p>")
    if summary.get("barcodes"):
        parts.append(f"<p style='color:#4b5563'>Barcodes: {_e(summary['barcodes']['note'])}</p>")
    if summary.get("color"):
        parts.append(f"<p style='color:#4b5563'>Colour ({_e(summary['color']['mode'])}): "
                     f"{_e(summary['color']['note'])}</p>")

    if "annotated_comparison.png" in images:
        parts.append("<h2 style='page-break-before: always'>Annotated comparison</h2>"
                     + _image_tag("annotated_comparison.png", _resource_width(images["annotated_comparison.png"])))
    if "difference_heatmap.png" in images:
        parts.append("<h2>Difference heatmap</h2>"
                     + _image_tag("difference_heatmap.png", _resource_width(images["difference_heatmap.png"], 0.9)))
    if "color_heatmap.png" in images:
        parts.append("<h2 style='page-break-before: always'>Colour difference (ΔE00 vs artwork)</h2>"
                     f"<p>{_e(summary['color']['note'])}</p>"
                     + _image_tag("color_heatmap.png", _resource_width(images["color_heatmap.png"], 0.9)))
    crops = [n for n in images if n.startswith("defect_")]
    if crops:
        parts.append("<h2 style='page-break-before: always'>Defect evidence</h2>"
                     "<p>Each row: original artwork | printed label | difference, enlarged; the defect is outlined "
                     "in red.</p>")
        defects = {f"defect_{d.id:02d}.png": d for d in record.defects.defects} if record.defects else {}
        for name in crops:
            d = defects.get(name)
            if d is not None:
                mark = record.marks.get(d.id, OperatorMark.UNREVIEWED)
                parts.append(f"<p><b>#{d.id}</b> {_e(d.category_label)} – {_e(d.severity.value)}, "
                             f"{_e(d.reliability.value)}, operator: {_e(mark.label)}</p>")
            parts.append(_image_tag(name, min(int(CONTENT_WIDTH_PX * 0.8), images[name].shape[1] // 2)))
        hidden = len(record.defects.defects) - len(crops) if record.defects else 0
        if hidden > 0:
            parts.append(f"<p>{hidden} more defect crop(s) are saved as PNG files next to this report.</p>")

    setup = summary["setup"]
    parts.append("<h2 style='page-break-before: always'>Inspection setup</h2>")
    parts.append(f"<p>Profile: <b>{_e(setup['profile'] or 'none')}</b>"
                 + (" (changed since it was saved)" if setup.get("profile_modified") else "")
                 + (f" – {_e(setup['profile_file'])}" if setup.get("profile_file") else "") + "</p>")
    zone_rows = []
    for r in setup["regions"]:
        zone_rows.append([r["name"], "critical region" if r["kind"] == "critical" else "ignored region",
                          _fraction_box(record, r), "not inspected" if r["kind"] == "ignore" else "strict"])
    for z in setup["text_zones"]:
        zone_rows.append([z["name"], "text zone" + (" (critical)" if z["critical"] else ""), _fraction_box(record, z),
                          z["expected_text"] or "as in the artwork"])
    for z in setup["barcode_zones"]:
        zone_rows.append([z["name"], f"barcode zone, {z['format']}" + (" (critical)" if z["critical"] else ""),
                          _fraction_box(record, z), z["expected_value"] or "as in the artwork"])
    if zone_rows:
        parts.append(_table(["Name", "Type", "Position x, y, w × h (mm)", "Expected"], zone_rows, [20, 25, 25, 30]))
    else:
        parts.append("<p>No regions or zones: the whole label is inspected with the general settings.</p>")

    settings = summary["settings"]
    parts.append("<h3>Settings</h3><p>All values used for this inspection (the same as in inspection.json).</p>")
    titles = {
        "decision": "Acceptance criteria", "defects": "Defect detection and severity",
        "comparison": "Comparison", "text": "Text verification", "barcodes": "Barcodes", "color": "Colour tolerances",
    }
    for key, title in titles.items():
        values = settings.get(key)
        if values:
            pairs = [(k.replace("_", " "), _number(v)) for k, v in values.items()]
            parts.append(f"<h4>{_e(title)}</h4>" + _key_values(pairs, 3))
    parts.append(
        "<p style='color:#6b7280'>Generated offline on this computer. Uncertain findings are never passed "
        "automatically: they give REVIEW. Barcode results are decoding checks, not ISO/IEC 15416 / 15415 grades; "
        "colour results without calibration are approximate.</p>"
    )
    return (
        "<html><head><style>"
        "h1 { font-size: 16pt; margin: 0; } h2 { font-size: 12pt; margin-top: 14px; } "
        "h3 { font-size: 10pt; margin-top: 10px; } h4 { font-size: 9pt; margin-top: 8px; } "
        "td, th { font-size: 7.5pt; } p, li { font-size: 8.5pt; }"
        "</style></head><body>" + "".join(parts) + "</body></html>"
    )


def _fraction_box(record: InspectionRecord, zone: dict[str, Any]) -> str:
    """Position of a zone (fractions of the artwork) in mm, or in % without a scale."""
    original = record.original
    px_per_mm = record.defects.px_per_mm if record.defects is not None else None
    scale = grid_scale(record)
    if original is not None and px_per_mm and scale:
        w_mm = original.working.shape[1] * scale[0] / px_per_mm
        h_mm = original.working.shape[0] * scale[1] / px_per_mm
        return (f"{zone['x'] * w_mm:.1f}, {zone['y'] * h_mm:.1f}, "
                f"{zone['width'] * w_mm:.1f} × {zone['height'] * h_mm:.1f}")
    return (f"{zone['x']:.1%}, {zone['y']:.1%}, {zone['width']:.1%} × {zone['height']:.1%} of the artwork")


def _qimage(image: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(image[:, :, ::-1]) if image.ndim == 3 else np.ascontiguousarray(image)
    h, w = rgb.shape[:2]
    fmt = QImage.Format.Format_RGB888 if rgb.ndim == 3 else QImage.Format.Format_Grayscale8
    return QImage(rgb.data, w, h, rgb.strides[0], fmt).copy()


def write_pdf(path: Path, record: InspectionRecord, summary: dict[str, Any], images: dict[str, np.ndarray]) -> int:
    """Write the report to ``path``; returns the number of pages. Raises ReportError."""
    _ensure_app()
    path = Path(path)
    writer = QPdfWriter(str(path))
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    writer.setPageMargins(QMarginsF(MARGIN_MM, MARGIN_MM, MARGIN_MM, MARGIN_MM), QPageLayout.Unit.Millimeter)
    writer.setResolution(RESOLUTION_DPI)
    writer.setTitle(f"{record.config.report.title} – {summary['report']['printed_file'] or 'inspection'}")
    writer.setCreator(f"{summary['application']['name']} {summary['application']['version']}")

    doc = QTextDocument()
    doc.documentLayout().setPaintDevice(writer)
    font = QFont(QGuiApplication.font())
    font.setPointSizeF(FONT_PT)
    doc.setDefaultFont(font)
    for name, image in images.items():
        doc.addResource(QTextDocument.ResourceType.ImageResource, QUrl(f"img://{name}"), _qimage(image))
    doc.setHtml(report_html(record, summary, images))

    page = writer.pageLayout().paintRectPixels(RESOLUTION_DPI)
    footer = FOOTER_MM / 25.4 * RESOLUTION_DPI
    content_h = page.height() - footer
    doc.setPageSize(QSizeF(page.width(), content_h))
    pages = doc.pageCount()
    footer_text = (
        f"{record.config.report.title} · {summary['report']['printed_file'] or '–'} · "
        f"{summary['report']['inspected_at']} · {summary['verdict']['status']}"
    )
    painter = QPainter()
    if not painter.begin(writer):
        raise ReportError(f"Could not write the PDF report to {path}.")
    try:
        small = QFont(font)
        small.setPointSizeF(7)
        for index in range(pages):
            if index:
                writer.newPage()
            painter.save()
            painter.translate(0, -index * content_h)
            doc.drawContents(painter, QRectF(0, index * content_h, page.width(), content_h))
            painter.restore()
            painter.setPen(QColor("#6b7280"))
            painter.setFont(small)
            line_y = content_h + footer * 0.25
            painter.drawLine(QPointF(0, line_y), QPointF(page.width(), line_y))
            box = QRectF(0, line_y, page.width(), footer * 0.75)
            painter.drawText(box, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), footer_text)
            painter.drawText(box, int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                             f"Page {index + 1} of {pages}")
    finally:
        painter.end()
    if not path.is_file() or path.stat().st_size == 0:
        raise ReportError(f"The PDF report could not be written to {path}.")
    return pages
