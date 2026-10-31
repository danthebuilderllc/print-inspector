"""Saving an inspection: a folder with the reports, tables and images.

Layout of ``<parent>/<photo name>_<date-time>_<VERDICT>/``:

* ``report.pdf`` - the formatted inspection report (core/report_pdf.py);
* ``inspection.json`` - all inspection data: verdict and reasons, input
  files, date and time, profile, all check results and metrics, every finding
  with its coordinates, severity and operator decision, regions and zones,
  and the settings used;
* ``findings.csv`` - one row per finding of every inspection (defects #n,
  text Tn, barcodes Bn, colour Cn) with position, severity, operator decision
  and the inspection it belongs to (file names, date, verdict) - opens in Excel
  and can be concatenated across inspections;
* ``defects.csv`` - the structural defects with all their measurements;
* ``text.csv`` - one row per text finding, if the text was inspected;
* ``colors.csv`` / ``color_findings.csv`` and ``color_heatmap*.png`` - the
  colour areas, colour findings and ΔE00 heatmaps, if colour was analysed;
* ``barcodes.csv`` - one row per barcode (value expected / decoded, status,
  measurements) and ``barcode_findings.csv``, if barcodes were inspected;
* ``annotated_comparison.png`` - artwork | print with every finding and region
  marked; ``difference_heatmap.png`` - the difference score with a scale;
* the images passed in (e.g. the registered print);
* ``defect_<n>.png`` - original | printed | difference crops of each defect.

Everything is written locally; nothing is uploaded.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from print_inspector.core.barcode_inspection import NOT_ISO
from print_inspector.core.barcode_zones import zones_to_data as barcode_zones_to_data
from print_inspector.core.color_inspection import APPROXIMATE_NOTE, heatmap_image
from print_inspector.core.decision import OperatorMark
from print_inspector.core.defects import Defect
from print_inspector.core.image_io import LoadedImage
from print_inspector.core.inspection_record import (  # noqa: F401 - InspectionRecord is re-exported
    DISMISSED,
    NEEDS_REVIEW,
    REJECTS,
    InspectionRecord,
    collect_findings,
    grid_scale,
)
from print_inspector.core.regions import regions_to_data
from print_inspector.core.report_images import annotated_comparison, difference_heatmap
from print_inspector.core.report_pdf import ReportError, write_pdf
from print_inspector.core.text_zones import zones_to_data as text_zones_to_data

SUMMARY_FILE = "inspection.json"
REPORT_FILE = "report.pdf"
FINDINGS_FILE = "findings.csv"
ANNOTATED_FILE = "annotated_comparison.png"
HEATMAP_FILE = "difference_heatmap.png"
DEFECTS_FILE = "defects.csv"
TEXT_FILE = "text.csv"
BARCODES_FILE = "barcodes.csv"
BARCODE_FINDINGS_FILE = "barcode_findings.csv"
FINDINGS_COLUMNS = (
    "id", "check", "problem", "description", "severity", "detection", "operator", "verdict", "region_or_zone",
    "x_mm", "y_mm", "width_mm", "height_mm", "artwork_x_px", "artwork_y_px", "artwork_width_px",
    "artwork_height_px", "original_file", "printed_file", "inspected_at", "inspection_status", "profile",
)
BARCODE_CSV_COLUMNS = (
    "symbol", "name", "zone", "critical", "format", "expected_value", "expected_from", "artwork_value",
    "print_value", "print_format", "status", "strips_decoded", "unused_error_correction", "ec_level",
    "wrong_modules", "bar_width_change_modules", "findings",
)
COLORS_FILE = "colors.csv"
COLOR_FINDINGS_FILE = "color_findings.csv"
COLOR_CSV_COLUMNS = (
    "area", "paper", "x_mm", "y_mm", "area_mm2", "design_L", "design_a", "design_b",
    "print_L", "print_a", "print_b", "delta_e00", "local_spread_delta_e00", "mode",
)
COLOR_FINDING_COLUMNS = (
    "id", "problem", "delta_e00", "peak_delta_e00", "dL", "da", "db", "area_mm2", "severity", "detection",
    "operator", "x_mm", "y_mm", "mode", "notes",
)
BARCODE_FINDING_COLUMNS = (
    "id", "problem", "symbol", "zone", "format", "expected", "decoded", "severity", "detection", "operator",
    "x_mm", "y_mm", "notes",
)
TEXT_CSV_COLUMNS = (
    "id", "problem", "expected", "read", "severity", "detection", "operator", "zone", "critical",
    "x_mm", "y_mm", "artwork_ocr_confidence", "print_ocr_confidence", "ink_missing", "image_score", "notes",
)
CSV_COLUMNS = (
    "id", "category", "category_verified", "severity", "detection", "operator",
    "x_mm", "y_mm", "length_mm", "thickness_mm", "area_mm2", "scale_estimated",
    "artwork_x_px", "artwork_y_px", "grid_x", "grid_y", "grid_width", "grid_height",
    "score", "max_difference", "delta_e",
)


class ExportError(RuntimeError):
    """The results could not be written."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(_jsonable(k)): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def results_folder_name(record: InspectionRecord) -> str:
    stem = record.printed.path.stem if record.printed is not None else "inspection"
    stem = re.sub(r"[^\w.-]+", "_", stem).strip("._") or "inspection"
    return f"{stem[:60]}_{record.created:%Y%m%d-%H%M%S}_{record.decision.status.name}"


def _check(result) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "level": result.level.value,
        "issues": [{"code": i.code, "severity": i.severity.value, "message": i.message} for i in result.issues],
        "metrics": _jsonable(result.metrics),
    }


def _file(image: LoadedImage | None) -> dict[str, Any] | None:
    if image is None:
        return None
    return {
        "path": str(image.path),
        "format": image.file_format,
        "size_bytes": image.file_size_bytes,
        "width": image.width,
        "height": image.height,
        "dpi": list(image.dpi) if image.dpi else None,
    }


def defect_position_mm(defect: Defect, px_per_mm: float) -> tuple[float, float]:
    """Defect centre in mm from the artwork's top-left corner."""
    return defect.centroid[0] / px_per_mm, defect.centroid[1] / px_per_mm


def _defect_entry(d: Defect, record: InspectionRecord) -> dict[str, Any]:
    report = record.defects
    assert report is not None
    entry = {f.name: _jsonable(getattr(d, f.name)) for f in dataclasses.fields(d) if f.name != "evidence"}
    entry["category_label"] = d.category_label
    entry["operator"] = record.marks.get(d.id, OperatorMark.UNREVIEWED).value
    entry["position_mm"] = list(defect_position_mm(d, report.px_per_mm))
    scale = grid_scale(record)
    if scale is not None:
        entry["artwork_centroid_px"] = [d.centroid[0] / scale[0], d.centroid[1] / scale[1]]
    return entry


def _text_summary(record: InspectionRecord) -> dict[str, Any] | None:
    report = record.text
    if report is None:
        return None
    px_per_mm = record.defects.px_per_mm if record.defects is not None else None
    findings = []
    for f in report.findings:
        entry = _jsonable(f)
        entry["label"] = f.label
        entry["description"] = f.describe()
        entry["operator"] = record.text_marks.get(f.id, OperatorMark.UNREVIEWED).value
        if px_per_mm:
            entry["position_mm"] = [(f.x + f.width / 2) / px_per_mm, (f.y + f.height / 2) / px_per_mm]
        findings.append(entry)
    return {
        "available": report.available,
        "engine": report.engine,
        "level": report.level.value,
        "issues": [{"code": i.code, "severity": i.severity.value, "message": i.message} for i in report.issues],
        "lines": [_jsonable(line) for line in report.lines],
        "findings": findings,
        "unread_regions": [list(b) for b in report.unread_regions],
        "notes": list(report.notes),
        "seconds": report.seconds,
    }


def text_csv_rows(record: InspectionRecord) -> list[dict[str, Any]]:
    report = record.text
    if report is None:
        return []
    px_per_mm = record.defects.px_per_mm if record.defects is not None else None
    rows = []
    for f in report.findings:
        rows.append({
            "id": f.label,
            "problem": f.kind.label,
            "expected": f.expected,
            "read": f.read,
            "severity": f.severity.value,
            "detection": "reliable" if f.reliable else "uncertain",
            "operator": record.text_marks.get(f.id, OperatorMark.UNREVIEWED).value,
            "zone": f.zone or "",
            "critical": "yes" if f.critical else "no",
            "x_mm": round((f.x + f.width / 2) / px_per_mm, 2) if px_per_mm else "",
            "y_mm": round((f.y + f.height / 2) / px_per_mm, 2) if px_per_mm else "",
            "artwork_ocr_confidence": "" if f.artwork_confidence is None else round(f.artwork_confidence, 1),
            "print_ocr_confidence": "" if f.print_confidence is None else round(f.print_confidence, 1),
            "ink_missing": round(f.ink_loss, 3),
            "image_score": round(f.image_score, 2),
            "notes": " ".join(f.notes),
        })
    return rows


def _mm(record: InspectionRecord, x: float, y: float) -> tuple[Any, Any]:
    px_per_mm = record.defects.px_per_mm if record.defects is not None else None
    return (round(x / px_per_mm, 2), round(y / px_per_mm, 2)) if px_per_mm else ("", "")


def _barcode_summary(record: InspectionRecord) -> dict[str, Any] | None:
    report = record.barcodes
    if report is None:
        return None
    findings = []
    for f in report.findings:
        entry = _jsonable(f)
        entry["label"] = f.label
        entry["description"] = f.describe()
        entry["operator"] = record.barcode_marks.get(f.id, OperatorMark.UNREVIEWED).value
        findings.append(entry)
    return {
        "available": report.available,
        "library": report.library,
        "note": NOT_ISO,
        "level": report.level.value,
        "issues": [{"code": i.code, "severity": i.severity.value, "message": i.message} for i in report.issues],
        "symbols": [_jsonable(s) for s in report.symbols],
        "findings": findings,
        "notes": list(report.notes),
        "seconds": report.seconds,
    }


def barcode_csv_rows(record: InspectionRecord) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    report = record.barcodes
    if report is None:
        return [], []
    symbols = [{
        "symbol": s.id, "name": s.name, "zone": s.zone or "", "critical": "yes" if s.critical else "no",
        "format": s.format_label, "expected_value": s.expected_value, "expected_from": s.expected_source,
        "artwork_value": s.artwork_value, "print_value": s.print_value, "print_format": s.print_format,
        "status": s.status.value,
        "strips_decoded": f"{s.strips[0]}/{s.strips[1]}" if s.strips else "",
        "unused_error_correction": "" if s.unused_ec is None else round(s.unused_ec, 2),
        "ec_level": s.ec_level,
        "wrong_modules": "" if s.module_errors is None else f"{s.module_errors}/{s.module_count}",
        "bar_width_change_modules": "" if s.bar_growth_modules is None else round(s.bar_growth_modules, 2),
        "findings": " ".join(f"B{i}" for i in s.finding_ids),
    } for s in report.symbols]
    findings = []
    for f in report.findings:
        x_mm, y_mm = _mm(record, f.x + f.width / 2, f.y + f.height / 2)
        findings.append({
            "id": f.label, "problem": f.kind.label, "symbol": f.symbol_id or "", "zone": f.zone or "",
            "format": f.format, "expected": f.expected, "decoded": f.decoded, "severity": f.severity.value,
            "detection": "reliable" if f.reliable else "uncertain",
            "operator": record.barcode_marks.get(f.id, OperatorMark.UNREVIEWED).value,
            "x_mm": x_mm, "y_mm": y_mm, "notes": " ".join(f.notes),
        })
    return symbols, findings


def _color_summary(record: InspectionRecord) -> dict[str, Any] | None:
    report = record.color
    if report is None:
        return None
    findings = []
    for f in report.findings:
        entry = _jsonable(f)
        entry["label"] = f.label
        entry["description"] = f.describe()
        entry["operator"] = record.color_marks.get(f.id, OperatorMark.UNREVIEWED).value
        findings.append(entry)
    return {
        "mode": report.mode,
        "note": "Calibrated measurement (CIEDE2000, CIELAB D50)." if report.calibrated else APPROXIMATE_NOTE,
        "mode_reasons": list(report.mode_reasons),
        "profile": report.profile,
        "paper": {
            "artwork_lab": report.paper_design_lab, "photo_lab": report.paper_photo_lab,
            "delta_e00": report.paper_photo_delta_e, "channel_balance_rgb": report.channel_balance,
        },
        "lighting_range": list(report.lighting_range),
        "global_shift_lab": report.global_shift,
        "measured_fraction": report.meaningful_fraction,
        "issues": [{"code": i.code, "severity": i.severity.value, "message": i.message} for i in report.issues],
        "areas": [_jsonable(a) for a in report.areas],
        "findings": findings,
        "notes": list(report.notes),
        "seconds": report.seconds,
    }


def color_csv_rows(record: InspectionRecord) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    report = record.color
    if report is None:
        return [], []
    areas = []
    for a in report.areas:
        x_mm, y_mm = _mm(record, a.x + a.width / 2, a.y + a.height / 2)
        areas.append({
            "area": a.id, "paper": "yes" if a.paper else "no", "x_mm": x_mm, "y_mm": y_mm,
            "area_mm2": round(a.area_mm2, 1),
            "design_L": round(a.design_lab[0], 2), "design_a": round(a.design_lab[1], 2), "design_b": round(a.design_lab[2], 2),
            "print_L": round(a.measured_lab[0], 2), "print_a": round(a.measured_lab[1], 2), "print_b": round(a.measured_lab[2], 2),
            "delta_e00": round(a.delta_e, 2), "local_spread_delta_e00": round(a.local_spread, 2), "mode": report.mode,
        })
    findings = []
    for f in report.findings:
        x_mm, y_mm = _mm(record, f.x + f.width / 2, f.y + f.height / 2)
        findings.append({
            "id": f.label, "problem": f.kind.label, "delta_e00": round(f.delta_e, 2),
            "peak_delta_e00": round(f.peak_delta_e, 2), "dL": round(f.dl, 2), "da": round(f.da, 2),
            "db": round(f.db, 2), "area_mm2": round(f.area_mm2, 2), "severity": f.severity.value,
            "detection": "reliable" if f.reliable else "uncertain",
            "operator": record.color_marks.get(f.id, OperatorMark.UNREVIEWED).value,
            "x_mm": x_mm, "y_mm": y_mm, "mode": report.mode, "notes": " ".join(f.notes),
        })
    return areas, findings


def _file_name(image: LoadedImage | None) -> str:
    return image.path.name if image is not None else ""


def finding_csv_rows(record: InspectionRecord) -> list[dict[str, Any]]:
    """One row per finding of every inspection, with the inspection it belongs to on each row."""
    px_per_mm = record.defects.px_per_mm if record.defects is not None else None
    scale = grid_scale(record)
    context = {
        "original_file": _file_name(record.original),
        "printed_file": _file_name(record.printed),
        "inspected_at": record.created.isoformat(timespec="seconds"),
        "inspection_status": record.decision.status.name,
        "profile": record.profile,
    }
    rows = []
    for f in collect_findings(record):
        x, y, w, h = f.box
        cx, cy = f.centre
        rows.append({
            "id": f.label, "check": f.inspection, "problem": f.problem, "description": f.description,
            "severity": f.severity, "detection": "reliable" if f.reliable else "uncertain",
            "operator": f.operator.value, "verdict": f.verdict, "region_or_zone": f.area,
            "x_mm": round(cx / px_per_mm, 2) if px_per_mm else "",
            "y_mm": round(cy / px_per_mm, 2) if px_per_mm else "",
            "width_mm": round(w / px_per_mm, 2) if px_per_mm else "",
            "height_mm": round(h / px_per_mm, 2) if px_per_mm else "",
            "artwork_x_px": round(cx / scale[0], 1) if scale else "",
            "artwork_y_px": round(cy / scale[1], 1) if scale else "",
            "artwork_width_px": round(w / scale[0], 1) if scale else "",
            "artwork_height_px": round(h / scale[1], 1) if scale else "",
            **context,
        })
    return rows


def _findings_overview(record: InspectionRecord) -> dict[str, int]:
    rows = collect_findings(record)
    return {
        "total": len(rows),
        "structural": sum(r.inspection == "structural" for r in rows),
        "text": sum(r.inspection == "text" for r in rows),
        "barcode": sum(r.inspection == "barcode" for r in rows),
        "colour": sum(r.inspection == "colour" for r in rows),
        "rejects": sum(r.verdict == REJECTS for r in rows),
        "needs_review": sum(r.verdict == NEEDS_REVIEW for r in rows),
        "dismissed": sum(r.verdict == DISMISSED for r in rows),
        "reviewed": sum(r.operator is not OperatorMark.UNREVIEWED for r in rows),
        "critical": sum(r.severity == "critical" for r in rows),
        "major": sum(r.severity == "major" for r in rows),
        "minor": sum(r.severity == "minor" for r in rows),
    }


def _operator_review(record: InspectionRecord) -> dict[str, Any]:
    decisions = [
        {"finding": r.label, "check": r.inspection, "decision": r.operator.value, "decision_label": r.operator.label,
         "automatic_severity": r.severity, "verdict": r.verdict}
        for r in collect_findings(record) if r.operator is not OperatorMark.UNREVIEWED
    ]
    return {"inspector": record.config.report.inspector, "decisions": decisions}


def build_summary(record: InspectionRecord) -> dict[str, Any]:
    decision = record.decision
    report = record.defects
    cfg = record.config
    return {
        "application": {"name": cfg.app.name, "version": cfg.app.version},
        "created": record.created.isoformat(timespec="seconds"),
        "report": {
            "title": cfg.report.title,
            "organisation": cfg.report.organisation,
            "inspector": cfg.report.inspector,
            "original_file": _file_name(record.original),
            "printed_file": _file_name(record.printed),
            "inspected_at": record.created.isoformat(timespec="seconds"),
            "profile": record.profile,
        },
        "findings_overview": _findings_overview(record),
        "operator_review": _operator_review(record),
        "verdict": {
            "status": decision.status.name,
            "reasons": list(decision.reasons),
            "failing_defects": list(decision.failing),
            "defects_needing_review": list(decision.review),
            "minor_defects": decision.minor_count,
            "failing_text_findings": list(decision.text_failing),
            "text_findings_needing_review": list(decision.text_review),
            "failing_barcode_findings": list(decision.barcode_failing),
            "barcode_findings_needing_review": list(decision.barcode_review),
            "failing_color_findings": list(decision.color_failing),
            "color_findings_needing_review": list(decision.color_review),
            "check_errors": list(record.errors),
        },
        "files": {"original": _file(record.original), "printed": _file(record.printed)},
        "checks": {
            "capture_quality": _check(record.quality),
            "alignment": _check(record.alignment),
            "registration": _check(record.registration),
            "comparison": _check(record.comparison),
        },
        "defects": None if report is None else {
            "px_per_mm": report.px_per_mm,
            "scale_estimated": report.scale_estimated,
            "items": [_defect_entry(d, record) for d in report.defects],
            "ignored_as_noise": [_jsonable(f) for f in report.filtered],
        },
        "text": _text_summary(record),
        "barcodes": _barcode_summary(record),
        "color": _color_summary(record),
        "setup": {
            "profile": record.profile,
            "profile_file": record.profile_file,
            "profile_modified": record.profile_modified,
            "regions": regions_to_data(record.regions),
            "text_zones": text_zones_to_data(record.text_zones),
            "barcode_zones": barcode_zones_to_data(record.barcode_zones),
        },
        "settings": {
            "sensitivity": record.sensitivity,
            "decision": _jsonable(cfg.decision),
            "defects": _jsonable(cfg.defects),
            "comparison": _jsonable(cfg.comparison),
            "text": _jsonable(cfg.text),
            "barcodes": _jsonable(cfg.barcodes),
            "color": _jsonable(cfg.color),
        },
    }


def defect_csv_rows(record: InspectionRecord) -> list[dict[str, Any]]:
    report = record.defects
    if report is None:
        return []
    scale = grid_scale(record)
    rows = []
    for d in report.defects:
        x_mm, y_mm = defect_position_mm(d, report.px_per_mm)
        rows.append({
            "id": d.id,
            "category": d.category.label,
            "category_verified": "yes" if d.category_certain else "no",
            "severity": d.severity.value,
            "detection": d.reliability.value,
            "operator": record.marks.get(d.id, OperatorMark.UNREVIEWED).value,
            "x_mm": round(x_mm, 2),
            "y_mm": round(y_mm, 2),
            "length_mm": round(d.length_mm, 3),
            "thickness_mm": round(d.thickness_mm, 3),
            "area_mm2": round(d.area_mm2, 4),
            "scale_estimated": "yes" if report.scale_estimated else "no",
            "artwork_x_px": round(d.centroid[0] / scale[0], 1) if scale else "",
            "artwork_y_px": round(d.centroid[1] / scale[1], 1) if scale else "",
            "grid_x": d.x,
            "grid_y": d.y,
            "grid_width": d.width,
            "grid_height": d.height,
            "score": round(d.score, 2),
            "max_difference": round(d.max_difference, 1),
            "delta_e": round(d.delta_e, 2),
        })
    return rows


def evidence_strip(defect: Defect, scale: int = 4) -> np.ndarray:
    """Original | printed | difference crops side by side, enlarged without smoothing."""
    e = defect.evidence
    diff = cv2.applyColorMap(np.clip(e.difference.astype(np.float32) * 3, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    tiles = []
    for image in (e.artwork, e.printed, diff):
        h, w = image.shape[:2]
        big = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(
            cv2.resize(e.region, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(big, contours, -1, (0, 0, 255), 1)
        tiles.append(big)
        tiles.append(np.full((big.shape[0], 8, 3), 255, np.uint8))
    return np.hstack(tiles[:-1])


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ExportError(f"Could not encode {path.name}.")
    encoded.tofile(str(path))  # works with non-ASCII paths on Windows, unlike cv2.imwrite


def _write_csv(path: Path, columns, rows) -> None:
    # utf-8-sig so that Excel recognises the encoding (µ, ², non-ASCII file names).
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def report_images(record: InspectionRecord) -> dict[str, np.ndarray]:
    """The images every saved inspection gets (annotated comparison, heatmaps), by file name."""
    images: dict[str, np.ndarray] = {}
    annotated = annotated_comparison(record)
    if annotated is not None:
        images[ANNOTATED_FILE] = annotated
    heatmap = difference_heatmap(record)
    if heatmap is not None:
        images[HEATMAP_FILE] = heatmap
    if record.color is not None:
        limit = record.config.color.heatmap_max_delta_e
        background = record.registration.registered if record.registration is not None else None
        images["color_heatmap.png"] = heatmap_image(record.color.heatmap, limit, background)
        images["color_heatmap_local.png"] = heatmap_image(record.color.local_heatmap, limit, background)
    return images


def write_results(parent: Path, record: InspectionRecord, images: dict[str, np.ndarray]) -> Path:
    """Write the results folder under ``parent`` and return its path.

    ``images`` maps extra file names (``*.png``) to BGR uint8 images. Raises ExportError.
    """
    parent = Path(parent)
    if not parent.is_dir():
        raise ExportError(f"The folder does not exist: {parent}")
    folder = parent / results_folder_name(record)
    suffix = 1
    while folder.exists():
        suffix += 1
        folder = parent / f"{results_folder_name(record)}_{suffix}"
    try:
        folder.mkdir()
        summary = build_summary(record)
        (folder / SUMMARY_FILE).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )
        _write_csv(folder / FINDINGS_FILE, FINDINGS_COLUMNS, finding_csv_rows(record))
        _write_csv(folder / DEFECTS_FILE, CSV_COLUMNS, defect_csv_rows(record))
        if record.text is not None:
            _write_csv(folder / TEXT_FILE, TEXT_CSV_COLUMNS, text_csv_rows(record))
        if record.color is not None:
            area_rows, color_rows = color_csv_rows(record)
            _write_csv(folder / COLORS_FILE, COLOR_CSV_COLUMNS, area_rows)
            _write_csv(folder / COLOR_FINDINGS_FILE, COLOR_FINDING_COLUMNS, color_rows)
        if record.barcodes is not None:
            symbol_rows, finding_rows = barcode_csv_rows(record)
            _write_csv(folder / BARCODES_FILE, BARCODE_CSV_COLUMNS, symbol_rows)
            _write_csv(folder / BARCODE_FINDINGS_FILE, BARCODE_FINDING_COLUMNS, finding_rows)
        generated = report_images(record)
        for name, image in {**generated, **images}.items():
            _write_png(folder / name, image)
        crops: dict[str, np.ndarray] = {}
        if record.defects is not None:
            for d in record.defects.defects:
                crop = evidence_strip(d)
                _write_png(folder / f"defect_{d.id:02d}.png", crop)
                if len(crops) < record.config.report.max_evidence_crops:
                    crops[f"defect_{d.id:02d}.png"] = crop
    except OSError as exc:
        raise ExportError(f"Could not write the results to {folder}: {exc.strerror or exc}") from exc
    if record.config.report.pdf:
        try:
            write_pdf(folder / REPORT_FILE, record, summary, {**generated, **crops})
        except (ReportError, OSError) as exc:
            raise ExportError(f"The results were saved to {folder}, but the PDF report failed: {exc}") from exc
    return folder
