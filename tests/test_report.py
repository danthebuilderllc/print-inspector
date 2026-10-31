"""Inspection reports: findings of every kind, the CSV list, report images and the PDF report."""
import csv
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from print_inspector import synthetic as syn
from print_inspector.config import load_config
from print_inspector.core.comparison import compare_to_artwork
from print_inspector.core.decision import OperatorMark, decide
from print_inspector.core.defects import assess_defects
from print_inspector.core.image_io import ImageRole, LoadedImage
from print_inspector.core.inspection_record import DISMISSED, REJECTS, STRUCTURAL, collect_findings
from print_inspector.core.models import InspectionStatus, QualityLevel
from print_inspector.core.regions import InspectionRegion, RegionKind, apply_to_defects, inspection_mask
from print_inspector.core.report_images import (
    FINDING_COLOURS,
    STATUS_COLOURS,
    annotated_comparison,
    ascii_text,
    difference_heatmap,
)
from print_inspector.core.report_pdf import report_html, write_pdf
from print_inspector.core.results_export import (
    ANNOTATED_FILE,
    FINDINGS_COLUMNS,
    FINDINGS_FILE,
    HEATMAP_FILE,
    REPORT_FILE,
    SUMMARY_FILE,
    InspectionRecord,
    build_summary,
    finding_csv_rows,
    report_images,
    write_results,
)
from print_inspector.core.text_zones import TextZone
from tests.test_comparison import ARTWORK, full_mask, simulate

PX_PER_MM = 12.0
IGNORED = InspectionRegion("Variable data", RegionKind.IGNORE, 0.02, 0.02, 0.15, 0.1)


def _image(name: str, role: ImageRole, pixels: np.ndarray) -> LoadedImage:
    return LoadedImage(Path("C:/labels") / name, role, "PNG", 12345, pixels, pixels, dpi=(304.8, 304.8))


@pytest.fixture(scope="module")
def record():
    config = load_config(user_path=None)
    config = dataclasses.replace(config, report=dataclasses.replace(
        config.report, inspector="R. Müller", organisation="Example Labels"))
    printed, truth = syn.add_defects(ARTWORK, ["extra_mark", "missing_region"], seed=4)
    registered = simulate(printed)
    h, w = ARTWORK.shape[:2]
    x, y, bw, bh = truth[0].bbox
    critical = InspectionRegion("Dose", RegionKind.CRITICAL, (x - 20) / w, (y - 20) / h, (bw + 40) / w, (bh + 40) / h)
    regions = (critical, IGNORED)
    mask = inspection_mask(full_mask(ARTWORK.shape), regions)
    comparison = compare_to_artwork(registered, ARTWORK, mask, config.comparison, 0.9)
    report = assess_defects(comparison, ARTWORK, registered, PX_PER_MM, False, config.defects)
    report = apply_to_defects(report, regions, ARTWORK.shape, config.decision.critical_region_severity)
    assert len(report.defects) >= 2
    marks = {report.defects[0].id: OperatorMark.CONFIRMED, report.defects[-1].id: OperatorMark.DISMISSED}
    decision = decide(None, None, None, comparison, report, marks, config.decision)
    registration = SimpleNamespace(reference=ARTWORK, registered=registered, mask=mask,
                                   level=QualityLevel.GOOD, issues=(), metrics={"residual_p95_px": 0.12})
    return InspectionRecord(
        original=_image("artwork_é.png", ImageRole.ORIGINAL, ARTWORK),
        printed=_image("photo 01.jpg", ImageRole.PRINTED, registered),
        quality=None, alignment=None, registration=registration, comparison=comparison, defects=report,
        marks=marks, decision=decision, sensitivity="high", config=config, regions=regions,
        text_zones=(TextZone("Lot", 0.1, 0.8, 0.3, 0.1, expected_text="LOT 12345"),),
        profile="Product 4711", profile_file="C:/profiles/Product_4711.json",
    )


def test_findings_of_every_kind_in_one_list(record):
    rows = collect_findings(record)
    assert [r.label for r in rows] == [f"#{d.id}" for d in record.defects.defects]
    first, last = rows[0], rows[-1]
    assert first.inspection == STRUCTURAL and first.operator is OperatorMark.CONFIRMED and first.verdict == REJECTS
    assert last.verdict == DISMISSED
    assert any(r.area == "Dose" for r in rows)


def test_findings_csv_rows(record):
    rows = finding_csv_rows(record)
    d = record.defects.defects[0]
    row = rows[0]
    assert list(row) == list(FINDINGS_COLUMNS)
    assert row["id"] == f"#{d.id}" and row["operator"] == "confirmed" and row["verdict"] == "rejects"
    assert row["x_mm"] == round(d.centroid[0] / PX_PER_MM, 2) and row["artwork_x_px"] == round(d.centroid[0], 1)
    assert row["width_mm"] == round(d.width / PX_PER_MM, 2)
    assert row["original_file"] == "artwork_é.png" and row["printed_file"] == "photo 01.jpg"
    assert row["inspection_status"] == record.decision.status.name and row["profile"] == "Product 4711"
    assert row["inspected_at"] == record.created.isoformat(timespec="seconds")


def test_annotated_comparison_image(record):
    image = annotated_comparison(record)
    h, w = ARTWORK.shape[:2]
    assert image.dtype == np.uint8 and image.shape[1] > 2 * w and image.shape[0] > h
    colours = {tuple(int(v) for v in px) for px in image.reshape(-1, 3)[::7]}
    assert STATUS_COLOURS[record.decision.status] in colours  # verdict badge in the header
    assert FINDING_COLOURS[STRUCTURAL] in colours  # defect boxes
    assert annotated_comparison(dataclasses.replace(record, registration=None)) is None


def test_difference_heatmap_image(record):
    image = difference_heatmap(record)
    h, w = ARTWORK.shape[:2]
    assert image.shape[0] > h and image.shape[1] > w
    d = record.defects.defects[0]
    title = image.shape[0] - h  # the title band above the map
    cx, cy = int(d.centroid[0]), int(d.centroid[1]) + title
    paper = image[title + 5, w // 2].astype(int)
    assert abs(int(image[cy, cx].astype(int).sum()) - int(paper.sum())) > 60  # the defect stands out
    assert difference_heatmap(dataclasses.replace(record, comparison=None)) is None


def test_ascii_text_for_image_captions():
    assert ascii_text("Müller – ΔE") == "Muller ? ?E"


def test_summary_has_report_header_setup_and_operator_review(record):
    summary = build_summary(record)
    info = summary["report"]
    assert info["original_file"] == "artwork_é.png" and info["printed_file"] == "photo 01.jpg"
    assert info["inspected_at"] == record.created.isoformat(timespec="seconds") and info["profile"] == "Product 4711"
    assert info["inspector"] == "R. Müller"
    overview = summary["findings_overview"]
    assert overview["structural"] == len(record.defects.defects) and overview["reviewed"] == 2
    assert overview["dismissed"] == 1 and overview["rejects"] >= 1
    decisions = summary["operator_review"]["decisions"]
    assert {d["decision"] for d in decisions} == {"confirmed", "dismissed"}
    setup = summary["setup"]
    assert [r["name"] for r in setup["regions"]] == ["Dose", "Variable data"]
    assert setup["text_zones"][0]["expected_text"] == "LOT 12345"
    assert setup["profile_file"].endswith("Product_4711.json")
    assert summary["settings"]["sensitivity"] == "high"
    assert summary["settings"]["decision"]["critical_region_severity"] == "major"


def test_report_html_contains_every_required_item(record):
    summary = build_summary(record)
    html = report_html(record, summary, report_images(record))
    for text in ("artwork_é.png", "photo 01.jpg", summary["report"]["inspected_at"], "Photo capture quality",
                 "Label alignment", "Registration to the artwork", "Detected defects (structural)",
                 "Position (mm)", "Severity", "Operator decision", "Confirmed defect", "Dismissed",
                 record.decision.status.name, "Product 4711", "Acceptance criteria", "Colour tolerances",
                 "Dose", "Variable data", "LOT 12345", "R. Müller", "img://annotated_comparison.png",
                 "img://difference_heatmap.png"):
        assert text in html, text


def test_pdf_report(qapp, tmp_path, record):
    from PySide6.QtPdf import QPdfDocument

    summary = build_summary(record)
    images = report_images(record)
    path = tmp_path / "report.pdf"
    pages = write_pdf(path, record, summary, images)
    assert path.read_bytes().startswith(b"%PDF") and pages >= 3
    doc = QPdfDocument()
    doc.load(str(path))
    assert doc.pageCount() == pages
    text = " ".join(doc.getAllText(i).text() for i in range(pages))
    text = " ".join(text.split())
    d = record.defects.defects[0]
    for item in ("Print Inspection Report", "artwork_é.png", "photo 01.jpg", record.decision.status.name,
                 "Photo capture quality", "Label alignment", "Registration to the artwork",
                 f"#{d.id}", f"{d.centroid[0] / PX_PER_MM:.1f}", d.severity.value, "Confirmed defect",
                 "Product 4711", "Acceptance criteria", f"Page 1 of {pages}", "Annotated comparison",
                 "Difference heatmap"):
        assert item in text, item


def test_write_results_with_pdf_and_without(qapp, tmp_path, record):
    folder = write_results(tmp_path, record, {})
    for name in (REPORT_FILE, SUMMARY_FILE, FINDINGS_FILE, ANNOTATED_FILE, HEATMAP_FILE, "defects.csv"):
        assert (folder / name).is_file(), name
    with open(folder / FINDINGS_FILE, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(record.defects.defects) and rows[0]["check"] == "structural"
    summary = json.loads((folder / SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["report"]["printed_file"] == "photo 01.jpg"

    config = dataclasses.replace(record.config, report=dataclasses.replace(record.config.report, pdf=False))
    folder = write_results(tmp_path, dataclasses.replace(record, config=config), {})
    assert not (folder / REPORT_FILE).exists() and (folder / FINDINGS_FILE).is_file()


def test_pdf_shows_a_limited_number_of_defect_crops(record):
    config = dataclasses.replace(record.config, report=dataclasses.replace(record.config.report, max_evidence_crops=1))
    limited = dataclasses.replace(record, config=config)
    from print_inspector.core.results_export import evidence_strip

    first = record.defects.defects[0]
    html = report_html(limited, build_summary(limited), {f"defect_{first.id:02d}.png": evidence_strip(first)})
    assert f"{len(record.defects.defects) - 1} more defect crop(s)" in html


def test_status_colours_cover_every_verdict():
    assert set(STATUS_COLOURS) == set(InspectionStatus)
