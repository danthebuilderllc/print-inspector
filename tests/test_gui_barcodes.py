"""Barcode inspection in the application (full pipeline from a synthetic phone photo)."""
import csv
import dataclasses
import json

import cv2
import pytest
import shiboken6
from PIL import Image
from PySide6.QtCore import QPointF, QRectF

from print_inspector import synthetic as syn
from print_inspector.core.barcode_inspection import BarcodeFindingKind, SymbolStatus
from print_inspector.core.barcode_zones import load_zones, zones_path
from print_inspector.core.decision import OperatorMark
from print_inspector.core.image_io import ImageRole
from print_inspector.core.models import InspectionStatus
from print_inspector.gui.barcode_panel import COL_BARCODE_OPERATOR
from print_inspector.gui.main_window import MainWindow
from tests.barcode_fakes import make_barcode_label, paint
from tests.conftest import write_image

TIMEOUT_MS = 60000


@pytest.fixture(scope="module")
def label():
    return make_barcode_label()


@pytest.fixture
def files(tmp_path, label):
    art = tmp_path / "label.png"
    Image.fromarray(label.image[:, :, ::-1].copy()).save(art, dpi=(304.8, 304.8))
    paths = {}
    gap = label.image.copy()
    x, y, w, h = label.boxes["ean"]
    paint(gap, (x + 60, y, 12, h))
    wrong = make_barcode_label(values={"qr": "https://example.com/p?id=99999"}).image
    for name, printed in (("good", label.image), ("ean_gap", gap), ("wrong_qr", wrong)):
        photo = syn.make_photo(printed, photo_size=(2400, 1800), roll_deg=-8, tilt_x_deg=10, label_fraction=0.75,
                               render=dict(noise_sigma=2.0, jpeg_quality=92))
        paths[name] = write_image(tmp_path / f"{name}.jpg", photo.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return art, paths


@pytest.fixture
def window(qtbot, app_config):
    config = dataclasses.replace(
        app_config,
        text=dataclasses.replace(app_config.text, enabled=False),
        color=dataclasses.replace(app_config.color, enabled=False),
    )
    win = MainWindow(config)
    qtbot.addWidget(win)
    win.show()
    yield win
    if shiboken6.isValid(win._thread_pool):
        win._thread_pool.waitForDone(TIMEOUT_MS)


def inspect(qtbot, window, art, photo):
    with qtbot.waitSignal(window.image_loaded, timeout=TIMEOUT_MS):
        window.load_image_file(ImageRole.ORIGINAL, art)
    with qtbot.waitSignal(window.barcodes_done, timeout=TIMEOUT_MS) as blocker:
        window.load_image_file(ImageRole.PRINTED, photo)
    return blocker.args[0]


def test_correct_barcodes_pass(qtbot, window, files):
    art, paths = files
    report = inspect(qtbot, window, art, paths["good"])
    assert [s.status for s in report.symbols] == [SymbolStatus.OK] * 3
    assert report.findings == ()
    assert window.defect_report().defects == ()  # dense bars give no false image defects
    assert window.status_indicator.status is InspectionStatus.PASS
    panel = window.barcode_panel
    assert panel.symbols_table.rowCount() == 3 and panel.symbols_table.item(0, 5).text() == "ok"
    assert "10/10 strips" in panel.symbols_table.item(0, 6).text()
    assert "ZXing-C++" in panel.summary.text()
    assert window.barcode_section.level_label.text().endswith("GOOD")
    assert window.status_area.progress.value() == window.status_area.progress.maximum() == 5  # + barcodes


def test_damaged_barcode_fails_and_is_highlighted(qtbot, window, files, tmp_path):
    art, paths = files
    report = inspect(qtbot, window, art, paths["ean_gap"])
    ean = report.symbols[0]
    assert ean.status is SymbolStatus.NOT_DECODED
    kinds = {f.kind for f in report.findings}
    assert kinds == {BarcodeFindingKind.NOT_DECODED, BarcodeFindingKind.DAMAGED_BARS}
    assert all(f.reliable for f in report.findings)
    assert window.status_indicator.status is InspectionStatus.FAIL
    assert set(window.decision().barcode_failing) == {f.id for f in report.findings}
    assert len(window.defect_report().defects) <= 3  # the gap itself, not every bar edge

    # Purple B boxes on the viewers; clicking the damaged bars selects that finding.
    window.set_printed_view_mode("registered")
    labels = [item.text() for item in window.printed_panel.view._box_items if hasattr(item, "text")]
    assert {"B1", "B2"} <= set(labels)
    bars = next(f for f in report.findings if f.kind is BarcodeFindingKind.DAMAGED_BARS)
    centre = QPointF(bars.x + bars.width / 2, bars.y + bars.height / 2)
    window._on_view_clicked("overlay", centre)
    # The smallest box under the pointer wins: the damaged bars or the image defect at the gap.
    selected = window.selected_barcode_finding() if window._selected_kind == "barcode" else window.selected_defect()
    assert selected is not None
    assert not (window._selected_kind == "barcode" and selected.kind is BarcodeFindingKind.NOT_DECODED)
    assert QRectF(selected.x, selected.y, selected.width, selected.height).adjusted(-6, -6, 6, 6).contains(centre)
    window.select_barcode_finding(bars.id)
    assert window.side_tabs.currentWidget() is window.barcode_panel
    assert not window.barcode_panel.print_crop.pixmap().isNull()

    window.mark_actions[OperatorMark.NEEDS_REVIEW].trigger()
    assert window.barcode_marks() == {bars.id: OperatorMark.NEEDS_REVIEW}
    row = next(r for r in range(window.barcode_panel.findings_table.rowCount())
               if window.barcode_panel.findings_table.item(r, 0).text() == bars.label)
    assert window.barcode_panel.findings_table.item(row, COL_BARCODE_OPERATOR).text() == "Needs review"

    out = tmp_path / "results"
    out.mkdir()
    with qtbot.waitSignal(window.results_saved, timeout=TIMEOUT_MS) as saved:
        window.save_results_to(out)
    folder = saved.args[0]
    summary = json.loads((folder / "inspection.json").read_text(encoding="utf-8"))
    assert summary["barcodes"]["library"].startswith("ZXing-C++") and "not an ISO" in summary["barcodes"]["note"]
    assert summary["barcodes"]["symbols"][0]["status"] == "not decoded"
    with open(folder / "barcodes.csv", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["status"] for r in rows] == ["not decoded", "ok", "ok"] and rows[0]["strips_decoded"] == "0/10"
    with open(folder / "barcode_findings.csv", encoding="utf-8-sig", newline="") as handle:
        findings = list(csv.DictReader(handle))
    assert {f["problem"] for f in findings} == {"Not decodable", "Damaged bars"}


def test_wrong_content_fails(qtbot, window, files):
    art, paths = files
    report = inspect(qtbot, window, art, paths["wrong_qr"])
    qr = report.symbols[2]
    assert qr.status is SymbolStatus.WRONG_CONTENT and qr.print_value.endswith("99999")
    assert any(f.kind is BarcodeFindingKind.WRONG_CONTENT and f.reliable for f in report.findings)
    assert window.status_indicator.status is InspectionStatus.FAIL


def test_barcode_zones_are_drawn_saved_and_used(qtbot, window, files, label):
    art, paths = files
    inspect(qtbot, window, art, paths["good"])
    dialog = window.create_barcode_zone_dialog()
    qtbot.addWidget(dialog)
    x, y, w, h = label.boxes["ean"]
    dialog.add_zone_rect(QRectF(x - 30, y - 30, w + 60, h + 60))
    dialog.table.item(0, 0).setText("GTIN")
    dialog.table.cellWidget(0, 1).setCurrentIndex(dialog.table.cellWidget(0, 1).findData("ean13"))
    dialog.table.item(0, 2).setText("4006381333932")  # wrong check digit
    dialog.accept()
    assert "check digit" in dialog.error_label.text()
    dialog.table.item(0, 2).setText("5901234123457")  # valid, but not what is printed
    zones = dialog.zones()
    assert dialog.validate() is None and zones[0].format == "ean13"
    with qtbot.waitSignal(window.barcodes_done, timeout=TIMEOUT_MS) as blocker:
        window.apply_barcode_zones(zones)
    assert load_zones(zones_path(art)) == zones
    report = blocker.args[0]
    gtin = next(s for s in report.symbols if s.name == "GTIN")
    assert gtin.status is SymbolStatus.WRONG_CONTENT and gtin.expected_source == "zone" and gtin.critical
    assert window.status_indicator.status is InspectionStatus.FAIL
    window.reset()
    with qtbot.waitSignal(window.image_loaded, timeout=TIMEOUT_MS):
        window.load_image_file(ImageRole.ORIGINAL, art)
    assert window.barcode_zones() == zones


def test_barcode_settings_rerun_only_the_barcodes(qtbot, window, files):
    art, paths = files
    inspect(qtbot, window, art, paths["good"])
    comparison = window.comparison_result()
    config = window.config()
    with qtbot.waitSignal(window.barcodes_done, timeout=TIMEOUT_MS):
        window.apply_settings(dataclasses.replace(config, barcodes=dataclasses.replace(config.barcodes, min_scan_success=0.5)))
    assert window.comparison_result() is comparison
    window.apply_settings(dataclasses.replace(config, barcodes=dataclasses.replace(config.barcodes, enabled=False)))
    assert window.barcode_report() is None
    assert window.status_indicator.status is InspectionStatus.PASS
