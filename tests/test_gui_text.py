"""Text inspection in the application (full pipeline, scripted OCR engine)."""
import csv
import dataclasses
import json

import cv2
import pytest
import shiboken6
from PIL import Image
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QMessageBox

from print_inspector import synthetic as syn
from print_inspector.core.decision import OperatorMark
from print_inspector.core.image_io import ImageRole
from print_inspector.core.models import InspectionStatus, QualityLevel
from print_inspector.core.text_inspection import TextFindingKind
from print_inspector.core.text_zones import load_zones, zones_path
from print_inspector.gui.main_window import MainWindow
from print_inspector.gui.text_panel import COL_TEXT_OPERATOR
from tests.conftest import write_image
from tests.ocr_fakes import ScriptedOcr, erase, make_text_label

TIMEOUT_MS = 60000
LOT, NAME = 0, 3


@pytest.fixture(scope="module")
def label():
    return make_text_label()


@pytest.fixture
def files(tmp_path, label):
    art = tmp_path / "label.png"
    Image.fromarray(label.image[:, :, ::-1].copy()).save(art, dpi=(304.8, 304.8))  # 12 px per mm
    paths = {}
    defective = label.image.copy()
    erase(defective, label.char_box(LOT, 8))  # the 5 of LOT 12345
    for name, printed in (("good", label.image), ("missing5", defective)):
        photo = syn.make_photo(printed, photo_size=(2400, 1800), roll_deg=-8, tilt_x_deg=10, label_fraction=0.75,
                               render=dict(noise_sigma=2.0, jpeg_quality=92))
        paths[name] = write_image(tmp_path / f"{name}.jpg", photo.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return art, paths


def make_window(qtbot, app_config, engine, **text_changes):
    config = dataclasses.replace(
        app_config,
        text=dataclasses.replace(app_config.text, **text_changes),
        barcodes=dataclasses.replace(app_config.barcodes, enabled=False),
        color=dataclasses.replace(app_config.color, enabled=False),
    )
    win = MainWindow(config, ocr_engine=engine)
    qtbot.addWidget(win)
    win.show()
    return win


@pytest.fixture
def cleanup():
    windows = []
    yield windows.append
    for win in windows:
        if shiboken6.isValid(win._thread_pool):
            win._thread_pool.waitForDone(TIMEOUT_MS)


def inspect(qtbot, window, art, photo):
    with qtbot.waitSignal(window.image_loaded, timeout=TIMEOUT_MS):
        window.load_image_file(ImageRole.ORIGINAL, art)
    with qtbot.waitSignal(window.text_done, timeout=TIMEOUT_MS) as blocker:
        window.load_image_file(ImageRole.PRINTED, photo)
    return blocker.args[0]


def test_missing_digit_fails_with_text_finding(qtbot, app_config, label, files, cleanup, tmp_path):
    art, paths = files
    window = make_window(qtbot, app_config, ScriptedOcr(label, reads={LOT: ("LOT 1234", 90)}))
    cleanup(window)
    report = inspect(qtbot, window, art, paths["missing5"])
    assert window.text_report() is report and report.available
    (finding,) = report.findings
    assert finding.kind is TextFindingKind.MISSING_CHARACTER and finding.expected == "5" and finding.reliable
    decision = window.decision()
    assert decision.status is InspectionStatus.FAIL and decision.text_failing == (finding.id,)
    assert "T1" in window.status_reason_label.text()

    # Text tab: findings and lines; Checks tab: the text section; progress counts the text stage.
    panel = window.text_panel
    assert panel.findings_table.rowCount() == 1 and panel.findings_table.item(0, 0).text() == "T1"
    assert panel.lines_table.rowCount() == 4
    assert panel.lines_table.item(0, 1).text() == "mismatch" and panel.lines_table.item(0, 3).text() == "LOT 1234"
    assert "Scripted OCR" in panel.summary.text()
    assert window.text_section.level_label.text().endswith("GOOD")
    area = window.status_area
    assert area.progress.value() == area.progress.maximum() == 5
    assert "text" in area.progress_label.full_text()

    # Orange "T1" box on the viewers; clicking it selects the finding.
    window.set_printed_view_mode("registered")
    labels = [item for item in window.printed_panel.view._box_items if hasattr(item, "text")]
    assert "T1" in [item.text() for item in labels]
    centre = QPointF(finding.x + finding.width / 2, finding.y + finding.height / 2)
    window._on_view_clicked("printed", centre)
    assert window.selected_text_finding() is finding
    assert window.side_tabs.currentWidget() is panel
    assert not panel.artwork_crop.pixmap().isNull() and "Missing character" in panel.title.full_text()

    # The operator overrules with the keyboard command for the selected finding.
    window.mark_actions[OperatorMark.DISMISSED].trigger()
    assert window.text_marks() == {finding.id: OperatorMark.DISMISSED}
    assert panel.findings_table.item(0, COL_TEXT_OPERATOR).text() == "Dismissed"
    assert decision.failing  # the image defect at the same place still fails the label
    window.set_mark(window.defect_report().defects[0].id, OperatorMark.DISMISSED)
    assert window.status_indicator.status is InspectionStatus.PASS

    # Saved results contain the text inspection.
    out = tmp_path / "results"
    out.mkdir()
    with qtbot.waitSignal(window.results_saved, timeout=TIMEOUT_MS) as saved:
        window.save_results_to(out)
    folder = saved.args[0]
    summary = json.loads((folder / "inspection.json").read_text(encoding="utf-8"))
    assert summary["text"]["engine"] == "Scripted OCR"
    assert summary["text"]["findings"][0]["operator"] == "dismissed"
    assert summary["text"]["lines"][0]["print_text"] == "LOT 1234"
    with open(folder / "text.csv", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["id"] == "T1" and rows[0]["expected"] == "5" and rows[0]["detection"] == "reliable"


def test_uncertain_ocr_gives_review_until_the_operator_decides(qtbot, app_config, label, files, cleanup):
    art, paths = files
    window = make_window(qtbot, app_config, ScriptedOcr(label, reads={NAME: ("PRINT INSPEKTOR", 85)}))
    cleanup(window)
    report = inspect(qtbot, window, art, paths["good"])
    (finding,) = report.findings
    assert not finding.reliable
    assert window.status_indicator.status is InspectionStatus.REVIEW
    assert window.decision().text_review == (finding.id,)
    window.select_text_finding(finding.id)
    window.text_panel.buttons[OperatorMark.DISMISSED].click()
    assert window.status_indicator.status is InspectionStatus.PASS
    window.text_panel.buttons[OperatorMark.CONFIRMED].click()
    assert window.status_indicator.status is InspectionStatus.FAIL


def test_no_verdict_until_the_text_is_read(qtbot, app_config, label, files, cleanup):
    art, paths = files
    window = make_window(qtbot, app_config, ScriptedOcr(label, delay_s=0.3))
    cleanup(window)
    with qtbot.waitSignal(window.image_loaded, timeout=TIMEOUT_MS):
        window.load_image_file(ImageRole.ORIGINAL, art)
    with qtbot.waitSignal(window.comparison_done, timeout=TIMEOUT_MS):
        window.load_image_file(ImageRole.PRINTED, paths["good"])
    assert "reading text (OCR)" in window.running_stages()
    assert window.status_indicator.status is InspectionStatus.NOT_INSPECTED
    assert "Reading the text" in window.status_reason_label.text()
    with qtbot.waitSignal(window.text_done, timeout=TIMEOUT_MS):
        pass
    assert window.status_indicator.status is InspectionStatus.PASS


def test_missing_ocr_engine_gives_review(qtbot, app_config, files, cleanup, tmp_path):
    art, paths = files
    window = make_window(qtbot, app_config, None, tesseract_path=str(tmp_path / "no-tesseract"))
    cleanup(window)
    report = inspect(qtbot, window, art, paths["good"])
    assert not report.available and report.level is QualityLevel.POOR
    assert window.status_indicator.status is InspectionStatus.REVIEW
    assert "not available" in window.status_reason_label.text()
    assert "not available" in window.text_panel.summary.text()


def test_missing_ocr_engine_when_not_required_only_warns(qtbot, app_config, files, cleanup, tmp_path):
    art, paths = files
    window = make_window(qtbot, app_config, None, tesseract_path=str(tmp_path / "no-tesseract"), required=False)
    cleanup(window)
    report = inspect(qtbot, window, art, paths["good"])
    assert report.level is QualityLevel.MARGINAL
    assert window.status_indicator.status is InspectionStatus.PASS


def test_text_zones_are_drawn_saved_and_used(qtbot, app_config, label, files, cleanup):
    art, paths = files
    engine = ScriptedOcr(label)
    window = make_window(qtbot, app_config, engine)
    cleanup(window)
    inspect(qtbot, window, art, paths["good"])
    dialog = window.create_text_zone_dialog()
    qtbot.addWidget(dialog)
    x, y, w, h = label.line_box(NAME)
    dialog.add_zone_rect(QRectF(x - 10, y - 10, w + 20, h + 20))
    dialog.add_zone_rect(QRectF(5, 5, 2, 2))  # too small: ignored
    dialog.table.item(0, 0).setText("Product name")
    dialog.table.item(0, 1).setText("PRINT INSPECTOR")
    zones = dialog.zones()
    assert len(zones) == 1 and zones[0].name == "Product name" and zones[0].critical
    with qtbot.waitSignal(window.text_done, timeout=TIMEOUT_MS) as blocker:
        window.apply_text_zones(zones)
    assert load_zones(zones_path(art)) == zones  # saved next to the artwork
    report = blocker.args[0]
    zone_lines = [line for line in report.lines if line.zone == "Product name"]
    assert len(zone_lines) == 1 and zone_lines[0].expected_source == "zone" and zone_lines[0].critical
    assert any(c.mode == "zone" for c in engine.calls)

    # Loading the artwork again restores the zones.
    window.reset()
    assert window.text_zones() == ()
    with qtbot.waitSignal(window.image_loaded, timeout=TIMEOUT_MS):
        window.load_image_file(ImageRole.ORIGINAL, art)
    assert window.text_zones() == zones


def test_duplicate_zone_names_are_refused(qtbot, app_config, label):
    from print_inspector.gui.text_zone_dialog import TextZoneDialog

    dialog = TextZoneDialog(label.image, (), app_config.ui)
    qtbot.addWidget(dialog)
    dialog.add_zone_rect(QRectF(10, 10, 100, 40))
    dialog.add_zone_rect(QRectF(10, 100, 100, 40))
    dialog.table.item(1, 0).setText("Zone 1")
    dialog.accept()
    assert dialog.result() != dialog.DialogCode.Accepted and "own name" in dialog.error_label.text()
    dialog.table.selectRow(1)
    dialog.delete_selected()
    assert len(dialog.zones()) == 1


def test_invalid_zone_file_is_reported(qtbot, app_config, label, files, cleanup):
    art, _ = files
    zones_path(art).write_text("{broken", encoding="utf-8")
    window = make_window(qtbot, app_config, ScriptedOcr(label))
    cleanup(window)
    with qtbot.waitSignal(window.image_loaded, timeout=TIMEOUT_MS):
        window.load_image_file(ImageRole.ORIGINAL, art)
    assert window.text_zones() == ()
    assert window.banner.isVisible() and "cannot be read" in window.banner.text()


def test_text_settings_rerun_only_the_text(qtbot, app_config, label, files, cleanup, monkeypatch):
    art, paths = files
    window = make_window(qtbot, app_config, ScriptedOcr(label, reads={NAME: ("PRINT INSPEKTOR", 85)}))
    cleanup(window)
    inspect(qtbot, window, art, paths["good"])
    comparison = window.comparison_result()
    config = window.config()
    with qtbot.waitSignal(window.text_done, timeout=TIMEOUT_MS):
        window.apply_settings(dataclasses.replace(config, text=dataclasses.replace(config.text, min_word_confidence=50.0)))
    assert window.comparison_result() is comparison
    window.apply_settings(dataclasses.replace(config, text=dataclasses.replace(config.text, enabled=False)))
    assert window.text_report() is None
    assert window.status_indicator.status is InspectionStatus.PASS  # no text inspection, no text findings


def test_zone_save_failure_is_reported(qtbot, app_config, label, files, cleanup, monkeypatch):
    art, _ = files
    window = make_window(qtbot, app_config, ScriptedOcr(label))
    cleanup(window)
    with qtbot.waitSignal(window.image_loaded, timeout=TIMEOUT_MS):
        window.load_image_file(ImageRole.ORIGINAL, art)
    shown = []
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: shown.append(a[2])))
    monkeypatch.setattr("print_inspector.gui.main_window.save_zones", lambda *a: (_ for _ in ()).throw(OSError("read-only")))
    from print_inspector.core.text_zones import TextZone

    window.apply_text_zones((TextZone("Lot", 0.1, 0.1, 0.3, 0.1),))
    assert shown and "read-only" in shown[0]
    assert len(window.text_zones()) == 1  # still used for this session
