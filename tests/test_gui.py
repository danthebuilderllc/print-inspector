import dataclasses
import json
import io

import cv2
import numpy as np
import pytest
import shiboken6
from PIL import Image
from PySide6.QtWidgets import QMessageBox

from print_inspector.core.capture_quality import QualityLevel
from print_inspector.core.image_io import ImageRole
from print_inspector.core.models import InspectionStatus
from print_inspector import synthetic as syn
from print_inspector.core.alignment import AlignmentMethod
from print_inspector.gui.main_window import COL_OPERATOR, DEFECT_COLUMNS, VIEW_ALIGNED, VIEW_PHOTO, VIEW_REGISTERED, MainWindow
from tests.conftest import make_label_bgr, make_label_photo, write_image

TIMEOUT_MS = 20000


@pytest.fixture
def window(qtbot, app_config):
    # Text, barcode and colour inspection have their own tests (test_gui_text/_barcodes/_color.py).
    config = dataclasses.replace(
        app_config,
        text=dataclasses.replace(app_config.text, enabled=False),
        barcodes=dataclasses.replace(app_config.barcodes, enabled=False),
        color=dataclasses.replace(app_config.color, enabled=False),
    )
    win = MainWindow(config)
    qtbot.addWidget(win)
    win.show()
    yield win
    # Let background workers finish. pytest-qt may already have closed the window,
    # whose closeEvent waits for the workers itself.
    if shiboken6.isValid(win._thread_pool):
        win._thread_pool.waitForDone(TIMEOUT_MS)


@pytest.fixture
def error_boxes(monkeypatch):
    """Capture error dialogs instead of blocking on them."""
    shown = []
    monkeypatch.setattr(
        QMessageBox, "critical", staticmethod(lambda parent, title, text, *a, **k: shown.append(text))
    )
    return shown


def _load(qtbot, window, role, path):
    with qtbot.waitSignal(window.image_loaded, timeout=TIMEOUT_MS) as blocker:
        window.load_image_file(role, path)
    assert blocker.args == [role]


def _load_printed_and_assess(qtbot, window, path):
    with qtbot.waitSignal(window.quality_assessed, timeout=TIMEOUT_MS) as blocker:
        window.load_image_file(ImageRole.PRINTED, path)
    return blocker.args[0]


def test_initial_state(window):
    assert window.status_indicator.status is InspectionStatus.NOT_INSPECTED
    assert window.status_indicator.text() == "NOT INSPECTED"
    assert not window.compare_action.isEnabled()
    assert not window.save_action.isEnabled()
    assert not window.reset_action.isEnabled()
    assert not window.zoom_in_action.isEnabled()
    assert window.defect_table.rowCount() == 0
    assert window.quality_section.issue_list.count() == 0
    for panel in (window.original_panel, window.printed_panel, window.overlay_panel):
        assert not panel.view.has_image()
    assert window.sensitivity_combo.currentData() == "medium"


def test_load_original_displays_full_resolution_and_file_info(qtbot, window, label_png):
    _load(qtbot, window, ImageRole.ORIGINAL, label_png)
    loaded = window.loaded_image(ImageRole.ORIGINAL)
    assert loaded is not None and loaded.path == label_png
    size = window.original_panel.view.image_size()
    assert (size.width(), size.height()) == (600, 400)  # no downscaling for display
    info = window.original_panel.info_label.text()
    assert "artwork.png" in info and "600 × 400 px" in info and "PNG" in info and "8-bit" in info
    assert window.reset_action.isEnabled()
    assert not window.compare_action.isEnabled()
    # Artwork alone triggers no capture-quality check.
    assert window.quality_report() is None


def test_artwork_dpi_shown(qtbot, window, tmp_path):
    out = io.BytesIO()
    Image.fromarray(np.full((600, 1181, 3), 255, np.uint8)).save(out, "PNG", dpi=(300, 300))
    path = tmp_path / "art300.png"
    path.write_bytes(out.getvalue())
    _load(qtbot, window, ImageRole.ORIGINAL, path)
    assert "300 dpi" in window.original_panel.info_label.text()
    assert "100.0 × 50.8 mm" in window.original_panel.info_label.text()


def test_good_photo_quality_shown_without_verdict(qtbot, window, photo_jpg):
    report = _load_printed_and_assess(qtbot, window, photo_jpg)
    assert window.quality_report() is report
    # 4.3 MP generated photo: below the 12 MP recommendation, otherwise fine.
    assert report.level is QualityLevel.MARGINAL
    assert "low_megapixels" in report.issue_codes()
    assert window.status_indicator.status is InspectionStatus.NOT_INSPECTED
    assert "MARGINAL" in window.quality_section.level_label.text()
    assert window.quality_section.issue_list.count() == len(report.issues)
    assert "Edge blur" in window.quality_section.metrics_label.text()
    assert window.printed_panel.warning_label.isVisible()
    window.set_printed_view_mode(VIEW_PHOTO)
    assert "JPEG" in window.printed_panel.info_label.text()
    region = window.printed_panel.view.region()
    assert region is not None and region.width() > 1000


def test_poor_photo_gives_review(qtbot, window, tmp_path):
    blurred = cv2.GaussianBlur(make_label_photo(), (0, 0), 6.0)
    path = write_image(tmp_path / "blurred.jpg", blurred, [cv2.IMWRITE_JPEG_QUALITY, 92])
    report = _load_printed_and_assess(qtbot, window, path)
    assert report.level is QualityLevel.POOR
    assert window.status_indicator.status is InspectionStatus.REVIEW
    assert "too poor" in window.status_reason_label.text()
    assert "POOR" in window.quality_section.level_label.text()
    assert any("CRITICAL" in window.quality_section.issue_list.item(i).text() for i in range(window.quality_section.issue_list.count()))


def test_blank_photo_gives_review(qtbot, window, tmp_path):
    path = write_image(tmp_path / "blank.jpg", np.full((1500, 2000, 3), 210, np.uint8))
    report = _load_printed_and_assess(qtbot, window, path)
    assert "no_content" in report.issue_codes()
    assert window.status_indicator.status is InspectionStatus.REVIEW


def test_artwork_dpi_triggers_reassessment_with_real_label_size(qtbot, window, photo_jpg, tmp_path):
    report = _load_printed_and_assess(qtbot, window, photo_jpg)
    assert report.metrics.label_size_assumed
    out = io.BytesIO()
    Image.fromarray(np.full((945, 1260, 3), 255, np.uint8)).save(out, "PNG", dpi=(400, 400))
    art = tmp_path / "art400.png"
    art.write_bytes(out.getvalue())
    with qtbot.waitSignal(window.quality_assessed, timeout=TIMEOUT_MS) as blocker:
        window.load_image_file(ImageRole.ORIGINAL, art)
    updated = blocker.args[0]
    assert not updated.metrics.label_size_assumed
    assert updated.metrics.label_long_side_mm == pytest.approx(80.0, abs=0.1)
    assert "artwork size" in window.quality_section.metrics_label.text()


def test_load_failure_reports_error_and_keeps_panel_empty(qtbot, window, tmp_path, error_boxes):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"garbage" * 100)
    with qtbot.waitSignal(window.image_load_failed, timeout=TIMEOUT_MS) as blocker:
        window.load_image_file(ImageRole.ORIGINAL, bad)
    assert blocker.args[0] is ImageRole.ORIGINAL
    assert error_boxes and "not a supported image" in error_boxes[0]
    assert not window.original_panel.view.has_image()
    assert window.loaded_image(ImageRole.ORIGINAL) is None


def test_truncated_photo_rejected(qtbot, window, tmp_path, error_boxes):
    ok, encoded = cv2.imencode(".jpg", make_label_photo(800, 600))
    path = tmp_path / "cut.jpg"
    path.write_bytes(encoded.tobytes()[: encoded.size * 3 // 4])
    with qtbot.waitSignal(window.image_load_failed, timeout=TIMEOUT_MS):
        window.load_image_file(ImageRole.PRINTED, path)
    assert "incomplete" in error_boxes[0]
    assert window.quality_report() is None


def test_failed_reload_keeps_previous_image(qtbot, window, label_png, tmp_path, error_boxes):
    _load(qtbot, window, ImageRole.ORIGINAL, label_png)
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"garbage" * 100)
    with qtbot.waitSignal(window.image_load_failed, timeout=TIMEOUT_MS):
        window.load_image_file(ImageRole.ORIGINAL, bad)
    assert window.loaded_image(ImageRole.ORIGINAL).path == label_png
    assert window.original_panel.view.has_image()
    assert "artwork.png" in window.original_panel.info_label.text()


def test_printed_rejects_bmp(qtbot, window, tmp_path, error_boxes):
    path = write_image(tmp_path / "photo.bmp", make_label_bgr())
    with qtbot.waitSignal(window.image_load_failed, timeout=TIMEOUT_MS):
        window.load_image_file(ImageRole.PRINTED, path)
    assert "Unsupported file type" in error_boxes[0]


def test_reset_clears_everything(qtbot, window, label_png, tmp_path):
    _load(qtbot, window, ImageRole.ORIGINAL, label_png)
    blurred = cv2.GaussianBlur(make_label_photo(), (0, 0), 6.0)
    path = write_image(tmp_path / "blurred.jpg", blurred)
    _load_printed_and_assess(qtbot, window, path)
    assert window.status_indicator.status is InspectionStatus.REVIEW
    window.reset()
    assert window.loaded_image(ImageRole.ORIGINAL) is None
    assert window.loaded_image(ImageRole.PRINTED) is None
    assert window.quality_report() is None
    assert window.quality_section.issue_list.count() == 0
    assert window.printed_panel.view.region() is None
    assert not window.printed_panel.warning_label.isVisible()
    assert window.status_indicator.status is InspectionStatus.NOT_INSPECTED
    assert not window.reset_action.isEnabled()


def test_reset_discards_in_flight_work(qtbot, window, photo_jpg):
    window.load_image_file(ImageRole.PRINTED, photo_jpg)
    window.reset()
    qtbot.waitUntil(lambda: window._thread_pool.activeThreadCount() == 0, timeout=TIMEOUT_MS)
    qtbot.wait(100)  # deliver any queued signal from the worker
    assert window.loaded_image(ImageRole.PRINTED) is None
    assert window.quality_report() is None
    assert not window.printed_panel.view.has_image()


def test_newer_load_supersedes_older(qtbot, window, tmp_path):
    first = write_image(tmp_path / "first.png", make_label_bgr(600, 400))
    second = write_image(tmp_path / "second.png", make_label_bgr(300, 200))
    window.load_image_file(ImageRole.ORIGINAL, first)
    window.load_image_file(ImageRole.ORIGINAL, second)
    qtbot.waitUntil(lambda: not window.is_loading(ImageRole.ORIGINAL), timeout=TIMEOUT_MS)
    qtbot.waitUntil(lambda: window._thread_pool.activeThreadCount() == 0, timeout=TIMEOUT_MS)
    qtbot.wait(100)
    assert window.loaded_image(ImageRole.ORIGINAL).path == second


def test_zoom_and_fit(qtbot, window, tmp_path):
    big = write_image(tmp_path / "big.png", cv2.resize(make_label_bgr(), (3000, 2000)))
    _load(qtbot, window, ImageRole.ORIGINAL, big)
    view = window.original_panel.view
    assert view.is_fit_mode
    fit_zoom = view.zoom
    assert 0 < fit_zoom < 1

    window.zoom_in()
    assert view.zoom == pytest.approx(fit_zoom * 1.25)
    assert not view.is_fit_mode
    window.zoom_out()
    window.zoom_out()
    assert view.zoom == pytest.approx(fit_zoom / 1.25)

    window.fit_to_window()
    assert view.is_fit_mode
    assert view.zoom == pytest.approx(fit_zoom)
    assert window.original_panel.zoom_label.text().endswith("%")


def test_zoom_is_clamped(qtbot, window, label_png, app_config):
    _load(qtbot, window, ImageRole.ORIGINAL, label_png)
    view = window.original_panel.view
    view.set_zoom(1e6)
    assert view.zoom == app_config.ui.max_zoom
    view.set_zoom(1e-9)
    assert view.zoom == app_config.ui.min_zoom


def test_display_shares_pixels_with_working_image(qtbot, window, label_png):
    """The display path wraps the working array instead of copying it."""
    from print_inspector.gui.workers import bgr8_to_qimage

    _load(qtbot, window, ImageRole.ORIGINAL, label_png)
    working = window.loaded_image(ImageRole.ORIGINAL).working
    qimage = bgr8_to_qimage(working)
    assert (qimage.width(), qimage.height()) == (600, 400)
    rgb = qimage.pixelColor(30, 250)  # inside the red box
    b, g, r = working[250, 30]
    assert (rgb.red(), rgb.green(), rgb.blue()) == (r, g, b)


@pytest.mark.parametrize(
    "status, text",
    [
        (InspectionStatus.PASS, "PASS"),
        (InspectionStatus.FAIL, "FAIL"),
        (InspectionStatus.REVIEW, "REVIEW"),
        (InspectionStatus.NOT_INSPECTED, "NOT INSPECTED"),
    ],
)
def test_status_indicator_labels(window, status, text):
    window.status_indicator.set_status(status)
    assert window.status_indicator.text() == text
    assert window.status_indicator.status is status


# ---------------------------------------------------------------- label detection and alignment


@pytest.fixture
def artwork_and_photo(tmp_path):
    artwork = syn.make_artwork()
    photo = syn.make_photo(artwork, photo_size=(2000, 1500), roll_deg=-18, tilt_x_deg=20)
    art_path = write_image(tmp_path / "art.png", artwork)
    photo_path = write_image(tmp_path / "label.jpg", photo.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return art_path, photo_path, photo


def _load_both_and_align(qtbot, window, art_path, photo_path):
    _load(qtbot, window, ImageRole.ORIGINAL, art_path)
    with qtbot.waitSignal(window.alignment_done, timeout=TIMEOUT_MS) as blocker:
        window.load_image_file(ImageRole.PRINTED, photo_path)
    return blocker.args[0]


def test_photo_is_aligned_automatically(qtbot, window, artwork_and_photo):
    art_path, photo_path, photo = artwork_and_photo
    result = _load_both_and_align(qtbot, window, art_path, photo_path)
    assert window.alignment_result() is result
    assert result.metrics.method is AlignmentMethod.BOUNDARY
    assert result.is_reliable
    # Until registration finishes, the perspective-corrected label is shown at full output size.
    window.set_printed_view_mode(VIEW_ALIGNED)
    size = window.printed_panel.view.image_size()
    assert (size.width(), size.height()) == result.metrics.output_size
    assert "Perspective-corrected label" in window.printed_panel.info_label.text()
    assert "Alignment:" in window.alignment_section.level_label.text()
    assert "Method" in window.alignment_section.metrics_label.text()
    rms, worst = syn.rectification_error(result.homography, result.metrics.output_size, photo)
    assert worst < 1.5


def test_switch_to_photo_shows_detected_corners(qtbot, window, artwork_and_photo):
    art_path, photo_path, photo = artwork_and_photo
    result = _load_both_and_align(qtbot, window, art_path, photo_path)
    window.set_printed_view_mode(VIEW_PHOTO)
    size = window.printed_panel.view.image_size()
    assert (size.width(), size.height()) == (2000, 1500)
    polygon = np.array(window.printed_panel.view.polygon())
    assert np.allclose(polygon, result.metrics.corners)
    window.set_printed_view_mode(VIEW_ALIGNED)
    assert window.printed_panel.view.polygon() is None


def test_manual_corners_replace_automatic(qtbot, window, artwork_and_photo):
    art_path, photo_path, photo = artwork_and_photo
    _load_both_and_align(qtbot, window, art_path, photo_path)
    dialog = window.create_corner_dialog()
    assert dialog is not None and dialog.corners() is not None  # pre-filled from detection
    dialog.set_corners(photo.corners)
    with qtbot.waitSignal(window.alignment_done, timeout=TIMEOUT_MS) as blocker:
        window.apply_manual_corners(dialog.corners())
    result = blocker.args[0]
    assert result.metrics.method is AlignmentMethod.MANUAL
    assert result.is_reliable
    # Detect Label goes back to automatic detection.
    with qtbot.waitSignal(window.alignment_done, timeout=TIMEOUT_MS) as blocker:
        window.detect_label()
    assert blocker.args[0].metrics.method is AlignmentMethod.BOUNDARY


def test_bad_manual_corners_give_review(qtbot, window, artwork_and_photo):
    art_path, photo_path, photo = artwork_and_photo
    _load_both_and_align(qtbot, window, art_path, photo_path)
    centre = photo.corners.mean(axis=0)
    with qtbot.waitSignal(window.alignment_done, timeout=TIMEOUT_MS) as blocker:
        window.apply_manual_corners(centre + (photo.corners - centre) * 0.7)
    assert not blocker.args[0].is_reliable
    assert window.status_indicator.status is InspectionStatus.REVIEW
    assert "aligned" in window.status_reason_label.text()


def test_unlocatable_label_gives_review_and_offers_manual(qtbot, window, tmp_path):
    path = write_image(tmp_path / "plain.jpg", np.full((1500, 2000, 3), 140, np.uint8))
    with qtbot.waitSignal(window.alignment_done, timeout=TIMEOUT_MS) as blocker:
        window.load_image_file(ImageRole.PRINTED, path)
    result = blocker.args[0]
    assert "label_not_found" in result.issue_codes()
    assert window.status_indicator.status is InspectionStatus.REVIEW
    assert window.manual_corners_action.isEnabled()
    assert window.printed_view_mode() == VIEW_REGISTERED
    # Nothing to align: the photo is shown instead, with a hint.
    assert "Mark Label Corners" in window.printed_panel.info_label.text()
    assert window.create_corner_dialog().corners() is None


def test_detection_actions_need_a_photo(window):
    assert not window.detect_action.isEnabled()
    assert not window.manual_corners_action.isEnabled()
    assert window.create_corner_dialog() is None


def test_reset_clears_alignment(qtbot, window, artwork_and_photo):
    art_path, photo_path, _ = artwork_and_photo
    _load_both_and_align(qtbot, window, art_path, photo_path)
    window.reset()
    assert window.alignment_result() is None
    assert window.alignment_section.issue_list.count() == 0
    assert not window.printed_panel.view.has_image()
    assert not window.detect_action.isEnabled()


# ---------------------------------------------------------------- registration and overlay


def _load_both_and_register(qtbot, window, art_path, photo_path):
    _load(qtbot, window, ImageRole.ORIGINAL, art_path)
    with qtbot.waitSignal(window.registration_done, timeout=TIMEOUT_MS) as blocker:
        window.load_image_file(ImageRole.PRINTED, photo_path)
    return blocker.args[0]


def test_photo_is_registered_after_alignment(qtbot, window, artwork_and_photo):
    art_path, photo_path, _ = artwork_and_photo
    result = _load_both_and_register(qtbot, window, art_path, photo_path)
    assert window.registration_result() is result
    assert result.is_reliable, [i.message for i in result.issues]
    assert window.printed_view_mode() == VIEW_REGISTERED
    size = window.printed_panel.view.image_size()
    assert (size.width(), size.height()) == result.metrics.grid_size
    assert "Registered to artwork" in window.printed_panel.info_label.text()
    assert "Residual" in window.registration_section.metrics_label.text()
    assert window.adjust_registration_action.isEnabled()
    # The synthetic label is pure white (blown out), so capture quality alone may ask for
    # REVIEW; neither alignment nor registration may be a reason.
    reasons = window.status_reason_label.text()
    assert "aligned" not in reasons and "registered" not in reasons


def test_overlay_panel_modes_and_opacity(qtbot, window, artwork_and_photo):
    art_path, photo_path, _ = artwork_and_photo
    result = _load_both_and_register(qtbot, window, art_path, photo_path)
    assert window.overlay_panel.view.has_image()
    gw, gh = result.metrics.grid_size
    window.set_overlay("blend", 0)
    assert np.array_equal(window.overlay_image()[result.mask > 0], result.reference[result.mask > 0])
    window.overlay_panel.view.set_zoom(4.0)
    window.set_overlay("blend", 100)
    assert np.array_equal(window.overlay_image()[result.mask > 0], result.registered[result.mask > 0])
    assert window.overlay_panel.view.zoom == pytest.approx(4.0)  # adjusting keeps the inspection zoom
    window.set_overlay("checkerboard", 50)
    assert window.overlay_image().shape == (gh, gw, 3)


def test_manual_registration_adjustment(qtbot, window, artwork_and_photo):
    art_path, photo_path, _ = artwork_and_photo
    good = _load_both_and_register(qtbot, window, art_path, photo_path)
    dialog = window.create_registration_dialog()
    assert dialog is not None
    dialog.dx.setValue(1.6)
    dialog.refine_box.setChecked(False)
    with qtbot.waitSignal(window.registration_done, timeout=TIMEOUT_MS) as blocker:
        window.apply_registration_adjustment(
            good.homography, dialog.adjustment(), good.metrics.grid_size, dialog.refine()
        )
    shifted = blocker.args[0]
    assert shifted.metrics.residual_p95_px == pytest.approx(1.6, abs=0.2)
    assert not shifted.is_reliable
    assert window.status_indicator.status is InspectionStatus.REVIEW
    assert "registered" in window.status_reason_label.text()


def test_reset_clears_registration(qtbot, window, artwork_and_photo):
    art_path, photo_path, _ = artwork_and_photo
    _load_both_and_register(qtbot, window, art_path, photo_path)
    window.reset()
    assert window.registration_result() is None
    assert not window.overlay_panel.view.has_image()
    assert window.registration_section.issue_list.count() == 0
    assert not window.adjust_registration_action.isEnabled()


def test_no_registration_without_artwork(qtbot, window, artwork_and_photo):
    _, photo_path, _ = artwork_and_photo
    with qtbot.waitSignal(window.alignment_done, timeout=TIMEOUT_MS):
        window.load_image_file(ImageRole.PRINTED, photo_path)
    window._thread_pool.waitForDone(TIMEOUT_MS)
    qtbot.wait(50)
    assert window.registration_result() is None
    assert not window.adjust_registration_action.isEnabled()
    assert not window.overlay_panel.view.has_image()


# ---------------------------------------------------------------- comparison


@pytest.fixture
def defective_label(tmp_path):
    artwork = syn.make_artwork()
    printed, defects = syn.add_defects(artwork, ["extra_mark", "missing_region", "small_dot"], seed=5)
    photo = syn.make_photo(printed, photo_size=(2000, 1500), roll_deg=-10, tilt_x_deg=12, label_fraction=0.75)
    art_path = write_image(tmp_path / "art.png", artwork)
    photo_path = write_image(tmp_path / "defects.jpg", photo.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return art_path, photo_path, defects


def _load_and_compare(qtbot, window, art_path, photo_path):
    _load(qtbot, window, ImageRole.ORIGINAL, art_path)
    with qtbot.waitSignal(window.comparison_done, timeout=TIMEOUT_MS * 2) as blocker:
        window.load_image_file(ImageRole.PRINTED, photo_path)
    return blocker.args[0]


def test_comparison_runs_after_registration_and_fills_list(qtbot, window, defective_label):
    art_path, photo_path, defects = defective_label
    result = _load_and_compare(qtbot, window, art_path, photo_path)
    report = window.defect_report()
    assert window.comparison_result() is result
    assert len(report.defects) >= len(defects)
    assert window.defect_table.rowCount() == len(report.defects)
    assert window.defect_table.item(0, 0).text() == "1"
    assert window.defect_table.item(0, 1).text() == report.defects[0].category_label
    assert window.defect_table.item(0, COL_OPERATOR).text() == "Not reviewed"
    assert window.defect_table.item(0, DEFECT_COLUMNS.index("Position (mm)")).text().count(",") == 1
    assert window.overlay_mode() == "defects"
    assert window.compare_action.isEnabled()
    assert "Candidates" in window.comparison_section.metrics_label.text()
    assert window.status_indicator.status is not InspectionStatus.PASS


def test_selecting_defect_zooms_and_highlights(qtbot, window, defective_label):
    art_path, photo_path, _ = defective_label
    _load_and_compare(qtbot, window, art_path, photo_path)
    report = window.defect_report()
    target = report.defects[-1]
    window.defect_table.selectRow(len(report.defects) - 1)
    assert window._highlight == target.id
    assert window.overlay_panel.view.zoom >= 1.0
    centre = window.overlay_panel.view.mapToScene(window.overlay_panel.view.viewport().rect().center())
    assert abs(centre.x() - (target.x + target.width / 2)) < 40
    assert abs(centre.y() - (target.y + target.height / 2)) < 40


def test_comparison_views_switchable(qtbot, window, defective_label):
    art_path, photo_path, _ = defective_label
    result = _load_and_compare(qtbot, window, art_path, photo_path)
    gw, gh = result.metrics.grid_size
    for mode in ("score", "gray", "structural", "edge_map", "expected", "blend", "checkerboard"):
        window.set_overlay(mode, 60)
        assert window.overlay_image().shape == (gh, gw, 3)


def test_sensitivity_change_reruns_comparison(qtbot, window, defective_label):
    art_path, photo_path, _ = defective_label
    _load_and_compare(qtbot, window, art_path, photo_path)
    with qtbot.waitSignal(window.comparison_done, timeout=TIMEOUT_MS) as blocker:
        window.sensitivity_combo.setCurrentIndex(window.sensitivity_combo.findData("high"))
    assert blocker.args[0].metrics.sensitivity == "high"
    with qtbot.waitSignal(window.comparison_done, timeout=TIMEOUT_MS) as blocker:
        window.compare_action.trigger()
    assert blocker.args[0].metrics.sensitivity == "high"


def test_reset_clears_comparison(qtbot, window, defective_label):
    art_path, photo_path, _ = defective_label
    _load_and_compare(qtbot, window, art_path, photo_path)
    window.reset()
    assert window.comparison_result() is None
    assert window.defect_table.rowCount() == 0
    assert not window.compare_action.isEnabled()


# ---------------------------------------------------------------- defects, operator marks and verdict

from print_inspector.core.decision import OperatorMark  # noqa: E402


@pytest.fixture
def inspection_files(tmp_path):
    """Artwork with DPI and off-white stock (a realistic, not blown-out photo), plus two prints."""
    artwork = np.minimum(syn.make_artwork(), 236)
    art_path = tmp_path / "art300.png"
    Image.fromarray(artwork[:, :, ::-1].copy()).save(art_path, dpi=(300, 300))
    paths = {}
    for name, printed in (
        ("good", artwork),
        ("defective", syn.add_defects(artwork, ["extra_mark", "small_dot"], seed=3)[0]),
    ):
        photo = syn.make_photo(
            printed, photo_size=(2400, 1800), roll_deg=-10, tilt_x_deg=12, label_fraction=0.75,
            render=dict(noise_sigma=2.0, jpeg_quality=92),
        )
        paths[name] = write_image(tmp_path / f"{name}.jpg", photo.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return art_path, paths


def test_good_label_passes(qtbot, window, inspection_files):
    art_path, paths = inspection_files
    _load_and_compare(qtbot, window, art_path, paths["good"])
    assert window.defect_report().defects == ()
    assert window.status_indicator.status is InspectionStatus.PASS
    assert window.decision().status is InspectionStatus.PASS
    assert "No defect meets the rejection criteria" in window.status_reason_label.text()


def test_defective_label_fails_and_operator_can_overrule(qtbot, window, inspection_files):
    art_path, paths = inspection_files
    _load_and_compare(qtbot, window, art_path, paths["defective"])
    report = window.defect_report()
    rejecting = report.defects[0]
    assert rejecting.severity.value in ("major", "critical") and rejecting.is_reliable
    assert window.status_indicator.status is InspectionStatus.FAIL
    assert window.decision().failing == (rejecting.id,)

    # Red numbered boxes on the registered photo.
    window.set_printed_view_mode(VIEW_REGISTERED)
    assert window.printed_panel.view.boxes() == len(report.defects)

    # Select -> side-by-side evidence; dismiss -> PASS; needs review -> REVIEW; clear -> FAIL.
    window.defect_table.selectRow(0)
    assert window.defect_detail.defect() is rejecting
    assert not window.defect_detail.images["printed"].pixmap().isNull()
    window.defect_detail.buttons[OperatorMark.DISMISSED].click()
    assert window.marks() == {rejecting.id: OperatorMark.DISMISSED}
    assert window.defect_table.item(0, COL_OPERATOR).text() == "Dismissed"
    assert window.status_indicator.status is InspectionStatus.PASS
    window.set_mark(rejecting.id, OperatorMark.NEEDS_REVIEW)
    assert window.status_indicator.status is InspectionStatus.REVIEW
    window.set_mark(rejecting.id, OperatorMark.UNREVIEWED)
    assert window.status_indicator.status is InspectionStatus.FAIL


def test_confirming_an_uncertain_finding_fails(qtbot, window, inspection_files, monkeypatch):
    art_path, paths = inspection_files
    strict = dataclasses.replace(window._config.defects, reliable_min_score=1000.0)  # nothing counts as reliable
    monkeypatch.setattr(window, "_config", dataclasses.replace(window._config, defects=strict))
    _load_and_compare(qtbot, window, art_path, paths["defective"])
    first = window.defect_report().defects[0]
    assert not first.is_reliable
    assert window.status_indicator.status is InspectionStatus.REVIEW
    window.set_mark(first.id, OperatorMark.CONFIRMED)
    assert window.status_indicator.status is InspectionStatus.FAIL
    assert "Confirmed" in window.status_reason_label.text()


def test_new_comparison_clears_marks(qtbot, window, inspection_files):
    art_path, paths = inspection_files
    _load_and_compare(qtbot, window, art_path, paths["defective"])
    window.set_mark(1, OperatorMark.DISMISSED)
    with qtbot.waitSignal(window.comparison_done, timeout=TIMEOUT_MS):
        window.compare_action.trigger()
    assert window.marks() == {}
    assert window.defect_detail.defect() is None


# ---------------------------------------------------------------- Step 7: interface

from PySide6.QtCore import QPointF, QSettings, Qt  # noqa: E402
from PySide6.QtGui import QKeySequence  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QToolBar  # noqa: E402

from print_inspector.config import config_with_overrides, read_user_overrides  # noqa: E402


def test_toolbar_status_area_and_shortcuts(window):
    toolbar = window.findChild(QToolBar, "mainToolbar")
    texts = [a.text() for a in toolbar.actions() if a.text()]
    required = ["Load Original…", "Load Printed…", "Compare", "Reset", "Save Results…", "Settings…"]
    assert [t for t in texts if t in required] == required
    assert window.alignment_button.menu().actions() == [
        window.detect_action, window.manual_corners_action, window.adjust_registration_action
    ]
    shortcuts = dict(window.shortcut_summary())
    native = lambda s: QKeySequence(s).toString(QKeySequence.SequenceFormat.NativeText)  # noqa: E731
    assert shortcuts["Compare"] == "F5"
    assert shortcuts["Load Original"] == native("Ctrl+O")
    assert shortcuts["Next Defect"] == "N" and shortcuts["Dismiss Defect"] == "D"
    assert shortcuts["Show Defect Markers"] == "B"
    assert shortcuts["Fit to Window"] == native("Ctrl+0")
    # Status area before anything is loaded.
    area = window.status_area
    assert area.quality.word() == "-" and area.alignment.word() == "-"
    assert area.progress_label.full_text() == "Waiting for images."
    assert window.status_indicator is area.verdict
    assert not window.banner.isVisible()
    assert window.settings_action.isEnabled()
    for action in window.mark_actions.values():
        assert not action.isEnabled()  # nothing selected


def test_initial_window_fits_the_screen(window):
    available = window.screen().availableGeometry()
    assert window.width() <= available.width() and window.height() <= available.height()


def test_window_layout_is_remembered(qtbot, app_config, tmp_path):
    settings = QSettings(str(tmp_path / "window.ini"), QSettings.Format.IniFormat)
    first = MainWindow(app_config, window_settings=settings)
    qtbot.addWidget(first)
    first.show()
    first.resize(760, 540)
    first.close()
    settings.sync()
    second = MainWindow(app_config, window_settings=settings)
    qtbot.addWidget(second)
    assert (second.width(), second.height()) == (760, 540)


def test_dropping_a_file_on_a_viewer_loads_it(qtbot, window, label_png):
    with qtbot.waitSignal(window.image_loaded, timeout=TIMEOUT_MS) as blocker:
        window.original_panel.file_dropped.emit(label_png)
    assert blocker.args == [ImageRole.ORIGINAL]
    assert window.loaded_image(ImageRole.ORIGINAL).path == label_png


def test_progress_and_badges_through_the_inspection(qtbot, window, inspection_files):
    art_path, paths = inspection_files
    _load(qtbot, window, ImageRole.ORIGINAL, art_path)
    assert window.status_area.progress_label.full_text() == "Load a photo of the printed label."
    with qtbot.waitSignal(window.comparison_done, timeout=TIMEOUT_MS * 2):
        window.load_image_file(ImageRole.PRINTED, paths["good"])
        assert window.status_area.progress_label.full_text().startswith("Running:")
        assert not window.save_action.isEnabled()  # not while work is running
    area = window.status_area
    assert area.quality.word() in ("GOOD", "MARGINAL")
    assert area.alignment.word() in ("GOOD", "MARGINAL")
    assert area.progress_label.full_text().startswith("Inspection complete")
    assert area.progress.value() == area.progress.maximum() == 4
    assert area.verdict.text() == "PASS"
    assert "No defect meets the rejection criteria" in area.verdict_reason.full_text()
    assert window.save_action.isEnabled()


def test_defect_navigation_markers_and_synchronised_views(qtbot, window, inspection_files):
    art_path, paths = inspection_files
    _load_and_compare(qtbot, window, art_path, paths["defective"])
    report = window.defect_report()
    n = len(report.defects)
    assert n >= 2
    sx, sy = window._grid_scale()

    # Numbered boxes on all three viewers; the toggle hides and shows them.
    window.set_printed_view_mode(VIEW_REGISTERED)
    assert window.original_panel.view.boxes() == n
    assert window.printed_panel.view.boxes() == n
    window.show_defects_action.trigger()
    assert window.original_panel.view.boxes() == 0 and window.printed_panel.view.boxes() == 0
    window.show_defects_action.trigger()
    assert window.original_panel.view.boxes() == n

    # Next / previous defect select rows and zoom all views to the same spot.
    window.next_defect_action.trigger()
    first = report.defects[0]
    assert window.selected_defect() is first
    assert window.defect_table.currentRow() == 0
    centre, zoom = window.original_panel.view.view_state()
    assert centre.x() == pytest.approx(first.x / sx + first.width / sx / 2, abs=3)
    assert centre.y() == pytest.approx(first.y / sy + first.height / sy / 2, abs=3)
    overlay_zoom = window.overlay_panel.view.zoom
    assert zoom == pytest.approx(overlay_zoom * sx, rel=0.05)  # same on-screen size
    window.next_defect_action.trigger()
    assert window.selected_defect() is report.defects[1]
    window.previous_defect_action.trigger()
    window.previous_defect_action.trigger()
    assert window.selected_defect() is report.defects[-1]  # wraps around

    # Zooming one view zooms the others (synchronised), unless synchronisation is off.
    window.zoom_in()
    assert window.original_panel.view.zoom == pytest.approx(window.overlay_panel.view.zoom * sx, rel=0.05)
    window.sync_action.trigger()
    before = window.original_panel.view.zoom
    window.overlay_panel.view.set_zoom(window.overlay_panel.view.zoom * 2)
    assert window.original_panel.view.zoom == pytest.approx(before)
    window.sync_action.trigger()

    # Marks from the keyboard commands act on the selected defect.
    window.mark_actions[OperatorMark.DISMISSED].trigger()
    assert window.marks() == {report.defects[-1].id: OperatorMark.DISMISSED}
    assert window.defect_table.item(n - 1, COL_OPERATOR).text() == "Dismissed"
    window.mark_actions[OperatorMark.UNREVIEWED].trigger()
    assert window.marks() == {}


def test_clicking_a_defect_box_selects_it(qtbot, window, inspection_files):
    art_path, paths = inspection_files
    _load_and_compare(qtbot, window, art_path, paths["defective"])
    target = window.defect_report().defects[-1]
    view = window.overlay_panel.view
    view.fit_to_window()
    centre = QPointF(target.x + target.width / 2, target.y + target.height / 2)
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, view.mapFromScene(centre))
    assert window.selected_defect() is target
    assert window.defect_detail.defect() is target
    # A click on the artwork viewer (artwork coordinates) works too.
    first = window.defect_report().defects[0]
    sx, sy = window._grid_scale()
    window._on_view_clicked("original", QPointF((first.x + first.width / 2) / sx, (first.y + first.height / 2) / sy))
    assert window.selected_defect() is first
    # Empty paper selects nothing new.
    window._on_view_clicked("overlay", QPointF(2, 2))
    assert window.selected_defect() is first


def test_save_results_writes_folder(qtbot, window, inspection_files, tmp_path):
    art_path, paths = inspection_files
    _load_and_compare(qtbot, window, art_path, paths["defective"])
    first = window.defect_report().defects[0]
    window.set_mark(first.id, OperatorMark.CONFIRMED)
    out = tmp_path / "results"
    out.mkdir()
    with qtbot.waitSignal(window.results_saved, timeout=TIMEOUT_MS) as blocker:
        window.save_action.setEnabled(True)
        window.save_results_to(out)
    folder = blocker.args[0]
    assert folder.parent == out and folder.name.endswith("_FAIL") and folder.name.startswith("defective_")
    summary = json.loads((folder / "inspection.json").read_text(encoding="utf-8"))
    assert summary["verdict"]["status"] == "FAIL"
    assert summary["files"]["printed"]["path"] == str(paths["defective"])
    assert summary["defects"]["items"][0]["operator"] == "confirmed"
    assert summary["defects"]["scale_estimated"] is False  # artwork has 300 dpi
    for name in ("defects.csv", "findings.csv", "registered_print.png", "annotated_comparison.png",
                 "difference_heatmap.png", "report.pdf", "defect_01.png"):
        assert (folder / name).is_file(), name
    assert window.banner.isVisible() and window.banner.kind() == "info"
    assert window.save_action.isEnabled()


def test_save_failure_is_reported(qtbot, window, inspection_files, tmp_path, error_boxes):
    art_path, paths = inspection_files
    _load_and_compare(qtbot, window, art_path, paths["good"])
    with qtbot.waitSignal(window.results_save_failed, timeout=TIMEOUT_MS):
        window.save_results_to(tmp_path / "missing")
    assert error_boxes and "does not exist" in error_boxes[0]


def test_settings_apply_and_are_saved(qtbot, window, inspection_files, tmp_path):
    art_path, paths = inspection_files
    _load_and_compare(qtbot, window, art_path, paths["defective"])
    assert window.status_indicator.status is InspectionStatus.FAIL
    settings_file = tmp_path / "user" / "config.json"
    window._user_config_path = settings_file

    # Severity limits change -> the comparison re-runs; everything becomes minor -> PASS.
    lenient = {"defects": {
        "severity_major_size_mm": 50.0, "severity_critical_size_mm": 100.0,
        "severity_major_area_mm2": 500.0, "severity_critical_area_mm2": 1000.0, "text_major_size_mm": 50.0,
    }, "text": {"enabled": False}, "barcodes": {"enabled": False}, "color": {"enabled": False}}
    with qtbot.waitSignal(window.comparison_done, timeout=TIMEOUT_MS):
        window.apply_settings(config_with_overrides(lenient), lenient)
    assert all(d.severity.value == "minor" for d in window.defect_report().defects)
    assert window.status_indicator.status is InspectionStatus.PASS
    assert read_user_overrides(settings_file) == lenient

    # A decision-only change re-evaluates the verdict without comparing again.
    comparison = window.comparison_result()
    strict = {**lenient, "decision": {"max_minor_defects": 0}}
    window.apply_settings(config_with_overrides(strict), strict)
    assert window.comparison_result() is comparison
    assert window.status_indicator.status is InspectionStatus.FAIL
    assert read_user_overrides(settings_file) == strict


def test_settings_dialog_starts_from_the_saved_file(qtbot, window, tmp_path):
    settings_file = tmp_path / "config.json"
    settings_file.write_text('{"decision": {"max_minor_defects": 5}}', encoding="utf-8")
    window._user_config_path = settings_file
    dialog = window.create_settings_dialog()
    qtbot.addWidget(dialog)
    dialog.editors[("decision", "fail_min_severity")].setCurrentIndex(
        dialog.editors[("decision", "fail_min_severity")].findData("critical")
    )
    dialog.accept()
    assert dialog.overrides() == {"decision": {"max_minor_defects": 5, "fail_min_severity": "critical"}}

    settings_file.write_text("{not json", encoding="utf-8")
    broken = window.create_settings_dialog()
    qtbot.addWidget(broken)
    assert "invalid" in broken.notice_label.text()


def test_compare_repeats_label_detection_when_needed(qtbot, window, label_png, tmp_path):
    _load(qtbot, window, ImageRole.ORIGINAL, label_png)
    path = write_image(tmp_path / "plain.jpg", np.full((1500, 2000, 3), 140, np.uint8))
    with qtbot.waitSignal(window.alignment_done, timeout=TIMEOUT_MS):
        window.load_image_file(ImageRole.PRINTED, path)
    assert window.compare_action.isEnabled()
    assert window.status_area.alignment.word() == "POOR"
    with qtbot.waitSignal(window.alignment_done, timeout=TIMEOUT_MS):
        window.compare_action.trigger()
    assert window.status_area.progress_label.full_text().startswith("Stopped")


def test_background_failure_shows_banner_and_review(qtbot, window, inspection_files):
    art_path, paths = inspection_files
    _load_and_compare(qtbot, window, art_path, paths["good"])
    window._on_comparison_failed(window._comparison_generation, "Comparison failed: out of memory")
    assert window.banner.isVisible() and window.banner.kind() == "error"
    assert "out of memory" in window.banner.text()
    assert window.status_indicator.status is InspectionStatus.REVIEW
    window.reset()
    assert not window.banner.isVisible()
