"""Colour analysis in the application (full pipeline from a synthetic phone photo)."""
import csv
import dataclasses
import json

import cv2
import numpy as np
import pytest
import shiboken6
from PIL import Image

from print_inspector import synthetic as syn
from print_inspector.core.color_calibration import COLORCHECKER_24, chart_field_corners, load_profile, render_chart
from print_inspector.core.color_inspection import ColorFindingKind
from print_inspector.core.decision import OperatorMark
from print_inspector.core.image_io import ImageRole
from print_inspector.core.models import InspectionStatus
from print_inspector.gui.color_panel import COL_COLOR_OPERATOR
from print_inspector.gui.main_window import COLOR_VIEWS, MainWindow
from tests.color_fakes import PATCHES, make_color_label, recolour
from tests.conftest import write_image

TIMEOUT_MS = 60000


@pytest.fixture
def files(tmp_path):
    artwork = make_color_label()
    art = tmp_path / "label.png"
    Image.fromarray(artwork[:, :, ::-1].copy()).save(art, dpi=(304.8, 304.8))
    spot = artwork.copy()
    x, y, w, h = PATCHES["red"][1]
    recolour(spot, (x + 100, y + 60, 80, 80), (40, 90, 200))
    paths = {}
    for name, printed in (("good", artwork), ("spot", spot)):
        photo = syn.make_photo(printed, photo_size=(2400, 1800), roll_deg=-8, tilt_x_deg=10, label_fraction=0.75,
                               render=dict(noise_sigma=2.0, jpeg_quality=92))
        paths[name] = write_image(tmp_path / f"{name}.jpg", photo.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    chart = np.full((700, 900, 3), 200, np.uint8)
    chart[100:100 + render_chart(COLORCHECKER_24).shape[0], 100:100 + render_chart(COLORCHECKER_24).shape[1]] = render_chart(COLORCHECKER_24)
    paths["chart"] = write_image(tmp_path / "chart.png", chart)
    return art, paths


@pytest.fixture
def window(qtbot, app_config, tmp_path):
    config = dataclasses.replace(
        app_config,
        text=dataclasses.replace(app_config.text, enabled=False),
        barcodes=dataclasses.replace(app_config.barcodes, enabled=False),
    )
    win = MainWindow(config, color_profile_path=tmp_path / "profile" / "color_profile.json")
    qtbot.addWidget(win)
    win.show()
    yield win
    if shiboken6.isValid(win._thread_pool):
        win._thread_pool.waitForDone(TIMEOUT_MS)


def inspect(qtbot, window, art, photo):
    with qtbot.waitSignal(window.image_loaded, timeout=TIMEOUT_MS):
        window.load_image_file(ImageRole.ORIGINAL, art)
    with qtbot.waitSignal(window.color_done, timeout=TIMEOUT_MS) as blocker:
        window.load_image_file(ImageRole.PRINTED, photo)
    return blocker.args[0]


def test_good_print_passes_with_approximate_colour(qtbot, window, files):
    art, paths = files
    report = inspect(qtbot, window, art, paths["good"])
    assert report.mode == "approximate" and report.findings == ()
    assert window.status_indicator.status is InspectionStatus.PASS
    panel = window.color_panel
    assert "APPROXIMATE" in panel.mode_label.text() and "not a print colour measurement" in panel.mode_label.text()
    assert panel.areas_table.rowCount() == len(report.areas) >= 5
    assert window.color_section.level_label.text().endswith("APPROXIMATE")
    # Heatmap views in the difference viewer.
    for key in COLOR_VIEWS:
        window.set_overlay(key, 50)
        assert window.overlay_image().shape == window.registration_result().registered.shape
        assert "ΔE00" in window.overlay_panel.info_label.text()
    assert "APPROXIMATE" in window.overlay_panel.info_label.text()


def test_local_colour_change_fails_and_can_be_overruled(qtbot, window, files, tmp_path):
    art, paths = files
    report = inspect(qtbot, window, art, paths["spot"])
    local = [f for f in report.findings if f.kind is ColorFindingKind.LOCAL_CHANGE]
    assert len(local) == 1 and local[0].reliable
    assert window.status_indicator.status is InspectionStatus.FAIL
    assert local[0].id in window.decision().color_failing
    window.select_color_finding(local[0].id)
    assert window.side_tabs.currentWidget() is window.color_panel
    labels = [item.text() for item in window.printed_panel.view._box_items if hasattr(item, "text")]
    assert local[0].label in labels
    window.mark_actions[OperatorMark.DISMISSED].trigger()
    assert window.color_marks() == {local[0].id: OperatorMark.DISMISSED}
    assert window.color_panel.findings_table.item(0, COL_COLOR_OPERATOR).text() == "Dismissed"

    out = tmp_path / "results"
    out.mkdir()
    with qtbot.waitSignal(window.results_saved, timeout=TIMEOUT_MS) as saved:
        window.save_results_to(out)
    folder = saved.args[0]
    summary = json.loads((folder / "inspection.json").read_text(encoding="utf-8"))
    assert summary["color"]["mode"] == "approximate" and "not a print colour measurement" in summary["color"]["note"]
    assert summary["color"]["findings"][0]["operator"] == "dismissed"
    with open(folder / "colors.csv", encoding="utf-8-sig", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == len(report.areas)
    with open(folder / "color_findings.csv", encoding="utf-8-sig", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["problem"] == "Local colour change"
    assert (folder / "color_heatmap.png").is_file() and (folder / "color_heatmap_local.png").is_file()


def test_calibration_from_a_chart_photo(qtbot, window, files):
    art, paths = files
    inspect(qtbot, window, art, paths["good"])
    dialog = window.create_calibration_dialog()
    qtbot.addWidget(dialog)
    assert "No calibration" in dialog.current_label.text()
    assert dialog.use_chart_photo(paths["chart"])
    assert dialog.calibrate_with(chart_field_corners(COLORCHECKER_24) + 100)
    profile = dialog.profile()
    assert profile is not None and profile.source == "separate photo" and profile.loo_mean_delta_e < 3
    assert "accepted" in dialog.result_label.text() and dialog.patch_table.rowCount() == 24
    assert dialog.calibrate_with(chart_field_corners(COLORCHECKER_24)[[2, 3, 0, 1]] + 100) is False
    assert "lightness order" in dialog.error_label.text()

    with qtbot.waitSignal(window.color_done, timeout=TIMEOUT_MS) as blocker:
        window.apply_color_profile(profile)
    report = blocker.args[0]
    assert report.mode == "calibrated" and report.calibrated
    assert load_profile(window._color_profile_path) == profile  # kept for the next start
    assert "CALIBRATED" in window.color_panel.mode_label.text()

    with qtbot.waitSignal(window.color_done, timeout=TIMEOUT_MS) as blocker:
        window.apply_color_profile(None)
    assert blocker.args[0].mode == "approximate"
    assert not window._color_profile_path.exists()


def test_colour_settings_rerun_only_the_colour(qtbot, window, files):
    art, paths = files
    inspect(qtbot, window, art, paths["good"])
    comparison = window.comparison_result()
    config = window.config()
    with qtbot.waitSignal(window.color_done, timeout=TIMEOUT_MS):
        window.apply_settings(dataclasses.replace(config, color=dataclasses.replace(config.color, local_delta_e=4.0)))
    assert window.comparison_result() is comparison
    window.apply_settings(dataclasses.replace(config, color=dataclasses.replace(config.color, enabled=False)))
    assert window.color_report() is None and window.status_indicator.status is InspectionStatus.PASS
