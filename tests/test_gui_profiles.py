"""Regions, inspection profiles and reports in the application (full pipeline from synthetic phone photos)."""
import dataclasses
import json

import cv2
import numpy as np
import pytest
import shiboken6
from PIL import Image
from PySide6.QtCore import QRectF

from print_inspector import synthetic as syn
from print_inspector.config import read_user_overrides
from print_inspector.core.image_io import ImageRole
from print_inspector.core.models import InspectionStatus
from print_inspector.core.profiles import find_profile, load_profile
from print_inspector.core.regions import RegionKind, load_regions, regions_path
from print_inspector.core.text_zones import TextZone, load_zones, zones_path
from print_inspector.gui.main_window import MainWindow
from tests.conftest import write_image

TIMEOUT_MS = 120000


@pytest.fixture
def files(tmp_path):
    artwork = np.minimum(syn.make_artwork(), 236)
    art = tmp_path / "art300.png"
    Image.fromarray(artwork[:, :, ::-1].copy()).save(art, dpi=(300, 300))
    printed, truth = syn.add_defects(artwork, ["extra_mark", "small_dot"], seed=3)
    photo = syn.make_photo(printed, photo_size=(2400, 1800), roll_deg=-10, tilt_x_deg=12, label_fraction=0.75,
                           render=dict(noise_sigma=2.0, jpeg_quality=92))
    path = write_image(tmp_path / "defective.jpg", photo.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return art, path, {d.kind: d.bbox for d in truth}


@pytest.fixture
def window(qtbot, app_config, tmp_path):
    config = dataclasses.replace(
        app_config,
        text=dataclasses.replace(app_config.text, enabled=False),
        barcodes=dataclasses.replace(app_config.barcodes, enabled=False),
        color=dataclasses.replace(app_config.color, enabled=False),
    )
    win = MainWindow(config, user_config_path=tmp_path / "user" / "config.json", profiles_dir=tmp_path / "profiles")
    qtbot.addWidget(win)
    win.show()
    yield win
    if shiboken6.isValid(win._thread_pool):
        win._thread_pool.waitForDone(TIMEOUT_MS)


def inspect(qtbot, window, art, photo):
    with qtbot.waitSignal(window.image_loaded, timeout=TIMEOUT_MS):
        window.load_image_file(ImageRole.ORIGINAL, art)
    with qtbot.waitSignal(window.comparison_done, timeout=TIMEOUT_MS):
        window.load_image_file(ImageRole.PRINTED, photo)


def defect_at(window, box):
    """The defect covering ``box`` (artwork pixels), or None."""
    sx, sy = window._grid_scale()
    x, y, w, h = box
    cx, cy = (x + w / 2) * sx, (y + h / 2) * sy
    return next((d for d in window.defect_report().defects
                 if d.x - 4 <= cx <= d.x + d.width + 4 and d.y - 4 <= cy <= d.y + d.height + 4), None)


def region_rect(box, pad=25):
    x, y, w, h = box
    return QRectF(x - pad, y - pad, w + 2 * pad, h + 2 * pad)


def draw_regions(qtbot, window, *regions):
    """Draw (box, kind) regions in the dialog and apply them; returns the regions."""
    dialog = window.create_region_dialog()
    qtbot.addWidget(dialog)
    for row, (box, kind) in enumerate(regions):
        dialog.add_zone_rect(region_rect(box))
        combo = dialog.table.cellWidget(row, 1)
        combo.setCurrentIndex(combo.findData(kind.value))
    assert dialog.validate() is None
    with qtbot.waitSignal(window.comparison_done, timeout=TIMEOUT_MS):
        window.apply_regions(dialog.regions())
    return dialog.regions()


def test_regions_change_the_inspection_and_are_saved_with_the_artwork(qtbot, window, files):
    art, photo, truth = files
    inspect(qtbot, window, art, photo)
    dot = defect_at(window, truth["small_dot"])
    mark = defect_at(window, truth["extra_mark"])
    assert dot is not None and mark is not None and dot.severity.value == "minor"

    regions = draw_regions(qtbot, window, (truth["extra_mark"], RegionKind.IGNORE),
                           (truth["small_dot"], RegionKind.CRITICAL))
    assert [r.name for r in regions] == ["Region 1", "Region 2"]
    assert defect_at(window, truth["extra_mark"]) is None  # ignored: not inspected
    dot = defect_at(window, truth["small_dot"])
    assert dot.severity.value == "major" and dot.critical_region == "Region 2"
    assert window.status_indicator.status is InspectionStatus.FAIL
    assert load_regions(regions_path(art)) == regions
    labels = [item.text() for item in window.overlay_panel.view._box_items if hasattr(item, "text")]
    assert "Ignored: Region 1" in labels and "Critical: Region 2" in labels

    # Loading the artwork again brings the regions back.
    window.reset()
    assert window.regions() == ()
    with qtbot.waitSignal(window.image_loaded, timeout=TIMEOUT_MS):
        window.load_image_file(ImageRole.ORIGINAL, art)
    assert window.regions() == regions


def test_profile_save_close_and_reload(qtbot, window, files, tmp_path):
    art, photo, truth = files
    inspect(qtbot, window, art, photo)
    regions = draw_regions(qtbot, window, (truth["small_dot"], RegionKind.CRITICAL))
    zones = (TextZone("Lot", 0.05, 0.05, 0.3, 0.1, expected_text="LOT 1"),)
    window.apply_text_zones(zones)  # saved next to the artwork (no profile yet)
    assert load_zones(zones_path(art)) == zones
    config = window.config()
    strict = dataclasses.replace(config, decision=dataclasses.replace(config.decision, max_minor_defects=0))
    window.apply_settings(strict, None)
    with qtbot.waitSignal(window.comparison_done, timeout=TIMEOUT_MS):
        window.sensitivity_combo.setCurrentIndex(window.sensitivity_combo.findData("high"))

    dialog = window.create_profile_dialog()
    qtbot.addWidget(dialog)
    assert dialog.entries() == [] and not dialog.load_button.isEnabled()
    assert dialog.save_current("Product 4711", "Front label")
    window.finish_profile_dialog(dialog)
    assert window.inspection_profile().name == "Product 4711" and not window.profile_modified()
    assert "Profile: Product 4711" in window.windowTitle() and "modified" not in window.windowTitle()
    stored = load_profile(find_profile(tmp_path / "profiles", "Product 4711").path)
    assert stored.sensitivity == "high" and stored.regions == regions and stored.text_zones == zones
    assert stored.settings["decision"]["max_minor_defects"] == 0 and stored.artwork_name == "art300.png"

    # With a profile in use, edits belong to the profile (not to the files next to the artwork).
    window.apply_text_zones(())
    assert load_zones(zones_path(art)) == zones
    assert window.profile_modified() and "(modified)" in window.windowTitle()
    window.apply_text_zones(zones)
    assert not window.profile_modified()

    # Closing the profile restores this computer's settings and the artwork's own regions and zones.
    with qtbot.waitSignal(window.comparison_done, timeout=TIMEOUT_MS):
        window.close_inspection_profile()
    assert window.inspection_profile() is None and "Profile" not in window.windowTitle()
    assert window.config().decision.max_minor_defects == 0  # applied before the profile existed
    assert window.sensitivity_combo.currentData() == "medium"
    assert window.regions() == regions and window._text_zones == zones
    window.apply_regions(())  # the artwork's own regions are cleared ...
    assert window.regions() == ()

    # ... and loading the profile brings everything back.
    dialog = window.create_profile_dialog()
    qtbot.addWidget(dialog)
    assert dialog.select_name("Product 4711")
    with qtbot.waitSignal(window.comparison_done, timeout=TIMEOUT_MS):
        assert dialog.load_selected()
        window.finish_profile_dialog(dialog)
    assert window.regions() == regions and window.sensitivity_combo.currentData() == "high"
    assert window.config().decision.max_minor_defects == 0
    assert defect_at(window, truth["small_dot"]).critical_region == "Region 1"


def test_settings_with_a_profile_in_use(qtbot, window, files, tmp_path):
    art, photo, _ = files
    inspect(qtbot, window, art, photo)
    dialog = window.create_profile_dialog()
    qtbot.addWidget(dialog)
    dialog.save_current("Label B")
    window.finish_profile_dialog(dialog)

    settings = window.create_settings_dialog()
    qtbot.addWidget(settings)
    assert "Profile 'Label B' is in use" in settings.notice_label.text()
    settings.editors[("decision", "max_minor_defects")].setValue(7)  # stored in the profile
    settings.editors[("text", "tesseract_path")].setText("C:/OCR")  # this computer's path
    settings.editors[("report", "inspector")].setText("J. Doe")  # not part of a profile
    settings.accept()
    window.accept_settings_dialog(settings)
    assert window.config().decision.max_minor_defects == 7 and window.config().text.tesseract_path == "C:/OCR"
    assert window.profile_modified()
    saved = read_user_overrides(tmp_path / "user" / "config.json")
    assert saved["text"]["tesseract_path"] == "C:/OCR" and saved["report"]["inspector"] == "J. Doe"
    assert "decision" not in saved  # the profile's setting is not written to this computer's file
    assert window.save_inspection_profile() and not window.profile_modified()
    assert load_profile(window._profile_path).settings["decision"]["max_minor_defects"] == 7


def test_saved_results_name_the_profile_and_contain_the_report(qtbot, window, files, tmp_path):
    art, photo, truth = files
    inspect(qtbot, window, art, photo)
    draw_regions(qtbot, window, (truth["small_dot"], RegionKind.CRITICAL))
    dialog = window.create_profile_dialog()
    qtbot.addWidget(dialog)
    dialog.save_current("Label C")
    window.finish_profile_dialog(dialog)
    out = tmp_path / "results"
    out.mkdir()
    with qtbot.waitSignal(window.results_saved, timeout=TIMEOUT_MS) as saved:
        window.save_results_to(out)
    folder = saved.args[0]
    for name in ("report.pdf", "findings.csv", "annotated_comparison.png", "difference_heatmap.png",
                 "inspection.json", "defects.csv"):
        assert (folder / name).is_file(), name
    summary = json.loads((folder / "inspection.json").read_text(encoding="utf-8"))
    assert summary["report"]["profile"] == "Label C" and summary["setup"]["profile_modified"] is False
    assert summary["setup"]["regions"][0]["kind"] == "critical"
    assert summary["report"]["original_file"] == "art300.png" and summary["report"]["printed_file"] == "defective.jpg"


def test_profile_library_import_export_delete(qtbot, window, files, tmp_path):
    art, _, _ = files
    with qtbot.waitSignal(window.image_loaded, timeout=TIMEOUT_MS):
        window.load_image_file(ImageRole.ORIGINAL, art)
    dialog = window.create_profile_dialog()
    qtbot.addWidget(dialog)
    answers = []
    dialog.confirm = lambda title, text: answers.append(title) or True
    assert not dialog.save_current("  ")
    assert "name" in dialog.error_label.text()
    assert dialog.save_current("Label D")
    exported = tmp_path / "share" / "label_d.json"
    exported.parent.mkdir()
    assert dialog.export_selected(exported) and load_profile(exported).name == "Label D"
    assert dialog.delete_selected() and dialog.entries() == [] and answers == ["Delete Profile"]
    assert dialog.import_file(exported) and [e.name for e in dialog.entries()] == ["Label D"]
    assert dialog.import_file(exported) and answers[-1] == "Replace Profile"  # asks before replacing
    bad = tmp_path / "share" / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert not dialog.import_file(bad) and "not a Print Inspector profile" in dialog.error_label.text()


def test_profile_for_an_artwork_of_another_shape_warns(qtbot, window, files, tmp_path):
    art, _, _ = files
    with qtbot.waitSignal(window.image_loaded, timeout=TIMEOUT_MS):
        window.load_image_file(ImageRole.ORIGINAL, art)
    profile = dataclasses.replace(
        window.current_profile("Square label"), artwork_size=(1000, 1000),
        text_zones=(TextZone("Lot", 0.1, 0.1, 0.2, 0.1),),
    )
    assert window.load_inspection_profile(tmp_path / "square.json", profile)
    assert window.banner.isVisible() and window.banner.kind() == "warning"
    assert "different shape" in window.banner.text()
