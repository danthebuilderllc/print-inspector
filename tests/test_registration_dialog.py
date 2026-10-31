import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from print_inspector import synthetic as syn
from print_inspector.core.alignment import align_label
from print_inspector.core.registration import ManualAdjustment, register_to_artwork
from print_inspector.gui.registration_dialog import NUDGE_PX, RegistrationAdjustDialog


@pytest.fixture(scope="module")
def registered(app_config_module):
    artwork = syn.make_artwork()
    photo = syn.make_photo(artwork, photo_size=(1600, 1200), roll_deg=8, tilt_x_deg=10, label_fraction=0.75)
    alignment = align_label(photo.image, app_config_module.alignment, artwork)
    result = register_to_artwork(photo.image, artwork, alignment, app_config_module.registration)
    return photo, result


@pytest.fixture(scope="module")
def app_config_module():
    from print_inspector.config import load_config

    return load_config(user_path=None)


@pytest.fixture
def dialog(qtbot, registered, app_config_module):
    photo, result = registered
    dlg = RegistrationAdjustDialog(photo.image, result.reference, result.homography, app_config_module.ui)
    qtbot.addWidget(dlg)
    dlg.show()
    return dlg


def test_starts_without_adjustment(dialog, registered):
    _, result = registered
    assert dialog.adjustment().is_identity()
    assert np.allclose(dialog.adjusted_homography(), result.homography)
    assert dialog.refine()
    preview = dialog.preview_image()
    assert preview is not None and preview.shape == result.reference.shape


def test_preview_without_adjustment_matches_registration(dialog, registered):
    _, result = registered
    preview = dialog.preview_image().astype(int)
    expected = result.registered.astype(int)
    inside = result.mask > 0
    # Linear preview vs. cubic final resampling: nearly identical where comparable.
    assert np.mean(np.abs(preview - expected)[inside]) < 3.0


def test_spin_boxes_build_the_adjustment(dialog):
    dialog.dx.setValue(1.5)
    dialog.dy.setValue(-0.75)
    dialog.rotation.setValue(0.1)
    dialog.scale_x.setValue(100.2)
    adjustment = dialog.adjustment()
    assert adjustment == ManualAdjustment(dx=1.5, dy=-0.75, rotation_deg=0.1, scale_x=1.002, scale_y=1.0)


def test_adjustment_moves_preview(dialog):
    before = dialog.preview_image().copy()
    dialog.dx.setValue(3.0)
    after = dialog.preview_image()
    # Content shifted 3 px right: compare a central strip.
    h, w = before.shape[:2]
    assert np.mean(np.abs(after[:, 103 : w - 100].astype(int) - before[:, 100 : w - 103].astype(int))) < 4.0


def test_arrow_keys_nudge(dialog, qtbot):
    QTest.keyClick(dialog, Qt.Key.Key_Right)
    QTest.keyClick(dialog, Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier)
    adjustment = dialog.adjustment()
    assert adjustment.dx == pytest.approx(NUDGE_PX)
    assert adjustment.dy == pytest.approx(1.0)


def test_reset(dialog):
    dialog.set_adjustment(ManualAdjustment(dx=4, rotation_deg=1))
    dialog.reset_button.click()
    assert dialog.adjustment().is_identity()


def test_overlay_controls(dialog):
    for index in range(dialog.mode_combo.count()):
        dialog.mode_combo.setCurrentIndex(index)
        dialog.opacity.setValue(30)
    assert dialog.view.has_image()
