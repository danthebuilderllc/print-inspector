import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialogButtonBox

from print_inspector import synthetic as syn
from print_inspector.core.label_detection import snap_corners_to_edges
from print_inspector.gui.corner_dialog import CornerSelectionDialog
from print_inspector.gui.workers import bgr8_to_qimage


@pytest.fixture(scope="module")
def photo():
    return syn.make_photo(syn.make_artwork(), photo_size=(1600, 1200), roll_deg=10, tilt_x_deg=12)


@pytest.fixture
def make_dialog(qtbot, app_config, photo):
    def factory(**kwargs):
        image = np.ascontiguousarray(photo.image)
        dialog = CornerSelectionDialog(
            bgr8_to_qimage(image), (1600, 1200), app_config.ui,
            snap=lambda c: snap_corners_to_edges(image, c, app_config.alignment), **kwargs,
        )
        dialog._keep_alive = image  # the QImage shares this buffer
        qtbot.addWidget(dialog)
        dialog.show()
        return dialog
    return factory


def _ok_enabled(dialog):
    return dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()


def test_empty_dialog_requires_four_corners(make_dialog):
    dialog = make_dialog()
    assert dialog.corners() is None
    assert not _ok_enabled(dialog)
    assert "0 of 4" in dialog.instructions.text()


def test_corners_in_any_order_are_ordered(make_dialog, photo):
    dialog = make_dialog()
    for index in (2, 0, 3, 1):
        dialog.add_corner(*photo.corners[index])
    assert _ok_enabled(dialog)
    assert np.allclose(dialog.corners(), photo.corners, atol=1e-6)
    dialog.add_corner(5, 5)  # a fifth click is ignored
    assert len(dialog.corners()) == 4


def test_drag_correction(make_dialog, photo):
    dialog = make_dialog(initial_corners=photo.corners)
    dialog.move_corner(0, photo.corners[0, 0] + 15, photo.corners[0, 1] - 10)
    moved = dialog.corners()
    assert np.allclose(moved[0], photo.corners[0] + [15, -10], atol=1e-6)
    assert np.allclose(moved[1:], photo.corners[1:], atol=1e-6)


def test_snap_to_edges(make_dialog, photo):
    rough = photo.corners + np.array([[7, -5], [-6, 6], [5, 7], [-6, -5]])
    dialog = make_dialog(initial_corners=rough)
    dialog.snap_to_edges()
    assert np.max(np.linalg.norm(dialog.corners() - photo.corners, axis=1)) < 1.0
    assert "snapped" in dialog.message_label.text()


def test_use_automatic_and_clear(make_dialog, photo):
    dialog = make_dialog(automatic_corners=photo.corners)
    assert dialog.automatic_button.isEnabled()
    dialog.use_automatic()
    assert np.allclose(dialog.corners(), photo.corners)
    dialog.clear_corners()
    assert dialog.corners() is None and not _ok_enabled(dialog)


def test_corners_clamped_to_image(make_dialog):
    dialog = make_dialog()
    dialog.add_corner(-50, -50)
    dialog.add_corner(5000, 10)
    dialog.add_corner(5000, 5000)
    dialog.add_corner(10, 5000)
    corners = dialog.corners()
    assert corners.min() >= -0.5 and corners[:, 0].max() <= 1599.5 and corners[:, 1].max() <= 1199.5


def test_mouse_clicks_place_corners(qtbot, make_dialog):
    dialog = make_dialog()
    view = dialog.view
    view.fit_to_window()
    targets = [(200.0, 150.0), (1400.0, 150.0), (1400.0, 1050.0), (200.0, 1050.0)]
    for x, y in targets:
        pos = view.mapFromScene(x + 0.5, y + 0.5)
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(pos))
    corners = dialog.corners()
    assert corners is not None
    tolerance = 1.5 / view.zoom  # one screen pixel, in image pixels
    assert np.max(np.abs(corners - np.array(targets))) <= tolerance
