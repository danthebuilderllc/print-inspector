"""Image viewer interaction: wheel zoom, clicks, view-change signals, drag and drop, synchronisation."""
import numpy as np
import pytest
from PySide6.QtCore import QMimeData, QPoint, QPointF, QRectF, Qt, QUrl
from PySide6.QtGui import QDropEvent, QWheelEvent
from PySide6.QtTest import QTest

from print_inspector.gui.image_panel import ImagePanel, ImageView, image_paths_from_mime
from print_inspector.gui.view_sync import ViewSynchronizer
from print_inspector.gui.workers import bgr8_to_qimage


@pytest.fixture
def make_view(qtbot, app_config):
    keep = []

    def make(width=800, height=600, cls=ImageView):
        image = np.full((height, width, 3), 200, np.uint8)
        keep.append(image)
        view = cls(app_config.ui)
        view.resize(400, 300)
        qtbot.addWidget(view)
        view.show()
        view.set_image(bgr8_to_qimage(image))
        return view

    return make


def _wheel(view: ImageView, delta_y: int, modifiers=Qt.KeyboardModifier.NoModifier) -> None:
    pos = QPointF(view.viewport().rect().center())
    event = QWheelEvent(
        pos, view.viewport().mapToGlobal(pos), QPoint(0, 0), QPoint(0, delta_y),
        Qt.MouseButton.NoButton, modifiers, Qt.ScrollPhase.NoScrollPhase, False,
    )
    view.wheelEvent(event)


def test_plain_mouse_wheel_zooms(make_view, app_config):
    view = make_view()
    start = view.zoom
    _wheel(view, 120)
    assert view.zoom == pytest.approx(start * app_config.ui.zoom_step)
    _wheel(view, -240)
    assert view.zoom == pytest.approx(start / app_config.ui.zoom_step)
    assert not view.is_fit_mode


def test_shift_wheel_does_not_zoom(make_view):
    view = make_view()
    view.set_zoom(2.0)
    _wheel(view, 120, Qt.KeyboardModifier.ShiftModifier)
    assert view.zoom == pytest.approx(2.0)


def test_view_changed_only_for_navigation(make_view):
    view = make_view()
    events = []
    view.view_changed.connect(lambda: events.append(1))
    view.fit_to_window()
    view.apply_view_state(QPointF(100, 100), 2.0)
    assert events == []  # silent changes
    view.set_zoom(3.0)
    assert events
    events.clear()
    view.horizontalScrollBar().setValue(view.horizontalScrollBar().value() + 25)
    assert events
    events.clear()
    view.show_area(QRectF(400, 300, 10, 10))
    assert events


def test_apply_view_state_centres_and_zooms(make_view):
    view = make_view()
    view.apply_view_state(QPointF(500, 200), 2.0)
    centre, zoom = view.view_state()
    assert zoom == pytest.approx(2.0)
    assert centre.x() == pytest.approx(500, abs=1.5) and centre.y() == pytest.approx(200, abs=1.5)


def test_click_reports_scene_position_but_drag_does_not(make_view):
    view = make_view()
    view.apply_view_state(QPointF(400, 300), 1.0)
    clicks = []
    view.image_clicked.connect(clicks.append)
    target = view.mapFromScene(QPointF(420, 310))
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, target)
    assert len(clicks) == 1
    assert clicks[0].x() == pytest.approx(420, abs=1.5) and clicks[0].y() == pytest.approx(310, abs=1.5)
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, target)
    QTest.mouseMove(view.viewport(), target + QPoint(40, 30))
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, target + QPoint(40, 30))
    assert len(clicks) == 1


def test_zoom_commands_keep_the_centre(make_view):
    view = make_view()
    view.apply_view_state(QPointF(300, 250), 1.0)
    view.zoom_in()
    centre, _ = view.view_state()
    assert centre.x() == pytest.approx(300, abs=2) and centre.y() == pytest.approx(250, abs=2)
    view.actual_size()
    assert view.zoom == pytest.approx(1.0)


# ---------------------------------------------------------------- synchronisation


def test_synchronizer_maps_between_resolutions(make_view):
    artwork = make_view(800, 600)
    grid = make_view(1600, 1200)  # comparison grid at 2x the artwork
    sync = ViewSynchronizer()
    sync.add_view(artwork)
    sync.add_view(grid)
    sync.set_scale(artwork, (1.0, 1.0))
    sync.set_scale(grid, (2.0, 2.0))

    grid.apply_view_state(QPointF(1000, 700), 4.0)
    grid.view_changed.emit()  # as after a user zoom/pan
    centre, zoom = artwork.view_state()
    assert zoom == pytest.approx(8.0)  # same on-screen size of the label
    assert centre.x() == pytest.approx(500, abs=1) and centre.y() == pytest.approx(350, abs=1)
    assert sync.leader() is grid

    artwork.set_zoom(2.0)  # user zooms the other view: the grid follows
    assert grid.zoom == pytest.approx(1.0)


def test_synchronizer_disabled_and_unscaled_views_are_left_alone(make_view):
    a, b, photo = make_view(), make_view(), make_view()
    sync = ViewSynchronizer()
    for view in (a, b, photo):
        sync.add_view(view)
    sync.set_scale(a, (1.0, 1.0))
    sync.set_scale(b, (1.0, 1.0))  # photo: no fixed relation to the artwork
    photo_zoom = photo.zoom
    a.set_zoom(3.0)
    assert b.zoom == pytest.approx(3.0)
    assert photo.zoom == pytest.approx(photo_zoom)
    assert sync.synced_views() == [a, b]

    sync.set_enabled(False)
    a.set_zoom(5.0)
    assert b.zoom == pytest.approx(3.0)
    sync.set_enabled(True)  # re-enabling aligns the others with the last navigated view
    assert b.zoom == pytest.approx(5.0)


# ---------------------------------------------------------------- drag and drop


def _drop(panel: ImagePanel, paths) -> None:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    event = QDropEvent(
        QPointF(10, 10), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    )
    panel.dropEvent(event)


def test_panel_drop_emits_single_file(qtbot, app_config, tmp_path):
    panel = ImagePanel("t", "drop here", app_config.ui, accept_drops=True)
    qtbot.addWidget(panel)
    assert panel.acceptDrops() and not panel.view.viewport().acceptDrops()
    dropped = []
    panel.file_dropped.connect(dropped.append)
    _drop(panel, [tmp_path / "a.png"])
    assert dropped == [tmp_path / "a.png"]
    _drop(panel, [tmp_path / "a.png", tmp_path / "b.png"])  # ambiguous: ignored
    assert len(dropped) == 1


def test_mime_paths_ignore_remote_urls():
    mime = QMimeData()
    mime.setUrls([QUrl("https://example.com/a.png")])
    assert image_paths_from_mime(mime) == []
