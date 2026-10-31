"""Zoomable image viewer and the titled panel that hosts it."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from PySide6.QtCore import QMimeData, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QImage,
    QMouseEvent,
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from print_inspector.config import UIConfig
from print_inspector.gui.elided_label import ElidedLabel


CLICK_TOLERANCE_PX = 4  # a press and release closer than this is a click, not a drag


class ImageView(QGraphicsView):
    """Shows one image at full resolution with zoom, fit and drag-to-pan.

    The pixmap is never downscaled: zooming changes only the view transform,
    so the full detail is available when zoomed in. Above 100 % pixels are
    drawn without smoothing so individual print defects stay crisp.

    ``view_changed`` is emitted when the user (or a navigation command such as
    ``show_area``) zooms or pans, so other views can follow; changes made by
    fitting, loading or :meth:`apply_view_state` are silent.
    """

    zoom_changed = Signal(float)
    view_changed = Signal()
    image_clicked = Signal(QPointF)  # scene (image pixel) coordinates

    def __init__(self, ui_config: UIConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ui = ui_config
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item: QGraphicsPixmapItem | None = None
        self._region_item: QGraphicsRectItem | None = None
        self._polygon_items: list[QGraphicsItem] = []
        self._box_items: list[QGraphicsItem] = []
        self._polygon_points: list[tuple[float, float]] | None = None
        self._zoom = 1.0
        self._fit_mode = True
        self._quiet = 0
        self._press_pos: QPointF | None = None
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(QColor("#3a3d41"))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.horizontalScrollBar().valueChanged.connect(self._on_scrolled)
        self.verticalScrollBar().valueChanged.connect(self._on_scrolled)

    @contextmanager
    def _silently(self):
        self._quiet += 1
        try:
            yield
        finally:
            self._quiet -= 1

    def _on_scrolled(self, _value: int) -> None:
        if not self._quiet and self._item is not None:
            self.view_changed.emit()

    @property
    def zoom(self) -> float:
        return self._zoom

    @property
    def is_fit_mode(self) -> bool:
        return self._fit_mode

    def has_image(self) -> bool:
        return self._item is not None

    def image_size(self) -> QSize:
        return self._item.pixmap().size() if self._item is not None else QSize()

    def set_image(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():
            raise ValueError("The image could not be converted for display.")
        self._scene.clear()
        self._region_item = None
        self._polygon_items = []
        self._box_items = []
        self._polygon_points = None
        with self._silently():
            self._item = self._scene.addPixmap(pixmap)
            self._scene.setSceneRect(self._item.boundingRect())
            self.fit_to_window()

    def update_image(self, image: QImage) -> None:
        """Replace the pixels but keep zoom and scroll position when the size is unchanged."""
        if self._item is None or self.image_size() != image.size():
            self.set_image(image)
            return
        pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():
            raise ValueError("The image could not be converted for display.")
        self._item.setPixmap(pixmap)

    def show_area(self, rect: QRectF, padding: float = 3.0, min_zoom: float = 1.0) -> None:
        """Zoom so that ``rect`` (image pixels), enlarged by ``padding`` x its size, fills the view."""
        if self._item is None:
            return
        w = max(rect.width(), 8.0) * padding
        h = max(rect.height(), 8.0) * padding
        viewport = self.viewport().rect()
        zoom = min(viewport.width() / w, viewport.height() / h)
        with self._silently():
            self.set_zoom(max(zoom, min_zoom))
            self.centerOn(rect.center())
        self.view_changed.emit()

    def view_state(self) -> tuple[QPointF, float]:
        """(scene point at the centre of the viewport, zoom)."""
        return self.mapToScene(self.viewport().rect().center()), self._zoom

    def apply_view_state(self, centre: QPointF, zoom: float) -> None:
        """Show ``centre`` at ``zoom`` without emitting ``view_changed``."""
        if self._item is None:
            return
        with self._silently():
            self.set_zoom(zoom)
            self.centerOn(centre)
    def set_region(self, rect: QRectF | None) -> None:
        """Outline a region (image pixel coordinates) over the image, or remove it."""
        if self._region_item is not None:
            self._scene.removeItem(self._region_item)
            self._region_item = None
        if rect is None or self._item is None:
            return
        pen = QPen(QColor("#22d3ee"), 2, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)  # constant on-screen width at any zoom
        self._region_item = self._scene.addRect(rect, pen)
        self._region_item.setZValue(1)

    def region(self) -> QRectF | None:
        return self._region_item.rect() if self._region_item is not None else None

    def set_polygon(self, points, colour: str = "#22c55e") -> None:
        """Outline a polygon given in image pixel coordinates (pixel centres at integers).

        Corner markers keep a constant on-screen size; the first corner is drawn
        larger to show the label's top-left. ``None`` removes the outline.
        """
        for item in self._polygon_items:
            self._scene.removeItem(item)
        self._polygon_items = []
        self._polygon_points = None
        if points is None or self._item is None:
            return
        pts = [(float(x), float(y)) for x, y in points]
        self._polygon_points = pts
        # Scene units are image pixels with pixel (0, 0) covering [0, 1).
        scene_pts = [QPointF(x + 0.5, y + 0.5) for x, y in pts]
        pen = QPen(QColor(colour), 2)
        pen.setCosmetic(True)
        self._polygon_items.append(self._scene.addPolygon(QPolygonF(scene_pts), pen))
        for index, p in enumerate(scene_pts):
            radius = 7 if index == 0 else 4
            marker = self._scene.addEllipse(-radius, -radius, 2 * radius, 2 * radius, pen, QBrush(QColor(colour)))
            marker.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            marker.setPos(p)
            self._polygon_items.append(marker)
        for item in self._polygon_items:
            item.setZValue(2)

    def set_boxes(self, boxes: list[tuple[QRectF, str, str]] | None) -> None:
        """Draw labelled rectangles (image pixel coordinates, label, colour); None removes them."""
        for item in self._box_items:
            self._scene.removeItem(item)
        self._box_items = []
        if not boxes or self._item is None:
            return
        for rect, label, colour in boxes:
            pen = QPen(QColor(colour), 2)
            pen.setCosmetic(True)
            box = self._scene.addRect(rect.adjusted(-3, -3, 3, 3), pen)
            box.setZValue(3)
            text = self._scene.addSimpleText(label)
            text.setBrush(QBrush(QColor(colour)))
            text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            text.setPos(rect.left() - 3, rect.top() - 3)
            text.setZValue(3)
            self._box_items.extend([box, text])

    def boxes(self) -> int:
        return len(self._box_items) // 2

    def polygon(self) -> list[tuple[float, float]] | None:
        return self._polygon_points

    def clear(self) -> None:
        with self._silently():
            self._scene.clear()
            self._item = None
            self._region_item = None
            self._polygon_items = []
            self._box_items = []
            self._polygon_points = None
            self.resetTransform()
            self._zoom = 1.0
            self._fit_mode = True

    def fit_to_window(self) -> None:
        if self._item is None:
            return
        self._fit_mode = True
        if self._item.boundingRect().isEmpty() or self.viewport().rect().isEmpty():
            return
        with self._silently():
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = self.transform().m11()
            self._after_zoom()

    @contextmanager
    def _anchored_at_centre(self):
        anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        try:
            yield
        finally:
            self.setTransformationAnchor(anchor)

    def zoom_in(self) -> None:
        with self._anchored_at_centre():
            self.set_zoom(self._zoom * self._ui.zoom_step)

    def zoom_out(self) -> None:
        with self._anchored_at_centre():
            self.set_zoom(self._zoom / self._ui.zoom_step)

    def actual_size(self) -> None:
        """100 %: one image pixel per screen pixel."""
        with self._anchored_at_centre():
            self.set_zoom(1.0)

    def set_zoom(self, factor: float) -> None:
        if self._item is None:
            return
        factor = min(max(factor, self._ui.min_zoom), self._ui.max_zoom)
        self._fit_mode = False
        self.setTransform(QTransform.fromScale(factor, factor))
        self._zoom = factor
        self._after_zoom()

    def _after_zoom(self) -> None:
        if self._item is not None:
            mode = (
                Qt.TransformationMode.SmoothTransformation
                if self._zoom < 1.0
                else Qt.TransformationMode.FastTransformation
            )
            self._item.setTransformationMode(mode)
        self.zoom_changed.emit(self._zoom)
        if not self._quiet:
            self.view_changed.emit()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt override
        """The wheel zooms around the mouse pointer; a sideways scroll (or Shift + wheel) pans."""
        steps = event.angleDelta().y() / 120.0
        if self._item is not None and steps and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.set_zoom(self._zoom * self._ui.zoom_step**steps)
            event.accept()
            return
        super().wheelEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        with self._silently():
            super().resizeEvent(event)
            if self._fit_mode:
                self.fit_to_window()

    def _is_click_target(self, event: QMouseEvent) -> bool:
        """Whether a click at ``event`` should be reported (subclasses exclude their own items)."""
        return True

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        self._press_pos = event.position() if event.button() == Qt.MouseButton.LeftButton else None
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        press, self._press_pos = self._press_pos, None
        target = self._is_click_target(event)
        super().mouseReleaseEvent(event)
        if press is None or not target or event.button() != Qt.MouseButton.LeftButton or self._item is None:
            return
        if (event.position() - press).manhattanLength() <= CLICK_TOLERANCE_PX:
            self.image_clicked.emit(self.mapToScene(event.position().toPoint()))


def image_paths_from_mime(mime: QMimeData) -> list[Path]:
    """Local files in a drag-and-drop payload."""
    if not mime.hasUrls():
        return []
    return [Path(url.toLocalFile()) for url in mime.urls() if url.isLocalFile()]


class ImagePanel(QFrame):
    """A titled panel showing either a placeholder message or an :class:`ImageView`.

    With ``accept_drops`` the panel takes an image file dragged onto it and
    emits ``file_dropped`` (loading it is the window's business).
    """

    file_dropped = Signal(Path)

    def __init__(
        self,
        title: str,
        placeholder: str,
        ui_config: UIConfig,
        parent: QWidget | None = None,
        accept_drops: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("imagePanel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._placeholder_text = placeholder

        self.title_label = QLabel(title)
        self.title_label.setObjectName("panelTitle")
        self.zoom_label = QLabel("")
        self.zoom_label.setObjectName("panelZoom")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.zoom_label.setMinimumWidth(self.zoom_label.fontMetrics().horizontalAdvance("6400 %") + 4)
        self.title_label.setMinimumWidth(1)  # may be clipped on very narrow screens rather than widen the window
        header = QHBoxLayout()
        header.addWidget(self.title_label, 1)
        header.addWidget(self.zoom_label)
        self._controls = QHBoxLayout()  # view switches etc. on their own row, so narrow panels still fit
        self._controls.setContentsMargins(0, 0, 0, 0)

        self.info_label = QLabel("")
        self.info_label.setObjectName("panelInfo")
        self.info_label.setWordWrap(True)
        self.warning_label = ElidedLabel()
        self.warning_label.setObjectName("panelWarning")
        self.warning_label.hide()

        self.placeholder_label = QLabel(placeholder)
        self.placeholder_label.setObjectName("panelPlaceholder")
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setWordWrap(True)
        self.view = ImageView(ui_config)
        self.view.zoom_changed.connect(self._on_zoom_changed)
        if accept_drops:
            # The panel handles drops itself; the view would swallow them.
            self.view.setAcceptDrops(False)
            self.view.viewport().setAcceptDrops(False)
            self.setAcceptDrops(True)

        self._stack = QStackedWidget()
        self._stack.addWidget(self.placeholder_label)
        self._stack.addWidget(self.view)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        layout.addLayout(header)
        layout.addLayout(self._controls)
        layout.addWidget(self.info_label)
        layout.addWidget(self.warning_label)
        layout.addWidget(self._stack, 1)

    def add_header_widget(self, widget: QWidget, stretch: int = 0) -> None:
        """Place a control (e.g. a view switch) in the row below the title."""
        self._controls.addWidget(widget, stretch)

    def set_image(self, image: QImage, info: str, warning: str | None = None) -> None:
        self._stack.setCurrentWidget(self.view)
        self.view.set_image(image)
        self.set_info(info, warning)

    def update_image(self, image: QImage) -> None:
        """Swap the displayed pixels, keeping zoom and position (e.g. while adjusting an overlay)."""
        self._stack.setCurrentWidget(self.view)
        self.view.update_image(image)

    def set_info(self, info: str, warning: str | None = None) -> None:
        self.info_label.setText(info)
        self.set_warning(warning)

    def set_warning(self, warning: str | None, details: str = "") -> None:
        """Show a warning above the image; long ``warning`` text is shortened, ``details`` go in the tooltip."""
        self.warning_label.set_full_text(warning or "")
        self.warning_label.setToolTip(details or warning or "")
        self.warning_label.setVisible(bool(warning))

    def clear(self) -> None:
        self.view.clear()
        self._stack.setCurrentWidget(self.placeholder_label)
        self.placeholder_label.setText(self._placeholder_text)
        self.zoom_label.setText("")
        self.set_info("")

    def _on_zoom_changed(self, zoom: float) -> None:
        self.zoom_label.setText(f"{zoom * 100:.0f} %")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt override
        if len(image_paths_from_mime(event.mimeData())) == 1:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt override
        paths = image_paths_from_mime(event.mimeData())
        if len(paths) != 1:
            event.ignore()
            return
        event.acceptProposedAction()
        self.file_dropped.emit(paths[0])
