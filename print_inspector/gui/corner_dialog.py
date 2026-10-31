"""Dialog for marking the label's four corners by hand."""
from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QMouseEvent, QPen, QPolygonF
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPolygonItem,
    QGraphicsSimpleTextItem,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from print_inspector.config import UIConfig
from print_inspector.core.label_detection import order_corners
from print_inspector.gui.image_panel import ImageView

HANDLE_RADIUS = 8
_COLOUR = QColor("#f97316")


def _to_scene(x: float, y: float) -> QPointF:
    """Image pixel coordinates (pixel centres at integers) to scene coordinates."""
    return QPointF(x + 0.5, y + 0.5)


def _from_scene(p: QPointF) -> tuple[float, float]:
    return p.x() - 0.5, p.y() - 0.5


class CornerHandle(QGraphicsEllipseItem):
    """A draggable corner marker with a constant on-screen size."""

    def __init__(self, on_moved: Callable[[], None]) -> None:
        super().__init__(-HANDLE_RADIUS, -HANDLE_RADIUS, 2 * HANDLE_RADIUS, 2 * HANDLE_RADIUS)
        self._on_moved = on_moved
        pen = QPen(QColor("white"), 2)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(_COLOUR))
        self.setZValue(10)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )

    def itemChange(self, change, value):  # noqa: N802 - Qt override
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._on_moved()
        return super().itemChange(change, value)


class CornerView(ImageView):
    """Image view that reports clicks on the image (not drags, not clicks on handles)."""

    def _is_click_target(self, event: QMouseEvent) -> bool:
        return not isinstance(self.itemAt(event.position().toPoint()), CornerHandle)


class CornerSelectionDialog(QDialog):
    """Place four corner handles on the photo and drag them to correct their position.

    ``initial_corners`` (e.g. from automatic detection) pre-places the handles.
    ``snap`` refines corners to nearby label edges; it returns None if no edge is found.
    """

    corners_changed = Signal()

    def __init__(
        self,
        photo: QImage,
        image_size: tuple[int, int],
        ui_config: UIConfig,
        initial_corners: np.ndarray | None = None,
        automatic_corners: np.ndarray | None = None,
        snap: Callable[[np.ndarray], np.ndarray | None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manual Label Corners")
        self.resize(1100, 800)
        self._image_size = image_size
        self._automatic = automatic_corners
        self._snap = snap
        self._handles: list[CornerHandle] = []
        self._labels: list[QGraphicsSimpleTextItem] = []

        self.view = CornerView(ui_config)
        self.view.set_image(photo)
        self.view.image_clicked.connect(self._on_image_clicked)
        self._outline = QGraphicsPolygonItem()
        pen = QPen(_COLOUR, 2)
        pen.setCosmetic(True)
        self._outline.setPen(pen)
        self._outline.setZValue(5)
        self.view.scene().addItem(self._outline)

        self.instructions = QLabel()
        self.instructions.setWordWrap(True)
        self.message_label = QLabel("")
        self.message_label.setStyleSheet("color: #92400e;")

        self.clear_button = QPushButton("Clear Corners")
        self.clear_button.clicked.connect(self.clear_corners)
        self.automatic_button = QPushButton("Use Automatic Detection")
        self.automatic_button.setEnabled(automatic_corners is not None)
        self.automatic_button.clicked.connect(self.use_automatic)
        self.snap_button = QPushButton("Snap to Label Edges")
        self.snap_button.setToolTip("Move the corners onto the nearest straight label edges.")
        self.snap_button.clicked.connect(self.snap_to_edges)
        self.fit_button = QPushButton("Fit to Window")
        self.fit_button.clicked.connect(self.view.fit_to_window)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply Corners")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        tools = QHBoxLayout()
        for button in (self.clear_button, self.automatic_button, self.snap_button, self.fit_button):
            tools.addWidget(button)
        tools.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self.instructions)
        layout.addLayout(tools)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.message_label)
        layout.addWidget(self.buttons)

        if initial_corners is not None:
            self.set_corners(initial_corners)
        self._update_state()

    # ---------------------------------------------------------------- corners

    def corners(self) -> np.ndarray | None:
        """The four corners in image pixel coordinates, ordered TL, TR, BR, BL, or None."""
        if len(self._handles) != 4:
            return None
        return order_corners(np.array([_from_scene(h.pos()) for h in self._handles]))

    def set_corners(self, corners: np.ndarray) -> None:
        self.clear_corners()
        for x, y in np.asarray(corners, np.float64).reshape(4, 2):
            self.add_corner(x, y)

    def add_corner(self, x: float, y: float) -> None:
        if len(self._handles) >= 4:
            return
        w, h = self._image_size
        x = min(max(x, -0.5), w - 0.5)
        y = min(max(y, -0.5), h - 0.5)
        handle = CornerHandle(self._on_handle_moved)
        handle.setPos(_to_scene(x, y))
        self.view.scene().addItem(handle)
        self._handles.append(handle)
        self._on_handle_moved()

    def move_corner(self, index: int, x: float, y: float) -> None:
        self._handles[index].setPos(_to_scene(x, y))

    def clear_corners(self) -> None:
        for item in self._handles + self._labels:
            self.view.scene().removeItem(item)
        self._handles = []
        self._labels = []
        self.message_label.setText("")
        self._on_handle_moved()

    def use_automatic(self) -> None:
        if self._automatic is not None:
            self.set_corners(self._automatic)
            self.message_label.setText("Corners reset to the automatic detection.")

    def snap_to_edges(self) -> None:
        corners = self.corners()
        if corners is None or self._snap is None:
            return
        snapped = self._snap(corners)
        if snapped is None:
            self.message_label.setText(
                "No clear label edge was found near the corners. Place them more precisely and try again."
            )
            return
        self.set_corners(snapped)
        self.message_label.setText("Corners snapped to the label edges. Check them before applying.")

    # ---------------------------------------------------------------- internals

    def _on_image_clicked(self, scene_pos: QPointF) -> None:
        if len(self._handles) < 4:
            self.add_corner(*_from_scene(scene_pos))

    def _on_handle_moved(self) -> None:
        for label in self._labels:
            self.view.scene().removeItem(label)
        self._labels = []
        ordered = self.corners()
        if ordered is not None:
            self._outline.setPolygon(QPolygonF([_to_scene(x, y) for x, y in ordered]))
            for name, (x, y) in zip(("TL", "TR", "BR", "BL"), ordered):
                text = QGraphicsSimpleTextItem(name)
                text.setBrush(QBrush(QColor("white")))
                text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
                text.setPos(_to_scene(x, y))
                text.setZValue(11)
                self.view.scene().addItem(text)
                self._labels.append(text)
        else:
            self._outline.setPolygon(QPolygonF([h.pos() for h in self._handles]))
        self._update_state()
        self.corners_changed.emit()

    def _update_state(self) -> None:
        count = len(self._handles)
        if count < 4:
            self.instructions.setText(
                f"Click the label's four corners in any order ({count} of 4 placed). "
                "The mouse wheel zooms; drag the image to pan."
            )
        else:
            self.instructions.setText(
                "Drag a corner to correct it. Zoom in (mouse wheel) to place corners precisely. "
                "The label's orientation is matched to the artwork automatically."
            )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(count == 4)
        self.snap_button.setEnabled(count == 4 and self._snap is not None)
