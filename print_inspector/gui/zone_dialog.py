"""Drawing and editing zones on the artwork (base of the text and barcode zone dialogs)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QMouseEvent, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGraphicsItem,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from print_inspector.config import UIConfig
from print_inspector.gui.image_panel import ImageView
from print_inspector.gui.workers import bgr8_to_qimage

ZONE_COLOUR = "#f97316"
MIN_ZONE_PX = 4
COL_NAME = 0


@dataclass(frozen=True)
class Column:
    header: str
    kind: str  # "text", "check" or "choice"
    choices: tuple[tuple[str, str], ...] = ()  # (label, value) for "choice"
    default: object = ""
    stretch: bool = False


class ZoneView(ImageView):
    """Image view on which a rectangle can be dragged (in draw mode)."""

    def __init__(self, ui_config: UIConfig, on_drawn, parent: QWidget | None = None) -> None:
        super().__init__(ui_config, parent)
        self._on_drawn = on_drawn
        self._drawing = False
        self._start: QPointF | None = None
        self._rubber = None

    def set_drawing(self, drawing: bool) -> None:
        self._drawing = drawing
        self.setDragMode(QGraphicsView.DragMode.NoDrag if drawing else QGraphicsView.DragMode.ScrollHandDrag)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor if drawing else Qt.CursorShape.OpenHandCursor)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._drawing and event.button() == Qt.MouseButton.LeftButton and self.has_image():
            self._start = self.mapToScene(event.position().toPoint())
            pen = QPen(QColor(ZONE_COLOUR), 2, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self._rubber = self.scene().addRect(QRectF(self._start, self._start), pen)
            self._rubber.setZValue(5)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._start is not None and self._rubber is not None:
            self._rubber.setRect(QRectF(self._start, self.mapToScene(event.position().toPoint())).normalized())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._start is not None and self._rubber is not None:
            rect = QRectF(self._start, self.mapToScene(event.position().toPoint())).normalized()
            self.scene().removeItem(self._rubber)
            self._start, self._rubber = None, None
            self._on_drawn(rect)
            return
        super().mouseReleaseEvent(event)


class ZoneEditorDialog(QDialog):
    """Rectangles on the artwork with a row of values each; subclasses turn rows into zones.

    Column 0 is always the zone name. Rectangles are in artwork pixels.
    """

    columns: tuple[Column, ...] = ()
    hint_text = ""
    item_name = "Zone"  # default names are "<item_name> <n>"

    def __init__(self, title: str, artwork: np.ndarray, ui_config: UIConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._artwork = np.ascontiguousarray(artwork)
        self._height, self._width = artwork.shape[:2]
        self._rects: list[QRectF] = []
        self._items: list[QGraphicsItem] = []

        self.view = ZoneView(ui_config, self.add_zone_rect)
        self.view.set_image(bgr8_to_qimage(self._artwork))
        self.table = QTableWidget(0, len(self.columns))
        self.table.setHorizontalHeaderLabels([c.header for c in self.columns])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        for index, column in enumerate(self.columns):
            if column.stretch:
                self.table.horizontalHeader().setSectionResizeMode(index, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._redraw)
        self.draw_button = QPushButton("Draw Zone")
        self.draw_button.setCheckable(True)
        self.draw_button.setToolTip("Drag a rectangle on the artwork.")
        self.draw_button.toggled.connect(self.view.set_drawing)
        self.delete_button = QPushButton("Delete Zone")
        self.delete_button.clicked.connect(self.delete_selected)
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #b91c1c;")
        self.error_label.setWordWrap(True)
        hint = QLabel(self.hint_text)
        hint.setWordWrap(True)
        hint.setObjectName("sideNote")

        buttons = QHBoxLayout()
        buttons.addWidget(self.draw_button)
        buttons.addWidget(self.delete_button)
        buttons.addStretch(1)
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(self.table, 1)
        side_layout.addLayout(buttons)
        side_layout.addWidget(hint)
        side_layout.addWidget(self.error_label)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.view)
        split.addWidget(side)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(split, 1)
        layout.addWidget(box)
        width = self.fontMetrics().averageCharWidth() * 150
        self.resize(width, int(width * 0.6))

    # ---------------------------------------------------------------- rows

    def add_row(self, rect: QRectF, values: list) -> None:
        """A zone at ``rect`` (artwork pixels) with one value per column."""
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, (column, value) in enumerate(zip(self.columns, values)):
            if column.kind == "check":
                item = QTableWidgetItem()
                item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                item.setCheckState(Qt.CheckState.Checked if value else Qt.CheckState.Unchecked)
                self.table.setItem(row, col, item)
            elif column.kind == "choice":
                combo = QComboBox()
                for label, data in column.choices:
                    combo.addItem(label, data)
                combo.setCurrentIndex(max(0, combo.findData(value)))
                self.table.setCellWidget(row, col, combo)
            else:
                self.table.setItem(row, col, QTableWidgetItem(str(value)))
        self._rects.append(rect)
        self._redraw()

    def row_values(self, row: int) -> list:
        values = []
        for col, column in enumerate(self.columns):
            if column.kind == "check":
                values.append(self.table.item(row, col).checkState() == Qt.CheckState.Checked)
            elif column.kind == "choice":
                values.append(self.table.cellWidget(row, col).currentData())
            else:
                values.append((self.table.item(row, col).text() or "").strip())
        return values

    def rows(self) -> list[tuple[tuple[float, float, float, float], list]]:
        """(x, y, width, height as fractions of the artwork, values) per zone."""
        out = []
        for row, rect in enumerate(self._rects):
            values = self.row_values(row)
            values[COL_NAME] = values[COL_NAME] or f"{self.item_name} {row + 1}"
            fractions = (rect.x() / self._width, rect.y() / self._height,
                         rect.width() / self._width, rect.height() / self._height)
            out.append((fractions, values))
        return out

    def rect_of(self, x: float, y: float, w: float, h: float) -> QRectF:
        return QRectF(round(x * self._width), round(y * self._height), round(w * self._width), round(h * self._height))

    def add_zone_rect(self, rect: QRectF) -> None:
        """Add a zone for ``rect`` (artwork pixels), clipped to the artwork."""
        rect = rect.intersected(QRectF(0, 0, self._width, self._height))
        if rect.width() < MIN_ZONE_PX or rect.height() < MIN_ZONE_PX:
            return
        values = [c.default for c in self.columns]
        values[COL_NAME] = f"{self.item_name} {len(self._rects) + 1}"
        self.add_row(rect, values)
        self.table.selectRow(self.table.rowCount() - 1)
        self.table.editItem(self.table.item(self.table.rowCount() - 1, COL_NAME))

    def delete_selected(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self._rects):
            self.table.removeRow(row)
            del self._rects[row]
            self._redraw()

    def _redraw(self) -> None:
        scene = self.view.scene()
        for item in self._items:
            scene.removeItem(item)
        self._items = []
        selected = self.table.currentRow()
        for row, rect in enumerate(self._rects):
            colour = QColor("#facc15" if row == selected else ZONE_COLOUR)
            pen = QPen(colour, 2)
            pen.setCosmetic(True)
            fill = QColor(colour)
            fill.setAlpha(40)
            item = scene.addRect(rect, pen, QBrush(fill))
            item.setZValue(3)
            name = self.table.item(row, COL_NAME)
            label = scene.addSimpleText(name.text() if name else "")
            label.setBrush(QBrush(colour))
            label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            label.setPos(rect.topLeft() + QPointF(0, -16 / max(self.view.zoom, 1e-3)))
            label.setZValue(3)
            self._items += [item, label]

    def validate(self) -> str | None:
        """An error message, or None if the zones can be used (subclasses add checks)."""
        names = [values[COL_NAME] for _, values in self.rows()]
        if len(set(names)) != len(names):
            return f"Each {self.item_name.lower()} needs its own name."
        return None

    def accept(self) -> None:  # noqa: D102 - validates first
        message = self.validate()
        if message:
            self.error_label.setText(message)
            return
        super().accept()
