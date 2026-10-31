"""Side-panel tab for the text inspection: findings, lines read by OCR, and the selected finding."""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from print_inspector.core.decision import OperatorMark
from print_inspector.core.text_inspection import LineStatus, TextFinding, TextReport
from print_inspector.gui.elided_label import ElidedLabel

FINDING_COLUMNS = ("#", "Problem", "Expected", "Read", "Severity", "Operator", "Detection", "OCR conf.")
LINE_COLUMNS = ("Line", "Status", "Artwork text", "Printed text", "OCR conf.", "Zone")
COL_TEXT_OPERATOR = FINDING_COLUMNS.index("Operator")
CROP_DISPLAY_PX = 260
_STATUS_COLOURS = {
    LineStatus.MATCH: "#14532d",
    LineStatus.MISMATCH: "#b91c1c",
    LineStatus.UNCERTAIN: "#92400e",
    LineStatus.UNREADABLE: "#92400e",
}


def _conf(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def _crop_pixmap(image: np.ndarray, box: tuple[int, int, int, int], margin: int) -> QPixmap:
    """The box with a margin, enlarged without smoothing, with the box outlined."""
    h_img, w_img = image.shape[:2]
    x, y, w, h = box
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(w_img, x + w + margin), min(h_img, y + h + margin)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return QPixmap()
    scale = max(1, CROP_DISPLAY_PX // max(crop.shape[1], 1), 1)
    scale = min(scale, max(1, 120 // max(crop.shape[0], 1)))
    big = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(big, ((x - x0) * scale, (y - y0) * scale), ((x - x0 + w) * scale - 1, (y - y0 + h) * scale - 1),
                  (0, 140, 255), 1)
    big = np.ascontiguousarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
    qimage = QImage(big.data, big.shape[1], big.shape[0], int(big.strides[0]), QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


class TextPanel(QWidget):
    finding_selected = Signal(int)
    line_selected = Signal(int)
    mark_requested = Signal(int, object)  # finding id, OperatorMark
    edit_zones_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._report: TextReport | None = None
        self._selected: int | None = None
        self._images: tuple[np.ndarray, np.ndarray] | None = None

        self.summary = QLabel("Text is inspected after the comparison.")
        self.summary.setWordWrap(True)
        self.summary.setObjectName("sideNote")
        self.zones_button = QPushButton("Edit Text Zones…")
        self.zones_button.setToolTip("Mark critical text zones on the artwork (Ctrl+T).")
        self.zones_button.clicked.connect(self.edit_zones_requested)
        self.findings_table = self._table(FINDING_COLUMNS)
        self.findings_table.itemSelectionChanged.connect(self._on_finding_clicked)
        self.lines_table = self._table(LINE_COLUMNS)
        self.lines_table.itemSelectionChanged.connect(self._on_line_clicked)

        self.title = ElidedLabel()
        self.title.setStyleSheet("font-weight: bold;")
        self.artwork_crop = QLabel()
        self.print_crop = QLabel()
        for label in (self.artwork_crop, self.print_crop):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("background: #e5e7eb;")
            label.setMinimumHeight(40)
        self.notes = QLabel("")
        self.notes.setWordWrap(True)
        self.notes.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.mark_label = QLabel("")
        self.buttons: dict[OperatorMark, QPushButton] = {}
        grid = QGridLayout()
        for index, (mark, text) in enumerate((
            (OperatorMark.CONFIRMED, "Confirm"), (OperatorMark.DISMISSED, "Dismiss"),
            (OperatorMark.NEEDS_REVIEW, "Needs review"), (OperatorMark.UNREVIEWED, "Clear mark"),
        )):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, m=mark: self._request(m))
            grid.addWidget(button, index // 2, index % 2)
            self.buttons[mark] = button

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(self.summary, 1)
        row.addWidget(self.zones_button, 0, Qt.AlignmentFlag.AlignTop)
        top_layout.addLayout(row)
        top_layout.addWidget(QLabel("<b>Text findings</b>"))
        top_layout.addWidget(self.findings_table, 1)
        top_layout.addWidget(QLabel("<b>Text read by OCR</b> (artwork / print)"))
        top_layout.addWidget(self.lines_table, 1)

        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.addWidget(self.title)
        detail_layout.addWidget(QLabel("Original"))
        detail_layout.addWidget(self.artwork_crop)
        detail_layout.addWidget(QLabel("Printed"))
        detail_layout.addWidget(self.print_crop)
        detail_layout.addWidget(self.notes)
        detail_layout.addWidget(self.mark_label)
        detail_layout.addLayout(grid)
        detail_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(detail)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        split = QSplitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        split.addWidget(top)
        split.addWidget(scroll)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(split)
        self.show_finding(None, OperatorMark.UNREVIEWED)

    @staticmethod
    def _table(columns) -> QTableWidget:
        table = QTableWidget(0, len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        table.setMinimumHeight(table.fontMetrics().height() * 5)
        return table

    # ---------------------------------------------------------------- content

    def report(self) -> TextReport | None:
        return self._report

    def clear(self, message: str = "Text is inspected after the comparison.") -> None:
        self._report = None
        self._images = None
        self.summary.setText(message)
        self.findings_table.setRowCount(0)
        self.lines_table.setRowCount(0)
        self.show_finding(None, OperatorMark.UNREVIEWED)

    def set_report(
        self, report: TextReport, marks: dict[int, OperatorMark], reference: np.ndarray, registered: np.ndarray,
    ) -> None:
        self._report = report
        self._images = (reference, registered)
        parts = []
        if report.available:
            reliable = sum(f.reliable for f in report.findings)
            parts.append(
                f"{report.engine}: {len(report.lines)} text line(s), {len(report.findings)} finding(s) "
                f"({reliable} reliable, {len(report.findings) - reliable} uncertain), {report.seconds:.1f} s."
            )
        parts += [i.message for i in report.issues] + list(report.notes)
        self.summary.setText(" ".join(parts))
        self.fill(marks)

    def fill(self, marks: dict[int, OperatorMark]) -> None:
        report = self._report
        if report is None:
            return
        table = self.findings_table
        table.blockSignals(True)
        table.setRowCount(len(report.findings))
        for row, f in enumerate(report.findings):
            mark = marks.get(f.id, OperatorMark.UNREVIEWED)
            values = (
                f.label, f.kind.label, f.expected, f.read, f.severity.value, mark.label,
                "reliable" if f.reliable else "uncertain",
                f"{_conf(f.artwork_confidence)} / {_conf(f.print_confidence)}",
            )
            tip = f.describe() + "\n" + "\n".join(f.notes) + (f"\nZone: {f.zone}" if f.zone else "")
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, f.id)
                item.setToolTip(tip)
                if mark is OperatorMark.DISMISSED:
                    item.setForeground(QBrush(QColor("#9ca3af")))
                elif f.severity.value == "critical":
                    item.setForeground(QBrush(QColor("#b91c1c")))
                table.setItem(row, col, item)
            if f.id == self._selected:
                table.selectRow(row)
        table.blockSignals(False)

        lines = self.lines_table
        lines.blockSignals(True)
        lines.setRowCount(len(report.lines))
        for row, line in enumerate(report.lines):
            zone = (line.zone or "") + (" (critical)" if line.critical and line.zone else "")
            if line.critical and not line.zone:
                zone = "critical (numbers)"
            values = (
                str(line.id), line.status.value, line.expected_text, line.print_text,
                f"{_conf(line.artwork_confidence)} / {_conf(line.print_confidence)}", zone,
            )
            tip = "\n".join(line.notes) or ("Expected text from the zone." if line.expected_source == "zone" else "")
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, line.id)
                if tip:
                    item.setToolTip(tip)
                if col == 1:
                    item.setForeground(QBrush(QColor(_STATUS_COLOURS[line.status])))
                lines.setItem(row, col, item)
        lines.blockSignals(False)

    def selected(self) -> int | None:
        return self._selected

    def select(self, finding_id: int | None) -> None:
        self._selected = finding_id
        table = self.findings_table
        table.blockSignals(True)
        table.clearSelection()
        for row in range(table.rowCount()):
            if table.item(row, 0).data(Qt.ItemDataRole.UserRole) == finding_id:
                table.selectRow(row)
                table.scrollToItem(table.item(row, 0))
        table.blockSignals(False)

    def show_finding(self, finding: TextFinding | None, mark: OperatorMark) -> None:
        for button in self.buttons.values():
            button.setEnabled(finding is not None)
        if finding is None or self._images is None:
            self.title.set_full_text("Select a text finding.")
            self.artwork_crop.clear()
            self.print_crop.clear()
            self.notes.setText("")
            self.mark_label.setText("")
            return
        reference, registered = self._images
        box = (finding.x, finding.y, finding.width, finding.height)
        margin = max(6, finding.height)
        self.artwork_crop.setPixmap(_crop_pixmap(reference, box, margin))
        self.print_crop.setPixmap(_crop_pixmap(registered, box, margin))
        self.title.set_full_text(
            f"{finding.label}  {finding.describe()} · {finding.severity.value.upper()} · "
            f"{'reliable' if finding.reliable else 'uncertain'}"
        )
        conf = f"OCR confidence: artwork {_conf(finding.artwork_confidence)}, print {_conf(finding.print_confidence)}."
        image = f"Image: comparison score {finding.image_score:.1f}, character ink missing {finding.ink_loss:.0%}."
        zone = f"Zone: {finding.zone}." if finding.zone else ("Critical text (contains numbers)." if finding.critical else "")
        self.notes.setText("\n".join([*finding.notes, conf, image, zone]).strip())
        self.mark_label.setText(f"Operator: {mark.label}")

    def _on_finding_clicked(self) -> None:
        items = self.findings_table.selectedItems()
        if items:
            self.finding_selected.emit(int(items[0].data(Qt.ItemDataRole.UserRole)))

    def _on_line_clicked(self) -> None:
        items = self.lines_table.selectedItems()
        if items:
            self.line_selected.emit(int(items[0].data(Qt.ItemDataRole.UserRole)))

    def _request(self, mark: OperatorMark) -> None:
        if self._selected is not None:
            self.mark_requested.emit(self._selected, mark)
