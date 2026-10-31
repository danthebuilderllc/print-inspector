"""Side-panel tab for the barcode inspection: symbols, findings and the selected finding."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from print_inspector.core.barcode_inspection import NOT_ISO, BarcodeFinding, BarcodeReport, SymbolStatus
from print_inspector.core.barcodes import FORMAT_LABELS
from print_inspector.core.decision import OperatorMark
from print_inspector.gui.elided_label import ElidedLabel
from print_inspector.gui.text_panel import TextPanel, _crop_pixmap

SYMBOL_COLUMNS = ("#", "Name", "Type", "Expected", "Decoded (print)", "Status", "Decoding margin", "Damage")
FINDING_COLUMNS = ("#", "Problem", "Barcode", "Severity", "Operator", "Detection")
COL_BARCODE_OPERATOR = FINDING_COLUMNS.index("Operator")
_STATUS_COLOURS = {
    SymbolStatus.OK: "#14532d",
    SymbolStatus.DAMAGED: "#92400e",
    SymbolStatus.WRONG_CONTENT: "#b91c1c",
    SymbolStatus.WRONG_FORMAT: "#b91c1c",
    SymbolStatus.NOT_DECODED: "#b91c1c",
    SymbolStatus.NO_REFERENCE: "#92400e",
}


def _margin(symbol) -> str:
    if symbol.strips is not None:
        return f"{symbol.strips[0]}/{symbol.strips[1]} strips"
    if symbol.unused_ec is not None:
        return f"{symbol.unused_ec:.0%} EC unused ({symbol.ec_level})"
    return "-"


def _damage(symbol) -> str:
    if symbol.module_errors is not None:
        return f"{symbol.module_errors} of {symbol.module_count} modules"
    if symbol.bar_growth_modules is not None:
        return f"bars {symbol.bar_growth_modules:+.2f} module"
    return "-"


class BarcodePanel(QWidget):
    finding_selected = Signal(int)
    symbol_selected = Signal(int)
    mark_requested = Signal(int, object)  # finding id, OperatorMark
    edit_zones_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._report: BarcodeReport | None = None
        self._selected: int | None = None
        self._images: tuple[np.ndarray, np.ndarray] | None = None

        self.summary = QLabel("Barcodes are inspected after the comparison.")
        self.summary.setWordWrap(True)
        self.summary.setObjectName("sideNote")
        self.zones_button = QPushButton("Edit Barcode Zones…")
        self.zones_button.setToolTip("Mark where barcodes must be, with their type and value (Ctrl+Shift+B).")
        self.zones_button.clicked.connect(self.edit_zones_requested)
        self.symbols_table = TextPanel._table(SYMBOL_COLUMNS)
        self.symbols_table.itemSelectionChanged.connect(self._on_symbol_clicked)
        self.findings_table = TextPanel._table(FINDING_COLUMNS)
        self.findings_table.itemSelectionChanged.connect(self._on_finding_clicked)
        disclaimer = QLabel(NOT_ISO)
        disclaimer.setWordWrap(True)
        disclaimer.setObjectName("sideNote")

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
        top_layout.addWidget(QLabel("<b>Barcodes</b>"))
        top_layout.addWidget(self.symbols_table, 1)
        top_layout.addWidget(QLabel("<b>Barcode findings</b>"))
        top_layout.addWidget(self.findings_table, 1)
        top_layout.addWidget(disclaimer)

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

    def report(self) -> BarcodeReport | None:
        return self._report

    def clear(self, message: str = "Barcodes are inspected after the comparison.") -> None:
        self._report = None
        self._images = None
        self.summary.setText(message)
        self.symbols_table.setRowCount(0)
        self.findings_table.setRowCount(0)
        self.show_finding(None, OperatorMark.UNREVIEWED)

    def set_report(
        self, report: BarcodeReport, marks: dict[int, OperatorMark], reference: np.ndarray, registered: np.ndarray,
    ) -> None:
        self._report = report
        self._images = (reference, registered)
        parts = []
        if report.available:
            decoded = sum(1 for s in report.symbols if s.print_value)
            parts.append(
                f"{report.library}: {len(report.symbols)} barcode(s), {decoded} decoded in the print, "
                f"{len(report.findings)} finding(s), {report.seconds:.1f} s."
            )
        parts += [i.message for i in report.issues] + list(report.notes)
        self.summary.setText(" ".join(parts))
        self.fill(marks)

    def fill(self, marks: dict[int, OperatorMark]) -> None:
        report = self._report
        if report is None:
            return
        table = self.symbols_table
        table.blockSignals(True)
        table.setRowCount(len(report.symbols))
        for row, s in enumerate(report.symbols):
            name = s.name + (" (critical)" if s.critical else "")
            expected = s.expected_value + (f"  [{s.expected_source}]" if s.expected_source else "")
            decoded = s.print_value + (
                f"  ({FORMAT_LABELS[s.print_format]})" if s.print_format and s.print_format != s.format else ""
            )
            values = (str(s.id), name, s.format_label, expected, decoded, s.status.value, _margin(s), _damage(s))
            tip = "\n".join(s.notes)
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, s.id)
                if tip:
                    item.setToolTip(tip)
                if col == 5:
                    item.setForeground(QBrush(QColor(_STATUS_COLOURS[s.status])))
                table.setItem(row, col, item)
        table.blockSignals(False)

        table = self.findings_table
        table.blockSignals(True)
        table.setRowCount(len(report.findings))
        for row, f in enumerate(report.findings):
            mark = marks.get(f.id, OperatorMark.UNREVIEWED)
            symbol = report.symbol(f.symbol_id) if f.symbol_id is not None else None
            values = (
                f.label, f.describe(), symbol.name if symbol else FORMAT_LABELS.get(f.format, ""),
                f.severity.value, mark.label, "reliable" if f.reliable else "uncertain",
            )
            tip = "\n".join(f.notes)
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

    def show_finding(self, finding: BarcodeFinding | None, mark: OperatorMark) -> None:
        for button in self.buttons.values():
            button.setEnabled(finding is not None)
        if finding is None or self._images is None:
            self.title.set_full_text("Select a barcode finding.")
            self.artwork_crop.clear()
            self.print_crop.clear()
            self.notes.setText("")
            self.mark_label.setText("")
            return
        reference, registered = self._images
        box = (finding.x, finding.y, finding.width, finding.height)
        margin = max(8, min(finding.width, finding.height) // 2)
        self.artwork_crop.setPixmap(_crop_pixmap(reference, box, margin))
        self.print_crop.setPixmap(_crop_pixmap(registered, box, margin))
        self.title.set_full_text(
            f"{finding.label}  {finding.describe()} · {finding.severity.value.upper()} · "
            f"{'reliable' if finding.reliable else 'uncertain'}"
        )
        zone = f"Zone: {finding.zone}." if finding.zone else ""
        self.notes.setText("\n".join([*finding.notes, zone]).strip())
        self.mark_label.setText(f"Operator: {mark.label}")

    def _on_finding_clicked(self) -> None:
        items = self.findings_table.selectedItems()
        if items:
            self.finding_selected.emit(int(items[0].data(Qt.ItemDataRole.UserRole)))

    def _on_symbol_clicked(self) -> None:
        items = self.symbols_table.selectedItems()
        if items:
            self.symbol_selected.emit(int(items[0].data(Qt.ItemDataRole.UserRole)))

    def _request(self, mark: OperatorMark) -> None:
        if self._selected is not None:
            self.mark_requested.emit(self._selected, mark)
