"""Side-panel tab for the colour analysis: mode, lighting / white balance, colour areas and findings."""
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

from print_inspector.core.color import describe_shift, lab_to_bgr
from print_inspector.core.color_inspection import APPROXIMATE_NOTE, ColorFinding, ColorReport
from print_inspector.core.decision import OperatorMark
from print_inspector.gui.elided_label import ElidedLabel
from print_inspector.gui.text_panel import TextPanel, _crop_pixmap

AREA_COLUMNS = ("#", "Artwork", "Print", "ΔE00", "ΔL*", "Δa*", "Δb*", "Area (mm²)", "Local ΔE00")
FINDING_COLUMNS = ("#", "Problem", "ΔE00", "Change", "Severity", "Operator", "Detection")
COL_COLOR_OPERATOR = FINDING_COLUMNS.index("Operator")


def swatch(lab) -> QColor:
    b, g, r = (int(round(v)) for v in np.clip(lab_to_bgr(np.asarray(lab, np.float64)), 0, 255))
    return QColor(r, g, b)


class ColorPanel(QWidget):
    finding_selected = Signal(int)
    area_selected = Signal(int)
    mark_requested = Signal(int, object)  # finding id, OperatorMark
    calibration_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._report: ColorReport | None = None
        self._selected: int | None = None
        self._images: tuple[np.ndarray, np.ndarray] | None = None

        self.mode_label = QLabel("Colour is analysed after the comparison.")
        self.mode_label.setWordWrap(True)
        self.calibration_button = QPushButton("Colour Calibration…")
        self.calibration_button.setToolTip("Calibrate the camera with a photographed colour chart (Ctrl+K).")
        self.calibration_button.clicked.connect(self.calibration_requested)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setObjectName("sideNote")
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.areas_table = TextPanel._table(AREA_COLUMNS)
        self.areas_table.itemSelectionChanged.connect(self._on_area_clicked)
        self.findings_table = TextPanel._table(FINDING_COLUMNS)
        self.findings_table.itemSelectionChanged.connect(self._on_finding_clicked)

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
        top_layout.addWidget(self.mode_label)  # stacked: the panel must stay narrow on small screens
        row = QHBoxLayout()
        row.addWidget(self.calibration_button)
        row.addStretch(1)
        top_layout.addLayout(row)
        top_layout.addWidget(self.summary)
        top_layout.addWidget(QLabel("<b>Colour findings</b>"))
        top_layout.addWidget(self.findings_table, 1)
        areas_heading = QLabel("<b>Colour areas</b> (flat colours of the artwork, CIELAB D50)")
        areas_heading.setWordWrap(True)
        top_layout.addWidget(areas_heading)
        top_layout.addWidget(self.areas_table, 1)

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

    def report(self) -> ColorReport | None:
        return self._report

    def clear(self, message: str = "Colour is analysed after the comparison.") -> None:
        self._report = None
        self._images = None
        self.mode_label.setText(message)
        self.mode_label.setStyleSheet("")
        self.summary.setText("")
        self.areas_table.setRowCount(0)
        self.findings_table.setRowCount(0)
        self.show_finding(None, OperatorMark.UNREVIEWED)

    def set_report(self, report: ColorReport, marks, reference: np.ndarray, registered: np.ndarray) -> None:
        self._report = report
        self._images = (reference, registered)
        if report.calibrated:
            self.mode_label.setText(f"<b>CALIBRATED</b> colour measurement · {report.profile}")
            self.mode_label.setStyleSheet("QLabel { background: #dcfce7; color: #14532d; padding: 6px; border-radius: 4px; }")
        else:
            self.mode_label.setText(f"<b>APPROXIMATE</b> · {APPROXIMATE_NOTE}")
            self.mode_label.setStyleSheet("QLabel { background: #fef3c7; color: #78350f; padding: 6px; border-radius: 4px; }")
        lines = list(report.mode_reasons)
        if report.paper_design_lab is not None and report.paper_photo_delta_e is not None:
            balance = report.channel_balance
            lines.append(
                f"Paper in the photo: ΔE00 {report.paper_photo_delta_e:.1f} from the artwork white"
                + (f" (R/G {balance[0]:.2f}, B/G {balance[2]:.2f})" if balance else "") + "."
            )
        lines.append(f"Lighting across the label: {report.lighting_range[0]:.2f} - {report.lighting_range[1]:.2f}.")
        if report.global_shift is not None:
            lines.append(f"Inks overall: {describe_shift(*report.global_shift)} "
                         f"(ΔL* {report.global_shift[0]:+.1f}, Δa* {report.global_shift[1]:+.1f}, Δb* {report.global_shift[2]:+.1f}).")
        lines.append(f"Colour measured on {report.meaningful_fraction:.0%} of the label (flat areas away from edges "
                     "and structural defects); {:.1f} s.".format(report.seconds))
        lines += [i.message for i in report.issues if i.code != "color_uncalibrated"] + list(report.notes)
        self.summary.setText("\n".join(lines))
        self.fill(marks)

    def fill(self, marks) -> None:
        report = self._report
        if report is None:
            return
        table = self.areas_table
        table.blockSignals(True)
        table.setRowCount(len(report.areas))
        for row, a in enumerate(report.areas):
            diff = np.subtract(a.measured_lab, a.design_lab)
            values = (
                f"{a.id}{' (paper)' if a.paper else ''}", "", "", f"{a.delta_e:.1f}",
                f"{diff[0]:+.1f}", f"{diff[1]:+.1f}", f"{diff[2]:+.1f}", f"{a.area_mm2:.0f}", f"{a.local_spread:.1f}",
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, a.id)
                if col == 1:
                    item.setBackground(QBrush(swatch(a.design_lab)))
                    item.setToolTip("L* {:.1f}  a* {:.1f}  b* {:.1f}".format(*a.design_lab))
                elif col == 2:
                    item.setBackground(QBrush(swatch(a.measured_lab)))
                    item.setToolTip("L* {:.1f}  a* {:.1f}  b* {:.1f}".format(*a.measured_lab))
                table.setItem(row, col, item)
        table.blockSignals(False)

        table = self.findings_table
        table.blockSignals(True)
        table.setRowCount(len(report.findings))
        for row, f in enumerate(report.findings):
            mark = marks.get(f.id, OperatorMark.UNREVIEWED)
            values = (
                f.label, f.kind.label, f"{f.delta_e:.1f}", describe_shift(f.dl, f.da, f.db), f.severity.value,
                mark.label, "reliable" if f.reliable else "uncertain",
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, f.id)
                item.setToolTip("\n".join(f.notes))
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

    def show_finding(self, finding: ColorFinding | None, mark: OperatorMark) -> None:
        for button in self.buttons.values():
            button.setEnabled(finding is not None)
        if finding is None or self._images is None:
            self.title.set_full_text("Select a colour finding.")
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
        lab = ""
        if finding.expected_lab is not None and finding.measured_lab is not None:
            lab = ("Expected L* {:.1f} a* {:.1f} b* {:.1f}; ".format(*finding.expected_lab)
                   + "measured L* {:.1f} a* {:.1f} b* {:.1f}.".format(*finding.measured_lab))
        self.notes.setText("\n".join([*finding.notes, lab, f"Area {finding.area_mm2:.1f} mm²."]).strip())
        self.mark_label.setText(f"Operator: {mark.label}")

    def _on_finding_clicked(self) -> None:
        items = self.findings_table.selectedItems()
        if items:
            self.finding_selected.emit(int(items[0].data(Qt.ItemDataRole.UserRole)))

    def _on_area_clicked(self) -> None:
        items = self.areas_table.selectedItems()
        if items:
            self.area_selected.emit(int(items[0].data(Qt.ItemDataRole.UserRole)))

    def _request(self, mark: OperatorMark) -> None:
        if self._selected is not None:
            self.mark_requested.emit(self._selected, mark)
