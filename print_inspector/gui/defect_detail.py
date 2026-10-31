"""Side-by-side view of one defect with its evidence and the operator's judgement."""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from print_inspector.core.decision import OperatorMark
from print_inspector.core.defects import Defect
from print_inspector.gui.elided_label import ElidedLabel

CROP_DISPLAY_PX = 150  # side length of each enlarged crop


def _to_pixmap(bgr: np.ndarray, region: np.ndarray | None, size: int) -> QPixmap:
    """Enlarge a crop (nearest neighbour, so pixels stay visible) and outline the region."""
    h, w = bgr.shape[:2]
    scale = max(1, size // max(h, w))
    big = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    if region is not None:
        mask = cv2.resize(region, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(big, contours, -1, (0, 0, 255), 1)
    big = np.ascontiguousarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
    image = QImage(big.data, big.shape[1], big.shape[0], int(big.strides[0]), QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class DefectDetailWidget(QWidget):
    """Artwork, printed and difference crops of the selected defect, plus mark buttons."""

    mark_requested = Signal(int, object)  # defect id, OperatorMark

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._defect: Defect | None = None
        self.title = ElidedLabel()
        self.title.setStyleSheet("font-weight: bold;")
        self.images: dict[str, QLabel] = {}
        grid = QGridLayout()
        for col, (key, caption) in enumerate((("artwork", "Original"), ("printed", "Printed"), ("difference", "Difference"))):
            image = QLabel()
            image.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image.setMinimumSize(CROP_DISPLAY_PX // 2, CROP_DISPLAY_PX // 2)
            image.setStyleSheet("background: #e5e7eb;")
            label = QLabel(caption)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(label, 0, col)
            grid.addWidget(image, 1, col)
            self.images[key] = image
        self.evidence = QLabel("")
        self.evidence.setObjectName("metrics")
        self.evidence.setWordWrap(True)
        self.evidence.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.mark_label = QLabel("")

        self.buttons: dict[OperatorMark, QPushButton] = {}
        row = QGridLayout()  # 2 x 2, so the panel stays narrow
        for index, (mark, text, tip) in enumerate((
            (OperatorMark.CONFIRMED, "Confirm defect", "This is a real printing defect."),
            (OperatorMark.DISMISSED, "Dismiss", "Not a defect (e.g. dust, reflection, acceptable variation)."),
            (OperatorMark.NEEDS_REVIEW, "Needs review", "Cannot decide here; someone else must look."),
            (OperatorMark.UNREVIEWED, "Clear mark", "Back to the automatic assessment."),
        )):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(lambda _=False, m=mark: self._request(m))
            row.addWidget(button, index // 2, index % 2)
            self.buttons[mark] = button

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.title)
        layout.addLayout(grid)
        layout.addWidget(self.evidence)
        layout.addWidget(self.mark_label)
        layout.addLayout(row)
        self.show_defect(None, OperatorMark.UNREVIEWED)

    def defect(self) -> Defect | None:
        return self._defect

    def show_defect(self, defect: Defect | None, mark: OperatorMark, scale_estimated: bool = False) -> None:
        self._defect = defect
        for button in self.buttons.values():
            button.setEnabled(defect is not None)
        if defect is None:
            self.title.set_full_text("Select a defect in the list.")
            for label in self.images.values():
                label.clear()
            self.evidence.setText("")
            self.mark_label.setText("")
            return
        e = defect.evidence
        size = CROP_DISPLAY_PX
        self.images["artwork"].setPixmap(_to_pixmap(e.artwork, None, size))
        self.images["printed"].setPixmap(_to_pixmap(e.printed, e.region, size))
        diff = cv2.applyColorMap(np.clip(e.difference.astype(np.float32) * 3, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        self.images["difference"].setPixmap(_to_pixmap(diff, e.region, size))
        self.title.set_full_text(
            f"#{defect.id}  {defect.category_label} · {defect.severity.value.upper()} · "
            f"{'reliably detected' if defect.is_reliable else 'uncertain detection'}"
        )
        mm = "mm (estimated scale)" if scale_estimated else "mm"
        self.evidence.setText("\n".join([
            f"Category     {defect.category_note}",
            f"Category is  {'likely' if defect.category_certain else 'NOT verified'}",
            f"Detection    {'; '.join(defect.reliability_notes)}",
            f"Severity     {'; '.join(defect.severity_notes)}",
            f"Size         {defect.length_mm:.2f} × {defect.thickness_mm:.2f} {mm}, {defect.area_mm2:.3f} mm² "
            f"({defect.width} × {defect.height} px, {defect.area_px} px)",
            f"Difference   max {defect.max_difference:.0f}, mean {defect.mean_difference:.0f} gray levels; "
            f"ΔE {defect.delta_e:.1f} (ΔL {defect.delta_l:+.1f}, ΔC {defect.delta_c:.1f})",
            f"Evidence     score {defect.score:.1f} ({defect.peak_sigma:.1f} × noise), SSIM {defect.min_ssim:.2f}, "
            f"edges missing {defect.missing_edge_px} / extra {defect.extra_edge_px}",
        ]))
        self.mark_label.setText(f"Operator: {mark.label}")

    def _request(self, mark: OperatorMark) -> None:
        if self._defect is not None:
            self.mark_requested.emit(self._defect.id, mark)
