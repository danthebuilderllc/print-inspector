"""PASS / FAIL / REVIEW verdict badge."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QWidget

from print_inspector.core.models import InspectionStatus

# (background, text) colours per status
_COLOURS: dict[InspectionStatus, tuple[str, str]] = {
    InspectionStatus.NOT_INSPECTED: ("#6b7280", "#ffffff"),
    InspectionStatus.PASS: ("#15803d", "#ffffff"),
    InspectionStatus.FAIL: ("#b91c1c", "#ffffff"),
    InspectionStatus.REVIEW: ("#d97706", "#111111"),
}


class StatusIndicator(QLabel):
    """Coloured verdict word. ``compact`` gives a lower badge for the status strip."""

    def __init__(self, parent: QWidget | None = None, compact: bool = False) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._font_pt = 13 if compact else 17
        self.setMinimumHeight(self.fontMetrics().height() * (2 if compact else 3))
        self._status = InspectionStatus.NOT_INSPECTED
        self.set_status(InspectionStatus.NOT_INSPECTED)

    @property
    def status(self) -> InspectionStatus:
        return self._status

    def set_status(self, status: InspectionStatus) -> None:
        self._status = status
        background, foreground = _COLOURS[status]
        self.setText(status.label)
        self.setAccessibleName(f"Inspection result: {status.label}")
        self.setStyleSheet(
            f"QLabel {{ background-color: {background}; color: {foreground}; "
            f"font-size: {self._font_pt}pt; font-weight: bold; border-radius: 6px; padding: 4px 12px; }}"
        )
