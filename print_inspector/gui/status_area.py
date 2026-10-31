"""Bottom status area: capture quality, alignment quality, progress and the verdict."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from print_inspector.core.models import QualityLevel
from print_inspector.gui.elided_label import ElidedLabel
from print_inspector.gui.status_indicator import StatusIndicator

# level: (word, background, foreground)
LEVEL_STYLE: dict[QualityLevel | None, tuple[str, str, str]] = {
    None: ("-", "#e5e7eb", "#374151"),
    QualityLevel.GOOD: ("GOOD", "#dcfce7", "#14532d"),
    QualityLevel.MARGINAL: ("MARGINAL", "#fef3c7", "#78350f"),
    QualityLevel.POOR: ("POOR", "#fee2e2", "#7f1d1d"),
}


def level_style_sheet(level: QualityLevel | None) -> str:
    _, background, foreground = LEVEL_STYLE[level]
    return (
        f"QLabel {{ background: {background}; color: {foreground}; font-weight: bold; "
        "border-radius: 4px; padding: 4px 8px; }"
    )


def _caption(text: str) -> ElidedLabel:
    label = ElidedLabel()
    label.setObjectName("statusCaption")
    label.set_full_text(text)
    return label


class CheckBadge(QWidget):
    """'<title>' over a coloured level word and a one-line detail."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self.badge = QLabel()
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail = ElidedLabel()
        self.detail.setObjectName("statusDetail")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(_caption(title))
        row = QHBoxLayout()
        row.addWidget(self.badge)
        row.addWidget(self.detail, 1)
        layout.addLayout(row)
        self.set_level(None)

    def set_level(self, level: QualityLevel | None, word: str | None = None, detail: str = "") -> None:
        self.badge.setText(word or LEVEL_STYLE[level][0])
        self.badge.setStyleSheet(level_style_sheet(level))
        self.badge.setAccessibleName(f"{self._title}: {self.badge.text()}")
        self.detail.set_full_text(detail)

    def word(self) -> str:
        return self.badge.text()


class StatusArea(QFrame):
    """The strip at the bottom of the window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statusArea")
        self.quality = CheckBadge("Capture quality")
        self.alignment = CheckBadge("Alignment quality")
        self.progress_label = ElidedLabel()
        self.progress_label.setObjectName("statusDetail")
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(10)
        progress_box = QVBoxLayout()
        progress_box.setSpacing(2)
        progress_box.addWidget(_caption("Inspection progress"))
        progress_box.addWidget(self.progress)
        progress_box.addWidget(self.progress_label)

        self.verdict = StatusIndicator(compact=True)
        self.verdict_reason = ElidedLabel()
        self.verdict_reason.setObjectName("statusDetail")
        verdict_box = QVBoxLayout()
        verdict_box.setSpacing(2)
        verdict_box.addWidget(self.verdict)
        verdict_box.addWidget(self.verdict_reason)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(14)
        layout.addWidget(self.quality, 3)
        layout.addWidget(self.alignment, 3)
        layout.addLayout(progress_box, 4)
        layout.addLayout(verdict_box, 4)
        self.set_progress(0, 0, "Waiting for images.", busy=False)

    def set_progress(self, done: int, total: int, text: str, busy: bool) -> None:
        if busy and total == 0:
            self.progress.setRange(0, 0)  # indeterminate
        else:
            self.progress.setRange(0, max(total, 1))
            self.progress.setValue(min(done, max(total, 1)) if total else 0)
        self.progress_label.set_full_text(text)
        self.progress.setAccessibleName(f"Inspection progress: {text}")
