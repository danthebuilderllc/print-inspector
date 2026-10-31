"""Manual registration fallback: nudge the photo onto the artwork with a live overlay."""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from print_inspector.config import UIConfig
from print_inspector.core.registration import ManualAdjustment
from print_inspector.gui.image_panel import ImageView
from print_inspector.gui.overlay import OVERLAY_MODES, make_overlay
from print_inspector.gui.workers import bgr8_to_qimage

PREVIEW_MAX_SIDE = 2000
NUDGE_PX = 0.25


class RegistrationAdjustDialog(QDialog):
    """Adjust offset, rotation and scale of the registered photo relative to the artwork.

    ``base_homography`` maps comparison-grid pixels to photo pixels (the current
    registration). The preview resamples the photo directly with the adjusted
    mapping (without the local correction), so what you see is what is applied.
    """

    def __init__(
        self,
        photo: np.ndarray,
        reference: np.ndarray,
        base_homography: np.ndarray,
        ui_config: UIConfig,
        initial: ManualAdjustment | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Adjust Registration")
        self.resize(1200, 850)
        self._photo = photo
        self._base = np.asarray(base_homography, np.float64)
        gh, gw = reference.shape[:2]
        self._grid_size = (gw, gh)
        k = min(1.0, PREVIEW_MAX_SIDE / max(gw, gh))
        self._preview_size = (max(8, round(gw * k)), max(8, round(gh * k)))
        # preview pixel -> grid pixel (pixel-centre convention)
        self._preview_to_grid = np.array(
            [[gw / self._preview_size[0], 0, 0.5 * gw / self._preview_size[0] - 0.5],
             [0, gh / self._preview_size[1], 0.5 * gh / self._preview_size[1] - 0.5],
             [0, 0, 1.0]]
        )
        self._reference = np.ascontiguousarray(
            cv2.resize(reference, self._preview_size, interpolation=cv2.INTER_AREA) if k < 1 else reference
        )
        self._preview: np.ndarray | None = None
        self._overlay: np.ndarray | None = None

        self.view = ImageView(ui_config)
        self.dx = self._spin(-200, 200, 0.05, 2, " px")
        self.dy = self._spin(-200, 200, 0.05, 2, " px")
        self.rotation = self._spin(-10, 10, 0.01, 3, " °")
        self.scale_x = self._spin(90, 110, 0.01, 3, " %")
        self.scale_y = self._spin(90, 110, 0.01, 3, " %")
        self.mode_combo = QComboBox()
        for key, label in OVERLAY_MODES.items():
            self.mode_combo.addItem(label, key)
        self.mode_combo.setCurrentIndex(self.mode_combo.findData("edges"))
        self.mode_combo.currentIndexChanged.connect(lambda _: self._redraw_overlay())
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(70)
        self.opacity.valueChanged.connect(lambda _: self._redraw_overlay())
        self.refine_box = QCheckBox("Refine automatically from this position")
        self.refine_box.setChecked(True)
        self.refine_box.setToolTip(
            "Runs intensity-based and local refinement starting from your adjustment. "
            "Untick to apply exactly what you set."
        )
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(lambda: self.set_adjustment(ManualAdjustment()))

        form = QFormLayout()
        form.addRow("Move right", self.dx)
        form.addRow("Move down", self.dy)
        form.addRow("Rotate", self.rotation)
        form.addRow("Width", self.scale_x)
        form.addRow("Height", self.scale_y)
        form.addRow("Overlay", self.mode_combo)
        form.addRow("Opacity", self.opacity)
        controls = QVBoxLayout()
        controls.addLayout(form)
        controls.addWidget(self.refine_box)
        controls.addWidget(self.reset_button)
        hint = QLabel(
            "Arrow keys move the photo by 0.25 px (Shift: 1 px). Zoom in (mouse wheel) on text "
            "and fine lines: the magenta artwork edges should sit exactly on the printed edges."
        )
        hint.setWordWrap(True)
        controls.addWidget(hint)
        controls.addStretch(1)
        side = QWidget()
        side.setLayout(controls)
        side.setFixedWidth(300)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        body = QHBoxLayout()
        body.addWidget(self.view, 1)
        body.addWidget(side)
        layout = QVBoxLayout(self)
        layout.addLayout(body, 1)
        layout.addWidget(self.buttons)

        for key, ddx, ddy in (("Left", -1, 0), ("Right", 1, 0), ("Up", 0, -1), ("Down", 0, 1)):
            for modifier, step in (("", NUDGE_PX), ("Shift+", 1.0)):
                shortcut = QShortcut(QKeySequence(modifier + key), self)
                shortcut.activated.connect(lambda ddx=ddx, ddy=ddy, step=step: self.nudge(ddx * step, ddy * step))

        for spin in (self.dx, self.dy, self.rotation, self.scale_x, self.scale_y):
            spin.valueChanged.connect(lambda _: self._update_preview())
        self.set_adjustment(initial or ManualAdjustment())
        self.view.fit_to_window()

    @staticmethod
    def _spin(low: float, high: float, step: float, decimals: int, suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(False)
        return spin

    # ---------------------------------------------------------------- state

    def adjustment(self) -> ManualAdjustment:
        return ManualAdjustment(
            dx=self.dx.value(),
            dy=self.dy.value(),
            rotation_deg=self.rotation.value(),
            scale_x=self.scale_x.value() / 100.0,
            scale_y=self.scale_y.value() / 100.0,
        )

    def set_adjustment(self, adjustment: ManualAdjustment) -> None:
        spins = (self.dx, self.dy, self.rotation, self.scale_x, self.scale_y)
        values = (adjustment.dx, adjustment.dy, adjustment.rotation_deg, adjustment.scale_x * 100, adjustment.scale_y * 100)
        for spin, value in zip(spins, values):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._update_preview()

    def refine(self) -> bool:
        return self.refine_box.isChecked()

    def adjusted_homography(self) -> np.ndarray:
        """Grid -> photo mapping with the adjustment applied."""
        return self._base @ self.adjustment().sampling_matrix(self._grid_size)

    def nudge(self, dx: float, dy: float) -> None:
        self.dx.setValue(self.dx.value() + dx)
        self.dy.setValue(self.dy.value() + dy)

    def preview_image(self) -> np.ndarray | None:
        return self._preview

    # ---------------------------------------------------------------- drawing

    def _update_preview(self) -> None:
        mapping = self.adjusted_homography() @ self._preview_to_grid
        self._preview = cv2.warpPerspective(
            self._photo, mapping, self._preview_size,
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT,
        )
        self._redraw_overlay()

    def _redraw_overlay(self) -> None:
        if self._preview is None:
            return
        self._overlay = make_overlay(
            self._reference, self._preview, None, self.mode_combo.currentData(), self.opacity.value() / 100.0
        )
        self.view.update_image(bgr8_to_qimage(self._overlay))
