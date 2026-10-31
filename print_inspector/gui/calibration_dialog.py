"""Colour calibration with a photographed reference chart."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from print_inspector.config import ColorConfig, ImageLoadingConfig, UIConfig
from print_inspector.core.color_calibration import (
    COLORCHECKER_24,
    SAME_PHOTO,
    SEPARATE_PHOTO,
    CalibrationError,
    CalibrationProfile,
    ChartReference,
    calibrate,
    load_chart_reference,
)
from print_inspector.core.image_io import ImageLoadError, ImageRole, LoadedImage, load_image
from print_inspector.gui.corner_dialog import CornerSelectionDialog
from print_inspector.gui.workers import bgr8_to_qimage

INSTRUCTIONS = (
    "Photograph the chart with the same phone, settings and lighting as the labels - best in the same photo. "
    "Lock exposure and white balance if the camera allows it. Then mark the four outer corners of the patch "
    "field, starting at the chart's top-left patch (dark skin on a ColorChecker), clockwise."
)


class CalibrationDialog(QDialog):
    """Create, check and choose the colour profile; ``profile()`` after acceptance (None = no calibration)."""

    def __init__(
        self,
        current_photo: LoadedImage | None,
        current_profile: CalibrationProfile | None,
        ui_config: UIConfig,
        color_config: ColorConfig,
        loading_config: ImageLoadingConfig,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Colour Calibration")
        self._ui = ui_config
        self._cfg = color_config
        self._loading = loading_config
        self._current_photo = current_photo
        self._chart_photo: LoadedImage | None = None
        self._chart: ChartReference = COLORCHECKER_24
        self._profile: CalibrationProfile | None = current_profile
        self._removed = False

        self.current_label = QLabel()
        self.current_label.setWordWrap(True)
        self.same_photo = QRadioButton("Chart in the current printed photo")
        self.separate_photo = QRadioButton("Separate chart photo")
        self.same_photo.setEnabled(current_photo is not None)
        (self.same_photo if current_photo is not None else self.separate_photo).setChecked(True)
        self.load_photo_button = QPushButton("Load Chart Photo…")
        self.load_photo_button.clicked.connect(self._choose_photo)
        self.photo_label = QLabel("")
        self.chart_combo = QComboBox()
        self.chart_combo.addItem(COLORCHECKER_24.name, "builtin")
        self.chart_combo.addItem("Load chart reference file…", "file")
        self.chart_combo.activated.connect(self._on_chart_chosen)
        self.mark_button = QPushButton("Mark Chart Corners and Calibrate…")
        self.mark_button.clicked.connect(self._mark_corners)
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.patch_table = QTableWidget(0, 2)
        self.patch_table.setHorizontalHeaderLabels(("Patch", "ΔE00 (leave-one-out)"))
        self.patch_table.verticalHeader().setVisible(False)
        self.patch_table.horizontalHeader().setStretchLastSection(True)
        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #b91c1c;")
        instructions = QLabel(INSTRUCTIONS)
        instructions.setWordWrap(True)
        instructions.setObjectName("sideNote")

        photo_row = QHBoxLayout()
        photo_row.addWidget(self.separate_photo)
        photo_row.addWidget(self.load_photo_button)
        photo_row.addWidget(self.photo_label, 1)
        form = QFormLayout()
        form.addRow("Chart", self.chart_combo)
        self.remove_button = QPushButton("Remove Calibration")
        self.remove_button.setToolTip("Analyse colours as approximate again.")
        self.remove_button.clicked.connect(self._remove)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.addButton(self.remove_button, QDialogButtonBox.ButtonRole.ResetRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.current_label)
        layout.addWidget(instructions)
        layout.addWidget(self.same_photo)
        layout.addLayout(photo_row)
        layout.addLayout(form)
        layout.addWidget(self.mark_button)
        layout.addWidget(self.result_label)
        layout.addWidget(self.patch_table, 1)
        layout.addWidget(self.error_label)
        layout.addWidget(buttons)
        self._show_profile()
        width = self.fontMetrics().averageCharWidth() * 90
        self.resize(width, int(width * 0.9))

    # ---------------------------------------------------------------- state

    def profile(self) -> CalibrationProfile | None:
        return None if self._removed else self._profile

    def removed(self) -> bool:
        return self._removed

    def _show_profile(self) -> None:
        if self._removed or self._profile is None:
            self.current_label.setText("<b>No calibration</b>: colours are analysed as approximate.")
            self.patch_table.setRowCount(0)
            return
        p = self._profile
        ok = p.loo_mean_delta_e <= self._cfg.max_calibration_delta_e
        verdict = "usable" if ok else f"<b>not used</b> (limit ΔE00 {self._cfg.max_calibration_delta_e:.1f})"
        self.current_label.setText(f"<b>Profile</b>: {p.describe()} - {verdict}")
        self.patch_table.setRowCount(len(p.patch_delta_e))
        names = self._chart.names if len(self._chart.names) == len(p.patch_delta_e) else ()
        for row, value in enumerate(p.patch_delta_e):
            self.patch_table.setItem(row, 0, QTableWidgetItem(names[row] if names else str(row + 1)))
            item = QTableWidgetItem(f"{value:.1f}")
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.patch_table.setItem(row, 1, item)

    # ---------------------------------------------------------------- inputs

    def _choose_photo(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Chart Photo", "", "Photos (*.jpg *.jpeg *.png)")
        if path:
            self.use_chart_photo(Path(path))

    def use_chart_photo(self, path: Path) -> bool:
        try:
            self._chart_photo = load_image(path, ImageRole.PRINTED, self._loading)
        except ImageLoadError as exc:
            self.error_label.setText(str(exc))
            return False
        self.photo_label.setText(self._chart_photo.path.name)
        self.separate_photo.setChecked(True)
        self.error_label.setText("")
        return True

    def _on_chart_chosen(self, index: int) -> None:
        if self.chart_combo.itemData(index) != "file":
            self._chart = COLORCHECKER_24
            return
        path, _ = QFileDialog.getOpenFileName(self, "Chart Reference", "", "Chart reference (*.json)")
        if not path or not self.use_chart_reference(Path(path)):
            self.chart_combo.setCurrentIndex(0)
            self._chart = COLORCHECKER_24

    def use_chart_reference(self, path: Path) -> bool:
        try:
            self._chart = load_chart_reference(path)
        except CalibrationError as exc:
            self.error_label.setText(str(exc))
            return False
        self.chart_combo.insertItem(0, self._chart.name, "loaded")
        self.chart_combo.setCurrentIndex(0)
        self.error_label.setText("")
        return True

    def _source(self) -> tuple[LoadedImage | None, str]:
        if self.same_photo.isChecked():
            return self._current_photo, SAME_PHOTO
        return self._chart_photo, SEPARATE_PHOTO

    # ---------------------------------------------------------------- calibration

    def create_corner_dialog(self) -> CornerSelectionDialog | None:
        photo, _ = self._source()
        if photo is None:
            self.error_label.setText("Load a chart photo first.")
            return None
        dialog = CornerSelectionDialog(bgr8_to_qimage(photo.working), (photo.width, photo.height), self._ui, parent=self)
        dialog.setWindowTitle("Mark the Colour Chart's Patch Field")
        return dialog

    def _mark_corners(self) -> None:
        dialog = self.create_corner_dialog()
        if dialog is not None and dialog.exec() == QDialog.DialogCode.Accepted and dialog.corners() is not None:
            self.calibrate_with(dialog.corners())

    def calibrate_with(self, corners: np.ndarray) -> bool:
        """Calibrate from the chosen photo with the chart's corners (photo pixels). Shows the result."""
        photo, source = self._source()
        if photo is None:
            self.error_label.setText("Load a chart photo first.")
            return False
        try:
            profile = calibrate(photo.working, corners, self._chart, source, photo.path.name)
        except CalibrationError as exc:
            self.error_label.setText(str(exc))
            return False
        self._profile = profile
        self._removed = False
        self.error_label.setText("")
        ok = profile.loo_mean_delta_e <= self._cfg.max_calibration_delta_e
        self.result_label.setText(
            ("Calibration accepted. " if ok else "This calibration is NOT accurate enough and will not be used. ")
            + f"Fit ΔE00 {profile.fit_mean_delta_e:.1f} mean; expected on other colours (leave-one-out) "
            f"{profile.loo_mean_delta_e:.1f} mean, {profile.loo_max_delta_e:.1f} max. " + " ".join(profile.notes)
        )
        self._show_profile()
        return True

    def _remove(self) -> None:
        self._removed = True
        self.result_label.setText("The calibration will be removed when you press OK.")
        self._show_profile()
