"""Settings dialog: the acceptance criteria and inspection settings operators may change.

Only the settings listed in ``SETTING_FIELDS`` are shown; everything else
stays as configured in the settings file. The result is a set of overrides of
the bundled defaults (values equal to the default are left out), validated
with the same rules as the settings file before the dialog closes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from print_inspector.config import (
    SENSITIVITY_LEVELS,
    SEVERITY_LEVELS,
    AppConfig,
    ConfigError,
    config_with_overrides,
    load_config,
)


@dataclass(frozen=True)
class SettingField:
    section: str
    key: str
    label: str
    help: str
    kind: str  # "choice", "int", "float", "bool" or "text"
    choices: tuple[str, ...] = ()
    minimum: float = 0.0
    maximum: float = 1000.0
    decimals: int = 2
    suffix: str = ""


GROUPS: tuple[tuple[str, tuple[SettingField, ...]], ...] = (
    ("Acceptance criteria (verdict)", (
        SettingField("decision", "fail_min_severity", "Reject (FAIL) from severity",
                     "A reliably detected or confirmed defect of this severity or worse fails the label.",
                     "choice", SEVERITY_LEVELS),
        SettingField("decision", "max_minor_defects", "Minor defects allowed",
                     "More reliable minor defects than this fail the label.", "int", maximum=1000),
        SettingField("decision", "allow_pass_with_warnings", "Allow PASS with capture warnings",
                     "If off, a label with capture or alignment warnings (not critical problems) gets REVIEW "
                     "instead of PASS.", "bool"),
        SettingField("decision", "critical_region_severity", "Severity in critical regions at least",
                     "Defects and colour findings touching a critical region are at least this severe; uncertain "
                     "ones there always give REVIEW.", "choice", SEVERITY_LEVELS),
    )),
    ("Defect severity", (
        SettingField("defects", "severity_major_size_mm", "Major from length",
                     "Defects at least this long are major (unless their area makes them worse).",
                     "float", minimum=0.01, maximum=100, suffix=" mm"),
        SettingField("defects", "severity_critical_size_mm", "Critical from length",
                     "Defects at least this long are critical.", "float", minimum=0.01, maximum=500, suffix=" mm"),
        SettingField("defects", "severity_major_area_mm2", "Major from area",
                     "Defects with at least this area are major.", "float", minimum=0.001, maximum=1000,
                     decimals=3, suffix=" mm²"),
        SettingField("defects", "severity_critical_area_mm2", "Critical from area",
                     "Defects with at least this area are critical.", "float", minimum=0.001, maximum=10000,
                     decimals=3, suffix=" mm²"),
        SettingField("defects", "text_major_size_mm", "Major from length in text",
                     "Stricter limit inside text-like areas (a broken character changes meaning).",
                     "float", minimum=0.01, maximum=100, suffix=" mm"),
        SettingField("defects", "faint_difference", "Faint difference below",
                     "Defects whose largest difference is below this are rated one level lower.",
                     "float", minimum=1, maximum=255, decimals=0, suffix=" gray levels"),
    )),
    ("Defect grouping and noise", (
        SettingField("defects", "merge_distance_mm", "Merge regions closer than",
                     "Difference regions closer than this are reported as one defect.",
                     "float", minimum=0.01, maximum=50, suffix=" mm"),
        SettingField("defects", "noise_max_area_mm2", "Ignore as noise up to",
                     "Regions this small are ignored only if they are also weak and not reliably detected.",
                     "float", minimum=0.0001, maximum=10, decimals=4, suffix=" mm²"),
    )),
    ("Text inspection (OCR)", (
        SettingField("text", "enabled", "Inspect text with OCR",
                     "Read the text of the artwork and the print with Tesseract and compare it.", "bool"),
        SettingField("text", "required", "REVIEW if OCR is not available",
                     "Without a working OCR engine the label cannot be passed automatically.", "bool"),
        SettingField("text", "min_word_confidence", "Minimum OCR confidence",
                     "OCR readings below this confidence never count as a defect on their own (REVIEW).",
                     "float", minimum=0, maximum=100, decimals=0),
        SettingField("text", "critical_min_confidence", "Minimum OCR confidence, critical text",
                     "The same for critical text zones and (if enabled) lines with numbers.",
                     "float", minimum=0, maximum=100, decimals=0),
        SettingField("text", "partial_min_fraction", "Partial character from ink missing",
                     "Share of a character's ink that must be missing to report it.", "float",
                     minimum=0.01, maximum=1, decimals=2),
        SettingField("text", "critical_partial_min_fraction", "Same, in critical text",
                     "Stricter limit for critical text.", "float", minimum=0.01, maximum=1, decimals=2),
        SettingField("text", "numbers_are_critical", "Treat lines with numbers as critical",
                     "Lot numbers, dates, prices and doses are checked strictly.", "bool"),
        SettingField("text", "languages", "OCR languages",
                     "Tesseract language codes, e.g. eng or eng+deu (the .traineddata files must be installed).",
                     "text"),
        SettingField("text", "tesseract_path", "Tesseract folder",
                     "Empty: use the bundled copy, the standard install folder or the PATH.", "text"),
    )),
    ("Barcode inspection", (
        SettingField("barcodes", "enabled", "Inspect barcodes and QR codes",
                     "Decode EAN-13, UPC-A, Code 128 and QR codes with the local ZXing-C++ library.", "bool"),
        SettingField("barcodes", "required", "REVIEW if the barcode library is missing",
                     "Without the library the label cannot be passed automatically.", "bool"),
        SettingField("barcodes", "min_scan_success", "Strips that must decode",
                     "Share of horizontal strips across a linear barcode that must decode on their own.",
                     "float", minimum=0, maximum=1, decimals=2),
        SettingField("barcodes", "min_unused_ec", "QR error correction left unused",
                     "Share of a QR code's error correction that must remain unused.",
                     "float", minimum=0, maximum=1, decimals=2),
        SettingField("barcodes", "max_bar_growth_modules", "Bar width change accepted",
                     "Average bars wider or narrower than in the artwork, in modules.",
                     "float", minimum=0.01, maximum=5, decimals=2, suffix=" module"),
    )),
    ("Colour analysis", (
        SettingField("color", "enabled", "Analyse colour differences",
                     "CIELAB / ΔE00 analysis of flat colour areas, separate from the structural inspection.", "bool"),
        SettingField("color", "local_delta_e", "Local colour change from",
                     "ΔE00 between a spot and the same artwork colour elsewhere on the label.",
                     "float", minimum=0.5, maximum=50, decimals=1),
        SettingField("color", "patch_tolerance_delta_e", "Colour tolerance (calibrated)",
                     "ΔE00 from the artwork colour accepted when the camera is calibrated.",
                     "float", minimum=0.5, maximum=50, decimals=1),
        SettingField("color", "patch_critical_delta_e", "Critical colour deviation (calibrated)",
                     "ΔE00 from the artwork colour that is critical.", "float", minimum=0.5, maximum=80, decimals=1),
        SettingField("color", "approximate_review_delta_e", "Uncalibrated: REVIEW from",
                     "Without calibration only deviations this large are reported, as uncertain (REVIEW).",
                     "float", minimum=1, maximum=80, decimals=1),
        SettingField("color", "max_calibration_delta_e", "Calibration accepted up to",
                     "Largest leave-one-out mean ΔE00 of a colour calibration that is still used.",
                     "float", minimum=0.5, maximum=20, decimals=1),
    )),
    ("Inspection", (
        SettingField("inspection", "sensitivity", "Default sensitivity",
                     "Sensitivity used when the application starts.", "choice", SENSITIVITY_LEVELS),
        SettingField("capture_quality", "assumed_label_long_side_mm", "Label length if unknown",
                     "Used for sizes in mm when the artwork has no usable DPI (sizes are then marked '~').",
                     "float", minimum=10, maximum=500, decimals=0, suffix=" mm"),
    )),
    ("Reports", (
        SettingField("report", "pdf", "Write a PDF report", "Save report.pdf with the results.", "bool"),
        SettingField("report", "title", "Report title", "Heading of the PDF report.", "text"),
        SettingField("report", "organisation", "Organisation", "Printed above the title (optional).", "text"),
        SettingField("report", "inspector", "Inspector", "Name printed with the operator decisions (optional).",
                     "text"),
        SettingField("report", "max_evidence_crops", "Defect crops in the PDF",
                     "At most this many defect close-ups in the PDF; all are saved as PNG files.", "int", maximum=500),
    )),
)
SETTING_FIELDS: tuple[SettingField, ...] = tuple(f for _, group in GROUPS for f in group)


def config_value(config: AppConfig, f: SettingField) -> Any:
    return getattr(getattr(config, f.section), f.key)


def merge_values(
    overrides: Mapping[str, Any], values: Mapping[tuple[str, str], Any], defaults: AppConfig
) -> dict[str, Any]:
    """``overrides`` with the dialog ``values`` applied; values equal to the default are removed."""
    result = {k: dict(v) if isinstance(v, Mapping) else v for k, v in overrides.items()}
    for f in SETTING_FIELDS:
        value = values[(f.section, f.key)]
        section = result.setdefault(f.section, {})
        if value == config_value(defaults, f):
            section.pop(f.key, None)
        else:
            section[f.key] = value
        if not section:
            result.pop(f.section)
    return result


class SettingsDialog(QDialog):
    """Edit the operator-facing settings; ``result_config()``/``overrides()`` after acceptance."""

    def __init__(
        self,
        current: AppConfig,
        overrides: Mapping[str, Any],
        settings_path: Path | None,
        parent: QWidget | None = None,
        notice: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self._defaults = load_config(user_path=None)
        self._base_overrides = dict(overrides)
        self._result: AppConfig | None = None
        self._overrides: dict[str, Any] | None = None
        self.editors: dict[tuple[str, str], QWidget] = {}

        body = QWidget()
        body_layout = QVBoxLayout(body)
        for title, group_fields in GROUPS:
            box = QGroupBox(title)
            form = QFormLayout(box)
            for f in group_fields:
                editor = self._make_editor(f)
                editor.setToolTip(f.help)
                label = QLabel(f.label)
                label.setToolTip(f.help)
                form.addRow(label, editor)
                self.editors[(f.section, f.key)] = editor
            body_layout.addWidget(box)
        body_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)

        where = (
            f"Saved for this Windows user in:\n{settings_path}" if settings_path is not None
            else "Applies to this session only."
        )
        info = QLabel(
            "Changes to defect settings re-run the comparison, which clears operator marks; changes to the "
            "text settings re-run the text inspection. "
            f"Other settings stay as in the settings file.\n{where}"
        )
        info.setWordWrap(True)
        info.setObjectName("sideNote")
        self.notice_label = QLabel(notice or "")
        self.notice_label.setWordWrap(True)
        self.notice_label.setStyleSheet("color: #92400e;")
        self.notice_label.setVisible(bool(notice))
        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #b91c1c; font-weight: bold;")
        self.error_label.hide()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.defaults_button = QPushButton("Restore Defaults")
        self.defaults_button.setToolTip("Show the default values (nothing is saved until you press OK).")
        buttons.addButton(self.defaults_button, QDialogButtonBox.ButtonRole.ResetRole)
        self.defaults_button.clicked.connect(lambda: self.set_values(self._defaults))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.notice_label)
        layout.addWidget(scroll, 1)
        layout.addWidget(info)
        layout.addWidget(self.error_label)
        layout.addWidget(buttons)
        self.set_values(current)
        width = self.fontMetrics().averageCharWidth() * 80
        self.resize(width, int(width * 1.1))

    def _make_editor(self, f: SettingField) -> QWidget:
        if f.kind == "choice":
            combo = QComboBox()
            for choice in f.choices:
                combo.addItem(choice.capitalize(), choice)
            return combo
        if f.kind == "bool":
            return QCheckBox()
        if f.kind == "text":
            return QLineEdit()
        if f.kind == "int":
            spin = QSpinBox()
            spin.setRange(int(f.minimum), int(f.maximum))
            spin.setSuffix(f.suffix)
            return spin
        spin = QDoubleSpinBox()
        spin.setDecimals(f.decimals)
        spin.setRange(f.minimum, f.maximum)
        spin.setSingleStep(10 ** -f.decimals * 10 if f.decimals else 1)
        spin.setSuffix(f.suffix)
        return spin

    def set_values(self, config: AppConfig) -> None:
        for f in SETTING_FIELDS:
            editor = self.editors[(f.section, f.key)]
            value = config_value(config, f)
            if isinstance(editor, QComboBox):
                editor.setCurrentIndex(editor.findData(value))
            elif isinstance(editor, QCheckBox):
                editor.setChecked(bool(value))
            elif isinstance(editor, QLineEdit):
                editor.setText(str(value))
            else:
                editor.setValue(value)

    def values(self) -> dict[tuple[str, str], Any]:
        out: dict[tuple[str, str], Any] = {}
        for f in SETTING_FIELDS:
            editor = self.editors[(f.section, f.key)]
            if isinstance(editor, QComboBox):
                out[(f.section, f.key)] = editor.currentData()
            elif isinstance(editor, QCheckBox):
                out[(f.section, f.key)] = editor.isChecked()
            elif isinstance(editor, QLineEdit):
                out[(f.section, f.key)] = editor.text().strip()
            elif isinstance(editor, QSpinBox):
                out[(f.section, f.key)] = int(editor.value())
            else:
                out[(f.section, f.key)] = round(float(editor.value()), f.decimals)
        return out

    def accept(self) -> None:  # noqa: D102 - validates before closing
        overrides = merge_values(self._base_overrides, self.values(), self._defaults)
        try:
            config = config_with_overrides(overrides)
        except ConfigError as exc:
            self.error_label.setText(f"These settings cannot be used: {exc}")
            self.error_label.show()
            return
        self._overrides = overrides
        self._result = config
        super().accept()

    def result_config(self) -> AppConfig | None:
        return self._result

    def overrides(self) -> dict[str, Any] | None:
        return self._overrides
