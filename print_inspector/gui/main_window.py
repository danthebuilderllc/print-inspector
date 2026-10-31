"""Main application window.

Layout
    * toolbars: Load Original, Load Printed, Compare, Manual Alignment, Reset,
      Save Results, Settings and sensitivity; view controls;
    * three viewers with synchronised zoom and pan: original artwork, aligned
      print, difference result;
    * side panel (dockable): the defect list with position, size, score,
      severity and operator status, the selected defect side by side, and the
      details of every check;
    * status area: capture quality, alignment quality, progress and the
      PASS / FAIL / REVIEW verdict.

All image processing runs in a background thread pool (``gui/workers.py``)
using the functions in ``print_inspector.core``; this module only starts the
work, keeps the results and displays them. Results of superseded work are
discarded using generation counters.
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSettings, Qt, QThreadPool, QUrl, Signal
from PySide6.QtGui import QAction, QBrush, QCloseEvent, QColor, QGuiApplication, QImage, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from print_inspector.config import (
    SENSITIVITY_LEVELS,
    AppConfig,
    ConfigError,
    config_data,
    config_overrides,
    config_with_overrides,
    parse_config,
    read_user_overrides,
    write_user_overrides,
)
from print_inspector.core.alignment import AlignmentMethod, AlignmentResult
from print_inspector.core.capture_quality import QualityReport, artwork_label_long_side_mm
from print_inspector.core.comparison import ComparisonResult
from print_inspector.core.decision import InspectionDecision, OperatorMark, alignment_blocks, decide
from print_inspector.core.defects import Defect, DefectReport, grid_px_per_mm
from print_inspector.core.image_io import ImageRole, LoadedImage, allowed_extensions
from print_inspector.core.label_detection import snap_corners_to_edges
from print_inspector.core.models import InspectionStatus, QualityLevel, Severity
from print_inspector.core.registration import ManualAdjustment, RegistrationResult, comparison_grid
from print_inspector.core.barcode_inspection import BarcodeFinding, BarcodeReport
from print_inspector.core.barcode_zones import BarcodeZone, BarcodeZoneError
from print_inspector.core.barcode_zones import load_zones as load_barcode_zones
from print_inspector.core.barcode_zones import save_zones as save_barcode_zones
from print_inspector.core.barcode_zones import zones_path as barcode_zones_path
from print_inspector.core.color_calibration import CalibrationError, CalibrationProfile, load_profile, save_profile
from print_inspector.core.color_inspection import ColorFinding, ColorReport, heatmap_image
from print_inspector.core import profiles as inspection_profiles
from print_inspector.core.ocr import OcrEngine
from print_inspector.core.profiles import InspectionProfile, ProfileError
from print_inspector.core.regions import (
    InspectionRegion,
    RegionError,
    RegionKind,
    load_regions,
    regions_path,
    save_regions,
)
from print_inspector.core.results_export import InspectionRecord, defect_position_mm
from print_inspector.core.text_inspection import TextFinding, TextReport
from print_inspector.core.text_zones import TextZone, TextZoneError, load_zones, save_zones, zones_path
from print_inspector.gui.barcode_panel import BarcodePanel
from print_inspector.gui.barcode_zone_dialog import BarcodeZoneDialog
from print_inspector.gui.calibration_dialog import CalibrationDialog
from print_inspector.gui.check_section import (
    CheckSection,
    describe_image,
    format_alignment_metrics,
    format_comparison_metrics,
    format_quality_metrics,
    format_registration_metrics,
    opt,
)
from print_inspector.gui.color_panel import ColorPanel
from print_inspector.gui.comparison_view import COMPARISON_VIEWS, render_comparison
from print_inspector.gui.corner_dialog import CornerSelectionDialog
from print_inspector.gui.defect_detail import DefectDetailWidget
from print_inspector.gui.image_panel import ImagePanel, ImageView
from print_inspector.gui.overlay import OVERLAY_MODES, make_overlay
from print_inspector.gui.profile_dialog import ProfileDialog
from print_inspector.gui.region_dialog import RegionDialog
from print_inspector.gui.registration_dialog import RegistrationAdjustDialog
from print_inspector.gui.settings_dialog import SettingsDialog
from print_inspector.gui.status_area import StatusArea
from print_inspector.gui.text_panel import TextPanel
from print_inspector.gui.text_zone_dialog import TextZoneDialog
from print_inspector.gui.view_sync import ViewSynchronizer
from print_inspector.gui.workers import (
    AlignmentWorker,
    BarcodeInspectionWorker,
    ColorAnalysisWorker,
    ComparisonWorker,
    DisplayableImage,
    ImageLoadWorker,
    QualityWorker,
    RegistrationWorker,
    SaveResultsWorker,
    TextInspectionWorker,
    bgr8_to_qimage,
)

log = logging.getLogger(__name__)

DEFECT_COLUMNS = ("#", "Category", "Severity", "Operator", "Position (mm)", "Size (mm)", "Score", "Detection")
COL_OPERATOR = DEFECT_COLUMNS.index("Operator")
_BOX_COLOURS = {
    "active": "#ef4444", "dismissed": "#9ca3af", "selected": "#facc15", "text": "#f97316", "barcode": "#a855f7",
    "color": "#14b8a6", "critical": "#dc2626", "ignored": "#6b7280",
}
COLOR_VIEWS = {"color_artwork": "Colour ΔE00 vs artwork", "color_local": "Colour change (local)"}
VIEW_REGISTERED = "registered"
VIEW_ALIGNED = "aligned"
VIEW_PHOTO = "photo"
WORKER_STACK_BYTES = 16 * 1024 * 1024
CLICK_PADDING_SCREEN_PX = 6  # a click this close to a defect box selects it

MARK_SHORTCUTS = {
    OperatorMark.CONFIRMED: "C",
    OperatorMark.DISMISSED: "D",
    OperatorMark.NEEDS_REVIEW: "R",
    OperatorMark.UNREVIEWED: "U",
}
MOUSE_HELP = (
    ("Mouse wheel", "Zoom in or out at the pointer"),
    ("Drag", "Pan"),
    ("Shift + mouse wheel", "Scroll sideways"),
    ("Click a defect (red, n), text (orange, Tn), barcode (purple, Bn) or colour (teal, Cn) box",
     "Select it and zoom to it"),
    ("Drop an image on a viewer", "Load it as the original (1) or printed (2) image"),
)

_STYLE = """
#imagePanel { background: #f4f5f7; border: 1px solid #d0d4da; border-radius: 6px; }
#panelTitle { font-size: 11pt; font-weight: bold; color: #1f2937; }
#panelZoom { color: #4b5563; }
#panelInfo { color: #374151; }
#panelWarning { color: #92400e; background: #fef3c7; border-radius: 4px; padding: 4px; }
#panelPlaceholder { color: #6b7280; font-size: 10pt; background: #e5e7eb; border-radius: 4px; }
#sideHeading { font-weight: bold; color: #1f2937; margin-top: 6px; }
#sideNote { color: #6b7280; }
#metrics { color: #374151; font-family: Consolas, "Courier New", monospace; }
#statusArea { background: #f9fafb; border: 1px solid #d0d4da; border-radius: 6px; }
#statusCaption { color: #4b5563; font-weight: bold; }
#statusDetail { color: #374151; }
#banner_error { background: #fee2e2; color: #7f1d1d; border: 1px solid #fca5a5; border-radius: 4px; }
#banner_info { background: #e0f2fe; color: #0c4a6e; border: 1px solid #7dd3fc; border-radius: 4px; }
#banner_warning { background: #fef3c7; color: #78350f; border: 1px solid #fcd34d; border-radius: 4px; }
"""


class MessageBanner(QWidget):
    """A dismissible one-message bar above the viewers (errors from background work, notices)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setOpenExternalLinks(True)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        close = QToolButton()
        close.setText("✕")
        close.setAutoRaise(True)
        close.setToolTip("Hide this message")
        close.clicked.connect(self.hide)
        self._frame = QWidget()
        inner = QHBoxLayout(self._frame)
        inner.setContentsMargins(8, 4, 4, 4)
        inner.addWidget(self.label, 1)
        inner.addWidget(close, 0, Qt.AlignmentFlag.AlignTop)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._frame)
        self._kind = "info"
        self.hide()

    def show_message(self, text: str, kind: str = "error") -> None:
        self._kind = kind
        self._frame.setObjectName(f"banner_{kind}")
        self._frame.style().unpolish(self._frame)  # re-apply the window style sheet for the new name
        self._frame.style().polish(self._frame)
        self.label.setText(text)
        self.show()

    def kind(self) -> str:
        return self._kind

    def text(self) -> str:
        return self.label.text()


class MainWindow(QMainWindow):
    image_loaded = Signal(object)  # ImageRole
    image_load_failed = Signal(object, str)  # ImageRole, message
    quality_assessed = Signal(object)  # QualityReport
    quality_failed = Signal(str)
    alignment_done = Signal(object)  # AlignmentResult
    alignment_failed = Signal(str)
    registration_done = Signal(object)  # RegistrationResult
    registration_failed = Signal(str)
    comparison_done = Signal(object)  # ComparisonResult
    comparison_failed = Signal(str)
    decision_changed = Signal(object)  # InspectionDecision
    text_done = Signal(object)  # TextReport
    text_failed = Signal(str)
    barcodes_done = Signal(object)  # BarcodeReport
    barcodes_failed = Signal(str)
    color_done = Signal(object)  # ColorReport
    color_failed = Signal(str)
    results_saved = Signal(object)  # Path of the results folder
    results_save_failed = Signal(str)

    def __init__(
        self,
        config: AppConfig,
        parent: QWidget | None = None,
        user_config_path: Path | None = None,
        window_settings: QSettings | None = None,
        ocr_engine: OcrEngine | None = None,
        color_profile_path: Path | None = None,
        profiles_dir: Path | None = None,
    ) -> None:
        """``ocr_engine``: use this engine for the text inspection instead of finding Tesseract.

        ``color_profile_path``: where the colour calibration profile is kept (None: the
        ``color.calibration_profile`` setting, else the calibration lasts for this session only).
        ``profiles_dir``: the inspection profile library (default: ``<user data dir>/profiles``).
        """
        super().__init__(parent)
        self._config = config
        self._base_config = config  # the settings of this computer, without an inspection profile
        self._profiles_dir = Path(profiles_dir) if profiles_dir is not None else inspection_profiles.profiles_dir()
        self._profile: InspectionProfile | None = None
        self._profile_path: Path | None = None
        self._regions: tuple[InspectionRegion, ...] = ()
        self._ocr_engine = ocr_engine
        self._injected_engine = ocr_engine is not None
        self._user_config_path = user_config_path
        self._window_settings = window_settings
        self._images: dict[ImageRole, LoadedImage | None] = {role: None for role in ImageRole}
        self._generation: dict[ImageRole, int] = {role: 0 for role in ImageRole}
        self._loading: set[ImageRole] = set()
        self._workers: dict[tuple[ImageRole, int], ImageLoadWorker] = {}
        self._quality_generation = 0
        self._quality_workers: dict[int, QualityWorker] = {}
        self._quality_report: QualityReport | None = None
        self._alignment_generation = 0
        self._alignment_workers: dict[int, AlignmentWorker] = {}
        self._alignment: AlignmentResult | None = None
        self._automatic_corners: np.ndarray | None = None
        self._manual_corners: np.ndarray | None = None
        self._registration_generation = 0
        self._registration_workers: dict[int, RegistrationWorker] = {}
        self._registration: RegistrationResult | None = None
        self._overlay_image: np.ndarray | None = None
        self._comparison_generation = 0
        self._comparison_workers: dict[int, ComparisonWorker] = {}
        self._comparison: ComparisonResult | None = None
        self._defects: DefectReport | None = None
        self._marks: dict[int, OperatorMark] = {}
        self._decision: InspectionDecision | None = None
        self._highlight: int | None = None
        self._printed_previews: dict[str, QImage] = {}
        self._printed_shown: str | None = None
        self._check_errors: dict[str, str] = {}  # stage -> reason; any entry forces REVIEW
        self._save_worker: SaveResultsWorker | None = None
        self._text_generation = 0
        self._text_workers: dict[int, TextInspectionWorker] = {}
        self._text_report: TextReport | None = None
        self._text_marks: dict[int, OperatorMark] = {}
        self._text_highlight: int | None = None
        self._text_zones: tuple[TextZone, ...] = ()
        self._barcode_generation = 0
        self._barcode_workers: dict[int, BarcodeInspectionWorker] = {}
        self._barcode_report: BarcodeReport | None = None
        self._barcode_marks: dict[int, OperatorMark] = {}
        self._barcode_highlight: int | None = None
        self._barcode_zones: tuple[BarcodeZone, ...] = ()
        self._color_generation = 0
        self._color_workers: dict[int, ColorAnalysisWorker] = {}
        self._color_report: ColorReport | None = None
        self._color_marks: dict[int, OperatorMark] = {}
        self._color_highlight: int | None = None
        if color_profile_path is None and config.color.calibration_profile:
            color_profile_path = Path(config.color.calibration_profile)
        self._color_profile_path = color_profile_path
        self._color_profile: CalibrationProfile | None = None
        self._selected_kind = "defect"  # what the keyboard marks act on: "defect", "text", "barcode" or "color"
        self._results_dir: Path | None = None
        # Own pool with an explicit stack size: NumPy's LAPACK routines overflow the small
        # default thread stacks (512 KB on macOS, 1 MB on Windows) and crash the process.
        self._thread_pool = QThreadPool(self)
        self._thread_pool.setStackSize(WORKER_STACK_BYTES)
        self._last_dir = Path.home()
        self._sync = ViewSynchronizer(self)

        self.setWindowTitle(f"{config.app.name} {config.app.version}")
        self.setStyleSheet(_STYLE)
        self._create_actions()
        self._create_central_widget()
        self._create_side_panel()
        self._create_toolbars()
        self._create_menus()
        self._set_initial_geometry()
        self.statusBar().showMessage("Load the original artwork and a photo of the printed label.")
        self._load_color_profile()
        self._update_status()

    # ------------------------------------------------------------------ UI setup

    def _action(
        self, text: str, shortcut: str | QKeySequence.StandardKey | None = None, slot=None,
        tip: str = "", icon: QStyle.StandardPixmap | None = None, checkable: bool = False,
    ) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        if icon is not None:
            action.setIcon(self.style().standardIcon(icon))
        if tip:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        action.setCheckable(checkable)
        if slot is not None:
            action.triggered.connect(slot)
        self.addAction(action)  # shortcuts work even when the action is only in a menu
        return action

    def _create_actions(self) -> None:
        SP = QStyle.StandardPixmap
        self.load_original_action = self._action(
            "Load Original…", "Ctrl+O", lambda: self._choose_file(ImageRole.ORIGINAL),
            "Load the original digital label artwork (PNG, JPEG, BMP, TIFF).", SP.SP_DirOpenIcon,
        )
        self.load_printed_action = self._action(
            "Load Printed…", "Ctrl+Shift+O", lambda: self._choose_file(ImageRole.PRINTED),
            "Load a smartphone photo of the printed label (JPEG, PNG).", SP.SP_DialogOpenButton,
        )
        self.compare_action = self._action(
            "Compare", "F5", self.compare,
            "Run the inspection, or compare again with the current sensitivity.", SP.SP_MediaPlay,
        )
        self.reset_action = self._action(
            "Reset", "Ctrl+R", self.reset, "Clear both images and all results.", SP.SP_BrowserReload,
        )
        self.save_action = self._action(
            "Save Results…", QKeySequence.StandardKey.Save, self.save_results,
            "Save the verdict, defect list and images to a folder.", SP.SP_DialogSaveButton,
        )
        self.settings_action = self._action(
            "Settings…", "Ctrl+,", self.open_settings,
            "Acceptance criteria, severity limits and other settings.", SP.SP_FileDialogDetailedView,
        )
        self.detect_action = self._action(
            "Detect Label Again", "Ctrl+D", self.detect_label,
            "Find the label automatically and correct perspective (discards manual corners).",
        )
        self.manual_corners_action = self._action(
            "Mark Label Corners…", "Ctrl+M", self._open_manual_corners,
            "Mark or correct the label's four corners by hand.",
        )
        self.adjust_registration_action = self._action(
            "Adjust Registration…", "Ctrl+J", self._open_registration_dialog,
            "Fine-tune how the photo lies on the artwork by hand.",
        )
        self.zoom_in_action = self._action("Zoom In", None, self.zoom_in, "Zoom in (mouse wheel up)")
        self.zoom_in_action.setShortcuts([QKeySequence("Ctrl++"), QKeySequence("Ctrl+=")])
        self.zoom_out_action = self._action("Zoom Out", "Ctrl+-", self.zoom_out, "Zoom out (mouse wheel down)")
        self.fit_action = self._action("Fit to Window", "Ctrl+0", self.fit_to_window, "Show the whole image")
        self.actual_size_action = self._action(
            "Actual Pixels", "Ctrl+1", self.actual_size, "One image pixel per screen pixel"
        )
        self.sync_action = self._action(
            "Synchronize Views", "Ctrl+L", None,
            "Zoom and pan all viewers together on the same part of the label.", checkable=True,
        )
        self.sync_action.setChecked(True)
        self.sync_action.toggled.connect(self._sync.set_enabled)
        self.show_defects_action = self._action(
            "Show Defect Markers", "B", None, "Show or hide the numbered defect boxes.", checkable=True,
        )
        self.show_defects_action.setChecked(True)
        self.show_defects_action.toggled.connect(lambda _: self._refresh_defect_markers())
        self.next_defect_action = self._action("Next Defect", "N", lambda: self.step_defect(1))
        self.previous_defect_action = self._action("Previous Defect", "P", lambda: self.step_defect(-1))
        self.mark_actions: dict[OperatorMark, QAction] = {}
        for mark, text in (
            (OperatorMark.CONFIRMED, "Confirm Defect"),
            (OperatorMark.DISMISSED, "Dismiss Defect"),
            (OperatorMark.NEEDS_REVIEW, "Defect Needs Review"),
            (OperatorMark.UNREVIEWED, "Clear Mark"),
        ):
            self.mark_actions[mark] = self._action(
                text, MARK_SHORTCUTS[mark], lambda _=False, m=mark: self._mark_selected(m),
                "Applies to the selected defect.",
            )
        self.text_zones_action = self._action(
            "Text Zones…", "Ctrl+T", self._open_text_zones,
            "Mark critical text zones on the artwork (checked strictly, optionally against exact text).",
        )
        self.barcode_zones_action = self._action(
            "Barcode Zones…", "Ctrl+Shift+B", self._open_barcode_zones,
            "Mark where barcodes must be, with their type and expected value.",
        )
        self.color_calibration_action = self._action(
            "Colour Calibration…", "Ctrl+K", self._open_color_calibration,
            "Calibrate the camera with a photographed colour chart (otherwise colours are approximate).",
        )
        self.regions_action = self._action(
            "Inspection Regions…", "Ctrl+Shift+R", self._open_regions,
            "Mark critical regions (inspected strictly) and ignored regions (not inspected) on the artwork.",
        )
        self.profiles_action = self._action(
            "Profiles…", "Ctrl+P", self.open_profiles,
            "Load, save, import or export inspection profiles for recurring label designs.",
        )
        self.save_profile_action = self._action(
            "Save Profile", "Ctrl+Shift+S", self.save_inspection_profile,
            "Save the current settings, regions and zones to the profile in use (or as a new profile).",
        )
        self.close_profile_action = self._action(
            "Close Profile", None, self.close_inspection_profile,
            "Stop using the profile: back to this computer's settings and the zones saved with the artwork.",
        )
        self.shortcuts_action = self._action("Keyboard and Mouse…", "F1", self._show_shortcuts)
        self.about_action = self._action("About Print Inspector", None, self._show_about)
        self.exit_action = self._action("Exit", "Alt+F4", self.close)

    def _create_toolbars(self) -> None:
        toolbar = QToolBar("Main toolbar", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toolbar.addAction(self.load_original_action)
        toolbar.addAction(self.load_printed_action)
        toolbar.addSeparator()
        toolbar.addAction(self.compare_action)
        alignment_menu = QMenu("Manual Alignment", self)
        alignment_menu.addAction(self.detect_action)
        alignment_menu.addAction(self.manual_corners_action)
        alignment_menu.addAction(self.adjust_registration_action)
        self.alignment_button = QToolButton()
        self.alignment_button.setText("Manual Alignment")
        self.alignment_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogListView))
        self.alignment_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.alignment_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.alignment_button.setMenu(alignment_menu)
        self.alignment_button.setToolTip("Correct the label corners or the registration by hand.")
        toolbar.addWidget(self.alignment_button)
        toolbar.addSeparator()
        toolbar.addAction(self.reset_action)
        toolbar.addAction(self.save_action)
        toolbar.addAction(self.settings_action)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" Sensitivity: "))
        self.sensitivity_combo = QComboBox()
        for level in SENSITIVITY_LEVELS:
            self.sensitivity_combo.addItem(level.capitalize(), level)
        self.sensitivity_combo.setCurrentIndex(SENSITIVITY_LEVELS.index(self._config.inspection.sensitivity))
        self.sensitivity_combo.setToolTip(
            "Comparison sensitivity: High finds fainter and smaller differences (more candidates), "
            "Low only clear ones. Changing it compares again."
        )
        self.sensitivity_combo.currentIndexChanged.connect(lambda _: self.run_comparison())
        toolbar.addWidget(self.sensitivity_combo)
        self.addToolBar(toolbar)

        view_bar = QToolBar("View", self)
        view_bar.setObjectName("viewToolbar")
        view_bar.setMovable(False)
        view_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        for action in (self.zoom_in_action, self.zoom_out_action, self.fit_action, self.actual_size_action):
            view_bar.addAction(action)
        view_bar.addSeparator()
        view_bar.addAction(self.sync_action)
        view_bar.addAction(self.show_defects_action)
        self.addToolBar(view_bar)

    def _create_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.load_original_action)
        file_menu.addAction(self.load_printed_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.settings_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        inspect_menu = self.menuBar().addMenu("&Inspection")
        inspect_menu.addAction(self.compare_action)
        inspect_menu.addSeparator()
        inspect_menu.addAction(self.detect_action)
        inspect_menu.addAction(self.manual_corners_action)
        inspect_menu.addAction(self.adjust_registration_action)
        inspect_menu.addSeparator()
        inspect_menu.addAction(self.regions_action)
        inspect_menu.addAction(self.text_zones_action)
        inspect_menu.addAction(self.barcode_zones_action)
        inspect_menu.addAction(self.color_calibration_action)
        inspect_menu.addSeparator()
        inspect_menu.addAction(self.reset_action)

        profile_menu = self.menuBar().addMenu("&Profiles")
        profile_menu.addAction(self.profiles_action)
        profile_menu.addAction(self.save_profile_action)
        profile_menu.addAction(self.close_profile_action)

        defects_menu = self.menuBar().addMenu("&Defects")
        defects_menu.addAction(self.next_defect_action)
        defects_menu.addAction(self.previous_defect_action)
        defects_menu.addSeparator()
        for action in self.mark_actions.values():
            defects_menu.addAction(action)

        view_menu = self.menuBar().addMenu("&View")
        for action in (self.zoom_in_action, self.zoom_out_action, self.fit_action, self.actual_size_action):
            view_menu.addAction(action)
        view_menu.addSeparator()
        view_menu.addAction(self.sync_action)
        view_menu.addAction(self.show_defects_action)
        view_menu.addSeparator()
        panel_action = self.side_dock.toggleViewAction()
        panel_action.setText("Inspection Details Panel")
        panel_action.setShortcut(QKeySequence("F9"))
        view_menu.addAction(panel_action)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self.shortcuts_action)
        help_menu.addAction(self.about_action)

    def _create_central_widget(self) -> None:
        ui = self._config.ui
        self.original_panel = ImagePanel(
            "1 · Original Artwork", "Load the original digital artwork\n(Ctrl+O, or drop the file here)",
            ui, accept_drops=True,
        )
        self.printed_panel = ImagePanel(
            "2 · Aligned Print", "Load a photo of the printed label\n(Ctrl+Shift+O, or drop the file here)",
            ui, accept_drops=True,
        )
        self.original_panel.file_dropped.connect(lambda p: self.load_image_file(ImageRole.ORIGINAL, p))
        self.printed_panel.file_dropped.connect(lambda p: self.load_image_file(ImageRole.PRINTED, p))
        self.printed_view_combo = QComboBox()
        self.printed_view_combo.addItem("Registered to artwork", VIEW_REGISTERED)
        self.printed_view_combo.addItem("Perspective-corrected", VIEW_ALIGNED)
        self.printed_view_combo.addItem("Original photo", VIEW_PHOTO)
        self.printed_view_combo.setToolTip("Compare the corrected label with the photo as captured.")
        self.printed_view_combo.currentIndexChanged.connect(lambda _: self._refresh_printed_panel())
        self.printed_panel.add_header_widget(self.printed_view_combo, 1)
        self.overlay_panel = ImagePanel(
            "3 · Difference Result",
            "Differences appear here once the photo\nis registered to the artwork.",
            ui,
        )
        self.overlay_mode_combo = QComboBox()
        for key, label in COMPARISON_VIEWS.items():
            self.overlay_mode_combo.addItem(label, key)
        self.overlay_mode_combo.insertSeparator(self.overlay_mode_combo.count())
        for key, label in COLOR_VIEWS.items():
            self.overlay_mode_combo.addItem(label, key)
        self.overlay_mode_combo.insertSeparator(self.overlay_mode_combo.count())
        for key, label in OVERLAY_MODES.items():
            self.overlay_mode_combo.addItem(f"Overlay: {label}", key)
        self.overlay_mode_combo.setToolTip("Comparison result view, or an overlay of artwork and registered photo.")
        self.overlay_mode_combo.currentIndexChanged.connect(lambda _: self._refresh_overlay())
        self.overlay_opacity = QSlider(Qt.Orientation.Horizontal)
        self.overlay_opacity.setRange(0, 100)
        self.overlay_opacity.setValue(50)
        self.overlay_opacity.setMinimumWidth(60)
        self.overlay_opacity.setMaximumWidth(120)
        self.overlay_opacity.setToolTip("Display gain of difference maps / overlay opacity")
        self.overlay_opacity.valueChanged.connect(lambda _: self._refresh_overlay())
        self.overlay_panel.add_header_widget(self.overlay_mode_combo, 2)
        self.overlay_panel.add_header_widget(self.overlay_opacity, 1)
        for combo in (self.printed_view_combo, self.overlay_mode_combo):
            # Shrink with the panel instead of setting its minimum width by the longest entry.
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
        self._panels: dict[ImageRole, ImagePanel] = {
            ImageRole.ORIGINAL: self.original_panel,
            ImageRole.PRINTED: self.printed_panel,
        }
        for key, panel in (("original", self.original_panel), ("printed", self.printed_panel),
                           ("overlay", self.overlay_panel)):
            self._sync.add_view(panel.view)
            panel.view.image_clicked.connect(lambda point, k=key: self._on_view_clicked(k, point))

        self.views_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.views_splitter.setChildrenCollapsible(False)
        for panel in (self.original_panel, self.printed_panel, self.overlay_panel):
            panel.setMinimumWidth(self.fontMetrics().averageCharWidth() * 18)
            self.views_splitter.addWidget(panel)
            self.views_splitter.setStretchFactor(self.views_splitter.count() - 1, 1)

        self.banner = MessageBanner()
        self.status_area = StatusArea()
        self.status_indicator = self.status_area.verdict

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 4)
        layout.setSpacing(6)
        layout.addWidget(self.banner)
        layout.addWidget(self.views_splitter, 1)
        layout.addWidget(self.status_area)
        self.setCentralWidget(container)

    def _create_side_panel(self) -> None:
        def heading(text: str) -> QLabel:
            label = QLabel(text)
            label.setObjectName("sideHeading")
            return label

        def note(text: str) -> QLabel:
            label = QLabel(text)
            label.setObjectName("sideNote")
            label.setWordWrap(True)
            return label

        # Defects tab: verdict reasons, list, navigation, selected defect.
        self.status_reason_label = QLabel("")
        self.status_reason_label.setWordWrap(True)
        self.status_reason_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.defects_summary = note("")
        self.defect_table = QTableWidget(0, len(DEFECT_COLUMNS))
        self.defect_table.setHorizontalHeaderLabels(DEFECT_COLUMNS)
        self.defect_table.verticalHeader().setVisible(False)
        self.defect_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.defect_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.defect_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.defect_table.setAlternatingRowColors(True)
        header = self.defect_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.defect_table.setMinimumHeight(self.fontMetrics().height() * 6)
        self.defect_table.itemSelectionChanged.connect(self._on_defect_selected)
        self.previous_button = QPushButton("◀ Previous")
        self.previous_button.setToolTip("Previous defect (P)")
        self.previous_button.clicked.connect(lambda: self.step_defect(-1))
        self.next_button = QPushButton("Next ▶")
        self.next_button.setToolTip("Next defect (N)")
        self.next_button.clicked.connect(lambda: self.step_defect(1))
        nav = QHBoxLayout()
        nav.addWidget(self.previous_button)
        nav.addWidget(self.next_button)
        nav.addStretch(1)

        self.defect_detail = DefectDetailWidget()
        self.defect_detail.mark_requested.connect(self.set_mark)
        for mark, button in self.defect_detail.buttons.items():
            button.setToolTip(f"{button.toolTip()} (key {MARK_SHORTCUTS[mark]})")

        list_part = QWidget()
        list_layout = QVBoxLayout(list_part)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.addWidget(heading("Verdict"))
        list_layout.addWidget(self.status_reason_label)
        list_layout.addWidget(heading("Defects (click a row or a box to inspect)"))
        list_layout.addWidget(self.defects_summary)
        list_layout.addWidget(self.defect_table, 1)
        list_layout.addLayout(nav)
        detail_part = QWidget()
        detail_layout = QVBoxLayout(detail_part)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.addWidget(heading("Selected defect"))
        detail_layout.addWidget(self.defect_detail)
        detail_layout.addWidget(note(
            "Categories marked 'possible' or 'uncertain' are not verified. Confirm, dismiss or flag "
            "defects for review; your marks override the automatic assessment. The verdict follows "
            "the acceptance criteria in Settings."
        ))
        detail_layout.addStretch(1)
        detail_scroll = QScrollArea()
        detail_scroll.setWidget(detail_part)
        detail_scroll.setWidgetResizable(True)
        detail_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        defects_split = QSplitter(Qt.Orientation.Vertical)
        defects_split.setChildrenCollapsible(False)
        defects_split.addWidget(list_part)
        defects_split.addWidget(detail_scroll)
        defects_split.setStretchFactor(0, 1)
        defects_split.setStretchFactor(1, 1)

        # Checks tab: details of every check stage.
        self.quality_section = CheckSection("Capture quality")
        self.alignment_section = CheckSection("Alignment")
        self.registration_section = CheckSection("Registration")
        self.comparison_section = CheckSection("Comparison")
        self.text_section = CheckSection("Text (OCR)")
        self.barcode_section = CheckSection("Barcodes")
        self.color_section = CheckSection("Colour")
        checks = QWidget()
        checks_layout = QVBoxLayout(checks)
        for title, section in (
            ("Capture quality (printed photo)", self.quality_section),
            ("Label detection and alignment", self.alignment_section),
            ("Registration to artwork", self.registration_section),
            ("Comparison with artwork", self.comparison_section),
            ("Text inspection (OCR)", self.text_section),
            ("Barcode inspection", self.barcode_section),
            ("Colour analysis", self.color_section),
        ):
            checks_layout.addWidget(heading(title))
            checks_layout.addWidget(section)
        checks_layout.addStretch(1)
        checks_scroll = QScrollArea()
        checks_scroll.setWidget(checks)
        checks_scroll.setWidgetResizable(True)

        self.side_tabs = QTabWidget()
        self.text_panel = TextPanel()
        self.text_panel.finding_selected.connect(self.select_text_finding)
        self.text_panel.line_selected.connect(self.show_text_line)
        self.text_panel.mark_requested.connect(self.set_text_mark)
        self.text_panel.edit_zones_requested.connect(self._open_text_zones)
        for mark, button in self.text_panel.buttons.items():
            button.setToolTip(f"Key {MARK_SHORTCUTS[mark]}")
        self.barcode_panel = BarcodePanel()
        self.barcode_panel.finding_selected.connect(self.select_barcode_finding)
        self.barcode_panel.symbol_selected.connect(self.show_barcode_symbol)
        self.barcode_panel.mark_requested.connect(self.set_barcode_mark)
        self.barcode_panel.edit_zones_requested.connect(self._open_barcode_zones)
        for mark, button in self.barcode_panel.buttons.items():
            button.setToolTip(f"Key {MARK_SHORTCUTS[mark]}")
        self.side_tabs.addTab(defects_split, "Defects")
        self.side_tabs.addTab(self.text_panel, "Text")
        self.color_panel = ColorPanel()
        self.color_panel.finding_selected.connect(self.select_color_finding)
        self.color_panel.area_selected.connect(self.show_color_area)
        self.color_panel.mark_requested.connect(self.set_color_mark)
        self.color_panel.calibration_requested.connect(self._open_color_calibration)
        for mark, button in self.color_panel.buttons.items():
            button.setToolTip(f"Key {MARK_SHORTCUTS[mark]}")
        self.side_tabs.addTab(self.barcode_panel, "Barcodes")
        self.side_tabs.addTab(self.color_panel, "Colour")
        self.side_tabs.addTab(checks_scroll, "Checks")
        self.side_dock = QDockWidget("Inspection Details", self)
        self.side_dock.setObjectName("inspectionDetails")
        self.side_dock.setWidget(self.side_tabs)
        self.side_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.side_dock.setMinimumWidth(self.fontMetrics().averageCharWidth() * 32)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.side_dock)

    def _set_initial_geometry(self) -> None:
        """Configured size, limited to the screen; restore the last layout if one was saved."""
        settings = self._window_settings
        if settings is not None and settings.value("geometry") is not None:
            self.restoreGeometry(settings.value("geometry"))
            if settings.value("state") is not None:
                self.restoreState(settings.value("state"))
            return
        screen = self.screen() or QGuiApplication.primaryScreen()
        width, height = self._config.ui.window_width, self._config.ui.window_height
        if screen is not None:
            available = screen.availableGeometry()
            width = min(width, int(available.width() * 0.95))
            height = min(height, int(available.height() * 0.92))
        self.resize(width, height)
        self._size_dock_on_show = True

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        if getattr(self, "_size_dock_on_show", False):  # dock sizes only take effect once laid out
            self._size_dock_on_show = False
            self.resizeDocks(
                [self.side_dock], [max(self.side_dock.minimumWidth(), int(self.width() * 0.28))],
                Qt.Orientation.Horizontal,
            )

    # ------------------------------------------------------------------ state

    def loaded_image(self, role: ImageRole) -> LoadedImage | None:
        return self._images[role]

    def quality_report(self) -> QualityReport | None:
        return self._quality_report

    def alignment_result(self) -> AlignmentResult | None:
        return self._alignment

    def registration_result(self) -> RegistrationResult | None:
        return self._registration

    def comparison_result(self) -> ComparisonResult | None:
        return self._comparison

    def defect_report(self) -> DefectReport | None:
        return self._defects

    def decision(self) -> InspectionDecision | None:
        return self._decision

    def marks(self) -> dict[int, OperatorMark]:
        return dict(self._marks)

    def selected_defect(self) -> Defect | None:
        return self._defects.by_id(self._highlight) if self._defects and self._highlight is not None else None

    def text_report(self) -> TextReport | None:
        return self._text_report

    def text_marks(self) -> dict[int, OperatorMark]:
        return dict(self._text_marks)

    def text_zones(self) -> tuple[TextZone, ...]:
        return self._text_zones

    def selected_text_finding(self) -> TextFinding | None:
        if self._text_report is None or self._text_highlight is None:
            return None
        return self._text_report.by_id(self._text_highlight)

    def barcode_report(self) -> BarcodeReport | None:
        return self._barcode_report

    def barcode_marks(self) -> dict[int, OperatorMark]:
        return dict(self._barcode_marks)

    def barcode_zones(self) -> tuple[BarcodeZone, ...]:
        return self._barcode_zones

    def selected_barcode_finding(self) -> BarcodeFinding | None:
        if self._barcode_report is None or self._barcode_highlight is None:
            return None
        return self._barcode_report.by_id(self._barcode_highlight)

    def color_report(self) -> ColorReport | None:
        return self._color_report

    def color_marks(self) -> dict[int, OperatorMark]:
        return dict(self._color_marks)

    def color_profile(self) -> CalibrationProfile | None:
        return self._color_profile

    def selected_color_finding(self) -> ColorFinding | None:
        if self._color_report is None or self._color_highlight is None:
            return None
        return self._color_report.by_id(self._color_highlight)

    def _color_pending(self) -> bool:
        return (
            self._config.color.enabled and self._comparison is not None
            and self._color_report is None and "color" not in self._check_errors
        )

    def _barcode_pending(self) -> bool:
        return (
            self._config.barcodes.enabled and self._comparison is not None
            and self._barcode_report is None and "barcodes" not in self._check_errors
        )

    def _text_pending(self) -> bool:
        """Text inspection is enabled and still to come for the current comparison."""
        return (
            self._config.text.enabled and self._comparison is not None
            and self._text_report is None and "text" not in self._check_errors
        )

    def is_loading(self, role: ImageRole | None = None) -> bool:
        return bool(self._loading) if role is None else role in self._loading

    def config(self) -> AppConfig:
        return self._config

    def view_synchronizer(self) -> ViewSynchronizer:
        return self._sync

    def running_stages(self) -> list[str]:
        """Human-readable names of the work in progress."""
        stages = []
        if self._loading:
            stages.append("loading images")
        if self._quality_generation in self._quality_workers:
            stages.append("checking capture quality")
        if self._alignment_generation in self._alignment_workers:
            stages.append("detecting the label")
        if self._registration_generation in self._registration_workers:
            stages.append("registering to the artwork")
        if self._comparison_generation in self._comparison_workers:
            stages.append("comparing and assessing defects")
        if self._text_generation in self._text_workers:
            stages.append("reading text (OCR)")
        if self._barcode_generation in self._barcode_workers:
            stages.append("reading barcodes")
        if self._color_generation in self._color_workers:
            stages.append("analysing colour")
        if self._save_worker is not None:
            stages.append("saving results")
        return stages

    def _both_images(self) -> bool:
        return all(self._images[role] is not None for role in ImageRole)

    def _update_actions(self) -> None:
        has_any_image = any(self._images.values())
        has_photo = self._images[ImageRole.PRINTED] is not None
        busy = bool(self.running_stages())
        can_compare = self._both_images() and not self._loading
        self.compare_action.setEnabled(can_compare)
        self.compare_action.setToolTip(
            "Run the inspection, or compare again with the current sensitivity (F5)." if can_compare
            else "Load the original artwork and a photo of the printed label first."
        )
        can_save = (
            self._both_images() and not busy
            and (self._quality_report is not None or self._alignment is not None or bool(self._check_errors))
        )
        self.save_action.setEnabled(can_save)
        self.detect_action.setEnabled(has_photo)
        self.manual_corners_action.setEnabled(has_photo)
        self.adjust_registration_action.setEnabled(self._registration_base() is not None)
        self.alignment_button.setEnabled(has_photo)
        self.reset_action.setEnabled(has_any_image or bool(self._loading))
        for action in (self.zoom_in_action, self.zoom_out_action, self.fit_action, self.actual_size_action):
            action.setEnabled(has_any_image)
        has_defects = bool(self._defects and self._defects.defects)
        for action in (self.next_defect_action, self.previous_defect_action):
            action.setEnabled(has_defects)
        self.previous_button.setEnabled(has_defects)
        self.next_button.setEnabled(has_defects)
        for action in self.mark_actions.values():
            action.setEnabled(
                self.selected_defect() is not None or self.selected_text_finding() is not None
                or self.selected_barcode_finding() is not None or self.selected_color_finding() is not None
            )
        self.barcode_zones_action.setEnabled(self._images[ImageRole.ORIGINAL] is not None)
        self.barcode_panel.zones_button.setEnabled(self._images[ImageRole.ORIGINAL] is not None)
        self.text_zones_action.setEnabled(self._images[ImageRole.ORIGINAL] is not None)
        self.text_panel.zones_button.setEnabled(self._images[ImageRole.ORIGINAL] is not None)
        self.regions_action.setEnabled(self._images[ImageRole.ORIGINAL] is not None)
        self.close_profile_action.setEnabled(self._profile is not None)

    def _update_status(self) -> None:
        self._update_title()
        text = self._text_report if self._config.text.enabled else None
        barcodes = self._barcode_report if self._config.barcodes.enabled else None
        color = self._color_report if self._config.color.enabled else None
        decision = decide(
            self._quality_report, self._alignment, self._registration, self._comparison,
            self._defects, self._marks, self._config.decision, text, self._text_marks,
            barcodes, self._barcode_marks, color, self._color_marks,
        )
        pending = [what for what, waiting in (("the text (OCR)", self._text_pending()),
                                              ("the barcodes", self._barcode_pending()),
                                              ("the colours", self._color_pending())) if waiting]
        if self._check_errors:
            decision = InspectionDecision(
                InspectionStatus.REVIEW, decision.reasons + tuple(self._check_errors.values()),
                (), decision.review, decision.minor_count, (), decision.text_review, (), decision.barcode_review,
                (), decision.color_review,
            )
        elif pending:
            if decision.status is InspectionStatus.PASS:  # no verdict before everything has been checked
                decision = InspectionDecision(
                    InspectionStatus.NOT_INSPECTED, (f"Reading {' and '.join(pending)}…",), (), (),
                    decision.minor_count,
                )
            else:
                decision = dataclasses.replace(
                    decision, reasons=decision.reasons + (f"(Still reading {' and '.join(pending)}.)",)
                )
        self._decision = decision
        self.status_indicator.set_status(decision.status)
        reasons = " ".join(decision.reasons)
        self.status_reason_label.setText(reasons or "No verdict yet: load both images to inspect.")
        self.status_area.verdict_reason.set_full_text(reasons)
        self._refresh_status_area()
        self._update_actions()
        self.decision_changed.emit(decision)

    def _refresh_status_area(self) -> None:
        """Capture quality, alignment quality and progress in the bottom strip."""
        area = self.status_area
        running = self.running_stages()
        has_photo = self._images[ImageRole.PRINTED] is not None
        has_art = self._images[ImageRole.ORIGINAL] is not None

        report = self._quality_report
        if "checking capture quality" in running:
            area.quality.set_level(None, "CHECKING…", "Checking the photo…")
        elif "quality" in self._check_errors:
            area.quality.set_level(QualityLevel.POOR, "FAILED", self._check_errors["quality"])
        elif report is not None:
            detail = report.issues[0].message if report.issues else "No capture problems found."
            if len(report.issues) > 1:
                detail += f" (+{len(report.issues) - 1} more)"
            area.quality.set_level(report.level, None, detail)
        else:
            area.quality.set_level(None, None, "No photo loaded." if not has_photo else "")

        alignment, registration = self._alignment, self._registration
        if "detecting the label" in running or "registering to the artwork" in running:
            text = "Detecting the label…" if "detecting the label" in running else "Registering to the artwork…"
            area.alignment.set_level(None, "CHECKING…", text)
        elif "alignment" in self._check_errors or "registration" in self._check_errors:
            reason = self._check_errors.get("alignment") or self._check_errors.get("registration")
            area.alignment.set_level(QualityLevel.POOR, "FAILED", reason)
        elif registration is not None and alignment is not None:
            level = QualityLevel.POOR if alignment_blocks(alignment, registration) else registration.level
            issues = registration.issues or alignment.issues
            detail = (
                issues[0].message if issues
                else f"Registered; 95 % of points within {opt(registration.metrics.residual_p95_px, '.2f')} px."
            )
            area.alignment.set_level(level, None, detail)
        elif alignment is not None:
            detail = alignment.issues[0].message if alignment.issues else "Label found and corrected."
            if not has_art and alignment.rectified is not None:
                detail += " Load the artwork to register."
            area.alignment.set_level(alignment.level, None, detail)
        else:
            area.alignment.set_level(None, None, "")

        extra_stages = int(self._config.text.enabled) + int(self._config.barcodes.enabled) + int(self._config.color.enabled)
        total = (2 if has_photo else 0) + (2 + extra_stages if has_photo and has_art else 0)
        stopped = alignment is not None and alignment.rectified is None
        compared = self._comparison is not None or "comparison" in self._check_errors or stopped
        done = sum((
            report is not None or "quality" in self._check_errors,
            alignment is not None or "alignment" in self._check_errors,
            registration is not None or "registration" in self._check_errors or stopped,
            compared,
        ) + tuple(
            report is not None or key in self._check_errors or stopped or "comparison" in self._check_errors
            for enabled, report, key in (
                (self._config.text.enabled, self._text_report, "text"),
                (self._config.barcodes.enabled, self._barcode_report, "barcodes"),
                (self._config.color.enabled, self._color_report, "color"),
            ) if enabled
        )[: total])
        if running:
            text = "Running: " + ", ".join(running) + "…"
        elif not has_photo and not has_art:
            text = "Waiting for images."
        elif not has_photo:
            text = "Load a photo of the printed label."
        elif not has_art:
            text = "Load the original artwork to compare."
        elif self._comparison is not None and not (self._text_pending() or self._barcode_pending() or self._color_pending()):
            text = f"Inspection complete (comparison {self._comparison.metrics.seconds:.1f} s"
            if self._text_report is not None and self._text_report.available:
                text += f", text {self._text_report.seconds:.1f} s"
            if self._barcode_report is not None and self._barcode_report.available:
                text += f", barcodes {self._barcode_report.seconds:.1f} s"
            if self._color_report is not None:
                text += f", colour {self._color_report.seconds:.1f} s"
            text += ")."
        elif done >= total:
            text = "Stopped before the comparison - see the verdict."
        else:
            text = "Ready - press Compare (F5)."
        area.set_progress(done, total, text, busy=bool(running))

    def _start_worker(self, worker) -> None:
        self._thread_pool.start(worker)
        self._refresh_status_area()
        self._update_actions()

    # ------------------------------------------------------------------ messages

    def show_error(self, text: str) -> None:
        """Non-blocking error message above the viewers (for failures of background work)."""
        self.banner.show_message(text, "error")
        self.statusBar().showMessage(text, 10000)

    def _clear_banner(self) -> None:
        self.banner.hide()

    # ------------------------------------------------------------------ loading

    def _choose_file(self, role: ImageRole) -> None:
        patterns = " ".join(f"*{ext}" for ext in allowed_extensions(role, self._config.image_loading))
        if role is ImageRole.ORIGINAL:
            caption, filter_name = "Load Original Artwork", "Artwork images"
        else:
            caption, filter_name = "Load Printed Label Photo", "Photos"
        path, _ = QFileDialog.getOpenFileName(self, caption, str(self._last_dir), f"{filter_name} ({patterns})")
        if path:
            self.load_image_file(role, Path(path))

    def load_image_file(self, role: ImageRole, path: Path) -> None:
        """Start loading ``path`` in the background. Emits image_loaded or image_load_failed."""
        path = Path(path)
        self._generation[role] += 1
        generation = self._generation[role]
        self._loading.add(role)
        if path.parent.is_dir():
            self._last_dir = path.parent

        worker = ImageLoadWorker(path, role, generation, self._config.image_loading)
        worker.signals.succeeded.connect(self._on_image_loaded)
        worker.signals.failed.connect(self._on_image_failed)
        self._workers[(role, generation)] = worker

        self._panels[role].set_info(f"Loading {path.name}…")
        self.statusBar().showMessage(f"Loading {role.label}: {path.name}…")
        self._start_worker(worker)

    def _on_image_loaded(self, role: ImageRole, generation: int, result: DisplayableImage) -> None:
        self._workers.pop((role, generation), None)
        if generation != self._generation[role]:
            return  # superseded by a newer load or a reset
        self._loading.discard(role)
        image = result.image
        self._clear_banner()

        if role is ImageRole.ORIGINAL:
            try:
                self.original_panel.set_image(result.preview, describe_image(image))
            except ValueError as exc:
                self._report_load_failure(role, str(exc))
                return
            self._images[role] = image
            self._artwork_setup(image)
        else:
            self._images[role] = image
            self._manual_corners = None
            self._automatic_corners = None
            self._clear_alignment()
            self._printed_previews = {VIEW_PHOTO: result.preview}
            self._printed_shown = None
            self._refresh_printed_panel()

        self.statusBar().showMessage(
            f"Loaded {role.label}: {image.path.name} ({image.width} × {image.height}, {image.file_format})"
        )
        self.image_loaded.emit(role)
        # The photo needs (re)checking; new artwork changes label size, orientation and verification.
        if self._images[ImageRole.PRINTED] is not None:
            self._start_quality_assessment()
            self._start_alignment(self._manual_corners)
        else:
            self._update_sync_scales()
        self._update_status()

    def _on_image_failed(self, role: ImageRole, generation: int, message: str) -> None:
        self._workers.pop((role, generation), None)
        if generation != self._generation[role]:
            return
        self._loading.discard(role)
        self._report_load_failure(role, message)

    def _report_load_failure(self, role: ImageRole, message: str) -> None:
        current = self._images[role]
        if current is not None:
            if role is ImageRole.PRINTED:
                self._printed_shown = None
                self._refresh_printed_panel()
            else:
                self._panels[role].set_info(describe_image(current))
        else:
            self._panels[role].clear()
        self._update_status()
        self.statusBar().showMessage(f"Could not load the {role.label}.")
        self.image_load_failed.emit(role, message)
        QMessageBox.critical(self, "Could Not Load Image", message)

    # ------------------------------------------------------------------ printed panel

    def printed_view_mode(self) -> str:
        return self.printed_view_combo.currentData()

    def set_printed_view_mode(self, mode: str) -> None:
        self.printed_view_combo.setCurrentIndex(self.printed_view_combo.findData(mode))

    def _refresh_printed_panel(self) -> None:
        """Show the registered label, the perspective-corrected label or the original photo."""
        photo = self._images[ImageRole.PRINTED]
        panel = self.printed_panel
        if photo is None:
            panel.clear()
            self._printed_shown = None
            self._update_sync_scales()
            return
        alignment = self._alignment
        wanted = self.printed_view_mode()
        fallback = [wanted] + [k for k in (VIEW_REGISTERED, VIEW_ALIGNED, VIEW_PHOTO) if k != wanted]
        key = next(k for k in fallback if k in self._printed_previews)

        if key == VIEW_REGISTERED:
            m = self._registration.metrics
            w, h = m.grid_size
            info = (
                f"Registered to artwork: {w} × {h} px ({m.grid_scale:.2f}× artwork pixels)\n"
                f"{m.method.label} · from {photo.path.name}\n"
                "Resampled once from the original photo pixels."
            )
        elif key == VIEW_ALIGNED:
            m = alignment.metrics
            w, h = m.output_size
            scale = f" ({m.scale_to_artwork:.2f}× artwork size)" if m.scale_to_artwork else ""
            info = f"Perspective-corrected label: {w} × {h} px{scale}\n{m.method.label} · from {photo.path.name}"
        else:
            info = describe_image(photo)
            if alignment is None and self._alignment_workers:
                info += "\nDetecting the label…"
            elif alignment is not None and alignment.rectified is None:
                info += "\nThe label could not be aligned. Use Manual Alignment → Mark Label Corners."
        if key != wanted and wanted != VIEW_PHOTO:
            info += "\n(The selected view is not available yet.)"

        if self._printed_shown != key:
            panel.set_image(self._printed_previews[key], info)
            self._printed_shown = key
        else:
            panel.set_info(info)
        self._show_quality_warning_on_panel()

        panel.view.set_boxes(self._defect_boxes() if key == VIEW_REGISTERED else None)
        if key == VIEW_PHOTO:
            region = self._quality_report.metrics.content_region if self._quality_report else None
            panel.view.set_region(QRectF(region.x, region.y, region.width, region.height) if region else None)
            corners = alignment.metrics.corners if alignment is not None else None
            colour = "#22c55e" if alignment is not None and alignment.is_reliable else "#ef4444"
            panel.view.set_polygon(corners, colour)
        else:
            panel.view.set_region(None)
            panel.view.set_polygon(None)
        self._update_sync_scales()

    # ------------------------------------------------------------------ view synchronisation

    def _grid_scale(self) -> tuple[float, float] | None:
        """Comparison-grid pixels per artwork pixel."""
        artwork = self._images[ImageRole.ORIGINAL]
        registration = self._registration
        if artwork is None or registration is None or registration.reference is None:
            return None
        gh, gw = registration.reference.shape[:2]
        ah, aw = artwork.working.shape[:2]
        return gw / aw, gh / ah

    def _update_sync_scales(self) -> None:
        artwork = self._images[ImageRole.ORIGINAL]
        grid = self._grid_scale()
        printed = None
        if artwork is not None:
            ah, aw = artwork.working.shape[:2]
            if self._printed_shown == VIEW_REGISTERED:
                printed = grid
            elif self._printed_shown == VIEW_ALIGNED and self._alignment is not None and self._alignment.rectified is not None:
                oh, ow = self._alignment.rectified.shape[:2]
                printed = (ow / aw, oh / ah)
        self._sync.set_scale(self.original_panel.view, (1.0, 1.0) if artwork is not None else None)
        self._sync.set_scale(self.printed_panel.view, printed)
        self._sync.set_scale(self.overlay_panel.view, grid if self._overlay_image is not None else None)

    # ------------------------------------------------------------------ capture quality

    def _start_quality_assessment(self) -> None:
        printed = self._images[ImageRole.PRINTED]
        if printed is None:
            return
        self._quality_generation += 1
        generation = self._quality_generation
        self._quality_report = None
        self._check_errors.pop("quality", None)
        label_mm = artwork_label_long_side_mm(self._images[ImageRole.ORIGINAL], self._config.capture_quality)
        worker = QualityWorker(printed, generation, self._config.capture_quality, label_mm)
        worker.signals.succeeded.connect(self._on_quality_ready)
        worker.signals.failed.connect(self._on_quality_failed)
        self._quality_workers[generation] = worker
        self.quality_section.set_busy()
        self._start_worker(worker)

    def _on_quality_ready(self, generation: int, report: QualityReport) -> None:
        self._quality_workers.pop(generation, None)
        if generation != self._quality_generation:
            return
        self._quality_report = report
        self.quality_section.set_level(report.level)
        self.quality_section.set_issues(report.issues, "No capture problems found.")
        self.quality_section.metrics_label.setText(format_quality_metrics(report.metrics))
        self._refresh_printed_panel()
        self._update_status()
        self.statusBar().showMessage(f"Capture quality: {report.level.value.upper()}")
        self.quality_assessed.emit(report)

    def _on_quality_failed(self, generation: int, message: str) -> None:
        self._quality_workers.pop(generation, None)
        if generation != self._quality_generation:
            return
        self._quality_report = None
        self.quality_section.set_error(message)
        # An unassessed photo cannot be trusted.
        self._check_errors["quality"] = "Capture quality could not be assessed."
        self._update_status()
        self.show_error(f"Capture quality could not be assessed: {message}")
        self.quality_failed.emit(message)

    def _show_quality_warning_on_panel(self) -> None:
        report = self._quality_report
        if report is None or not report.issues:
            self.printed_panel.set_warning(None)
            return
        criticals = sum(i.severity is Severity.CRITICAL for i in report.issues)
        warnings_ = len(report.issues) - criticals
        parts = []
        if criticals:
            parts.append(f"{criticals} critical")
        if warnings_:
            parts.append(f"{warnings_} warning{'s' if warnings_ != 1 else ''}")
        # One line (screen space is short); the full list is in the tooltip and the Checks tab.
        self.printed_panel.set_warning(
            f"Capture quality {report.level.value.upper()} ({', '.join(parts)}): {report.issues[0].message}",
            "\n".join(f"• {i.message}" for i in report.issues),
        )

    # ------------------------------------------------------------------ alignment

    def detect_label(self) -> None:
        """Run automatic detection again, discarding manual corners."""
        self._manual_corners = None
        self._start_alignment(None)

    def apply_manual_corners(self, corners: np.ndarray) -> None:
        """Align using user-placed corners (photo pixel coordinates)."""
        self._manual_corners = np.asarray(corners, np.float64).reshape(4, 2)
        self._start_alignment(self._manual_corners)

    def create_corner_dialog(self) -> CornerSelectionDialog | None:
        photo = self._images[ImageRole.PRINTED]
        if photo is None:
            return None
        if self._manual_corners is not None:
            initial = self._manual_corners
        elif self._alignment is not None and self._alignment.metrics.corners is not None:
            initial = self._alignment.metrics.corners
        else:
            initial = None
        alignment_cfg = self._config.alignment
        return CornerSelectionDialog(
            bgr8_to_qimage(photo.working),
            (photo.width, photo.height),
            self._config.ui,
            initial_corners=initial,
            automatic_corners=self._automatic_corners,
            snap=lambda c: snap_corners_to_edges(photo.working, c, alignment_cfg),
            parent=self,
        )

    def _open_manual_corners(self) -> None:
        dialog = self.create_corner_dialog()
        if dialog is None:
            return
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.corners() is not None:
            self.apply_manual_corners(dialog.corners())

    def _start_alignment(self, manual_corners: np.ndarray | None) -> None:
        photo = self._images[ImageRole.PRINTED]
        if photo is None:
            return
        self._alignment_generation += 1
        generation = self._alignment_generation
        self._alignment = None
        self._check_errors.pop("alignment", None)
        self._printed_previews.pop(VIEW_ALIGNED, None)
        self._clear_registration()
        worker = AlignmentWorker(
            photo, self._images[ImageRole.ORIGINAL], generation, self._config.alignment, manual_corners
        )
        worker.signals.succeeded.connect(self._on_alignment_ready)
        worker.signals.failed.connect(self._on_alignment_failed)
        self._alignment_workers[generation] = worker
        self.alignment_section.set_busy()
        self._start_worker(worker)
        self._refresh_printed_panel()
        self._update_status()

    def _on_alignment_ready(self, generation: int, result: AlignmentResult) -> None:
        self._alignment_workers.pop(generation, None)
        if generation != self._alignment_generation:
            return
        self._alignment = result
        if result.metrics.method not in (None, AlignmentMethod.MANUAL):
            self._automatic_corners = result.metrics.corners
        if result.rectified is not None:
            self._printed_previews[VIEW_ALIGNED] = bgr8_to_qimage(result.rectified)
        self._printed_shown = None  # the aligned image changed; redraw
        self.alignment_section.set_level(result.level)
        self.alignment_section.set_issues(result.issues, "Label found and aligned; no problems detected.")
        self.alignment_section.metrics_label.setText(format_alignment_metrics(result.metrics))
        self._refresh_printed_panel()
        self.statusBar().showMessage(f"Alignment: {result.level.value.upper()}")
        if result.rectified is not None and self._images[ImageRole.ORIGINAL] is not None:
            self._start_registration()
        self._update_status()
        self.alignment_done.emit(result)

    def _on_alignment_failed(self, generation: int, message: str) -> None:
        self._alignment_workers.pop(generation, None)
        if generation != self._alignment_generation:
            return
        self._alignment = None
        self.alignment_section.set_error(message)
        self._refresh_printed_panel()
        self._check_errors["alignment"] = "The label could not be aligned."
        self._update_status()
        self.show_error(f"The label could not be aligned: {message}")
        self.alignment_failed.emit(message)

    def _clear_alignment(self) -> None:
        self._alignment_generation += 1  # results of in-flight work will be ignored
        self._alignment = None
        self._check_errors.pop("alignment", None)
        self._printed_previews.pop(VIEW_ALIGNED, None)
        self.alignment_section.clear()

    # ------------------------------------------------------------------ registration

    def _registration_base(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(reference image, grid -> photo homography) to adjust manually, if possible."""
        artwork = self._images[ImageRole.ORIGINAL]
        if artwork is None or self._alignment is None or self._alignment.homography is None:
            return None
        if self._registration is not None and self._registration.homography is not None:
            return self._registration.reference, self._registration.homography
        size, start = comparison_grid(self._alignment, artwork.working.shape, self._config.registration)
        return None if size is None else (None, start)

    def _start_registration(self, initial: np.ndarray | None = None, refine: bool = True) -> None:
        photo = self._images[ImageRole.PRINTED]
        artwork = self._images[ImageRole.ORIGINAL]
        if photo is None or artwork is None or self._alignment is None or self._alignment.rectified is None:
            return
        self._clear_registration()
        generation = self._registration_generation
        worker = RegistrationWorker(
            photo, artwork, self._alignment, generation, self._config.registration, initial, refine
        )
        worker.signals.succeeded.connect(self._on_registration_ready)
        worker.signals.failed.connect(self._on_registration_failed)
        self._registration_workers[generation] = worker
        self.registration_section.set_busy()
        self._start_worker(worker)

    def _on_registration_ready(self, generation: int, result: RegistrationResult) -> None:
        self._registration_workers.pop(generation, None)
        if generation != self._registration_generation:
            return
        self._registration = result
        if result.registered is not None:
            self._printed_previews[VIEW_REGISTERED] = bgr8_to_qimage(result.registered)
        self._printed_shown = None
        self.registration_section.set_level(result.level)
        self.registration_section.set_issues(result.issues, "Registered and verified; no problems detected.")
        self.registration_section.metrics_label.setText(format_registration_metrics(result.metrics))
        self._refresh_printed_panel()
        self._refresh_overlay(fit=True)
        self.statusBar().showMessage(f"Registration: {result.level.value.upper()}")
        if result.registered is not None:
            self.run_comparison()
        self._update_status()
        self.registration_done.emit(result)

    def _on_registration_failed(self, generation: int, message: str) -> None:
        self._registration_workers.pop(generation, None)
        if generation != self._registration_generation:
            return
        self._registration = None
        self.registration_section.set_error(message)
        self._check_errors["registration"] = "The photo could not be registered to the artwork."
        self._update_status()
        self.show_error(f"The photo could not be registered to the artwork: {message}")
        self.registration_failed.emit(message)

    def _clear_registration(self) -> None:
        self._clear_comparison()
        self._registration_generation += 1  # results of in-flight work will be ignored
        self._registration = None
        self._check_errors.pop("registration", None)
        self._printed_previews.pop(VIEW_REGISTERED, None)
        self._overlay_image = None
        self.registration_section.clear()
        self.overlay_panel.clear()
        self.original_panel.view.set_boxes(None)

    def create_registration_dialog(self) -> RegistrationAdjustDialog | None:
        base = self._registration_base()
        photo = self._images[ImageRole.PRINTED]
        artwork = self._images[ImageRole.ORIGINAL]
        if base is None or photo is None or artwork is None:
            return None
        reference, homography = base
        if reference is None:
            size, _ = comparison_grid(self._alignment, artwork.working.shape, self._config.registration)
            reference = np.ascontiguousarray(cv2.resize(artwork.working, size, interpolation=cv2.INTER_AREA))
        return RegistrationAdjustDialog(photo.working, reference, homography, self._config.ui, parent=self)

    def apply_registration_adjustment(
        self, base_homography: np.ndarray, adjustment: ManualAdjustment, grid_size: tuple[int, int], refine: bool
    ) -> None:
        """Register again from a manually adjusted starting point (verified like any registration)."""
        self._start_registration(base_homography @ adjustment.sampling_matrix(grid_size), refine)

    def _open_registration_dialog(self) -> None:
        dialog = self.create_registration_dialog()
        if dialog is None:
            return
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._start_registration(dialog.adjusted_homography(), dialog.refine())

    def overlay_mode(self) -> str:
        return self.overlay_mode_combo.currentData()

    def set_overlay(self, mode: str, opacity_percent: int) -> None:
        self.overlay_mode_combo.setCurrentIndex(self.overlay_mode_combo.findData(mode))
        self.overlay_opacity.setValue(opacity_percent)
        self._refresh_overlay()

    def overlay_image(self) -> np.ndarray | None:
        return self._overlay_image

    def _refresh_overlay(self, fit: bool = False) -> None:
        result = self._registration
        if result is None or result.registered is None:
            return
        mode = self.overlay_mode()
        strength = self.overlay_opacity.value() / 100.0
        comparison = self._comparison
        if mode in COLOR_VIEWS and self._color_report is None:
            mode = "defects" if comparison is not None else "edges"  # colour still being analysed
        if mode in COLOR_VIEWS:
            report = self._color_report
            values = report.heatmap if mode == "color_artwork" else report.local_heatmap
            limit = self._config.color.heatmap_max_delta_e * (1.5 - strength)
            self._overlay_image = np.ascontiguousarray(heatmap_image(values, limit, result.registered))
            info = (
                f"{COLOR_VIEWS[mode]} · ΔE00 0 (dark) to {limit:.0f} (yellow) · "
                + ("calibrated" if report.calibrated else "APPROXIMATE (uncalibrated)")
                + "\nGray: not measured (edges, fine detail, structural defects)."
            )
        elif mode in COMPARISON_VIEWS:
            if comparison is None:
                mode = "edges"  # comparison still running: show the alignment overlay meanwhile
                info = "Comparing with the artwork…"
            else:
                c = comparison.metrics
                n = len(self._defects.defects) if self._defects else 0
                info = f"{COMPARISON_VIEWS[mode]} · {n} defect{'s' if n != 1 else ''} · sensitivity {c.sensitivity}"
        else:
            m = result.metrics
            info = (
                f"Artwork vs. registered photo · residual 95%: {opt(m.residual_p95_px, '.2f')} px\n"
                "Darkened areas are outside the comparison mask."
            )
        if mode in COLOR_VIEWS:
            pass
        elif mode in COMPARISON_VIEWS:
            markers = self.show_defects_action.isChecked()
            self._overlay_image = render_comparison(
                comparison, result.registered, result.mask, mode, strength, self._highlight,
                self._defects if markers else None, self._marks,
            )
        else:
            self._overlay_image = make_overlay(result.reference, result.registered, result.mask, mode, strength)
        image = bgr8_to_qimage(self._overlay_image)
        if fit or not self.overlay_panel.view.has_image():
            self.overlay_panel.set_image(image, info)
        else:
            self.overlay_panel.update_image(image)
            self.overlay_panel.set_info(info)
        self.overlay_panel.view.set_boxes(self._defect_boxes(defects=False))
        self._update_sync_scales()

    # ------------------------------------------------------------------ comparison

    def compare(self) -> None:
        """Run whatever the inspection still needs, or compare again with the current settings.

        With a registered photo this re-runs the comparison (e.g. after changing the
        sensitivity); otherwise it repeats registration or label detection.
        """
        if not self._both_images():
            self.statusBar().showMessage("Load the original artwork and a photo of the printed label first.")
            return
        pipeline = {"detecting the label", "registering to the artwork", "comparing and assessing defects"}
        if pipeline & set(self.running_stages()):
            self.statusBar().showMessage("The inspection is already running.", 5000)
            return
        self._clear_banner()
        if self._registration is not None and self._registration.registered is not None:
            self.run_comparison()
        elif self._alignment is not None and self._alignment.rectified is not None:
            self._start_registration()
            self._update_status()
        else:
            if self._quality_report is None:
                self._start_quality_assessment()
            self._start_alignment(self._manual_corners)

    def run_comparison(self) -> None:
        """(Re)compare the registered photo with the artwork using the current sensitivity."""
        registration = self._registration
        if registration is None or registration.registered is None:
            return
        self._clear_comparison()
        generation = self._comparison_generation
        artwork = self._images[ImageRole.ORIGINAL]
        px_per_mm, estimated = grid_px_per_mm(
            registration.metrics.grid_scale,
            artwork.dpi,
            artwork_label_long_side_mm(artwork, self._config.capture_quality),
            max(artwork.width, artwork.height),
            self._config.capture_quality.assumed_label_long_side_mm,
        )
        worker = ComparisonWorker(
            registration, generation, self._config.comparison, self.sensitivity_combo.currentData(),
            self._config.defects, px_per_mm, estimated, self._regions, self._config.decision.critical_region_severity,
        )
        worker.signals.succeeded.connect(self._on_comparison_ready)
        worker.signals.failed.connect(self._on_comparison_failed)
        self._comparison_workers[generation] = worker
        self.comparison_section.set_busy()
        self._start_worker(worker)
        self._refresh_overlay()
        self._update_status()

    def _on_comparison_ready(self, generation: int, result: ComparisonResult, report: DefectReport) -> None:
        self._comparison_workers.pop(generation, None)
        if generation != self._comparison_generation:
            return
        self._comparison = result
        self._defects = report
        self._marks = {}
        count = len(report.defects)
        self.comparison_section.set_level(result.level)
        self.comparison_section.set_issues(
            result.issues,
            "No differences found." if not result.candidates else f"{len(result.candidates)} difference regions found.",
        )
        self.comparison_section.metrics_label.setText(format_comparison_metrics(result.metrics))
        self._fill_defect_table()
        if self.overlay_mode() not in COMPARISON_VIEWS:
            self.overlay_mode_combo.setCurrentIndex(self.overlay_mode_combo.findData("defects"))
        self._refresh_overlay()
        self._refresh_defect_markers()
        if self._config.text.enabled:
            self.run_text_inspection()
        if self._config.barcodes.enabled:
            self.run_barcode_inspection()
        if self._config.color.enabled:
            self.run_color_analysis()
        self._update_status()
        self.statusBar().showMessage(f"Comparison: {count} defect{'s' if count != 1 else ''}")
        self.comparison_done.emit(result)

    def _on_comparison_failed(self, generation: int, message: str) -> None:
        self._comparison_workers.pop(generation, None)
        if generation != self._comparison_generation:
            return
        self._comparison = None
        self.comparison_section.set_error(message)
        self._check_errors["comparison"] = "The print could not be compared with the artwork."
        self._update_status()
        self.show_error(message)
        self.comparison_failed.emit(message)

    def _clear_comparison(self) -> None:
        self._comparison_generation += 1  # results of in-flight work will be ignored
        self._comparison = None
        self._defects = None
        self._marks = {}
        self._highlight = None
        self._check_errors.pop("comparison", None)
        self.comparison_section.clear()
        self.defect_table.setRowCount(0)
        self.defects_summary.setText("")
        self.defect_detail.show_defect(None, OperatorMark.UNREVIEWED)
        self.original_panel.view.set_boxes(None)
        self._clear_text()
        self._clear_barcodes()
        self._clear_color()

    # ------------------------------------------------------------------ defects

    def _fill_defect_table(self) -> None:
        report = self._defects
        self.defect_table.blockSignals(True)
        self.defect_table.setRowCount(0)
        if report is None:
            self.defect_table.blockSignals(False)
            return
        self.defect_table.setRowCount(len(report.defects))
        scale = self._grid_scale() or (1.0, 1.0)
        approx = " ~" if report.scale_estimated else ""
        for row, d in enumerate(report.defects):
            mark = self._marks.get(d.id, OperatorMark.UNREVIEWED)
            x_mm, y_mm = defect_position_mm(d, report.px_per_mm)
            values = (
                str(d.id),
                d.category_label,
                d.severity.value,
                mark.label,
                f"{x_mm:.1f}, {y_mm:.1f}{approx}",
                f"{d.length_mm:.2f} × {d.thickness_mm:.2f}{approx}",
                f"{d.score:.1f}",
                d.reliability.value,
            )
            tooltip = (
                f"{d.category_note}\n"
                f"Centre {x_mm:.2f}, {y_mm:.2f} mm from the artwork's top-left corner "
                f"(artwork pixel {d.centroid[0] / scale[0]:.0f}, {d.centroid[1] / scale[1]:.0f}).\n"
                f"Score {d.score:.1f} (1 = detection threshold); largest difference {d.max_difference:.0f} "
                f"gray levels; ΔE {d.delta_e:.1f}."
                + ("\nSizes and positions use an assumed label size." if report.scale_estimated else "")
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(tooltip)
                item.setData(Qt.ItemDataRole.UserRole, d.id)
                if col in (0, DEFECT_COLUMNS.index("Score")):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if mark is OperatorMark.DISMISSED:
                    item.setForeground(QBrush(QColor("#9ca3af")))
                elif d.severity.value == "critical":
                    item.setForeground(QBrush(QColor("#b91c1c")))
                self.defect_table.setItem(row, col, item)
            if d.id == self._highlight:
                self.defect_table.selectRow(row)
        self.defect_table.blockSignals(False)
        counts = {s: sum(1 for d in report.defects if d.severity.value == s) for s in ("critical", "major", "minor")}
        summary = (
            f"{len(report.defects)} defect(s): {counts['critical']} critical, {counts['major']} major, "
            f"{counts['minor']} minor."
        )
        if report.filtered:
            summary += f" {len(report.filtered)} tiny weak region(s) ignored as noise."
        if report.scale_estimated:
            summary += " Sizes in mm are estimated (artwork has no usable DPI)."
        self.defects_summary.setText(summary)

    def _defect_boxes(
        self, scale: tuple[float, float] = (1.0, 1.0), defects: bool = True
    ) -> list[tuple[QRectF, str, str]] | None:
        """Numbered defect and text-finding boxes in a view with ``scale`` grid pixels per view pixel."""
        if not self.show_defects_action.isChecked():
            return None
        sx, sy = scale
        boxes = []
        for f in self._text_report.findings if self._text_report is not None else ():
            mark = self._text_marks.get(f.id, OperatorMark.UNREVIEWED)
            colour = (
                _BOX_COLOURS["selected"] if f.id == self._text_highlight and self._selected_kind == "text"
                else _BOX_COLOURS["dismissed"] if mark is OperatorMark.DISMISSED
                else _BOX_COLOURS["text"]
            )
            boxes.append((QRectF(f.x / sx, f.y / sy, f.width / sx, f.height / sy), f.label, colour))
        for f in self._barcode_report.findings if self._barcode_report is not None else ():
            mark = self._barcode_marks.get(f.id, OperatorMark.UNREVIEWED)
            colour = (
                _BOX_COLOURS["selected"] if f.id == self._barcode_highlight and self._selected_kind == "barcode"
                else _BOX_COLOURS["dismissed"] if mark is OperatorMark.DISMISSED
                else _BOX_COLOURS["barcode"]
            )
            boxes.append((QRectF(f.x / sx, f.y / sy, f.width / sx, f.height / sy), f.label, colour))
        for f in self._color_report.findings if self._color_report is not None else ():
            mark = self._color_marks.get(f.id, OperatorMark.UNREVIEWED)
            colour = (
                _BOX_COLOURS["selected"] if f.id == self._color_highlight and self._selected_kind == "color"
                else _BOX_COLOURS["dismissed"] if mark is OperatorMark.DISMISSED
                else _BOX_COLOURS["color"]
            )
            boxes.append((QRectF(f.x / sx, f.y / sy, f.width / sx, f.height / sy), f.label, colour))
        reference = self._registration.reference if self._registration is not None else None
        for region in self._regions if reference is not None else ():
            x, y, w, h = region.rect(reference.shape[1], reference.shape[0])
            boxes.append((QRectF(x / sx, y / sy, w / sx, h / sy), f"{region.kind.label}: {region.name}",
                          _BOX_COLOURS["ignored" if region.kind is RegionKind.IGNORE else "critical"]))
        for d in (self._defects.defects if self._defects is not None and defects else ()):
            mark = self._marks.get(d.id, OperatorMark.UNREVIEWED)
            colour = (
                _BOX_COLOURS["selected"] if d.id == self._highlight and self._selected_kind == "defect"
                else _BOX_COLOURS["dismissed"] if mark is OperatorMark.DISMISSED
                else _BOX_COLOURS["active"]
            )
            boxes.append((QRectF(d.x / sx, d.y / sy, d.width / sx, d.height / sy), str(d.id), colour))
        return boxes or None

    def _refresh_defect_markers(self) -> None:
        """Redraw the boxes on all three viewers (after selection, marks or the overlay toggle)."""
        grid = self._grid_scale()
        self.original_panel.view.set_boxes(self._defect_boxes(grid) if grid else None)
        if self._printed_shown == VIEW_REGISTERED:
            self.printed_panel.view.set_boxes(self._defect_boxes())
        self._refresh_overlay()

    def color_finding_at(self, grid_point: QPointF, tolerance: float = 4.0) -> ColorFinding | None:
        if self._color_report is None:
            return None
        hits = [
            f for f in self._color_report.findings
            if QRectF(f.x, f.y, f.width, f.height).adjusted(-tolerance, -tolerance, tolerance, tolerance)
            .contains(grid_point)
        ]
        return min(hits, key=lambda f: f.width * f.height) if hits else None

    def barcode_finding_at(self, grid_point: QPointF, tolerance: float = 4.0) -> BarcodeFinding | None:
        if self._barcode_report is None:
            return None
        hits = [
            f for f in self._barcode_report.findings
            if QRectF(f.x, f.y, f.width, f.height).adjusted(-tolerance, -tolerance, tolerance, tolerance)
            .contains(grid_point)
        ]
        return min(hits, key=lambda f: f.width * f.height) if hits else None

    def text_finding_at(self, grid_point: QPointF, tolerance: float = 4.0) -> TextFinding | None:
        if self._text_report is None:
            return None
        hits = [
            f for f in self._text_report.findings
            if QRectF(f.x, f.y, f.width, f.height).adjusted(-tolerance, -tolerance, tolerance, tolerance)
            .contains(grid_point)
        ]
        return min(hits, key=lambda f: f.width * f.height) if hits else None

    def defect_at(self, grid_point: QPointF, tolerance: float = 4.0) -> Defect | None:
        """The smallest defect whose box (grown by ``tolerance`` grid pixels) contains the point."""
        if self._defects is None:
            return None
        hits = [
            d for d in self._defects.defects
            if QRectF(d.x, d.y, d.width, d.height).adjusted(-tolerance, -tolerance, tolerance, tolerance)
            .contains(grid_point)
        ]
        return min(hits, key=lambda d: d.width * d.height) if hits else None

    def _on_view_clicked(self, panel: str, point: QPointF) -> None:
        """A click on a viewer selects the defect or text finding under the pointer."""
        if (self._defects is None and self._text_report is None and self._barcode_report is None
                and self._color_report is None) or not self.show_defects_action.isChecked():
            return
        grid = self._grid_scale()
        if grid is None:
            return
        if panel == "original":
            view, grid_point = self.original_panel.view, QPointF(point.x() * grid[0], point.y() * grid[1])
        elif panel == "printed" and self._printed_shown == VIEW_REGISTERED:
            view, grid_point = self.printed_panel.view, point
        elif panel == "overlay":
            view, grid_point = self.overlay_panel.view, point
        else:
            return
        scale = grid[0] if panel == "original" else 1.0
        tolerance = max(4.0, CLICK_PADDING_SCREEN_PX / max(view.zoom, 1e-6) * scale)
        candidates = [
            (item, select) for item, select in (
                (self.defect_at(grid_point, tolerance), self.select_defect),
                (self.text_finding_at(grid_point, tolerance), self.select_text_finding),
                (self.barcode_finding_at(grid_point, tolerance), self.select_barcode_finding),
                (self.color_finding_at(grid_point, tolerance), self.select_color_finding),
            ) if item is not None
        ]
        if candidates:  # the smallest box under the pointer
            item, select = min(candidates, key=lambda c: c[0].width * c[0].height)
            select(item.id)

    def _on_defect_selected(self) -> None:
        items = self.defect_table.selectedItems()
        if not items or self._defects is None:
            return
        self.select_defect(int(items[0].data(Qt.ItemDataRole.UserRole)))

    def _row_of(self, defect_id: int) -> int | None:
        for row in range(self.defect_table.rowCount()):
            item = self.defect_table.item(row, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == defect_id:
                return row
        return None

    def select_defect(self, defect_id: int) -> None:
        """Highlight a defect, zoom all views onto it and show its evidence side by side."""
        defect = self._defects.by_id(defect_id) if self._defects else None
        if defect is None:
            return
        self._highlight = defect.id
        self._selected_kind = "defect"
        row = self._row_of(defect.id)
        if row is not None and self.defect_table.currentRow() != row:
            self.defect_table.blockSignals(True)
            self.defect_table.selectRow(row)
            self.defect_table.blockSignals(False)
        if row is not None:
            self.defect_table.scrollToItem(self.defect_table.item(row, 0))
        if self.overlay_mode() not in COMPARISON_VIEWS:
            self.overlay_mode_combo.setCurrentIndex(self.overlay_mode_combo.findData("defects"))
        self._refresh_defect_markers()
        self._zoom_to(QRectF(defect.x, defect.y, defect.width, defect.height))
        self.defect_detail.show_defect(
            defect, self._marks.get(defect.id, OperatorMark.UNREVIEWED), self._defects.scale_estimated
        )
        self.side_tabs.setCurrentIndex(0)
        self._update_actions()

    def _zoom_to(self, rect: QRectF, padding: float = 6.0) -> None:
        """Zoom all viewers to ``rect`` (comparison-grid pixels)."""
        self.overlay_panel.view.show_area(rect, padding=padding, min_zoom=1.0)  # synced views follow
        if not self._sync.is_enabled():
            if self._printed_shown == VIEW_REGISTERED:
                self.printed_panel.view.show_area(rect, padding=padding, min_zoom=1.0)
            grid = self._grid_scale()
            if grid is not None:
                art = QRectF(rect.x() / grid[0], rect.y() / grid[1], rect.width() / grid[0], rect.height() / grid[1])
                self.original_panel.view.show_area(art, padding=padding, min_zoom=1.0 / grid[0])

    def step_defect(self, step: int) -> None:
        """Select the next (+1) or previous (-1) defect in the list."""
        if self._defects is None or not self._defects.defects:
            return
        ids = [d.id for d in self._defects.defects]
        if self._highlight in ids:
            index = (ids.index(self._highlight) + step) % len(ids)
        else:
            index = 0 if step > 0 else len(ids) - 1
        self.select_defect(ids[index])

    def _mark_selected(self, mark: OperatorMark) -> None:
        if self._selected_kind == "text" and self.selected_text_finding() is not None:
            self.set_text_mark(self._text_highlight, mark)
            return
        if self._selected_kind == "barcode" and self.selected_barcode_finding() is not None:
            self.set_barcode_mark(self._barcode_highlight, mark)
            return
        if self._selected_kind == "color" and self.selected_color_finding() is not None:
            self.set_color_mark(self._color_highlight, mark)
            return
        defect = self.selected_defect()
        if defect is None:
            self.statusBar().showMessage("Select a defect first.", 4000)
            return
        self.set_mark(defect.id, mark)

    def set_mark(self, defect_id: int, mark: OperatorMark) -> None:
        """Record the operator's judgement of a defect and update the verdict."""
        if self._defects is None or self._defects.by_id(defect_id) is None:
            return
        if mark is OperatorMark.UNREVIEWED:
            self._marks.pop(defect_id, None)
        else:
            self._marks[defect_id] = mark
        log.info("Operator marked defect #%d: %s", defect_id, mark.value)
        self._fill_defect_table()
        self._refresh_defect_markers()
        defect = self._defects.by_id(defect_id)
        if self._highlight == defect_id:
            self.defect_detail.show_defect(defect, mark, self._defects.scale_estimated)
        self._update_status()
        self.statusBar().showMessage(f"Defect #{defect_id}: {mark.label}", 4000)

    # ------------------------------------------------------------------ text inspection

    def _load_text_zones(self, artwork: LoadedImage) -> None:
        try:
            self._text_zones = load_zones(zones_path(artwork.path))
        except TextZoneError as exc:
            self._text_zones = ()
            self.show_error(f"{exc} The zones are ignored.")
            return
        if self._text_zones:
            log.info("Loaded %d text zone(s) for %s", len(self._text_zones), artwork.path.name)

    def run_text_inspection(self) -> None:
        """(Re)inspect the text of the compared print in the background."""
        registration, comparison = self._registration, self._comparison
        if registration is None or registration.registered is None or comparison is None:
            return
        self._clear_text()
        generation = self._text_generation
        worker = TextInspectionWorker(
            registration, comparison, self._text_zones, self._defects.px_per_mm, generation,
            self._config.text, self._config.defects, self._ocr_engine,
            self._config.barcodes.formats if self._config.barcodes.enabled else (), self._regions,
        )
        worker.signals.succeeded.connect(self._on_text_ready)
        worker.signals.failed.connect(self._on_text_failed)
        self._text_workers[generation] = worker
        self.text_section.set_busy()
        self.text_panel.clear("Reading the text…")
        self._start_worker(worker)
        self._update_status()

    def _on_text_ready(self, generation: int, report: TextReport, engine: OcrEngine | None) -> None:
        self._text_workers.pop(generation, None)
        if generation != self._text_generation:
            return
        if engine is not None:
            self._ocr_engine = engine  # found once, reused for later inspections
        self._text_report = report
        self._text_marks = {}
        ok = "Text checked; no problems found." if report.available else ""
        self.text_section.set_level(report.level)
        self.text_section.set_issues(report.issues, ok)
        reliable = sum(f.reliable for f in report.findings)
        self.text_section.metrics_label.setText("\n".join([
            f"Engine       {report.engine or 'none'}",
            f"Lines        {len(report.lines)} ({sum(line.critical for line in report.lines)} critical), "
            f"{len(self._text_zones)} zone(s)",
            f"Findings     {len(report.findings)} ({reliable} reliable)",
            f"Unread areas {len(report.unread_regions)}",
            f"Time         {report.seconds:.1f} s",
        ]))
        self.text_panel.set_report(report, self._text_marks, self._registration.reference, self._registration.registered)
        self._refresh_defect_markers()
        self._update_status()
        self.statusBar().showMessage(
            f"Text: {len(report.findings)} finding(s)" if report.available else "Text inspection not available."
        )
        self.text_done.emit(report)

    def _on_text_failed(self, generation: int, message: str) -> None:
        self._text_workers.pop(generation, None)
        if generation != self._text_generation:
            return
        self.text_section.set_error(message)
        self.text_panel.clear(message)
        self._check_errors["text"] = "The text could not be inspected."
        self._update_status()
        self.show_error(message)
        self.text_failed.emit(message)

    def _clear_text(self) -> None:
        self._text_generation += 1  # results of in-flight work will be ignored
        self._text_report = None
        self._text_marks = {}
        self._text_highlight = None
        if self._selected_kind == "text":
            self._selected_kind = "defect"
        self._check_errors.pop("text", None)
        self.text_section.clear()
        self.text_panel.clear()

    def select_text_finding(self, finding_id: int) -> None:
        """Highlight a text finding, zoom all views onto it and show it in the Text tab."""
        finding = self._text_report.by_id(finding_id) if self._text_report else None
        if finding is None:
            return
        self._text_highlight = finding.id
        self._selected_kind = "text"
        self.text_panel.select(finding.id)
        self.text_panel.show_finding(finding, self._text_marks.get(finding.id, OperatorMark.UNREVIEWED))
        self.side_tabs.setCurrentWidget(self.text_panel)
        self._refresh_defect_markers()
        self._zoom_to(QRectF(finding.x, finding.y, finding.width, finding.height), padding=4.0)
        self._update_actions()

    def show_text_line(self, line_id: int) -> None:
        line = self._text_report.line(line_id) if self._text_report else None
        if line is not None:
            self._zoom_to(QRectF(line.x, line.y, line.width, line.height), padding=1.4)

    def set_text_mark(self, finding_id: int, mark: OperatorMark) -> None:
        """Record the operator's judgement of a text finding and update the verdict."""
        finding = self._text_report.by_id(finding_id) if self._text_report else None
        if finding is None:
            return
        if mark is OperatorMark.UNREVIEWED:
            self._text_marks.pop(finding_id, None)
        else:
            self._text_marks[finding_id] = mark
        log.info("Operator marked text finding %s: %s", finding.label, mark.value)
        self.text_panel.fill(self._text_marks)
        if self._text_highlight == finding_id:
            self.text_panel.show_finding(finding, mark)
        self._refresh_defect_markers()
        self._update_status()
        self.statusBar().showMessage(f"Text finding {finding.label}: {mark.label}", 4000)

    def create_text_zone_dialog(self) -> TextZoneDialog | None:
        artwork = self._images[ImageRole.ORIGINAL]
        if artwork is None:
            return None
        return TextZoneDialog(artwork.working, self._text_zones, self._config.ui, parent=self)

    def _open_text_zones(self) -> None:
        dialog = self.create_text_zone_dialog()
        if dialog is not None and dialog.exec() == QDialog.DialogCode.Accepted:
            self.apply_text_zones(dialog.zones())

    def apply_text_zones(self, zones: tuple[TextZone, ...]) -> None:
        """Use these zones, save them next to the artwork, and inspect the text again."""
        artwork = self._images[ImageRole.ORIGINAL]
        if artwork is None:
            return
        self._text_zones = tuple(zones)
        path = zones_path(artwork.path)
        try:
            if self._profile is None:
                save_zones(path, self._text_zones)
        except (OSError, TextZoneError) as exc:
            QMessageBox.critical(
                self, "Could Not Save Text Zones",
                f"The zones could not be saved to {path}:\n{exc}\n\nThey apply until the artwork is reloaded.",
            )
        self.statusBar().showMessage(
            f"{len(self._text_zones)} text zone(s) for {artwork.path.name}" + self._kept_where(), 5000
        )
        if self._config.text.enabled and self._comparison is not None:
            self.run_text_inspection()
        self._update_status()  # the title shows whether the profile in use changed

    # ------------------------------------------------------------------ barcode inspection

    def _load_barcode_zones(self, artwork: LoadedImage) -> None:
        try:
            self._barcode_zones = load_barcode_zones(barcode_zones_path(artwork.path))
        except BarcodeZoneError as exc:
            self._barcode_zones = ()
            self.show_error(f"{exc} The barcode zones are ignored.")

    def run_barcode_inspection(self) -> None:
        """(Re)inspect the barcodes of the compared print in the background."""
        registration, comparison = self._registration, self._comparison
        if registration is None or registration.registered is None or comparison is None:
            return
        self._clear_barcodes()
        generation = self._barcode_generation
        worker = BarcodeInspectionWorker(
            registration, comparison, self._barcode_zones, generation, self._config.barcodes, self._regions
        )
        worker.signals.succeeded.connect(self._on_barcodes_ready)
        worker.signals.failed.connect(self._on_barcodes_failed)
        self._barcode_workers[generation] = worker
        self.barcode_section.set_busy()
        self.barcode_panel.clear("Reading the barcodes…")
        self._start_worker(worker)
        self._update_status()

    def _on_barcodes_ready(self, generation: int, report: BarcodeReport) -> None:
        self._barcode_workers.pop(generation, None)
        if generation != self._barcode_generation:
            return
        self._barcode_report = report
        self._barcode_marks = {}
        self.barcode_section.set_level(report.level)
        self.barcode_section.set_issues(report.issues, "Barcodes checked." if report.available else "")
        decoded = sum(1 for s in report.symbols if s.print_value)
        self.barcode_section.metrics_label.setText("\n".join([
            f"Library      {report.library or 'none'}",
            f"Barcodes     {len(report.symbols)} ({decoded} decoded in the print), {len(self._barcode_zones)} zone(s)",
            f"Findings     {len(report.findings)} ({sum(f.reliable for f in report.findings)} reliable)",
            f"Time         {report.seconds:.1f} s",
            "Not an ISO/IEC 15416 / 15415 grade.",
        ]))
        self.barcode_panel.set_report(
            report, self._barcode_marks, self._registration.reference, self._registration.registered
        )
        self._refresh_defect_markers()
        self._update_status()
        self.statusBar().showMessage(
            f"Barcodes: {len(report.findings)} finding(s)" if report.available else "Barcode inspection not available."
        )
        self.barcodes_done.emit(report)

    def _on_barcodes_failed(self, generation: int, message: str) -> None:
        self._barcode_workers.pop(generation, None)
        if generation != self._barcode_generation:
            return
        self.barcode_section.set_error(message)
        self.barcode_panel.clear(message)
        self._check_errors["barcodes"] = "The barcodes could not be inspected."
        self._update_status()
        self.show_error(message)
        self.barcodes_failed.emit(message)

    def _clear_barcodes(self) -> None:
        self._barcode_generation += 1  # results of in-flight work will be ignored
        self._barcode_report = None
        self._barcode_marks = {}
        self._barcode_highlight = None
        if self._selected_kind == "barcode":
            self._selected_kind = "defect"
        self._check_errors.pop("barcodes", None)
        self.barcode_section.clear()
        self.barcode_panel.clear()

    def select_barcode_finding(self, finding_id: int) -> None:
        """Highlight a barcode finding, zoom all views onto it and show it in the Barcodes tab."""
        finding = self._barcode_report.by_id(finding_id) if self._barcode_report else None
        if finding is None:
            return
        self._barcode_highlight = finding.id
        self._selected_kind = "barcode"
        self.barcode_panel.select(finding.id)
        self.barcode_panel.show_finding(finding, self._barcode_marks.get(finding.id, OperatorMark.UNREVIEWED))
        self.side_tabs.setCurrentWidget(self.barcode_panel)
        self._refresh_defect_markers()
        self._zoom_to(QRectF(finding.x, finding.y, finding.width, finding.height), padding=1.6)
        self._update_actions()

    def show_barcode_symbol(self, symbol_id: int) -> None:
        symbol = self._barcode_report.symbol(symbol_id) if self._barcode_report else None
        if symbol is not None:
            self._zoom_to(QRectF(symbol.x, symbol.y, symbol.width, symbol.height), padding=1.3)

    def set_barcode_mark(self, finding_id: int, mark: OperatorMark) -> None:
        """Record the operator's judgement of a barcode finding and update the verdict."""
        finding = self._barcode_report.by_id(finding_id) if self._barcode_report else None
        if finding is None:
            return
        if mark is OperatorMark.UNREVIEWED:
            self._barcode_marks.pop(finding_id, None)
        else:
            self._barcode_marks[finding_id] = mark
        log.info("Operator marked barcode finding %s: %s", finding.label, mark.value)
        self.barcode_panel.fill(self._barcode_marks)
        if self._barcode_highlight == finding_id:
            self.barcode_panel.show_finding(finding, mark)
        self._refresh_defect_markers()
        self._update_status()
        self.statusBar().showMessage(f"Barcode finding {finding.label}: {mark.label}", 4000)

    def create_barcode_zone_dialog(self) -> BarcodeZoneDialog | None:
        artwork = self._images[ImageRole.ORIGINAL]
        if artwork is None:
            return None
        return BarcodeZoneDialog(artwork.working, self._barcode_zones, self._config.ui, parent=self)

    def _open_barcode_zones(self) -> None:
        dialog = self.create_barcode_zone_dialog()
        if dialog is not None and dialog.exec() == QDialog.DialogCode.Accepted:
            self.apply_barcode_zones(dialog.zones())

    def apply_barcode_zones(self, zones: tuple[BarcodeZone, ...]) -> None:
        """Use these zones, save them next to the artwork, and inspect the barcodes again."""
        artwork = self._images[ImageRole.ORIGINAL]
        if artwork is None:
            return
        self._barcode_zones = tuple(zones)
        path = barcode_zones_path(artwork.path)
        try:
            if self._profile is None:
                save_barcode_zones(path, self._barcode_zones)
        except (OSError, BarcodeZoneError) as exc:
            QMessageBox.critical(
                self, "Could Not Save Barcode Zones",
                f"The zones could not be saved to {path}:\n{exc}\n\nThey apply until the artwork is reloaded.",
            )
        self.statusBar().showMessage(
            f"{len(self._barcode_zones)} barcode zone(s) for {artwork.path.name}" + self._kept_where(), 5000
        )
        if self._config.barcodes.enabled and self._comparison is not None:
            self.run_barcode_inspection()
        self._update_status()

    # ------------------------------------------------------------------ colour analysis

    def _load_color_profile(self) -> None:
        if self._color_profile_path is None:
            return
        try:
            self._color_profile = load_profile(self._color_profile_path)
        except CalibrationError as exc:
            self._color_profile = None
            self.show_error(f"{exc} Colours are analysed as approximate.")
        if self._color_profile is not None:
            log.info("Colour profile: %s", self._color_profile.describe())

    def run_color_analysis(self) -> None:
        """(Re)analyse the colours of the compared print in the background."""
        registration = self._registration
        if registration is None or registration.registered is None or self._defects is None:
            return
        self._clear_color()
        generation = self._color_generation
        worker = ColorAnalysisWorker(
            registration, self._defects, generation, self._config.color, self._color_profile, self._regions,
            self._config.decision.critical_region_severity,
        )
        worker.signals.succeeded.connect(self._on_color_ready)
        worker.signals.failed.connect(self._on_color_failed)
        self._color_workers[generation] = worker
        self.color_section.set_busy()
        self.color_panel.clear("Analysing the colours…")
        self._start_worker(worker)
        self._update_status()

    def _on_color_ready(self, generation: int, report: ColorReport) -> None:
        self._color_workers.pop(generation, None)
        if generation != self._color_generation:
            return
        self._color_report = report
        self._color_marks = {}
        self.color_section.set_level(report.level, "CALIBRATED" if report.calibrated else "APPROXIMATE")
        self.color_section.set_issues(report.issues, "Colour analysed; no lighting problems found.")
        self.color_section.metrics_label.setText("\n".join([
            f"Mode         {report.mode}" + (f" ({report.profile})" if report.profile else ""),
            f"Areas        {len(report.areas)} flat colour areas, {report.meaningful_fraction:.0%} of the label measured",
            f"Findings     {len(report.findings)} ({sum(f.reliable for f in report.findings)} reliable)",
            f"Lighting     {report.lighting_range[0]:.2f} - {report.lighting_range[1]:.2f}",
            f"Time         {report.seconds:.1f} s",
        ]))
        self.color_panel.set_report(report, self._color_marks, self._registration.reference, self._registration.registered)
        self._refresh_defect_markers()
        self._update_status()
        self.statusBar().showMessage(f"Colour ({report.mode}): {len(report.findings)} finding(s)")
        self.color_done.emit(report)

    def _on_color_failed(self, generation: int, message: str) -> None:
        self._color_workers.pop(generation, None)
        if generation != self._color_generation:
            return
        self.color_section.set_error(message)
        self.color_panel.clear(message)
        self._check_errors["color"] = "The colours could not be analysed."
        self._update_status()
        self.show_error(message)
        self.color_failed.emit(message)

    def _clear_color(self) -> None:
        self._color_generation += 1  # results of in-flight work will be ignored
        self._color_report = None
        self._color_marks = {}
        self._color_highlight = None
        if self._selected_kind == "color":
            self._selected_kind = "defect"
        self._check_errors.pop("color", None)
        self.color_section.clear()
        self.color_panel.clear()

    def select_color_finding(self, finding_id: int) -> None:
        finding = self._color_report.by_id(finding_id) if self._color_report else None
        if finding is None:
            return
        self._color_highlight = finding.id
        self._selected_kind = "color"
        self.color_panel.select(finding.id)
        self.color_panel.show_finding(finding, self._color_marks.get(finding.id, OperatorMark.UNREVIEWED))
        self.side_tabs.setCurrentWidget(self.color_panel)
        self._refresh_defect_markers()
        self._zoom_to(QRectF(finding.x, finding.y, finding.width, finding.height), padding=1.6)
        self._update_actions()

    def show_color_area(self, area_id: int) -> None:
        area = next((a for a in self._color_report.areas if a.id == area_id), None) if self._color_report else None
        if area is not None:
            self._zoom_to(QRectF(area.x, area.y, area.width, area.height), padding=1.3)

    def set_color_mark(self, finding_id: int, mark: OperatorMark) -> None:
        finding = self._color_report.by_id(finding_id) if self._color_report else None
        if finding is None:
            return
        if mark is OperatorMark.UNREVIEWED:
            self._color_marks.pop(finding_id, None)
        else:
            self._color_marks[finding_id] = mark
        log.info("Operator marked colour finding %s: %s", finding.label, mark.value)
        self.color_panel.fill(self._color_marks)
        if self._color_highlight == finding_id:
            self.color_panel.show_finding(finding, mark)
        self._refresh_defect_markers()
        self._update_status()
        self.statusBar().showMessage(f"Colour finding {finding.label}: {mark.label}", 4000)

    def create_calibration_dialog(self) -> CalibrationDialog:
        return CalibrationDialog(
            self._images[ImageRole.PRINTED], self._color_profile, self._config.ui, self._config.color,
            self._config.image_loading, parent=self,
        )

    def _open_color_calibration(self) -> None:
        dialog = self.create_calibration_dialog()
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.apply_color_profile(dialog.profile())

    def apply_color_profile(self, profile: CalibrationProfile | None) -> None:
        """Use (and keep) this colour profile, or none; analyse the colours again."""
        self._color_profile = profile
        path = self._color_profile_path
        if path is not None:
            try:
                if profile is None:
                    if path.exists():
                        path.unlink()
                else:
                    save_profile(path, profile)
            except OSError as exc:
                QMessageBox.critical(
                    self, "Could Not Save the Colour Profile",
                    f"The profile could not be saved to {path}:\n{exc}\n\nIt applies until the application is closed.",
                )
        self.statusBar().showMessage(
            f"Colour calibration: {profile.describe()}" if profile else "Colour calibration removed.", 8000
        )
        if self._config.color.enabled and self._comparison is not None:
            self.run_color_analysis()

    # ------------------------------------------------------------------ regions and profiles

    def _kept_where(self) -> str:
        return f" (in profile '{self._profile.name}' – save the profile to keep them)." if self._profile else "."

    def _artwork_setup(self, artwork: LoadedImage) -> None:
        """Regions and zones for a newly loaded artwork: the profile's, or those saved next to the file."""
        if self._profile is not None:
            warning = inspection_profiles.shape_warning(self._profile, artwork.width, artwork.height)
            if warning:
                self.banner.show_message(warning, "warning")
            return
        self._load_text_zones(artwork)
        self._load_barcode_zones(artwork)
        try:
            self._regions = load_regions(regions_path(artwork.path))
        except RegionError as exc:
            self._regions = ()
            self.show_error(f"{exc} The regions are ignored.")
        suggestion = next(
            (e.profile for e in inspection_profiles.list_profiles(self._profiles_dir)
             if e.profile is not None and e.profile.artwork_name == artwork.path.name), None,
        )
        if suggestion is not None:
            self.statusBar().showMessage(
                f"Profile '{suggestion.name}' was made for {artwork.path.name} – load it with Profiles… (Ctrl+P).",
                15000,
            )

    def regions(self) -> tuple[InspectionRegion, ...]:
        return self._regions

    def create_region_dialog(self) -> RegionDialog | None:
        artwork = self._images[ImageRole.ORIGINAL]
        if artwork is None:
            return None
        return RegionDialog(artwork.working, self._regions, self._config.ui, parent=self)

    def _open_regions(self) -> None:
        dialog = self.create_region_dialog()
        if dialog is not None and dialog.exec() == QDialog.DialogCode.Accepted:
            self.apply_regions(dialog.regions())

    def apply_regions(self, regions: tuple[InspectionRegion, ...]) -> None:
        """Use these regions (saved next to the artwork unless a profile is in use) and inspect again."""
        artwork = self._images[ImageRole.ORIGINAL]
        if artwork is None:
            return
        self._regions = tuple(regions)
        path = regions_path(artwork.path)
        if self._profile is None:
            try:
                save_regions(path, self._regions)
            except (OSError, RegionError) as exc:
                QMessageBox.critical(
                    self, "Could Not Save Regions",
                    f"The regions could not be saved to {path}:\n{exc}\n\nThey apply until the artwork is reloaded.",
                )
        self.statusBar().showMessage(
            f"{len(self._regions)} region(s) for {artwork.path.name}" + self._kept_where(), 5000
        )
        if self._registration is not None and self._registration.registered is not None:
            self.run_comparison()  # every inspection follows the comparison
        self._update_status()

    def inspection_profile(self) -> InspectionProfile | None:
        return self._profile

    def profile_modified(self) -> bool:
        """True if the setup in use differs from the profile in use."""
        if self._profile is None:
            return False
        return not inspection_profiles.matches(
            self._profile, self._config, self.sensitivity_combo.currentData(), self._regions, self._text_zones,
            self._barcode_zones,
        )

    def _update_title(self) -> None:
        title = f"{self._config.app.name} {self._config.app.version}"
        if self._profile is not None:
            title += f" – Profile: {self._profile.name}" + (" (modified)" if self.profile_modified() else "")
        self.setWindowTitle(title)

    def current_profile(
        self, name: str, description: str = "", previous: InspectionProfile | None = None
    ) -> InspectionProfile:
        """A profile of the setup in use (settings, sensitivity, regions and zones)."""
        artwork = self._images[ImageRole.ORIGINAL]
        if artwork is not None:
            artwork_name, size = artwork.path.name, (artwork.width, artwork.height)
        elif previous is not None:
            artwork_name, size = previous.artwork_name, previous.artwork_size
        else:
            artwork_name, size = "", None
        return inspection_profiles.create_profile(
            name, description, self._config, self.sensitivity_combo.currentData(), self._regions,
            self._text_zones, self._barcode_zones, artwork_name, size, previous,
        )

    def create_profile_dialog(self) -> ProfileDialog:
        return ProfileDialog(
            self._profiles_dir, self.current_profile, self._profile.name if self._profile else "", parent=self
        )

    def open_profiles(self) -> None:
        dialog = self.create_profile_dialog()
        dialog.exec()
        self.finish_profile_dialog(dialog)

    def finish_profile_dialog(self, dialog: ProfileDialog) -> None:
        """Load the profile chosen in the dialog, or adopt the one saved there."""
        if dialog.chosen() is not None:
            self.load_inspection_profile(*dialog.chosen())
        elif dialog.saved() is not None:
            self._profile_path, self._profile = dialog.saved()
            self._update_actions()
            self._update_status()
            self.statusBar().showMessage(f"Profile '{self._profile.name}' saved to {self._profile_path}", 8000)

    def load_inspection_profile(self, path: Path, profile: InspectionProfile | None = None) -> bool:
        """Use a profile: its settings over this computer's, its sensitivity, regions and zones."""
        try:
            profile = profile or inspection_profiles.load_profile(path)
            config = inspection_profiles.apply_profile(profile, self._base_config)
        except ProfileError as exc:
            self.show_error(str(exc))
            return False
        self._profile, self._profile_path = profile, Path(path)
        self._use_setup(config, profile.sensitivity, profile.regions, profile.text_zones, profile.barcode_zones)
        artwork = self._images[ImageRole.ORIGINAL]
        warning = inspection_profiles.shape_warning(profile, artwork.width, artwork.height) if artwork else None
        if warning:
            self.banner.show_message(warning, "warning")
        log.info("Inspection profile '%s' loaded from %s", profile.name, path)
        self.statusBar().showMessage(f"Profile '{profile.name}': {profile.summary()}", 8000)
        return True

    def save_inspection_profile(self) -> bool:
        """Save the setup in use to the profile in use; without one, open the profile library."""
        if self._profile is None or self._profile_path is None:
            self.open_profiles()
            return self._profile is not None
        profile = self.current_profile(self._profile.name, self._profile.description, self._profile)
        try:
            inspection_profiles.save_profile(self._profile_path, profile)
        except OSError as exc:
            QMessageBox.critical(self, "Could Not Save the Profile", f"{self._profile_path}:\n{exc}")
            return False
        self._profile = profile
        self._update_status()
        self.statusBar().showMessage(f"Profile '{profile.name}' saved to {self._profile_path}", 8000)
        return True

    def close_inspection_profile(self) -> None:
        """Back to this computer's settings and the regions and zones saved with the artwork."""
        if self._profile is None:
            return
        name = self._profile.name
        self._profile, self._profile_path = None, None
        self._regions, self._text_zones, self._barcode_zones = (), (), ()
        artwork = self._images[ImageRole.ORIGINAL]
        if artwork is not None:
            self._artwork_setup(artwork)
        self._use_setup(self._base_config, self._base_config.inspection.sensitivity, self._regions,
                        self._text_zones, self._barcode_zones)
        self.statusBar().showMessage(f"Profile '{name}' closed.", 5000)

    def _use_setup(self, config: AppConfig, sensitivity: str, regions, text_zones, barcode_zones) -> None:
        """Switch settings, sensitivity, regions and zones at once and inspect again."""
        old, self._config = self._config, config
        if (old.text.tesseract_path, old.text.tessdata_dir, old.text.languages) != (
            config.text.tesseract_path, config.text.tessdata_dir, config.text.languages
        ) and not self._injected_engine:
            self._ocr_engine = None
        self._regions, self._text_zones, self._barcode_zones = tuple(regions), tuple(text_zones), tuple(barcode_zones)
        self.sensitivity_combo.blockSignals(True)
        self.sensitivity_combo.setCurrentIndex(SENSITIVITY_LEVELS.index(sensitivity))
        self.sensitivity_combo.blockSignals(False)
        if self._registration is not None and self._registration.registered is not None:
            self.run_comparison()  # the text, barcode and colour inspections follow
        self._update_actions()
        self._update_status()

    # ------------------------------------------------------------------ saving

    def inspection_record(self) -> InspectionRecord:
        return InspectionRecord(
            original=self._images[ImageRole.ORIGINAL],
            printed=self._images[ImageRole.PRINTED],
            quality=self._quality_report,
            alignment=self._alignment,
            registration=self._registration,
            comparison=self._comparison,
            defects=self._defects,
            marks=dict(self._marks),
            decision=self._decision,
            sensitivity=self.sensitivity_combo.currentData(),
            config=self._config,
            errors=tuple(self._check_errors.values()),
            text=self._text_report if self._config.text.enabled else None,
            text_marks=dict(self._text_marks),
            barcodes=self._barcode_report if self._config.barcodes.enabled else None,
            barcode_marks=dict(self._barcode_marks),
            color=self._color_report if self._config.color.enabled else None,
            color_marks=dict(self._color_marks),
            regions=self._regions,
            text_zones=self._text_zones,
            barcode_zones=self._barcode_zones,
            profile=self._profile.name if self._profile is not None else "",
            profile_file=str(self._profile_path) if self._profile_path is not None else "",
            profile_modified=self.profile_modified(),
        )

    def save_results(self) -> None:
        if not self.save_action.isEnabled():
            return
        start = self._results_dir or self._last_dir
        folder = QFileDialog.getExistingDirectory(self, "Save Results - Choose a Folder", str(start))
        if folder:
            self.save_results_to(Path(folder))

    def save_results_to(self, parent: Path) -> None:
        """Write the results folder under ``parent`` in the background (results_saved / results_save_failed)."""
        if self._save_worker is not None:
            return
        registration = self._registration
        images: dict[str, np.ndarray] = {}
        if registration is not None and registration.registered is not None:
            images["registered_print.png"] = registration.registered  # annotated images come with the report
        self._results_dir = Path(parent)
        worker = SaveResultsWorker(Path(parent), self.inspection_record(), images)
        worker.signals.succeeded.connect(self._on_results_saved)
        worker.signals.failed.connect(self._on_results_save_failed)
        self._save_worker = worker
        self.statusBar().showMessage("Saving results…")
        self._start_worker(worker)

    def _on_results_saved(self, folder: Path) -> None:
        self._save_worker = None
        self._update_status()
        url = QUrl.fromLocalFile(str(folder)).toString()
        self.banner.show_message(f"Results saved to <a href=\"{url}\">{folder}</a>", "info")
        self.statusBar().showMessage(f"Results saved to {folder}", 10000)
        self.results_saved.emit(folder)

    def _on_results_save_failed(self, message: str) -> None:
        self._save_worker = None
        self._update_status()
        self.statusBar().showMessage("Saving the results failed.", 10000)
        self.results_save_failed.emit(message)
        QMessageBox.critical(self, "Could Not Save Results", message)

    # ------------------------------------------------------------------ settings

    def create_settings_dialog(self) -> SettingsDialog:
        """The dialog edits the settings file: it starts from the values saved there.

        With an inspection profile in use it edits the settings in use instead: those a
        profile stores belong to the profile (Save Profile keeps them), the others are
        saved to the settings file as usual.
        """
        if self._profile is not None:
            return SettingsDialog(
                self._config, config_overrides(self._config), self._user_config_path, parent=self,
                notice=f"Profile '{self._profile.name}' is in use. Changes to the inspection settings apply to "
                       "this profile: use Profiles ▸ Save Profile to keep them. Paths (Tesseract folder) and "
                       "settings a profile does not contain are saved for this computer.",
            )
        current, overrides, notice = self._config, {}, None
        if self._user_config_path is not None:
            try:
                overrides = read_user_overrides(self._user_config_path)
                current = config_with_overrides(overrides)
            except ConfigError as exc:
                overrides = {}
                notice = f"Your settings file is invalid and will be replaced when you press OK.\n{exc}"
        return SettingsDialog(current, overrides, self._user_config_path, parent=self, notice=notice)

    def open_settings(self) -> None:
        dialog = self.create_settings_dialog()
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.accept_settings_dialog(dialog)

    def accept_settings_dialog(self, dialog: SettingsDialog) -> None:
        """Apply an accepted settings dialog (see create_settings_dialog)."""
        config = dialog.result_config()
        if config is None:
            return
        if self._profile is None:
            self.apply_settings(config, dialog.overrides())
            return
        # The settings a profile does not store go to this computer's settings.
        base, new = config_data(self._base_config), config_data(config)
        for section, values in new.items():
            for key, value in values.items():
                if not inspection_profiles.is_profile_setting(section, key):
                    base[section][key] = value
        old_base, self._base_config = self._base_config, parse_config(base)
        self.apply_settings(config, config_overrides(self._base_config) if self._base_config != old_base else None)

    def apply_settings(self, config: AppConfig, overrides: dict | None = None) -> None:
        """Use new settings; save them if a settings file is configured; re-run what they affect."""
        if overrides is not None and self._user_config_path is not None:
            try:
                write_user_overrides(self._user_config_path, overrides)
            except OSError as exc:
                QMessageBox.critical(
                    self, "Could Not Save Settings",
                    f"The settings could not be saved to {self._user_config_path}:\n{exc}\n\n"
                    "They apply until the application is closed.",
                )
        old, self._config = self._config, config
        if self._profile is None:
            self._base_config = config
        log.info("Settings changed")
        if old.capture_quality != config.capture_quality and self._images[ImageRole.PRINTED] is not None:
            self._start_quality_assessment()
        rerun = old.defects != config.defects or old.comparison != config.comparison or (
            old.capture_quality.assumed_label_long_side_mm != config.capture_quality.assumed_label_long_side_mm
        )
        if (old.text.tesseract_path, old.text.tessdata_dir, old.text.languages) != (
            config.text.tesseract_path, config.text.tessdata_dir, config.text.languages
        ) and not self._injected_engine:
            self._ocr_engine = None  # find Tesseract again with the new settings
        if rerun and self._registration is not None and self._registration.registered is not None:
            self.run_comparison()  # the text and barcode inspections follow the comparison
        elif self._comparison is not None:
            if old.text != config.text and config.text.enabled:
                self.run_text_inspection()
            elif old.text != config.text:
                self._clear_text()
            if old.color != config.color and config.color.enabled:
                self.run_color_analysis()
            elif old.color != config.color:
                self._clear_color()
            if old.barcodes != config.barcodes and config.barcodes.enabled:
                self.run_barcode_inspection()
            elif old.barcodes != config.barcodes:
                self._clear_barcodes()
        self._update_status()
        self.statusBar().showMessage("Settings applied.", 5000)

    # ------------------------------------------------------------------ commands

    def reset(self) -> None:
        for role in ImageRole:
            self._generation[role] += 1  # results of in-flight loads will be ignored
            self._images[role] = None
        self._loading.clear()
        self._quality_generation += 1
        self._quality_report = None
        self._check_errors.clear()
        self.quality_section.clear()
        self._clear_alignment()
        self._clear_registration()
        self._clear_comparison()
        self._manual_corners = None
        self._automatic_corners = None
        self._printed_previews = {}
        self._printed_shown = None
        if self._profile is None:  # a profile's regions and zones stay in use
            self._text_zones = ()
            self._barcode_zones = ()
            self._regions = ()
        for panel in (self.original_panel, self.printed_panel, self.overlay_panel):
            panel.clear()
        self.defect_table.setRowCount(0)
        self._clear_banner()
        self._update_sync_scales()
        self._update_status()
        self.statusBar().showMessage("Reset. Load the original artwork and a printed label photo.")

    def _views_with_images(self) -> list[ImageView]:
        panels = (self.original_panel, self.printed_panel, self.overlay_panel)
        return [p.view for p in panels if p.view.has_image()]

    def _zoom_targets(self) -> list[ImageView]:
        """Views to zoom: with synchronisation, the lead view (the others follow) plus unsynced views."""
        views = self._views_with_images()
        if not self._sync.is_enabled():
            return views
        synced = self._sync.synced_views()
        leader = self._sync.leader()
        return [v for v in views if v not in synced] + ([leader] if leader is not None else [])

    def zoom_in(self) -> None:
        for view in self._zoom_targets():
            view.zoom_in()

    def zoom_out(self) -> None:
        for view in self._zoom_targets():
            view.zoom_out()

    def actual_size(self) -> None:
        for view in self._zoom_targets():
            view.actual_size()

    def fit_to_window(self) -> None:
        for view in self._views_with_images():
            view.fit_to_window()

    def shortcut_summary(self) -> list[tuple[str, str]]:
        """(command, key) for every command with a keyboard shortcut."""
        rows = []
        for action in self.actions():
            keys = ", ".join(k.toString(QKeySequence.SequenceFormat.NativeText) for k in action.shortcuts())
            if keys:
                rows.append((action.text().replace("&", "").rstrip("…"), keys))
        panel = self.side_dock.toggleViewAction()
        rows.append((panel.text(), panel.shortcut().toString(QKeySequence.SequenceFormat.NativeText)))
        return rows

    def _show_shortcuts(self) -> None:
        def table(rows) -> str:
            return "<table cellspacing='4'>" + "".join(
                f"<tr><td>{a}</td><td><b>{b}</b></td></tr>" for a, b in rows
            ) + "</table>"

        QMessageBox.information(
            self, "Keyboard and Mouse",
            "<b>Keyboard</b>" + table(self.shortcut_summary()) + "<br><b>Mouse</b>" + table(MOUSE_HELP),
        )

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About Print Inspector",
            f"<b>{self._config.app.name}</b> {self._config.app.version}<br><br>"
            "Compares original label artwork with a photo of the printed label.<br><br>"
            "Runs entirely offline. No images or data leave this computer.<br><br>"
            "<b>Capabilities:</b> image loading and viewing, smartphone photo quality checks, label "
            "detection with perspective correction, precise registration to the artwork with "
            "verification, manual fallbacks for both, a hybrid comparison engine, defect localisation "
            "with severity and operator review, a PASS / FAIL / REVIEW verdict from configurable "
            "acceptance criteria, offline text inspection (Tesseract OCR combined with a character-level "
            "image check, critical text zones), offline barcode and QR code inspection (ZXing-C++; "
            "decoding and damage checks, not ISO grading), colour difference analysis (CIELAB, ΔE00; "
            "approximate unless calibrated with a colour chart), critical and ignored regions, reusable "
            "inspection profiles stored as local files, and saving the results with a PDF report, annotated "
            "images, a findings CSV and the JSON data.",
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        if self._window_settings is not None:
            self._window_settings.setValue("geometry", self.saveGeometry())
            self._window_settings.setValue("state", self.saveState())
        # Let in-flight work finish so threads are not torn down mid-write.
        self._thread_pool.waitForDone(5000)
        super().closeEvent(event)
