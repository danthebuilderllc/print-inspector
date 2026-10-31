"""Background workers that keep heavy work off the GUI thread."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal
from PySide6.QtGui import QImage

from print_inspector.config import (
    AlignmentConfig,
    CaptureQualityConfig,
    ComparisonConfig,
    DefectsConfig,
    ImageLoadingConfig,
    RegistrationConfig,
    TextConfig,
    BarcodeConfig,
    ColorConfig,
)
from print_inspector.core.alignment import AlignmentResult, align_label
from print_inspector.core.capture_quality import assess_capture_quality
from print_inspector.core.barcode_inspection import inspect_barcodes, locate_symbols
from print_inspector.core.barcode_inspection import unavailable_report as barcode_unavailable_report
from print_inspector.core.barcode_zones import BarcodeZone
from print_inspector.core.barcodes import BarcodeUnavailable
from print_inspector.core.color_calibration import CalibrationProfile
from print_inspector.core.color_inspection import inspect_color
from print_inspector.core.comparison import ComparisonResult, compare_to_artwork
from print_inspector.core.defects import DefectReport
from print_inspector.core.defects import assess_defects, defect_rows
from print_inspector.core.image_io import ImageLoadError, ImageRole, LoadedImage, load_image
from print_inspector.core.registration import RegistrationResult, register_to_artwork
from print_inspector.core.ocr import OcrEngine, OcrUnavailable, create_engine
from print_inspector.core.regions import (
    InspectionRegion,
    RegionKind,
    apply_to_color,
    apply_to_defects,
    boxes,
    inspection_mask,
)
from print_inspector.core.results_export import ExportError, InspectionRecord, write_results
from print_inspector.core.text_inspection import inspect_text, unavailable_report
from print_inspector.core.text_zones import TextZone

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DisplayableImage:
    """A loaded image plus a QImage view of its working pixels.

    ``preview`` shares memory with ``image.working`` (no copy), so it is only
    valid while this object, which keeps ``image`` alive, is referenced.
    """

    image: LoadedImage
    preview: QImage


def bgr8_to_qimage(bgr: np.ndarray) -> QImage:
    """Wrap a C-contiguous HxWx3 uint8 BGR array in a QImage without copying.

    The caller must keep ``bgr`` alive for as long as the QImage is used.
    """
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 uint8 BGR array, got {bgr.shape} {bgr.dtype}")
    if not bgr.flags.c_contiguous:
        raise ValueError("Expected a C-contiguous array")
    height, width, _ = bgr.shape
    return QImage(bgr.data, width, height, int(bgr.strides[0]), QImage.Format.Format_BGR888)


class ImageLoadSignals(QObject):
    # role, generation, DisplayableImage
    succeeded = Signal(object, int, object)
    # role, generation, user-readable message
    failed = Signal(object, int, str)


class ImageLoadWorker(QRunnable):
    """Loads and decodes one image file in a thread-pool thread.

    ``generation`` lets the window ignore results that were superseded by a
    newer load or a reset while this worker was running.
    """

    def __init__(
        self, path: Path, role: ImageRole, generation: int, config: ImageLoadingConfig
    ) -> None:
        super().__init__()
        self.setAutoDelete(False)
        self.signals = ImageLoadSignals()
        self._path = path
        self._role = role
        self._generation = generation
        self._config = config

    def run(self) -> None:
        try:
            loaded = load_image(self._path, self._role, self._config)
            preview = bgr8_to_qimage(loaded.working)
        except ImageLoadError as exc:
            log.warning("Failed to load %s: %s", self._path, exc)
            self.signals.failed.emit(self._role, self._generation, str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - report anything to the user, never crash
            log.exception("Unexpected error loading %s", self._path)
            self.signals.failed.emit(
                self._role, self._generation, f"Unexpected error while loading the image: {exc}"
            )
            return
        log.info("Loaded %s %s: %s", self._role.value, self._path, loaded.summary())
        self.signals.succeeded.emit(
            self._role, self._generation, DisplayableImage(image=loaded, preview=preview)
        )


class QualitySignals(QObject):
    succeeded = Signal(int, object)  # generation, QualityReport
    failed = Signal(int, str)  # generation, message


class QualityWorker(QRunnable):
    """Runs capture-quality assessment of the printed photo in a thread-pool thread."""

    def __init__(
        self,
        image: LoadedImage,
        generation: int,
        config: CaptureQualityConfig,
        label_long_side_mm: float | None,
    ) -> None:
        super().__init__()
        self.setAutoDelete(False)
        self.signals = QualitySignals()
        self._image = image
        self._generation = generation
        self._config = config
        self._label_long_side_mm = label_long_side_mm

    def run(self) -> None:
        try:
            report = assess_capture_quality(
                self._image.working, self._config, self._label_long_side_mm
            )
        except Exception as exc:  # noqa: BLE001 - a failed check must surface, not crash
            log.exception("Capture quality assessment failed for %s", self._image.path)
            self.signals.failed.emit(self._generation, f"Quality assessment failed: {exc}")
            return
        log.info(
            "Capture quality of %s: %s (%s)",
            self._image.path.name,
            report.level.value,
            ", ".join(sorted(report.issue_codes())) or "no issues",
        )
        self.signals.succeeded.emit(self._generation, report)


class AlignmentSignals(QObject):
    succeeded = Signal(int, object)  # generation, AlignmentResult
    failed = Signal(int, str)  # generation, message


class AlignmentWorker(QRunnable):
    """Detects the label, corrects perspective and checks the alignment in a thread-pool thread."""

    def __init__(
        self,
        photo: LoadedImage,
        artwork: LoadedImage | None,
        generation: int,
        config: AlignmentConfig,
        manual_corners: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.setAutoDelete(False)
        self.signals = AlignmentSignals()
        self._photo = photo
        self._artwork = artwork
        self._generation = generation
        self._config = config
        self._manual_corners = manual_corners

    def run(self) -> None:
        try:
            result = align_label(
                self._photo.working,
                self._config,
                artwork=None if self._artwork is None else self._artwork.working,
                manual_corners=self._manual_corners,
            )
        except Exception as exc:  # noqa: BLE001 - a failed alignment must surface, not crash
            log.exception("Alignment failed for %s", self._photo.path)
            self.signals.failed.emit(self._generation, f"Alignment failed: {exc}")
            return
        method = result.metrics.method.value if result.metrics.method else "none"
        log.info(
            "Alignment of %s: %s via %s (%s)",
            self._photo.path.name,
            result.level.value,
            method,
            ", ".join(sorted(result.issue_codes())) or "no issues",
        )
        self.signals.succeeded.emit(self._generation, result)


class RegistrationSignals(QObject):
    succeeded = Signal(int, object)  # generation, RegistrationResult
    failed = Signal(int, str)  # generation, message


class RegistrationWorker(QRunnable):
    """Registers the photo precisely to the artwork in a thread-pool thread."""

    def __init__(
        self,
        photo: LoadedImage,
        artwork: LoadedImage,
        alignment: AlignmentResult,
        generation: int,
        config: RegistrationConfig,
        initial_homography: np.ndarray | None = None,
        refine: bool = True,
    ) -> None:
        super().__init__()
        self.setAutoDelete(False)
        self.signals = RegistrationSignals()
        self._photo = photo
        self._artwork = artwork
        self._alignment = alignment
        self._generation = generation
        self._config = config
        self._initial = initial_homography
        self._refine = refine

    def run(self) -> None:
        try:
            result = register_to_artwork(
                self._photo.working,
                self._artwork.working,
                self._alignment,
                self._config,
                initial_homography=self._initial,
                refine=self._refine,
            )
        except Exception as exc:  # noqa: BLE001 - a failed registration must surface, not crash
            log.exception("Registration failed for %s", self._photo.path)
            self.signals.failed.emit(self._generation, f"Registration failed: {exc}")
            return
        m = result.metrics
        log.info(
            "Registration of %s: %s (%s), residual p95 %s px, coverage %s",
            self._photo.path.name,
            result.level.value,
            ", ".join(sorted(result.issue_codes())) or "no issues",
            "n/a" if m.residual_p95_px is None else f"{m.residual_p95_px:.2f}",
            "n/a" if m.coverage is None else f"{m.coverage:.0%}",
        )
        self.signals.succeeded.emit(self._generation, result)


class ComparisonSignals(QObject):
    succeeded = Signal(int, object, object)  # generation, ComparisonResult, DefectReport
    failed = Signal(int, str)  # generation, message


class ComparisonWorker(QRunnable):
    """Compares the registered photo with the artwork and assesses the defects, in a thread-pool thread."""

    def __init__(
        self,
        registration: RegistrationResult,
        generation: int,
        config: ComparisonConfig,
        sensitivity: str,
        defects_config: DefectsConfig,
        px_per_mm: float,
        scale_estimated: bool,
        regions: tuple[InspectionRegion, ...] = (),
        critical_region_severity: str = "major",
    ) -> None:
        """Ignored ``regions`` are not compared; defects in critical ones are raised in severity."""
        super().__init__()
        self.setAutoDelete(False)
        self.signals = ComparisonSignals()
        self._regions = regions
        self._critical_region_severity = critical_region_severity
        self._registration = registration
        self._generation = generation
        self._config = config
        self._sensitivity = sensitivity
        self._defects_config = defects_config
        self._px_per_mm = px_per_mm
        self._scale_estimated = scale_estimated

    def run(self) -> None:
        r = self._registration
        try:
            result = compare_to_artwork(
                r.registered,
                r.reference,
                inspection_mask(r.mask, self._regions),
                self._config,
                blur_sigma_px=r.metrics.blur_match_sigma_px or 0.0,
                alignment_residual_px=r.metrics.residual_p95_px,
                sensitivity=self._sensitivity,
            )
            report = assess_defects(
                result, r.reference, r.registered, self._px_per_mm, self._scale_estimated, self._defects_config
            )
            report = apply_to_defects(report, self._regions, r.reference.shape, self._critical_region_severity)
        except Exception as exc:  # noqa: BLE001 - a failed comparison must surface, not crash
            log.exception("Comparison failed")
            self.signals.failed.emit(self._generation, f"Comparison failed: {exc}")
            return
        log.info(
            "Comparison (%s sensitivity): %d candidates -> %d defects (%d filtered as noise), %.1f s",
            self._sensitivity, len(result.candidates), len(report.defects), len(report.filtered),
            result.metrics.seconds,
        )
        for row in defect_rows(report):
            log.info("Defect %s", row)
        self.signals.succeeded.emit(self._generation, result, report)


class SaveSignals(QObject):
    succeeded = Signal(object)  # Path of the results folder
    failed = Signal(str)


class SaveResultsWorker(QRunnable):
    """Writes the results folder (JSON, CSV, PNG images) in a thread-pool thread."""

    def __init__(self, parent: Path, record: InspectionRecord, images: dict[str, np.ndarray]) -> None:
        super().__init__()
        self.setAutoDelete(False)
        self.signals = SaveSignals()
        self._parent = parent
        self._record = record
        self._images = images

    def run(self) -> None:
        try:
            folder = write_results(self._parent, self._record, self._images)
        except ExportError as exc:
            log.error("Saving results failed: %s", exc)
            self.signals.failed.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - a failed save must surface, not crash
            log.exception("Saving results failed")
            self.signals.failed.emit(f"Saving the results failed: {exc}")
            return
        log.info("Results saved to %s", folder)
        self.signals.succeeded.emit(folder)


class TextSignals(QObject):
    succeeded = Signal(int, object, object)  # generation, TextReport, OcrEngine (or None)
    failed = Signal(int, str)


class TextInspectionWorker(QRunnable):
    """Finds Tesseract (unless an engine is given) and inspects the text, in a thread-pool thread.

    A missing OCR engine is not an error of the worker: it yields a report saying
    the text could not be inspected (REVIEW if text inspection is required).
    """

    def __init__(
        self,
        registration: RegistrationResult,
        comparison: ComparisonResult,
        zones: tuple[TextZone, ...],
        px_per_mm: float,
        generation: int,
        config: TextConfig,
        defects_config: DefectsConfig,
        engine: OcrEngine | None,
        barcode_formats: tuple[str, ...] = (),
        regions: tuple[InspectionRegion, ...] = (),
    ) -> None:
        """Barcodes of ``barcode_formats`` found in the artwork and ignored ``regions`` are not read."""
        super().__init__()
        self.setAutoDelete(False)
        self.signals = TextSignals()
        self._regions = regions
        self._barcode_formats = barcode_formats
        self._registration = registration
        self._comparison = comparison
        self._zones = zones
        self._px_per_mm = px_per_mm
        self._generation = generation
        self._config = config
        self._defects_config = defects_config
        self._engine = engine

    def run(self) -> None:
        engine = self._engine
        try:
            if engine is None:
                try:
                    engine = create_engine(self._config)
                except OcrUnavailable as exc:
                    log.warning("Text inspection not available: %s", exc)
                    self.signals.succeeded.emit(self._generation, unavailable_report(str(exc), self._config.required), None)
                    return
            r, c = self._registration, self._comparison
            exclude: tuple = ()
            if self._barcode_formats:
                try:
                    exclude = tuple(s.box for s in locate_symbols(r.reference, self._barcode_formats))
                except BarcodeUnavailable:
                    exclude = ()
            gh, gw = r.reference.shape[:2]
            exclude += boxes(self._regions, RegionKind.IGNORE, gw, gh)
            report = inspect_text(
                r.reference, r.registered, c.expected, c.combined_score, inspection_mask(r.mask, self._regions),
                self._px_per_mm,
                self._zones, engine, self._config, self._defects_config, exclude,
            )
        except Exception as exc:  # noqa: BLE001 - a failed text inspection must surface, not crash
            log.exception("Text inspection failed")
            self.signals.failed.emit(self._generation, f"Text inspection failed: {exc}")
            return
        for f in report.findings:
            log.info("Text finding %s: %s (%s, %s)", f.label, f.describe(), f.severity.value,
                     "reliable" if f.reliable else "uncertain")
        self.signals.succeeded.emit(self._generation, report, engine)


class BarcodeSignals(QObject):
    succeeded = Signal(int, object)  # generation, BarcodeReport
    failed = Signal(int, str)


class BarcodeInspectionWorker(QRunnable):
    """Inspects the barcodes of the registered print, in a thread-pool thread.

    A missing decoding library yields a report saying so (REVIEW if barcode
    inspection is required), not a failure of the worker.
    """

    def __init__(
        self,
        registration: RegistrationResult,
        comparison: ComparisonResult,
        zones: tuple[BarcodeZone, ...],
        generation: int,
        config: BarcodeConfig,
        regions: tuple[InspectionRegion, ...] = (),
    ) -> None:
        """Barcodes in ignored ``regions`` are skipped unless a zone asks for them."""
        super().__init__()
        self.setAutoDelete(False)
        self.signals = BarcodeSignals()
        self._regions = regions
        self._registration = registration
        self._comparison = comparison
        self._zones = zones
        self._generation = generation
        self._config = config

    def run(self) -> None:
        r = self._registration
        try:
            gh, gw = r.reference.shape[:2]
            report = inspect_barcodes(
                r.reference, r.registered, self._comparison.combined_score, r.mask, self._zones, self._config,
                boxes(self._regions, RegionKind.IGNORE, gw, gh),
            )
        except BarcodeUnavailable as exc:
            log.warning("Barcode inspection not available: %s", exc)
            report = barcode_unavailable_report(str(exc), self._config.required)
        except Exception as exc:  # noqa: BLE001 - a failed barcode inspection must surface, not crash
            log.exception("Barcode inspection failed")
            self.signals.failed.emit(self._generation, f"Barcode inspection failed: {exc}")
            return
        for s in report.symbols:
            log.info("Barcode %s (%s): %s, print '%s'", s.name, s.format_label, s.status.value, s.print_value)
        for f in report.findings:
            log.info("Barcode finding %s: %s (%s, %s)", f.label, f.describe(), f.severity.value,
                     "reliable" if f.reliable else "uncertain")
        self.signals.succeeded.emit(self._generation, report)


class ColorSignals(QObject):
    succeeded = Signal(int, object)  # generation, ColorReport
    failed = Signal(int, str)


class ColorAnalysisWorker(QRunnable):
    """Colour difference analysis of the registered print, in a thread-pool thread."""

    def __init__(
        self,
        registration: RegistrationResult,
        defects: DefectReport,
        generation: int,
        config: ColorConfig,
        profile: CalibrationProfile | None,
        regions: tuple[InspectionRegion, ...] = (),
        critical_region_severity: str = "major",
    ) -> None:
        """Ignored ``regions`` are not measured; findings in critical ones are raised in severity."""
        super().__init__()
        self.setAutoDelete(False)
        self.signals = ColorSignals()
        self._regions = regions
        self._critical_region_severity = critical_region_severity
        self._registration = registration
        self._defects = defects
        self._generation = generation
        self._config = config
        self._profile = profile

    def run(self) -> None:
        r = self._registration
        try:
            report = inspect_color(
                r.reference, r.registered, inspection_mask(r.mask, self._regions), self._defects,
                self._defects.px_per_mm, self._profile, self._config,
            )
            report = apply_to_color(report, self._regions, r.reference.shape, self._critical_region_severity)
        except Exception as exc:  # noqa: BLE001 - a failed colour analysis must surface, not crash
            log.exception("Colour analysis failed")
            self.signals.failed.emit(self._generation, f"Colour analysis failed: {exc}")
            return
        for f in report.findings:
            log.info("Colour finding %s: %s (%s, %s, %s)", f.label, f.describe(), f.severity.value,
                     "reliable" if f.reliable else "uncertain", report.mode)
        self.signals.succeeded.emit(self._generation, report)

