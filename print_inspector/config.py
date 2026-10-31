"""Application configuration: loading, validation and path helpers.

Defaults live in ``config/default_config.json``. An optional user override
file (``<user data dir>/config.json``) may change any subset of the values.
Every value is validated; unknown keys are rejected so typos are not
silently ignored.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
import typing
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

APP_NAME = "Print Inspector"
APP_DIR_NAME = "PrintInspector"
DEFAULT_CONFIG_RELATIVE_PATH = Path("config") / "default_config.json"
SENSITIVITY_LEVELS = ("low", "medium", "high")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
# File extensions the image loader can handle, mapped to the container format.
EXTENSION_FORMATS: dict[str, str] = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or invalid."""


def resource_path(relative: str | Path) -> Path:
    """Return the absolute path of a bundled resource.

    Works both from source and from a PyInstaller build, where resources are
    unpacked under ``sys._MEIPASS``.
    """
    base = getattr(sys, "_MEIPASS", None)
    root = Path(base) if base else Path(__file__).resolve().parent.parent
    return root / relative


def user_data_dir() -> Path:
    """Per-user local folder for logs and the optional config override."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        return Path(base) / APP_DIR_NAME
    return Path.home() / f".{APP_DIR_NAME.lower()}"


def default_user_config_path() -> Path:
    return user_data_dir() / "config.json"


@dataclass(frozen=True)
class AppInfo:
    name: str
    version: str


@dataclass(frozen=True)
class ImageLoadingConfig:
    original_extensions: tuple[str, ...]
    printed_extensions: tuple[str, ...]
    max_file_size_mb: float
    max_pixels: int


@dataclass(frozen=True)
class CaptureQualityConfig:
    """Thresholds for smartphone photo quality checks (see core/capture_quality.py).

    ``*_warning`` values mark a usable but degraded photo; ``*_critical`` values
    mark a photo that cannot be inspected reliably (result: REVIEW).
    """

    # Global analysis (exposure, lighting, content) runs on a copy downscaled to this size.
    analysis_max_side_px: int
    min_megapixels_warning: float
    # Canny thresholds (8-bit gradient units) used to find printed content edges.
    edge_low_threshold: float
    edge_high_threshold: float
    min_content_edge_pixels: int
    # Blur: estimated Gaussian edge spread (sigma, in full-resolution pixels).
    blur_tile_size_px: int
    blur_sample_tiles: int
    blur_contrast_window_px: int
    blur_min_edge_contrast: float
    blur_min_samples: int
    blur_warning_sigma_px: float
    blur_critical_sigma_px: float
    sharp_edge_sigma_px: float
    # Overexposure / glare: share of the label area at or above clip_level (0-255).
    clip_level: int
    overexposure_warning_fraction: float
    overexposure_critical_fraction: float
    glare_warning_fraction: float
    glare_critical_fraction: float
    # Underexposure: share of label area at or below dark_level, and the 99th
    # percentile brightness (the brightest parts, normally the label stock).
    dark_level: int
    underexposure_warning_dark_fraction: float
    underexposure_critical_dark_fraction: float
    underexposure_warning_highlight: float
    underexposure_critical_highlight: float
    # Uneven lighting: (brightest - darkest) / brightest of the fitted light field.
    lighting_grid_cells: int
    lighting_warning_unevenness: float
    lighting_critical_unevenness: float
    # Incomplete capture: content edge density in a border band relative to the label interior.
    border_band_fraction: float
    cutoff_warning_ratio: float
    cutoff_critical_ratio: float
    # Resolution across the label (pixels per millimetre, after blur).
    min_pixels_per_mm_warning: float
    min_pixels_per_mm_critical: float
    assumed_label_long_side_mm: float
    artwork_size_min_mm: float
    artwork_size_max_mm: float


INTERPOLATION_METHODS = ("linear", "cubic", "lanczos")


@dataclass(frozen=True)
class AlignmentConfig:
    """Label detection, perspective correction and alignment checks (core/alignment.py).

    ``*_warning`` / ``*_critical`` pairs grade the alignment like capture quality:
    any critical finding makes the alignment unreliable (result: REVIEW).
    """

    # Boundary detection runs on a copy downscaled to this size; corners are then refined at full resolution.
    detection_max_side_px: int
    min_label_area_fraction: float
    background_band_fraction: float
    # Edge refinement: search distance (detection-image px) along each side's normal.
    edge_search_px: float
    edge_samples_per_side: int
    edge_min_contrast: float
    edge_refine_iterations: int
    min_edge_support_warning: float
    min_edge_support_critical: float
    max_edge_fit_rms_px_warning: float
    max_edge_fit_rms_px_critical: float
    # Geometry of the detected quadrilateral.
    max_corner_angle_deviation_deg_warning: float
    max_corner_angle_deviation_deg_critical: float
    max_side_ratio_warning: float
    max_side_ratio_critical: float
    # Estimated physical aspect ratio vs. the artwork (relative difference).
    aspect_tolerance_warning: float
    aspect_tolerance_critical: float
    # Focal length / photo long side, used when it cannot be recovered from the label
    # (one pair of label sides parallel in the photo). Typical phone main camera: 0.75-0.8.
    assumed_focal_length_factor: float
    # Output image.
    interpolation: str
    max_output_long_side_px: int
    # Verification against the artwork (low-resolution structure comparison).
    verify_max_side_px: int
    orientation_min_margin: float
    min_similarity_warning: float
    min_similarity_critical: float
    residual_grid_cells: int
    residual_min_response: float
    residual_min_tiles: int
    residual_warning_fraction: float
    residual_critical_fraction: float
    # Feature-matching fallback (SIFT + RANSAC homography).
    feature_max_side_px: int
    feature_max_keypoints: int
    feature_ratio_test: float
    feature_ransac_threshold_px: float
    feature_min_inliers: int
    feature_min_inlier_ratio: float
    corner_agreement_warning_fraction: float
    # Largest artwork-vs-label scale difference accepted as bleed/trim (share, per axis).
    max_artwork_scale_deviation: float


@dataclass(frozen=True)
class RegistrationConfig:
    """Precise registration of the photo to the artwork (core/registration.py).

    Distances are in comparison-grid pixels (artwork pixels scaled to the photo's resolution).
    """

    max_grid_long_side_px: int
    interpolation: str
    # Global correction by feature matching (SIFT + MAGSAC++).
    feature_max_side_px: int
    feature_max_keypoints: int
    feature_ratio_test: float
    feature_ransac_threshold_px: float
    feature_min_inliers: int
    # Inliers must span this share of the grid area; clustered matches extrapolate badly.
    feature_min_spread: float
    max_global_correction_fraction: float
    # Intensity-based homography refinement (ECC), coarse to fine.
    ecc_levels: int
    ecc_max_side_px: int
    ecc_iterations: int
    ecc_epsilon: float
    ecc_gauss_filter_size: int
    # Full homography only with this many 2D-structured tiles spread over the label; else affine.
    ecc_homography_min_tiles: int
    # The artwork is blurred to match the photo before local measurements.
    blur_match_max_sigma_px: float
    # Pixels differing more than this misalignment (x gradient) could explain are ignored by ECC
    # as likely print defects (4x this value before the coarse ECC levels).
    outlier_edge_tolerance_px: float
    # Local (tile) measurements: phase correlation + Lucas-Kanade.
    tile_size_px: int
    tile_presmooth_sigma_px: float
    tile_min_structure: float
    tile_min_isotropy: float
    tile_max_shift_px: float
    tile_max_residual: float
    # Smooth local correction (lens distortion, slight non-flatness): polynomial up to this degree.
    local_poly_degree: int
    local_min_tiles: int
    local_min_rms_px: float
    local_min_improvement: float
    local_warning_correction_px: float
    # Comparison mask and verification.
    mask_border_px: int
    residual_p95_warning_px: float
    residual_p95_critical_px: float
    coverage_warning: float
    coverage_critical: float
    min_verified_tiles: int
    min_similarity_warning: float
    min_similarity_critical: float
    min_mask_fraction_warning: float
    min_mask_fraction_critical: float


@dataclass(frozen=True)
class ComparisonConfig:
    """Hybrid comparison of the registered photo with the artwork (core/comparison.py).

    ``*_sigma`` thresholds are in units of the locally measured photo noise /
    print texture, so they adapt to the camera and to halftone areas.
    """

    # Tiled processing (bounded memory on large images); margin covers all filter footprints.
    tile_size_px: int
    tile_margin_px: int
    # Additional comparisons at 1/2, 1/4, ... resolution (0 = full resolution only).
    coarse_levels: int
    # Expected residual misalignment (px); the registration's measured residual is used if larger.
    alignment_tolerance_px: float
    # Cap for the edge jitter measured per block (ragged halftone edges, local misregistration).
    max_edge_jitter_px: float
    # Allowed blur mismatch between photo and model, as half the difference of variances (px^2).
    blur_tolerance_px2: float
    # Noise / texture is estimated per block of this size.
    noise_block_px: int
    noise_floor: float
    lighting_poly_degree: int
    # Detection with hysteresis: a candidate needs a pixel above *_high_sigma and extends to *_low_sigma.
    fine_high_sigma: float
    fine_low_sigma: float
    coarse_high_sigma: float
    coarse_low_sigma: float
    min_abs_difference: float
    # Confirmation: a difference counts only with structural, edge or contrast evidence.
    ssim_sigma: float
    ssim_confirm: float
    edge_low_threshold: float
    edge_high_threshold: float
    edge_tolerance_px: int
    contrast_window_px: int
    contrast_confirm_ratio: float
    # Grouping into candidates.
    min_component_area_px: int
    merge_gap_px: int
    max_candidates: int
    candidate_area_warning_fraction: float
    candidate_area_critical_fraction: float
    # Self-check: mean SSIM of print vs. expected image at the coarsest scale.
    min_model_similarity_warning: float
    min_model_similarity_critical: float
    # Threshold multipliers for the Low / High sensitivity settings (Medium = 1).
    sensitivity_low_factor: float
    sensitivity_high_factor: float


SEVERITY_LEVELS = ("minor", "major", "critical")


@dataclass(frozen=True)
class DefectsConfig:
    """Defect localisation, categorisation and severity (core/defects.py).

    Sizes are in millimetres on the label (from the artwork DPI, otherwise from
    the assumed label size of the capture-quality settings).
    """

    # Candidates closer than this are one defect (e.g. fragments of a broken character).
    merge_distance_mm: float
    # A missing-ink and an extra-ink region this close are one shifted/misshapen element.
    shape_pair_distance_mm: float
    # A region is dropped as noise only if it is smaller than this AND weaker than
    # noise_max_score AND not reliably detected.
    noise_max_area_mm2: float
    noise_max_score: float
    # Text-like areas of the artwork (characters of this height in rows).
    text_min_height_mm: float
    text_max_height_mm: float
    # Edge difference: thin regions lying almost entirely along artwork edges.
    edge_band_px: int
    edge_min_fraction: float
    edge_max_thickness_px: float
    # Colour difference: chroma change (Lab) dominating the lightness change.
    color_min_delta_c: float
    color_max_lightness_ratio: float
    # A category is "likely" only if its deciding measurement reaches this share.
    likely_min_fraction: float
    # Reliable detection: strong score and difference, plus structural evidence.
    reliable_min_score: float
    reliable_min_difference: float
    reliable_max_ssim: float
    # Severity (largest dimension / area); text has its own size limit.
    severity_major_size_mm: float
    severity_critical_size_mm: float
    severity_major_area_mm2: float
    severity_critical_area_mm2: float
    text_major_size_mm: float
    # Differences fainter than this (gray levels) are one severity level lower.
    faint_difference: float
    # Margin around each defect in the stored evidence images (grid pixels).
    evidence_margin_px: int


@dataclass(frozen=True)
class DecisionConfig:
    """Acceptance criteria for PASS / FAIL / REVIEW (core/decision.py)."""

    # Defects of this severity or worse reject the label (if reliable or confirmed).
    fail_min_severity: str
    # More reliable/confirmed minor defects than this also reject the label.
    max_minor_defects: int
    # PASS is allowed when capture/alignment checks only gave warnings (not critical).
    allow_pass_with_warnings: bool
    # Defects and colour findings in a critical region are at least this severe, and
    # uncertain ones there always need a person (core/regions.py).
    critical_region_severity: str = "major"


@dataclass(frozen=True)
class TextConfig:
    """Offline text inspection with a locally installed Tesseract (core/ocr.py, core/text_inspection.py).

    Confidences are Tesseract word confidences (0-100). "Critical" lines are lines in
    critical text zones and, with ``numbers_are_critical``, lines containing digits.
    """

    enabled: bool
    # Without a working OCR engine the verdict is REVIEW (otherwise only a warning).
    required: bool
    # Empty = search: bundled copy, standard install folders, PATH.
    tesseract_path: str
    tessdata_dir: str
    languages: str  # Tesseract language codes, e.g. "eng" or "eng+deu"
    page_psm: int  # page segmentation mode for finding text in the artwork (11 = sparse text)
    line_psm: int  # for reading one line (7)
    zone_psm: int  # for reading a zone with expected text (6 = block)
    timeout_s: float  # per OCR call
    max_parallel: int  # OCR calls run at the same time
    min_text_height_mm: float  # smallest text height the artwork is upscaled for
    target_line_height_px: int  # lines are scaled to this height before OCR
    line_margin: float  # crop margin around a line, as a fraction of its height
    min_detect_confidence: float  # artwork words below this are not treated as text
    min_word_confidence: float  # OCR readings below this are uncertain
    critical_min_confidence: float  # same, in critical lines
    min_ocr_line_height_px: float  # lines lower than this (comparison grid) are too small for reliable OCR
    corroboration_score: float  # combined comparison score that confirms an OCR difference in the image
    ink_loss_ratio: float  # printed ink darker than this share of the expected darkness counts as present
    min_ink_contrast: float  # gray levels between paper and ink for a character to be checked
    partial_min_fraction: float  # share of a character's ink missing = partially printed
    critical_partial_min_fraction: float
    partial_reliable_fraction: float  # missing share that is unambiguous from the image alone
    partial_min_area_px: int  # missing pixels (comparison grid) below this are ignored
    numbers_are_critical: bool
    severity_missing_word: str
    severity_missing_character: str
    severity_incorrect_number: str
    severity_incorrect_character: str
    severity_missing_punctuation: str
    severity_extra_text: str
    severity_partial_character: str
    severity_unreadable_text: str
    critical_line_severity: str  # findings in critical lines are at least this severe


BARCODE_FORMATS = ("ean13", "upca", "code128", "qr")


@dataclass(frozen=True)
class BarcodeConfig:
    """Offline barcode and QR code inspection with ZXing-C++ (core/barcodes.py, core/barcode_inspection.py).

    These are decoding checks and measurements from a photo, not ISO/IEC 15416 /
    15415 print-quality grades (which need a calibrated verifier).
    """

    enabled: bool
    # Without the decoding library the verdict is REVIEW (otherwise only a warning).
    required: bool
    formats: tuple[str, ...]  # symbologies looked for: ean13, upca, code128, qr
    region_margin_modules: float  # search margin around each symbol, in modules
    scan_strips: int  # a linear symbol is also decoded in this many horizontal strips
    min_scan_success: float  # share of strips that must decode
    min_unused_ec: float  # QR: share of the error correction that must remain unused
    quiet_zone_modules: float  # linear symbols: quiet zone checked on each side (at most what the artwork has)
    qr_quiet_zone_modules: float
    max_bar_growth_modules: float  # average bar width change (ink spread) still accepted
    min_damage_modules2: float  # damaged area (in square modules) worth reporting
    damage_corroboration_score: float  # comparison score that confirms damage in the image
    severity_wrong_content: str
    severity_not_decoded: str
    severity_damage: str  # damage that the decoding still copes with
    severity_damage_affecting_decoding: str
    severity_quiet_zone: str
    severity_low_margin: str
    severity_bar_width: str
    severity_wrong_format: str
    severity_unexpected_barcode: str
    critical_zone_severity: str  # findings in critical barcode zones are at least this severe


@dataclass(frozen=True)
class ColorConfig:
    """Local colour difference analysis (core/color*.py). Colour differences are CIEDE2000 (ΔE00) in CIELAB D50.

    Without a colour calibration the measurements are approximate (phone camera
    processing, lighting and white balance); they never count as print colour
    verification and can at most lead to REVIEW.
    """

    enabled: bool
    smoothing_mm: float  # colours are averaged over this scale (halftone dots, noise)
    edge_margin_px: float  # distance from artwork edges, where misregistration dominates
    flat_tolerance: float  # artwork colour variation (gray levels) still counted as a flat colour area
    min_patch_area_mm2: float  # flat colour areas smaller than this are not measured
    same_color_delta_e: float  # artwork colours closer than this are one colour
    paper_min_lightness: float  # L* of the artwork colour that counts as unprinted paper
    local_delta_e: float  # a colour this far from the same colour elsewhere on the label is a local change
    local_reliable_delta_e: float
    local_min_area_mm2: float
    patch_tolerance_delta_e: float  # calibrated: a colour area this far from the artwork is a deviation
    patch_critical_delta_e: float
    approximate_review_delta_e: float  # uncalibrated: only deviations this large are reported (REVIEW)
    global_shift_share: float  # share of colour areas shifted the same way = global shift
    white_balance_note_delta_e: float  # paper this far from the artwork white: white balance / lighting note
    max_calibration_delta_e: float  # calibration accepted if its leave-one-out mean ΔE00 is at most this
    calibration_profile: str  # profile file; empty = the profile saved by the application, if any
    exposure_from_paper: bool  # profiles from a separate chart photo: scale exposure on the paper (luminance only)
    heatmap_max_delta_e: float  # ΔE00 shown at full colour in the heatmaps
    severity_local_change: str
    severity_patch_deviation: str
    severity_approximate_deviation: str
    severity_global_shift: str


@dataclass(frozen=True)
class InspectionConfig:
    sensitivity: str


@dataclass(frozen=True)
class ReportConfig:
    """Saved results and the PDF report (core/results_export.py, core/report_pdf.py)."""

    pdf: bool  # write report.pdf with the results
    title: str
    organisation: str  # printed in the report header (optional)
    inspector: str  # name of the person inspecting, printed with the operator decisions (optional)
    max_evidence_crops: int  # defect crops shown in the PDF (all are saved as PNG files)


@dataclass(frozen=True)
class UIConfig:
    window_width: int
    window_height: int
    zoom_step: float
    min_zoom: float
    max_zoom: float


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    max_bytes: int
    backup_count: int


@dataclass(frozen=True)
class AppConfig:
    app: AppInfo
    image_loading: ImageLoadingConfig
    capture_quality: CaptureQualityConfig
    alignment: AlignmentConfig
    registration: RegistrationConfig
    comparison: ComparisonConfig
    defects: DefectsConfig
    decision: DecisionConfig
    text: TextConfig
    barcodes: BarcodeConfig
    color: ColorConfig
    inspection: InspectionConfig
    report: ReportConfig
    ui: UIConfig
    logging: LoggingConfig


_SECTION_KEYS: dict[str, tuple[str, ...]] = {
    "app": ("name", "version"),
    "image_loading": (
        "original_extensions",
        "printed_extensions",
        "max_file_size_mb",
        "max_pixels",
    ),
    "capture_quality": tuple(f.name for f in fields(CaptureQualityConfig)),
    "alignment": tuple(f.name for f in fields(AlignmentConfig)),
    "registration": tuple(f.name for f in fields(RegistrationConfig)),
    "comparison": tuple(f.name for f in fields(ComparisonConfig)),
    "defects": tuple(f.name for f in fields(DefectsConfig)),
    "decision": tuple(f.name for f in fields(DecisionConfig)),
    "text": tuple(f.name for f in fields(TextConfig)),
    "barcodes": tuple(f.name for f in fields(BarcodeConfig)),
    "color": tuple(f.name for f in fields(ColorConfig)),
    "inspection": ("sensitivity",),
    "report": tuple(f.name for f in fields(ReportConfig)),
    "ui": ("window_width", "window_height", "zoom_step", "min_zoom", "max_zoom"),
    "logging": ("level", "max_bytes", "backup_count"),
}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read configuration file {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"Invalid JSON in {path} (line {exc.lineno}, column {exc.colno}): {exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a JSON object.")
    return data


def _check_keys(section: Mapping[str, Any], name: str, allowed: tuple[str, ...]) -> None:
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise ConfigError(f"Unknown setting(s) in '{name}': {', '.join(unknown)}")


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"Section '{name}' is missing or is not an object.")
    _check_keys(value, name, _SECTION_KEYS[name])
    return value


def _get(section: Mapping[str, Any], section_name: str, key: str, kind: type) -> Any:
    full = f"{section_name}.{key}"
    if key not in section:
        raise ConfigError(f"Missing setting '{full}'.")
    value = section[key]
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind is float:
        ok = is_number
    elif kind is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    else:
        ok = isinstance(value, kind)
    if not ok:
        raise ConfigError(f"Setting '{full}' must be of type {kind.__name__}, got {value!r}.")
    return float(value) if kind is float else value


def _positive(value: float, name: str) -> None:
    if not value > 0:
        raise ConfigError(f"Setting '{name}' must be greater than 0, got {value!r}.")


def _extensions(section: Mapping[str, Any], section_name: str, key: str) -> tuple[str, ...]:
    raw = _get(section, section_name, key, list)
    if not raw:
        raise ConfigError(f"Setting '{section_name}.{key}' must list at least one extension.")
    result: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.startswith(".") or len(item) < 2:
            raise ConfigError(
                f"Invalid extension {item!r} in '{section_name}.{key}' (expected e.g. \".png\")."
            )
        if item.lower() not in EXTENSION_FORMATS:
            raise ConfigError(
                f"Unsupported extension {item!r} in '{section_name}.{key}'. "
                f"Supported: {', '.join(EXTENSION_FORMATS)}."
            )
        result.append(item.lower())
    return tuple(dict.fromkeys(result))


def _parse_capture_quality(s: Mapping[str, Any]) -> CaptureQualityConfig:
    name = "capture_quality"
    hints = typing.get_type_hints(CaptureQualityConfig)
    values = {f.name: _get(s, name, f.name, hints[f.name]) for f in fields(CaptureQualityConfig)}
    cfg = CaptureQualityConfig(**values)
    for key, value in values.items():
        _positive(value, f"{name}.{key}")

    for key in (
        "overexposure_warning_fraction",
        "overexposure_critical_fraction",
        "glare_warning_fraction",
        "glare_critical_fraction",
        "underexposure_warning_dark_fraction",
        "underexposure_critical_dark_fraction",
        "lighting_warning_unevenness",
        "lighting_critical_unevenness",
        "border_band_fraction",
        "blur_min_edge_contrast",
    ):
        if values[key] > 1.0:
            raise ConfigError(f"Setting '{name}.{key}' must be between 0 and 1.")
    for key in (
        "clip_level",
        "dark_level",
        "underexposure_warning_highlight",
        "underexposure_critical_highlight",
    ):
        if values[key] > 255:
            raise ConfigError(f"Setting '{name}.{key}' must be between 0 and 255.")
    if cfg.blur_contrast_window_px < 3 or cfg.blur_contrast_window_px % 2 == 0:
        raise ConfigError(f"Setting '{name}.blur_contrast_window_px' must be an odd number >= 3.")
    if cfg.border_band_fraction >= 0.25:
        raise ConfigError(f"Setting '{name}.border_band_fraction' must be below 0.25.")

    # (lower, higher) pairs: a worse photo moves a metric past warning before critical.
    ordered = [
        ("edge_low_threshold", "edge_high_threshold"),
        ("blur_warning_sigma_px", "blur_critical_sigma_px"),
        ("overexposure_warning_fraction", "overexposure_critical_fraction"),
        ("glare_warning_fraction", "glare_critical_fraction"),
        ("underexposure_warning_dark_fraction", "underexposure_critical_dark_fraction"),
        ("underexposure_critical_highlight", "underexposure_warning_highlight"),
        ("lighting_warning_unevenness", "lighting_critical_unevenness"),
        ("cutoff_warning_ratio", "cutoff_critical_ratio"),
        ("min_pixels_per_mm_critical", "min_pixels_per_mm_warning"),
        ("artwork_size_min_mm", "artwork_size_max_mm"),
    ]
    for low, high in ordered:
        if not values[low] < values[high]:
            raise ConfigError(f"Setting '{name}.{low}' must be less than '{name}.{high}'.")
    return cfg


def _parse_alignment(s: Mapping[str, Any]) -> AlignmentConfig:
    name = "alignment"
    hints = typing.get_type_hints(AlignmentConfig)
    values = {f.name: _get(s, name, f.name, hints[f.name]) for f in fields(AlignmentConfig)}
    values["interpolation"] = values["interpolation"].lower()
    cfg = AlignmentConfig(**values)
    for key, value in values.items():
        if not isinstance(value, str):
            _positive(value, f"{name}.{key}")
    if cfg.interpolation not in INTERPOLATION_METHODS:
        raise ConfigError(
            f"Setting '{name}.interpolation' must be one of {', '.join(INTERPOLATION_METHODS)}."
        )
    for key in (
        "min_label_area_fraction",
        "min_edge_support_warning",
        "min_edge_support_critical",
        "min_similarity_warning",
        "min_similarity_critical",
        "residual_min_response",
        "feature_ratio_test",
        "feature_min_inlier_ratio",
        "orientation_min_margin",
    ):
        if values[key] > 1.0:
            raise ConfigError(f"Setting '{name}.{key}' must be between 0 and 1.")
    if cfg.background_band_fraction >= 0.25:
        raise ConfigError(f"Setting '{name}.background_band_fraction' must be below 0.25.")
    if cfg.max_corner_angle_deviation_deg_critical >= 90:
        raise ConfigError(f"Setting '{name}.max_corner_angle_deviation_deg_critical' must be below 90.")
    if cfg.max_side_ratio_warning <= 1.0:
        raise ConfigError(f"Setting '{name}.max_side_ratio_warning' must be greater than 1.")
    if cfg.residual_grid_cells < 2:
        raise ConfigError(f"Setting '{name}.residual_grid_cells' must be at least 2.")
    if cfg.residual_min_tiles < 4:
        raise ConfigError(f"Setting '{name}.residual_min_tiles' must be at least 4.")

    ordered = [
        ("min_edge_support_critical", "min_edge_support_warning"),
        ("max_edge_fit_rms_px_warning", "max_edge_fit_rms_px_critical"),
        ("max_corner_angle_deviation_deg_warning", "max_corner_angle_deviation_deg_critical"),
        ("max_side_ratio_warning", "max_side_ratio_critical"),
        ("aspect_tolerance_warning", "aspect_tolerance_critical"),
        ("min_similarity_critical", "min_similarity_warning"),
        ("residual_warning_fraction", "residual_critical_fraction"),
    ]
    for low, high in ordered:
        if not values[low] < values[high]:
            raise ConfigError(f"Setting '{name}.{low}' must be less than '{name}.{high}'.")
    return cfg


def parse_config(data: Mapping[str, Any]) -> AppConfig:
    """Validate a raw configuration mapping and build an :class:`AppConfig`."""
    _check_keys(data, "top level", tuple(_SECTION_KEYS))

    s = _section(data, "app")
    app = AppInfo(name=_get(s, "app", "name", str), version=_get(s, "app", "version", str))

    s = _section(data, "image_loading")
    image_loading = ImageLoadingConfig(
        original_extensions=_extensions(s, "image_loading", "original_extensions"),
        printed_extensions=_extensions(s, "image_loading", "printed_extensions"),
        max_file_size_mb=_get(s, "image_loading", "max_file_size_mb", float),
        max_pixels=_get(s, "image_loading", "max_pixels", int),
    )
    _positive(image_loading.max_file_size_mb, "image_loading.max_file_size_mb")
    _positive(image_loading.max_pixels, "image_loading.max_pixels")

    capture_quality = _parse_capture_quality(_section(data, "capture_quality"))
    alignment = _parse_alignment(_section(data, "alignment"))
    registration = _parse_registration(_section(data, "registration"))
    comparison = _parse_comparison(_section(data, "comparison"))
    defects = _parse_defects(_section(data, "defects"))
    decision = _parse_decision(_section(data, "decision"))
    text = _parse_text(_section(data, "text"))
    barcodes = _parse_barcodes(_section(data, "barcodes"))
    color = _parse_color(_section(data, "color"))

    s = _section(data, "inspection")
    sensitivity = _get(s, "inspection", "sensitivity", str).lower()
    if sensitivity not in SENSITIVITY_LEVELS:
        raise ConfigError(
            f"Setting 'inspection.sensitivity' must be one of {', '.join(SENSITIVITY_LEVELS)}."
        )
    inspection = InspectionConfig(sensitivity=sensitivity)

    s = _section(data, "report")
    report = ReportConfig(
        pdf=_get(s, "report", "pdf", bool),
        title=_get(s, "report", "title", str).strip() or "Print Inspection Report",
        organisation=_get(s, "report", "organisation", str).strip(),
        inspector=_get(s, "report", "inspector", str).strip(),
        max_evidence_crops=_get(s, "report", "max_evidence_crops", int),
    )
    if report.max_evidence_crops < 0:
        raise ConfigError("Setting 'report.max_evidence_crops' must be 0 or greater.")

    s = _section(data, "ui")
    ui = UIConfig(
        window_width=_get(s, "ui", "window_width", int),
        window_height=_get(s, "ui", "window_height", int),
        zoom_step=_get(s, "ui", "zoom_step", float),
        min_zoom=_get(s, "ui", "min_zoom", float),
        max_zoom=_get(s, "ui", "max_zoom", float),
    )
    _positive(ui.window_width, "ui.window_width")
    _positive(ui.window_height, "ui.window_height")
    _positive(ui.min_zoom, "ui.min_zoom")
    if ui.zoom_step <= 1.0:
        raise ConfigError("Setting 'ui.zoom_step' must be greater than 1.0.")
    if ui.max_zoom <= ui.min_zoom:
        raise ConfigError("Setting 'ui.max_zoom' must be greater than 'ui.min_zoom'.")

    s = _section(data, "logging")
    level = _get(s, "logging", "level", str).upper()
    if level not in LOG_LEVELS:
        raise ConfigError(f"Setting 'logging.level' must be one of {', '.join(LOG_LEVELS)}.")
    logging_cfg = LoggingConfig(
        level=level,
        max_bytes=_get(s, "logging", "max_bytes", int),
        backup_count=_get(s, "logging", "backup_count", int),
    )
    _positive(logging_cfg.max_bytes, "logging.max_bytes")
    if logging_cfg.backup_count < 0:
        raise ConfigError("Setting 'logging.backup_count' must be 0 or greater.")

    return AppConfig(
        app=app,
        image_loading=image_loading,
        capture_quality=capture_quality,
        alignment=alignment,
        registration=registration,
        comparison=comparison,
        defects=defects,
        decision=decision,
        text=text,
        barcodes=barcodes,
        color=color,
        inspection=inspection,
        report=report,
        ui=ui,
        logging=logging_cfg,
    )


def _parse_registration(s: Mapping[str, Any]) -> RegistrationConfig:
    name = "registration"
    hints = typing.get_type_hints(RegistrationConfig)
    values = {f.name: _get(s, name, f.name, hints[f.name]) for f in fields(RegistrationConfig)}
    values["interpolation"] = values["interpolation"].lower()
    cfg = RegistrationConfig(**values)
    for key, value in values.items():
        if not isinstance(value, str):
            _positive(value, f"{name}.{key}")
    if cfg.interpolation not in INTERPOLATION_METHODS:
        raise ConfigError(
            f"Setting '{name}.interpolation' must be one of {', '.join(INTERPOLATION_METHODS)}."
        )
    for key in (
        "feature_ratio_test",
        "feature_min_spread",
        "max_global_correction_fraction",
        "tile_min_isotropy",
        "local_min_improvement",
        "coverage_warning",
        "coverage_critical",
        "min_similarity_warning",
        "min_similarity_critical",
        "min_mask_fraction_warning",
        "min_mask_fraction_critical",
    ):
        if values[key] > 1.0:
            raise ConfigError(f"Setting '{name}.{key}' must be between 0 and 1.")
    if cfg.ecc_gauss_filter_size % 2 == 0:
        raise ConfigError(f"Setting '{name}.ecc_gauss_filter_size' must be an odd number.")
    if cfg.tile_size_px < 32:
        raise ConfigError(f"Setting '{name}.tile_size_px' must be at least 32.")
    if cfg.local_poly_degree > 5:
        raise ConfigError(f"Setting '{name}.local_poly_degree' must be 5 or less.")
    ordered = [
        ("residual_p95_warning_px", "residual_p95_critical_px"),
        ("coverage_critical", "coverage_warning"),
        ("min_similarity_critical", "min_similarity_warning"),
        ("min_mask_fraction_critical", "min_mask_fraction_warning"),
    ]
    for low, high in ordered:
        if not values[low] < values[high]:
            raise ConfigError(f"Setting '{name}.{low}' must be less than '{name}.{high}'.")
    return cfg


def _parse_comparison(s: Mapping[str, Any]) -> ComparisonConfig:
    name = "comparison"
    hints = typing.get_type_hints(ComparisonConfig)
    values = {f.name: _get(s, name, f.name, hints[f.name]) for f in fields(ComparisonConfig)}
    cfg = ComparisonConfig(**values)
    for key, value in values.items():
        if key != "coarse_levels":
            _positive(value, f"{name}.{key}")
    if cfg.max_edge_jitter_px < cfg.alignment_tolerance_px:
        raise ConfigError(f"Setting '{name}.max_edge_jitter_px' must be at least '{name}.alignment_tolerance_px'.")
    if not 0 <= cfg.coarse_levels <= 4:
        raise ConfigError(f"Setting '{name}.coarse_levels' must be between 0 and 4.")
    if cfg.tile_size_px < 128:
        raise ConfigError(f"Setting '{name}.tile_size_px' must be at least 128.")
    if cfg.contrast_window_px % 2 == 0:
        raise ConfigError(f"Setting '{name}.contrast_window_px' must be an odd number.")
    if cfg.ssim_confirm > 1.0 or cfg.min_model_similarity_warning > 1.0:
        raise ConfigError(f"Setting '{name}.ssim_confirm' must be between 0 and 1.")
    if cfg.contrast_confirm_ratio <= 1.0:
        raise ConfigError(f"Setting '{name}.contrast_confirm_ratio' must be greater than 1.")
    if not cfg.sensitivity_high_factor < 1.0 < cfg.sensitivity_low_factor:
        raise ConfigError(
            f"Settings '{name}.sensitivity_high_factor' must be below 1 and "
            f"'{name}.sensitivity_low_factor' above 1."
        )
    ordered = [
        ("fine_low_sigma", "fine_high_sigma"),
        ("coarse_low_sigma", "coarse_high_sigma"),
        ("edge_low_threshold", "edge_high_threshold"),
        ("candidate_area_warning_fraction", "candidate_area_critical_fraction"),
        ("min_model_similarity_critical", "min_model_similarity_warning"),
    ]
    for low, high in ordered:
        if not values[low] < values[high]:
            raise ConfigError(f"Setting '{name}.{low}' must be less than '{name}.{high}'.")
    if cfg.candidate_area_critical_fraction > 1.0:
        raise ConfigError(f"Setting '{name}.candidate_area_critical_fraction' must be at most 1.")
    if cfg.tile_margin_px * 4 >= cfg.tile_size_px:
        raise ConfigError(f"Setting '{name}.tile_margin_px' must be less than a quarter of the tile size.")
    return cfg


def _parse_defects(s: Mapping[str, Any]) -> DefectsConfig:
    name = "defects"
    hints = typing.get_type_hints(DefectsConfig)
    values = {f.name: _get(s, name, f.name, hints[f.name]) for f in fields(DefectsConfig)}
    cfg = DefectsConfig(**values)
    for key, value in values.items():
        _positive(value, f"{name}.{key}")
    for key in ("edge_min_fraction", "color_max_lightness_ratio", "likely_min_fraction", "reliable_max_ssim"):
        if values[key] > 1.0:
            raise ConfigError(f"Setting '{name}.{key}' must be between 0 and 1.")
    ordered = [
        ("text_min_height_mm", "text_max_height_mm"),
        ("severity_major_size_mm", "severity_critical_size_mm"),
        ("severity_major_area_mm2", "severity_critical_area_mm2"),
    ]
    for low, high in ordered:
        if not values[low] < values[high]:
            raise ConfigError(f"Setting '{name}.{low}' must be less than '{name}.{high}'.")
    return cfg


def _parse_decision(s: Mapping[str, Any]) -> DecisionConfig:
    name = "decision"
    severity = _get(s, name, "fail_min_severity", str).lower()
    if severity not in SEVERITY_LEVELS:
        raise ConfigError(f"Setting '{name}.fail_min_severity' must be one of {', '.join(SEVERITY_LEVELS)}.")
    max_minor = _get(s, name, "max_minor_defects", int)
    if max_minor < 0:
        raise ConfigError(f"Setting '{name}.max_minor_defects' must be 0 or greater.")
    region_severity = _get(s, name, "critical_region_severity", str).lower()
    if region_severity not in SEVERITY_LEVELS:
        raise ConfigError(f"Setting '{name}.critical_region_severity' must be one of {', '.join(SEVERITY_LEVELS)}.")
    return DecisionConfig(
        fail_min_severity=severity,
        max_minor_defects=max_minor,
        allow_pass_with_warnings=_get(s, name, "allow_pass_with_warnings", bool),
        critical_region_severity=region_severity,
    )


def _parse_text(s: Mapping[str, Any]) -> TextConfig:
    name = "text"
    hints = typing.get_type_hints(TextConfig)
    values = {f.name: _get(s, name, f.name, hints[f.name]) for f in fields(TextConfig)}
    cfg = TextConfig(**values)
    if not re.fullmatch(r"[A-Za-z_]+(\+[A-Za-z_]+)*", cfg.languages):
        raise ConfigError(f"Setting '{name}.languages' must be Tesseract language codes such as 'eng' or 'eng+deu'.")
    for key in ("page_psm", "line_psm", "zone_psm"):
        if not 0 <= values[key] <= 13:
            raise ConfigError(f"Setting '{name}.{key}' must be a Tesseract page segmentation mode (0-13).")
    for key in ("timeout_s", "max_parallel", "min_text_height_mm", "min_ocr_line_height_px",
                "corroboration_score", "min_ink_contrast", "partial_min_area_px"):
        _positive(values[key], f"{name}.{key}")
    if cfg.target_line_height_px < 16:
        raise ConfigError(f"Setting '{name}.target_line_height_px' must be at least 16.")
    for key in ("min_detect_confidence", "min_word_confidence", "critical_min_confidence"):
        if not 0 <= values[key] <= 100:
            raise ConfigError(f"Setting '{name}.{key}' must be between 0 and 100.")
    for key in ("line_margin", "ink_loss_ratio", "partial_min_fraction", "critical_partial_min_fraction",
                "partial_reliable_fraction"):
        if not 0 < values[key] <= 1:
            raise ConfigError(f"Setting '{name}.{key}' must be between 0 and 1.")
    if cfg.critical_min_confidence < cfg.min_word_confidence:
        raise ConfigError(f"Setting '{name}.critical_min_confidence' must be at least '{name}.min_word_confidence'.")
    if cfg.critical_partial_min_fraction > cfg.partial_min_fraction:
        raise ConfigError(
            f"Setting '{name}.critical_partial_min_fraction' must not exceed '{name}.partial_min_fraction'."
        )
    for key in values:
        if key.startswith("severity_") or key == "critical_line_severity":
            values[key] = values[key].lower()
            if values[key] not in SEVERITY_LEVELS:
                raise ConfigError(f"Setting '{name}.{key}' must be one of {', '.join(SEVERITY_LEVELS)}.")
    return TextConfig(**values)


def _parse_barcodes(s: Mapping[str, Any]) -> BarcodeConfig:
    name = "barcodes"
    values: dict[str, Any] = {}
    for f in fields(BarcodeConfig):
        if f.name == "formats":
            raw = s.get("formats")
            if not isinstance(raw, list) or not raw or not all(isinstance(v, str) for v in raw):
                raise ConfigError(f"Setting '{name}.formats' must be a non-empty list of names.")
            unknown = sorted({v.lower() for v in raw} - set(BARCODE_FORMATS))
            if unknown:
                raise ConfigError(
                    f"Setting '{name}.formats' has unsupported format(s) {', '.join(unknown)}; "
                    f"supported: {', '.join(BARCODE_FORMATS)}."
                )
            values["formats"] = tuple(dict.fromkeys(v.lower() for v in raw))
            continue
        kind = typing.get_type_hints(BarcodeConfig)[f.name]
        values[f.name] = _get(s, name, f.name, kind)
    for key in ("region_margin_modules", "quiet_zone_modules", "qr_quiet_zone_modules", "max_bar_growth_modules",
                "min_damage_modules2", "damage_corroboration_score"):
        _positive(values[key], f"{name}.{key}")
    if not 3 <= values["scan_strips"] <= 50:
        raise ConfigError(f"Setting '{name}.scan_strips' must be between 3 and 50.")
    for key in ("min_scan_success", "min_unused_ec"):
        if not 0 <= values[key] <= 1:
            raise ConfigError(f"Setting '{name}.{key}' must be between 0 and 1.")
    for key in values:
        if key.startswith("severity_") or key == "critical_zone_severity":
            values[key] = values[key].lower()
            if values[key] not in SEVERITY_LEVELS:
                raise ConfigError(f"Setting '{name}.{key}' must be one of {', '.join(SEVERITY_LEVELS)}.")
    return BarcodeConfig(**values)


def _parse_color(s: Mapping[str, Any]) -> ColorConfig:
    name = "color"
    hints = typing.get_type_hints(ColorConfig)
    values = {f.name: _get(s, name, f.name, hints[f.name]) for f in fields(ColorConfig)}
    for key, value in values.items():
        if hints[key] is float:
            _positive(value, f"{name}.{key}")
    if not 0 < values["global_shift_share"] <= 1:
        raise ConfigError(f"Setting '{name}.global_shift_share' must be between 0 and 1.")
    if values["paper_min_lightness"] > 100:
        raise ConfigError(f"Setting '{name}.paper_min_lightness' must be at most 100.")
    for low, high in (("local_delta_e", "local_reliable_delta_e"), ("patch_tolerance_delta_e", "patch_critical_delta_e")):
        if values[low] > values[high]:
            raise ConfigError(f"Setting '{name}.{low}' must not exceed '{name}.{high}'.")
    for key in values:
        if key.startswith("severity_"):
            values[key] = values[key].lower()
            if values[key] not in SEVERITY_LEVELS:
                raise ConfigError(f"Setting '{name}.{key}' must be one of {', '.join(SEVERITY_LEVELS)}.")
    return ColorConfig(**values)


def load_config(default_path: Path | None = None, user_path: Path | None = None) -> AppConfig:
    """Load the bundled defaults, apply an optional user override, and validate.

    ``user_path`` is applied only if the file exists.
    """
    data = _read_json(default_path or resource_path(DEFAULT_CONFIG_RELATIVE_PATH))
    if user_path is not None and Path(user_path).is_file():
        data = deep_merge(data, _read_json(Path(user_path)))
    return parse_config(data)


# ---------------------------------------------------------------- user settings file


def default_config_data() -> dict[str, Any]:
    """The bundled default configuration as a raw mapping."""
    return _read_json(resource_path(DEFAULT_CONFIG_RELATIVE_PATH))


def read_user_overrides(path: Path) -> dict[str, Any]:
    """The user override file as a raw mapping; empty if the file does not exist."""
    path = Path(path)
    return _read_json(path) if path.is_file() else {}


def config_with_overrides(overrides: Mapping[str, Any]) -> AppConfig:
    """Defaults plus ``overrides``, validated (raises ConfigError)."""
    return parse_config(deep_merge(default_config_data(), overrides))


def config_data(config: AppConfig) -> dict[str, Any]:
    """``config`` as a raw mapping that ``parse_config`` accepts (the inverse of parsing)."""

    def raw(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return [raw(v) for v in value]
        return value

    return {
        f.name: {g.name: raw(getattr(getattr(config, f.name), g.name)) for g in fields(getattr(config, f.name))}
        for f in fields(config)
    }


def config_overrides(config: AppConfig) -> dict[str, Any]:
    """The values of ``config`` that differ from the bundled defaults, as an override mapping."""
    defaults = default_config_data()
    out: dict[str, Any] = {}
    for section, values in config_data(config).items():
        changed = {k: v for k, v in values.items() if defaults.get(section, {}).get(k) != v}
        if changed:
            out[section] = changed
    return out


def write_user_overrides(path: Path, overrides: Mapping[str, Any]) -> None:
    """Write the override file atomically (a crash never leaves half a file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(overrides, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
