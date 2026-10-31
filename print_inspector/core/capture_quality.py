"""Smartphone photo capture-quality assessment.

Decides, before any comparison, whether a photo of a printed label is good
enough to inspect. Every threshold comes from :class:`CaptureQualityConfig`
(section ``capture_quality`` of the configuration file).

Each issue is a WARNING (usable, but defects may be missed) or CRITICAL (the
photo cannot be inspected reliably). Any CRITICAL issue makes the report POOR,
and a POOR report recommends REVIEW. This module never recommends PASS.

Measurements
------------
Content region
    Canny edges on a downscaled grayscale copy. The 1st-99th percentile extent
    of the edge pixels approximates where the printed label is. This is a coarse
    estimate; precise label detection is a later step.
Blur
    Measured at full resolution in the tiles with the most edges. At each edge
    pixel the local contrast C (max - min in a window) and gradient magnitude g
    are measured. For a Gaussian-blurred step edge, g_max = C / (sigma * sqrt(2*pi)),
    so sigma = C / (sqrt(2*pi) * g). The median over all edge pixels is reported
    in pixels. A perfectly sharp edge measures about 0.8 px due to pixel sampling.
    Thin strokes make strong blur read low (measured on generated labels:
    true sigma 1 -> 1.3, 2 -> 2.0, 3 -> 2.6, 5 -> 4.2), so the configured
    thresholds are on this measured scale.
Exposure
    Share of the label area at or above ``clip_level`` (overexposure), largest
    connected blown-out blob (glare), share at or below ``dark_level`` and the
    99th-percentile brightness (underexposure).
Uneven lighting
    The label region is split into a grid. Each cell's 95th-percentile
    brightness is fitted with a smooth quadratic surface, starting from the
    brighter half of the cells; cells far darker than the fit (solid ink
    areas) are discarded iteratively. Unevenness is (max - min) / max of the
    fitted light field over the cells kept.
Incomplete capture
    Edge density in a thin band along each photo border, relative to the edge
    density inside the content region. Printed content running into the border
    suggests the label is cut off or captured with no margin.
Resolution
    Pixels per millimetre across the label's long side. The label size comes
    from the artwork's DPI when available, otherwise an assumed (conservative)
    size. Effective resolution is reduced in proportion to blur beyond
    ``sharp_edge_sigma_px``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from print_inspector.config import CaptureQualityConfig
from print_inspector.core.image_io import LoadedImage
from print_inspector.core.models import (
    InspectionStatus,
    QualityIssue,
    QualityLevel,
    Severity,
    level_from_issues,
    sort_issues,
)

SIDES = ("top", "bottom", "left", "right")
_SQRT_2PI = math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class Region:
    """Axis-aligned rectangle in full-resolution pixel coordinates."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class QualityMetrics:
    image_width: int
    image_height: int
    megapixels: float
    content_region: Region | None = None
    blur_sigma_px: float | None = None
    blur_samples: int = 0
    clipped_fraction: float | None = None
    largest_glare_fraction: float | None = None
    highlight_level: float | None = None
    dark_fraction: float | None = None
    lighting_unevenness: float | None = None
    border_content_ratios: dict[str, float] = field(default_factory=dict)
    label_long_side_px: float | None = None
    label_long_side_mm: float | None = None
    label_size_assumed: bool = True
    pixels_per_mm: float | None = None
    effective_pixels_per_mm: float | None = None


@dataclass(frozen=True)
class QualityReport:
    metrics: QualityMetrics
    issues: tuple[QualityIssue, ...]

    @property
    def level(self) -> QualityLevel:
        return level_from_issues(self.issues)

    @property
    def is_reliable(self) -> bool:
        return self.level is not QualityLevel.POOR

    @property
    def recommended_status(self) -> InspectionStatus | None:
        """REVIEW when the photo cannot be inspected reliably; otherwise None (no verdict yet)."""
        return InspectionStatus.REVIEW if self.level is QualityLevel.POOR else None

    def issue_codes(self) -> set[str]:
        return {i.code for i in self.issues}


def artwork_label_long_side_mm(artwork: LoadedImage | None, config: CaptureQualityConfig) -> float | None:
    """Physical long side of the label from the artwork DPI, if present and plausible."""
    if artwork is None:
        return None
    size = artwork.physical_size_mm
    if size is None:
        return None
    long_side = max(size)
    if not config.artwork_size_min_mm <= long_side <= config.artwork_size_max_mm:
        return None
    return long_side


def assess_capture_quality(
    bgr: np.ndarray,
    config: CaptureQualityConfig,
    label_long_side_mm: float | None = None,
) -> QualityReport:
    """Assess a photo given as an ``HxWx3`` uint8 BGR array (``LoadedImage.working``)."""
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError(f"Expected HxWx3 uint8 BGR image, got {bgr.shape} {bgr.dtype}")
    height, width = bgr.shape[:2]
    issues: list[QualityIssue] = []
    megapixels = width * height / 1_000_000
    if megapixels < config.min_megapixels_warning:
        issues.append(
            QualityIssue(
                "low_megapixels",
                Severity.WARNING,
                f"Photo is {megapixels:.1f} MP, below the recommended "
                f"{config.min_megapixels_warning:.0f} MP. Small defects may not be visible.",
            )
        )

    scale = min(1.0, config.analysis_max_side_px / max(height, width))
    small_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small_bgr = cv2.resize(bgr, small_size, interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr
    gray_small = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
    del small_bgr

    # Blown-out mask measured at full resolution, then averaged down.
    gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, clip_full = cv2.threshold(gray_full, config.clip_level - 1, 255, cv2.THRESH_BINARY)
    del gray_full
    clip_small = cv2.resize(clip_full, small_size, interpolation=cv2.INTER_AREA) if scale < 1.0 else clip_full
    del clip_full

    edges = cv2.Canny(
        cv2.GaussianBlur(gray_small, (0, 0), 1.0),
        config.edge_low_threshold,
        config.edge_high_threshold,
    )
    ys, xs = np.nonzero(edges)
    has_content = xs.size >= config.min_content_edge_pixels

    if has_content:
        x0, x1 = (int(v) for v in np.percentile(xs, [1, 99]))
        y0, y1 = (int(v) for v in np.percentile(ys, [1, 99]))
        x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    else:
        x0, y0, x1, y1 = 0, 0, small_size[0] - 1, small_size[1] - 1
        issues.append(
            QualityIssue(
                "no_content",
                Severity.CRITICAL,
                "No printed label content was found in the photo. Check that the label is in "
                "view, in focus and not washed out.",
            )
        )
    region_slice = (slice(y0, y1 + 1), slice(x0, x1 + 1))
    region_gray = gray_small[region_slice]
    region_clip = clip_small[region_slice]

    # ---- exposure
    clipped_fraction = float(np.mean(region_clip, dtype=np.float64) / 255.0)
    glare_fraction = _largest_blob_fraction(region_clip >= 128)
    highlight = float(np.percentile(region_gray, 99))
    dark_fraction = float(np.mean(region_gray <= config.dark_level))
    issues.extend(_exposure_issues(config, clipped_fraction, glare_fraction, highlight, dark_fraction))

    metrics = dict(
        image_width=width,
        image_height=height,
        megapixels=megapixels,
        clipped_fraction=clipped_fraction,
        largest_glare_fraction=glare_fraction,
        highlight_level=highlight,
        dark_fraction=dark_fraction,
    )
    if not has_content:
        return _report(metrics, issues)

    region = Region(
        x=int(x0 / scale),
        y=int(y0 / scale),
        width=int(round((x1 - x0 + 1) / scale)),
        height=int(round((y1 - y0 + 1) / scale)),
    )
    metrics["content_region"] = region

    # ---- uneven lighting
    unevenness = lighting_unevenness(region_gray, config.lighting_grid_cells)
    metrics["lighting_unevenness"] = unevenness
    if unevenness is not None:
        severity = _grade_high(unevenness, config.lighting_warning_unevenness, config.lighting_critical_unevenness)
        if severity:
            issues.append(
                QualityIssue(
                    "uneven_lighting",
                    severity,
                    f"Lighting across the label varies by about {unevenness:.0%}. Use even, "
                    "diffuse light and avoid shadows from the phone or your hand.",
                )
            )

    # ---- incomplete capture
    ratios = border_content_ratios(edges, region_slice, config.border_band_fraction)
    metrics["border_content_ratios"] = ratios
    for severity, threshold in (
        (Severity.CRITICAL, config.cutoff_critical_ratio),
        (Severity.WARNING, config.cutoff_warning_ratio),
    ):
        sides = [s for s in SIDES if ratios[s] >= threshold]
        if sides:
            issues.append(
                QualityIssue(
                    "incomplete_capture",
                    severity,
                    f"Printed content reaches the {', '.join(sides)} edge of the photo. The label "
                    "may be cut off. Capture the whole label with a margin around it.",
                )
            )
            break

    # ---- blur (full resolution)
    sigma, samples = estimate_blur_sigma(bgr, edges, (x0, y0, x1, y1), scale, config)
    metrics["blur_sigma_px"] = sigma
    metrics["blur_samples"] = samples
    if sigma is None:
        issues.append(
            QualityIssue(
                "blur_unmeasurable",
                Severity.CRITICAL,
                "Sharpness could not be measured: too few clear edges were found. The photo "
                "may be severely blurred or have very low contrast.",
            )
        )
    else:
        severity = _grade_high(sigma, config.blur_warning_sigma_px, config.blur_critical_sigma_px)
        if severity:
            issues.append(
                QualityIssue(
                    "blur",
                    severity,
                    f"Photo is blurred (edge blur about {sigma:.1f} px). Hold the phone steady, "
                    "tap to focus on the label, and use more light.",
                )
            )

    # ---- resolution
    assumed = label_long_side_mm is None
    long_mm = config.assumed_label_long_side_mm if assumed else float(label_long_side_mm)
    long_px = max(region.width, region.height)
    ppm = long_px / long_mm
    effective = ppm * min(1.0, config.sharp_edge_sigma_px / sigma) if sigma else ppm
    metrics.update(
        label_long_side_px=float(long_px),
        label_long_side_mm=long_mm,
        label_size_assumed=assumed,
        pixels_per_mm=ppm,
        effective_pixels_per_mm=effective,
    )
    severity = _grade_low(effective, config.min_pixels_per_mm_warning, config.min_pixels_per_mm_critical)
    if severity:
        basis = (
            f"assuming the label is up to {long_mm:.0f} mm long"
            if assumed
            else f"label length {long_mm:.0f} mm from the artwork"
        )
        issues.append(
            QualityIssue(
                "low_resolution",
                severity,
                f"Effective resolution is about {effective:.1f} px/mm ({basis}); at least "
                f"{config.min_pixels_per_mm_warning:.0f} px/mm is recommended. Move the camera "
                "closer so the label fills more of the frame.",
            )
        )

    return _report(metrics, issues)


def _report(metrics: dict, issues: list[QualityIssue]) -> QualityReport:
    return QualityReport(metrics=QualityMetrics(**metrics), issues=sort_issues(issues))


def _grade_high(value: float, warning: float, critical: float) -> Severity | None:
    if value >= critical:
        return Severity.CRITICAL
    if value >= warning:
        return Severity.WARNING
    return None


def _grade_low(value: float, warning: float, critical: float) -> Severity | None:
    if value < critical:
        return Severity.CRITICAL
    if value < warning:
        return Severity.WARNING
    return None


def _exposure_issues(
    config: CaptureQualityConfig,
    clipped_fraction: float,
    glare_fraction: float,
    highlight: float,
    dark_fraction: float,
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    over = _grade_high(
        clipped_fraction, config.overexposure_warning_fraction, config.overexposure_critical_fraction
    )
    if over:
        issues.append(
            QualityIssue(
                "overexposure",
                over,
                f"{clipped_fraction:.0%} of the label area is blown out to pure white, so detail "
                "there is lost. Reduce exposure or lighting.",
            )
        )
    glare = _grade_high(glare_fraction, config.glare_warning_fraction, config.glare_critical_fraction)
    # Widespread overexposure already explains a large blown-out area.
    if glare and over is not Severity.CRITICAL:
        issues.append(
            QualityIssue(
                "glare",
                glare,
                f"A glare spot covers about {glare_fraction:.1%} of the label. Change the light "
                "or camera angle to remove reflections.",
            )
        )
    by_highlight = _grade_low(
        highlight, config.underexposure_warning_highlight, config.underexposure_critical_highlight
    )
    by_dark = _grade_high(
        dark_fraction, config.underexposure_warning_dark_fraction, config.underexposure_critical_dark_fraction
    )
    under = Severity.CRITICAL if Severity.CRITICAL in (by_highlight, by_dark) else (by_highlight or by_dark)
    if under:
        issues.append(
            QualityIssue(
                "underexposure",
                under,
                f"Photo is too dark (brightest areas reach {highlight:.0f}/255; "
                f"{dark_fraction:.0%} of the label is near black). Add light or increase exposure.",
            )
        )
    return issues


def _largest_blob_fraction(mask: np.ndarray) -> float:
    if mask.size == 0 or not mask.any():
        return 0.0
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return 0.0
    return float(stats[1:, cv2.CC_STAT_AREA].max()) / mask.size


def lighting_unevenness(region_gray: np.ndarray, cells: int) -> float | None:
    """Relative brightness variation of the smooth light field over the region (0 = even)."""
    h, w = region_gray.shape
    if min(h, w) < cells * 4:
        return None
    ys = np.linspace(0, h, cells + 1).astype(int)
    xs = np.linspace(0, w, cells + 1).astype(int)
    values, cx, cy = [], [], []
    for i in range(cells):
        for j in range(cells):
            cell = region_gray[ys[i] : ys[i + 1], xs[j] : xs[j + 1]]
            values.append(float(np.percentile(cell, 95)))
            cx.append((xs[j] + xs[j + 1]) / (2.0 * w) - 0.5)
            cy.append((ys[i] + ys[i + 1]) / (2.0 * h) - 0.5)
    v = np.asarray(values)
    x = np.asarray(cx)
    y = np.asarray(cy)
    design = np.column_stack([np.ones_like(x), x, y, x * x, y * y, x * y])
    # Light falls on the substrate and ink can only be darker, so seed the fit
    # with the brighter half of the cells and let shaded paper cells rejoin.
    keep = v >= np.median(v)
    fit = v
    for _ in range(8):
        if keep.sum() < design.shape[1] + 2:
            return None
        coef, *_ = np.linalg.lstsq(design[keep], v[keep], rcond=None)
        fit = design @ coef
        residual = v - fit
        spread = 1.4826 * float(np.median(np.abs(residual[keep] - np.median(residual[keep])))) + 1.0
        # Cells much darker than the light field are solid ink, not shading.
        new_keep = residual > -2.5 * spread
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    field_values = fit[keep]
    top = float(field_values.max())
    if top <= 0:
        return None
    return float(np.clip((top - float(field_values.min())) / top, 0.0, 1.0))


def border_content_ratios(
    edges: np.ndarray, region_slice: tuple[slice, slice], band_fraction: float
) -> dict[str, float]:
    """Edge density in each border band divided by edge density in the content region."""
    h, w = edges.shape
    band = max(2, int(round(band_fraction * min(h, w))))
    present = edges > 0
    interior = float(present[region_slice].mean()) if present[region_slice].size else 0.0
    if interior <= 0:
        return {side: 0.0 for side in SIDES}
    bands = {
        "top": present[:band, :],
        "bottom": present[h - band :, :],
        "left": present[:, :band],
        "right": present[:, w - band :],
    }
    return {side: float(b.mean()) / interior for side, b in bands.items()}


def estimate_blur_sigma(
    bgr: np.ndarray,
    edges_small: np.ndarray,
    region_small: tuple[int, int, int, int],
    scale: float,
    config: CaptureQualityConfig,
) -> tuple[float | None, int]:
    """Median edge blur sigma (full-resolution px) over the most edge-rich tiles."""
    x0, y0, x1, y1 = region_small
    tile_small = max(8, int(round(config.blur_tile_size_px * scale)))
    candidates: list[tuple[int, int, int]] = []
    for ty in range(y0, y1 + 1, tile_small):
        for tx in range(x0, x1 + 1, tile_small):
            count = int(np.count_nonzero(edges_small[ty : ty + tile_small, tx : tx + tile_small]))
            if count:
                candidates.append((count, tx, ty))
    candidates.sort(reverse=True)

    full_h, full_w = bgr.shape[:2]
    samples: list[np.ndarray] = []
    for _, tx, ty in candidates[: config.blur_sample_tiles]:
        fx0, fy0 = int(tx / scale), int(ty / scale)
        fx1 = min(full_w, int((tx + tile_small) / scale))
        fy1 = min(full_h, int((ty + tile_small) / scale))
        if fx1 - fx0 < config.blur_contrast_window_px or fy1 - fy0 < config.blur_contrast_window_px:
            continue
        tile = cv2.cvtColor(bgr[fy0:fy1, fx0:fx1], cv2.COLOR_BGR2GRAY)
        samples.append(edge_blur_sigmas(tile, config))
    values = np.concatenate(samples) if samples else np.empty(0, dtype=np.float32)
    if values.size < config.blur_min_samples:
        return None, int(values.size)
    return float(np.median(values)), int(values.size)


def edge_blur_sigmas(gray_tile: np.ndarray, config: CaptureQualityConfig) -> np.ndarray:
    """Per-edge-pixel blur sigma estimates for one uint8 grayscale tile."""
    t = gray_tile.astype(np.float32) * (1.0 / 255.0)
    gx = cv2.Sobel(t, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
    gy = cv2.Sobel(t, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
    magnitude = cv2.magnitude(gx, gy)
    window = cv2.getStructuringElement(
        cv2.MORPH_RECT, (config.blur_contrast_window_px, config.blur_contrast_window_px)
    )
    contrast = cv2.dilate(t, window) - cv2.erode(t, window)
    edges = cv2.Canny(gray_tile, config.edge_low_threshold, config.edge_high_threshold)
    margin = config.blur_contrast_window_px // 2
    edges[:margin, :] = 0
    edges[-margin:, :] = 0
    edges[:, :margin] = 0
    edges[:, -margin:] = 0
    selected = (edges > 0) & (contrast >= config.blur_min_edge_contrast) & (magnitude > 1e-6)
    return contrast[selected] / (_SQRT_2PI * magnitude[selected])
