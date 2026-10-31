"""Local colour difference analysis - separate from the structural defect inspection.

**Where colour is measured.** Only where a colour difference is meaningful:
inside the flat colour areas of the artwork, away from edges (where
registration and blur dominate) and outside structural defects (missing or
extra print, shape and edge differences are not colour defects). Colours are
averaged over ``smoothing_mm`` in linear light, so halftone dots and noise
mix as they do for the eye.

**How the photo becomes CIELAB.**

* *Calibrated* - a colour profile from a photographed reference chart
  (``color_calibration``) whose leave-one-out error is within the limit maps
  the camera's linear RGB to XYZ D50. Smooth lighting falloff, measured on the
  unprinted paper, is divided out (relative, so the overall level is kept).
  Profiles from a *separate* chart photo may scale the exposure on the paper
  (luminance only). Absolute ΔE00 against the artwork is then a measurement.
* *Approximate* (no usable profile) - the photo is taken as sRGB and its white
  balance and exposure are corrected **on the unprinted paper only** (Bradford
  adaptation of the measured paper to the artwork white). Inks are never used
  to normalise, so a wrong ink stays visible - but camera processing, light
  source and the paper correction itself make the absolute values
  approximate. They are reported as such and never verify print colour.

**What is reported.**

1. *Local colour changes* - a colour that differs from the same artwork colour
   elsewhere on this label (ΔE00 of each pixel against the typical printed value
   of its colour). Relative within one photo, so camera effects largely cancel:
   meaningful in both modes.
2. *Colour area deviations* - the printed colour of a flat area against the
   artwork colour (ΔE00). Calibrated: against the tolerances. Approximate:
   only very large deviations, as uncertain findings (REVIEW).
3. *Global shift* - most colour areas deviate in the same direction.
   Approximate mode: probably lighting or camera, not the print.
4. *Lighting and white balance* - the paper's colour in the photo against the
   artwork white, its channel balance, and the lighting unevenness.

Heatmaps: ΔE00 against the artwork and ΔE00 against the same colour elsewhere
(NaN where not meaningful).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from enum import Enum

import cv2
import numpy as np

from print_inspector.config import ColorConfig
from print_inspector.core.color import (
    bgr_to_lab,
    bgr_to_linear_rgb,
    delta_e_2000,
    describe_shift,
    linear_rgb_to_xyz,
    xyz_to_lab,
    _BRADFORD,
)
from print_inspector.core.color_calibration import SEPARATE_PHOTO, CalibrationProfile
from print_inspector.core.comparison import _fit_gain, _norm_coords, _poly_terms
from print_inspector.core.defects import DefectCategory, DefectReport, DefectSeverity
from print_inspector.core.models import QualityIssue, QualityLevel, Severity, level_from_issues

log = logging.getLogger(__name__)

Box = tuple[int, int, int, int]
CALIBRATED = "calibrated"
APPROXIMATE = "approximate"
STRUCTURAL = frozenset({
    DefectCategory.MISSING_PRINT, DefectCategory.EXTRA_PRINT, DefectCategory.SHAPE_DIFFERENCE,
    DefectCategory.EDGE_DIFFERENCE, DefectCategory.POSSIBLE_TEXT,
})
MAX_COLOURS = 48
APPROXIMATE_NOTE = (
    "Approximate: uncalibrated phone photo (camera processing, lighting and white balance) - "
    "not a print colour measurement."
)


class ColorFindingKind(str, Enum):
    LOCAL_CHANGE = "local_change"
    AREA_DEVIATION = "area_deviation"
    GLOBAL_SHIFT = "global_shift"
    PAPER_COLOR = "paper_color"

    @property
    def label(self) -> str:
        return {
            ColorFindingKind.LOCAL_CHANGE: "Local colour change",
            ColorFindingKind.AREA_DEVIATION: "Colour deviation from artwork",
            ColorFindingKind.GLOBAL_SHIFT: "Global colour shift",
            ColorFindingKind.PAPER_COLOR: "Paper colour difference",
        }[self]


@dataclass(frozen=True)
class ColorFinding:
    id: int
    kind: ColorFindingKind
    x: int
    y: int
    width: int
    height: int
    delta_e: float  # ΔE00 (median over the region)
    peak_delta_e: float
    dl: float  # measured - reference
    da: float
    db: float
    area_mm2: float
    severity: DefectSeverity
    reliable: bool
    notes: tuple[str, ...]
    expected_lab: tuple[float, float, float] | None = None
    measured_lab: tuple[float, float, float] | None = None
    critical: bool = False

    @property
    def label(self) -> str:
        return f"C{self.id}"

    @property
    def is_reliable(self) -> bool:
        return self.reliable

    def describe(self) -> str:
        return f"{self.kind.label}: ΔE00 {self.delta_e:.1f} ({describe_shift(self.dl, self.da, self.db)})"


@dataclass(frozen=True)
class ColorArea:
    """A flat colour area of the artwork and its printed colour."""

    id: int
    group: int  # areas of the same artwork colour share a group
    x: int
    y: int
    width: int
    height: int
    area_mm2: float
    paper: bool
    design_lab: tuple[float, float, float]
    measured_lab: tuple[float, float, float]
    delta_e: float  # ΔE00 measured vs design
    local_spread: float  # 90th percentile of the local ΔE00 inside the area


@dataclass(frozen=True)
class ColorReport:
    mode: str  # CALIBRATED or APPROXIMATE
    mode_reasons: tuple[str, ...]
    profile: str  # description of the profile used, if any
    areas: tuple[ColorArea, ...]
    findings: tuple[ColorFinding, ...]
    issues: tuple[QualityIssue, ...]
    paper_design_lab: tuple[float, float, float] | None
    paper_photo_lab: tuple[float, float, float] | None  # before any correction (sRGB assumed / profile)
    paper_photo_delta_e: float | None
    channel_balance: tuple[float, float, float] | None  # paper R, G, B relative to G (linear)
    lighting_range: tuple[float, float]  # lighting gain over the label (1 = even)
    global_shift: tuple[float, float, float] | None  # median dL, da, db over the colour areas
    meaningful_fraction: float
    heatmap: np.ndarray  # float32 ΔE00 against the artwork, NaN where not meaningful
    local_heatmap: np.ndarray  # float32 ΔE00 against the same colour elsewhere, NaN where not meaningful
    seconds: float = 0.0
    notes: tuple[str, ...] = ()

    @property
    def calibrated(self) -> bool:
        return self.mode == CALIBRATED

    @property
    def level(self) -> QualityLevel:
        return level_from_issues(self.issues)

    @property
    def is_reliable(self) -> bool:
        return self.level is not QualityLevel.POOR

    def by_id(self, finding_id: int) -> ColorFinding | None:
        return next((f for f in self.findings if f.id == finding_id), None)


def profile_usable(profile: CalibrationProfile | None, cfg: ColorConfig) -> tuple[bool, str]:
    """Whether the profile may be used, and why not."""
    if profile is None:
        return False, "No colour calibration: the photo is not colour-calibrated."
    if profile.loo_mean_delta_e > cfg.max_calibration_delta_e:
        return False, (
            f"The colour calibration is not accurate enough (ΔE00 {profile.loo_mean_delta_e:.1f} "
            f"leave-one-out, limit {cfg.max_calibration_delta_e:.1f})."
        )
    return True, ""


# ---------------------------------------------------------------- helpers


def _bradford_to(source_xyz: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
    cone_src = _BRADFORD @ source_xyz
    cone_dst = _BRADFORD @ target_xyz
    return np.linalg.inv(_BRADFORD) @ np.diag(cone_dst / np.maximum(cone_src, 1e-9)) @ _BRADFORD


def _flat_mask(reference: np.ndarray, cfg: ColorConfig, margin_px: float) -> np.ndarray:
    flat = np.ones(reference.shape[:2], bool)
    for c in range(3):
        channel = reference[:, :, c].astype(np.float32)
        mean = cv2.blur(channel, (5, 5))
        var = cv2.blur(channel * channel, (5, 5)) - mean * mean
        flat &= np.sqrt(np.maximum(var, 0)) <= cfg.flat_tolerance
    radius = int(np.ceil(margin_px))
    if radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        flat = cv2.erode(flat.astype(np.uint8), kernel) > 0
    return flat


def _colour_groups(lab: np.ndarray, cfg: ColorConfig) -> tuple[np.ndarray, np.ndarray]:
    """Distinct artwork colours (Lab, K x 3) and each sample's group (-1: none)."""
    quantised = np.round(lab).astype(np.int32)
    keys, inverse, counts = np.unique(quantised, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    order = np.argsort(-counts)
    centres: list[np.ndarray] = []
    for k in order[: MAX_COLOURS * 8]:
        candidate = keys[k].astype(np.float64)
        if counts[k] < 20:
            break
        if centres and float(np.min(delta_e_2000(np.array(centres), candidate[None, :]))) < cfg.same_color_delta_e:
            continue
        centres.append(candidate)
        if len(centres) >= MAX_COLOURS:
            break
    if not centres:
        return np.zeros((0, 3)), np.full(len(lab), -1)
    centres_arr = np.array(centres)
    # Assign each distinct quantised colour, then every sample through it.
    distances = np.stack([delta_e_2000(keys.astype(np.float64), np.broadcast_to(c, keys.shape)) for c in centres_arr], axis=1)
    nearest = np.argmin(distances, axis=1)
    nearest[np.min(distances, axis=1) >= cfg.same_color_delta_e] = -1
    return centres_arr, nearest[inverse]


def _median_lab_of_linear(linear: np.ndarray, to_lab) -> np.ndarray:
    return to_lab(np.median(linear, axis=0)[None, :])[0]


# ---------------------------------------------------------------- main entry


def inspect_color(
    reference: np.ndarray,
    registered: np.ndarray,
    mask: np.ndarray,
    defects: DefectReport | None,
    px_per_mm: float,
    profile: CalibrationProfile | None,
    cfg: ColorConfig,
) -> ColorReport:
    """Colour analysis of the registered print against the artwork (comparison grid, BGR uint8)."""
    started = time.perf_counter()
    h, w = mask.shape
    sigma = max(0.8, cfg.smoothing_mm * px_per_mm / 2)
    px2_per_mm2 = px_per_mm * px_per_mm

    # ---- where colour is meaningful
    meaningful = _flat_mask(reference, cfg, cfg.edge_margin_px + 2 * sigma) & (mask > 0)
    if defects is not None:
        for d in defects.defects:
            if d.category in STRUCTURAL:
                pad = int(round(cfg.edge_margin_px + 2 * sigma))
                meaningful[max(0, d.y - pad) : d.y + d.height + pad, max(0, d.x - pad) : d.x + d.width + pad] = False
    ys, xs = np.nonzero(meaningful)
    issues: list[QualityIssue] = []
    notes: list[str] = []
    nan_map = np.full((h, w), np.nan, np.float32)

    # ---- artwork colours (design values, exact) and the print in linear light, smoothed
    ref_lab_px = bgr_to_lab(reference[ys, xs])
    photo_lin = cv2.GaussianBlur(bgr_to_linear_rgb(registered).astype(np.float32), (0, 0), sigma)
    ref_lin = cv2.GaussianBlur(bgr_to_linear_rgb(reference).astype(np.float32), (0, 0), sigma)
    photo_px = photo_lin[ys, xs].astype(np.float64)
    centres, group = _colour_groups(ref_lab_px, cfg) if len(ys) else (np.zeros((0, 3)), np.zeros(0, int))

    # ---- paper: the light, neutral artwork colour covering most area
    paper_group = None
    if len(centres):
        counts = np.bincount(group[group >= 0], minlength=len(centres))
        chroma = np.hypot(centres[:, 1], centres[:, 2])
        candidates = [k for k in range(len(centres)) if centres[k, 0] >= cfg.paper_min_lightness and chroma[k] <= 12]
        if candidates:
            paper_group = max(candidates, key=lambda k: counts[k])
    paper_sel = group == paper_group if paper_group is not None else np.zeros(len(ys), bool)
    if paper_sel.sum() < 200:
        paper_group = None
        paper_sel = np.zeros(len(ys), bool)

    # ---- lighting: smooth falloff measured on the paper, divided out (relative)
    lighting_range = (1.0, 1.0)
    if paper_group is not None:
        luminance = photo_px[paper_sel] @ np.array([0.2126, 0.7152, 0.0722])
        ratio = luminance / max(float(np.median(luminance)), 1e-6)
        coeffs = _fit_gain(xs[paper_sel].astype(np.float64), ys[paper_sel].astype(np.float64), ratio, (w, h), 2)
        u, v = _norm_coords(xs.astype(np.float64), ys.astype(np.float64), (w, h))
        gain = np.clip(_poly_terms(u, v, 2) @ coeffs, 0.2, 5.0)
        photo_px = photo_px / gain[:, None]
        lighting_range = (float(gain.min()), float(gain.max()))
    else:
        notes.append("No unprinted paper area in the artwork: lighting and white balance cannot be measured.")

    # ---- the photo in CIELAB
    usable, reason = profile_usable(profile, cfg)
    mode_reasons: list[str] = []
    paper_design = centres[paper_group] if paper_group is not None else None
    raw_xyz = profile.apply(photo_px) if usable else linear_rgb_to_xyz(photo_px)
    paper_raw_lab = None
    channel_balance = None
    if paper_group is not None:
        paper_rgb = np.median(photo_px[paper_sel], axis=0)
        channel_balance = tuple(float(v) for v in paper_rgb / max(float(paper_rgb[1]), 1e-6))
        paper_raw_lab = xyz_to_lab(np.median(raw_xyz[paper_sel], axis=0)[None, :])[0]
    if usable:
        mode = CALIBRATED
        xyz = raw_xyz
        if profile.source == SEPARATE_PHOTO and cfg.exposure_from_paper and paper_group is not None:
            target_y = float(xyz_from_lab(paper_design)[1])
            scale = target_y / max(float(np.median(raw_xyz[paper_sel][:, 1])), 1e-6)
            xyz = raw_xyz * scale
            notes.append(f"Exposure scaled by {scale:.2f} on the paper (profile from a separate chart photo).")
        if profile.chart_nominal:
            notes.append("The calibration uses nominal chart values; its accuracy depends on your chart.")
    else:
        mode = APPROXIMATE
        mode_reasons.append(reason)
        if paper_group is not None:
            measured_white = np.median(raw_xyz[paper_sel], axis=0)
            target_white = xyz_from_lab(paper_design)
            xyz = raw_xyz @ _bradford_to(measured_white, target_white).T
            mode_reasons.append("White balance and exposure corrected on the paper only (inks are not used).")
        else:
            xyz = raw_xyz
            mode_reasons.append("The photo is taken as sRGB without any correction.")
    photo_lab = xyz_to_lab(xyz)

    # ---- lighting / white balance report
    paper_delta = None
    if paper_raw_lab is not None and paper_design is not None:
        paper_delta = float(delta_e_2000(paper_raw_lab[None, :], paper_design[None, :])[0])
        if mode == APPROXIMATE and paper_delta >= cfg.white_balance_note_delta_e:
            issues.append(QualityIssue(
                "color_white_balance", Severity.WARNING,
                f"The paper in the photo is ΔE00 {paper_delta:.1f} from the artwork white "
                f"({describe_shift(*(paper_raw_lab - paper_design))}): lighting or white balance differs. "
                "It was corrected on the paper; a paper of a different shade cannot be told apart without calibration.",
            ))
    if lighting_range[1] / max(lighting_range[0], 1e-6) > 1.3:
        issues.append(QualityIssue(
            "color_uneven_lighting", Severity.WARNING,
            f"Uneven lighting ({lighting_range[0]:.2f} - {lighting_range[1]:.2f} across the label); corrected from "
            "the paper, but colours in the darker parts are less certain.",
        ))
    if mode == APPROXIMATE:
        issues.append(QualityIssue("color_uncalibrated", Severity.WARNING, APPROXIMATE_NOTE))

    # ---- heatmap against the artwork (smoothed the same way)
    ref_lab_smooth = xyz_to_lab(linear_rgb_to_xyz(ref_lin[ys, xs].astype(np.float64)))
    heat = nan_map.copy()
    if len(ys):
        heat[ys, xs] = delta_e_2000(photo_lab, ref_lab_smooth).astype(np.float32)

    # ---- local heatmap: against the typical print of the same artwork colour
    local = nan_map.copy()
    typical = np.zeros((len(centres), 3))
    for k in range(len(centres)):
        sel = group == k
        if sel.sum() < 50:
            continue
        typical[k] = np.median(photo_lab[sel], axis=0)
        local[ys[sel], xs[sel]] = delta_e_2000(photo_lab[sel], np.broadcast_to(typical[k], (int(sel.sum()), 3))).astype(np.float32)

    # ---- colour areas
    areas: list[ColorArea] = []
    min_area_px = cfg.min_patch_area_mm2 * px2_per_mm2
    label_img = np.full((h, w), -1, np.int32)
    label_img[ys, xs] = group
    index_img = np.full((h, w), -1, np.int64)
    index_img[ys, xs] = np.arange(len(ys))
    for k in range(len(centres)):
        count, labels, stats, _ = cv2.connectedComponentsWithStats((label_img == k).astype(np.uint8), connectivity=8)
        for i in range(1, count):
            x, y, bw, bh, area = (int(v) for v in stats[i])
            if area < min_area_px:
                continue
            idx = index_img[labels == i]
            measured = np.median(photo_lab[idx], axis=0)
            design = np.median(ref_lab_px[idx], axis=0)
            spread = float(np.nanpercentile(local[ys[idx], xs[idx]], 90)) if len(idx) else 0.0
            areas.append(ColorArea(
                len(areas) + 1, k, x, y, bw, bh, area / px2_per_mm2, k == paper_group,
                tuple(float(v) for v in design), tuple(float(v) for v in measured),
                float(delta_e_2000(measured[None, :], design[None, :])[0]), spread,
            ))

    findings: list[ColorFinding] = []

    # 1. Local colour changes (relative within this print).
    if len(ys):
        changed = np.nan_to_num(local, nan=0.0) >= cfg.local_delta_e
        changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(changed, connectivity=8)
        min_px = cfg.local_min_area_mm2 * px2_per_mm2
        for i in range(1, count):
            x, y, bw, bh, area = (int(v) for v in stats[i])
            if area < min_px:
                continue
            region = labels == i
            idx = index_img[region]
            idx = idx[idx >= 0]
            values = local[region]
            k = int(np.bincount(group[idx][group[idx] >= 0]).argmax()) if len(idx) else -1
            diff = np.median(photo_lab[idx], axis=0) - typical[k] if k >= 0 else np.zeros(3)
            median_de = float(np.nanmedian(values))
            reliable = median_de >= cfg.local_reliable_delta_e
            severity = DefectSeverity(cfg.severity_local_change)
            if median_de >= 2 * cfg.local_reliable_delta_e:
                severity = max(severity, DefectSeverity.CRITICAL, key=lambda s: s.rank)
            findings.append(ColorFinding(
                0, ColorFindingKind.LOCAL_CHANGE, x, y, bw, bh, median_de, float(np.nanmax(values)),
                float(diff[0]), float(diff[1]), float(diff[2]), area / px2_per_mm2, severity, reliable,
                (f"This area differs by ΔE00 {median_de:.1f} from the same artwork colour elsewhere on the label "
                 f"({describe_shift(*diff)}).",
                 "Relative within this photo, so camera and lighting largely cancel out; glare or a shadow on "
                 "this spot can also cause it." if not reliable else
                 "Relative within this photo, so camera and lighting largely cancel out."),
                tuple(float(v) for v in typical[k]) if k >= 0 else None,
                tuple(float(v) for v in typical[k] + diff) if k >= 0 else None,
            ))

    # 2. Colour areas against the artwork.
    ink_areas = [a for a in areas if not a.paper]
    for a in areas:
        if a.paper and mode == APPROXIMATE:
            continue  # the paper was used for the white balance
        diff = np.subtract(a.measured_lab, a.design_lab)
        if mode == CALIBRATED:
            if a.delta_e < cfg.patch_tolerance_delta_e:
                continue
            severity = DefectSeverity(cfg.severity_patch_deviation)
            if a.delta_e >= cfg.patch_critical_delta_e:
                severity = max(severity, DefectSeverity.CRITICAL, key=lambda s: s.rank)
            kind = ColorFindingKind.PAPER_COLOR if a.paper else ColorFindingKind.AREA_DEVIATION
            notes_f = (f"Calibrated measurement: ΔE00 {a.delta_e:.1f} from the artwork colour "
                       f"({describe_shift(*diff)}; tolerance {cfg.patch_tolerance_delta_e:.1f}).",)
            reliable = True
        else:
            if a.delta_e < cfg.approximate_review_delta_e:
                continue
            severity = DefectSeverity(cfg.severity_approximate_deviation)
            kind = ColorFindingKind.AREA_DEVIATION
            notes_f = (f"About ΔE00 {a.delta_e:.1f} from the artwork colour ({describe_shift(*diff)}).",
                       APPROXIMATE_NOTE + " A person must judge it; calibrate to measure.")
            reliable = False
        findings.append(ColorFinding(
            0, kind, a.x, a.y, a.width, a.height, a.delta_e, a.delta_e, *(float(v) for v in diff), a.area_mm2,
            severity, reliable, notes_f, a.design_lab, a.measured_lab,
        ))

    # 3. Global shift of the inks.
    global_shift = None
    if len(ink_areas) >= 3:
        diffs = np.array([np.subtract(a.measured_lab, a.design_lab) for a in ink_areas])
        global_shift = tuple(float(v) for v in np.median(diffs, axis=0))
        threshold = cfg.patch_tolerance_delta_e if mode == CALIBRATED else cfg.approximate_review_delta_e / 2
        mean_dir = np.median(diffs, axis=0)
        norm = np.linalg.norm(mean_dir)
        if norm > 0:
            cos = diffs @ mean_dir / (np.linalg.norm(diffs, axis=1) * norm + 1e-9)
            shifted = np.array([a.delta_e >= threshold for a in ink_areas]) & (cos > 0.7)
            if shifted.mean() >= cfg.global_shift_share:
                x0 = min(a.x for a in ink_areas)
                y0 = min(a.y for a in ink_areas)
                x1 = max(a.x + a.width for a in ink_areas)
                y1 = max(a.y + a.height for a in ink_areas)
                median_de = float(np.median([a.delta_e for a in ink_areas]))
                why = ("Calibrated: the inks as a whole are off." if mode == CALIBRATED else
                       "Uncalibrated: a shift common to all colours is most likely lighting or camera processing.")
                findings.append(ColorFinding(
                    0, ColorFindingKind.GLOBAL_SHIFT, x0, y0, x1 - x0, y1 - y0, median_de,
                    float(max(a.delta_e for a in ink_areas)), *global_shift,
                    float(sum(a.area_mm2 for a in ink_areas)), DefectSeverity(cfg.severity_global_shift),
                    mode == CALIBRATED,
                    (f"{int(shifted.sum())} of {len(ink_areas)} colour areas are shifted the same way "
                     f"({describe_shift(*global_shift)}).", why),
                ))

    order = sorted(range(len(findings)), key=lambda i: (-findings[i].severity.rank, -findings[i].delta_e))
    findings = [replace(findings[i], id=rank + 1) for rank, i in enumerate(order)]
    meaningful_fraction = float(meaningful.sum()) / float(max((mask > 0).sum(), 1))
    if meaningful_fraction < 0.05:
        notes.append("Few flat colour areas: colour differences could be measured on only a small part of the label.")
    seconds = time.perf_counter() - started
    log.info("Colour analysis (%s): %d colour areas, %d findings, %.1f s", mode, len(areas), len(findings), seconds)
    return ColorReport(
        mode, tuple(mode_reasons), profile.describe() if usable else "", tuple(areas), tuple(findings),
        tuple(issues),
        tuple(float(v) for v in paper_design) if paper_design is not None else None,
        tuple(float(v) for v in paper_raw_lab) if paper_raw_lab is not None else None,
        paper_delta, channel_balance, lighting_range, global_shift, meaningful_fraction,
        heat, local, seconds, tuple(notes),
    )


def xyz_from_lab(lab) -> np.ndarray:
    from print_inspector.core.color import lab_to_xyz

    return lab_to_xyz(np.asarray(lab, np.float64)[None, :])[0]


def heatmap_image(values: np.ndarray, max_delta_e: float, background: np.ndarray | None = None) -> np.ndarray:
    """BGR image of a ΔE00 map: 0 dark blue ... ``max_delta_e`` and above yellow; not meaningful: faded photo."""
    valid = np.isfinite(values)
    scaled = np.clip(np.nan_to_num(values, nan=0.0) / max_delta_e * 255, 0, 255).astype(np.uint8)
    coloured = cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS)
    if background is None:
        base = np.full_like(coloured, 128)
    else:
        gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        base = cv2.cvtColor((128 + gray.astype(np.float32) * 0.35).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return np.where(valid[:, :, None], coloured, base)
