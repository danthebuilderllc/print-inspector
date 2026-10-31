"""Hybrid comparison of the registered photo with the artwork.

Input: the outputs of registration - the photo resampled onto the comparison
grid, the artwork on the same grid and the comparison mask.

1. Expected print (what the artwork should look like in this photo)
   ``expected = gain(x, y) * T_c(blur(artwork))`` per colour channel c:
   - ``blur``: the photo's measured optical blur (from registration);
   - ``T_c``: monotonic tone curve per channel, fitted as binned medians of
     photo vs. artwork over the whole label (robust to defects);
   - ``gain``: smooth lighting field, a low-order polynomial fitted robustly on
     bare paper. Being so smooth, it cannot absorb local defects.
   Nothing is smoothed on the photo side, so tiny defects keep full contrast.

2. Noise and texture model
   The residual (photo - expected) is summarised per block by its median
   absolute deviation, ignoring pixels at artwork edges, and separately for
   two classes of artwork pixels: tints (screened in print, so textured) and
   solid ink / bare stock (smooth). Each pixel uses its own class's level, so
   a block that straddles a tint and paper does not lend the paper's low
   noise to the tint. All thresholds are in these units (sigma), so they
   adapt to camera, lighting and print texture, while a defect occupying part
   of a block still stands out.

3. Per-pixel evidence at each scale (full resolution and 1/2, 1/4, ...):
   - absolute difference (max over colour channels) minus an *edge allowance*.
     Residual misalignment, halftone screening of edges, ink spread and JPEG
     colour subsampling move or soften edges. By Taylor expansion such a change
     is bounded by ``jitter * |gradient| + (jitter^2 / 2 + blur_tol) * |curvature|``
     of the expected image (curvature matters on thin bars and fine text),
     taking the strongest channel's gradient and curvature because JPEG colour
     subsampling spreads an edge into every channel. ``jitter`` is, per colour channel, the larger of the configured tolerance
     and the edge jitter *measured* in each block (robust median of |residual|
     / gradient over edge pixels, capped). Edges are tolerated by exactly as
     much as they wander in this print, while the interiors of missing strokes
     and specks on bare stock, which have little gradient or curvature, are not;
   - structural similarity (SSIM) map;
   - local contrast change (ratio of local standard deviations), coarse scales;
   - edge maps (full resolution): artwork edges missing in the photo and photo
     edges absent from the artwork, each with a small positional tolerance.
   A pixel counts only when its normalised absolute difference is backed by
   structural evidence - SSIM or edges at full resolution; SSIM or local
   contrast at coarse scales - and exceeds a minimum absolute difference.
   Pixel subtraction alone never decides.

   Self-check: the mean SSIM between print and expected image at the coarsest
   scale (where halftone texture has averaged out) must be high. If the print
   does not resemble the artwork, the noise model would absorb the differences,
   so this is reported (critical below the configured level -> REVIEW).

4. Candidates: hysteresis (seed above the high threshold, extent down to the
   low threshold) over the multi-scale score, a small morphological closing to
   join fragments, then connected components. Each component is measured
   (area, scores, differences, polarity, SSIM, edges, contrast, shape) and
   given a *preliminary* kind. Deciding PASS/FAIL is not part of this step.

Large grids are processed in overlapping tiles; the overlap exceeds every
filter footprint, so tiling does not change the result.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from print_inspector.config import ComparisonConfig
from print_inspector.core.models import (
    InspectionStatus,
    QualityIssue,
    QualityLevel,
    Severity,
    level_from_issues,
    sort_issues,
)

log = logging.getLogger(__name__)

EDGE_MISSING = 1  # codes in ComparisonResult.edge_difference
EDGE_EXTRA = 2
_GRAY_WEIGHTS = np.array([0.114, 0.587, 0.299], np.float32)  # BGR

KIND_LABELS: dict[str, str] = {
    "missing_line": "Missing / broken line",
    "missing_element": "Missing graphic element",
    "missing_region": "Missing printed region",
    "broken_character": "Broken character / small missing ink",
    "extra_ink_mark": "Extra ink mark / unexpected shape",
    "small_dot": "Small unwanted dot",
    "shape_deviation": "Shape deviation (e.g. distorted logo)",
    "light_spot": "Lighter area on bare stock",
}


@dataclass(frozen=True)
class DefectCandidate:
    """A connected region where the print differs from the artwork (grid pixel units)."""

    id: int
    kind: str  # preliminary, see KIND_LABELS
    x: int
    y: int
    width: int
    height: int
    centroid: tuple[float, float]
    area_px: int
    score: float  # peak score relative to the low threshold (1.0 = detection limit)
    peak_sigma: float  # peak difference in noise units at the scale that detected it
    max_difference: float  # largest colour-channel difference, gray levels
    mean_difference: float  # mean (photo - expected) gray level; negative = darker than expected
    polarity: str  # "darker", "lighter" or "mixed"
    min_ssim: float
    missing_edge_px: int
    extra_edge_px: int
    contrast_ratio: float  # local contrast photo / expected
    scale: int  # 1 = full resolution, 2, 4, ...
    ink_fraction: float  # share of the region where the artwork has ink
    elongation: float  # length / width of the region

    @property
    def label(self) -> str:
        return KIND_LABELS.get(self.kind, self.kind)


@dataclass(frozen=True)
class ComparisonMetrics:
    grid_size: tuple[int, int]
    sensitivity: str
    threshold_factor: float
    noise_sigma: tuple[float, float, float]  # global, per BGR channel, on bare stock
    texture_sigma_max: float  # highest local noise/texture level found
    alignment_tolerance_px: float
    edge_jitter_max_px: float  # largest measured local edge jitter (capped)
    blur_sigma_px: float
    lighting_gain_range: tuple[float, float]
    scales: tuple[int, ...]
    tiles: int
    candidate_count: int
    candidates_dropped: int
    candidate_area_fraction: float
    compared_fraction: float
    model_similarity: float  # mean SSIM of print vs. expected at the coarsest scale
    seconds: float


@dataclass(frozen=True, eq=False)
class ComparisonResult:
    metrics: ComparisonMetrics
    issues: tuple[QualityIssue, ...]
    candidates: tuple[DefectCandidate, ...]
    expected: np.ndarray  # BGR uint8: the artwork as it should appear in this photo
    gray_difference: np.ndarray  # uint8 |photo - expected| (gray levels)
    structural_difference: np.ndarray  # uint8 (1 - SSIM) * 255
    edge_difference: np.ndarray  # uint8: 0, EDGE_MISSING or EDGE_EXTRA
    combined_score: np.ndarray  # float32, 1.0 = detection threshold, 0 outside the mask
    candidate_mask: np.ndarray  # uint8 255 inside candidates

    @property
    def level(self) -> QualityLevel:
        return level_from_issues(self.issues)

    @property
    def is_reliable(self) -> bool:
        return self.level is not QualityLevel.POOR

    @property
    def recommended_status(self) -> InspectionStatus | None:
        """REVIEW when the comparison itself is not trustworthy; never a PASS/FAIL verdict."""
        return None if self.is_reliable else InspectionStatus.REVIEW

    def candidates_of(self, kind: str) -> list[DefectCandidate]:
        return [c for c in self.candidates if c.kind == kind]


# ---------------------------------------------------------------- photometric model


def _poly_terms(u: np.ndarray, v: np.ndarray, degree: int) -> np.ndarray:
    return np.column_stack([u**i * v**j for i in range(degree + 1) for j in range(degree + 1 - i)])


def _norm_coords(xs: np.ndarray, ys: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    w, h = size
    return (xs - (w - 1) / 2.0) / (w / 2.0), (ys - (h - 1) / 2.0) / (h / 2.0)


_TONE_SPIKE_GRAY = 20.0  # a level that would lift the next levels by more than this is an outlier
_TONE_SPIKE_REACH = 8  # ... judged against this many following levels


def _drop_tone_spikes(levels: np.ndarray, medians: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove levels whose median lies far above the following levels.

    Such a level is dominated by a defect (e.g. bars missing in the print, so that
    edge pixels of the artwork are paper in the photo); the running maximum would
    otherwise carry it to every lighter level. Ordinary noise stays well below the limit.
    """
    n = len(medians)
    keep = np.ones(n, bool)
    for i in range(n - 1):
        following = medians[i + 1 : i + 1 + _TONE_SPIKE_REACH]
        if medians[i] - float(np.median(following)) > _TONE_SPIKE_GRAY:
            keep[i] = False
    return levels[keep], medians[keep]


def _fit_tone_curve(model: np.ndarray, target: np.ndarray, min_count: int = 20) -> np.ndarray:
    """Monotonic 256-entry LUT: median target value for each model value (interpolated)."""
    order = np.lexsort((target, model))
    xs, ys = model[order], target[order]
    starts = np.searchsorted(xs, np.arange(256), side="left")
    ends = np.searchsorted(xs, np.arange(256), side="right")
    counts = ends - starts
    known = np.flatnonzero(counts >= min_count)
    if known.size < 2:
        return np.arange(256, dtype=np.float32)
    medians = ys[starts[known] + counts[known] // 2].astype(np.float64)
    known, medians = _drop_tone_spikes(known, medians)
    lut = np.interp(np.arange(256), known, medians)
    return np.maximum.accumulate(lut).astype(np.float32)


def _fit_gain(
    xs: np.ndarray, ys: np.ndarray, ratio: np.ndarray, size: tuple[int, int], degree: int
) -> np.ndarray:
    """Robust polynomial fit of the lighting gain (IRLS with hard outlier rejection)."""
    u, v = _norm_coords(xs, ys, size)
    a = _poly_terms(u, v, degree)
    keep = np.ones(ratio.size, bool)
    coeffs = np.zeros(a.shape[1])
    coeffs[0] = 1.0
    for _ in range(4):
        if keep.sum() < a.shape[1] * 5:
            break
        coeffs, *_ = np.linalg.lstsq(a[keep], ratio[keep], rcond=None)
        residual = ratio - a @ coeffs
        mad = 1.4826 * float(np.median(np.abs(residual[keep]))) + 1e-4
        keep = np.abs(residual) < 3.0 * mad
    return coeffs


def _gain_tile(coeffs: np.ndarray, degree: int, size: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = box
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    u, v = _norm_coords(xx.ravel(), yy.ravel(), size)
    return (_poly_terms(u, v, degree) @ coeffs).reshape(y1 - y0, x1 - x0).astype(np.float32)


# ---------------------------------------------------------------- noise model


def _block_quantiles(values: np.ndarray, block: int, quantiles: tuple[float, ...]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Per-block valid-pixel counts and quantiles of an image whose invalid pixels are NaN.

    Sorting puts NaNs last, so the quantiles are read at positions within the valid
    count - thread-safe and free of all-NaN warnings (unlike ``np.nanmedian``).
    Blocks without valid pixels get NaN.
    """
    h, w = values.shape
    nby, nbx = -(-h // block), -(-w // block)
    padded = np.full((nby * block, nbx * block), np.nan, np.float32)
    padded[:h, :w] = values
    blocks = np.sort(padded.reshape(nby, block, nbx, block).transpose(0, 2, 1, 3).reshape(nby, nbx, -1), axis=2)
    counts = np.sum(~np.isnan(blocks), axis=2)
    out = []
    for q in quantiles:
        index = np.clip(np.floor(q * (counts - 1)).astype(np.int64), 0, blocks.shape[2] - 1)
        found = np.take_along_axis(blocks, index[:, :, None], axis=2)[:, :, 0]
        found[counts == 0] = np.nan
        out.append(found)
    return counts, out


def _block_sigma(residual: np.ndarray, valid: np.ndarray, block: int, floor: float) -> tuple[np.ndarray, float]:
    """Robust noise level per block, median-filtered across blocks; plus the global level.

    The larger of MAD x 1.4826 and the 90th percentile of |residual| / 1.645 is
    used (equal for Gaussian noise). JPEG compression and noise reduction make
    phone-photo residuals heavy-tailed, and the MAD alone then underestimates them.
    """
    counts, (median, p90) = _block_quantiles(np.where(valid, np.abs(residual), np.nan), block, (0.5, 0.9))
    sigma = np.maximum(1.4826 * median, p90 / 1.645)
    nby, nbx = sigma.shape
    values = np.abs(residual[valid])
    global_sigma = (
        max(floor, 1.4826 * float(np.median(values)), float(np.percentile(values, 90)) / 1.645)
        if values.size else floor
    )
    sigma[(counts < 0.2 * block * block) | np.isnan(sigma)] = global_sigma
    sigma = cv2.medianBlur(sigma.astype(np.float32), 3) if min(nby, nbx) >= 3 else sigma.astype(np.float32)
    return np.maximum(sigma, floor), global_sigma


def _curvature(channel: np.ndarray) -> np.ndarray:
    """Frobenius norm of the Hessian (gray levels / px^2)."""
    ixx = cv2.Sobel(channel, cv2.CV_32F, 2, 0, ksize=3, scale=0.25)
    iyy = cv2.Sobel(channel, cv2.CV_32F, 0, 2, ksize=3, scale=0.25)
    ixy = cv2.Sobel(channel, cv2.CV_32F, 1, 1, ksize=3, scale=0.25)
    return np.sqrt(ixx * ixx + iyy * iyy + 2.0 * ixy * ixy)


def _block_jitter(
    residual: np.ndarray, gradient: np.ndarray, valid: np.ndarray, block: int, base: float, cap: float
) -> np.ndarray:
    """Edge jitter per block in pixels: median |residual| / gradient over clear edge pixels."""
    edges = valid & (gradient >= 8.0)
    ratio = np.where(edges, np.abs(residual) / np.maximum(gradient, 1e-3), np.nan).astype(np.float32)
    counts, (jitter,) = _block_quantiles(ratio, block, (0.5,))
    nby, nbx = jitter.shape
    jitter[(counts < 10) | np.isnan(jitter)] = base
    jitter = cv2.medianBlur(jitter.astype(np.float32), 3) if min(nby, nbx) >= 3 else jitter.astype(np.float32)
    return np.clip(jitter, base, cap)


def _sigma_tile(blocks: np.ndarray, block: int, box: tuple[int, int, int, int]) -> np.ndarray:
    """Bilinear interpolation of the block noise grid at the pixels of ``box``."""
    y0, y1, x0, x1 = box
    xs = ((np.arange(x0, x1, dtype=np.float32) + 0.5) / block - 0.5)
    ys = ((np.arange(y0, y1, dtype=np.float32) + 0.5) / block - 0.5)
    map_x, map_y = np.meshgrid(xs, ys)
    return cv2.remap(blocks, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- tiling


def _tiles(h: int, w: int, size: int, margin: int):
    for y0 in range(0, h, size):
        for x0 in range(0, w, size):
            y1, x1 = min(h, y0 + size), min(w, x0 + size)
            outer = (max(0, y0 - margin), min(h, y1 + margin), max(0, x0 - margin), min(w, x1 + margin))
            yield (y0, y1, x0, x1), outer


# ---------------------------------------------------------------- per-scale analysis


@dataclass
class _ScaleOutput:
    z: np.ndarray  # confirmed, normalised difference (sigma units); 0 where unconfirmed
    gray_difference: np.ndarray | None = None
    structural: np.ndarray | None = None
    edges: np.ndarray | None = None
    tiles: int = 0
    ssim_sum: float = 0.0
    ssim_count: int = 0


def _analyse(
    photo: np.ndarray,
    expected: np.ndarray,
    mask: np.ndarray,
    sigma_blocks: list[tuple[np.ndarray, np.ndarray]],
    tint: np.ndarray,
    jitter_blocks: list[np.ndarray],
    cfg: ComparisonConfig,
    min_abs: float,
    full_resolution: bool,
) -> _ScaleOutput:
    h, w = mask.shape
    out = _ScaleOutput(z=np.zeros((h, w), np.float32))
    if full_resolution:
        out.gray_difference = np.zeros((h, w), np.uint8)
        out.structural = np.zeros((h, w), np.uint8)
        out.edges = np.zeros((h, w), np.uint8)
    block = cfg.noise_block_px
    tol = cfg.edge_tolerance_px
    tol_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    confirm_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 3, 2 * tol + 3))
    log_ratio = math.log(cfg.contrast_confirm_ratio)
    win = cfg.contrast_window_px
    for (y0, y1, x0, x1), (py0, py1, px0, px1) in _tiles(h, w, cfg.tile_size_px, cfg.tile_margin_px):
        out.tiles += 1
        tile_mask = mask[py0:py1, px0:px1] > 0
        if not tile_mask.any():
            continue
        ph = photo[py0:py1, px0:px1].astype(np.float32)
        ex = expected[py0:py1, px0:px1].astype(np.float32)
        absr = np.abs(ph - ex)
        is_tint = tint[py0:py1, px0:px1]
        sig = np.dstack([
            np.where(
                is_tint,
                _sigma_tile(tint_blocks, block, (py0, py1, px0, px1)),
                _sigma_tile(solid_blocks, block, (py0, py1, px0, px1)),
            )
            for solid_blocks, tint_blocks in sigma_blocks
        ])
        grad = np.dstack([
            cv2.magnitude(
                cv2.Sobel(ex[:, :, c], cv2.CV_32F, 1, 0, ksize=3, scale=0.125),
                cv2.Sobel(ex[:, :, c], cv2.CV_32F, 0, 1, ksize=3, scale=0.125),
            )
            for c in range(3)
        ])
        # JPEG stores colour at half resolution: at a colour edge every channel can be disturbed
        # in proportion to the strongest channel's edge, so the allowance uses the maximum.
        grad = cv2.dilate(grad.max(axis=2), np.ones((3, 3), np.uint8))[:, :, None]
        curvature = cv2.dilate(
            np.max(np.dstack([_curvature(ex[:, :, c]) for c in range(3)]), axis=2), np.ones((3, 3), np.uint8)
        )[:, :, None]
        jitter = np.dstack([_sigma_tile(j, block, (py0, py1, px0, px1)) for j in jitter_blocks])
        allowance = jitter * grad + (0.5 * jitter * jitter + cfg.blur_tolerance_px2) * curvature
        z = np.max((absr - allowance) / sig, axis=2)
        abs_max = np.max(absr, axis=2)

        g_ph = ph @ _GRAY_WEIGHTS
        g_ex = ex @ _GRAY_WEIGHTS
        _, ssim_map = structural_similarity(
            g_ex, g_ph, data_range=255.0, gaussian_weights=True, sigma=cfg.ssim_sigma,
            use_sample_covariance=False, full=True,
        )
        structural = np.clip(1.0 - ssim_map, 0.0, 1.0).astype(np.float32)
        core_mask = tile_mask[y0 - py0 : y1 - py0, x0 - px0 : x1 - px0]
        out.ssim_sum += float(ssim_map[y0 - py0 : y1 - py0, x0 - px0 : x1 - px0][core_mask].sum())
        out.ssim_count += int(core_mask.sum())

        def local_std(img: np.ndarray) -> np.ndarray:
            mean = cv2.boxFilter(img, cv2.CV_32F, (win, win))
            sq = cv2.boxFilter(img * img, cv2.CV_32F, (win, win))
            return np.sqrt(np.maximum(sq - mean * mean, 0.0))

        confirm = structural >= cfg.ssim_confirm
        if not full_resolution:
            # Local contrast confirms faint, larger changes at the coarse scales. At full
            # resolution paper and print texture alone can change it, so there only
            # structural evidence (SSIM, edges) confirms.
            c0 = 2.0 * sig.mean(axis=2)
            confirm |= np.abs(np.log((local_std(g_ph) + c0) / (local_std(g_ex) + c0))) >= log_ratio

        if full_resolution:
            e_ex = cv2.Canny(np.clip(g_ex, 0, 255).astype(np.uint8), cfg.edge_low_threshold, cfg.edge_high_threshold) > 0
            e_ph = cv2.Canny(np.clip(g_ph, 0, 255).astype(np.uint8), cfg.edge_low_threshold, cfg.edge_high_threshold) > 0
            missing = e_ex & ~(cv2.dilate(e_ph.astype(np.uint8), tol_kernel) > 0)
            extra = e_ph & ~(cv2.dilate(e_ex.astype(np.uint8), tol_kernel) > 0)
            confirm |= cv2.dilate((missing | extra).astype(np.uint8), confirm_kernel) > 0
            codes = np.zeros(missing.shape, np.uint8)
            codes[missing] = EDGE_MISSING
            codes[extra] = EDGE_EXTRA

        accepted = confirm & (abs_max >= min_abs) & tile_mask
        zc = np.where(accepted, np.maximum(z, 0.0), 0.0).astype(np.float32)
        cy0, cy1, cx0, cx1 = y0 - py0, y1 - py0, x0 - px0, x1 - px0
        out.z[y0:y1, x0:x1] = zc[cy0:cy1, cx0:cx1]
        if full_resolution:
            gd = np.abs(g_ph - g_ex)
            gd[~tile_mask] = 0
            out.gray_difference[y0:y1, x0:x1] = np.clip(gd[cy0:cy1, cx0:cx1], 0, 255).astype(np.uint8)
            structural[~tile_mask] = 0
            out.structural[y0:y1, x0:x1] = (structural[cy0:cy1, cx0:cx1] * 255).astype(np.uint8)
            codes[~tile_mask] = 0
            out.edges[y0:y1, x0:x1] = codes[cy0:cy1, cx0:cx1]
    return out


# ---------------------------------------------------------------- candidate description


def _classify(polarity: str, ink_fraction: float, area: int, elongation: float, small_area: float) -> str:
    """Preliminary kind from simple measurements (refined classification comes later)."""
    if polarity == "lighter":
        if ink_fraction >= 0.4:
            if elongation >= 4.0:
                return "missing_line"
            if area <= small_area:
                return "broken_character"
            return "missing_region" if elongation < 2.5 else "missing_element"
        return "light_spot"
    if polarity == "darker":
        if ink_fraction < 0.4:
            return "small_dot" if area <= small_area else "extra_ink_mark"
        return "shape_deviation"
    return "shape_deviation"


def _describe(
    index: int,
    stats: np.ndarray,
    labels: np.ndarray,
    score: np.ndarray,
    scale_map: np.ndarray,
    scale_low: dict[int, float],
    photo: np.ndarray,
    expected: np.ndarray,
    structural: np.ndarray,
    edges: np.ndarray,
    paper_level: float,
    small_area: float,
    min_abs: float,
) -> DefectCandidate:
    x, y, w, h, area = (int(v) for v in stats[index])
    comp = labels[y : y + h, x : x + w] == index
    sc = score[y : y + h, x : x + w]
    peak = float(sc[comp].max())
    py, px = np.unravel_index(np.argmax(np.where(comp, sc, -1.0)), sc.shape)
    scale = int(scale_map[y + py, x + px])
    ph = photo[y : y + h, x : x + w].astype(np.float32)
    ex = expected[y : y + h, x : x + w].astype(np.float32)
    diff = (ph @ _GRAY_WEIGHTS) - (ex @ _GRAY_WEIGHTS)
    d = diff[comp]
    # The core (at least half the peak difference) describes the defect itself;
    # the rest of the region is its soft halo.
    core = comp & (np.abs(diff) >= 0.5 * float(np.abs(d).max()))
    dc = diff[core]
    darker = float(np.mean(dc < -min_abs / 2))
    lighter = float(np.mean(dc > min_abs / 2))
    if darker >= 0.25 and lighter >= 0.25:
        polarity = "mixed"
    else:
        polarity = "darker" if float(dc.mean()) < 0 else "lighter"
    # Ink in the artwork under the core: for missing ink look at the artwork around it too.
    artwork_ink = (ex @ _GRAY_WEIGHTS) < paper_level - 40.0
    ink = float(np.mean(artwork_ink[core]))
    near = cv2.dilate(comp.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    e = edges[y : y + h, x : x + w][near]
    std_ph = float((ph @ _GRAY_WEIGHTS)[near].std())
    std_ex = float((ex @ _GRAY_WEIGHTS)[near].std())
    moments = cv2.moments(core.astype(np.uint8), binaryImage=True)
    if moments["m00"] > 0:
        cov = np.array([[moments["mu20"], moments["mu11"]], [moments["mu11"], moments["mu02"]]]) / moments["m00"]
        lam = np.linalg.eigvalsh(cov)
        elongation = float(math.sqrt(max(lam[1], 1e-6) / max(lam[0], 0.25 / 12)))
        cx, cy = moments["m10"] / moments["m00"] + x, moments["m01"] / moments["m00"] + y
    else:
        elongation, cx, cy = 1.0, x + w / 2.0, y + h / 2.0
    return DefectCandidate(
        id=0,
        kind=_classify(polarity, ink, int(np.count_nonzero(core)), elongation, small_area),
        x=x, y=y, width=w, height=h,
        centroid=(float(cx), float(cy)),
        area_px=area,
        score=peak,
        peak_sigma=peak * scale_low[scale],
        max_difference=float(np.max(np.abs(ph - ex), axis=2)[comp].max()),
        mean_difference=float(d.mean()),
        polarity=polarity,
        min_ssim=1.0 - float(structural[y : y + h, x : x + w][comp].max()) / 255.0,
        missing_edge_px=int(np.count_nonzero(e == EDGE_MISSING)),
        extra_edge_px=int(np.count_nonzero(e == EDGE_EXTRA)),
        contrast_ratio=(std_ph + 1.0) / (std_ex + 1.0),
        scale=scale,
        ink_fraction=ink,
        elongation=elongation,
    )


# ---------------------------------------------------------------- main entry point


def compare_to_artwork(
    registered: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    cfg: ComparisonConfig,
    blur_sigma_px: float = 0.0,
    alignment_residual_px: float | None = None,
    sensitivity: str = "medium",
) -> ComparisonResult:
    """Compare the registered photo with the artwork (same grid, BGR uint8) inside ``mask``."""
    started = time.perf_counter()
    for name, image in (("registered", registered), ("reference", reference)):
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"Expected HxWx3 uint8 BGR {name}, got {image.shape} {image.dtype}")
    if registered.shape != reference.shape or mask.shape != registered.shape[:2]:
        raise ValueError("Registered photo, reference and mask must have the same size")
    factor = {"low": cfg.sensitivity_low_factor, "medium": 1.0, "high": cfg.sensitivity_high_factor}[sensitivity]
    h, w = mask.shape
    size = (w, h)
    valid = mask > 0
    tolerance = max(cfg.alignment_tolerance_px, alignment_residual_px or 0.0)
    min_abs = cfg.min_abs_difference * factor

    # ---- expected print: blur, per-channel tone curves, lighting gain
    model = cv2.GaussianBlur(reference, (0, 0), blur_sigma_px) if blur_sigma_px > 0 else reference.copy()
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(valid.ravel())
    if idx.size == 0:
        raise ValueError("The comparison mask is empty")
    if idx.size > 400_000:
        idx = rng.choice(idx, 400_000, replace=False)
    ys, xs = np.divmod(idx, w)
    model_s = model.reshape(-1, 3)[idx]
    photo_s = registered.reshape(-1, 3)[idx].astype(np.float32)
    luts = [_fit_tone_curve(model_s[:, c], photo_s[:, c]) for c in range(3)]
    gain_coeffs = np.zeros((cfg.lighting_poly_degree + 1) * (cfg.lighting_poly_degree + 2) // 2)
    gain_coeffs[0] = 1.0
    for _ in range(2):  # alternate tone curves and lighting gain
        predicted = np.column_stack([luts[c][model_s[:, c]] for c in range(3)]) @ _GRAY_WEIGHTS
        observed = photo_s @ _GRAY_WEIGHTS
        paper_level = float(np.percentile(predicted, 95))
        paper = (predicted >= 0.85 * paper_level) & (predicted > 30)
        if paper.sum() >= 200:
            gain_coeffs = _fit_gain(
                xs[paper].astype(np.float64), ys[paper].astype(np.float64),
                observed[paper] / np.maximum(predicted[paper], 1.0), size, cfg.lighting_poly_degree,
            )
        u, v = _norm_coords(xs.astype(np.float64), ys.astype(np.float64), size)
        gain_s = np.maximum(_poly_terms(u, v, cfg.lighting_poly_degree) @ gain_coeffs, 0.05)
        luts = [_fit_tone_curve(model_s[:, c], np.clip(photo_s[:, c] / gain_s, 0, 255)) for c in range(3)]
    expected = np.empty_like(reference)
    gains = []
    for (y0, y1, x0, x1), _ in _tiles(h, w, cfg.tile_size_px, 0):
        gain = _gain_tile(gain_coeffs, cfg.lighting_poly_degree, size, (y0, y1, x0, x1))
        gains.append((float(gain.min()), float(gain.max())))
        tile = np.dstack([luts[c][model[y0:y1, x0:x1, c]] for c in range(3)]) * gain[:, :, None]
        expected[y0:y1, x0:x1] = np.clip(tile + 0.5, 0, 255).astype(np.uint8)
    expected.setflags(write=False)
    paper_level = float(np.percentile((expected.reshape(-1, 3)[idx].astype(np.float32) @ _GRAY_WEIGHTS), 95))

    # ---- multi-scale analysis
    scales = [2**level for level in range(cfg.coarse_levels + 1)]
    score = np.zeros((h, w), np.float32)
    seed = np.zeros((h, w), bool)
    scale_map = np.ones((h, w), np.uint8)
    scale_low: dict[int, float] = {}
    fine: _ScaleOutput | None = None
    noise_sigma = (0.0, 0.0, 0.0)
    texture_max = 0.0
    jitter_max = tolerance
    model_similarity = 1.0
    tiles = 0
    for s in scales:
        if s == 1:
            ph_s, ex_s, mk_s = registered, expected, mask
        else:
            small = (max(8, round(w / s)), max(8, round(h / s)))
            ph_s = cv2.resize(registered, small, interpolation=cv2.INTER_AREA)
            ex_s = cv2.resize(expected, small, interpolation=cv2.INTER_AREA)
            mk_s = ((cv2.resize(mask, small, interpolation=cv2.INTER_AREA) >= 250) * 255).astype(np.uint8)
        hs, ws = mk_s.shape
        # Noise per block, ignoring artwork edges (misalignment there is handled by the allowance).
        ex_gray = cv2.cvtColor(ex_s, cv2.COLOR_BGR2GRAY)
        edge_px = cv2.dilate(
            (cv2.magnitude(cv2.Sobel(ex_gray, cv2.CV_32F, 1, 0, scale=0.125), cv2.Sobel(ex_gray, cv2.CV_32F, 0, 1, scale=0.125)) > 3.0).astype(np.uint8),
            np.ones((5, 5), np.uint8),
        ) > 0
        quiet = (mk_s > 0) & ~edge_px
        ref_s = reference if s == 1 else cv2.resize(reference, (ws, hs), interpolation=cv2.INTER_AREA)
        # Screen dots spill about a dot width beyond tints (e.g. into knock-out text).
        tint = cv2.dilate(np.any((ref_s > 12) & (ref_s < 243), axis=2).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        blocks, globals_ = [], []
        for c in range(3):
            residual = ph_s[:, :, c].astype(np.float32) - ex_s[:, :, c].astype(np.float32)
            solid, g = _block_sigma(residual, quiet & ~tint, cfg.noise_block_px, cfg.noise_floor)
            textured, _ = _block_sigma(residual, quiet & tint, cfg.noise_block_px, cfg.noise_floor)
            blocks.append((solid, textured))
            globals_.append(g)
        jitter = []
        for c in range(3):
            ex_c = ex_s[:, :, c].astype(np.float32)
            grad_c = cv2.magnitude(
                cv2.Sobel(ex_c, cv2.CV_32F, 1, 0, scale=0.125), cv2.Sobel(ex_c, cv2.CV_32F, 0, 1, scale=0.125)
            )
            jitter.append(_block_jitter(
                ph_s[:, :, c].astype(np.float32) - ex_c, grad_c, mk_s > 0,
                cfg.noise_block_px, tolerance / s, cfg.max_edge_jitter_px / s,
            ))
        if s == 1:
            noise_sigma = tuple(float(g) for g in globals_)
            texture_max = float(max(max(a.max(), b.max()) for a, b in blocks))
            jitter_max = float(max(j.max() for j in jitter))
        high = (cfg.fine_high_sigma if s == 1 else cfg.coarse_high_sigma) * factor
        low = (cfg.fine_low_sigma if s == 1 else cfg.coarse_low_sigma) * factor
        scale_low[s] = low
        result = _analyse(ph_s, ex_s, mk_s, blocks, tint, jitter, cfg, min_abs, s == 1)
        tiles += result.tiles
        if result.ssim_count:
            model_similarity = result.ssim_sum / result.ssim_count  # the coarsest scale wins
        z = result.z if s == 1 else cv2.resize(result.z, (w, h), interpolation=cv2.INTER_LINEAR)
        relative = z / low
        better = relative > score
        score[better] = relative[better]
        scale_map[better] = s
        seed |= z >= high
        if s == 1:
            fine = result
        del result, z, relative, better
    score[~valid] = 0.0
    seed &= valid

    # ---- grouping into candidates
    low_mask = (score >= 1.0).astype(np.uint8)
    gap = cfg.merge_gap_px
    joined = cv2.morphologyEx(low_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1)))
    joined &= valid.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    seeded = np.zeros(count, bool)
    seeded[np.unique(labels[seed & (joined > 0)])] = True
    seeded[0] = False
    peaks = np.zeros(count, np.float32)
    np.maximum.at(peaks, labels.ravel(), score.ravel())
    keep = [i for i in range(1, count) if seeded[i] and stats[i, cv2.CC_STAT_AREA] >= cfg.min_component_area_px]
    keep.sort(key=lambda i: -float(peaks[i]))
    dropped = max(0, len(keep) - cfg.max_candidates)
    keep = keep[: cfg.max_candidates]
    small_area = (0.006 * max(w, h)) ** 2
    described = [
        _describe(
            i, stats, labels, score, scale_map, scale_low, registered, expected, fine.structural, fine.edges,
            paper_level, small_area, min_abs,
        )
        for i in keep
    ]
    candidates = [replace(c, id=n) for n, c in enumerate(described, start=1)]
    candidate_mask = (np.isin(labels, keep) * 255).astype(np.uint8)

    compared = int(valid.sum())
    area_fraction = float(np.count_nonzero(candidate_mask)) / max(compared, 1)
    issues: list[QualityIssue] = []
    if area_fraction >= cfg.candidate_area_warning_fraction:
        severity = Severity.CRITICAL if area_fraction >= cfg.candidate_area_critical_fraction else Severity.WARNING
        issues.append(QualityIssue(
            "extensive_differences", severity,
            f"Differences cover {area_fraction:.0%} of the label. Check that the right artwork is "
            "loaded and that registration is correct before trusting individual candidates.",
        ))
    if model_similarity < cfg.min_model_similarity_warning:
        severity = Severity.CRITICAL if model_similarity < cfg.min_model_similarity_critical else Severity.WARNING
        issues.append(QualityIssue(
            "model_mismatch", severity,
            f"The print resembles the artwork only weakly (similarity {model_similarity:.2f}). The noise "
            "model then becomes unreliable and real defects can be missed. Check the artwork and registration.",
        ))
    if dropped:
        issues.append(QualityIssue(
            "too_many_candidates", Severity.WARNING,
            f"{len(keep) + dropped} difference regions were found; only the {len(keep)} strongest are listed.",
        ))

    for array in (fine.gray_difference, fine.structural, fine.edges, score, candidate_mask):
        array.setflags(write=False)
    metrics = ComparisonMetrics(
        grid_size=size,
        sensitivity=sensitivity,
        threshold_factor=factor,
        noise_sigma=noise_sigma,
        texture_sigma_max=texture_max,
        alignment_tolerance_px=tolerance,
        edge_jitter_max_px=jitter_max,
        blur_sigma_px=blur_sigma_px,
        lighting_gain_range=(min(g[0] for g in gains), max(g[1] for g in gains)),
        scales=tuple(scales),
        tiles=tiles,
        candidate_count=len(candidates),
        candidates_dropped=dropped,
        candidate_area_fraction=area_fraction,
        compared_fraction=compared / mask.size,
        model_similarity=model_similarity,
        seconds=time.perf_counter() - started,
    )
    return ComparisonResult(
        metrics=metrics,
        issues=sort_issues(issues),
        candidates=tuple(candidates),
        expected=expected,
        gray_difference=fine.gray_difference,
        structural_difference=fine.structural,
        edge_difference=fine.edges,
        combined_score=score,
        candidate_mask=candidate_mask,
    )
