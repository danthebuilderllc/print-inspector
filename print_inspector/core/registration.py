"""Precise registration of the photographed label to the original artwork.

Comparison grid
---------------
Comparisons happen on a grid that covers the artwork exactly, scaled to the
photo's native resolution (``grid_scale`` grid pixels per artwork pixel, taken
from the perspective-corrected label). The photo is therefore not upsampled
beyond what it resolves, and tiny text and hairlines keep the detail the photo
has. ``reference`` is the artwork resampled once to this grid.

Estimation (coarse to fine; every stage samples the *original* photo afresh,
so estimation never works on repeatedly resampled images)
1. Start: the Step 3 perspective correction, scaled onto the grid.
2. Global correction by feature matching: SIFT keypoints, Lowe ratio test and
   a MAGSAC++ homography. Used only if the inliers are spread over the label:
   clustered matches can have a tiny reprojection error yet extrapolate tens
   of pixels wrong elsewhere (seen on low-feature artwork).
3. ECC refinement (``cv2.findTransformECC``) on a 3-level pyramid. ECC uses all
   pixels, so it also works on artwork with large flat areas. A full
   homography is estimated only when 2D detail covers the label; otherwise an
   affine model, because flat areas leave perspective terms unconstrained.
4. Photometric matching: the photo is softer than digital artwork, and the
   camera's tone curve applied to a soft edge shifts where the edge appears.
   The artwork is blurred to the photo's estimated sharpness and mapped
   through the measured tone curve before any local measurement; without
   this, local shifts are biased by up to 2 px (measured on synthetic data).
   ECC compares against this model and ignores pixels that differ far beyond
   what misalignment could explain: otherwise large print defects (a missing
   element, a misprinted logo) pull the whole registration by 1-2 px.
5. Local correction (only when justified): a grid of tiles is measured with
   phase correlation followed by Lucas-Kanade (sub-pixel), and a smooth
   polynomial displacement field is fitted. It is applied only if it predicts
   held-out tiles better than no correction (checkerboard cross-validation).
   This corrects residual lens distortion and slight non-flatness, which a
   homography cannot.

Output: one ``cv2.remap`` from the original photo pixels through the full
model (field, then homography), computed in strips to bound memory.

Tiles
-----
A tile is measured only where the noise-free artwork has structure. Blank
tiles give confident but meaningless shifts, so they are skipped. Tiles with
structure in one direction only (straight edges, bars, hairlines) are measured
along their edge normal - the direction in which misalignment produces false
defects - because the shift along the lines is undetermined.

Verification (independent of the estimation)
--------------------------------------------
Residual misalignment is measured on a *different* tile grid (offset by half a
tile, plus tiles placed along the comparison-area borders) after the final
resampling. Reported:
median / 95th percentile / max residual in grid pixels, and coverage (share
of detailed tiles that could be verified). High feature-match confidence is
never taken as proof of local accuracy.

Comparison mask: grid pixels whose source lies inside the photo and inside the
detected label outline, excluding a border along both the label outline and
the artwork edges (those pixels mix label and surroundings).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from print_inspector.config import RegistrationConfig
from print_inspector.core.alignment import AlignmentResult
from print_inspector.core.models import (
    InspectionStatus,
    QualityIssue,
    QualityLevel,
    Severity,
    level_from_issues,
    sort_issues,
)

log = logging.getLogger(__name__)

_INTERPOLATION = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}
_KERNEL_MARGIN = {"linear": 1, "cubic": 2, "lanczos": 4}
_STRIP_ROWS = 256


class RegistrationMethod(str, Enum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    MANUAL_REFINED = "manual_refined"

    @property
    def label(self) -> str:
        return {
            RegistrationMethod.AUTOMATIC: "Automatic",
            RegistrationMethod.MANUAL: "Manual adjustment",
            RegistrationMethod.MANUAL_REFINED: "Manual adjustment, refined automatically",
        }[self]


@dataclass(frozen=True)
class ManualAdjustment:
    """User correction applied on top of a registration, in grid pixels / degrees / percent.

    Positive ``dx``/``dy`` move the photo right/down relative to the artwork;
    rotation (clockwise, degrees) and scale act about the grid centre.
    """

    dx: float = 0.0
    dy: float = 0.0
    rotation_deg: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0

    def is_identity(self) -> bool:
        return self == ManualAdjustment()

    def sampling_matrix(self, grid_size: tuple[int, int]) -> np.ndarray:
        """Grid -> grid matrix telling where to sample the current registration."""
        gw, gh = grid_size
        cx, cy = (gw - 1) / 2.0, (gh - 1) / 2.0
        a = math.radians(self.rotation_deg)
        move = (
            np.array([[1, 0, cx + self.dx], [0, 1, cy + self.dy], [0, 0, 1.0]])
            @ np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
            @ np.diag([self.scale_x, self.scale_y, 1.0])
            @ np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
        )
        return np.linalg.inv(move)


@dataclass(frozen=True)
class RegistrationMetrics:
    method: RegistrationMethod | None = None
    grid_size: tuple[int, int] | None = None
    grid_scale: float | None = None  # grid pixels per artwork pixel
    feature_matches: int | None = None
    feature_inliers: int | None = None
    feature_rms_px: float | None = None
    features_used: bool = False
    ecc_model: str | None = None  # "homography" or "affine"
    ecc_correlation: float | None = None
    ecc_levels_completed: int = 0
    blur_match_sigma_px: float | None = None
    tone_curve_applied: bool = False
    outlier_fraction: float | None = None  # share of the label excluded from ECC as likely defects
    local_correction_applied: bool = False
    local_tiles: int = 0
    local_rms_before_px: float | None = None  # held-out tiles, without local correction
    local_rms_after_px: float | None = None  # held-out tiles, with local correction
    local_max_correction_px: float | None = None
    local_degree: int = 0
    correction_from_alignment_px: float | None = None
    residual_median_px: float | None = None
    residual_p95_px: float | None = None
    residual_max_px: float | None = None
    structured_tiles: int = 0
    verified_tiles: int = 0
    coverage: float | None = None
    structure_similarity: float | None = None
    mask_fraction: float | None = None

    @property
    def feature_inlier_ratio(self) -> float | None:
        if not self.feature_matches or self.feature_inliers is None:
            return None
        return self.feature_inliers / self.feature_matches


@dataclass(frozen=True, eq=False)
class RegistrationResult:
    metrics: RegistrationMetrics
    issues: tuple[QualityIssue, ...]
    registered: np.ndarray | None = None  # photo on the comparison grid, BGR uint8, read-only
    reference: np.ndarray | None = None  # artwork on the comparison grid, BGR uint8, read-only
    mask: np.ndarray | None = None  # uint8, 255 = comparable
    homography: np.ndarray | None = None  # grid -> photo (global part of the model)
    field: np.ndarray | None = None  # polynomial coefficients (terms x 2) of the local displacement, or None
    field_degree: int = 0
    tile_residuals: tuple[tuple[float, float, float, float], ...] = ()  # (x, y, dx, dy) verification tiles

    @property
    def level(self) -> QualityLevel:
        return level_from_issues(self.issues)

    @property
    def is_reliable(self) -> bool:
        return self.registered is not None and self.level is not QualityLevel.POOR

    @property
    def recommended_status(self) -> InspectionStatus | None:
        return None if self.is_reliable else InspectionStatus.REVIEW

    def issue_codes(self) -> set[str]:
        return {i.code for i in self.issues}

    def grid_to_photo(self, points: np.ndarray) -> np.ndarray:
        """Map grid pixel coordinates (N x 2) to photo pixel coordinates with the full model."""
        if self.homography is None or self.metrics.grid_size is None:
            raise ValueError("No registration model")
        pts = np.asarray(points, np.float64).reshape(-1, 2)
        if self.field is not None:
            pts = pts + _eval_field(self.field, self.field_degree, pts, self.metrics.grid_size)
        return cv2.perspectiveTransform(pts[None], self.homography)[0]


# ---------------------------------------------------------------- small helpers


def _scale_matrix(sx: float, sy: float) -> np.ndarray:
    """Pixel-centre convention: x' = (x + 0.5) * sx - 0.5."""
    return np.array([[sx, 0, 0.5 * sx - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1.0]])


def _sample(image: np.ndarray, grid_to_src: np.ndarray, size: tuple[int, int], flags: int = cv2.INTER_LINEAR):
    return cv2.warpPerspective(
        image, grid_to_src, size, flags=flags | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT
    )


def _corner_displacement(matrix: np.ndarray, size: tuple[int, int]) -> float:
    """Largest movement of the grid corners under a grid -> grid matrix."""
    w, h = size
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64)
    moved = cv2.perspectiveTransform(corners[None], matrix)[0]
    return float(np.max(np.linalg.norm(moved - corners, axis=1)))


def _zncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
    return float(np.sum(a * b) / denom) if denom > 0 else 0.0


def _gray32(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _poly_terms(u: np.ndarray, v: np.ndarray, degree: int) -> np.ndarray:
    return np.column_stack([u**i * v**j for i in range(degree + 1) for j in range(degree + 1 - i)])


def _normalised(points: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    w, h = size
    return (points[:, 0] - (w - 1) / 2.0) / (w / 2.0), (points[:, 1] - (h - 1) / 2.0) / (h / 2.0)


def _eval_field(coeffs: np.ndarray, degree: int, points: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    u, v = _normalised(points, size)
    return _poly_terms(u, v, degree) @ coeffs


# ---------------------------------------------------------------- comparison grid


def comparison_grid(
    alignment: AlignmentResult, artwork_shape: tuple[int, ...], cfg: RegistrationConfig
) -> tuple[tuple[int, int], np.ndarray]:
    """Grid size and the initial grid -> photo homography derived from the perspective correction."""
    if alignment.homography is None or alignment.metrics.output_size is None:
        raise ValueError("The label has not been aligned")
    ah, aw = artwork_shape[:2]
    rw, rh = alignment.metrics.output_size
    scale = min(rw / aw, cfg.max_grid_long_side_px / max(aw, ah))
    gw, gh = max(32, round(aw * scale)), max(32, round(ah * scale))
    grid_to_rect = _scale_matrix(rw / gw, rh / gh)
    return (gw, gh), np.linalg.inv(alignment.homography) @ grid_to_rect


# ---------------------------------------------------------------- estimation stages


def _feature_correction(ref_u8: np.ndarray, img_u8: np.ndarray, valid: np.ndarray, cfg: RegistrationConfig):
    """Grid -> grid homography from SIFT matches (reference -> sampled photo), or None."""
    gh, gw = ref_u8.shape
    s = min(1.0, cfg.feature_max_side_px / max(gw, gh))
    size = (max(16, round(gw * s)), max(16, round(gh * s)))
    a = cv2.resize(ref_u8, size, interpolation=cv2.INTER_AREA) if s < 1 else ref_u8
    b = cv2.resize(img_u8, size, interpolation=cv2.INTER_AREA) if s < 1 else img_u8
    m = cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST) if s < 1 else valid
    sift = cv2.SIFT_create(nfeatures=cfg.feature_max_keypoints)
    kp_a, des_a = sift.detectAndCompute(a, None)
    kp_b, des_b = sift.detectAndCompute(b, m)
    info = {"feature_matches": 0, "feature_inliers": 0}
    if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        return None, info
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_a, des_b, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < cfg.feature_ratio_test * p[1].distance]
    info["feature_matches"] = len(good)
    if len(good) < cfg.feature_min_inliers:
        return None, info
    back = _scale_matrix(1.0 / s, 1.0 / s) if s < 1 else np.eye(3)
    src = cv2.perspectiveTransform(np.array([kp_a[g.queryIdx].pt for g in good], np.float64)[None], back)[0]
    dst = cv2.perspectiveTransform(np.array([kp_b[g.trainIdx].pt for g in good], np.float64)[None], back)[0]
    h, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, cfg.feature_ransac_threshold_px / s, maxIters=10000, confidence=0.999)
    if h is None or mask is None:
        return None, info
    inliers = mask.ravel().astype(bool)
    info["feature_inliers"] = int(inliers.sum())
    projected = cv2.perspectiveTransform(src[inliers][None], h)[0]
    info["feature_rms_px"] = float(np.sqrt(np.mean(np.sum((projected - dst[inliers]) ** 2, axis=1))))
    if inliers.sum() < cfg.feature_min_inliers:
        return None, info
    hull = cv2.convexHull(dst[inliers].astype(np.float32))
    spread = float(cv2.contourArea(hull)) / (gw * gh)
    info["feature_spread"] = spread
    if spread < cfg.feature_min_spread:
        return None, info
    return h, info


def _ecc_correction(
    ref: np.ndarray, img: np.ndarray, valid: np.ndarray, cfg: RegistrationConfig, levels: int, motion: int
) -> tuple[np.ndarray, float | None, int]:
    """Grid -> grid correction (reference -> sampled photo) by coarse-to-fine ECC."""
    gh, gw = ref.shape
    top = min(1.0, cfg.ecc_max_side_px / max(gw, gh))
    scales = [top / 2**i for i in reversed(range(levels))]
    correction = np.eye(3)
    correlation, done = None, 0
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, cfg.ecc_iterations, cfg.ecc_epsilon)
    for k in scales:
        lw, lh = max(16, round(gw * k)), max(16, round(gh * k))
        to_level = _scale_matrix(lw / gw, lh / gh)
        t = cv2.resize(ref, (lw, lh), interpolation=cv2.INTER_AREA)
        i = cv2.resize(img, (lw, lh), interpolation=cv2.INTER_AREA)
        m = cv2.erode(cv2.resize(valid, (lw, lh), interpolation=cv2.INTER_NEAREST), np.ones((5, 5), np.uint8))
        warp = to_level @ correction @ np.linalg.inv(to_level)
        warp = (warp / warp[2, 2]).astype(np.float32)
        if motion != cv2.MOTION_HOMOGRAPHY:
            warp = warp[:2].copy()
        try:
            correlation, warp = cv2.findTransformECC(t, i, warp, motion, criteria, m, cfg.ecc_gauss_filter_size)
        except cv2.error as exc:
            log.info("ECC stopped at level %.3f: %s", k, str(exc).splitlines()[-1] if str(exc) else exc)
            break
        if warp.shape[0] == 2:
            warp = np.vstack([warp, [0, 0, 1]])
        correction = np.linalg.inv(to_level) @ warp.astype(np.float64) @ to_level
        done += 1
    return correction / correction[2, 2], (float(correlation) if correlation is not None else None), done


def _estimate_blur_match(ref: np.ndarray, img: np.ndarray, valid: np.ndarray, max_sigma: float) -> float:
    """Gaussian sigma that makes the reference best resemble the photo near edges."""
    grad = cv2.magnitude(cv2.Sobel(ref, cv2.CV_32F, 1, 0), cv2.Sobel(ref, cv2.CV_32F, 0, 1))
    near_edges = cv2.dilate((grad > 20).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    select = near_edges & (cv2.erode(valid, np.ones((7, 7), np.uint8)) > 0)
    if select.sum() < 500:
        return 0.0
    target = img[select]
    sigmas = np.arange(0.0, max_sigma + 1e-9, 0.25)
    scores = []
    for s in sigmas:
        blurred = cv2.GaussianBlur(ref, (0, 0), s) if s > 0 else ref
        scores.append(_zncc(blurred[select], target))
    best = int(np.argmax(scores))
    if 0 < best < len(sigmas) - 1:
        s0, s1, s2 = scores[best - 1], scores[best], scores[best + 1]
        denom = s0 - 2 * s1 + s2
        if denom < 0:
            return float(sigmas[best] + 0.25 * 0.5 * (s0 - s2) / denom)
    return float(sigmas[best])


@dataclass(frozen=True)
class _TileShift:
    x: float  # tile centre
    y: float
    dx: float  # displacement of the photo relative to the reference
    dy: float
    residual: float
    normal: tuple[float, float] | None = None  # set for one-directional tiles: only this component is known

    @property
    def magnitude(self) -> float:
        return math.hypot(self.dx, self.dy)


class _TileMeasurer:
    """Sub-pixel tile displacements: phase correlation start, Lucas-Kanade refinement."""

    def __init__(self, ref_raw: np.ndarray, ref: np.ndarray, img: np.ndarray, valid: np.ndarray, cfg: RegistrationConfig):
        self.cfg = cfg
        self.t = cfg.tile_size_px
        self.window = cv2.createHanningWindow((self.t, self.t), cv2.CV_32F)
        self.ref_raw = ref_raw
        self.ref = ref
        self.img = img
        self.valid = valid

    def classify(self, x0: int, y0: int) -> tuple[str, np.ndarray | None] | None:
        """("2d", None), ("1d", edge normal) or None (no usable structure) from the artwork tile."""
        tile = self.ref_raw[y0 : y0 + self.t, x0 : x0 + self.t]
        gx = cv2.Sobel(tile, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
        gy = cv2.Sobel(tile, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
        w = self.window
        wsum = float(w.sum())
        jxx, jyy, jxy = float((w * gx * gx).sum()), float((w * gy * gy).sum()), float((w * gx * gy).sum())
        lam, vec = np.linalg.eigh(np.array([[jxx, jxy], [jxy, jyy]]))
        lam_min, lam_max = float(lam[0]), float(lam[1])
        if lam_max / wsum < self.cfg.tile_min_structure:
            return None
        if lam_min / wsum >= self.cfg.tile_min_structure and lam_min / lam_max >= self.cfg.tile_min_isotropy:
            return "2d", None
        return "1d", vec[:, 1]

    def measure(self, x0: int, y0: int, normal: np.ndarray | None = None) -> _TileShift | None:
        t, cfg = self.t, self.cfg
        a = self.ref[y0 : y0 + t, x0 : x0 + t]
        b0 = self.img[y0 : y0 + t, x0 : x0 + t]
        (sx, sy), _ = cv2.phaseCorrelate(np.ascontiguousarray(a), np.ascontiguousarray(b0), self.window)
        # Gauss-Newton with the (noise-free) reference gradients: a fixed Jacobian keeps the
        # iteration stable; photo noise only makes the last steps jitter by ~0.01-0.03 px.
        a = (a - a.mean()) / (a.std() + 1e-6)
        w = self.window
        wsum = float(w.sum())
        gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
        gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
        if normal is None:
            jxx, jyy, jxy = float((w * gx * gx).sum()), float((w * gy * gy).sum()), float((w * gx * gy).sum())
            det = jxx * jyy - jxy * jxy
            if det <= 1e-9 * max(jxx * jyy, 1e-12):
                return None
            shift = np.array([sx, sy], np.float64)
        else:
            gn = gx * float(normal[0]) + gy * float(normal[1])
            jnn = float((w * gn * gn).sum())
            if jnn <= 1e-9:
                return None
            shift = float(np.dot([sx, sy], normal)) * normal  # the tangential part is undetermined
        if math.hypot(*shift) > cfg.tile_max_shift_px:
            return None
        residual, step_size = math.inf, math.inf
        for _ in range(10):
            move = np.float32([[1, 0, -(x0 + shift[0])], [0, 1, -(y0 + shift[1])]])
            b = cv2.warpAffine(self.img, move, (t, t), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            b = (b - b.mean()) / (b.std() + 1e-6)
            err = b - a
            residual = math.sqrt(float((w * err * err).sum()) / wsum)
            if normal is None:
                rx, ry = float((w * gx * err).sum()), float((w * gy * err).sum())
                step = -np.array([jyy * rx - jxy * ry, jxx * ry - jxy * rx]) / det
            else:
                step = -float((w * gn * err).sum()) / jnn * normal
            shift = shift + step
            step_size = math.hypot(*step)
            if math.hypot(*shift) > cfg.tile_max_shift_px:
                return None
            if step_size < 0.01:
                break
        if step_size >= 0.05:  # still moving: not converged
            return None
        if residual > cfg.tile_max_residual:
            return None
        n = None if normal is None else (float(normal[0]), float(normal[1]))
        return _TileShift(x0 + (t - 1) / 2.0, y0 + (t - 1) / 2.0, float(shift[0]), float(shift[1]), residual, n)

    def positions(self, offset: int, include_borders: bool) -> list[tuple[int, int]]:
        h, w = self.ref.shape
        t = self.t
        xs = list(range(offset, w - t + 1, t))
        ys = list(range(offset, h - t + 1, t))
        if include_borders:
            cols = np.flatnonzero(self.valid.any(axis=0))
            rows = np.flatnonzero(self.valid.any(axis=1))
            if cols.size and rows.size:
                for v, top in ((int(cols[0]), w - t), (int(cols[-1]) - t + 1, w - t)):
                    if 0 <= v <= top:
                        xs.append(v)
                for v, top in ((int(rows[0]), h - t), (int(rows[-1]) - t + 1, h - t)):
                    if 0 <= v <= top:
                        ys.append(v)
        return [(x, y) for y in sorted(set(ys)) for x in sorted(set(xs))]

    def grid(self, offset: int, include_borders: bool = False) -> tuple[list[_TileShift], int]:
        """Measure all usable tiles of a grid starting at ``offset``. Returns (shifts, structured count)."""
        t = self.t
        shifts, structured = [], 0
        for x0, y0 in self.positions(offset, include_borders):
            if np.count_nonzero(self.valid[y0 : y0 + t, x0 : x0 + t]) < 0.98 * t * t:
                continue
            kind = self.classify(x0, y0)
            if kind is None:
                continue
            structured += 1
            found = self.measure(x0, y0, kind[1])
            if found is not None:
                shifts.append(found)
        return shifts, structured

    def count_2d(self) -> tuple[int, float]:
        """Number of 2D-structured tiles and the share of the grid their centres span."""
        h, w = self.ref.shape
        t = self.t
        centres = []
        for x0, y0 in self.positions(0, False):
            if np.count_nonzero(self.valid[y0 : y0 + t, x0 : x0 + t]) < 0.98 * t * t:
                continue
            kind = self.classify(x0, y0)
            if kind is not None and kind[0] == "2d":
                centres.append((x0 + t / 2, y0 + t / 2))
        if len(centres) < 3:
            return len(centres), 0.0
        c = np.array(centres)
        span = (np.ptp(c[:, 0]) + t) * (np.ptp(c[:, 1]) + t) / (w * h)
        return len(centres), float(min(1.0, span))


def _field_rows(shifts: list[_TileShift], degree: int, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rows for the field coefficients [cx..., cy...]: two per 2D tile, one per 1D tile."""
    pts = np.array([[s.x, s.y] for s in shifts])
    u, v = _normalised(pts, size)
    terms = _poly_terms(u, v, degree)
    zeros = np.zeros_like(terms)
    rows, values = [], []
    for i, s in enumerate(shifts):
        if s.normal is None:
            rows.append(np.concatenate([terms[i], zeros[i]]))
            values.append(s.dx)
            rows.append(np.concatenate([zeros[i], terms[i]]))
            values.append(s.dy)
        else:
            nx, ny = s.normal
            rows.append(np.concatenate([nx * terms[i], ny * terms[i]]))
            values.append(nx * s.dx + ny * s.dy)
    return np.array(rows), np.array(values)


def _tile_errors(shifts: list[_TileShift], coeffs: np.ndarray | None, degree: int, size: tuple[int, int]) -> np.ndarray:
    """Remaining misalignment per tile after a field (None = no field); 1D tiles: normal component."""
    pts = np.array([[s.x, s.y] for s in shifts])
    d = np.array([[s.dx, s.dy] for s in shifts])
    if coeffs is not None:
        d = d - _eval_field(coeffs, degree, pts, size)
    out = np.hypot(d[:, 0], d[:, 1])
    for i, s in enumerate(shifts):
        if s.normal is not None:
            out[i] = abs(d[i, 0] * s.normal[0] + d[i, 1] * s.normal[1])
    return out


def _fit_field(shifts: list[_TileShift], degree: int, size: tuple[int, int]) -> np.ndarray | None:
    a, b = _field_rows(shifts, degree, size)
    n_terms = a.shape[1] // 2
    if len(b) < a.shape[1] + 2:
        return None
    coeffs, *_ = np.linalg.lstsq(a, b, rcond=None)
    errors = _tile_errors(shifts, coeffs.reshape(2, n_terms).T, degree, size)
    keep = errors <= max(0.3, 3.0 * 1.4826 * float(np.median(errors)))
    if not keep.all():
        kept = [s for s, k in zip(shifts, keep) if k]
        a, b = _field_rows(kept, degree, size)
        if len(b) >= a.shape[1] + 2:
            coeffs, *_ = np.linalg.lstsq(a, b, rcond=None)
    return coeffs.reshape(2, n_terms).T


def _local_field(
    shifts: list[_TileShift], tile: int, cfg: RegistrationConfig, size: tuple[int, int]
) -> tuple[np.ndarray | None, int, dict]:
    """Smooth displacement field, kept only if it predicts held-out tiles better than none.

    Polynomial degrees 1..``local_poly_degree`` are compared by checkerboard
    cross-validation; sparse artwork gets a simpler field rather than none.
    Returns (coefficients or None, degree, info).
    """
    info: dict = {"local_tiles": len(shifts)}
    if len(shifts) < cfg.local_min_tiles:
        return None, 0, info
    parity = [(int(s.x // tile) + int(s.y // tile)) % 2 for s in shifts]
    halves = [[s for s, p in zip(shifts, parity) if p == k] for k in (0, 1)]
    if not halves[0] or not halves[1]:
        return None, 0, info
    before = np.concatenate([_tile_errors(h, None, 0, size) for h in halves])
    rms_before = float(np.sqrt(np.mean(before**2)))
    info["local_rms_before_px"] = rms_before
    best_degree, best_rms = 0, rms_before
    for degree in range(1, cfg.local_poly_degree + 1):
        after = []
        for train, test in ((halves[0], halves[1]), (halves[1], halves[0])):
            coeffs = _fit_field(train, degree, size)
            if coeffs is None:
                break
            after.extend(_tile_errors(test, coeffs, degree, size))
        else:
            rms = float(np.sqrt(np.mean(np.square(after))))
            if rms < best_rms:
                best_degree, best_rms = degree, rms
    info["local_rms_after_px"] = best_rms
    if (
        best_degree == 0
        or rms_before < cfg.local_min_rms_px
        or best_rms > (1.0 - cfg.local_min_improvement) * rms_before
    ):
        return None, 0, info
    coeffs = _fit_field(shifts, best_degree, size)
    if coeffs is None:
        return None, 0, info
    w, h = size
    ys, xs = np.mgrid[0 : h : max(1, h // 16), 0 : w : max(1, w // 16)]
    probe = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    info["local_max_correction_px"] = float(
        np.max(np.linalg.norm(_eval_field(coeffs, best_degree, probe, size), axis=1))
    )
    info["local_degree"] = best_degree
    return coeffs, best_degree, info


def _tone_curve(model: np.ndarray, img: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
    """Monotonic look-up table mapping (blurred) artwork gray levels to photo gray levels."""
    select = cv2.erode(valid, np.ones((5, 5), np.uint8)) > 0
    x = model[select]
    y = img[select]
    if x.size < 1000:
        return None
    if x.size > 400_000:
        idx = np.random.default_rng(0).choice(x.size, 400_000, replace=False)
        x, y = x[idx], y[idx]
    edges = np.linspace(0, 256, 65)
    which = np.clip(np.digitize(x, edges) - 1, 0, 63)
    centres, medians = [], []
    for k in range(64):
        values = y[which == k]
        if values.size >= 30:
            centres.append(0.5 * (edges[k] + edges[k + 1]))
            medians.append(float(np.median(values)))
    if len(centres) < 4:
        return None
    medians = np.maximum.accumulate(np.array(medians))
    return np.interp(np.arange(256, dtype=np.float64), centres, medians).astype(np.float32)


def _apply_curve(image: np.ndarray, lut: np.ndarray | None) -> np.ndarray:
    if lut is None:
        return image
    return np.interp(image, np.arange(256, dtype=np.float32), lut).astype(np.float32)


def _photometric_model(
    ref: np.ndarray, img: np.ndarray, valid: np.ndarray, max_sigma: float
) -> tuple[np.ndarray, float, np.ndarray | None]:
    """(artwork as it should look in the photo, blur sigma, tone curve)."""
    sigma = _estimate_blur_match(ref, img, valid, max_sigma)
    blurred = cv2.GaussianBlur(ref, (0, 0), sigma) if sigma > 0 else ref
    curve = _tone_curve(blurred, img, valid)
    return _apply_curve(blurred, curve), sigma, curve


def _without_outliers(model: np.ndarray, img: np.ndarray, valid: np.ndarray, edge_tolerance_px: float) -> np.ndarray:
    """``valid`` minus pixels differing far more than misalignment could explain (likely defects).

    Differences at edges are allowed up to ``edge_tolerance_px * gradient``. If the
    outliers would remove more than 30 % of the area, the image is probably not yet
    aligned (or the artwork is wrong) and nothing is removed.
    """
    gradient = cv2.magnitude(
        cv2.Sobel(model, cv2.CV_32F, 1, 0, ksize=3, scale=0.125), cv2.Sobel(model, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
    )
    gradient = cv2.dilate(gradient, np.ones((3, 3), np.uint8))
    excess = np.abs(img - model) - edge_tolerance_px * gradient
    inside = valid > 0
    if not inside.any():
        return valid
    spread = 1.4826 * float(np.median(np.abs(img - model)[inside])) + 1.0
    outliers = (excess > max(25.0, 6.0 * spread)) & inside
    outliers = cv2.dilate(outliers.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    if np.count_nonzero(outliers & inside) > 0.3 * np.count_nonzero(inside):
        return valid
    trusted = valid.copy()
    trusted[outliers] = 0
    return trusted


# ---------------------------------------------------------------- final resampling


def _render(
    photo: np.ndarray,
    homography: np.ndarray,
    field: np.ndarray | None,
    degree: int,
    size: tuple[int, int],
    interpolation: str,
    photo_to_rect: np.ndarray,
    rect_size: tuple[int, int],
    border_rect_px: float,
    border_grid_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Single resampling of the original photo onto the grid, plus the comparison mask."""
    gw, gh = size
    ph, pw = photo.shape[:2]
    rw, rh = rect_size
    margin = _KERNEL_MARGIN[interpolation]
    out = np.zeros((gh, gw, 3), np.uint8)
    mask = np.zeros((gh, gw), np.uint8)
    xs = np.arange(gw, dtype=np.float64)
    hm = homography
    grid_border = max(0, int(math.ceil(border_grid_px)))
    for y0 in range(0, gh, _STRIP_ROWS):
        y1 = min(gh, y0 + _STRIP_ROWS)
        gx, gy = np.meshgrid(xs, np.arange(y0, y1, dtype=np.float64))
        # Artwork edge pixels mix label and surroundings: never compare them.
        in_grid = (gx >= grid_border) & (gx <= gw - 1 - grid_border) & (gy >= grid_border) & (gy <= gh - 1 - grid_border)
        if field is not None:
            pts = np.column_stack([gx.ravel(), gy.ravel()])
            d = _eval_field(field, degree, pts, size)
            gx = gx + d[:, 0].reshape(gx.shape)
            gy = gy + d[:, 1].reshape(gy.shape)
        den = hm[2, 0] * gx + hm[2, 1] * gy + hm[2, 2]
        px = (hm[0, 0] * gx + hm[0, 1] * gy + hm[0, 2]) / den
        py = (hm[1, 0] * gx + hm[1, 1] * gy + hm[1, 2]) / den
        out[y0:y1] = cv2.remap(
            photo, px.astype(np.float32), py.astype(np.float32), _INTERPOLATION[interpolation],
            borderMode=cv2.BORDER_CONSTANT,
        )
        rd = photo_to_rect[2, 0] * px + photo_to_rect[2, 1] * py + photo_to_rect[2, 2]
        rx = (photo_to_rect[0, 0] * px + photo_to_rect[0, 1] * py + photo_to_rect[0, 2]) / rd
        ry = (photo_to_rect[1, 0] * px + photo_to_rect[1, 1] * py + photo_to_rect[1, 2]) / rd
        inside = (
            in_grid
            & (den > 0)
            & (px >= margin) & (px <= pw - 1 - margin) & (py >= margin) & (py <= ph - 1 - margin)
            & (rx >= border_rect_px - 0.5) & (rx <= rw - 0.5 - border_rect_px)
            & (ry >= border_rect_px - 0.5) & (ry <= rh - 0.5 - border_rect_px)
        )
        mask[y0:y1][inside] = 255
    return out, mask


def _valid_sampling_mask(
    grid_to_photo: np.ndarray, photo_shape: tuple[int, ...], alignment: AlignmentResult, size: tuple[int, int]
) -> np.ndarray:
    """Where the sampled photo has data from inside the label (for estimation only)."""
    ph, pw = photo_shape[:2]
    in_photo = _sample(np.full((ph, pw), 255, np.uint8), grid_to_photo, size, cv2.INTER_NEAREST)
    rw, rh = alignment.metrics.output_size
    in_label = _sample(np.full((rh, rw), 255, np.uint8), alignment.homography @ grid_to_photo, size, cv2.INTER_NEAREST)
    return cv2.bitwise_and(in_photo, in_label)


# ---------------------------------------------------------------- main entry point


def register_to_artwork(
    photo: np.ndarray,
    artwork: np.ndarray,
    alignment: AlignmentResult,
    cfg: RegistrationConfig,
    initial_homography: np.ndarray | None = None,
    refine: bool = True,
) -> RegistrationResult:
    """Register the photo (HxWx3 uint8 BGR working image) to the artwork on the comparison grid.

    ``initial_homography`` (grid -> photo) starts from a manual adjustment; with
    ``refine=False`` it is used as is and only verified.
    """
    for name, image in (("photo", photo), ("artwork", artwork)):
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"Expected HxWx3 uint8 BGR {name}, got {image.shape} {image.dtype}")
    if initial_homography is None:
        method = RegistrationMethod.AUTOMATIC
    else:
        method = RegistrationMethod.MANUAL_REFINED if refine else RegistrationMethod.MANUAL
    if alignment is None or alignment.homography is None:
        return RegistrationResult(
            metrics=RegistrationMetrics(method=method),
            issues=(QualityIssue(
                "not_aligned", Severity.CRITICAL,
                "The label must be located and perspective-corrected before it can be registered.",
            ),),
        )

    size, start = comparison_grid(alignment, artwork.shape, cfg)
    gw, gh = size
    ah, aw = artwork.shape[:2]
    m: dict = {"method": method, "grid_size": size, "grid_scale": gw / aw}
    issues: list[QualityIssue] = []

    reference = cv2.resize(artwork, size, interpolation=cv2.INTER_AREA if gw < aw else cv2.INTER_LINEAR)
    reference.setflags(write=False)
    ref_gray = _gray32(reference)
    photo_gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    homography = start if initial_homography is None else np.asarray(initial_homography, np.float64)

    def sampled(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _sample(photo_gray, h, size), _valid_sampling_mask(h, photo.shape, alignment, size)

    diagonal = math.hypot(gw, gh)
    presmooth = cfg.tile_presmooth_sigma_px
    if refine:
        img, valid = sampled(homography)
        if initial_homography is None:
            correction, info = _feature_correction(ref_gray.astype(np.uint8), img, valid, cfg)
            m.update({k: v for k, v in info.items() if k != "feature_spread"})
            if correction is not None:
                if _corner_displacement(correction, size) <= cfg.max_global_correction_fraction * diagonal:
                    homography = homography @ correction
                    m["features_used"] = True
                    img, valid = sampled(homography)
                else:
                    log.info("Feature correction rejected as implausibly large")
        # Perspective terms are only trustworthy when 2D detail covers the label.
        tiles_2d, span = _TileMeasurer(ref_gray, ref_gray, ref_gray, valid, cfg).count_2d()
        motion = (
            cv2.MOTION_HOMOGRAPHY
            if tiles_2d >= cfg.ecc_homography_min_tiles and span >= 0.3
            else cv2.MOTION_AFFINE
        )
        m["ecc_model"] = "homography" if motion == cv2.MOTION_HOMOGRAPHY else "affine"
        # Print defects must not pull the registration: model the expected photo first
        # and leave pixels that differ far beyond misalignment out of ECC.
        img = img.astype(np.float32)
        model_ref = _photometric_model(ref_gray, img, valid, cfg.blur_match_max_sigma_px)[0]
        trusted = _without_outliers(model_ref, img, valid, cfg.outlier_edge_tolerance_px * 4)
        correction, correlation, done = _ecc_correction(model_ref, img, trusted, cfg, cfg.ecc_levels, motion)
        if done and _corner_displacement(correction, size) <= cfg.max_global_correction_fraction * diagonal:
            homography = homography @ correction
        m.update(ecc_correlation=correlation, ecc_levels_completed=done)
        if done < cfg.ecc_levels:
            issues.append(QualityIssue(
                "ecc_incomplete", Severity.WARNING,
                f"Intensity-based refinement stopped early ({done} of {cfg.ecc_levels} levels). "
                "Accuracy relies on the other stages; see the verification result.",
            ))

    # ---- photometric model of the photo: blur, then the camera tone curve
    img, valid = sampled(homography)
    img = img.astype(np.float32)
    model_ref, sigma, curve = _photometric_model(ref_gray, img, valid, cfg.blur_match_max_sigma_px)
    m.update(blur_match_sigma_px=sigma, tone_curve_applied=curve is not None)

    if refine:
        # Final global pass against the photometrically matched reference, excluding outliers.
        trusted = _without_outliers(model_ref, img, valid, cfg.outlier_edge_tolerance_px)
        correction, correlation, done = _ecc_correction(model_ref, img, trusted, cfg, 1, motion)
        if done and _corner_displacement(correction, size) <= cfg.max_global_correction_fraction * diagonal:
            homography = homography @ correction
            m["ecc_correlation"] = correlation
            img, valid = sampled(homography)
            img = img.astype(np.float32)
        m["outlier_fraction"] = 1.0 - float(np.count_nonzero(trusted)) / max(1, np.count_nonzero(valid))
    ref_matched = cv2.GaussianBlur(model_ref, (0, 0), presmooth)

    field, degree = None, 0
    if refine:
        measurer = _TileMeasurer(ref_gray, ref_matched, cv2.GaussianBlur(img, (0, 0), presmooth), valid, cfg)
        shifts, _ = measurer.grid(0, include_borders=True)
        field, degree, info = _local_field(shifts, cfg.tile_size_px, cfg, size)
        m.update(info)
        m["local_correction_applied"] = field is not None
        if field is not None and m.get("local_max_correction_px", 0.0) >= cfg.local_warning_correction_px:
            issues.append(QualityIssue(
                "local_distortion", Severity.WARNING,
                f"The photo needed a local correction of up to {m['local_max_correction_px']:.1f} px "
                "(lens distortion or a label that is not flat).",
            ))

    m["correction_from_alignment_px"] = _corner_displacement(np.linalg.inv(start) @ homography, size)

    # ---- the single resampling of the original photo
    rw, rh = alignment.metrics.output_size
    border_rect = cfg.mask_border_px * (rw / gw)
    registered, mask = _render(
        photo, homography, field, degree, size, cfg.interpolation,
        alignment.homography, (rw, rh), border_rect, cfg.mask_border_px,
    )
    registered.setflags(write=False)
    mask.setflags(write=False)
    mask_fraction = float(np.count_nonzero(mask)) / mask.size
    m["mask_fraction"] = mask_fraction

    # ---- verification on an offset tile grid, after the final resampling
    reg_gray = _gray32(registered)
    verifier = _TileMeasurer(ref_gray, ref_matched, cv2.GaussianBlur(reg_gray, (0, 0), presmooth), mask, cfg)
    checks, structured = verifier.grid(cfg.tile_size_px // 2, include_borders=True)
    m.update(structured_tiles=structured, verified_tiles=len(checks))
    tile_residuals = tuple((c.x, c.y, c.dx, c.dy) for c in checks)
    if checks:
        magnitudes = _tile_errors(checks, None, 0, size)
        m.update(
            residual_median_px=float(np.median(magnitudes)),
            residual_p95_px=float(np.percentile(magnitudes, 95)),
            residual_max_px=float(magnitudes.max()),
        )
    if structured:
        m["coverage"] = len(checks) / structured

    s = min(1.0, 800 / max(gw, gh))
    small = (max(8, round(gw * s)), max(8, round(gh * s)))
    sel = cv2.resize(mask, small, interpolation=cv2.INTER_NEAREST) > 0

    def structure(g: np.ndarray) -> np.ndarray:
        g = cv2.GaussianBlur(cv2.resize(g, small, interpolation=cv2.INTER_AREA), (0, 0), 1.0)
        return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))

    if sel.sum() > 100:
        m["structure_similarity"] = _zncc(structure(ref_gray)[sel], structure(reg_gray)[sel])

    issues.extend(_grade(RegistrationMetrics(**m), cfg))
    return RegistrationResult(
        metrics=RegistrationMetrics(**m),
        issues=sort_issues(issues),
        registered=registered,
        reference=reference,
        mask=mask,
        homography=homography,
        field=field,
        field_degree=degree,
        tile_residuals=tile_residuals,
    )


def _grade(m: RegistrationMetrics, cfg: RegistrationConfig) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    if m.residual_p95_px is not None:
        if m.residual_p95_px >= cfg.residual_p95_critical_px:
            severity = Severity.CRITICAL
        elif m.residual_p95_px >= cfg.residual_p95_warning_px:
            severity = Severity.WARNING
        else:
            severity = None
        if severity:
            issues.append(QualityIssue(
                "misaligned", severity,
                f"After registration, detailed areas are still offset from the artwork by up to "
                f"{m.residual_p95_px:.2f} px (95th percentile, max {m.residual_max_px:.2f} px). "
                "Misalignment of this size causes false defects along text and lines.",
            ))
    if m.structured_tiles == 0:
        issues.append(QualityIssue(
            "no_detail", Severity.WARNING,
            "The artwork has no detailed areas in which the alignment could be verified.",
        ))
    elif m.verified_tiles < cfg.min_verified_tiles:
        issues.append(QualityIssue(
            "unverified", Severity.CRITICAL,
            f"Alignment could be verified in only {m.verified_tiles} of {m.structured_tiles} detailed areas.",
        ))
    elif m.coverage is not None and m.coverage < cfg.coverage_warning:
        severity = Severity.CRITICAL if m.coverage < cfg.coverage_critical else Severity.WARNING
        issues.append(QualityIssue(
            "low_coverage", severity,
            f"Alignment could be verified in only {m.coverage:.0%} of the detailed areas "
            f"({m.verified_tiles} of {m.structured_tiles}). Elsewhere the print differs too much "
            "from the artwork to measure, or alignment failed.",
        ))
    if m.structure_similarity is not None:
        if m.structure_similarity < cfg.min_similarity_critical:
            severity = Severity.CRITICAL
        elif m.structure_similarity < cfg.min_similarity_warning:
            severity = Severity.WARNING
        else:
            severity = None
        if severity:
            issues.append(QualityIssue(
                "content_mismatch", severity,
                f"The registered print resembles the artwork only weakly (similarity "
                f"{m.structure_similarity:.2f}). Check that the correct artwork is loaded.",
            ))
    if m.mask_fraction is not None:
        if m.mask_fraction < cfg.min_mask_fraction_critical:
            severity = Severity.CRITICAL
        elif m.mask_fraction < cfg.min_mask_fraction_warning:
            severity = Severity.WARNING
        else:
            severity = None
        if severity:
            issues.append(QualityIssue(
                "small_comparison_area", severity,
                f"Only {m.mask_fraction:.0%} of the artwork area can be compared with the photo.",
            ))
    return issues
