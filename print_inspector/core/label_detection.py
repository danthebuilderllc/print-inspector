"""Locate the printed label's four corners in a smartphone photo.

Two independent methods are provided:

Boundary detection (primary, :func:`detect_label_boundary`)
    1. On a downscaled copy, candidate outlines come from (a) separating the
       label from the background by colour distance to the photo border, and
       (b) closed Canny edge contours. Each outline is reduced to a convex
       quadrilateral.
    2. Candidates are scored by how much of their outline lies on a real edge;
       the largest well-supported one is chosen.
    3. Each side is refined at full resolution: the edge is located with
       sub-pixel precision along many normals across the side, a straight line
       is fitted robustly, and the corners are the intersections of adjacent
       lines. Sampling only the middle 80 % of each side makes this tolerant of
       rounded corners.

Feature matching (fallback, :func:`match_artwork`, needs the artwork)
    SIFT keypoints of artwork and photo are matched (Lowe ratio test) and a
    homography is estimated with MAGSAC++. The label corners are the projected
    artwork corners. This works when the label edge is invisible (e.g. a
    white label on a white surface), but measures the artwork canvas, which
    can include bleed.

All corner arrays are ``4x2`` float64 in full-resolution photo pixel
coordinates (pixel centres at integers), ordered TL, TR, BR, BL.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from print_inspector.config import AlignmentConfig

log = logging.getLogger(__name__)

SIDE_NAMES = ("top", "right", "bottom", "left")


@dataclass(frozen=True)
class SideFit:
    """Result of fitting one label side at full resolution."""

    point: np.ndarray  # a point on the fitted line
    direction: np.ndarray  # unit direction of the line
    support: float  # share of sample positions where a consistent edge was found
    rms_px: float  # RMS distance of the edge points from the fitted line
    contrast: float  # median brightness step across the edge (0-255)


@dataclass(frozen=True)
class BoundaryDetection:
    corners: np.ndarray
    sides: tuple[SideFit, ...]  # top, right, bottom, left (photo orientation)
    source: str  # "segmentation" or "edges"

    @property
    def min_support(self) -> float:
        return min(s.support for s in self.sides)

    @property
    def max_rms_px(self) -> float:
        return max(s.rms_px for s in self.sides)


@dataclass(frozen=True)
class FeatureMatch:
    homography: np.ndarray  # artwork pixel coords -> photo pixel coords
    corners: np.ndarray
    matches: int
    inliers: int
    reprojection_rms_px: float

    @property
    def inlier_ratio(self) -> float:
        return self.inliers / self.matches if self.matches else 0.0


# ---------------------------------------------------------------- geometry helpers


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order four points clockwise (as seen in the image), starting nearest the top-left."""
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    pts = pts[np.argsort(angles)]  # increasing angle = clockwise with y pointing down
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    return np.roll(pts, -start, axis=0)


def is_convex_quad(corners: np.ndarray) -> bool:
    pts = np.asarray(corners, np.float64)
    signs = []
    for i in range(4):
        a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(np.sign(cross))
    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


def quad_area(corners: np.ndarray) -> float:
    return float(abs(cv2.contourArea(np.asarray(corners, np.float32))))


def _scale_points(points: np.ndarray, factor: float) -> np.ndarray:
    """Map pixel coordinates between resolutions (pixel-centre convention)."""
    return (np.asarray(points, np.float64) + 0.5) * factor - 0.5


def _line_intersection(p1, d1, p2, d2) -> np.ndarray | None:
    """Intersection of lines p1 + t*d1 and p2 + s*d2 (closed form; None if parallel)."""
    det = float(d1[0]) * float(-d2[1]) - float(-d2[0]) * float(d1[1])
    if abs(det) < 1e-9:
        return None
    rx, ry = float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1])
    t = (rx * float(-d2[1]) - float(-d2[0]) * ry) / det
    return np.array([float(p1[0]) + t * float(d1[0]), float(p1[1]) + t * float(d1[1])])


# ---------------------------------------------------------------- candidate outlines


def _polygon_to_quad(contour: np.ndarray) -> np.ndarray | None:
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    if perimeter <= 0:
        return None
    quad = None
    for eps in np.linspace(0.005, 0.08, 16):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            quad = order_corners(approx.reshape(4, 2))
            break
        if len(approx) < 4:
            break
    if quad is None:
        quad = order_corners(cv2.boxPoints(cv2.minAreaRect(hull)))
    return _fit_quad_sides(contour.reshape(-1, 2).astype(np.float64), quad)


def _fit_quad_sides(points: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Replace each side by a line fitted to the straight middle part of the outline.

    For rounded corners the polygon vertices sit on the arcs, so its sides run
    inside the true edges; the fitted lines meet at the virtual sharp corners.
    """
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        d = b - a
        length = float(np.hypot(*d))
        if length < 4:
            return quad
        t_hat = d / length
        normal = np.array([-t_hat[1], t_hat[0]])
        rel = points - a
        t = rel @ t_hat / length
        dist = np.abs(rel @ normal)
        near = points[(t > 0.15) & (t < 0.85) & (dist < 0.05 * length + 2.0)]
        if len(near) < 5:
            return quad
        vx, vy, x0, y0 = cv2.fitLine(near.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))
    corners = []
    for i in range(4):
        corner = _line_intersection(*lines[i - 1], *lines[i])
        if corner is None:
            return quad
        corners.append(corner)
    corners = np.array(corners)
    if not is_convex_quad(corners) or np.max(np.linalg.norm(corners - quad, axis=1)) > 0.25 * min(
        np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)
    ):
        return quad
    return corners


def _segmentation_candidates(small_bgr: np.ndarray, cfg: AlignmentConfig) -> list[np.ndarray]:
    h, w = small_bgr.shape[:2]
    lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    band = max(2, int(round(cfg.background_band_fraction * min(h, w))))
    border = np.concatenate(
        [lab[:band].reshape(-1, 3), lab[-band:].reshape(-1, 3), lab[:, :band].reshape(-1, 3), lab[:, -band:].reshape(-1, 3)]
    )
    background = np.median(border, axis=0)
    distance = np.sqrt(np.sum((lab - background) ** 2, axis=2))
    distance8 = cv2.GaussianBlur(np.clip(distance, 0, 255).astype(np.uint8), (0, 0), 1.5)
    otsu, _ = cv2.threshold(distance8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(float(otsu), cfg.edge_min_contrast)
    mask = (distance8 > threshold).astype(np.uint8) * 255
    k = max(3, int(round(min(h, w) * 0.01)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
        quad = _polygon_to_quad(contour)
        if quad is not None:
            candidates.append(quad)
    return candidates


def _edge_candidates(gray_small: np.ndarray, cfg: AlignmentConfig) -> list[np.ndarray]:
    blurred = cv2.GaussianBlur(gray_small, (0, 0), 1.2)
    edges = cv2.Canny(blurred, 2 * cfg.edge_min_contrast, 5 * cfg.edge_min_contrast)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area = cfg.min_label_area_fraction * gray_small.size
    candidates = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:25]:
        if cv2.contourArea(contour) < min_area:
            break
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            candidates.append(approx.reshape(4, 2).astype(np.float64))
        else:
            # Slightly curved sides (lens distortion, label not quite flat) or a ragged
            # outline: reduce the convex hull to four sides instead.
            hull = cv2.convexHull(contour)
            if cv2.contourArea(hull) > 0 and cv2.contourArea(contour) / cv2.contourArea(hull) > 0.85:
                quad = _polygon_to_quad(contour)
                if quad is not None:
                    candidates.append(quad)
    return candidates


def _outline_support(gradient: np.ndarray, quad: np.ndarray, threshold: float, samples: int = 40) -> float:
    """Share of points along the quad outline that lie on a strong edge (within ±2 px)."""
    h, w = gradient.shape
    hits = total = 0
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        d = b - a
        length = float(np.hypot(*d))
        if length < 1:
            return 0.0
        n = np.array([-d[1], d[0]]) / length
        for t in np.linspace(0.1, 0.9, samples):
            p = a + t * d
            best = 0.0
            for u in (-2, -1, 0, 1, 2):
                x, y = int(round(p[0] + u * n[0])), int(round(p[1] + u * n[1]))
                if 0 <= x < w and 0 <= y < h:
                    best = max(best, float(gradient[y, x]))
            hits += best >= threshold
            total += 1
    return hits / total if total else 0.0


# ---------------------------------------------------------------- full-resolution refinement


def _fit_side(gray: np.ndarray, a: np.ndarray, b: np.ndarray, search: float, cfg: AlignmentConfig) -> SideFit | None:
    h, w = gray.shape
    d = b - a
    length = float(np.hypot(*d))
    if length < 10:
        return None
    t_hat = d / length
    normal = np.array([-t_hat[1], t_hat[0]])
    n_samples = cfg.edge_samples_per_side
    ts = np.linspace(0.1, 0.9, n_samples)
    base = a[None, :] + ts[:, None] * d[None, :]
    half = int(np.ceil(search))
    us = np.arange(-half - 2, half + 3, dtype=np.float64)  # 2 extra each side for the derivative kernel
    coords = base[:, None, :] + us[None, :, None] * normal[None, None, :]

    inside = (
        (coords[..., 0].min(axis=1) >= 0)
        & (coords[..., 0].max(axis=1) <= w - 1)
        & (coords[..., 1].min(axis=1) >= 0)
        & (coords[..., 1].max(axis=1) <= h - 1)
    )
    x0 = int(max(0, np.floor(coords[..., 0].min()) - 1))
    y0 = int(max(0, np.floor(coords[..., 1].min()) - 1))
    x1 = int(min(w, np.ceil(coords[..., 0].max()) + 2))
    y1 = int(min(h, np.ceil(coords[..., 1].max()) + 2))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    crop = gray[y0:y1, x0:x1].astype(np.float32)
    map_x = (coords[..., 0] - x0).astype(np.float32)
    map_y = (coords[..., 1] - y0).astype(np.float32)
    profiles = cv2.remap(crop, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    deriv = profiles[:, 2:] - profiles[:, :-2]  # centred at us[1:-1]
    deriv = (deriv[:, :-2] + 2 * deriv[:, 1:-1] + deriv[:, 2:]) / 4.0  # centred at us[2:-2]
    offsets = us[2:-2]
    # Either polarity is accepted: artwork running to the label edge can make the
    # label darker than the background along part of a side and lighter along the rest.
    # Edges picked up from other content are removed by the robust line fit below.
    score = np.abs(deriv)
    k = np.argmax(score, axis=1)
    rows = np.arange(n_samples)
    lo = np.clip(k - 1, 0, score.shape[1] - 1)
    hi = np.clip(k + 1, 0, score.shape[1] - 1)
    s0, s1, s2 = score[rows, lo], score[rows, k], score[rows, hi]
    denom = s0 - 2 * s1 + s2
    delta = np.where(np.abs(denom) > 1e-6, 0.5 * (s0 - s2) / np.where(denom == 0, 1, denom), 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    u_peak = offsets[k] + delta

    near = np.clip(k + 2 - 4, 0, profiles.shape[1] - 1)  # profile index of offsets[k] is k + 2
    far = np.clip(k + 2 + 4, 0, profiles.shape[1] - 1)
    contrast = np.abs(profiles[rows, far] - profiles[rows, near])
    valid = inside & (contrast >= cfg.edge_min_contrast) & (k > 0) & (k < score.shape[1] - 1)
    if valid.sum() < max(6, int(0.2 * n_samples)):
        return None

    points = base[valid] + u_peak[valid, None] * normal[None, :]
    point, direction, residual = _robust_line(points)
    inliers = residual <= max(1.0, 3.0 * 1.4826 * float(np.median(residual)))
    if inliers.sum() >= 6:
        point, direction, residual = _robust_line(points[inliers])
    else:
        inliers = np.ones(len(points), bool)
    rms = float(np.sqrt(np.mean(residual**2)))
    return SideFit(
        point=point,
        direction=direction,
        support=float(inliers.sum()) / n_samples,
        rms_px=rms,
        contrast=float(np.median(contrast[valid])),
    )


def _robust_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    direction = np.array([vx, vy], np.float64)
    point = np.array([x0, y0], np.float64)
    diff = points - point
    residual = np.abs(diff[:, 0] * direction[1] - diff[:, 1] * direction[0])
    return point, direction, residual


def refine_corners(
    gray: np.ndarray, corners: np.ndarray, search_px: float, cfg: AlignmentConfig
) -> tuple[np.ndarray, tuple[SideFit, ...]] | None:
    """Snap an approximate quad to the label edges at full resolution.

    ``corners`` must be ordered TL, TR, BR, BL. Returns refined corners and the
    per-side fits, or None if any side has no usable edge.
    """
    current = np.asarray(corners, np.float64).copy()
    fits: tuple[SideFit, ...] = ()
    for iteration in range(cfg.edge_refine_iterations):
        search = search_px if iteration == 0 else max(3.0, search_px / 3.0)
        side_fits = []
        for i in range(4):
            fit = _fit_side(gray, current[i], current[(i + 1) % 4], search, cfg)
            if fit is None:
                return None
            side_fits.append(fit)
        new = []
        for i in range(4):
            prev, this = side_fits[i - 1], side_fits[i]
            corner = _line_intersection(prev.point, prev.direction, this.point, this.direction)
            if corner is None:
                return None
            new.append(corner)
        new = np.array(new)
        # Rounded corners legitimately move a lot when the sides are extended to meet;
        # a jump beyond that means a fit locked onto something else.
        shortest_side = min(np.linalg.norm(current[(i + 1) % 4] - current[i]) for i in range(4))
        if np.max(np.linalg.norm(new - current, axis=1)) > max(4.0 * search + 2.0, 0.1 * shortest_side):
            return None
        current = new
        fits = tuple(side_fits)
    if not is_convex_quad(current):
        return None
    return current, fits


def snap_corners_to_edges(bgr: np.ndarray, corners: np.ndarray, cfg: AlignmentConfig) -> np.ndarray | None:
    """Refine user-placed corners to the nearby label edges (used by the manual tool)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ordered = order_corners(corners)
    longest = max(np.linalg.norm(ordered[i] - ordered[(i + 1) % 4]) for i in range(4))
    search = max(6.0, 0.02 * longest)
    result = refine_corners(gray, ordered, search, cfg)
    return None if result is None else result[0]


# ---------------------------------------------------------------- boundary detection


def detect_label_boundary(bgr: np.ndarray, cfg: AlignmentConfig) -> BoundaryDetection | None:
    """Find the label outline; None if no plausible, edge-supported quadrilateral exists."""
    h, w = bgr.shape[:2]
    scale = min(1.0, cfg.detection_max_side_px / max(h, w))
    small = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_small, (0, 0), 1.0)
    gradient = cv2.magnitude(
        cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    )
    small_area = gray_small.size
    scored: list[tuple[float, float, np.ndarray, str]] = []
    for source, quads in (
        ("segmentation", _segmentation_candidates(small, cfg)),
        ("edges", _edge_candidates(gray_small, cfg)),
    ):
        for quad in quads:
            quad = order_corners(quad)
            area_fraction = quad_area(quad) / small_area
            if not cfg.min_label_area_fraction <= area_fraction <= 0.97 or not is_convex_quad(quad):
                continue
            support = _outline_support(gradient, quad, 2.0 * cfg.edge_min_contrast)
            scored.append((support, area_fraction, quad, source))
    if not scored:
        return None
    # Largest well-supported outline: the label itself rather than a printed frame inside it.
    good = [c for c in scored if c[0] >= 0.7]
    pool = good if good else [c for c in scored if c[0] >= 0.4]
    if not pool:
        return None
    support, area_fraction, quad, source = max(pool, key=lambda c: c[1] if good else c[0] * c[1])
    log.debug("Boundary candidate from %s: support %.2f, area %.2f", source, support, area_fraction)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    refined = refine_corners(gray, _scale_points(quad, 1.0 / scale), cfg.edge_search_px / scale, cfg)
    if refined is None:
        return None
    corners, sides = refined
    return BoundaryDetection(corners=corners, sides=sides, source=source)


# ---------------------------------------------------------------- feature matching


def match_artwork(artwork_bgr: np.ndarray, photo_bgr: np.ndarray, cfg: AlignmentConfig) -> FeatureMatch | None:
    """Estimate the artwork -> photo homography from SIFT matches, or None."""
    def prepare(img: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        s = min(1.0, cfg.feature_max_side_px / max(h, w))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if s < 1:
            gray = cv2.resize(gray, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        return gray, s

    art_gray, s_art = prepare(artwork_bgr)
    photo_gray, s_photo = prepare(photo_bgr)
    sift = cv2.SIFT_create(nfeatures=cfg.feature_max_keypoints)
    kp_a, des_a = sift.detectAndCompute(art_gray, None)
    kp_p, des_p = sift.detectAndCompute(photo_gray, None)
    if des_a is None or des_p is None or len(kp_a) < 4 or len(kp_p) < 4:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_a, des_p, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < cfg.feature_ratio_test * p[1].distance]
    if len(good) < cfg.feature_min_inliers:
        return None
    src = _scale_points(np.array([kp_a[m.queryIdx].pt for m in good]), 1.0 / s_art)
    dst = _scale_points(np.array([kp_p[m.trainIdx].pt for m in good]), 1.0 / s_photo)
    threshold = cfg.feature_ransac_threshold_px / s_photo
    homography, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, threshold, maxIters=10000, confidence=0.999)
    if homography is None or mask is None:
        return None
    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    if inliers < cfg.feature_min_inliers:
        return None
    projected = cv2.perspectiveTransform(src[inlier_mask][None], homography)[0]
    rms = float(np.sqrt(np.mean(np.sum((projected - dst[inlier_mask]) ** 2, axis=1))))
    ah, aw = artwork_bgr.shape[:2]
    art_corners = np.array([[-0.5, -0.5], [aw - 0.5, -0.5], [aw - 0.5, ah - 0.5], [-0.5, ah - 0.5]])
    corners = cv2.perspectiveTransform(art_corners[None], homography)[0]
    return FeatureMatch(
        homography=homography, corners=corners, matches=len(good), inliers=inliers, reprojection_rms_px=rms
    )
