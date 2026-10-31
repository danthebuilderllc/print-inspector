"""Perspective correction of the photographed label, with alignment quality checks.

Pipeline (:func:`align_label`)
------------------------------
1. Corners: from manual input, the detected label boundary, or (fallback)
   artwork feature matching. See ``label_detection.py``.
2. Geometry checks: convex quadrilateral, inside the photo, corner angles and
   opposite-side ratios (how steep the camera angle is).
3. Orientation: if the artwork is available, the four possible "which corner is
   top-left" choices are compared with the artwork and the best is kept; the
   margin to the runner-up measures how certain that choice is.
4. Aspect ratio: the label's physical width/height is estimated from the
   corners alone (Zhang & He, "Whiteboard scanning and image enhancement",
   2007; principal point assumed at the image centre) and compared with the
   artwork's aspect ratio.
5. Output: one ``cv2.warpPerspective`` from the *original* photo pixels to a
   canvas with the artwork's aspect ratio (or the estimated one) and enough
   pixels to keep the photo's native sampling along the label's longest sides
   (capped by ``max_output_long_side_px``). Scale is thereby normalised: the
   output is exactly ``scale_to_artwork`` times the artwork size. Later
   refinements must compose their transform with ``homography`` and warp the
   original photo again, so the pixels are interpolated only once.
6. Verification against the artwork (low resolution, structure images):
   global similarity, and local residual misalignment measured by phase
   correlation in a grid of tiles after removing the best global affine fit.

Every finding is a QualityIssue. Any CRITICAL issue makes the alignment
unreliable and recommends REVIEW. This module never recommends PASS.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from print_inspector.config import AlignmentConfig
from print_inspector.core.label_detection import (
    SIDE_NAMES,
    BoundaryDetection,
    FeatureMatch,
    detect_label_boundary,
    is_convex_quad,
    match_artwork,
    order_corners,
    quad_area,
)
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


class AlignmentMethod(str, Enum):
    BOUNDARY = "boundary"
    FEATURES = "features"
    MANUAL = "manual"

    @property
    def label(self) -> str:
        return {
            AlignmentMethod.BOUNDARY: "Automatic (label edges)",
            AlignmentMethod.FEATURES: "Automatic (artwork matching)",
            AlignmentMethod.MANUAL: "Manual corners",
        }[self]


@dataclass(frozen=True)
class AlignmentMetrics:
    method: AlignmentMethod | None = None
    corners: np.ndarray | None = None  # label TL, TR, BR, BL in photo pixels (after orientation)
    edge_support: tuple[float, ...] | None = None  # per side, photo orientation: top, right, bottom, left
    edge_fit_rms_px: float | None = None
    corner_angle_deviation_deg: float | None = None
    side_ratio: float | None = None
    estimated_aspect: float | None = None
    target_aspect: float | None = None
    aspect_error: float | None = None
    quarter_turns: int = 0
    orientation_margin: float | None = None
    feature_inliers: int | None = None
    feature_inlier_ratio: float | None = None
    feature_rms_px: float | None = None
    corner_agreement: float | None = None  # boundary vs. feature corners, share of the label diagonal
    similarity: float | None = None
    residual_fraction: float | None = None  # share of the label's long side
    residual_tiles: int = 0
    artwork_scale: tuple[float, float] | None = None  # leftover x/y scale vs. artwork (e.g. bleed)
    output_size: tuple[int, int] | None = None
    scale_to_artwork: float | None = None


@dataclass(frozen=True, eq=False)
class AlignmentResult:
    metrics: AlignmentMetrics
    issues: tuple[QualityIssue, ...]
    rectified: np.ndarray | None = None  # HxWx3 uint8 BGR, read-only
    valid_mask: np.ndarray | None = None  # HxW uint8, 255 where the output came from inside the photo
    homography: np.ndarray | None = None  # photo pixel coords -> rectified pixel coords

    @property
    def level(self) -> QualityLevel:
        return level_from_issues(self.issues)

    @property
    def is_reliable(self) -> bool:
        return self.rectified is not None and self.level is not QualityLevel.POOR

    @property
    def recommended_status(self) -> InspectionStatus | None:
        return None if self.is_reliable else InspectionStatus.REVIEW

    def issue_codes(self) -> set[str]:
        return {i.code for i in self.issues}


# ---------------------------------------------------------------- geometry


def estimate_aspect_ratio(
    corners: np.ndarray, image_size: tuple[int, int], assumed_focal_factor: float = 0.78
) -> float:
    """Physical width/height of the rectangle imaged at ``corners``; see :func:`aspect_ratio_and_focal`."""
    return aspect_ratio_and_focal(corners, image_size, assumed_focal_factor)[0]


def aspect_ratio_and_focal(
    corners: np.ndarray, image_size: tuple[int, int], assumed_focal_factor: float = 0.78
) -> tuple[float, bool]:
    """Physical width/height of the rectangle imaged at ``corners`` (TL, TR, BR, BL).

    Uses the method of Zhang & He: the focal length is recovered from the two
    vanishing points, assuming square pixels and the principal point at the
    image centre. When a pair of sides is parallel in the photo (a vanishing
    point at infinity), the focal length cannot be recovered and
    ``assumed_focal_factor`` x the photo's long side is used instead, and the
    result is only approximate (about ±10 % for very wide or tele lenses at
    steep angles). Returns (aspect, focal length recovered from the photo).
    """
    w_img, h_img = image_size
    u0, v0 = (w_img - 1) / 2.0, (h_img - 1) / 2.0
    tl, tr, br, bl = (np.array([x, y, 1.0]) for x, y in np.asarray(corners, np.float64))
    m1, m2, m3, m4 = tl, tr, bl, br
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)
    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1
    long_side = max(w_img, h_img)
    focal = assumed_focal_factor * long_side
    recovered = False
    # The vanishing points lie at n[:2] / n[2]. Only trust the recovered focal length when
    # both are within 20 photo widths; farther away it is ill-conditioned.
    scale2 = math.hypot(n2[0], n2[1])
    scale3 = math.hypot(n3[0], n3[1])
    if abs(n2[2]) * 20 * long_side > scale2 and abs(n3[2]) * 20 * long_side > scale3:
        f2 = -(
            (n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * u0 * u0)
            + (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0 * v0)
        ) / (n2[2] * n3[2])
        if (0.2 * long_side) ** 2 <= f2 <= (10.0 * long_side) ** 2:
            focal = math.sqrt(f2)
            recovered = True
    a_inv = np.linalg.inv(np.array([[focal, 0, u0], [0, focal, v0], [0, 0, 1.0]]))
    q = a_inv.T @ a_inv
    aspect = float(math.sqrt((n2 @ q @ n2) / (n3 @ q @ n3)))
    # Without perspective in either direction the focal length does not matter.
    if not recovered and abs(n2[2]) * 20 * long_side <= scale2 and abs(n3[2]) * 20 * long_side <= scale3:
        recovered = True
    return aspect, recovered


def corner_angle_deviation(corners: np.ndarray) -> float:
    """Largest deviation of an interior angle from 90 degrees, in degrees."""
    pts = np.asarray(corners, np.float64)
    worst = 0.0
    for i in range(4):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        worst = max(worst, abs(math.degrees(math.acos(np.clip(cos, -1.0, 1.0))) - 90.0))
    return worst


def side_lengths(corners: np.ndarray) -> np.ndarray:
    """Lengths of the top, right, bottom and left sides."""
    pts = np.asarray(corners, np.float64)
    return np.array([np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)])


def output_size_for(corners: np.ndarray, aspect: float, max_long_side: int) -> tuple[int, int]:
    """Output size with the given aspect that keeps the photo's densest sampling."""
    top, right, bottom, left = side_lengths(corners)
    native_w, native_h = max(top, bottom), max(left, right)
    out_w = max(native_w, native_h * aspect)
    out_h = out_w / aspect
    factor = min(1.0, max_long_side / max(out_w, out_h))
    return max(2, int(round(out_w * factor))), max(2, int(round(out_h * factor)))


def edge_corners(width: int, height: int) -> np.ndarray:
    """Outer edge corners (TL, TR, BR, BL) of a width x height pixel grid."""
    return np.array(
        [[-0.5, -0.5], [width - 0.5, -0.5], [width - 0.5, height - 0.5], [-0.5, height - 0.5]], np.float64
    )


def homography_from_corners(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Exact 4-point homography in double precision (OpenCV's helpers work in float32).

    Points are normalised (Hartley) before solving the 8x8 linear system.
    """
    def normaliser(pts: np.ndarray) -> np.ndarray:
        centre = pts.mean(axis=0)
        spread = np.sqrt(2.0) / max(np.mean(np.linalg.norm(pts - centre, axis=1)), 1e-12)
        return np.array([[spread, 0, -spread * centre[0]], [0, spread, -spread * centre[1]], [0, 0, 1.0]])

    src = np.asarray(src, np.float64).reshape(4, 2)
    dst = np.asarray(dst, np.float64).reshape(4, 2)
    t_src, t_dst = normaliser(src), normaliser(dst)
    s = (np.column_stack([src, np.ones(4)]) @ t_src.T)[:, :2]
    d = (np.column_stack([dst, np.ones(4)]) @ t_dst.T)[:, :2]
    a = np.zeros((8, 8))
    b = np.zeros(8)
    for i, ((x, y), (u, v)) in enumerate(zip(s, d)):
        a[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        a[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
        b[2 * i], b[2 * i + 1] = u, v
    try:
        h = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        raise ValueError("Corners do not define a valid perspective transform") from None
    homography = np.linalg.inv(t_dst) @ np.append(h, 1.0).reshape(3, 3) @ t_src
    return homography / homography[2, 2]


def rectify(
    photo: np.ndarray, corners: np.ndarray, size: tuple[int, int], interpolation: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single perspective warp of the photo. Returns (image, valid mask, homography)."""
    width, height = size
    homography = homography_from_corners(corners, edge_corners(width, height))
    image = cv2.warpPerspective(
        photo, homography, (width, height), flags=_INTERPOLATION[interpolation], borderMode=cv2.BORDER_CONSTANT
    )
    photo_h, photo_w = photo.shape[:2]
    outline = cv2.perspectiveTransform(edge_corners(photo_w, photo_h)[None], homography)[0]
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [np.round(outline).astype(np.int32)], 255)
    image.setflags(write=False)
    mask.setflags(write=False)
    return image, mask, homography


# ---------------------------------------------------------------- verification against the artwork


def _structure(gray: np.ndarray) -> np.ndarray:
    """Edge-strength image used to compare a photo with digital artwork."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def _zncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
    return float(np.sum(a * b) / denom) if denom > 0 else 0.0


@dataclass(frozen=True)
class _Verification:
    quarter_turns: int
    margin: float
    similarity: float
    residual_fraction: float | None
    residual_tiles: int
    artwork_scale: tuple[float, float] | None


def _axis_scale_offset(mapping: np.ndarray, width: int, height: int) -> tuple[float, float, float, float]:
    """Best per-axis fit x' = sx*x + tx, y' = sy*y + ty to a homography over a width x height grid."""
    xs, ys = np.meshgrid(np.linspace(0, width - 1, 5), np.linspace(0, height - 1, 5))
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    mapped = cv2.perspectiveTransform(pts[None].astype(np.float64), mapping)[0]
    sx, tx = np.polyfit(pts[:, 0], mapped[:, 0], 1)
    sy, ty = np.polyfit(pts[:, 1], mapped[:, 1], 1)
    return float(sx), float(tx), float(sy), float(ty)


def _verify_against_artwork(
    photo: np.ndarray,
    corners: np.ndarray,
    artwork: np.ndarray,
    cfg: AlignmentConfig,
    artwork_to_photo: np.ndarray | None = None,
) -> _Verification:
    """Choose the orientation and measure how well the rectified label matches the artwork.

    With ``artwork_to_photo`` (from feature matching), the artwork is first
    scaled/shifted per axis onto the rectified label, so artwork with bleed
    or a slightly different trim is compared like for like.
    """
    ah, aw = artwork.shape[:2]
    s_art = cfg.verify_max_side_px / max(aw, ah)
    vw, vh = max(8, round(aw * s_art)), max(8, round(ah * s_art))
    art_small = cv2.resize(cv2.cvtColor(artwork, cv2.COLOR_BGR2GRAY), (vw, vh), interpolation=cv2.INTER_AREA)
    art_struct = _structure(art_small)
    # verify-grid px -> artwork px (pixel-centre convention)
    small_to_art = np.array(
        [[aw / vw, 0, 0.5 * aw / vw - 0.5], [0, ah / vh, 0.5 * ah / vh - 0.5], [0, 0, 1.0]]
    )

    # Downscale the photo first (area averaging) so the small warp does not alias.
    longest = float(side_lengths(corners).max())
    s_photo = min(1.0, 2.0 * cfg.verify_max_side_px / longest)
    ph, pw = photo.shape[:2]
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    if s_photo < 1.0:
        gray = cv2.resize(gray, (round(pw * s_photo), round(ph * s_photo)), interpolation=cv2.INTER_AREA)
    small_corners = (corners + 0.5) * s_photo - 0.5
    target = edge_corners(vw, vh).astype(np.float32)

    to_small_photo = np.array([[s_photo, 0, 0.5 * s_photo - 0.5], [0, s_photo, 0.5 * s_photo - 0.5], [0, 0, 1.0]])

    scores, structures, references, scales = [], [], [], []
    for k in range(4):
        h_k = cv2.getPerspectiveTransform(np.roll(small_corners, -k, axis=0).astype(np.float32), target)
        warped = cv2.warpPerspective(gray, h_k, (vw, vh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        struct = _structure(warped)
        reference, scale = art_struct, None
        if artwork_to_photo is not None:
            # artwork (verify grid) -> artwork px -> photo px -> small photo px -> verify grid
            mapping = h_k.astype(np.float64) @ to_small_photo @ artwork_to_photo @ small_to_art
            sx, tx, sy, ty = _axis_scale_offset(mapping, vw, vh)
            if sx > 0 and sy > 0:
                reference = cv2.warpAffine(art_struct, np.float32([[sx, 0, tx], [0, sy, ty]]), (vw, vh))
                scale = (sx, sy)
        structures.append(struct)
        references.append(reference)
        scales.append(scale)
        scores.append(_zncc(struct, reference))
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    margin = float(scores[best] - scores[order[1]])

    residual, tiles, residual_scale = _residual_misalignment(references[best], structures[best], cfg)
    scale = scales[best] if scales[best] is not None else residual_scale
    return _Verification(best, margin, float(scores[best]), residual, tiles, scale)


def _residual_misalignment(
    art: np.ndarray, photo: np.ndarray, cfg: AlignmentConfig
) -> tuple[float | None, int, tuple[float, float] | None]:
    h, w = art.shape
    g = cfg.residual_grid_cells
    th, tw = h // g, w // g
    if th < 16 or tw < 16:
        return None, 0, None
    window = cv2.createHanningWindow((tw, th), cv2.CV_32F)
    centres, moved = [], []
    for i in range(g):
        for j in range(g):
            a = np.ascontiguousarray(art[i * th : (i + 1) * th, j * tw : (j + 1) * tw])
            b = np.ascontiguousarray(photo[i * th : (i + 1) * th, j * tw : (j + 1) * tw])
            if a.std() < 1e-3 or b.std() < 1e-3:
                continue
            (dx, dy), response = cv2.phaseCorrelate(a, b, window)
            if response < cfg.residual_min_response or abs(dx) > tw / 4 or abs(dy) > th / 4:
                continue
            c = np.array([(j + 0.5) * tw, (i + 0.5) * th])
            centres.append(c)
            moved.append(c + np.array([dx, dy]))
    if len(centres) < cfg.residual_min_tiles:
        return None, len(centres), None
    # Remove only per-axis scale and offset (what artwork bleed or a slightly
    # different trim produces). Rotation, shear and perspective left over from
    # inaccurate corners stay in the residual.
    c = np.asarray(centres)
    t = np.asarray(moved)
    residual = np.empty_like(t)
    scale = []
    for axis in range(2):
        design = np.column_stack([c[:, axis], np.ones(len(c))])
        coef, *_ = np.linalg.lstsq(design, t[:, axis], rcond=None)
        residual[:, axis] = t[:, axis] - design @ coef
        scale.append(float(coef[0]))
    rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    return rms / max(w, h), len(centres), (scale[0], scale[1])


# ---------------------------------------------------------------- grading helpers


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


def _boundary_issues(boundary: BoundaryDetection, cfg: AlignmentConfig) -> list[QualityIssue]:
    issues = []
    support = [s.support for s in boundary.sides]
    weakest = int(np.argmin(support))
    severity = _grade_low(min(support), cfg.min_edge_support_warning, cfg.min_edge_support_critical)
    if severity:
        issues.append(QualityIssue(
            "edge_support", severity,
            f"The label's {SIDE_NAMES[weakest]} edge (as seen in the photo) was found along only "
            f"{support[weakest]:.0%} of its length. Check the corners, or use Manual Corners.",
        ))
    severity = _grade_high(boundary.max_rms_px, cfg.max_edge_fit_rms_px_warning, cfg.max_edge_fit_rms_px_critical)
    if severity:
        issues.append(QualityIssue(
            "edge_straightness", severity,
            f"The label edges are not straight (deviation {boundary.max_rms_px:.1f} px). The label "
            "may be curved or creased, or the outline is uncertain.",
        ))
    return issues


# ---------------------------------------------------------------- main entry point


def align_label(
    photo: np.ndarray,
    cfg: AlignmentConfig,
    artwork: np.ndarray | None = None,
    manual_corners: np.ndarray | None = None,
) -> AlignmentResult:
    """Locate, rectify and verify the label in ``photo`` (HxWx3 uint8 BGR working image)."""
    if photo.ndim != 3 or photo.shape[2] != 3 or photo.dtype != np.uint8:
        raise ValueError(f"Expected HxWx3 uint8 BGR photo, got {photo.shape} {photo.dtype}")
    ph, pw = photo.shape[:2]
    issues: list[QualityIssue] = []
    m: dict = {}
    boundary: BoundaryDetection | None = None
    feature: FeatureMatch | None = None

    if artwork is not None:
        try:
            feature = match_artwork(artwork, photo, cfg)
        except cv2.error as exc:
            log.warning("Artwork feature matching failed: %s", exc)
    feature_ok = feature is not None and feature.inlier_ratio >= cfg.feature_min_inlier_ratio
    if feature is not None:
        m.update(
            feature_inliers=feature.inliers,
            feature_inlier_ratio=feature.inlier_ratio,
            feature_rms_px=feature.reprojection_rms_px,
        )

    if manual_corners is not None:
        corners = order_corners(manual_corners)
        method = AlignmentMethod.MANUAL
    else:
        boundary = detect_label_boundary(photo, cfg)
        boundary_ok = boundary is not None and not any(
            i.severity is Severity.CRITICAL for i in _boundary_issues(boundary, cfg)
        )
        if boundary_ok or (boundary is not None and not feature_ok):
            corners, method = boundary.corners, AlignmentMethod.BOUNDARY
        elif feature_ok:
            corners, method = order_corners(feature.corners), AlignmentMethod.FEATURES
            if boundary is not None:
                issues.append(QualityIssue(
                    "edges_unclear", Severity.WARNING,
                    "The label edges are unclear, so the label was located by matching the artwork. "
                    "If the artwork includes bleed, the crop follows the artwork rather than the label edge.",
                ))
        else:
            hint = " and the artwork could not be matched to the photo" if artwork is not None else ""
            issues.append(QualityIssue(
                "label_not_found", Severity.CRITICAL,
                f"The label could not be located automatically{hint}. Use Manual Corners to mark its "
                "four corners, or retake the photo against a plain, contrasting background.",
            ))
            return AlignmentResult(metrics=AlignmentMetrics(**m), issues=sort_issues(issues))
        if method is AlignmentMethod.BOUNDARY:
            m.update(edge_support=tuple(s.support for s in boundary.sides), edge_fit_rms_px=boundary.max_rms_px)
            issues.extend(_boundary_issues(boundary, cfg))
            if feature_ok:
                diagonal = float(np.linalg.norm(boundary.corners[2] - boundary.corners[0]))
                distances = [np.min(np.linalg.norm(boundary.corners - c, axis=1)) for c in feature.corners]
                agreement = float(max(distances)) / diagonal
                m["corner_agreement"] = agreement
                if agreement >= cfg.corner_agreement_warning_fraction:
                    issues.append(QualityIssue(
                        "corner_disagreement", Severity.WARNING,
                        f"The label edges and the artwork content disagree on the label position by "
                        f"{agreement:.1%} of its size. The artwork may include bleed, or a different "
                        "outline was detected.",
                    ))
    m["method"] = method

    # ---- geometry
    if not is_convex_quad(corners) or quad_area(corners) < 0.001 * pw * ph:
        issues.append(QualityIssue(
            "invalid_corners", Severity.CRITICAL,
            "The label corners do not form a valid four-sided shape. Mark the corners again.",
        ))
        m["corners"] = corners
        return AlignmentResult(metrics=AlignmentMetrics(**m), issues=sort_issues(issues))

    tolerance = 0.01 * max(pw, ph)
    if np.any(corners[:, 0] < -tolerance) or np.any(corners[:, 0] > pw - 1 + tolerance) or np.any(
        corners[:, 1] < -tolerance
    ) or np.any(corners[:, 1] > ph - 1 + tolerance):
        issues.append(QualityIssue(
            "label_outside_photo", Severity.CRITICAL,
            "Part of the label lies outside the photo. Capture the whole label with a margin around it.",
        ))

    deviation = corner_angle_deviation(corners)
    lengths = side_lengths(corners)
    side_ratio = float(max(lengths[0] / lengths[2], lengths[2] / lengths[0], lengths[1] / lengths[3], lengths[3] / lengths[1]))
    m.update(corner_angle_deviation_deg=deviation, side_ratio=side_ratio)
    severity = _grade_high(
        deviation, cfg.max_corner_angle_deviation_deg_warning, cfg.max_corner_angle_deviation_deg_critical
    )
    if severity:
        issues.append(QualityIssue(
            "steep_angle", severity,
            f"The label corners are skewed by up to {deviation:.0f}°: the photo was taken at a steep "
            "angle or the outline is wrong. Hold the phone parallel to the label.",
        ))
    severity = _grade_high(side_ratio, cfg.max_side_ratio_warning, cfg.max_side_ratio_critical)
    if severity:
        issues.append(QualityIssue(
            "strong_perspective", severity,
            f"Opposite label edges differ in length by a factor of {side_ratio:.2f}: strong perspective. "
            "Resolution varies across the label. Hold the phone parallel to the label.",
        ))

    # ---- orientation and verification against the artwork
    verification = None
    if artwork is not None:
        verification = _verify_against_artwork(
            photo, corners, artwork, cfg, feature.homography if feature_ok else None
        )
        corners = np.roll(corners, -verification.quarter_turns, axis=0)
        m.update(
            quarter_turns=verification.quarter_turns,
            orientation_margin=verification.margin,
            similarity=verification.similarity,
            residual_fraction=verification.residual_fraction,
            residual_tiles=verification.residual_tiles,
            artwork_scale=verification.artwork_scale,
        )
    m["corners"] = corners

    estimated, focal_recovered = aspect_ratio_and_focal(corners, (pw, ph), cfg.assumed_focal_length_factor)
    m["estimated_aspect"] = estimated
    if artwork is not None:
        target = artwork.shape[1] / artwork.shape[0]
        aspect_error = abs(estimated / target - 1.0)
        m.update(target_aspect=target, aspect_error=aspect_error)
        severity = _grade_high(aspect_error, cfg.aspect_tolerance_warning, cfg.aspect_tolerance_critical)
        if severity and not focal_recovered:
            # Tilted in one direction only: the estimate relies on an assumed lens and is
            # approximate, so it can only warn. Content checks below still catch wrong outlines.
            severity = Severity.WARNING
        if severity:
            approximate = "" if focal_recovered else " (approximate: the camera lens had to be assumed)"
            issues.append(QualityIssue(
                "aspect_mismatch", severity,
                f"The detected outline's proportions ({estimated:.3f}{approximate}) differ from the "
                f"artwork's ({target:.3f}) by {aspect_error:.0%}. The wrong outline may have been found, "
                "or the artwork includes bleed.",
            ))
    else:
        target = estimated
        m["target_aspect"] = target
        issues.append(QualityIssue(
            "not_verified", Severity.WARNING,
            "No artwork loaded: label orientation and alignment accuracy cannot be verified, and the "
            "label proportions are estimated from the photo.",
        ))

    if verification is not None:
        severity = _grade_low(verification.similarity, cfg.min_similarity_warning, cfg.min_similarity_critical)
        if severity:
            issues.append(QualityIssue(
                "content_mismatch", severity,
                f"The aligned label matches the artwork poorly (similarity {verification.similarity:.2f}). "
                "The wrong object may have been detected, or the label differs strongly from the artwork.",
            ))
        if verification.artwork_scale is not None:
            deviation = max(abs(s - 1.0) for s in verification.artwork_scale)
            if deviation > cfg.max_artwork_scale_deviation:
                sx, sy = verification.artwork_scale
                issues.append(QualityIssue(
                    "outline_mismatch", Severity.CRITICAL,
                    f"The detected outline and the artwork differ in size by {deviation:.0%} "
                    f"(x {sx:.2f}, y {sy:.2f}), more than bleed or trim would explain. A frame inside the "
                    "label or another object may have been detected. Use Manual Corners.",
                ))
        if verification.margin < cfg.orientation_min_margin:
            issues.append(QualityIssue(
                "orientation_uncertain", Severity.WARNING,
                "The label's orientation (which side is up) could not be determined with confidence.",
            ))
        if verification.residual_fraction is None:
            issues.append(QualityIssue(
                "residual_unverified", Severity.WARNING,
                "Alignment accuracy could not be measured: too few areas could be matched to the artwork.",
            ))
        else:
            severity = _grade_high(
                verification.residual_fraction, cfg.residual_warning_fraction, cfg.residual_critical_fraction
            )
            if severity:
                issues.append(QualityIssue(
                    "residual_distortion", severity,
                    f"After correction the label still deviates locally from the artwork by about "
                    f"{verification.residual_fraction:.1%} of its size. The corners may be inaccurate or "
                    "the label is not flat.",
                ))

    # ---- the single full-resolution warp
    size = output_size_for(corners, target, cfg.max_output_long_side_px)
    rectified, mask, homography = rectify(photo, corners, size, cfg.interpolation)
    m["output_size"] = size
    if artwork is not None:
        m["scale_to_artwork"] = size[0] / artwork.shape[1]

    return AlignmentResult(
        metrics=AlignmentMetrics(**m),
        issues=sort_issues(issues),
        rectified=rectified,
        valid_mask=mask,
        homography=homography,
    )
