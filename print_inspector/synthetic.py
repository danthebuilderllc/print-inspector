"""Synthetic label artwork and photos with exactly known geometry.

Used by the automated tests and the ``tools/`` demos to measure detection and
registration errors. Not used by the application at runtime.

Photos are made by placing the flat label in 3D (roll, tilt, distance),
projecting it with a pinhole camera whose principal point is the image centre
(like a phone camera), and rendering it over a textured background. Optional
effects: radial lens distortion, ink/tone differences (gain, offset, gamma)
and optical blur. The exact artwork -> photo mapping is kept for ground truth.

Coordinate convention (same as OpenCV): pixel centres are at integer
coordinates, so the outer edge of a W-pixel-wide image runs from -0.5 to W-0.5.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# ---------------------------------------------------------------- artwork


def make_artwork(width: int = 1200, height: int = 800) -> np.ndarray:
    """An asymmetric label design (BGR) with text, blocks, fine lines and a code-like pattern."""
    s = width / 1200.0
    img = np.full((height, width, 3), 255, np.uint8)
    cv2.rectangle(img, (0, 0), (width - 1, int(150 * s)), (150, 60, 20), -1)  # header band
    cv2.putText(img, "PRINT INSPECTOR", (int(40 * s), int(105 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                2.2 * s, (255, 255, 255), max(1, int(6 * s)), cv2.LINE_AA)
    lines = ["LOT 12345-A", "EXP 2027/06", "Net wt. 250 g"]
    for i, text in enumerate(lines):
        cv2.putText(img, text, (int(40 * s), int((260 + i * 110) * s)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6 * s, (20, 20, 20), max(1, int(4 * s)), cv2.LINE_AA)
    cv2.circle(img, (int(980 * s), int(330 * s)), int(110 * s), (40, 40, 210), -1)  # logo
    cv2.circle(img, (int(980 * s), int(330 * s)), int(55 * s), (255, 255, 255), -1)
    rng = np.random.default_rng(7)
    x = int(40 * s)
    while x < int(620 * s):  # barcode-like bars
        bar = int(rng.integers(1, 5) * s) or 1
        cv2.rectangle(img, (x, int(620 * s)), (x + bar - 1, int(760 * s)), (0, 0, 0), -1)
        x += bar + int(rng.integers(1, 4) * s) + 1
    for k in range(5):  # fine lines
        y = int((600 + k * 35) * s)
        cv2.line(img, (int(700 * s), y), (int(1160 * s), y), (0, 0, 0), 1, cv2.LINE_AA)
    return img


def make_small_text_artwork(width: int = 1200, height: int = 800) -> np.ndarray:
    """Dense small print: ingredient-style text 7-12 px high, hairlines and tiny symbols."""
    img = np.full((height, width, 3), 255, np.uint8)
    cv2.rectangle(img, (0, 0), (width - 1, 70), (40, 120, 40), -1)
    cv2.putText(img, "INGREDIENTS / ZUTATEN", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    rng = np.random.default_rng(11)
    words = ["water", "sugar", "E330", "salt", "0.5%", "vit.B12", "(soy)", "Ca", "lot#", "20mg", "keep", "dry"]
    y = 95
    row = 0
    while y < height - 120:
        x = 20
        scale = 0.35 + 0.05 * (row % 3)
        while x < width - 90:
            word = words[int(rng.integers(len(words)))]
            cv2.putText(img, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
            (w, _), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            x += w + int(rng.integers(6, 14))
        y += 18 + 2 * (row % 3)
        row += 1
    for k in range(4):  # hairlines
        cv2.line(img, (20, height - 100 + 18 * k), (width - 20, height - 100 + 18 * k), (0, 0, 0), 1, cv2.LINE_AA)
    for k in range(30):  # tiny recycling-style symbols
        cx = 30 + 38 * k
        cv2.circle(img, (cx, height - 20), 6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.line(img, (cx - 4, height - 20), (cx + 4, height - 20), (0, 0, 0), 1)
    return img


def make_detailed_artwork(width: int = 1200, height: int = 800) -> np.ndarray:
    """Detailed graphics: guilloche curves, a 2D code, gradients and fine hatching."""
    img = np.full((height, width, 3), 250, np.uint8)
    t = np.linspace(0, 2 * np.pi, 4000)
    for k in range(12):  # guilloche rosette
        r = 160 + 40 * np.sin(7 * t + k * 0.5)
        pts = np.column_stack([300 + r * np.cos(t), 330 + r * np.sin(t)]).astype(np.int32)
        cv2.polylines(img, [pts], True, (120 - 8 * k, 40 + 10 * k, 160), 1, cv2.LINE_AA)
    rng = np.random.default_rng(5)
    module = 10  # 2D code-like matrix, 25 x 25 modules
    code = rng.integers(0, 2, (25, 25))
    code[:7, :7] = code[:7, -7:] = code[-7:, :7] = 1
    for i in range(25):
        for j in range(25):
            if code[i, j]:
                cv2.rectangle(img, (650 + j * module, 80 + i * module), (659 + j * module, 89 + i * module), (0, 0, 0), -1)
    for y in range(420, 560):  # colour gradient band
        img[y, 600:1150] = (int(255 * (y - 420) / 140), 120, 255 - int(255 * (y - 420) / 140))
    for k in range(60):  # fine diagonal hatching
        cv2.line(img, (600 + 9 * k, 600), (560 + 9 * k, 760), (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(img, "Batch 0042-B  Made in EU", (40, 740), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def make_low_feature_artwork(width: int = 1200, height: int = 800) -> np.ndarray:
    """Few distinctive features: large flat colour areas, one soft shape, a short word."""
    img = np.full((height, width, 3), (230, 200, 160), np.uint8)
    cv2.rectangle(img, (0, int(height * 0.62)), (width - 1, height - 1), (60, 110, 200), -1)
    cv2.ellipse(img, (int(width * 0.7), int(height * 0.32)), (150, 110), 15, 0, 360, (250, 250, 250), -1, cv2.LINE_AA)
    cv2.putText(img, "PURE", (70, int(height * 0.85)), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (255, 255, 255), 6, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- camera model


@dataclass(frozen=True)
class SyntheticPhoto:
    image: np.ndarray  # BGR uint8 photo
    corners: np.ndarray  # 4x2 photo positions of the label's TL, TR, BR, BL edge corners
    homography: np.ndarray  # artwork pixel coords -> ideal (undistorted) photo pixel coords
    artwork_size: tuple[int, int]  # (width, height)
    distortion_k1: float = 0.0
    focal_px: float = 1.0

    def artwork_to_photo(self, points: np.ndarray) -> np.ndarray:
        """Exact photo position of artwork pixel coordinates (including lens distortion)."""
        ideal = cv2.perspectiveTransform(np.asarray(points, np.float64).reshape(1, -1, 2), self.homography)[0]
        return distort_points(ideal, self.image.shape[1], self.image.shape[0], self.distortion_k1, self.focal_px)


def artwork_edge_corners(width: int, height: int) -> np.ndarray:
    """Outer edge corners of an image, TL, TR, BR, BL, in pixel coordinates."""
    return np.array(
        [[-0.5, -0.5], [width - 0.5, -0.5], [width - 0.5, height - 0.5], [-0.5, height - 0.5]],
        dtype=np.float64,
    )


def distort_points(points: np.ndarray, width: int, height: int, k1: float, focal: float) -> np.ndarray:
    """Radial lens distortion: p_d = c + (p - c) * (1 + k1 * r^2), r normalised by the focal length."""
    pts = np.asarray(points, np.float64)
    if k1 == 0.0:
        return pts.copy()
    c = np.array([(width - 1) / 2.0, (height - 1) / 2.0])
    n = (pts - c) / focal
    r2 = np.sum(n * n, axis=1, keepdims=True)
    return c + n * (1.0 + k1 * r2) * focal


def project_label(
    label_aspect: float,
    photo_size: tuple[int, int],
    *,
    roll_deg: float = 0.0,
    tilt_x_deg: float = 0.0,
    tilt_y_deg: float = 0.0,
    label_fraction: float = 0.6,
    offset: tuple[float, float] = (0.0, 0.0),
    focal_factor: float = 0.8,
) -> np.ndarray:
    """Photo positions of a flat label's corners (TL, TR, BR, BL) seen by a pinhole camera.

    ``label_fraction``: approximate label width as a share of the photo's smaller side.
    ``offset``: label centre shift as a share of the photo size (moved in 3D, so the
    camera model stays consistent).
    """
    pw, ph = photo_size
    f = focal_factor * max(pw, ph)
    w, h = 1.0, 1.0 / label_aspect
    pts = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]])
    r, a, b = (math.radians(v) for v in (roll_deg, tilt_x_deg, tilt_y_deg))
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    rotated = pts @ (ry @ rx @ rz).T
    distance = f * w / (label_fraction * min(pw, ph))
    centre = np.array([offset[0] * pw * distance / f, offset[1] * ph * distance / f, distance])
    cam = rotated + centre
    if np.any(cam[:, 2] <= 0):
        raise ValueError("Label is behind the camera")
    return np.column_stack([f * cam[:, 0] / cam[:, 2] + pw / 2 - 0.5, f * cam[:, 1] / cam[:, 2] + ph / 2 - 0.5])


def render_photo(
    artwork: np.ndarray,
    photo_size: tuple[int, int],
    corners: np.ndarray,
    *,
    background_bgr: tuple[int, int, int] = (95, 105, 115),
    background_texture: float = 6.0,
    lighting_falloff: float = 0.1,
    blur_sigma: float = 0.6,
    noise_sigma: float = 2.0,
    ink_gain: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ink_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gamma: float = 1.0,
    distortion_k1: float = 0.0,
    focal_factor: float = 0.8,
    halftone_period: float = 0.0,
    paper_texture: float = 0.0,
    jpeg_quality: int = 0,
    seed: int = 0,
) -> SyntheticPhoto:
    """Render ``artwork`` onto a textured background with the given (undistorted) corner positions.

    ``ink_gain``/``ink_offset`` alter the printed colours (per BGR channel),
    ``gamma`` applies a camera tone curve, ``distortion_k1`` adds radial lens
    distortion (positive = pincushion, negative = barrel). ``halftone_period``
    (artwork pixels, 0 = off) prints tints as rotated dot screens, like offset
    or flexo printing; ``paper_texture`` adds fibre-like brightness variation
    (gray levels); ``jpeg_quality`` (1-100, 0 = off) applies JPEG compression.
    """
    pw, ph = photo_size
    ah, aw = artwork.shape[:2]
    homography = cv2.getPerspectiveTransform(
        artwork_edge_corners(aw, ah).astype(np.float32), np.asarray(corners, np.float32)
    ).astype(np.float64)
    rng = np.random.default_rng(seed)

    texture = rng.normal(0.0, 1.0, (ph // 8 + 1, pw // 8 + 1)).astype(np.float32)
    texture = cv2.resize(cv2.GaussianBlur(texture, (0, 0), 1.5), (pw, ph), interpolation=cv2.INTER_LINEAR)
    photo = np.empty((ph, pw, 3), np.float32)
    for c in range(3):
        photo[:, :, c] = background_bgr[c] + background_texture * texture

    printed = np.clip(artwork.astype(np.float32) * np.float32(ink_gain) + np.float32(ink_offset), 0, 255)
    print_homography = homography
    if halftone_period:
        k = 4  # render the screen at 4x artwork resolution
        # Nearest-neighbour upscaling: solid ink stays solid, only tints (and anti-aliased edge
        # pixels) are screened - as a RIP does.
        printed = halftone(cv2.resize(printed, (aw * k, ah * k), interpolation=cv2.INTER_NEAREST), halftone_period * k)
        printed = cv2.GaussianBlur(printed, (0, 0), 0.35 * k)  # lens blur before sampling (anti-aliasing)
        to_art = np.array([[1.0 / k, 0, 0.5 / k - 0.5], [0, 1.0 / k, 0.5 / k - 0.5], [0, 0, 1.0]])
        print_homography = homography @ to_art
    if paper_texture:
        grain = rng.normal(0.0, 1.0, printed.shape[:2]).astype(np.float32)
        grain = cv2.GaussianBlur(grain, (0, 0), 1.2)
        grain *= paper_texture / max(float(grain.std()), 1e-6)
        printed = np.clip(printed + grain[:, :, None], 0, 255)
    label = cv2.warpPerspective(printed, print_homography, (pw, ph), flags=cv2.INTER_LINEAR)
    # Anti-aliased coverage mask from a 4x supersampled polygon.
    ss = 4
    mask_big = np.zeros((ph * ss, pw * ss), np.uint8)
    poly = np.round((np.asarray(corners) + 0.5) * ss * 16 - 0.5 * 16).astype(np.int32)
    cv2.fillPoly(mask_big, [poly], 255, lineType=cv2.LINE_8, shift=4)
    mask = cv2.resize(mask_big, (pw, ph), interpolation=cv2.INTER_AREA).astype(np.float32)[:, :, None] / 255.0
    photo = photo * (1.0 - mask) + label * mask

    if lighting_falloff:
        yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
        rr = ((xx - pw / 2) ** 2 + (yy - ph / 2) ** 2) / ((pw / 2) ** 2 + (ph / 2) ** 2)
        photo *= (1.0 - lighting_falloff * rr)[:, :, None]
    if blur_sigma:
        photo = cv2.GaussianBlur(photo, (0, 0), blur_sigma)

    focal = focal_factor * max(pw, ph)
    if distortion_k1:
        # For each distorted output pixel find the ideal pixel (fixed-point inversion of the model).
        yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float64)
        c = np.array([(pw - 1) / 2.0, (ph - 1) / 2.0])
        dx, dy = (xx - c[0]) / focal, (yy - c[1]) / focal
        ux, uy = dx.copy(), dy.copy()
        for _ in range(8):
            factor = 1.0 + distortion_k1 * (ux * ux + uy * uy)
            ux, uy = dx / factor, dy / factor
        photo = cv2.remap(
            photo, (ux * focal + c[0]).astype(np.float32), (uy * focal + c[1]).astype(np.float32),
            cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT,
        )
    if gamma != 1.0:
        photo = 255.0 * np.power(np.clip(photo, 0, 255) / 255.0, gamma)
    if noise_sigma:
        photo += rng.normal(0.0, noise_sigma, photo.shape).astype(np.float32)
    image = np.clip(photo + 0.5, 0, 255).astype(np.uint8)
    if jpeg_quality:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    true_corners = distort_points(np.asarray(corners, np.float64), pw, ph, distortion_k1, focal)
    return SyntheticPhoto(
        image=image, corners=true_corners, homography=homography, artwork_size=(aw, ah),
        distortion_k1=distortion_k1, focal_px=focal,
    )


def make_photo(
    artwork: np.ndarray | None = None,
    photo_size: tuple[int, int] = (2000, 1500),
    seed: int = 0,
    render: dict | None = None,
    **pose,
) -> SyntheticPhoto:
    """Project and render in one call. ``pose`` goes to :func:`project_label`,
    ``render`` to :func:`render_photo`."""
    art = make_artwork() if artwork is None else artwork
    corners = project_label(art.shape[1] / art.shape[0], photo_size, **pose)
    return render_photo(art, photo_size, corners, seed=seed, **(render or {}))


def halftone(image: np.ndarray, period: float) -> np.ndarray:
    """Amplitude-modulated dot screen per channel (screen angles 15/75/45 degrees).

    Each BGR channel's ink coverage (1 - value/255) becomes round dots on a
    rotated grid with the given period (pixels). Solid ink and bare paper stay solid.
    """
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.empty_like(image, dtype=np.float32)
    for c, angle in enumerate((15.0, 75.0, 45.0)):
        a = math.radians(angle)
        u = (xx * math.cos(a) + yy * math.sin(a)) * (2 * math.pi / period)
        v = (-xx * math.sin(a) + yy * math.cos(a)) * (2 * math.pi / period)
        screen = 0.5 + 0.25 * (np.cos(u) + np.cos(v))  # 0..1 dot-shaped threshold surface
        coverage = 1.0 - image[:, :, c].astype(np.float32) / 255.0
        out[:, :, c] = np.where(coverage > screen, 0.0, 255.0)
        out[:, :, c][coverage >= 0.98] = 0.0
        out[:, :, c][coverage <= 0.02] = 255.0
    return out


# ---------------------------------------------------------------- print defects


@dataclass(frozen=True)
class SyntheticDefect:
    kind: str
    bbox: tuple[int, int, int, int]  # x, y, w, h in artwork pixels


def _paper_spot(artwork: np.ndarray, rng: np.random.Generator, clearance: int, taken: list) -> tuple[int, int]:
    """A random paper position at least ``clearance`` px from ink and earlier defects."""
    gray = cv2.cvtColor(artwork, cv2.COLOR_BGR2GRAY)
    paper = (gray >= int(gray.max()) - 15).astype(np.uint8)  # bare stock, white or tinted
    paper[:clearance + 20, :] = paper[-clearance - 20:, :] = 0
    paper[:, :clearance + 20] = paper[:, -clearance - 20:] = 0
    for x, y, w, h in taken:
        cv2.rectangle(paper, (x - clearance, y - clearance), (x + w + clearance, y + h + clearance), 0, -1)
    dist = cv2.distanceTransform(paper, cv2.DIST_L2, 5)
    ys, xs = np.nonzero(dist > clearance)
    if xs.size == 0:
        raise ValueError("No free paper area for a defect")
    i = int(rng.integers(xs.size))
    return int(xs[i]), int(ys[i])


def _ink_spot(artwork: np.ndarray, rng: np.random.Generator, region: tuple[int, int, int, int], dark: int = 100):
    """A random position on dark ink within ``region`` (x, y, w, h)."""
    x0, y0, w, h = region
    gray = cv2.cvtColor(artwork[y0:y0 + h, x0:x0 + w], cv2.COLOR_BGR2GRAY)
    core = cv2.erode((gray < dark).astype(np.uint8), np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(core)
    i = int(rng.integers(xs.size))
    return int(xs[i]) + x0, int(ys[i]) + y0


def add_defects(artwork: np.ndarray, kinds: list[str], seed: int = 0) -> tuple[np.ndarray, list[SyntheticDefect]]:
    """Copy of ``make_artwork()`` output with printing defects, plus their true locations.

    Kinds: missing_line, missing_element, broken_character, extra_mark, unexpected_shape,
    missing_region, distorted_logo, small_dot. Positions refer to the standard artwork layout.
    """
    rng = np.random.default_rng(seed)
    out = artwork.copy()
    defects: list[SyntheticDefect] = []
    taken: list[tuple[int, int, int, int]] = []
    paper = (255, 255, 255)
    for kind in kinds:
        if kind == "missing_line":  # a gap in one of the fine lines
            y = 600 + 35 * int(rng.integers(5))
            x = int(rng.integers(760, 1060))
            cv2.rectangle(out, (x, y - 2), (x + 30, y + 2), paper, -1)
            box = (x, y - 2, 31, 5)
        elif kind == "missing_element":  # the logo's lower part not printed
            cv2.rectangle(out, (900, 380), (1100, 450), paper, -1)
            box = (900, 380, 201, 71)
        elif kind == "broken_character":  # a 5 px gap in a letter stroke
            x, y = _ink_spot(artwork, rng, (40, 200, 520, 280))
            cv2.rectangle(out, (x - 2, y - 2), (x + 2, y + 2), paper, -1)
            box = (x - 2, y - 2, 5, 5)
        elif kind == "extra_mark":  # an ink splash on bare paper
            x, y = _paper_spot(artwork, rng, 30, taken)
            cv2.ellipse(out, (x, y), (9, 5), float(rng.uniform(0, 180)), 0, 360, (40, 40, 40), -1, cv2.LINE_AA)
            box = (x - 9, y - 9, 19, 19)
        elif kind == "unexpected_shape":  # a stray printed triangle
            x, y = _paper_spot(artwork, rng, 30, taken)
            pts = np.array([[x, y - 8], [x + 8, y + 7], [x - 8, y + 7]], np.int32)
            cv2.fillPoly(out, [pts], (150, 60, 20), cv2.LINE_AA)
            box = (x - 8, y - 8, 17, 16)
        elif kind == "missing_region":  # a patch of the header band left unprinted
            x = int(rng.integers(650, 1050))
            cv2.rectangle(out, (x, 20), (x + 60, 70), paper, -1)
            box = (x, 20, 61, 51)
        elif kind == "distorted_logo":  # the logo ring printed 4 px out of shape on one side
            region = out[200:460, 980:1100].copy()
            out[200:460, 984:1100] = region[:, :116]
            box = (980, 210, 120, 240)
        elif kind == "small_dot":  # a tiny ink speck
            x, y = _paper_spot(artwork, rng, 25, taken)
            cv2.circle(out, (x, y), 1, (30, 30, 30), -1, cv2.LINE_AA)
            box = (x - 2, y - 2, 5, 5)
        else:
            raise ValueError(f"Unknown defect kind: {kind}")
        taken.append(box)
        defects.append(SyntheticDefect(kind, box))
    return out, defects


# ---------------------------------------------------------------- error measurement


def rectification_error(
    photo_to_rectified: np.ndarray,
    output_size: tuple[int, int],
    synthetic: SyntheticPhoto,
    grid: int = 21,
) -> tuple[float, float]:
    """(RMS, max) geometric error of a rectification, in artwork pixels.

    A grid of artwork points is mapped artwork -> photo (ground truth) -> rectified
    (estimate) and compared with where those points belong in the rectified image.
    """
    aw, ah = synthetic.artwork_size
    ow, oh = output_size
    xs, ys = np.meshgrid(np.linspace(0, aw - 1, grid), np.linspace(0, ah - 1, grid))
    art_pts = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    photo_pts = synthetic.artwork_to_photo(art_pts)
    rect_pts = cv2.perspectiveTransform(photo_pts[None], photo_to_rectified)[0]
    sx, sy = ow / aw, oh / ah
    expected = np.column_stack([(art_pts[:, 0] + 0.5) * sx - 0.5, (art_pts[:, 1] + 0.5) * sy - 0.5])
    err_art = (rect_pts - expected) / np.array([sx, sy])
    dist = np.linalg.norm(err_art, axis=1)
    return float(np.sqrt(np.mean(dist**2))), float(dist.max())


def registration_error(
    grid_to_photo,
    grid_size: tuple[int, int],
    artwork_size: tuple[int, int],
    artwork_to_photo,
    mask: np.ndarray | None = None,
    step: int = 6,
) -> tuple[float, float, float]:
    """(RMS, 95th percentile, max) registration error in comparison-grid pixels.

    ``grid_to_photo`` is the estimated mapping, ``artwork_to_photo`` the true one.
    Only grid points inside ``mask`` (if given) are measured.
    """
    gw, gh = grid_size
    aw, ah = artwork_size
    ys, xs = np.mgrid[step // 2 : gh : step, step // 2 : gw : step]
    pts = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    if mask is not None:
        pts = pts[mask[pts[:, 1].astype(int), pts[:, 0].astype(int)] > 0]
    sx, sy = gw / aw, gh / ah
    art = np.column_stack([(pts[:, 0] + 0.5) / sx - 0.5, (pts[:, 1] + 0.5) / sy - 0.5])
    truth = artwork_to_photo(art)
    estimate = grid_to_photo(pts)
    # Convert photo-pixel errors to grid pixels with the local scale of the true mapping.
    art_dx = np.column_stack([art[:, 0] + 1.0 / sx, art[:, 1]])
    art_dy = np.column_stack([art[:, 0], art[:, 1] + 1.0 / sy])
    jx = artwork_to_photo(art_dx) - truth
    jy = artwork_to_photo(art_dy) - truth
    err = estimate - truth
    det = jx[:, 0] * jy[:, 1] - jx[:, 1] * jy[:, 0]
    gx = (err[:, 0] * jy[:, 1] - err[:, 1] * jy[:, 0]) / det
    gy = (jx[:, 0] * err[:, 1] - jx[:, 1] * err[:, 0]) / det
    dist = np.hypot(gx, gy)
    return float(np.sqrt(np.mean(dist**2))), float(np.percentile(dist, 95)), float(dist.max())
