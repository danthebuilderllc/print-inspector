"""Colour spaces and colour differences.

All colour comparison is in **CIELAB under D50** (2° observer), the reference
for graphic arts measurement (ISO 13655). The artwork and an uncalibrated
photo are taken to be **sRGB** (the colour space of digital artwork exports
and of phone JPEGs): sRGB values are linearised, converted to XYZ (D65) and
chromatically adapted to D50 with the Bradford transform, as an ICC sRGB
profile does. Differences are **CIEDE2000** (ΔE00), from scikit-image (checked
against the published test data of Sharma, Wu and Dalal, 2005).
"""
from __future__ import annotations

import numpy as np
from skimage.color import deltaE_ciede2000

# sRGB (IEC 61966-2-1) linear RGB -> XYZ (D65)
_SRGB_TO_XYZ_D65 = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_BRADFORD = np.array([
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
])
WHITE_D65 = np.array([0.95047, 1.0, 1.08883])
WHITE_D50 = np.array([0.96422, 1.0, 0.82521])


def _adaptation(source_white: np.ndarray, target_white: np.ndarray) -> np.ndarray:
    cone_src = _BRADFORD @ source_white
    cone_dst = _BRADFORD @ target_white
    return np.linalg.inv(_BRADFORD) @ np.diag(cone_dst / cone_src) @ _BRADFORD


D65_TO_D50 = _adaptation(WHITE_D65, WHITE_D50)
SRGB_TO_XYZ_D50 = D65_TO_D50 @ _SRGB_TO_XYZ_D65
XYZ_D50_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ_D50)


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    """sRGB values 0-255 (any shape) -> linear 0-1 (float64)."""
    v = np.asarray(values, np.float64) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    """Linear 0-1 -> sRGB 0-255 (float64, clipped)."""
    v = np.clip(np.asarray(values, np.float64), 0.0, 1.0)
    return 255.0 * np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1 / 2.4) - 0.055)


def bgr_to_linear_rgb(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 (H x W x 3 or N x 3) -> linear RGB 0-1."""
    return srgb_to_linear(np.asarray(bgr)[..., ::-1])


def linear_rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """Linear sRGB -> XYZ adapted to D50."""
    return np.asarray(rgb, np.float64) @ SRGB_TO_XYZ_D50.T


def xyz_to_lab(xyz: np.ndarray, white: np.ndarray = WHITE_D50) -> np.ndarray:
    t = np.asarray(xyz, np.float64) / white
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(t > eps, np.cbrt(t), (kappa * t + 16) / 116)
    lab = np.empty_like(f)
    lab[..., 0] = 116 * f[..., 1] - 16
    lab[..., 1] = 500 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200 * (f[..., 1] - f[..., 2])
    return lab


def lab_to_xyz(lab: np.ndarray, white: np.ndarray = WHITE_D50) -> np.ndarray:
    lab = np.asarray(lab, np.float64)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    eps, kappa = 216 / 24389, 24389 / 27
    out = np.empty_like(lab)
    out[..., 0] = np.where(fx**3 > eps, fx**3, (116 * fx - 16) / kappa)
    out[..., 1] = np.where(lab[..., 0] > kappa * eps, fy**3, lab[..., 0] / kappa)
    out[..., 2] = np.where(fz**3 > eps, fz**3, (116 * fz - 16) / kappa)
    return out * white


def bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    """sRGB BGR uint8 -> CIELAB D50."""
    return xyz_to_lab(linear_rgb_to_xyz(bgr_to_linear_rgb(bgr)))


def lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    """CIELAB D50 -> sRGB BGR (float 0-255, clipped; out-of-gamut colours are clipped)."""
    rgb = linear_to_srgb(lab_to_xyz(lab) @ XYZ_D50_TO_SRGB.T)
    return rgb[..., ::-1]


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference of two Lab arrays (last axis L, a, b)."""
    return deltaE_ciede2000(np.asarray(lab1, np.float64), np.asarray(lab2, np.float64), channel_axis=-1)


def describe_shift(dl: float, da: float, db: float) -> str:
    """Plain words for a Lab difference (measured minus expected)."""
    parts = []
    if abs(dl) >= 2:
        parts.append("lighter" if dl > 0 else "darker")
    if abs(da) >= 2:
        parts.append("redder" if da > 0 else "greener")
    if abs(db) >= 2:
        parts.append("yellower" if db > 0 else "bluer")
    return ", ".join(parts) or "about the same"
