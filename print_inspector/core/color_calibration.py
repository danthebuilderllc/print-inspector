"""Camera colour calibration with a physical colour reference chart.

A chart with known CIELAB (D50) patch values - e.g. a ColorChecker Classic
24 - is photographed **with the same phone, settings and lighting** as the
labels, ideally in the same photo. The operator marks the four outer corners
of the patch field; the patches are sampled and a mapping from the photo's
linear RGB to XYZ (D50) is fitted.

The mapping is a *root-polynomial* regression of degree 2 (Finlayson et al.,
2015): terms r, g, b, √(rg), √(gb), √(rb). It is exposure invariant (doubling
the light doubles every term) and needs few patches. Its accuracy is
estimated by **leave-one-out** cross-validation: each patch is predicted by
a fit without it. A profile whose leave-one-out mean ΔE00 exceeds the limit
in the settings is not used.

A profile describes one camera, its settings and one lighting. It cannot
check whether a later photo was taken the same way; locked exposure and white
balance (or a chart in every photo) are the operator's responsibility.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from print_inspector.config import user_data_dir
from print_inspector.core.color import bgr_to_linear_rgb, delta_e_2000, lab_to_xyz, xyz_to_lab

PROFILE_VERSION = 1
SAME_PHOTO = "same photo"
SEPARATE_PHOTO = "separate photo"


class CalibrationError(ValueError):
    """The chart could not be measured or the calibration is unusable."""


@dataclass(frozen=True)
class ChartReference:
    name: str
    rows: int
    cols: int
    names: tuple[str, ...]
    lab: tuple[tuple[float, float, float], ...]  # D50, row by row from the top-left patch
    nominal: bool  # generic published values rather than the measured values of this chart

    def lab_array(self) -> np.ndarray:
        return np.array(self.lab, np.float64)


# ColorChecker Classic 24, nominal CIELAB D50 values published by X-Rite for charts made
# from November 2014. Individual charts differ; prefer the data supplied with your chart.
COLORCHECKER_24 = ChartReference(
    name="ColorChecker Classic 24 (nominal values)",
    rows=4, cols=6,
    names=(
        "dark skin", "light skin", "blue sky", "foliage", "blue flower", "bluish green",
        "orange", "purplish blue", "moderate red", "purple", "yellow green", "orange yellow",
        "blue", "green", "red", "yellow", "magenta", "cyan",
        "white 9.5", "neutral 8", "neutral 6.5", "neutral 5", "neutral 3.5", "black 2",
    ),
    lab=(
        (37.54, 14.37, 14.92), (64.66, 19.27, 17.50), (49.32, -3.82, -22.54),
        (43.46, -12.74, 22.72), (54.94, 9.61, -24.79), (70.48, -32.26, -0.37),
        (62.73, 35.83, 56.50), (39.43, 10.75, -45.17), (50.57, 48.64, 16.67),
        (30.10, 22.54, -20.87), (71.77, -24.13, 58.19), (71.51, 18.24, 67.37),
        (28.37, 15.42, -49.80), (54.38, -39.72, 32.27), (42.43, 51.05, 28.62),
        (81.80, 2.67, 80.41), (50.63, 51.28, -14.12), (49.57, -29.71, -28.32),
        (95.19, -1.03, 2.93), (81.29, -0.57, 0.44), (66.89, -0.75, -0.06),
        (50.76, -0.13, 0.14), (35.63, -0.46, -0.48), (20.64, 0.07, -0.46),
    ),
    nominal=True,
)


def load_chart_reference(path: Path) -> ChartReference:
    """A chart description: JSON with name, rows, cols, illuminant "D50" and patches [{name, lab}] row by row."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows, cols = int(data["rows"]), int(data["cols"])
        patches = data["patches"]
        names = tuple(str(p.get("name", f"patch {i + 1}")) for i, p in enumerate(patches))
        lab = tuple(tuple(float(v) for v in p["lab"]) for p in patches)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CalibrationError(f"Chart reference {path.name} cannot be read: {exc}") from exc
    if str(data.get("illuminant", "D50")).upper() != "D50":
        raise CalibrationError("Chart reference values must be CIELAB D50 (as supplied for print measurement).")
    if rows < 2 or cols < 2 or len(lab) != rows * cols or any(len(v) != 3 for v in lab):
        raise CalibrationError(f"Chart reference {path.name} must have rows x cols patches with three Lab values each.")
    if len(lab) < 12:
        raise CalibrationError("A chart needs at least 12 patches for a calibration.")
    return ChartReference(str(data.get("name", path.stem)), rows, cols, names, lab, bool(data.get("nominal", False)))


def patch_centres(corners: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Centres of the patches (row by row) from the corners of the patch field (TL, TR, BR, BL)."""
    tl, tr, br, bl = np.asarray(corners, np.float64).reshape(4, 2)
    out = []
    for r in range(rows):
        v = (r + 0.5) / rows
        left, right = tl + (bl - tl) * v, tr + (br - tr) * v
        for c in range(cols):
            u = (c + 0.5) / cols
            out.append(left + (right - left) * u)
    return np.array(out)


def sample_chart(image_bgr: np.ndarray, corners: np.ndarray, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Median linear RGB of the middle of each patch, and each sample's relative spread."""
    tl, tr, br, bl = np.asarray(corners, np.float64).reshape(4, 2)
    pitch = min(np.linalg.norm(tr - tl) / cols, np.linalg.norm(bl - tl) / rows)
    half = max(1, int(round(pitch * 0.18)))  # sample the middle third of each patch
    h, w = image_bgr.shape[:2]
    rgb, spread = [], []
    for x, y in patch_centres(corners, rows, cols):
        x0, x1 = int(round(x)) - half, int(round(x)) + half + 1
        y0, y1 = int(round(y)) - half, int(round(y)) + half + 1
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            raise CalibrationError("A chart patch lies outside the photo; check the corners.")
        patch = bgr_to_linear_rgb(image_bgr[y0:y1, x0:x1]).reshape(-1, 3)
        median = np.median(patch, axis=0)
        rgb.append(median)
        luminance = patch @ np.array([0.2126, 0.7152, 0.0722])
        spread.append(float(np.std(luminance) / max(float(np.mean(luminance)), 1e-4)))
    return np.array(rgb), np.array(spread)


def _features(rgb: np.ndarray) -> np.ndarray:
    v = np.clip(np.asarray(rgb, np.float64), 0.0, None)
    r, g, b = v[..., 0], v[..., 1], v[..., 2]
    return np.stack([r, g, b, np.sqrt(r * g), np.sqrt(g * b), np.sqrt(r * b)], axis=-1)


def _fit(rgb: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    a = _features(rgb)
    ridge = 1e-6 * np.eye(a.shape[1])
    return np.linalg.solve(a.T @ a + ridge, a.T @ xyz)


@dataclass(frozen=True)
class CalibrationProfile:
    created: str
    chart: str
    chart_nominal: bool
    source: str  # SAME_PHOTO or SEPARATE_PHOTO
    photo: str
    matrix: tuple[tuple[float, float, float], ...]  # 6 x 3: root-polynomial terms -> XYZ D50
    fit_mean_delta_e: float
    fit_max_delta_e: float
    loo_mean_delta_e: float  # leave-one-out: the expected error on colours not in the chart
    loo_max_delta_e: float
    patch_delta_e: tuple[float, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def apply(self, linear_rgb: np.ndarray) -> np.ndarray:
        """Linear camera RGB -> XYZ D50."""
        return _features(linear_rgb) @ np.array(self.matrix)

    def describe(self) -> str:
        nominal = " (nominal chart values)" if self.chart_nominal else ""
        return (
            f"{self.chart}{nominal}, {self.source}, {self.created[:16].replace('T', ' ')}: "
            f"ΔE00 {self.loo_mean_delta_e:.1f} mean / {self.loo_max_delta_e:.1f} max (leave-one-out)"
        )


def calibrate(
    image_bgr: np.ndarray, corners: np.ndarray, chart: ChartReference, source: str, photo_name: str
) -> CalibrationProfile:
    """Measure the chart in the photo and fit the camera mapping. Raises CalibrationError."""
    rgb, spread = sample_chart(image_bgr, corners, chart.rows, chart.cols)
    reference = chart.lab_array()
    if np.median(spread) > 0.12 or np.mean(spread > 0.25) > 0.2:
        raise CalibrationError(
            "The chart patches are not uniform in the photo: the corners are misplaced, or there is glare."
        )
    measured_y = rgb @ np.array([0.2126, 0.7152, 0.0722])
    rank = np.corrcoef(np.argsort(np.argsort(measured_y)), np.argsort(np.argsort(reference[:, 0])))[0, 1]
    if rank < 0.85:
        raise CalibrationError(
            "The measured patches do not match the chart's lightness order: mark the corners starting at the "
            "chart's top-left patch, going clockwise."
        )
    notes = []
    clipped = int(np.sum(rgb.max(axis=1) >= 0.985))
    if clipped:
        notes.append(f"{clipped} patch(es) are at the top of the camera's range (overexposed).")
    xyz = lab_to_xyz(reference)
    matrix = _fit(rgb, xyz)
    fitted = xyz_to_lab(_features(rgb) @ matrix)
    errors = delta_e_2000(fitted, reference)
    loo = np.empty(len(rgb))
    for i in range(len(rgb)):
        others = np.arange(len(rgb)) != i
        m = _fit(rgb[others], xyz[others])
        loo[i] = float(delta_e_2000(xyz_to_lab(_features(rgb[i : i + 1]) @ m), reference[i : i + 1])[0])
    if chart.nominal:
        notes.append("Nominal chart values: the calibration is only as good as the match of your chart to them.")
    return CalibrationProfile(
        created=datetime.now().astimezone().isoformat(timespec="seconds"),
        chart=chart.name, chart_nominal=chart.nominal, source=source, photo=photo_name,
        matrix=tuple(tuple(float(v) for v in row) for row in matrix),
        fit_mean_delta_e=float(errors.mean()), fit_max_delta_e=float(errors.max()),
        loo_mean_delta_e=float(loo.mean()), loo_max_delta_e=float(loo.max()),
        patch_delta_e=tuple(float(v) for v in loo), notes=tuple(notes),
    )


def default_profile_path() -> Path:
    return user_data_dir() / "color_profile.json"


def save_profile(path: Path, profile: CalibrationProfile) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": PROFILE_VERSION, **asdict(profile)}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_profile(path: Path) -> CalibrationProfile | None:
    """The profile in ``path``; None if there is none. Raises CalibrationError if it is unreadable."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != PROFILE_VERSION:
            raise ValueError(f"unsupported version {data.get('version')}")
        matrix = tuple(tuple(float(v) for v in row) for row in data["matrix"])
        if len(matrix) != 6 or any(len(r) != 3 for r in matrix):
            raise ValueError("the matrix must be 6 x 3")
        return CalibrationProfile(
            created=str(data["created"]), chart=str(data["chart"]), chart_nominal=bool(data["chart_nominal"]),
            source=str(data["source"]), photo=str(data["photo"]), matrix=matrix,
            fit_mean_delta_e=float(data["fit_mean_delta_e"]), fit_max_delta_e=float(data["fit_max_delta_e"]),
            loo_mean_delta_e=float(data["loo_mean_delta_e"]), loo_max_delta_e=float(data["loo_max_delta_e"]),
            patch_delta_e=tuple(float(v) for v in data.get("patch_delta_e", ())),
            notes=tuple(str(n) for n in data.get("notes", ())),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CalibrationError(f"Colour profile {path.name} cannot be read: {exc}") from exc


def render_chart(chart: ChartReference, patch_px: int = 60, gap_px: int = 12) -> np.ndarray:
    """An sRGB image of the chart (for tests and as an on-screen reference). BGR uint8."""
    from print_inspector.core.color import lab_to_bgr

    h = chart.rows * patch_px + (chart.rows + 1) * gap_px
    w = chart.cols * patch_px + (chart.cols + 1) * gap_px
    image = np.full((h, w, 3), 30, np.uint8)
    colours = np.clip(np.round(lab_to_bgr(chart.lab_array())), 0, 255).astype(np.uint8)
    for i, colour in enumerate(colours):
        r, c = divmod(i, chart.cols)
        y, x = gap_px + r * (patch_px + gap_px), gap_px + c * (patch_px + gap_px)
        cv2.rectangle(image, (x, y), (x + patch_px - 1, y + patch_px - 1), colour.tolist(), -1)
    return image


def chart_field_corners(chart: ChartReference, patch_px: int = 60, gap_px: int = 12) -> np.ndarray:
    """Corners (TL, TR, BR, BL) of the patch field in ``render_chart``'s image (half a gap outside the patches)."""
    h = chart.rows * patch_px + (chart.rows + 1) * gap_px
    w = chart.cols * patch_px + (chart.cols + 1) * gap_px
    m = gap_px / 2
    return np.array([[m, m], [w - m, m], [w - m, h - m], [m, h - m]], np.float64)
