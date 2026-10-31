"""Precise registration demonstration with ground-truth error measurement.

For several synthetic labels (standard, small text, detailed graphics,
low-feature artwork, lens distortion, ink/tone differences) this runs label
detection, perspective correction and precise registration, then measures the
true registration error against the known geometry and compares it with what
the application reports about itself.

For each case an image is written showing a zoomed detail with the artwork
edges (magenta) over the photo:
    [ after perspective correction only | after precise registration ]

Usage (from the project root, with the virtual environment active):

    python tools/registration_demo.py [output_folder]

Runs fully offline. Default output folder: registration_demo_output
"""
from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from print_inspector import synthetic as syn  # noqa: E402
from print_inspector.config import load_config  # noqa: E402
from print_inspector.core.alignment import align_label  # noqa: E402
from print_inspector.core.registration import comparison_grid, register_to_artwork  # noqa: E402
from print_inspector.gui.overlay import make_overlay  # noqa: E402

PHOTO_SIZE = (2400, 1800)
POSE = dict(roll_deg=-14, tilt_x_deg=18, tilt_y_deg=8, label_fraction=0.75)
CASES = [
    ("01_standard", syn.make_artwork, {}, None),
    ("02_small_text", syn.make_small_text_artwork, {}, None),
    ("03_detailed_graphics", syn.make_detailed_artwork, {}, None),
    ("04_low_feature", syn.make_low_feature_artwork, {}, None),
    ("05_lens_distortion", syn.make_artwork, dict(distortion_k1=0.03), None),
    ("06_distortion_no_local_fix", syn.make_artwork, dict(distortion_k1=0.03), dict(local_min_rms_px=100.0)),
    ("07_ink_tone_blur", syn.make_detailed_artwork, dict(ink_gain=(0.9, 0.95, 1.05), ink_offset=(10, 5, -5), gamma=1.3, blur_sigma=1.2), None),
    ("08_small_text_distortion", syn.make_small_text_artwork, dict(distortion_k1=0.03), None),
]
CROP = 160  # grid pixels around the detail shown, enlarged 4x


def _edge_mask(reference: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    near = cv2.dilate((grad > 40).astype(np.uint8), np.ones((5, 5), np.uint8)) * 255
    return cv2.bitwise_and(mask, near)


def _detail(image: np.ndarray, centre: tuple[int, int], caption: str) -> np.ndarray:
    x, y = centre
    h, w = image.shape[:2]
    x0, y0 = max(0, min(w - CROP, x - CROP // 2)), max(0, min(h - CROP, y - CROP // 2))
    crop = cv2.resize(image[y0 : y0 + CROP, x0 : x0 + CROP], (CROP * 4, CROP * 4), interpolation=cv2.INTER_NEAREST)
    bar = np.full((34, crop.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, caption, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([bar, crop])


def main(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(user_path=None)
    rows = []
    for name, make_art, render, overrides in CASES:
        artwork = make_art()
        reg_cfg = dataclasses.replace(cfg.registration, **(overrides or {}))
        photo = syn.make_photo(artwork, photo_size=PHOTO_SIZE, render=render, **POSE)
        alignment = align_label(photo.image, cfg.alignment, artwork)
        start = time.perf_counter()
        result = register_to_artwork(photo.image, artwork, alignment, reg_cfg)
        seconds = time.perf_counter() - start
        m = result.metrics
        near = _edge_mask(result.reference, result.mask)
        _, initial = comparison_grid(alignment, artwork.shape, reg_cfg)
        before_map = lambda q: cv2.perspectiveTransform(q[None], initial)[0]  # noqa: E731
        _, before_p95, _ = syn.registration_error(before_map, m.grid_size, photo.artwork_size, photo.artwork_to_photo, near, 3)
        _, true_p95, true_max = syn.registration_error(result.grid_to_photo, m.grid_size, photo.artwork_size, photo.artwork_to_photo, near, 3)

        before_img = cv2.warpPerspective(
            photo.image, initial, m.grid_size, flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP
        )
        gw, gh = m.grid_size
        centre = (int(gw * 0.85), int(gh * 0.85))  # a corner region, where errors are largest
        sheet = np.hstack([
            _detail(make_overlay(result.reference, before_img, None, "edges", 1.0), centre,
                    f"Perspective-corrected only: p95 {before_p95:.2f} px"),
            np.full((CROP * 4 + 34, 12, 3), 255, np.uint8),
            _detail(make_overlay(result.reference, result.registered, result.mask, "edges", 1.0), centre,
                    f"Registered: p95 {true_p95:.2f} px"),
        ])
        ok, encoded = cv2.imencode(".png", sheet)
        if ok:
            encoded.tofile(str(output / f"{name}.png"))
        rows.append((name, before_p95, true_p95, true_max, m.residual_p95_px, m.coverage, m.local_correction_applied,
                     result.level.value, ", ".join(sorted(result.issue_codes())) or "-", seconds))

    header = (
        f"{'case':28s} {'start':>6s} {'true95':>7s} {'truemax':>7s} {'meas95':>7s} {'cover':>6s} "
        f"{'local':>5s} {'level':9s} {'time':>5s}  issues"
    )
    lines = [
        "Errors in comparison-grid pixels, measured near artwork edges (where misalignment causes",
        "false defects). start = after perspective correction only; true = after registration",
        "(ground truth); meas = what the application reports about itself.",
        "",
        header,
        "-" * len(header),
    ]
    for name, start, true95, truemax, meas, cover, local, level, issues, seconds in rows:
        meas_s = "-" if meas is None else f"{meas:.2f}"
        cover_s = "-" if cover is None else f"{cover:.0%}"
        lines.append(
            f"{name:28s} {start:6.2f} {true95:7.2f} {truemax:7.2f} {meas_s:>7s} {cover_s:>6s} "
            f"{'yes' if local else 'no':>5s} {level:9s} {seconds:4.1f}s  {issues}"
        )
    report = "\n".join(lines)
    print(report)
    (output / "summary.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\nDetail images written to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("registration_demo_output")))
