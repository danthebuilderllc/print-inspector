"""Before/after demonstration of label detection and perspective correction.

Generates synthetic label photos with known rotation, tilt and scale, runs the
application's alignment, measures the geometric error against the ground truth
and writes one comparison image per case:

    [ photo with true (blue) and detected (green/red) corners | aligned label | artwork ]

Usage (from the project root, with the virtual environment active):

    python tools/alignment_demo.py [output_folder]

Runs fully offline. Default output folder: alignment_demo_output
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from print_inspector import synthetic as syn  # noqa: E402
from print_inspector.config import load_config  # noqa: E402
from print_inspector.core.alignment import align_label  # noqa: E402

PHOTO_SIZE = (2400, 1800)
CASES = [
    ("01_frontal", dict()),
    ("02_rotated_25", dict(roll_deg=25)),
    ("03_tilted_30", dict(tilt_x_deg=30, roll_deg=-8)),
    ("04_tilt_and_rotation", dict(roll_deg=-35, tilt_x_deg=20, tilt_y_deg=15)),
    ("05_upside_down", dict(roll_deg=185, tilt_y_deg=12)),
    ("06_quarter_turn", dict(roll_deg=95, tilt_x_deg=15)),
    ("07_small_off_centre", dict(label_fraction=0.3, roll_deg=15, offset=(0.15, -0.12))),
    ("08_close_up", dict(label_fraction=0.85, roll_deg=4, tilt_x_deg=10)),
    ("09_steep_angle", dict(tilt_x_deg=50, roll_deg=10, label_fraction=0.6)),
]
PANEL_HEIGHT = 600


def _fit_height(image: np.ndarray, height: int) -> np.ndarray:
    scale = height / image.shape[0]
    return cv2.resize(image, (max(1, round(image.shape[1] * scale)), height), interpolation=cv2.INTER_AREA)


def _caption(image: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((36, image.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([bar, image])


def _draw_quad(image: np.ndarray, corners: np.ndarray, colour, thickness: int) -> None:
    pts = np.round(corners).astype(np.int32)
    cv2.polylines(image, [pts], True, colour, thickness, cv2.LINE_AA)
    cv2.circle(image, tuple(int(v) for v in pts[0]), thickness * 4, colour, -1, cv2.LINE_AA)


def main(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(user_path=None).alignment
    artwork = syn.make_artwork()
    rows = []
    for name, pose in CASES:
        photo = syn.make_photo(artwork, photo_size=PHOTO_SIZE, **pose)
        start = time.perf_counter()
        result = align_label(photo.image, cfg, artwork)
        seconds = time.perf_counter() - start

        before = photo.image.copy()
        _draw_quad(before, photo.corners, (255, 80, 0), 4)
        if result.metrics.corners is not None:
            colour = (0, 200, 0) if result.is_reliable else (0, 0, 230)
            _draw_quad(before, result.metrics.corners, colour, 2)
        panels = [_caption(_fit_height(before, PANEL_HEIGHT), "Before: photo (blue = true, green = detected)")]
        if result.rectified is not None:
            rms, worst = syn.rectification_error(result.homography, result.metrics.output_size, photo)
            w, h = result.metrics.output_size
            panels.append(_caption(_fit_height(result.rectified, PANEL_HEIGHT), f"After: aligned {w}x{h} px"))
        else:
            rms = worst = float("nan")
        panels.append(_caption(_fit_height(artwork, PANEL_HEIGHT), "Artwork"))
        sheet = np.hstack([np.pad(p, ((0, 0), (0, 10), (0, 0)), constant_values=255) for p in panels])
        ok, encoded = cv2.imencode(".png", sheet)
        if ok:
            encoded.tofile(str(output / f"{name}.png"))

        m = result.metrics
        rows.append((
            name,
            m.method.value if m.method else "-",
            rms,
            worst,
            m.quarter_turns * 90,
            m.similarity,
            m.residual_fraction,
            result.level.value,
            ", ".join(sorted(result.issue_codes())) or "-",
            seconds,
        ))

    header = (
        f"{'case':24s} {'method':9s} {'rms px':>7s} {'max px':>7s} {'turn':>5s} "
        f"{'simil.':>6s} {'resid.':>7s} {'level':9s} {'time':>6s}  issues"
    )
    lines = [
        "Geometric error = distance between where artwork points land in the aligned image",
        "and where they belong, in artwork pixels (artwork 1200 x 800 px).",
        "",
        header,
        "-" * len(header),
    ]
    for name, method, rms, worst, turn, sim, resid, level, issues, seconds in rows:
        sim_s = "-" if sim is None else f"{sim:.2f}"
        resid_s = "-" if resid is None else f"{resid:.2%}"
        lines.append(
            f"{name:24s} {method:9s} {rms:7.2f} {worst:7.2f} {turn:5d} {sim_s:>6s} {resid_s:>7s} "
            f"{level:9s} {seconds:5.2f}s  {issues}"
        )
    report = "\n".join(lines)
    print(report)
    (output / "summary.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\nComparison images written to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("alignment_demo_output")))
