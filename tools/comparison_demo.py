"""Detection performance of the comparison engine on synthetic print defects.

For several photo conditions (clean, noise + JPEG, halftone screening, tone /
lighting differences, paper texture) and several random seeds this renders

* a defect-free print  -> every candidate is a false positive, and
* a print with eight kinds of planted defects -> per-kind recall,

runs the full pipeline (label detection, perspective correction, registration,
comparison) and reports detection rates. A dot-size sweep estimates the
smallest detectable ink speck. Comparison images for one case are written
to the output folder.

Usage (from the project root, with the virtual environment active):

    python tools/comparison_demo.py [output_folder] [--quick]

Runs fully offline. Default output folder: comparison_demo_output
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from print_inspector import synthetic as syn  # noqa: E402
from print_inspector.config import load_config  # noqa: E402
from print_inspector.core.alignment import align_label  # noqa: E402
from print_inspector.core.comparison import compare_to_artwork  # noqa: E402
from print_inspector.core.decision import decide  # noqa: E402
from print_inspector.core.defects import assess_defects, grid_px_per_mm  # noqa: E402
from print_inspector.core.registration import register_to_artwork  # noqa: E402

KINDS = [
    "missing_line", "missing_element", "broken_character", "extra_mark",
    "unexpected_shape", "missing_region", "distorted_logo", "small_dot",
]
CONDITIONS = {
    "clean": dict(),
    "noise + JPEG": dict(noise_sigma=3.0, jpeg_quality=88),
    "halftone": dict(halftone_period=3.0, noise_sigma=2.0, jpeg_quality=92),
    "tone + lighting": dict(gamma=1.3, ink_gain=(0.9, 0.95, 1.05), ink_offset=(8, 4, -4), lighting_falloff=0.25, blur_sigma=1.0),
    "paper texture": dict(paper_texture=4.0, noise_sigma=2.5, jpeg_quality=90),
}
POSE = dict(roll_deg=-12, tilt_x_deg=15, tilt_y_deg=6, label_fraction=0.75)
PHOTO_SIZE = (2400, 1800)


def run_pipeline(printed: np.ndarray, artwork: np.ndarray, render: dict, seed: int, cfg):
    photo = syn.make_photo(printed, photo_size=PHOTO_SIZE, seed=seed, render=render, **POSE)
    alignment = align_label(photo.image, cfg.alignment, artwork)
    registration = register_to_artwork(photo.image, artwork, alignment, cfg.registration)
    comparison = compare_to_artwork(
        registration.registered, registration.reference, registration.mask, cfg.comparison,
        registration.metrics.blur_match_sigma_px, registration.metrics.residual_p95_px,
    )
    # The 1200 px artwork is treated as a 100 mm label (no DPI in the generated artwork).
    px_per_mm, _ = grid_px_per_mm(registration.metrics.grid_scale, None, None, artwork.shape[1], 100.0)
    report = assess_defects(comparison, registration.reference, registration.registered, px_per_mm, True, cfg.defects)
    # Capture quality is left out: the generated label stock is pure white (blown out by design).
    verdict = decide(None, alignment, registration, comparison, report, {}, cfg.decision)
    return registration, comparison, report, verdict


def overlaps(candidate, box, scale: float, pad: float = 4.0) -> bool:
    x, y, w, h = (v * scale for v in box)
    return not (
        candidate.x > x + w + pad or candidate.x + candidate.width < x - pad
        or candidate.y > y + h + pad or candidate.y + candidate.height < y - pad
    )


def draw_dot(image: np.ndarray, centre: tuple[int, int], diameter: float, value: int) -> None:
    """Anti-aliased dot of fractional diameter (supersampled)."""
    ss = 8
    r = int(np.ceil(diameter)) + 2
    x, y = centre
    patch = image[y - r : y + r + 1, x - r : x + r + 1]
    big = np.zeros(((2 * r + 1) * ss, (2 * r + 1) * ss), np.uint8)
    cv2.circle(big, ((r * ss) + ss // 2, (r * ss) + ss // 2), int(diameter * ss / 2), 255, -1)
    cover = cv2.resize(big, (2 * r + 1, 2 * r + 1), interpolation=cv2.INTER_AREA).astype(np.float32)[:, :, None] / 255
    patch[:] = (patch * (1 - cover) + value * cover).astype(np.uint8)


def main(output: Path, quick: bool) -> int:
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(user_path=None)
    artwork = syn.make_artwork()
    seeds = [0] if quick else [0, 1, 2]
    hits: dict[str, list[int]] = defaultdict(list)
    kinds_found: dict[str, list[str]] = defaultdict(list)
    false_good: dict[str, list[int]] = defaultdict(list)
    false_defective: dict[str, list[int]] = defaultdict(list)
    verdicts: dict[str, list[str]] = defaultdict(list)
    started = time.perf_counter()
    for cond, render in CONDITIONS.items():
        for seed in seeds:
            _, good, _, good_verdict = run_pipeline(artwork, artwork, render, seed, cfg)
            false_good[cond].append(len(good.candidates))
            verdicts["good"].append(good_verdict.status.value)
            printed, defects = syn.add_defects(artwork, KINDS, seed=seed + 10)
            registration, result, report, verdict = run_pipeline(printed, artwork, render, seed, cfg)
            verdicts["defective"].append(verdict.status.value)
            for kind in KINDS:  # one print per defect kind: is the verdict FAIL?
                single, _ = syn.add_defects(artwork, [kind], seed=seed + 20)
                verdicts[kind].append(run_pipeline(single, artwork, render, seed, cfg)[3].status.value)
            scale = registration.metrics.grid_scale
            for d in defects:
                matched = [c for c in result.candidates if overlaps(c, d.bbox, scale)]
                hits[d.kind].append(1 if matched else 0)
                if matched:
                    kinds_found[d.kind].append(matched[0].kind)
            false_defective[cond].append(
                sum(1 for c in result.candidates if not any(overlaps(c, d.bbox, scale) for d in defects))
            )
            if cond == "halftone" and seed == seeds[0]:
                save_visuals(output, registration, result)

    lines = ["Detection performance on synthetic prints (full pipeline)", ""]
    lines.append(f"{'condition':18s} {'false (good print)':>20s} {'false (defective)':>19s}")
    for cond in CONDITIONS:
        lines.append(f"{cond:18s} {sum(false_good[cond]):>12d} / {len(seeds)} prints {sum(false_defective[cond]):>8d} / {len(seeds)} prints")
    lines += ["", f"{'defect kind':18s} {'detected':>10s}   most common reported kind"]
    for kind in KINDS:
        found = hits[kind]
        common = max(set(kinds_found[kind]), key=kinds_found[kind].count) if kinds_found[kind] else "-"
        lines.append(f"{kind:18s} {sum(found):>4d} / {len(found):<4d}   {common}")

    lines += ["", "Verdicts (label = 100 mm; capture-quality check left out, see source)",
              f"{'print':18s} {'PASS':>5s} {'FAIL':>5s} {'REVIEW':>7s}"]
    for name in ["good", "defective"] + KINDS:
        v = verdicts[name]
        lines.append(f"{name:18s} {v.count('pass'):>5d} {v.count('fail'):>5d} {v.count('review'):>7d}")

    # ---- smallest detectable dot (on bare paper, clean and noise + JPEG conditions)
    lines += ["", "Small dot detection (dots on bare paper; label 1200 artwork px = 100 mm assumed)"]
    lines.append(f"{'diameter':>16s} {'dark ink':>14s} {'mid gray':>14s}")
    rng = np.random.default_rng(5)
    for diameter in (1.0, 1.5, 2.0, 3.0):
        row = []
        for value in (30, 150):
            detected = total = 0
            for seed in seeds:
                printed = artwork.copy()
                spots = []
                for _ in range(6):
                    x, y = syn._paper_spot(artwork, rng, 25, spots)
                    draw_dot(printed, (x, y), diameter, value)
                    spots.append((x - 3, y - 3, 7, 7))
                registration, result, _, _ = run_pipeline(printed, artwork, CONDITIONS["noise + JPEG"], seed, cfg)
                scale = registration.metrics.grid_scale
                detected += sum(1 for box in spots if any(overlaps(c, box, scale, 2) for c in result.candidates))
                total += len(spots)
            row.append(f"{detected}/{total}")
        lines.append(f"{diameter:5.1f} px ({diameter / 12:.2f} mm) {row[0]:>14s} {row[1]:>14s}")
    lines += ["", f"Total time {time.perf_counter() - started:.0f} s"]
    report = "\n".join(lines)
    print(report)
    (output / "summary.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\nImages written to {output.resolve()}")
    return 0


def save_visuals(output: Path, registration, result) -> None:
    def colour(map_u8: np.ndarray, gain: float) -> np.ndarray:
        return cv2.applyColorMap(np.clip(map_u8.astype(np.float32) * gain, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    boxes = registration.registered.copy()
    for c in result.candidates:
        cv2.rectangle(boxes, (c.x - 3, c.y - 3), (c.x + c.width + 3, c.y + c.height + 3), (0, 0, 255), 2)
        cv2.putText(boxes, str(c.id), (c.x, max(12, c.y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    edges = np.full_like(registration.registered, 255)
    edges[result.edge_difference == 1] = (0, 0, 255)
    edges[result.edge_difference == 2] = (255, 0, 0)
    images = {
        "1_candidates.png": boxes,
        "2_expected.png": result.expected,
        "3_gray_difference.png": colour(result.gray_difference, 4.0),
        "4_structural_difference.png": colour(result.structural_difference, 1.0),
        "5_edge_difference.png": edges,
        "6_combined_score.png": colour(np.clip(result.combined_score * 60, 0, 255), 1.0),
    }
    for name, image in images.items():
        ok, encoded = cv2.imencode(".png", image)
        if ok:
            encoded.tofile(str(output / name))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    raise SystemExit(main(Path(args[0]) if args else Path("comparison_demo_output"), "--quick" in sys.argv))
