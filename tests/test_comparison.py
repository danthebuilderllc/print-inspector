"""Comparison engine on known-good and deliberately modified print/artwork pairs.

Most tests simulate the *registered* photo directly on the artwork grid (fast,
exact ground truth): the printed artwork goes through camera blur, tone curve,
ink changes, noise, lighting, halftone screening, JPEG and optional residual
misregistration. A few tests run the full pipeline from a perspective photo.
"""
import dataclasses

import cv2
import numpy as np
import pytest

from print_inspector import synthetic as syn
from print_inspector.core.comparison import (
    EDGE_EXTRA,
    EDGE_MISSING,
    KIND_LABELS,
    compare_to_artwork,
)
from print_inspector.core.models import InspectionStatus, Severity

ARTWORK = syn.make_artwork()
KINDS = [
    "missing_line", "missing_element", "broken_character", "extra_mark",
    "unexpected_shape", "missing_region", "distorted_logo", "small_dot",
]


@pytest.fixture(scope="module")
def cfg():
    from print_inspector.config import load_config

    return load_config(user_path=None).comparison


def simulate(
    printed: np.ndarray,
    *,
    blur: float = 0.9,
    gamma: float = 1.0,
    ink_gain=(1.0, 1.0, 1.0),
    ink_offset=(0.0, 0.0, 0.0),
    noise: float = 2.0,
    lighting: float = 0.0,
    shift=(0.0, 0.0),
    halftone: float = 0.0,
    jpeg: int = 0,
    seed: int = 0,
) -> np.ndarray:
    """The printed artwork as the registration would deliver it (same grid as the artwork)."""
    rng = np.random.default_rng(seed)
    img = np.clip(printed.astype(np.float32) * np.float32(ink_gain) + np.float32(ink_offset), 0, 255)
    h, w = img.shape[:2]
    if halftone:
        k = 4
        big = syn.halftone(cv2.resize(img, (w * k, h * k), interpolation=cv2.INTER_NEAREST), halftone * k)
        img = cv2.resize(cv2.GaussianBlur(big, (0, 0), 0.35 * k), (w, h), interpolation=cv2.INTER_AREA)
    if shift != (0.0, 0.0):
        img = cv2.warpAffine(img, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]), (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    img = cv2.GaussianBlur(img, (0, 0), blur)
    if lighting:
        ramp = np.linspace(1.0, 1.0 - lighting, w, dtype=np.float32)[None, :, None]
        img *= ramp
    if gamma != 1.0:
        img = 255.0 * np.power(np.clip(img, 0, 255) / 255.0, gamma)
    img += rng.normal(0, noise, img.shape).astype(np.float32)
    out = np.clip(img + 0.5, 0, 255).astype(np.uint8)
    if jpeg:
        out = cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, jpeg])[1], cv2.IMREAD_COLOR)
    return out


def full_mask(shape, border=3):
    mask = np.zeros(shape[:2], np.uint8)
    mask[border:-border, border:-border] = 255
    return mask


def run(cfg, registered, reference=ARTWORK, blur=0.9, **kwargs):
    return compare_to_artwork(registered, reference, full_mask(reference.shape), cfg, blur_sigma_px=blur, **kwargs)


def overlaps(candidate, box, pad=3):
    x, y, w, h = box
    return not (
        candidate.x > x + w + pad or candidate.x + candidate.width < x - pad
        or candidate.y > y + h + pad or candidate.y + candidate.height < y - pad
    )


GOOD_CONDITIONS = {
    "clean": dict(),
    "noise + JPEG": dict(noise=3.0, jpeg=85),
    "tone curve and ink": dict(gamma=1.35, ink_gain=(0.9, 0.95, 1.05), ink_offset=(8, 4, -4)),
    "uneven lighting": dict(lighting=0.3),
    "halftone": dict(halftone=3.0, jpeg=92),
    "residual misregistration 0.2 px": dict(shift=(0.2, -0.13)),
    "softer photo": dict(blur=1.4),
}


# ---------------------------------------------------------------- known-good pairs


@pytest.mark.parametrize("condition", list(GOOD_CONDITIONS), ids=list(GOOD_CONDITIONS))
def test_good_print_has_no_candidates(cfg, condition):
    params = GOOD_CONDITIONS[condition]
    result = run(cfg, simulate(ARTWORK, **params), blur=params.get("blur", 0.9))
    assert result.candidates == (), [(c.kind, round(c.score, 2), c.x, c.y) for c in result.candidates]
    assert result.is_reliable
    assert result.recommended_status is None


def test_misregistration_tolerance_at_pipeline_resolution(cfg):
    """With the photo's typical sampling (grid ~1.2x artwork pixels) 0.3 px residual is tolerated."""
    finer = cv2.resize(ARTWORK, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
    registered = simulate(finer, shift=(0.3, -0.2))
    result = compare_to_artwork(registered, finer, full_mask(finer.shape), cfg, 0.9, alignment_residual_px=0.36)
    assert result.candidates == ()


def test_pixel_subtraction_alone_would_flag_these(cfg):
    """Lighting and tone differences are large in raw pixel terms, yet not defects."""
    registered = simulate(ARTWORK, lighting=0.3, gamma=1.35)
    raw = np.abs(registered.astype(int) - ARTWORK.astype(int)).max(axis=2)
    assert (raw > 30).mean() > 0.3  # naive subtraction: a third of the label "differs"
    assert run(cfg, registered).candidates == ()


# ---------------------------------------------------------------- deliberately modified pairs


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("condition", ["clean", "noise + JPEG", "halftone"])
def test_each_defect_kind_is_found(cfg, kind, condition):
    printed, defects = syn.add_defects(ARTWORK, [kind], seed=4)
    result = run(cfg, simulate(printed, **GOOD_CONDITIONS[condition]))
    box = defects[0].bbox
    assert any(overlaps(c, box) for c in result.candidates), f"{kind} at {box} not found"
    assert all(overlaps(c, box, pad=40) for c in result.candidates), "false candidates elsewhere"


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("missing_line", {"missing_line"}),
        ("broken_character", {"broken_character"}),
        ("extra_mark", {"extra_ink_mark", "small_dot"}),
        ("small_dot", {"small_dot"}),
        ("missing_region", {"missing_region", "missing_element"}),
        ("missing_element", {"missing_region", "missing_element"}),
    ],
)
def test_preliminary_kind(cfg, kind, expected):
    printed, defects = syn.add_defects(ARTWORK, [kind], seed=4)
    result = run(cfg, simulate(printed))
    hits = [c for c in result.candidates if overlaps(c, defects[0].bbox)]
    assert hits and hits[0].kind in expected, [c.kind for c in hits]
    assert hits[0].label == KIND_LABELS[hits[0].kind]


def test_all_defects_together_with_locations_and_scores(cfg):
    printed, defects = syn.add_defects(ARTWORK, KINDS, seed=7)
    result = run(cfg, simulate(printed, noise=2.5, jpeg=90))
    for d in defects:
        assert any(overlaps(c, d.bbox) for c in result.candidates), d.kind
    for c in result.candidates:
        assert c.score >= 1.0 and c.peak_sigma > 0 and c.max_difference >= cfg.min_abs_difference
        assert c.area_px > 0 and c.width > 0 and c.height > 0
        assert c.polarity in ("darker", "lighter", "mixed")
        assert 0.0 <= c.min_ssim <= 1.0
    assert [c.id for c in result.candidates] == list(range(1, len(result.candidates) + 1))
    scores = [c.score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)


def test_polarity_and_ink(cfg):
    printed, defects = syn.add_defects(ARTWORK, ["extra_mark", "missing_region"], seed=2)
    result = run(cfg, simulate(printed))
    extra = next(c for c in result.candidates if overlaps(c, defects[0].bbox))
    missing = next(c for c in result.candidates if overlaps(c, defects[1].bbox))
    assert extra.polarity == "darker" and extra.ink_fraction < 0.4 and extra.mean_difference < 0
    assert missing.polarity == "lighter" and missing.ink_fraction > 0.6 and missing.mean_difference > 0


@pytest.mark.parametrize("diameter_px, value", [(1.5, 30), (2.0, 140)])
def test_tiny_dots_are_not_smoothed_away(cfg, diameter_px, value):
    printed = ARTWORK.copy()
    big = cv2.resize(printed, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    cv2.circle(big, (400 * 8 + 4, 520 * 8 + 4), int(diameter_px * 4), (value,) * 3, -1)
    printed = cv2.resize(big, (ARTWORK.shape[1], ARTWORK.shape[0]), interpolation=cv2.INTER_AREA)
    result = run(cfg, simulate(printed, noise=2.0))
    assert any(overlaps(c, (398, 518, 5, 5)) for c in result.candidates)


def test_large_misregistration_creates_differences(cfg):
    """Registration would flag this; the comparison itself does not hide it either."""
    result = run(cfg, simulate(ARTWORK, shift=(2.0, 0.0)))
    assert len(result.candidates) > 0


def test_sensitivity_changes_thresholds(cfg):
    printed = ARTWORK.copy()
    cv2.circle(printed, (450, 530), 6, (243, 243, 243), -1)  # very faint gray mark on paper
    registered = simulate(printed, noise=2.0)
    low = run(cfg, registered, sensitivity="low")
    high = run(cfg, registered, sensitivity="high")
    assert high.metrics.threshold_factor < 1.0 < low.metrics.threshold_factor
    assert any(overlaps(c, (444, 524, 13, 13)) for c in high.candidates)
    assert not any(overlaps(c, (444, 524, 13, 13)) for c in low.candidates)


def test_wrong_artwork_gives_review(cfg):
    """Every pixel differs, so the self-calibrating noise model would absorb it: the self-check catches it."""
    other = syn.make_detailed_artwork()
    result = run(cfg, simulate(ARTWORK), reference=other)
    assert result.metrics.model_similarity < cfg.min_model_similarity_critical
    assert any(i.code == "model_mismatch" and i.severity is Severity.CRITICAL for i in result.issues)
    assert result.recommended_status is InspectionStatus.REVIEW


def test_model_similarity_high_for_real_prints(cfg):
    printed, _ = syn.add_defects(ARTWORK, KINDS, seed=7)
    for registered in (simulate(ARTWORK, halftone=3.0), simulate(printed)):
        result = run(cfg, registered)
        assert result.metrics.model_similarity > cfg.min_model_similarity_warning
        assert "model_mismatch" not in {i.code for i in result.issues}


# ---------------------------------------------------------------- outputs, tiling, inputs


def test_output_maps(cfg):
    printed, _ = syn.add_defects(ARTWORK, ["missing_line", "extra_mark"], seed=1)
    result = run(cfg, simulate(printed))
    h, w = ARTWORK.shape[:2]
    for name in ("gray_difference", "structural_difference", "edge_difference", "candidate_mask"):
        array = getattr(result, name)
        assert array.shape == (h, w) and array.dtype == np.uint8
        with pytest.raises(ValueError):
            array.flat[0] = 1
    assert result.combined_score.shape == (h, w) and result.combined_score.dtype == np.float32
    assert result.expected.shape == ARTWORK.shape
    assert set(np.unique(result.edge_difference)) <= {0, EDGE_MISSING, EDGE_EXTRA}
    assert result.combined_score[:3].max() == 0  # outside the mask
    assert result.candidate_mask.max() == 255
    assert result.metrics.candidate_count == len(result.candidates)


def test_tiled_processing_matches_single_tile(cfg):
    printed, _ = syn.add_defects(ARTWORK, ["small_dot", "missing_line", "extra_mark"], seed=3)
    registered = simulate(printed, noise=2.0)
    single = run(dataclasses.replace(cfg, tile_size_px=2048), registered)
    tiled = run(dataclasses.replace(cfg, tile_size_px=256), registered)
    assert tiled.metrics.tiles > single.metrics.tiles
    assert np.allclose(tiled.combined_score, single.combined_score, atol=1e-3)
    assert [(c.x, c.y, c.area_px) for c in tiled.candidates] == [(c.x, c.y, c.area_px) for c in single.candidates]


def test_photometric_model_reproduces_the_print(cfg):
    registered = simulate(ARTWORK, gamma=1.3, ink_gain=(0.9, 0.95, 1.05), lighting=0.2, noise=0.0)
    result = run(cfg, registered)
    inside = full_mask(ARTWORK.shape, 10) > 0
    error = np.abs(result.expected.astype(int) - registered.astype(int)).max(axis=2)[inside]
    assert np.median(error) <= 2 and np.percentile(error, 99) < 25


def test_rejects_bad_input(cfg):
    with pytest.raises(ValueError):
        compare_to_artwork(ARTWORK[:, :, 0], ARTWORK, full_mask(ARTWORK.shape), cfg)
    with pytest.raises(ValueError):
        compare_to_artwork(ARTWORK[:100], ARTWORK, full_mask(ARTWORK.shape), cfg)
    with pytest.raises(ValueError):
        compare_to_artwork(ARTWORK, ARTWORK, np.zeros(ARTWORK.shape[:2], np.uint8), cfg)


# ---------------------------------------------------------------- full pipeline


@pytest.fixture(scope="module")
def pipeline():
    from print_inspector.config import load_config
    from print_inspector.core.alignment import align_label
    from print_inspector.core.registration import register_to_artwork

    config = load_config(user_path=None)

    def go(printed, render):
        photo = syn.make_photo(printed, photo_size=(2400, 1800), render=render,
                               roll_deg=-12, tilt_x_deg=15, tilt_y_deg=6, label_fraction=0.75)
        alignment = align_label(photo.image, config.alignment, ARTWORK)
        registration = register_to_artwork(photo.image, ARTWORK, alignment, config.registration)
        result = compare_to_artwork(
            registration.registered, registration.reference, registration.mask, config.comparison,
            registration.metrics.blur_match_sigma_px, registration.metrics.residual_p95_px,
        )
        return registration, result

    return go


@pytest.mark.parametrize(
    "render", [dict(noise_sigma=3.0, jpeg_quality=88), dict(halftone_period=3.0, jpeg_quality=92)],
    ids=["noise + JPEG", "halftone"],
)
def test_full_pipeline_detection_performance(pipeline, render):
    _, good = pipeline(ARTWORK, render)
    assert len(good.candidates) == 0
    printed, defects = syn.add_defects(ARTWORK, KINDS, seed=11)
    registration, result = pipeline(printed, render)
    s = registration.metrics.grid_scale
    for d in defects:
        box = tuple(v * s for v in d.bbox)
        assert any(overlaps(c, box, pad=4) for c in result.candidates), d.kind
    false = [c for c in result.candidates if not any(overlaps(c, tuple(v * s for v in d.bbox), 4) for d in defects)]
    assert len(false) <= 1
    assert registration.is_reliable  # defects must not disturb the registration


def test_tone_curve_ignores_a_level_dominated_by_a_defect():
    """A sparse level whose pixels are mostly in a defect must not lift all lighter levels."""
    from print_inspector.core.comparison import _fit_tone_curve

    rng = np.random.default_rng(0)
    model = np.concatenate([np.full(5000, 20), np.arange(30, 230).repeat(40), np.full(5000, 236)]).astype(np.uint8)
    target = model.astype(np.float32) * 0.95 + rng.normal(0, 2, model.size).astype(np.float32)
    target[model == 105] = 233  # e.g. bar edges where the bars are missing in the print
    lut = _fit_tone_curve(model, target)
    assert lut[150] == pytest.approx(150 * 0.95, abs=4)  # not lifted to paper white
    assert lut[236] == pytest.approx(236 * 0.95, abs=4)
    noisy = model.astype(np.float32) + rng.normal(0, 3, model.size).astype(np.float32)
    assert np.all(np.diff(_fit_tone_curve(model, noisy)) >= 0)  # still monotonic
