"""Precise registration measured against synthetic photos with exactly known geometry.

"True error" is the distance (in comparison-grid pixels) between where the
registration maps a grid point in the photo and where that artwork point
really is, measured near artwork edges (where misalignment causes false
defects). "Measured" values are what the registration reports about itself.
"""
import dataclasses

import cv2
import numpy as np
import pytest

from print_inspector import synthetic as syn
from print_inspector.core.alignment import align_label
from print_inspector.core.models import InspectionStatus, QualityLevel
from print_inspector.core.registration import (
    ManualAdjustment,
    RegistrationMethod,
    _TileMeasurer,
    comparison_grid,
    register_to_artwork,
)

POSE = dict(roll_deg=-14, tilt_x_deg=18, tilt_y_deg=8, label_fraction=0.75)
PHOTO_SIZE = (2400, 1800)
ARTWORKS = {
    "standard": syn.make_artwork,
    "small_text": syn.make_small_text_artwork,
    "detailed": syn.make_detailed_artwork,
    "low_feature": syn.make_low_feature_artwork,
}


@pytest.fixture(scope="module")
def configs():
    from print_inspector.config import load_config

    return load_config(user_path=None)


def _scenario(cfg, artwork, render=None, registration_cfg=None):
    photo = syn.make_photo(artwork, photo_size=PHOTO_SIZE, render=render, **POSE)
    alignment = align_label(photo.image, cfg.alignment, artwork)
    result = register_to_artwork(photo.image, artwork, alignment, registration_cfg or cfg.registration)
    return photo, alignment, result


def _edge_mask(result) -> np.ndarray:
    gray = cv2.cvtColor(result.reference, cv2.COLOR_BGR2GRAY)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    near = cv2.dilate((grad > 40).astype(np.uint8), np.ones((5, 5), np.uint8)) * 255
    return cv2.bitwise_and(result.mask, near)


def _true_error(photo, result, mapping=None):
    """(rms, p95, max) true error near edges, in grid pixels."""
    return syn.registration_error(
        mapping or result.grid_to_photo, result.metrics.grid_size, photo.artwork_size,
        photo.artwork_to_photo, _edge_mask(result), step=3,
    )


@pytest.fixture(scope="module")
def scenarios(configs):
    cache = {}

    def get(name, **kwargs):
        key = (name, tuple(sorted((k, repr(v)) for k, v in kwargs.items())))
        if key not in cache:
            cache[key] = _scenario(configs, ARTWORKS[name](), **kwargs)
        return cache[key]

    return get


# ---------------------------------------------------------------- accuracy on different artwork


@pytest.mark.parametrize(
    "name, max_p95",
    [("standard", 0.15), ("small_text", 0.15), ("detailed", 0.15), ("low_feature", 0.5)],
)
def test_true_accuracy_near_edges(scenarios, name, max_p95):
    photo, _, result = scenarios(name)
    rms, p95, worst = _true_error(photo, result)
    assert p95 < max_p95, (rms, p95, worst)
    assert result.is_reliable, [i.message for i in result.issues]
    assert result.recommended_status is None


def test_registration_improves_on_perspective_correction(scenarios, configs):
    photo, alignment, result = scenarios("small_text")
    _, start = comparison_grid(alignment, syn.make_small_text_artwork().shape, configs.registration)
    _, start_p95, _ = _true_error(photo, result, lambda q: cv2.perspectiveTransform(q[None], start)[0])
    _, final_p95, _ = _true_error(photo, result)
    assert final_p95 < 0.3 * start_p95


@pytest.mark.parametrize("name", ["standard", "small_text", "detailed"])
def test_reported_residual_tracks_true_error(scenarios, name):
    photo, _, result = scenarios(name)
    _, true_p95, _ = _true_error(photo, result)
    m = result.metrics
    assert m.residual_p95_px is not None
    assert abs(m.residual_p95_px - true_p95) < 0.1
    assert m.coverage >= 0.7


def test_low_feature_artwork_does_not_trust_clustered_features(scenarios):
    """Matches cluster on the few features; their homography would extrapolate badly."""
    photo, _, result = scenarios("low_feature")
    m = result.metrics
    assert m.feature_inliers is not None and m.feature_inliers > 0
    assert not m.features_used
    assert m.ecc_model == "affine"
    _, p95, _ = _true_error(photo, result)
    assert p95 < 0.5


def test_small_text_keeps_full_detail(scenarios):
    """The grid matches the photo's resolution: no downsampling of the printed detail."""
    photo, alignment, result = scenarios("small_text")
    gw, gh = result.metrics.grid_size
    rw, rh = alignment.metrics.output_size
    assert abs(gw - rw) <= 1 and abs(gh - rh) <= 1
    reg = cv2.cvtColor(result.registered, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(reg, cv2.CV_64F)[result.mask > 0].var()
    rect = cv2.cvtColor(alignment.rectified, cv2.COLOR_BGR2GRAY)
    assert sharpness > 0.85 * cv2.Laplacian(rect, cv2.CV_64F).var()


# ---------------------------------------------------------------- lens distortion and photometric differences


def test_lens_distortion_corrected_locally(scenarios):
    photo, _, result = scenarios("standard", render=dict(distortion_k1=0.03))
    m = result.metrics
    assert m.local_correction_applied and m.local_degree >= 2
    _, p95, _ = _true_error(photo, result)
    assert p95 < 0.35
    assert "misaligned" not in result.issue_codes()


def test_lens_distortion_without_local_correction_is_reported(configs):
    """A global homography matches features well, yet leaves local errors: they must be detected."""
    no_local = dataclasses.replace(configs.registration, local_min_rms_px=100.0)
    photo, _, result = _scenario(
        configs, syn.make_artwork(), render=dict(distortion_k1=0.03), registration_cfg=no_local
    )
    assert result.metrics.features_used
    _, true_p95, _ = _true_error(photo, result)
    assert true_p95 > 0.5
    assert result.metrics.residual_p95_px >= configs.registration.residual_p95_warning_px
    assert "misaligned" in result.issue_codes()


def test_ink_tone_and_blur_differences(scenarios):
    photo, _, result = scenarios(
        "detailed",
        render=dict(ink_gain=(0.9, 0.95, 1.05), ink_offset=(10, 5, -5), gamma=1.3, blur_sigma=1.2),
    )
    m = result.metrics
    assert m.tone_curve_applied
    assert m.blur_match_sigma_px > 0.9
    _, p95, _ = _true_error(photo, result)
    assert p95 < 0.3
    assert result.is_reliable


# ---------------------------------------------------------------- verification detects misalignment


@pytest.mark.parametrize("offset, level", [(0.7, QualityLevel.MARGINAL), (1.6, QualityLevel.POOR)])
def test_injected_offset_is_measured(scenarios, configs, offset, level):
    photo, alignment, good = scenarios("standard")
    shifted = good.homography @ ManualAdjustment(dx=offset).sampling_matrix(good.metrics.grid_size)
    result = register_to_artwork(
        photo.image, syn.make_artwork(), alignment, configs.registration, initial_homography=shifted, refine=False
    )
    assert result.metrics.method is RegistrationMethod.MANUAL
    # Median = typical tile; the 95th percentile adds the per-tile measurement spread.
    assert result.metrics.residual_median_px == pytest.approx(offset, abs=0.1)
    assert offset - 0.1 <= result.metrics.residual_p95_px <= offset + 0.25
    assert result.level is level
    if level is QualityLevel.POOR:
        assert result.recommended_status is InspectionStatus.REVIEW


def test_manual_adjustment_then_refinement_recovers(scenarios, configs):
    photo, alignment, good = scenarios("standard")
    rough = good.homography @ ManualAdjustment(dx=2.5, dy=-1.5, rotation_deg=0.2).sampling_matrix(
        good.metrics.grid_size
    )
    result = register_to_artwork(
        photo.image, syn.make_artwork(), alignment, configs.registration, initial_homography=rough, refine=True
    )
    assert result.metrics.method is RegistrationMethod.MANUAL_REFINED
    _, p95, _ = _true_error(photo, result)
    assert p95 < 0.15
    assert result.is_reliable


def test_wrong_artwork_gives_review(configs):
    photo = syn.make_photo(syn.make_artwork(), photo_size=PHOTO_SIZE, **POSE)
    other = syn.make_detailed_artwork()
    alignment = align_label(photo.image, configs.alignment, other)
    result = register_to_artwork(photo.image, other, alignment, configs.registration)
    assert not result.is_reliable
    assert result.recommended_status is InspectionStatus.REVIEW
    assert result.issue_codes() & {"content_mismatch", "misaligned", "low_coverage", "unverified"}


def test_requires_alignment(configs):
    from print_inspector.core.alignment import AlignmentMetrics, AlignmentResult

    failed = AlignmentResult(metrics=AlignmentMetrics(), issues=())
    image = np.zeros((100, 100, 3), np.uint8)
    result = register_to_artwork(image, image, failed, configs.registration)
    assert "not_aligned" in result.issue_codes() and result.registered is None
    assert result.recommended_status is InspectionStatus.REVIEW


# ---------------------------------------------------------------- comparison mask and resampling


def test_mask_excludes_borders_and_matches_label(scenarios, configs):
    _, _, result = scenarios("standard")
    mask = result.mask
    border = configs.registration.mask_border_px
    assert mask[:border].max() == 0 and mask[-border:].max() == 0
    assert mask[:, :border].max() == 0 and mask[:, -border:].max() == 0
    assert result.metrics.mask_fraction > 0.95
    assert result.registered.shape[:2] == mask.shape == result.reference.shape[:2]


def test_bleed_area_is_masked_out(configs):
    artwork = syn.make_artwork()
    bleed = cv2.copyMakeBorder(artwork, 24, 24, 36, 36, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    photo = syn.make_photo(artwork, photo_size=PHOTO_SIZE, **POSE)
    alignment = align_label(photo.image, configs.alignment, bleed)
    result = register_to_artwork(photo.image, bleed, alignment, configs.registration)
    gw, gh = result.metrics.grid_size
    s = gw / bleed.shape[1]
    # The bleed margin (outside the physical label) is not compared.
    assert result.mask[:, : int(30 * s)].max() == 0
    assert result.mask[: int(18 * s), :].max() == 0
    assert result.mask[gh // 2, gw // 2] == 255
    assert result.is_reliable, [i.message for i in result.issues]


def test_single_resampling_from_original_photo(scenarios, configs):
    """The registered image is exactly one remap of the photo through the reported model."""
    photo, _, result = scenarios("standard", render=dict(distortion_k1=0.03))
    assert result.field is not None
    gw, gh = result.metrics.grid_size
    ys, xs = np.mgrid[0:gh, 0:gw]
    mapped = result.grid_to_photo(np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64))
    map_x = mapped[:, 0].reshape(gh, gw).astype(np.float32)
    map_y = mapped[:, 1].reshape(gh, gw).astype(np.float32)
    again = cv2.remap(photo.image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    diff = np.abs(again.astype(int) - result.registered.astype(int))[result.mask > 0]
    assert diff.max() <= 1


def test_outputs_are_read_only(scenarios):
    _, _, result = scenarios("standard")
    for array in (result.registered, result.reference, result.mask):
        with pytest.raises(ValueError):
            array.flat[0] = 0


# ---------------------------------------------------------------- building blocks


def test_manual_adjustment_moves_photo_as_labelled():
    sample = ManualAdjustment(dx=2.0, dy=-1.0).sampling_matrix((100, 80))
    point = cv2.perspectiveTransform(np.array([[[50.0, 40.0]]]), sample)[0, 0]
    assert np.allclose(point, [48.0, 41.0])  # content moves right/up: sample from left/below
    assert ManualAdjustment().is_identity()
    assert np.allclose(ManualAdjustment().sampling_matrix((100, 80)), np.eye(3))


def test_tile_classification(configs):
    cfg = configs.registration
    t = cfg.tile_size_px
    ref = np.full((3 * t, 3 * t), 255, np.float32)
    for y in range(t + 10, 2 * t - 10, 8):  # horizontal hairlines in the middle tile row
        ref[y, :] = 0
    cv2.putText(ref, "Ag8", (2 * t + 5, 2 * t + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)
    measurer = _TileMeasurer(ref, ref, ref, np.full(ref.shape, 255, np.uint8), cfg)
    assert measurer.classify(0, 0) is None  # blank: nothing to measure
    kind, normal = measurer.classify(0, t)
    assert kind == "1d" and abs(abs(normal[1]) - 1.0) < 0.01  # lines: only the vertical offset is known
    assert measurer.classify(2 * t, 2 * t)[0] == "2d"


def test_rejects_non_bgr_input(scenarios, configs):
    _, alignment, _ = scenarios("standard")
    with pytest.raises(ValueError):
        register_to_artwork(np.zeros((10, 10), np.uint8), syn.make_artwork(), alignment, configs.registration)


def test_never_recommends_pass(scenarios):
    for name in ("standard", "low_feature"):
        _, _, result = scenarios(name)
        assert result.recommended_status in (None, InspectionStatus.REVIEW)
