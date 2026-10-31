"""Perspective correction measured against synthetic photos with known geometry.

Geometric error = distance between where artwork points end up in the rectified
image and where they belong, in artwork pixels (see synthetic.rectification_error).
"""
import dataclasses

import cv2
import numpy as np
import pytest

from print_inspector import synthetic as syn
from print_inspector.core.alignment import (
    AlignmentMethod,
    align_label,
    aspect_ratio_and_focal,
    edge_corners,
    estimate_aspect_ratio,
    output_size_for,
    side_lengths,
)
from print_inspector.core.models import InspectionStatus, QualityLevel, Severity

# Pose, expected quarter turns needed to bring the label upright.
CASES = {
    "frontal": (dict(), 0),
    "roll 15": (dict(roll_deg=15), 0),
    "roll -35, tilt 20": (dict(roll_deg=-35, tilt_x_deg=20), 0),
    "tilt 30/15": (dict(roll_deg=5, tilt_x_deg=30, tilt_y_deg=15), 0),
    "upside down": (dict(roll_deg=188, tilt_y_deg=10), 2),
    "quarter turn": (dict(roll_deg=97, tilt_x_deg=12), 1),
    "three-quarter turn": (dict(roll_deg=-85, tilt_y_deg=-12), 3),
    "small, off-centre": (dict(label_fraction=0.3, roll_deg=20, offset=(0.15, -0.1)), 0),
    "large": (dict(label_fraction=0.85, roll_deg=3), 0),
}


@pytest.fixture(scope="module")
def artwork():
    return syn.make_artwork()


@pytest.fixture
def cfg(app_config):
    return app_config.alignment


def _severity(result, code):
    found = [i.severity for i in result.issues if i.code == code]
    return found[0] if found else None


# ---------------------------------------------------------------- geometry helpers


@pytest.mark.parametrize("aspect", [1.5, 1.0, 0.6])
@pytest.mark.parametrize("focal", [0.6, 0.8, 1.2])
@pytest.mark.parametrize(
    "pose", [dict(), dict(tilt_x_deg=35, roll_deg=10), dict(tilt_x_deg=25, tilt_y_deg=20, roll_deg=30)]
)
def test_aspect_ratio_recovered_for_any_lens(aspect, focal, pose):
    corners = syn.project_label(aspect, (4000, 3000), focal_factor=focal, **pose)
    estimated, recovered = aspect_ratio_and_focal(corners, (4000, 3000))
    assert recovered
    assert estimated == pytest.approx(aspect, rel=0.005)


def test_aspect_ratio_with_one_parallel_pair_is_approximate():
    """Pure forward tilt: the lens cannot be recovered, so an assumed focal length is used."""
    corners = syn.project_label(1.5, (4000, 3000), focal_factor=0.8, tilt_x_deg=35)
    estimated, recovered = aspect_ratio_and_focal(corners, (4000, 3000), assumed_focal_factor=0.78)
    assert not recovered
    assert estimated == pytest.approx(1.5, rel=0.02)
    assert estimate_aspect_ratio(corners, (4000, 3000), 0.78) == estimated


def test_approximate_aspect_mismatch_only_warns(artwork, cfg):
    photo = syn.make_photo(artwork, tilt_x_deg=35)
    wrong_lens = dataclasses.replace(cfg, assumed_focal_length_factor=0.4)
    result = align_label(photo.image, wrong_lens, artwork)
    assert result.metrics.aspect_error > cfg.aspect_tolerance_critical
    assert _severity(result, "aspect_mismatch") is Severity.WARNING
    assert "approximate" in [i.message for i in result.issues if i.code == "aspect_mismatch"][0]


def test_output_size_keeps_native_resolution():
    corners = np.array([[100, 100], [1300, 140], [1280, 900], [120, 860]], float)
    w, h = output_size_for(corners, 1.5, 100000)
    top, right, bottom, left = side_lengths(corners)
    assert w >= max(top, bottom) - 1 and h >= max(left, right) - 1
    assert w / h == pytest.approx(1.5, abs=0.01)


def test_output_size_is_capped():
    corners = np.array([[0, 0], [9000, 0], [9000, 6000], [0, 6000]], float)
    w, h = output_size_for(corners, 1.5, 3000)
    assert max(w, h) == 3000 and w / h == pytest.approx(1.5, abs=0.01)


# ---------------------------------------------------------------- automatic alignment


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_rectification_error_below_one_artwork_pixel(artwork, cfg, case):
    pose, turns = CASES[case]
    photo = syn.make_photo(artwork, **pose)
    result = align_label(photo.image, cfg, artwork)
    assert result.metrics.method is AlignmentMethod.BOUNDARY
    assert result.metrics.quarter_turns == turns
    rms, worst = syn.rectification_error(result.homography, result.metrics.output_size, photo)
    assert rms < 0.7 and worst < 1.2
    assert result.level is QualityLevel.GOOD, [i.message for i in result.issues]
    assert result.recommended_status is None


def test_output_matches_artwork_proportions_and_scale(artwork, cfg):
    photo = syn.make_photo(artwork, roll_deg=10, tilt_x_deg=20)
    result = align_label(photo.image, cfg, artwork)
    w, h = result.metrics.output_size
    assert result.rectified.shape == (h, w, 3)
    assert w / h == pytest.approx(artwork.shape[1] / artwork.shape[0], abs=0.005)
    assert result.metrics.scale_to_artwork == pytest.approx(w / artwork.shape[1])
    assert result.valid_mask.shape == (h, w) and result.valid_mask.min() == 255


def test_single_interpolation_from_original_photo(artwork, cfg):
    """The output is exactly one warp of the original photo with the reported homography."""
    photo = syn.make_photo(artwork, roll_deg=-20, tilt_y_deg=15)
    result = align_label(photo.image, cfg, artwork)
    expected = cv2.warpPerspective(photo.image, result.homography, result.metrics.output_size, flags=cv2.INTER_CUBIC)
    assert np.array_equal(result.rectified, expected)
    w, h = result.metrics.output_size
    mapped = cv2.perspectiveTransform(result.metrics.corners[None], result.homography)[0]
    assert np.allclose(mapped, edge_corners(w, h), atol=1e-6)


def test_rectified_image_is_read_only(artwork, cfg):
    result = align_label(syn.make_photo(artwork).image, cfg, artwork)
    with pytest.raises(ValueError):
        result.rectified[0, 0, 0] = 1


def test_linear_interpolation_option(artwork, cfg):
    photo = syn.make_photo(artwork, roll_deg=10)
    result = align_label(photo.image, dataclasses.replace(cfg, interpolation="linear"), artwork)
    expected = cv2.warpPerspective(photo.image, result.homography, result.metrics.output_size, flags=cv2.INTER_LINEAR)
    assert np.array_equal(result.rectified, expected)


def test_rectified_content_matches_artwork(artwork, cfg):
    photo = syn.make_photo(artwork, roll_deg=30, tilt_x_deg=20)
    result = align_label(photo.image, cfg, artwork)
    rect = cv2.cvtColor(result.rectified, cv2.COLOR_BGR2GRAY).astype(np.float32)
    art = cv2.resize(cv2.cvtColor(artwork, cv2.COLOR_BGR2GRAY), rect.shape[::-1], interpolation=cv2.INTER_AREA)
    corr = np.corrcoef(rect.ravel(), art.astype(np.float32).ravel())[0, 1]
    assert corr > 0.95
    assert result.metrics.similarity > 0.85


def test_feature_fallback_when_edges_are_invisible(artwork, cfg):
    corners = syn.project_label(1.5, (2000, 1500), roll_deg=12, tilt_x_deg=10)
    photo = syn.render_photo(artwork, (2000, 1500), corners, background_bgr=(250, 250, 250), background_texture=1.0)
    result = align_label(photo.image, cfg, artwork)
    assert result.metrics.method is AlignmentMethod.FEATURES
    rms, worst = syn.rectification_error(result.homography, result.metrics.output_size, photo)
    assert worst < 2.5
    assert result.is_reliable


def test_no_artwork_estimates_proportions_and_warns(artwork, cfg):
    photo = syn.make_photo(artwork, roll_deg=10, tilt_x_deg=25)
    result = align_label(photo.image, cfg, None)
    assert result.metrics.method is AlignmentMethod.BOUNDARY
    assert _severity(result, "not_verified") is Severity.WARNING
    assert result.level is QualityLevel.MARGINAL
    w, h = result.metrics.output_size
    assert w / h == pytest.approx(1.5, rel=0.01)
    rms, worst = syn.rectification_error(result.homography, (w, h), photo)
    assert worst < 2.0


def test_label_not_found_requires_manual_corners(cfg):
    blank = np.full((1500, 2000, 3), 120, np.uint8)
    result = align_label(blank, cfg, syn.make_artwork())
    assert "label_not_found" in result.issue_codes()
    assert result.rectified is None
    assert result.recommended_status is InspectionStatus.REVIEW
    assert "Manual Corners" in result.issues[0].message


def test_invisible_edges_without_artwork_fail_safely(artwork, cfg):
    corners = syn.project_label(1.5, (2000, 1500), roll_deg=12)
    photo = syn.render_photo(artwork, (2000, 1500), corners, background_bgr=(250, 250, 250), background_texture=1.0)
    result = align_label(photo.image, cfg, None)
    assert not result.is_reliable
    assert result.recommended_status is InspectionStatus.REVIEW


# ---------------------------------------------------------------- unreliable alignments are flagged


def test_wrong_artwork_is_flagged(artwork, cfg):
    photo = syn.make_photo(artwork, roll_deg=10)
    other = cv2.flip(syn.make_artwork(), 1)
    cv2.putText(other, "DIFFERENT", (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 12)
    result = align_label(photo.image, cfg, other)
    assert _severity(result, "content_mismatch") is Severity.CRITICAL
    assert result.recommended_status is InspectionStatus.REVIEW


@pytest.mark.parametrize(
    "offset_fraction, expected",
    [(0.0, QualityLevel.GOOD), (0.01, QualityLevel.MARGINAL), (0.03, QualityLevel.POOR)],
)
def test_inaccurate_corners_are_graded(artwork, cfg, offset_fraction, expected):
    photo = syn.make_photo(artwork, roll_deg=12, tilt_x_deg=15)
    diagonal = np.linalg.norm(photo.corners[2] - photo.corners[0])
    corners = photo.corners.copy()
    corners[1] += np.array([1, -1]) * offset_fraction * diagonal / np.sqrt(2)
    result = align_label(photo.image, cfg, artwork, manual_corners=corners)
    assert result.metrics.method is AlignmentMethod.MANUAL
    assert result.level is expected, [i.message for i in result.issues]


def test_exact_manual_corners_in_any_order(artwork, cfg):
    photo = syn.make_photo(artwork, roll_deg=190, tilt_y_deg=10)
    result = align_label(photo.image, cfg, artwork, manual_corners=photo.corners[[3, 1, 0, 2]])
    rms, worst = syn.rectification_error(result.homography, result.metrics.output_size, photo)
    assert worst < 0.5
    assert result.metrics.quarter_turns == 2


def test_inner_outline_is_rejected(artwork, cfg):
    photo = syn.make_photo(artwork, roll_deg=12, tilt_x_deg=15)
    centre = photo.corners.mean(axis=0)
    inner = centre + (photo.corners - centre) * 0.75
    result = align_label(photo.image, cfg, artwork, manual_corners=inner)
    assert _severity(result, "outline_mismatch") is Severity.CRITICAL


def test_artwork_with_bleed_is_not_rejected(artwork, cfg):
    photo = syn.make_photo(artwork, roll_deg=12, tilt_x_deg=15)
    bleed = cv2.copyMakeBorder(artwork, 24, 24, 36, 36, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    result = align_label(photo.image, cfg, bleed)
    assert result.is_reliable, [i.message for i in result.issues]
    assert "corner_disagreement" in result.issue_codes()
    assert result.metrics.artwork_scale == pytest.approx((1.06, 1.06), abs=0.01)


def test_label_partly_outside_photo(artwork, cfg):
    photo = syn.make_photo(artwork, roll_deg=5, label_fraction=0.7, offset=(0.3, 0.0))
    assert photo.corners[:, 0].max() > 2000
    result = align_label(photo.image, cfg, artwork, manual_corners=photo.corners)
    assert _severity(result, "label_outside_photo") is Severity.CRITICAL
    assert result.valid_mask.min() == 0  # part of the output has no photo data


def test_steep_angle_is_flagged(artwork, cfg):
    photo = syn.make_photo(artwork, tilt_x_deg=60, roll_deg=10, label_fraction=0.9)
    result = align_label(photo.image, cfg, artwork, manual_corners=photo.corners)
    assert result.metrics.side_ratio > cfg.max_side_ratio_warning
    assert "strong_perspective" in result.issue_codes()


def test_degenerate_manual_corners(cfg):
    photo = np.full((600, 800, 3), 200, np.uint8)
    collinear = np.array([[10, 10], [200, 10], [400, 10], [600, 10]], float)
    result = align_label(photo, cfg, None, manual_corners=collinear)
    assert "invalid_corners" in result.issue_codes()
    assert result.rectified is None and not result.is_reliable


def test_never_recommends_pass(artwork, cfg):
    for image in (syn.make_photo(artwork).image, np.full((800, 1000, 3), 90, np.uint8)):
        assert align_label(image, cfg, artwork).recommended_status in (None, InspectionStatus.REVIEW)


def test_rejects_non_bgr_input(cfg):
    with pytest.raises(ValueError):
        align_label(np.zeros((10, 10), np.uint8), cfg)
