"""Capture-quality checks on generated photos with one controlled problem each."""
import dataclasses

import cv2
import numpy as np
import pytest

from print_inspector.core.capture_quality import (
    QualityLevel,
    Severity,
    artwork_label_long_side_mm,
    assess_capture_quality,
    lighting_unevenness,
)
from print_inspector.core.image_io import ImageRole, LoadedImage
from print_inspector.core.models import InspectionStatus
from tests.conftest import make_label_photo

# The generated photo is 4.3 MP; the label spans ~1680 px. With an 80 mm label
# that is ~21 px/mm, comfortably above the default 15 px/mm recommendation.
LABEL_MM = 80.0


@pytest.fixture(scope="module")
def photo() -> np.ndarray:
    return make_label_photo()


@pytest.fixture
def cfg(app_config):
    # Generated photos are smaller than real 12 MP captures; keep that warning out of the way.
    return dataclasses.replace(app_config.capture_quality, min_megapixels_warning=1.0)


def _severity(report, code):
    found = [i.severity for i in report.issues if i.code == code]
    return found[0] if found else None


def test_good_photo_has_no_issues(photo, cfg):
    report = assess_capture_quality(photo, cfg, LABEL_MM)
    assert report.issues == ()
    assert report.level is QualityLevel.GOOD
    assert report.is_reliable
    assert report.recommended_status is None  # no verdict: certainly not PASS


def test_content_region_matches_label(photo, cfg):
    region = assess_capture_quality(photo, cfg, LABEL_MM).metrics.content_region
    h, w = photo.shape[:2]
    assert region is not None
    # Label occupies 15 %..85 % of the frame in both directions.
    assert abs(region.x - 0.15 * w) < 0.03 * w
    assert abs(region.y - 0.15 * h) < 0.03 * h
    assert abs(region.width - 0.70 * w) < 0.05 * w
    assert abs(region.height - 0.70 * h) < 0.05 * h


@pytest.mark.parametrize("sigma, low, high", [(1.0, 1.0, 1.6), (2.0, 1.7, 2.4), (5.0, 3.5, 5.5)])
def test_blur_estimate_tracks_true_blur(photo, cfg, sigma, low, high):
    blurred = cv2.GaussianBlur(photo, (0, 0), sigma)
    measured = assess_capture_quality(blurred, cfg, LABEL_MM).metrics.blur_sigma_px
    assert low <= measured <= high


def test_sharp_photo_measures_near_pixel_limit(photo, cfg):
    assert assess_capture_quality(photo, cfg, LABEL_MM).metrics.blur_sigma_px < 1.1


def test_moderate_blur_warns(photo, cfg):
    report = assess_capture_quality(cv2.GaussianBlur(photo, (0, 0), 2.2), cfg, LABEL_MM)
    assert _severity(report, "blur") is Severity.WARNING


def test_heavy_blur_is_critical_and_review(photo, cfg):
    report = assess_capture_quality(cv2.GaussianBlur(photo, (0, 0), 5.0), cfg, LABEL_MM)
    assert _severity(report, "blur") is Severity.CRITICAL
    assert report.level is QualityLevel.POOR
    assert report.recommended_status is InspectionStatus.REVIEW


def test_extreme_blur_is_never_reported_as_sharp(photo, cfg):
    report = assess_capture_quality(cv2.GaussianBlur(photo, (0, 0), 25.0), cfg, LABEL_MM)
    assert report.level is QualityLevel.POOR
    assert report.issue_codes() & {"blur", "blur_unmeasurable", "no_content"}


def test_blur_thresholds_are_configurable(photo, cfg):
    blurred = cv2.GaussianBlur(photo, (0, 0), 5.0)
    lenient = dataclasses.replace(
        cfg, blur_warning_sigma_px=50, blur_critical_sigma_px=60, min_pixels_per_mm_critical=0.5,
        min_pixels_per_mm_warning=1,
    )
    assert "blur" not in assess_capture_quality(blurred, lenient, LABEL_MM).issue_codes()


def test_overexposure_critical(photo, cfg):
    bright = np.clip(photo.astype(np.float32) * 1.6, 0, 255).astype(np.uint8)
    report = assess_capture_quality(bright, cfg, LABEL_MM)
    assert _severity(report, "overexposure") is Severity.CRITICAL
    assert report.recommended_status is InspectionStatus.REVIEW
    assert report.metrics.clipped_fraction > 0.5


def test_glare_spot_detected(photo, cfg):
    glare = photo.copy()
    cv2.ellipse(glare, (1300, 900), (200, 140), 0, 0, 360, (255, 255, 255), -1)
    report = assess_capture_quality(glare, cfg, LABEL_MM)
    assert _severity(report, "glare") is Severity.CRITICAL
    assert "overexposure" not in report.issue_codes() or _severity(report, "overexposure") is Severity.WARNING


def test_small_glare_spot_warns(photo, cfg):
    glare = photo.copy()
    cv2.ellipse(glare, (1300, 900), (70, 50), 0, 0, 360, (255, 255, 255), -1)
    assert _severity(assess_capture_quality(glare, cfg, LABEL_MM), "glare") is Severity.WARNING


def test_exposure_thresholds_are_configurable(photo, cfg):
    bright = np.clip(photo.astype(np.float32) * 1.6, 0, 255).astype(np.uint8)
    lenient = dataclasses.replace(
        cfg,
        overexposure_warning_fraction=0.98,
        overexposure_critical_fraction=0.99,
        glare_warning_fraction=0.98,
        glare_critical_fraction=0.99,
    )
    codes = assess_capture_quality(bright, lenient, LABEL_MM).issue_codes()
    assert "overexposure" not in codes and "glare" not in codes


def test_underexposure_critical(photo, cfg):
    dark = (photo.astype(np.float32) * 0.35).astype(np.uint8)
    report = assess_capture_quality(dark, cfg, LABEL_MM)
    assert _severity(report, "underexposure") is Severity.CRITICAL
    assert report.recommended_status is InspectionStatus.REVIEW


def test_mild_underexposure_warns(photo, cfg):
    dim = (photo.astype(np.float32) * 0.55).astype(np.uint8)
    assert _severity(assess_capture_quality(dim, cfg, LABEL_MM), "underexposure") is Severity.WARNING


def test_uneven_lighting_detected(photo, cfg):
    ramp = np.linspace(1.0, 0.25, photo.shape[1], dtype=np.float32)[None, :, None]
    shaded = (photo * ramp).astype(np.uint8)
    report = assess_capture_quality(shaded, cfg, LABEL_MM)
    assert _severity(report, "uneven_lighting") is Severity.CRITICAL
    assert report.metrics.lighting_unevenness > 0.45


def test_mild_uneven_lighting_warns(photo, cfg):
    ramp = np.linspace(1.0, 0.6, photo.shape[1], dtype=np.float32)[None, :, None]
    shaded = (photo * ramp).astype(np.uint8)
    assert _severity(assess_capture_quality(shaded, cfg, LABEL_MM), "uneven_lighting") is Severity.WARNING


def test_solid_ink_area_is_not_mistaken_for_shadow():
    region = np.full((400, 400), 235, np.uint8)
    region[50:150, 50:350] = 30  # large black print block
    region[250:300, 100:200] = 90
    assert lighting_unevenness(region, 8) < 0.1


def test_label_cut_off_at_left_edge(photo, cfg):
    cropped = np.ascontiguousarray(photo[:, 900:])
    report = assess_capture_quality(cropped, cfg, LABEL_MM)
    issue = [i for i in report.issues if i.code == "incomplete_capture"]
    assert issue and issue[0].severity is Severity.CRITICAL
    assert "left" in issue[0].message and "right" not in issue[0].message
    assert report.recommended_status is InspectionStatus.REVIEW


def test_label_cut_off_at_top_and_bottom(photo, cfg):
    cropped = np.ascontiguousarray(photo[400:1250])
    report = assess_capture_quality(cropped, cfg, LABEL_MM)
    assert report.metrics.border_content_ratios["top"] >= cfg.cutoff_warning_ratio
    assert "incomplete_capture" in report.issue_codes()


def test_blank_photo_has_no_content(cfg):
    blank = np.full((1200, 1600, 3), 200, np.uint8)
    report = assess_capture_quality(blank, cfg, LABEL_MM)
    assert _severity(report, "no_content") is Severity.CRITICAL
    assert report.recommended_status is InspectionStatus.REVIEW


def test_low_resolution_critical(photo, cfg):
    small = cv2.resize(photo, (600, 450), interpolation=cv2.INTER_AREA)
    report = assess_capture_quality(small, cfg, LABEL_MM)
    assert _severity(report, "low_resolution") is Severity.CRITICAL
    assert report.metrics.pixels_per_mm < cfg.min_pixels_per_mm_critical


def test_resolution_uses_assumed_label_size_without_artwork(photo, cfg):
    report = assess_capture_quality(photo, cfg, None)
    m = report.metrics
    assert m.label_size_assumed and m.label_long_side_mm == cfg.assumed_label_long_side_mm
    assert m.pixels_per_mm == pytest.approx(m.label_long_side_px / cfg.assumed_label_long_side_mm)
    # 1680 px over an assumed 150 mm is ~11 px/mm: below the 15 px/mm recommendation.
    assert _severity(report, "low_resolution") is Severity.WARNING
    assert "assuming" in [i for i in report.issues if i.code == "low_resolution"][0].message


def test_blur_lowers_effective_resolution(photo, cfg):
    sharp = assess_capture_quality(photo, cfg, LABEL_MM).metrics
    blurred = assess_capture_quality(cv2.GaussianBlur(photo, (0, 0), 2.0), cfg, LABEL_MM).metrics
    assert blurred.effective_pixels_per_mm < sharp.effective_pixels_per_mm * 0.7


def test_low_megapixels_warning(photo, app_config):
    report = assess_capture_quality(photo, app_config.capture_quality, LABEL_MM)
    assert _severity(report, "low_megapixels") is Severity.WARNING
    assert report.level is QualityLevel.MARGINAL
    assert report.recommended_status is None


def test_critical_issues_listed_first(cfg):
    blank = np.full((300, 400, 3), 200, np.uint8)
    report = assess_capture_quality(blank, dataclasses.replace(cfg, min_megapixels_warning=12), LABEL_MM)
    severities = [i.severity for i in report.issues]
    assert severities == sorted(severities, key=lambda s: s is not Severity.CRITICAL)


def test_never_recommends_pass(photo, cfg):
    variants = [photo, cv2.GaussianBlur(photo, (0, 0), 4.0), np.full_like(photo, 0)]
    for image in variants:
        assert assess_capture_quality(image, cfg, LABEL_MM).recommended_status in (
            None,
            InspectionStatus.REVIEW,
        )


def test_input_not_modified(photo, cfg):
    before = photo.copy()
    assess_capture_quality(photo, cfg, LABEL_MM)
    assert np.array_equal(photo, before)


def test_rejects_non_bgr_input(cfg):
    with pytest.raises(ValueError):
        assess_capture_quality(np.zeros((10, 10), np.uint8), cfg)


def _artwork_with_dpi(dpi):
    data = np.zeros((600, 1181, 3), np.uint8)
    data.setflags(write=False)
    return LoadedImage(
        path=__import__("pathlib").Path("art.png"),
        role=ImageRole.ORIGINAL,
        file_format="PNG",
        file_size_bytes=1,
        original=data,
        working=data,
        dpi=dpi,
    )


def test_artwork_label_size_from_dpi(cfg):
    assert artwork_label_long_side_mm(_artwork_with_dpi((300.0, 300.0)), cfg) == pytest.approx(100.0, abs=0.1)
    assert artwork_label_long_side_mm(_artwork_with_dpi(None), cfg) is None
    # 1181 px at 1 dpi would be a 30 m label: ignored as implausible.
    assert artwork_label_long_side_mm(_artwork_with_dpi((1.0, 1.0)), cfg) is None
    assert artwork_label_long_side_mm(None, cfg) is None
