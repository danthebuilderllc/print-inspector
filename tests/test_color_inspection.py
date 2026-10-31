"""Colour difference analysis: calibrated vs approximate, local changes, white balance, lighting."""
import dataclasses

import numpy as np
import pytest

from print_inspector.config import load_config
from print_inspector.core.color_calibration import (
    COLORCHECKER_24,
    SAME_PHOTO,
    SEPARATE_PHOTO,
    calibrate,
    chart_field_corners,
    render_chart,
)
from print_inspector.core.color_inspection import (
    APPROXIMATE,
    CALIBRATED,
    ColorFindingKind as K,
    heatmap_image,
    inspect_color,
)
from print_inspector.core.comparison import compare_to_artwork
from print_inspector.core.decision import OperatorMark, decide
from print_inspector.core.defects import assess_defects
from print_inspector.core.models import InspectionStatus
from tests.color_fakes import PATCHES, camera, make_color_label, recolour
from tests.test_comparison import full_mask, simulate

PX_PER_MM = 12.0
WARM = dict(gains=(1.15, 1.0, 0.8), exposure=0.85, crosstalk=0.1)


@pytest.fixture(scope="module")
def config():
    return load_config(user_path=None)


@pytest.fixture(scope="module")
def artwork():
    return make_color_label()


@pytest.fixture(scope="module")
def profile():
    chart = COLORCHECKER_24
    return calibrate(camera(render_chart(chart), **WARM), chart_field_corners(chart), chart, SAME_PHOTO, "chart.jpg")


def analyse(config, artwork, printed, cam=None, profile=None, cfg=None, defects=None, **sim):
    photo = simulate(camera(printed, **(cam or {})), **sim)
    return inspect_color(artwork, photo, full_mask(artwork.shape), defects, PX_PER_MM, profile, cfg or config.color)


def verdict(config, artwork, report, marks=None):
    comparison = compare_to_artwork(artwork, artwork, full_mask(artwork.shape), config.comparison, 0.0)
    defects = assess_defects(comparison, artwork, artwork, PX_PER_MM, False, config.defects)
    return decide(None, None, None, comparison, defects, {}, config.decision, color_report=report, color_marks=marks or {})


def area_at(report, box):
    x, y, w, h = box
    return next(a for a in report.areas if a.x <= x + w / 2 <= a.x + a.width and a.y <= y + h / 2 <= a.y + a.height)


# ---------------------------------------------------------------- approximate (uncalibrated)


def test_clean_print_matches(config, artwork):
    report = analyse(config, artwork, artwork.copy())
    assert report.mode == APPROXIMATE and not report.calibrated
    assert "No colour calibration" in report.mode_reasons[0]
    assert "color_uncalibrated" in [i.code for i in report.issues]
    assert len(report.areas) == 6 and sum(a.paper for a in report.areas) == 1
    assert all(a.delta_e < 1.0 for a in report.areas)
    assert report.findings == ()
    assert 0.5 < report.meaningful_fraction < 1.0
    # Heatmaps: measured inside the colour areas, NaN on edges and fine text.
    x, y, w, h = PATCHES["red"][1]
    assert np.isfinite(report.heatmap[y + h // 2, x + w // 2]) and np.isnan(report.heatmap[y, x])
    assert np.isnan(report.local_heatmap[525, 900])  # inside the text
    assert verdict(config, artwork, report).status is InspectionStatus.PASS


def test_camera_white_balance_is_identified_not_reported_as_defects(config, artwork):
    report = analyse(config, artwork, artwork.copy(), WARM)
    codes = [i.code for i in report.issues]
    assert "color_white_balance" in codes
    assert report.paper_photo_delta_e > 5
    r, g, b = report.channel_balance
    assert r > 1.1 and b < 0.9  # warm light / white balance
    assert report.findings == ()  # approximate deviations below the REVIEW threshold
    assert max(a.delta_e for a in report.areas) < config.color.approximate_review_delta_e
    assert verdict(config, artwork, report).status is InspectionStatus.PASS


def test_local_colour_change_is_found_in_both_modes(config, artwork, profile):
    printed = artwork.copy()
    x, y, w, h = PATCHES["red"][1]
    recolour(printed, (x + 100, y + 60, 80, 80), (40, 90, 200))  # an orange spot in the red
    for prof in (None, profile):
        report = analyse(config, artwork, printed, WARM, prof)
        (f,) = report.findings
        assert f.kind is K.LOCAL_CHANGE and f.reliable and f.delta_e > config.color.local_reliable_delta_e
        assert x + 90 <= f.x <= x + 110 and f.width <= 100 and 5 < f.area_mm2 < 60
        assert "yellower" in f.describe()
    assert verdict(config, artwork, report).status is InspectionStatus.FAIL


def test_wrong_ink_is_not_normalised_away_but_uncalibrated_only_reviews(config, artwork):
    printed = artwork.copy()
    recolour(printed, PATCHES["blue"][1], (170, 60, 110))  # blue printed violet
    report = analyse(config, artwork, printed, WARM)
    (f,) = report.findings
    assert f.kind is K.AREA_DEVIATION and not f.reliable and f.delta_e > 15
    assert "Approximate" in f.notes[1]
    decision = verdict(config, artwork, report)
    assert decision.status is InspectionStatus.REVIEW and decision.color_review == (f.id,)
    assert "uncalibrated" in " ".join(decision.reasons)
    assert verdict(config, artwork, report, {f.id: OperatorMark.CONFIRMED}).status is InspectionStatus.FAIL


def test_small_deviations_are_not_claimed_without_calibration(config, artwork):
    printed = artwork.copy()
    recolour(printed, PATCHES["blue"][1], (170, 120, 60))  # lighter blue, about ΔE00 11
    report = analyse(config, artwork, printed, WARM)
    assert report.findings == ()


# ---------------------------------------------------------------- calibrated


def test_calibration_makes_colours_measurable(config, artwork, profile):
    report = analyse(config, artwork, artwork.copy(), WARM, profile)
    assert report.mode == CALIBRATED and report.calibrated
    assert "leave-one-out" in report.profile
    assert all(a.delta_e < 2.5 for a in report.areas)
    assert "color_uncalibrated" not in [i.code for i in report.issues]
    assert report.findings == ()


@pytest.mark.parametrize("bgr, severity", [((170, 120, 60), "critical"), ((60, 60, 160), "major")])
def test_calibrated_deviations_are_reliable(config, artwork, profile, bgr, severity):
    printed = artwork.copy()
    box = PATCHES["blue"][1] if bgr[0] > 100 else PATCHES["red"][1]
    recolour(printed, box, bgr)
    report = analyse(config, artwork, printed, WARM, profile)
    deviations = [f for f in report.findings if f.kind is K.AREA_DEVIATION]
    assert len(deviations) == 1 and deviations[0].reliable and deviations[0].severity.value == severity
    assert "Calibrated measurement" in deviations[0].notes[0]
    assert verdict(config, artwork, report).status is InspectionStatus.FAIL


def test_inaccurate_profile_is_not_used(config, artwork, profile):
    bad = dataclasses.replace(profile, loo_mean_delta_e=7.5)
    report = analyse(config, artwork, artwork.copy(), WARM, bad)
    assert report.mode == APPROXIMATE and "not accurate enough" in report.mode_reasons[0]


def test_separate_chart_photo_scales_exposure_on_paper(config, artwork, profile):
    separate = dataclasses.replace(profile, source=SEPARATE_PHOTO)
    darker = dict(WARM, exposure=0.6)  # the label photo is darker than the chart photo
    report = analyse(config, artwork, artwork.copy(), darker, separate)
    assert any("Exposure scaled" in n for n in report.notes)
    assert all(a.delta_e < 3.0 for a in report.areas)


def test_global_ink_shift_calibrated(config, artwork, profile):
    printed = artwork.copy()
    for name, (bgr, box) in PATCHES.items():
        recolour(printed, box, tuple(int(v * 0.7) for v in bgr))  # every ink much darker
    report = analyse(config, artwork, printed, WARM, profile)
    shift = [f for f in report.findings if f.kind is K.GLOBAL_SHIFT]
    assert len(shift) == 1 and shift[0].reliable and shift[0].dl < -3
    assert "darker" in shift[0].describe()


# ---------------------------------------------------------------- lighting, structure, edge cases


def test_uneven_lighting_is_corrected_and_reported(config, artwork):
    report = analyse(config, artwork, artwork.copy(), lighting=0.5)
    assert report.lighting_range[1] / report.lighting_range[0] > 1.3
    assert "color_uneven_lighting" in [i.code for i in report.issues]
    assert report.findings == ()  # the falloff is not a local colour change


def test_structural_defects_are_left_to_the_structural_inspection(config, artwork):
    printed = artwork.copy()
    x, y, w, h = PATCHES["green"][1]
    recolour(printed, (x + 100, y + 70, 70, 70), (245, 245, 245))  # missing ink: a structural defect
    photo = simulate(printed)
    comparison = compare_to_artwork(photo, artwork, full_mask(artwork.shape), config.comparison, 0.9)
    defects = assess_defects(comparison, artwork, photo, PX_PER_MM, False, config.defects)
    assert any(d.category.value == "missing_print" for d in defects.defects)
    with_structure = inspect_color(artwork, photo, full_mask(artwork.shape), defects, PX_PER_MM, None, config.color)
    without = inspect_color(artwork, photo, full_mask(artwork.shape), None, PX_PER_MM, None, config.color)
    assert any(f.kind is K.LOCAL_CHANGE for f in without.findings)
    assert not any(f.kind is K.LOCAL_CHANGE for f in with_structure.findings)
    assert np.isnan(with_structure.local_heatmap[y + 105, x + 135])


def test_full_bleed_artwork_without_paper(config):
    artwork = np.full((400, 600, 3), (40, 40, 200), np.uint8)
    report = inspect_color(artwork, simulate(artwork), full_mask(artwork.shape), None, PX_PER_MM, None, config.color)
    assert report.paper_design_lab is None and report.channel_balance is None
    assert any("No unprinted paper" in n for n in report.notes)
    assert "without any correction" in report.mode_reasons[1]


def test_thresholds_are_configurable(config, artwork):
    printed = artwork.copy()
    recolour(printed, PATCHES["blue"][1], (170, 120, 60))
    strict = dataclasses.replace(config.color, approximate_review_delta_e=8.0)
    report = analyse(config, artwork, printed, WARM, cfg=strict)
    assert [f.kind for f in report.findings] == [K.AREA_DEVIATION]


def test_heatmap_image():
    values = np.full((10, 10), np.nan, np.float32)
    values[2:5, 2:5] = 20.0
    image = heatmap_image(values, 20.0)
    assert image.shape == (10, 10, 3) and image.dtype == np.uint8
    assert tuple(image[0, 0]) == (128, 128, 128)
    assert image[3, 3, 1] > 200  # full scale: yellow (viridis)
