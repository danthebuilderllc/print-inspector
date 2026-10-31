"""Defect localisation, categories, severity, reliability and evidence on known modifications.

The registered photo is simulated on the artwork grid (see test_comparison.simulate)
with 12 grid pixels per mm, i.e. the 1200 px artwork is a 100 mm label.
"""
import dataclasses

import cv2
import numpy as np
import pytest

from print_inspector import synthetic as syn
from print_inspector.core.comparison import compare_to_artwork
from print_inspector.core.defects import (
    DefectCategory,
    DefectSeverity,
    DetectionReliability,
    assess_defects,
    defect_rows,
    grid_px_per_mm,
    text_like_mask,
)
from tests.test_comparison import ARTWORK, full_mask, simulate

PX_PER_MM = 12.0


@pytest.fixture(scope="module")
def config():
    from print_inspector.config import load_config

    return load_config(user_path=None)


def analyse(config, printed, artwork=ARTWORK, defects_cfg=None, **sim):
    registered = simulate(printed, **sim)
    comparison = compare_to_artwork(registered, artwork, full_mask(artwork.shape), config.comparison, 0.9)
    return assess_defects(comparison, artwork, registered, PX_PER_MM, False, defects_cfg or config.defects)


def only(report):
    assert len(report.defects) == 1, [(d.category, d.x, d.y) for d in report.defects]
    return report.defects[0]


def contains(defect, box, pad=6):
    x, y, w, h = box
    return defect.x - pad <= x and defect.y - pad <= y and defect.x + defect.width + pad >= x + w and defect.y + defect.height + pad >= y + h


def local_colour_patch(dl=0, da=-10, db=-35):
    printed = ARTWORK.copy()
    patch = np.zeros(ARTWORK.shape[:2], bool)
    patch[230:280, 880:960] = True
    patch &= np.all(ARTWORK == (40, 40, 210), axis=2)
    lab = cv2.cvtColor(printed, cv2.COLOR_BGR2LAB).astype(np.int16)
    lab[patch] += np.int16([dl, da, db])
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------- categories


def test_missing_print(config):
    printed, truth = syn.add_defects(ARTWORK, ["missing_region"], seed=4)
    d = only(analyse(config, printed))
    assert d.category is DefectCategory.MISSING_PRINT and d.category_certain
    assert d.polarity == "lighter" and d.ink_fraction > 0.9
    assert contains(d, truth[0].bbox)


@pytest.mark.parametrize("kind", ["extra_mark", "unexpected_shape", "small_dot"])
def test_extra_print(config, kind):
    printed, truth = syn.add_defects(ARTWORK, [kind], seed=4)
    d = only(analyse(config, printed))
    assert d.category is DefectCategory.EXTRA_PRINT and d.category_certain
    assert d.polarity == "darker" and d.ink_fraction < 0.2
    assert contains(d, truth[0].bbox)


def test_shape_difference_from_shifted_element(config):
    artwork = ARTWORK.copy()
    cv2.rectangle(artwork, (440, 520), (460, 540), (30, 30, 30), -1)
    printed = artwork.copy()
    printed[518:545, 435:470] = 255
    cv2.rectangle(printed, (446, 520), (466, 540), (30, 30, 30), -1)
    d = only(analyse(config, printed, artwork))
    assert d.category is DefectCategory.SHAPE_DIFFERENCE
    assert d.polarity == "mixed"


def test_edge_difference(config):
    printed = ARTWORK.copy()
    printed[150:152, 300:500] = (150, 60, 20)  # header ink spread 2 px along its lower edge
    d = only(analyse(config, printed))
    assert d.category is DefectCategory.EDGE_DIFFERENCE
    assert d.edge_fraction >= config.defects.edge_min_fraction
    assert d.thickness_px <= config.defects.edge_max_thickness_px


def test_possible_text_defect_is_never_verified(config):
    printed, truth = syn.add_defects(ARTWORK, ["broken_character"], seed=4)
    d = only(analyse(config, printed))
    assert d.category is DefectCategory.POSSIBLE_TEXT
    assert not d.category_certain
    assert "not verified" in d.category_note
    assert d.text_overlap >= 0.5
    assert d.category_label == "Possible text defect"


def test_possible_color_defect_is_never_verified(config):
    d = only(analyse(config, local_colour_patch()))
    assert d.category is DefectCategory.POSSIBLE_COLOR and not d.category_certain
    assert d.delta_c >= config.defects.color_min_delta_c
    assert abs(d.delta_l) <= config.defects.color_max_lightness_ratio * d.delta_c


def test_unknown_visual_difference(config):
    tinted = ARTWORK.copy()
    tinted[480:580, 380:560] = (225, 228, 230)  # tinted stock, no ink
    printed = tinted.copy()
    cv2.circle(printed, (470, 530), 6, (255, 255, 255), -1)
    d = only(analyse(config, printed, tinted))
    assert d.category is DefectCategory.UNKNOWN and not d.category_certain
    assert d.category_label.endswith("(uncertain)")


def test_good_print_has_no_defects(config):
    report = analyse(config, ARTWORK, noise=2.5, jpeg=90)
    assert report.defects == () and report.filtered == ()


# ---------------------------------------------------------------- grouping and noise


def test_close_fragments_are_one_defect_distant_ones_are_not(config):
    for gap, expected in ((3, 1), (20, 2)):
        printed = ARTWORK.copy()
        cv2.circle(printed, (450, 530), 2, (30, 30, 30), -1)
        cv2.circle(printed, (454 + gap, 530), 2, (30, 30, 30), -1)
        assert len(analyse(config, printed).defects) == expected


def test_small_strong_dot_is_not_removed_as_noise(config):
    printed, _ = syn.add_defects(ARTWORK, ["small_dot"], seed=4)
    report = analyse(config, printed)
    assert len(report.defects) == 1 and report.filtered == ()


def test_noise_removal_follows_configured_thresholds(config):
    printed, _ = syn.add_defects(ARTWORK, ["small_dot"], seed=4)
    strict = dataclasses.replace(
        config.defects, noise_max_area_mm2=10.0, noise_max_score=1000.0, reliable_min_score=1000.0
    )
    report = analyse(config, printed, defects_cfg=strict)
    assert report.defects == ()
    assert len(report.filtered) == 1 and "isolated noise" in report.filtered[0].reason


def test_reliable_regions_are_never_removed_as_noise(config):
    printed, _ = syn.add_defects(ARTWORK, ["small_dot"], seed=4)
    cfg = dataclasses.replace(config.defects, noise_max_area_mm2=10.0, noise_max_score=1000.0)
    report = analyse(config, printed, defects_cfg=cfg)
    assert len(report.defects) == 1 and report.defects[0].is_reliable


# ---------------------------------------------------------------- measurements


def test_measurements_in_pixels_and_mm(config):
    printed, truth = syn.add_defects(ARTWORK, ["missing_region"], seed=4)
    d = only(analyse(config, printed))
    x, y, w, h = truth[0].bbox  # 61 x 51 px = 5.1 x 4.25 mm
    # The detected box includes the blur halo; the half-maximum core is the defect itself.
    assert w <= d.width <= w + 14 and h <= d.height <= h + 14
    assert d.length_px == pytest.approx(w, abs=3) and d.thickness_px == pytest.approx(h, abs=3)
    assert d.length_mm == pytest.approx(d.length_px / PX_PER_MM)
    assert d.area_mm2 == pytest.approx(d.area_px / PX_PER_MM**2)
    assert d.area_mm2 == pytest.approx(w * h / PX_PER_MM**2, rel=0.1)
    assert d.length_px >= d.thickness_px
    assert x <= d.centroid[0] <= x + w and y <= d.centroid[1] <= y + h


def test_difference_magnitude(config):
    printed, _ = syn.add_defects(ARTWORK, ["extra_mark"], seed=4)
    d = only(analyse(config, printed))
    assert d.max_difference > 150 and d.mean_difference > 30
    assert d.signed_difference < 0 and d.delta_l < 0
    assert d.delta_e > 30 and d.score > 1.0 and d.peak_sigma > 0


# ---------------------------------------------------------------- severity and reliability


def test_severity_by_size(config):
    printed, _ = syn.add_defects(ARTWORK, ["extra_mark"], seed=4)  # ~2 mm splash
    assert only(analyse(config, printed)).severity is DefectSeverity.CRITICAL
    small = ARTWORK.copy()
    cv2.circle(small, (450, 530), 1, (30, 30, 30), -1)  # ~0.25 mm speck
    assert only(analyse(config, small)).severity is DefectSeverity.MINOR
    lenient = dataclasses.replace(
        config.defects, severity_major_size_mm=50, severity_critical_size_mm=60,
        severity_major_area_mm2=50, severity_critical_area_mm2=60,
    )
    assert only(analyse(config, printed, defects_cfg=lenient)).severity is DefectSeverity.MINOR


def test_text_has_stricter_size_limit(config):
    printed, _ = syn.add_defects(ARTWORK, ["broken_character"], seed=4)
    d = only(analyse(config, printed))
    assert d.severity is not DefectSeverity.MINOR
    relaxed = dataclasses.replace(config.defects, text_major_size_mm=5.0, severity_major_size_mm=5.0, severity_major_area_mm2=5.0)
    assert only(analyse(config, printed, defects_cfg=relaxed)).severity is DefectSeverity.MINOR


def test_faint_differences_are_one_level_lower(config):
    printed, _ = syn.add_defects(ARTWORK, ["extra_mark"], seed=4)
    cfg = dataclasses.replace(config.defects, faint_difference=255.0)
    d = only(analyse(config, printed, defects_cfg=cfg))
    assert d.severity is DefectSeverity.MAJOR
    assert any("faint" in note for note in d.severity_notes)


def test_reliability_is_separate_from_category(config):
    printed, _ = syn.add_defects(ARTWORK, ["extra_mark"], seed=4)
    d = only(analyse(config, printed))
    assert d.reliability is DetectionReliability.RELIABLE and d.category_certain
    strict = dataclasses.replace(config.defects, reliable_min_score=1000.0)
    u = only(analyse(config, printed, defects_cfg=strict))
    assert u.reliability is DetectionReliability.UNCERTAIN and not u.is_reliable
    assert u.category is DefectCategory.EXTRA_PRINT and u.category_certain  # category judged separately
    assert any("close to the detection threshold" in n for n in u.reliability_notes)


# ---------------------------------------------------------------- evidence, ordering, helpers


def test_evidence_crops(config):
    printed, _ = syn.add_defects(ARTWORK, ["extra_mark"], seed=4)
    d = only(analyse(config, printed))
    e = d.evidence
    shapes = {a.shape[:2] for a in (e.artwork, e.expected, e.printed, e.difference, e.score, e.region)}
    assert len(shapes) == 1
    ox, oy = e.origin
    h, w = e.artwork.shape[:2]
    assert np.array_equal(e.artwork, ARTWORK[oy : oy + h, ox : ox + w])
    assert e.region.max() == 255 and e.region[d.y - oy, :].max() == 255 or e.region.sum() > 0
    margin = config.defects.evidence_margin_px
    assert w >= d.width + margin and h >= d.height + margin


def test_most_severe_first_and_numbered(config):
    printed, _ = syn.add_defects(ARTWORK, ["small_dot", "missing_region", "missing_line"], seed=6)
    report = analyse(config, printed)
    ranks = [d.severity.rank for d in report.defects]
    assert ranks == sorted(ranks, reverse=True)
    assert [d.id for d in report.defects] == list(range(1, len(report.defects) + 1))
    assert report.by_id(1) is report.defects[0] and report.by_id(99) is None
    rows = defect_rows(report)
    assert rows[0]["id"] == 1 and set(rows[0]) >= {"category", "severity", "reliability", "length_mm"}


def test_text_like_mask_finds_text_not_barcodes(config):
    gray = cv2.cvtColor(ARTWORK, cv2.COLOR_BGR2GRAY)
    mask = text_like_mask(gray, PX_PER_MM, config.defects)
    assert mask[230:260, 60:300].mean() > 100  # "LOT 12345-A"
    assert mask[640:740, 60:500].mean() < 10  # barcode
    assert mask[30:120, 60:600].mean() > 100  # knock-out header text


def test_px_per_mm_from_dpi_or_assumed_size():
    assert grid_px_per_mm(1.5, (300.0, 300.0), 101.6, 1200, 150.0) == (pytest.approx(1.5 * 300 / 25.4), False)
    value, estimated = grid_px_per_mm(1.5, None, None, 1200, 150.0)
    assert estimated and value == pytest.approx(1.5 * 1200 / 150.0)


def test_rejects_bad_scale(config):
    printed, _ = syn.add_defects(ARTWORK, ["extra_mark"], seed=4)
    registered = simulate(printed)
    comparison = compare_to_artwork(registered, ARTWORK, full_mask(ARTWORK.shape), config.comparison, 0.9)
    with pytest.raises(ValueError):
        assess_defects(comparison, ARTWORK, registered, 0.0, False, config.defects)
