"""Camera colour calibration with a reference chart."""
import json

import numpy as np
import pytest

from print_inspector.core.color import bgr_to_lab, delta_e_2000, xyz_to_lab
from print_inspector.core.color_calibration import (
    COLORCHECKER_24,
    SAME_PHOTO,
    SEPARATE_PHOTO,
    CalibrationError,
    calibrate,
    chart_field_corners,
    load_chart_reference,
    load_profile,
    patch_centres,
    render_chart,
    save_profile,
)
from tests.color_fakes import camera

CHART = COLORCHECKER_24
WARM = dict(gains=(1.15, 1.0, 0.8), exposure=0.85, crosstalk=0.1)


def test_patch_centres_follow_the_corners():
    centres = patch_centres(np.array([[0, 0], [60, 0], [60, 40], [0, 40]]), 4, 6)
    assert centres[0] == pytest.approx([5, 5]) and centres[5] == pytest.approx([55, 5])
    assert centres[-1] == pytest.approx([55, 35]) and len(centres) == 24


def test_calibration_corrects_a_simulated_camera():
    photo = camera(render_chart(CHART), **WARM)
    profile = calibrate(photo, chart_field_corners(CHART), CHART, SAME_PHOTO, "chart.jpg")
    assert profile.fit_mean_delta_e < 1.0
    assert profile.loo_mean_delta_e < 1.5 and len(profile.patch_delta_e) == 24
    assert profile.chart_nominal and any("Nominal" in n for n in profile.notes)
    # Without calibration the same camera is several ΔE00 off on the chart.
    raw = bgr_to_lab(photo[np.round(patch_centres(chart_field_corners(CHART), 4, 6)[:, 1]).astype(int),
                           np.round(patch_centres(chart_field_corners(CHART), 4, 6)[:, 0]).astype(int)])
    assert float(np.mean(delta_e_2000(raw, CHART.lab_array()))) > 4.0
    # Applied to the chart photo, the profile reproduces the reference values.
    from print_inspector.core.color_calibration import sample_chart

    rgb, _ = sample_chart(photo, chart_field_corners(CHART), 4, 6)
    corrected = xyz_to_lab(profile.apply(rgb))
    assert float(np.mean(delta_e_2000(corrected, CHART.lab_array()))) < 1.0


def test_wrong_corner_order_is_detected():
    photo = render_chart(CHART)
    corners = chart_field_corners(CHART)[[2, 3, 0, 1]]  # chart upside down
    with pytest.raises(CalibrationError, match="lightness order"):
        calibrate(photo, corners, CHART, SAME_PHOTO, "x")


def test_misplaced_corners_are_detected():
    photo = render_chart(CHART)
    corners = chart_field_corners(CHART) + np.array([30.0, 30.0])  # half a patch off: samples straddle edges
    with pytest.raises(CalibrationError):
        calibrate(photo, corners, CHART, SAME_PHOTO, "x")


def test_chart_outside_photo():
    with pytest.raises(CalibrationError, match="outside"):
        calibrate(render_chart(CHART), chart_field_corners(CHART) * 3, CHART, SAME_PHOTO, "x")


def test_profile_round_trip(tmp_path):
    profile = calibrate(camera(render_chart(CHART), **WARM), chart_field_corners(CHART), CHART, SEPARATE_PHOTO, "c.jpg")
    path = tmp_path / "profile.json"
    assert load_profile(path) is None
    save_profile(path, profile)
    assert load_profile(path) == profile
    assert "leave-one-out" in profile.describe() and "separate photo" in profile.describe()
    path.write_text('{"version": 1}', encoding="utf-8")
    with pytest.raises(CalibrationError, match="cannot be read"):
        load_profile(path)


def test_chart_reference_file(tmp_path):
    data = {
        "name": "My chart #1234", "rows": 4, "cols": 6, "illuminant": "D50",
        "patches": [{"name": n, "lab": list(v)} for n, v in zip(CHART.names, CHART.lab)],
    }
    path = tmp_path / "chart.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    chart = load_chart_reference(path)
    assert chart.name == "My chart #1234" and not chart.nominal and chart.lab == CHART.lab
    data["illuminant"] = "D65"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CalibrationError, match="D50"):
        load_chart_reference(path)
    data["illuminant"] = "D50"
    data["patches"] = data["patches"][:10]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CalibrationError):
        load_chart_reference(path)
