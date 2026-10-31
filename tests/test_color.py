"""Colour spaces and CIEDE2000."""
import numpy as np
import pytest

from print_inspector.core.color import (
    bgr_to_lab,
    delta_e_2000,
    describe_shift,
    lab_to_bgr,
    lab_to_xyz,
    linear_to_srgb,
    srgb_to_linear,
    xyz_to_lab,
    WHITE_D50,
)

# Test data of Sharma, Wu and Dalal (2005), "The CIEDE2000 color-difference formula".
SHARMA = [
    ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
    ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
    ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
    ((50.0, -1.0, 2.0), (50.0, 0.0, 0.0), 2.3669),
    ((50.0, 2.4900, -0.0010), (50.0, -2.4900, 0.0009), 7.1792),
    ((50.0, 2.4900, -0.0010), (50.0, -2.4900, 0.0011), 7.2195),
    ((50.0, -0.0010, 2.4900), (50.0, 0.0011, -2.4900), 4.7461),
    ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
    ((50.0, 2.5, 0.0), (61.0, -5.0, 29.0), 22.8977),
    ((50.0, 2.5, 0.0), (56.0, -27.0, -3.0), 31.9030),
    ((50.0, 2.5, 0.0), (50.0, 3.1736, 0.5854), 1.0000),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
    ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
]


@pytest.mark.parametrize("lab1, lab2, expected", SHARMA)
def test_ciede2000_reference_data(lab1, lab2, expected):
    assert float(delta_e_2000(np.array(lab1), np.array(lab2))) == pytest.approx(expected, abs=1e-4)
    assert float(delta_e_2000(np.array(lab2), np.array(lab1))) == pytest.approx(expected, abs=1e-4)


def test_srgb_white_and_black_in_d50():
    white = bgr_to_lab(np.array([255, 255, 255], np.uint8))
    assert white == pytest.approx([100.0, 0.0, 0.0], abs=1e-3)  # adapted to D50: neutral stays neutral
    assert bgr_to_lab(np.array([0, 0, 0], np.uint8)) == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    gray = bgr_to_lab(np.array([119, 119, 119], np.uint8))
    assert gray[0] == pytest.approx(50.0, abs=0.5) and abs(gray[1]) < 1e-3 and abs(gray[2]) < 1e-3


def test_srgb_red_matches_the_icc_value():
    # sRGB red in CIELAB D50 (ICC sRGB profile, Bradford): about L 54.3, a 80.8, b 69.9
    red = bgr_to_lab(np.array([0, 0, 255], np.uint8))
    assert red == pytest.approx([54.29, 80.80, 69.89], abs=0.1)


def test_round_trips():
    rng = np.random.default_rng(1)
    bgr = rng.integers(0, 256, (500, 3)).astype(np.uint8)
    back = lab_to_bgr(bgr_to_lab(bgr))
    assert np.abs(back - bgr).max() < 0.01
    lab = np.array([[30.0, 20.0, -40.0], [90.0, -5.0, 10.0], [5.0, 1.0, 1.0]])
    assert xyz_to_lab(lab_to_xyz(lab)) == pytest.approx(lab, abs=1e-9)
    assert xyz_to_lab(WHITE_D50) == pytest.approx([100.0, 0.0, 0.0], abs=1e-9)
    values = np.arange(256, dtype=np.float64)
    assert linear_to_srgb(srgb_to_linear(values)) == pytest.approx(values, abs=1e-9)


def test_describe_shift():
    assert describe_shift(-5, 0, 3) == "darker, yellower"
    assert describe_shift(0.5, -0.5, 0.2) == "about the same"
    assert describe_shift(3, -3, -3) == "lighter, greener, bluer"
