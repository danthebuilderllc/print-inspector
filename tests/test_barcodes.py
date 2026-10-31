"""The decoding layer (ZXing-C++) and barcode zone files."""
import builtins
import json

import numpy as np
import pytest

from print_inspector.core import barcodes
from print_inspector.core.barcode_zones import BarcodeZone, BarcodeZoneError, load_zones, save_zones, zones_path
from print_inspector.core.barcodes import (
    BarcodeUnavailable,
    canonical_value,
    check_available,
    decode,
    gtin_check_digit_ok,
    render,
    same_symbology,
)


def on_stock(symbol: np.ndarray, pad: int = 40) -> np.ndarray:
    image = np.full((symbol.shape[0] + 2 * pad, symbol.shape[1] + 2 * pad), 255, np.uint8)
    image[pad : pad + symbol.shape[0], pad : pad + symbol.shape[1]] = symbol
    return image


@pytest.mark.parametrize("fmt, value", [
    ("ean13", "4006381333931"), ("upca", "036000291452"), ("code128", "LOT-12345/A"), ("qr", "Ünïcødé ✓ 12345"),
])
def test_round_trip(fmt, value):
    (symbol,) = decode(on_stock(render(fmt, value, 3)))
    assert same_symbology(symbol.format, fmt)
    assert canonical_value(symbol.format, symbol.text) == canonical_value(fmt, value)
    x, y, w, h = symbol.box
    assert 30 <= x <= 80 and 30 <= y <= 80 and w > 50


def test_qr_details():
    (symbol,) = decode(on_stock(render("qr", "https://example.com", 4)))
    assert symbol.version == 2 and symbol.ec_level in "LMQH" and symbol.unused_ec == 1.0
    assert len(symbol.corners) == 4


def test_formats_are_restricted():
    image = on_stock(render("code128", "ABC-1", 3))
    assert decode(image, ("ean13", "qr")) == []
    assert len(decode(image, ("code128",))) == 1


def test_small_print_needs_the_thorough_retry():
    import cv2

    symbol = render("qr", "small", 2)
    blurred = cv2.GaussianBlur(cv2.resize(on_stock(symbol, 10), None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA), (0, 0), 0.6)
    found = decode(blurred)
    assert [s.text for s in found] == ["small"]
    assert decode(np.full((5, 5), 255, np.uint8)) == []


def test_gtin_helpers():
    assert canonical_value("upca", "036000291452") == "0036000291452"
    assert canonical_value("ean13", "0036000291452") == "0036000291452"
    assert canonical_value("code128", "  a b ") == "  a b "
    assert gtin_check_digit_ok("4006381333931") and gtin_check_digit_ok("036000291452")
    assert not gtin_check_digit_ok("4006381333932") and not gtin_check_digit_ok("12345")
    assert same_symbology("ean13", "upca") and not same_symbology("ean13", "code128")


def test_invalid_content_cannot_be_rendered():
    with pytest.raises(ValueError, match="EAN-13"):
        render("ean13", "4006381333932")


def test_missing_library_is_reported(monkeypatch):
    real_import = builtins.__import__

    def no_zxing(name, *args, **kwargs):
        if name == "zxingcpp":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_zxing)
    with pytest.raises(BarcodeUnavailable, match="zxing-cpp is not installed"):
        decode(np.zeros((20, 20), np.uint8))
    with pytest.raises(BarcodeUnavailable):
        check_available()


def test_library_version():
    assert barcodes.library_version().startswith("ZXing-C++ ")


# ---------------------------------------------------------------- zone files


def test_zone_round_trip(tmp_path):
    path = zones_path(tmp_path / "label.png")
    assert path.name == "label.png.barcodes.json" and load_zones(path) == ()
    zones = (
        BarcodeZone("GTIN", 0.05, 0.05, 0.3, 0.25, "ean13", "4006381333931"),
        BarcodeZone("Link", 0.6, 0.4, 0.3, 0.4, "qr", "", critical=False),
        BarcodeZone("Anything", 0.0, 0.8, 0.5, 0.2),
    )
    save_zones(path, zones)
    assert load_zones(path) == zones
    assert json.loads(path.read_text(encoding="utf-8"))["zones"][0]["format"] == "ean13"
    save_zones(path, ())
    assert not path.exists()


@pytest.mark.parametrize("zone, message", [
    (BarcodeZone("", 0, 0, 0.1, 0.1), "no name"),
    (BarcodeZone("a", 0, 0, 0.1, 0.1, "pdf417"), "unsupported format"),
    (BarcodeZone("a", 0.95, 0, 0.1, 0.1), "beyond"),
    (BarcodeZone("a", 0, 0, 0.1, 0.1, "ean13", "4006381333932"), "check digit"),
    (BarcodeZone("a", 0, 0, 0.1, 0.1, "upca", "03600029145"), "12 or 13 digits"),
])
def test_invalid_zones(tmp_path, zone, message):
    with pytest.raises(BarcodeZoneError, match=message):
        save_zones(tmp_path / "a.png.barcodes.json", (zone,))


def test_unreadable_zone_file(tmp_path):
    path = tmp_path / "a.png.barcodes.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(BarcodeZoneError, match="'zones' list"):
        load_zones(path)
