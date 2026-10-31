"""Barcode and QR code inspection on synthetic labels: correct, damaged and incorrect codes.

The codes are rendered with the same local library (ZXing-C++); the print is
simulated on the comparison grid (blur, noise) with damage painted in.
"""
import dataclasses

import numpy as np
import pytest

from print_inspector.config import load_config
from print_inspector.core import barcode_inspection
from print_inspector.core.barcode_inspection import (
    BarcodeFindingKind as K,
    SymbolStatus,
    inspect_barcodes,
    unavailable_report,
)
from print_inspector.core.barcode_zones import BarcodeZone
from print_inspector.core.barcodes import render
from print_inspector.core.comparison import compare_to_artwork
from print_inspector.core.decision import OperatorMark, decide
from print_inspector.core.defects import assess_defects
from print_inspector.core.models import InspectionStatus, QualityLevel
from tests.barcode_fakes import SIZE, STOCK, make_barcode_label, paint, place
from tests.test_comparison import full_mask, simulate


@pytest.fixture(scope="module")
def config():
    return load_config(user_path=None)


@pytest.fixture(scope="module")
def label():
    return make_barcode_label()


def run(config, artwork, printed, zones=(), cfg=None, seed=0):
    registered = simulate(printed, seed=seed)
    mask = full_mask(artwork.shape)
    comparison = compare_to_artwork(registered, artwork, mask, config.comparison, 0.9)
    report = inspect_barcodes(artwork, registered, comparison.combined_score, mask, tuple(zones), cfg or config.barcodes)
    return report, comparison, registered


def verdict(config, artwork, comparison, registered, report, marks=None):
    defects = assess_defects(comparison, artwork, registered, 12.0, False, config.defects)
    dismissed = {d.id: OperatorMark.DISMISSED for d in defects.defects}  # judge the barcode findings alone
    return decide(None, None, None, comparison, defects, dismissed, config.decision, None, None, report, marks or {})


def symbol(report, name):
    return next(s for s in report.symbols if s.name == name)


def kinds(report):
    return sorted({f.kind for f in report.findings}, key=lambda k: k.value)


def zone_around(box, name="zone", pad=40, **kwargs):
    x, y, w, h = box
    return BarcodeZone(name, (x - pad) / SIZE[0], (y - pad) / SIZE[1], (w + 2 * pad) / SIZE[0], (h + 2 * pad) / SIZE[1], **kwargs)


# ---------------------------------------------------------------- correct codes


def test_correct_codes_decode_and_match(config, label):
    report, comparison, registered = run(config, label.image, label.image.copy())
    assert report.available and report.library.startswith("ZXing-C++")
    assert [s.name for s in report.symbols] == ["EAN-13 1", "Code 128 1", "QR Code 1"]
    assert [s.status for s in report.symbols] == [SymbolStatus.OK] * 3
    assert symbol(report, "EAN-13 1").print_value == "4006381333931"
    assert symbol(report, "Code 128 1").print_value == "LOT-12345/A"
    assert symbol(report, "QR Code 1").print_value == "https://example.com/p?id=12345"
    assert symbol(report, "EAN-13 1").strips == (10, 10)
    qr = symbol(report, "QR Code 1")
    assert qr.unused_ec == 1.0 and qr.module_errors == 0 and qr.module_count == 25 * 25
    assert abs(symbol(report, "EAN-13 1").bar_growth_modules) < 0.2
    assert report.findings == ()
    assert verdict(config, label.image, comparison, registered, report).status is InspectionStatus.PASS


def test_upc_a_is_compared_as_gtin(config):
    upc = make_barcode_label(("upc",))
    report, *_ = run(config, upc.image, upc.image.copy(), zones=[zone_around(upc.boxes["upc"], "UPC", format="upca", expected_value="036000291452")])
    s = symbol(report, "UPC")
    assert s.status is SymbolStatus.OK and s.expected_source == "zone"
    assert s.print_value.endswith("036000291452")


def test_rotated_code_is_analysed(config):
    artwork = np.full((SIZE[1], SIZE[0], 3), STOCK, np.uint8)
    place(artwork, np.ascontiguousarray(np.rot90(render("code128", "ROTATED-42", 3))), (500, 150))
    printed = artwork.copy()
    report, *_ = run(config, artwork, printed)
    (s,) = report.symbols
    assert s.status is SymbolStatus.OK and s.print_value == "ROTATED-42" and s.strips == (10, 10)


# ---------------------------------------------------------------- damaged codes


def test_bar_gap_through_the_whole_code_is_not_decodable(config, label):
    printed = label.image.copy()
    x, y, w, h = label.boxes["ean"]
    paint(printed, (x + 60, y, 12, h))
    report, comparison, registered = run(config, label.image, printed)
    s = symbol(report, "EAN-13 1")
    assert s.status is SymbolStatus.NOT_DECODED and s.strips[0] == 0
    not_decoded = next(f for f in report.findings if f.kind is K.NOT_DECODED)
    assert not_decoded.reliable and not_decoded.severity.value == "critical"
    assert "damage visible" in not_decoded.notes[0] and "not an ISO" in not_decoded.notes[1]
    bars = next(f for f in report.findings if f.kind is K.DAMAGED_BARS)
    assert bars.reliable and x + 55 <= bars.x <= x + 65 and bars.width <= 20
    decision = verdict(config, label.image, comparison, registered, report)
    assert decision.status is InspectionStatus.FAIL and not_decoded.id in decision.barcode_failing


def test_local_bar_damage_still_decodes_but_is_reported(config, label):
    printed = label.image.copy()
    x, y, w, h = label.boxes["ean"]
    paint(printed, (x + 60, y + 40, 12, 40))
    report, *_ = run(config, label.image, printed)
    s = symbol(report, "EAN-13 1")
    assert s.status is SymbolStatus.DAMAGED and s.print_value == "4006381333931"
    assert s.strips[0] < s.strips[1]
    (f,) = report.findings
    assert f.kind is K.DAMAGED_BARS and f.reliable and f.severity.value == "major"  # stops the strips through it
    assert f.y >= y + 30 and f.y + f.height <= y + 90


def test_small_qr_damage_is_minor_while_error_correction_copes(config, label):
    printed = label.image.copy()
    x, y, w, h = label.boxes["qr"]
    paint(printed, (x + 60, y + 60, 40, 40))  # 5 x 5 modules
    report, comparison, registered = run(config, label.image, printed)
    s = symbol(report, "QR Code 1")
    assert s.status is SymbolStatus.DAMAGED and s.print_value == "https://example.com/p?id=12345"
    assert 0 < s.module_errors <= 25 and s.unused_ec < 1.0
    (f,) = report.findings
    assert f.kind is K.DAMAGED_MODULES and f.reliable and f.severity.value == "minor"
    assert x + 50 <= f.x <= x + 110
    assert verdict(config, label.image, comparison, registered, report).status is InspectionStatus.PASS


def test_large_qr_damage_is_not_decodable(config, label):
    printed = label.image.copy()
    x, y, w, h = label.boxes["qr"]
    paint(printed, (x + 30, y + 30, 120, 120))
    report, *_ = run(config, label.image, printed)
    assert symbol(report, "QR Code 1").status is SymbolStatus.NOT_DECODED
    assert kinds(report) == [K.DAMAGED_MODULES, K.NOT_DECODED]
    modules = [f for f in report.findings if f.kind is K.DAMAGED_MODULES]
    assert len(modules) == 1 and modules[0].severity.value == "major"  # one area, grouped


def test_low_qr_error_correction_reserve(config, label):
    printed = label.image.copy()
    x, y, w, h = label.boxes["qr"]
    paint(printed, (x + 60, y + 60, 40, 40))
    strict = dataclasses.replace(config.barcodes, min_unused_ec=0.9)
    report, *_ = run(config, label.image, printed, cfg=strict)
    margin = [f for f in report.findings if f.kind is K.LOW_DECODE_MARGIN]
    assert len(margin) == 1 and margin[0].reliable  # explained by the visible damage
    assert all(f.severity.value == "major" for f in report.findings if f.kind is K.DAMAGED_MODULES)


def test_mark_in_the_quiet_zone(config, label):
    printed = label.image.copy()
    x, y, w, h = label.boxes["code128"]
    paint(printed, (x + w + 8, y + 70, 10, 12), 20)
    report, *_ = run(config, label.image, printed)
    (f,) = report.findings
    assert f.kind is K.QUIET_ZONE and f.reliable and f.x >= x + w
    assert symbol(report, "Code 128 1").status is SymbolStatus.DAMAGED


def test_spread_bars_are_measured(config, label):
    printed = label.image.copy()
    x, y, w, h = label.boxes["ean"]
    region = printed[y : y + h, x - 2 : x + w + 2]
    dark = (region[:, :, 0] < 128).astype(np.uint8)
    import cv2

    spread = cv2.dilate(dark, np.ones((1, 3), np.uint8))  # every bar one pixel wider on each side (2/3 module)
    region[spread > 0] = 20
    report, *_ = run(config, label.image, printed)
    s = symbol(report, "EAN-13 1")
    assert s.bar_growth_modules > 0.4
    assert K.BAR_WIDTH in kinds(report)
    assert not next(f for f in report.findings if f.kind is K.BAR_WIDTH).reliable


# ---------------------------------------------------------------- incorrect content


def test_wrong_ean_content_fails(config, label):
    printed = make_barcode_label(values={"ean": "5901234123457"}).image
    report, comparison, registered = run(config, label.image, printed)
    s = symbol(report, "EAN-13 1")
    assert s.status is SymbolStatus.WRONG_CONTENT and s.print_value == "5901234123457"
    wrong = next(f for f in report.findings if f.kind is K.WRONG_CONTENT)
    assert wrong.reliable and (wrong.expected, wrong.decoded) == ("4006381333931", "5901234123457")
    assert verdict(config, label.image, comparison, registered, report).status is InspectionStatus.FAIL


def test_wrong_qr_content_fails(config, label):
    printed = make_barcode_label(values={"qr": "https://example.com/p?id=99999"}).image
    report, *_ = run(config, label.image, printed)
    assert symbol(report, "QR Code 1").status is SymbolStatus.WRONG_CONTENT


def test_zone_expected_value_is_checked(config, label):
    zone = zone_around(label.boxes["code128"], "Lot code", format="code128", expected_value="LOT-99999/A")
    report, comparison, registered = run(config, label.image, label.image.copy(), zones=[zone])
    s = symbol(report, "Lot code")
    assert s.status is SymbolStatus.WRONG_CONTENT and s.expected_source == "zone" and s.critical
    assert {f.kind for f in report.findings if f.symbol_id == s.id} == {K.WRONG_CONTENT, K.ARTWORK_MISMATCH}
    mismatch = next(f for f in report.findings if f.kind is K.ARTWORK_MISMATCH)
    assert not mismatch.reliable and mismatch.severity.value == "critical"


def test_wrong_barcode_type(config, label):
    zone = zone_around(label.boxes["code128"], "Lot code", format="qr")
    report, *_ = run(config, label.image, label.image.copy(), zones=[zone])
    s = symbol(report, "Lot code")
    assert s.status is SymbolStatus.WRONG_FORMAT
    assert next(f for f in report.findings if f.symbol_id == s.id).kind is K.WRONG_FORMAT


def test_unexpected_barcode_in_the_print(config):
    artwork = make_barcode_label(("ean",)).image
    printed = make_barcode_label(("ean", "qr")).image
    report, *_ = run(config, artwork, printed)
    extra = [f for f in report.findings if f.kind is K.UNEXPECTED_BARCODE]
    assert len(extra) == 1 and extra[0].decoded == "https://example.com/p?id=12345"


def test_required_zone_without_barcode_in_print(config, label):
    blank = np.full_like(label.image, STOCK)
    zone = BarcodeZone("GTIN", 0.05, 0.05, 0.3, 0.3, "ean13", "4006381333931")
    report, comparison, registered = run(config, blank, blank.copy(), zones=[zone])
    s = symbol(report, "GTIN")
    assert s.status is SymbolStatus.NOT_DECODED and "zone value" in s.notes[0]
    f = next(f for f in report.findings if f.kind is K.NOT_DECODED)
    assert not f.reliable  # nothing visible explains it
    assert verdict(config, blank, comparison, registered, report).status is InspectionStatus.REVIEW


# ---------------------------------------------------------------- decoding failures without visible damage


def test_decoding_failure_without_damage_needs_review(config, label, monkeypatch):
    real = barcode_inspection.decode
    calls = {"n": 0}

    def flaky(image, formats=(), thorough=True):
        calls["n"] += 1
        return real(image, formats, thorough) if calls["n"] == 1 else []  # the artwork decodes, the photo not

    monkeypatch.setattr(barcode_inspection, "decode", flaky)
    report, comparison, registered = run(config, label.image, label.image.copy())
    not_decoded = [f for f in report.findings if f.kind is K.NOT_DECODED]
    assert len(not_decoded) == 3 and not any(f.reliable for f in not_decoded)
    assert "photo may be too blurred" in not_decoded[0].notes[0]
    decision = verdict(config, label.image, comparison, registered, report)
    assert decision.status is InspectionStatus.REVIEW and set(decision.barcode_review) == {f.id for f in not_decoded}
    assert "scanner" in " ".join(decision.reasons)
    # The operator checked the codes with a scanner.
    marks = {f.id: OperatorMark.DISMISSED for f in not_decoded}
    assert verdict(config, label.image, comparison, registered, report, marks).status is InspectionStatus.PASS


def test_unavailable_library_gives_review(config, label):
    report = unavailable_report("zxing-cpp is not installed.", True)
    assert not report.available and report.level is QualityLevel.POOR
    _, comparison, registered = run(config, label.image, label.image.copy())
    decision = verdict(config, label.image, comparison, registered, report)
    assert decision.status is InspectionStatus.REVIEW and "not available" in decision.reasons[0]
    assert unavailable_report("x", False).level is QualityLevel.MARGINAL
