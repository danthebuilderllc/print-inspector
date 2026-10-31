"""Text inspection: OCR results combined with the character-level image check.

The print is simulated on the comparison grid with real defects painted into
the image; the OCR engine is the scripted stand-in from ``ocr_fakes`` (what a
real OCR would read). With Tesseract installed, ``test_ocr.py`` also runs the
real engine.
"""
import cv2
import numpy as np
import pytest

from print_inspector.config import load_config
from print_inspector.core.comparison import compare_to_artwork
from print_inspector.core.decision import OperatorMark, decide
from print_inspector.core.defects import assess_defects
from print_inspector.core.models import InspectionStatus, QualityLevel
from print_inspector.core.text_inspection import (
    LineStatus,
    TextFindingKind as K,
    glyph_losses,
    group_lines,
    inspect_text,
    prepare_for_ocr,
    unavailable_report,
)
from print_inspector.core.ocr import OcrWord
from print_inspector.core.text_zones import TextZone
from tests.ocr_fakes import SIZE, ScriptedOcr, erase, make_text_label
from tests.test_comparison import full_mask, simulate

PX_PER_MM = 12.0
LOT, EXP, NET, NAME = 0, 1, 2, 3


@pytest.fixture(scope="module")
def config():
    return load_config(user_path=None)


@pytest.fixture(scope="module")
def label():
    return make_text_label()


def run(config, label, printed, engine, zones=(), text_cfg=None, seed=0):
    registered = simulate(printed, seed=seed)
    mask = full_mask(label.image.shape)
    comparison = compare_to_artwork(registered, label.image, mask, config.comparison, 0.9)
    report = inspect_text(
        label.image, registered, comparison.expected, comparison.combined_score, mask, PX_PER_MM,
        tuple(zones), engine, text_cfg or config.text, config.defects,
    )
    return report, comparison, registered


def verdict(config, label, comparison, registered, report, marks=None):
    defects = assess_defects(comparison, label.image, registered, PX_PER_MM, False, config.defects)
    return decide(None, None, None, comparison, defects, {}, config.decision, report, marks or {})


def inside(finding, box, pad=6):
    x, y, w, h = box
    return finding.x - pad <= x + w / 2 <= finding.x + finding.width + pad and finding.y - pad <= y + h / 2 <= finding.y + finding.height + pad


def only(report):
    assert len(report.findings) == 1, [(f.kind, f.expected, f.read, f.reliable) for f in report.findings]
    return report.findings[0]


# ---------------------------------------------------------------- clean print


def test_clean_print_matches(config, label):
    report, comparison, registered = run(config, label, label.image.copy(), ScriptedOcr(label))
    assert report.available and report.engine == "Scripted OCR"
    assert report.findings == ()
    assert [line.status for line in report.lines] == [LineStatus.MATCH] * 4
    assert [line.expected_text for line in report.lines] == label.lines
    assert all(line.artwork_confidence == 95 and line.print_confidence == 95 for line in report.lines)
    assert [line.critical for line in report.lines] == [True, True, True, False]  # digits -> critical
    assert verdict(config, label, comparison, registered, report).status is InspectionStatus.PASS


# ---------------------------------------------------------------- the required text defects


def test_missing_character_in_number(config, label):
    printed = label.image.copy()
    erase(printed, label.char_box(LOT, 8))  # the 5 of LOT 12345
    report, comparison, registered = run(config, label, printed, ScriptedOcr(label, reads={LOT: ("LOT 1234", 92)}))
    f = only(report)
    assert f.kind is K.MISSING_CHARACTER and f.expected == "5" and f.read == ""
    assert f.reliable and f.severity.value == "critical" and f.critical
    assert inside(f, label.char_box(LOT, 8))
    assert f.ink_loss > 0.9 and f.print_confidence == 92 and f.artwork_confidence == 95
    assert report.line(f.line_id).status is LineStatus.MISMATCH
    decision = verdict(config, label, comparison, registered, report)
    assert decision.status is InspectionStatus.FAIL and f.id in decision.text_failing


def test_incorrect_number(config, label):
    printed = label.image.copy()
    x, y, w, h = label.char_box(LOT, 6)  # the 3
    erase(printed, (x, y, w, h))
    cv2.putText(printed, "8", (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 20, 20), 3, cv2.LINE_AA)
    report, *_ = run(config, label, printed, ScriptedOcr(label, reads={LOT: ("LOT 12845", 90)}))
    f = only(report)
    assert (f.kind, f.expected, f.read) == (K.INCORRECT_NUMBER, "3", "8")
    assert f.reliable and inside(f, label.char_box(LOT, 6))


def test_missing_punctuation(config, label):
    printed = label.image.copy()
    erase(printed, label.char_box(EXP, 8))  # the /
    report, *_ = run(config, label, printed, ScriptedOcr(label, reads={EXP: ("EXP 2027 06", 90)}))
    f = only(report)
    assert (f.kind, f.expected) == (K.MISSING_PUNCTUATION, "/")
    assert f.reliable and f.severity.value == "critical"  # a date: punctuation inside a number
    assert inside(f, label.char_box(EXP, 8))


def test_missing_word(config, label):
    printed = label.image.copy()
    for k in (4, 5, 6):  # "wt."
        erase(printed, label.char_box(NET, k))
    report, *_ = run(config, label, printed, ScriptedOcr(label, reads={NET: ("Net 250 g", 91)}))
    f = only(report)
    assert (f.kind, f.expected) == (K.MISSING_WORD, "wt.")
    assert f.reliable
    word_box = label.word_boxes(NET)[1][1]
    assert f.x <= word_box[0] + 4 and f.x + f.width >= word_box[0] + word_box[2] - 4


def test_partially_printed_character_found_by_the_image(config, label):
    printed = label.image.copy()
    erase(printed, label.char_box(NAME, 1), fraction_y=(0.5, 1.0))  # lower half of the R
    report, comparison, registered = run(config, label, printed, ScriptedOcr(label))  # OCR still reads it
    f = only(report)
    assert f.kind is K.PARTIAL_CHARACTER and f.expected == "R"
    assert 0.25 < f.ink_loss < 0.8 and f.reliable
    assert f.severity.value == "major" and not f.critical
    assert inside(f, label.char_box(NAME, 1))
    assert verdict(config, label, comparison, registered, report).status is InspectionStatus.FAIL


# ---------------------------------------------------------------- uncertainty -> REVIEW, not FAIL


def test_ocr_difference_without_image_evidence_is_uncertain(config, label):
    report, comparison, registered = run(
        config, label, label.image.copy(), ScriptedOcr(label, reads={NAME: ("PRINT INSPEKTOR", 88)})
    )
    f = only(report)
    assert (f.kind, f.expected, f.read) == (K.INCORRECT_CHARACTER, "C", "K")
    assert not f.reliable and "image comparison shows no difference" in f.notes[0]
    decision = verdict(config, label, comparison, registered, report)
    assert decision.status is InspectionStatus.REVIEW and decision.text_review == (f.id,)
    # The operator decides.
    assert verdict(config, label, comparison, registered, report, {f.id: OperatorMark.DISMISSED}).status is InspectionStatus.PASS
    assert verdict(config, label, comparison, registered, report, {f.id: OperatorMark.CONFIRMED}).status is InspectionStatus.FAIL


def test_typical_ocr_confusion_is_uncertain(config, label):
    report, comparison, registered = run(config, label, label.image.copy(), ScriptedOcr(label, reads={LOT: ("LOT I2345", 90)}))
    f = only(report)
    assert f.kind is K.INCORRECT_NUMBER and not f.reliable
    assert any("OCR confusion" in n for n in f.notes)
    assert verdict(config, label, comparison, registered, report).status is InspectionStatus.REVIEW


def test_low_print_confidence_on_critical_text_needs_review(config, label):
    report, comparison, registered = run(config, label, label.image.copy(), ScriptedOcr(label, reads={LOT: ("L0T 1Z34S", 41)}))
    assert report.findings and not any(f.reliable for f in report.findings)
    assert verdict(config, label, comparison, registered, report).status is InspectionStatus.REVIEW


def test_low_print_confidence_on_ordinary_text_with_clean_image_is_only_noted(config, label):
    report, comparison, registered = run(config, label, label.image.copy(), ScriptedOcr(label, reads={NAME: ("PRlNT lNSPECT0R", 35)}))
    assert report.findings == ()
    line = report.lines[NAME]
    assert line.status is LineStatus.UNCERTAIN and "image comparison found no difference" in line.notes[0]
    assert verdict(config, label, comparison, registered, report).status is InspectionStatus.PASS


def test_unreadable_print_text_in_critical_line_is_a_finding(config, label):
    report, comparison, registered = run(config, label, label.image.copy(), ScriptedOcr(label, reads={EXP: ("", 0)}))
    f = only(report)
    assert f.kind is K.UNREADABLE_TEXT and not f.reliable and f.critical
    assert report.line(f.line_id).status is LineStatus.UNCERTAIN
    assert verdict(config, label, comparison, registered, report).status is InspectionStatus.REVIEW


def test_artwork_ocr_uncertain_falls_back_to_the_image(config, label):
    printed = label.image.copy()
    erase(printed, label.char_box(NAME, 3), fraction_y=(0.0, 0.6))  # top of the N
    engine = ScriptedOcr(label, artwork_reads={NAME: ("PR1NT 1NSPECTOR", 30)}, reads={NAME: ("PR1NT 1NSPECTOR", 30)})
    report, *_ = run(config, label, printed, engine)
    line = report.lines[NAME]
    assert "could not read the artwork" in line.notes[0]
    (f,) = report.findings
    assert f.kind is K.PARTIAL_CHARACTER and f.reliable  # found by the image check alone
    assert inside(f, label.char_box(NAME, 3))


def test_ocr_errors_are_reported(config, label):
    report, comparison, registered = run(config, label, label.image.copy(), ScriptedOcr(label, fail={NAME}))
    assert "ocr_errors" in [i.code for i in report.issues] and report.level is QualityLevel.MARGINAL
    assert report.lines[NAME].status is LineStatus.UNREADABLE
    assert verdict(config, label, comparison, registered, report).status is InspectionStatus.PASS  # not critical


def test_text_ocr_did_not_find_is_still_checked_by_image(config, label):
    printed = label.image.copy()
    erase(printed, label.char_box(NAME, 10), fraction_x=(0.0, 0.7))  # most of the S
    report, *_ = run(config, label, printed, ScriptedOcr(label, hidden={NAME}))
    assert len(report.lines) == 3
    assert any(box[1] <= label.line_box(NAME)[1] + 10 <= box[1] + box[3] for box in report.unread_regions)
    (f,) = report.findings
    assert f.kind in (K.PARTIAL_CHARACTER, K.MISSING_CHARACTER) and inside(f, label.char_box(NAME, 10))
    assert any("could not read" in n for n in f.notes)


# ---------------------------------------------------------------- critical text zones


def _zone_around(label, line, pad=12, **kwargs):
    x, y, w, h = label.line_box(line)
    return TextZone(
        kwargs.pop("name", "zone"), (x - pad) / SIZE[0], (y - pad) / SIZE[1],
        (w + 2 * pad) / SIZE[0], (h + 2 * pad) / SIZE[1], **kwargs,
    )


def test_critical_zone_is_stricter(config, label):
    printed = label.image.copy()
    erase(printed, label.char_box(NAME, 8), fraction_y=(0.0, 0.15))  # top of the S: about 14 % of its ink
    plain, *_ = run(config, label, printed, ScriptedOcr(label))
    assert plain.findings == ()  # below the normal 20 % limit
    zone = _zone_around(label, NAME, name="Product name")
    strict, *_ = run(config, label, printed, ScriptedOcr(label), zones=[zone])
    f = only(strict)  # above the 10 % limit for critical text
    assert f.kind is K.PARTIAL_CHARACTER and inside(f, label.char_box(NAME, 8))
    assert config.text.critical_partial_min_fraction <= f.ink_loss < config.text.partial_min_fraction
    assert f.zone == "Product name" and f.critical and f.severity.value == "critical"
    assert strict.lines[NAME].zone == "Product name" and strict.lines[NAME].critical


def test_zone_with_expected_text(config, label):
    printed = label.image.copy()
    erase(printed, label.char_box(LOT, 8))
    zone = _zone_around(label, LOT, name="Lot", expected_text="LOT 12345")
    report, *_ = run(config, label, printed, ScriptedOcr(label, reads={LOT: ("LOT 1234", 90)}), zones=[zone])
    zone_lines = [line for line in report.lines if line.zone == "Lot"]
    assert len(zone_lines) == 1 and zone_lines[0].expected_source == "zone"
    kinds = {f.kind for f in report.findings if f.zone == "Lot"}
    assert K.MISSING_CHARACTER in kinds
    assert any(f.reliable for f in report.findings if f.zone == "Lot")


def test_artwork_differing_from_expected_text_needs_review(config, label):
    zone = _zone_around(label, LOT, name="Lot", expected_text="LOT 99999")
    report, comparison, registered = run(config, label, label.image.copy(), ScriptedOcr(label), zones=[zone])
    mismatch = [f for f in report.findings if f.kind is K.ARTWORK_MISMATCH]
    assert len(mismatch) == 1 and not mismatch[0].reliable
    assert verdict(config, label, comparison, registered, report).status is InspectionStatus.REVIEW


def test_zone_without_detected_text_is_read_as_a_block(config, label):
    zone = _zone_around(label, NAME, name="Name")
    engine = ScriptedOcr(label, hidden={NAME})
    report, *_ = run(config, label, label.image.copy(), engine, zones=[zone])
    zone_lines = [line for line in report.lines if line.zone == "Name"]
    assert len(zone_lines) == 1 and zone_lines[0].expected_text == "PRINT INSPECTOR"
    assert any(c.mode == "zone" for c in engine.calls)
    assert report.findings == ()


# ---------------------------------------------------------------- building blocks


def test_unavailable_report():
    required = unavailable_report("Tesseract not found.", True)
    assert not required.available and required.level is QualityLevel.POOR and not required.is_reliable
    optional = unavailable_report("Tesseract not found.", False)
    assert optional.level is QualityLevel.MARGINAL and optional.is_reliable


def test_unavailable_required_ocr_gives_review(config, label):
    report, comparison, registered = run(config, label, label.image.copy(), ScriptedOcr(label))
    decision = verdict(config, label, comparison, registered, unavailable_report("missing", True))
    assert decision.status is InspectionStatus.REVIEW and "not available" in decision.reasons[0]


def test_group_lines_by_baseline_and_zone():
    words = [
        OcrWord("LOT", 90, 10, 10, 40, 20), OcrWord("123", 90, 60, 12, 40, 20),
        OcrWord("EXP", 90, 10, 60, 40, 20), OcrWord("far", 90, 400, 10, 40, 20),
    ]
    lines = group_lines(words, lambda w: None)
    assert [[w.text for w in ws] for ws, _ in lines] == [["LOT", "123"], ["far"], ["EXP"]]  # top to bottom
    zone = object()
    split = group_lines(words, lambda w: zone if w.text == "123" else None)
    assert [[w.text for w in ws] for ws, _ in split] == [["LOT"], ["far"], ["123"], ["EXP"]]


def test_prepare_for_ocr_scales_inverts_and_pads():
    light_text = np.full((20, 50), 30, np.uint8)
    light_text[5:15, 10:40] = 220
    out = prepare_for_ocr(light_text, 2.0, invert=True)
    assert out.shape == (40 + 24, 100 + 24)
    assert out[0, 0] == 255 and out[12 + 20, 12 + 50] < 40  # border white, text dark


def test_glyph_losses_measure_missing_ink(config, label):
    printed = label.image.copy()
    erase(printed, label.char_box(LOT, 0))  # the L
    registered = simulate(printed)
    comparison = compare_to_artwork(registered, label.image, full_mask(label.image.shape), config.comparison, 0.9)
    gray = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # noqa: E731
    glyphs = glyph_losses(label.line_box(LOT), gray(label.image), gray(comparison.expected), gray(registered),
                          full_mask(label.image.shape), config.text)
    assert len(glyphs) >= 8
    lost = max(glyphs, key=lambda g: g.loss)
    assert lost.loss > 0.9 and abs(lost.box[0] - label.char_box(LOT, 0)[0]) <= 6  # font side bearing
    assert sorted(g.loss for g in glyphs)[-2] < 0.1  # the other characters are intact
