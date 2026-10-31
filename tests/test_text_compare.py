"""Text comparison: expected (artwork) text against the text read from the print."""
import pytest

from print_inspector.core.text_compare import TextDifferenceKind as K
from print_inspector.core.text_compare import compare_text, is_punctuation, normalize_text


def kinds(expected, read):
    return [(d.kind, d.expected, d.read) for d in compare_text(expected, read)]


def test_identical_and_spacing_only_differences_are_equal():
    assert compare_text("LOT 12345", "LOT 12345") == []
    assert compare_text("PRINT INSPECTOR", "PRINTINSPECTOR") == []
    assert compare_text("Net  wt.", " Net wt. ") == []


def test_missing_character_in_number():
    (d,) = compare_text("LOT 12345", "LOT 1234")
    assert (d.kind, d.expected, d.in_number) == (K.MISSING_CHARACTER, "5", True)
    assert "LOT 12345"[d.start : d.end] == "5"


def test_incorrect_number():
    (d,) = compare_text("LOT 12345", "LOT 12845")
    assert (d.kind, d.expected, d.read, d.confusable) == (K.INCORRECT_NUMBER, "3", "8", False)


def test_missing_punctuation():
    assert kinds("EXP 2027/06", "EXP 2027 06") == [(K.MISSING_PUNCTUATION, "/", "")]
    assert kinds("Hello, world", "Hello world") == [(K.MISSING_PUNCTUATION, ",", "")]
    (d,) = compare_text("Price 12.50", "Price 1250")
    assert d.kind is K.MISSING_PUNCTUATION and d.in_number  # a decimal point changes the number


def test_missing_word():
    assert kinds("Net wt. 250 g", "Net 250 g") == [(K.MISSING_WORD, "wt.", "")]
    assert kinds("BEST BEFORE END", "BEST END") == [(K.MISSING_WORD, "BEFORE", "")]
    assert kinds("LOT 12345", "12345") == [(K.MISSING_WORD, "LOT", "")]


def test_whole_line_missing_gives_one_finding_per_word():
    assert kinds("Net wt. 250 g", "") == [
        (K.MISSING_WORD, "Net", ""), (K.MISSING_WORD, "wt.", ""), (K.MISSING_WORD, "250", ""), (K.MISSING_WORD, "g", ""),
    ]


def test_extra_text():
    (d,) = compare_text("ABC", "ABXC")
    assert d.kind is K.EXTRA_TEXT and d.read == "X" and d.start == d.end == 2


def test_incorrect_and_missing_letters():
    assert kinds("Hello", "Helo") == [(K.MISSING_CHARACTER, "l", "")]
    assert kinds("INSPECTOR", "INSPEKTOR") == [(K.INCORRECT_CHARACTER, "C", "K")]


@pytest.mark.parametrize("expected, read", [
    ("LOT 10", "LOT IO"), ("modern", "rnodern"), ("SIZE", "5IZE"), ("cost", "Cost"), ("1.5", "1,5"),
])
def test_typical_ocr_confusions_are_marked(expected, read):
    differences = compare_text(expected, read)
    assert differences and all(d.confusable for d in differences)


def test_real_differences_are_not_confusions():
    assert not any(d.confusable for d in compare_text("LOT 12345", "LOT 12845"))
    assert not any(d.confusable for d in compare_text("INSPECTOR", "INSPEKTOR"))


def test_normalisation():
    assert normalize_text("“Net” wt. – 250 g") == '"Net" wt. - 250 g'
    assert normalize_text("ﬁne") == "fine"  # ligature
    assert is_punctuation("./-") and not is_punctuation("a.") and not is_punctuation("")
