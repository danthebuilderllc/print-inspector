"""Comparison of the expected text (artwork) with the text read from the print.

Words are aligned first, so a dropped word is reported as a missing word;
words that differ are then compared character by character. Differences
only in spacing are ignored (OCR often merges or splits words). Each
difference records whether it involves a number and whether it is a typical
OCR confusion (0/O, 1/l/I, 5/S, rn/m, a case change ...), which makes it
uncertain.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum


class TextDifferenceKind(str, Enum):
    MISSING_WORD = "missing_word"
    MISSING_CHARACTER = "missing_character"
    INCORRECT_NUMBER = "incorrect_number"
    INCORRECT_CHARACTER = "incorrect_character"
    MISSING_PUNCTUATION = "missing_punctuation"
    EXTRA_TEXT = "extra_text"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").capitalize()


@dataclass(frozen=True)
class TextDifference:
    kind: TextDifferenceKind
    expected: str  # the expected characters (empty for extra text)
    read: str  # what was read instead (empty if missing)
    start: int  # span in the normalised expected text; start == end for an insertion
    end: int
    in_number: bool
    confusable: bool  # a typical OCR confusion, so possibly not a print defect

    def describe(self) -> str:
        number = " in a number" if self.in_number else ""
        if self.kind is TextDifferenceKind.EXTRA_TEXT:
            return f"Extra text '{self.read}'"
        if not self.read:
            return f"{self.kind.label} '{self.expected}'{number}"
        return f"{self.kind.label}: expected '{self.expected}', read '{self.read}'"


_REPLACEMENTS = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "´": "'", "`": "'",
    "“": '"', "”": '"', "„": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
    " ": " ", " ": " ", " ": " ",
}
_CONFUSABLE_GROUPS = (
    "0OoDQ", "1Il|i!", "5Ss", "8B", "2Zz", "6Gb", "9gq", "7T", ".,", "'\"", "Cc", "Vv", "Ww", "Xx",
    "Uu", "Kk", "Pp", "Yy", "Jj", "Mm", "Nn", "Hh", "Ee", "Ff", "Aa", "Rr", "Ll", "Tt",
)
_CONFUSABLE_SEQUENCES = {
    frozenset(p) for p in (("rn", "m"), ("cl", "d"), ("vv", "w"), ("VV", "W"), ("li", "h"), ("ri", "n"), ("nn", "m"))
}


def normalize_text(text: str) -> str:
    """Unicode-normalised text with uniform quotes, dashes and single spaces."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(_REPLACEMENTS.get(c, c) for c in text)
    return re.sub(r"\s+", " ", text).strip()


def is_punctuation(text: str) -> bool:
    return bool(text) and all(unicodedata.category(c).startswith(("P", "S")) for c in text)


def _confusable(expected: str, read: str) -> bool:
    if frozenset((expected, read)) in _CONFUSABLE_SEQUENCES:
        return True
    if len(expected) != len(read) or not expected:
        return False
    return all(e == r or any(e in g and r in g for g in _CONFUSABLE_GROUPS) for e, r in zip(expected, read))


def _tokens(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens, spans = [], []
    for m in re.finditer(r"\S+", text):
        tokens.append(m.group())
        spans.append((m.start(), m.end()))
    return tokens, spans


def compare_text(expected: str, read: str) -> list[TextDifference]:
    """Differences between the expected text and the text read from the print."""
    e, r = normalize_text(expected), normalize_text(read)
    if e == r:
        return []
    e_tokens, e_spans = _tokens(e)
    r_tokens, _ = _tokens(r)
    out: list[TextDifference] = []
    matcher = SequenceMatcher(None, e_tokens, r_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            for k in range(i1, i2):
                token = e_tokens[k]
                kind = TextDifferenceKind.MISSING_PUNCTUATION if is_punctuation(token) else TextDifferenceKind.MISSING_WORD
                out.append(TextDifference(kind, token, "", *e_spans[k], any(c.isdigit() for c in token), False))
        elif tag == "insert":
            at = e_spans[i1][0] if i1 < len(e_spans) else len(e)
            extra = " ".join(r_tokens[j1:j2])
            out.append(TextDifference(TextDifferenceKind.EXTRA_TEXT, "", extra, at, at, False, is_punctuation(extra)))
        else:
            out.extend(_character_differences(e, e_tokens[i1:i2], e_spans[i1:i2], r_tokens[j1:j2]))
    return out


def _character_differences(
    e: str, tokens: list[str], spans: list[tuple[int, int]], read_tokens: list[str]
) -> list[TextDifference]:
    """Character differences of a block of expected words against the read words (spaces ignored)."""
    chars, positions = [], []
    for token, (start, _) in zip(tokens, spans):
        for offset, c in enumerate(token):
            chars.append(c)
            positions.append(start + offset)
    expected = "".join(chars)
    read = "".join(read_tokens)
    if expected == read:
        return []  # only the spacing differs
    whole_tokens = {(s, t - 1): token for token, (s, t) in zip(tokens, spans)}
    out = []
    matcher = SequenceMatcher(None, expected, read, autojunk=False)
    for tag, a1, a2, b1, b2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        exp, got = expected[a1:a2], read[b1:b2]
        neighbours = expected[max(0, a1 - 1) : a2 + 1]
        in_number = any(c.isdigit() for c in exp + got) or (
            is_punctuation(exp) and sum(c.isdigit() for c in neighbours) >= 2
        )
        if tag == "insert":
            at = positions[a1] if a1 < len(positions) else (positions[-1] + 1 if positions else 0)
            out.append(TextDifference(TextDifferenceKind.EXTRA_TEXT, "", got, at, at, in_number, is_punctuation(got)))
            continue
        start, end = positions[a1], positions[a2 - 1] + 1
        if tag == "delete":
            if (start, end - 1) in whole_tokens and not is_punctuation(exp):
                kind = TextDifferenceKind.MISSING_WORD
            elif is_punctuation(exp):
                kind = TextDifferenceKind.MISSING_PUNCTUATION
            else:
                kind = TextDifferenceKind.MISSING_CHARACTER
            out.append(TextDifference(kind, e[start:end], "", start, end, in_number, False))
        else:  # replace
            if any(c.isdigit() for c in exp + got):
                kind = TextDifferenceKind.INCORRECT_NUMBER
            else:
                kind = TextDifferenceKind.INCORRECT_CHARACTER
            out.append(TextDifference(kind, e[start:end], got, start, end, in_number, _confusable(exp, got)))
    return out
