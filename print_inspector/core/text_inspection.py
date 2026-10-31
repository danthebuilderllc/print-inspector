"""Text inspection: OCR of the artwork and the print, combined with a character-level image check.

1. **Find the text in the artwork.** Tesseract reads the artwork (on the
   comparison grid, enlarged for small text); words are grouped into lines.
   Text-like areas that OCR could not read are kept as "unread regions".
2. **Read the same place on the print.** The photo is registered to the
   artwork, so each artwork line is cropped at the same coordinates from the
   artwork and from the print. Both crops are prepared identically (best
   colour channel, contrast stretch, dark text on light, scaled to a fixed
   line height) and read as single lines.
3. **Compare the texts** (``text_compare``): missing words, missing characters,
   incorrect numbers or characters, missing punctuation, extra text.
4. **Check every character in the image**, independently of OCR: each
   character of the artwork is compared with the print; a character with a
   share of its ink missing is partially printed.
5. **Combine.** An OCR difference is *reliable* only when both readings are
   confident, it is not a typical OCR confusion, the text is large enough,
   and the image comparison shows a difference at that place. Otherwise it
   is *uncertain*: the verdict becomes REVIEW (if the finding could reject the
   label), never FAIL on OCR alone. An image-only finding is reliable when the
   missing share is unambiguous.

**Critical lines** - lines in critical text zones and (``numbers_are_critical``)
lines containing digits - need higher OCR confidence, are checked for smaller
ink losses, and their findings are at least ``critical_line_severity``. A
critical line that OCR cannot read is itself a finding (REVIEW). A zone with
``expected_text`` is compared with that text.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import Enum

import cv2
import numpy as np

from print_inspector.config import DefectsConfig, TextConfig
from print_inspector.core.defects import DefectSeverity, text_like_mask
from print_inspector.core.models import QualityIssue, QualityLevel, Severity, level_from_issues
from print_inspector.core.ocr import OcrContext, OcrEngine, OcrError, OcrWord
from print_inspector.core.text_compare import (
    TextDifference,
    TextDifferenceKind,
    compare_text,
    normalize_text,
)
from print_inspector.core.text_zones import TextZone

log = logging.getLogger(__name__)

Box = tuple[int, int, int, int]  # x, y, width, height on the comparison grid
_PAD = 12  # white border around OCR crops (Tesseract needs a margin)
_STRONG_SCORE_FACTOR = 3.0  # image-only findings this far above the detection threshold are reliable


class TextFindingKind(str, Enum):
    MISSING_WORD = "missing_word"
    MISSING_CHARACTER = "missing_character"
    INCORRECT_NUMBER = "incorrect_number"
    INCORRECT_CHARACTER = "incorrect_character"
    MISSING_PUNCTUATION = "missing_punctuation"
    EXTRA_TEXT = "extra_text"
    PARTIAL_CHARACTER = "partial_character"
    UNREADABLE_TEXT = "unreadable_text"
    ARTWORK_MISMATCH = "artwork_mismatch"

    @property
    def label(self) -> str:
        return {
            TextFindingKind.MISSING_WORD: "Missing word",
            TextFindingKind.MISSING_CHARACTER: "Missing character",
            TextFindingKind.INCORRECT_NUMBER: "Incorrect number",
            TextFindingKind.INCORRECT_CHARACTER: "Incorrect character",
            TextFindingKind.MISSING_PUNCTUATION: "Missing punctuation",
            TextFindingKind.EXTRA_TEXT: "Extra text",
            TextFindingKind.PARTIAL_CHARACTER: "Partially printed character",
            TextFindingKind.UNREADABLE_TEXT: "Text not readable",
            TextFindingKind.ARTWORK_MISMATCH: "Artwork differs from expected text",
        }[self]


_FROM_DIFFERENCE = {
    TextDifferenceKind.MISSING_WORD: TextFindingKind.MISSING_WORD,
    TextDifferenceKind.MISSING_CHARACTER: TextFindingKind.MISSING_CHARACTER,
    TextDifferenceKind.INCORRECT_NUMBER: TextFindingKind.INCORRECT_NUMBER,
    TextDifferenceKind.INCORRECT_CHARACTER: TextFindingKind.INCORRECT_CHARACTER,
    TextDifferenceKind.MISSING_PUNCTUATION: TextFindingKind.MISSING_PUNCTUATION,
    TextDifferenceKind.EXTRA_TEXT: TextFindingKind.EXTRA_TEXT,
}


class LineStatus(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"  # a reliable finding
    UNCERTAIN = "uncertain"  # findings that need a person, or OCR could not confirm the text
    UNREADABLE = "unreadable"  # no usable OCR reading


@dataclass(frozen=True)
class TextFinding:
    id: int
    kind: TextFindingKind
    line_id: int | None
    zone: str | None
    critical: bool
    expected: str
    read: str
    x: int
    y: int
    width: int
    height: int
    severity: DefectSeverity
    reliable: bool
    notes: tuple[str, ...]
    artwork_confidence: float | None
    print_confidence: float | None
    image_score: float  # peak combined comparison score at the place (1 = detection threshold)
    ink_loss: float  # largest share of a character's ink missing there (0-1)

    @property
    def label(self) -> str:
        return f"T{self.id}"

    @property
    def is_reliable(self) -> bool:
        return self.reliable

    def describe(self) -> str:
        if self.kind is TextFindingKind.EXTRA_TEXT:
            return f"Extra text '{self.read}'"
        if self.kind in (TextFindingKind.PARTIAL_CHARACTER, TextFindingKind.UNREADABLE_TEXT):
            return f"{self.kind.label}" + (f" ('{self.expected}')" if self.expected else "")
        if self.read:
            return f"{self.kind.label}: expected '{self.expected}', read '{self.read}'"
        return f"{self.kind.label} '{self.expected}'"


@dataclass(frozen=True)
class TextLine:
    id: int
    x: int
    y: int
    width: int
    height: int
    zone: str | None
    critical: bool
    expected_text: str  # artwork reading, or the zone's expected text
    expected_source: str  # "artwork OCR" or "zone"
    artwork_text: str
    artwork_confidence: float | None
    print_text: str
    print_confidence: float | None
    status: LineStatus
    finding_ids: tuple[int, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class TextReport:
    available: bool
    engine: str
    lines: tuple[TextLine, ...]
    findings: tuple[TextFinding, ...]
    issues: tuple[QualityIssue, ...]
    unread_regions: tuple[Box, ...] = ()
    seconds: float = 0.0
    notes: tuple[str, ...] = ()

    @property
    def level(self) -> QualityLevel:
        return level_from_issues(self.issues)

    @property
    def is_reliable(self) -> bool:
        return self.level is not QualityLevel.POOR

    def by_id(self, finding_id: int) -> TextFinding | None:
        return next((f for f in self.findings if f.id == finding_id), None)

    def line(self, line_id: int) -> TextLine | None:
        return next((line for line in self.lines if line.id == line_id), None)


def unavailable_report(reason: str, required: bool) -> TextReport:
    """Text inspection could not run (no OCR engine); REVIEW if text inspection is required."""
    severity = Severity.CRITICAL if required else Severity.WARNING
    message = f"Text inspection (OCR) is not available: {reason}"
    return TextReport(False, "", (), (), (QualityIssue("ocr_unavailable", severity, message),))


# ---------------------------------------------------------------- image helpers


@dataclass
class _Glyph:
    box: Box
    loss: float
    missing_px: int
    core_px: int


def _clip(x0: float, y0: float, x1: float, y1: float, w: int, h: int) -> tuple[int, int, int, int]:
    return max(0, int(np.floor(x0))), max(0, int(np.floor(y0))), min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))


def _best_channel(crop: np.ndarray) -> int | None:
    """Colour channel with the most contrast (None = luminance), so coloured text reads well."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    spreads = [np.percentile(gray, 99) - np.percentile(gray, 1)]
    spreads += [np.percentile(crop[:, :, c], 99) - np.percentile(crop[:, :, c], 1) for c in range(3)]
    best = int(np.argmax(spreads))
    return None if best == 0 or spreads[best] < 1.15 * spreads[0] else best - 1


def _channel(crop: np.ndarray, channel: int | None) -> np.ndarray:
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if channel is None else np.ascontiguousarray(crop[:, :, channel])


def prepare_for_ocr(gray: np.ndarray, scale: float, invert: bool) -> np.ndarray:
    """Contrast-stretched, dark-on-light, scaled crop with a white border."""
    lo, hi = np.percentile(gray, 1), np.percentile(gray, 99)
    stretched = np.clip((gray.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1.0)), 0, 255).astype(np.uint8)
    if invert:
        stretched = 255 - stretched
    if abs(scale - 1.0) > 1e-3:
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        stretched = cv2.resize(stretched, None, fx=scale, fy=scale, interpolation=interp)
    return cv2.copyMakeBorder(stretched, _PAD, _PAD, _PAD, _PAD, cv2.BORDER_CONSTANT, value=255)


def _light_text(gray: np.ndarray) -> bool:
    """True for light text on a dark background (the background is the majority)."""
    lo, hi = np.percentile(gray, 2), np.percentile(gray, 98)
    return float(np.median(gray)) < (lo + hi) / 2


def glyph_losses(
    box: Box, ref_gray: np.ndarray, exp_gray: np.ndarray, reg_gray: np.ndarray, mask: np.ndarray, cfg: TextConfig
) -> list[_Glyph]:
    """Share of each artwork character's ink that is missing in the print, inside ``box``.

    Characters are the connected ink shapes of the artwork. Each is compared on its
    core (at least half its peak darkness in the expected print, which follows the
    photo's tone and blur); a core pixel is missing when the print has less than
    ``ink_loss_ratio`` of the expected darkness within one pixel (registration jitter).
    """
    h_img, w_img = ref_gray.shape
    x, y, w, h = box
    x0, y0, x1, y1 = _clip(x - 2, y - 2, x + w + 2, y + h + 2, w_img, h_img)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return []
    r = ref_gray[y0:y1, x0:x1].astype(np.float32)
    e = exp_gray[y0:y1, x0:x1].astype(np.float32)
    p = reg_gray[y0:y1, x0:x1].astype(np.float32)
    m = mask[y0:y1, x0:x1] > 0
    if _light_text(r):
        r, e, p = 255 - r, 255 - e, 255 - p
    paper_r, ink_r = np.percentile(r, 90), np.percentile(r, 3)
    if paper_r - ink_r < cfg.min_ink_contrast:
        return []
    ink = ((paper_r - r) >= 0.5 * (paper_r - ink_r)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    de = np.percentile(e, 90) - e
    dp = np.percentile(p, 90) - p
    dp_near = cv2.dilate(dp, np.ones((3, 3), np.uint8))
    out = []
    for i in range(1, count):
        gx, gy, gw, gh, area = (int(v) for v in stats[i])
        if area < 4:
            continue
        sl = (slice(gy, gy + gh), slice(gx, gx + gw))
        comp = labels[sl] == i
        de_c = de[sl]
        peak = float(de_c[comp].max())
        if peak < cfg.min_ink_contrast:
            continue
        core = comp & (de_c >= 0.5 * peak) & m[sl]
        core_px = int(core.sum())
        if core_px < 3:
            continue
        missing = core & (dp_near[sl] < cfg.ink_loss_ratio * de_c)
        missing_px = int(missing.sum())
        out.append(_Glyph((x0 + gx, y0 + gy, gw, gh), missing_px / core_px, missing_px, core_px))
    return out


def _overlaps(a: Box, b: Box, pad: int = 0) -> bool:
    return not (
        a[0] > b[0] + b[2] + pad or a[0] + a[2] + pad < b[0] or a[1] > b[1] + b[3] + pad or a[1] + a[3] + pad < b[1]
    )


def _peak_score(score: np.ndarray, box: Box, pad: int = 2) -> float:
    h, w = score.shape
    x0, y0, x1, y1 = _clip(box[0] - pad, box[1] - pad, box[0] + box[2] + pad, box[1] + box[3] + pad, w, h)
    return float(score[y0:y1, x0:x1].max()) if x1 > x0 and y1 > y0 else 0.0


# ---------------------------------------------------------------- OCR helpers


@dataclass
class _Reading:
    words: list[OcrWord]  # grid coordinates
    error: str | None = None

    @property
    def text(self) -> str:
        return normalize_text(" ".join(w.text for w in self.words))

    @property
    def confidence(self) -> float | None:
        confs = [w.confidence for w in self.words if w.confidence >= 0]
        return min(confs) if confs else None


def _to_grid(words: list[OcrWord], x0: int, y0: int, scale: float) -> list[OcrWord]:
    return [
        replace(
            w,
            x=int(round(x0 + (w.x - _PAD) / scale)), y=int(round(y0 + (w.y - _PAD) / scale)),
            width=max(1, int(round(w.width / scale))), height=max(1, int(round(w.height / scale))),
        )
        for w in words
    ]


@dataclass
class _Line:
    id: int
    box: Box
    words: list[OcrWord]  # artwork words found on the page (grid)
    zone: TextZone | None
    psm: int
    expected_text: str = ""  # from the zone, if given
    artwork: _Reading = field(default_factory=lambda: _Reading([]))
    printed: _Reading = field(default_factory=lambda: _Reading([]))


def _read(engine: OcrEngine, image: np.ndarray, box: Box, psm: int, source: str, mode: str,
          scale: float, channel: int | None, invert: bool) -> _Reading:
    x, y, w, h = box
    crop = image[y : y + h, x : x + w]
    prepared = prepare_for_ocr(_channel(crop, channel), scale, invert)
    try:
        words = engine.recognize(prepared, psm, OcrContext(source, mode, box))
    except OcrError as exc:
        return _Reading([], str(exc))
    return _Reading(sorted(_to_grid(words, x, y, scale), key=lambda wd: (wd.block, wd.paragraph, wd.line, wd.x)))


def _detect_words(
    reference: np.ndarray, px_per_mm: float, engine: OcrEngine, cfg: TextConfig
) -> list[OcrWord]:
    gh, gw = reference.shape[:2]
    scale = cfg.target_line_height_px * 0.6 / max(cfg.min_text_height_mm * px_per_mm, 1e-6)
    scale = float(np.clip(scale, 1.0, 4.0))
    scale = min(scale, 8000.0 / max(gh, gw))
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    prepared = prepare_for_ocr(gray, scale, _light_text(gray))
    words = engine.recognize(prepared, cfg.page_psm, OcrContext("artwork", "page", (0, 0, gw, gh)))
    out = []
    for w in _to_grid(words, 0, 0, scale):
        if w.confidence < cfg.min_detect_confidence or not any(c.isalnum() for c in w.text):
            continue
        if w.height < 3 or w.height > 0.5 * gh or w.width > 0.98 * gw:
            continue
        out.append(w)
    return out


def _zone_of(word: OcrWord, zones: list[tuple[TextZone, Box]]) -> TextZone | None:
    cx, cy = word.x + word.width / 2, word.y + word.height / 2
    for zone, (x, y, w, h) in zones:
        if x <= cx < x + w and y <= cy < y + h:
            return zone
    return None


def group_lines(words: list[OcrWord], zone_of) -> list[tuple[list[OcrWord], TextZone | None]]:
    """Words on one baseline (and in the same zone) form a line."""
    n = len(words)
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    zones = [zone_of(w) for w in words]
    for i in range(n):
        a = words[i]
        for j in range(i + 1, n):
            b = words[j]
            if zones[i] is not zones[j]:
                continue
            overlap = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
            hmin, hmax = min(a.height, b.height), max(a.height, b.height)
            if overlap < 0.5 * hmin or hmax > 2.0 * hmin:
                continue
            gap = max(a.x, b.x) - min(a.x + a.width, b.x + b.width)
            if gap <= 1.5 * hmax:
                parent[root(i)] = root(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    lines = [(sorted((words[i] for i in idx), key=lambda w: w.x), zones[idx[0]]) for idx in groups.values()]
    return sorted(lines, key=lambda item: (min(w.y for w in item[0]), min(w.x for w in item[0])))


def _union(words: list[OcrWord]) -> Box:
    x0 = min(w.x for w in words)
    y0 = min(w.y for w in words)
    x1 = max(w.x + w.width for w in words)
    y1 = max(w.y + w.height for w in words)
    return x0, y0, x1 - x0, y1 - y0


def _span_box(reading: _Reading, start: int, end: int, fallback: Box) -> tuple[Box, list[OcrWord]]:
    """Grid box of characters [start, end) of the reading's normalised text, and the words involved."""
    words = reading.words
    if not words:
        return fallback, []
    pieces, pos = [], 0
    for w in words:
        token = normalize_text(w.text)
        pieces.append((pos, pos + len(token), w, token))
        pos += len(token) + 1
    if start == end:  # insertion point
        for s, t, w, token in pieces:
            if s <= start <= t:
                frac = (start - s) / max(len(token), 1)
                cx = w.x + frac * w.width
                return (int(cx - 0.3 * w.height), w.y, max(2, int(0.6 * w.height)), w.height), [w]
        w = words[-1]
        return (w.x + w.width, w.y, max(2, int(0.6 * w.height)), w.height), [w]
    boxes, involved = [], []
    for s, t, w, token in pieces:
        a, b = max(start, s), min(end, t)
        if a >= b:
            continue
        n = max(len(token), 1)
        bx0 = w.x + (a - s) / n * w.width
        bx1 = w.x + (b - s) / n * w.width
        boxes.append((bx0, w.y, bx1, w.y + w.height))
        involved.append(w)
    if not boxes:
        return fallback, []
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return (int(np.floor(x0)), int(np.floor(y0)), max(1, int(np.ceil(x1 - x0))), max(1, int(np.ceil(y1 - y0)))), involved


def _min_conf(words: list[OcrWord]) -> float | None:
    confs = [w.confidence for w in words if w.confidence >= 0]
    return min(confs) if confs else None


def _words_in(words: list[OcrWord], box: Box) -> list[OcrWord]:
    return [w for w in words if _overlaps((w.x, w.y, w.width, w.height), box)]


def _severity_for(kind: TextFindingKind, in_number: bool, critical: bool, cfg: TextConfig) -> DefectSeverity:
    name = {
        TextFindingKind.MISSING_WORD: cfg.severity_missing_word,
        TextFindingKind.MISSING_CHARACTER: cfg.severity_missing_character,
        TextFindingKind.INCORRECT_NUMBER: cfg.severity_incorrect_number,
        TextFindingKind.INCORRECT_CHARACTER: cfg.severity_incorrect_character,
        TextFindingKind.MISSING_PUNCTUATION: cfg.severity_missing_punctuation,
        TextFindingKind.EXTRA_TEXT: cfg.severity_extra_text,
        TextFindingKind.PARTIAL_CHARACTER: cfg.severity_partial_character,
        TextFindingKind.UNREADABLE_TEXT: cfg.severity_unreadable_text,
        TextFindingKind.ARTWORK_MISMATCH: cfg.critical_line_severity,
    }[kind]
    severity = DefectSeverity(name)
    if in_number and kind is not TextFindingKind.EXTRA_TEXT:
        severity = max(severity, DefectSeverity(cfg.severity_incorrect_number), key=lambda s: s.rank)
    if critical:
        severity = max(severity, DefectSeverity(cfg.critical_line_severity), key=lambda s: s.rank)
    return severity


# ---------------------------------------------------------------- main entry


def inspect_text(
    reference: np.ndarray,
    registered: np.ndarray,
    expected: np.ndarray,
    combined_score: np.ndarray,
    mask: np.ndarray,
    px_per_mm: float,
    zones: tuple[TextZone, ...],
    engine: OcrEngine,
    cfg: TextConfig,
    defects_cfg: DefectsConfig,
    exclude: tuple[Box, ...] = (),
) -> TextReport:
    """Inspect the text of the registered print against the artwork (all images on the comparison grid).

    ``exclude``: areas that are not text (barcodes), ignored by OCR and the unread-text check.
    """
    started = time.perf_counter()
    gh, gw = reference.shape[:2]
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    exp_gray = cv2.cvtColor(expected, cv2.COLOR_BGR2GRAY)
    reg_gray = cv2.cvtColor(registered, cv2.COLOR_BGR2GRAY)
    zone_rects = [(z, z.rect(gw, gh)) for z in zones]

    # 1. Text in the artwork.
    try:
        page_words = _detect_words(reference, px_per_mm, engine, cfg)
    except OcrError as exc:
        issue = QualityIssue("ocr_failed", Severity.CRITICAL, f"OCR of the artwork failed: {exc}")
        return TextReport(True, engine.name, (), (), (issue,), seconds=time.perf_counter() - started)
    page_words = [
        w for w in page_words
        if not any(_overlaps((w.x, w.y, w.width, w.height), box) for box in exclude)
    ]
    expected_zones = [(z, r) for z, r in zone_rects if z.expected_text.strip()]
    free_words = [w for w in page_words if _zone_of(w, expected_zones) is None]
    lines: list[_Line] = []
    for words, zone in group_lines(free_words, lambda w: _zone_of(w, zone_rects)):
        lines.append(_Line(len(lines) + 1, _union(words), words, zone, cfg.line_psm))
    covered = {id(line.zone) for line in lines if line.zone is not None}
    for zone, rect in zone_rects:
        if zone.expected_text.strip():
            lines.append(_Line(len(lines) + 1, rect, [], zone, cfg.zone_psm, zone.expected_text))
        elif id(zone) not in covered:  # a critical zone whose text OCR did not find: read it as a block
            lines.append(_Line(len(lines) + 1, rect, [], zone, cfg.zone_psm))

    # 2. Read every line on the artwork and on the print, prepared identically.
    def read_line(line: _Line) -> None:
        x, y, w, h = line.box
        text_h = float(np.median([wd.height for wd in line.words])) if line.words else None
        if text_h is None:  # zone: estimate the character height from the artwork
            glyphs = glyph_losses(line.box, ref_gray, ref_gray, ref_gray, np.full_like(ref_gray, 255), cfg)
            text_h = float(np.median([g.box[3] for g in glyphs])) if glyphs else max(h / 2, 8.0)
        margin = cfg.line_margin * text_h
        x0, y0, x1, y1 = _clip(x - margin, y - margin, x + w + margin, y + h + margin, gw, gh)
        crop_box = (x0, y0, x1 - x0, y1 - y0)
        scale = float(np.clip(cfg.target_line_height_px / max(text_h, 1.0), 0.5, 6.0))
        ref_crop = reference[y0:y1, x0:x1]
        channel = _best_channel(ref_crop)
        invert = _light_text(_channel(ref_crop, channel))
        mode = "zone" if line.psm == cfg.zone_psm and not line.words else "line"
        line.artwork = _read(engine, reference, crop_box, line.psm, "artwork", mode, scale, channel, invert)
        line.printed = _read(engine, registered, crop_box, line.psm, "print", mode, scale, channel, invert)

    with ThreadPoolExecutor(max_workers=max(1, cfg.max_parallel)) as pool:
        list(pool.map(read_line, lines))

    # 3./4./5. Compare, check characters in the image, combine.
    findings: list[TextFinding] = []
    line_results: list[TextLine] = []
    errors = 0
    for line in lines:
        f_before = len(findings)
        new, status, notes, expected_text, source = _assess_line(
            line, ref_gray, exp_gray, reg_gray, combined_score, mask, cfg
        )
        findings.extend(new)
        errors += bool(line.artwork.error or line.printed.error)
        x, y, w, h = line.box
        zone = line.zone
        line_results.append(TextLine(
            id=line.id, x=x, y=y, width=w, height=h,
            zone=zone.name if zone else None,
            critical=_is_critical(line, cfg),
            expected_text=expected_text, expected_source=source,
            artwork_text=line.artwork.text, artwork_confidence=line.artwork.confidence,
            print_text=line.printed.text, print_confidence=line.printed.confidence,
            status=status, finding_ids=tuple(range(f_before, len(findings))), notes=tuple(notes),
        ))

    # Text-like areas OCR did not read: checked by the image comparison only.
    unread: list[Box] = []
    text_mask = text_like_mask(ref_gray, px_per_mm, defects_cfg)
    if text_mask.any():
        for line in lines:
            x, y, w, h = line.box
            pad = max(2, int(0.4 * h))
            cv2.rectangle(text_mask, (x - pad, y - pad), (x + w + pad, y + h + pad), 0, -1)
        for x, y, w, h in exclude:
            cv2.rectangle(text_mask, (x - 4, y - 4), (x + w + 4, y + h + 4), 0, -1)
        count, _, stats, _ = cv2.connectedComponentsWithStats((text_mask > 0).astype(np.uint8), connectivity=8)
        for i in range(1, count):
            box = tuple(int(v) for v in stats[i][:4])
            if stats[i][4] < 20:
                continue
            unread.append(box)
            for g in glyph_losses(box, ref_gray, exp_gray, reg_gray, mask, cfg):
                finding = _glyph_finding(g, None, None, False, "", combined_score, cfg)
                if finding is not None:
                    findings.append(replace(finding, notes=finding.notes + ("In text that OCR could not read.",)))

    # Order by severity, then position; number T1, T2 ...
    order = sorted(range(len(findings)), key=lambda i: (-findings[i].severity.rank, findings[i].y, findings[i].x))
    new_id = {old: rank + 1 for rank, old in enumerate(order)}
    findings = [replace(findings[i], id=new_id[i]) for i in order]
    line_results = [replace(line, finding_ids=tuple(sorted(new_id[i] for i in line.finding_ids))) for line in line_results]

    issues = []
    if errors:
        issues.append(QualityIssue(
            "ocr_errors", Severity.WARNING,
            f"OCR failed for {errors} text line(s); they were checked by the image comparison only.",
        ))
    notes = []
    if not lines:
        notes.append("No text was found in the artwork.")
    if unread:
        notes.append(f"{len(unread)} text-like area(s) could not be read by OCR; checked by image comparison only.")
    seconds = time.perf_counter() - started
    log.info(
        "Text inspection: %d lines, %d findings (%d reliable), %d unread areas, %.1f s",
        len(line_results), len(findings), sum(f.reliable for f in findings), len(unread), seconds,
    )
    return TextReport(True, engine.name, tuple(line_results), tuple(findings), tuple(issues), tuple(unread),
                      seconds, tuple(notes))


def _is_critical(line: _Line, cfg: TextConfig) -> bool:
    if line.zone is not None and line.zone.critical:
        return True
    text = line.expected_text or line.artwork.text
    return cfg.numbers_are_critical and any(c.isdigit() for c in text)


def _glyph_finding(
    g: _Glyph, line_id: int | None, zone: str | None, critical: bool, expected: str,
    combined_score: np.ndarray, cfg: TextConfig,
) -> TextFinding | None:
    threshold = cfg.critical_partial_min_fraction if critical else cfg.partial_min_fraction
    if g.loss < threshold or g.missing_px < cfg.partial_min_area_px:
        return None
    score = _peak_score(combined_score, g.box)
    # Unambiguous: a large share missing, or a clear difference in the comparison engine too.
    reliable = g.core_px >= 10 and (
        g.loss >= cfg.partial_reliable_fraction or score >= _STRONG_SCORE_FACTOR * cfg.corroboration_score
    )
    kind = TextFindingKind.MISSING_CHARACTER if g.loss >= 0.85 else TextFindingKind.PARTIAL_CHARACTER
    notes = [f"{g.loss:.0%} of the character's ink is missing in the print ({g.missing_px} px); "
             f"comparison score {score:.1f}."]
    if not reliable:
        notes.append("Small or faint loss: needs a person to judge.")
    x, y, w, h = g.box
    return TextFinding(
        0, kind, line_id, zone, critical, expected, "", x, y, w, h,
        _severity_for(kind, False, critical, cfg), reliable, tuple(notes), None, None, score, g.loss,
    )


def _assess_line(
    line: _Line, ref_gray, exp_gray, reg_gray, combined_score, mask, cfg: TextConfig
) -> tuple[list[TextFinding], LineStatus, list[str], str, str]:
    critical = _is_critical(line, cfg)
    zone = line.zone.name if line.zone else None
    min_conf = cfg.critical_min_confidence if critical else cfg.min_word_confidence
    art, prn = line.artwork, line.printed
    notes: list[str] = []
    findings: list[TextFinding] = []
    small = line.box[3] < cfg.min_ocr_line_height_px if line.words else False

    # Image check of every character, independent of OCR.
    glyphs = glyph_losses(line.box, ref_gray, exp_gray, reg_gray, mask, cfg)
    image_trouble = any(g.loss >= cfg.partial_min_fraction and g.missing_px >= cfg.partial_min_area_px for g in glyphs)
    image_trouble |= _peak_score(combined_score, line.box) >= cfg.corroboration_score

    if line.expected_text:
        expected_text, source = normalize_text(line.expected_text), "zone"
        reference_reading = prn  # positions of differences: on the print reading
        art_conf = art.confidence
        if art.error is None and art.text and compare_text(expected_text, art.text):
            findings.append(TextFinding(
                0, TextFindingKind.ARTWORK_MISMATCH, line.id, zone, True, expected_text, art.text,
                *line.box, _severity_for(TextFindingKind.ARTWORK_MISMATCH, False, True, cfg), False,
                ("OCR of the artwork does not give the zone's expected text: the artwork, the expected "
                 "text or the OCR reading is wrong.",), art_conf, prn.confidence, 0.0, 0.0,
            ))
    else:
        expected_text, source = art.text, "artwork OCR"
        reference_reading = art
        art_conf = art.confidence

    artwork_ok = bool(expected_text) and (source == "zone" or (art_conf is not None and art_conf >= min_conf))
    print_conf = prn.confidence
    print_ok = bool(prn.text) and print_conf is not None and print_conf >= min_conf and prn.error is None

    if art.error or prn.error:
        notes.append(f"OCR failed: {art.error or prn.error}")
    if not artwork_ok:
        notes.append("OCR could not read the artwork text reliably; checked by the image comparison only.")
    elif not print_ok and not image_trouble:
        notes.append("OCR could not read the print reliably, but the image comparison found no difference.")

    differences: list[TextDifference] = []
    if artwork_ok and (print_ok or image_trouble or source == "zone"):
        differences = compare_text(expected_text, prn.text)

    used_glyphs: set[int] = set()
    for d in differences:
        kind = _FROM_DIFFERENCE[d.kind]
        reading = prn if source == "zone" else reference_reading
        box, involved = _span_box(reading, d.start, d.end, line.box) if source != "zone" else (line.box, [])
        a_conf = _min_conf(involved) if involved else art_conf
        near_print = _words_in(prn.words, box)
        p_conf = _min_conf(near_print) if near_print else print_conf
        score = _peak_score(combined_score, box)
        hits = [i for i, g in enumerate(glyphs) if _overlaps(g.box, box, 1)]
        loss = max((glyphs[i].loss for i in hits), default=0.0)
        used_glyphs.update(i for i in hits if glyphs[i].loss >= cfg.critical_partial_min_fraction)
        corroborated = score >= cfg.corroboration_score or loss >= (
            cfg.critical_partial_min_fraction if critical else cfg.partial_min_fraction
        )
        if source == "zone":
            corroborated = corroborated or image_trouble
        if kind is TextFindingKind.EXTRA_TEXT and d.confusable and not corroborated:
            log.debug("Ignoring uncorroborated extra punctuation '%s' (OCR noise)", d.read)
            continue
        f_notes = []
        if a_conf is not None and a_conf < min_conf and source != "zone":
            f_notes.append(f"artwork read with low OCR confidence ({a_conf:.0f})")
        if p_conf is None or p_conf < min_conf:
            f_notes.append("print read with low OCR confidence" + (f" ({p_conf:.0f})" if p_conf is not None else ""))
        if d.confusable:
            f_notes.append(f"'{d.expected}' / '{d.read}' is a typical OCR confusion")
        if small:
            f_notes.append("text is small for reliable OCR")
        if not corroborated:
            f_notes.append("the image comparison shows no difference here")
        reliable = not f_notes
        notes_out = (
            ["Confirmed by the image comparison" + (f" (score {score:.1f}" + (f", {loss:.0%} ink missing)" if loss else ")"))]
            if reliable else ["Uncertain: " + "; ".join(f_notes) + "."]
        )
        findings.append(TextFinding(
            0, kind, line.id, zone, critical, d.expected, d.read, *box,
            _severity_for(kind, d.in_number, critical, cfg), reliable, tuple(notes_out),
            a_conf, p_conf, score, loss,
        ))

    # Characters that lost ink but no OCR difference covers them.
    for i, g in enumerate(glyphs):
        if i in used_glyphs or any(_overlaps(g.box, (f.x, f.y, f.width, f.height)) for f in findings):
            continue
        guess = ""
        for w in reference_reading.words if source != "zone" else ():
            if _overlaps((w.x, w.y, w.width, w.height), g.box):
                text = normalize_text(w.text)
                idx = int((g.box[0] + g.box[2] / 2 - w.x) / max(w.width, 1) * len(text))
                guess = text[min(max(idx, 0), len(text) - 1)] if text else ""
                break
        finding = _glyph_finding(g, line.id, zone, critical, guess, combined_score, cfg)
        if finding is not None:
            findings.append(finding)

    if critical and (not artwork_ok or (not print_ok and not differences)):
        if not any(f.kind is not TextFindingKind.ARTWORK_MISMATCH for f in findings):
            what = "the artwork" if not artwork_ok else "the print"
            findings.append(TextFinding(
                0, TextFindingKind.UNREADABLE_TEXT, line.id, zone, True, expected_text, prn.text, *line.box,
                _severity_for(TextFindingKind.UNREADABLE_TEXT, False, True, cfg), False,
                (f"Critical text: OCR could not read {what} with enough confidence (needs {min_conf:.0f}); "
                 "a person must check it.",),
                art_conf, print_conf, _peak_score(combined_score, line.box), 0.0,
            ))

    if any(f.reliable for f in findings):
        status = LineStatus.MISMATCH
    elif findings:
        status = LineStatus.UNCERTAIN
    elif not artwork_ok or not print_ok:
        status = LineStatus.UNREADABLE if (art.error or prn.error or not prn.text) else LineStatus.UNCERTAIN
    else:
        status = LineStatus.MATCH
    return findings, status, notes, expected_text, source
