"""Barcode and QR code inspection of the registered print.

1. **Locate** the barcodes: ZXing-C++ decodes the artwork on the comparison
   grid; barcode zones (``barcode_zones``) add where a code must be, its
   format and optionally its exact value.
2. **Decode the print** in the same region (the photo is registered to the
   artwork) and **compare** the value with the expected one (the zone's value,
   else the artwork's). EAN-13 / UPC-A are compared as GTIN-13.
3. **Decoding failures** are findings. They are *reliable* only when damage in
   the image explains them; otherwise the photo may simply be unsuitable
   (blur, glare, too small) and the verdict is REVIEW.
4. **Damage**, measured against the artwork:

   * linear codes: the print is binarised and compared with the artwork bars
     (registration jitter of one pixel across the bars is tolerated);
     damaged areas larger than ``min_damage_modules2`` are reported. The code
     is also decoded in ``scan_strips`` horizontal strips - the share that
     decodes shows how robust it is in this photo. The average bar width
     change (ink spread) and dark marks in the quiet zones are measured;
   * QR codes: every module of the artwork's grid is sampled in the print;
     wrong modules are grouped into damaged areas (finder patterns are
     reported separately), and the unused error correction reported by the
     decoder shows how much reserve is left.

These are decoding checks and measurements from a phone photo. They are
**not** ISO/IEC 15416 (linear) or 15415 (2D) print-quality grades, which
require a calibrated verifier with defined illumination and aperture.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from enum import Enum

import cv2
import numpy as np

from print_inspector.config import BarcodeConfig
from print_inspector.core.barcode_zones import ANY_FORMAT, BarcodeZone
from print_inspector.core.barcodes import (
    EAN_UPC,
    EAN_UPC_MODULES,
    FORMAT_LABELS,
    LINEAR,
    Symbol,
    canonical_value,
    decode,
    library_version,
    same_symbology,
)
from print_inspector.core.defects import DefectSeverity
from print_inspector.core.models import QualityIssue, QualityLevel, Severity, level_from_issues

log = logging.getLogger(__name__)

Box = tuple[int, int, int, int]
# Share of a QR code's codewords that each error correction level can restore.
QR_EC_CAPACITY = {"L": 0.07, "M": 0.15, "Q": 0.25, "H": 0.30}
NOT_ISO = "Decoding check and measurements from a photo - not an ISO/IEC 15416 / 15415 grade."


class BarcodeFindingKind(str, Enum):
    WRONG_CONTENT = "wrong_content"
    NOT_DECODED = "not_decoded"
    WRONG_FORMAT = "wrong_format"
    DAMAGED_BARS = "damaged_bars"
    DAMAGED_MODULES = "damaged_modules"
    QUIET_ZONE = "quiet_zone"
    LOW_DECODE_MARGIN = "low_decode_margin"
    BAR_WIDTH = "bar_width"
    ARTWORK_MISMATCH = "artwork_mismatch"
    UNEXPECTED_BARCODE = "unexpected_barcode"

    @property
    def label(self) -> str:
        return {
            BarcodeFindingKind.WRONG_CONTENT: "Wrong content",
            BarcodeFindingKind.NOT_DECODED: "Not decodable",
            BarcodeFindingKind.WRONG_FORMAT: "Wrong barcode type",
            BarcodeFindingKind.DAMAGED_BARS: "Damaged bars",
            BarcodeFindingKind.DAMAGED_MODULES: "Damaged QR modules",
            BarcodeFindingKind.QUIET_ZONE: "Quiet zone obstructed",
            BarcodeFindingKind.LOW_DECODE_MARGIN: "Low decoding margin",
            BarcodeFindingKind.BAR_WIDTH: "Bar width deviation",
            BarcodeFindingKind.ARTWORK_MISMATCH: "Artwork does not match zone",
            BarcodeFindingKind.UNEXPECTED_BARCODE: "Unexpected barcode",
        }[self]


class SymbolStatus(str, Enum):
    OK = "ok"
    DAMAGED = "damaged"  # decodes with the right content, but damage or little reserve
    WRONG_CONTENT = "wrong content"
    WRONG_FORMAT = "wrong format"
    NOT_DECODED = "not decoded"
    NO_REFERENCE = "no reference"  # nothing to compare with (no value in the artwork or the zone)


@dataclass(frozen=True)
class BarcodeFinding:
    id: int
    kind: BarcodeFindingKind
    symbol_id: int | None
    zone: str | None
    format: str
    critical: bool
    expected: str
    decoded: str
    x: int
    y: int
    width: int
    height: int
    severity: DefectSeverity
    reliable: bool
    notes: tuple[str, ...]
    image_score: float = 0.0

    @property
    def label(self) -> str:
        return f"B{self.id}"

    @property
    def is_reliable(self) -> bool:
        return self.reliable

    def describe(self) -> str:
        if self.kind is BarcodeFindingKind.WRONG_CONTENT:
            return f"Wrong content: expected '{self.expected}', decoded '{self.decoded}'"
        return self.kind.label


@dataclass(frozen=True)
class SymbolResult:
    id: int
    name: str
    zone: str | None
    critical: bool
    format: str  # expected format key ("" if any)
    x: int
    y: int
    width: int
    height: int
    expected_value: str
    expected_source: str  # "zone", "artwork" or "" (none)
    artwork_value: str
    print_value: str
    print_format: str
    status: SymbolStatus
    scan_success: float | None = None  # linear: share of strips that decode
    strips: tuple[int, int] | None = None  # (decoded, total)
    unused_ec: float | None = None  # QR: from the decoder, print
    ec_level: str = ""
    module_errors: int | None = None  # QR: wrong modules
    module_count: int | None = None
    bar_growth_modules: float | None = None  # linear: average bar width change (+ = wider)
    module_px: float | None = None
    finding_ids: tuple[int, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def format_label(self) -> str:
        return FORMAT_LABELS.get(self.format or self.print_format, "any")


@dataclass(frozen=True)
class BarcodeReport:
    available: bool
    library: str
    symbols: tuple[SymbolResult, ...]
    findings: tuple[BarcodeFinding, ...]
    issues: tuple[QualityIssue, ...]
    seconds: float = 0.0
    notes: tuple[str, ...] = ()

    @property
    def level(self) -> QualityLevel:
        return level_from_issues(self.issues)

    @property
    def is_reliable(self) -> bool:
        return self.level is not QualityLevel.POOR

    def by_id(self, finding_id: int) -> BarcodeFinding | None:
        return next((f for f in self.findings if f.id == finding_id), None)

    def symbol(self, symbol_id: int) -> SymbolResult | None:
        return next((s for s in self.symbols if s.id == symbol_id), None)


def unavailable_report(reason: str, required: bool) -> BarcodeReport:
    severity = Severity.CRITICAL if required else Severity.WARNING
    issue = QualityIssue("barcodes_unavailable", severity, f"Barcode inspection is not available: {reason}")
    return BarcodeReport(False, "", (), (), (issue,))


def locate_symbols(reference: np.ndarray, formats: tuple[str, ...]) -> list[Symbol]:
    """Barcodes in the artwork (comparison grid). Raises BarcodeUnavailable."""
    return decode(reference, formats)


# ---------------------------------------------------------------- geometry helpers


def _clip_box(box: Box, w: int, h: int) -> Box:
    x0, y0 = max(0, box[0]), max(0, box[1])
    x1, y1 = min(w, box[0] + box[2]), min(h, box[1] + box[3])
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def _grow(box: Box, margin: float, w: int, h: int) -> Box:
    m = int(round(margin))
    return _clip_box((box[0] - m, box[1] - m, box[2] + 2 * m, box[3] + 2 * m), w, h)


def _centre(box: Box) -> tuple[float, float]:
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def _inside(point: tuple[float, float], box: Box) -> bool:
    return box[0] <= point[0] < box[0] + box[2] and box[1] <= point[1] < box[1] + box[3]


def _peak(score: np.ndarray, box: Box) -> float:
    x, y, w, h = _clip_box(box, score.shape[1], score.shape[0])
    return float(score[y : y + h, x : x + w].max()) if w and h else 0.0


def _union(boxes: list[Box]) -> Box:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return x0, y0, x1 - x0, y1 - y0


def _overlaps(a: Box, b: Box) -> bool:
    return not (a[0] >= b[0] + b[2] or b[0] >= a[0] + a[2] or a[1] >= b[1] + b[3] or b[1] >= a[1] + a[3])


def _gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _module_px(symbol: Symbol, reference_gray: np.ndarray) -> float | None:
    """Module (narrowest bar / QR cell) size in grid pixels, from the artwork."""
    x, y, w, h = symbol.box
    if symbol.format == "qr":
        if symbol.version is None:
            return None
        side = float(np.hypot(symbol.corners[1][0] - symbol.corners[0][0], symbol.corners[1][1] - symbol.corners[0][1]))
        return side / (17 + 4 * symbol.version)
    across = w if _bars_vertical(symbol) else h
    if symbol.format in EAN_UPC:
        return across / EAN_UPC_MODULES
    # Code 128: the narrowest runs along the middle line of the artwork symbol are one module.
    crop = reference_gray[y : y + h, x : x + w]
    if crop.size == 0:
        return None
    line = crop[crop.shape[0] // 2, :] if _bars_vertical(symbol) else crop[:, crop.shape[1] // 2]
    dark = line < 128
    edges = np.flatnonzero(np.diff(dark.astype(np.int8))) + 1
    runs = np.diff(np.concatenate(([0], edges, [len(dark)])))
    runs = runs[1:-1] if len(runs) > 2 else runs
    if len(runs) == 0:
        return None
    return float(np.percentile(runs, 10))


def _bars_vertical(symbol: Symbol) -> bool:
    return round(symbol.orientation / 90) % 2 == 0


# ---------------------------------------------------------------- linear symbols


@dataclass
class _LinearAnalysis:
    module_px: float
    strips_ok: int = 0
    strips_total: int = 0
    failing_strips: list[Box] = field(default_factory=list)
    damage: list[tuple[Box, float]] = field(default_factory=list)  # box, area in square modules
    quiet_zone: list[tuple[Box, float]] = field(default_factory=list)
    bar_growth_modules: float | None = None


def _rotate_for_bars(a: np.ndarray, vertical: bool) -> np.ndarray:
    return a if vertical else np.ascontiguousarray(np.rot90(a))


def _box_back(box: Box, vertical: bool, crop_w: int, ox: int, oy: int) -> Box:
    """A box in the (possibly rotated) crop back to grid coordinates."""
    x, y, w, h = box
    if not vertical:  # np.rot90: rotated (row r, col c) = original (row c, col W-1-r)
        x, y, w, h = crop_w - y - h, x, h, w
    return x + ox, y + oy, w, h


def _analyse_linear(
    art: Symbol, expected: str, ref_gray: np.ndarray, reg_gray: np.ndarray, mask: np.ndarray, cfg: BarcodeConfig,
) -> _LinearAnalysis | None:
    module = _module_px(art, ref_gray)
    if module is None or module < 1.0:
        return None
    vertical = _bars_vertical(art)
    gh, gw = ref_gray.shape
    x, y, w, h = art.box
    quiet = cfg.quiet_zone_modules * module
    pad_along = int(round(quiet + 2))
    if vertical:
        crop = _clip_box((x - pad_along, y, w + 2 * pad_along, h), gw, gh)
    else:
        crop = _clip_box((x, y - pad_along, w, h + 2 * pad_along), gw, gh)
    cx, cy, cw, ch = crop
    if cw < 8 or ch < 8:
        return None
    ref = _rotate_for_bars(ref_gray[cy : cy + ch, cx : cx + cw], vertical)
    prn = _rotate_for_bars(reg_gray[cy : cy + ch, cx : cx + cw], vertical)
    valid = _rotate_for_bars(mask[cy : cy + ch, cx : cx + cw], vertical) > 0
    rows, cols = ref.shape
    # Columns of the symbol (between the guard bars) within the rotated crop.
    bar_cols = np.flatnonzero((ref < 128).any(axis=0))
    if len(bar_cols) == 0:
        return None
    c0, c1 = int(bar_cols[0]), int(bar_cols[-1]) + 1
    analysis = _LinearAnalysis(module)

    art_dark = ref < 128
    threshold, _ = cv2.threshold(prn[:, max(0, c0 - 2) : c1 + 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    prn_dark = prn < threshold
    # Damage: the print differs from the artwork, even allowing one pixel of shift across the bars.
    left = np.roll(art_dark, 1, axis=1)
    right = np.roll(art_dark, -1, axis=1)
    wrong = (prn_dark != art_dark) & (prn_dark != left) & (prn_dark != right) & valid
    wrong[:, :c0] = False
    wrong[:, c1:] = False
    wrong = cv2.morphologyEx(wrong.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    min_area = cfg.min_damage_modules2 * module * module
    count, _, stats, _ = cv2.connectedComponentsWithStats(wrong, connectivity=8)
    for i in range(1, count):
        bx, by, bw, bh, area = (int(v) for v in stats[i])
        if area >= min_area:
            analysis.damage.append((_box_back((bx, by, bw, bh), vertical, cw, cx, cy), area / (module * module)))

    # Quiet zones: dark print where the artwork is clear on both sides of the symbol.
    clear = ~art_dark.any(axis=0)
    for side_cols in (np.arange(max(0, c0 - int(quiet)), c0), np.arange(c1, min(cols, c1 + int(quiet)))):
        side_cols = side_cols[clear[side_cols]] if len(side_cols) else side_cols
        if len(side_cols) == 0:
            continue
        zone = np.zeros_like(art_dark)
        zone[:, side_cols] = True
        marks = (prn_dark & zone & valid).astype(np.uint8)
        marks = cv2.morphologyEx(marks, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(marks, connectivity=8)
        for i in range(1, n):
            bx, by, bw, bh, area = (int(v) for v in st[i])
            if area >= min_area:
                analysis.quiet_zone.append((_box_back((bx, by, bw, bh), vertical, cw, cx, cy), area / (module * module)))

    # Average bar width change: extra dark pixels per bar along each row.
    middle = art_dark[rows // 2, c0:c1]
    bars = int(np.count_nonzero(np.diff(np.concatenate(([0], middle.astype(np.int8)))) == 1))
    if bars:
        growth = (prn_dark[:, c0:c1].sum(axis=1) - art_dark[:, c0:c1].sum(axis=1)) / bars
        analysis.bar_growth_modules = float(np.median(growth)) / module

    # Decoding in horizontal strips (a damaged area stops the strips through it).
    n_strips = cfg.scan_strips
    strip_h = rows / n_strips
    target = canonical_value(art.format, expected) if expected else None
    lo, hi = max(0, c0 - int(quiet)), min(cols, c1 + int(quiet))
    for i in range(n_strips):
        r0, r1 = int(round(i * strip_h)), int(round((i + 1) * strip_h))
        if r1 - r0 < 1:
            continue
        strip = prn[r0:r1, lo:hi]
        strip = cv2.resize(strip, (strip.shape[1], max(24, strip.shape[0])), interpolation=cv2.INTER_NEAREST)
        found = decode(strip, (art.format,) if art.format not in EAN_UPC else ("ean13", "upca"), thorough=False)
        ok = any(target is None or canonical_value(s.format, s.text) == target for s in found)
        analysis.strips_total += 1
        if ok:
            analysis.strips_ok += 1
        else:
            analysis.failing_strips.append(_box_back((lo, r0, hi - lo, r1 - r0), vertical, cw, cx, cy))
    return analysis


# ---------------------------------------------------------------- QR codes


@dataclass
class _QrAnalysis:
    size: int
    module_px: float
    errors: np.ndarray  # bool (size x size)
    finder_errors: int
    clusters: list[tuple[Box, int]]  # box, wrong modules
    quiet_zone: list[tuple[Box, float]]


def _qr_point(corners, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = corners  # TL, TR, BR, BL
    top_x, top_y = x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
    bot_x, bot_y = x3 + (x2 - x3) * u, y3 + (y2 - y3) * u
    return top_x + (bot_x - top_x) * v, top_y + (bot_y - top_y) * v


def _analyse_qr(art: Symbol, ref_gray: np.ndarray, reg_gray: np.ndarray, mask: np.ndarray, cfg: BarcodeConfig) -> _QrAnalysis | None:
    if art.version is None:
        return None
    n = 17 + 4 * art.version
    module = _module_px(art, ref_gray)
    if module is None or module < 1.5:
        return None
    centres = (np.arange(n) + 0.5) / n
    u, v = np.meshgrid(centres, centres)
    px, py = _qr_point(art.corners, u, v)
    r = max(1, int(round(0.25 * module)))
    kernel = (2 * r + 1, 2 * r + 1)
    ref_mean = cv2.blur(ref_gray.astype(np.float32), kernel)
    reg_mean = cv2.blur(reg_gray.astype(np.float32), kernel)
    xi = np.clip(np.round(px).astype(int), 0, ref_gray.shape[1] - 1)
    yi = np.clip(np.round(py).astype(int), 0, ref_gray.shape[0] - 1)
    art_dark = ref_mean[yi, xi] < 128
    values = reg_mean[yi, xi]
    if art_dark.all() or (~art_dark).all():
        return None
    threshold = (np.median(values[art_dark]) + np.median(values[~art_dark])) / 2
    errors = ((values < threshold) != art_dark) & (mask[yi, xi] > 0)
    finder = np.zeros((n, n), bool)
    finder[:8, :8] = finder[:8, -8:] = finder[-8:, :8] = True
    clusters = []
    # Wrong modules within one module of each other form one damaged area (an erased patch also
    # covers modules that were light anyway).
    grouped = cv2.dilate(errors.astype(np.uint8), np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    half = module / 2
    for i in range(1, count):
        cells = np.argwhere((labels == i) & errors)
        xs = px[cells[:, 0], cells[:, 1]]
        ys = py[cells[:, 0], cells[:, 1]]
        box = (int(xs.min() - half), int(ys.min() - half), int(xs.max() - xs.min() + module) + 1, int(ys.max() - ys.min() + module) + 1)
        clusters.append((box, len(cells)))

    # Quiet zone: dark print in a ring around the symbol where the artwork is clear.
    quiet = []
    x, y, w, h = art.box
    ring = int(round(cfg.qr_quiet_zone_modules * module))
    outer = _clip_box((x - ring, y - ring, w + 2 * ring, h + 2 * ring), ref_gray.shape[1], ref_gray.shape[0])
    ox, oy, ow, oh = outer
    zone = np.ones((oh, ow), bool)
    ix0, iy0 = x - ox, y - oy
    zone[max(0, iy0) : iy0 + h, max(0, ix0) : ix0 + w] = False
    ref_crop, reg_crop = ref_gray[oy : oy + oh, ox : ox + ow], reg_gray[oy : oy + oh, ox : ox + ow]
    marks = zone & (ref_crop >= 128) & (reg_crop < threshold) & (mask[oy : oy + oh, ox : ox + ow] > 0)
    marks = cv2.morphologyEx(marks.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    min_area = cfg.min_damage_modules2 * module * module
    count, _, st, _ = cv2.connectedComponentsWithStats(marks, connectivity=8)
    for i in range(1, count):
        bx, by, bw, bh, area = (int(val) for val in st[i])
        if area >= min_area:
            quiet.append(((bx + ox, by + oy, bw, bh), area / (module * module)))
    return _QrAnalysis(n, module, errors, int((errors & finder).sum()), clusters, quiet)


# ---------------------------------------------------------------- main entry


@dataclass
class _Target:
    name: str
    zone: BarcodeZone | None
    region: Box
    art: Symbol | None
    format: str  # expected format key or "" (any)
    expected: str
    source: str


def _targets(art_symbols: list[Symbol], zones: tuple[BarcodeZone, ...], gw: int, gh: int) -> list[_Target]:
    targets, used = [], set()
    for zone in zones:
        rect = zone.rect(gw, gh)
        inside = [
            s for s in art_symbols
            if _inside(_centre(s.box), rect) and (zone.format == ANY_FORMAT or same_symbology(s.format, zone.format))
        ]
        art = max(inside, key=lambda s: s.box[2] * s.box[3]) if inside else None
        if art is not None:
            used.add(id(art))
        fmt = zone.format if zone.format != ANY_FORMAT else (art.format if art else "")
        if zone.expected_value:
            expected, source = zone.expected_value, "zone"
        elif art is not None:
            expected, source = art.text, "artwork"
        else:
            expected, source = "", ""
        targets.append(_Target(zone.name, zone, art.box if art else rect, art, fmt, expected, source))
    counters: dict[str, int] = {}
    for s in sorted(art_symbols, key=lambda s: (s.box[1], s.box[0])):
        if id(s) in used:
            continue
        counters[s.format] = counters.get(s.format, 0) + 1
        targets.append(_Target(f"{s.label} {counters[s.format]}", None, s.box, s, s.format, s.text, "artwork"))
    return targets


def _severity(name: str, critical: bool, cfg: BarcodeConfig) -> DefectSeverity:
    severity = DefectSeverity(name)
    if critical:
        severity = max(severity, DefectSeverity(cfg.critical_zone_severity), key=lambda s: s.rank)
    return severity


def inspect_barcodes(
    reference: np.ndarray,
    registered: np.ndarray,
    combined_score: np.ndarray,
    mask: np.ndarray,
    zones: tuple[BarcodeZone, ...],
    cfg: BarcodeConfig,
    ignore: tuple[Box, ...] = (),
) -> BarcodeReport:
    """Inspect the barcodes of the registered print (all images on the comparison grid).

    Barcodes found in the artwork with their centre in an ``ignore`` box (ignored
    regions) are not inspected unless a zone asks for them. Raises
    BarcodeUnavailable if the decoding library is missing.
    """
    started = time.perf_counter()
    gh, gw = reference.shape[:2]
    ref_gray, reg_gray = _gray(reference), _gray(registered)
    art_symbols = [s for s in locate_symbols(reference, cfg.formats)]
    targets = [
        t for t in _targets(art_symbols, zones, gw, gh)
        if t.zone is not None or not any(_inside(_centre(t.region), box) for box in ignore)
    ]

    findings: list[BarcodeFinding] = []
    results: list[SymbolResult] = []
    for number, t in enumerate(targets, start=1):
        findings_before = len(findings)
        critical = bool(t.zone and t.zone.critical)
        zone_name = t.zone.name if t.zone else None
        module = _module_px(t.art, ref_gray) if t.art else None
        margin = cfg.region_margin_modules * (module or max(t.region[2], t.region[3]) / 60)
        search = _grow(t.region, margin, gw, gh)
        sx, sy, sw, sh = search
        notes: list[str] = []

        def add(kind, severity, reliable, box, finding_notes, expected="", decoded="", fmt=None):
            findings.append(BarcodeFinding(
                0, kind, number, zone_name, fmt or t.format, critical, expected, decoded, *box,
                _severity(severity, critical, cfg), reliable, tuple(finding_notes), _peak(combined_score, box),
            ))

        # Decode the print in the region of the symbol.
        found = [s.moved(sx, sy) for s in decode(registered[sy : sy + sh, sx : sx + sw], cfg.formats)]
        centre = _centre(t.region)
        found.sort(key=lambda s: np.hypot(*np.subtract(_centre(s.box), centre)))
        printed = next((s for s in found if not t.format or same_symbology(s.format, t.format)), None)
        other = next((s for s in found if t.format and not same_symbology(s.format, t.format)), None)

        expected_canonical = canonical_value(t.format or (printed.format if printed else ""), t.expected) if t.expected else ""
        if t.art is not None and t.source == "zone":
            art_value = canonical_value(t.art.format, t.art.text)
            if art_value != expected_canonical:
                add(BarcodeFindingKind.ARTWORK_MISMATCH, cfg.severity_wrong_content, False, t.region,
                    [f"The artwork's barcode encodes '{t.art.text}', the zone expects '{t.expected}': the "
                     "artwork or the zone value is wrong."], t.expected, t.art.text)
        if t.art is None:
            notes.append("No barcode was decoded in the artwork here; the print is compared with the zone value."
                          if t.expected else "No barcode was decoded in the artwork here and the zone gives no value.")

        # Damage measured against the artwork.
        linear = qr = None
        if t.art is not None and t.art.format in LINEAR:
            # Strips measure how robustly the print decodes, so they must give the print's own value.
            linear = _analyse_linear(t.art, printed.text if printed else t.art.text, ref_gray, reg_gray, mask, cfg)
        elif t.art is not None and t.art.format == "qr":
            qr = _analyse_qr(t.art, ref_gray, reg_gray, mask, cfg)
        damage_boxes: list[tuple[Box, float]] = []
        if linear is not None:
            damage_boxes = linear.damage
        elif qr is not None:
            damage_boxes = [(b, float(c)) for b, c in qr.clusters]
        corroborated_damage = [
            (b, a) for b, a in damage_boxes if _peak(combined_score, b) >= cfg.damage_corroboration_score
        ]

        status = SymbolStatus.OK
        decoded_value = printed.text if printed else ""
        if printed is None:
            if other is not None:
                status = SymbolStatus.WRONG_FORMAT
                add(BarcodeFindingKind.WRONG_FORMAT, cfg.severity_wrong_format, True, other.box,
                    [f"Expected {FORMAT_LABELS[t.format]}, the print has {other.label} '{other.text}'."],
                    t.expected, other.text, other.format)
            elif t.art is None and not t.expected:
                status = SymbolStatus.NO_REFERENCE
            else:
                status = SymbolStatus.NOT_DECODED
                reasons = []
                if corroborated_damage:
                    reasons.append(f"damage visible in the image ({len(corroborated_damage)} area(s))")
                if qr is not None and qr.finder_errors:
                    reasons.append(f"{qr.finder_errors} finder pattern module(s) wrong")
                reliable = bool(reasons)
                note = (
                    "The print could not be decoded; " + "; ".join(reasons) + "." if reliable else
                    "The print could not be decoded, but no damage is visible: the photo may be too blurred, "
                    "small or reflective. Check with a scanner."
                )
                add(BarcodeFindingKind.NOT_DECODED, cfg.severity_not_decoded, reliable, t.region, [note, NOT_ISO],
                    t.expected)
        else:
            value = canonical_value(printed.format, printed.text)
            if expected_canonical and value != expected_canonical:
                status = SymbolStatus.WRONG_CONTENT
                add(BarcodeFindingKind.WRONG_CONTENT, cfg.severity_wrong_content, True, printed.box,
                    ["The barcode decodes (with valid check digit / error correction) to a different value."],
                    t.expected, printed.text, printed.format)
            elif not expected_canonical:
                status = SymbolStatus.NO_REFERENCE
                notes.append("Decoded, but there is no expected value to compare with.")

        # Damage findings (also when the code decodes - it may fail on a scanner).
        decoding_hit = printed is None or (
            linear is not None and linear.strips_total and linear.strips_ok / linear.strips_total < cfg.min_scan_success
        ) or (printed is not None and printed.unused_ec is not None and printed.unused_ec < cfg.min_unused_ec)
        kind = BarcodeFindingKind.DAMAGED_MODULES if qr is not None else BarcodeFindingKind.DAMAGED_BARS
        for box, amount in damage_boxes:
            score = _peak(combined_score, box)
            reliable = score >= cfg.damage_corroboration_score
            affects = decoding_hit or (
                linear is not None and any(_overlaps(box, s) for s in linear.failing_strips)
            )
            unit = "wrong module(s)" if qr is not None else "square modules"
            add(kind, cfg.severity_damage_affecting_decoding if affects else cfg.severity_damage, reliable, box,
                [f"{amount:.0f} {unit} differ from the artwork (comparison score {score:.1f})"
                 + ("; the decoding is affected." if affects else "; the code still decodes in this photo."),
                 NOT_ISO])
        if qr is not None and qr.finder_errors and not any(f.kind is BarcodeFindingKind.NOT_DECODED for f in findings[findings_before:]):
            notes.append(f"{qr.finder_errors} module(s) of the finder patterns differ from the artwork.")

        quiet_boxes = (linear.quiet_zone if linear else []) + (qr.quiet_zone if qr else [])
        for box, area in quiet_boxes:
            score = _peak(combined_score, box)
            add(BarcodeFindingKind.QUIET_ZONE, cfg.severity_quiet_zone, score >= cfg.damage_corroboration_score, box,
                [f"Dark marks ({area:.1f} square modules) in the quiet zone, which is clear in the artwork."])

        strips = None
        scan_success = None
        if linear is not None and linear.strips_total:
            strips = (linear.strips_ok, linear.strips_total)
            scan_success = linear.strips_ok / linear.strips_total
            if printed is not None and scan_success < cfg.min_scan_success:
                boxes = linear.failing_strips
                explained = any(_overlaps(b, s) for b, _ in corroborated_damage for s in boxes)
                add(BarcodeFindingKind.LOW_DECODE_MARGIN, cfg.severity_low_margin, explained, _union(boxes),
                    [f"Only {linear.strips_ok} of {linear.strips_total} strips across the code decode "
                     f"(needed {cfg.min_scan_success:.0%})."
                     + ("" if explained else " No damage is visible there; the photo may be the cause."), NOT_ISO])
            if linear.bar_growth_modules is not None and abs(linear.bar_growth_modules) > cfg.max_bar_growth_modules:
                wider = "wider" if linear.bar_growth_modules > 0 else "narrower"
                add(BarcodeFindingKind.BAR_WIDTH, cfg.severity_bar_width, False, t.art.box,
                    [f"Bars are on average {abs(linear.bar_growth_modules):.2f} module {wider} than in the artwork "
                     "(ink spread or a thin print; also depends on photo sharpness).", NOT_ISO])
        if printed is not None and printed.unused_ec is not None and printed.unused_ec < cfg.min_unused_ec:
            explained = bool(corroborated_damage)
            add(BarcodeFindingKind.LOW_DECODE_MARGIN, cfg.severity_low_margin, explained, printed.box,
                [f"The decoder needed {1 - printed.unused_ec:.0%} of the error correction "
                 f"(at most {1 - cfg.min_unused_ec:.0%} allowed)."
                 + ("" if explained else " No damage is visible; the photo may be the cause."), NOT_ISO])

        if status is SymbolStatus.OK and len(findings) > findings_before:
            status = SymbolStatus.DAMAGED
        x, y, w, h = t.region
        results.append(SymbolResult(
            id=number, name=t.name, zone=zone_name, critical=critical, format=t.format, x=x, y=y, width=w, height=h,
            expected_value=t.expected, expected_source=t.source, artwork_value=t.art.text if t.art else "",
            print_value=decoded_value, print_format=printed.format if printed else "", status=status,
            scan_success=scan_success, strips=strips,
            unused_ec=printed.unused_ec if printed else None, ec_level=(printed or t.art).ec_level if (printed or t.art) else "",
            module_errors=int(qr.errors.sum()) if qr else None, module_count=qr.size * qr.size if qr else None,
            bar_growth_modules=linear.bar_growth_modules if linear else None,
            module_px=module, finding_ids=tuple(range(findings_before, len(findings))), notes=tuple(notes),
        ))

    # Barcodes in the print where the artwork has none.
    regions = [_grow((r.x, r.y, r.width, r.height), 10, gw, gh) for r in results]
    for s in decode(registered, cfg.formats):
        if any(_inside(_centre(s.box), region) for region in regions):
            continue
        findings.append(BarcodeFinding(
            0, BarcodeFindingKind.UNEXPECTED_BARCODE, None, None, s.format, False, "", s.text, *s.box,
            DefectSeverity(cfg.severity_unexpected_barcode), True,
            (f"{s.label} '{s.text}' in the print where the artwork has no barcode.",), _peak(combined_score, s.box),
        ))

    order = sorted(range(len(findings)), key=lambda i: (-findings[i].severity.rank, findings[i].y, findings[i].x))
    new_id = {old: rank + 1 for rank, old in enumerate(order)}
    findings = [replace(findings[i], id=new_id[i]) for i in order]
    results = [replace(r, finding_ids=tuple(sorted(new_id[i] for i in r.finding_ids))) for r in results]
    notes = [] if targets else ["No barcode was found in the artwork."]
    seconds = time.perf_counter() - started
    log.info("Barcode inspection: %d symbol(s), %d finding(s), %.1f s", len(results), len(findings), seconds)
    return BarcodeReport(True, library_version(), tuple(results), tuple(findings), (), seconds, tuple(notes))
