"""Defect localisation, categorisation, severity and evidence.

Input: the comparison result (candidate mask, maps, candidates) and the
registered photo / artwork on the comparison grid.

1. Grouping: candidate regions closer than ``merge_distance_mm`` form one defect
   (fragments of one broken character, the two sides of a shifted edge). Then a
   region of missing ink and a region of extra ink within ``shape_pair_distance_mm``
   of each other are joined: together they are one element printed shifted or
   out of shape (a *shape difference*), not two unrelated defects.
2. Noise: a region is removed only if it is smaller than ``noise_max_area_mm2``,
   weaker than ``noise_max_score`` *and* not reliably detected. Removed regions
   are kept in ``DefectReport.filtered`` with the reason - nothing disappears silently.
3. Measurements: bounding box of the detected region; area and oriented
   length / thickness (pixels and mm) of its *half-maximum core* - pixels with
   at least half the peak colour difference. Optical blur and the detection
   hysteresis widen the detected region by a few pixels; the half-maximum
   extent recovers the size of the printed defect itself. Also gray-level and
   colour (CIE Lab ΔE, ΔL, ΔC) differences, scores and evidence crops.
4. Two separate judgements:
   - *detection reliability* - how sure we are that something is different
     (score, difference size, structural evidence);
   - *category certainty* - how sure we are about what it is. "Possible text
     defect" and "Possible color defect" are never claimed as verified: the
     text area is recognised heuristically and no character recognition is done.
5. Severity (minor / major / critical) from size in mm, with a stricter size
   limit in text areas, lowered one level for faint differences.

Units: grid pixels unless suffixed ``_mm``/``_mm2``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

import cv2
import numpy as np

from print_inspector.config import DefectsConfig
from print_inspector.core.comparison import ComparisonResult

_GRAY = np.array([0.114, 0.587, 0.299], np.float32)  # BGR weights


class DefectCategory(str, Enum):
    MISSING_PRINT = "missing_print"
    EXTRA_PRINT = "extra_print"
    SHAPE_DIFFERENCE = "shape_difference"
    EDGE_DIFFERENCE = "edge_difference"
    POSSIBLE_TEXT = "possible_text_defect"
    POSSIBLE_COLOR = "possible_color_defect"
    UNKNOWN = "unknown_difference"

    @property
    def label(self) -> str:
        return {
            DefectCategory.MISSING_PRINT: "Missing print",
            DefectCategory.EXTRA_PRINT: "Extra print",
            DefectCategory.SHAPE_DIFFERENCE: "Shape difference",
            DefectCategory.EDGE_DIFFERENCE: "Edge difference",
            DefectCategory.POSSIBLE_TEXT: "Possible text defect",
            DefectCategory.POSSIBLE_COLOR: "Possible color defect",
            DefectCategory.UNKNOWN: "Unknown visual difference",
        }[self]


class DefectSeverity(str, Enum):
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return ("minor", "major", "critical").index(self.value)


class DetectionReliability(str, Enum):
    RELIABLE = "reliable"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, eq=False)
class DefectEvidence:
    """Image crops around a defect (all the same size), for the operator and reports."""

    origin: tuple[int, int]  # top-left of the crops on the comparison grid
    artwork: np.ndarray  # BGR, artwork on the grid
    expected: np.ndarray  # BGR, artwork as it should look in this photo
    printed: np.ndarray  # BGR, registered photo
    difference: np.ndarray  # uint8 |printed - expected| (gray levels)
    score: np.ndarray  # uint8 combined score x 50 (50 = detection threshold)
    region: np.ndarray  # uint8 255 = defect region


@dataclass(frozen=True, eq=False)
class Defect:
    id: int
    category: DefectCategory
    category_certain: bool
    category_note: str
    reliability: DetectionReliability
    reliability_notes: tuple[str, ...]
    severity: DefectSeverity
    severity_notes: tuple[str, ...]
    x: int
    y: int
    width: int
    height: int
    centroid: tuple[float, float]
    area_px: int  # half-maximum core (see module docstring)
    length_px: float  # oriented extent of the core along the defect
    thickness_px: float  # oriented extent of the core across the defect
    area_mm2: float
    length_mm: float
    thickness_mm: float
    score: float  # peak, 1.0 = detection threshold
    peak_sigma: float
    max_difference: float  # largest colour-channel difference, gray levels
    mean_difference: float  # mean |gray difference| over the region
    signed_difference: float  # mean (printed - expected) gray in the core; < 0 = darker
    delta_e: float  # mean CIE76 ΔE (Lab) in the core
    delta_l: float
    delta_c: float
    polarity: str
    ink_fraction: float
    edge_fraction: float
    text_overlap: float
    min_ssim: float
    missing_edge_px: int
    extra_edge_px: int
    scale: int
    candidate_ids: tuple[int, ...]
    evidence: DefectEvidence
    critical_region: str = ""  # name of the critical region it touches (core/regions.py)

    @property
    def category_label(self) -> str:
        label = self.category.label
        if not self.category_certain and not label.startswith("Possible"):
            label += " (uncertain)"
        return label

    @property
    def is_reliable(self) -> bool:
        return self.reliability is DetectionReliability.RELIABLE


@dataclass(frozen=True)
class FilteredRegion:
    x: int
    y: int
    width: int
    height: int
    area_mm2: float
    score: float
    reason: str


@dataclass(frozen=True, eq=False)
class DefectReport:
    defects: tuple[Defect, ...]
    filtered: tuple[FilteredRegion, ...]
    px_per_mm: float
    scale_estimated: bool  # True when mm values rest on an assumed label size
    text_mask: np.ndarray = field(repr=False)

    def by_id(self, defect_id: int) -> Defect | None:
        return next((d for d in self.defects if d.id == defect_id), None)


# ---------------------------------------------------------------- artwork analysis


def text_like_mask(gray: np.ndarray, px_per_mm: float, cfg: DefectsConfig) -> np.ndarray:
    """Areas of the artwork that look like text: character-sized shapes in rows.

    Dark-on-light and light-on-dark (knock-out) characters are both considered.
    Heuristic only - it marks where a defect *may* affect text.
    """
    h_min, h_max = cfg.text_min_height_mm * px_per_mm, cfg.text_max_height_mm * px_per_mm
    boxes = []
    for binary in (gray < 128, gray >= 128):
        count, _, stats, centres = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
        for i in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[i])
            if not h_min <= h <= h_max or not 0.08 * h <= w <= 2.0 * h or area < 6:
                continue
            if not 0.1 <= area / (w * h) <= 0.9:
                continue
            boxes.append((x, y, w, h, centres[i][0], centres[i][1]))
    mask = np.zeros(gray.shape, np.uint8)
    if len(boxes) < 2:
        return mask
    b = np.array(boxes, np.float64)
    hh = b[:, 3]
    dx = np.abs(b[:, 4][:, None] - b[:, 4][None, :])
    dy = np.abs(b[:, 5][:, None] - b[:, 5][None, :])
    ratio = hh[None, :] / hh[:, None]
    neighbour = (dx <= 3.0 * hh[:, None]) & (dy <= 0.5 * hh[:, None]) & (ratio >= 0.5) & (ratio <= 2.0)
    np.fill_diagonal(neighbour, False)
    for (x, y, w, h, _, _), keep in zip(boxes, neighbour.any(axis=1)):
        if keep:
            pad = int(round(0.3 * h))
            cv2.rectangle(mask, (int(x) - pad, int(y) - pad), (int(x + w) + pad, int(y + h) + pad), 255, -1)
    return mask


def _edge_band(gray: np.ndarray, band: int) -> np.ndarray:
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    return cv2.dilate(edges, kernel) > 0


def _pair_shape_changes(
    labels: np.ndarray,
    members: dict[int, list],
    mask: np.ndarray,
    registered: np.ndarray,
    expected: np.ndarray,
    distance: float,
) -> tuple[np.ndarray, dict[int, list]]:
    """Join a missing-ink group and an extra-ink group lying within ``distance`` px."""
    info = {}
    for group in members:
        ys, xs = np.nonzero((labels == group) & mask)
        if xs.size == 0:
            continue
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        r = (labels[y0:y1, x0:x1] == group) & mask[y0:y1, x0:x1]
        diff = (registered[y0:y1, x0:x1].astype(np.float32) @ _GRAY) - (expected[y0:y1, x0:x1].astype(np.float32) @ _GRAY)
        info[group] = ((x0, y0, x1, y1), float(np.mean(diff[r])))
    parent = {g: g for g in members}

    def root(g: int) -> int:
        while parent[g] != g:
            g = parent[g]
        return g

    groups = [g for g in members if g in info]
    for i, a in enumerate(groups):
        for b in groups[i + 1 :]:
            (ax0, ay0, ax1, ay1), da = info[a]
            (bx0, by0, bx1, by1), db = info[b]
            if da * db >= 0:  # same polarity: not a shifted element
                continue
            gap = max(0, max(ax0, bx0) - min(ax1, bx1), max(ay0, by0) - min(ay1, by1))
            if gap <= distance:
                parent[root(b)] = root(a)
    if all(root(g) == g for g in parent):
        return labels, members
    merged: dict[int, list] = {}
    relabel = np.arange(int(labels.max()) + 1)
    for g in members:
        r = root(g)
        merged.setdefault(r, []).extend(members[g])
        relabel[g] = r
    return relabel[labels], merged


# ---------------------------------------------------------------- per-region assessment


def _lab(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def _categorise(
    polarity_share: tuple[float, float],
    ink_fraction: float,
    edge_fraction: float,
    thickness: float,
    text_overlap: float,
    delta_l: float,
    delta_c: float,
    cfg: DefectsConfig,
) -> tuple[DefectCategory, bool, str]:
    darker, lighter = polarity_share
    likely = cfg.likely_min_fraction
    # A colour change with little lightness change.
    if delta_c >= cfg.color_min_delta_c and abs(delta_l) <= cfg.color_max_lightness_ratio * delta_c:
        return DefectCategory.POSSIBLE_COLOR, False, (
            f"Mainly a colour change (ΔC {delta_c:.1f}, ΔL {delta_l:+.1f}); colour accuracy is not "
            "measured against a colour standard."
        )
    if darker >= 0.25 and lighter >= 0.25:
        base, certain, note = DefectCategory.SHAPE_DIFFERENCE, True, (
            f"Both missing and extra ink ({lighter:.0%} lighter, {darker:.0%} darker)."
        )
    elif lighter > darker:
        certain = lighter >= likely and ink_fraction >= likely
        if ink_fraction >= 0.5:
            base, note = DefectCategory.MISSING_PRINT, f"Lighter than expected where the artwork has ink ({ink_fraction:.0%})."
        else:
            base, certain, note = DefectCategory.UNKNOWN, False, "Lighter than expected on bare stock (glare, paper flaw?)."
    else:
        certain = darker >= likely and ink_fraction <= 1.0 - likely
        if ink_fraction <= 0.5:
            base, note = DefectCategory.EXTRA_PRINT, f"Darker than expected where the artwork has no ink ({1 - ink_fraction:.0%})."
        else:
            base, note = DefectCategory.EXTRA_PRINT, "Darker than expected on existing ink (heavier ink?)."
            certain = False
    if (
        base in (DefectCategory.SHAPE_DIFFERENCE, DefectCategory.MISSING_PRINT, DefectCategory.EXTRA_PRINT)
        and edge_fraction >= cfg.edge_min_fraction
        and thickness <= cfg.edge_max_thickness_px
    ):
        base, certain, note = DefectCategory.EDGE_DIFFERENCE, edge_fraction >= likely, (
            f"A thin band ({thickness:.1f} px) along artwork edges ({edge_fraction:.0%} of the region)."
        )
    if text_overlap >= 0.5 and base is not DefectCategory.UNKNOWN:
        return DefectCategory.POSSIBLE_TEXT, False, (
            f"In a text-like area ({text_overlap:.0%}): {base.label.lower()}. Text content is not read, "
            "so whether characters changed is not verified."
        )
    return base, certain, note


def _severity(
    length_mm: float, area_mm2: float, text_overlap: float, max_difference: float, cfg: DefectsConfig
) -> tuple[DefectSeverity, tuple[str, ...]]:
    notes = []
    level = DefectSeverity.MINOR
    if length_mm >= cfg.severity_major_size_mm or area_mm2 >= cfg.severity_major_area_mm2:
        level = DefectSeverity.MAJOR
    if length_mm >= cfg.severity_critical_size_mm or area_mm2 >= cfg.severity_critical_area_mm2:
        level = DefectSeverity.CRITICAL
    notes.append(f"{length_mm:.2f} mm long, {area_mm2:.3f} mm²")
    if text_overlap >= 0.5 and length_mm >= cfg.text_major_size_mm and level is DefectSeverity.MINOR:
        level = DefectSeverity.MAJOR
        notes.append(f"in a text-like area (limit {cfg.text_major_size_mm:.2f} mm)")
    if max_difference < cfg.faint_difference and level is not DefectSeverity.MINOR:
        level = DefectSeverity.MINOR if level is DefectSeverity.MAJOR else DefectSeverity.MAJOR
        notes.append(f"faint ({max_difference:.0f} gray levels): one level lower")
    return level, tuple(notes)


def _reliability(score: float, max_difference: float, min_ssim: float, edge_changes: int, cfg: DefectsConfig):
    notes = []
    ok = True
    if score < cfg.reliable_min_score:
        ok = False
        notes.append(f"score {score:.1f} close to the detection threshold (reliable from {cfg.reliable_min_score:.1f})")
    if max_difference < cfg.reliable_min_difference:
        ok = False
        notes.append(f"difference only {max_difference:.0f} gray levels (reliable from {cfg.reliable_min_difference:.0f})")
    structural = min_ssim <= cfg.reliable_max_ssim or edge_changes > 0
    if not structural:
        ok = False
        notes.append("little structural evidence (SSIM high, no edge change)")
    if ok:
        notes.append(f"score {score:.1f}, difference {max_difference:.0f} gray levels, structural change confirmed")
    return (DetectionReliability.RELIABLE if ok else DetectionReliability.UNCERTAIN), tuple(notes)


# ---------------------------------------------------------------- main entry point


def assess_defects(
    comparison: ComparisonResult,
    reference: np.ndarray,
    registered: np.ndarray,
    px_per_mm: float,
    scale_estimated: bool,
    cfg: DefectsConfig,
) -> DefectReport:
    """Turn comparison candidates into measured, categorised defects with evidence."""
    if px_per_mm <= 0:
        raise ValueError("px_per_mm must be positive")
    h, w = comparison.candidate_mask.shape
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    text = text_like_mask(ref_gray, px_per_mm, cfg)
    band = _edge_band(ref_gray, cfg.edge_band_px)
    # Bare stock level: bright end of the expected print (a quarter-resolution sample suffices).
    paper_level = float(np.percentile(cv2.cvtColor(comparison.expected[::4, ::4], cv2.COLOR_BGR2GRAY), 95))

    # ---- grouping: dilate by half the merge distance, connected components
    mask = comparison.candidate_mask > 0
    radius = max(0, int(round(cfg.merge_distance_mm * px_per_mm / 2)))
    grouped = mask.astype(np.uint8)
    if radius:
        grouped = cv2.dilate(grouped, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    count, labels = cv2.connectedComponents(grouped, connectivity=8)

    members: dict[int, list] = {}
    for cand in comparison.candidates:
        sub = labels[cand.y : cand.y + cand.height, cand.x : cand.x + cand.width]
        hit = sub[mask[cand.y : cand.y + cand.height, cand.x : cand.x + cand.width]]
        if hit.size:
            members.setdefault(int(np.bincount(hit).argmax()), []).append(cand)
    labels, members = _pair_shape_changes(
        labels, members, mask, registered, comparison.expected, cfg.shape_pair_distance_mm * px_per_mm
    )

    defects: list[Defect] = []
    filtered: list[FilteredRegion] = []
    margin = cfg.evidence_margin_px
    for group, cands in members.items():
        region = (labels == group) & mask
        ys, xs = np.nonzero(region)
        if xs.size == 0:
            continue
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        r = region[y0:y1, x0:x1]

        printed = registered[y0:y1, x0:x1].astype(np.float32)
        expected = comparison.expected[y0:y1, x0:x1].astype(np.float32)
        gray_diff = printed @ _GRAY - expected @ _GRAY
        lab_p, lab_e = _lab(registered[y0:y1, x0:x1]), _lab(comparison.expected[y0:y1, x0:x1])
        d_lab = lab_p - lab_e
        de = np.sqrt(np.sum(d_lab * d_lab, axis=2))
        core = r & (de >= 0.5 * float(de[r].max()))
        cy_, cx_ = np.nonzero(core)
        area = int(core.sum())
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.column_stack([cx_, cy_]).astype(np.float32))
        length, thickness = max(rw, rh, 1.0), max(min(rw, rh), 1.0)
        area_mm2 = area / px_per_mm**2
        dl = float(d_lab[:, :, 0][core].mean())
        dc = float(np.sqrt(d_lab[:, :, 1] ** 2 + d_lab[:, :, 2] ** 2)[core].mean())
        signed = gray_diff[core]
        darker = float(np.mean(signed < -5))
        lighter = float(np.mean(signed > 5))
        polarity = "mixed" if darker >= 0.25 and lighter >= 0.25 else ("darker" if darker > lighter else "lighter")
        ink = float(np.mean((expected @ _GRAY)[core] < paper_level - 40.0))
        edge_fraction = float(band[y0:y1, x0:x1][r].mean())
        text_overlap = float((text[y0:y1, x0:x1] > 0)[r].mean())
        max_difference = float(np.abs(printed - expected).max(axis=2)[r].max())

        score = max(c.score for c in cands)
        edges = sum(c.missing_edge_px + c.extra_edge_px for c in cands)
        min_ssim = min(c.min_ssim for c in cands)
        reliability, reliability_notes = _reliability(score, max_difference, min_ssim, edges, cfg)
        if (
            area_mm2 < cfg.noise_max_area_mm2
            and score < cfg.noise_max_score
            and reliability is DetectionReliability.UNCERTAIN
        ):
            filtered.append(FilteredRegion(
                x0, y0, x1 - x0, y1 - y0, area_mm2, score,
                f"isolated noise: {area_mm2:.4f} mm² < {cfg.noise_max_area_mm2} mm² and score "
                f"{score:.2f} < {cfg.noise_max_score}",
            ))
            continue

        category, certain, note = _categorise(
            (darker, lighter), ink, edge_fraction, thickness, text_overlap, dl, dc, cfg
        )
        length_mm = length / px_per_mm
        severity, severity_notes = _severity(length_mm, area_mm2, text_overlap, max_difference, cfg)

        ex0, ey0 = max(0, x0 - margin), max(0, y0 - margin)
        ex1, ey1 = min(w, x1 + margin), min(h, y1 + margin)
        crop = (slice(ey0, ey1), slice(ex0, ex1))
        region_crop = np.zeros((ey1 - ey0, ex1 - ex0), np.uint8)
        region_crop[y0 - ey0 : y1 - ey0, x0 - ex0 : x1 - ex0][r] = 255
        evidence = DefectEvidence(
            origin=(ex0, ey0),
            artwork=reference[crop].copy(),
            expected=comparison.expected[crop].copy(),
            printed=registered[crop].copy(),
            difference=comparison.gray_difference[crop].copy(),
            score=np.clip(comparison.combined_score[crop] * 50.0, 0, 255).astype(np.uint8),
            region=region_crop,
        )
        moments = cv2.moments(r.astype(np.uint8), binaryImage=True)
        cx = moments["m10"] / moments["m00"] + x0 if moments["m00"] else (x0 + x1) / 2
        cy = moments["m01"] / moments["m00"] + y0 if moments["m00"] else (y0 + y1) / 2
        defects.append(Defect(
            id=0,
            category=category,
            category_certain=certain,
            category_note=note,
            reliability=reliability,
            reliability_notes=reliability_notes,
            severity=severity,
            severity_notes=severity_notes,
            x=x0, y=y0, width=x1 - x0, height=y1 - y0,
            centroid=(float(cx), float(cy)),
            area_px=area,
            length_px=float(length),
            thickness_px=float(thickness),
            area_mm2=area_mm2,
            length_mm=length_mm,
            thickness_mm=thickness / px_per_mm,
            score=score,
            peak_sigma=max(c.peak_sigma for c in cands),
            max_difference=max_difference,
            mean_difference=float(np.abs(gray_diff)[r].mean()),
            signed_difference=float(signed.mean()),
            delta_e=float(de[core].mean()),
            delta_l=dl,
            delta_c=dc,
            polarity=polarity,
            ink_fraction=ink,
            edge_fraction=edge_fraction,
            text_overlap=text_overlap,
            min_ssim=min_ssim,
            missing_edge_px=sum(c.missing_edge_px for c in cands),
            extra_edge_px=sum(c.extra_edge_px for c in cands),
            scale=min(c.scale for c in cands),
            candidate_ids=tuple(sorted(c.id for c in cands)),
            evidence=evidence,
        ))

    # Most severe first, then strongest; ids follow that order.
    defects.sort(key=lambda d: (-d.severity.rank, -d.score))
    numbered = tuple(replace(d, id=n) for n, d in enumerate(defects, start=1))
    text.setflags(write=False)
    return DefectReport(
        defects=numbered, filtered=tuple(filtered), px_per_mm=px_per_mm,
        scale_estimated=scale_estimated, text_mask=text,
    )


def grid_px_per_mm(
    grid_scale: float,
    artwork_dpi: tuple[float, float] | None,
    artwork_long_side_mm: float | None,
    artwork_long_side_px: int,
    assumed_long_side_mm: float,
) -> tuple[float, bool]:
    """(comparison-grid pixels per mm, estimated?) from the artwork DPI or an assumed label size."""
    if artwork_dpi is not None and artwork_long_side_mm is not None:
        return grid_scale * artwork_dpi[0] / 25.4, False
    return grid_scale * artwork_long_side_px / assumed_long_side_mm, True


def defect_rows(report: DefectReport) -> list[dict]:
    """Plain-dict summary of the defects (for logging and later reports)."""
    rows = []
    for d in report.defects:
        rows.append({
            "id": d.id,
            "category": d.category.value,
            "category_certain": d.category_certain,
            "reliability": d.reliability.value,
            "severity": d.severity.value,
            "bbox_px": (d.x, d.y, d.width, d.height),
            "length_mm": round(d.length_mm, 3),
            "area_mm2": round(d.area_mm2, 4),
            "max_difference": round(d.max_difference, 1),
            "delta_e": round(d.delta_e, 2),
            "score": round(d.score, 2),
        })
    return rows
