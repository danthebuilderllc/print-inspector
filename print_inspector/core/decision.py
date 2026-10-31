"""Inspection decisions: PASS, FAIL or REVIEW.

Trust gate (``preliminary_status``): an inspection whose photo, label
alignment, registration or comparison cannot be trusted is REVIEW.

Precedence between alignment (Step 3) and registration (Step 4): a verified
registration measures the actual alignment directly, so Step 3's indirect
accuracy estimates (edge fit, proportions, residual, similarity) no longer
block. Findings that registration cannot repair still do: part of the label
outside the photo, invalid or missing corners, or an outline that does not
match the label (the comparison mask is derived from that outline).

Verdict (``decide``), given the defects and the operator's marks:

* Operator marks override the automatic assessment of a defect: DISMISSED
  defects are ignored, CONFIRMED defects count as certain, NEEDS_REVIEW ones
  require a person.
* A defect *meets the rejection criteria* when its severity is at least
  ``fail_min_severity``; too many minor defects (``max_minor_defects``) also do.

1. FAIL - a confirmed defect meets the rejection criteria (the operator has
   seen it, so this holds even when the photo checks were not perfect).
2. REVIEW - the inspection inputs are not reliable.
3. FAIL - a reliably detected, unreviewed defect meets the rejection criteria.
4. REVIEW - a potentially rejecting finding is uncertain (any uncertain
   finding in a critical text line, barcode zone or critical region), or the
   operator asked for review.
5. PASS - nothing above applies (with capture/alignment warnings only if
   ``allow_pass_with_warnings``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from print_inspector.config import DecisionConfig
from print_inspector.core.barcode_inspection import BarcodeReport
from print_inspector.core.color_inspection import ColorReport
from print_inspector.core.alignment import AlignmentResult
from print_inspector.core.capture_quality import QualityReport
from print_inspector.core.comparison import ComparisonResult
from print_inspector.core.defects import DefectReport, DefectSeverity
from print_inspector.core.models import InspectionStatus, QualityLevel, Severity
from print_inspector.core.registration import RegistrationResult
from print_inspector.core.text_inspection import TextReport

# Step 3 findings that a verified registration does not resolve.
ALIGNMENT_BLOCKERS = frozenset({"label_not_found", "invalid_corners", "label_outside_photo", "outline_mismatch"})


class OperatorMark(str, Enum):
    UNREVIEWED = "unreviewed"
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"
    NEEDS_REVIEW = "needs_review"

    @property
    def label(self) -> str:
        return {
            OperatorMark.UNREVIEWED: "Not reviewed",
            OperatorMark.CONFIRMED: "Confirmed defect",
            OperatorMark.DISMISSED: "Dismissed",
            OperatorMark.NEEDS_REVIEW: "Needs review",
        }[self]


@dataclass(frozen=True)
class InspectionDecision:
    status: InspectionStatus
    reasons: tuple[str, ...]
    failing: tuple[int, ...] = ()  # defect ids that reject the label
    review: tuple[int, ...] = ()  # defect ids that need a person
    minor_count: int = 0
    text_failing: tuple[int, ...] = ()  # text finding ids that reject the label
    text_review: tuple[int, ...] = ()  # text finding ids that need a person
    barcode_failing: tuple[int, ...] = ()  # barcode finding ids that reject the label
    barcode_review: tuple[int, ...] = ()  # barcode finding ids that need a person
    color_failing: tuple[int, ...] = ()  # colour finding ids that reject the label
    color_review: tuple[int, ...] = ()  # colour finding ids that need a person


def alignment_blocks(alignment: AlignmentResult, registration: RegistrationResult | None) -> bool:
    if registration is None or not registration.is_reliable:
        return not alignment.is_reliable
    return any(i.severity is Severity.CRITICAL and i.code in ALIGNMENT_BLOCKERS for i in alignment.issues)


def preliminary_status(
    quality: QualityReport | None,
    alignment: AlignmentResult | None,
    registration: RegistrationResult | None = None,
    comparison: ComparisonResult | None = None,
) -> tuple[InspectionStatus, list[str]]:
    """Trust gate from the checks available so far (None = not run yet). Never PASS."""
    reasons = []
    if quality is not None and not quality.is_reliable:
        reasons.append("The photo quality is too poor to inspect reliably.")
    if alignment is not None and alignment_blocks(alignment, registration):
        reasons.append("The label could not be aligned reliably.")
    if registration is not None and not registration.is_reliable:
        reasons.append("The photo could not be registered precisely enough to the artwork.")
    if comparison is not None and not comparison.is_reliable:
        reasons.append("The comparison found differences too extensive to assess individually.")
    if reasons:
        return InspectionStatus.REVIEW, reasons
    return InspectionStatus.NOT_INSPECTED, []


def _has_warnings(*checks) -> bool:
    return any(c is not None and getattr(c, "level", None) is QualityLevel.MARGINAL for c in checks)


@dataclass
class _Groups:
    """Findings (defects "#n", text findings "Tn") sorted by what they mean for the verdict."""

    confirmed_fail: list[tuple[str, int]] = field(default_factory=list)
    auto_fail: list[tuple[str, int]] = field(default_factory=list)
    uncertain: list[tuple[str, int]] = field(default_factory=list)
    review: list[tuple[str, int]] = field(default_factory=list)
    minor_reliable: list[tuple[str, int]] = field(default_factory=list)
    minor_total: list[tuple[str, int]] = field(default_factory=list)
    uncertain_in_region: list[tuple[str, int]] = field(default_factory=list)  # minor, uncertain, critical area

    def add(self, key: tuple[str, int], rejecting: bool, reliable: bool, mark: OperatorMark,
            uncertain_needs_person: bool = False) -> None:
        if mark is OperatorMark.DISMISSED:
            return
        if mark is OperatorMark.CONFIRMED:
            if rejecting:
                self.confirmed_fail.append(key)
            else:
                self.minor_reliable.append(key)
                self.minor_total.append(key)
        elif mark is OperatorMark.NEEDS_REVIEW:
            self.review.append(key)
        elif rejecting:
            (self.auto_fail if reliable else self.uncertain).append(key)
        else:
            self.minor_total.append(key)
            if reliable:
                self.minor_reliable.append(key)
            elif uncertain_needs_person:
                self.uncertain.append(key)
                self.uncertain_in_region.append(key)


_PREFIX = {"defect": "#", "text": "T", "barcode": "B", "color": "C"}


def _label(key: tuple[str, int]) -> str:
    return f"{_PREFIX[key[0]]}{key[1]}"


def _ids(keys: list[tuple[str, int]], kind: str) -> tuple[int, ...]:
    return tuple(k[1] for k in keys if k[0] == kind)


def decide(
    quality: QualityReport | None,
    alignment: AlignmentResult | None,
    registration: RegistrationResult | None,
    comparison: ComparisonResult | None,
    report: DefectReport | None,
    marks: dict[int, OperatorMark],
    cfg: DecisionConfig,
    text_report: TextReport | None = None,
    text_marks: dict[int, OperatorMark] | None = None,
    barcode_report: BarcodeReport | None = None,
    barcode_marks: dict[int, OperatorMark] | None = None,
    color_report: ColorReport | None = None,
    color_marks: dict[int, OperatorMark] | None = None,
) -> InspectionDecision:
    """PASS / FAIL / REVIEW for a finished comparison; the trust gate before that.

    Text findings (``text_report``), barcode findings (``barcode_report``) and
    colour findings (``color_report``) count like defects. An uncertain text finding in a critical line, or an
    uncertain barcode finding in a critical zone, needs a person even when it is
    minor. A text or barcode inspection that could not run although it is
    required makes the inputs unreliable.
    """
    gate, gate_reasons = preliminary_status(quality, alignment, registration, comparison)
    if comparison is None or report is None:
        return InspectionDecision(gate, tuple(gate_reasons))
    for extra_report in (text_report, barcode_report, color_report):
        if extra_report is not None and not extra_report.is_reliable:
            gate = InspectionStatus.REVIEW
            gate_reasons = gate_reasons + [
                i.message for i in extra_report.issues if i.severity is Severity.CRITICAL
            ]

    fail_rank = DefectSeverity(cfg.fail_min_severity).rank
    groups = _Groups()
    for d in report.defects:
        groups.add(("defect", d.id), d.severity.rank >= fail_rank, d.is_reliable,
                   marks.get(d.id, OperatorMark.UNREVIEWED), uncertain_needs_person=bool(d.critical_region))
    text_marks = text_marks or {}
    for f in text_report.findings if text_report is not None else ():
        groups.add(("text", f.id), f.severity.rank >= fail_rank, f.reliable,
                   text_marks.get(f.id, OperatorMark.UNREVIEWED), uncertain_needs_person=f.critical)
    barcode_marks = barcode_marks or {}
    for f in barcode_report.findings if barcode_report is not None else ():
        groups.add(("barcode", f.id), f.severity.rank >= fail_rank, f.reliable,
                   barcode_marks.get(f.id, OperatorMark.UNREVIEWED), uncertain_needs_person=f.critical)
    color_marks = color_marks or {}
    for f in color_report.findings if color_report is not None else ():
        groups.add(("color", f.id), f.severity.rank >= fail_rank, f.reliable,
                   color_marks.get(f.id, OperatorMark.UNREVIEWED), uncertain_needs_person=f.critical)

    def ids(values: list[tuple[str, int]]) -> str:
        return ", ".join(_label(v) for v in values)

    def result(status, reasons, failing=(), review=()) -> InspectionDecision:
        failing, review = list(failing), list(review)
        return InspectionDecision(
            status, tuple(reasons), _ids(failing, "defect"), _ids(review, "defect"), len(groups.minor_total),
            _ids(failing, "text"), _ids(review, "text"), _ids(failing, "barcode"), _ids(review, "barcode"),
            _ids(failing, "color"), _ids(review, "color"),
        )

    g = groups
    severity = cfg.fail_min_severity
    if g.confirmed_fail:
        return result(
            InspectionStatus.FAIL, (f"Confirmed defect(s) {ids(g.confirmed_fail)} of {severity} severity or worse.",),
            g.confirmed_fail, g.review + g.uncertain,
        )
    if gate is InspectionStatus.REVIEW:
        extra = []
        if g.auto_fail or g.uncertain:
            extra.append(f"Possible rejecting defect(s) {ids(g.auto_fail + g.uncertain)} - confirm or dismiss them.")
        return result(InspectionStatus.REVIEW, gate_reasons + extra, (), g.auto_fail + g.uncertain + g.review)
    if g.auto_fail:
        return result(
            InspectionStatus.FAIL, (f"Defect(s) {ids(g.auto_fail)} of {severity} severity or worse, reliably detected.",),
            g.auto_fail, g.review + g.uncertain,
        )
    if len(g.minor_reliable) > cfg.max_minor_defects:
        return result(
            InspectionStatus.FAIL,
            (f"{len(g.minor_reliable)} minor defects (at most {cfg.max_minor_defects} allowed).",),
            g.minor_reliable, g.review + g.uncertain,
        )
    reasons = []
    uncertain_text = [k for k in g.uncertain if k[0] == "text"]
    uncertain_barcodes = [k for k in g.uncertain if k[0] == "barcode"]
    uncertain_colors = [k for k in g.uncertain if k[0] == "color"]
    uncertain_defects = [k for k in g.uncertain if k[0] == "defect"]
    in_regions = [k for k in uncertain_defects if k in g.uncertain_in_region]
    uncertain_defects = [k for k in uncertain_defects if k not in g.uncertain_in_region]
    if uncertain_defects:
        reasons.append(f"Uncertain finding(s) {ids(uncertain_defects)} of {severity} severity or worse need a person.")
    if in_regions:
        reasons.append(f"Uncertain finding(s) {ids(in_regions)} in a critical region need a person.")
    if uncertain_text:
        reasons.append(
            f"Uncertain text finding(s) {ids(uncertain_text)}: OCR and the image comparison do not agree "
            "clearly - a person must check them."
        )
    if uncertain_barcodes:
        reasons.append(
            f"Uncertain barcode finding(s) {ids(uncertain_barcodes)}: not confirmed by visible damage (the photo may "
            "be the cause) - a person must check them, e.g. with a scanner."
        )
    if uncertain_colors:
        reasons.append(
            f"Colour finding(s) {ids(uncertain_colors)} from an uncalibrated photo or uncertain: approximate only - "
            "a person must judge the colour (or calibrate the camera to measure it)."
        )
    if g.review:
        reasons.append(f"Defect(s) {ids(g.review)} marked for review.")
    if len(g.minor_total) > cfg.max_minor_defects:
        reasons.append(f"{len(g.minor_total)} minor findings (at most {cfg.max_minor_defects} allowed), some uncertain.")
    if reasons:
        return result(InspectionStatus.REVIEW, reasons, (), g.uncertain + g.review)
    if not cfg.allow_pass_with_warnings and _has_warnings(quality, alignment, registration, comparison, text_report):
        return result(
            InspectionStatus.REVIEW,
            ("No rejecting defects, but the photo checks gave warnings and PASS requires clean checks.",),
        )
    note = f" ({len(g.minor_total)} minor finding(s) within the allowed {cfg.max_minor_defects})" if g.minor_total else ""
    return result(InspectionStatus.PASS, (f"No defect meets the rejection criteria{note}.",))
