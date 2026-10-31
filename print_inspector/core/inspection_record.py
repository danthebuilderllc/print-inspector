"""One inspection as a whole, and its findings of every kind in one list (for the reports)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from print_inspector.config import AppConfig
from print_inspector.core.alignment import AlignmentResult
from print_inspector.core.barcode_inspection import BarcodeReport
from print_inspector.core.barcode_zones import BarcodeZone
from print_inspector.core.capture_quality import QualityReport
from print_inspector.core.color_inspection import ColorReport
from print_inspector.core.comparison import ComparisonResult
from print_inspector.core.decision import InspectionDecision, OperatorMark
from print_inspector.core.defects import DefectReport
from print_inspector.core.image_io import LoadedImage
from print_inspector.core.regions import InspectionRegion, critical_region_at
from print_inspector.core.registration import RegistrationResult
from print_inspector.core.text_inspection import TextReport
from print_inspector.core.text_zones import TextZone

STRUCTURAL, TEXT, BARCODE, COLOUR = "structural", "text", "barcode", "colour"
REJECTS, NEEDS_REVIEW, DISMISSED, WITHIN_LIMITS = "rejects", "needs review", "dismissed", "within limits"


@dataclass(frozen=True)
class InspectionRecord:
    """Everything known about one inspection (any stage may be missing)."""

    original: LoadedImage | None
    printed: LoadedImage | None
    quality: QualityReport | None
    alignment: AlignmentResult | None
    registration: RegistrationResult | None
    comparison: ComparisonResult | None
    defects: DefectReport | None
    marks: dict[int, OperatorMark]
    decision: InspectionDecision
    sensitivity: str
    config: AppConfig
    errors: tuple[str, ...] = ()
    text: TextReport | None = None
    text_marks: dict[int, OperatorMark] = field(default_factory=dict)
    barcodes: BarcodeReport | None = None
    barcode_marks: dict[int, OperatorMark] = field(default_factory=dict)
    color: ColorReport | None = None
    color_marks: dict[int, OperatorMark] = field(default_factory=dict)
    regions: tuple[InspectionRegion, ...] = ()
    text_zones: tuple[TextZone, ...] = ()
    barcode_zones: tuple[BarcodeZone, ...] = ()
    profile: str = ""  # name of the inspection profile in use ("" = none)
    profile_file: str = ""
    profile_modified: bool = False  # settings changed since the profile was loaded or saved
    created: datetime = field(default_factory=lambda: datetime.now().astimezone())


def grid_scale(record: InspectionRecord) -> tuple[float, float] | None:
    """Comparison grid pixels per artwork pixel (x, y)."""
    if record.registration is None or record.original is None or record.registration.reference is None:
        return None
    gh, gw = record.registration.reference.shape[:2]
    ah, aw = record.original.working.shape[:2]
    return gw / aw, gh / ah


def grid_shape(record: InspectionRecord) -> tuple[int, int] | None:
    if record.registration is None or record.registration.reference is None:
        return None
    return record.registration.reference.shape[:2]


@dataclass(frozen=True)
class FindingRow:
    """A finding of any inspection, in the terms every report uses."""

    label: str  # "#3", "T1", "B2", "C1"
    inspection: str  # STRUCTURAL, TEXT, BARCODE or COLOUR
    id: int
    problem: str
    description: str
    severity: str
    reliable: bool
    operator: OperatorMark
    verdict: str  # REJECTS, NEEDS_REVIEW, DISMISSED or WITHIN_LIMITS
    box: tuple[int, int, int, int]  # x, y, width, height on the comparison grid
    centre: tuple[float, float]  # on the comparison grid
    area: str = ""  # critical region or zone it belongs to


def _verdict(mark: OperatorMark, finding_id: int, failing: tuple[int, ...], review: tuple[int, ...]) -> str:
    if mark is OperatorMark.DISMISSED:
        return DISMISSED
    if finding_id in failing:
        return REJECTS
    if finding_id in review:
        return NEEDS_REVIEW
    return WITHIN_LIMITS


def collect_findings(record: InspectionRecord) -> list[FindingRow]:
    """All findings: structural defects, then text, barcode and colour findings."""
    d = record.decision
    rows: list[FindingRow] = []
    if record.defects is not None:
        for f in record.defects.defects:
            mark = record.marks.get(f.id, OperatorMark.UNREVIEWED)
            rows.append(FindingRow(
                f"#{f.id}", STRUCTURAL, f.id, f.category_label,
                f"{f.category_label}, {f.length_mm:.2f} × {f.thickness_mm:.2f} mm, {f.area_mm2:.3f} mm²",
                f.severity.value, f.is_reliable, mark, _verdict(mark, f.id, d.failing, d.review),
                (f.x, f.y, f.width, f.height), f.centroid, f.critical_region,
            ))
    if record.text is not None:
        for f in record.text.findings:
            mark = record.text_marks.get(f.id, OperatorMark.UNREVIEWED)
            rows.append(FindingRow(
                f.label, TEXT, f.id, f.kind.label, f.describe(), f.severity.value, f.reliable, mark,
                _verdict(mark, f.id, d.text_failing, d.text_review), (f.x, f.y, f.width, f.height),
                (f.x + f.width / 2, f.y + f.height / 2), f.zone or "",
            ))
    if record.barcodes is not None:
        for f in record.barcodes.findings:
            mark = record.barcode_marks.get(f.id, OperatorMark.UNREVIEWED)
            rows.append(FindingRow(
                f.label, BARCODE, f.id, f.kind.label, f.describe(), f.severity.value, f.reliable, mark,
                _verdict(mark, f.id, d.barcode_failing, d.barcode_review), (f.x, f.y, f.width, f.height),
                (f.x + f.width / 2, f.y + f.height / 2), f.zone or "",
            ))
    if record.color is not None:
        shape = grid_shape(record)
        for f in record.color.findings:
            mark = record.color_marks.get(f.id, OperatorMark.UNREVIEWED)
            region = None
            if f.critical and shape is not None:
                region = critical_region_at(record.regions, (f.x, f.y, f.width, f.height), shape[1], shape[0])
            rows.append(FindingRow(
                f.label, COLOUR, f.id, f.kind.label, f.describe(), f.severity.value, f.reliable, mark,
                _verdict(mark, f.id, d.color_failing, d.color_review), (f.x, f.y, f.width, f.height),
                (f.x + f.width / 2, f.y + f.height / 2), region.name if region else "",
            ))
    return rows
