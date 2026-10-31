"""Critical and ignored regions: storage, and their effect on the comparison, defects, verdict and barcodes."""
import dataclasses
import json

import numpy as np
import pytest

from print_inspector import synthetic as syn
from print_inspector.config import load_config
from print_inspector.core.barcode_inspection import inspect_barcodes
from print_inspector.core.barcode_zones import BarcodeZone
from print_inspector.core.comparison import compare_to_artwork
from print_inspector.core.decision import OperatorMark, decide
from print_inspector.core.defects import DefectSeverity, DetectionReliability, assess_defects
from print_inspector.core.models import InspectionStatus
from print_inspector.core.regions import (
    InspectionRegion,
    RegionError,
    RegionKind,
    apply_to_color,
    apply_to_defects,
    boxes,
    critical_region_at,
    inspection_mask,
    load_regions,
    regions_path,
    save_regions,
)
from tests.barcode_fakes import make_barcode_label
from tests.test_comparison import ARTWORK, full_mask, simulate

PX_PER_MM = 12.0


@pytest.fixture(scope="module")
def config():
    return load_config(user_path=None)


def around(box, shape, kind, name="R", pad=30):
    """A region covering ``box`` (pixels) with ``pad`` pixels to spare, as fractions of ``shape``."""
    h, w = shape[:2]
    x, y, bw, bh = box
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    return InspectionRegion(name, kind, x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h)


def analyse(config, printed, regions=()):
    registered = simulate(printed)
    mask = inspection_mask(full_mask(ARTWORK.shape), regions)
    comparison = compare_to_artwork(registered, ARTWORK, mask, config.comparison, 0.9)
    report = assess_defects(comparison, ARTWORK, registered, PX_PER_MM, False, config.defects)
    report = apply_to_defects(report, regions, ARTWORK.shape, config.decision.critical_region_severity)
    return comparison, report


def touches(defect, box, pad=6):
    x, y, w, h = box
    return not (defect.x > x + w + pad or defect.x + defect.width + pad < x or
                defect.y > y + h + pad or defect.y + defect.height + pad < y)


# ---------------------------------------------------------------- storage


def test_regions_round_trip_next_to_the_artwork(tmp_path):
    artwork = tmp_path / "label.png"
    path = regions_path(artwork)
    assert path.name == "label.png.regions.json"
    assert load_regions(path) == ()
    regions = (
        InspectionRegion("Logo", RegionKind.CRITICAL, 0.1, 0.1, 0.3, 0.2),
        InspectionRegion("Lot area", RegionKind.IGNORE, 0.6, 0.7, 0.3, 0.2),
    )
    save_regions(path, regions)
    assert json.loads(path.read_text(encoding="utf-8"))["regions"][1]["kind"] == "ignore"
    assert load_regions(path) == regions
    save_regions(path, ())
    assert not path.exists()


@pytest.mark.parametrize("content, message", [
    ("{", "cannot be read"),
    ('{"regions": [{"name": "A", "kind": "maybe", "x": 0, "y": 0, "width": 0.1, "height": 0.1}]}', "unknown type"),
    ('{"regions": [{"name": "A", "kind": "ignore", "x": 0.95, "y": 0, "width": 0.1, "height": 0.1}]}', "beyond"),
    ('{"regions": [{"name": " ", "kind": "ignore", "x": 0, "y": 0, "width": 0.1, "height": 0.1}]}', "no name"),
    ('{"regions": 5}', "no list"),
])
def test_invalid_region_files_are_rejected(tmp_path, content, message):
    path = tmp_path / "a.png.regions.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(RegionError, match=message):
        load_regions(path)


def test_duplicate_region_names_are_rejected(tmp_path):
    region = InspectionRegion("A", RegionKind.IGNORE, 0, 0, 0.1, 0.1)
    with pytest.raises(RegionError, match="twice"):
        save_regions(tmp_path / "r.json", (region, dataclasses.replace(region, x=0.5)))


def test_inspection_mask_removes_only_ignored_regions():
    mask = np.full((100, 200), 255, np.uint8)
    regions = (InspectionRegion("I", RegionKind.IGNORE, 0.5, 0.5, 0.25, 0.5),
               InspectionRegion("C", RegionKind.CRITICAL, 0.0, 0.0, 0.25, 0.25))
    out = inspection_mask(mask, regions)
    assert out is not mask and mask.min() == 255  # the registration mask stays untouched
    assert out[50:100, 100:150].max() == 0 and out[:50].min() == 255 and out[0:25, 0:50].min() == 255
    assert inspection_mask(mask, regions[1:]) is mask
    assert boxes(regions, RegionKind.IGNORE, 200, 100) == ((100, 50, 50, 50),)
    assert critical_region_at(regions, (40, 20, 10, 10), 200, 100).name == "C"
    assert critical_region_at(regions, (60, 30, 10, 10), 200, 100) is None


# ---------------------------------------------------------------- in the inspection


def test_ignored_region_hides_a_real_defect(config):
    printed, truth = syn.add_defects(ARTWORK, ["extra_mark"], seed=4)
    box = truth[0].bbox
    _, report = analyse(config, printed)
    assert any(touches(d, box) for d in report.defects)
    _, ignored = analyse(config, printed, (around(box, ARTWORK.shape, RegionKind.IGNORE),))
    assert not any(touches(d, box) for d in ignored.defects)


def test_critical_region_raises_severity_and_fails(config):
    printed, truth = syn.add_defects(ARTWORK, ["small_dot"], seed=2)
    box = truth[0].bbox
    comparison, report = analyse(config, printed)
    defect = next(d for d in report.defects if touches(d, box))
    assert defect.severity is DefectSeverity.MINOR and defect.critical_region == ""
    plain = decide(None, None, None, comparison, report, {}, config.decision)
    assert plain.status is InspectionStatus.PASS  # one reliable minor defect is allowed

    region = around(box, ARTWORK.shape, RegionKind.CRITICAL, "Dose")
    comparison, report = analyse(config, printed, (region,))
    defect = next(d for d in report.defects if touches(d, box))
    assert defect.severity is DefectSeverity.MAJOR and defect.critical_region == "Dose"
    assert "critical region 'Dose'" in defect.severity_notes[-1] and "minor to major" in defect.severity_notes[-1]
    assert defect.is_reliable
    decision = decide(None, None, None, comparison, report, {}, config.decision)
    assert decision.status is InspectionStatus.FAIL and defect.id in decision.failing


def test_uncertain_minor_defect_in_a_critical_region_needs_review(config):
    printed, truth = syn.add_defects(ARTWORK, ["small_dot"], seed=2)
    region = around(truth[0].bbox, ARTWORK.shape, RegionKind.CRITICAL, "Dose")
    comparison, report = analyse(config, printed, (region,))
    defect = next(d for d in report.defects if d.critical_region)
    # Make it uncertain and minor (as a faint defect would be); outside a region it would pass.
    weak = dataclasses.replace(defect, reliability=DetectionReliability.UNCERTAIN, severity=DefectSeverity.MINOR)
    report = dataclasses.replace(report, defects=(weak,))
    decision = decide(None, None, None, comparison, report, {}, config.decision)
    assert decision.status is InspectionStatus.REVIEW and decision.review == (weak.id,)
    assert "critical region" in decision.reasons[0]
    outside = dataclasses.replace(report, defects=(dataclasses.replace(weak, critical_region=""),))
    assert decide(None, None, None, comparison, outside, {}, config.decision).status is InspectionStatus.PASS
    dismissed = decide(None, None, None, comparison, report, {weak.id: OperatorMark.DISMISSED}, config.decision)
    assert dismissed.status is InspectionStatus.PASS


def test_critical_region_severity_is_configurable(config):
    printed, truth = syn.add_defects(ARTWORK, ["small_dot"], seed=2)
    region = around(truth[0].bbox, ARTWORK.shape, RegionKind.CRITICAL)
    _, report = analyse(config, printed)
    raised = apply_to_defects(report, (region,), ARTWORK.shape, "critical")
    defect = next(d for d in raised.defects if d.critical_region)
    assert defect.severity is DefectSeverity.CRITICAL
    assert raised.defects[0] is defect  # sorted by severity again


def test_colour_findings_in_critical_regions():
    from print_inspector.core.color_inspection import ColorFinding, ColorFindingKind, ColorReport

    finding = ColorFinding(1, ColorFindingKind.LOCAL_CHANGE, 10, 10, 20, 20, 7.0, 9.0, 1, 2, 3, 3.0,
                           DefectSeverity.MINOR, False, ("note",))
    fields = {f.name: None for f in dataclasses.fields(ColorReport)}
    report = ColorReport(**{**fields, "findings": (finding,)})
    region = InspectionRegion("Brand red", RegionKind.CRITICAL, 0, 0, 0.5, 0.5)
    out = apply_to_color(report, (region,), (100, 100), "major")
    f = out.findings[0]
    assert f.critical and f.severity is DefectSeverity.MAJOR and "Brand red" in f.notes[-1]
    assert apply_to_color(report, (), (100, 100), "major") is report


def test_barcodes_in_ignored_regions_are_skipped_unless_zoned(config):
    label = make_barcode_label()
    registered = simulate(label.image)
    mask = full_mask(label.image.shape)
    comparison = compare_to_artwork(registered, label.image, mask, config.comparison, 0.9)
    h, w = label.image.shape[:2]
    full = inspect_barcodes(label.image, registered, comparison.combined_score, mask, (), config.barcodes)
    assert len(full.symbols) == 3
    qr = around(label.boxes["qr"], label.image.shape, RegionKind.IGNORE, "QR", pad=10)
    ignore = boxes((qr,), RegionKind.IGNORE, w, h)
    skipped = inspect_barcodes(label.image, registered, comparison.combined_score, mask, (), config.barcodes, ignore)
    assert len(skipped.symbols) == 2 and all(s.format != "qr" for s in skipped.symbols)
    x, y, bw, bh = label.boxes["qr"]
    zone = BarcodeZone("QR", (x - 20) / w, (y - 20) / h, (bw + 40) / w, (bh + 40) / h)
    zoned = inspect_barcodes(label.image, registered, comparison.combined_score, mask, (zone,), config.barcodes,
                             ignore)
    assert len(zoned.symbols) == 3
