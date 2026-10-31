"""Saving inspection results: summary JSON, defect CSV and images."""
import csv
import json

import numpy as np
import pytest

from print_inspector import synthetic as syn
from print_inspector.config import load_config
from print_inspector.core.comparison import compare_to_artwork
from print_inspector.core.decision import InspectionDecision, OperatorMark, decide
from print_inspector.core.defects import assess_defects
from print_inspector.core.models import InspectionStatus
from print_inspector.core.results_export import (
    CSV_COLUMNS,
    DEFECTS_FILE,
    SUMMARY_FILE,
    ExportError,
    InspectionRecord,
    _jsonable,
    evidence_strip,
    results_folder_name,
    write_results,
)
from tests.test_comparison import ARTWORK, full_mask, simulate

PX_PER_MM = 12.0


@pytest.fixture(scope="module")
def config():
    return load_config(user_path=None)


@pytest.fixture(scope="module")
def analysed(config):
    printed, truth = syn.add_defects(ARTWORK, ["extra_mark", "missing_region"], seed=4)
    registered = simulate(printed)
    comparison = compare_to_artwork(registered, ARTWORK, full_mask(ARTWORK.shape), config.comparison, 0.9)
    report = assess_defects(comparison, ARTWORK, registered, PX_PER_MM, False, config.defects)
    return comparison, report


def _record(config, comparison=None, report=None, marks=None, decision=None):
    marks = marks or {}
    if decision is None:
        decision = decide(None, None, None, comparison, report, marks, config.decision)
    return InspectionRecord(
        original=None, printed=None, quality=None, alignment=None, registration=None,
        comparison=comparison, defects=report, marks=marks, decision=decision,
        sensitivity="medium", config=config,
    )


def test_writes_summary_csv_and_images(tmp_path, config, analysed):
    comparison, report = analysed
    assert len(report.defects) >= 2
    first = report.defects[0]
    record = _record(config, comparison, report, {first.id: OperatorMark.DISMISSED})
    images = {"registered_print.png": np.full((40, 60, 3), 128, np.uint8)}
    folder = write_results(tmp_path, record, images)

    assert folder.parent == tmp_path and folder.name.endswith(record.decision.status.name)
    summary = json.loads((folder / SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["verdict"]["status"] == record.decision.status.name
    assert summary["verdict"]["reasons"] == list(record.decision.reasons)
    items = summary["defects"]["items"]
    assert [i["id"] for i in items] == [d.id for d in report.defects]
    assert items[0]["operator"] == "dismissed" and items[1]["operator"] == "unreviewed"
    assert "evidence" not in items[0]
    assert items[0]["position_mm"] == pytest.approx([first.centroid[0] / PX_PER_MM, first.centroid[1] / PX_PER_MM])
    assert summary["checks"]["comparison"]["level"] == comparison.level.value
    assert summary["settings"]["decision"]["fail_min_severity"] == config.decision.fail_min_severity

    with open(folder / DEFECTS_FILE, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(CSV_COLUMNS)
    assert len(rows) == len(report.defects)
    assert rows[0]["operator"] == "dismissed" and rows[0]["severity"] == first.severity.value
    assert float(rows[0]["x_mm"]) == pytest.approx(first.centroid[0] / PX_PER_MM, abs=0.01)

    assert (folder / "registered_print.png").is_file()
    for d in report.defects:
        assert (folder / f"defect_{d.id:02d}.png").is_file()


def test_second_save_gets_its_own_folder(tmp_path, config):
    record = _record(config, decision=InspectionDecision(InspectionStatus.REVIEW, ("Photo too poor.",)))
    first = write_results(tmp_path, record, {})
    second = write_results(tmp_path, record, {})
    assert first != second and second.name.endswith("_2")
    summary = json.loads((first / SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["defects"] is None and summary["checks"]["alignment"] is None
    with open(first / DEFECTS_FILE, encoding="utf-8-sig") as handle:
        assert handle.read().strip() == ",".join(CSV_COLUMNS)


def test_missing_parent_folder_is_an_error(tmp_path, config):
    record = _record(config, decision=InspectionDecision(InspectionStatus.REVIEW, ()))
    with pytest.raises(ExportError, match="does not exist"):
        write_results(tmp_path / "nope", record, {})


def test_folder_name_is_safe(config):
    from dataclasses import replace
    from pathlib import Path
    from types import SimpleNamespace

    record = _record(config, decision=InspectionDecision(InspectionStatus.FAIL, ()))
    record = replace(record, printed=SimpleNamespace(path=Path("C:/photos/Label: lot*12?.jpg")))
    name = results_folder_name(record)
    assert name.startswith("Label_lot_12_") and name.endswith("_FAIL")
    assert not set(name) & set('<>:"/\\|?*')


def test_jsonable_handles_numpy_enums_and_non_finite():
    data = _jsonable({
        "a": np.float32(1.5), "b": np.array([[1, 2]]), "c": InspectionStatus.PASS,
        "d": float("inf"), "e": (np.int64(3),), "f": np.float64("nan"),
    })
    assert data == {"a": 1.5, "b": [[1, 2]], "c": "pass", "d": None, "e": [3], "f": None}
    json.dumps(data, allow_nan=False)


def test_evidence_strip_is_three_enlarged_crops(analysed):
    _, report = analysed
    d = report.defects[0]
    strip = evidence_strip(d, scale=3)
    h, w = d.evidence.printed.shape[:2]
    assert strip.shape == (h * 3, w * 3 * 3 + 16, 3)
