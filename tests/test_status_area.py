"""Bottom status area widgets."""
from print_inspector.core.models import InspectionStatus, QualityLevel
from print_inspector.gui.status_area import ElidedLabel, StatusArea


def test_badges_progress_and_verdict(qtbot):
    area = StatusArea()
    qtbot.addWidget(area)
    assert area.quality.word() == "-" and area.alignment.word() == "-"
    assert area.progress_label.full_text() == "Waiting for images."
    area.quality.set_level(QualityLevel.MARGINAL, None, "Photo has 4 MP.")
    assert area.quality.word() == "MARGINAL"
    assert area.quality.detail.toolTip() == "Photo has 4 MP."
    area.alignment.set_level(QualityLevel.POOR, "FAILED", "Not aligned")
    assert area.alignment.word() == "FAILED"

    area.set_progress(1, 4, "Running", busy=True)
    assert (area.progress.minimum(), area.progress.maximum(), area.progress.value()) == (0, 4, 1)
    area.set_progress(0, 0, "Loading", busy=True)
    assert area.progress.maximum() == 0  # indeterminate
    area.set_progress(4, 4, "Done", busy=False)
    assert area.progress.value() == 4

    area.verdict.set_status(InspectionStatus.FAIL)
    assert area.verdict.text() == "FAIL"


def test_elided_label_keeps_full_text(qtbot):
    label = ElidedLabel()
    qtbot.addWidget(label)
    label.resize(60, 20)
    text = "A long reason that cannot fit into sixty pixels of width at all"
    label.set_full_text(text)
    assert label.full_text() == text and label.toolTip() == text
    assert label.text() != text and label.text().endswith("…")
