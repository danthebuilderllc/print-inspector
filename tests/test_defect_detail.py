
import pytest

from print_inspector import synthetic as syn
from print_inspector.core.comparison import compare_to_artwork
from print_inspector.core.decision import OperatorMark
from print_inspector.core.defects import assess_defects
from print_inspector.gui.defect_detail import DefectDetailWidget
from tests.test_comparison import ARTWORK, full_mask, simulate


@pytest.fixture(scope="module")
def defect():
    from print_inspector.config import load_config

    config = load_config(user_path=None)
    printed, _ = syn.add_defects(ARTWORK, ["broken_character"], seed=4)
    registered = simulate(printed)
    comparison = compare_to_artwork(registered, ARTWORK, full_mask(ARTWORK.shape), config.comparison, 0.9)
    return assess_defects(comparison, ARTWORK, registered, 12.0, True, config.defects).defects[0]


@pytest.fixture
def widget(qtbot):
    w = DefectDetailWidget()
    qtbot.addWidget(w)
    w.show()
    return w


def test_empty_state(widget):
    assert widget.defect() is None
    assert all(not b.isEnabled() for b in widget.buttons.values())
    assert widget.images["artwork"].pixmap().isNull()


def test_shows_original_and_printed_side_by_side(widget, defect):
    widget.show_defect(defect, OperatorMark.UNREVIEWED, scale_estimated=True)
    for key in ("artwork", "printed", "difference"):
        assert not widget.images[key].pixmap().isNull()
    sizes = {widget.images[k].pixmap().size().toTuple() for k in ("artwork", "printed", "difference")}
    assert len(sizes) == 1  # same enlargement for all three
    assert f"#{defect.id}" in widget.title.full_text()
    text = widget.evidence.text()
    assert "NOT verified" in text and "estimated scale" in text
    assert "Not reviewed" in widget.mark_label.text()


def test_buttons_request_marks(widget, defect):
    widget.show_defect(defect, OperatorMark.UNREVIEWED)
    received = []
    widget.mark_requested.connect(lambda defect_id, mark: received.append((defect_id, mark)))
    marks = (OperatorMark.CONFIRMED, OperatorMark.DISMISSED, OperatorMark.NEEDS_REVIEW, OperatorMark.UNREVIEWED)
    for mark in marks:
        widget.buttons[mark].click()
    assert received == [(defect.id, mark) for mark in marks]
