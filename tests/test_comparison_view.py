import numpy as np
import pytest

from print_inspector import synthetic as syn
from print_inspector.core.comparison import compare_to_artwork
from print_inspector.core.decision import OperatorMark
from print_inspector.core.defects import assess_defects
from print_inspector.gui.comparison_view import COMPARISON_VIEWS, DISMISSED_COLOUR, render_comparison
from tests.test_comparison import ARTWORK, full_mask, simulate


@pytest.fixture(scope="module")
def compared():
    from print_inspector.config import load_config

    config = load_config(user_path=None)
    printed, defects = syn.add_defects(ARTWORK, ["extra_mark", "missing_line"], seed=2)
    registered = simulate(printed)
    mask = full_mask(ARTWORK.shape)
    result = compare_to_artwork(registered, ARTWORK, mask, config.comparison, 0.9)
    report = assess_defects(result, ARTWORK, registered, 12.0, False, config.defects)
    return result, registered, mask, report


def test_every_view_renders(compared):
    result, registered, mask, report = compared
    for mode in COMPARISON_VIEWS:
        image = render_comparison(result, registered, mask, mode, 0.5, report=report)
        assert image.shape == registered.shape and image.dtype == np.uint8 and image.flags.c_contiguous


def test_defects_are_boxed_highlighted_and_dismissed_gray(compared):
    result, registered, mask, report = compared
    d = report.defects[0]
    edge = (d.y - 4, d.x + d.width // 2)
    plain = render_comparison(result, registered, mask, "defects", 0.5, report=report)
    assert tuple(plain[edge]) == (0, 0, 255)  # red box
    highlighted = render_comparison(result, registered, mask, "defects", 0.5, highlight=d.id, report=report)
    assert tuple(highlighted[edge]) == (0, 255, 255)
    dismissed = render_comparison(
        result, registered, mask, "defects", 0.5, report=report, marks={d.id: OperatorMark.DISMISSED}
    )
    assert tuple(dismissed[edge]) == DISMISSED_COLOUR


def test_edge_view_colours(compared):
    result, registered, mask, _ = compared
    image = render_comparison(result, registered, mask, "edge_map", 0.5)
    missing = result.edge_difference == 1
    if missing.any():
        assert tuple(image[missing][0]) == (0, 0, 230)


def test_gain_increases_brightness(compared):
    result, registered, mask, _ = compared
    weak = render_comparison(result, registered, mask, "gray", 0.0).astype(int).sum()
    strong = render_comparison(result, registered, mask, "gray", 1.0).astype(int).sum()
    assert strong > weak


def test_inputs_unchanged_and_unknown_mode(compared):
    result, registered, mask, _ = compared
    before = registered.copy()
    render_comparison(result, registered, mask, "defects", 0.5)
    assert np.array_equal(registered, before)
    with pytest.raises(ValueError):
        render_comparison(result, registered, mask, "nonsense", 0.5)
