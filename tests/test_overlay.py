import numpy as np
import pytest

from print_inspector.gui.overlay import OVERLAY_MODES, make_overlay


@pytest.fixture
def pair():
    reference = np.full((60, 80, 3), 255, np.uint8)
    reference[20:40, 20:60] = 0
    photo = np.full((60, 80, 3), 200, np.uint8)
    photo[22:42, 20:60] = 30  # 2 px lower than the artwork
    return reference, photo


def test_all_modes_produce_same_size_bgr(pair):
    reference, photo = pair
    for mode in OVERLAY_MODES:
        out = make_overlay(reference, photo, None, mode, 0.5)
        assert out.shape == reference.shape and out.dtype == np.uint8
        assert out.flags.c_contiguous


def test_blend_opacity_extremes(pair):
    reference, photo = pair
    assert np.array_equal(make_overlay(reference, photo, None, "blend", 0.0), reference)
    assert np.array_equal(make_overlay(reference, photo, None, "blend", 1.0), photo)


def test_edges_drawn_in_magenta_on_photo(pair):
    reference, photo = pair
    out = make_overlay(reference, photo, None, "edges", 1.0)
    magenta = np.all(out == [255, 0, 255], axis=2)
    assert magenta[20, 30] or magenta[19, 30]  # top edge of the artwork block
    assert np.array_equal(out[5, 5], photo[5, 5])  # away from edges: photo unchanged


def test_difference_highlights_misalignment(pair):
    reference, photo = pair
    out = make_overlay(reference, photo, None, "difference", 0.5).astype(int).sum(axis=2)
    assert out[41, 30] > out[30, 30]  # strip covered only by the shifted photo


def test_checkerboard_alternates(pair):
    reference, photo = pair
    out = make_overlay(reference, photo, None, "checkerboard", 0.5, checker_px=10)
    assert np.array_equal(out[0, 0], reference[0, 0])
    assert np.array_equal(out[0, 10], photo[0, 10])


def test_outside_mask_is_darkened(pair):
    reference, photo = pair
    mask = np.full(reference.shape[:2], 255, np.uint8)
    mask[:, :10] = 0
    out = make_overlay(reference, photo, mask, "blend", 1.0)
    assert out[5, 5].sum() < photo[5, 5].sum()
    assert np.array_equal(out[5, 50], photo[5, 50])


def test_input_not_modified_and_errors(pair):
    reference, photo = pair
    before = photo.copy()
    make_overlay(reference, photo, None, "edges", 1.0)
    assert np.array_equal(photo, before)
    with pytest.raises(ValueError):
        make_overlay(reference, photo, None, "nonsense", 0.5)
    with pytest.raises(ValueError):
        make_overlay(reference, photo[:10], None, "blend", 0.5)
