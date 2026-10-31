"""Label boundary detection and feature matching on photos with known corner positions."""
import cv2
import numpy as np
import pytest

from print_inspector import synthetic as syn
from print_inspector.core.label_detection import (
    detect_label_boundary,
    is_convex_quad,
    match_artwork,
    order_corners,
    refine_corners,
    snap_corners_to_edges,
)

POSES = {
    "frontal": dict(),
    "roll 15": dict(roll_deg=15),
    "roll -35, tilt 20": dict(roll_deg=-35, tilt_x_deg=20),
    "tilt 30/15": dict(roll_deg=5, tilt_x_deg=30, tilt_y_deg=15),
    "upside down": dict(roll_deg=188, tilt_y_deg=10),
    "quarter turn": dict(roll_deg=97, tilt_x_deg=12),
    "small, off-centre": dict(label_fraction=0.3, roll_deg=20, offset=(0.15, -0.1)),
    "large": dict(label_fraction=0.85, roll_deg=3),
}


@pytest.fixture(scope="module")
def artwork():
    return syn.make_artwork()


@pytest.fixture
def cfg(app_config):
    return app_config.alignment


def _max_corner_error(found: np.ndarray, truth: np.ndarray) -> float:
    """Largest distance from each true corner to the nearest found corner (order-independent)."""
    return max(float(np.min(np.linalg.norm(found - t, axis=1))) for t in truth)


def test_order_corners_any_input_order():
    square = np.array([[10, 10], [110, 12], [108, 90], [12, 88]], float)
    for perm in ([2, 0, 3, 1], [3, 2, 1, 0], [1, 3, 0, 2]):
        assert np.array_equal(order_corners(square[perm]), square)


def test_is_convex_quad():
    assert is_convex_quad(np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float))
    assert not is_convex_quad(np.array([[0, 0], [10, 10], [10, 0], [0, 10]], float))  # bow-tie
    assert not is_convex_quad(np.array([[0, 0], [10, 0], [2, 2], [0, 10]], float))  # dent


@pytest.mark.parametrize("pose", list(POSES), ids=list(POSES))
def test_boundary_corners_sub_pixel(artwork, cfg, pose):
    photo = syn.make_photo(artwork, **POSES[pose])
    found = detect_label_boundary(photo.image, cfg)
    assert found is not None
    assert _max_corner_error(found.corners, photo.corners) < 1.0
    assert found.min_support >= cfg.min_edge_support_warning
    assert found.max_rms_px < cfg.max_edge_fit_rms_px_warning


def test_rounded_corners(artwork, cfg):
    """Corners are the intersections of the straight sides, not the rounded outline."""
    background = (95, 105, 115)
    rounded = artwork.copy()
    h, w = rounded.shape[:2]
    r = 60
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (w - 1 - r, h - 1 - r), (r, h - 1 - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1)
    rounded[mask == 0] = background
    corners = syn.project_label(w / h, (2000, 1500), roll_deg=-12, tilt_x_deg=15)
    photo = syn.render_photo(rounded, (2000, 1500), corners, background_bgr=background)
    found = detect_label_boundary(photo.image, cfg)
    assert found is not None
    assert _max_corner_error(found.corners, photo.corners) < 1.5


def test_printed_frame_inside_label_is_not_chosen(artwork, cfg):
    framed = artwork.copy()
    h, w = framed.shape[:2]
    cv2.rectangle(framed, (40, 40), (w - 41, h - 41), (0, 0, 0), 6)
    photo = syn.make_photo(framed, roll_deg=8, tilt_y_deg=10)
    found = detect_label_boundary(photo.image, cfg)
    assert found is not None
    assert _max_corner_error(found.corners, photo.corners) < 1.0


def test_low_contrast_background_is_not_trusted(artwork, cfg):
    corners = syn.project_label(1.5, (2000, 1500), roll_deg=12, tilt_x_deg=10)
    photo = syn.render_photo(artwork, (2000, 1500), corners, background_bgr=(250, 250, 250), background_texture=1.0)
    found = detect_label_boundary(photo.image, cfg)
    assert found is None or found.min_support < cfg.min_edge_support_critical


def test_blank_photo_has_no_boundary(cfg):
    assert detect_label_boundary(np.full((1200, 1600, 3), 120, np.uint8), cfg) is None


def test_refine_corners_recovers_from_offsets(artwork, cfg):
    photo = syn.make_photo(artwork, roll_deg=20, tilt_x_deg=15)
    rng = np.random.default_rng(3)
    rough = photo.corners + rng.uniform(-6, 6, (4, 2))
    gray = cv2.cvtColor(photo.image, cv2.COLOR_BGR2GRAY)
    refined, sides = refine_corners(gray, rough, 12.0, cfg)
    assert _max_corner_error(refined, photo.corners) < 1.0
    assert len(sides) == 4


def test_snap_manual_corners_to_edges(artwork, cfg):
    photo = syn.make_photo(artwork, roll_deg=-25, tilt_y_deg=20)
    rough = photo.corners + np.array([[8, -6], [-7, 5], [6, 8], [-5, -8]])
    snapped = snap_corners_to_edges(photo.image, rough[[2, 0, 3, 1]], cfg)  # any click order
    assert snapped is not None
    assert _max_corner_error(snapped, photo.corners) < 1.0


def test_snap_returns_none_without_edges(cfg):
    blank = np.full((600, 800, 3), 128, np.uint8)
    corners = np.array([[100, 100], [700, 100], [700, 500], [100, 500]], float)
    assert snap_corners_to_edges(blank, corners, cfg) is None


def test_feature_match_locates_label_without_visible_edges(artwork, cfg):
    corners = syn.project_label(1.5, (2000, 1500), roll_deg=12, tilt_x_deg=10)
    photo = syn.render_photo(artwork, (2000, 1500), corners, background_bgr=(250, 250, 250), background_texture=1.0)
    match = match_artwork(artwork, photo.image, cfg)
    assert match is not None
    assert match.inliers >= cfg.feature_min_inliers
    assert match.inlier_ratio >= cfg.feature_min_inlier_ratio
    assert _max_corner_error(match.corners, photo.corners) < 3.0


def test_feature_match_rejects_unrelated_artwork(cfg):
    photo = syn.make_photo(syn.make_artwork(), roll_deg=5)
    rng = np.random.default_rng(0)
    unrelated = cv2.GaussianBlur(rng.integers(0, 255, (800, 1200, 3), dtype=np.uint8), (0, 0), 3)
    match = match_artwork(unrelated, photo.image, cfg)
    assert match is None or match.inlier_ratio < cfg.feature_min_inlier_ratio
