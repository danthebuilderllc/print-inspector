"""Alignment overlay images (artwork vs. registered photo) for visual inspection."""
from __future__ import annotations

import cv2
import numpy as np

OVERLAY_MODES: dict[str, str] = {
    "blend": "Blend",
    "edges": "Artwork edges on photo",
    "difference": "Difference",
    "checkerboard": "Checkerboard",
}
_EDGE_COLOUR = np.array([255, 0, 255], np.float32)  # magenta (BGR)


def make_overlay(
    reference: np.ndarray,
    photo: np.ndarray,
    mask: np.ndarray | None,
    mode: str,
    opacity: float,
    checker_px: int = 48,
) -> np.ndarray:
    """A new HxWx3 uint8 BGR image combining artwork and registered photo.

    ``opacity`` (0-1): blend - share of the photo; edges - strength of the
    artwork outline; difference - amplification; checkerboard - unused.
    Areas outside ``mask`` (not comparable) are darkened.
    """
    if reference.shape != photo.shape:
        raise ValueError("Reference and photo must have the same size")
    opacity = float(np.clip(opacity, 0.0, 1.0))
    if mode == "blend":
        out = cv2.addWeighted(reference, 1.0 - opacity, photo, opacity, 0.0)
    elif mode == "edges":
        edges = cv2.Canny(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), 50, 150) > 0
        out = photo.copy()
        out[edges] = np.clip(
            (1.0 - opacity) * out[edges].astype(np.float32) + opacity * _EDGE_COLOUR, 0, 255
        ).astype(np.uint8)
    elif mode == "difference":
        diff = cv2.absdiff(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY))
        gain = 1.0 + 7.0 * opacity
        out = cv2.applyColorMap(cv2.convertScaleAbs(diff, alpha=gain), cv2.COLORMAP_INFERNO)
    elif mode == "checkerboard":
        h, w = reference.shape[:2]
        yy, xx = np.indices((h, w))
        photo_cells = ((yy // checker_px) + (xx // checker_px)) % 2 == 1
        out = reference.copy()
        out[photo_cells] = photo[photo_cells]
    else:
        raise ValueError(f"Unknown overlay mode: {mode}")
    if mask is not None:
        outside = mask == 0
        out[outside] = (out[outside].astype(np.float32) * 0.3 + np.float32([60, 20, 20])).astype(np.uint8)
    return np.ascontiguousarray(out)
