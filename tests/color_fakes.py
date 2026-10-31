"""Synthetic colour labels and a simulated phone camera (white balance, exposure, channel crosstalk)."""
from __future__ import annotations

import cv2
import numpy as np

from print_inspector.core.color import bgr_to_linear_rgb, linear_to_srgb

SIZE = (1200, 800)
PAPER = (245, 245, 245)
PATCHES = {  # name: (BGR, box)
    "red": ((40, 40, 200), (80, 80, 300, 220)),
    "red2": ((40, 40, 200), (80, 420, 300, 220)),
    "blue": ((170, 90, 30), (460, 80, 300, 220)),
    "green": ((70, 150, 40), (460, 420, 300, 220)),
    "yellow": ((40, 200, 235), (840, 80, 280, 220)),
}


def make_color_label() -> np.ndarray:
    image = np.full((SIZE[1], SIZE[0], 3), PAPER, np.uint8)
    for colour, (x, y, w, h) in PATCHES.values():
        cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), colour, -1)
    cv2.putText(image, "COLOUR 123", (840, 520), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (30, 30, 30), 3, cv2.LINE_AA)
    return image


def camera(image: np.ndarray, gains=(1.0, 1.0, 1.0), exposure: float = 1.0, crosstalk: float = 0.0) -> np.ndarray:
    """What an uncalibrated phone might record: white balance gains (R, G, B), exposure, channel mixing."""
    linear = bgr_to_linear_rgb(image)
    mix = np.full((3, 3), crosstalk / 2) + np.eye(3) * (1 - crosstalk)
    out = (linear @ mix.T) * np.array(gains) * exposure
    return np.clip(np.round(linear_to_srgb(out)[..., ::-1]), 0, 255).astype(np.uint8)


def recolour(image: np.ndarray, box, bgr) -> None:
    x, y, w, h = box
    image[y : y + h, x : x + w] = bgr
