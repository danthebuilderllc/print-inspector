"""Images that visualise a comparison result (for the GUI)."""
from __future__ import annotations

import cv2
import numpy as np

from print_inspector.core.comparison import EDGE_EXTRA, EDGE_MISSING, ComparisonResult
from print_inspector.core.decision import OperatorMark
from print_inspector.core.defects import DefectReport

COMPARISON_VIEWS: dict[str, str] = {
    "defects": "Defects",
    "score": "Combined score",
    "gray": "Gray difference",
    "structural": "Structural difference (SSIM)",
    "edge_map": "Edge difference",
    "expected": "Expected print",
}
DEFECT_COLOUR = (0, 0, 255)  # red (BGR)
HIGHLIGHT_COLOUR = (0, 255, 255)  # yellow
DISMISSED_COLOUR = (160, 160, 160)  # gray
_OUTLINE_COLOUR = (255, 255, 0)  # cyan


def _colour_map(values: np.ndarray, gain: float) -> np.ndarray:
    scaled = np.clip(values.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)


def _darken_outside(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is not None:
        outside = mask == 0
        image[outside] = (image[outside].astype(np.float32) * 0.3 + np.float32([60, 20, 20])).astype(np.uint8)
    return image


def defect_colour(mark: OperatorMark, highlighted: bool) -> tuple[int, int, int]:
    if highlighted:
        return HIGHLIGHT_COLOUR
    return DISMISSED_COLOUR if mark is OperatorMark.DISMISSED else DEFECT_COLOUR


def render_comparison(
    result: ComparisonResult,
    registered: np.ndarray,
    mask: np.ndarray | None,
    mode: str,
    strength: float,
    highlight: int | None = None,
    report: DefectReport | None = None,
    marks: dict[int, OperatorMark] | None = None,
) -> np.ndarray:
    """A new BGR uint8 image for ``mode`` (see COMPARISON_VIEWS).

    ``strength`` (0-1) sets the display gain of the difference maps.
    ``highlight`` is the id of a defect to emphasise; dismissed defects are drawn gray.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    marks = marks or {}
    if mode == "defects":
        out = registered.copy()
        for d in (report.defects if report is not None else ()):
            pad = 4
            colour = defect_colour(marks.get(d.id, OperatorMark.UNREVIEWED), d.id == highlight)
            thickness = 3 if d.id == highlight else 2
            cv2.rectangle(out, (d.x - pad, d.y - pad), (d.x + d.width + pad, d.y + d.height + pad), colour, thickness)
            cv2.putText(out, str(d.id), (d.x - pad, max(14, d.y - pad - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    elif mode == "score":
        out = _colour_map(result.combined_score, 85.0 * (0.25 + 1.5 * strength))
        contours, _ = cv2.findContours(result.candidate_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, _OUTLINE_COLOUR, 1)
    elif mode == "gray":
        out = _colour_map(result.gray_difference, 1.0 + 9.0 * strength)
    elif mode == "structural":
        out = _colour_map(result.structural_difference, 0.5 + 2.0 * strength)
    elif mode == "edge_map":
        base = cv2.cvtColor(cv2.cvtColor(result.expected, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        out = (255 - (255 - base.astype(np.float32)) * 0.25).astype(np.uint8)  # faint artwork
        out[result.edge_difference == EDGE_MISSING] = (0, 0, 230)  # artwork edges missing in the print
        out[result.edge_difference == EDGE_EXTRA] = (230, 80, 0)  # print edges not in the artwork
    elif mode == "expected":
        out = result.expected.copy()
    else:
        raise ValueError(f"Unknown comparison view: {mode}")
    return np.ascontiguousarray(_darken_outside(out, mask))
