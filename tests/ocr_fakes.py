"""Synthetic text labels with known character positions, and a scripted stand-in for Tesseract.

The stand-in returns the words a real OCR engine would return for each region:
the artwork text with exact word boxes, and for the print what the test says
OCR reads there. Everything else - cropping, preparation, text comparison,
the character ink check and the decision - runs for real.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from print_inspector.core.ocr import OcrContext, OcrError, OcrWord
from print_inspector.core.text_inspection import _PAD

FONT = cv2.FONT_HERSHEY_SIMPLEX
SIZE = (1200, 800)  # artwork / grid size (w, h); 12 px per mm -> 100 mm label


@dataclass
class TextLabel:
    image: np.ndarray
    lines: list[str]
    char_boxes: list[list[tuple[int, int, int, int] | None]]  # per line, per character (None for spaces)

    def word_boxes(self, index: int) -> list[tuple[str, tuple[int, int, int, int]]]:
        text, boxes = self.lines[index], self.char_boxes[index]
        words, current = [], []
        for c, box in zip(text + " ", boxes + [None]):
            if c == " ":
                if current:
                    xs = [b[0] for _, b in current] + [b[0] + b[2] for _, b in current]
                    ys = [b[1] for _, b in current] + [b[1] + b[3] for _, b in current]
                    words.append(("".join(ch for ch, _ in current), (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))))
                current = []
            else:
                current.append((c, box))
        return words

    def line_box(self, index: int) -> tuple[int, int, int, int]:
        boxes = [b for _, b in self.word_boxes(index)]
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes)
        y1 = max(b[1] + b[3] for b in boxes)
        return x0, y0, x1 - x0, y1 - y0

    def char_box(self, line: int, char: int) -> tuple[int, int, int, int]:
        return self.char_boxes[line][char]


def make_text_label(
    lines=("LOT 12345", "EXP 2027/06", "Net wt. 250 g", "PRINT INSPECTOR"),
    scale: float = 1.6, thickness: int = 3, left: int = 80, top: int = 140, gap: int = 140,
) -> TextLabel:
    image = np.full((SIZE[1], SIZE[0], 3), 236, np.uint8)
    cv2.rectangle(image, (900, 120), (1100, 320), (40, 40, 200), -1)  # a non-text element
    char_boxes = []
    for i, text in enumerate(lines):
        baseline_y = top + i * gap
        cv2.putText(image, text, (left, baseline_y), FONT, scale, (20, 20, 20), thickness, cv2.LINE_AA)
        boxes = []
        for k, c in enumerate(text):
            if c == " ":
                boxes.append(None)
                continue
            x_start = left + cv2.getTextSize(text[:k], FONT, scale, thickness)[0][0] if k else left
            x_end = left + cv2.getTextSize(text[: k + 1], FONT, scale, thickness)[0][0]
            glyph = np.zeros((200, 200), np.uint8)
            cv2.putText(glyph, c, (20, 120), FONT, scale, 255, thickness, cv2.LINE_AA)
            ys = np.nonzero(glyph.max(axis=1) > 60)[0]
            y0, y1 = (baseline_y - 120 + int(ys[0]), baseline_y - 120 + int(ys[-1]) + 1) if len(ys) else (baseline_y - 10, baseline_y)
            boxes.append((int(x_start), y0, max(1, int(x_end - x_start)), y1 - y0))
        char_boxes.append(boxes)
    return TextLabel(image, list(lines), char_boxes)


def erase(image: np.ndarray, box, fraction_x=(0.0, 1.0), fraction_y=(0.0, 1.0), pad=2, colour=236) -> None:
    """Paint over part of a character box with label stock (missing ink)."""
    x, y, w, h = box
    x0, x1 = int(x + fraction_x[0] * w) - (pad if fraction_x[0] == 0 else 0), int(x + fraction_x[1] * w) + (pad if fraction_x[1] == 1 else 0)
    y0, y1 = int(y + fraction_y[0] * h) - (pad if fraction_y[0] == 0 else 0), int(y + fraction_y[1] * h) + (pad if fraction_y[1] == 1 else 0)
    image[max(0, y0) : y1, max(0, x0) : x1] = colour


@dataclass
class ScriptedOcr:
    """Answers OCR calls from the label's known text.

    ``reads`` maps a line index to what OCR reads on the print (text, confidence);
    ``artwork_reads`` likewise for the artwork. ``hidden`` lines are not found on the
    artwork page; ``fail`` lines raise an OCR error.
    """

    label: TextLabel
    reads: dict[int, tuple[str, float]] = field(default_factory=dict)
    artwork_reads: dict[int, tuple[str, float]] = field(default_factory=dict)
    confidence: float = 95.0
    hidden: set[int] = field(default_factory=set)
    fail: set[int] = field(default_factory=set)
    name: str = "Scripted OCR"
    delay_s: float = 0.0  # simulate slow OCR
    calls: list[OcrContext] = field(default_factory=list)
    grid_scale: float = 1.0  # comparison grid pixels per artwork pixel (taken from the page call)

    def _scaled(self, box):
        k = self.grid_scale
        return tuple(int(round(v * k)) for v in box)

    def _words(self, index: int, source: str) -> list[tuple[str, tuple[int, int, int, int], float]]:
        artwork_words = [(t, self._scaled(b)) for t, b in self.label.word_boxes(index)]
        override = (self.reads if source == "print" else self.artwork_reads).get(index)
        if override is None:
            return [(t, b, self.confidence) for t, b in artwork_words]
        text, conf = override
        tokens = text.split()
        if not tokens:
            return []
        if len(tokens) == len(artwork_words):
            return [(t, b, conf) for t, (_, b) in zip(tokens, artwork_words)]
        x, y, w, h = self._scaled(self.label.line_box(index))
        total = sum(len(t) for t in tokens) + len(tokens) - 1
        out, pos = [], 0
        for t in tokens:
            x0 = x + pos / total * w
            x1 = x + (pos + len(t)) / total * w
            out.append((t, (int(x0), y, max(1, int(x1 - x0)), h), conf))
            pos += len(t) + 1
        return out

    def recognize(self, image: np.ndarray, psm: int, context: OcrContext) -> list[OcrWord]:
        self.calls.append(context)
        if self.delay_s:
            time.sleep(self.delay_s)
        rx, ry, rw, rh = context.region
        scale = (image.shape[1] - 2 * _PAD) / rw
        if context.mode == "page":
            self.grid_scale = rw / self.label.image.shape[1]
            indices = [i for i in range(len(self.label.lines)) if i not in self.hidden]
        else:
            indices = [self._line_at(context.region)]
            if indices[0] is None:
                return []
            if indices[0] in self.fail:
                raise OcrError("scripted failure")
        words = []
        for line_no, index in enumerate(indices):
            source = "artwork" if context.source == "artwork" else "print"
            for word_no, (text, (x, y, w, h), conf) in enumerate(self._words(index, source)):
                words.append(OcrWord(
                    text, conf,
                    int(round((x - rx) * scale + _PAD)), int(round((y - ry) * scale + _PAD)),
                    int(round(w * scale)), int(round(h * scale)),
                    block=1, paragraph=1, line=line_no + 1,
                ))
        return words

    def _line_at(self, region) -> int | None:
        rx, ry, rw, rh = region
        best, best_overlap = None, 0
        for i in range(len(self.label.lines)):
            x, y, w, h = self._scaled(self.label.line_box(i))
            ox = min(rx + rw, x + w) - max(rx, x)
            oy = min(ry + rh, y + h) - max(ry, y)
            if ox > 0 and oy > 0 and ox * oy > best_overlap:
                best, best_overlap = i, ox * oy
        return best
