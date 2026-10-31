"""Synthetic labels with barcodes (rendered with the same local library) and controlled damage."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from print_inspector.core.barcodes import render

SIZE = (1200, 800)  # artwork / grid size (w, h); 12 px per mm -> 100 mm label
STOCK = 236

SYMBOLS = {
    "ean": ("ean13", "4006381333931", 3, (60, 60)),
    "upc": ("upca", "036000291452", 3, (60, 300)),
    "code128": ("code128", "LOT-12345/A", 3, (520, 60)),
    "qr": ("qr", "https://example.com/p?id=12345", 8, (700, 360)),
}


@dataclass
class BarcodeLabel:
    image: np.ndarray
    boxes: dict[str, tuple[int, int, int, int]]  # symbol (bars / modules, without quiet zone) per name
    module: dict[str, int]


def place(image: np.ndarray, symbol: np.ndarray, origin, bar_height: int | None = None) -> tuple[int, int, int, int]:
    """Paste a rendered symbol (white quiet zone made label stock); returns the dark symbol's box."""
    if bar_height is not None and symbol.shape[0] < bar_height:
        symbol = cv2.resize(symbol, (symbol.shape[1], bar_height), interpolation=cv2.INTER_NEAREST)
    x, y = origin
    h, w = symbol.shape
    patch = np.where(symbol < 128, 20, STOCK).astype(np.uint8)
    image[y : y + h, x : x + w] = patch[:, :, None]
    ys, xs = np.nonzero(symbol < 128)
    return x + int(xs.min()), y + int(ys.min()), int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1


def make_barcode_label(names=("ean", "code128", "qr"), values: dict | None = None) -> BarcodeLabel:
    image = np.full((SIZE[1], SIZE[0], 3), STOCK, np.uint8)
    cv2.putText(image, "BARCODES", (60, 760), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3, cv2.LINE_AA)
    boxes, modules = {}, {}
    for name in names:
        fmt, value, module, origin = SYMBOLS[name]
        value = (values or {}).get(name, value)
        symbol = render(fmt, value, module)
        boxes[name] = place(image, symbol, origin, None if fmt == "qr" else 180)
        modules[name] = module
    return BarcodeLabel(image, boxes, modules)


def paint(image: np.ndarray, box, value=STOCK) -> None:
    x, y, w, h = box
    image[y : y + h, x : x + w] = value
