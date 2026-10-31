"""Barcode and QR code decoding with ZXing-C++ (the ``zxing-cpp`` package).

The library is compiled code inside the Python package: decoding is local and
needs no network, no service and no licence key. It is imported only when
needed, so the rest of the application works (with a clear message) if it is
missing.

Supported here: EAN-13, UPC-A, Code 128 and QR Code. EAN-13 and UPC-A values
are compared as 13-digit GTINs (a UPC-A is an EAN-13 with a leading 0, and
ZXing may report it either way).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import metadata

import cv2
import numpy as np

log = logging.getLogger(__name__)

FORMAT_LABELS = {"ean13": "EAN-13", "upca": "UPC-A", "code128": "Code 128", "qr": "QR Code"}
LINEAR = frozenset({"ean13", "upca", "code128"})
EAN_UPC = frozenset({"ean13", "upca"})
EAN_UPC_MODULES = 95  # modules between the outer guard bars of EAN-13 / UPC-A


class BarcodeUnavailable(RuntimeError):
    """The decoding library is not installed or cannot be loaded."""


def _zxing():
    try:
        import zxingcpp
    except ImportError as exc:  # also covers a broken native library
        raise BarcodeUnavailable(
            "The barcode library zxing-cpp is not installed (pip install -r requirements.txt)."
        ) from exc
    return zxingcpp


def library_version() -> str:
    try:
        return f"ZXing-C++ {metadata.version('zxing-cpp')}"
    except metadata.PackageNotFoundError:
        return "ZXing-C++"


def check_available() -> str:
    """The library's name and version. Raises BarcodeUnavailable."""
    _zxing()
    return library_version()


@dataclass(frozen=True)
class Symbol:
    """One decoded barcode, in the pixel coordinates of the decoded image."""

    format: str  # ean13, upca, code128, qr
    text: str
    corners: tuple[tuple[float, float], ...]  # top-left, top-right, bottom-right, bottom-left of the symbol
    orientation: int  # degrees, as reported by the decoder
    ec_level: str = ""
    version: int | None = None  # QR version (size = 17 + 4 * version modules)
    unused_ec: float | None = None  # QR: share of the error correction capacity not needed (1 = no errors)

    @property
    def box(self) -> tuple[int, int, int, int]:
        xs = [p[0] for p in self.corners]
        ys = [p[1] for p in self.corners]
        x0, y0 = int(np.floor(min(xs))), int(np.floor(min(ys)))
        return x0, y0, int(np.ceil(max(xs))) - x0 + 1, int(np.ceil(max(ys))) - y0 + 1

    @property
    def label(self) -> str:
        return FORMAT_LABELS[self.format]

    def moved(self, dx: float, dy: float, scale: float = 1.0) -> Symbol:
        """The same symbol in another image: ``(x / scale + dx, y / scale + dy)``."""
        corners = tuple((x / scale + dx, y / scale + dy) for x, y in self.corners)
        return Symbol(self.format, self.text, corners, self.orientation, self.ec_level, self.version, self.unused_ec)


def canonical_value(format_key: str, text: str) -> str:
    """Value for comparison: EAN-13 / UPC-A as a 13-digit GTIN, otherwise the text itself."""
    if format_key in EAN_UPC:
        digits = "".join(c for c in text if c.isdigit())
        return digits.zfill(13) if len(digits) <= 13 else digits
    return text


def same_symbology(a: str, b: str) -> bool:
    return a == b or (a in EAN_UPC and b in EAN_UPC)


def gtin_check_digit_ok(value: str) -> bool:
    """Modulo-10 check digit of an EAN-13 / UPC-A (as GTIN-13)."""
    digits = canonical_value("ean13", value)
    if len(digits) != 13 or not digits.isdigit():
        return False
    total = sum(int(d) * (3 if i % 2 else 1) for i, d in enumerate(digits[:12]))
    return (10 - total % 10) % 10 == int(digits[12])


def _format_key(zx, fmt) -> str | None:
    table = {
        zx.BarcodeFormat.EAN13: "ean13",
        zx.BarcodeFormat.UPCA: "upca",
        zx.BarcodeFormat.Code128: "code128",
        zx.BarcodeFormat.QRCode: "qr",
    }
    return table.get(fmt)


def _to_symbol(zx, result) -> Symbol | None:
    key = _format_key(zx, result.format)
    if key is None or not result.valid:
        return None
    p = result.position
    corners = tuple((float(q.x), float(q.y)) for q in (p.top_left, p.top_right, p.bottom_right, p.bottom_left))
    extra = getattr(result, "extra", None) or {}
    version = extra.get("Version")
    try:
        version = int(version) if version not in (None, "") else None
    except (TypeError, ValueError):
        version = None
    unused = extra.get("UEC")
    return Symbol(
        key, result.text, corners, int(result.orientation), str(result.ec_level or ""), version,
        float(unused) if isinstance(unused, (int, float)) else None,
    )


def decode(image: np.ndarray, formats: tuple[str, ...] = tuple(FORMAT_LABELS), thorough: bool = True) -> list[Symbol]:
    """Decode the barcodes in a gray or BGR uint8 image. Raises BarcodeUnavailable.

    ``thorough`` retries with a global binarizer, a contrast-stretched copy and a
    2x enlarged copy when the first attempt finds nothing (photos of small codes).
    """
    zx = _zxing()
    wanted = {
        "ean13": zx.BarcodeFormat.EAN13, "upca": zx.BarcodeFormat.UPCA,
        "code128": zx.BarcodeFormat.Code128, "qr": zx.BarcodeFormat.QRCode,
    }
    zx_formats = tuple(wanted[f] for f in formats)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray)
    if gray.shape[0] < 8 or gray.shape[1] < 8:
        return []

    def attempt(img: np.ndarray, binarizer=None) -> list[Symbol]:
        kwargs = {"formats": zx_formats, "try_rotate": True, "try_downscale": True, "try_invert": True}
        if binarizer is not None:
            kwargs["binarizer"] = binarizer
        found = [_to_symbol(zx, r) for r in zx.read_barcodes(img, **kwargs)]
        return [s for s in found if s is not None]

    symbols = attempt(gray)
    if symbols or not thorough:
        return symbols
    symbols = attempt(gray, zx.Binarizer.GlobalHistogram)
    if symbols:
        return symbols
    lo, hi = np.percentile(gray, 1), np.percentile(gray, 99)
    stretched = np.clip((gray.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1.0)), 0, 255).astype(np.uint8)
    symbols = attempt(stretched)
    if symbols:
        return symbols
    if max(gray.shape) <= 3000:
        big = cv2.resize(stretched, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        return [s.moved(0, 0, 2.0) for s in attempt(big)]
    return []


def render(format_key: str, value: str, module_px: int = 3, quiet_zone: bool = True) -> np.ndarray:
    """A clean grayscale image of a symbol (for tests and demonstrations). Raises BarcodeUnavailable / ValueError."""
    zx = _zxing()
    fmt = {
        "ean13": zx.BarcodeFormat.EAN13, "upca": zx.BarcodeFormat.UPCA,
        "code128": zx.BarcodeFormat.Code128, "qr": zx.BarcodeFormat.QRCode,
    }[format_key]
    try:
        barcode = zx.create_barcode(value, fmt)
    except Exception as exc:  # noqa: BLE001 - the library raises its own error type for invalid content
        raise ValueError(f"Cannot encode '{value}' as {FORMAT_LABELS[format_key]}: {exc}") from exc
    return np.array(zx.write_barcode_to_image(barcode, scale=module_px, add_hrt=False, add_quiet_zones=quiet_zone))
