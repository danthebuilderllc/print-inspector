"""Local image file loading and the internal image representation.

Internal representation (:class:`LoadedImage`)
-----------------------------------------------
``original``
    The decoded pixels exactly as stored in the file, at native bit depth and
    channel layout, with the EXIF orientation applied (a lossless pixel
    rearrangement; no values change). Layout follows OpenCV conventions:

    * ``HxW``     grayscale
    * ``HxWx3``   BGR
    * ``HxWx4``   BGRA (see ``alpha_premultiplied``)

    dtype is ``uint8``, ``uint16`` or ``float32``. Read-only.

``working``
    The representation all processing stages consume: ``HxWx3`` ``uint8`` BGR,
    C-contiguous, read-only. 16-bit data is scaled to 8 bits (rounded),
    grayscale is expanded to three equal channels, and transparent pixels are
    composited onto white label stock. When the original already is 8-bit BGR,
    ``working`` *is* ``original`` (same memory, no copy).

``alpha``
    ``HxW`` ``uint8`` coverage (255 = opaque) when the file had transparency,
    otherwise ``None``.

Both arrays are read-only, so processing stages can never alter the loaded
data in place; they must create new arrays.

Why OpenCV + Pillow
-------------------
Pixels are decoded by OpenCV from an in-memory buffer (works with non-ASCII
Windows paths, keeps 16-bit data). Pillow reads only header metadata (EXIF
orientation, DPI, TIFF ExtraSamples) without decoding pixels. Behaviour of
OpenCV 4.10 that this module compensates for (verified by tests):

* ``IMREAD_UNCHANGED`` ignores EXIF orientation for JPEG and PNG, so it is
  applied here. For TIFF, OpenCV applies the orientation tag itself.
* A truncated JPEG decodes without error (missing rows are filled with grey),
  so JPEGs are checked for a complete entropy-coded stream and end marker.
* 8-bit RGBA TIFFs with unassociated alpha are returned premultiplied.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from print_inspector.config import EXTENSION_FORMATS, ImageLoadingConfig

log = logging.getLogger(__name__)

_EXIF_ORIENTATION_TAG = 0x0112
_TIFF_EXTRA_SAMPLES_TAG = 338
_SUPPORTED_DTYPES = (np.uint8, np.uint16, np.float32)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ImageRole(str, Enum):
    ORIGINAL = "original"
    PRINTED = "printed"

    @property
    def label(self) -> str:
        return "original artwork" if self is ImageRole.ORIGINAL else "printed photograph"


class ImageLoadError(Exception):
    """Raised with a user-readable message when an image cannot be loaded."""


@dataclass(frozen=True, eq=False)
class LoadedImage:
    """A decoded image. See the module docstring for the representation."""

    path: Path
    role: ImageRole
    file_format: str
    file_size_bytes: int
    original: np.ndarray
    working: np.ndarray
    alpha: np.ndarray | None = None
    alpha_premultiplied: bool = False
    exif_orientation: int = 1
    dpi: tuple[float, float] | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def height(self) -> int:
        return int(self.original.shape[0])

    @property
    def width(self) -> int:
        return int(self.original.shape[1])

    @property
    def channels(self) -> int:
        return 1 if self.original.ndim == 2 else int(self.original.shape[2])

    @property
    def bit_depth(self) -> int:
        return self.original.dtype.itemsize * 8

    @property
    def color_mode(self) -> str:
        return {1: "Grayscale", 3: "RGB", 4: "RGBA"}[self.channels]

    @property
    def has_alpha(self) -> bool:
        return self.alpha is not None

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1_000_000

    @property
    def physical_size_mm(self) -> tuple[float, float] | None:
        """(width, height) in mm from the file's DPI, if the file declares one."""
        if self.dpi is None:
            return None
        return (self.width / self.dpi[0] * 25.4, self.height / self.dpi[1] * 25.4)

    def gray(self) -> np.ndarray:
        """A new ``HxW`` uint8 grayscale array computed from ``working``."""
        return cv2.cvtColor(self.working, cv2.COLOR_BGR2GRAY)

    def summary(self) -> str:
        size_mb = self.file_size_bytes / (1024 * 1024)
        return (
            f"{self.width} × {self.height} px  ·  {self.megapixels:.1f} MP  ·  {self.file_format}  ·  "
            f"{self.color_mode}  ·  {self.bit_depth}-bit  ·  {size_mb:.1f} MB"
        )


@dataclass(frozen=True)
class _HeaderInfo:
    size: tuple[int, int] | None = None
    orientation: int = 1
    dpi: tuple[float, float] | None = None
    tiff_extra_samples: int | None = None
    note: str | None = None


def allowed_extensions(role: ImageRole, config: ImageLoadingConfig) -> tuple[str, ...]:
    if role is ImageRole.ORIGINAL:
        return config.original_extensions
    return config.printed_extensions


def sniff_format(buffer: np.ndarray) -> str:
    """Identify the container format from the file's leading bytes."""
    head = buffer[:16].tobytes()
    if head.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if head.startswith(_PNG_SIGNATURE):
        return "PNG"
    if head.startswith(b"BM"):
        return "BMP"
    if head[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        return "TIFF"
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"heic", b"heix", b"hevc", b"heim", b"heis", b"mif1", b"msf1"):
            return "HEIC"
        return "ISO-BMFF"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "WEBP"
    if head[:4] in (b"GIF8",):
        return "GIF"
    return "UNKNOWN"


def jpeg_is_complete(buffer: np.ndarray) -> bool:
    """True if the JPEG has well-formed segments and reaches an end-of-image marker.

    Walks header segments by their lengths and scans each entropy-coded scan for
    the next real marker (skipping 0xFF00 stuffing and restart markers).
    Data appended after the end marker (e.g. motion-photo video) is allowed.
    """
    n = int(buffer.size)
    if n < 4 or buffer[0] != 0xFF or buffer[1] != 0xD8:
        return False
    pos = 2
    while pos + 1 < n:
        if buffer[pos] != 0xFF:
            return False
        marker = int(buffer[pos + 1])
        if marker == 0xFF:  # fill byte
            pos += 1
            continue
        if marker == 0xD9:
            return True
        if 0xD0 <= marker <= 0xD7 or marker == 0x01:
            pos += 2
            continue
        if pos + 3 >= n:
            return False
        length = (int(buffer[pos + 2]) << 8) | int(buffer[pos + 3])
        segment_end = pos + 2 + length
        if length < 2 or segment_end > n:
            return False
        if marker == 0xDA:  # start of scan: entropy-coded data follows
            pos = _next_jpeg_marker(buffer, segment_end)
            if pos < 0:
                return False
        else:
            pos = segment_end
    return False


def _next_jpeg_marker(buffer: np.ndarray, start: int) -> int:
    data = buffer[start:]
    if data.size < 2:
        return -1
    ff = np.flatnonzero(data[:-1] == 0xFF)
    following = data[ff + 1]
    is_marker = (following != 0x00) & (following != 0xFF) & ~((following >= 0xD0) & (following <= 0xD7))
    hits = ff[is_marker]
    return start + int(hits[0]) if hits.size else -1


def png_has_end_chunk(buffer: np.ndarray) -> bool:
    """True if the PNG chunk sequence is intact up to the IEND chunk."""
    n = int(buffer.size)
    pos = len(_PNG_SIGNATURE)
    while pos + 8 <= n:
        length = int.from_bytes(buffer[pos : pos + 4].tobytes(), "big")
        chunk_type = buffer[pos + 4 : pos + 8].tobytes()
        end = pos + 12 + length
        if end > n:
            return False
        if chunk_type == b"IEND":
            return True
        pos = end
    return False


def _read_header(path: Path, file_format: str) -> _HeaderInfo:
    """Read size, EXIF orientation, DPI and TIFF ExtraSamples without decoding pixels."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            # Damaged metadata only costs us orientation/DPI; pixel decoding is checked separately.
            warnings.simplefilter("ignore", UserWarning)
            with Image.open(path) as im:
                size = (int(im.size[0]), int(im.size[1]))
                orientation = 1
                if file_format == "JPEG" or (file_format == "PNG" and "exif" in im.info):
                    orientation = int(im.getexif().get(_EXIF_ORIENTATION_TAG, 1) or 1)
                elif file_format == "TIFF":
                    orientation = int(im.tag_v2.get(274, 1) or 1)
                if orientation not in range(1, 9):
                    orientation = 1
                dpi = None
                raw_dpi = im.info.get("dpi")
                if raw_dpi and len(raw_dpi) == 2:
                    x, y = float(raw_dpi[0]), float(raw_dpi[1])
                    if x > 0 and y > 0:
                        dpi = (x, y)
                extra = None
                if file_format == "TIFF":
                    value = im.tag_v2.get(_TIFF_EXTRA_SAMPLES_TAG)
                    if isinstance(value, tuple) and value:
                        value = value[0]
                    extra = int(value) if value is not None else None
                return _HeaderInfo(size, orientation, dpi, extra)
    except Image.DecompressionBombError:
        raise ImageLoadError(f"{path.name} declares an extremely large image size.") from None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        log.info("Metadata of %s could not be read: %s", path, exc)
        return _HeaderInfo(note="File metadata (orientation, DPI) could not be read; defaults assumed.")


def apply_exif_orientation(image: np.ndarray, orientation: int) -> np.ndarray:
    """Return the image transformed so it displays upright for EXIF orientation 1-8."""
    if orientation == 2:
        return cv2.flip(image, 1)
    if orientation == 3:
        return cv2.rotate(image, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(image, 0)
    if orientation == 5:
        return cv2.transpose(image)
    if orientation == 6:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.transpose(image), -1)
    if orientation == 8:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def make_working_bgr8(
    original: np.ndarray, alpha_premultiplied: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """Build the ``working`` BGR uint8 image and optional alpha from ``original``.

    Returns ``original`` itself (no copy) when it already is 8-bit BGR.
    """
    if original.dtype == np.uint8:
        data8 = original
    elif original.dtype == np.uint16:
        data8 = cv2.convertScaleAbs(original, alpha=1.0 / 257.0)
    elif original.dtype == np.float32:
        values = np.nan_to_num(original, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(values, 0.0, None, out=values)
        peak = float(values.max()) if values.size else 1.0
        data8 = cv2.convertScaleAbs(values, alpha=255.0 if peak <= 1.0 else 255.0 / peak)
    else:
        raise ValueError(f"Unsupported pixel type: {original.dtype}")

    if data8.ndim == 2:
        return cv2.cvtColor(data8, cv2.COLOR_GRAY2BGR), None
    if data8.shape[2] == 3:
        return data8, None
    if data8.shape[2] != 4:
        raise ValueError(f"Unsupported channel count: {data8.shape[2]}")

    colour = np.ascontiguousarray(data8[:, :, :3])
    alpha = np.ascontiguousarray(data8[:, :, 3])
    alpha3 = cv2.merge([alpha, alpha, alpha])
    if alpha_premultiplied:
        composite = cv2.add(colour, cv2.subtract(np.full_like(alpha3, 255), alpha3))
    else:
        composite = cv2.add(
            cv2.multiply(colour, alpha3, scale=1.0 / 255.0),
            cv2.subtract(np.full_like(alpha3, 255), alpha3),
        )
    return composite, alpha


def load_image(path: str | Path, role: ImageRole, config: ImageLoadingConfig) -> LoadedImage:
    """Load an image file for the given role, or raise :class:`ImageLoadError`."""
    path = Path(path)
    ext = path.suffix.lower()
    allowed = allowed_extensions(role, config)
    if ext not in allowed:
        raise ImageLoadError(
            f"Unsupported file type '{path.suffix or '(none)'}' for the {role.label}. "
            f"Supported types: {', '.join(allowed)}."
        )
    if not path.is_file():
        raise ImageLoadError(f"File not found: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ImageLoadError(f"Could not access {path.name}: {exc}") from exc
    if size == 0:
        raise ImageLoadError(f"{path.name} is empty (0 bytes).")
    if size > config.max_file_size_mb * 1024 * 1024:
        raise ImageLoadError(
            f"{path.name} is {size / (1024 * 1024):.0f} MB, above the "
            f"{config.max_file_size_mb:.0f} MB limit."
        )

    try:
        buffer = np.fromfile(path, dtype=np.uint8)
    except MemoryError:
        raise ImageLoadError(f"Not enough memory to read {path.name}.") from None
    except OSError as exc:
        raise ImageLoadError(f"Could not read {path.name}: {exc}") from exc

    file_format = sniff_format(buffer)
    if file_format == "HEIC":
        raise ImageLoadError(
            f"{path.name} is a HEIC/HEIF photo, which is not supported. On iPhone, set "
            "Settings > Camera > Formats to 'Most Compatible', or export the photo as JPEG."
        )
    if file_format not in set(EXTENSION_FORMATS.values()):
        raise ImageLoadError(
            f"{path.name} is not a supported image file (detected: {file_format.lower()}). "
            "The file may be corrupt or have the wrong extension."
        )
    allowed_formats = {EXTENSION_FORMATS[e] for e in allowed}
    if file_format not in allowed_formats:
        raise ImageLoadError(
            f"{path.name} contains {file_format} data, which is not accepted for the "
            f"{role.label} (accepted: {', '.join(sorted(allowed_formats))})."
        )
    notes: list[str] = []
    if EXTENSION_FORMATS[ext] != file_format:
        notes.append(f"File extension is {path.suffix} but the content is {file_format}.")

    if file_format == "JPEG" and not jpeg_is_complete(buffer):
        raise ImageLoadError(
            f"{path.name} is incomplete or damaged: the JPEG data ends before the image does. "
            "Copy the photo from the phone again."
        )
    if file_format == "PNG" and not png_has_end_chunk(buffer):
        raise ImageLoadError(f"{path.name} is incomplete or damaged (PNG data is truncated).")

    header = _read_header(path, file_format)
    if header.note:
        notes.append(header.note)
    if header.size is not None and header.size[0] * header.size[1] > config.max_pixels:
        pixels = header.size[0] * header.size[1]
        raise ImageLoadError(
            f"{path.name} has {pixels / 1e6:.0f} MP, above the {config.max_pixels / 1e6:.0f} MP limit."
        )

    try:
        decoded = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    except MemoryError:
        raise ImageLoadError(f"Not enough memory to decode {path.name}.") from None
    except cv2.error as exc:
        raise ImageLoadError(f"Could not decode {path.name}: {exc}") from exc
    del buffer
    if decoded is None or decoded.size == 0:
        raise ImageLoadError(
            f"{path.name} could not be decoded as an image. The file may be corrupt or truncated."
        )

    if decoded.ndim == 3 and decoded.shape[2] == 1:
        decoded = decoded[:, :, 0]
    if decoded.ndim not in (2, 3) or (decoded.ndim == 3 and decoded.shape[2] not in (3, 4)):
        raise ImageLoadError(f"{path.name} has an unsupported channel layout {decoded.shape}.")
    if decoded.dtype not in _SUPPORTED_DTYPES:
        raise ImageLoadError(f"{path.name} has an unsupported pixel type ({decoded.dtype}).")
    pixels = int(decoded.shape[0]) * int(decoded.shape[1])
    if pixels > config.max_pixels:
        raise ImageLoadError(
            f"{path.name} has {pixels / 1e6:.0f} MP, above the {config.max_pixels / 1e6:.0f} MP limit."
        )

    # OpenCV already applies the TIFF orientation tag; JPEG/PNG need it applied here.
    if file_format in ("JPEG", "PNG") and header.orientation != 1:
        decoded = apply_exif_orientation(decoded, header.orientation)

    # Unassociated 8-bit alpha is premultiplied by OpenCV's TIFF reader;
    # associated alpha (ExtraSamples = 1) is premultiplied in the file itself.
    premultiplied = (
        file_format == "TIFF"
        and decoded.ndim == 3
        and decoded.shape[2] == 4
        and (header.tiff_extra_samples == 1 or (header.tiff_extra_samples == 2 and decoded.dtype == np.uint8))
    )

    original = np.ascontiguousarray(decoded)
    original.setflags(write=False)
    try:
        working, alpha = make_working_bgr8(original, premultiplied)
    except MemoryError:
        raise ImageLoadError(f"Not enough memory to prepare {path.name} for processing.") from None
    working = np.ascontiguousarray(working)
    working.setflags(write=False)
    if alpha is not None:
        alpha.setflags(write=False)

    return LoadedImage(
        path=path,
        role=role,
        file_format=file_format,
        file_size_bytes=size,
        original=original,
        working=working,
        alpha=alpha,
        alpha_premultiplied=premultiplied,
        exif_orientation=header.orientation,
        dpi=header.dpi,
        notes=tuple(notes),
    )
