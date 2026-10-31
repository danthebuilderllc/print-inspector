"""Shared test fixtures. Test images are generated on the fly - no files are downloaded."""
from __future__ import annotations

import os
from pathlib import Path

# Run Qt without a visible display (must be set before a QApplication exists).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":
    # On Windows the offscreen platform reads fonts from Qt's own fonts folder, which the PySide6 wheels do
    # not have: without this, text is not drawn (empty PDF text) and widgets get odd sizes.
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from print_inspector.config import AppConfig, load_config  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_user_dir(tmp_path, monkeypatch):
    """Keep tests away from the real user's config and log folder."""
    user_dir = tmp_path / "localappdata"
    monkeypatch.setenv("LOCALAPPDATA", str(user_dir))
    monkeypatch.setenv("APPDATA", str(user_dir))


@pytest.fixture
def app_config() -> AppConfig:
    return load_config(user_path=None)


def make_label_bgr(width: int = 600, height: int = 400) -> np.ndarray:
    """A synthetic label artwork: white stock, black text, a red box, a fine line."""
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(img, "LOT 12345", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    cv2.rectangle(img, (30, 200), (250, 330), (0, 0, 220), -1)
    cv2.line(img, (300, 250), (570, 250), (0, 0, 0), 1)
    return img


def make_label_photo(width: int = 2400, height: int = 1800, seed: int = 0) -> np.ndarray:
    """A synthetic, well-captured photo: a white label on a grey table with sensor noise.

    The label spans the central 70 % of the frame in both directions.
    """
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), (95, 105, 115), np.uint8)
    lx0, ly0, lx1, ly1 = int(width * 0.15), int(height * 0.15), int(width * 0.85), int(height * 0.85)
    cv2.rectangle(img, (lx0, ly0), (lx1, ly1), (235, 238, 240), -1)
    s = width / 2400
    lines = ["PRINT INSPECTOR", "LOT 12345-A", "EXP 2027/06", "Net wt. 250 g"]
    for i, text in enumerate(lines):
        cv2.putText(
            img,
            text,
            (lx0 + int(60 * s), ly0 + int((150 + i * 170) * s)),
            cv2.FONT_HERSHEY_SIMPLEX,
            (3.2 - 0.4 * i) * s,
            (20, 20, 20),
            max(1, int(6 * s)),
            cv2.LINE_AA,
        )
    cv2.rectangle(
        img, (lx1 - int(500 * s), ly0 + int(80 * s)), (lx1 - int(80 * s), ly0 + int(420 * s)), (40, 40, 200), -1
    )
    for k in range(6):
        y = ly0 + int((520 + k * 40) * s)
        cv2.line(img, (lx1 - int(500 * s), y), (lx1 - int(80 * s), y), (0, 0, 0), max(1, int((2 + k % 2) * s)), cv2.LINE_AA)
    noise = rng.normal(0.0, 2.0, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def write_image(path: Path, image: np.ndarray, params: list[int] | None = None) -> Path:
    """Encode with OpenCV and write via NumPy so non-ASCII paths work on Windows."""
    ok, encoded = cv2.imencode(path.suffix, image, params or [])
    assert ok, f"encoding {path.suffix} failed"
    encoded.tofile(str(path))
    return path


@pytest.fixture
def label_png(tmp_path) -> Path:
    return write_image(tmp_path / "artwork.png", make_label_bgr())


@pytest.fixture
def label_jpg(tmp_path) -> Path:
    return write_image(tmp_path / "photo.jpg", make_label_bgr(), [cv2.IMWRITE_JPEG_QUALITY, 95])


@pytest.fixture
def photo_jpg(tmp_path) -> Path:
    """A realistic label photo saved as JPEG."""
    return write_image(tmp_path / "label_photo.jpg", make_label_photo(), [cv2.IMWRITE_JPEG_QUALITY, 92])
