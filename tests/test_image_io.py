import dataclasses
import io

import cv2
import numpy as np
import pytest
from PIL import Image, ImageOps

from print_inspector.core.image_io import (
    ImageLoadError,
    ImageRole,
    apply_exif_orientation,
    jpeg_is_complete,
    load_image,
    make_working_bgr8,
    sniff_format,
)
from tests.conftest import make_label_bgr, write_image


def _asymmetric_bgr(width: int = 60, height: int = 40) -> np.ndarray:
    """Every pixel distinct, so any flip or rotation is detectable."""
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def _pil_bytes(rgb_array: np.ndarray, fmt: str, orientation: int | None = None, **save_kwargs) -> bytes:
    image = Image.fromarray(rgb_array)
    if orientation is not None:
        exif = image.getexif()
        exif[0x0112] = orientation
        save_kwargs["exif"] = exif
    out = io.BytesIO()
    image.save(out, fmt, **save_kwargs)
    return out.getvalue()


def _upright_reference(data: bytes) -> np.ndarray:
    """Pillow's EXIF-transposed result, converted to BGR, as the independent reference."""
    with Image.open(io.BytesIO(data)) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    return rgb[:, :, ::-1]


# ---------------------------------------------------------------- formats and dimensions


@pytest.mark.parametrize("ext", [".png", ".bmp", ".tif", ".tiff"])
def test_lossless_formats_roundtrip_exactly(tmp_path, app_config, ext):
    source = make_label_bgr()
    path = write_image(tmp_path / f"art{ext}", source)
    loaded = load_image(path, ImageRole.ORIGINAL, app_config.image_loading)
    assert loaded.original.shape == source.shape
    assert np.array_equal(loaded.original, source)
    assert loaded.file_format == {".png": "PNG", ".bmp": "BMP", ".tif": "TIFF", ".tiff": "TIFF"}[ext]


@pytest.mark.parametrize("ext", [".jpg", ".jpeg"])
def test_jpeg_loads_for_both_roles(tmp_path, app_config, ext):
    source = make_label_bgr()
    path = write_image(tmp_path / f"img{ext}", source, [cv2.IMWRITE_JPEG_QUALITY, 95])
    for role in ImageRole:
        loaded = load_image(path, role, app_config.image_loading)
        assert loaded.original.shape == source.shape
        assert loaded.file_format == "JPEG"
        assert np.mean(np.abs(loaded.working.astype(int) - source.astype(int))) < 3


def test_dimensions_preserved_for_large_odd_sized_image(tmp_path, app_config):
    source = np.full((3001, 4003, 3), 200, np.uint8)
    path = write_image(tmp_path / "big.png", source)
    loaded = load_image(path, ImageRole.PRINTED, app_config.image_loading)
    assert (loaded.width, loaded.height) == (4003, 3001)
    assert loaded.working.shape == (3001, 4003, 3)


# ---------------------------------------------------------------- internal representation


def test_8bit_bgr_working_shares_memory_with_original(label_png, app_config):
    loaded = load_image(label_png, ImageRole.ORIGINAL, app_config.image_loading)
    assert loaded.working is loaded.original
    assert loaded.alpha is None


def test_arrays_are_read_only(label_png, app_config):
    loaded = load_image(label_png, ImageRole.ORIGINAL, app_config.image_loading)
    with pytest.raises(ValueError):
        loaded.original[0, 0, 0] = 1
    with pytest.raises(ValueError):
        loaded.working[0, 0, 0] = 1


def test_grayscale_8bit(tmp_path, app_config):
    gray = make_label_bgr()[:, :, 1].copy()
    path = write_image(tmp_path / "gray.png", gray)
    loaded = load_image(path, ImageRole.ORIGINAL, app_config.image_loading)
    assert loaded.original.ndim == 2 and loaded.color_mode == "Grayscale"
    assert np.array_equal(loaded.original, gray)
    assert loaded.working.shape == gray.shape + (3,)
    assert all(np.array_equal(loaded.working[:, :, c], gray) for c in range(3))


def test_grayscale_jpeg_stays_single_channel(tmp_path, app_config):
    gray = make_label_bgr()[:, :, 1].copy()
    path = write_image(tmp_path / "gray.jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 95])
    loaded = load_image(path, ImageRole.PRINTED, app_config.image_loading)
    assert loaded.channels == 1
    assert loaded.working.shape == gray.shape + (3,)


def test_16bit_gray_keeps_original_and_scales_working(tmp_path, app_config):
    source = np.linspace(0, 65535, 300 * 200).reshape(200, 300).astype(np.uint16)
    path = write_image(tmp_path / "deep.png", source)
    loaded = load_image(path, ImageRole.ORIGINAL, app_config.image_loading)
    assert loaded.original.dtype == np.uint16 and loaded.bit_depth == 16
    assert np.array_equal(loaded.original, source)
    expected = np.round(source / 257.0).astype(np.uint8)
    assert loaded.working.dtype == np.uint8
    assert np.array_equal(loaded.working[:, :, 0], expected)


def test_16bit_color_tiff(tmp_path, app_config):
    source = (make_label_bgr().astype(np.uint16) * 257)
    path = write_image(tmp_path / "deep.tif", source)
    loaded = load_image(path, ImageRole.ORIGINAL, app_config.image_loading)
    assert loaded.original.dtype == np.uint16
    assert np.array_equal(loaded.original, source)
    assert np.array_equal(loaded.working, make_label_bgr())


def test_float32_tiff(tmp_path, app_config):
    # Single channel: OpenCV stores 3-channel float TIFFs with lossy LogLuv encoding.
    source = np.zeros((20, 30), np.float32)
    source[:, 15:] = 1.0
    source[0, 0] = 0.5
    path = write_image(tmp_path / "float.tif", source)
    loaded = load_image(path, ImageRole.ORIGINAL, app_config.image_loading)
    assert loaded.original.dtype == np.float32 and loaded.bit_depth == 32
    assert np.array_equal(loaded.original, source)
    assert tuple(loaded.working[0, 20]) == (255, 255, 255)
    assert tuple(loaded.working[0, 0]) == (128, 128, 128)


def test_float_above_one_is_normalised_to_peak():
    working, _ = make_working_bgr8(np.array([[0.0, 2.0, 4.0]], np.float32))
    assert list(working[0, :, 0]) == [0, 128, 255]


def test_rgba_png_composited_on_white_with_alpha_kept(tmp_path, app_config):
    source = np.zeros((10, 20, 4), np.uint8)
    source[..., :3] = (50, 100, 200)
    source[:, :5, 3] = 0  # transparent
    source[:, 5:10, 3] = 128  # half
    source[:, 10:, 3] = 255  # opaque
    path = write_image(tmp_path / "logo.png", source)
    loaded = load_image(path, ImageRole.ORIGINAL, app_config.image_loading)
    assert loaded.has_alpha and loaded.color_mode == "RGBA"
    assert np.array_equal(loaded.original, source)
    assert np.array_equal(loaded.alpha, source[:, :, 3])
    assert tuple(loaded.working[0, 0]) == (255, 255, 255)
    assert tuple(loaded.working[0, 15]) == (50, 100, 200)
    half = loaded.working[0, 7].astype(int)
    expected = np.array([50, 100, 200]) * 128 / 255 + 255 * 127 / 255
    assert np.all(np.abs(half - expected) <= 1)


def test_rgba_tiff_with_unassociated_alpha_is_not_darkened(tmp_path, app_config):
    """OpenCV premultiplies such TIFFs; the composite must still match the PNG result."""
    rgba = np.zeros((10, 20, 4), np.uint8)
    rgba[..., :3] = (200, 100, 50)  # RGB order for Pillow
    rgba[..., 3] = 128
    tif = tmp_path / "logo.tif"
    tif.write_bytes(_pil_bytes(rgba, "TIFF"))
    png = tmp_path / "logo.png"
    png.write_bytes(_pil_bytes(rgba, "PNG"))
    from_tif = load_image(tif, ImageRole.ORIGINAL, app_config.image_loading)
    from_png = load_image(png, ImageRole.ORIGINAL, app_config.image_loading)
    assert from_tif.alpha_premultiplied and not from_png.alpha_premultiplied
    diff = np.abs(from_tif.working.astype(int) - from_png.working.astype(int))
    assert diff.max() <= 1


def test_make_working_rejects_unsupported_dtype():
    with pytest.raises(ValueError):
        make_working_bgr8(np.zeros((2, 2), dtype=np.int32))


# ---------------------------------------------------------------- orientation


@pytest.mark.parametrize("orientation", range(1, 9))
def test_png_exif_orientation_matches_pillow(tmp_path, app_config, orientation):
    rgb = _asymmetric_bgr()[:, :, ::-1].copy()
    data = _pil_bytes(rgb, "PNG", orientation)
    path = tmp_path / "o.png"
    path.write_bytes(data)
    loaded = load_image(path, ImageRole.PRINTED, app_config.image_loading)
    assert loaded.exif_orientation == orientation
    assert np.array_equal(loaded.original, _upright_reference(data))


@pytest.mark.parametrize("orientation", range(1, 9))
def test_jpeg_exif_orientation_matches_pillow(tmp_path, app_config, orientation):
    rgb = cv2.resize(make_label_bgr(), (120, 80))[:, :, ::-1].copy()
    data = _pil_bytes(rgb, "JPEG", orientation, quality=95)
    path = tmp_path / "o.jpg"
    path.write_bytes(data)
    loaded = load_image(path, ImageRole.PRINTED, app_config.image_loading)
    reference = _upright_reference(data)
    assert loaded.original.shape == reference.shape
    # Different JPEG decoders differ by a few levels; geometry errors would be far larger.
    assert np.mean(np.abs(loaded.original.astype(int) - reference.astype(int))) < 4


@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
def test_tiff_orientation_applied_exactly_once(tmp_path, app_config, orientation):
    rgb = _asymmetric_bgr()[:, :, ::-1].copy()
    data = _pil_bytes(rgb, "TIFF", orientation)
    path = tmp_path / "o.tif"
    path.write_bytes(data)
    loaded = load_image(path, ImageRole.ORIGINAL, app_config.image_loading)
    assert np.array_equal(loaded.original, _upright_reference(data))


def test_apply_exif_orientation_handles_16bit_rgba():
    data = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
    assert apply_exif_orientation(data, 6).shape == (3, 2, 4)
    assert np.array_equal(apply_exif_orientation(data, 5), data.transpose(1, 0, 2))


# ---------------------------------------------------------------- metadata


def test_dpi_and_physical_size(tmp_path, app_config):
    rgb = np.full((600, 1181, 3), 255, np.uint8)
    path = tmp_path / "art.png"
    path.write_bytes(_pil_bytes(rgb, "PNG", dpi=(300, 300)))
    loaded = load_image(path, ImageRole.ORIGINAL, app_config.image_loading)
    assert loaded.dpi == pytest.approx((300, 300), abs=0.1)
    width_mm, height_mm = loaded.physical_size_mm
    assert width_mm == pytest.approx(100.0, abs=0.1)
    assert height_mm == pytest.approx(50.8, abs=0.1)


def test_no_dpi_means_no_physical_size(label_png, app_config):
    loaded = load_image(label_png, ImageRole.ORIGINAL, app_config.image_loading)
    assert loaded.dpi is None and loaded.physical_size_mm is None


def test_summary_mentions_size_and_type(label_png, app_config):
    loaded = load_image(label_png, ImageRole.ORIGINAL, app_config.image_loading)
    text = loaded.summary()
    assert "600 × 400 px" in text and "PNG" in text and "RGB" in text and "8-bit" in text


def test_non_ascii_path(tmp_path, app_config):
    folder = tmp_path / "Étiquettes_标签"
    folder.mkdir()
    path = write_image(folder / "étiquette.png", make_label_bgr())
    assert load_image(path, ImageRole.ORIGINAL, app_config.image_loading).width == 600


# ---------------------------------------------------------------- invalid and corrupt files


def test_truncated_jpeg_detected(tmp_path, app_config):
    ok, encoded = cv2.imencode(".jpg", make_label_bgr())
    data = encoded.tobytes()
    path = tmp_path / "cut.jpg"
    path.write_bytes(data[: int(len(data) * 0.9)])
    with pytest.raises(ImageLoadError, match="incomplete"):
        load_image(path, ImageRole.PRINTED, app_config.image_loading)


def test_progressive_jpeg_accepted(tmp_path, app_config):
    path = write_image(tmp_path / "prog.jpg", make_label_bgr(), [cv2.IMWRITE_JPEG_PROGRESSIVE, 1])
    assert load_image(path, ImageRole.PRINTED, app_config.image_loading).width == 600


def test_jpeg_with_trailing_data_accepted(tmp_path, app_config):
    """Phones append data (e.g. motion-photo video) after the JPEG end marker."""
    ok, encoded = cv2.imencode(".jpg", make_label_bgr())
    path = tmp_path / "motion.jpg"
    path.write_bytes(encoded.tobytes() + b"\x00\x00\x00\x18ftypmp42" + bytes(5000))
    assert load_image(path, ImageRole.PRINTED, app_config.image_loading).width == 600


def test_jpeg_with_restart_markers_accepted(tmp_path, app_config):
    path = write_image(tmp_path / "rst.jpg", make_label_bgr(), [cv2.IMWRITE_JPEG_RST_INTERVAL, 4])
    assert load_image(path, ImageRole.PRINTED, app_config.image_loading).width == 600


def test_jpeg_is_complete_unit():
    ok, encoded = cv2.imencode(".jpg", make_label_bgr())
    assert jpeg_is_complete(encoded.ravel())
    assert not jpeg_is_complete(encoded.ravel()[:-2])
    assert not jpeg_is_complete(np.frombuffer(b"\xff\xd8\xff", np.uint8))


@pytest.mark.parametrize("ext", [".png", ".bmp", ".tif"])
def test_truncated_lossless_detected(tmp_path, app_config, ext):
    ok, encoded = cv2.imencode(ext, make_label_bgr())
    data = encoded.tobytes()
    path = tmp_path / f"cut{ext}"
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(ImageLoadError):
        load_image(path, ImageRole.ORIGINAL, app_config.image_loading)


def test_heic_renamed_to_jpg_gets_specific_message(tmp_path, app_config):
    path = tmp_path / "IMG_0001.jpg"
    path.write_bytes(b"\x00\x00\x00\x18ftypheic" + bytes(200))
    with pytest.raises(ImageLoadError, match="HEIC"):
        load_image(path, ImageRole.PRINTED, app_config.image_loading)


def test_webp_renamed_rejected(tmp_path, app_config):
    path = tmp_path / "photo.png"
    path.write_bytes(b"RIFF\x00\x00\x00\x00WEBPVP8 " + bytes(200))
    with pytest.raises(ImageLoadError, match="webp"):
        load_image(path, ImageRole.PRINTED, app_config.image_loading)


def test_png_content_with_jpg_extension_loads_with_note(tmp_path, app_config):
    ok, encoded = cv2.imencode(".png", make_label_bgr())
    path = tmp_path / "misnamed.jpg"
    path.write_bytes(encoded.tobytes())
    loaded = load_image(path, ImageRole.PRINTED, app_config.image_loading)
    assert loaded.file_format == "PNG"
    assert any("content is PNG" in note for note in loaded.notes)


def test_bmp_content_disguised_as_png_rejected_for_photo(tmp_path, app_config):
    ok, encoded = cv2.imencode(".bmp", make_label_bgr())
    path = tmp_path / "disguised.png"
    path.write_bytes(encoded.tobytes())
    with pytest.raises(ImageLoadError, match="BMP data"):
        load_image(path, ImageRole.PRINTED, app_config.image_loading)


def test_sniff_format():
    assert sniff_format(np.frombuffer(b"\xff\xd8\xff\xe0" + bytes(12), np.uint8)) == "JPEG"
    assert sniff_format(np.frombuffer(b"MM\x00*" + bytes(12), np.uint8)) == "TIFF"
    assert sniff_format(np.frombuffer(b"hello world 12345", np.uint8)) == "UNKNOWN"


def test_printed_role_rejects_bmp_extension(tmp_path, app_config):
    path = write_image(tmp_path / "photo.bmp", make_label_bgr())
    with pytest.raises(ImageLoadError, match="Unsupported file type"):
        load_image(path, ImageRole.PRINTED, app_config.image_loading)


def test_extension_check_is_case_insensitive(tmp_path, app_config):
    path = write_image(tmp_path / "upper.png", make_label_bgr())
    upper = path.rename(tmp_path / "UPPER.PNG")
    assert load_image(upper, ImageRole.PRINTED, app_config.image_loading).width == 600


def test_missing_file(tmp_path, app_config):
    with pytest.raises(ImageLoadError, match="not found"):
        load_image(tmp_path / "nope.png", ImageRole.ORIGINAL, app_config.image_loading)


def test_empty_file(tmp_path, app_config):
    path = tmp_path / "empty.png"
    path.write_bytes(b"")
    with pytest.raises(ImageLoadError, match="empty"):
        load_image(path, ImageRole.ORIGINAL, app_config.image_loading)


def test_garbage_file(tmp_path, app_config):
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"this is not an image" * 50)
    with pytest.raises(ImageLoadError, match="not a supported image"):
        load_image(path, ImageRole.PRINTED, app_config.image_loading)


def test_corrupt_png_body(tmp_path, app_config):
    ok, encoded = cv2.imencode(".png", make_label_bgr())
    data = bytearray(encoded.tobytes())
    data[100:400] = bytes(300)  # damage compressed data, chunk structure intact
    path = tmp_path / "damaged.png"
    path.write_bytes(bytes(data))
    with pytest.raises(ImageLoadError):
        load_image(path, ImageRole.ORIGINAL, app_config.image_loading)


def test_file_size_limit(label_png, app_config):
    tiny = dataclasses.replace(app_config.image_loading, max_file_size_mb=1e-6)
    with pytest.raises(ImageLoadError, match="limit"):
        load_image(label_png, ImageRole.ORIGINAL, tiny)


def test_pixel_limit_checked_from_header(label_png, app_config):
    tiny = dataclasses.replace(app_config.image_loading, max_pixels=1000)
    with pytest.raises(ImageLoadError, match="MP limit"):
        load_image(label_png, ImageRole.ORIGINAL, tiny)
