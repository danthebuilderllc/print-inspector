"""The Tesseract interface: output parsing, locating the program, running it.

Tests marked ``tesseract`` use a real local Tesseract and are skipped when it
is not installed (see README, "Text inspection"). The others use a small
stand-in program that answers like Tesseract, so the command-line handling is
tested everywhere except Windows (where the stand-in would need to be an .exe).
"""
import dataclasses
import stat
import sys
import textwrap

import cv2
import numpy as np
import pytest

from print_inspector.config import load_config
from print_inspector.core.ocr import (
    EXECUTABLE_NAME,
    OcrContext,
    OcrError,
    OcrUnavailable,
    TesseractEngine,
    candidate_executables,
    create_engine,
    find_tesseract,
    inspect_installation,
    parse_language_list,
    parse_tsv,
)

TSV = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
    "1\t1\t0\t0\t0\t0\t0\t0\t640\t120\t-1\t\n"
    "2\t1\t1\t0\t0\t0\t20\t30\t300\t40\t-1\t\n"
    "4\t1\t1\t1\t1\t0\t20\t30\t300\t40\t-1\t\n"
    "5\t1\t1\t1\t1\t1\t20\t30\t90\t40\t96.512695\tLOT\n"
    "5\t1\t1\t1\t1\t2\t130\t31\t190\t39\t91.3\t12345\n"
    "5\t1\t1\t1\t1\t3\t330\t31\t5\t39\t12\t \n"
    "5\t1\t2\t1\t1\t1\t20\t90\t60\t20\t88\t\"EXP\"\n"
)


@pytest.fixture
def text_cfg():
    return load_config(user_path=None).text


def test_parse_tsv_words_only():
    words = parse_tsv(TSV)
    assert [w.text for w in words] == ["LOT", "12345", '"EXP"']
    lot = words[0]
    assert (lot.x, lot.y, lot.width, lot.height) == (20, 30, 90, 40)
    assert lot.confidence == pytest.approx(96.512695)
    assert (words[2].block, words[2].line) == (2, 1)


def test_parse_tsv_integer_confidence_and_empty_output():
    assert parse_tsv("") == []
    assert parse_tsv(TSV.replace("96.512695", "96"))[0].confidence == 96


def test_parse_tsv_rejects_other_output():
    with pytest.raises(OcrError):
        parse_tsv("Error opening data file eng.traineddata\n")


def test_parse_language_list():
    out = 'List of available languages in "C:\\Program Files\\Tesseract-OCR/tessdata/" (3):\neng\nosd\ndeu\n'
    assert parse_language_list(out) == ("eng", "osd", "deu")


def test_explicit_path_is_the_only_candidate(text_cfg, tmp_path):
    cfg = dataclasses.replace(text_cfg, tesseract_path=str(tmp_path))
    assert candidate_executables(cfg) == [tmp_path / EXECUTABLE_NAME]
    exe = tmp_path / "my-tesseract"
    cfg = dataclasses.replace(text_cfg, tesseract_path=str(exe))
    assert candidate_executables(cfg) == [exe]
    with pytest.raises(OcrUnavailable, match="not installed|not found"):
        find_tesseract(dataclasses.replace(text_cfg, tesseract_path=str(tmp_path / "missing")))


def test_bundled_copy_is_searched(text_cfg):
    candidates = candidate_executables(text_cfg)
    assert any(c.parent.name == "tesseract" for c in candidates)


# ---------------------------------------------------------------- stand-in program

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="the stand-in program is a shell script")


def _stand_in(tmp_path, languages=("eng", "osd"), fail=False):
    """A script that answers --version, --list-langs and recognition like Tesseract does."""
    folder = tmp_path / "Tesseract-OCR"
    (folder / "tessdata").mkdir(parents=True)
    log = tmp_path / "args.log"
    exe = folder / "tesseract"
    langs = "\\n".join(languages)
    exe.write_text(textwrap.dedent(f"""\
        #!/bin/sh
        echo "$@" >> "{log}"
        case "$1" in
          --version) echo "tesseract 5.3.4"; echo " leptonica-1.84.1"; exit 0;;
          --list-langs) printf 'List of available languages in "%s/" (2):\\n{langs}\\n' "$3"; exit 0;;
        esac
        {"echo 'Error: image unreadable' >&2; exit 1" if fail else ""}
        test -f "$1" || exit 2
        printf 'level\\tpage_num\\tblock_num\\tpar_num\\tline_num\\tword_num\\tleft\\ttop\\twidth\\theight\\tconf\\ttext\\n'
        printf '5\\t1\\t1\\t1\\t1\\t1\\t12\\t12\\t50\\t30\\t93.5\\tLOT\\n'
        """))
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return exe, log


@posix_only
def test_stand_in_installation_is_found_and_used(text_cfg, tmp_path):
    exe, log = _stand_in(tmp_path)
    cfg = dataclasses.replace(text_cfg, tesseract_path=str(exe.parent))
    installation = find_tesseract(cfg)
    assert installation.executable == exe and installation.version == "tesseract 5.3.4"
    assert installation.tessdata_dir == exe.parent / "tessdata"
    assert installation.languages == ("eng", "osd")
    engine = create_engine(cfg)
    assert engine.name == "Tesseract 5.3.4"
    words = engine.recognize(np.full((60, 120), 255, np.uint8), 7, OcrContext("print", "line", (0, 0, 10, 10)))
    assert [(w.text, w.confidence) for w in words] == [("LOT", 93.5)]
    call = log.read_text().splitlines()[-1]
    assert "stdout" in call and "--psm 7" in call and "-l eng" in call and "--oem 1" in call
    assert f"--tessdata-dir {exe.parent / 'tessdata'}" in call and "-c tessedit_create_tsv=1" in call


@posix_only
def test_missing_language_data_is_reported(text_cfg, tmp_path):
    exe, _ = _stand_in(tmp_path, languages=("osd",))
    cfg = dataclasses.replace(text_cfg, tesseract_path=str(exe), languages="eng+deu")
    with pytest.raises(OcrUnavailable, match="eng.traineddata, deu.traineddata"):
        inspect_installation(exe, cfg)


@posix_only
def test_recognition_failure_raises_ocr_error(text_cfg, tmp_path):
    exe, _ = _stand_in(tmp_path, fail=True)
    cfg = dataclasses.replace(text_cfg, tesseract_path=str(exe))
    engine = TesseractEngine(find_tesseract(cfg), cfg)
    with pytest.raises(OcrError, match="image unreadable"):
        engine.recognize(np.zeros((10, 10), np.uint8), 7, OcrContext("print", "line", (0, 0, 10, 10)))


@posix_only
def test_not_tesseract_is_rejected(text_cfg, tmp_path):
    fake = tmp_path / "tesseract"
    fake.write_text("#!/bin/sh\necho hello\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    with pytest.raises(OcrUnavailable, match="did not respond like Tesseract"):
        inspect_installation(fake, text_cfg)


# ---------------------------------------------------------------- real Tesseract (if installed)


def _real_engine():
    try:
        return create_engine(load_config(user_path=None).text)
    except OcrUnavailable:
        return None


REAL = _real_engine()
tesseract = pytest.mark.skipif(REAL is None, reason="Tesseract OCR with eng.traineddata is not installed")


@tesseract
def test_real_tesseract_reads_a_line():
    image = np.full((80, 420), 255, np.uint8)
    cv2.putText(image, "LOT 12345", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3, cv2.LINE_AA)
    words = REAL.recognize(image, 7, OcrContext("artwork", "line", (0, 0, 420, 80)))
    assert " ".join(w.text for w in words) == "LOT 12345"
    assert all(w.confidence > 60 for w in words)


@tesseract
def test_real_tesseract_finds_a_missing_digit():
    from print_inspector.core.comparison import compare_to_artwork
    from print_inspector.core.text_inspection import TextFindingKind, inspect_text
    from tests.ocr_fakes import erase, make_text_label
    from tests.test_comparison import full_mask, simulate

    config = load_config(user_path=None)
    label = make_text_label()
    printed = label.image.copy()
    erase(printed, label.char_box(0, 8))
    registered = simulate(printed)
    mask = full_mask(label.image.shape)
    comparison = compare_to_artwork(registered, label.image, mask, config.comparison, 0.9)
    report = inspect_text(label.image, registered, comparison.expected, comparison.combined_score, mask, 12.0,
                          (), REAL, config.text, config.defects)
    assert [line.expected_text for line in report.lines][:3] == label.lines[:3]
    missing = [f for f in report.findings if f.kind is TextFindingKind.MISSING_CHARACTER]
    assert missing and missing[0].expected == "5" and missing[0].reliable


# ---------------------------------------------------------------- bundling for offline packaging


def test_bundle_copies_program_libraries_and_selected_languages(tmp_path):
    from tools.bundle_tesseract import bundle

    source = tmp_path / "Tesseract-OCR"
    (source / "tessdata" / "configs").mkdir(parents=True)
    (source / EXECUTABLE_NAME).write_bytes(b"exe")
    (source / "libtesseract-5.dll").write_bytes(b"dll")
    (source / "LICENSE").write_text("Apache-2.0")
    for lang in ("eng", "deu", "osd"):
        (source / "tessdata" / f"{lang}.traineddata").write_bytes(b"data")
    target = tmp_path / "bundle"
    files = bundle(source, target, ["eng"])
    names = sorted(str(f.relative_to(target)).replace("\\", "/") for f in files)
    assert names == sorted([EXECUTABLE_NAME, "libtesseract-5.dll", "LICENSE", "tessdata/eng.traineddata"])
    with pytest.raises(FileNotFoundError, match="fra.traineddata"):
        bundle(source, target, ["fra"])
