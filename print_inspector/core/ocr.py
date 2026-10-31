"""Offline OCR with a locally installed Tesseract.

The ``tesseract`` command-line program is run directly (no Python wrapper):
the image is written to a private temporary folder, Tesseract writes its
word table (TSV) to standard output, and the table is parsed here. Tesseract
reads only local files - its program, its DLLs and the language data
(``<lang>.traineddata``) - and makes no network connections.

Where Tesseract is looked for, in order:

1. ``text.tesseract_path`` from the settings (the program or its folder);
2. a copy bundled with the application: ``tesseract/`` next to the program
   (or in the PyInstaller bundle), see ``tools/bundle_tesseract.py``;
3. the standard Windows install folders (``C:\\Program Files\\Tesseract-OCR``);
4. the ``PATH``.

The language data is taken from ``text.tessdata_dir``, else from the
``tessdata`` folder next to the program, else from Tesseract's own default.
"""
from __future__ import annotations

import csv
import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from print_inspector.config import TextConfig, resource_path

log = logging.getLogger(__name__)

EXECUTABLE_NAME = "tesseract.exe" if sys.platform == "win32" else "tesseract"
BUNDLED_DIR = "tesseract"
TSV_COLUMNS = (
    "level", "page_num", "block_num", "par_num", "line_num", "word_num",
    "left", "top", "width", "height", "conf", "text",
)
_WORD_LEVEL = 5


class OcrUnavailable(RuntimeError):
    """No usable OCR engine: Tesseract or its language data is missing or broken."""


class OcrError(RuntimeError):
    """One OCR call failed."""


@dataclass(frozen=True)
class OcrWord:
    """One recognised word, in the pixel coordinates of the image given to the engine."""

    text: str
    confidence: float  # 0-100; -1 if the engine gives none
    x: int
    y: int
    width: int
    height: int
    block: int = 0
    paragraph: int = 0
    line: int = 0


@dataclass(frozen=True)
class OcrContext:
    """What an OCR call is for (used for logging; test engines use it to know the region)."""

    source: str  # "artwork" or "print"
    mode: str  # "page", "line" or "zone"
    region: tuple[int, int, int, int]  # x, y, width, height on the comparison grid


class OcrEngine(Protocol):
    name: str

    def recognize(self, image: np.ndarray, psm: int, context: OcrContext) -> list[OcrWord]:
        """Words in ``image`` (uint8 gray or BGR). Raises OcrError."""
        ...


def parse_tsv(text: str) -> list[OcrWord]:
    """Word rows of Tesseract's TSV output (other levels and empty words are skipped)."""
    lines = text.splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    if tuple(header[: len(TSV_COLUMNS)]) != TSV_COLUMNS:
        raise OcrError(f"Unexpected OCR output (header {header[:4]}…).")
    words = []
    reader = csv.reader(io.StringIO("\n".join(lines[1:])), delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        if len(row) < len(TSV_COLUMNS) - 1:
            continue
        try:
            level = int(row[0])
            values = [int(v) for v in row[1:10]]
            conf = float(row[10])
        except ValueError:
            continue
        text_value = "\t".join(row[11:]).strip() if len(row) > 11 else ""
        if level != _WORD_LEVEL or not text_value:
            continue
        _, block, par, line, _, left, top, width, height = values
        words.append(OcrWord(text_value, conf, left, top, width, height, block, par, line))
    return words


@dataclass(frozen=True)
class TesseractInstallation:
    executable: Path
    tessdata_dir: Path | None
    version: str
    languages: tuple[str, ...]


def _run(args: list[str], timeout: float) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["OMP_THREAD_LIMIT"] = "1"  # one thread per call; calls run in parallel instead
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0  # no console window flashing
    return subprocess.run(
        args, capture_output=True, timeout=timeout, env=env, creationflags=flags, check=False,
        stdin=subprocess.DEVNULL,
    )


def candidate_executables(cfg: TextConfig) -> list[Path]:
    candidates: list[Path] = []
    if cfg.tesseract_path:
        path = Path(cfg.tesseract_path)
        candidates.append(path / EXECUTABLE_NAME if path.is_dir() else path)
        return candidates  # an explicit setting is not second-guessed
    candidates.append(resource_path(BUNDLED_DIR) / EXECUTABLE_NAME)
    candidates.append(Path(sys.executable).resolve().parent / BUNDLED_DIR / EXECUTABLE_NAME)
    if sys.platform == "win32":
        for var in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
            base = os.environ.get(var)
            if base:
                sub = Path(base) / "Programs" if var == "LOCALAPPDATA" else Path(base)
                candidates.append(sub / "Tesseract-OCR" / EXECUTABLE_NAME)
    found = shutil.which("tesseract")
    if found:
        candidates.append(Path(found))
    unique = []
    for c in candidates:
        if c not in unique:
            unique.append(c)
    return unique


def _tessdata_for(executable: Path, cfg: TextConfig) -> Path | None:
    if cfg.tessdata_dir:
        return Path(cfg.tessdata_dir)
    local = executable.parent / "tessdata"
    return local if local.is_dir() else None


def parse_language_list(output: str) -> tuple[str, ...]:
    """Languages from ``tesseract --list-langs`` (first line is a heading)."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return tuple(line for line in lines[1:] if not line.lower().startswith("list of"))


def inspect_installation(executable: Path, cfg: TextConfig) -> TesseractInstallation:
    """Check one Tesseract program and its language data. Raises OcrUnavailable."""
    if not executable.is_file():
        raise OcrUnavailable(f"Tesseract was not found at {executable}.")
    tessdata = _tessdata_for(executable, cfg)
    try:
        version_run = _run([str(executable), "--version"], 15)
        lang_args = [str(executable), "--list-langs"]
        if tessdata is not None:
            lang_args += ["--tessdata-dir", str(tessdata)]
        lang_run = _run(lang_args, 15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OcrUnavailable(f"Tesseract at {executable} could not be started: {exc}") from exc
    version_text = (version_run.stdout or version_run.stderr).decode("utf-8", "replace").strip()
    if version_run.returncode != 0 or not version_text.lower().startswith("tesseract"):
        raise OcrUnavailable(f"{executable} did not respond like Tesseract.")
    version = version_text.splitlines()[0].strip()
    # Tesseract 4 printed the language list to stderr.
    languages = parse_language_list((lang_run.stdout or lang_run.stderr).decode("utf-8", "replace"))
    missing = [lang for lang in cfg.languages.split("+") if lang not in languages]
    if missing:
        where = tessdata or "Tesseract's default tessdata folder"
        raise OcrUnavailable(
            f"Tesseract language data missing: {', '.join(m + '.traineddata' for m in missing)} "
            f"(looked in {where})."
        )
    return TesseractInstallation(executable, tessdata, version, languages)


def find_tesseract(cfg: TextConfig) -> TesseractInstallation:
    """The first working Tesseract with the configured languages. Raises OcrUnavailable."""
    problems = []
    for candidate in candidate_executables(cfg):
        if not candidate.is_file():
            continue
        try:
            installation = inspect_installation(candidate, cfg)
        except OcrUnavailable as exc:
            problems.append(str(exc))
            continue
        log.info("OCR: %s at %s (languages %s)", installation.version, installation.executable, ", ".join(installation.languages))
        return installation
    if problems:
        raise OcrUnavailable(" ".join(problems))
    raise OcrUnavailable(
        "Tesseract OCR is not installed. Install it (see README, 'Text inspection') "
        "or set its folder in Settings."
    )


class TesseractEngine:
    """Runs the Tesseract program for each image."""

    def __init__(self, installation: TesseractInstallation, cfg: TextConfig) -> None:
        self.installation = installation
        self._cfg = cfg
        self.name = installation.version.replace("tesseract", "Tesseract", 1)

    def recognize(self, image: np.ndarray, psm: int, context: OcrContext) -> list[OcrWord]:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise OcrError("The image could not be prepared for OCR.")
        with tempfile.TemporaryDirectory(prefix="print_inspector_ocr_") as folder:
            path = Path(folder) / "crop.png"
            encoded.tofile(str(path))
            args = [str(self.installation.executable), str(path), "stdout"]
            if self.installation.tessdata_dir is not None:
                args += ["--tessdata-dir", str(self.installation.tessdata_dir)]
            # TSV output is switched on directly rather than with the "tsv" config file, which
            # separately downloaded language folders (e.g. tessdata_best) do not contain.
            args += [
                "-l", self._cfg.languages, "--psm", str(psm), "--oem", "1", "--dpi", "300",
                "-c", "preserve_interword_spaces=1", "-c", "tessedit_create_tsv=1",
            ]
            try:
                result = _run(args, self._cfg.timeout_s)
            except subprocess.TimeoutExpired as exc:
                raise OcrError(f"OCR took longer than {self._cfg.timeout_s:.0f} s ({context.source} {context.mode}).") from exc
            except OSError as exc:
                raise OcrError(f"Tesseract could not be started: {exc}") from exc
        if result.returncode != 0:
            message = result.stderr.decode("utf-8", "replace").strip().splitlines()
            raise OcrError(f"Tesseract failed: {message[-1] if message else f'exit code {result.returncode}'}")
        return parse_tsv(result.stdout.decode("utf-8", "replace"))


def create_engine(cfg: TextConfig) -> TesseractEngine:
    """Find Tesseract and return an engine. Raises OcrUnavailable."""
    return TesseractEngine(find_tesseract(cfg), cfg)
