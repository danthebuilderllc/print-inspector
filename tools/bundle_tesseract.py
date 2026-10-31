"""Copy a local Tesseract installation into the project for offline packaging.

Creates ``tesseract/`` in the project root with the program, its DLLs, the
licence files and only the requested language data, then checks that the copy
works. Print Inspector uses this copy before any other Tesseract (from source
and, once bundled by PyInstaller, from the packaged application), so the
packaged app needs no separate OCR installation.

Usage (Windows, from the project root, with the virtual environment active):

    python tools/bundle_tesseract.py
    python tools/bundle_tesseract.py --source "C:\\Program Files\\Tesseract-OCR" --languages eng+deu
    python tools/bundle_tesseract.py --tessdata D:\\tessdata_best   (take the language files from elsewhere)

Nothing is downloaded. On macOS/Linux the program's shared libraries live
outside its folder, so a copy made there only works on that machine; bundle
on Windows for the Windows application.
"""
from __future__ import annotations

import argparse
import dataclasses
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from print_inspector.config import load_config  # noqa: E402
from print_inspector.core.ocr import (  # noqa: E402
    BUNDLED_DIR,
    EXECUTABLE_NAME,
    OcrUnavailable,
    candidate_executables,
    inspect_installation,
)

LICENCE_PATTERNS = ("LICENSE*", "COPYING*", "*.license", "doc/*LICENSE*")


def default_source() -> Path | None:
    cfg = dataclasses.replace(load_config(user_path=None).text, tesseract_path="")
    for candidate in candidate_executables(cfg):
        if candidate.is_file() and candidate.parent.name != BUNDLED_DIR:
            return candidate.parent
    return None


def bundle(source: Path, target: Path, languages: list[str], tessdata: Path | None = None) -> list[Path]:
    """Copy program, libraries, licences and language data from ``source`` to ``target``. Returns the files."""
    exe = source / EXECUTABLE_NAME
    if not exe.is_file():
        raise FileNotFoundError(f"{EXECUTABLE_NAME} not found in {source}")
    tessdata = tessdata or source / "tessdata"
    missing = [lang for lang in languages if not (tessdata / f"{lang}.traineddata").is_file()]
    if missing:
        raise FileNotFoundError(f"Language data not found in {tessdata}: {', '.join(m + '.traineddata' for m in missing)}")
    if target.exists():
        shutil.rmtree(target)
    (target / "tessdata").mkdir(parents=True)
    copied = [Path(shutil.copy2(exe, target / exe.name))]
    for library in sorted(source.glob("*.dll")):
        copied.append(Path(shutil.copy2(library, target / library.name)))
    for pattern in LICENCE_PATTERNS:
        for licence in source.glob(pattern):
            if licence.is_file():
                copied.append(Path(shutil.copy2(licence, target / licence.name)))
    for lang in languages:
        copied.append(Path(shutil.copy2(tessdata / f"{lang}.traineddata", target / "tessdata" / f"{lang}.traineddata")))
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, help="Tesseract folder (default: the installed one)")
    parser.add_argument("--tessdata", type=Path, help="folder with the .traineddata files (default: <source>/tessdata)")
    parser.add_argument("--languages", default=None, help="e.g. eng or eng+deu (default: from the settings)")
    parser.add_argument("--target", type=Path, default=Path(__file__).resolve().parent.parent / BUNDLED_DIR)
    args = parser.parse_args(argv)

    cfg = load_config(user_path=None).text
    languages = (args.languages or cfg.languages).split("+")
    source = args.source or default_source()
    if source is None:
        print("No Tesseract installation found. Install it first (see README) or pass --source.")
        return 1
    try:
        files = bundle(source, args.target, languages, args.tessdata)
    except (FileNotFoundError, OSError) as exc:
        print(f"Bundling failed: {exc}")
        return 1
    size = sum(f.stat().st_size for f in files) / 1e6
    print(f"Copied {len(files)} files ({size:.0f} MB) from {source} to {args.target}")
    try:
        check_cfg = dataclasses.replace(cfg, tesseract_path=str(args.target), tessdata_dir="", languages="+".join(languages))
        installation = inspect_installation(args.target / EXECUTABLE_NAME, check_cfg)
    except OcrUnavailable as exc:
        print(f"The copy does not work: {exc}")
        return 1
    print(f"OK: {installation.version}, languages {', '.join(installation.languages)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
