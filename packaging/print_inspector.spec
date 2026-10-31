# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build of Print Inspector: one folder, windowed, fully offline.

Build with packaging/build_windows.ps1 (Windows), or directly:

    pyinstaller --noconfirm --clean --distpath dist --workpath build packaging/print_inspector.spec

Output: dist/Print Inspector/Print Inspector.exe plus the _internal folder.

Contents:
  - Python, the application and its libraries (NumPy, SciPy, OpenCV, scikit-image, Pillow, PySide6, zxing-cpp)
  - the Qt plugins the application needs (platform, styles, image formats), collected by PyInstaller's PySide6 hooks
  - config/default_config.json (the default settings)
  - tesseract/ (program, DLLs, licences and language data) if tools/bundle_tesseract.py created it
  - nothing that communicates over a network: Qt Network, WebEngine and the other unused Qt modules are excluded

There are no model files to bundle: the inspection uses no trained models
except Tesseract's language data, which is inside tesseract/tessdata.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 - SPECPATH is set by PyInstaller
# What is left out of the build (shared with tests/test_deployment.py).
_spec = importlib.util.spec_from_file_location("bundle_rules", ROOT / "packaging" / "bundle_rules.py")
rules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rules)

APP_NAME = "Print Inspector"
VERSION = json.loads((ROOT / "config" / "default_config.json").read_text(encoding="utf-8"))["app"]["version"]
ICON = os.environ.get("PRINT_INSPECTOR_ICON") or str(ROOT / "build" / "icon" / "print_inspector.ico")
TESSERACT = ROOT / "tesseract"

datas = [(str(ROOT / "config" / "default_config.json"), "config")]
if TESSERACT.is_dir():
    # Copied as data: the program is started as a separate process (see core/ocr.py), not linked.
    for path in TESSERACT.rglob("*"):
        if path.is_file():
            datas.append((str(path), str(Path("tesseract") / path.relative_to(TESSERACT).parent)))
    print(f"Print Inspector: bundling Tesseract from {TESSERACT}")
else:
    print("Print Inspector: WARNING - no tesseract/ folder; the application will report text inspection "
          "as not available (REVIEW). Run tools/bundle_tesseract.py first to include OCR.")
if os.environ.get("PRINT_INSPECTOR_REQUIRE_TESSERACT") == "1" and not TESSERACT.is_dir():
    raise SystemExit("tesseract/ is required for this build (PRINT_INSPECTOR_REQUIRE_TESSERACT=1)")

a = Analysis(  # noqa: F821
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=rules.HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=rules.EXCLUDES,
    noarchive=False,
    optimize=0,
)


removed = sorted({entry[0] for entry in a.binaries + a.datas if rules.unwanted(entry[0])})
a.binaries = [entry for entry in a.binaries if not rules.unwanted(entry[0])]
a.datas = [entry for entry in a.datas if not rules.unwanted(entry[0])]
print(f"Print Inspector: left out {len(removed)} unneeded Qt / OpenCV files (network, QML, virtual keyboard, PDF)")

pyz = PYZ(a.pure)  # noqa: F821

version_info = None
if sys.platform == "win32":
    from PyInstaller.utils.win32.versioninfo import (
        FixedFileInfo, StringFileInfo, StringStruct, StringTable, VarFileInfo, VarStruct, VSVersionInfo,
    )

    numbers = tuple(int(n) for n in (VERSION.split(".") + ["0"] * 4)[:4])
    version_info = VSVersionInfo(
        ffi=FixedFileInfo(filevers=numbers, prodvers=numbers),
        kids=[
            StringFileInfo([StringTable("040904B0", [
                StringStruct("ProductName", APP_NAME),
                StringStruct("FileDescription", f"{APP_NAME} - offline label print inspection"),
                StringStruct("FileVersion", VERSION),
                StringStruct("ProductVersion", VERSION),
                StringStruct("InternalName", "PrintInspector"),
                StringStruct("OriginalFilename", f"{APP_NAME}.exe"),
            ])]),
            VarFileInfo([VarStruct("Translation", [0x0409, 1200])]),
        ],
    )

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # compressed executables are often flagged by virus scanners
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON if Path(ICON).is_file() and sys.platform == "win32" else None,
    version=version_info,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
