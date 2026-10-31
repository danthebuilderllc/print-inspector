"""Render the application icon (print_inspector/gui/app_icon.py) into a Windows .ico file.

Usage: python packaging/make_icon.py [output.ico]   (default: build/icon/print_inspector.ico)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_app = None  # kept alive while painting


def main(argv: list[str] | None = None) -> int:
    from PIL import Image
    from PySide6.QtGui import QGuiApplication

    from print_inspector.gui.app_icon import ICON_SIZES, icon_image

    argv = sys.argv[1:] if argv is None else argv
    target = Path(argv[0]) if argv else ROOT / "build" / "icon" / "print_inspector.ico"
    global _app
    _app = QGuiApplication.instance() or QGuiApplication([])  # QPainter needs an application object
    target.parent.mkdir(parents=True, exist_ok=True)
    largest = icon_image(max(ICON_SIZES))
    png = target.with_suffix(".png")
    if not largest.save(str(png)):
        print(f"Could not write {png}")
        return 1
    with Image.open(png) as image:
        image.save(target, format="ICO", sizes=[(s, s) for s in ICON_SIZES])
    print(f"Icon written to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
