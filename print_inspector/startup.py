"""Start-up checks and error reporting that work before Qt or any other library is loaded.

Uses the standard library only, so that a broken installation (a missing DLL,
a deleted file, a damaged Qt plugin folder) still produces a readable message
box and a log file instead of a window that flashes and disappears.
"""
from __future__ import annotations

import faulthandler
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

from print_inspector.config import APP_DIR_NAME, APP_NAME, user_data_dir  # standard library only

STARTUP_LOG = "startup_error.log"
CRASH_LOG = "crash.log"

_crash_file = None  # kept open for faulthandler


def is_frozen() -> bool:
    """True when running as the packaged (PyInstaller) application."""
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    """Folder with the bundled resources (``_internal`` in a packaged build, the project root otherwise)."""
    base = getattr(sys, "_MEIPASS", None)
    return Path(base) if base else Path(__file__).resolve().parent.parent


def log_dir() -> Path:
    """The folder of the application log."""
    return user_data_dir() / "logs"


# Qt's platform plugin; the PySide6 wheels keep the plugins in PySide6/plugins (Windows) or PySide6/Qt/plugins.
PLATFORM_PLUGINS = {"win32": "qwindows.dll", "darwin": "libqcocoa.dylib", "linux": "libqxcb.so"}


def required_files() -> list[tuple[Path, ...]]:
    """Files without which the packaged application cannot start, relative to the bundle folder.

    Each entry lists alternative locations; one of them must exist.
    """
    files: list[tuple[Path, ...]] = [(Path("config") / "default_config.json",)]
    plugin = PLATFORM_PLUGINS.get(sys.platform)
    if is_frozen() and plugin:
        files.append(tuple(Path("PySide6", *sub, "plugins", "platforms", plugin) for sub in ((), ("Qt",))))
    return files


def missing_files(base: Path | None = None) -> list[Path]:
    base = bundle_dir() if base is None else base
    return [base / options[0] for options in required_files() if not any((base / f).is_file() for f in options)]


def write_error_log(text: str, name: str = STARTUP_LOG) -> Path | None:
    """Append ``text`` to a log in the user's log folder (or the temp folder). Returns the file, or None."""
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    entry = f"=== {stamp} {APP_NAME} ({sys.executable})\n{text.rstrip()}\n\n"
    for folder in (log_dir(), Path(tempfile.gettempdir()) / APP_DIR_NAME):
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / name
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(entry)
            return path
        except OSError:
            continue
    return None


def show_native_error(title: str, message: str) -> None:
    """A message box without Qt (Windows), or a line on stderr elsewhere / when that fails."""
    if sys.platform == "win32":
        try:
            import ctypes

            MB_ICONERROR, MB_SETFOREGROUND = 0x10, 0x10000
            ctypes.windll.user32.MessageBoxW(None, message, title, MB_ICONERROR | MB_SETFOREGROUND)
            return
        except (AttributeError, OSError):
            pass
    if sys.stderr is not None:
        print(f"{title}: {message}", file=sys.stderr)


def explain(exc: BaseException) -> str:
    """A message for the operator: what went wrong and what to do."""
    if isinstance(exc, ImportError):
        what = (
            f"A component of the application could not be loaded ({exc}).\n\n"
            "The installation is incomplete or damaged, or blocked by security software. "
            "Reinstall Print Inspector by extracting the complete folder again."
        )
        if sys.platform == "win32" and "DLL" in str(exc):
            what += (" If the problem remains, install the Microsoft Visual C++ 2015-2022 "
                     "Redistributable (x64) from the installation media.")
        return what
    return f"Print Inspector could not start:\n{type(exc).__name__}: {exc}"


def report_startup_error(exc: BaseException) -> int:
    """Log and show an error that prevented the start. Returns the process exit code."""
    details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    path = write_error_log(details)
    message = explain(exc)
    if path is not None:
        message += f"\n\nDetails were written to:\n{path}"
    show_native_error(APP_NAME, message)
    return 2


def report_missing_files(missing: list[Path]) -> int:
    listing = "\n".join(str(p) for p in missing)
    path = write_error_log(f"Missing files:\n{listing}")
    message = (
        "The installation is incomplete. These files are missing:\n\n"
        f"{listing}\n\nExtract the complete Print Inspector folder again (do not copy the .exe on its own)."
    )
    if path is not None:
        message += f"\n\nDetails were written to:\n{path}"
    show_native_error(APP_NAME, message)
    return 3


def enable_crash_log() -> Path | None:
    """Write a Python traceback of every thread to ``crash.log`` if native code crashes."""
    global _crash_file
    if _crash_file is not None:
        return Path(_crash_file.name)
    try:
        folder = log_dir()
        folder.mkdir(parents=True, exist_ok=True)
        _crash_file = open(folder / CRASH_LOG, "a", encoding="utf-8")  # noqa: SIM115 - must stay open
        faulthandler.enable(_crash_file)
        return folder / CRASH_LOG
    except (OSError, RuntimeError):
        return None
