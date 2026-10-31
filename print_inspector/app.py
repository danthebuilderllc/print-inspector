"""Application start-up: configuration, logging, error handling and the Qt event loop."""
from __future__ import annotations

import logging
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QMessageBox

from print_inspector.config import (
    APP_DIR_NAME,
    APP_NAME,
    AppConfig,
    ConfigError,
    default_user_config_path,
    load_config,
    user_data_dir,
)
from print_inspector import offline
from print_inspector.core.color_calibration import default_profile_path
from print_inspector.gui.app_icon import app_icon
from print_inspector.gui.main_window import MainWindow
from print_inspector.logging_setup import setup_logging

log = logging.getLogger(__name__)


def load_config_with_fallback() -> tuple[AppConfig, str | None]:
    """Load defaults plus the user override; if the override is invalid, use defaults.

    Returns the config and an optional warning for the user. Raises ConfigError
    only if the bundled defaults themselves are broken.
    """
    user_path = default_user_config_path()
    try:
        return load_config(user_path=user_path), None
    except ConfigError as exc:
        config = load_config(user_path=None)
        return config, f"Your settings file was ignored because it is invalid:\n{user_path}\n\n{exc}"


def _set_windows_app_id() -> None:
    """Give the app its own taskbar identity on Windows instead of python.exe's."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(f"{APP_DIR_NAME}.App")
    except (AttributeError, OSError):
        pass


def _install_excepthook() -> None:
    def handle(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        log.critical("Unhandled exception", exc_info=(exc_type, exc, tb))
        if QApplication.instance() is not None:
            details = "".join(traceback.format_exception(exc_type, exc, tb))
            box = QMessageBox(QMessageBox.Icon.Critical, APP_NAME, f"An unexpected error occurred:\n{exc}")
            box.setDetailedText(details)
            box.exec()

    sys.excepthook = handle


USAGE = f"""{APP_NAME}

  "Print Inspector.exe"                      start the application
  "Print Inspector.exe" --self-test          check the installation on generated test labels
        [--output FOLDER]                    where to write the results (default: the user data folder)
        [--quiet]                            no message box at the end (exit code 0 = no check failed)
  "Print Inspector.exe" --version            show the version
"""


@dataclass(frozen=True)
class Options:
    self_test: bool = False
    output: Path | None = None
    quiet: bool = False
    version: bool = False
    help: bool = False
    error: str = ""


def parse_options(argv: list[str]) -> Options:
    """Our own options (Qt's, such as -platform, are left to Qt). A windowed program has no console for argparse."""
    values: dict = {}
    args = list(argv[1:])
    while args:
        arg = args.pop(0)
        if arg == "--self-test":
            values["self_test"] = True
        elif arg == "--quiet":
            values["quiet"] = True
        elif arg == "--version":
            values["version"] = True
        elif arg in ("--help", "-h", "/?"):
            values["help"] = True
        elif arg == "--output" or arg.startswith("--output="):
            value = arg.partition("=")[2] if "=" in arg else (args.pop(0) if args else "")
            if not value:
                return Options(error="--output needs a folder")
            values["output"] = Path(value)
    return Options(**values)


def _tell(text: str, quiet: bool = False, error: bool = False) -> None:
    """Print to the console if there is one; otherwise show a message box (unless quiet)."""
    if sys.stdout is not None:
        print(text, file=sys.stderr if error else sys.stdout)
    elif not quiet:
        (QMessageBox.critical if error else QMessageBox.information)(None, APP_NAME, text)


def run_self_test(options: Options) -> int:
    from print_inspector.selftest import run_self_test as run

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = options.output or user_data_dir() / "self-test" / stamp
    setup_logging(load_config(user_path=None).logging, output)
    log.info("Self-test starting, output %s", output)
    report = run(output)
    log.info(report.summary())
    if sys.stdout is not None:
        print(report.text())
    if not options.quiet:
        box = QMessageBox(QMessageBox.Icon.Information if report.ok else QMessageBox.Icon.Critical, APP_NAME,
                          f"{report.summary()}.\n\nThe full report is in:\n{output}")
        box.setDetailedText(report.text())
        box.exec()
    return 0 if report.ok else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    options = parse_options(argv)
    offline.install_network_guard()  # before anything else could reach out
    _set_windows_app_id()
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_DIR_NAME)
    app.setWindowIcon(app_icon())
    # The same look on every Windows version and theme: the window's colours assume a light palette.
    app.setStyle("Fusion")
    app.setPalette(app.style().standardPalette())

    if options.error or options.help:
        _tell(f"{options.error}\n\n{USAGE}" if options.error else USAGE, options.quiet, bool(options.error))
        return 2 if options.error else 0
    if options.version:
        _tell(f"{APP_NAME} {load_config(user_path=None).app.version}", options.quiet)
        return 0
    if options.self_test:
        return run_self_test(options)

    try:
        config, config_warning = load_config_with_fallback()
    except ConfigError as exc:
        QMessageBox.critical(None, APP_NAME, f"The application configuration is invalid:\n\n{exc}")
        return 1

    log_file = setup_logging(config.logging, user_data_dir() / "logs")
    _install_excepthook()
    log.info("%s %s starting (log file: %s)", config.app.name, config.app.version, log_file)
    log.info("Offline mode: outbound network access is %s", "blocked" if offline.guard_installed() else "NOT blocked")

    # Window size and panel layout are remembered in a local file next to the logs.
    window_settings = QSettings(str(user_data_dir() / "window.ini"), QSettings.Format.IniFormat)
    window = MainWindow(
        config, user_config_path=default_user_config_path(), window_settings=window_settings,
        color_profile_path=Path(config.color.calibration_profile) if config.color.calibration_profile
        else default_profile_path(),
    )
    window.show()
    if config_warning:
        log.warning(config_warning)
        QMessageBox.warning(window, APP_NAME, config_warning)
    return app.exec()
