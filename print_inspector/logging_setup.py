"""Local logging to a rotating file (and the console when one exists)."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from print_inspector.config import LoggingConfig

LOG_FILE_NAME = "print_inspector.log"
_HANDLER_MARK = "_print_inspector_handler"


def setup_logging(config: LoggingConfig, log_dir: Path) -> Path | None:
    """Configure the root logger. Returns the log file path, or None if file logging failed.

    Safe to call more than once: handlers installed by a previous call are replaced.
    """
    root = logging.getLogger()
    root.setLevel(config.level)
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            root.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter("%(asctime)s %(levelname)-7s [%(name)s] %(message)s")

    # A windowed PyInstaller build has no console: sys.stderr is None there.
    if sys.stderr is not None:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(formatter)
        setattr(console, _HANDLER_MARK, True)
        root.addHandler(console)

    log_file: Path | None = Path(log_dir) / LOG_FILE_NAME
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
    except OSError as exc:
        logging.getLogger(__name__).warning("File logging disabled (%s): %s", log_dir, exc)
        return None
    file_handler.setFormatter(formatter)
    setattr(file_handler, _HANDLER_MARK, True)
    root.addHandler(file_handler)
    return log_file
