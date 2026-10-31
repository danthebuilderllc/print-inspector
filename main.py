"""Print Inspector launcher.

Run with ``python main.py``. This file is also the PyInstaller entry point:
it checks the installation and reports any start-up error in a message box and
a log file before the application (and Qt) is loaded.
"""
from print_inspector import startup


def run() -> int:
    missing = startup.missing_files()
    if missing:
        return startup.report_missing_files(missing)
    startup.enable_crash_log()
    try:
        from print_inspector.app import main

        return main()
    except Exception as exc:  # noqa: BLE001 - whatever stops the start must reach the operator
        return startup.report_startup_error(exc)


if __name__ == "__main__":
    raise SystemExit(run())
