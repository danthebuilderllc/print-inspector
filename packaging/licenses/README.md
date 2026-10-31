# Licence texts that the bundled packages do not ship

The PySide6 / shiboken6 wheels (Qt for Python, used under the GNU LGPL v3)
contain no licence text. The LGPL v3 requires that a copy of it, and of the
GNU GPL v3 it refers to, is given to everyone who receives the application.

Put the official, unmodified texts in this folder:

| File           | Official source                              |
|----------------|----------------------------------------------|
| `LGPL-3.0.txt` | https://www.gnu.org/licenses/lgpl-3.0.txt    |
| `GPL-3.0.txt`  | https://www.gnu.org/licenses/gpl-3.0.txt     |

Either download them yourself, or run the build once with
`-FetchLicenseTexts` (packaging/build_windows.ps1), which downloads exactly
these two files on the build computer. Commit them to version control; the
application itself never downloads anything.

`packaging/collect_licenses.py` copies them to `licenses/qt-pyside6/` in the
build. A release build (`-Release`) stops if they are missing.
