"""Copy the licence texts of everything bundled with Print Inspector into the build.

Usage (from the project root, in the build environment):

    python packaging/collect_licenses.py "dist/Print Inspector/licenses"

Writes one folder per bundled Python distribution (the runtime packages of
requirements.txt and everything they depend on) with the licence files that
the package ships, plus Python's own licence, Tesseract's licence files (if
tesseract/ exists), the licence texts kept in packaging/licenses/ (the
GNU LGPL and GPL for Qt / PySide6, whose wheels ship no licence text) and
THIRD_PARTY_NOTICES.md, and an index (INDEX.txt) with name, version and
declared licence of each component. Reads installed package metadata and
local files only; nothing is downloaded.

With --strict the exit code is 1 if a required licence text is missing.
"""
from __future__ import annotations

import re
import shutil
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTRA_TEXTS = ROOT / "packaging" / "licenses"
# Qt for Python is used under the LGPL v3, which refers to the GPL v3: both texts must ship with the app.
REQUIRED_TEXTS = ("LGPL-3.0.txt", "GPL-3.0.txt")
LICENCE_FILE = re.compile(r"(^|/)(LICEN[CS]E|COPYING|NOTICE|AUTHORS|COPYRIGHT)[^/]*$|\.license$", re.IGNORECASE)
# Build- and test-only packages in requirements.txt: not part of the application.
NOT_BUNDLED = {"pytest", "pytest-qt", "pyinstaller", "pyinstaller-hooks-contrib", "pip", "setuptools", "wheel",
               "altgraph", "macholib", "pefile", "pywin32-ctypes", "iniconfig", "pluggy", "colorama"}
QT_DISTRIBUTIONS = {"pyside6", "pyside6-essentials", "pyside6-addons", "shiboken6"}
# Some Qt for Python wheels (e.g. the Windows ones) ship only the text of Qt's commercial licence. Print
# Inspector uses Qt under the LGPL v3, so that text is left out and the LGPL/GPL texts are added instead.
QT_COMMERCIAL = re.compile(r"commercial", re.IGNORECASE)


@dataclass
class Component:
    name: str
    version: str
    licence: str
    homepage: str
    files: list[str]


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def runtime_requirements(requirements: Path) -> list[str]:
    """Package names of requirements.txt, without the test and packaging tools."""
    names = []
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name = _normalise(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0])
            if name not in NOT_BUNDLED:
                names.append(name)
    return names


def _dependencies(dist: metadata.Distribution) -> list[str]:
    names = []
    for requirement in dist.requires or ():
        if "extra ==" in requirement:
            continue  # optional extras are not installed for the application
        marker = requirement.partition(";")[2].strip()
        if marker and ("sys_platform" in marker or "platform_system" in marker or "python_version" in marker):
            try:
                from packaging.markers import Marker  # noqa: PLC0415 - optional, present with pip/PyInstaller

                if not Marker(marker).evaluate():
                    continue
            except Exception:  # noqa: BLE001 - when in doubt, include the dependency
                pass
        names.append(_normalise(re.split(r"[<>=!~\[; (]", requirement.strip(), maxsplit=1)[0]))
    return names


def bundled_distributions(roots: list[str]) -> list[metadata.Distribution]:
    """The root distributions and their installed dependencies (each once)."""
    seen: dict[str, metadata.Distribution] = {}
    queue = list(roots)
    while queue:
        name = queue.pop(0)
        if name in seen or name in NOT_BUNDLED:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        seen[name] = dist
        queue.extend(_dependencies(dist))
    return sorted(seen.values(), key=lambda d: _normalise(d.metadata["Name"]))


def _licence(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    expression = meta.get("License-Expression")
    if expression:
        return expression
    classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or () if c.startswith("License ::")]
    text = (meta.get("License") or "").strip()
    if classifiers:
        return "; ".join(classifiers)
    return text.splitlines()[0][:120] if text else "see licence files"


def _homepage(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    if meta.get("Home-page"):
        return meta["Home-page"]
    for url in meta.get_all("Project-URL") or ():
        label, _, link = url.partition(",")
        if label.strip().lower() in ("homepage", "home", "source", "source code", "repository"):
            return link.strip()
    return ""


def copy_distribution_licences(dist: metadata.Distribution, target: Path) -> Component:
    name, version = dist.metadata["Name"], dist.version
    folder = target / f"{_normalise(name)}-{version}"
    copied = []
    for file in dist.files or ():
        if not LICENCE_FILE.search(str(file).replace("\\", "/")):
            continue
        if _normalise(name) in QT_DISTRIBUTIONS and QT_COMMERCIAL.search(str(file)):
            continue
        source = Path(dist.locate_file(file))
        if source.is_file():
            folder.mkdir(parents=True, exist_ok=True)
            relative = str(file).replace("\\", "/").split(".dist-info/")[-1].replace("/", "__")
            shutil.copy2(source, folder / relative)
            copied.append(relative)
    meta_licence = (dist.metadata.get("License") or "").strip()
    if not copied and meta_licence and len(meta_licence) > 200:  # full text in the metadata only
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "LICENSE-from-metadata.txt").write_text(meta_licence, encoding="utf-8")
        copied.append("LICENSE-from-metadata.txt")
    return Component(name, version, _licence(dist), _homepage(dist), copied)


def collect(target: Path, requirements: Path = ROOT / "requirements.txt") -> list[Component]:
    target.mkdir(parents=True, exist_ok=True)
    distributions = bundled_distributions(runtime_requirements(requirements))
    components = [copy_distribution_licences(d, target) for d in distributions]

    python_licence = next((p for p in (Path(sys.base_prefix) / "LICENSE.txt", Path(sys.base_prefix) / "LICENSE")
                           if p.is_file()), None)
    if python_licence is None:  # POSIX installations keep it in the standard library folder
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        candidate = Path(sys.base_prefix) / "lib" / version / "LICENSE.txt"
        python_licence = candidate if candidate.is_file() else None
    python_files = []
    if python_licence is not None:
        (target / "python").mkdir(exist_ok=True)
        shutil.copy2(python_licence, target / "python" / "LICENSE.txt")
        python_files.append("LICENSE.txt")
    components.insert(0, Component("Python", sys.version.split()[0], "PSF-2.0 (Python Software Foundation License)",
                                   "https://www.python.org", python_files))

    tesseract = ROOT / "tesseract"
    if tesseract.is_dir():
        files = [p for p in tesseract.iterdir() if p.is_file() and LICENCE_FILE.search(p.name)]
        if files:
            (target / "tesseract").mkdir(exist_ok=True)
            for f in files:
                shutil.copy2(f, target / "tesseract" / f.name)
        components.append(Component(
            "Tesseract OCR (and the libraries in its folder)", "bundled",
            "Apache-2.0 (Tesseract); see the files for the libraries", "https://github.com/tesseract-ocr/tesseract",
            [f.name for f in files]))

    qt_files = []
    for name in REQUIRED_TEXTS:
        source = EXTRA_TEXTS / name
        if source.is_file():
            (target / "qt-pyside6").mkdir(exist_ok=True)
            shutil.copy2(source, target / "qt-pyside6" / name)
            qt_files.append(name)
    for c in components:
        if _normalise(c.name) in QT_DISTRIBUTIONS:
            c.files += [f"../qt-pyside6/{n}" for n in qt_files]

    notices = ROOT / "THIRD_PARTY_NOTICES.md"
    if notices.is_file():
        shutil.copy2(notices, target / notices.name)

    lines = ["Print Inspector - bundled third-party components", "=" * 48, "",
             "Each folder holds the licence files shipped by that component. See THIRD_PARTY_NOTICES.md for",
             "the obligations that apply when the application is redistributed.", ""]
    for c in components:
        lines.append(f"{c.name} {c.version}")
        lines.append(f"    Licence:  {c.licence}")
        if c.homepage:
            lines.append(f"    Source:   {c.homepage}")
        lines.append(f"    Files:    {', '.join(c.files) if c.files else 'none shipped - see THIRD_PARTY_NOTICES.md'}")
        lines.append("")
    (target / "INDEX.txt").write_text("\n".join(lines), encoding="utf-8")
    return components


def missing_required_texts() -> list[str]:
    return [n for n in REQUIRED_TEXTS if not (EXTRA_TEXTS / n).is_file()]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    strict = "--strict" in argv
    args = [a for a in argv if a != "--strict"]
    if len(args) != 1:
        print(__doc__)
        return 2
    components = collect(Path(args[0]))
    without = [c.name for c in components if not c.files]
    print(f"Licence files of {len(components)} components written to {args[0]}")
    if without:
        print("No licence file found for: " + ", ".join(without) + " (see THIRD_PARTY_NOTICES.md)")
    missing = missing_required_texts()
    if missing:
        print(f"MISSING licence text(s) in {EXTRA_TEXTS}: {', '.join(missing)} - see packaging/licenses/README.md")
        return 1 if strict else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
