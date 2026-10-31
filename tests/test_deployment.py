"""Offline deployment: network guard, start-up error handling, command line, build rules, licences, self-test."""
import ast
import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest

import main as launcher
from print_inspector import __version__, offline, startup
from print_inspector.app import Options, parse_options
from print_inspector.config import load_config

ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


rules = _load(ROOT / "packaging" / "bundle_rules.py", "bundle_rules")
licences = _load(ROOT / "packaging" / "collect_licenses.py", "collect_licenses")


@pytest.fixture
def guard():
    offline.uninstall_network_guard()
    offline.clear_blocked_attempts()
    offline.install_network_guard()
    yield
    offline.uninstall_network_guard()
    offline.clear_blocked_attempts()


# ---------------------------------------------------------------- network guard


def test_guard_refuses_outbound_connections_and_name_lookups(guard):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(offline.NetworkBlocked):
            sock.connect(("203.0.113.7", 443))  # TEST-NET-3: never reached, refused before any packet
        with pytest.raises(offline.NetworkBlocked):
            sock.connect_ex(("198.51.100.1", 80))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp, pytest.raises(offline.NetworkBlocked):
        udp.sendto(b"x", ("192.0.2.1", 53))
    with pytest.raises(offline.NetworkBlocked):
        socket.getaddrinfo("example.com", 443)
    with pytest.raises(offline.NetworkBlocked):
        socket.create_connection(("example.com", 80), timeout=1)
    attempts = offline.blocked_attempts()
    assert [a.operation for a in attempts] == ["connect", "connect", "sendto", "name lookup", "name lookup"]
    assert "203.0.113.7" in attempts[0].target and attempts[3].target == "example.com"
    assert isinstance(offline.NetworkBlocked("x"), OSError)  # code handling OSError handles it


def test_guard_allows_loopback(guard):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=5) as client:
            peer, _ = server.accept()
            client.sendall(b"ok")
            assert peer.recv(2) == b"ok"
            peer.close()
    a, b = socket.socketpair()
    a.close()
    b.close()
    assert socket.getaddrinfo("localhost", 80)
    assert offline.blocked_attempts() == []


def test_guard_install_is_idempotent_and_reversible(guard):
    original = offline._originals["connect"]
    offline.install_network_guard()
    assert offline._originals["connect"] is original
    offline.uninstall_network_guard()
    assert socket.socket.connect is original and not offline.guard_installed()


NETWORK_MODULES = {"socket", "ssl", "http", "urllib", "requests", "urllib3", "httpx", "aiohttp", "ftplib", "smtplib",
                   "poplib", "imaplib", "telnetlib", "xmlrpc", "webbrowser", "asyncio", "socketserver"}


def _imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_application_code_contains_no_network_code():
    files = sorted((ROOT / "print_inspector").rglob("*.py")) + [ROOT / "main.py"]
    offenders = {}
    for path in files:
        if path.name == "offline.py":
            continue  # the guard itself
        found = {n for n in _imports(path) if n.split(".")[0] in NETWORK_MODULES or "QtNetwork" in n
                 or "QtWebEngine" in n or n.endswith("QDesktopServices")}
        if found:
            offenders[str(path.relative_to(ROOT))] = sorted(found)
    assert offenders == {}


def test_links_opened_by_the_application_are_local_files():
    """The only clickable link (results saved) points to a local folder."""
    text = (ROOT / "print_inspector" / "gui" / "main_window.py").read_text(encoding="utf-8")
    assert "QUrl.fromLocalFile(str(folder))" in text
    assert "http://" not in text and "https://" not in text


# ---------------------------------------------------------------- start-up errors


@pytest.fixture
def shown(monkeypatch):
    messages = []
    monkeypatch.setattr(startup, "show_native_error", lambda title, text: messages.append((title, text)))
    return messages


def test_startup_error_is_logged_and_shown(shown):
    try:
        raise ImportError("DLL load failed while importing cv2: The specified module could not be found.")
    except ImportError as exc:
        code = startup.report_startup_error(exc)
    assert code == 2
    (title, text), = shown
    assert title == "Print Inspector" and "incomplete or damaged" in text
    log = startup.log_dir() / startup.STARTUP_LOG
    assert str(log) in text and "DLL load failed" in log.read_text(encoding="utf-8")


def test_explanations():
    assert "Reinstall" in startup.explain(ModuleNotFoundError("No module named 'scipy._cyutility'"))
    assert startup.explain(ValueError("bad")).endswith("ValueError: bad")


def test_missing_files_are_reported(tmp_path, shown, monkeypatch):
    assert startup.missing_files(ROOT) == []  # running from source: the defaults are present
    missing = startup.missing_files(tmp_path)
    assert missing == [tmp_path / "config" / "default_config.json"]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "default_config.json").write_text("{}")
    assert startup.missing_files(tmp_path) == [tmp_path / "PySide6" / "plugins" / "platforms" / "qwindows.dll"]
    plugin = tmp_path / "PySide6" / "Qt" / "plugins" / "platforms" / "qwindows.dll"
    plugin.parent.mkdir(parents=True)
    plugin.write_bytes(b"")
    assert startup.missing_files(tmp_path) == []  # either plugin layout is accepted
    assert startup.report_missing_files([plugin]) == 3
    assert "qwindows.dll" in shown[0][1] and "Extract the complete" in shown[0][1]


def test_launcher_reports_missing_files_and_start_errors(monkeypatch, shown):
    monkeypatch.setattr(startup, "missing_files", lambda: [Path("config/default_config.json")])
    assert launcher.run() == 3
    monkeypatch.setattr(startup, "missing_files", lambda: [])
    monkeypatch.setattr(startup, "enable_crash_log", lambda: None)

    def broken_main():
        raise RuntimeError("the configuration folder is not readable")

    import print_inspector.app

    monkeypatch.setattr(print_inspector.app, "main", broken_main)
    assert launcher.run() == 2
    assert "RuntimeError: the configuration folder is not readable" in shown[-1][1]


def test_error_log_falls_back_to_the_temp_folder(monkeypatch, tmp_path):
    blocked = tmp_path / "file-not-folder"
    blocked.write_text("")
    monkeypatch.setattr(startup, "log_dir", lambda: blocked / "logs")
    monkeypatch.setattr(startup.tempfile, "gettempdir", lambda: str(tmp_path / "temp"))
    path = startup.write_error_log("boom")
    assert path == tmp_path / "temp" / "PrintInspector" / startup.STARTUP_LOG and "boom" in path.read_text()


# ---------------------------------------------------------------- command line and version


@pytest.mark.parametrize(
    "argv, expected",
    [
        ([], Options()),
        (["--self-test"], Options(self_test=True)),
        (["--self-test", "--quiet", "--output", "D:/checks"],
         Options(self_test=True, quiet=True, output=Path("D:/checks"))),
        (["--output=C:/x y", "--self-test"], Options(self_test=True, output=Path("C:/x y"))),
        (["--version"], Options(version=True)),
        (["/?"], Options(help=True)),
        (["-platform", "offscreen"], Options()),  # Qt's own options are left to Qt
        (["--output"], Options(error="--output needs a folder")),
    ],
)
def test_parse_options(argv, expected):
    assert parse_options(["Print Inspector.exe", *argv]) == expected


def test_versions_agree():
    assert __version__ == load_config(user_path=None).app.version == "0.12.0"


# ---------------------------------------------------------------- build rules


@pytest.mark.parametrize(
    "dest",
    [
        "PySide6/Qt6Network.dll", "PySide6/Qt6Qml.dll", "PySide6/Qt6QuickControls2.dll", "PySide6/Qt6Pdf.dll",
        "PySide6/Qt6VirtualKeyboard.dll", "PySide6/Qt6WebEngineCore.dll",
        "PySide6/plugins/generic/qtuiotouchplugin.dll", "PySide6/plugins/tls/qschannelbackend.dll",
        "PySide6/plugins/networkinformation/qnetworklistmanager.dll",
        "PySide6/plugins/platforminputcontexts/qtvirtualkeyboardplugin.dll",
        "PySide6/plugins/imageformats/qpdf.dll",
        "PySide6/Qt/lib/QtNetwork.framework/Versions/A/QtNetwork",
        "PySide6/Qt/lib/QtQml.framework/Resources/Info.plist",
        "cv2/opencv_videoio_ffmpeg4100_64.dll",
    ],
)
def test_build_leaves_out_network_and_unused_qt_parts(dest):
    assert rules.unwanted(dest) and rules.unwanted(dest.replace("/", "\\"))


@pytest.mark.parametrize(
    "dest",
    [
        "PySide6/Qt6Core.dll", "PySide6/Qt6Gui.dll", "PySide6/Qt6Widgets.dll", "PySide6/Qt6Svg.dll",
        "PySide6/plugins/platforms/qwindows.dll", "PySide6/plugins/styles/qmodernwindowsstyle.dll",
        "PySide6/plugins/imageformats/qjpeg.dll", "PySide6/plugins/imageformats/qtiff.dll",
        "PySide6/Qt/plugins/platforms/libqcocoa.dylib", "cv2/cv2.pyd", "zxingcpp.cp311-win_amd64.pyd",
        "tesseract/tesseract.exe", "tesseract/tessdata/eng.traineddata", "config/default_config.json",
        "numpy.libs/libopenblas64_.dll", "PySide6/QtGui.pyd", "python311.dll", "PySide6/pyside6.abi3.dll",
    ],
)
def test_build_keeps_what_the_application_needs(dest):
    assert not rules.unwanted(dest)


def test_excluded_python_modules():
    for module in ("PySide6.QtNetwork", "PySide6.QtWebEngineCore", "PySide6.QtQml", "requests", "urllib3",
                   "pytest", "tkinter"):
        assert module in rules.EXCLUDES
    for module in ("PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets"):
        assert module not in rules.EXCLUDES
    assert "print_inspector.selftest" in rules.HIDDEN_IMPORTS


def test_spec_file():
    text = (ROOT / "packaging" / "print_inspector.spec").read_text(encoding="utf-8")
    compile(text, "print_inspector.spec", "exec")
    for needle in ('ROOT / "config" / "default_config.json"', 'ROOT / "tesseract"', "console=False", "upx=False",
                   'APP_NAME = "Print Inspector"', "rules.unwanted", "rules.EXCLUDES", "rules.HIDDEN_IMPORTS"):
        assert needle in text, needle


def test_build_script_runs_the_self_test_and_checks():
    text = (ROOT / "packaging" / "build_windows.ps1").read_text(encoding="utf-8")
    for needle in ("--self-test --quiet --output", "requirements.txt", "tools\\bundle_tesseract.py", "pytest",
                   "packaging\\print_inspector.spec", "collect_licenses.py", "Compress-Archive", "Get-FileHash",
                   "3 11 True", "--no-index", "<Networking>Disable</Networking>"):
        assert needle in text, needle


# ---------------------------------------------------------------- licences and icon


def test_runtime_requirements_exclude_build_and_test_tools():
    names = licences.runtime_requirements(ROOT / "requirements.txt")
    assert {"numpy", "opencv-python-headless", "scikit-image", "pillow", "pyside6", "zxing-cpp"} <= set(names)
    assert not {"pytest", "pytest-qt", "pyinstaller"} & set(names)


def test_licences_are_collected(tmp_path, monkeypatch):
    monkeypatch.setattr(licences, "EXTRA_TEXTS", tmp_path / "texts")
    components = licences.collect(tmp_path / "licenses")
    names = {c.name.lower().replace("_", "-") for c in components}
    assert {"python", "numpy", "opencv-python-headless", "scikit-image", "pyside6", "zxing-cpp", "scipy"} <= names
    index = (tmp_path / "licenses" / "INDEX.txt").read_text(encoding="utf-8")
    assert "zxing-cpp" in index and "Apache" in index
    numpy = next(c for c in components if c.name == "numpy")
    assert numpy.files and (tmp_path / "licenses" / f"numpy-{numpy.version}" / numpy.files[0]).is_file()
    assert (tmp_path / "licenses" / "THIRD_PARTY_NOTICES.md").is_file()
    assert licences.missing_required_texts() == ["LGPL-3.0.txt", "GPL-3.0.txt"]
    assert licences.main(["--strict", str(tmp_path / "again")]) == 1

    (tmp_path / "texts").mkdir()
    for name in licences.REQUIRED_TEXTS:
        (tmp_path / "texts" / name).write_text(f"{name} text")
    components = licences.collect(tmp_path / "with-texts")
    pyside = next(c for c in components if c.name == "PySide6")
    # The LGPL/GPL texts are added whatever the wheel ships; Qt's commercial licence text is never included.
    assert pyside.files[-2:] == ["../qt-pyside6/LGPL-3.0.txt", "../qt-pyside6/GPL-3.0.txt"]
    assert not [f for f in pyside.files if "commercial" in f.lower()]
    assert (tmp_path / "with-texts" / "qt-pyside6" / "GPL-3.0.txt").is_file()
    assert licences.main(["--strict", str(tmp_path / "strict")]) == 0


def test_icon_file(qapp, tmp_path):
    from PIL import Image

    from print_inspector.gui.app_icon import ICON_SIZES, app_icon

    icon_tool = _load(ROOT / "packaging" / "make_icon.py", "make_icon")
    target = tmp_path / "icon" / "print_inspector.ico"
    assert icon_tool.main([str(target)]) == 0
    with Image.open(target) as ico:
        assert {(s, s) for s in ICON_SIZES} <= set(ico.info["sizes"])
    assert len(app_icon().availableSizes()) == len(ICON_SIZES)


# ---------------------------------------------------------------- self-test


def test_self_test_of_the_application(qapp, tmp_path):
    from print_inspector.selftest import REPORT_JSON, REPORT_TEXT, CheckStatus, run_self_test

    offline.uninstall_network_guard()
    offline.clear_blocked_attempts()
    try:
        report = run_self_test(tmp_path / "selftest", timeout_s=240)
    finally:
        offline.uninstall_network_guard()
    statuses = {c.name: c.status for c in report.checks}
    assert report.ok, report.text()
    assert list(statuses) == ["Application startup", "Image loading", "Label alignment", "Defect detection",
                              "Text inspection", "Barcode inspection", "Colour inspection",
                              "PASS / FAIL / REVIEW verdict", "Report export", "Inspection profiles",
                              "Offline operation"]
    ocr = statuses["Text inspection"]
    assert ocr in (CheckStatus.PASS, CheckStatus.SKIPPED)
    assert all(s is CheckStatus.PASS for name, s in statuses.items() if name != "Text inspection")
    if ocr is CheckStatus.SKIPPED:
        assert "PASSED with skipped checks" in report.summary() and report.skipped
    data = json.loads((tmp_path / "selftest" / REPORT_JSON).read_text(encoding="utf-8"))
    assert data["ok"] is True and data["checks"][0]["status"] == "PASS"
    assert "Offline operation" in (tmp_path / "selftest" / REPORT_TEXT).read_text(encoding="utf-8")
    results = list((tmp_path / "selftest" / "results").iterdir())
    assert len(results) == 1 and (results[0] / "report.pdf").is_file()


def test_self_test_reports_a_failed_stage(qapp, tmp_path, monkeypatch):
    """A broken stage gives FAIL (with the reason), and the stages that depend on it are not passed."""
    from print_inspector import selftest

    def broken(folder):
        raise OSError("disk full")

    monkeypatch.setattr(selftest, "make_test_labels", broken)
    try:
        report = selftest.run_self_test(tmp_path / "broken", timeout_s=60)
    finally:
        offline.uninstall_network_guard()
    statuses = {c.name: c.status.value for c in report.checks}
    assert not report.ok and statuses["Application startup"] == "PASS" and statuses["Image loading"] == "FAIL"
    assert "disk full" in report.checks[1].detail
    assert statuses["Report export"] == "FAIL" and "Not run" in report.checks[8].detail
    assert "FAILED" in report.summary()
