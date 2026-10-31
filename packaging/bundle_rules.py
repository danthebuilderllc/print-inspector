"""What the PyInstaller build leaves out, and the imports it must add (used by print_inspector.spec).

Everything that could communicate over a network is left out: Qt Network and
the modules and plugins that need it (QML, Quick, the virtual keyboard, the
TUIO touch plugin, TLS and network-information plugins), web engines, and
Python HTTP clients. Tested in tests/test_deployment.py.
"""
from __future__ import annotations

import re

# Qt modules the application does not use. Network-capable ones are excluded on purpose.
QT_EXCLUDES = [
    f"PySide6.{m}" for m in (
        "QtNetwork", "QtNetworkAuth", "QtWebEngineCore", "QtWebEngineWidgets", "QtWebEngineQuick", "QtWebChannel",
        "QtWebSockets", "QtWebView", "QtHttpServer", "QtRemoteObjects", "QtBluetooth", "QtNfc", "QtPositioning",
        "QtLocation", "QtSerialPort", "QtSerialBus", "QtSensors", "QtMultimedia", "QtMultimediaWidgets",
        "QtSpatialAudio", "QtTextToSpeech", "QtQml", "QtQuick", "QtQuick3D", "QtQuickControls2", "QtQuickWidgets",
        "Qt3DCore", "Qt3DRender", "Qt3DInput", "Qt3DLogic", "Qt3DAnimation", "Qt3DExtras", "QtCharts",
        "QtDataVisualization", "QtGraphs", "QtSql", "QtTest", "QtDesigner", "QtHelp", "QtUiTools", "QtPdf",
        "QtPdfWidgets", "QtSvg", "QtSvgWidgets", "QtOpenGL", "QtOpenGLWidgets", "QtScxml", "QtStateMachine",
        "QtConcurrent", "QtDBus", "QtAxContainer", "QtXml",
    )
]
EXCLUDES = QT_EXCLUDES + [
    "tkinter", "_tkinter", "matplotlib", "IPython", "jupyter", "notebook", "pytest", "pytestqt", "_pytest",
    "tests", "tools", "pooch", "requests", "urllib3", "httpx", "dask", "pandas", "sympy",
]
# Imports PyInstaller cannot see. scipy._cyutility: shared Cython helpers of SciPy >= 1.16, imported from
# compiled code only. print_inspector.selftest: imported on demand (--self-test).
HIDDEN_IMPORTS = ["print_inspector.selftest", "scipy._cyutility"]


# Qt libraries that must not be in the build (Windows: Qt6Network.dll, macOS: QtNetwork.framework/...):
# networking, and the QML / Quick / virtual keyboard / PDF stack that would bring networking back in.
UNWANTED_QT_LIBRARIES = re.compile(
    r"^(lib)?qt6?(network|networkauth|qml\w*|quick\w*|virtualkeyboard\w*|pdf\w*|websockets|webengine\w*|"
    r"webchannel\w*|webview\w*|httpserver|remoteobjects|bluetooth|nfc|positioning\w*|location|serialport|"
    r"serialbus|sensors\w*|multimedia\w*)(\.framework|\.dll|\.dylib|\.so[.\d]*)?$"
)
# Qt plugin folders the application does not need. "generic" holds the TUIO touch plugin, which listens on a
# UDP port; "platforminputcontexts" the virtual keyboard (needs QML and Network); "tls" and
# "networkinformation" are network plugins.
UNWANTED_QT_PLUGINS = {"generic", "platforminputcontexts", "tls", "networkinformation", "qmltooling", "sqldrivers",
                       "multimedia", "position", "sensors", "webview", "designer", "virtualkeyboard"}


def unwanted(dest: str) -> bool:
    """True for a file (its path inside the build) that must be left out."""
    parts = [p.lower() for p in dest.replace("\\", "/").split("/")]
    name = parts[-1]
    # OpenCV's FFmpeg video plugin (LGPL) is loaded on demand for video files only; the application reads none.
    if name.startswith("opencv_videoio_ffmpeg"):
        return True
    # Qt's PDF image-format plugin needs Qt Pdf (excluded); reports are written with QtGui's QPdfWriter.
    if name.startswith(("qpdf", "libqpdf")):
        return True
    if "plugins" in parts[:-1] and parts[parts.index("plugins") + 1] in UNWANTED_QT_PLUGINS:
        return True
    return any(UNWANTED_QT_LIBRARIES.match(part) for part in parts)
