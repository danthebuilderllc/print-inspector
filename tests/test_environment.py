"""Checks that the installed environment matches requirements.txt expectations."""
import sys


def test_python_version():
    assert sys.version_info >= (3, 11)


def test_dependencies_import():
    import cv2
    import numpy
    import PySide6
    import skimage

    assert int(cv2.__version__.split(".")[0]) >= 4
    assert int(numpy.__version__.split(".")[0]) >= 1
    assert PySide6.__version__.startswith("6.")
    assert skimage.__version__


def test_opencv_is_headless_build():
    """The GUI build of OpenCV bundles its own Qt, which can clash with PySide6."""
    import cv2

    info = cv2.getBuildInformation()
    gui_line = next((line for line in info.splitlines() if line.strip().startswith("GUI:")), "")
    assert "QT" not in gui_line.upper(), f"Non-headless OpenCV detected: {gui_line.strip()}"
