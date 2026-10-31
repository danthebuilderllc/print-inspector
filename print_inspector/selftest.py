"""Self-test of the installed application: ``Print Inspector.exe --self-test``.

Runs the real application window (the same code the operator uses, without
showing it) on generated label images with known content and known defects,
and checks every stage: start-up and bundled components, image loading, label
alignment, defect detection, text, barcode and colour inspection, the verdict,
report export and inspection profiles - and that nothing tried to use the
network. Nothing is downloaded and nothing outside the chosen output folder
is changed (the user's settings and profile library are not touched).

A check is PASS only if it ran and gave the expected result. A check that
could not run (for example text inspection when no Tesseract is bundled) is
SKIPPED, never PASS.
"""
from __future__ import annotations

import csv
import json
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from print_inspector import offline, startup
from print_inspector.config import APP_NAME, AppConfig, load_config, resource_path
from print_inspector.core.decision import OperatorMark

REPORT_JSON = "self_test.json"
REPORT_TEXT = "self_test.txt"
ARTWORK_FILE = "selftest_artwork.png"
GOOD_FILE = "selftest_good_print.jpg"
DEFECTIVE_FILE = "selftest_defective_print.jpg"
EAN = "4006381333931"
LOT_TEXT = "LOT 12345-A"
ARTWORK_DPI = 304.8  # 12 px per mm: a 100 x 66.7 mm label
# Areas of the generated artwork (x, y, w, h in artwork pixels, 1200 x 800).
LOT_BOX = (25, 210, 390, 75)  # "LOT 12345-A" spans x 40-388, y 224-276
BARCODE_ORIGIN = (40, 600)
LOGO_CENTRE, LOGO_RADIUS = (980, 330), 110


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    detail: str
    seconds: float = 0.0


@dataclass
class SelfTestReport:
    started: str
    version: str
    frozen: bool
    executable: str
    platform: str
    output_dir: str
    checks: list[CheckResult] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status is CheckStatus.FAIL]

    @property
    def skipped(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status is CheckStatus.SKIPPED]

    @property
    def ok(self) -> bool:
        """No check failed (skipped checks are listed separately; they did not pass)."""
        return bool(self.checks) and not self.failed

    def summary(self) -> str:
        counts = {s: sum(c.status is s for c in self.checks) for s in CheckStatus}
        verdict = "FAILED" if self.failed else ("PASSED with skipped checks" if self.skipped else "PASSED")
        return (f"Self-test {verdict}: {counts[CheckStatus.PASS]} passed, {counts[CheckStatus.FAIL]} failed, "
                f"{counts[CheckStatus.SKIPPED]} skipped")

    def text(self) -> str:
        lines = [
            f"{APP_NAME} {self.version} self-test",
            f"Started:    {self.started}",
            f"Program:    {self.executable} ({'packaged' if self.frozen else 'from source'})",
            f"System:     {self.platform}",
            f"Output:     {self.output_dir}",
            "",
        ]
        for c in self.checks:
            lines.append(f"[{c.status.value:7}] {c.name} ({c.seconds:.1f} s)")
            lines.extend(f"          {line}" for line in c.detail.splitlines())
        lines += ["", f"{self.summary()} in {self.seconds:.0f} s"]
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["checks"] = [dict(asdict(c), status=c.status.value) for c in self.checks]
        data["summary"] = self.summary()
        data["ok"] = self.ok
        return data


class _CheckFailed(Exception):
    pass


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise _CheckFailed(message)


# ---------------------------------------------------------------- test images


@dataclass
class TestLabels:
    artwork: Path
    good: Path
    defective: Path
    defect_box: tuple[int, int, int, int]  # extra mark in the defective print (artwork pixels)
    logo_box: tuple[int, int, int, int]  # recoloured in the defective print


def make_test_labels(folder: Path) -> TestLabels:
    """Artwork with text, a logo, fine lines and a real EAN-13; a good and a defective phone photo."""
    from PIL import Image

    from print_inspector import synthetic as syn
    from print_inspector.core.barcodes import render

    artwork = syn.make_artwork()
    artwork[590:790, 30:680] = 255  # the decorative bars make way for a real barcode
    symbol = render("ean13", EAN, 3)
    symbol = cv2.resize(symbol, (symbol.shape[1], 170), interpolation=cv2.INTER_NEAREST)
    x, y = BARCODE_ORIGIN
    artwork[y : y + symbol.shape[0], x : x + symbol.shape[1]] = np.where(symbol < 128, 20, 255)[:, :, None]
    artwork = np.minimum(artwork, 236).astype(np.uint8)  # label stock, not pure white

    printed, truth = syn.add_defects(artwork, ["extra_mark"], seed=3)
    cx, cy = LOGO_CENTRE
    ring = np.zeros(artwork.shape[:2], np.uint8)
    cv2.circle(ring, LOGO_CENTRE, LOGO_RADIUS - 4, 255, -1)
    cv2.circle(ring, LOGO_CENTRE, LOGO_RADIUS // 2 + 4, 0, -1)
    printed[ring > 0] = (40, 150, 225)  # red logo printed orange

    folder.mkdir(parents=True, exist_ok=True)
    art_path = folder / ARTWORK_FILE
    Image.fromarray(artwork[:, :, ::-1].copy()).save(art_path, dpi=(ARTWORK_DPI, ARTWORK_DPI))
    paths = []
    for name, image, seed in ((GOOD_FILE, artwork, 1), (DEFECTIVE_FILE, printed, 2)):
        photo = syn.make_photo(image, photo_size=(2400, 1800), roll_deg=-8, tilt_x_deg=10, label_fraction=0.75,
                               seed=seed, render=dict(noise_sigma=2.0, jpeg_quality=92))
        path = folder / name
        if not cv2.imwrite(str(path), photo.image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Could not write {path}")
        paths.append(path)
    r = LOGO_RADIUS
    return TestLabels(art_path, paths[0], paths[1], truth[0].bbox, (cx - r, cy - r, 2 * r, 2 * r))


# ---------------------------------------------------------------- driving the window


class _Recorder:
    """Collects the window's result signals so the test can wait for a complete inspection."""

    DONE = ("comparison_done", "text_done", "barcodes_done", "color_done")
    FAILED = ("image_load_failed", "alignment_failed", "registration_failed", "comparison_failed",
              "text_failed", "barcodes_failed", "color_failed")

    def __init__(self, window) -> None:
        self.events: list[tuple[str, object]] = []
        for name in self.DONE + self.FAILED + ("image_loaded", "results_saved", "results_save_failed"):
            getattr(window, name).connect(lambda *args, n=name: self.events.append((n, args)))

    def names(self) -> list[str]:
        return [n for n, _ in self.events]

    def failures(self) -> list[str]:
        return [f"{n}: {args[-1]}" for n, args in self.events if n in self.FAILED or n == "results_save_failed"]


def _wait(condition: Callable[[], bool], timeout_s: float) -> bool:
    from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

    timer = QTimer()
    timer.start(50)  # wakes the loop below regularly
    deadline = time.monotonic() + timeout_s
    try:
        while not condition():
            if time.monotonic() > deadline:
                return False
            QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents
                                           | QEventLoop.ProcessEventsFlag.WaitForMoreEvents, 100)
        return True
    finally:
        timer.stop()


def _inspect(window, recorder: _Recorder, photo: Path, timeout_s: float) -> None:
    """Load the printed photo and wait until every enabled inspection has reported."""
    from print_inspector.core.image_io import ImageRole

    cfg = window.config()
    expected = ["comparison_done"]
    expected += ["text_done"] if cfg.text.enabled else []
    expected += ["barcodes_done"] if cfg.barcodes.enabled else []
    expected += ["color_done"] if cfg.color.enabled else []
    recorder.events.clear()
    window.load_image_file(ImageRole.PRINTED, photo)

    def finished() -> bool:
        names = recorder.names()
        if recorder.failures():
            return True
        return all(e in names for e in expected) and not window.running_stages()

    _expect(_wait(finished, timeout_s), f"The inspection did not finish within {timeout_s:.0f} s "
                                        f"(still running: {', '.join(window.running_stages()) or 'none'})")
    _expect(not recorder.failures(), "; ".join(recorder.failures()))


def _box_hits(window, box: tuple[int, int, int, int], items) -> list:
    """Findings (with x, y, width, height in comparison grid pixels) overlapping an artwork box."""
    sx, sy = window._grid_scale()
    x, y, w, h = box
    x0, y0, x1, y1 = x * sx - 6, y * sy - 6, (x + w) * sx + 6, (y + h) * sy + 6
    return [i for i in items if i.x < x1 and i.x + i.width > x0 and i.y < y1 and i.y + i.height > y0]


# ---------------------------------------------------------------- the checks


class _Run:
    def __init__(self, output_dir: Path, timeout_s: float, config: AppConfig) -> None:
        self.output_dir = output_dir
        self.timeout_s = timeout_s
        self.config = config
        self.window = None
        self.recorder: _Recorder | None = None
        self.labels: TestLabels | None = None
        self.good: dict = {}
        self.defective: dict = {}

    def startup(self) -> str:
        import PySide6
        import skimage
        from PySide6.QtGui import QImageReader
        from PySide6.QtWidgets import QApplication

        from print_inspector.core.barcodes import library_version
        from print_inspector.gui.main_window import MainWindow

        missing = startup.missing_files()
        _expect(not missing, "Missing bundled files: " + ", ".join(str(m) for m in missing))
        _expect(resource_path("config/default_config.json").is_file(), "Default configuration not found")
        app = QApplication.instance()
        _expect(app is not None, "No Qt application")
        formats = {bytes(f).decode() for f in QImageReader.supportedImageFormats()}
        _expect({"png", "jpg", "bmp"} <= formats, f"Qt image format plugins missing (found: {sorted(formats)})")
        self.window = MainWindow(
            self.config, user_config_path=self.output_dir / "settings" / "config.json",
            profiles_dir=self.output_dir / "profiles",
        )
        self.recorder = _Recorder(self.window)
        return (f"Qt {PySide6.__version__} ({app.platformName()} platform), OpenCV {cv2.__version__}, "
                f"NumPy {np.__version__}, scikit-image {skimage.__version__}, {library_version()}, "
                f"Python {platform.python_version()}\nImage formats: {', '.join(sorted(formats))}")

    def image_loading(self) -> str:
        from print_inspector.core.image_io import ImageRole

        self.labels = make_test_labels(self.output_dir / "images")
        self.recorder.events.clear()
        self.window.load_image_file(ImageRole.ORIGINAL, self.labels.artwork)
        _expect(_wait(lambda: "image_loaded" in self.recorder.names() or bool(self.recorder.failures()),
                      self.timeout_s), "The artwork did not load")
        _expect(not self.recorder.failures(), "; ".join(self.recorder.failures()))
        art = self.window.loaded_image(ImageRole.ORIGINAL)
        _expect(art is not None and (art.width, art.height) == (1200, 800), "Artwork loaded with a wrong size")
        self._set_zones()
        start = time.monotonic()
        _inspect(self.window, self.recorder, self.labels.good, self.timeout_s)
        self.good["seconds"] = time.monotonic() - start
        photo = self.window.loaded_image(ImageRole.PRINTED)
        _expect(photo is not None and (photo.width, photo.height) == (2400, 1800), "Photo loaded with a wrong size")
        return (f"Artwork {art.path.name} {art.width} x {art.height} px ({art.file_format}), photo {photo.path.name} "
                f"{photo.width} x {photo.height} px ({photo.file_format}); "
                f"full inspection {self.good['seconds']:.1f} s")

    def _set_zones(self) -> None:
        from print_inspector.core.barcode_zones import BarcodeZone
        from print_inspector.core.text_zones import TextZone

        x, y, w, h = LOT_BOX
        text_zone = TextZone("Lot", x / 1200, y / 800, w / 1200, h / 800, expected_text=LOT_TEXT)
        bx, by = BARCODE_ORIGIN
        barcode_zone = BarcodeZone("EAN", (bx - 10) / 1200, (by - 10) / 800, 340 / 1200, 195 / 800,
                                   format="ean13", expected_value=EAN)
        self.window.apply_text_zones((text_zone,))
        self.window.apply_barcode_zones((barcode_zone,))

    def alignment(self) -> str:
        w = self.window
        alignment, registration = w.alignment_result(), w.registration_result()
        _expect(alignment is not None and alignment.metrics.corners is not None, "The label was not found in the photo")
        _expect(alignment.level.value != "poor", f"Alignment quality {alignment.level.value}: "
                + "; ".join(i.message for i in alignment.issues))
        _expect(registration is not None and registration.registered is not None, "Registration failed")
        _expect(registration.level.value != "poor", f"Registration quality {registration.level.value}: "
                + "; ".join(i.message for i in registration.issues))
        m = registration.metrics
        method = alignment.metrics.method.value if alignment.metrics.method else "?"
        return (f"Label found ({method}), alignment {alignment.level.value}; registration {registration.level.value}, "
                f"residual p95 {m.residual_p95_px:.2f} px, coverage {m.coverage:.0%}")

    def defects_good(self) -> None:
        w = self.window
        self.good.update(defects=w.defect_report(), decision=w.decision(), text=w.text_report(),
                         barcodes=w.barcode_report(), color=w.color_report())

    def defect_detection(self) -> str:
        good = self.good["defects"]
        _expect(good is not None, "No defect report for the good print")
        _expect(not good.defects, f"{len(good.defects)} false defect(s) on the good print")
        start = time.monotonic()
        _inspect(self.window, self.recorder, self.labels.defective, self.timeout_s)
        w = self.window
        self.defective.update(seconds=time.monotonic() - start, defects=w.defect_report(), decision=w.decision(),
                              text=w.text_report(), barcodes=w.barcode_report(), color=w.color_report())
        report = self.defective["defects"]
        hits = _box_hits(w, self.labels.defect_box, report.defects)
        _expect(bool(hits), f"The extra mark at {self.labels.defect_box} was not found "
                f"({len(report.defects)} defect(s) elsewhere)")
        d = hits[0]
        self.defective["mark_id"] = d.id
        return (f"Good print: no defects. Defective print: {len(report.defects)} defect(s); extra mark found as "
                f"#{d.id} {d.category.value}, {d.severity.value}, {d.reliability.value}")

    def text_inspection(self) -> tuple[CheckStatus, str]:
        good = self.good["text"]
        _expect(good is not None, "No text report")
        if not good.available:
            reason = "; ".join(i.message for i in good.issues)
            return CheckStatus.SKIPPED, f"{reason}\nBundle Tesseract (tools/bundle_tesseract.py) to inspect text."
        _expect(not good.findings, "Text findings on the good print: " + "; ".join(f.describe() for f in good.findings))
        read = " / ".join(line.print_text for line in good.lines if line.print_text)
        _expect(LOT_TEXT in read, f"'{LOT_TEXT}' was not read (read: {read})")
        return CheckStatus.PASS, f"{good.engine}: {len(good.lines)} line(s) read, no findings on the good print"

    def barcode_inspection(self) -> str:
        good = self.good["barcodes"]
        _expect(good is not None and good.available, "Barcode inspection not available: "
                + ("; ".join(i.message for i in good.issues) if good else "no report"))
        values = [s.print_value for s in good.symbols]
        _expect(EAN in values, f"EAN-13 {EAN} not decoded (decoded: {values})")
        statuses = {s.status.value for s in good.symbols}
        _expect(statuses == {"ok"} and not good.findings,
                f"Unexpected result on the good print: {statuses}; " + "; ".join(f.describe() for f in good.findings))
        return f"{good.library}: EAN-13 {EAN} decoded and matches the expected value"

    def color_inspection(self) -> str:
        good, bad = self.good["color"], self.defective["color"]
        _expect(good is not None and bad is not None, "No colour report")
        reliable = [f for f in good.findings if f.reliable]
        _expect(not reliable, "Colour findings on the good print: " + "; ".join(f.describe() for f in reliable))
        hits = _box_hits(self.window, self.labels.logo_box, bad.findings)
        _expect(bool(hits), f"The recoloured logo was not found ({len(bad.findings)} colour finding(s) elsewhere)")
        return (f"{good.mode.lower()} mode; good print: no reliable findings; recoloured logo found: "
                f"{hits[0].describe()}")

    def verdict(self) -> str:
        good, bad = self.good["decision"], self.defective["decision"]
        ocr = self.good["text"] is not None and self.good["text"].available
        expected_good = "PASS" if ocr else "REVIEW"  # missing OCR must never pass a label
        _expect(good.status.name == expected_good, f"Good print: {good.status.name}, expected {expected_good} "
                f"({'; '.join(good.reasons)})")
        mark = self.defective["mark_id"]
        if ocr:
            _expect(bad.status.name == "FAIL", f"Defective print: {bad.status.name}, expected FAIL "
                    f"({'; '.join(bad.reasons)})")
        else:  # the defect is shown for confirmation, not passed and not failed without a person
            _expect(bad.status.name == "REVIEW" and mark in bad.review,
                    f"Defective print: {bad.status.name}, expected REVIEW listing #{mark} ({'; '.join(bad.reasons)})")
        # The operator confirms the extra mark: a confirmed defect always fails the label.
        self.window.set_mark(mark, OperatorMark.CONFIRMED)
        confirmed = self.window.decision()
        _expect(confirmed.status.name == "FAIL" and mark in confirmed.failing,
                f"After confirming #{mark}: {confirmed.status.name} ({'; '.join(confirmed.reasons)})")
        if ocr:
            return f"Good print PASS; defective print FAIL; confirming defect #{mark} keeps FAIL"
        return (f"Good print REVIEW and defective print REVIEW (text inspection is required and no OCR is "
                f"available, so no label is passed); after the operator confirms defect #{mark}: FAIL")

    def report_export(self) -> str:
        results = self.output_dir / "results"
        results.mkdir(parents=True, exist_ok=True)
        self.recorder.events.clear()
        self.window.save_results_to(results)
        _expect(_wait(lambda: any(n.startswith("results_save") for n in self.recorder.names()), self.timeout_s),
                "Saving the results did not finish")
        _expect(not self.recorder.failures(), "; ".join(self.recorder.failures()))
        folder = next(args[0] for n, args in self.recorder.events if n == "results_saved")
        names = ("report.pdf", "inspection.json", "findings.csv", "defects.csv", "annotated_comparison.png",
                 "difference_heatmap.png")
        missing = [n for n in names if not (folder / n).is_file()]
        _expect(not missing, f"Missing in {folder}: {', '.join(missing)}")
        _expect((folder / "report.pdf").read_bytes()[:5] == b"%PDF-", "report.pdf is not a PDF file")
        summary = json.loads((folder / "inspection.json").read_text(encoding="utf-8"))
        _expect(summary["report"]["printed_file"] == DEFECTIVE_FILE, "inspection.json names the wrong photo")
        _expect(summary["verdict"]["status"] == "FAIL", f"inspection.json verdict {summary['verdict']['status']}")
        with open(folder / "findings.csv", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        confirmed = [r for r in rows if r["operator"] == "confirmed"]
        _expect([r["id"] for r in confirmed] == [f"#{self.defective['mark_id']}"],
                f"findings.csv operator decisions: {[(r['id'], r['operator']) for r in rows]}")
        for name in ("annotated_comparison.png", "difference_heatmap.png"):
            _expect(cv2.imread(str(folder / name)) is not None, f"{name} cannot be read")
        size = (folder / "report.pdf").stat().st_size / 1e3
        return (f"{folder.name}: report.pdf ({size:.0f} kB), findings.csv ({len(rows)} findings, operator decision "
                f"included), inspection.json (verdict FAIL), annotated comparison and heatmap images")

    def profiles(self) -> str:
        from print_inspector.core import profiles as lib

        w = self.window
        folder = self.output_dir / "profiles"
        profile = w.current_profile("Self-test label", "Created by the self-test")
        path = lib.path_for(folder, profile.name)
        lib.save_profile(path, profile)
        entries = lib.list_profiles(folder)
        _expect([e.name for e in entries] == [profile.name], f"Profile library lists {[e.name for e in entries]}")
        loaded = lib.load_profile(path)
        _expect(loaded == profile, "The reloaded profile differs from the saved one")
        _expect(len(loaded.text_zones) == 1 and loaded.barcode_zones[0].expected_value == EAN,
                "Zones were not stored in the profile")
        self.recorder.events.clear()
        _expect(w.load_inspection_profile(path), "The profile could not be used")
        _expect(_wait(lambda: not w.running_stages(), self.timeout_s), "Inspection with the profile did not finish")
        _expect(w.inspection_profile().name == profile.name and not w.profile_modified(),
                "The window does not use the profile unchanged")
        return f"Saved, listed, reloaded and applied {path.name} ({loaded.summary()})"

    def network(self) -> str:
        attempts = offline.blocked_attempts()
        _expect(offline.guard_installed(), "The network guard is not active")
        _expect(not attempts,
                "Network access was attempted: " + "; ".join(f"{a.operation} {a.target}" for a in attempts))
        loaded = sorted(m for m in sys.modules if m.split(".")[0] in {"requests", "urllib3", "httpx", "aiohttp"}
                        or m in {"PySide6.QtNetwork", "PySide6.QtWebEngineCore", "http.client", "ssl"})
        _expect(not loaded, f"Network modules loaded: {', '.join(loaded)}")
        bundled = []
        if startup.is_frozen():
            base = startup.bundle_dir()
            bundled = [p.name for p in base.rglob("*") if "qt6network" in p.name.lower()
                       or "qtnetwork" in p.name.lower()]
            _expect(not bundled, f"Qt network library bundled: {', '.join(bundled)}")
        return "No network access attempted; no network modules loaded" + (
            "; Qt network library not bundled" if startup.is_frozen() else "")

    def close(self) -> None:
        if self.window is not None:
            self.window._thread_pool.waitForDone(int(self.timeout_s * 1000))
            self.window.close()
            self.window.deleteLater()


CHECKS: tuple[tuple[str, str], ...] = (
    ("Application startup", "startup"),
    ("Image loading", "image_loading"),
    ("Label alignment", "alignment"),
    ("Defect detection", "defect_detection"),
    ("Text inspection", "text_inspection"),
    ("Barcode inspection", "barcode_inspection"),
    ("Colour inspection", "color_inspection"),
    ("PASS / FAIL / REVIEW verdict", "verdict"),
    ("Report export", "report_export"),
    ("Inspection profiles", "profiles"),
    ("Offline operation", "network"),
)
# A check that cannot run without the result of an earlier one.
_NEEDS = {"image_loading": "startup", "alignment": "image_loading", "defect_detection": "alignment",
          "text_inspection": "defect_detection", "barcode_inspection": "defect_detection",
          "color_inspection": "defect_detection", "verdict": "defect_detection",
          "report_export": "defect_detection", "profiles": "image_loading"}


def run_self_test(output_dir: Path, timeout_s: float = 300.0, config: AppConfig | None = None) -> SelfTestReport:
    """Run all checks (a QApplication must exist). Writes self_test.json / .txt to ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or load_config(user_path=None)  # the shipped defaults, not this computer's settings
    offline.install_network_guard()
    report = SelfTestReport(
        started=datetime.now().astimezone().isoformat(timespec="seconds"), version=config.app.version,
        frozen=startup.is_frozen(), executable=sys.executable,
        platform=f"{platform.platform()} ({platform.machine()})", output_dir=str(output_dir),
    )
    run = _Run(output_dir, timeout_s, config)
    passed: set[str] = set()
    start_all = time.monotonic()
    try:
        for name, method in CHECKS:
            needed = _NEEDS.get(method)
            start = time.monotonic()
            if needed and needed not in passed:
                report.checks.append(CheckResult(name, CheckStatus.FAIL, f"Not run: '{needed}' did not succeed"))
                continue
            try:
                if method == "image_loading":
                    detail = run.image_loading()
                    run.defects_good()
                    status = CheckStatus.PASS
                else:
                    result = getattr(run, method)()
                    status, detail = result if isinstance(result, tuple) else (CheckStatus.PASS, result)
            except _CheckFailed as exc:
                status, detail = CheckStatus.FAIL, str(exc)
            except Exception as exc:  # noqa: BLE001 - every error is a failed check with its traceback
                status, detail = CheckStatus.FAIL, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            if status is CheckStatus.PASS:
                passed.add(method)
            elif status is CheckStatus.SKIPPED and method == "text_inspection":
                passed.add(method)
            report.checks.append(CheckResult(name, status, detail, time.monotonic() - start))
    finally:
        run.close()
    report.seconds = time.monotonic() - start_all
    (output_dir / REPORT_JSON).write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / REPORT_TEXT).write_text(report.text(), encoding="utf-8")
    return report
