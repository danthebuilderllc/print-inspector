# Print Inspector

A Windows desktop application that compares original label artwork against a
smartphone photo of the printed label and highlights printing defects.

All processing happens on the local computer. The application makes no network
connections, downloads no models, and sends no telemetry.

## Development status

**Step 12 (current): the offline Windows application.**
`packaging\build_windows.bat` builds **Print Inspector.exe**, a standalone
program that needs no Python and no internet access. It bundles Python, all
libraries, the Qt plugins, the default settings, Tesseract OCR with its language
data, and the licence texts. The build leaves out all network components. The
application blocks outbound network access at start-up, reports start-up
errors in a message box and a log file, and has a built-in **self-test**
(`Print Inspector.exe --self-test`) that checks every inspection stage on
generated labels. The installation, usage and verification guide is
[docs/WINDOWS_DEPLOYMENT.md](docs/WINDOWS_DEPLOYMENT.md); the licence
obligations are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Step 11 added inspection reports and reusable inspection profiles. Save Results now writes a **PDF inspection report**, an **annotated comparison
image**, a **difference heatmap**, a **CSV list of all findings** and the
**JSON inspection data**. **Inspection profiles** store the settings, regions
and zones of a recurring label design in local files; **critical and ignored
regions** make parts of a label stricter or exclude them.

Step 10 added the colour analysis: flat colour areas are compared in
**CIELAB (D50) with ΔE00**, separately from the structural inspection. Without
a **colour calibration** every colour value is marked *approximate*.

| Capability | Status |
|---|---|
| Load original artwork (PNG, JPG, JPEG, BMP, TIF, TIFF) | Done |
| Load printed photo (JPG, JPEG, PNG) | Done |
| EXIF orientation, 16-bit, grayscale, transparency, DPI | Done |
| Detection of corrupt, truncated, mislabelled and HEIC files | Done |
| Full-resolution viewing, mouse-wheel zoom, fit to window, actual pixels, pan | Done |
| Synchronised zoom and pan of the three viewers | Done |
| Defect markers on all viewers (toggle), click a box to select | Done |
| Status area: capture quality, alignment quality, progress, verdict | Done |
| Settings dialog (acceptance criteria, severity limits), saved per user | Done |
| Save results: JSON summary, CSV defect list, images, per-defect evidence | Done |
| Drag and drop of image files, keyboard shortcuts, remembered window layout | Done |
| Photo quality: blur, overexposure, glare, underexposure, uneven lighting, resolution, cut-off label | Done |
| Automatic label detection, perspective/rotation/scale correction, cropping | Done |
| Fallback by matching the artwork when label edges are not visible | Done |
| Manual corner tool with snapping to edges | Done |
| Alignment quality checks; REVIEW for unreliable alignment | Done |
| Precise sub-pixel registration to the artwork, incl. lens-distortion correction | Done |
| Verified registration accuracy, comparison mask, REVIEW when insufficient | Done |
| Alignment overlay (blend / edges / difference / checkerboard, adjustable opacity) | Done |
| Manual registration adjustment | Done |
| Hybrid comparison (SSIM, differences, edges, local contrast, multi-scale) | Done |
| Defect candidates with location, size, preliminary kind and scores | Done |
| Defects: grouping, measurement in mm, category, severity, evidence | Done |
| Detection reliability separate from category certainty | Done |
| Operator marks (confirm / dismiss / needs review), side-by-side evidence | Done |
| PASS / FAIL / REVIEW from configurable acceptance criteria | Done |
| Comparison views (candidates, score, gray / SSIM / edge maps, expected print) | Done |
| PDF inspection report, annotated comparison image, difference heatmap, findings CSV, JSON data | Done |
| Inspection profiles (save / load / import / export, local files) | Done |
| Critical and ignored regions | Done |
| Text inspection with local Tesseract OCR + character image check | Done (tested with a scripted OCR engine; see "Text inspection") |
| Critical text zones, optional exact expected text | Done |
| Barcode / QR inspection (EAN-13, UPC-A, Code 128, QR) with ZXing-C++, damage checks, barcode zones | Done (not ISO grading) |
| Colour analysis: CIELAB D50, ΔE00 heatmaps, local changes, white balance / lighting | Done (approximate without calibration) |
| Camera colour calibration with a reference chart (ColorChecker or own chart data) | Done |
| Inspection capabilities 1–14 from the specification | See the coverage table below |
| Standalone Windows build (PyInstaller), build script, ZIP / installer | Done (build verified on macOS only; see "Verifying Step 12") |
| Self-test of the installed application, start-up error handling, network guard | Done |

### Coverage of the 14 required inspection capabilities

"Visual" means the defect is found as a difference from the artwork, measured,
categorised and judged. "OCR" means the text is also read and compared (Step 8);
"Decoded" means barcodes are decoded and checked (Step 9). "Colour" means the
colour analysis (Step 10): relative within the photo, absolute only when calibrated.

| # | Capability | Status |
|---|---|---|
| 1 | Missing printed text | OCR (missing word / character) + visual |
| 2 | Incorrect characters and numbers | OCR (incorrect character / number), confirmed by the image comparison |
| 3 | Missing punctuation | OCR (missing punctuation) + visual |
| 4 | Broken or incomplete characters | Character ink check (partially printed character) + visual |
| 5 | Missing graphic elements | Visual (tested) |
| 6 | Broken fine lines | Visual (tested) |
| 7 | Distorted logos | Visual: shape difference / missing + extra print (tested) |
| 8 | Extra ink dots or marks | Visual (tested, down to 0.08 mm on synthetic prints) |
| 9 | Missing ink | Visual (tested) |
| 10 | Printing smears | Visual as extra print (not tested specifically) |
| 11 | Unexpected colour differences | Colour: local changes (relative, both modes); deviations from the artwork and global shifts - measured when calibrated, approximate otherwise |
| 12 | Barcode and QR code defects | Decoded and compared; damaged bars / modules, quiet zone, decoding margin (tested). No ISO grading |
| 13 | Printing misregistration | Appears as edge / shape differences; not measured per ink yet |
| 14 | Artwork positioning errors | Not measured yet (registration compensates the position) |

## Project structure

Files marked *(planned)* are not created yet. They will be added in later steps.

```
PrintInspector/                     <- project root (this folder)
├── main.py                         Entry point: python main.py (also the PyInstaller entry)
├── requirements.txt                Pinned dependencies
├── pytest.ini                      Test settings
├── README.md
├── config/
│   └── default_config.json         Default settings (bundled with the app)
├── print_inspector/
│   ├── __init__.py
│   ├── app.py                      Start-up: config, logging, error handling, event loop
│   ├── config.py                   Config loading and validation, path helpers
│   ├── synthetic.py                Synthetic labels/photos with known geometry (tests, demo, self-test)
│   ├── logging_setup.py            Rotating local log file
│   ├── startup.py                  Installation check, start-up error message and log (before Qt loads)
│   ├── offline.py                  Network guard: blocks and logs outbound connections
│   ├── selftest.py                 Self-test of the installed application (--self-test)
│   ├── core/                       Image processing (no GUI code)
│   │   ├── __init__.py
│   │   ├── models.py               InspectionStatus (NOT_INSPECTED / PASS / FAIL / REVIEW)
│   │   ├── image_io.py             Image loading and internal representation
│   │   ├── capture_quality.py      Smartphone photo quality checks
│   │   ├── label_detection.py      Label corners: edge-based detection, artwork matching
│   │   ├── alignment.py            Perspective correction and alignment checks
│   │   ├── registration.py         Precise registration to the artwork, verification, mask
│   │   ├── comparison.py           Hybrid comparison engine, defect candidates
│   │   ├── defects.py              Defect regions, measurements, category, severity, evidence
│   │   ├── decision.py             PASS / FAIL / REVIEW, operator marks
│   │   ├── results_export.py       Save Results: JSON data, CSV lists, images, PDF
│   │   ├── inspection_record.py    One inspection as a whole; all findings in one list
│   │   ├── report_images.py        Annotated comparison image, difference heatmap
│   │   ├── report_pdf.py           PDF inspection report (Qt, offline)
│   │   ├── regions.py              Critical and ignored regions (stored next to the artwork)
│   │   ├── profiles.py             Inspection profiles and the local profile library
│   │   ├── ocr.py                  Tesseract: locating it, running it, reading its output
│   │   ├── text_compare.py         Expected vs. read text: missing words, characters, numbers ...
│   │   ├── text_zones.py           Critical text zones (stored next to the artwork)
│   │   ├── text_inspection.py      Text inspection: OCR + character image check + reliability
│   │   ├── barcodes.py             ZXing-C++ decoding, GTIN helpers
│   │   ├── barcode_zones.py        Barcode zones (stored next to the artwork)
│   │   ├── barcode_inspection.py   Barcode inspection: decode, compare, damage, quiet zone, margin
│   │   ├── color.py                sRGB -> CIELAB D50 (Bradford), CIEDE2000
│   │   ├── color_calibration.py    Reference chart, root-polynomial camera calibration, profiles
│   │   └── color_inspection.py     Colour analysis: areas, local changes, deviations, heatmaps
│   └── gui/
│       ├── __init__.py
│       ├── main_window.py          Main window: toolbars, viewers, side panel, status area
│       ├── image_panel.py          Zoomable full-resolution image viewer, drag and drop
│       ├── view_sync.py            Synchronised zoom and pan between the viewers
│       ├── status_area.py          Bottom strip: quality, alignment, progress, verdict
│       ├── check_section.py        Check details (Checks tab) and metric formatting
│       ├── settings_dialog.py      Settings dialog
│       ├── elided_label.py         One-line label that shortens long text
│       ├── text_panel.py           Text tab: findings, lines read, selected finding
│       ├── barcode_panel.py        Barcodes tab: symbols, findings, selected finding
│       ├── color_panel.py          Colour tab: mode, lighting / white balance, areas, findings
│       ├── calibration_dialog.py   Colour calibration with a chart photo
│       ├── zone_dialog.py          Drawing zones on the artwork (shared by the three dialogs below)
│       ├── text_zone_dialog.py     Critical text zones
│       ├── barcode_zone_dialog.py  Barcode zones
│       ├── region_dialog.py        Critical and ignored regions
│       ├── profile_dialog.py       Profile library: load, save, delete, import, export
│       ├── corner_dialog.py        Manual corner selection
│       ├── registration_dialog.py  Manual registration adjustment
│       ├── overlay.py              Alignment overlay images
│       ├── comparison_view.py      Comparison result images
│       ├── defect_detail.py        Side-by-side defect evidence and operator marks
│       ├── status_indicator.py     Verdict badge
│       ├── app_icon.py             Application icon (drawn in code)
│       └── workers.py              Background work (loading, checks, comparison, saving)
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 Fixtures. Test images are generated, nothing is downloaded.
│   ├── test_environment.py         Checks installed dependencies
│   ├── test_config.py
│   ├── test_image_io.py
│   ├── test_capture_quality.py
│   ├── test_label_detection.py
│   ├── test_alignment.py
│   ├── test_decision.py
│   ├── test_corner_dialog.py
│   ├── test_registration.py
│   ├── test_registration_dialog.py
│   ├── test_overlay.py
│   ├── test_comparison.py
│   ├── test_comparison_view.py
│   ├── test_defects.py
│   ├── test_defect_detail.py
│   ├── test_image_view.py          Wheel zoom, clicks, synchronisation, drag and drop
│   ├── test_status_area.py
│   ├── test_settings_dialog.py
│   ├── test_results_export.py
│   ├── test_text_compare.py
│   ├── test_text_inspection.py
│   ├── test_text_zones.py
│   ├── test_ocr.py                 Tesseract interface; real-OCR tests run when it is installed
│   ├── ocr_fakes.py                Text labels with known characters, scripted OCR engine
│   ├── test_gui_text.py
│   ├── test_barcodes.py            Decoding layer, GTIN helpers, zone files
│   ├── test_barcode_inspection.py  Correct, damaged and incorrect barcodes
│   ├── barcode_fakes.py            Synthetic labels with barcodes
│   ├── test_gui_barcodes.py
│   ├── test_color.py               Colour spaces, CIEDE2000 (Sharma et al. test data)
│   ├── test_color_calibration.py
│   ├── test_color_inspection.py    Local changes, wrong inks, white balance, lighting, calibrated vs approximate
│   ├── color_fakes.py              Colour labels and a simulated phone camera
│   ├── test_gui_color.py
│   ├── test_regions.py             Region files; effect on defects, verdict, barcodes
│   ├── test_profiles.py            Profile contents, validation, applying, library
│   ├── test_report.py              Findings list, CSV, report images, PDF contents
│   ├── test_gui_profiles.py        Regions, profiles and reports in the application
│   ├── test_deployment.py          Network guard, start-up errors, build rules, licences, self-test
│   └── test_gui.py
├── tools/
│   ├── alignment_demo.py           Before/after images and error table on synthetic photos
│   ├── registration_demo.py        Registration accuracy vs. ground truth, detail images
│   ├── comparison_demo.py          Detection performance on synthetic defects
│   └── bundle_tesseract.py         Copies Tesseract + language data into tesseract/ for packaging
├── tesseract/                      (created by bundle_tesseract.py; not in version control)
├── THIRD_PARTY_NOTICES.md          Bundled components, licences, redistribution obligations
├── docs/
│   └── WINDOWS_DEPLOYMENT.md       Installation, usage, clean-PC / offline / network verification, build, troubleshooting
├── packaging/
│   ├── build_windows.ps1           Windows build script (tests, Tesseract, PyInstaller, licences, self-test, ZIP)
│   ├── build_windows.bat           Wrapper: packaging\build_windows.bat [options]
│   ├── print_inspector.spec        PyInstaller configuration (one folder, windowed)
│   ├── bundle_rules.py             What the build leaves out (network, QML, unused Qt), hidden imports
│   ├── collect_licenses.py         Licence texts of all bundled components -> licenses\
│   ├── make_icon.py                Writes the .ico file of the executable
│   ├── installer.iss               Inno Setup script (optional setup program)
│   ├── check_network.ps1           Records network connection attempts of the installed app (Windows audit)
│   └── licenses/                   Licence texts the wheels do not ship (GNU LGPL / GPL for Qt)
├── build/, dist/, .venv-build/     (created by the build; not in version control)
```

## Installation (Windows 10/11)

**Operators** install the built application (ZIP or setup program) without
Python: see [docs/WINDOWS_DEPLOYMENT.md](docs/WINDOWS_DEPLOYMENT.md). The steps
below set up the **development environment** (running from source, tests, building).

You need an internet connection only for this one-time setup. After that, the
application runs fully offline.

1. Install **Python 3.11 (64-bit)** from python.org. During setup, tick "Add python.exe to PATH".
2. Open **Command Prompt** in the project folder, for example `C:\PrintInspector`.
3. Create and activate a virtual environment, then install the dependencies:

   ```bat
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

**Installing on a computer with no internet access:** on a connected Windows PC
with the same Python version, run `pip download -r requirements.txt -d wheels`. Copy
the `wheels` folder to the offline PC, then run
`pip install --no-index --find-links wheels -r requirements.txt`.

### Tesseract OCR (for the text inspection)

The text inspection uses the **Tesseract** program (open source, Apache 2.0),
run locally. No Python package is needed for it.

1. Download the Windows installer of **Tesseract 5** (64-bit) from the
   University of Mannheim builds: <https://github.com/UB-Mannheim/tesseract/wiki>
   (e.g. `tesseract-ocr-w64-setup-5.x.x.exe`). Copy it to the offline PC if
   necessary.
2. Run it with the default folder `C:\Program Files\Tesseract-OCR`. English
   (`eng.traineddata`) is included; tick further languages under
   "Additional language data" if your labels need them (then set
   *Settings → OCR languages*, e.g. `eng+deu`).
3. Check it: `"C:\Program Files\Tesseract-OCR\tesseract.exe" --version`.

Print Inspector finds Tesseract by itself, in this order: the folder set in
*Settings → Tesseract folder*; the copy bundled with the application
(`tesseract\` next to the program, see below); `C:\Program Files\Tesseract-OCR`
(also `Program Files (x86)` and `%LOCALAPPDATA%\Programs`); the `PATH`. The
*Text (OCR)* section of the **Checks** tab shows which engine was used.

Optional, more accurate language data: download `eng.traineddata` from the
`tessdata_best` repository (<https://github.com/tesseract-ocr/tessdata_best>),
put it in a folder of its own and set `text.tessdata_dir` to that folder in
`config.json` (slower, usually more accurate on small text).

**Packaging Tesseract with the application (fully offline, no separate install):**

```bat
python tools\bundle_tesseract.py
```

This copies `tesseract.exe`, its DLLs, its licence and only the configured
language files into `tesseract\` in the project (the tool prints the total size)
and checks that the copy works. The application uses this copy first; the
PyInstaller build includes the folder (`build_windows.bat -TesseractSource …` runs this tool). Options:
`--source <Tesseract folder>`, `--languages eng+deu`, `--tessdata <folder>`.

If Tesseract is missing, the inspection still runs, but the verdict is
**REVIEW** ("Text inspection (OCR) is not available") because the text could
not be checked. To inspect without OCR, untick *Settings → Inspect text with
OCR* or *REVIEW if OCR is not available*.

## Launch

```bat
.venv\Scripts\activate
python main.py
```

### Keyboard shortcuts

Help → Keyboard and Mouse (F1) shows this list in the application.

| Action | Shortcut |
|---|---|
| Load original / printed image | Ctrl+O / Ctrl+Shift+O (or drop the file on viewer 1 / 2) |
| Compare (run or repeat the inspection) | F5 |
| Reset | Ctrl+R |
| Save results | Ctrl+S |
| Settings | Ctrl+, |
| Detect label again / mark corners / adjust registration | Ctrl+D / Ctrl+M / Ctrl+J |
| Zoom in / out | Mouse wheel, or Ctrl++ / Ctrl+- |
| Fit to window / actual pixels | Ctrl+0 / Ctrl+1 |
| Synchronise views on/off | Ctrl+L |
| Defect markers on/off | B |
| Next / previous defect | N / P |
| Confirm / dismiss / needs review / clear mark (selected defect) | C / D / R / U |
| Inspection details panel on/off | F9 |
| Critical text zones | Ctrl+T |
| Barcode zones | Ctrl+Shift+B |
| Colour calibration | Ctrl+K |
| Inspection regions (critical / ignored) | Ctrl+Shift+R |
| Profiles… / Save profile | Ctrl+P / Ctrl+Shift+S |
| Pan | Drag the image; Shift + wheel scrolls sideways |

## Running the tests

```bat
.venv\Scripts\activate
python -m pytest
```

The GUI tests run without opening windows, using Qt's offscreen platform.

## Configuration

The defaults are in `config/default_config.json`. **Settings…** (Ctrl+,) edits
the acceptance criteria, severity limits, grouping/noise limits, the default
sensitivity and the assumed label size; it validates the values and saves only
those that differ from the defaults to
`%LOCALAPPDATA%\PrintInspector\config.json`. Other settings can be added to that
file by hand; include only the values you want to change. For example:

```json
{ "inspection": { "sensitivity": "high" } }
```

If the override file is invalid, the application shows a warning and uses the
defaults. Unknown keys are reported, so a typo is not silently ignored.

Logs are written to `%LOCALAPPDATA%\PrintInspector\logs\print_inspector.log`.
The window size and panel layout are remembered in
`%LOCALAPPDATA%\PrintInspector\window.ini` (delete it to restore the default layout).

## How images are represented

Each loaded image keeps two arrays. Both are read-only, so no processing step
can change them in place.

| Array | Contents |
|---|---|
| `original` | Pixels exactly as stored in the file: native size, bit depth (8/16-bit, float) and channels (gray, BGR, BGRA). Rotated upright by the EXIF orientation tag (pixel values unchanged). |
| `working` | What every processing step uses: 3-channel 8-bit BGR. 16-bit is scaled to 8-bit, gray is expanded to 3 channels, and transparency is composited onto white. For an ordinary 8-bit colour photo this **is** the original array (no copy). |
| `alpha` | Transparency mask, if the file had one. |

The screen preview wraps `working` directly, with no extra copy. Full details are
in the docstring of `print_inspector/core/image_io.py`.

The loader checks file contents, not just the extension. It reports clear errors
for truncated JPEGs (OpenCV would otherwise show them with a grey bottom),
damaged PNG/TIFF/BMP files, iPhone HEIC photos renamed to `.jpg`, and other
formats in disguise.

## Photo quality checks

When a printed photo is loaded, it is checked in the background. Results appear
under **Capture quality** in the right-hand panel and above the photo. The
detected label area is outlined with a dashed cyan box.

| Check | What it measures | Settings in `capture_quality` |
|---|---|---|
| Blur | Edge spread (sigma, px) at full resolution in the most detailed areas | `blur_warning_sigma_px`, `blur_critical_sigma_px` |
| Overexposure | Share of label area at or above `clip_level` | `overexposure_*_fraction` |
| Glare | Largest blown-out spot, as a share of the label area | `glare_*_fraction` |
| Underexposure | 99th-percentile brightness and near-black share | `underexposure_*` |
| Uneven lighting | Variation of the smooth light field across the label (solid ink areas are ignored) | `lighting_*_unevenness` |
| Cut-off label | Printed content running into the photo border | `cutoff_*_ratio`, `border_band_fraction` |
| Resolution | Pixels per mm across the label, reduced for blur | `min_pixels_per_mm_*` |
| Megapixels | Photo size | `min_megapixels_warning` |

- **Warning** means usable, but small defects may be missed. Capture quality shows MARGINAL.
- **Critical** means the photo cannot be inspected reliably. Capture quality shows POOR and the result shows **REVIEW**.

**Label size for the resolution check:** if the artwork file declares a DPI (PNG
pHYs, TIFF or JPEG resolution), the label's real size is taken from it. Otherwise
the label is assumed to be up to 150 mm long. That makes the estimate cautious:
it may warn when a smaller label is in fact fine.

**Calibration:** default thresholds were set on generated test photos, not on
your real captures. Measured blur reads low for strong blur (true 5 px reads as
about 4.2 px), and the thresholds are set on that measured scale. Please
re-check the thresholds with your own photos (see below).

**Tips for good captures:** use a plain background that contrasts with the label,
leave a margin around the label, use even diffuse light, hold the phone parallel
to the label, and tap to focus.

## Verifying Step 2

1. Run `python -m pytest`. All tests should pass.
2. Run `python main.py`, load an artwork, then load:
   - a sharp, well-lit photo: Capture quality GOOD or MARGINAL; result NOT INSPECTED.
   - a deliberately blurred photo (move while shooting): critical blur; result **REVIEW**.
   - a photo with the label cut off at one edge: "reaches the … edge"; **REVIEW**.
   - a photo with flash glare: glare warning or critical.
   - a phone photo taken in portrait: displayed upright.
3. For your typical good photos, note the values listed under Capture quality
   (edge blur, px/mm, light field variation). If good photos get warnings, adjust thresholds in
   `%LOCALAPPDATA%\PrintInspector\config.json`, for example:

   ```json
   { "capture_quality": { "blur_warning_sigma_px": 2.3 } }
   ```

## Label detection and perspective correction

When a printed photo is loaded (and again when the artwork changes), the label
is located and corrected in the background. Use the **Aligned label / Original
photo** switch above the printed view to compare before and after. In the photo
view, the detected outline is drawn in green (reliable) or red (unreliable). The
larger dot marks the label's top-left corner.

**How the corners are found**

1. **Label edges (primary).** The label is separated from the background, its
   outline is reduced to four sides, and each side is refined at full resolution
   by locating the edge along 80 cross-sections and fitting a straight line.
   Corners are where the lines meet, so rounded corners are handled.
2. **Artwork matching (fallback).** If the edges are not clear (for example a
   white label on a white surface), SIFT features of the artwork are matched to
   the photo. This needs the artwork to be loaded.
3. **Manual corners (Ctrl+M).** Click the four corners in any order, drag them
   to correct them, and optionally click **Snap to Label Edges**.

**The aligned image**

- One perspective warp from the original photo pixels corrects perspective,
  rotation and scale and crops the label. The pixels are resampled only once.
  The interpolation is set by `alignment.interpolation` (cubic by default).
- The output has the artwork's proportions and keeps the photo's full resolution
  across the label, up to `max_output_long_side_px`. Its size is an exact
  multiple of the artwork size (shown as "× artwork size").
- Which way is up is decided by comparing all four orientations with the artwork.

**Alignment checks** (thresholds in the `alignment` section of the config)

| Indicator | Meaning |
|---|---|
| Edge found | Share of each side where a consistent edge was located |
| Edge fit | How straight the located edges are (curved or creased labels, wrong outline) |
| Geometry | Corner angles and opposite-side ratio, which show how steep the camera angle was |
| Proportions | Label width/height estimated from the photo alone vs. the artwork |
| Orientation margin | How clearly one orientation beats the others |
| Similarity | Structural match between the aligned label and the artwork |
| Residual | Local misalignment left after correction, measured in a 4×4 grid |
| Artwork fit | Size difference between label and artwork (bleed); over 10% means a wrong outline |

Any critical finding marks the alignment POOR and the result **REVIEW**.
Measured on generated labels, the checks gave the following results. A corner
off by 1% of the label size gives MARGINAL, 2–3% gives POOR. Artwork with bleed
is accepted with a note. A frame printed inside the label is rejected.

**Known limits**

- Rectangular labels are the primary case. Other shapes need the artwork
  fallback or manual corners.
- If the label is tilted in one direction only, its proportions cannot be
  measured exactly from the photo. A typical phone focal length is assumed
  (`assumed_focal_length_factor`), and a mismatch then only produces a warning.
- Without artwork, orientation and accuracy cannot be verified (warning).
- Labels that are not flat are not unwarped. They show up as edge-fit or
  residual warnings.

**Before/after demonstration:** `python tools/alignment_demo.py` writes
comparison images and an error table to `alignment_demo_output/`.

## Verifying Step 3

1. Run `python -m pytest`. All tests should pass.
2. Run `python tools/alignment_demo.py` and look at the images in
   `alignment_demo_output/`. The error table lists the geometric error per case.
3. Run `python main.py`, load an artwork, then photos of the printed label:
   straight on, rotated, tilted, upside down, and on a plain and on a busy
   background. For each, switch between **Aligned label** and **Original photo**
   and read the Alignment section.
4. For a photo where detection fails or is wrong, open **Manual Corners**,
   place or drag the corners, use **Snap to Label Edges**, and apply.

## Precise registration to the artwork

After perspective correction, and whenever artwork is loaded, the photo is
registered to the artwork in the background. The printed panel then shows
**Registered to artwork** by default. Its other views are **Perspective-corrected**
and **Original photo**.

**Comparison grid.** Comparisons use a grid covering the artwork exactly, at
the photo's own resolution. Small text and hairlines keep all the detail the
photo has, and the photo is not enlarged beyond what it resolves.

**How the photo is registered** (each step starts from the original photo pixels)

1. Start from the perspective correction.
2. **Feature matching** (SIFT + MAGSAC++ homography). It is used only if the
   matches are spread over the label. Clustered matches can look very
   confident while being tens of pixels wrong elsewhere.
3. **ECC refinement** on a 3-level pyramid, using all pixels, so it also works
   on artwork with large flat areas. A full perspective model is used only when
   detail covers the label; otherwise an affine model.
4. **Photometric matching.** The artwork is blurred to the photo's sharpness and
   mapped through the camera's tone curve. Without this, edge positions are
   biased by up to 2 px.
5. **Local correction** for lens distortion or a label that is not quite flat:
   sub-pixel tile measurements (phase correlation + Lucas-Kanade) and a smooth
   polynomial field. The field is applied only if it predicts held-out tiles
   better than no correction.
6. **One resampling** of the original photo through the complete model.

**Verification, independent of the estimation.** The remaining misalignment is
measured after the final resampling on a separate set of tiles, including
along the label borders. Blank tiles are skipped. Tiles with lines in one
direction only are measured across the lines, because that is the direction
in which misalignment causes false defects.

| Indicator | Meaning | Settings (`registration`) |
|---|---|---|
| Residual 95% | Remaining misalignment in detailed areas (grid px) | `residual_p95_warning_px` (0.5), `residual_p95_critical_px` (1.0) |
| Verified | Share of detailed areas whose alignment could be measured | `coverage_warning`, `coverage_critical` |
| Similarity | Structural match of print and artwork (wrong artwork) | `min_similarity_*` |
| Compared | Share of the artwork area inside the comparison mask | `min_mask_fraction_*` |
| Local fix | Size of the lens/flatness correction (warning above 3 px) | `local_warning_correction_px` |

Any critical finding marks registration POOR and the result **REVIEW**. A
verified registration supersedes the indirect accuracy estimates of the label
alignment step. Problems that registration cannot repair, such as part of the
label being outside the photo, still give REVIEW.

**Comparison mask.** Only grid pixels taken from inside the photo and inside the
detected label are compared. A 3 px border along the label outline and the
artwork edges is excluded. Bleed in the artwork lies outside the mask.

**Alignment overlay** (third panel). The mode can be Blend, Artwork edges on
photo (magenta), Difference, or Checkerboard; the slider sets opacity, edge
strength or gain. Zoom in on text and hairlines: the magenta edges should sit
exactly on the printed edges. Areas outside the comparison mask are darkened.

**Manual fallback: Adjust Registration (Ctrl+J).** You can move, rotate and scale
the photo with sub-pixel steps (arrow keys: 0.25 px, Shift: 1 px) while watching
a live overlay. Optionally, automatic refinement can continue from your
position. The result is verified like any other registration.

**Accuracy on generated labels** (`python tools/registration_demo.py`; errors near
artwork edges, comparison-grid pixels):

| Case | After perspective correction | After registration (true, 95%) | Reported (95%) |
|---|---|---|---|
| Standard label | 0.46 | 0.02 | 0.04 |
| Small text | 0.77 | 0.01 | 0.03 |
| Detailed graphics | 0.34 | 0.02 | 0.05 |
| Low-feature artwork | 0.93 | 0.37 | 0.10 |
| Lens distortion | 1.13 | 0.24 | 0.11 |
| Lens distortion, local fix disabled | 1.13 | 0.78 | 1.02 → flagged |
| Ink, tone curve and blur differences | 0.45 | 0.05 | 0.06 |

Large print defects (a missing element, a misprinted logo) are excluded from the
intensity-based refinement as outliers. Before this was added, they pulled the
registration by 1–2 px; this was found while developing Step 5.

**Known limits**

- On artwork with very few features, the reported residual can be optimistic.
  It is measured only where detail exists (0.10 px reported vs. 0.37 px true in
  the low-feature case). Misalignment in flat areas cannot cause false defects,
  but edges between large flat areas are only partly verified.
- Results come from generated photos. Real photos have paper texture, halftone
  dots and ink spread. Check the reported residual and the overlay on your own
  labels before relying on the thresholds.
- Labels that are strongly curved (bottles, cans) are outside the smooth local
  correction and will be reported as misaligned.

## Verifying Step 4

1. Run `python -m pytest`. All tests should pass.
2. Run `python tools/registration_demo.py` and look at the table and the detail images.
3. Run `python main.py`, load artwork and a photo, wait for **Registration**, then:
   - In panel 3, choose *Artwork edges on photo*, zoom in on small text and
     hairlines, and check that the magenta edges sit on the printed edges.
   - Read the Registration metrics: residual 95% should be below 0.5 px.
   - Open **Adjust Registration**, untick *Refine automatically*, move the photo
     by 1.5 px and apply. The result must become **REVIEW** with the measured
     offset. Use **Detect Label** to return to automatic registration.

## Comparison engine

After registration, and again when you press **Compare (F5)** or change the
**Sensitivity**, the registered print is compared with the artwork. Panel 3 then
shows the defect candidates, and the list on the right fills. Click a row to zoom
both views onto that candidate.

**1. Expected print.** The artwork is turned into what it should look like in this
photo. It is blurred to the photo's sharpness, each colour channel is mapped
through a fitted tone curve, and it is multiplied by a smooth lighting field (a
low-order polynomial, too smooth to absorb defects). The photo itself is never
smoothed.

**2. Noise and texture model.** Noise is measured per block, separately for tints
(halftone screens are textured) and for solid ink and bare stock (smooth). All
thresholds are multiples of this local noise level (sigma). Halftone and paper
texture therefore raise the threshold only where they occur.

**3. Evidence per pixel**, at full, ½ and ¼ resolution:

| Method | Role |
|---|---|
| Absolute difference (max over colour channels) | Main evidence, minus an edge allowance: jitter × gradient + (jitter²/2 + blur tolerance) × curvature |
| Edge jitter, measured per block and colour channel | How far edges wander in this print (halftone edges, ink spread, JPEG colour bleed), capped at 1.5 px |
| SSIM map | Confirms structural change (all scales) |
| Edge-map comparison (artwork edges missing / extra edges) | Confirms at full resolution |
| Local contrast | Confirms faint, larger changes at coarse scales |
| Multi-scale | Full resolution for specks and breaks; coarse scales for faint larger areas |

A difference counts only if structural evidence confirms it and it exceeds a
minimum absolute difference. **Pixel subtraction alone never decides.**

**4. Candidates.** A candidate must contain at least one pixel above the high
threshold, and it extends down to the low threshold (hysteresis). A small closing
joins fragments, then connected components are measured. Each candidate gets:
location, size, area, score (1.0 = detection threshold), peak in noise units, peak
grey-level difference, darker/lighter, SSIM, missing/extra edge pixels, contrast
ratio, the scale at which it was found, ink coverage and elongation. A
**preliminary kind** is also given: missing line, missing element / region, broken
character, extra ink mark, small dot, shape deviation or light spot.

**Self-checks (REVIEW).** If the print does not resemble the artwork (mean SSIM of
print vs. expected image at the coarsest scale), the noise model would absorb the
differences, so a critical *model mismatch* is reported. Candidates covering more
than 25% of the label are also critical.

**Views in panel 3:** Defect candidates, Combined score, Gray difference,
Structural difference (SSIM), Edge difference (red = missing in print, blue =
extra in print), Expected print, plus the alignment overlays. The slider sets the
display gain.

**Sensitivity:** Low / Medium / High multiply the thresholds by 1.3 / 1 / 0.75
(`sensitivity_*_factor`).

**Large images** are processed in overlapping tiles of 1024 px. The overlap exceeds
every filter, so the result is identical to processing in one piece (tested).

**Detection performance on synthetic prints** (`python tools/comparison_demo.py`;
full pipeline from a perspective photo; 5 conditions × 3 random seeds):

| | Result |
|---|---|
| Planted defects found (8 kinds × 15 prints) | **120 / 120** |
| False candidates on defect-free prints (clean, noise + JPEG, halftone, tone/lighting, paper texture) | **0 in 15 prints** |
| False candidates on prints with defects | 1 in 15 prints |
| Smallest dark speck reliably found | 0.08 mm (1 artwork px) |
| Smallest mid-gray speck reliably found | 0.12 mm (8 of 18 found at 0.08 mm) |

The defect kinds are: missing line segment, missing graphic element, broken
character, extra ink mark, unexpected printed shape, missing printed region,
distorted logo and small dot.

**Known limits**

- These numbers come from generated prints and photos. Real presses, papers and
  cameras will differ. Verify with real good and defective labels before relying
  on them.
- Very fine repeating detail (1-pixel bars on the comparison grid) tolerates only
  about 0.2 px of residual misregistration; at typical photo sampling, 0.3 px.
  Registration normally reaches 0.02–0.05 px there.
- Inside halftone tints, the threshold is higher. A faint defect within a tint
  must stand out from the screen texture.
- Colour accuracy (ΔE) is not evaluated yet: tone curves are fitted per channel,
  so a uniform colour shift of a whole ink is treated as normal.
- The preliminary kinds are heuristic. Classification and the PASS / FAIL decision
  come in the next step.

## Verifying Step 5

1. Run `python -m pytest`. All tests should pass.
2. Run `python tools/comparison_demo.py` (about 3 minutes; `--quick` for one seed)
   and look at the table and the images in `comparison_demo_output/`.
3. Run `python main.py` with artwork and a photo of a **good** label: the candidate
   list should be empty or contain only weak entries (score close to 1).
4. Mark a printed label with a fine pen (a dot, a stroke through a letter), or
   scratch away a little ink, and photograph it. The candidates should appear at
   those places; click them to zoom in.
5. Try the sensitivity settings and the panel 3 views.

## Defects, severity and the verdict

After each comparison, the difference regions become **defects**. They are listed
on the right, numbered most severe first, and drawn as red numbered boxes on the
registered photo and in panel 3 (*Defects* view). Click a defect to zoom both
views onto it and to see the **original and printed regions side by side**, with
the difference and all evidence.

**From candidates to defects** (settings in `defects`)

- **Grouping:** regions closer than `merge_distance_mm` (0.3 mm) are one defect.
  A missing-ink region and an extra-ink region within `shape_pair_distance_mm`
  (1 mm) are one shifted or misshapen element.
- **Noise:** a region is removed only if it is smaller than `noise_max_area_mm2`
  **and** weaker than `noise_max_score` **and** not reliably detected. Removed
  regions are counted in the summary; a small but clear speck is always kept.
- **Size** is measured on the half-maximum core (pixels with at least half the
  peak difference). This recovers the printed defect without the blur halo
  (within 3 px in the tests). The box shows the full detected extent. Millimetres
  come from the artwork DPI; without DPI they are estimated from the assumed label
  size and marked "~".

**Categories.** The category and how certain it is are shown separately from how
reliably the difference was detected:

| Category | Evidence | Certainty |
|---|---|---|
| Missing print | lighter than expected where the artwork has ink | likely if clear |
| Extra print | darker than expected on bare stock (or heavier ink) | likely if clear |
| Shape difference | missing and extra ink side by side | likely |
| Edge difference | thin band along artwork edges | likely if almost all on edges |
| Possible text defect | in a text-like area of the artwork | **never verified** (no OCR) |
| Possible color defect | chroma change much larger than lightness change | **never verified** |
| Unknown visual difference | anything else (e.g. lighter on bare stock) | uncertain |

**Detection reliability:** a defect is *reliable* if its score reaches
`reliable_min_score` (2× the detection threshold), its difference reaches
`reliable_min_difference` grey levels, **and** structural evidence confirms it
(SSIM ≤ `reliable_max_ssim` or edge changes). Otherwise it is *uncertain*.

**Severity:**
- **Major:** the length reaches `severity_major_size_mm` (0.5 mm) or the area
  reaches `severity_major_area_mm2` (0.1 mm²). In text areas, `text_major_size_mm`
  (0.2 mm) is enough.
- **Critical:** the length reaches 2 mm or the area 1 mm².
- **Minor:** anything smaller.
- Differences fainter than `faint_difference` are one level lower.

**Operator marks:** *Confirm defect*, *Dismiss*, *Needs review* and *Clear mark*.
Marks override the automatic assessment of that defect and are reset when the
comparison runs again.

**Verdict** (settings in `decision`):

1. **FAIL:** an operator-confirmed defect of `fail_min_severity` (major) or worse.
2. **REVIEW:** the photo, alignment, registration or comparison is not reliable.
3. **FAIL:** a *reliably detected*, unreviewed defect of major severity or worse,
   or more than `max_minor_defects` (3) reliable/confirmed minor defects.
4. **REVIEW:** an *uncertain* finding of major severity or worse, a defect marked
   *Needs review*, or too many minor findings when some are uncertain.
5. **PASS:** none of the above. Capture warnings (not critical) are allowed when
   `allow_pass_with_warnings` is true.

**Verdict accuracy on generated prints** (`python tools/comparison_demo.py`; 5
photo conditions × 3 seeds; capture-quality check left out because the generated
stock is pure white):

| Print | PASS | FAIL | REVIEW |
|---|---|---|---|
| Defect-free | 15 | 0 | 0 |
| All 8 defect types | 0 | 15 | 0 |
| Missing line / missing element / broken character / extra mark / unexpected shape / missing region / distorted logo (each separately) | 0 | 15 each | 0 |
| One 0.25 mm speck | 15 | 0 | 0 |

The single speck is detected and listed every time, as a *minor* defect. It is
below the 0.5 mm major size, and one minor defect is within the allowed 3. To
reject such specks, lower `severity_major_size_mm` or set `max_minor_defects` to 0.

**Limits**

- The thresholds are starting values chosen for generated images. Set them to
  your own acceptance criteria and check them with real good and defective labels.
- Text defects are *possible* only: the text area is recognised by shape, and the
  characters are not read.
- Colour is judged relative to the rest of the same print. A uniform shift of a
  whole ink, or a colour that matches no standard, is not detected.
- Operator marks are kept only for the current comparison. Saving them with the
  results comes with reporting.

## Verifying Step 6

1. Run `python -m pytest`. All tests should pass.
2. Run `python tools/comparison_demo.py` (about 10 minutes, or `--quick`) and check
   the verdict table.
3. Run `python main.py` with your artwork (ideally with DPI) and a photo of a good
   label: the result should be PASS, or REVIEW with a stated reason.
4. Photograph a label with a pen mark or scratched ink: the result should be FAIL,
   with red numbered boxes. Click the defect and compare original and printed
   side by side.
5. Dismiss the rejecting defect: the result should change to PASS. Mark it *Needs
   review*: REVIEW. Clear the mark: FAIL again.

## The interface

```
┌ File  Inspection  Defects  View  Help ─────────────────────────────────────────────┐
│ Load Original  Load Printed │ Compare  Manual Alignment ▾ │ Reset  Save Results     │
│ Settings │ Sensitivity [Medium]   Zoom In  Zoom Out  Fit  Actual  Sync  Markers     │
├───────────────┬───────────────┬────────────────────┬────────────────────────────────┤
│ 1 Original    │ 2 Aligned     │ 3 Difference       │ Inspection Details (dockable)  │
│   Artwork     │   Print       │   Result           │  [Defects] [Checks]            │
│               │ [view ▾]      │ [view ▾] [gain]    │  Verdict reasons               │
│   ┌─┐1        │   ┌─┐1        │   ┌─┐1             │  # Category Severity Operator  │
│   └─┘         │   └─┘         │   └─┘              │    Position Size Score ...     │
│               │               │                    │  ◀ Previous  Next ▶            │
│               │               │                    │  Original | Printed | Diff     │
│               │               │                    │  Confirm  Dismiss  Review  Clear│
├───────────────┴───────────────┴────────────────────┤                                │
│ Capture quality  Alignment quality  Progress ▓▓▓░  │ PASS / FAIL / REVIEW            │
└────────────────────────────────────────────────────┴────────────────────────────────┘
```

* **Viewers.** All three show full-resolution pixels. The mouse wheel zooms at
  the pointer; dragging pans. With **Synchronize Views** on (default), zooming
  or panning any viewer moves the others to the same part of the label at the
  same size on screen, although they have different pixel sizes (artwork
  pixels vs. the comparison grid). The original photo (viewer 2 → "Original
  photo") has no fixed relation to the artwork and is not synchronised.
* **Text.** Text findings appear as orange boxes "T1", "T2"… on the viewers and
  in the **Text** tab (see "Text inspection").
* **Barcodes.** Barcode findings appear as purple boxes "B1", "B2"… and in the
  **Barcodes** tab (see "Barcode inspection").
* **Colour.** Colour findings appear as teal boxes "C1", "C2"… and in the
  **Colour** tab; the difference viewer has two ΔE00 heatmap views (see
  "Colour analysis").
* **Defects.** Numbered boxes appear on all three viewers (red; yellow =
  selected; gray = dismissed); **B** hides them. Clicking a box or a list row
  selects the defect, zooms all viewers to it and shows original / printed /
  difference crops side by side. The list shows category, severity, operator
  status, position (mm from the artwork's top-left corner), size, score
  (1 = detection threshold) and detection reliability; the tooltip adds the
  artwork pixel position, largest difference and ΔE.
* **Status area.** Capture quality and alignment quality (registration
  supersedes the label-detection estimate) with their first finding, the
  progress of the background work, and the verdict with its reason. All
  details are in the **Checks** tab.
* **Compare** runs whatever the inspection still needs: compare again (e.g.
  after changing the sensitivity), or repeat registration / label detection.
  The inspection also starts automatically as soon as both images are loaded.
* **Errors** from background work appear in a red bar above the viewers and
  force REVIEW; failures to load or save a file open a message box.
* **Responsiveness.** Loading, checks, registration, comparison and saving run
  in a background thread pool; the window stays usable and outdated results
  are discarded.
* **Screen sizes.** The window opens at the configured size, limited to the
  screen, and remembers its layout. It fits screens down to about 800 × 600
  logical pixels (e.g. 1366 × 768 at 150 % scaling): panel headers stack,
  long texts are shortened with "…" (full text in the tooltip), the side panel
  can be resized, moved to the left, floated or hidden (F9). The Fusion style
  is used so the interface looks the same on Windows 10 and 11 and in dark mode.

## Saving results

**Save Results…** (Ctrl+S) asks for a folder and creates
`<photo name>_<date-time>_<VERDICT>/` in it:

| File | Content |
|---|---|
| `report.pdf` | The formatted inspection report (see "Inspection reports") |
| `inspection.json` | All inspection data: report header (files, date and time, profile, inspector), verdict and reasons, every check with its findings and metrics, every finding with operator decision, position and measurements, overview counts, operator decisions, regions and zones, settings used |
| `findings.csv` | One row per finding of every inspection (#n defects, Tn text, Bn barcodes, Cn colour): check, problem, severity, detection, operator decision, verdict effect, region or zone, position and size in mm and artwork pixels, plus the file names, date, verdict and profile on each row |
| `defects.csv` | One row per structural defect with all its measurements (opens in Excel) |
| `text.csv` | One row per text finding (if the text was inspected); `inspection.json` also has every line read |
| `colors.csv`, `color_findings.csv`, `color_heatmap.png`, `color_heatmap_local.png` | Colour areas (artwork / printed Lab, ΔE00), colour findings, heatmaps; marked calibrated or approximate |
| `barcodes.csv`, `barcode_findings.csv` | One row per barcode (expected / decoded value, status, decoding margin, damage) and per barcode finding |
| `annotated_comparison.png` | Artwork \| registered print with every finding, critical and ignored region marked, and a header with verdict, file names and date |
| `difference_heatmap.png` | The combined difference score over the print with a colour scale (1 = detection threshold) |
| `registered_print.png` | The photo registered to the artwork |
| `defect_NN.png` | Original \| printed \| difference crops of each defect, enlarged |

Saving runs in the background and is available once both images are loaded
and no work is running; an inspection that ended in REVIEW can be saved as
well. Operator marks are saved as they are at that moment.

## Verifying Step 7

1. `python -m pytest`: all tests pass (about 5 minutes).
2. `python main.py`, then drop the artwork on viewer 1 and the photo on
   viewer 2. The progress bar runs through the stages while the window stays
   responsive, then the badges and the verdict fill in.
3. Turn the mouse wheel over any viewer: all three zoom together on the same
   spot. Drag to pan; Ctrl+L switches synchronisation off and on.
4. Click a red box (or press N): all viewers zoom to the defect and the side
   panel shows it side by side. Press D (dismiss) or C (confirm) and watch the
   verdict; B hides the markers.
5. Settings…: set "Reject (FAIL) from severity" to Critical and press OK: the
   verdict is re-evaluated; change a severity limit: the comparison re-runs.
6. Save Results…: check the folder contents listed above.
7. Make the window small (or use a 1366 × 768 screen at 150 %): everything
   stays reachable.

## Text inspection

After the comparison, the text is inspected in the background ("reading text
(OCR)" in the progress area). The verdict is shown only when this is done.

**How it works**

1. **Text in the artwork.** Tesseract reads the artwork (enlarged so that
   1 mm text is large enough for OCR). Words are grouped into lines. Text-like
   areas that OCR cannot read are kept as *unread areas* and are still checked
   by the image comparison.
2. **The same place on the print.** Because the photo is registered to the
   artwork, each line is cut out at the same coordinates from the artwork and
   from the print, prepared identically (best colour channel, contrast, dark
   text on light, scaled to 48 px line height) and read as a single line. The
   artwork reading is the expected text, so OCR quirks of a font affect both
   sides alike.
3. **Text comparison.** Words are aligned first, then characters:
   *missing word*, *missing character*, *incorrect number*, *incorrect
   character*, *missing punctuation*, *extra text*. Differences only in
   spacing are ignored.
4. **Character image check, without OCR.** Every character of the artwork is
   compared with the print: if a share of its ink is missing (20 %; 10 % in
   critical text), it is a *partially printed character* (or a missing one if
   almost all is gone). This works for fonts and sizes that OCR cannot read.
5. **Reliability.** A text difference is *reliable* only if both OCR
   readings are confident (≥ 60, critical text ≥ 80), it is not a typical OCR
   confusion (0/O, 1/l/I, 5/S, rn/m, a case change …), the text is at least
   12 px high in the photo, and the image comparison shows a difference at
   that place. Otherwise it is *uncertain* and the verdict is **REVIEW**, not
   FAIL. OCR confidence is shown for every line and finding.

**Critical text.** Text in a *critical text zone*, and (setting *Treat lines
with numbers as critical*, on by default) every line containing digits, is
checked strictly: higher OCR confidence, the 10 % ink limit, all findings at
least *critical*, any uncertain finding gives REVIEW, and if OCR cannot read
it at all, that is itself a finding (REVIEW).

**Critical text zones.** *Inspection → Text Zones…* (Ctrl+T, or *Edit Text
Zones…* in the Text tab) shows the artwork: press *Draw Zone* and drag a
rectangle around the text, name it, and optionally enter the **exact
expected text** (then the print is compared with that text, and an artwork
that reads differently is reported as well). Zones are saved next to the
artwork as `<artwork file>.zones.json` and reused for every inspection of it.

**Text tab.** Findings (T1, T2 … with expected and read text, severity,
operator status, reliability, OCR confidence of artwork / print), every line
read (artwork text, printed text, confidence, status), and the selected
finding with original and printed crops and the Confirm / Dismiss / Needs
review / Clear buttons (keys C / D / R / U work for the selected finding).

**Settings** (*Text inspection (OCR)*): on/off, REVIEW without OCR, minimum
confidences, ink limits, numbers as critical, languages, Tesseract folder.
All further values are in the `text` section of the settings file (severities
per kind of finding, page segmentation modes, scaling, time-out, parallel OCR
calls).

**Limits**

* OCR is unreliable for very small text (below about 1 mm cap height in a
  typical phone photo), stylised or script fonts, vertical or curved text,
  and heavily damaged print. Such text is checked by the image comparison
  only; in critical text it gives REVIEW.
* The comparison is with the *artwork's* text as OCR reads it. A wrong text
  in the artwork itself is not detected unless a zone gives the expected text.
* Character positions within a word are estimated from the word box, so a
  finding's box may be off by a character in proportional fonts.
* **Tested here with a scripted OCR engine** (it returns what Tesseract would
  read for each region) on synthetic labels with real image defects; all
  image processing, text comparison and decision logic ran for real. The
  Tesseract interface was tested with a stand-in program. The two tests with
  the real Tesseract (`tests/test_ocr.py`) are skipped until Tesseract is
  installed; they run automatically afterwards.

## Verifying Step 8

1. Install Tesseract (above), then `python -m pytest`: all tests pass and the
   two real-OCR tests in `tests/test_ocr.py` are no longer skipped.
2. `python main.py`, load a label: the Checks tab, *Text (OCR)*, names the
   Tesseract version; the Text tab lists every line with artwork and printed
   text and the OCR confidences.
3. Inspect a label with a missing or altered digit in a lot number: a
   reliable *missing character* / *incorrect number* finding (orange box) and
   FAIL.
4. Inspect a photo where the text is blurred: uncertain findings or
   *text not readable* in critical text, verdict REVIEW.
5. Draw a zone around a lot number with its exact text (Ctrl+T); inspect
   again: the zone line is compared with that text.
6. Rename the Tesseract folder (or set a wrong one in Settings): REVIEW with
   "Text inspection (OCR) is not available".

## Barcode inspection

**Library.** Barcodes are decoded with **ZXing-C++** through its Python
package `zxing-cpp` (Apache 2.0), installed with the other requirements
(`pip install -r requirements.txt`). The wheel contains the compiled
library: decoding is local, with no network access, service or licence key.
For a PC without internet, include it in the `wheels` folder (see
Installation); PyInstaller packages it with the application.

**Supported:** EAN-13, UPC-A, Code 128, QR Code (setting `barcodes.formats`).
EAN-13 and UPC-A values are compared as 13-digit GTINs.

**How it works** (after the comparison, in the background, "reading barcodes"):

1. **Locate.** The artwork is decoded on the comparison grid: every supported
   code found there is inspected. **Barcode zones** add where a code must be.
2. **Decode the print** in the same region (the photo is registered to the
   artwork) and **compare** with the expected value: the zone's value if
   given, otherwise the value decoded from the artwork.
3. **Decoding failures.** A code that cannot be decoded is a finding. It is
   *reliable* (FAIL) only when damage visible in the image explains it;
   otherwise the photo may be the cause (blur, glare, too small) and the
   verdict is **REVIEW** - check the code with a scanner.
4. **Damage**, measured against the artwork:
   * *Linear codes:* the print is binarised and compared with the artwork bars
     (one pixel of registration jitter across the bars is tolerated); damaged
     areas of at least 0.5 square modules are reported. The code is also
     decoded in 10 horizontal strips: the share that decodes is the
     *decoding margin* in this photo (at least 70 %). The average bar width
     change (ink spread) and dark marks in the quiet zones are measured.
   * *QR codes:* every module of the artwork's module grid is sampled in the
     print; wrong modules are grouped into damaged areas (finder pattern
     damage is noted), and the decoder reports how much of the error
     correction was needed (at least 25 % must remain unused).
   * Damage that does not stop the code decoding is *minor*; damage that
     affects decoding is *major*.
5. **Wrong type** (e.g. a QR code where the zone requires Code 128) and
   **unexpected barcodes** (in the print where the artwork has none) are
   reported too.

**Not ISO grading.** ISO/IEC 15416 (linear) and 15415 (2D) grades need a
calibrated verifier with defined illumination, aperture and reflectance
calibration. The measurements here come from a phone photo; they show
whether the code decodes, whether its content is right and where it is
damaged, but they are no substitute for a verifier grade. Every result says
so.

**Barcode zones.** *Inspection → Barcode Zones…* (Ctrl+Shift+B, or *Edit
Barcode Zones…* in the Barcodes tab): drag a rectangle around a code, name
it, choose its type and, optionally, enter the **expected value** (e.g. the
GTIN from the product master data; EAN-13 / UPC-A values need a correct
check digit). A zone must contain a decodable code; if the artwork itself
encodes a different value than the zone, that is reported (REVIEW). Zones
are saved next to the artwork as `<artwork file>.barcodes.json`. Findings in
critical zones (default) are at least *critical*, and uncertain ones give
REVIEW.

**Barcodes tab.** Every code (name, type, expected value and where it comes
from, value decoded from the print, status, decoding margin, damage) and every
finding (B1, B2 … with severity, operator status and reliability); the
selected finding with original and printed crops and Confirm / Dismiss /
Needs review / Clear (keys C / D / R / U).

**Settings** (*Barcode inspection*): on/off, REVIEW without the library,
strips that must decode, QR error correction left unused, accepted bar width
change. More in the `barcodes` section of the settings file (formats, strips,
quiet zones, severities per finding).

**Interplay with the other checks.** Barcode areas are excluded from the text
inspection (bars are not text; the digits printed under a code are still
read). The image comparison also reports damage in a code as defects.

**Limits**

* Small codes need enough photo resolution (several photo pixels per module;
  the minimum has not been measured yet). Blurred photos make decoding fail;
  without visible damage that gives REVIEW, not FAIL.
* Linear codes are analysed when they are horizontal or vertical in the
  artwork; other angles are decoded but the bars are not measured.
* Bar width change depends on photo sharpness and exposure; it is reported as
  *uncertain* and is minor by default.
* Tested on synthetic labels (codes rendered with the same library, printed
  and photographed by the simulator, including perspective, blur, noise and
  JPEG). Real labels and real phone photos have not been tested yet.

**Fix in the comparison engine (Step 5) found while testing barcodes.** A large
defect in a dense high-contrast pattern (all bars missing over a 1 mm wide strip)
could distort the tone curve of the expected print, so that every bar edge on
the label was reported as a difference (about 140 false defects): one sparse
level of the tone curve, dominated by the missing bars, was carried to every
lighter level by the curve's monotonic step. Levels lying more than 20 gray
levels above the following levels are now ignored; curves without such a
spike are unchanged. That photo now gives exactly the one real defect, and
all earlier comparison tests still pass.

## Verifying Step 9

1. `pip install -r requirements.txt` (adds `zxing-cpp`), then
   `python -m pytest`: all tests pass.
2. `python main.py` with a label that has barcodes: the Barcodes tab lists
   each code with its decoded value and "10/10 strips" (linear) or
   "100 % EC unused" (QR); verdict PASS.
3. Photograph a print with a scratched or partly missing barcode: damaged
   bars / modules (purple boxes); if it no longer decodes, *Not decodable*
   and FAIL.
4. Draw a barcode zone with the GTIN from your product data
   (Ctrl+Shift+B): a print with another GTIN gives *Wrong content* and FAIL.
5. A blurred photo of a good code: *Not decodable* (uncertain) and REVIEW,
   with the advice to check it with a scanner.

## Colour analysis

**Separate from the structural inspection.** It runs after the comparison
(progress "analysing colour") and has its own tab, findings (C1, C2 …),
heatmaps and settings. Areas of structural defects (missing or extra print,
shape, edge or text differences) are left to the structural inspection and
are not measured as colour.

**Colour spaces.** The artwork and an uncalibrated photo are taken as **sRGB**.
Values are linearised, converted to XYZ and adapted to **D50** with the
Bradford transform (as an ICC sRGB profile does), then to **CIELAB D50**, the
reference for print measurement. Colour differences are **CIEDE2000 (ΔE00)**,
computed with scikit-image and checked against the published test data of
Sharma, Wu and Dalal (2005).

**Where ΔE00 is meaningful.** Only inside the flat colour areas of the
artwork, at least 3 px plus the smoothing away from edges (registration and
blur dominate there), with colours averaged over 0.2 mm in linear light (so
halftone dots mix as they do for the eye). Fine text, edges, gradients and
images are not measured (gray in the heatmaps). Flat areas of at least 4 mm²
are listed as *colour areas*.

**Approximate (default) vs calibrated.**

| | Approximate (no calibration) | Calibrated |
|---|---|---|
| Photo to CIELAB | taken as sRGB; white balance and exposure corrected **on the unprinted paper only** | profile from a photographed reference chart |
| Inks used for correction | never - a wrong ink stays visible | never |
| Local colour changes | reported (relative within the photo) | reported |
| Deviation from the artwork | only very large ones (ΔE00 ≥ 15), always *uncertain* → REVIEW | against the tolerance (ΔE00 5 major, 10 critical), *reliable* → FAIL |
| Paper colour | cannot be judged (used for the white balance) | measured |
| Shown as | **APPROXIMATE** - "not a print colour measurement" | **CALIBRATED** with the profile's accuracy |

**What is reported.**

* *Local colour change* - a spot that differs from the same artwork colour
  elsewhere on the label (ΔE00 ≥ 6, reliable from 10, at least 1 mm²).
  Camera and lighting affect the whole photo alike and largely cancel out;
  glare or a shadow on the spot can also cause it.
* *Colour deviation from artwork* - see the table.
* *Global colour shift* - most colour areas deviate the same way (e.g. all
  inks darker). Uncalibrated, this is most likely lighting or camera
  processing and is reported as such.
* *Lighting and white balance* - the paper's colour in the photo against the
  artwork white (ΔE00 and R/G, B/G balance: e.g. warm light), and the
  lighting falloff across the label (measured on the paper and divided out).

**Heatmaps.** In the difference viewer (viewer 3): *Colour ΔE00 vs artwork*
and *Colour change (local)*; dark = no difference, yellow = the heatmap limit
(the slider changes it), gray = not measured. Saved with the results.

**Colour calibration.** *Inspection → Colour Calibration…* (Ctrl+K, or the
button in the Colour tab):

1. Photograph a colour reference chart **with the same phone, settings and
   lighting** as the labels - best in the same photo as the label (*Chart in
   the current printed photo*), otherwise as a separate photo (*Separate chart
   photo*). Lock exposure and white balance if the camera app allows it.
2. Choose the chart: the built-in **ColorChecker Classic 24** (nominal values
   published by X-Rite for charts made from November 2014), or better *Load
   chart reference file…* with the measured values supplied with your chart:
   JSON `{"name", "rows", "cols", "illuminant": "D50", "patches": [{"name",
   "lab": [L, a, b]}, …]}`, patches row by row from the top left.
3. *Mark Chart Corners and Calibrate…*: place the four outer corners of the
   patch field, starting at the top-left patch, clockwise. The patches are
   sampled, the order is checked (lightness ranks), and a root-polynomial
   mapping (camera RGB → XYZ D50) is fitted. The dialog shows the expected
   error on other colours (**leave-one-out** ΔE00) per patch.
4. A profile is used only if its leave-one-out mean ΔE00 is at most 3
   (setting). It is kept in `%LOCALAPPDATA%\PrintInspector\color_profile.json`
   (or the file in `color.calibration_profile`) until removed.

A profile is valid only for the camera, settings and lighting it was made
with; the application cannot check that a later photo was taken the same way.
Profiles from a separate chart photo scale the exposure on the paper
(luminance only, setting `exposure_from_paper`).

**Settings** (*Colour analysis*): on/off, local change threshold, calibrated
tolerance and critical deviation, the uncalibrated REVIEW threshold, the
calibration accuracy limit. More in the `color` section of the settings file
(smoothing, edge margin, minimum areas, severities, heatmap scale).

**Limits**

* Even calibrated, a phone photo is not a spectrophotometer: no defined
  measurement geometry, backing or UV condition, and gloss or glare change
  the result. Use a spectrophotometer for colour approval.
* Large colour areas only: small areas, fine detail, gradients and photos in
  the artwork are not measured.
* When one of two equally large areas of the same colour changes, the
  "typical" value lies between them and the local change is underestimated;
  the deviation from the artwork (calibrated) still shows it.
* Artwork colours are taken as sRGB; CMYK or spot colour definitions (Pantone)
  are not read from the artwork file.
* Tested on synthetic labels with a simulated phone camera (white balance,
  exposure, channel crosstalk) and through the photo simulator; not yet on
  real labels, cameras or charts.

## Verifying Step 10

1. `python -m pytest`: all tests pass.
2. `python main.py`, inspect a label: the Colour tab says **APPROXIMATE**, lists
   the colour areas with artwork and printed swatches and ΔE00, and the paper /
   white balance / lighting summary. Try the two colour heatmap views.
3. Photograph the same label under warm and cold light: the white balance note
   changes, but no colour findings appear.
4. Mark a spot on a coloured area with a coloured pen: a *local colour change*
   (teal box) and FAIL.
5. Photograph a ColorChecker with the label (Ctrl+K, chart in the current
   photo): the dialog shows the leave-one-out ΔE00; after OK the Colour tab
   says **CALIBRATED** and lists measured deviations.

## Inspection reports

Every saved inspection contains five reports of the same inspection: the PDF
report, the annotated comparison image, the difference heatmap, the findings
CSV and the JSON data. The PDF is built from the same data as
`inspection.json`, so the two never disagree. It is laid out with Qt
(`QTextDocument` + `QPdfWriter`, part of PySide6): no extra package, no
network. Every page has a footer with the photo name, date, verdict and page
number.

| Required content | PDF | JSON | CSV / images |
|---|---|---|---|
| Original and printed file names | page 1 | `report`, `files` | every CSV row; image header |
| Inspection date and time | page 1 and footer | `report.inspected_at` | every CSV row; image header |
| Capture quality results | page 1 (level, problems, metrics) | `checks.capture_quality` | – |
| Alignment quality results | page 1 (alignment and registration) | `checks.alignment`, `checks.registration` | – |
| Inspection settings | last pages (all values), profile, regions, zones | `settings`, `setup` | profile on every CSV row |
| Detected defect count | "Result" | `findings_overview` | one row per finding |
| Coordinates and severity | findings table (mm and artwork px) | every finding | `findings.csv`; boxes in the image |
| PASS / FAIL / REVIEW | page 1 badge and footer | `verdict` | every CSV row; image header |
| Operator review decisions | findings table, decisions list, inspector name | `operator_review`, each finding | `operator` column; "ok?" label in the image |

The PDF also shows the defect close-ups (up to *Defect crops in the PDF*, all
are saved as PNG files), the colour heatmap if colour was analysed, and the
notes that barcode results are not ISO grades and uncalibrated colours are
approximate.

Settings ▸ *Reports*: PDF on/off, report title, organisation, inspector name,
number of defect crops in the PDF.

## Critical and ignored regions

*Inspection ▸ Inspection Regions…* (Ctrl+Shift+R): drag rectangles on the
artwork and choose the type:

* **Critical**: a structural defect or colour finding touching the region is at
  least of the severity set in Settings ▸ *Severity in critical regions at
  least* (major by default, so a reliable one fails the label). An uncertain
  finding there always gives REVIEW, even when it is minor.
* **Ignored**: not inspected at all. The region is removed from the
  comparison mask (no defects, no colour measurement), words touching it are not
  read, and barcodes found there in the artwork are skipped (barcodes in a
  barcode zone are still inspected). Use it only for areas that may
  legitimately differ, such as variable data printed later.

Regions appear as red dashed (critical) or gray (ignored) boxes on the
viewers and in the annotated image. Without a profile they are saved next to
the artwork in `<artwork>.regions.json`, like the text and barcode zones. Text
and barcodes keep their own critical zones.

## Inspection profiles

A profile holds everything that belongs to a **label design**, so a recurring
label is inspected the same way every time:

| Stored in a profile | Settings |
|---|---|
| Sensitivity and sensitivity thresholds | sensitivity, `comparison`, `defects` |
| Acceptance criteria | `decision` (incl. critical-region severity) |
| Inspection zones, critical regions, ignored regions | regions, text zones, barcode zones |
| Text verification settings | `text` and the text zones with expected text |
| Barcode expected values | barcode zones with format and expected value, `barcodes` |
| Colour tolerances | `color` |

Not stored: paths that belong to one computer (Tesseract folder, tessdata
folder, colour calibration file), capture/alignment/registration settings, the
window layout and logging. They stay as configured on the computer that loads
the profile.

**Profiles…** (Ctrl+P) opens the library:

* **Save Profile** stores the current setup under a name; the profile is then
  in use (shown in the window title).
* **Load** applies a profile: its settings over this computer's, its
  sensitivity, regions and zones. The comparison runs again.
* **Import… / Export…** copy a profile file to or from any folder (a network
  share, a USB stick); imported profiles are validated first.
* **Delete** removes the profile file.

While a profile is in use, changes to settings, sensitivity, regions or zones
belong to the profile: the title shows *(modified)* until you use
*Profiles ▸ Save Profile* (Ctrl+Shift+S). Settings a profile does not hold
(e.g. the Tesseract folder, the report inspector name) are still saved for this
computer. *Profiles ▸ Close Profile* returns to this computer's settings and
the regions and zones saved next to the artwork. If a profile was made for an
artwork of a different shape, a warning asks you to check its regions and zones.
When an artwork is loaded for which a profile exists (same file name), the
status bar points to it.

Profiles are JSON files in `%LOCALAPPDATA%\PrintInspector\profiles\`. A
profile from a newer version, with unknown settings, invalid values or invalid
zones is rejected with a message; settings added in later versions take
this computer's values. No cloud storage or database is involved.

## Verifying Step 11

1. `python -m pytest`: all tests pass.
2. Inspect a label, then *Inspection Regions…*: draw an **ignored** region over
   a defect and a **critical** region over a small (minor) defect. The first
   disappears, the second becomes major and the label fails.
3. *Profiles…* → name it → **Save Profile**: the title shows the profile.
   Change the sensitivity: *(modified)*; Ctrl+Shift+S saves it again.
4. *Close Profile*, then *Profiles…* → **Load**: settings, sensitivity,
   regions and zones come back and the inspection runs again.
5. **Export…** the profile, delete it, **Import…** it again.
6. Confirm one finding and dismiss another, then **Save Results…**: open
   `report.pdf` (verdict, files, date, capture and alignment results, findings
   with positions, severity and your decisions, images, settings),
   `annotated_comparison.png`, `difference_heatmap.png`, `findings.csv` (in
   Excel) and `inspection.json`.


## The offline Windows application

`packaging\build_windows.bat` turns the source into **Print Inspector.exe**: a
standalone, windowed program in one folder (`dist\Print Inspector\`), plus a
ZIP with a SHA-256 checksum and optionally a setup program. Python is not
needed on the inspection computers. The full guide covers installation, use,
verification on a clean PC and without network, the network audit, the build
and troubleshooting: [docs/WINDOWS_DEPLOYMENT.md](docs/WINDOWS_DEPLOYMENT.md).

| Requirement | How it is met |
|---|---|
| PyInstaller build | `packaging/print_inspector.spec`: one folder (faster start than one file, LGPL-friendly), windowed, no UPX |
| Python libraries | Collected by PyInstaller. `HIDDEN_IMPORTS` in `packaging/bundle_rules.py` adds what it cannot see (`scipy._cyutility`, found by the test build) |
| Qt plugins | Platform, styles, image formats and icon engines are collected by PyInstaller's PySide6 hooks. The application checks at start-up that the platform plugin is present |
| OCR binaries and language data | `tesseract\` (from `tools/bundle_tesseract.py`) is bundled into `_internal\tesseract`, which the application searches first |
| Model files | None needed. The inspection uses no trained models apart from Tesseract's language data, which is bundled with it |
| Default configuration | `config/default_config.json` is bundled into `_internal\config` |
| Start-up error handling | `main.py` and `print_inspector/startup.py`, using the standard library only: missing files, DLL/import errors and other start-up errors give a message box plus `startup_error.log`; native crashes write `crash.log` |
| No downloads, no activation, no cloud | There is no network code in the application (checked by a test). The build leaves out Qt Network, QML/Quick, the virtual keyboard, the TUIO touch plugin (which listens on UDP) and the TLS plugins. `print_inspector/offline.py` blocks and logs outbound connections and DNS lookups from Python |
| Data stays local | Results go to the folders the operator chooses; settings, profiles, logs and self-test output go to `%LOCALAPPDATA%\PrintInspector` |

Command line of the built program:

```
"Print Inspector.exe"                                   start the application
"Print Inspector.exe" --self-test [--output DIR] [--quiet]   check the installation (exit code 0 = no check failed)
"Print Inspector.exe" --version
```

The **self-test** (`print_inspector/selftest.py`) runs the real main window
(not shown) on generated labels with known content. It checks:
- start-up and the bundled plugins;
- image loading, label alignment and registration;
- defect detection (a good print and one with an added mark);
- text inspection with a text zone (SKIPPED, not passed, without Tesseract);
- barcode inspection (an EAN-13 with its expected value);
- colour inspection (a recoloured logo);
- the verdicts, including an operator confirmation;
- report export (PDF, JSON, CSV, images);
- inspection profiles (save, list, reload, apply);
- offline operation (no network attempts, no network libraries loaded or bundled).

The build script runs it on the built program. Run it again on every inspection computer.

Licences and redistribution obligations: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Two items need action before distributing outside your organisation:
- The GNU LGPL/GPL texts for Qt must be placed in `packaging/licenses/`. `-FetchLicenseTexts` downloads them on the build computer, and `-Release` refuses to build without them.
- Qt's third-party attributions and an LGPL source offer must be added.

## Verifying Step 12

**What was actually run during development.** The development computer is a Mac (Apple Silicon), not Windows:

| Test | Result |
|---|---|
| Full test suite from source (`python -m pytest`) | 771 passed, 2 skipped (the real-Tesseract tests: Tesseract is not installed) |
| `tests/test_deployment.py`, 59 tests: network guard, no network code, start-up errors, command line, build rules, licence collection, icon, in-process self-test | Passed |
| PyInstaller build with `packaging/print_inspector.spec` (macOS arm64) | Built. Found and fixed a missing hidden import (`scipy._cyutility`); without it the built program failed at start, and the start-up error handling reported it correctly |
| Built program: `--self-test` | 10 passed, 0 failed, 1 skipped (text inspection: no Tesseract bundled) |
| Built program: `--self-test` with **all networking denied** by the operating system (macOS `sandbox-exec`, `deny network*`) | Same result. The OS log shows no internet or DNS access by the program. Its only denied network operation was a local Unix socket to the macOS system logger, opened by macOS's own logging |
| Built program: normal start, `--version`, a copy with `config/default_config.json` removed | Window starts and logs "Offline mode: outbound network access is blocked"; version shown; damaged copy: message plus `startup_error.log`, exit code 3 |
| Bundle contents | No Qt Network, QML/Quick, virtual keyboard, PDF or TUIO touch libraries (39 files left out) |

**Not run yet:** nothing on Windows.
- **Not run on Windows:** `build_windows.ps1`, `check_network.ps1`, the installer script, the Windows Sandbox test, and the built `.exe` itself.
- **Not tested with Tesseract:** the OCR part of the self-test.
- **No real photos:** only synthetic labels and photos so far.

Do these on Windows before relying on the build:

1. On a Windows 10/11 x64 build PC with Python 3.11 x64 and Tesseract:
   `packaging\build_windows.bat -TesseractSource "C:\Program Files\Tesseract-OCR"`.
   The build must end with all self-test checks PASS (none skipped).
2. Double-click `dist\PrintInspector-clean-test.wsb`. This is a clean Windows Sandbox: no Python and networking disabled. The self-test must pass; confirm inside that `where python` finds no Python.
3. In an administrator PowerShell, run `packaging\check_network.ps1` and then `check_network.ps1 -Interactive` (with a real inspection). Expected: "No network connection attempts … were recorded."
4. On a clean PC or VM with networking disabled, inspect real labels, save results and profiles, and fill in the release checklist (section 12 of docs/WINDOWS_DEPLOYMENT.md).
