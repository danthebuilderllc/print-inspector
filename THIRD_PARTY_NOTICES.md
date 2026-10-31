# Third-party components and redistribution obligations

Print Inspector (the Windows application built by `packaging/build_windows.ps1`)
contains the components listed below. The licence texts are in the
`licenses\` folder of the application (`licenses\INDEX.txt` lists every
component with its version and licence; `licenses\requirements-lock.txt` lists
the exact package versions of the build).

This is a technical summary for whoever distributes the application. It is not legal advice.
Before distributing the application outside your organisation, have it reviewed by
someone qualified. The licence for Print Inspector's own code has not been chosen yet;
that is the owner's decision.

## Summary: what to do when distributing

| # | Obligation | Applies because of | Done by the build? |
|---|-----------|--------------------|--------------------|
| 1 | Ship the licence and notice texts of every component | all components | Yes: `licenses\` (collect_licenses.py), apart from item 4 |
| 2 | Ship the GNU LGPL v3 and GNU GPL v3 texts | Qt for Python (PySide6, shiboken6), Qt | Only if `packaging\licenses\LGPL-3.0.txt` and `GPL-3.0.txt` exist. A `-Release` build stops without them |
| 3 | Offer the source code of the LGPL libraries in the exact versions shipped, for at least 3 years, or ship it with the application | Qt 6.7.3 / PySide6 6.7.3 / shiboken6 6.7.3; LGPL libraries in the Tesseract folder, if any | No: keep `qt-everywhere-src-6.7.3` and `pyside-setup-everywhere-src-6.7.3` (and the Tesseract build's sources, see below) with each release and state how to get them |
| 4 | Ship Qt's third-party attributions (FreeType, HarfBuzz, libpng, zlib, PCRE2, double-conversion, md4c, Unicode data, …) | Qt libraries bundled | No: the PySide6 wheels do not contain them. Add the attributions for QtCore, QtGui, QtWidgets, QtSvg, QtOpenGL, QtDBus and the bundled plugins from "Licenses Used in Qt 6.7" (Qt documentation) to `packaging\licenses\` |
| 5 | Let users replace the LGPL libraries with modified versions, and do not forbid reverse engineering for that purpose | LGPL v3 | Yes: one-folder build, Qt and PySide6 stay separate DLL files. Do not add licence terms that forbid it |
| 6 | Keep the Apache-2.0 NOTICE files and licence | OpenCV, zxing-cpp, Tesseract | Yes (OpenCV, zxing-cpp; Tesseract's licence files are copied by bundle_tesseract.py) |
| 7 | Do not use GPL-only Qt modules | Qt Virtual Keyboard, Qt Charts and others are GPL v3 / commercial only | Yes: the build leaves out the virtual keyboard and every other unused Qt module (packaging\bundle_rules.py) |

## Components

### Python 3.11 (runtime, standard library)
- Licence: PSF License 2.0 (permissive).
- Obligation: include the licence. Done: `licenses\python\LICENSE.txt`.
- Python's Windows runtime includes OpenSSL (`libcrypto`, `libssl`; Apache-2.0), libffi (MIT), SQLite (public domain) and others. Their notices are part of Python's LICENSE.txt. Print Inspector does not import `ssl`.

### Microsoft Visual C++ runtime (vcruntime140.dll, msvcp140.dll, …)
- Shipped with Python and the PySide6 wheels, and collected by PyInstaller.
- Licence: Microsoft Visual Studio licence terms. These DLLs are "Distributable Code" and may be redistributed unmodified as part of an application.

### Qt 6.7.3 via PySide6 6.7.3 and shiboken6 6.7.3 (Qt for Python)
- Licence: LGPL v3 (also available under GPL v2/v3 or a commercial Qt licence).
- Bundled parts:
  - QtCore, QtGui, QtWidgets;
  - QtSvg, QtOpenGL, QtDBus where a plugin needs them;
  - plugins: platform (`qwindows`), styles, image formats, icon engines.
- Left out by the build: Qt Network, QML/Quick, the virtual keyboard (GPL-only), PDF, WebEngine and all other unused modules.
- Obligations: items 2–5 above. The one-folder layout keeps `Qt6*.dll` as separate files that a user can replace (LGPL "suitable shared library mechanism").
- The "Installation Information" requirement of LGPL v3 / GPL v3 §6 applies to consumer "User Products" (devices). It normally does not apply to software installed on an office or production PC.

### OpenCV 4.10 (opencv-python-headless 4.10.0.84)
- Licence: Apache-2.0. The wheel's `LICENSE-3RD-PARTY.txt` lists its bundled libraries: libjpeg-turbo, libpng, libtiff, libwebp, OpenJPEG, OpenEXR, zlib, Intel IPP ICV, and others.
- Obligation: include LICENSE and the third-party file. Done.
- FFmpeg (LGPL 2.1) comes with the Windows wheel as a separate plugin, `opencv_videoio_ffmpeg*.dll`, used only for video files. The build leaves it out, so no FFmpeg obligation applies to the Windows application. (macOS and Linux wheels link FFmpeg into OpenCV itself; that only matters for builds on those systems.)

### NumPy 1.26.4 and SciPy
- Licence: BSD-3-Clause.
- The Windows wheels bundle:
  - OpenBLAS (BSD-3-Clause);
  - the GCC Fortran/quadmath runtime (GPL v3 with the GCC Runtime Library Exception, which allows binary redistribution without GPL obligations).
- Obligation: include the licence files (their LICENSE.txt contains the bundled-library texts). Done.

### scikit-image 0.24.0 and its dependencies
- scikit-image: BSD-3-Clause.
- Dependencies: imageio (BSD-2-Clause), tifffile (BSD-3-Clause), networkx (BSD-3-Clause), lazy_loader (BSD-3-Clause), packaging (Apache-2.0 or BSD-2-Clause).
- Obligation: include the licences. Done.

### Pillow 10.4.0
- Licence: MIT-CMU (HPND).
- The Windows wheel bundles libjpeg-turbo, libpng, zlib, libtiff, libwebp, FreeType (FreeType Licence), Little CMS, OpenJPEG, HarfBuzz/raqm and others. Their notices are in Pillow's LICENSE.
- Obligation: include the licence. Done.

### zxing-cpp 3.1.1 (barcode decoding)
- Licence: Apache-2.0. Obligation: include the licence (and NOTICE, if one is shipped). Done.

### Tesseract OCR (optional; only if `tesseract\` was bundled)
- Tesseract and the `tessdata` language files: Apache-2.0. bundle_tesseract.py copies Tesseract's licence files into the bundle.
- The usual Windows build (UB Mannheim) contains many DLLs, all copied by bundle_tesseract.py. Among them:
  - Leptonica (BSD-2-Clause), libpng, libjpeg, libtiff, libwebp, OpenJPEG, giflib, zlib, ICU (Unicode licence), libarchive (BSD), libcurl (curl licence);
  - GCC runtime libraries (GPL v3 with runtime exception);
  - for the training tools only, LGPL libraries such as GLib, Pango and Cairo.
- Obligations:
  1. Before a release, list the DLLs in `tesseract\` and include each one's licence.
  2. For every LGPL DLL, either meet the LGPL obligations (item 3), or remove the DLLs that `tesseract.exe` does not need. Check with `dumpbin /dependents tesseract.exe` before removing any.
- libcurl lets Tesseract read an image from a URL. Print Inspector always passes Tesseract a local temporary file, so this is never used.

### PyInstaller 6.10 bootloader
- Licence: GPL v2 with the PyInstaller bootloader exception. The exception allows distributing the built application under any licence; the bootloader's licence places no obligations on it.

### Inno Setup 6 (optional installer)
- Free to use, including commercially. The setup programs it builds carry no licence obligations from Inno Setup.

### Fonts
- No fonts are bundled. Qt uses the fonts installed with Windows.
- The PDF reports embed subsets of the fonts they use. The standard Windows fonts (Segoe UI, Arial) allow embedding for printing and viewing.

## No online licensing, activation or telemetry

None of the components above needs online activation or licence keys, and none
of them phones home in this build. The application blocks outbound network
access at start-up. See "Verifying that nothing communicates over the network"
in `INSTALLATION AND USAGE.md` (docs/WINDOWS_DEPLOYMENT.md in the source).
