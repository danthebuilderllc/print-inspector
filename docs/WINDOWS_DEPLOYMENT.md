# Print Inspector for Windows: installation, use and verification

Print Inspector compares label artwork with a smartphone photo of the printed
label and reports PASS, FAIL or REVIEW. The Windows application:

- runs **without Python** and **without internet access**;
- needs **no activation** or licence key and has **no cloud services, update checks or telemetry**;
- **keeps all inspection data on the computer**, in the folders you choose and in your Windows user profile.

Contents:

1. [What you receive](#1-what-you-receive)
2. [System requirements](#2-system-requirements)
3. [Installation](#3-installation)
4. [First start: the self-test](#4-first-start-the-self-test)
5. [Using Print Inspector](#5-using-print-inspector)
6. [Where data is stored, uninstalling](#6-where-data-is-stored-uninstalling)
7. [Verifying on a clean Windows computer](#7-verifying-on-a-clean-windows-computer-without-python)
8. [Testing with networking disabled](#8-testing-with-networking-disabled)
9. [Verifying that nothing communicates over the network](#9-verifying-that-nothing-communicates-over-the-network)
10. [Building the application](#10-building-the-application)
11. [Troubleshooting](#11-troubleshooting)
12. [Release checklist](#12-release-checklist)

---

## 1. What you receive

| File | Purpose |
|------|---------|
| `PrintInspector-<version>-win64.zip` | The application folder (portable, no installation needed) |
| `PrintInspector-<version>-win64.zip.sha256` | Checksum, to check the ZIP arrived intact |
| `PrintInspector-<version>-setup.exe` (optional) | Installer: Start-menu entries and uninstaller |
| `PrintInspector-clean-test.wsb` | Windows Sandbox file for the clean-computer test (section 7) |

The application folder `Print Inspector\` contains:

```
Print Inspector\
├── Print Inspector.exe          the program
├── _internal\                   Python runtime, libraries, Qt, settings, Tesseract OCR (never edit or remove)
│   ├── config\default_config.json
│   ├── tesseract\               OCR program and language data (if the build included OCR)
│   └── PySide6\ cv2\ numpy\ ...
├── licenses\                    licence texts of all bundled components, INDEX.txt, requirements-lock.txt
├── THIRD_PARTY_NOTICES.md       third-party components and redistribution obligations
├── INSTALLATION AND USAGE.md    this guide
├── BUILD-INFO.txt               version, build date, whether OCR is included
└── SELF-TEST-AT-BUILD.txt       self-test result on the build computer
```

## 2. System requirements

- Windows 10 or Windows 11, 64-bit (x64). Older Windows versions are not supported.
- 8 GB of RAM recommended (4 GB minimum). About 400 MB of disk space for the application, plus space for saved results.
- Screen resolution of at least 1366 × 768.
- No Python, no internet connection, no administrator rights (for the ZIP or a per-user installation).

## 3. Installation

### 3a. From the ZIP (portable)

1. Copy the ZIP to the computer (USB stick, network share or internal software distribution).
2. Optional but recommended: check the file is intact. In PowerShell:
   ```powershell
   Get-FileHash .\PrintInspector-0.12.0-win64.zip -Algorithm SHA256
   ```
   The value must match the `.sha256` file.
3. If the ZIP came from the internet or an e-mail, right-click it → **Properties** → tick **Unblock** → OK.
   This avoids security warnings for every file.
4. Right-click → **Extract All…** and choose a folder with a short path:
   - for yourself only: `%LOCALAPPDATA%\Programs`;
   - for all users: `C:\Program Files` (needs administrator rights).

   You get the folder `…\Print Inspector\`.
5. Start `Print Inspector.exe`. For convenience, right-click it → **Show more options** → **Send to** → **Desktop (create shortcut)**.

> Do **not** start the program from inside the ZIP, and do not copy `Print Inspector.exe` on its own:
> it needs the `_internal` folder next to it.

### 3b. With the setup program

Run `PrintInspector-<version>-setup.exe`:
- By default it installs for the current user, without administrator rights, into `%LOCALAPPDATA%\Programs\Print Inspector`. Choose "all users" in the first dialog to install into `C:\Program Files\Print Inspector`.
- It creates Start-menu entries **Print Inspector**, **Print Inspector Self-Test** and **Installation and Usage**, plus a desktop shortcut if you choose one.
- It needs no internet connection.

### Windows security prompts

The executable is not signed unless your organisation signs it during the build (section 10). On first start,
Windows SmartScreen may show **"Windows protected your PC"**. Click **More info** → **Run anyway**, but only
if the file came from your trusted source. To avoid the prompt for everyone, sign the build with your
organisation's code-signing certificate, or have IT allow the folder.

## 4. First start: the self-test

Run the self-test once on every computer after installing, and after every update:

- Start menu → **Print Inspector Self-Test**, or
- in a command prompt: `"C:\…\Print Inspector\Print Inspector.exe" --self-test`

The self-test generates test labels with known content and known defects. It inspects them with the real
application and checks each stage:

| Check | What is verified |
|-------|------------------|
| Application startup | Bundled files, Qt platform and image-format plugins, library versions, main window |
| Image loading | Artwork PNG with resolution, phone photo JPEG |
| Label alignment | Label found in the photo, perspective corrected, registered to the artwork |
| Defect detection | No false defects on the good print; the added mark is found on the defective print |
| Text inspection | OCR reads the lot number in a text zone (needs bundled Tesseract) |
| Barcode inspection | EAN-13 decoded and compared with the expected value of a barcode zone |
| Colour inspection | No colour finding on the good print; the recoloured logo is found |
| PASS / FAIL / REVIEW | Correct verdicts, including after the operator confirms a defect |
| Report export | PDF report, JSON data, CSV findings, annotated image and heatmap written |
| Inspection profiles | Profile saved, listed, reloaded and applied |
| Offline operation | No network access attempted; no network libraries loaded or bundled |

At the end a message box shows the summary and the result folder: `%LOCALAPPDATA%\PrintInspector\self-test\<date-time>\`,
with `self_test.txt`, `self_test.json`, the generated images and the saved report.

- **PASSED**: every check ran and passed.
- **PASSED with skipped checks**: something could not be checked. Usually text inspection, when the build does not include Tesseract. A skipped check did *not* pass. Without OCR, the application gives REVIEW instead of PASS for every label whose text must be inspected.
- **FAILED**: do not use the installation for inspections. See section 11.

Command-line options (for scripted checks the exit code is 0 when no check failed):

```
"Print Inspector.exe" --self-test [--output <folder>] [--quiet]
"Print Inspector.exe" --version
```

## 5. Using Print Inspector

1. **Load Original…** (Ctrl+O): the approved artwork (PNG, TIFF, BMP or JPEG; PNG or TIFF with the correct resolution is best).
2. **Load Printed…** (Ctrl+Shift+O): a smartphone photo of the printed label. Take it flat and evenly lit, with the whole label visible and sharp.
3. The checks run on their own: photo quality, label alignment, registration, comparison, defect detection, and text, barcode and colour inspection. The status area shows progress and the verdict:
   - **PASS**: no defect that rejects the label, and every required check was reliable.
   - **FAIL**: a defect, text error, barcode error or colour deviation that rejects the label.
   - **REVIEW**: a person must decide (uncertain finding, unreliable photo, or a check that could not run).
4. Review the findings in the side panel tabs (Defects, Text, Barcodes, Colour) and mark each one **Confirmed** or **Dismissed**. Your decisions update the verdict and are saved with the results.
5. **Save Results** (Ctrl+S) writes a results folder:
   - `report.pdf`, `inspection.json`, `findings.csv`;
   - `annotated_comparison.png`, `difference_heatmap.png`;
   - defect close-ups and the CSV lists for each check.
6. For recurring label designs, set up text zones, barcode zones with expected values, and critical and ignored regions. Then save everything as an **inspection profile** (Ctrl+P) and load it next time.

Settings (Ctrl+,) holds sensitivity, acceptance criteria, colour tolerances, the report header (organisation and
inspector) and the Tesseract folder. F1 shows all keyboard shortcuts.

## 6. Where data is stored, uninstalling

| Data | Location |
|------|----------|
| Inspection results | The folder you choose in Save Results |
| Text zones, barcode zones, regions for an artwork | Next to the artwork file (`<artwork>.zones.json`, `.barcodes.json`, `.regions.json`) |
| Personal settings | `%LOCALAPPDATA%\PrintInspector\config.json` |
| Inspection profiles | `%LOCALAPPDATA%\PrintInspector\profiles\` |
| Colour calibration | `%LOCALAPPDATA%\PrintInspector\color_profile.json` |
| Window layout | `%LOCALAPPDATA%\PrintInspector\window.ini` |
| Log files | `%LOCALAPPDATA%\PrintInspector\logs\` (`print_inspector.log`, `startup_error.log`, `crash.log`) |
| Self-test results | `%LOCALAPPDATA%\PrintInspector\self-test\` |

Nothing is written to the registry (the setup program only adds its uninstall entry) or sent anywhere.
To keep data on the computer, save results to local folders. A network share is only accessed if you choose one.

**Uninstall:**
- ZIP installation: delete the `Print Inspector` folder.
- Setup installation: Windows Settings → Apps → Print Inspector → Uninstall.

Your settings, profiles and logs stay in `%LOCALAPPDATA%\PrintInspector`. Delete that folder to remove them too.

## 7. Verifying on a clean Windows computer (without Python)

The goal is to show that the application runs on a computer that has never had Python, Qt, OpenCV or Tesseract
installed. Use one of these:

### Option A: Windows Sandbox (recommended; Windows 10/11 Pro or Enterprise)

Windows Sandbox is a fresh Windows every time it starts. It has no Python, no Visual C++ redistributable and no
programs of yours.

1. Turn it on once: Control Panel → Programs → **Turn Windows features on or off** → **Windows Sandbox**, then restart.
2. The build creates `dist\PrintInspector-clean-test.wsb`. Double-click it. The sandbox:
   - starts **with networking disabled**;
   - maps the application folder read-only as `C:\PrintInspector`;
   - runs the self-test automatically.
3. Wait for the self-test message box. Then, inside the sandbox, confirm that Python is absent: open a command prompt and run
   ```
   where python & where py & dir "C:\Program Files\Python*"
   ```
   Expect "INFO: Could not find files", or only the Microsoft Store stub `…\WindowsApps\python.exe`. That stub is not Python: it only opens the Store.
4. The self-test results are in `dist\sandbox-results\selftest\` on the host. Keep `self_test.txt` with your release records.
5. Optionally also inspect a real label: copy an artwork and a photo into `dist\sandbox-results` before starting. Inside the sandbox they are in `C:\Results`.

### Option B: a clean virtual machine or spare PC

1. Install Windows 10/11 x64 from Microsoft media. Add no development tools, Python, Visual C++ redistributables or other Qt programs.
2. Check: `where python` finds nothing, or only the Store stub. Under Settings → Apps, no Python, no "Microsoft Visual C++ 2015-2022 Redistributable", no Tesseract.
3. Disconnect the network (section 8), copy and extract the ZIP, run the self-test, then inspect a few real labels and save the results.
4. Record the Windows version (`winver`), the self-test summary and the files you checked.

## 8. Testing with networking disabled

Do the self-test and at least one complete real inspection (load, inspect, review, save results, save and load a
profile) with all network access switched off:

- **Windows Sandbox:** the generated `.wsb` file already contains `<Networking>Disable</Networking>`.
- **Physical PC or VM:**
  - Switch off Wi-Fi and unplug the network cable, or enable airplane mode; or
  - in an administrator PowerShell:
    ```powershell
    Get-NetAdapter | Disable-NetAdapter -Confirm:$false   # turn back on: Enable-NetAdapter
    ```
  - For a VM, disconnect the virtual network adapter.

Confirm the network is really off: `Test-NetConnection 1.1.1.1 -Port 443` must report `TcpTestSucceeded : False`.

Expected behaviour: everything works exactly as with a network. There are no waiting times, no error messages and no "offline" warnings.

## 9. Verifying that nothing communicates over the network

Print Inspector contains no network code:
- no HTTP client, no update check, licence check or telemetry;
- the build leaves out Qt Network, WebEngine, QML and the plugins that use the network;
- at start-up, the application blocks all outbound connections and DNS lookups made from Python code, and logs any attempt (`Blocked outbound network …` in `print_inspector.log`).

You can verify this independently with Windows' own tools. Check both `Print Inspector.exe` and
`_internal\tesseract\tesseract.exe`, which the application starts for OCR.

1. **Automated check with Windows auditing (recommended).** In an administrator PowerShell:
   ```powershell
   powershell -ExecutionPolicy Bypass -File packaging\check_network.ps1 -AppDir "C:\…\Print Inspector"
   powershell -ExecutionPolicy Bypass -File packaging\check_network.ps1 -AppDir "C:\…\Print Inspector" -Interactive
   ```
   The script:
   - switches on the Windows audit of network connections (Security log events 5156/5157);
   - blocks both programs in the firewall;
   - runs the self-test (or the application, with `-Interactive`: use it normally and close it);
   - lists every connection Windows recorded for them, then restores the previous settings.

   Expected: **"No network connection attempts by Print Inspector or Tesseract were recorded."**
   (The script is in the source package under `packaging\`.)
2. **Sysinternals Process Monitor** (free, Microsoft). Filter `Process Name is Print Inspector.exe` and `Process Name is tesseract.exe`, plus `Operation begins with TCP` and `Operation begins with UDP`. Use the application for a full inspection. Expected: no events.
3. **Resource Monitor** (`resmon`) → Network tab, or **Sysinternals TCPView**, while the application runs. Expected: Print Inspector does not appear under Network Activity, TCP Connections or Listening Ports. In PowerShell:
   ```powershell
   $p = (Get-Process "Print Inspector").Id
   Get-NetTCPConnection -OwningProcess $p -ErrorAction SilentlyContinue
   Get-NetUDPEndpoint  -OwningProcess $p -ErrorAction SilentlyContinue
   ```
   Expected: no output.
4. **Firewall block rule** (for daily use, if your policy requires it):
   ```powershell
   New-NetFirewallRule -DisplayName "Print Inspector - no network" -Direction Outbound -Action Block -Program "C:\…\Print Inspector\Print Inspector.exe"
   New-NetFirewallRule -DisplayName "Print Inspector OCR - no network" -Direction Outbound -Action Block -Program "C:\…\Print Inspector\_internal\tesseract\tesseract.exe"
   ```
   The application works unchanged with these rules.
5. **Contents of the build:** no network libraries are bundled. Run this in the application folder:
   ```powershell
   Get-ChildItem -Recurse "_internal" | Where-Object Name -match "Qt6Network|WebEngine|Qt6Qml|qtuiotouch|qschannel"
   ```
   Expected: no output.

What you may see from **Windows itself** (not from Print Inspector): SmartScreen checking an unsigned program on
first start, Microsoft Defender cloud scanning, or certificate checks for signed files. These run under Windows
processes (`smartscreen.exe`, `MsMpEng.exe`, `svchost.exe`), are controlled by your Windows and IT policies, and
also happen for every other program.

## 10. Building the application

The build computer needs internet access only to download the Python packages, unless you use a wheel folder.
The built application needs none.

### Prerequisites (build computer)

1. Windows 10/11 x64.
2. **Python 3.11, 64-bit**, from python.org (tick "py launcher"; adding Python to PATH is optional).
3. **Tesseract OCR** for Windows, if text inspection is wanted. For example, the UB Mannheim installer, with the language data you need (`eng`, plus e.g. `deu`).
4. Optional: **Inno Setup 6**, for a setup program.
5. The Print Inspector source folder.

### Build

In a command prompt in the source folder:

```bat
packaging\build_windows.bat -TesseractSource "C:\Program Files\Tesseract-OCR"
```

The script:
1. creates `.venv-build` and installs `requirements.txt`;
2. runs the test suite;
3. copies Tesseract and its language data into `tesseract\`;
4. creates the icon and runs PyInstaller (`packaging\print_inspector.spec`);
5. adds the licence texts, this guide and `BUILD-INFO.txt`;
6. **runs the self-test of the built program**;
7. writes the ZIP with its SHA-256 checksum and the Windows Sandbox file.

Output: `dist\Print Inspector\`, `dist\PrintInspector-<version>-win64.zip`, and `dist\PrintInspector-clean-test.wsb`.

| Option | Effect |
|--------|--------|
| `-TesseractSource <folder>` | Bundle this Tesseract installation (otherwise an existing `tesseract\` folder is used) |
| `-Languages eng+deu` | OCR languages to bundle (default: from the settings) |
| `-NoTesseract` | Build without OCR on purpose (text inspection then gives REVIEW) |
| `-Wheelhouse <folder>` | Install the packages from a local wheel folder (offline build computer) |
| `-Python <python.exe>` | Use this Python 3.11 instead of `py -3.11` |
| `-SkipTests` | Skip the test suite (development builds only) |
| `-Release` | Distribution build: tests, Tesseract and the GNU licence texts required; all self-test checks must pass |
| `-FetchLicenseTexts` | Download the official LGPL-3.0 / GPL-3.0 texts into `packaging\licenses\` (once) |
| `-Installer` | Also build `PrintInspector-<version>-setup.exe` with Inno Setup 6 |
| `-Clean` | Recreate the build environment and remove old output |

A release build, typically:

```bat
packaging\build_windows.bat -Release -TesseractSource "C:\Program Files\Tesseract-OCR" -Installer
```

### Building on a computer without internet access

On a computer with internet access and the same Python 3.11 x64:

```bat
py -3.11 -m pip download -r requirements.txt -d wheels --only-binary=:all: --platform win_amd64 --python-version 3.11
```

Copy `wheels\` to the build computer, then:

```bat
packaging\build_windows.bat -Wheelhouse wheels -TesseractSource "C:\Program Files\Tesseract-OCR"
```

Also place `LGPL-3.0.txt` and `GPL-3.0.txt` in `packaging\licenses\` (see the README there).

### Code signing (optional, recommended)

Sign `dist\Print Inspector\Print Inspector.exe` (and the setup program) with your organisation's certificate
before creating the ZIP. Signing does not change how the application works offline:

```bat
signtool sign /fd SHA256 /a "dist\Print Inspector\Print Inspector.exe"
```

The timestamp option (`/tr <timestamp server URL>`) contacts a timestamp server *during signing*, not when the
application runs. Leave it out if the build computer has no internet access.

### Files of the build system

| File | Purpose |
|------|---------|
| `packaging\build_windows.ps1`, `build_windows.bat` | The build script and a double-click wrapper |
| `packaging\print_inspector.spec` | PyInstaller configuration (one folder, windowed, no UPX) |
| `packaging\bundle_rules.py` | What the build leaves out (network, QML, unused Qt modules) and the hidden imports |
| `packaging\collect_licenses.py` | Copies the licence texts of all bundled components |
| `packaging\make_icon.py` | Creates the `.ico` file from the icon drawn in `print_inspector\gui\app_icon.py` |
| `packaging\installer.iss` | Inno Setup script for the optional setup program |
| `packaging\check_network.ps1` | Records any network connection attempts of the installed application (section 9) |
| `packaging\licenses\` | Licence texts the bundled packages do not ship (GNU LGPL / GPL for Qt) |
| `tools\bundle_tesseract.py` | Copies a Tesseract installation and its language data into `tesseract\` |

## 11. Troubleshooting

### On the inspection computer

Log files are in `%LOCALAPPDATA%\PrintInspector\logs\`: `print_inspector.log` (normal log),
`startup_error.log` (errors that stopped the start) and `crash.log` (crashes in native code). Include them
when you report a problem.

| Symptom | Cause and solution |
|---------|--------------------|
| Nothing happens, or a window flashes and disappears | Read `startup_error.log` and `crash.log`. Often the cause is security software blocking or quarantining files in `_internal`: restore them, or ask IT to allow the folder. |
| "The installation is incomplete. These files are missing: …" | The program was started from inside the ZIP, `Print Inspector.exe` was copied without `_internal`, or antivirus removed a file. Extract the complete ZIP again. |
| "A component of the application could not be loaded (DLL load failed …)" | Extract again to a short path (e.g. `C:\Tools\Print Inspector`; very long paths can break DLL loading). Check antivirus quarantine. If it persists, install the Microsoft Visual C++ 2015–2022 Redistributable (x64) from your IT software store. |
| "This application failed to start because no Qt platform plugin could be initialized" | `_internal\PySide6\plugins\platforms` is missing or damaged: extract again. Remove the environment variables `QT_PLUGIN_PATH` and `QT_QPA_PLATFORM_PLUGIN_PATH` if other software set them (System → Advanced → Environment Variables). |
| "Windows protected your PC" (SmartScreen) | Unsigned program. Click More info → Run anyway, if the source is trusted. Sign the build to avoid this (section 10). |
| Antivirus reports the program as suspicious | False positives are common with PyInstaller programs. Verify the checksum, then ask IT to allow the folder or submit it to the vendor. Signing reduces this. |
| First start is slow (10–30 s) | Antivirus scans the files on first use. Later starts are faster. |
| "Text inspection (OCR) is not available" / text results always REVIEW | The build has no Tesseract (see `BUILD-INFO.txt`), or `_internal\tesseract` is missing. Use a build with OCR, or set a local Tesseract folder in Settings → Text. |
| "Your settings file was ignored because it is invalid" | `%LOCALAPPDATA%\PrintInspector\config.json` is damaged. Delete it (the defaults are used) and set your settings again. |
| A profile is listed in red / "Cannot be used" | The profile file is damaged or from an incompatible version. Delete it or import a good copy. |
| Saving results fails | No write permission in the chosen folder, or the disk is full. Choose another folder. |
| The self-test FAILED | Open `self_test.txt` in the folder shown. Each failed check gives the reason. Reinstall (extract the complete ZIP again) and run it again. If it still fails, send `self_test.txt` and the logs to whoever built the application. |
| Text is tiny or blurry on a high-resolution screen | Windows display scaling is supported. Check Settings → Display → Scale, and sign out and in again after changing it. |

### On the build computer

| Symptom | Cause and solution |
|---------|--------------------|
| "Python 3.11 not found" / "Python 3.11 64-bit is required" | Install Python 3.11 x64 from python.org with the py launcher, or pass `-Python C:\Path\python.exe`. |
| "running scripts is disabled on this system" | Use `build_windows.bat`, which starts PowerShell with `-ExecutionPolicy Bypass` for this script only. |
| Installing the requirements fails | No internet access or a proxy: use `-Wheelhouse` (section 10). |
| Tests fail | The build stops. Fix the cause first; the output names the failing tests. `-SkipTests` is for development builds only. |
| "No Tesseract to bundle" (with `-Release`) | Install Tesseract and pass `-TesseractSource`, or build without `-Release`. |
| "MISSING licence text(s)" (with `-Release`) | Put `LGPL-3.0.txt` and `GPL-3.0.txt` in `packaging\licenses\`, or run once with `-FetchLicenseTexts`. |
| PyInstaller succeeds, but the self-test of the built program fails with "could not be loaded" / "No module named …" | A module is imported in a way PyInstaller cannot see. Add it to `HIDDEN_IMPORTS` in `packaging\bundle_rules.py` and build again. `build\pyinstaller\print_inspector\warn-print_inspector.txt` lists the modules PyInstaller could not find. |
| The self-test of the built program shows "Text inspection SKIPPED" | The build has no Tesseract. That is expected with `-NoTesseract`, and an error with `-Release`. |
| `ISCC.exe not found` | Install Inno Setup 6, or build without `-Installer`. |

## 12. Release checklist

Fill this in for every version you distribute. Keep it with the ZIP.

| Item | Result |
|------|--------|
| Version / build date (`BUILD-INFO.txt`) | |
| Build command used | |
| Test suite on the build computer: passed / failed / skipped | |
| Self-test of the built program (`SELF-TEST-AT-BUILD.txt`): all checks PASS, none skipped | |
| Clean computer (Windows Sandbox or clean VM): Windows version, Python absent confirmed | |
| Self-test on the clean computer **with networking disabled** | |
| Real labels inspected on the clean computer (artwork / photo / verdict as expected) | |
| `check_network.ps1` (self-test and interactive): no connection attempts | |
| Licences: `licenses\INDEX.txt` complete, LGPL/GPL texts present, Qt attributions added, LGPL source offer arranged | |
| ZIP SHA-256 recorded | |
| Signed (if required) | |
