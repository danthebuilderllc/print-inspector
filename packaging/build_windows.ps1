<#
.SYNOPSIS
    Builds the offline Windows application "Print Inspector" (folder, ZIP and optional installer).

.DESCRIPTION
    Run from any folder on a 64-bit Windows 10/11 build computer with Python 3.11 (64-bit) installed:

        powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -TesseractSource "C:\Program Files\Tesseract-OCR"

    Steps:
      1. creates a separate build environment (.venv-build) and installs requirements.txt
         (from PyPI, or from a local wheel folder with -Wheelhouse for an offline build computer)
      2. runs the test suite
      3. copies Tesseract OCR and its language data into tesseract\ (tools\bundle_tesseract.py)
      4. builds dist\Print Inspector\ with PyInstaller (packaging\print_inspector.spec)
      5. adds licence texts, the installation guide and a build record
      6. runs the built application's self-test (Print Inspector.exe --self-test)
      7. writes dist\PrintInspector-<version>-win64.zip and its SHA-256 checksum
      8. optionally builds an installer with Inno Setup 6 (-Installer)

    Only the build computer needs internet access (for step 1, unless -Wheelhouse is used, and for
    -FetchLicenseTexts). The built application never downloads anything.

.PARAMETER Python
    Python 3.11 (64-bit) to use, e.g. "C:\Python311\python.exe". Default: the "py -3.11" launcher.
.PARAMETER Wheelhouse
    Folder with all required wheel files (pip download -r requirements.txt -d wheels). Installs without internet.
.PARAMETER TesseractSource
    Tesseract installation to bundle (folder containing tesseract.exe). Default: an existing tesseract\ folder.
.PARAMETER Languages
    Tesseract languages to bundle, e.g. "eng+deu". Default: from config\default_config.json.
.PARAMETER NoTesseract
    Build without OCR on purpose: the application then reports text inspection as not available (REVIEW).
.PARAMETER SkipTests
    Do not run the test suite before building (not allowed with -Release).
.PARAMETER Release
    Strict mode for a build that will be distributed: tests, Tesseract and the GNU licence texts are
    required, and every self-test check must pass (none skipped).
.PARAMETER FetchLicenseTexts
    Download the official LGPL-3.0 and GPL-3.0 texts from gnu.org into packaging\licenses (once).
.PARAMETER Installer
    Also build a setup program with Inno Setup 6 (must be installed; https://jrsoftware.org).
.PARAMETER Clean
    Recreate the build environment and remove previous build output.
#>
[CmdletBinding()]
param(
    [string]$Python = "",
    [string]$Wheelhouse = "",
    [string]$TesseractSource = "",
    [string]$Languages = "",
    [switch]$NoTesseract,
    [switch]$SkipTests,
    [switch]$Release,
    [switch]$FetchLicenseTexts,
    [switch]$Installer,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Paths given on the command line are relative to the folder the build was started from.
if ($Wheelhouse) { $Wheelhouse = (Resolve-Path $Wheelhouse).Path }
if ($TesseractSource) { $TesseractSource = (Resolve-Path $TesseractSource).Path }
if ($Python -and (Test-Path $Python)) { $Python = (Resolve-Path $Python).Path }

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$AppName = "Print Inspector"
$Venv = Join-Path $Root ".venv-build"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
$Dist = Join-Path $Root "dist"
$BuildDir = Join-Path $Root "build"
$AppDir = Join-Path $Dist $AppName
$Exe = Join-Path $AppDir "$AppName.exe"
$Tesseract = Join-Path $Root "tesseract"
$LicenceTexts = Join-Path $Root "packaging\licenses"

function Write-Step([string]$Text) { Write-Host "`n=== $Text" -ForegroundColor Cyan }
function Fail([string]$Text) { Write-Host "BUILD FAILED: $Text" -ForegroundColor Red; exit 1 }

# Runs a program and stops the build if it fails (PowerShell does not do this for native programs).
function Invoke-Checked([string]$Program, [string[]]$Arguments, [string]$What) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { Fail "$What (exit code $LASTEXITCODE)" }
}

if ($Release -and $SkipTests) { Fail "-Release runs the tests; do not combine it with -SkipTests." }
if ($Release -and $NoTesseract) { Fail "-Release requires Tesseract; do not combine it with -NoTesseract." }
if (-not [Environment]::Is64BitOperatingSystem) { Fail "A 64-bit Windows is required." }

# ------------------------------------------------------------------ 1. build environment
Write-Step "Python and build environment"
if ($Python) {
    $BasePython = @($Python)
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $BasePython = @("py", "-3.11")
} else {
    Fail "Python 3.11 not found. Install it from python.org (64-bit) or pass -Python <path to python.exe>."
}
$BaseExe = $BasePython[0]
$BaseArgs = @($BasePython | Select-Object -Skip 1)
# No quotes inside the code: Windows PowerShell 5.1 mangles quotes in arguments of native programs.
$Check = & $BaseExe @BaseArgs -c 'import sys; print(sys.version_info[0], sys.version_info[1], sys.maxsize > 2**32)'
if ($LASTEXITCODE -ne 0) { Fail "Could not run Python ($($BasePython -join ' '))." }
if ("$Check".Trim() -ne "3 11 True") { Fail "Python 3.11 64-bit is required, found (major minor 64-bit): $Check" }

if ($Clean) {
    foreach ($Folder in @($Venv, $BuildDir, $AppDir)) {
        if (Test-Path $Folder) { Remove-Item -Recurse -Force $Folder }
    }
}
if (-not (Test-Path $VenvPython)) {
    Invoke-Checked $BaseExe ($BaseArgs + @("-m", "venv", $Venv)) "Creating the build environment"
}
# Large downloads (PySide6 is several hundred MB): wait longer and retry on slow or unstable connections.
$PipArgs = @("-m", "pip", "install", "--disable-pip-version-check", "--timeout", "120", "--retries", "10",
    "-r", "requirements.txt")
if ($Wheelhouse) {
    if (-not (Test-Path $Wheelhouse)) { Fail "Wheel folder not found: $Wheelhouse" }
    $PipArgs += @("--no-index", "--find-links", $Wheelhouse)
}
Invoke-Checked $VenvPython $PipArgs "Installing the requirements"
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
$Lock = Join-Path $BuildDir "requirements-lock.txt"
& $VenvPython -m pip freeze --disable-pip-version-check | Set-Content -Encoding UTF8 $Lock
$Version = (Get-Content -Raw -Encoding UTF8 "config\default_config.json" | ConvertFrom-Json).app.version
Write-Host "Print Inspector version $Version"

# ------------------------------------------------------------------ 2. tests
if (-not $SkipTests) {
    Write-Step "Test suite"
    Invoke-Checked $VenvPython @("-m", "pytest", "-q") "Tests failed"
} else {
    Write-Host "Tests skipped (-SkipTests)." -ForegroundColor Yellow
}

# ------------------------------------------------------------------ 3. Tesseract OCR
Write-Step "Tesseract OCR"
if ($NoTesseract) {
    if (Test-Path $Tesseract) { Remove-Item -Recurse -Force $Tesseract }
    Write-Host "Building WITHOUT OCR (-NoTesseract): text inspection will report 'not available' (REVIEW)." -ForegroundColor Yellow
} elseif ($TesseractSource) {
    $BundleArgs = @("tools\bundle_tesseract.py", "--source", $TesseractSource)
    if ($Languages) { $BundleArgs += @("--languages", $Languages) }
    Invoke-Checked $VenvPython $BundleArgs "Copying Tesseract"
} elseif (Test-Path (Join-Path $Tesseract "tesseract.exe")) {
    Write-Host "Using the existing tesseract\ folder."
} else {
    if ($Release) { Fail "No Tesseract to bundle. Pass -TesseractSource <folder> (or -NoTesseract without -Release)." }
    Write-Host "No Tesseract found: building without OCR. Pass -TesseractSource to include it." -ForegroundColor Yellow
}
$WithOcr = Test-Path (Join-Path $Tesseract "tesseract.exe")
$env:PRINT_INSPECTOR_REQUIRE_TESSERACT = if ($WithOcr) { "1" } else { "0" }

# ------------------------------------------------------------------ licence texts
if ($FetchLicenseTexts) {
    Write-Step "Downloading the GNU licence texts (build computer only)"
    New-Item -ItemType Directory -Force -Path $LicenceTexts | Out-Null
    $Texts = [ordered]@{ "lgpl-3.0.txt" = "LGPL-3.0.txt"; "gpl-3.0.txt" = "GPL-3.0.txt" }
    foreach ($Source in $Texts.Keys) {
        $Target = Join-Path $LicenceTexts $Texts[$Source]
        Invoke-WebRequest -UseBasicParsing -Uri "https://www.gnu.org/licenses/$Source" -OutFile $Target
        Write-Host "Saved $Target"
    }
}

# ------------------------------------------------------------------ 4. PyInstaller
Write-Step "Application icon"
$Icon = Join-Path $BuildDir "icon\print_inspector.ico"
Invoke-Checked $VenvPython @("packaging\make_icon.py", $Icon) "Creating the icon"
$env:PRINT_INSPECTOR_ICON = $Icon

Write-Step "PyInstaller"
if (Test-Path $AppDir) { Remove-Item -Recurse -Force $AppDir }
Invoke-Checked $VenvPython @("-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", $Dist,
    "--workpath", (Join-Path $BuildDir "pyinstaller"), "packaging\print_inspector.spec") "PyInstaller"
if (-not (Test-Path $Exe)) { Fail "$Exe was not created." }

# ------------------------------------------------------------------ 5. licences, guide, build record
Write-Step "Licences and documents"
$CollectArgs = @("packaging\collect_licenses.py", (Join-Path $AppDir "licenses"))
if ($Release) { $CollectArgs = @("packaging\collect_licenses.py", "--strict", (Join-Path $AppDir "licenses")) }
Invoke-Checked $VenvPython $CollectArgs "Collecting licences (see packaging\licenses\README.md)"
Copy-Item "docs\WINDOWS_DEPLOYMENT.md" (Join-Path $AppDir "INSTALLATION AND USAGE.md")
Copy-Item "THIRD_PARTY_NOTICES.md" (Join-Path $AppDir "THIRD_PARTY_NOTICES.md")
Copy-Item $Lock (Join-Path $AppDir "licenses\requirements-lock.txt")
$PyVersion = & $VenvPython -c "import sys; print(sys.version.split()[0])"
@(
    "$AppName $Version",
    "Built:        $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') on $env:COMPUTERNAME",
    "Python:       $PyVersion (64-bit)",
    "Tesseract:    $(if ($WithOcr) { 'included (tesseract\ folder)' } else { 'NOT included - text inspection reports REVIEW' })",
    "Packages:     licenses\requirements-lock.txt",
    "Self-test:    run '$AppName.exe --self-test' after installing"
) | Set-Content -Encoding UTF8 (Join-Path $AppDir "BUILD-INFO.txt")

# ------------------------------------------------------------------ 6. self-test of the built application
Write-Step "Self-test of the built application"
$SelfTestDir = Join-Path $BuildDir "selftest"
if (Test-Path $SelfTestDir) { Remove-Item -Recurse -Force $SelfTestDir }
$Process = Start-Process -FilePath $Exe -ArgumentList "--self-test --quiet --output `"$SelfTestDir`"" -Wait -PassThru
$Report = Join-Path $SelfTestDir "self_test.txt"
if (Test-Path $Report) { Get-Content -Encoding UTF8 $Report | Write-Host }
if ($Process.ExitCode -ne 0) {
    Fail "The self-test of the built application failed (exit code $($Process.ExitCode)). See $SelfTestDir and %LOCALAPPDATA%\PrintInspector\logs\startup_error.log"
}
$Result = Get-Content -Raw -Encoding UTF8 (Join-Path $SelfTestDir "self_test.json") | ConvertFrom-Json
$Skipped = @($Result.checks | Where-Object { $_.status -eq "SKIPPED" })
if ($Skipped.Count -gt 0) {
    $Names = ($Skipped | ForEach-Object { $_.name }) -join ", "
    if ($Release) { Fail "Release builds must pass every self-test check; skipped: $Names" }
    Write-Host "Skipped self-test checks: $Names" -ForegroundColor Yellow
}
Copy-Item $Report (Join-Path $AppDir "SELF-TEST-AT-BUILD.txt")

# ------------------------------------------------------------------ 7. ZIP and checksum
Write-Step "ZIP archive"
$Zip = Join-Path $Dist "PrintInspector-$Version-win64.zip"
if (Test-Path $Zip) { Remove-Item -Force $Zip }
Compress-Archive -Path $AppDir -DestinationPath $Zip -CompressionLevel Optimal
$Hash = (Get-FileHash -Algorithm SHA256 $Zip).Hash
"$Hash  $(Split-Path -Leaf $Zip)" | Set-Content -Encoding ASCII "$Zip.sha256"
Write-Host "$Zip`nSHA-256 $Hash"

# Windows Sandbox configuration: a clean Windows without Python and without network for the acceptance test.
$SandboxResults = Join-Path $Dist "sandbox-results"
New-Item -ItemType Directory -Force -Path $SandboxResults | Out-Null
$Wsb = Join-Path $Dist "PrintInspector-clean-test.wsb"
@"
<Configuration>
  <Networking>Disable</Networking>
  <vGPU>Disable</vGPU>
  <ClipboardRedirection>Disable</ClipboardRedirection>
  <MappedFolders>
    <MappedFolder>
      <HostFolder>$([Security.SecurityElement]::Escape($AppDir))</HostFolder>
      <SandboxFolder>C:\PrintInspector</SandboxFolder>
      <ReadOnly>true</ReadOnly>
    </MappedFolder>
    <MappedFolder>
      <HostFolder>$([Security.SecurityElement]::Escape($SandboxResults))</HostFolder>
      <SandboxFolder>C:\Results</SandboxFolder>
      <ReadOnly>false</ReadOnly>
    </MappedFolder>
  </MappedFolders>
  <LogonCommand>
    <Command>cmd.exe /c start "" "C:\PrintInspector\Print Inspector.exe" --self-test --output C:\Results\selftest</Command>
  </LogonCommand>
</Configuration>
"@ | Set-Content -Encoding UTF8 $Wsb
Write-Host "Clean-computer test: double-click $Wsb (Windows Sandbox; results in $SandboxResults)"

# ------------------------------------------------------------------ 8. installer (optional)
if ($Installer) {
    Write-Step "Installer (Inno Setup)"
    $Iscc = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    ) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
    if (-not $Iscc) { Fail "Inno Setup 6 (ISCC.exe) not found. Install it or build without -Installer." }
    Invoke-Checked $Iscc @("/Qp", "/DAppVersion=$Version", "/DSourceDir=$AppDir", "/DOutputDir=$Dist",
        "packaging\installer.iss") "Inno Setup"
    $Setup = Join-Path $Dist "PrintInspector-$Version-setup.exe"
    $SetupHash = (Get-FileHash -Algorithm SHA256 $Setup).Hash
    "$SetupHash  $(Split-Path -Leaf $Setup)" | Set-Content -Encoding ASCII "$Setup.sha256"
    Write-Host "$Setup`nSHA-256 $SetupHash"
}

Write-Step "Done"
Write-Host "Application folder: $AppDir"
Write-Host "Copy the ZIP (or the setup program) to the target computer; see docs\WINDOWS_DEPLOYMENT.md."
