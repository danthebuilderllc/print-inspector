@echo off
rem Builds the offline Windows application Print Inspector. All arguments are passed on, e.g.
rem   packaging\build_windows.bat -TesseractSource "C:\Program Files\Tesseract-OCR"
rem   packaging\build_windows.bat -Release -Installer
rem See packaging\build_windows.ps1 for every option.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_windows.ps1" %*
exit /b %ERRORLEVEL%
