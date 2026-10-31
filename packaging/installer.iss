; Inno Setup 6 script for Print Inspector (optional; packaging\build_windows.ps1 -Installer runs it).
; Installs the PyInstaller folder, creates Start menu (and optionally desktop) shortcuts and an
; uninstaller. No internet access, no services, no registry settings besides the uninstall entry.
; Inspection data, settings and profiles in %LOCALAPPDATA%\PrintInspector are kept on uninstall.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\Print Inspector"
#endif
#ifndef OutputDir
  #define OutputDir "..\dist"
#endif

[Setup]
AppId={{6F1C2E7A-4B8D-4E21-9A57-3C0F5D8B1E42}
AppName=Print Inspector
AppVersion={#AppVersion}
AppVerName=Print Inspector {#AppVersion}
AppPublisher=Print Inspector
DefaultDirName={autopf}\Print Inspector
DefaultGroupName=Print Inspector
DisableProgramGroupPage=yes
; Per-user installation without administrator rights by default; "all users" can be chosen.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#OutputDir}
OutputBaseFilename=PrintInspector-{#AppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\Print Inspector.exe
InfoBeforeFile={#SourceDir}\THIRD_PARTY_NOTICES.md

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Print Inspector"; Filename: "{app}\Print Inspector.exe"
Name: "{group}\Print Inspector Self-Test"; Filename: "{app}\Print Inspector.exe"; Parameters: "--self-test"
Name: "{group}\Installation and Usage"; Filename: "{app}\INSTALLATION AND USAGE.md"
Name: "{autodesktop}\Print Inspector"; Filename: "{app}\Print Inspector.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Print Inspector.exe"; Description: "Start Print Inspector"; Flags: nowait postinstall skipifsilent
