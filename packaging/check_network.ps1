<#
.SYNOPSIS
    Shows whether Print Inspector (or its Tesseract OCR) attempts any network connection, using Windows' own
    connection auditing. Run as administrator on the computer where Print Inspector is installed.

.DESCRIPTION
    1. Switches on the Windows audit of network connections ("Filtering Platform Connection"; events 5156 =
       connection allowed, 5157 = connection blocked) and, unless -NoBlock, adds firewall rules that block all
       outbound traffic of Print Inspector.exe and tesseract.exe.
    2. Runs the application: its self-test (default), or the normal application with -Interactive, in which
       case you inspect labels as usual and close the window when done.
    3. Lists every connection Windows recorded for those two programs during the run.
    4. Restores the previous audit setting and removes the firewall rules.

    Expected result: "No network connection attempts ... were recorded."

    Usage (administrator PowerShell):
        powershell -ExecutionPolicy Bypass -File check_network.ps1 -AppDir "C:\Program Files\Print Inspector"
        powershell -ExecutionPolicy Bypass -File check_network.ps1 -AppDir "..." -Interactive

.PARAMETER AppDir
    The Print Inspector folder (containing "Print Inspector.exe"). Default: ..\dist\Print Inspector.
.PARAMETER Interactive
    Start the application normally instead of the self-test; the check ends when you close it.
.PARAMETER NoBlock
    Do not add blocking firewall rules (audit only: shows allowed connections as event 5156).
#>
#Requires -RunAsAdministrator
[CmdletBinding()]
param(
    [string]$AppDir = (Join-Path (Split-Path -Parent $PSScriptRoot) "dist\Print Inspector"),
    [switch]$Interactive,
    [switch]$NoBlock
)

$ErrorActionPreference = "Stop"
$Exe = Join-Path $AppDir "Print Inspector.exe"
if (-not (Test-Path $Exe)) { throw "Print Inspector.exe not found in $AppDir" }
$Programs = @($Exe) + @(Get-ChildItem -Path $AppDir -Recurse -Filter "tesseract.exe" -ErrorAction SilentlyContinue |
    ForEach-Object { $_.FullName })
# "Filtering Platform Connection" by GUID: the subcategory name is translated on non-English Windows.
$Subcategory = "{0CCE9226-69AE-11D9-BED3-505054503030}"
$RuleGroup = "Print Inspector network check"

# Remember the current audit setting ("No Auditing", "Success", "Failure" or "Success and Failure").
$Before = (auditpol /get /subcategory:$Subcategory /r | ConvertFrom-Csv)."Inclusion Setting"
auditpol /set /subcategory:$Subcategory /success:enable /failure:enable | Out-Null
if (-not $NoBlock) {
    foreach ($Program in $Programs) {
        New-NetFirewallRule -DisplayName "$RuleGroup - block $(Split-Path -Leaf $Program)" -Group $RuleGroup `
            -Direction Outbound -Program $Program -Action Block | Out-Null
    }
}
$Start = Get-Date
Write-Host "Watching: $($Programs -join ', ')"
try {
    if ($Interactive) {
        Write-Host "Use Print Inspector as usual (load images, inspect, save results, profiles). Close it to finish."
        Start-Process -FilePath $Exe -Wait
    } else {
        $Output = Join-Path $env:TEMP ("PrintInspector-network-check-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
        Write-Host "Running the self-test (results in $Output) ..."
        $Process = Start-Process -FilePath $Exe -ArgumentList "--self-test --quiet --output `"$Output`"" -Wait -PassThru
        $Summary = Join-Path $Output "self_test.txt"
        if (Test-Path $Summary) { Get-Content $Summary | Select-Object -Last 1 | Write-Host }
        Write-Host "Self-test exit code: $($Process.ExitCode)"
    }
    Start-Sleep -Seconds 2  # let the event log catch up
} finally {
    $Flags = @{ "No Auditing" = @("disable", "disable"); "Success" = @("enable", "disable");
                "Failure" = @("disable", "enable"); "Success and Failure" = @("enable", "enable") }
    $Restore = if ($Flags.ContainsKey($Before)) { $Flags[$Before] } else { @("disable", "disable") }
    auditpol /set /subcategory:$Subcategory /success:$($Restore[0]) /failure:$($Restore[1]) | Out-Null
    if (-not $NoBlock) { Remove-NetFirewallRule -Group $RuleGroup -ErrorAction SilentlyContinue }
}

$Names = $Programs | ForEach-Object { [regex]::Escape((Split-Path -Leaf $_).ToLower()) }
$Pattern = "(" + ($Names -join "|") + ")"
$Events = @(Get-WinEvent -FilterHashtable @{ LogName = "Security"; Id = 5156, 5157; StartTime = $Start } `
        -ErrorAction SilentlyContinue | Where-Object { $_.Message.ToLower() -match $Pattern })
if ($Events.Count -eq 0) {
    Write-Host "`nNo network connection attempts by Print Inspector or Tesseract were recorded." -ForegroundColor Green
    exit 0
}
Write-Host "`n$($Events.Count) connection event(s) recorded:" -ForegroundColor Yellow
foreach ($Event in $Events) {
    $Data = @{}
    ([xml]$Event.ToXml()).Event.EventData.Data | ForEach-Object { $Data[$_.Name] = $_.'#text' }
    $Kind = if ($Event.Id -eq 5157) { "BLOCKED" } else { "allowed" }
    $Local = if ($Data["DestAddress"] -match "^(127\.|::1$)") { " (loopback, stays on this computer)" } else { "" }
    Write-Host ("{0:HH:mm:ss} {1,-8} {2} -> {3}:{4} protocol {5}{6}" -f $Event.TimeCreated, $Kind,
        (Split-Path -Leaf $Data["Application"]), $Data["DestAddress"], $Data["DestPort"], $Data["Protocol"], $Local)
}
exit 1
