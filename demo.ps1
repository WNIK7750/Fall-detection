param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8765,
    [string]$SourceVideo = "clips/s01_1050_90.mp4",
    [switch]$NoLLM,
    [switch]$NoBrowser,
    [switch]$NoRealtime
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$Python = "python"
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Python = $VenvPython
}

$ServerUrl = "ws://$HostName`:$Port/ws/events"
$DashboardUrl = "http://$HostName`:$Port/"

Write-Host "Starting monitor server on $DashboardUrl"
$ServerJob = Start-Job -Name "fall-demo-monitor" -ScriptBlock {
    param($Root, $Python, $HostName, $Port)
    Set-Location $Root
    & $Python monitor_server.py --host $HostName --port $Port
} -ArgumentList $Root, $Python, $HostName, $Port

try {
    $Ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        try {
            Invoke-WebRequest -Uri $DashboardUrl -UseBasicParsing -TimeoutSec 1 | Out-Null
            $Ready = $true
            break
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $Ready) {
        throw "Monitor server did not become ready at $DashboardUrl"
    }

    if (-not $NoBrowser) {
        Start-Process $DashboardUrl
    }

    $ArgsList = @(
        "playmode_event_loop.py",
        "--source-video", $SourceVideo,
        "--server-url", $ServerUrl,
        "--stop-after-event"
    )
    if ($NoLLM) {
        $ArgsList += "--no-llm"
    }
    if (-not $NoRealtime) {
        $ArgsList += "--realtime"
    }

    Write-Host "Running playmode client..."
    & $Python @ArgsList
    Write-Host ""
    Write-Host "Demo complete. Dashboard remains open while this script is running."
    Write-Host "Press Enter to stop the monitor server."
    [void][Console]::ReadLine()
}
finally {
    Stop-Job $ServerJob -ErrorAction SilentlyContinue | Out-Null
    Remove-Job $ServerJob -Force -ErrorAction SilentlyContinue | Out-Null
}
