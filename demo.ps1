param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8765,
    [switch]$NoLLM,
    [switch]$NoBrowser
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
        "continuous_event_client.py",
        "--roadshow-loop",
        "--start-near-fall",
        "--pre-roll-frames", "300",
        "--clip-fps", "30",
        "--event-pre-frames", "90",
        "--event-post-frames", "151",
        "--event-clip-frames", "241",
        "--llm-sheet-frames", "9",
        "--llm-sheet-stride", "29",
        "--realtime",
        "--scenarios", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23",
        "--server-url", $ServerUrl,
        "--log-every", "120"
    )
    if ($NoLLM) {
        $ArgsList += "--no-llm"
    }

    Write-Host "Running live dataset camera loop..."
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
