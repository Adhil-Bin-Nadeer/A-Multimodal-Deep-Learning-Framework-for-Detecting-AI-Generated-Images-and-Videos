param(
    [int]$Port = 7860,
    [int]$Threads = 8
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$BackendDir = Join-Path $RepoRoot "backend"

if (-not (Test-Path $PythonExe)) {
    Write-Error "Python executable not found at $PythonExe"
}

if (-not (Test-Path $BackendDir)) {
    Write-Error "Backend directory not found at $BackendDir"
}

$env:PORT = "$Port"
$env:WAITRESS_THREADS = "$Threads"
Set-Location $BackendDir

Write-Host "Starting backend production server (Waitress) on port $Port with $Threads threads..."
Write-Host "Using Python: $PythonExe"

& $PythonExe "serve.py"
