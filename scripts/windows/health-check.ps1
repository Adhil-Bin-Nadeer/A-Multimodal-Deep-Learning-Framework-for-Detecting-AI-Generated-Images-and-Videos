param(
    [string]$BaseUrl = "http://127.0.0.1:7860"
)

$ErrorActionPreference = "Stop"

$healthUrl = "$BaseUrl/api/health"
Write-Host "Checking backend health at $healthUrl"

$response = Invoke-RestMethod -Uri $healthUrl -Method Get
$response | ConvertTo-Json -Depth 8
