$ErrorActionPreference = "Stop"

$environmentPython = Join-Path $PSScriptRoot ".venv-mlagents\Scripts\python.exe"
$pluginPath = Join-Path $PSScriptRoot "mlagents-gtrxl"

if (-not (Test-Path $environmentPython)) {
    throw "The ML-Agents environment was not found. Run setup-mlagents-gpu.ps1 first or create .venv-mlagents."
}

& $environmentPython -m pip install -e $pluginPath
if ($LASTEXITCODE -ne 0) {
    throw "Could not install the PathSearch GTrXL trainer plugin."
}

Write-Host "Installed the experimental GTrXL PPO trainer plugin."
