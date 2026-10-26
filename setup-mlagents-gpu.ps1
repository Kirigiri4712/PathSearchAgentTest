$ErrorActionPreference = "Stop"

$environmentPath = Join-Path $PSScriptRoot ".venv-mlagents"
$pythonPath = Join-Path $environmentPath "Scripts\python.exe"

if (-not (Test-Path $pythonPath)) {
    & py -3.10 -m venv $environmentPath
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the ML-Agents virtual environment with Python 3.10."
    }
}

& $pythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Could not upgrade pip in the ML-Agents virtual environment."
}

# ML-Agents 1.1.0 supports PyTorch through 2.8.0. Use the official CUDA 12.8 wheel.
& $pythonPath -m pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) {
    throw "Could not install CUDA-enabled PyTorch 2.8.0."
}

& $pythonPath -m pip install "mlagents==1.1.0"
if ($LASTEXITCODE -ne 0) {
    throw "Could not install ML-Agents 1.1.0."
}

& $pythonPath -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "The ML-Agents virtual environment has dependency conflicts."
}

& $pythonPath -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not detected'); assert torch.cuda.is_available(), 'CUDA is unavailable to PyTorch'"
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch could not use the NVIDIA GPU."
}

Write-Host "GPU training environment is ready. Start training with:"
Write-Host ".venv-mlagents\Scripts\mlagents-learn.exe Assets/config/trainer.yaml --torch-device=cuda --run-id=<name>"
