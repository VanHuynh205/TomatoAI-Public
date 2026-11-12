# Windows x64 bootstrap. Never downloads weights or invokes training.
[CmdletBinding()]
param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
try {
    if ([Environment]::OSVersion.Platform -ne 'Win32NT' -or -not [Environment]::Is64BitOperatingSystem) {
        throw 'Tomato Desktop requires Windows x64. macOS and Linux desktop hosts are not supported.'
    }
    $bundle = Join-Path $PSScriptRoot 'ai/artifacts/dinov2-v1-280'
    $required = @('classifier.safetensors', 'config.json', 'dinov2-release.json', 'leaf-classifier.json', 'model.safetensors', 'preprocessor_config.json')
    $missing = @($required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $bundle $_) -PathType Leaf) })
    if ($missing.Count) {
        throw ('Missing authorized DINOv2 v1 bundle files: ' + ($missing -join ', ') + '. Download dinov2-v1-280-model.zip from GitHub Release v0.1.0-dinov2-v1-280 and extract it into ai/artifacts; see README.md and MODEL_CARD.md. Weights are release assets, not Git blobs. Training-data rights remain unresolved.')
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw 'Install uv from https://docs.astral.sh/uv/getting-started/installation/ and rerun this script.'
    }
    Push-Location $PSScriptRoot
    try {
        $python = Join-Path $PSScriptRoot '.venv-runtime/Scripts/python.exe'
        if (-not $CheckOnly) {
            if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
                & uv venv .venv-runtime --python 3.12.14
                if ($LASTEXITCODE -ne 0) { throw 'uv venv failed.' }
            }
            & uv pip sync --python $python --torch-backend cpu desktop/requirements-runtime-cpu.lock
            if ($LASTEXITCODE -ne 0) { throw 'CPU dependency installation failed.' }
        }
        if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
            throw 'Runtime environment missing. Run ./Start.ps1 without -CheckOnly first.'
        }
        & $python -c "import struct,sys; assert sys.version_info[:3] == (3,12,14) and struct.calcsize('P') == 8, 'CPython 3.12.14 x64 required'; from pathlib import Path; from ai.dinov2_inference import validate_dinov2_export; from desktop.release import validate_care_export; from desktop.prerequisites import require_webview2; print(validate_dinov2_export(Path('ai/artifacts/dinov2-v1-280'))); validate_care_export(Path('infra/knowledge'), Path('infra/knowledge')); require_webview2(); print('Runtime inputs verified.')"
        if ($LASTEXITCODE -ne 0) { throw 'Preflight failed. Check the bundle, care files, Python version and WebView2 prerequisites.' }
        if (-not $CheckOnly) {
            & $python -m desktop
            if ($LASTEXITCODE -ne 0) { throw 'Tomato Desktop exited with an error.' }
        }
    } finally { Pop-Location }
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
