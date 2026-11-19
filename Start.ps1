# Windows x64 bootstrap. Never downloads weights or invokes training.
[CmdletBinding()]
param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
try {
    if ([Environment]::OSVersion.Platform -ne 'Win32NT' -or -not [Environment]::Is64BitOperatingSystem) {
        throw 'Tomato Desktop requires Windows x64. macOS and Linux desktop hosts are not supported.'
    }
    $bundle = Join-Path $PSScriptRoot 'ai/artifacts/cnn-bigru-20261118-g2-convnext_tiny_axial_bigru'
    $required = @('cnn-gru-release.json', 'leaf-classifier.json', 'model.safetensors')
    $missing = @($required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $bundle $_) -PathType Leaf) })
    if ($missing.Count) {
        throw ('Missing trained G2 bundle files: ' + ($missing -join ', ') + '. Extract TomatoDesktop-G2-model.zip at the repository root. See README.md and MODEL_CARD.md for the expected paths.')
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
        & $python -c "import struct,sys; assert sys.version_info[:3] == (3,12,14) and struct.calcsize('P') == 8, 'CPython 3.12.14 x64 required'; from pathlib import Path; from ai.cnn_gru_inference import validate_cnn_gru_export; from desktop.server import DEFAULT_MODEL_BUNDLE; from desktop.release import validate_care_export; from desktop.prerequisites import require_webview2; assert Path(sys.argv[1]).resolve() == Path(DEFAULT_MODEL_BUNDLE).resolve(), 'Bootstrap bundle path differs from desktop runtime'; print(validate_cnn_gru_export(Path(DEFAULT_MODEL_BUNDLE), require_trained=True)['model_version']); validate_care_export(Path('infra/knowledge'), Path('infra/knowledge')); require_webview2(); print('Runtime inputs verified.')" $bundle
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
