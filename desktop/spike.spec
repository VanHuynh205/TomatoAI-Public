# D1 synthetic-only onedir; this is not the release specification.
from pathlib import Path

root = Path(SPECPATH).parent
a = Analysis(
    [str(root / 'desktop/spike.py')],
    pathex=[str(root)],
    binaries=[],
    datas=[
        (str(root / 'frontend/tests/fixtures/spike.html'), 'frontend/tests/fixtures'),
        (str(root / 'desktop/tests/fixtures/spike.png'), 'desktop/tests/fixtures'),
    ],
    hiddenimports=['webview.platforms.winforms', 'webview.platforms.edgechromium'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'webview.platforms.mshtml', 'webview.platforms.cef', 'webview.platforms.qt',
        'webview.platforms.gtk', 'webview.platforms.cocoa', 'webview.platforms.android',
        'backend', 'ai', 'torch', 'torchvision', 'PIL', 'pytest', 'mypy', 'ruff',
    ],
    noarchive=False,
    optimize=0,
)
# edgechromium resolves all three SDK native directories even on x64 at import.
# Remove only Android and legacy MSHTML assets, which this host never uses.
a.datas = [item for item in a.datas if not any(part in item[0] for part in (
    'pywebview-android.jar', 'WebBrowserInterop',
))]
a.binaries = [item for item in a.binaries if not any(part in item[0] for part in (
    'WebBrowserInterop',
))]
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name='TomatoSpike',
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, disable_windowed_traceback=True,
)
coll = COLLECT(
    exe, a.binaries, a.datas, strip=False, upx=False, name='TomatoSpike',
)
