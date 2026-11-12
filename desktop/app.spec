# D3 Windows x64 onedir. Build with the dedicated CPU environment and verify afterwards.
# ruff: noqa: F821 -- PyInstaller supplies the spec globals.
import re
import struct
import sys
from importlib.metadata import distribution, version
from pathlib import Path, PurePosixPath

from PyInstaller.utils.hooks import (
    collect_dynamic_libs,
    copy_metadata,
    get_package_paths,
)

root = Path(SPECPATH).resolve().parent
if sys.platform != 'win32' or struct.calcsize('P') != 8 or sys.version_info[:3] != (3, 12, 14):
    raise SystemExit('Build requires Windows x64 / CPython 3.12.14.')

sys.path.insert(0, str(root))
from desktop.release import validate_care_export
from desktop.server import DEFAULT_MODEL_BUNDLE
from ai.dinov2_inference import DINO_V2_FILES

validate_care_export(root / 'infra/knowledge', root / 'infra/knowledge')

lock_path = root / 'desktop/requirements-runtime-cpu.lock'
pins = dict(re.findall(r'^([A-Za-z0-9_.-]+)==([^\s]+)$', lock_path.read_text(), re.MULTILINE))
for name, expected in pins.items():
    if version(name) != expected:
        raise SystemExit(f'{name}: expected {expected}; use requirements-runtime-cpu.lock.')

datas = [
    (str(root / 'frontend/dist'), 'frontend'),
    (str(root / 'infra/knowledge/care-v1.json'), 'knowledge'),
    (str(root / 'infra/knowledge/allowlist.json'), 'knowledge'),
    (str(root / 'desktop/notices'), 'notices'),
    (str(root / 'LICENSE'), 'notices/project'),
    (str(root / 'THIRD_PARTY_NOTICES.md'), 'notices/project'),
    (str(root / 'NOTICE'), 'notices/project'),
    (str(root / 'DATASET_ATTRIBUTIONS.md'), 'notices/project'),
    (str(root / 'MODEL_CARD.md'), 'notices/project'),
    (str(root / 'infra/knowledge/CONTENT-LICENSE.md'), 'notices/care'),
]
datas += [(str(root / DEFAULT_MODEL_BUNDLE / name), 'model') for name in DINO_V2_FILES]
for source, _ in datas:
    if not Path(source).exists():
        raise SystemExit(f'Missing release input: {source}')

# Preserve the exact wheel's full license tree, including bundled native libraries.
for name in pins:
    dist = distribution(name)
    datas += copy_metadata(name)
    for entry in dist.files or ():
        relative = PurePosixPath(str(entry).replace('\\', '/'))
        if re.fullmatch(r'(licen[cs]e|copying|notice|authors|copyright)(\..*)?', relative.name, re.IGNORECASE):
            source = Path(dist.locate_file(entry))
            if not source.is_file():
                raise SystemExit(f'Missing dependency notice: {name}/{entry}')
            datas.append((str(source), str(PurePosixPath('notices/python') / name / relative.parent)))
    metadata = next(entry for entry in dist.files if str(entry).replace('\\', '/').endswith('.dist-info/METADATA'))
    datas.append((str(dist.locate_file(metadata)), f'notices/python/{name}'))
datas += copy_metadata('tomato-desktop')
python_license = Path(sys.base_prefix) / 'LICENSE.txt'
if not python_license.is_file():
    raise SystemExit('Missing CPython distribution LICENSE.txt.')
datas.append((str(python_license), 'notices/cpython'))
datas.append((str(lock_path), 'notices'))

# The pinned torchvision wheel uses stable extensions; the upstream hook still names _C.
_, torchvision_path = get_package_paths('torchvision')
binaries = collect_dynamic_libs('torchvision')
for name in ('_C_stable.pyd', 'image_stable.pyd'):
    source = Path(torchvision_path) / name
    if not source.is_file():
        raise SystemExit(f'Missing torchvision CPU extension: {name}')
    binaries.append((str(source), 'torchvision'))

a = Analysis(
    [str(root / 'desktop/__main__.py')],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        'webview.platforms.winforms', 'webview.platforms.edgechromium',
        'uvicorn.protocols.http.h11_impl', 'uvicorn.lifespan.on',
        'transformers.models.dinov2.configuration_dinov2',
        'transformers.models.dinov2.modeling_dinov2',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'webview.platforms.mshtml', 'webview.platforms.cef', 'webview.platforms.qt',
        'webview.platforms.gtk', 'webview.platforms.cocoa', 'webview.platforms.android',
        'ai.train', 'ai.evaluate', 'ai.experiments', 'ai.evaluation_protocol',
        'ai.benchmark', 'ai.split', 'desktop.spike',
        'desktop.tests', 'backend.tests', 'tests', 'pytest', 'mypy', 'ruff',
        'torch.testing._internal', 'torch._dynamo.test_case', 'torch._dynamo.testing',
        'torch._inductor.test_case', 'torch.distributed.testing', 'torch.fx.passes.tests',
        'IPython', 'notebook', 'scipy', 'pandas', 'tensorboard', 'pynvml',
        'ultralytics', 'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
    ],
    noarchive=False,
    optimize=0,
)
# Torch 2.14 imports these helpers during CPU initialization; keep the remaining test suite excluded.
_, torch_path = get_package_paths('torch')
for module, filename in (
    ('torch.testing._internal', '__init__.py'),
    ('torch.testing._internal.logging_tensor', 'logging_tensor.py'),
    ('torch.testing._internal.common_dtype', 'common_dtype.py'),
):
    a.pure.append((module, str(Path(torch_path) / 'testing/_internal' / filename), 'PYMODULE'))

# Keep all WebView2 architecture directories: edgechromium resolves them on import.
a.datas = [item for item in a.datas if not item[0].replace('\\', '/').startswith('torch/fx/passes/tests/') and not any(part in item[0] for part in (
    'pywebview-android.jar', 'WebBrowserInterop',
))]
a.binaries = [item for item in a.binaries if 'WebBrowserInterop' not in item[0]]
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name='TomatoDesktop',
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, disable_windowed_traceback=True, contents_directory='_internal',
)
coll = COLLECT(
    exe, a.binaries, a.datas, strip=False, upx=False, name='TomatoDesktop',
)
