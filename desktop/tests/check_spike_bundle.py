"""Post-build assertion check; run after PyInstaller, without starting a GUI."""

import sys
from pathlib import Path
from unittest.mock import patch

from PyInstaller.archive.readers import CArchiveReader  # type: ignore[import-untyped]
from webview import util


def check_bundle(root: Path) -> None:
    assert (root / "TomatoSpike.exe").is_file()
    internal = root / "_internal"
    assert (internal / "frontend/tests/fixtures/spike.html").is_file()
    assert (internal / "desktop/tests/fixtures/spike.png").is_file()
    with patch.object(util, "__file__", str(internal / "webview/util.py")):
        # edgechromium imports resolve all three directories unconditionally in 6.2.1.
        for architecture in ("win-arm64", "win-x64", "win-x86"):
            native = Path(util.interop_dll_path(architecture))
            assert (native / "WebView2Loader.dll").is_file()
    files = [item for item in root.rglob("*") if item.is_file()]
    assert not any(
        part in {"backend", "ai", "infra", "library", "torch", "torchvision"}
        for item in files
        for part in item.relative_to(root).parts
    )
    assert not any(
        item.suffix in {".pth", ".pt", ".onnx", ".jpg", ".jpeg", ".csv"}
        or "WebBrowserInterop" in item.name
        or item.name == "pywebview-android.jar"
        for item in files
    )
    archive = CArchiveReader(str(root / "TomatoSpike.exe"))
    pyz = archive.open_embedded_archive("PYZ.pyz")
    assert pyz is not None
    assert not any(
        name.split(".")[0] in {"backend", "ai", "torch", "torchvision", "pytest"}
        or name in {"webview.platforms.mshtml", "webview.platforms.cef"}
        for name in pyz.toc
    )
    print(
        f"Bundle checks passed: {len(files)} files; SDK paths and exclusions verified."
    )


if __name__ == "__main__":
    check_bundle(Path(sys.argv[1]).resolve())
