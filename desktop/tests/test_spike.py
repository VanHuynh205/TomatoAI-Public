"""D1 boundaries: real resource staging; mocked Windows/runtime GUI only."""

import ctypes
import importlib
import subprocess
import sys
import threading
import winreg
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self
from unittest.mock import Mock

import pytest


def test_spike_import_does_not_require_api_or_ml_packages() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "class NoApplicationPackages:\n"
                " def find_spec(self, fullname, *args):\n"
                "  if fullname.split('.')[0] in {'ai', 'backend', 'torch', 'fastapi', 'uvicorn'}:\n"
                "   raise ImportError('D1 must not import ' + fullname)\n"
                "sys.meta_path.insert(0, NoApplicationPackages())\n"
                "import desktop.spike\n"
            ),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_missing_webview2_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = importlib.import_module("desktop.launcher")
    monkeypatch.setattr(winreg, "OpenKey", Mock(side_effect=FileNotFoundError))
    download = Mock(side_effect=AssertionError("Runtime must never download"))
    monkeypatch.setattr("urllib.request.urlopen", download)
    with pytest.raises(RuntimeError, match="WebView2.*offline"):
        launcher.require_webview2()
    download.assert_not_called()


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[Any, str, str], Any]:
    values: dict[tuple[Any, str, str], Any] = {}

    class Key:
        def __init__(self, hive: Any, path: str) -> None:
            self.hive, self.path = hive, path

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            pass

    def query(key: Key, name: str) -> tuple[Any, int]:
        try:
            return values[key.hive, key.path, name], winreg.REG_SZ
        except KeyError:
            raise FileNotFoundError from None

    monkeypatch.setattr(winreg, "OpenKey", Key)
    monkeypatch.setattr(winreg, "QueryValueEx", query)
    values[
        winreg.HKEY_LOCAL_MACHINE,
        r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full",
        "Release",
    ] = 528449
    return values


@pytest.mark.parametrize(
    "hive,path",
    [
        (
            winreg.HKEY_CURRENT_USER,
            r"SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
        ),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
        ),
    ],
)
def test_installed_evergreen_runtime_is_accepted(
    registry: dict[Any, Any], hive: Any, path: str
) -> None:
    registry[hive, path, "pv"] = "156.0.4314.50"
    importlib.import_module("desktop.launcher").require_webview2()


@pytest.mark.parametrize("version", ["0.0.0.0", "85.0.0.0", "broken", ""])
def test_unusable_runtime_is_rejected(registry: dict[Any, Any], version: str) -> None:
    registry[
        winreg.HKEY_CURRENT_USER,
        r"SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
        "pv",
    ] = version
    with pytest.raises(RuntimeError, match="WebView2"):
        importlib.import_module("desktop.launcher").require_webview2()


def test_dotnet_requirement_fails_before_gui(registry: dict[Any, Any]) -> None:
    registry[
        winreg.HKEY_LOCAL_MACHINE,
        r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full",
        "Release",
    ] = 0
    with pytest.raises(RuntimeError, match=r"\.NET"):
        importlib.import_module("desktop.launcher").require_webview2()


@pytest.mark.parametrize(
    "missing,expected",
    [
        (
            "dotnet",
            (
                "WebView2 cần .NET Framework 4.6.2 trở lên. "
                "Hãy dùng bộ cài .NET offline phù hợp rồi mở lại ứng dụng."
            ),
        ),
        (
            "webview2",
            (
                "Không tìm thấy Microsoft Edge WebView2 Runtime phù hợp. "
                "Hãy dùng bộ cài WebView2 Evergreen Standalone offline x64 rồi mở lại ứng dụng. "
                "Ứng dụng không tự tải hoặc cài runtime."
            ),
        ),
    ],
)
def test_main_preserves_prerequisite_instructions_before_startup(
    registry: dict[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
    expected: str,
) -> None:
    launcher = importlib.import_module("desktop.launcher")
    import webview

    if missing == "dotnet":
        registry[
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full",
            "Release",
        ] = 0
    messages: list[str] = []
    bind, compose, server, create, start = (Mock() for _ in range(5))
    monkeypatch.setattr(launcher, "_show_startup_error", messages.append)
    monkeypatch.setattr(launcher, "bind_loopback_socket", bind)
    monkeypatch.setattr(launcher, "create_desktop_app", compose)
    monkeypatch.setattr(launcher.uvicorn, "Server", server)
    monkeypatch.setattr(webview, "create_window", create)
    monkeypatch.setattr(webview, "start", start)
    assert launcher.main() == 1
    assert messages == [expected]
    for operation in (bind, compose, server, create, start):
        operation.assert_not_called()


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    spike = importlib.import_module("desktop.spike")
    import webview

    guilib = importlib.import_module("webview.guilib")

    require = Mock()
    error = Mock()
    initialize = Mock(return_value=SimpleNamespace(renderer="edgechromium"))
    create = Mock()
    start = Mock()
    monkeypatch.setattr(spike, "require_webview2", require)
    monkeypatch.setattr(spike, "_show_startup_error", error)
    monkeypatch.setattr(guilib, "initialize", initialize)
    monkeypatch.setattr(webview, "create_window", create)
    monkeypatch.setattr(webview, "start", start)
    # Restore sys.modules after main's process-level fail-closed import guard.
    monkeypatch.setitem(sys.modules, "webview.platforms.mshtml", None)
    return SimpleNamespace(
        spike=spike,
        require=require,
        error=error,
        initialize=initialize,
        create=create,
        start=start,
    )


def test_main_stages_only_fixtures_and_uses_local_http(
    host: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    staged: list[Path] = []

    def create(title: str, url: str, **kwargs: Any) -> None:
        index = Path(url)
        staged.append(index.parent)
        assert threading.current_thread() is threading.main_thread()
        assert title == "Tomato — thử đóng gói"
        assert sorted(item.name for item in index.parent.iterdir()) == [
            "spike.html",
            "spike.png",
        ]
        assert "Bản thử đóng gói – chưa có mô hình bệnh" in index.read_text(
            encoding="utf-8"
        )
        assert (
            (index.parent / "spike.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        )
        assert "js_api" not in kwargs

    host.create.side_effect = create
    assert host.spike.main() == 0
    host.start.assert_called_once_with(gui="edgechromium", http_server=True)
    host.error.assert_not_called()
    assert staged and not staged[0].exists()


def test_missing_runtime_never_initializes_or_creates_gui(
    host: SimpleNamespace,
) -> None:
    host.require.side_effect = RuntimeError("WebView2: dùng bộ cài offline")
    assert host.spike.main() == 1
    host.initialize.assert_not_called()
    host.create.assert_not_called()
    host.start.assert_not_called()
    assert "offline" in host.error.call_args.args[0]


@pytest.mark.parametrize(
    "missing",
    ["frontend/tests/fixtures/spike.html", "desktop/tests/fixtures/spike.png"],
)
def test_missing_bundle_asset_fails_before_gui(
    host: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing: str
) -> None:
    root = Path(__file__).resolve().parents[2]
    import shutil

    for relative in (
        "frontend/tests/fixtures/spike.html",
        "desktop/tests/fixtures/spike.png",
    ):
        if relative != missing:
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, target)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert host.spike.main() == 1
    host.create.assert_not_called()
    assert Path(missing).name in host.error.call_args.args[0]


def test_bundle_paths_do_not_depend_on_cwd(
    host: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(sys, "_MEIPASS", str(root), raising=False)
    monkeypatch.chdir(tmp_path)
    assert host.spike.main() == 0
    host.start.assert_called_once()


def test_startup_exception_cleans_staging_and_returns_failure(
    host: SimpleNamespace,
) -> None:
    host.start.side_effect = RuntimeError("fixture startup failure")
    assert host.spike.main() == 1
    index = Path(host.create.call_args.kwargs["url"])
    assert not index.parent.exists()
    assert "fixture startup failure" in host.error.call_args.args[0]


def test_no_legacy_renderer_fallback(host: SimpleNamespace) -> None:
    host.initialize.return_value.renderer = "mshtml"
    assert host.spike.main() == 1
    host.create.assert_not_called()
    host.start.assert_not_called()
    assert "WebView2" in host.error.call_args.args[0]


def test_worker_thread_does_not_open_gui(host: SimpleNamespace) -> None:
    results: list[int] = []
    worker = threading.Thread(target=lambda: results.append(host.spike.main()))
    worker.start()
    worker.join(timeout=5)
    assert results == [1]
    host.require.assert_not_called()
    host.create.assert_not_called()


def test_error_dialog_uses_windows_without_webview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spike = importlib.import_module("desktop.spike")
    message_box = Mock(return_value=1)
    monkeypatch.setattr(ctypes.windll.user32, "MessageBoxW", message_box)
    spike._show_startup_error("WebView2: dùng bộ cài offline")
    args = message_box.call_args.args
    assert args[0] is None
    assert "WebView2: dùng bộ cài offline" in args[1]
    assert args[3] & 0x10
