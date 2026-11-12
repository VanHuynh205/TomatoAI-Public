"""D2 uses real sockets/API threads; GUI and OS browser effects are isolated."""

import importlib
import io
import socket
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Self, cast
from unittest.mock import Mock

import httpx
import pytest
from PIL import Image

from ai.contracts import Prediction
from backend.app import create_app
from backend.care import CareCatalog
from backend.tests.conftest import FakePredictor


def test_prebound_socket_and_two_instances() -> None:
    launcher = importlib.import_module("desktop.launcher")
    with (
        launcher.bind_loopback_socket() as first,
        launcher.bind_loopback_socket() as second,
    ):
        assert first.getsockname()[0] == second.getsockname()[0] == "127.0.0.1"
        assert first.getsockname()[1] != second.getsockname()[1]
        with socket.socket() as competitor, pytest.raises(OSError):
            competitor.bind(first.getsockname())


@pytest.mark.parametrize(
    "url",
    [
        "file:///C:/secret",
        "javascript:alert(1)",
        "data:text/html,hi",
        "//example.org",
        "https://name:password@example.org/",
        "https://example.org:0/",
        "https://example.org:bad/",
        "https://example.org\\@evil.org",
        "https://example.org\n/",
        "https://",
        "",
        " https://example.org",
    ],
)
def test_bridge_rejects_unsafe_url(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    launcher = importlib.import_module("desktop.launcher")
    opened = Mock()
    monkeypatch.setattr("webbrowser.open", opened)
    bridge = launcher.SessionBridge("session-one")
    with pytest.raises(ValueError):
        bridge.open_external_url(url)
    opened.assert_not_called()


def test_bridge_keeps_native_token_and_opens_safe_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = importlib.import_module("desktop.launcher")
    opened = Mock(return_value=True)
    monkeypatch.setattr("webbrowser.open", opened)
    bridge = launcher.SessionBridge("native-session")
    assert bridge.get_session_token() == "native-session"
    bridge.open_external_url("https://example.org/care#symptoms")
    opened.assert_called_once_with("https://example.org/care#symptoms", new=2)


class Event:
    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def __iadd__(self, handler: Any) -> Self:
        self.handlers.append(handler)
        return self

    def fire(self, *args: Any) -> None:
        for handler in self.handlers:
            handler(*args)


@pytest.fixture
def host_case(
    monkeypatch: pytest.MonkeyPatch,
    predictor: FakePredictor,
    care_catalog: CareCatalog,
) -> SimpleNamespace:
    launcher = importlib.import_module("desktop.launcher")
    import webview

    case = SimpleNamespace(
        app=None,
        windows=[],
        origin="",
        token=webview.token,
        sockets=[],
        threads=[],
        start=None,
    )
    monkeypatch.setattr(launcher, "require_webview2", lambda: None)
    monkeypatch.setattr(
        launcher, "_show_startup_error", lambda message: pytest.fail(message)
    )
    from contextlib import nullcontext

    monkeypatch.setattr(
        launcher, "_secure_webview", lambda origin: nullcontext(threading.Event())
    )

    original_bind = launcher.bind_loopback_socket

    def bind() -> socket.socket:
        sock = cast(socket.socket, original_bind())
        case.sockets.append(sock)
        return sock

    def compose(assets: Path, model: Path, care: Path, token: str, origin: str) -> Any:
        case.origin = origin
        case.app = create_app(
            predictor=predictor,
            care_catalog=care_catalog,
            session_token=token,
            allowed_origin=origin,
        )
        return case.app

    def create_window(title: str, **kwargs: Any) -> Any:
        assert threading.current_thread() is threading.main_thread()
        assert kwargs["url"] == case.origin
        assert kwargs["js_api"].get_session_token() == case.token
        window = SimpleNamespace(
            events=SimpleNamespace(closing=Event()), destroy=Mock()
        )
        case.windows.append(window)
        return window

    def start(**kwargs: Any) -> None:
        assert threading.current_thread() is threading.main_thread()
        assert kwargs["gui"] == "edgechromium"
        assert kwargs["http_server"] is False
        assert kwargs["private_mode"] is True
        case.threads = [t for t in threading.enumerate() if t.name == "tomato-api"]
        assert len(case.threads) == 1
        if case.start:
            case.start()
        case.windows[-1].events.closing.fire()

    monkeypatch.setattr(launcher, "bind_loopback_socket", bind)
    monkeypatch.setattr(launcher, "create_desktop_app", compose)
    monkeypatch.setattr(webview, "create_window", create_window)
    monkeypatch.setattr(webview, "start", start)
    return case


def test_three_open_close_cycles_are_clean(host_case: SimpleNamespace) -> None:
    launcher = importlib.import_module("desktop.launcher")

    def check_health() -> None:
        with httpx.Client(
            base_url=host_case.origin, timeout=2, trust_env=False
        ) as client:
            assert client.get("/api/v1/health").status_code == 401
            assert (
                client.get(
                    "/api/v1/health", headers={"X-Session-Token": "old-token"}
                ).status_code
                == 401
            )
            response = client.get(
                "/api/v1/health", headers={"X-Session-Token": host_case.token}
            )
            assert response.json() == {
                "schema_version": "1",
                "ready": True,
                "model_version": "fixture-model-v1",
                "knowledge_version": "fixture-care-v1",
                "reason": None,
            }

    host_case.start = check_health
    for _ in range(3):
        assert launcher.main() == 0
        assert all(not thread.is_alive() for thread in host_case.threads)
        assert host_case.sockets[-1].fileno() == -1
    assert len(host_case.windows) == 3


def test_source_host_selects_trained_yolo_by_default(
    host_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = importlib.import_module("desktop.launcher")
    compose = Mock(wraps=launcher.create_desktop_app)
    monkeypatch.setattr(launcher, "create_desktop_app", compose)

    assert launcher.main() == 0
    assert compose.call_args.args[1] == (
        launcher.resource_root() / "ai/artifacts/dinov2-v1-280"
    )


def test_source_host_accepts_explicit_model_directory(
    host_case: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = importlib.import_module("desktop.launcher")
    compose = Mock(wraps=launcher.create_desktop_app)
    monkeypatch.setattr(launcher, "create_desktop_app", compose)

    assert launcher.main(model_dir=tmp_path) == 0
    assert compose.call_args.args[1] == tmp_path


def test_frozen_host_uses_validated_internal_layout(
    host_case: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = importlib.import_module("desktop.launcher")
    install = tmp_path / "Cà chua có dấu"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(install / "_internal"), raising=False)
    monkeypatch.setattr(sys, "executable", str(install / "TomatoDesktop.exe"))
    check = Mock()
    monkeypatch.setattr(launcher, "validate_release", check)
    compose = Mock(wraps=launcher.create_desktop_app)
    monkeypatch.setattr(launcher, "create_desktop_app", compose)
    assert launcher.main(model_dir=tmp_path / "candidate") == 0
    check.assert_called_once_with(install)
    assert compose.call_args.args[:3] == (
        install / "_internal/frontend",
        install / "_internal/model",
        install / "_internal/knowledge/care-v1.json",
    )


def test_close_during_inference(
    host_case: SimpleNamespace, predictor: FakePredictor
) -> None:
    launcher = importlib.import_module("desktop.launcher")
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = predictor.predict

    def blocked(image: Image.Image) -> Prediction:
        entered.set()
        assert release.wait(5)
        result = original(image)
        finished.set()
        return result

    predictor.predict = blocked  # type: ignore[method-assign]
    encoded = io.BytesIO()
    Image.new("RGB", (4, 4)).save(encoded, format="PNG")

    def request() -> int:
        return httpx.post(
            host_case.origin + "/api/v1/predict",
            headers={"X-Session-Token": host_case.token, "Origin": host_case.origin},
            files={"image": ("leaf.png", encoded.getvalue(), "image/png")},
            timeout=6,
            trust_env=False,
        ).status_code

    with ThreadPoolExecutor(max_workers=1) as pool:
        results: list[Any] = []

        def close() -> None:
            results.append(pool.submit(request))
            assert entered.wait(3)
            host_case.windows[-1].events.closing.fire()
            assert host_case.app.state.runtime.stopping.is_set()
            assert host_case.app.state.runtime.slot.locked()
            release.set()

        host_case.start = close
        try:
            assert launcher.main() == 0
            assert results[0].result(timeout=3) == 200
        finally:
            release.set()
    assert finished.is_set()
    assert not host_case.app.state.runtime.slot.locked()
    assert all(not thread.is_alive() for thread in host_case.threads)


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("missing care"),
        ValueError("C:/private/model.pth secret-session"),
        RuntimeError("C:/private/care.json secret-token"),
    ],
)
def test_startup_failure_never_opens_business_window(
    host_case: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    launcher = importlib.import_module("desktop.launcher")
    messages: list[str] = []
    monkeypatch.setattr(launcher, "create_desktop_app", Mock(side_effect=failure))
    monkeypatch.setattr(launcher, "_show_startup_error", messages.append)
    assert launcher.main() == 1
    assert host_case.windows == []
    assert host_case.sockets[0].fileno() == -1
    assert messages == [
        (
            "Không thể mở ứng dụng. Kiểm tra WebView2, giao diện, mô hình và thẻ chăm sóc trong gói. "
            "Ứng dụng không tự tải tài nguyên."
        )
    ]


def test_navigation_and_session_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = importlib.import_module("desktop.launcher")
    import webview
    import webview.guilib
    import webview.platforms

    order: list[str] = []

    class NativeBase:
        def on_webview_ready(self, sender: Any, args: Any) -> None:
            assert len(sender.CoreWebView2.FrameNavigationStarting.handlers) == 1
            order.append("first navigation")
            sender.CoreWebView2.NewWindowRequested += self.on_new_window_request

        def on_new_window_request(self, sender: Any, args: Any) -> None:
            raise AssertionError("upstream handler must never open an external URL")

    native: Any = ModuleType("webview.platforms.edgechromium")
    native.EdgeChrome = NativeBase
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setattr(webview.platforms, "edgechromium", native, raising=False)
    monkeypatch.setattr(
        importlib.import_module("webview.guilib"),
        "initialize",
        lambda gui: SimpleNamespace(renderer="edgechromium"),
    )
    opener = Mock(
        side_effect=AssertionError("native navigation must not open OS browser")
    )
    monkeypatch.setattr("webbrowser.open", opener)
    settings = dict(webview.settings)
    with launcher._secure_webview("http://127.0.0.1:8765") as failed:
        edge = native.EdgeChrome()
        sender = SimpleNamespace(
            CoreWebView2=SimpleNamespace(
                FrameNavigationStarting=Event(), NewWindowRequested=Event()
            )
        )
        edge.on_webview_ready(sender, SimpleNamespace(IsSuccess=True))
        assert not failed.is_set()
        for url, expected_cancel in [
            ("http://127.0.0.1:8765/", False),
            ("http://127.0.0.1:8765/#sources", False),
            ("https://example.org", True),
            ("http://127.0.0.1:8766/", True),
            ("http://localhost:8765/", True),
            ("http://127.0.0.1:8765@evil.org/", True),
            ("file:///C:/index.html", True),
            ("data:text/html,<h1>outside</h1>", True),
        ]:
            navigation = SimpleNamespace(Uri=url, Cancel=False, IsRedirected=True)
            edge.on_navigation_start(sender, navigation)
            assert navigation.Cancel is expected_cancel
            sender.CoreWebView2.FrameNavigationStarting.fire(sender, navigation)
            assert navigation.Cancel is expected_cancel
        popup = SimpleNamespace(Uri="https://example.org/", Handled=False)
        sender.CoreWebView2.NewWindowRequested.fire(sender, popup)
        assert popup.Handled is True
        assert webview.settings["ALLOW_FILE_URLS"] is False
        assert webview.settings["ALLOW_DOWNLOADS"] is False
    assert native.EdgeChrome is NativeBase
    assert dict(webview.settings) == settings
    assert order == ["first navigation"]
    opener.assert_not_called()


def test_webview_initialization_failure_closes_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = importlib.import_module("desktop.launcher")
    import webview.guilib
    import webview.platforms

    class NativeBase:
        def on_webview_ready(self, sender: Any, args: Any) -> None:
            raise AssertionError("failed initialization must not navigate")

    native: Any = ModuleType("webview.platforms.edgechromium")
    native.EdgeChrome = NativeBase
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setattr(webview.platforms, "edgechromium", native, raising=False)
    monkeypatch.setattr(
        importlib.import_module("webview.guilib"),
        "initialize",
        lambda gui: SimpleNamespace(renderer="edgechromium"),
    )
    messages: list[str] = []
    monkeypatch.setattr(launcher, "_show_startup_error", messages.append)
    destroy = Mock()
    with launcher._secure_webview("http://127.0.0.1:8765") as failed:
        edge = native.EdgeChrome()
        edge.pywebview_window = SimpleNamespace(destroy=destroy)
        edge.on_webview_ready(None, SimpleNamespace(IsSuccess=False))
        assert failed.is_set()
    destroy.assert_called_once_with()
    assert messages == ["WebView2 không khởi tạo được. Hãy kiểm tra runtime offline."]


@pytest.mark.parametrize(
    "change", ["model", "knowledge", "schema", "unauthorized", "not_ready"]
)
def test_readiness_rejects_wrong_server(
    host_case: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    launcher = importlib.import_module("desktop.launcher")
    real_client = httpx.Client
    messages: list[str] = []
    health = {
        "schema_version": "1",
        "ready": True,
        "model_version": "fixture-model-v1",
        "knowledge_version": "fixture-care-v1",
        "reason": None,
    }
    if change in {"model", "knowledge"}:
        health[change + "_version"] = "wrong"
    if change == "schema":
        health["schema_version"] = "2"
    if change == "not_ready":
        health["ready"] = False
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            401 if change == "unauthorized" else 200, json=health
        )
    )
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: real_client(transport=transport, **kwargs)
    )
    monkeypatch.setattr(launcher, "_show_startup_error", messages.append)
    monkeypatch.setattr(launcher, "READINESS_TIMEOUT", 0.2)
    assert launcher.main() == 1
    assert not host_case.windows
    assert messages
    assert all(sock.fileno() == -1 for sock in host_case.sockets)


@pytest.mark.parametrize("worker_finishes", [True, False], ids=["drained", "stuck"])
def test_close_during_request_exits_host_and_releases_port(
    tmp_path: Path,
    care_catalog: CareCatalog,
    worker_finishes: bool,
    record_testsuite_property: Callable[[str, object], None],
) -> None:
    # Real host, Uvicorn and HTTP; the window and inference duration are controlled.
    marker = tmp_path / "port.txt"
    code = textwrap.dedent("""
        import io, os, sys, threading, time
        from contextlib import nullcontext
        from pathlib import Path
        from types import SimpleNamespace
        import httpx, webview
        from PIL import Image
        from backend.app import create_app
        from backend.care import CareCatalog
        from backend.tests.conftest import FakePredictor
        from desktop.tests.test_lifecycle import Event
        from desktop import launcher

        entered, release = threading.Event(), threading.Event()
        class ControlledPredictor(FakePredictor):
            def predict(self, image):
                entered.set()
                release.wait()
                return super().predict(image)

        window = SimpleNamespace(events=SimpleNamespace(closing=Event()))
        current = {}
        def compose(assets, model, care, token, origin):
            current.update(token=token, origin=origin)
            return create_app(predictor=ControlledPredictor(), care_catalog=CareCatalog.load(Path(sys.argv[1])), session_token=token, allowed_origin=origin)
        def request():
            body = io.BytesIO()
            Image.new('RGB', (4, 4)).save(body, format='PNG')
            response = httpx.post(current['origin'] + '/api/v1/predict', headers={'Origin': current['origin'], 'X-Session-Token': current['token']}, files={'image': ('leaf.png', body.getvalue(), 'image/png')}, timeout=20, trust_env=False)
            response.raise_for_status()
            current['result'] = response.json()['class_id']
        def start(**kwargs):
            client = threading.Thread(target=request, daemon=True)
            client.start()
            assert entered.wait(4)
            Path(sys.argv[2]).write_text(f"{os.getpid()} {current['origin'].rsplit(':', 1)[1]} {time.monotonic()}", encoding='utf8')
            window.events.closing.fire()
            if sys.argv[3] == 'drain':
                release.set()
                client.join(5)
                assert current.get('result') == 'healthy'
        launcher.require_webview2 = lambda: None
        launcher._secure_webview = lambda origin: nullcontext(threading.Event())
        launcher.create_desktop_app = compose
        launcher._show_startup_error = lambda message: sys.exit(5)
        webview.create_window = lambda *args, **kwargs: window
        webview.start = start
        sys.exit(launcher.main())
    """)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(tmp_path / "care.json"),
            str(marker),
            "drain" if worker_finishes else "stall",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    ended_at = time.monotonic()
    assert child.returncode == (0 if worker_finishes else 1), child.stderr
    pid, port, closed_at = marker.read_text().split()
    elapsed = ended_at - float(closed_at)
    if worker_finishes:
        assert elapsed < 10
    else:
        assert "API worker exceeded shutdown deadline." in child.stderr
        # Include OS process teardown and the parent's wait after the 10s deadline.
        assert 10 <= elapsed < 12
    with socket.socket() as probe, pytest.raises(ConnectionRefusedError):
        probe.settimeout(6)
        probe.connect(("127.0.0.1", int(port)))
    mode = "drained" if worker_finishes else "stuck"
    record_testsuite_property(f"{mode}_host_pid", pid)
    record_testsuite_property(f"{mode}_port", port)
    record_testsuite_property(f"{mode}_close_to_exit_seconds", round(elapsed, 3))
