"""Windows host with one retained loopback socket and bounded shutdown."""

import ctypes
import logging
import os
import socket
import sys
import threading
import time
import webbrowser
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import uvicorn

from backend.app import Runtime
from backend.schemas import HealthResponse
from backend.security import validate_boundary
from desktop.prerequisites import require_webview2
from desktop.release import validate_release
from desktop.server import DEFAULT_MODEL_BUNDLE, create_desktop_app

READINESS_TIMEOUT = 30.0
SHUTDOWN_TIMEOUT = 10.0


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[1]


def bind_loopback_socket() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind(("127.0.0.1", 0))
        sock.setblocking(False)
        return sock
    except BaseException:
        sock.close()
        raise


class SessionBridge:
    def __init__(self, token: str) -> None:
        self._token = token

    def get_session_token(self) -> str:
        return self._token

    def open_external_url(self, url: str) -> None:
        if (
            not isinstance(url, str)
            or not url
            or any(
                char.isspace() or ord(char) < 32 or ord(char) == 127 or char == "\\"
                for char in url
            )
        ):
            raise ValueError("Địa chỉ nguồn không hợp lệ.")
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port == 0
            or "%" in parsed.netloc
        ):
            raise ValueError("Chỉ mở nguồn HTTP(S) không chứa thông tin đăng nhập.")
        if not webbrowser.open(url, new=2):
            raise RuntimeError(
                "Không mở được trình duyệt. Hãy kiểm tra trình duyệt mặc định."
            )


def _same_origin(url: str, origin: str) -> bool:
    try:
        parsed = urlsplit(url)
        return (
            f"{parsed.scheme}://{parsed.netloc}" == origin
            and parsed.username is None
            and parsed.password is None
            and not any(char.isspace() or char == "\\" for char in url)
        )
    except ValueError:
        return False


@contextmanager
def _secure_webview(origin: str) -> Iterator[threading.Event]:
    import webview
    from webview.guilib import initialize

    if version("pywebview") != "6.2.1":
        raise RuntimeError("Cần pywebview 6.2.1 đã kiểm chứng cho host này.")
    # The pinned loader can otherwise fall back to MSHTML despite the requested GUI.
    sys.modules["webview.platforms.mshtml"] = None  # type: ignore[assignment]
    gui = initialize("edgechromium")
    if gui.renderer != "edgechromium":
        raise RuntimeError("Không thể khởi tạo WebView2; không dùng renderer cũ.")
    from webview.platforms import edgechromium

    original = edgechromium.EdgeChrome
    failed = threading.Event()

    # shortcut: native callbacks depend on pywebview 6.2.1; revalidate before upgrading.
    class LocalEdgeChrome(original):  # type: ignore[misc, valid-type]
        def on_navigation_start(self, sender: Any, args: Any) -> None:
            args.Cancel = not _same_origin(str(args.Uri), origin)

        def on_new_window_request(self, sender: Any, args: Any) -> None:
            args.Handled = True

        def on_webview_ready(self, sender: Any, args: Any) -> None:
            if not args.IsSuccess:
                failed.set()
                _show_startup_error(
                    "WebView2 không khởi tạo được. Hãy kiểm tra runtime offline."
                )
                self.pywebview_window.destroy()
                return
            # Install frame guards before the base callback performs first navigation.
            sender.CoreWebView2.FrameNavigationStarting += self.on_navigation_start
            super().on_webview_ready(sender, args)

    settings = {
        "ALLOW_DOWNLOADS": False,
        "ALLOW_FILE_URLS": False,
        "OPEN_EXTERNAL_LINKS_IN_BROWSER": False,
        "IGNORE_SSL_ERRORS": False,
        "REMOTE_DEBUGGING_PORT": None,
    }
    previous = {key: webview.settings[key] for key in settings}
    try:
        edgechromium.EdgeChrome = LocalEdgeChrome  # type: ignore[misc]
        webview.settings.update(settings)
        yield failed
    finally:
        edgechromium.EdgeChrome = original  # type: ignore[misc]
        webview.settings.update(previous)


def _wait_ready(
    thread: threading.Thread,
    origin: str,
    token: str,
    runtime: Runtime,
) -> None:
    deadline = time.monotonic() + READINESS_TIMEOUT
    with httpx.Client(timeout=1.0, trust_env=False, follow_redirects=False) as client:
        while time.monotonic() < deadline:
            if not thread.is_alive():
                raise RuntimeError("API không khởi động được.")
            try:
                response = client.get(
                    origin + "/api/v1/health", headers={"X-Session-Token": token}
                )
                if response.status_code in {401, 403}:
                    raise RuntimeError("Phiên API không khớp.")
                if response.status_code == 200:
                    health = HealthResponse.model_validate(response.json())
                    if (
                        health.ready
                        and health.model_version == runtime.predictor.model_version
                        and health.knowledge_version
                        == runtime.catalog.knowledge_version
                    ):
                        return
                    raise RuntimeError("Phiên bản API/model/chăm sóc không khớp.")
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
    raise RuntimeError("API chưa sẵn sàng sau 30 giây.")


def _show_startup_error(message: str) -> None:
    ctypes.windll.user32.MessageBoxW(None, message, "Tomato — lỗi khởi động", 0x10)


def main(*, model_dir: Path | None = None) -> int:
    sock: socket.socket | None = None
    thread: threading.Thread | None = None
    server: uvicorn.Server | None = None
    runtime: Runtime | None = None
    close_deadline: float | None = None
    status = 1

    def closing() -> None:
        nonlocal close_deadline
        if close_deadline is None:
            close_deadline = time.monotonic() + SHUTDOWN_TIMEOUT
        if runtime is not None:
            runtime.stopping.set()
        if server is not None:
            server.should_exit = True

    try:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Cửa sổ phải được khởi động trên main thread.")
        root = resource_root()
        frozen = getattr(sys, "frozen", False)
        if frozen:
            try:
                release_dir = Path(sys.executable).parent
                if root != release_dir / "_internal":
                    raise ValueError("invalid onedir resource layout")
                validate_release(release_dir)
            except (OSError, ValueError):
                _show_startup_error(
                    "Gói ứng dụng thiếu tài nguyên hoặc kiểm tra toàn vẹn thất bại. "
                    "Hãy cài lại bản phát hành đầy đủ. Ứng dụng không tự tải tài nguyên."
                )
                return 1
        try:
            require_webview2()
        except RuntimeError as error:
            _show_startup_error(str(error))
            return 1
        import webview

        sock = bind_loopback_socket()
        origin = f"http://127.0.0.1:{sock.getsockname()[1]}"
        validate_boundary(webview.token, origin)
        bundle_dir = (
            root / "model" if frozen else model_dir or root / DEFAULT_MODEL_BUNDLE
        )
        app = create_desktop_app(
            root / ("frontend" if frozen else "frontend/dist"),
            bundle_dir,
            root
            / ("knowledge/care-v1.json" if frozen else "infra/knowledge/care-v1.json"),
            webview.token,
            origin,
        )
        runtime = app.state.runtime
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                workers=1,
                reload=False,
                access_log=False,
                log_config=None,
                log_level="error",
                proxy_headers=False,
                server_header=False,
                ws="none",
                timeout_graceful_shutdown=int(SHUTDOWN_TIMEOUT),
            )
        )
        thread = threading.Thread(
            target=server.run,
            kwargs={"sockets": [sock]},
            name="tomato-api",
            daemon=True,
        )
        thread.start()
        assert runtime is not None
        _wait_ready(thread, origin, webview.token, runtime)
        with _secure_webview(origin) as native_failed:
            window = webview.create_window(
                "Tomato — nhận diện lá cà chua",
                url=origin,
                js_api=SessionBridge(webview.token),
                width=1080,
                height=760,
                min_size=(640, 480),
                background_color="#f3f6f5",
            )
            if window is None:
                raise RuntimeError("Không tạo được cửa sổ WebView2.")
            window.events.closing += closing
            webview.start(
                gui="edgechromium", http_server=False, private_mode=True, debug=False
            )
            status = 1 if native_failed.is_set() else 0
    except Exception:  # noqa: BLE001 -- startup failures use a sanitized native dialog
        # Do not expose paths, token, uploaded data or model exceptions in the dialog.
        _show_startup_error(
            "Không thể mở ứng dụng. Kiểm tra WebView2, giao diện, mô hình và thẻ chăm sóc trong gói. "
            "Ứng dụng không tự tải tài nguyên."
        )
    finally:
        user_closed = close_deadline is not None
        if runtime is not None:
            runtime.stopping.set()
        if server is not None:
            server.should_exit = True
        deadline = (
            close_deadline
            if close_deadline is not None
            else time.monotonic() + SHUTDOWN_TIMEOUT
        )
        try:
            if thread is not None:
                thread.join(max(0.0, deadline - time.monotonic()))
        finally:
            if sock is not None:
                sock.close()
        if thread is not None and thread.is_alive():
            logging.getLogger(__name__).error("API worker exceeded shutdown deadline.")
            if user_closed:
                os._exit(1)
            status = 1
    return status
