"""D1 packaging spike: only synthetic frontend fixtures, no API or model."""

import ctypes
import shutil
import sys
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

from desktop.prerequisites import require_webview2


def _show_startup_error(message: str) -> None:
    ctypes.windll.user32.MessageBoxW(
        None,
        f"Không thể mở bản thử đóng gói.\n\n{message}",
        "Tomato — lỗi khởi động",
        0x10,
    )


def main() -> int:
    try:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Cửa sổ phải được khởi động trên main thread.")
        require_webview2()
        root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
        html = root / "frontend/tests/fixtures/spike.html"
        image = root / "desktop/tests/fixtures/spike.png"
        for asset in (html, image):
            if not asset.is_file():
                raise RuntimeError(
                    f"Thiếu tài nguyên bản thử: {asset.name}. Hãy build lại gói."
                )

        with TemporaryDirectory(prefix="tomato-spike-") as staging:
            index = Path(staging) / "spike.html"
            shutil.copyfile(html, index)
            shutil.copyfile(image, Path(staging) / "spike.png")

            import webview
            from webview.guilib import initialize

            # pywebview 6.2.1 can fall back even with gui='edgechromium'; fail closed.
            sys.modules["webview.platforms.mshtml"] = None  # type: ignore[assignment]
            gui = initialize("edgechromium")
            if gui.renderer != "edgechromium":
                raise RuntimeError(
                    "Không thể khởi tạo WebView2; không dùng renderer cũ."
                )
            webview.create_window(
                "Tomato — thử đóng gói",
                url=str(index),
                width=920,
                height=760,
                min_size=(640, 480),
                background_color="#f5f7f3",
            )
            # shortcut: builtin HTTP is process-lived; add bounded shutdown in D2.
            webview.start(gui="edgechromium", http_server=True)
        return 0
    except Exception as exc:  # noqa: BLE001 -- startup must fail via a native error dialog
        _show_startup_error(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
