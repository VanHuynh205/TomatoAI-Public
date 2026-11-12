"""Report import failures even when the Windows host has no console."""

import atexit
import ctypes
import gc


def main() -> int:
    try:
        from desktop.launcher import main as launch
    except Exception:  # noqa: BLE001 -- native dependency imports may raise any exception
        ctypes.windll.user32.MessageBoxW(
            None,
            "Không thể nạp thành phần khởi động. Hãy cài lại bản phát hành Windows x64 đầy đủ. "
            "Ứng dụng không tự tải tài nguyên.",
            "Tomato — lỗi khởi động",
            0x10,
        )
        return 1
    if __name__ == "__main__":
        # Python.NET repeatedly scans the heap on exit. Keep imports out of those scans.
        gc.collect()
        gc.freeze()
        atexit.register(gc.unfreeze)
    return launch()


if __name__ == "__main__":
    raise SystemExit(main())
