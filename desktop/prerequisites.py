"""Windows prerequisite probe shared by the D1 spike and the application host."""

import winreg


def require_webview2() -> None:
    """Probe prerequisites without importing pywebview's fallback-capable WinForms."""
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full",
        ) as key:
            release, _ = winreg.QueryValueEx(key, "Release")
        if int(release) < 394802:
            raise ValueError(".NET too old")
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "WebView2 cần .NET Framework 4.6.2 trở lên. "
            "Hãy dùng bộ cài .NET offline phù hợp rồi mở lại ứng dụng."
        ) from exc

    # Match pywebview 6.2.1's Evergreen registry locations, without GUI side effects.
    client = r"Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
    for hive, path in (
        (winreg.HKEY_CURRENT_USER, rf"SOFTWARE\{client}"),
        (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\WOW6432Node\{client}"),
    ):
        try:
            with winreg.OpenKey(hive, path) as key:
                version, _ = winreg.QueryValueEx(key, "pv")
            parts = tuple(int(part) for part in str(version).split("."))
            if len(parts) == 4 and parts[0] >= 86:
                return
        except (OSError, TypeError, ValueError):
            continue
    raise RuntimeError(
        "Không tìm thấy Microsoft Edge WebView2 Runtime phù hợp. "
        "Hãy dùng bộ cài WebView2 Evergreen Standalone offline x64 rồi mở lại ứng dụng. "
        "Ứng dụng không tự tải hoặc cài runtime."
    )
