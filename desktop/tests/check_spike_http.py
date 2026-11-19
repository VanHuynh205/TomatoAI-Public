"""Characterize pinned pywebview's HTTP containment; no native GUI is opened."""

import http.client
import shutil
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

from webview.http import start_server


def check_http() -> None:
    source = Path(__file__).resolve().parents[2]
    with TemporaryDirectory(prefix="d1-http-check-") as directory:
        root = Path(directory)
        shutil.copyfile(
            source / "frontend/tests/fixtures/spike.html", root / "spike.html"
        )
        shutil.copyfile(source / "desktop/tests/fixtures/spike.png", root / "spike.png")
        address, common, _ = start_server([str(root / "spike.html")])
        assert common is not None and Path(common) == root
        port = urlsplit(address).port
        assert port is not None

        def get(path: str) -> tuple[int, bytes]:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            try:
                connection.request("GET", path)
                response = connection.getresponse()
                return response.status, response.read()
            finally:
                connection.close()

        deadline = time.monotonic() + 5
        while True:
            try:
                status, html = get("/spike.html")
                break
            except ConnectionRefusedError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
        assert status == 200
        assert "Bản thử đóng gói – chưa có mô hình bệnh" in html.decode("utf-8")
        assert get("/spike.png")[0] == 200
        for path in (
            "/pyproject.toml",
            "/backend/schemas.py",
            "/../pyproject.toml",
            "/%2e%2e/pyproject.toml",
        ):
            status, _ = get(path)
            assert status in (403, 404), (path, status)
            print(path, status)
        print("Builtin HTTP: staged HTML/PNG=200; source and parent paths denied.")


if __name__ == "__main__":
    check_http()
