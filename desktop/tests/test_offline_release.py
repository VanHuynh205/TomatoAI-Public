"""A candidate must carry the local resources required by its runtime."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from ai.artifacts import load_bundle
from ai.yolo import ULTRALYTICS_VERSION
from desktop import release
from desktop.tests.test_package import release_dir as release_dir  # noqa: PLC0414


def test_manifest_rejects_external_base(release_dir: Path) -> None:
    index = release_dir / "_internal/frontend/index.html"
    index.write_text(
        '<base href="https://example.invalid/"><script src="/assets/app.js"></script>',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frontend.*base"):
        release.build_manifest(release_dir)


@pytest.mark.parametrize(
    "missing",
    ["cfg/default.yaml", "cfg/models/26/yolo26-cls.yaml", "dependency", "version"],
)
def test_manifest_requires_bundled_yolo_runtime(
    release_dir: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    def yolo_bundle(bundle_dir: Path, *, require_trained: bool) -> SimpleNamespace:
        bundle = load_bundle(bundle_dir, require_trained=require_trained)
        return SimpleNamespace(
            metadata={**bundle.metadata, "architecture": "yolo26s_cls"}
        )

    monkeypatch.setattr(release, "load_bundle", yolo_bundle)
    internal = release_dir / "_internal"
    for name in ("cfg/default.yaml", "cfg/models/26/yolo26-cls.yaml"):
        if name != missing:
            path = internal / "ultralytics" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture: true\n", encoding="utf-8")
    if missing != "dependency":
        version = "0.1" if missing == "version" else ULTRALYTICS_VERSION
        path = internal / f"ultralytics-{version}.dist-info/METADATA"
        path.parent.mkdir()
        path.write_text(f"Name: ultralytics\nVersion: {version}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="bundled YOLO runtime"):
        release.build_manifest(release_dir)
