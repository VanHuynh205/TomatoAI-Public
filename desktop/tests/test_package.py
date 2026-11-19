"""Synthetic packaging contracts, never a trained-model or redistribution proof."""

import csv
import hashlib
import importlib
import json
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from torch import nn


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def release_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, care_payload: dict[str, Any]
) -> Path:
    root = tmp_path / "Cà chua có dấu"
    internal = root / "_internal"
    for name in ("frontend/assets", "model", "knowledge", "notices"):
        (internal / name).mkdir(parents=True)
    (root / "TomatoDesktop.exe").write_bytes(b"synthetic executable contract")
    (internal / "frontend/index.html").write_text(
        '<html><script type="module" src="/assets/app.js"></script></html>',
        encoding="utf-8",
    )
    (internal / "frontend/assets/app.js").write_text("console.log('Tomato')")
    (internal / "notices/THIRD-PARTY.txt").write_text("Synthetic notices only")

    # Simulate a trained declaration with tiny CPU tensors; no real model is touched.
    monkeypatch.setattr(
        "ai.artifacts.build_model", lambda architecture, classes: nn.Linear(4, classes)
    )
    torch.save(nn.Linear(4, 2).state_dict(), internal / "model/model.pth")
    class_ids = ["healthy", "early_blight"]
    config = json.loads(
        (Path(__file__).resolve().parents[2] / "ai/configs/E01.json").read_text()
    )
    write_json(
        internal / "model/metadata.json",
        {
            "schema_version": 1,
            "architecture": "efficientnet_b0",
            "model_version": "unit-model-v1",
            "class_ids": class_ids,
            "preprocessing": config["preprocessing"],
            "dataset_version": "unit-data-v1",
            "dataset_sha256": "a" * 64,
            "split_hash": "b" * 64,
            "weights_sha256": hashlib.sha256(
                (internal / "model/model.pth").read_bytes()
            ).hexdigest(),
            "threshold": 0.5,
            "class_map_sha256": hashlib.sha256(
                json.dumps(class_ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "validation_decision_sha256": "c" * 64,
            "artifact_kind": "trained",
        },
    )
    cards = [{**care_payload, "class_id": label} for label in class_ids]
    sources = tmp_path / "care-sources"
    drafts = sources / "drafts"
    drafts.mkdir(parents=True)
    for card in cards:
        write_json(drafts / (card["class_id"] + ".json"), card)
    write_json(
        sources / "allowlist.json",
        {
            "schema_version": 1,
            "knowledge_version": "unit-care-v1",
            "allowed_use": "offline_reference_vn",
            "cards": [
                {
                    "class_id": label,
                    "file": label + ".json",
                    "sha256": hashlib.sha256(
                        (drafts / (label + ".json")).read_bytes()
                    ).hexdigest(),
                }
                for label in class_ids
            ],
        },
    )
    from desktop.build_care import export_care

    export_care(drafts, sources / "allowlist.json", internal / "knowledge/care-v1.json")
    shutil.copyfile(sources / "allowlist.json", internal / "knowledge/allowlist.json")
    monkeypatch.setattr("desktop.release._CARE_SOURCES", sources, raising=False)
    for name, version in {
        "fastapi": "0.139.0",
        "pydantic": "2.13.4",
        "uvicorn": "0.54.0",
        "httpx": "0.28.1",
        "pillow": "12.3.0",
        "torch": "2.14.1+cpu",
        "torchvision": "0.29.1+cpu",
        "pywebview": "6.2.1",
    }.items():
        metadata = internal / f"{name}-{version}.dist-info/METADATA"
        metadata.parent.mkdir()
        metadata.write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        )
    return root


def seal(root: Path) -> dict[str, Any]:
    release = importlib.import_module("desktop.release")
    manifest = release.build_manifest(root)
    write_json(root / "release-manifest.json", manifest)
    return manifest  # type: ignore[no-any-return]


def test_manifest_hashes_every_packaged_file(release_dir: Path) -> None:
    release = importlib.import_module("desktop.release")
    manifest = seal(release_dir)
    assert manifest["schema_version"] == 1
    assert manifest["app_version"] == "0.1.0"
    assert manifest["api_version"] == "1"
    assert manifest["model_version"] == "unit-model-v1"
    assert manifest["knowledge_version"] == "unit-care-v1"
    assert manifest["architecture"] == "x64"
    assert manifest["build_os"]["system"] == "Windows"
    assert manifest["webview2"]["minimum_major"] == 86
    assert manifest["dependencies"]["torch"] == "2.14.1+cpu"
    expected = {
        path.relative_to(release_dir).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in release_dir.rglob("*")
        if path.is_file() and path.name != "release-manifest.json"
    }
    assert manifest["files"] == expected
    assert release.validate_release(release_dir) is None


def test_runtime_validation_does_not_require_build_sources(
    release_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    release = importlib.import_module("desktop.release")
    seal(release_dir)
    monkeypatch.setattr(release, "_CARE_SOURCES", tmp_path / "unshipped-sources")
    assert release.validate_release(release_dir) is None


def test_resource_paths_and_readonly_install(
    release_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launcher = importlib.import_module("desktop.launcher")
    release = importlib.import_module("desktop.release")
    seal(release_dir)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(release_dir / "_internal"), raising=False)
    assert launcher.resource_root() == release_dir / "_internal"

    before = {p: p.read_bytes() for p in release_dir.rglob("*") if p.is_file()}
    original_open = Path.open
    writes: list[Path] = []

    def guarded_open(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if path.is_relative_to(release_dir) and any(c in mode for c in "wax+"):
            writes.append(path)
            raise PermissionError("read-only install")
        return original_open(path, mode, *args, **kwargs)

    try:
        for path in before:
            path.chmod(stat.S_IREAD)
        monkeypatch.setattr(Path, "open", guarded_open)
        release.validate_release(release_dir)
        assert writes == []
        assert {p: p.read_bytes() for p in before} == before
    finally:
        for path in before:
            path.chmod(stat.S_IREAD | stat.S_IWRITE)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows NTFS ACL contract")
def test_release_validation_under_deny_write_acl(release_dir: Path) -> None:
    release = importlib.import_module("desktop.release")
    seal(release_dir)
    workspace = Path(__file__).resolve().parents[2]
    scratch = workspace / ".scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="package-acl-", dir=scratch) as temporary:
        directory = Path(temporary).resolve()
        assert directory.is_relative_to(workspace.resolve())
        install = directory / "installation"
        shutil.copytree(release_dir, install)
        before = {p: p.read_bytes() for p in install.rglob("*") if p.is_file()}
        identity = subprocess.run(
            ["whoami", "/user", "/fo", "csv", "/nh"],
            check=True,
            capture_output=True,
            text=True,
        )
        sid = next(csv.reader(identity.stdout.splitlines()))[1]
        try:
            denied = subprocess.run(
                [
                    "icacls",
                    str(install),
                    "/deny",
                    f"*{sid}:(OI)(CI)(WD,AD,WEA,WA,DC)",
                    "/q",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if denied.returncode in {5, 1314}:
                pytest.skip("Current Windows token lacks WRITE_DAC privilege")
            denied.check_returncode()
            with pytest.raises(PermissionError):
                (install / "write-probe.txt").write_text("must fail")
            with pytest.raises(PermissionError):
                (install / "TomatoDesktop.exe").open("ab")
            release.validate_release(install)
            assert {p: p.read_bytes() for p in before} == before
        finally:
            subprocess.run(
                ["icacls", str(install), "/remove:d", f"*{sid}", "/t", "/q"],
                check=True,
                capture_output=True,
                text=True,
            )
        (install / "write-probe.txt").write_text("ACL restored")


def test_source_resource_root_ignores_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = importlib.import_module("desktop.launcher")
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.chdir(tmp_path)
    assert launcher.__file__ is not None
    assert launcher.resource_root() == Path(launcher.__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "missing",
    [
        "TomatoDesktop.exe",
        "_internal/frontend/index.html",
        "_internal/frontend/assets/app.js",
        "_internal/model/model.pth",
        "_internal/model/metadata.json",
        "_internal/knowledge/care-v1.json",
        "_internal/knowledge/allowlist.json",
        "_internal/notices/THIRD-PARTY.txt",
    ],
)
def test_missing_runtime_and_invalid_assets(release_dir: Path, missing: str) -> None:
    release = importlib.import_module("desktop.release")
    seal(release_dir)
    (release_dir / missing).unlink()
    with pytest.raises(ValueError):
        release.validate_release(release_dir)


@pytest.mark.parametrize(
    "change",
    ["changed", "added", "omitted", "hash", "schema", "versions", "dependency"],
)
def test_manifest_rejects_tampering(release_dir: Path, change: str) -> None:
    release = importlib.import_module("desktop.release")
    manifest = seal(release_dir)
    if change == "changed":
        (release_dir / "_internal/frontend/assets/app.js").write_text("changed")
    elif change == "added":
        (release_dir / "extra.txt").write_text("not inventoried")
    elif change == "omitted":
        del manifest["files"]["TomatoDesktop.exe"]
    elif change == "hash":
        manifest["files"]["TomatoDesktop.exe"] = "0" * 64
    elif change == "schema":
        manifest["schema_version"] = 2
    elif change == "versions":
        manifest["model_version"] = "different-model"
    else:
        manifest["dependencies"]["torch"] = "2.14.1+cu126"
    write_json(release_dir / "release-manifest.json", manifest)
    with pytest.raises(ValueError):
        release.validate_release(release_dir)


@pytest.mark.parametrize(
    "entry",
    [
        "../outside",
        "/absolute",
        "C:/outside",
        "_internal\\model\\model.pth",
        "_internal/../TomatoDesktop.exe",
        "release-manifest.json",
    ],
)
def test_unsafe_manifest_paths_fail(release_dir: Path, entry: str) -> None:
    release = importlib.import_module("desktop.release")
    manifest = seal(release_dir)
    manifest["files"][entry] = "a" * 64
    write_json(release_dir / "release-manifest.json", manifest)
    with pytest.raises(ValueError):
        release.validate_release(release_dir)


@pytest.mark.parametrize(
    "change",
    [
        "model_fixture",
        "care_fixture",
        "model_schema",
        "weights_hash",
        "care_schema",
        "head",
        "missing_care",
        "allowlist",
        "frontend_fixture",
    ],
)
def test_builder_reuses_authoritative_validators(
    release_dir: Path, change: str
) -> None:
    release = importlib.import_module("desktop.release")
    internal = release_dir / "_internal"
    if change == "frontend_fixture":
        (internal / "frontend/index.html").write_text("<h1>Tomato fixture</h1>")
    elif change in {"model_fixture", "model_schema", "weights_hash", "head"}:
        path = internal / "model/metadata.json"
        metadata = json.loads(path.read_text())
        if change == "model_fixture":
            metadata["artifact_kind"] = "fixture"
        elif change == "model_schema":
            metadata["schema_version"] = 2
        elif change == "weights_hash":
            metadata["weights_sha256"] = "0" * 64
        else:
            metadata["class_ids"].append("extra_class")
            metadata["class_map_sha256"] = hashlib.sha256(
                json.dumps(metadata["class_ids"], separators=(",", ":")).encode()
            ).hexdigest()
        write_json(path, metadata)
    else:
        path = internal / "knowledge/care-v1.json"
        care = json.loads(path.read_text())
        if change == "care_fixture":
            care["artifact_kind"] = "fixture"
        elif change == "care_schema":
            care["schema_version"] = 2
        elif change == "missing_care":
            care["cards"].pop()
        else:
            path = internal / "knowledge/allowlist.json"
            care = json.loads(path.read_text())
            care["knowledge_version"] = "wrong"
        write_json(path, care)
    with pytest.raises(ValueError):
        release.build_manifest(release_dir)


@pytest.mark.parametrize("change", ["bundle", "allowlist_hash", "draft"])
def test_builder_rejects_care_changes_without_version_change(
    release_dir: Path, change: str
) -> None:
    release = importlib.import_module("desktop.release")
    if change == "bundle":
        path = release_dir / "_internal/knowledge/care-v1.json"
        care = json.loads(path.read_text(encoding="utf-8"))
        care["cards"][0]["summary"] = "Unapproved care content"
    elif change == "allowlist_hash":
        path = release_dir / "_internal/knowledge/allowlist.json"
        care = json.loads(path.read_text(encoding="utf-8"))
        care["cards"][0]["sha256"] = "0" * 64
    else:
        path = release._CARE_SOURCES / "drafts/early_blight.json"
        care = json.loads(path.read_text(encoding="utf-8"))
        care["summary"] = "Unapproved care content"
    write_json(path, care)
    with pytest.raises(ValueError, match="care|allowlist"):
        release.build_manifest(release_dir)


@pytest.mark.parametrize(
    "name",
    [
        "_internal/tests/check.py",
        "_internal/frontend/fixtures/a.png",
        "_internal/torch/lib/torch_cuda.dll",
        "_internal/torch/lib/cudnn64_9.dll",
        "_internal/model/other.pth",
    ],
)
def test_builder_rejects_test_cuda_and_extra_weights(
    release_dir: Path, name: str
) -> None:
    release = importlib.import_module("desktop.release")
    path = release_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"forbidden")
    with pytest.raises(ValueError):
        release.build_manifest(release_dir)


def test_reparse_directory_rejected(
    release_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = importlib.import_module("desktop.release")
    link = release_dir / "linked"
    link.mkdir()
    original = Path.is_junction
    # The sandbox forbids creating actual symlinks and junctions; probe the guard.
    monkeypatch.setattr(
        Path, "is_junction", lambda path: path == link or original(path)
    )
    with pytest.raises(ValueError, match="symlink|junction|reparse"):
        release.build_manifest(release_dir)


def test_real_junction_rejected(release_dir: Path, tmp_path: Path) -> None:
    import _winapi

    release = importlib.import_module("desktop.release")
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("unchanged")
    junction = release_dir / "_internal/redirected"
    _winapi.CreateJunction(str(outside), str(junction))
    try:
        with pytest.raises(ValueError, match="symlink|junction|reparse"):
            release.build_manifest(release_dir)
        with pytest.raises(ValueError, match="symlink|junction|reparse"):
            release.build_manifest(junction)
    finally:
        junction.rmdir()
    assert sentinel.read_text() == "unchanged"


def test_builder_rejects_missing_referenced_script(release_dir: Path) -> None:
    release = importlib.import_module("desktop.release")
    index = release_dir / "_internal/frontend/index.html"
    index.write_text('<script type="module" src="/assets/missing.js"></script>')
    with pytest.raises(ValueError, match="frontend"):
        release.build_manifest(release_dir)


def test_builder_rejects_fixture_in_compiled_asset(release_dir: Path) -> None:
    release = importlib.import_module("desktop.release")
    script = release_dir / "_internal/frontend/assets/app.js"
    script.write_text("window.fixture = true;")
    with pytest.raises(ValueError, match="fixture"):
        release.build_manifest(release_dir)


def test_manifest_cli(release_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    release = importlib.import_module("desktop.release")
    assert release.main(["build-manifest", "--directory", str(release_dir)]) == 0
    assert release.main(["verify", "--directory", str(release_dir)]) == 0
    (release_dir / "TomatoDesktop.exe").write_bytes(b"changed")
    assert release.main(["verify", "--directory", str(release_dir)]) != 0
    assert "hash" in capsys.readouterr().err.lower()


@pytest.mark.parametrize(
    "failure",
    [
        "manifest hash mismatch",
        "fixture bundle forbidden in release",
        "missing care for model class",
    ],
)
def test_frozen_launcher_validates_before_composition(
    release_dir: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    launcher = importlib.import_module("desktop.launcher")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(release_dir / "_internal"), raising=False)
    monkeypatch.setattr(sys, "executable", str(release_dir / "TomatoDesktop.exe"))
    check = Mock(side_effect=ValueError(failure))
    monkeypatch.setattr(launcher, "validate_release", check)
    messages: list[str] = []
    monkeypatch.setattr(launcher, "_show_startup_error", messages.append)
    compose, bind, prerequisite = Mock(), Mock(), Mock()
    monkeypatch.setattr(launcher, "create_desktop_app", compose)
    monkeypatch.setattr(launcher, "bind_loopback_socket", bind)
    monkeypatch.setattr(launcher, "require_webview2", prerequisite)
    assert launcher.main() == 1
    check.assert_called_once_with(release_dir)
    compose.assert_not_called()
    bind.assert_not_called()
    prerequisite.assert_not_called()
    assert messages == [
        (
            "Gói ứng dụng thiếu tài nguyên hoặc kiểm tra toàn vẹn thất bại. "
            "Hãy cài lại bản phát hành đầy đủ. Ứng dụng không tự tải tài nguyên."
        )
    ]


def test_manifest_hash_precedes_weight_loading(
    release_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = importlib.import_module("desktop.release")
    seal(release_dir)
    (release_dir / "_internal/model/model.pth").write_bytes(b"changed weights")
    loader = Mock(side_effect=AssertionError("unverified weights reached loader"))
    monkeypatch.setattr(release, "load_bundle", loader)
    with pytest.raises(ValueError, match="hash"):
        release.validate_release(release_dir)
    loader.assert_not_called()


def test_manifest_builder_will_not_replace_sealed_release(release_dir: Path) -> None:
    release = importlib.import_module("desktop.release")
    seal(release_dir)
    path = release_dir / "release-manifest.json"
    before = path.read_bytes()
    assert release.main(["build-manifest", "--directory", str(release_dir)]) == 1
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["missing", "cuda"])
def test_bundled_dependency_metadata_is_authoritative(
    release_dir: Path, change: str
) -> None:
    release = importlib.import_module("desktop.release")
    path = release_dir / "_internal/torch-2.14.1+cpu.dist-info/METADATA"
    if change == "missing":
        path.unlink()
    else:
        path.write_text("Name: torch\nVersion: 2.14.1+cu126\n")
    with pytest.raises(ValueError):
        release.build_manifest(release_dir)


def test_duplicate_manifest_json_keys_rejected(release_dir: Path) -> None:
    release = importlib.import_module("desktop.release")
    seal(release_dir)
    path = release_dir / "release-manifest.json"
    path.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        release.validate_release(release_dir)


def _build_cnn_gru_release_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, care_payload: dict[str, Any]
) -> Path:
    from safetensors.torch import save_file

    from ai.cnn_gru_inference import CNN_GRU_MANIFEST, EXPORT_LIMITS
    from ai.cnn_gru_model import IMAGENET_MEAN, IMAGENET_STD, build_cnn_gru_model
    from ai.dinov3_field import CLASSES

    root = tmp_path / "TomatoCnnGru"
    internal = root / "_internal"
    for name in ("frontend/assets", "model", "knowledge", "notices"):
        (internal / name).mkdir(parents=True)
    (root / "TomatoDesktop.exe").write_bytes(b"synthetic executable contract")
    (internal / "frontend/index.html").write_text(
        '<html><script type="module" src="/assets/app.js"></script></html>',
        encoding="utf-8",
    )
    (internal / "frontend/assets/app.js").write_text(
        "console.log('Tomato')", encoding="utf-8"
    )
    (internal / "notices/THIRD-PARTY.txt").write_text(
        "Synthetic notices only", encoding="utf-8"
    )

    model = build_cnn_gru_model(
        "resnet50", len(CLASSES), use_gru_head=False, hidden_dim=16, proj_dim=32
    )
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        internal / "model/model.safetensors",
    )
    write_json(
        internal / "model/leaf-classifier.json",
        {
            "schema_version": 1,
            "architecture": "resnet50",
            "use_gru_head": False,
            "hidden_dim": 16,
            "proj_dim": 32,
            "model_version": "cnn-gru-pkg-v1",
            "class_ids": list(CLASSES),
            "size": 288,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "test_time_horizontal_flip": True,
            "temperature": 1.1,
            "threshold": 0.55,
            "artifact_kind": "trained",
            "guardrail_passed": True,
        },
    )
    write_json(
        internal / f"model/{CNN_GRU_MANIFEST}",
        {
            "schema_version": 1,
            "model_version": "cnn-gru-pkg-v1",
            "architecture": "resnet50",
            "evaluation_size": 288,
            "temperature": 1.1,
            "threshold": 0.55,
            "artifact_kind": "trained",
            "sha256": {
                name: hashlib.sha256(
                    (internal / "model" / name).read_bytes()
                ).hexdigest()
                for name in EXPORT_LIMITS
            },
        },
    )

    care_classes = list(CLASSES[:-1])
    sources = tmp_path / "care-sources-cnn-gru"
    drafts = sources / "drafts"
    drafts.mkdir(parents=True)
    for label in care_classes:
        write_json(drafts / f"{label}.json", {**care_payload, "class_id": label})
    write_json(
        sources / "allowlist.json",
        {
            "schema_version": 1,
            "knowledge_version": "unit-care-v1",
            "allowed_use": "offline_reference_vn",
            "cards": [
                {
                    "class_id": label,
                    "file": f"{label}.json",
                    "sha256": hashlib.sha256(
                        (drafts / f"{label}.json").read_bytes()
                    ).hexdigest(),
                }
                for label in care_classes
            ],
        },
    )
    from desktop.build_care import export_care

    export_care(drafts, sources / "allowlist.json", internal / "knowledge/care-v1.json")
    shutil.copyfile(sources / "allowlist.json", internal / "knowledge/allowlist.json")
    monkeypatch.setattr("desktop.release._CARE_SOURCES", sources, raising=False)
    for name, version in {
        "fastapi": "0.139.0",
        "pydantic": "2.13.4",
        "uvicorn": "0.54.0",
        "httpx": "0.28.1",
        "pillow": "13.0.0",
        "torch": "2.14.1+cpu",
        "torchvision": "0.29.1+cpu",
        "pywebview": "6.2.1",
        "safetensors": "0.7.0",
        "numpy": "2.5.4",
    }.items():
        metadata = internal / f"{name}-{version}.dist-info/METADATA"
        metadata.parent.mkdir()
        metadata.write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
            encoding="utf-8",
        )
    return root


def test_cnn_gru_release_manifest_builds_and_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, care_payload: dict[str, Any]
) -> None:
    release = importlib.import_module("desktop.release")
    root = _build_cnn_gru_release_dir(tmp_path, monkeypatch, care_payload)
    manifest = seal(root)
    assert manifest["model_version"] == "cnn-gru-pkg-v1"
    assert manifest["dependencies"]["safetensors"] == "0.7.0"
    assert manifest["dependencies"]["numpy"] == "2.5.4"
    assert release.validate_release(root) is None


@pytest.mark.parametrize("passed", [False, None])
def test_cnn_gru_release_requires_passing_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    care_payload: dict[str, Any],
    passed: bool | None,
) -> None:
    release = importlib.import_module("desktop.release")
    root = _build_cnn_gru_release_dir(tmp_path, monkeypatch, care_payload)
    model = root / "_internal/model"
    metadata_path = model / "leaf-classifier.json"
    metadata = json.loads(metadata_path.read_bytes())
    if passed is None:
        metadata.pop("guardrail_passed")
    else:
        metadata["guardrail_passed"] = passed
    write_json(metadata_path, metadata)
    manifest_path = model / "cnn-gru-release.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["sha256"][metadata_path.name] = hashlib.sha256(
        metadata_path.read_bytes()
    ).hexdigest()
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="selection guardrail"):
        release.build_manifest(root)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_safetensors",
        "missing_numpy",
        "fixture_model",
        "mixed_legacy",
        "mixed_dinov2",
    ],
)
def test_cnn_gru_release_rejects_missing_deps_fixture_and_mixed_formats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    care_payload: dict[str, Any],
    defect: str,
) -> None:
    release = importlib.import_module("desktop.release")
    root = _build_cnn_gru_release_dir(tmp_path, monkeypatch, care_payload)
    internal = root / "_internal"
    if defect == "missing_safetensors":
        shutil.rmtree(internal / "safetensors-0.7.0.dist-info")
    elif defect == "missing_numpy":
        shutil.rmtree(internal / "numpy-2.5.4.dist-info")
    elif defect == "fixture_model":
        meta_path = internal / "model/leaf-classifier.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["artifact_kind"] = "fixture"
        write_json(meta_path, meta)
        rel_path = internal / "model/cnn-gru-release.json"
        rel = json.loads(rel_path.read_text(encoding="utf-8"))
        rel["artifact_kind"] = "fixture"
        rel["sha256"]["leaf-classifier.json"] = hashlib.sha256(
            meta_path.read_bytes()
        ).hexdigest()
        write_json(rel_path, rel)
    elif defect == "mixed_legacy":
        (internal / "model/model.pth").write_bytes(b"extra")
    else:
        (internal / "model/dinov2-release.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        release.build_manifest(root)
