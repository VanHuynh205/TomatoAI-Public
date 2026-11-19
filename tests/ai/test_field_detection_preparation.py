"""Regression checks for metadata-only exclusions in the field training study."""

import ast
import hashlib
import importlib.util
import io
import json
import textwrap
import zipfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docs/progress/yolo26s-field-colab-2026-11-10/protect_splits.py"
spec = importlib.util.spec_from_file_location("protect_field_splits", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

sample_spec = importlib.util.spec_from_file_location(
    "audit_source_samples",
    ROOT / "docs/research/yolo26s-field-colab-2026-11-10/audit_source_samples.py",
)
assert sample_spec is not None and sample_spec.loader is not None
samples = importlib.util.module_from_spec(sample_spec)
sample_spec.loader.exec_module(samples)


def load_script(name: str, relative: str) -> ModuleType:
    script_spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert script_spec is not None and script_spec.loader is not None
    loaded = importlib.util.module_from_spec(script_spec)
    script_spec.loader.exec_module(loaded)
    return loaded


plantdoc = load_script("audit_plantdoc", "docs/progress/yolo26s-field-colab-2026-11-10/audit_plantdoc_detection.py")
greenhouse = load_script("audit_greenhouse", "docs/research/yolo26s-field-colab-2026-11-10/greenhouse_audit.py")
packager = load_script("package_baseline", "docs/progress/yolo26s-field-colab-2026-11-10/package_baseline.py")
uploader = load_script("colab_baseline_check", "docs/progress/yolo26s-field-colab-2026-11-10/colab_baseline_check.py")


@contextmanager
def stream_response(chunks: Iterable[bytes]) -> Iterator[SimpleNamespace]:
    yield SimpleNamespace(raise_for_status=lambda: None, iter_content=lambda _size: iter(chunks))


def test_protection_is_transitive_across_groups_hashes_and_review_links() -> None:
    rows = [
        {"id": "a", "component_id": "a", "sha256": "image1"},
        {"id": "b", "component_id": "a"},
        {"id": "c", "pixel_sha256": "pixels1"},
        {"id": "d", "pixel_sha256": "pixels1"},
        {"id": "e", "sha256": "image1"},
        {"id": "f"},
        {"id": "g"},
    ]
    assert module.closure(rows, {"a"}, [("b", "c")]) == {"a", "b", "c", "d", "e"}


def test_empty_hashes_do_not_join_unrelated_images() -> None:
    assert module.closure([{"id": "a", "sha256": ""}, {"id": "b", "sha256": ""}], {"a"}, []) == {"a"}


def test_unknown_relation_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown record"):
        module.closure([{"id": "a"}], {"a"}, [("a", "missing")])


def test_old_component_link_protects_renamed_member() -> None:
    rows = [{"id": "a", "v1_component_id": "old"}, {"id": "b", "v1_component_id": "old"}]
    assert module.closure(rows, {"a"}, []) == {"a", "b"}


def test_single_image_download_accepts_matching_archive_member() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("train/photo.jpg", b"image bytes")
    assert samples.extract_image(buffer.getvalue(), "photo.jpg") == b"image bytes"


@pytest.mark.parametrize("names", [["wrong.jpg"], ["photo.jpg", "extra.jpg"]])
def test_single_image_download_rejects_wrong_or_multiple_members(names: list[str]) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in names:
            archive.writestr(name, b"image bytes")
    with pytest.raises(ValueError, match="unexpected single-image archive"):
        samples.extract_image(buffer.getvalue(), "photo.jpg")


def test_single_image_download_limits_uncompressed_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(samples, "LIMIT", 4)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("photo.jpg", b"x" * 1000)
    with pytest.raises(ValueError, match="unexpected single-image archive"):
        samples.extract_image(buffer.getvalue(), "photo.jpg")


def test_cached_sample_must_match_publisher_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(samples, "OUTPUT", tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "sample.jpg").write_bytes(b"changed image")
    with pytest.raises(ValueError, match="publisher image checksum mismatch"):
        samples.fetch({"id": "sample", "expected_sha256": "0" * 64})


def test_development_rejects_test_id_before_accessing_image_path() -> None:
    with pytest.raises(ValueError, match="test-protected record"):
        plantdoc.validate_member({"id": "reserved"}, "development", {"plantdoc_test_protected_ids": ["reserved"]})


def test_plantdoc_rejects_modified_source_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = plantdoc.HERE / "protected-splits.json"
    (tmp_path / source.name).write_bytes(source.read_bytes())
    (tmp_path / "plantdoc-source-train-labels.csv").write_text("modified labels", encoding="utf-8")
    monkeypatch.setattr(plantdoc, "HERE", tmp_path)
    with pytest.raises(ValueError, match="source CSV checksum mismatch"):
        plantdoc.main()


def test_plantdoc_rejects_modified_protection_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "protected-splits.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(plantdoc, "HERE", tmp_path)
    with pytest.raises(ValueError, match="split snapshot checksum mismatch"):
        plantdoc.main()


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "utf-16-be", "utf-16-le"])
def test_voc_rejects_dtd_in_all_supported_encodings(encoding: str) -> None:
    raw = '<!DOCTYPE annotation [<!ENTITY x "injected">]><annotation>&x;</annotation>'.encode(encoding)
    with pytest.raises(ValueError, match="DTD and entities"):
        greenhouse.parse_voc(raw)


def test_metadata_download_stops_at_byte_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    def chunks() -> Iterator[bytes]:
        yield b"12345"
        raise AssertionError("download consumed bytes beyond its limit")
    monkeypatch.setattr(greenhouse.requests, "get", lambda *a, **kw: stream_response(chunks()))
    with pytest.raises(ValueError, match="byte limit"):
        greenhouse.request("https://example.invalid", max_bytes=4)


def test_truncated_jpeg_never_enters_sample_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), "green").save(buffer, format="JPEG")
    raw = buffer.getvalue()[:-500]
    monkeypatch.setattr(samples, "OUTPUT", tmp_path)
    monkeypatch.setattr(samples.requests, "get", lambda *a, **kw: stream_response([raw]))
    with pytest.raises(OSError, match="truncated"):
        samples.fetch({"id": "sample", "filename": "photo.jpg", "url": "https://example.invalid", "width": 256, "height": 256})
    assert not (tmp_path / "raw/sample.jpg").exists()
    assert not (tmp_path / "download-results.json").exists()


def test_fieldplant_cache_cannot_replace_first_seen_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(samples, "OUTPUT", tmp_path)
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw/sample.jpg").write_bytes(b"changed bytes")
    (tmp_path / "download-results.json").write_text(json.dumps([{"id": "sample", "sha256": "0" * 64}]), encoding="utf-8")
    with pytest.raises(ValueError, match="previously audited image checksum changed"):
        samples.fetch({"id": "sample"})


def _synthetic_baseline() -> tuple[dict[str, bytes], bytes, bytes, str]:
    contents = {
        "ai/__init__.py": b"# synthetic baseline\n",
        "docs/progress/yolo26s-field-colab-2026-11-10/protected-splits.json": b"{}\n",
    }
    rebuilt, _ = packager.package_bytes(contents)
    old_buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(rebuilt)) as src, zipfile.ZipFile(old_buffer, "w") as dst:
        manifest_sha256 = hashlib.sha256(src.read("upload-manifest.json")).hexdigest()
        for name in src.namelist():
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 9, 0, 0, 0))
            dst.writestr(info, src.read(name), compress_type=zipfile.ZIP_DEFLATED)
    return contents, rebuilt, old_buffer.getvalue(), manifest_sha256


def test_repacked_baseline_keeps_sealed_members_despite_zip_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contents, rebuilt, old_bytes, manifest_sha256 = _synthetic_baseline()
    monkeypatch.setattr(uploader, "EXPECTED_MANIFEST_SHA256", manifest_sha256)
    assert rebuilt != old_bytes
    uploader.verify_package(rebuilt)
    assert packager.package_bytes(contents)[0] == rebuilt
    with zipfile.ZipFile(io.BytesIO(rebuilt)) as archive:
        assert all(info.date_time == (2026, 11, 10, 0, 0, 0) for info in archive.infolist())


def test_baseline_uploader_rejects_unsealed_manifest() -> None:
    payload, _ = packager.package_bytes({"unapproved.py": b"not baseline"})
    with pytest.raises(ValueError, match="sealed member manifest SHA-256 mismatch"):
        uploader.verify_package(payload)


def test_baseline_member_checksum_cannot_be_bypassed_by_repacking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, old_bytes, manifest_sha256 = _synthetic_baseline()
    monkeypatch.setattr(uploader, "EXPECTED_MANIFEST_SHA256", manifest_sha256)
    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(old_bytes)) as old, zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as changed:
        for name in old.namelist():
            changed.writestr(name, b"changed" if name == "ai/__init__.py" else old.read(name))
    with pytest.raises(ValueError, match="Baseline member mismatch: ai/__init__.py"):
        uploader.verify_package(buffer.getvalue())


def test_baseline_smoke_subprocess_remains_valid_after_notebook_guard() -> None:
    assert uploader.__file__ is not None
    tree = ast.parse(Path(uploader.__file__).read_text(encoding="utf-8"))
    smoke = next(node.value.value for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "smoke" for target in node.targets) and isinstance(node.value, ast.Constant))
    assert isinstance(smoke, str)
    compile(textwrap.dedent(smoke), "baseline-smoke", "exec")
