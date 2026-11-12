"""Read-only onedir integrity checks. Manifest hashes are not signatures."""

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import struct
import sys
import tempfile
from html.parser import HTMLParser
from importlib.metadata import distributions
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Any, Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ai.artifacts import load_bundle
from ai.dinov2_inference import DINO_V2_FILES, DINO_V2_MANIFEST, validate_dinov2_export
from ai.dinov3_field import CLASSES
from backend.care import CareCatalog, _Bundle, _json, _read
from backend.schemas import HealthResponse
from desktop.build_care import _Allowlist, export_care
from desktop.server import _asset_directory

APP_VERSION = "0.1.0"
API_VERSION = HealthResponse.model_fields["schema_version"].default
MANIFEST_NAME = "release-manifest.json"
_CARE_SOURCES = Path(__file__).resolve().parents[1] / "infra/knowledge"
_MAX_MANIFEST_BYTES = 8_000_000
_MAX_FILES = 30_000
_MAX_FILE_BYTES = 2_000_000_000
_MAX_RELEASE_BYTES = 8_000_000_000
_REQUIRED_DEPENDENCIES = {
    "fastapi",
    "pydantic",
    "uvicorn",
    "httpx",
    "pillow",
    "torch",
    "torchvision",
    "pywebview",
}
_WEBVIEW2 = {
    "runtime": "Evergreen Standalone",
    "architecture": "x64",
    "minimum_major": 86,
    "installation": "offline_separate",
}
_REQUIRED_FILES = {
    "TomatoDesktop.exe",
    "_internal/frontend/index.html",
    "_internal/knowledge/care-v1.json",
    "_internal/knowledge/allowlist.json",
}
_LEGACY_MODEL_FILES = {"_internal/model/metadata.json", "_internal/model/model.pth"}
_DINO_V2_MODEL_FILES = {f"_internal/model/{name}" for name in DINO_V2_FILES}
_Hash = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]
_Version = Annotated[
    str, StringConstraints(strict=True, min_length=1, max_length=128, pattern=r"^\S+$")
]


class _BuildOS(BaseModel):
    model_config = ConfigDict(extra="forbid")
    system: Literal["Windows"]
    release: _Version


class _WebView2(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runtime: Literal["Evergreen Standalone"]
    architecture: Literal["x64"]
    minimum_major: int = Field(strict=True, ge=86, le=86)
    installation: Literal["offline_separate"]


class _Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = Field(strict=True, ge=1, le=1)
    app_version: _Version
    api_version: Literal["1"]
    model_version: _Version
    knowledge_version: _Version
    dependencies: dict[_Version, _Version] = Field(min_length=1, max_length=1000)
    build_os: _BuildOS
    architecture: Literal["x64"]
    python_version: Annotated[
        str, StringConstraints(strict=True, pattern=r"^3\.12\.\d+$")
    ]
    webview2: _WebView2
    files: dict[str, _Hash] = Field(min_length=1, max_length=_MAX_FILES)


def _root(directory: Path) -> Path:
    text = str(directory)
    if text.replace("\\", "/").startswith("//") or "://" in text or "\x00" in text:
        raise ValueError("release directory must be local")
    root = Path(os.path.abspath(directory))
    if root.resolve(strict=True) != root:
        raise ValueError("release directory must not traverse symlinks or junctions")
    if not root.is_dir():
        raise ValueError("release directory missing")
    return root


def _safe_name(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or ":" in name
        or path.is_absolute()
        or path.as_posix() != name
        or any(
            part in {".", ".."}
            or part.rstrip(". ") != part
            or PureWindowsPath(part).is_reserved()
            or any(ord(char) < 32 for char in part)
            for part in path.parts
        )
    ):
        raise ValueError("unsafe manifest path")


def _files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    seen: set[str] = set()
    pending = [root]
    total_bytes = 0
    while pending:
        for path in sorted(pending.pop().iterdir()):
            name = path.relative_to(root).as_posix()
            _safe_name(name)
            if name.casefold() in seen:
                raise ValueError("duplicate case-insensitive release path")
            seen.add(name.casefold())
            info = path.lstat()
            if (
                path.is_symlink()
                or path.is_junction()
                or getattr(info, "st_file_attributes", 0) & 0x400
                or path.resolve(strict=True) != path
            ):
                raise ValueError(
                    "release contains a symlink, junction or reparse point"
                )
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
                continue
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("release contains a non-regular file")
            if name == MANIFEST_NAME:
                continue
            if name != "TomatoDesktop.exe" and not name.startswith("_internal/"):
                raise ValueError(f"unexpected release file: {name}")
            parts = {part.casefold() for part in path.relative_to(root).parts}
            relative = path.relative_to(root).parts
            license_tree = name.startswith("_internal/notices/") or (
                len(relative) > 3
                and relative[1].endswith(".dist-info")
                and relative[2] == "licenses"
            )
            if not license_tree and (
                parts & {"test", "tests", "fixture", "fixtures"}
                or re.search(
                    r"(?:^|[/_.-])(?:fixture|fixtures|spike)(?:[/_.-]|$)",
                    name,
                    re.IGNORECASE,
                )
            ):
                raise ValueError(f"test/fixture asset forbidden in release: {name}")
            if (
                path.suffix.lower() in {".dll", ".pyd", ".so"}
                and re.match(
                    r"(?:torch_cuda|c10_cuda|cudnn|cublas|cudart|cusparse|cufft|curand|nvrtc|nvjitlink|nvjpeg|cusolver|cupti|nvcuda)",
                    path.name,
                    re.IGNORECASE,
                )
                or "nvidia" in parts
            ):
                raise ValueError(f"CUDA runtime forbidden in CPU release: {name}")
            if (
                path.suffix.lower() in {".pth", ".pt", ".ckpt", ".safetensors"}
                and name not in _LEGACY_MODEL_FILES | _DINO_V2_MODEL_FILES
            ):
                raise ValueError(f"extra weights forbidden in release: {name}")
            total_bytes += info.st_size
            if (
                info.st_size > _MAX_FILE_BYTES
                or total_bytes > _MAX_RELEASE_BYTES
                or len(files) >= _MAX_FILES
            ):
                raise ValueError("release resource limit")
            files[name] = path
    return dict(sorted(files.items()))


def _dependencies(internal: Path) -> dict[str, str]:
    result = {}
    for distribution in distributions(path=[str(internal)]):
        raw_name = distribution.metadata.get("Name", "")
        name = re.sub(r"[-_.]+", "-", raw_name).lower()
        value = distribution.metadata.get("Version", "")
        if not name or not value or name in result:
            raise ValueError("invalid or duplicate bundled dependency metadata")
        if name.startswith("nvidia-") or "+cu" in value and "+cpu" not in value:
            raise ValueError("CUDA dependency forbidden in CPU release")
        result[name] = value
    if not _REQUIRED_DEPENDENCIES <= result.keys():
        raise ValueError("missing bundled runtime dependency metadata")
    if any(not result[name].endswith("+cpu") for name in ("torch", "torchvision")):
        raise ValueError("torch and torchvision must be CPU builds")
    return dict(sorted(result.items()))


class _FrontendReferences(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "base" and "href" in values:
            raise ValueError("frontend must not set a base href")
        field = "src" if tag == "script" else "href"
        if (
            tag == "script"
            or tag == "link"
            and values.get("rel") in {"stylesheet", "modulepreload"}
        ):
            reference = values.get(field)
            if reference:
                self.paths.append(reference)


def _frontend(internal: Path, files: dict[str, Path]) -> None:
    assets = _asset_directory(internal / "frontend")
    index = assets / "index.html"
    if not 0 < index.stat().st_size <= 1_000_000:
        raise ValueError("frontend index size limit")
    parser = _FrontendReferences()
    parser.feed(index.read_text(encoding="utf-8"))
    for reference in parser.paths:
        parsed = urlsplit(reference)
        name = unquote(parsed.path).removeprefix("/")
        if parsed.scheme or parsed.netloc:
            raise ValueError("frontend must use bundled offline assets")
        _safe_name(name)
        if "_internal/frontend/" + name not in files:
            raise ValueError("missing referenced frontend asset")
    for name, path in files.items():
        if name.startswith("_internal/frontend/") and path.suffix.lower() in {
            ".html",
            ".js",
            ".css",
            ".svg",
        }:
            if path.stat().st_size > 20_000_000:
                raise ValueError("frontend text asset size limit")
            if re.search(
                r"\bfixtures?\b|tomatospike|spike\.html",
                path.read_text(encoding="utf-8"),
                re.IGNORECASE,
            ):
                raise ValueError("fixture frontend forbidden in release")


def _release_metadata(root: Path, files: dict[str, Path]) -> dict[str, Any]:
    dinov2 = f"_internal/model/{DINO_V2_MANIFEST}" in files
    model_files = _DINO_V2_MODEL_FILES if dinov2 else _LEGACY_MODEL_FILES
    missing = (_REQUIRED_FILES | model_files) - files.keys()
    if missing:
        raise ValueError("missing release file: " + ", ".join(sorted(missing)))
    if files.keys() & (_LEGACY_MODEL_FILES | _DINO_V2_MODEL_FILES) != model_files:
        raise ValueError("multiple model formats in release")
    if not any(name.startswith("_internal/notices/") for name in files):
        raise ValueError("missing release notices")
    if not any(name.startswith("_internal/frontend/assets/") for name in files):
        raise ValueError("missing frontend assets")
    internal = root / "_internal"
    _frontend(internal, files)
    dependencies = _dependencies(internal)
    if dinov2:
        if not {"transformers", "safetensors"} <= dependencies.keys():
            raise ValueError("bundled DINOv2 runtime is incomplete")
        model_version = validate_dinov2_export(internal / "model")
        class_ids = CLASSES[:-1]
    else:
        model = load_bundle(internal / "model", require_trained=True)
        model_version = model.metadata["model_version"]
        class_ids = model.metadata["class_ids"]
    bundle = _Bundle.model_validate(_json(_read(internal / "knowledge/care-v1.json")))
    catalog = CareCatalog(bundle)
    if catalog.artifact_kind != "curated":
        raise ValueError("fixture care forbidden in release")
    catalog.validate_model_classes(class_ids)
    allowed = _Allowlist.model_validate(
        _json(_read(internal / "knowledge/allowlist.json"))
    )
    ids = [entry.class_id for entry in allowed.cards]
    if (
        allowed.knowledge_version != catalog.knowledge_version
        or len(ids) != len(set(ids))
        or len({entry.file for entry in allowed.cards}) != len(ids)
        or set(ids) != {card.class_id for card in bundle.cards}
    ):
        raise ValueError("care allowlist/version mismatch")
    return {
        "app_version": APP_VERSION,
        "api_version": API_VERSION,
        "model_version": model_version,
        "knowledge_version": catalog.knowledge_version,
        "dependencies": dependencies,
    }


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_FILE_BYTES:
            raise ValueError("release file resource limit")
        value = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("release changed during hashing")
    return value


def validate_care_export(knowledge_dir: Path, sources_dir: Path) -> None:
    """Build-time approval proof; source drafts are never required at runtime."""
    allowlist = sources_dir / "allowlist.json"
    if _read(knowledge_dir / "allowlist.json") != _read(allowlist):
        raise ValueError("care allowlist differs from approved build sources")
    with tempfile.TemporaryDirectory(
        prefix="tomato-care-", dir=sources_dir.parent
    ) as directory:
        expected = Path(directory) / "care-v1.json"
        export_care(sources_dir / "drafts", allowlist, expected)
        if _read(knowledge_dir / "care-v1.json") != _read(expected):
            raise ValueError("care bundle differs from allowlisted source export")


def build_manifest(release_dir: Path) -> dict[str, Any]:
    """Describe a valid onedir candidate without writing to the install directory."""
    root = _root(release_dir)
    files = _files(root)
    metadata = _release_metadata(root, files)
    validate_care_export(root / "_internal/knowledge", _CARE_SOURCES)
    if (
        platform.machine().lower() not in {"amd64", "x86_64"}
        or struct.calcsize("P") != 8
    ):
        raise ValueError("release requires an x64 build host")
    return _Manifest.model_validate(
        {
            "schema_version": 1,
            **metadata,
            "build_os": {"system": platform.system(), "release": platform.release()},
            "architecture": "x64",
            "python_version": platform.python_version(),
            "webview2": _WEBVIEW2,
            "files": {name: _hash(path) for name, path in files.items()},
        }
    ).model_dump()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate manifest JSON key")
        result[key] = value
    return result


def validate_release(release_dir: Path) -> None:
    """Fail before composition for changed files, incompatible or fixture resources."""
    root = _root(release_dir)
    files = _files(root)
    path = root / MANIFEST_NAME
    try:
        with path.open("rb") as stream:
            content = stream.read(_MAX_MANIFEST_BYTES + 1)
        if not 0 < len(content) <= _MAX_MANIFEST_BYTES:
            raise ValueError("release manifest size limit")
        manifest = _Manifest.model_validate(
            json.loads(content, object_pairs_hook=_unique_object)
        )
    except (OSError, UnicodeError, RecursionError) as error:
        raise ValueError("missing or invalid release manifest") from error
    for name in manifest.files:
        _safe_name(name)
    if set(manifest.files) != set(files):
        raise ValueError("manifest file inventory mismatch")
    for name, expected in manifest.files.items():
        if _hash(files[name]) != expected:
            raise ValueError(f"manifest hash mismatch: {name}")
    metadata = _release_metadata(root, files)
    for key, expected in metadata.items():
        if getattr(manifest, key) != expected:
            raise ValueError(f"release {key} mismatch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build-manifest", "verify"))
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "build-manifest":
            manifest = build_manifest(args.directory)
            # Existing releases stay immutable; rebuild into a new candidate directory.
            with (_root(args.directory) / MANIFEST_NAME).open(
                "x", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        else:
            validate_release(args.directory)
    except (OSError, ValueError) as error:
        print(f"Release verification failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"event": args.command, "directory": str(args.directory)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
