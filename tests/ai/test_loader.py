"""A real M1→M2 fixture; no raw project images or sealed artifacts are edited."""

import csv
import hashlib
import json
import pickle
import struct
import zlib
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ai.data import CLASSES, ManifestDataset, audit_manifest
from ai.preprocessing import build_transform
from ai.split import prepare_dataset, verify_seal
from tests.ai.conftest import AuditCase


@pytest.fixture
def sealed_images(
    audit_case: AuditCase, request: pytest.FixtureRequest
) -> tuple[Path, Path]:
    c = audit_case
    base = c.rows[0]
    c.rows = []
    for index, (_, label, _) in enumerate(CLASSES):
        prefix = (
            ("pv:" if index < 4 else "plantdoc:")
            if getattr(request, "param", False)
            else ""
        )
        for number in range(3):
            name = f"class-{index}-{number}.png"
            raw = c.root / name
            Image.new("RGB", (32, 24), (index * 20, number * 60, 80)).save(raw)
            c.rows.append(
                {
                    **base,
                    "record_id": f"{prefix}sample-{index}-{number}",
                    "source_file": name,
                    "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                    "label_original": label,
                    "leaf_group_id": f"group-{index}-{number}",
                }
            )
    c.write_rows()
    assert audit_manifest(c.csv, c.root, c.out, c.provenance) == {
        "accepted": 24,
        "excluded": 0,
    }
    sheet = c.out / "review-sheets" / "sheet.png"
    sheet.parent.mkdir()
    Image.new("RGB", (8, 8)).save(sheet)
    ids = [r["record_id"] for r in c.rows]
    review = {
        "schema_version": 1,
        "reviewer": "pytest synthetic only",
        "method": "fixture",
        "checked_at": "2026-11-02",
        "expert_verified": False,
        "input_manifest_sha256": hashlib.sha256(
            (c.out / "manifest.csv").read_bytes()
        ).hexdigest(),
        "sample_ids": ids,
        "quarantine_sample_ids": [],
        "notes": "Synthetic only",
        "limitations": "No real image review",
        "contact_sheets": [
            {
                "file": "review-sheets/sheet.png",
                "sha256": hashlib.sha256(sheet.read_bytes()).hexdigest(),
                "sample_ids": ids,
            }
        ],
    }
    prepare_dataset(c.out, review)
    return c.out, c.root


def manifest_rows(root: Path, split: str) -> list[dict[str, str]]:
    with (root / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        return [row for row in csv.DictReader(stream) if row["split"] == split]


@pytest.mark.parametrize("sealed_images", [True], indirect=True)
def test_yolo_training_loaders_never_open_test_images(
    sealed_images: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from torch.utils.data import WeightedRandomSampler

    from ai.train import _training_loaders

    root, raw = sealed_images
    config = json.loads(
        (Path(__file__).resolve().parents[2] / "ai/configs/E09.json").read_text()
    )
    config["dataset_dir"] = str(root)
    forbidden = {
        (raw / row["source_path"]).resolve() for row in manifest_rows(root, "test")
    }
    opened: set[Path] = set()
    original_open = Path.open

    def guarded_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        resolved = path.resolve()
        assert resolved not in forbidden, "training opened a TEST image"
        if resolved.parent == raw.resolve() and resolved.suffix == ".png":
            opened.add(resolved)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(AssertionError, match="TEST image"):
        next(iter(forbidden)).open("rb")
    loaders = _training_loaders(config)
    assert isinstance(loaders[0].sampler, WeightedRandomSampler)
    for loader, split in zip(loaders, ("train", "val"), strict=True):
        expected = [row["sample_id"] for row in manifest_rows(root, split)]
        samples = [sample for _, _, batch in loader for sample in batch]
        assert len(samples) == len(expected) == 8
        assert set(samples) <= set(expected)
        if split == "val":
            assert samples == expected
    assert opened and not opened.intersection(forbidden)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_loader_exact_split_order_and_batch(
    sealed_images: tuple[Path, Path],
    transform_spec: dict[str, Any],
    split: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, _ = sealed_images
    rows = manifest_rows(root, split)
    monkeypatch.chdir(tmp_path)
    data = ManifestDataset(root, split, build_transform(transform_spec))
    assert data.class_ids == (
        "bacterial_spot",
        "early_blight",
        "late_blight",
        "leaf_mold",
        "septoria_leaf_spot",
        "spider_mites",
        "yellow_leaf_curl",
        "healthy",
    )
    assert len(data) == len(rows) == 8
    expected = {
        "bacterial_spot": 0,
        "early_blight": 1,
        "late_blight": 2,
        "leaf_mold": 3,
        "septoria_leaf_spot": 4,
        "spider_mites": 5,
        "yellow_leaf_curl": 6,
        "healthy": 7,
    }
    for i, row in enumerate(rows):
        tensor, label, sample = data[i]
        assert tuple(tensor.shape) == (3, 224, 224)
        assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
        assert type(label) is int and label == expected[row["class_id"]]
        assert sample == row["sample_id"]
    tensors, labels, samples = next(
        iter(
            DataLoader(cast(Dataset[tuple[torch.Tensor, int, str]], data), batch_size=8)
        )
    )
    assert tuple(tensors.shape) == (8, 3, 224, 224)
    assert labels.tolist() == [expected[row["class_id"]] for row in rows]
    assert list(samples) == [row["sample_id"] for row in rows]
    restored = pickle.loads(pickle.dumps(data))
    assert torch.equal(restored[0][0], data[0][0])


@pytest.mark.parametrize("when", ["before", "after"])
@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_loader_checks_raw_every_read(
    sealed_images: tuple[Path, Path],
    transform_spec: dict[str, Any],
    when: str,
    damage: str,
) -> None:
    root, raw = sealed_images
    path = raw / manifest_rows(root, "train")[0]["source_path"]
    data = (
        ManifestDataset(root, "train", build_transform(transform_spec))
        if when == "after"
        else None
    )
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"changed")
    verify_seal(root)
    if data is None:
        data = ManifestDataset(root, "train", build_transform(transform_spec))
    with pytest.raises(
        FileNotFoundError if damage == "missing" else ValueError,
        match=None if damage == "missing" else "source_sha256",
    ):
        data[0]


@pytest.mark.parametrize("split", ["", "all", "Train", None])
def test_invalid_split(sealed_images: tuple[Path, Path], split: Any) -> None:
    with pytest.raises(ValueError, match="split"):
        ManifestDataset(sealed_images[0], split, Image.Image.copy)


def test_noncallable_transform(sealed_images: tuple[Path, Path]) -> None:
    invalid: Any = None
    with pytest.raises((ValueError, TypeError), match="transform"):
        ManifestDataset(sealed_images[0], "train", invalid)


def test_constructor_does_not_decode_raw(
    sealed_images: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("constructor decoded an image")

    monkeypatch.setattr(Image, "open", forbidden)
    assert len(ManifestDataset(sealed_images[0], "train", Image.Image.copy)) == 8


def test_hash_and_decode_share_handle(
    sealed_images: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data = ManifestDataset(sealed_images[0], "train", Image.Image.copy)
    digest, open_image = hashlib.file_digest, Image.open
    hashed: list[Any] = []

    def track(stream: Any, *args: Any, **kwargs: Any) -> Any:
        hashed.append(stream)
        return digest(stream, *args, **kwargs)

    def same_handle(stream: Any, *args: Any, **kwargs: Any) -> Any:
        assert len(hashed) == 1 and stream is hashed[0]
        assert stream.tell() == 0
        return open_image(stream, *args, **kwargs)

    monkeypatch.setattr(hashlib, "file_digest", track)
    monkeypatch.setattr(Image, "open", same_handle)
    image, _, sample = data[0]
    assert image.mode == "RGB" and image.size == (32, 24)
    assert sample.startswith("sample-")
    assert hashed[0].closed


@pytest.mark.parametrize("tamper", ["unsealed", "metadata"])
def test_bad_seal_rejected(sealed_images: tuple[Path, Path], tamper: str) -> None:
    root, _ = sealed_images
    if tamper == "unsealed":
        (root / "seal.json").unlink()
    else:
        with (root / "manifest.csv").open("a", encoding="utf-8") as stream:
            stream.write("\nchanged")
    with pytest.raises(ValueError):
        ManifestDataset(root, "train", Image.Image.copy)


def reseal_fixture(root: Path, name: str) -> None:
    """Tests only: change synthetic covered bytes to exercise loader trust boundaries."""
    path = root / "seal.json"
    seal = json.loads(path.read_text(encoding="utf-8"))
    seal["hashes"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    path.write_text(json.dumps(seal), encoding="utf-8")


def replace_fixture_image(root: Path, raw: Path, content: bytes) -> None:
    """Revise only synthetic metadata to model a matching-hash malformed input."""
    selected = manifest_rows(root, "train")[0]
    (raw / selected["source_path"]).write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    manifest = root / "manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    for row in rows:
        if row["sample_id"] == selected["sample_id"]:
            row["source_sha256"] = digest
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    split_path = root / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    for assignment in split["assignments"]:
        if assignment["sample_id"] == selected["sample_id"]:
            assignment["source_sha256"] = digest
    split_path.write_text(json.dumps(split), encoding="utf-8")
    reseal_fixture(root, "manifest.csv")
    reseal_fixture(root, "split.json")
    verify_seal(root)


@pytest.mark.parametrize(
    "value",
    ["", "/raw", "C:raw", "C:/raw", "\\raw", "\\\\server\\raw", "bad\u0000root", 1],
)
def test_source_root_rejected(sealed_images: tuple[Path, Path], value: Any) -> None:
    root, _ = sealed_images
    path = root / "provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance["source_root"] = value
    path.write_text(json.dumps(provenance), encoding="utf-8")
    reseal_fixture(root, path.name)
    with pytest.raises(ValueError, match="source_root"):
        ManifestDataset(root, "train", Image.Image.copy)


@pytest.mark.parametrize("damage", ["decode", "oversized", "bomb"])
def test_decode_and_pixel_limit(sealed_images: tuple[Path, Path], damage: str) -> None:
    root, raw = sealed_images
    if damage == "decode":
        content = b"invalid image"
    else:

        def chunk(kind: bytes, payload: bytes) -> bytes:
            return (
                struct.pack(">I", len(payload))
                + kind
                + payload
                + struct.pack(">I", zlib.crc32(kind + payload))
            )

        size = (5000, 4001) if damage == "oversized" else (20000, 20000)
        content = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", *size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", b"")
            + chunk(b"IEND", b"")
        )
    replace_fixture_image(root, raw, content)
    data = ManifestDataset(root, "train", Image.Image.copy)
    with pytest.raises(
        ValueError, match="decode_error" if damage == "decode" else "pixel_limit"
    ):
        data[0]


@pytest.mark.parametrize("mode", ["L", "RGBA", "EXIF"])
def test_loader_rgb_and_orientation(
    sealed_images: tuple[Path, Path], mode: str, tmp_path: Path
) -> None:
    root, raw = sealed_images
    path = tmp_path / "mode.png"
    image = Image.new(
        "RGB" if mode == "EXIF" else mode,
        (32, 24),
        (90, 80, 70) if mode == "EXIF" else (90, 80, 70, 0) if mode == "RGBA" else 90,
    )
    if mode == "EXIF":
        image.getexif()[274] = 6
    image.save(path, exif=image.getexif())
    replace_fixture_image(root, raw, path.read_bytes())
    output, _, _ = ManifestDataset(root, "train", Image.Image.copy)[0]
    assert output.mode == "RGB"
    assert output.size == ((24, 32) if mode == "EXIF" else (32, 24))
    assert output.getpixel((0, 0)) == ((90, 90, 90) if mode == "L" else (90, 80, 70))


def test_symlink_retarget_is_checked(
    sealed_images: tuple[Path, Path], tmp_path: Path
) -> None:
    root, raw = sealed_images
    path = raw / manifest_rows(root, "train")[0]["source_path"]
    data = ManifestDataset(root, "train", Image.Image.copy)
    outside = tmp_path / "outside.png"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("host does not permit symbolic links")
    with pytest.raises(ValueError, match="outside_root"):
        data[0]
