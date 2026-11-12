"""M1 metadata audit; source images remain read-only and no training gate is sealed."""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import warnings
from collections.abc import Callable
from datetime import date
from pathlib import Path, PureWindowsPath
from typing import Any, BinaryIO
from urllib.parse import urlsplit

from PIL import Image, ImageOps, UnidentifiedImageError

CLASSES = (
    ("bacterial_spot", "Tomato___Bacterial_spot", "Đốm vi khuẩn"),
    ("early_blight", "Tomato___Early_blight", "Đốm vòng / cháy lá sớm"),
    ("late_blight", "Tomato___Late_blight", "Mốc sương"),
    ("leaf_mold", "Tomato___Leaf_Mold", "Mốc lá"),
    ("septoria_leaf_spot", "Tomato___Septoria_leaf_spot", "Đốm lá Septoria"),
    (
        "spider_mites",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tổn thương do nhện hai chấm",
    ),
    (
        "yellow_leaf_curl",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Triệu chứng xoăn vàng lá theo nhãn TYLCV",
    ),
    ("healthy", "Tomato___healthy", "Lá khỏe theo nhãn dữ liệu"),
)
LABEL_IDS = {label: class_id for class_id, label, _ in CLASSES}
REQUIRED_COLUMNS = {
    "schema_version",
    "record_id",
    "source_file",
    "sha256",
    "label_original",
    "leaf_group_id",
    "group_method",
    "group_evidence",
    "record_status",
}
MANIFEST_COLUMNS = (
    "schema_version",
    "sample_id",
    "source_dataset",
    "source_version",
    "source_path",
    "source_sha256",
    "pixel_sha256",
    "label_original",
    "class_id",
    "group_id",
    "group_evidence",
    "split",
    "derived_path",
    "bbox",
)
MAX_PIXELS = 20_000_000
PIXEL_HASH_CONVENTION = (
    "SHA-256 of b'RGB\\0' + ASCII('<width>x<height>') + b'\\0' + "
    "row-major RGB bytes after Pillow EXIF transpose and RGB conversion"
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _nonblank(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _http_url(value: Any) -> bool:
    if not _nonblank(value):
        return False
    try:
        parsed = urlsplit(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.hostname)
    except ValueError:
        return False


def _source_path(root: Path, value: str) -> Path:
    if not value.strip() or "\x00" in value:
        raise ValueError("invalid_path")
    if (
        Path(value).is_absolute()
        or PureWindowsPath(value).drive
        or PureWindowsPath(value).root
    ):
        raise ValueError("outside_root")
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("outside_root")
    return resolved


def _validate_provenance(
    provenance: dict[str, Any], root: Path, manifest_hash: str
) -> None:
    if (
        not isinstance(provenance, dict)
        or type(provenance.get("schema_version")) is not int
        or provenance["schema_version"] != 1
    ):
        raise ValueError("provenance schema_version must be 1")
    for field in ("source_dataset", "source_version", "snapshot_version"):
        if not _nonblank(provenance.get(field)):
            raise ValueError(f"provenance {field} is required")
    if not _http_url(provenance.get("source_url")):
        raise ValueError("provenance source_url must be HTTP(S) with a host")
    authors = provenance.get("authors")
    if (
        not isinstance(authors, list)
        or not authors
        or not all(_nonblank(a) for a in authors)
    ):
        raise ValueError("provenance authors must be nonempty")
    scope = provenance.get("use_scope")
    if (
        not isinstance(scope, list)
        or "local_research_audit" not in scope
        or not all(_nonblank(s) for s in scope)
    ):
        raise ValueError("provenance use_scope must include local_research_audit")
    if provenance.get("input_manifest_sha256") != manifest_hash:
        raise ValueError("provenance input_manifest_sha256 mismatch")
    review = provenance.get("review")
    if (
        not isinstance(review, dict)
        or not _nonblank(review.get("reviewer"))
        or not _nonblank(review.get("method"))
        or review.get("expert_verified") is not False
    ):
        raise ValueError(
            "provenance review requires reviewer, method and expert_verified=false"
        )
    checked_at = review.get("checked_at")
    if not isinstance(checked_at, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}", checked_at
    ):
        raise ValueError("provenance review checked_at must be YYYY-MM-DD")
    try:
        date.fromisoformat(checked_at)
    except ValueError as error:
        raise ValueError("provenance review checked_at is invalid") from error
    license_info = provenance.get("license")
    if (
        not isinstance(license_info, dict)
        or license_info.get("status") != "verified"
        or not _nonblank(license_info.get("identifier"))
    ):
        raise ValueError("license must have a verified status and identifier")
    evidence = license_info.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("license evidence is required")
    for item in evidence:
        if (
            not isinstance(item, dict)
            or not _nonblank(item.get("file"))
            or not _http_url(item.get("url"))
            or not _nonblank(item.get("locator"))
            or not isinstance(item.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", item["sha256"])
        ):
            raise ValueError("license evidence requires file, sha256, URL and locator")
        try:
            path = _source_path(root, item["file"])
        except ValueError as error:
            raise ValueError(f"license evidence path: {error}") from error
        if not path.is_file() or _sha256(path) != item["sha256"].lower():
            raise ValueError("license evidence missing or hash mismatch")


def _decode_rgb(stream: BinaryIO) -> Image.Image:
    """Return detached, oriented pixels; audit and loader share the decode limit."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(stream) as image:
                if image.width * image.height > MAX_PIXELS:
                    raise Image.DecompressionBombError("pixel_limit")
                image.load()
                with ImageOps.exif_transpose(image) as oriented:
                    return oriented.convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("pixel_limit") from error
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
        EOFError,
        NotImplementedError,
    ) as error:
        raise ValueError("decode_error") from error


def _pixel_hash_stream(stream: BinaryIO) -> tuple[str, str]:
    try:
        with _decode_rgb(stream) as rgb:
            prefix = f"RGB\0{rgb.width}x{rgb.height}\0".encode("ascii")
            digest = hashlib.sha256(prefix)
            digest.update(rgb.tobytes())
            return digest.hexdigest(), ""
    except ValueError as error:
        return "", str(error)


def _pixel_hash(path: Path) -> tuple[str, str]:
    with path.open("rb") as stream:
        return _pixel_hash_stream(stream)


def _audit_row(row: dict[str, str], root: Path) -> tuple[str, str]:
    if row["label_original"] not in LABEL_IDS:
        return "", "unsupported_label"
    status = row["record_status"]
    if status != "ready_for_split":
        reason = (
            f"status_{status}"
            if status in {"needs_review", "excluded_exact_duplicate"}
            else "status_not_ready"
        )
        return "", reason
    if not row["leaf_group_id"].strip():
        return "", "missing_group"
    if row["group_method"] != "casefold_nfc_stem_lookup":
        return "", "unsupported_group_method"
    try:
        evidence = json.loads(row["group_evidence"])
    except json.JSONDecodeError:
        return "", "invalid_group_evidence"
    if evidence == []:
        return "", "missing_group_evidence"
    if (
        not isinstance(evidence, list)
        or not evidence
        or any(
            not isinstance(item, dict)
            or not _nonblank(item.get("source_file"))
            or not _nonblank(item.get("source_record"))
            for item in evidence
        )
    ):
        return "", "invalid_group_evidence"
    try:
        path = _source_path(root, row["source_file"])
    except ValueError as error:
        return "", str(error)
    except OSError:
        return "", "read_error"
    try:
        if not path.is_file():
            return "", "missing_file"
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            if source_hash != row["sha256"].lower():
                return "", "hash_mismatch"
            stream.seek(0)
            pixel_hash = _pixel_hash_stream(stream)
            after = os.fstat(stream.fileno())
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                return "", "source_changed"
            return pixel_hash
    except FileNotFoundError:
        return "", "missing_file"
    except OSError:
        return "", "read_error"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def audit_manifest(
    input_csv: Path, source_root: Path, output_dir: Path, provenance: dict[str, Any]
) -> dict[str, int]:
    """Publish four metadata artifacts only after structural and permission checks pass."""
    root = source_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("source_root must be a directory")
    output = output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("output directory already exists")
    if output.is_relative_to(root):
        raise ValueError("output must be outside source_root")
    manifest_hash = _sha256(input_csv)
    _validate_provenance(provenance, root, manifest_hash)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    counts = {"accepted": 0, "excluded": 0}
    seen: set[str] = set()
    try:
        with (
            input_csv.open(encoding="utf-8-sig", newline="") as source,
            (staging / "manifest.csv").open(
                "w", encoding="utf-8", newline=""
            ) as accepted,
            (staging / "exclusions.csv").open(
                "w", encoding="utf-8", newline=""
            ) as excluded,
        ):
            reader = csv.DictReader(source, strict=True)
            columns = reader.fieldnames
            if (
                not columns
                or len(set(columns)) != len(columns)
                or not REQUIRED_COLUMNS.issubset(columns)
            ):
                raise ValueError("manifest missing or duplicate required headers")
            manifest_writer = csv.DictWriter(accepted, fieldnames=MANIFEST_COLUMNS)
            exclusion_writer = csv.DictWriter(
                excluded,
                fieldnames=("sample_id", "source_path", "label_original", "reason"),
            )
            manifest_writer.writeheader()
            exclusion_writer.writeheader()
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(
                        f"manifest malformed row at line {reader.line_num}"
                    )
                if row["schema_version"] not in {"1", "1.0"}:
                    raise ValueError("manifest schema_version must be 1 or 1.0")
                sample_id = row["record_id"]
                if not sample_id.strip() or sample_id in seen:
                    raise ValueError("manifest blank or duplicate record_id")
                seen.add(sample_id)
                pixel_hash, reason = _audit_row(row, root)
                if reason:
                    exclusion_writer.writerow(
                        {
                            "sample_id": sample_id,
                            "source_path": row["source_file"],
                            "label_original": row["label_original"],
                            "reason": reason,
                        }
                    )
                    counts["excluded"] += 1
                    continue
                manifest_writer.writerow(
                    {
                        "schema_version": 1,
                        "sample_id": sample_id,
                        "source_dataset": provenance["source_dataset"],
                        "source_version": provenance["source_version"],
                        "source_path": Path(row["source_file"]).as_posix(),
                        "source_sha256": row["sha256"].lower(),
                        "pixel_sha256": pixel_hash,
                        "label_original": row["label_original"],
                        "class_id": LABEL_IDS[row["label_original"]],
                        "group_id": row["leaf_group_id"],
                        "group_evidence": row["group_evidence"],
                        "split": "",
                        "derived_path": "",
                        "bbox": "",
                    }
                )
                counts["accepted"] += 1
        if _sha256(input_csv) != manifest_hash:
            raise ValueError("manifest changed during audit")
        _write_json(
            staging / "label_map.json",
            {
                "schema_version": 1,
                "classes": [
                    {
                        "index": i,
                        "class_id": class_id,
                        "label_original": label,
                        "name_vi": name,
                    }
                    for i, (class_id, label, name) in enumerate(CLASSES)
                ],
            },
        )
        _write_json(
            staging / "provenance.json",
            {
                **provenance,
                "input_manifest_sha256": manifest_hash,
                "counts": counts,
                "source_root": Path(os.path.relpath(root, output)).as_posix(),
                "pixel_hash_convention": PIXEL_HASH_CONVENTION,
                "audit_stage": "M1-only",
                "split_sealed": False,
            },
        )
        if output.exists() or output.is_symlink():
            raise ValueError("output directory already exists")
        staging.rename(output)
    except (csv.Error, UnicodeDecodeError) as error:
        raise ValueError(f"manifest CSV invalid: {error}") from error
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return counts


class ManifestDataset:
    """Sealed map-style samples; training callers must limit themselves to train/val."""

    def __init__(
        self, dataset_dir: Path, split: str, transform: Callable[[Image.Image], Any]
    ) -> None:
        from ai.split import _csv, _json, verify_seal

        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        if not callable(transform):
            raise TypeError("transform must be callable")
        root = dataset_dir.resolve()
        verify_seal(root)
        source = _json(root, "provenance.json").get("source_root")
        if (
            not isinstance(source, str)
            or not source.strip()
            or "\x00" in source
            or (
                Path(source).is_absolute()
                or PureWindowsPath(source).drive
                or PureWindowsPath(source).root
            )
        ):
            raise ValueError("source_root must be relative")
        self.source_root = (root / source).resolve(strict=True)
        if not self.source_root.is_dir():
            raise ValueError("source_root must be a directory")
        classes = _json(root, "label_map.json")["classes"]
        self.class_ids = tuple(item["class_id"] for item in classes)
        self._indices = {item["class_id"]: item["index"] for item in classes}
        self._rows = [
            row
            for row in _csv(root, "manifest.csv", MANIFEST_COLUMNS)
            if row["split"] == split
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> tuple[Any, int, str]:
        row = self._rows[index]
        path = _source_path(self.source_root, row["source_path"])
        with path.open("rb") as stream:
            if (
                hashlib.file_digest(stream, "sha256").hexdigest()
                != row["source_sha256"]
            ):
                raise ValueError(f"source_sha256 mismatch: {row['sample_id']}")
            stream.seek(0)
            with _decode_rgb(stream) as rgb:
                result = self.transform(rgb)
        return result, self._indices[row["class_id"]], row["sample_id"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser(
        "audit", help="Audit source manifest into a new metadata directory"
    )
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--source-root", type=Path, required=True)
    audit.add_argument("--provenance", type=Path, required=True)
    audit.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
        counts = audit_manifest(args.input, args.source_root, args.out, provenance)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"audit: {error}", file=sys.stderr)
        return 1
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
