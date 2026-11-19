"""Public audit contract: quarantine, permission gates, immutable source and CLI."""

import csv
import hashlib
import json
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any, BinaryIO

import pytest
from PIL import Image

from ai import data
from ai.data import audit_manifest
from tests.ai.conftest import AuditCase


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("schema_version", ["1.0", "1"])
def test_input_schema_versions_emit_canonical_schema(
    audit_case: AuditCase, schema_version: str
) -> None:
    c = audit_case
    for row in c.rows:
        row["schema_version"] = schema_version
    c.write_rows()
    assert audit_manifest(c.csv, c.root, c.out, c.provenance) == {
        "accepted": 1,
        "excluded": 2,
    }
    assert [row["schema_version"] for row in read_csv(c.out / "manifest.csv")] == ["1"]
    for filename in ["label_map.json", "provenance.json"]:
        assert (
            json.loads((c.out / filename).read_text(encoding="utf-8"))["schema_version"]
            == 1
        )


@pytest.mark.parametrize(
    "schema_version",
    ["2", "1.1", "1.00", "1e0", "+1", "01", "1.0.1", "1.0x", " 1", "1 ", ""],
)
def test_unknown_input_schema_aborts_atomically(
    audit_case: AuditCase, schema_version: str
) -> None:
    c = audit_case
    c.rows[0]["schema_version"] = "1"
    c.rows[1]["schema_version"] = schema_version
    c.write_rows()
    with pytest.raises(ValueError, match="manifest schema_version"):
        audit_manifest(c.csv, c.root, c.out, c.provenance)
    assert not c.out.exists()
    assert not list(c.out.parent.glob(".audit-*"))


def test_unsupported_dds_is_quarantined_without_losing_valid_png(
    audit_case: AuditCase,
) -> None:
    c = audit_case
    payload = bytearray(128)
    payload[:4] = b"DDS "
    struct.pack_into("<I", payload, 4, 124)
    struct.pack_into("<3I", payload, 8, 0, 8, 8)
    struct.pack_into("<4I", payload, 76, 32, 4, int.from_bytes(b"ZZZZ", "little"), 0)
    (c.root / "broken.png").write_bytes(payload)
    c.rows = [c.rows[0], {**c.rows[1], "sha256": hashlib.sha256(payload).hexdigest()}]
    c.write_rows()
    before = {p: p.read_bytes() for p in c.root.iterdir()}

    assert audit_manifest(c.csv, c.root, c.out, c.provenance) == {
        "accepted": 1,
        "excluded": 1,
    }
    assert [row["sample_id"] for row in read_csv(c.out / "manifest.csv")] == [
        "fixture:leaf"
    ]
    exclusions = read_csv(c.out / "exclusions.csv")
    assert len(exclusions) == 1
    assert exclusions[0]["sample_id"] == "fixture:broken"
    assert exclusions[0]["reason"] == "decode_error"
    assert {p: p.read_bytes() for p in c.root.iterdir()} == before


def test_audit_preserves_raw_and_excludes_invalid(audit_case: AuditCase) -> None:
    c = audit_case
    before = {p: p.read_bytes() for p in c.root.iterdir()}
    assert audit_manifest(c.csv, c.root, c.out, c.provenance) == {
        "accepted": 1,
        "excluded": 2,
    }
    assert {p: p.read_bytes() for p in c.root.iterdir()} == before
    assert [r["reason"] for r in read_csv(c.out / "exclusions.csv")] == [
        "decode_error",
        "outside_root",
    ]
    rows = read_csv(c.out / "manifest.csv")
    assert len(rows) == 1
    assert rows[0] == {
        "schema_version": "1",
        "sample_id": "fixture:leaf",
        "source_dataset": "synthetic-fixture",
        "source_version": "fixture-v1",
        "source_path": "leaf.png",
        "source_sha256": hashlib.sha256(before[c.raw]).hexdigest(),
        "pixel_sha256": "33e702ec57e4b11d09485a5a9d4001015adc51efb108e734ebbb7b867fc78ec8",
        "label_original": "Tomato___Bacterial_spot",
        "class_id": "bacterial_spot",
        "group_id": "fixture:group-1",
        "group_evidence": c.rows[0]["group_evidence"],
        "split": "",
        "derived_path": "",
        "bbox": "",
    }
    assert {p.name for p in c.out.iterdir()} == {
        "manifest.csv",
        "exclusions.csv",
        "label_map.json",
        "provenance.json",
    }
    provenance = json.loads((c.out / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["input_manifest_sha256"] == c.provenance["input_manifest_sha256"]
    assert (c.out / provenance["source_root"]).resolve() == c.root.resolve()
    assert provenance["counts"] == {"accepted": 1, "excluded": 2}
    assert provenance["audit_stage"] == "M1-only"
    assert provenance["split_sealed"] is False
    assert provenance["review"]["expert_verified"] is False
    assert provenance["notes"] == c.provenance["notes"]


@pytest.mark.parametrize("operation", ["file_digest", "_pixel_hash_stream"])
def test_unreadable_image_is_quarantined_without_losing_valid_png(
    audit_case: AuditCase, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    c = audit_case
    good = c.root / "good.png"
    good.write_bytes(c.raw.read_bytes())
    c.rows = [c.rows[0], {**c.rows[0], "record_id": "good", "source_file": good.name}]
    c.write_rows()
    before = {p: p.read_bytes() for p in c.root.iterdir()}
    module = hashlib if operation == "file_digest" else data
    original = getattr(module, operation)

    def inaccessible(stream: BinaryIO, *args: Any) -> Any:
        if Path(stream.name) == c.raw:
            raise PermissionError("synthetic read denial")
        return original(stream, *args)

    monkeypatch.setattr(module, operation, inaccessible)
    assert audit_manifest(c.csv, c.root, c.out, c.provenance) == {
        "accepted": 1,
        "excluded": 1,
    }
    assert [row["sample_id"] for row in read_csv(c.out / "manifest.csv")] == ["good"]
    assert read_csv(c.out / "exclusions.csv")[0]["reason"] == "read_error"
    assert {p: p.read_bytes() for p in c.root.iterdir()} == before


@pytest.mark.parametrize(
    "mutation", ["missing", "pending", "evidence", "hash", "escape"]
)
def test_missing_license_blocks_release(audit_case: AuditCase, mutation: str) -> None:
    c = audit_case
    if mutation == "missing":
        del c.provenance["license"]
    elif mutation == "pending":
        c.provenance["license"]["status"] = "pending"
    elif mutation == "evidence":
        c.provenance["license"]["evidence"] = []
    elif mutation == "hash":
        c.provenance["license"]["evidence"][0]["sha256"] = "0" * 64
    else:
        c.provenance["license"]["evidence"][0]["file"] = "../outside.png"
    with pytest.raises(ValueError, match="license"):
        audit_manifest(c.csv, c.root, c.out, c.provenance)
    assert not c.out.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("source_dataset", " "),
        ("source_version", ""),
        ("snapshot_version", ""),
        ("source_url", "file:///tmp/a"),
        ("source_url", "https:///no-host"),
        ("authors", []),
        ("authors", [""]),
        ("use_scope", ["redistribution"]),
        ("input_manifest_sha256", "0" * 64),
        (
            "review",
            {
                "reviewer": "fixture",
                "method": "check",
                "checked_at": "2026-02-30",
                "expert_verified": False,
            },
        ),
        (
            "review",
            {
                "reviewer": "fixture",
                "method": "check",
                "checked_at": "2026-11-02",
                "expert_verified": True,
            },
        ),
    ],
)
def test_invalid_provenance_blocks_output(
    audit_case: AuditCase, field: str, value: Any
) -> None:
    c = audit_case
    c.provenance[field] = value
    with pytest.raises(ValueError, match="provenance"):
        audit_manifest(c.csv, c.root, c.out, c.provenance)
    assert not c.out.exists()


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("sha256", "0" * 64, "hash_mismatch"),
        ("sha256", "not-hex", "hash_mismatch"),
        ("leaf_group_id", "", "missing_group"),
        ("group_method", "unresolved", "unsupported_group_method"),
        ("group_evidence", "[]", "missing_group_evidence"),
        ("group_evidence", "invalid-json", "invalid_group_evidence"),
        ("group_evidence", '[{"source_file":"map.json"}]', "invalid_group_evidence"),
        (
            "group_evidence",
            '[{"source_file":"map.json","source_record":2}]',
            "invalid_group_evidence",
        ),
        ("label_original", "Tomato___Target_Spot", "unsupported_label"),
        ("record_status", "needs_review", "status_needs_review"),
        (
            "record_status",
            "excluded_exact_duplicate",
            "status_excluded_exact_duplicate",
        ),
        ("record_status", "unexpected", "status_not_ready"),
        ("source_file", "missing.png", "missing_file"),
        ("source_file", "", "invalid_path"),
        ("source_file", "C:\\outside.png", "outside_root"),
    ],
)
def test_quarantine_reason(
    audit_case: AuditCase, field: str, value: str, reason: str
) -> None:
    c = audit_case
    c.rows = [{**c.rows[0], field: value}]
    c.write_rows()
    assert audit_manifest(c.csv, c.root, c.out, c.provenance) == {
        "accepted": 0,
        "excluded": 1,
    }
    assert read_csv(c.out / "exclusions.csv")[0]["reason"] == reason


def test_explicit_mapping_ignores_legacy_status_code(audit_case: AuditCase) -> None:
    c = audit_case
    labels = [
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___healthy",
    ]
    c.rows = [
        {
            **c.rows[0],
            "record_id": f"fixture:{i}",
            "label_original": label,
            "status_code": "99",
        }
        for i, label in enumerate(labels)
    ]
    c.write_rows()
    assert audit_manifest(c.csv, c.root, c.out, c.provenance) == {
        "accepted": 8,
        "excluded": 0,
    }
    expected = [
        "bacterial_spot",
        "early_blight",
        "late_blight",
        "leaf_mold",
        "septoria_leaf_spot",
        "spider_mites",
        "yellow_leaf_curl",
        "healthy",
    ]
    assert [r["class_id"] for r in read_csv(c.out / "manifest.csv")] == expected
    mapping = json.loads((c.out / "label_map.json").read_text(encoding="utf-8"))
    assert mapping["schema_version"] == 1
    assert [r["class_id"] for r in mapping["classes"]] == expected
    assert [r["index"] for r in mapping["classes"]] == list(range(8))
    assert [r["label_original"] for r in mapping["classes"]] == labels
    assert mapping["classes"][0]["name_vi"] == "Đốm vi khuẩn"


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_pixel_hash_normalizes_rgb(audit_case: AuditCase, mode: str) -> None:
    c = audit_case
    image = Image.new("RGB", (8, 8), (10, 10, 10)).convert(mode)
    if mode == "P":
        image = Image.new("P", (8, 8), 0)
        image.putpalette([10, 10, 10] + [0] * 765)
    image.save(c.raw)
    c.rows = [{**c.rows[0], "sha256": hashlib.sha256(c.raw.read_bytes()).hexdigest()}]
    c.write_rows()
    audit_manifest(c.csv, c.root, c.out, c.provenance)
    expected = hashlib.sha256(b"RGB\x008x8\x00" + bytes([10, 10, 10]) * 64).hexdigest()
    assert read_csv(c.out / "manifest.csv")[0]["pixel_sha256"] == expected


def test_pixel_hash_applies_exif_orientation(audit_case: AuditCase) -> None:
    c = audit_case
    image = Image.new("RGB", (2, 1))
    image.putdata([(255, 0, 0), (0, 255, 0)])
    exif = Image.Exif()
    exif[274] = 6
    image.save(c.raw, exif=exif)
    before = c.raw.read_bytes()
    c.rows = [{**c.rows[0], "sha256": hashlib.sha256(before).hexdigest()}]
    c.write_rows()
    audit_manifest(c.csv, c.root, c.out, c.provenance)
    assert read_csv(c.out / "manifest.csv")[0]["pixel_sha256"] == (
        "63ce976da26f60068a194807a5c94f7c1c492d24ace796d433fb33bdcadf36bf"
    )
    assert c.raw.read_bytes() == before


@pytest.mark.parametrize("size", [(5000, 4001), (20000, 20000)])
def test_pixel_limit_prevents_loading_large_images(
    audit_case: AuditCase, size: tuple[int, int]
) -> None:
    c = audit_case

    # Header-only PNG exercises the pre-load limit without allocating a huge image.
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    c.raw.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", *size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", b"")
        + chunk(b"IEND", b"")
    )
    c.rows = [{**c.rows[0], "sha256": hashlib.sha256(c.raw.read_bytes()).hexdigest()}]
    c.write_rows()
    before_limit = Image.MAX_IMAGE_PIXELS
    assert audit_manifest(c.csv, c.root, c.out, c.provenance) == {
        "accepted": 0,
        "excluded": 1,
    }
    assert read_csv(c.out / "exclusions.csv")[0]["reason"] == "pixel_limit"
    assert Image.MAX_IMAGE_PIXELS == before_limit


@pytest.mark.parametrize(
    "problem",
    [
        "missing_header",
        "duplicate_header",
        "extra_field",
        "short_row",
        "bad_csv",
        "duplicate_id",
        "schema",
        "blank_id",
    ],
)
def test_malformed_input_aborts_atomically(audit_case: AuditCase, problem: str) -> None:
    c = audit_case
    if problem == "duplicate_id":
        c.rows[1]["record_id"] = c.rows[0]["record_id"]
    elif problem == "schema":
        c.rows[1]["schema_version"] = "2"
    elif problem == "blank_id":
        c.rows[1]["record_id"] = ""
    c.write_rows()
    if problem == "missing_header":
        c.csv.write_text("record_id,source_file\na,leaf.png\n", encoding="utf-8")
    elif problem == "duplicate_header":
        c.csv.write_text(
            c.csv.read_text(encoding="utf-8").replace("source_uuid,", "record_id,", 1),
            encoding="utf-8",
        )
    elif problem == "extra_field":
        with c.csv.open("a", encoding="utf-8") as stream:
            stream.write("," * 31 + "extra\n")
    elif problem == "short_row":
        with c.csv.open("a", encoding="utf-8") as stream:
            stream.write("1,short\n")
    elif problem == "bad_csv":
        with c.csv.open("a", encoding="utf-8") as stream:
            stream.write('1,"unclosed')
    c.provenance["input_manifest_sha256"] = hashlib.sha256(
        c.csv.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="manifest|schema|record_id|CSV"):
        audit_manifest(c.csv, c.root, c.out, c.provenance)
    assert not c.out.exists()
    assert not list(c.out.parent.glob(".audit-*"))


@pytest.mark.parametrize(
    "kind", ["empty", "populated", "source_child", "source_itself"]
)
def test_output_confinement(audit_case: AuditCase, kind: str) -> None:
    c = audit_case
    if kind in {"empty", "populated"}:
        c.out.mkdir()
        if kind == "populated":
            (c.out / "keep.txt").write_bytes(b"keep")
    else:
        c.out = c.root / "derived" if kind == "source_child" else c.root
    before = {p: p.read_bytes() for p in c.root.iterdir()}
    with pytest.raises(ValueError, match="output"):
        audit_manifest(c.csv, c.root, c.out, c.provenance)
    assert {p: p.read_bytes() for p in c.root.iterdir()} == before
    if kind == "populated":
        assert (c.out / "keep.txt").read_bytes() == b"keep"


def test_symlink_escape_is_excluded(audit_case: AuditCase) -> None:
    c = audit_case
    link = c.root / "linked.png"
    try:
        link.symlink_to(c.root.parent / "outside.png")
    except OSError as error:
        if os.name == "nt" and error.winerror == 1314:
            pytest.skip("Windows symlink privilege unavailable (1314)")
        raise
    c.rows = [{**c.rows[0], "source_file": "linked.png"}]
    c.write_rows()
    assert audit_manifest(c.csv, c.root, c.out, c.provenance) == {
        "accepted": 0,
        "excluded": 1,
    }
    assert read_csv(c.out / "exclusions.csv")[0]["reason"] == "outside_root"


@pytest.mark.skipif(os.name != "nt", reason="Windows junction test")
def test_junction_escape_is_excluded(audit_case: AuditCase) -> None:
    c = audit_case
    outside = c.root.parent / "junction-target"
    outside.mkdir()
    Image.new("RGB", (8, 8)).save(outside / "outside.png")
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(c.root / "linked"), str(outside)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode and "Access is denied" in result.stderr:
        pytest.skip("Windows sandbox denies junction creation")
    assert result.returncode == 0, result.stdout + result.stderr
    c.rows = [{**c.rows[0], "source_file": "linked/outside.png"}]
    c.write_rows()
    assert audit_manifest(c.csv, c.root, c.out, c.provenance) == {
        "accepted": 0,
        "excluded": 1,
    }
    assert read_csv(c.out / "exclusions.csv")[0]["reason"] == "outside_root"


def test_output_is_deterministic(audit_case: AuditCase) -> None:
    c = audit_case
    second = c.out.with_name("audit-second")
    audit_manifest(c.csv, c.root, c.out, c.provenance)
    audit_manifest(c.csv, c.root, second, c.provenance)
    assert {p.name: p.read_bytes() for p in c.out.iterdir()} == {
        p.name: p.read_bytes() for p in second.iterdir()
    }


@pytest.mark.parametrize("valid", [True, False])
def test_cli_fixture_audit(audit_case: AuditCase, valid: bool) -> None:
    c = audit_case
    if not valid:
        c.provenance["license"]["status"] = "pending"
    provenance_path = c.root.parent / "provenance-input.json"
    provenance_path.write_text(json.dumps(c.provenance), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai.data",
            "audit",
            "--input",
            str(c.csv),
            "--source-root",
            str(c.root),
            "--provenance",
            str(provenance_path),
            "--out",
            str(c.out),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
    )
    if valid:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {"accepted": 1, "excluded": 2}
        assert result.stderr == ""
    else:
        assert result.returncode != 0
        assert "license" in result.stderr
        assert "Traceback" not in result.stderr
        assert not c.out.exists()
