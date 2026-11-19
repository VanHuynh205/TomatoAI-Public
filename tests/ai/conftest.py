"""Synthetic inputs only; no dataset permissions are inferred by these fixtures."""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from PIL import Image


@pytest.fixture
def transform_spec() -> dict[str, Any]:
    profiles = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "docs/experiments/preprocessing-v1.json"
        ).read_text(encoding="utf-8")
    )
    return dict(profiles["reference_profiles"]["efficientnet_b0"])


@dataclass
class AuditCase:
    csv: Path
    root: Path
    out: Path
    provenance: dict[str, Any]
    raw: Path
    rows: list[dict[str, str]]

    def write_rows(self) -> None:
        with self.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.provenance["input_manifest_sha256"] = hashlib.sha256(
            self.csv.read_bytes()
        ).hexdigest()


@pytest.fixture
def audit_case(tmp_path: Path) -> AuditCase:
    root = tmp_path / "source"
    root.mkdir()
    raw = root / "leaf.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(raw)
    (root / "broken.png").write_bytes(b"not an image")
    Image.new("RGB", (8, 8)).save(tmp_path / "outside.png")
    permission = root / "permission.txt"
    permission.write_text(
        "Synthetic fixture permission; not a real license.", encoding="utf-8"
    )
    row = {
        "schema_version": "1.0",
        "record_id": "fixture:leaf",
        "source_file": "leaf.png",
        "source_record": "1",
        "source_uuid": "fixture-uuid",
        "label_original": "Tomato___Bacterial_spot",
        "status_code": "9",
        "variant": "color",
        "bytes": str(raw.stat().st_size),
        "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "decode_ok": "True",
        "image_format": "PNG",
        "width": "8",
        "height": "8",
        "mode": "RGB",
        "channels": "3",
        "exif_keys": "[]",
        "pixel_sha256": "untrusted-old-hash",
        "near_hash": "unused",
        "decode_error": "",
        "lookup_key": "fixture leaf",
        "leaf_group_id": "fixture:group-1",
        "group_method": "casefold_nfc_stem_lookup",
        "group_evidence": json.dumps(
            [
                {"source_file": "leaf-map.json", "source_record": "fixture leaf"},
                {
                    "source_file": "leaf_grouping/filtered_leafmaps/class.csv",
                    "source_record": "1",
                },
            ]
        ),
        "flags": "[]",
        "duplicate_of": "",
        "split_component_id": "not-a-leaf-group",
        "processed_file": "images/leaf.png",
        "record_status": "ready_for_split",
        "ready_for_training": "False",
    }
    rows = [row]
    for sample_id, path, content in [
        ("fixture:broken", "broken.png", b"not an image"),
        ("fixture:outside", "../outside.png", (tmp_path / "outside.png").read_bytes()),
    ]:
        rows.append(
            {
                **row,
                "record_id": sample_id,
                "source_file": path,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    case = AuditCase(
        csv=tmp_path / "input.csv",
        root=root,
        out=tmp_path / "audit",
        raw=raw,
        rows=rows,
        provenance={
            "schema_version": 1,
            "source_dataset": "synthetic-fixture",
            "source_version": "fixture-v1",
            "snapshot_version": "fixture-snapshot-v1",
            "source_url": "https://example.com/synthetic-fixture",
            "authors": ["Synthetic fixture author"],
            "license": {
                "identifier": "LicenseRef-Synthetic-Fixture",
                "status": "verified",
                "evidence": [
                    {
                        "file": "permission.txt",
                        "sha256": hashlib.sha256(permission.read_bytes()).hexdigest(),
                        "url": "https://example.com/synthetic-permission",
                        "locator": "Synthetic fixture only, line 1",
                    }
                ],
            },
            "use_scope": ["local_research_audit"],
            "review": {
                "reviewer": "pytest synthetic fixture",
                "method": "automated fixture consistency check; not legal review",
                "checked_at": "2026-11-02",
                "expert_verified": False,
            },
            "notes": "Synthetic fixture; conveys no rights to any real dataset.",
        },
    )
    case.write_rows()
    return case
