"""Exercise the FieldPlant selective ZIP reader and holdout exclusions offline."""

import io
import json
import os
import zipfile
from contextlib import AbstractContextManager
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "docs/research/yolo26s-field-colab-2026-11-10"


@pytest.fixture
def source(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPT_DIR))
    return import_module("fieldplant_source_audit")


class RangeFixture:
    def __init__(self, raw: bytes):
        self.raw = raw
        self.reads: list[tuple[int, int]] = []
        self.status = 206
        self.range_header = ""
        self.etag = '"fixture"'

    def get(
        self, url: str, *, headers: dict[str, str], timeout: tuple[int, int], stream: bool
    ) -> AbstractContextManager[SimpleNamespace]:
        del timeout
        assert stream and headers["Accept-Encoding"] == "identity"
        start, last = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        self.reads.append((start, last + 1))
        response = SimpleNamespace(
            status_code=self.status,
            url=url,
            headers={
                "Content-Range": self.range_header or f"bytes {start}-{last}/{len(self.raw)}",
                "ETag": self.etag,
            },
            raw=io.BytesIO(self.raw[start : last + 1]),
        )

        class Response:
            def __enter__(self) -> SimpleNamespace:
                return response

            def __exit__(self, *args: object) -> None:
                response.raw.close()

        return Response()


def test_reads_only_selected_local_records_and_zip64_directory(source: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    data = io.BytesIO()
    with monkeypatch.context() as writer:
        writer.setattr(zipfile, "ZIP64_LIMIT", 32)
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("train/not-tomato.jpg", b"x" * 150)
            archive.writestr("train/tomato.jpg", b"tomato" * 30)
    fixture = RangeFixture(data.getvalue())
    reader = source.RangeReader(fixture, "https://fixture.invalid", len(fixture.raw), fixture.etag)
    with zipfile.ZipFile(reader) as archive:
        row = {"archive_member": "train/tomato.jpg"}
        [(start, end, [selected])] = source.member_ranges(archive, [row])
        assert selected is row
        assert start > archive.getinfo("train/not-tomato.jpg").header_offset + 150
        initial_reads = len(fixture.reads)
        reader.window = (start, reader.fetch(start, end))
        assert archive.read(row["archive_member"]) == b"tomato" * 30
        assert fixture.reads[initial_reads:] == [(start, end)]
        assert row["zip_entry"]["uncompressed_bytes"] == 180
        assert reader.seek(0, os.SEEK_END) == len(fixture.raw)


@pytest.mark.parametrize(("attribute", "value"), [("status", 200), ("range_header", "bytes 1-9/10"), ("etag", "changed")])
def test_rejects_unverified_range(source: ModuleType, attribute: str, value: object) -> None:
    fixture = RangeFixture(b"0123456789")
    reader = source.RangeReader(fixture, "https://fixture.invalid", len(fixture.raw), fixture.etag)
    setattr(fixture, attribute, value)
    with pytest.raises(ValueError, match="unexpected ZIP range"):
        reader.fetch(0, 10)
    assert reader.transfer_bytes == 0


def test_keeps_original_labels_and_propagates_audit_exclusions(source: ModuleType) -> None:
    rows, _ = source.source_records()
    for row in rows:
        row["acquisition"] = {"status": "not_acquired"}
    by_id = {row["id"]: row for row in rows}
    boxes = [box for row in rows for box in row["annotations"]]
    assert len(boxes) == 2786
    assert sum(box["target_status"] == "excluded_healthy" for box in boxes) == 279
    assert {box["source_label"] for box in boxes} >= {"Tomato Brown Spots", "Tomato blight leaf"}
    assert not any(row["training_admitted"] or row["final_holdout_eligible"] for row in rows)
    source.record_groups(
        rows,
        {"records": [{"id": "fieldplant-3216", "source": "fieldplant-v11-kaggle-v1"}]},
        [],
        {"known_byte_sha256": [], "known_pixel_sha256": [], "source_records": []},
        {"protected_byte_sha256": [], "protected_pixel_sha256": []},
    )
    assert by_id["fieldplant-3216"]["provisional_group"] == by_id["fieldplant-3323"]["provisional_group"]
    assert by_id["fieldplant-3323"]["group_exclusion_reasons"] == ["prior_visual_audit"]


def test_marks_conflicting_annotations_on_identical_pixels(source: ModuleType) -> None:
    rows, _ = source.source_records()
    by_id = {row["id"]: row for row in rows}
    keys = ["fieldplant-280", "fieldplant-2722"]
    [conflict] = source.annotation_conflicts(rows, [keys])
    assert conflict == {
        "ids": keys,
        "box_counts": [1, 1],
        "source_category_ids": [[26], [22]],
        "box_counts_differ": False,
        "category_sets_differ": True,
        "status": "unresolved",
    }
    assert {by_id[key]["annotation_conflict_status"] for key in keys} == {
        "unresolved_exact_pixel_duplicate",
    }


def test_resolution_quarantines_entire_group_and_keeps_prior_exposure(source: ModuleType) -> None:
    def record(key: int, group: int, pixels: str, category: int, reasons: list[str]) -> dict[str, object]:
        return {
            "id": f"fieldplant-{key}",
            "image": {"id": key},
            "acquisition": {"pixel_sha256": pixels},
            "provisional_group": f"fieldplant-{group}",
            "group_exclusion_reasons": reasons,
            "annotations": [{
                "category_id": category,
                "bbox": [0, 0, 1, 1],
                "source_label": "Tomato healthy" if category == 25 else "Tomato Brown Spots",
                "target_status": "excluded_healthy" if category == 25 else "disease_scope_pending",
            }],
        }

    exposure = ["prior_visual_audit"]
    rows = [
        record(1, 1, "same", 22, exposure),
        record(2, 1, "same", 25, exposure),
        record(3, 1, "related", 22, exposure),
        record(4, 4, "disease", 22, []),
        record(5, 5, "healthy", 25, []),
        record(6, 6, "previously-opened", 22, exposure),
    ]
    conflicts = source.annotation_conflicts(rows, [["fieldplant-1", "fieldplant-2"]])
    manifest = {"records": rows, "grouping": {"annotation_conflicts_on_exact_pixels": conflicts}}
    original_manifest = json.dumps(manifest)
    ledger = source.conflict_resolution(manifest)

    assert json.dumps(manifest) == original_manifest
    assert ledger["groups"]["fieldplant-1"] == {
        "conflicting_record_ids": ["fieldplant-1", "fieldplant-2"],
        "holdout_exclusion_reasons": ["prior_visual_audit", "unresolved_exact_pixel_annotation_conflict"],
        "records": dict.fromkeys(("fieldplant-1", "fieldplant-2", "fieldplant-3"), "conflict_quarantine"),
    }
    assert ledger["groups"]["fieldplant-4"]["records"] == {"fieldplant-4": "disease_visual_review_candidate"}
    assert ledger["groups"]["fieldplant-5"]["records"] == {"fieldplant-5": "negative_visual_review_candidate"}
    assert ledger["groups"]["fieldplant-6"]["records"] == {"fieldplant-6": "prior_exposure_group"}
    assert ledger["summary"]["conflict_quarantine"]["unique_pixel_images"] == 2
    assert ledger["summary"]["disease_visual_review_candidate"]["source_classes"] == {
        "Tomato Brown Spots": {"unique_pixel_images": 1, "provisional_groups": 1, "source_boxes": 1},
    }
    assert not ledger["training_admitted"] and not ledger["final_holdout_established"]
    rows.reverse()
    conflicts[0]["ids"].reverse()
    assert json.dumps(source.conflict_resolution(manifest)) == json.dumps(ledger)


def test_resolution_rejects_unpinned_manifest(
    source: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(source, "OUTPUT", tmp_path)
    (tmp_path / "source-manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="source manifest SHA-256 changed"):
        source.resolve_conflicts()
    assert not (tmp_path / "conflict-resolution.json").exists()
