"""Known-exposure guards must not confuse old test protection with model exposure."""

import json
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

guard = import_module("docs.progress.yolo26s-field-colab-2026-11-10.master_exposure")


def test_transitive_cross_source_exposure_keeps_protection_purposes_separate() -> None:
    records: dict[str, Any] = {}
    guard.add_record(records, "baseline", "a" * 64)
    guard.add_record(records, "copy", "a" * 64, [("source_scene", "scene1")])
    guard.add_record(records, "same_scene", "b" * 64, [("source_scene", "scene1")])
    guard.add_record(records, "crop", "c" * 64)
    guard.add_record(records, "old_reserved_test", "d" * 64)
    guard.add_record(records, "unrelated", "e" * 64)
    result = guard.exclusion_sets(records, [("same_scene", "crop")], {"old_reserved_test"}, {"baseline"})
    assert result["blocked_from_training"] == ["old_reserved_test"]
    assert result["excluded_from_final_holdout"] == ["baseline", "copy", "crop", "same_scene"]


def test_filename_groups_are_source_scoped_and_empty_values_never_join() -> None:
    records: dict[str, Any] = {}
    guard.add_record(records, "a", "a" * 64, [("source_a_filename", "IMG_1"), ("pixel", "")])
    guard.add_record(records, "b", "b" * 64, [("source_b_filename", "IMG_1"), ("pixel", "")])
    result = guard.exclusion_sets(records, [], set(), {"a"})
    assert result["excluded_from_final_holdout"] == ["a"]


@pytest.mark.parametrize("mode", ["unknown_seed", "unknown_edge"])
def test_unknown_identity_fails_closed(mode: str) -> None:
    records: dict[str, Any] = {}
    guard.add_record(records, "a", "a" * 64)
    with pytest.raises(ValueError, match="unknown"):
        guard.exclusion_sets(records, [("a", "missing")] if mode == "unknown_edge" else [],
                             set(), {"missing"} if mode == "unknown_seed" else {"a"})


def test_merge_keeps_all_pixel_hashes_and_rejects_changed_image_bytes() -> None:
    records: dict[str, Any] = {}
    guard.add_record(records, "a", "a" * 64, [("pixel", "p1")])
    guard.add_record(records, "a", "a" * 64, [("pixel", "p2")])
    guard.add_record(records, "b", "b" * 64, [("pixel", "p2")])
    assert guard.exclusion_sets(records, [], set(), {"a"})["excluded_from_final_holdout"] == ["a", "b"]
    with pytest.raises(ValueError, match="Conflicting image hash"):
        guard.add_record(records, "a", "c" * 64)


@pytest.mark.parametrize("digest", ["", "bad", "g" * 64])
def test_invalid_image_hash_is_rejected(digest: str) -> None:
    with pytest.raises(ValueError, match="invalid image SHA-256"):
        guard.add_record({}, "a", digest)


def test_versioned_output_never_overwrites_changed_evidence(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    first = guard.write_immutable(path, {"excluded": ["a"]})
    before = path.read_bytes()
    assert guard.write_immutable(path, {"excluded": ["a"]}) == first
    with pytest.raises(ValueError, match="new reviewed version"):
        guard.write_immutable(path, {"excluded": []})
    assert path.read_bytes() == before


def test_real_ledgers_preserve_all_old_guards_and_new_review_reservations() -> None:
    result = guard.build()
    protected = json.loads((guard.HERE / "protected-splits.json").read_bytes())
    baseline = json.loads((guard.HERE / "baseline-exposure.json").read_bytes())
    train = set(result["blocked_from_training"])
    final = set(result["excluded_from_final_holdout"])
    assert {f"plantdoc:{id_}" for id_ in protected["plantdoc_train_protected_ids"]} <= train
    assert {row["sample_id"] for row in protected["plantvillage_protected_records"]} <= train
    assert {row["sample_id"] for row in baseline["source_records"]} <= final
    imtiaz = json.loads((guard.ROOT / "infra/derived/yolo26s-imtiaz-official-export-v1/visual-review-v1/review.json").read_bytes())
    assert {"imtiaz:" + row["member"] for row in imtiaz["final_holdout_excluded_members"]} <= final
    assert {row["id"] for row in result["records"] if row["id"].startswith("sichuan-")} <= final
    assert result["training_ready"] is False
    assert result["final_holdout_established"] is False
