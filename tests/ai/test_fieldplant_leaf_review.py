"""Only previously exposed, nonconflicting source records enter this review."""

import importlib.util
import sys
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).resolve().parents[2] / "docs/research/yolo26s-field-colab-2026-11-10"
sys.path.insert(0, str(DIRECTORY))
SPEC = importlib.util.spec_from_file_location("fieldplant_leaf_review", DIRECTORY / "fieldplant_leaf_review.py")
assert SPEC is not None and SPEC.loader is not None
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def test_review_does_not_expose_fresh_candidates_or_conflicting_records():
    manifest = {"records": [{"id": "seen", "image": {"id": 2}}, {"id": "conflict", "image": {"id": 1}}, {"id": "fresh", "image": {"id": 3}}]}
    ledger = {"groups": {"a": {"records": {"seen": "prior_exposure_group"}}, "b": {"records": {"conflict": "conflict_quarantine"}}, "c": {"records": {"fresh": "disease_visual_review_candidate"}}}}
    assert review.candidates(manifest, ledger) == [manifest["records"][0]]
    manifest["records"].append(manifest["records"][0])
    with pytest.raises(ValueError, match="missing or repeated"):
        review.candidates(manifest, ledger)


def test_sealed_review_cannot_be_overwritten(tmp_path):
    path = tmp_path / "review.json"
    original = {"training_admitted": False}
    expected = review.immutable_json(path, original)
    assert review.immutable_json(path, original) == expected
    with pytest.raises(ValueError, match="Evidence changed"):
        review.immutable_json(path, {"training_admitted": True})


def test_fresh_batch_closes_related_exposure_and_avoids_conflict_links():
    rows = [{"id": key, "provisional_group": group, "annotations": [{"source_label": label}]} for key, group, label in [
        ("a", "same", "Tomato Brown Spots"), ("b", "same", "Tomato Brown Spots"),
        ("c", "c", "Tomato blight leaf"), ("d", "d", "Tomato blight leaf"),
        ("e", "e", "Tomato blight leaf"), ("f", "f", "Tomato leaf yellow virus")]]
    ledger = {"groups": {"all": {"records": {key: "conflict_quarantine" if key == "e" else "disease_visual_review_candidate" for key in "abcdef"}}}}
    near = {"candidate_groups": [
        {"fieldplant_records": ["b", "c"], "exclusion_witnesses": {}},
        {"fieldplant_records": ["d", "e"], "exclusion_witnesses": {"conflict": "e"}}]}
    selected, reserved = review.fresh_batch({"records": rows}, ledger, near, per_label=1)
    assert len(selected) == 1
    assert selected[0]["id"] in {"a", "b"}
    assert reserved == ["a", "b", "c"]
    assert "f" not in reserved
