"""A filename suspicion blocks fitting without being relabelled baseline exposure."""

from copy import deepcopy
from importlib import import_module

import pytest

guard = import_module("docs.progress.yolo26s-field-colab-2026-11-10.candidate_split_guard")


def fixture():
    master = {
        "records": [
            {"id": id_, "sha256": digest * 64, "keys": [["byte", digest * 64]]}
            for id_, digest in (("bryan:v63:train/suspect.jpg", "a"), ("related", "b"),
                               ("baseline_train", "c"), ("old_test", "d"), ("fresh", "e"))
        ],
        "explicit_relations": [("bryan:v63:train/suspect.jpg", "related")],
        "blocked_from_training": ["old_test"],
        "excluded_from_final_holdout": ["baseline_train"],
    }
    audit = {
        "exports": {"v63": {"records": [{"member": "train/suspect.jpg", "sha256": "a" * 64}]}},
        "overlap_with_existing_data": {"v63": {"protected_plantdoc_filename_candidates": ["train/suspect.jpg"]}},
    }
    return master, audit


def test_filename_quarantine_propagates_without_inventing_holdout_exposure():
    master, audit = fixture()
    before = deepcopy(master)
    quarantine = guard.training_quarantine(master, audit)
    assert quarantine["quarantined_from_training"] == ["bryan:v63:train/suspect.jpg", "related"]
    assert master == before
    with pytest.raises(ValueError, match="protected or unresolved"):
        guard.validate_records(master, quarantine, [{"id": "related", "sha256": "b" * 64, "split": "train"}])
    assert guard.validate_records(master, quarantine, [{"id": "baseline_train", "sha256": "c" * 64, "split": "train"}])["train"] == 1


@pytest.mark.parametrize("id_,digest,split,message", [
    ("missing", "f", "train", "Unknown"),
    ("fresh", "f", "train", "hash mismatch"),
    ("fresh", "e", "test", "invalid split"),
    ("old_test", "d", "train", "protected"),
    ("baseline_train", "c", "final_holdout", "known exposure"),
])
def test_invalid_candidate_fails_closed(id_, digest, split, message):
    master, audit = fixture()
    with pytest.raises(ValueError, match=message):
        guard.validate_records(master, guard.training_quarantine(master, audit), [{"id": id_, "sha256": digest * 64, "split": split}])


@pytest.mark.parametrize("other_split", ["development", "final_holdout"])
def test_candidate_relatives_cannot_cross_splits(other_split):
    master, _ = fixture()
    with pytest.raises(ValueError, match="related group|candidate development relatives"):
        guard.validate_records(master, {"quarantined_from_training": []}, [
            {"id": "bryan:v63:train/suspect.jpg", "sha256": "a" * 64, "split": "train"},
            {"id": "related", "sha256": "b" * 64, "split": other_split},
        ])


def test_real_bryan_candidates_and_exact_byte_protections():
    inputs = guard.load_inputs()
    master, audit = inputs[guard.MASTER], inputs[guard.BRYAN]
    before = deepcopy(master["excluded_from_final_holdout"])
    quarantine = guard.training_quarantine(master, audit)
    seeds = set(quarantine["seed_ids"])
    assert sum(id_.startswith("bryan:v63:") for id_ in seeds) == 254
    assert sum(id_.startswith("bryan:fork_v1:") for id_ in seeds) == 123
    assert len(set(master["blocked_from_training"]) | set(quarantine["quarantined_from_training"])) == 4014
    exact = audit["overlap_with_existing_data"]["fork_v1"]["protected_exact_byte_matches"]
    assert len(exact) == 7
    assert {"bryan:fork_v1:" + member for member in exact} <= set(master["blocked_from_training"])
    assert master["excluded_from_final_holdout"] == before
    assert not seeds & set(before)
    row = next(row for row in audit["exports"]["v63"]["records"] if row["split"] == "train" and "bryan:v63:" + row["member"] in seeds)
    with pytest.raises(ValueError, match="protected or unresolved"):
        guard.validate_records(master, quarantine, [{"id": "bryan:v63:" + row["member"], "sha256": row["sha256"], "split": "train"}])
