"""Verify the curated train-only Sichuan leaf admission against publisher evidence."""

import json
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def subset(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(ROOT / "docs/research/yolo26s-field-colab-2026-11-10"))
    return import_module("sichuan_leaf_subset")


@pytest.fixture
def manifests(subset: ModuleType) -> tuple[dict[str, Any], dict[str, Any]]:
    admission = json.loads((subset.OUTPUT / "admission.json").read_bytes())
    candidates = json.loads((subset.OUTPUT / "candidates.json").read_bytes())
    return admission, candidates


def test_verifies_curated_train_admission(
    subset: ModuleType, manifests: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    admission, candidates = manifests
    subset.verify_admission(admission, candidates)
    (record,) = admission["records"]
    assert record["id"] == "sichuan-Gray_0617_013"
    assert record["boxes"][0]["xyxy"] == [843, 390, 3662, 2832]
    assert record["exif_transposed"] == {
        "width": 3024,
        "height": 4032,
        "boxes": [{"name": "GrayMold_Leaf", "xyxy": [192, 843, 2634, 3662], "difficult": "0"}],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "sichuan-Gray_0509_018"),
        ("group", "different-greenhouse"),
        ("role", "development"),
        ("final_holdout_eligible", True),
        ("image_sha256", "0" * 64),
        ("annotation_sha256", "0" * 64),
        ("local_image", "../substituted.jpg"),
        ("width", 3024),
        ("exif_orientation", 1),
        ("boxes", [{"name": "GrayMold_Fruit", "xyxy": [843, 390, 3662, 2832], "difficult": "0"}]),
        ("boxes", [{"name": "GrayMold_Leaf", "xyxy": [844, 390, 3662, 2832], "difficult": "0"}]),
        ("exif_transposed", {"width": 4032, "height": 3024, "boxes": []}),
    ],
)
def test_rejects_changed_admission(
    subset: ModuleType, manifests: tuple[dict[str, Any], dict[str, Any]], field: str, value: object
) -> None:
    admission, candidates = manifests
    admission["records"][0][field] = value
    with pytest.raises(ValueError, match="Sichuan admission"):
        subset.verify_admission(admission, candidates)


def test_rejects_holdout_in_nonadmitted_greenhouse_member(
    subset: ModuleType, manifests: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    admission, candidates = manifests
    candidates["records"][0]["final_holdout_eligible"] = True
    with pytest.raises(ValueError, match="Sichuan source group"):
        subset.verify_admission(admission, candidates)


def test_prior_review_marker_only_applies_to_holdout(
    subset: ModuleType, manifests: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    admission, candidates = manifests
    (record,) = admission["records"]
    source = next(row for row in candidates["records"] if row["id"] == record["id"])
    source["eligibility"] = "prior_audit_quarantine"
    subset.verify_admission(admission, candidates)


def test_disease_only_admission_preserves_source_boxes_and_train_only_policy(
    subset: ModuleType, manifests: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    _, candidates = manifests
    admission = subset.build_disease_only_admission(candidates)
    subset.verify_admission(admission, candidates, disease_only=True)
    assert {r["id"] for r in admission["records"]} == {
        "sichuan-Gray_0617_011", "sichuan-Gray_0617_012", "sichuan-Gray_0617_013",
    }
    record = next(r for r in admission["records"] if r["id"].endswith("011"))
    assert record["boxes"] == [{"name": "GrayMold_Leaf", "xyxy": [2365, 1, 3317, 1255], "difficult": "0"}]
    assert record["exif_transposed"]["boxes"][0]["xyxy"] == [1769, 2365, 3023, 3317]
    assert all(r["role"] == "train_only" and not r["final_holdout_eligible"] for r in admission["records"])
    with pytest.raises(ValueError, match="reviewed candidate"):
        subset.verify_admission(admission, candidates)
    record["boxes"] = [{"name": "GrayMold_Fruit", "xyxy": [2365, 1, 3317, 1255], "difficult": "0"}]
    with pytest.raises(ValueError, match="publisher data"):
        subset.verify_admission(admission, candidates, disease_only=True)


def test_v3_preserves_old_records_and_adds_naturally_truncated_leaf(
    subset: ModuleType, manifests: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    _, candidates = manifests
    old = subset.build_disease_only_admission(candidates)
    revised = subset.build_disease_only_admission(candidates, version=3)
    subset.verify_admission(revised, candidates, disease_only=True, version=3)
    assert revised["records"][1:] == old["records"]
    added = revised["records"][0]
    assert added["id"] == "sichuan-Gray_0617_010"
    assert added["boxes"] == [{"name": "GrayMold_Leaf", "xyxy": [2982, 910, 4032, 2129], "difficult": "0"}]
    assert added["exif_transposed"]["boxes"][0]["xyxy"] == [895, 2982, 2114, 4032]
    assert added["role"] == "train_only" and added["final_holdout_eligible"] is False
    with pytest.raises(ValueError, match="reviewed candidate"):
        subset.verify_admission(revised, candidates, disease_only=True)
