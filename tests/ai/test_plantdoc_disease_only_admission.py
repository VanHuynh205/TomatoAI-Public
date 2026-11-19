import importlib.util
from pathlib import Path
from typing import Any

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "docs/research/yolo26s-field-colab-2026-11-10/plantdoc_leaf_admission.py"
)
spec = importlib.util.spec_from_file_location("plantdoc_leaf_admission", SCRIPT)
assert spec and spec.loader
admission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(admission)


@pytest.fixture
def sample() -> dict[str, Any]:
    return {
        "id": "PDfixture",
        "split": "train",
        "source_split": "train",
        "source_label": "Tomato Early blight leaf",
        "source_box_count": 1,
        "group_id": "group-a",
        "boxes": [
            {
                "source_class": "Tomato Early blight leaf",
                "class_id": "early_blight",
                "xyxy": [0, 3, 100, 50],
                "xywhn": [0.5, 0.53, 1.0, 0.94],
            }
        ],
    }


def test_partial_leaf_keeps_publisher_box_and_provenance(
    sample: dict[str, Any],
) -> None:
    labels = [(51, "Tomato Early blight leaf")]
    targets, dropped = admission.disease_only_labels(
        sample, "visible_target_leaves_covered", labels
    )
    assert targets == [{**sample["boxes"][0], "source_box_id": 51}]
    assert dropped == []
    assert "source_box_id" not in sample["boxes"][0]


def test_publisher_healthy_retains_source_ids_as_negative(
    sample: dict[str, Any],
) -> None:
    sample["source_label"] = "Tomato leaf"
    sample["boxes"][0].update(source_class="Tomato leaf", class_id="healthy")
    labels = [(9, "Tomato leaf")]
    assert admission.disease_only_labels(
        sample, "publisher_healthy_negative", labels
    ) == (
        [],
        [9],
    )
    with pytest.raises(ValueError, match="conflicts with publisher labels"):
        admission.disease_only_labels(sample, "visible_target_leaves_covered", labels)


def test_disease_negative_and_source_annotations_must_agree(
    sample: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="conflicts with publisher labels"):
        admission.disease_only_labels(
            sample, "publisher_healthy_negative", [(51, "Tomato Early blight leaf")]
        )
    with pytest.raises(ValueError, match="Publisher class mismatch"):
        admission.disease_only_labels(
            sample, "visible_target_leaves_covered", [(51, "Tomato leaf late blight")]
        )
    with pytest.raises(ValueError, match="Publisher box count mismatch"):
        admission.disease_only_labels(sample, "visible_target_leaves_covered", [])


def test_retained_exclusions_do_not_reach_visual_rereview() -> None:
    decisions = {
        "a": {"visual_review": "unclear_leaf_geometry"},
        "b": {"visual_review": "unboxed_foreground_leaf"},
        "c": {"visual_review": "non_leaf_or_multi_leaf_box"},
        "d": {"visual_review": "third_party_stock_mark"},
        "e": {"visual_review": None},
    }
    rows = [{"id": name} for name in decisions]
    assert admission.disease_review_candidates(rows, decisions) == rows[:2]


def test_summary_counts_independent_groups_per_disease(sample: dict[str, Any]) -> None:
    rows = [sample, {**sample, "id": "b"}, {**sample, "id": "c", "boxes": []}]
    train = admission.disease_summary(rows)["train"]
    assert train["classes"]["early_blight"] == {"images": 2, "groups": 1, "boxes": 2}
    assert train["classes"]["yellow_leaf_curl"] == {
        "images": 0,
        "groups": 0,
        "boxes": 0,
    }
    assert "healthy" not in train["classes"]
    assert train["disease_negative_images"] == train["disease_negative_groups"] == 1
