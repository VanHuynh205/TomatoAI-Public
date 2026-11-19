"""Scene selection must preserve existing disease/organ annotation scope."""

import importlib.util
import sys
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).resolve().parents[2] / "docs/research/yolo26s-field-colab-2026-11-10"
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location("sichuan_wilt_preparation", DIRECTORY / "prepare_sichuan_wilt.py")
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.path.pop(0)


def row(name, labels):
    return {"filename": name, "boxes": [{"name": label, "xyxy": [1, 2, 3, 4]} for label in labels], "image_exists": True, "error": None}


def test_select_only_nonempty_publisher_wilt_leaf_scenes_without_rewriting_boxes():
    accepted = row("leaf.jpg", ["Wilt_Leaf", "Wilt_Leaf"])
    records = [row("empty.jpg", []), row("mixed.jpg", ["Wilt_Leaf", "Wilt_Stem"]), row("viral.jpg", ["Viral_Leaf"]), accepted]
    assert module.select_records(records) == [accepted]
    assert module.select_records(records)[0] is accepted


def test_reject_duplicate_or_invalid_selected_source():
    selected = row("leaf.jpg", ["Wilt_Leaf"])
    with pytest.raises(ValueError, match="Duplicate"):
        module.select_records([selected, selected])
    selected["image_exists"] = False
    with pytest.raises(ValueError, match="Invalid"):
        module.select_records([selected])
