"""The runtime smoke fixture must not borrow final-test or healthy targets."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "docs/progress/yolo26s-field-colab-2026-11-10"
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location("runtime_smoke_preparation", DIRECTORY / "prepare_runtime_smoke.py")
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.path.pop(0)


def inputs():
    return json.loads(module.MANIFEST.read_bytes()), json.loads((DIRECTORY / "protected-splits.json").read_bytes())


def test_fixture_has_real_disease_labels_and_disjoint_groups():
    manifest, protection = inputs()
    rows = module.select_records(manifest, protection)
    assert len(rows) == 15
    assert len({row["group_id"] for row in rows}) == 15
    assert {box["class_id"] for row in rows for box in row["boxes"]} == set(module.CLASSES)
    assert all(not row["final_holdout_eligible"] for row in rows)


def test_rejects_test_record_even_with_accepted_review_flag():
    manifest, protection = inputs()
    first = module.select_records(manifest, protection)[0]
    protection["plantdoc_test_protected_ids"].append(first["id"])
    with pytest.raises(ValueError, match="Protected or final-holdout"):
        module.select_records(manifest, protection)


def test_rejects_protected_development_record_from_train():
    manifest, protection = inputs()
    first = module.select_records(manifest, protection)[0]
    protection["plantdoc_train_protected_ids"].append(first["id"])
    with pytest.raises(ValueError, match="Protected development"):
        module.select_records(manifest, protection)
