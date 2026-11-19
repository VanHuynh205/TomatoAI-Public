import json
import runpy
from pathlib import Path
from typing import Any

import pytest

from ai.experiments import summarize_validation
from tests.ai.test_experiments import summary_pair as summary_pair  # noqa: PLC0414
from tests.ai.test_experiments import trained_run as trained_run  # noqa: PLC0414
from tests.ai.test_loader import sealed_images as sealed_images  # noqa: PLC0414

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "docs/progress/yolo-image/compare_validation.py"


@pytest.fixture
def config() -> dict[str, Any]:
    value: dict[str, Any] = json.loads((ROOT / "ai/configs/E09.json").read_text())
    value["run_id"] = "synthetic-yolo"
    value.pop("sample_prefix_weights")
    return value


def test_yolo_only_summary_preserves_app_default(summary_pair: list[Path]) -> None:
    summary = summarize_validation(summary_pair)
    assert summary["split"] == "val"
    assert {row["architecture"] for row in summary["runs"]} == {"yolo26s_cls"}
    assert summary["selected_by_architecture"] == {
        "yolo26s_cls": summary["selected_run"]
    }
    assert summary["decisions"]["yolo26s_cls"]["run_id"] == summary["selected_run"]
    assert summary["app_default_architecture"] == "efficientnet_b0"


def test_p02_lineage_rejects_internally_consistent_config_drift() -> None:
    compare = runpy.run_path(str(HELPER))
    expected = {
        "config": {"seed": 42, "batch": 8},
        "source_hashes": {"train.py": "locked"},
        "environment": {"device": "cuda", "torch": "pinned"},
    }
    compare["assert_lineage"](expected, **expected)
    for field, changed in (
        ("config", {"seed": 43, "batch": 8}),
        ("source_hashes", {"train.py": "changed"}),
        ("environment", {"device": "cuda", "torch": "different"}),
    ):
        with pytest.raises(ValueError, match=field):
            compare["assert_lineage"](expected | {field: changed}, **expected)


def test_comparison_keeps_source_membership_and_threshold_denominator(
    tmp_path: Path,
) -> None:
    compare = runpy.run_path(str(HELPER))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,class_id,split\n"
        "pv:a,healthy,val\nplantdoc:b,spot,val\nplantdoc:c,healthy,val\n"
        "plantdoc:sealed,spot,test\n",
        encoding="utf-8",
    )
    rows = compare["validation_rows"](manifest)
    predictions = {"pv:a": (0, 0.96), "plantdoc:b": (0, 0.61), "plantdoc:c": (0, 0.88)}
    report = compare["compare_run"](rows, predictions, ["healthy", "spot"], 0.8)
    plantdoc = report["plantdoc"]
    assert plantdoc["at_0_5"]["confusion_matrix"] == [[1, 0], [1, 0]]
    assert plantdoc["at_0_5"]["macro_f1_supported_classes"] == pytest.approx(1 / 3)
    assert plantdoc["at_0_5"]["coverage"] == 1
    assert plantdoc["at_0_5"]["accepted_accuracy"] == 0.5
    assert plantdoc["at_selected"]["coverage"] == 0.5
    assert plantdoc["at_selected"]["accepted_accuracy"] == 1
    assert report["pooled"]["at_selected"]["coverage"] == pytest.approx(2 / 3)
    no_threshold = compare["compare_run"](rows, predictions, ["healthy", "spot"], None)
    assert no_threshold["pooled"]["at_selected"]["accepted_accuracy"] is None
    assert no_threshold["pooled"]["at_selected"]["coverage"] == 0
    with pytest.raises(ValueError, match="membership"):
        compare["compare_run"](
            rows,
            {**predictions, "plantdoc:sealed": (1, 0.99)},
            ["healthy", "spot"],
            0.8,
        )


def test_saved_validation_rejects_reordered_or_substituted_ids(tmp_path: Path) -> None:
    compare = runpy.run_path(str(HELPER))
    rows = [
        {"sample_id": "pv:a", "class_id": "healthy"},
        {"sample_id": "plantdoc:b", "class_id": "spot"},
    ]
    entries = [
        {
            "sample_id": row["sample_id"],
            "target_index": i,
            "prediction_index": 0,
            "logits": [0, 0],
            "probabilities": [0.5, 0.5],
        }
        for i, row in enumerate(rows)
    ]
    path = tmp_path / "validation.json"
    payload = {"split": "val", "class_ids": ["healthy", "spot"], "sample_count": 2}
    for records in (list(reversed(entries)), [entries[0], entries[0]]):
        path.write_text(
            json.dumps(payload | {"predictions": records}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="order/identity/label"):
            compare["saved_predictions"](path, rows, payload["class_ids"])
    path.write_text(json.dumps(payload | {"predictions": entries}), encoding="utf-8")
    assert compare["saved_predictions"](path, rows, payload["class_ids"]) == {
        "pv:a": (0, 0.5),
        "plantdoc:b": (0, 0.5),
    }
