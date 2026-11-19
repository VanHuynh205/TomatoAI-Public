import json
import runpy
from pathlib import Path

import pytest

VERIFY = (
    Path(__file__).resolve().parents[2] / "docs/progress/yolo-image/verify_closeout.py"
)


def test_closeout_rejects_coherently_reassigned_winner() -> None:
    verify = runpy.run_path(str(VERIFY))["verify_selection"]
    metrics = {
        "E09": {"macro_f1": 0.91, "loss": 0.1},
        "E10": {"macro_f1": 0.95, "loss": 0.2},
    }
    summary = {
        "runs": [{"run_id": run, "metrics": values} for run, values in metrics.items()],
        "selected_run": "E10",
        "selected_by_architecture": {"yolo26s_cls": "E10"},
        "decisions": {"yolo26s_cls": {"run_id": "E10"}},
    }
    comparison = {"decision": {"selected_run": "E10"}}
    assert verify(summary, comparison, metrics) == "E10"

    forged = summary | {
        "selected_run": "E09",
        "selected_by_architecture": {"yolo26s_cls": "E09"},
        "decisions": {"yolo26s_cls": {"run_id": "E09"}},
    }
    with pytest.raises(ValueError, match="winner"):
        verify(forged, {"decision": {"selected_run": "E09"}}, metrics)


def test_closeout_checkpoint_tie_uses_loss_then_frozen() -> None:
    verify = runpy.run_path(str(VERIFY))["verify_selection"]
    metrics = {
        "E09": {"macro_f1": 0.95, "loss": 0.1},
        "E10": {"macro_f1": 0.95, "loss": 0.2},
    }
    summary = {
        "runs": [{"run_id": run, "metrics": values} for run, values in metrics.items()],
        "selected_run": "E09",
        "selected_by_architecture": {"yolo26s_cls": "E09"},
        "decisions": {"yolo26s_cls": {"run_id": "E09"}},
    }
    comparison = {"decision": {"selected_run": "E09"}}
    assert verify(summary, comparison, metrics) == "E09"
    metrics["E10"]["loss"] = 0.09
    tuned_winner = summary | {
        "selected_run": "E10",
        "selected_by_architecture": {"yolo26s_cls": "E10"},
        "decisions": {"yolo26s_cls": {"run_id": "E10"}},
    }
    assert verify(tuned_winner, {"decision": {"selected_run": "E10"}}, metrics) == "E10"
    metrics["E10"]["loss"] = 0.1
    assert verify(summary, comparison, metrics) == "E09"


def test_closeout_rejects_comparison_metric_drift(tmp_path: Path) -> None:
    read = runpy.run_path(str(VERIFY))["read_archived_comparison"]
    comparison = read(VERIFY.with_name("comparison.json"))
    comparison["records"]["E10"]["plantdoc"]["at_0_5"]["macro_f1_supported_classes"] = (
        0.99
    )
    forged = tmp_path / "comparison.json"
    forged.write_text(json.dumps(comparison), encoding="utf-8")
    with pytest.raises(ValueError, match="comparison hash"):
        read(forged)
