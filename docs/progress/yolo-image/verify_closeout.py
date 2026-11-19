"""Verify the archived P02 comparison before accepting the image closeout."""

import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ("E09", "E10")
COMPARISON_SHA256 = "c5fe942ed832bbc422697f67eb8917049cd2f953c8a4ea200653ed966ea89c1b"
P02_ARCHIVED_SOURCES = frozenset(
    {
        "ai/data.py",
        "ai/split.py",
        "ai/train.py",
        "backend/images.py",
        "desktop/launcher.py",
        "desktop/release.py",
        "desktop/server.py",
        "desktop/tests/browser_server.py",
        "desktop/tests/test_lifecycle.py",
        "tests/ai/test_data.py",
    }
)


def read_archived_comparison(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != COMPARISON_SHA256:
        raise ValueError("Archived comparison hash mismatch")
    comparison: dict[str, Any] = json.loads(content)
    return comparison


def verify_selection(
    summary: dict[str, Any],
    comparison: dict[str, Any],
    metrics: dict[str, dict[str, float]],
) -> str:
    # E09 comes first so an exact tie selects the frozen run.
    winner = max(
        RUNS, key=lambda run: (metrics[run]["macro_f1"], -metrics[run]["loss"])
    )
    if (
        len(summary["runs"]) != len(RUNS)
        or {row["run_id"]: row["metrics"] for row in summary["runs"]} != metrics
        or summary["selected_run"] != winner
        or summary["selected_by_architecture"] != {"yolo26s_cls": winner}
        or summary["decisions"]["yolo26s_cls"]["run_id"] != winner
        or comparison["decision"]["selected_run"] != winner
    ):
        raise ValueError("P02 winner does not match verified checkpoint metrics")
    return winner


def main() -> None:
    from ai.experiments import summarize_validation

    def read(path: Path) -> dict[str, Any]:
        value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8-sig"))
        return value

    def sha256(path: Path) -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    protocol = read(HERE / "protocol-lock.json")
    revision = read(HERE / "protocol-lock-v2.json")
    evaluation = read(HERE / "evaluation-lock.json")
    comparison = read_archived_comparison(HERE / "comparison.json")
    locked = (
        protocol["files_sha256"] | revision["files_sha256"] | evaluation["files_sha256"]
    )
    if sha256(HERE / "protocol-lock.json") != revision["base_lock_sha256"]:
        raise ValueError("P02 base lock hash mismatch")
    for entries in (locked, comparison["input_sha256"]):
        for name, digest in entries.items():
            source_root = HERE / "p02-source" if name in P02_ARCHIVED_SOURCES else ROOT
            if sha256(source_root / name) != digest:
                raise ValueError(f"Archived input hash mismatch: {name}")

    summary = read(ROOT / "docs/experiments/validation-summary-yolo-v1.json")
    fresh = summarize_validation([ROOT / "ai/runs" / run for run in RUNS])
    metrics = {row["run_id"]: row["metrics"] for row in fresh["runs"]}
    winner = verify_selection(summary, comparison, metrics)
    decision = comparison["decision"]
    if (
        summary != fresh
        or fresh["app_default_architecture"] != "efficientnet_b0"
        or decision["app_default_model_version"] != "b0-v1"
        or decision["research_gate_pass"] is not False
        or decision["eligible_for_task6_review"] is not False
        or decision["test_opened"] is not False
    ):
        raise ValueError("Archived summary or image closeout decision mismatch")
    print(
        json.dumps(
            {
                "status": "passed",
                "selected_run": winner,
                "eligible_for_task6_review": False,
                "locked_hashes": len(locked),
                "comparison_input_hashes": len(comparison["input_sha256"]),
                "comparison_sha256": COMPARISON_SHA256,
            }
        )
    )


if __name__ == "__main__":
    main()
