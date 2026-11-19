"""Compare two DINOv2 exports on aligned, held-out predictions."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ai.compare_dinov2_field import aligned, paired_interval
from ai.dinov2_field import NEGATIVE
from ai.dinov3_field import load_manifest, metrics


def compare(
    rows: list[dict[str, str]],
    prior: list[dict[str, str]],
    candidate: list[dict[str, str]],
) -> dict[str, Any]:
    slices = {
        "all": list(range(len(rows))),
        "field_tomato": [
            i
            for i, row in enumerate(rows)
            if row["domain"] == "field" and row["class_id"] != NEGATIVE
        ],
        "controlled_tomato": [
            i
            for i, row in enumerate(rows)
            if row["domain"] == "controlled" and row["class_id"] != NEGATIVE
        ],
        "field_mixed_tomato": [
            i
            for i, row in enumerate(rows)
            if row["domain"] == "field_mixed" and row["class_id"] != NEGATIVE
        ],
        "not_tomato": [i for i, row in enumerate(rows) if row["class_id"] == NEGATIVE],
    }
    for source in sorted({row["source"] for row in rows}):
        slices[f"source/{source}"] = [
            i for i, row in enumerate(rows) if row["source"] == source
        ]
    result: dict[str, Any] = {}
    for name, indices in slices.items():
        if not indices:
            continue
        subset = [rows[i] for i in indices]
        truth = [row["class_id"] for row in subset]
        before = [prior[i]["prediction"] for i in indices]
        after = [candidate[i]["prediction"] for i in indices]
        prior_score = metrics(truth, before)
        candidate_score = metrics(truth, after)
        groups = len({(row["source"], row["group"]) for row in subset})
        result[name] = {
            "images": len(subset),
            "groups": groups,
            "label_support": dict(sorted(Counter(truth).items())),
            "prior": prior_score,
            "candidate": candidate_score,
            "delta_accuracy": candidate_score["accuracy"] - prior_score["accuracy"],
            "delta_macro_f1": candidate_score["macro_f1"] - prior_score["macro_f1"],
            "delta_macro_f1_ci95_group_bootstrap": (
                paired_interval(subset, after, before, stratify_labels=True)
                if groups > 1
                else None
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = load_manifest(args.data, str(args.manifest.resolve()))
    if {row["split"] for row in rows} != {"test"}:
        raise ValueError("Only test rows can enter a paired comparison")
    report = compare(rows, aligned(args.prior, rows), aligned(args.candidate, rows))
    output = {
        "test_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "bootstrap": "Paired group resampling within source and group label set, 2000 draws.",
        "slices": report,
    }
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    for name, item in report.items():
        print(
            f"{name}: n={item['images']}, macro-F1 {item['prior']['macro_f1']:.4f} -> "
            f"{item['candidate']['macro_f1']:.4f}, accuracy {item['prior']['accuracy']:.4f} -> "
            f"{item['candidate']['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
