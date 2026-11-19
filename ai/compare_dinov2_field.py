"""Score the current desktop classifier and DINOv2 on the same sealed test images."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ai.dinov2_field import NEGATIVE, summarize
from ai.dinov3_field import CLASSES, load_manifest, metrics
from ai.inference import Predictor


def baseline(
    data: Path, bundle: Path, output: Path, rows: list[dict[str, str]]
) -> None:
    torch.set_num_threads(4)
    predictor = Predictor(bundle)
    if predictor.class_ids != CLASSES[:-1]:
        raise ValueError("Baseline class order does not match the shared taxonomy")
    fields = [
        "path",
        "sha256",
        "class_id",
        "source",
        "domain",
        "group",
        "prediction",
        "confidence",
        "status",
    ]
    pending = output.with_suffix(".pending")
    with pending.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            image_path = data / row["path"]
            if hashlib.sha256(image_path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Image changed after test freeze: {image_path}")
            with Image.open(image_path) as image:
                result = predictor.predict(image)
            top = result.top_k[0]
            writer.writerow(
                {
                    **row,
                    "prediction": top.class_id,
                    "confidence": top.score,
                    "status": result.status,
                }
            )
            if index % 250 == 0:
                print(f"{predictor.model_version}: {index}/{len(rows)}", flush=True)
    pending.replace(output)


def aligned(path: Path, expected: list[dict[str, str]]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    if len(predictions) != len(expected):
        raise ValueError(f"Prediction count differs: {path}")
    for row, truth in zip(predictions, expected, strict=True):
        if any(
            row[key] != truth[key]
            for key in ("path", "sha256", "class_id", "source", "domain", "group")
        ):
            raise ValueError(
                f"Prediction does not match sealed test row: {truth['path']}"
            )
    return predictions


def paired_interval(
    rows: list[dict[str, str]],
    first: list[str],
    second: list[str],
    draws: int = 2000,
    *,
    stratify_labels: bool = False,
) -> list[float]:
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        groups[row["source"]][row["group"]].append(i)
    strata: list[list[list[int]]] = []
    for source_groups in groups.values():
        by_labels: dict[frozenset[str], list[list[int]]] = defaultdict(list)
        for members in source_groups.values():
            labels = (
                frozenset(rows[i]["class_id"] for i in members)
                if stratify_labels
                else frozenset()
            )
            by_labels[labels].append(members)
        strata.extend(by_labels.values())
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(draws):
        indices = []
        for stratum in strata:
            for index in rng.integers(len(stratum), size=len(stratum)):
                indices.extend(stratum[index])
        targets = [rows[i]["class_id"] for i in indices]
        new = metrics(targets, [first[i] for i in indices])["macro_f1"]
        old = metrics(targets, [second[i] for i in indices])["macro_f1"]
        differences.append(new - old)
    return [float(value) for value in np.quantile(differences, [0.025, 0.975])]


def compare(
    rows: list[dict[str, str]], old: list[dict[str, str]], new: list[dict[str, str]]
) -> dict[str, Any]:
    old_labels = [row["prediction"] for row in old]
    new_labels = [row["prediction"] for row in new]
    indices = [
        i
        for i, row in enumerate(rows)
        if row["domain"] == "field" and row["class_id"] != NEGATIVE
    ]
    field_rows = [rows[i] for i in indices]
    new_field = [new_labels[i] for i in indices]
    old_field = [old_labels[i] for i in indices]
    targets = [row["class_id"] for row in field_rows]
    current = summarize(rows, old_labels)
    candidate = summarize(rows, new_labels)
    negative_indices = [i for i, row in enumerate(rows) if row["class_id"] == NEGATIVE]
    return {
        "baseline": current,
        "dinov2": candidate,
        "dinov2_closed_set_field": metrics(
            targets, [new[i]["closed_set"] for i in indices]
        ),
        "field_macro_f1_delta": candidate["field_tomato"]["macro_f1"]
        - current["field_tomato"]["macro_f1"],
        "field_macro_f1_delta_ci95_group_bootstrap": paired_interval(
            field_rows, new_field, old_field
        ),
        "baseline_negative_accepted_rate_at_published_threshold": (
            sum(old[i]["status"] == "ok" for i in negative_indices)
            / len(negative_indices)
        ),
        "field_groups_by_source": {
            source: len({row["group"] for row in field_rows if row["source"] == source})
            for source in sorted({row["source"] for row in field_rows})
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--bundle", type=Path, default=Path("ai/artifacts/yolo26s-cls-v1")
    )
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--dinov2-predictions", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    rows = load_manifest(args.data, "test.csv")
    if {row["split"] for row in rows} != {"test"}:
        raise ValueError("Expected a held-out test-only manifest")
    if not args.baseline_predictions.is_file():
        baseline(args.data, args.bundle, args.baseline_predictions, rows)
    old = aligned(args.baseline_predictions, rows)
    print(
        f"Baseline field macro-F1: {summarize(rows, [row['prediction'] for row in old])['field_tomato']['macro_f1']:.4f}"
    )
    if args.dinov2_predictions:
        if args.report is None:
            parser.error("--report is required with --dinov2-predictions")
        report = compare(rows, old, aligned(args.dinov2_predictions, rows))
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"DINOv2 field macro-F1 delta: {report['field_macro_f1_delta']:+.4f}")


if __name__ == "__main__":
    main()
