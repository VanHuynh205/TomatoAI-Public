"""Compare the frozen Colab DINOv3 export with DINOv2 v1 224 and P07."""

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from ai.compare_dinov2_field import aligned, paired_interval
from ai.dinov2_field import NEGATIVE
from ai.dinov3_field import CLASSES, metrics

RUN = Path(__file__).resolve().parent
DERIVED = RUN.parent
THRESHOLD = 0.5
BACTERIAL_SPOT = "bacterial_spot"
SEPTORIA = "septoria_leaf_spot"
HEALTHY = "healthy"
DISEASES = set(CLASSES) - {HEALTHY, NEGATIVE}
MODEL_VERSION = "dinov3-vits16-field-224-v1"

BENCHMARKS = {
    "legacy": (
        DERIVED / "dinov2-field-v2/test.csv",
        DERIVED / "dinov2-colab-v1/results/test-predictions.csv",
        DERIVED / "dinov2-colab-v1/yolo26s-cls-p07-predictions.csv",
    ),
    "external": (
        DERIVED / "dinov2-external-v1/test.csv",
        DERIVED / "dinov2-improvement-v1/results/external-prior.csv",
        RUN / "p07-external.csv",
    ),
}


def locked_rows(path: Path, lock: dict[str, Any]) -> list[dict[str, str]]:
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != lock["sha256"]:
        raise ValueError(f"Manifest SHA-256 differs from holdout lock: {path}")
    rows = list(csv.DictReader(io.StringIO(contents.decode("utf-8"), newline="")))
    if len(rows) != lock["rows"] or {row["split"] for row in rows} != {"test"}:
        raise ValueError(f"Expected {lock['rows']} sealed test rows: {path}")
    return rows


def read_predictions(
    path: Path, rows: list[dict[str, str]], model: str
) -> tuple[list[str], list[bool]]:
    labels, accepted = [], []
    allowed = CLASSES[:-1] if model == "p07" else CLASSES
    for index, row in enumerate(aligned(path, rows), start=2):
        label = row.get("prediction")
        if label is None or label not in allowed:
            raise ValueError(f"Invalid prediction label at {path}:{index}")
        try:
            confidence = float(row["confidence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid confidence at {path}:{index}") from exc
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"Confidence must be finite and in [0, 1] at {path}:{index}"
            )

        passes = label != NEGATIVE and confidence >= THRESHOLD
        if model == "dinov2":
            closed_set = row.get("closed_set")
            if closed_set not in CLASSES[:-1] or (
                label != NEGATIVE and closed_set != label
            ):
                raise ValueError(f"Invalid DINOv2 closed-set label at {path}:{index}")
        else:
            status = row.get("status")
            if status != ("ok" if passes else "uncertain"):
                raise ValueError(
                    f"Status disagrees with threshold policy at {path}:{index}"
                )
            passes = status == "ok"
        labels.append(label)
        accepted.append(passes)
    return labels, accepted


def rate(count: int, total: int) -> dict[str, int | float | None]:
    return {"count": count, "total": total, "rate": count / total if total else None}


def field_errors(rows: list[dict[str, str]], predictions: list[str]) -> dict[str, Any]:
    support = Counter(row["class_id"] for row in rows)
    errors: Counter[tuple[str, str]] = Counter()
    bacterial_groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row, prediction in zip(rows, predictions, strict=True):
        target = row["class_id"]
        errors[target, prediction] += 1
        if target == BACTERIAL_SPOT:
            bacterial_groups[row["source"], row["group"]].append(
                prediction == BACTERIAL_SPOT
            )
    return {
        "bacterial_spot_recall": rate(
            errors[BACTERIAL_SPOT, BACTERIAL_SPOT], support[BACTERIAL_SPOT]
        ),
        "bacterial_spot_groups": len(bacterial_groups),
        "bacterial_spot_mean_group_recall": (
            fmean(fmean(group) for group in bacterial_groups.values())
            if bacterial_groups
            else None
        ),
        "bacterial_spot_as_septoria": rate(
            errors[BACTERIAL_SPOT, SEPTORIA], support[BACTERIAL_SPOT]
        ),
        "septoria_as_bacterial_spot": rate(
            errors[SEPTORIA, BACTERIAL_SPOT], support[SEPTORIA]
        ),
        "healthy_as_disease": rate(
            sum(errors[HEALTHY, disease] for disease in DISEASES), support[HEALTHY]
        ),
    }


def compare_benchmark(
    rows: list[dict[str, str]], prediction_paths: dict[str, Path]
) -> dict[str, Any]:
    positive = [i for i, row in enumerate(rows) if row["class_id"] != NEGATIVE]
    negative = [i for i, row in enumerate(rows) if row["class_id"] == NEGATIVE]
    field = [i for i in positive if rows[i]["domain"] == "field"]
    field_rows = [rows[i] for i in field]
    targets = [row["class_id"] for row in rows]
    field_targets = [targets[i] for i in field]
    healthy_field = [i for i in field if targets[i] == HEALTHY]
    by_source: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_source[row["source"]].append(i)
    field_by_source: dict[str, list[int]] = defaultdict(list)
    for i in field:
        field_by_source[rows[i]["source"]].append(i)

    scores = {}
    field_labels = {}
    for model, path in prediction_paths.items():
        labels, accepted = read_predictions(path, rows, model)
        field_labels[model] = [labels[i] for i in field]
        accepted_field = [i for i in field if accepted[i]]
        scores[model] = {
            "all_images_top1": metrics(targets, labels),
            "positive_field_top1": metrics(field_targets, field_labels[model]),
            "positive_field_by_source": {
                source: metrics([targets[i] for i in ids], [labels[i] for i in ids])
                for source, ids in sorted(field_by_source.items())
            },
            "all_images_by_source": {
                source: metrics([targets[i] for i in ids], [labels[i] for i in ids])
                for source, ids in sorted(by_source.items())
            },
            "positive_field_errors": field_errors(field_rows, field_labels[model]),
            "positive_field_errors_by_source": {
                source: field_errors([rows[i] for i in ids], [labels[i] for i in ids])
                for source, ids in sorted(field_by_source.items())
            },
            "threshold_policy": {
                "positive_all_acceptance": rate(
                    sum(accepted[i] for i in positive), len(positive)
                ),
                "positive_field_acceptance": rate(len(accepted_field), len(field)),
                "accepted_positive_field_accuracy": rate(
                    sum(labels[i] == targets[i] for i in accepted_field),
                    len(accepted_field),
                ),
                "healthy_field_accepted_as_disease": rate(
                    sum(accepted[i] and labels[i] in DISEASES for i in healthy_field),
                    len(healthy_field),
                ),
                "negative_rejection": rate(
                    sum(not accepted[i] for i in negative), len(negative)
                ),
            },
        }

    candidate = scores["dinov3"]["positive_field_top1"]
    deltas = {}
    for model in ("dinov2", "p07"):
        baseline = scores[model]["positive_field_top1"]
        deltas[model] = {
            "accuracy": candidate["accuracy"] - baseline["accuracy"],
            "macro_f1": candidate["macro_f1"] - baseline["macro_f1"],
            "macro_f1_ci95_group_bootstrap": paired_interval(
                field_rows,
                field_labels["dinov3"],
                field_labels[model],
                stratify_labels=True,
            ),
        }
    return {
        "images": len(rows),
        "images_by_domain": dict(
            sorted(Counter(row["domain"] for row in rows).items())
        ),
        "positive_field_images": len(field),
        "positive_field_groups_by_source": {
            source: len({rows[i]["group"] for i in ids})
            for source, ids in sorted(field_by_source.items())
        },
        "negative_images": len(negative),
        "models": scores,
        "dinov3_minus_baseline_positive_field": deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Run from the repository root with python -m infra.derived.dinov3-colab-v1.compare_results.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=RUN,
        help="Directory with both prediction CSVs and frozen-model.json from the Colab test export.",
    )
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    lock_bytes = (RUN / "holdout-lock.json").read_bytes()
    lock_sha = hashlib.sha256(lock_bytes).hexdigest()
    receipt = json.loads((RUN / "input-receipt.json").read_text(encoding="utf-8"))
    if lock_sha != receipt["holdout_lock_sha256"]:
        raise ValueError(
            "Holdout lock SHA-256 differs from the experiment input receipt"
        )
    lock = json.loads(lock_bytes)
    if lock["status"] != "historical_benchmarks_locked_before_training":
        raise ValueError("Expected holdout lock created before training")
    frozen_path = args.export_dir / "frozen-model.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if (
        frozen.get("status") != "frozen_before_holdout_prediction"
        or frozen.get("model_version") != MODEL_VERSION
        or frozen.get("holdout_lock_sha256") != lock_sha
    ):
        raise ValueError(
            f"Frozen model receipt does not match this experiment: {frozen_path}"
        )

    benchmarks = {}
    for name, (manifest, dinov2, p07) in BENCHMARKS.items():
        sealed = lock["manifests"][name]
        rows = locked_rows(manifest, sealed)
        predictions = {
            "dinov3": args.export_dir / f"dinov3-{name}-predictions.csv",
            "dinov2": dinov2,
            "p07": p07,
        }
        benchmarks[name] = {
            "manifest_sha256": sealed["sha256"],
            "predictions_sha256": {
                model: hashlib.sha256(path.read_bytes()).hexdigest()
                for model, path in predictions.items()
            },
            **compare_benchmark(rows, predictions),
        }

    report = {
        "scope": lock["notes"],
        "holdout_lock_sha256": lock_sha,
        "dinov3_model_version": MODEL_VERSION,
        "dinov3_selected_epoch": frozen["selected_epoch"],
        "output_classes": {"dinov3": CLASSES, "dinov2": CLASSES, "p07": CLASSES[:-1]},
        "inference": {
            "dinov3": "224 px, single pass.",
            "dinov2": "v1 at 224 px, averages original and horizontally flipped logits before softmax.",
            "p07": "YOLO26s-cls P07 at 224 px, single pass.",
        },
        "acceptance_threshold": THRESHOLD,
        "threshold_policy_definition": "Top-1 tomato label and confidence >= 0.5. DINOv2 acceptance is computed from its saved scores. Top-1 metrics count every image regardless of acceptance.",
        "positive_field_definition": "domain == field and class_id != not_tomato_leaf",
        "macro_f1_definition": "Mean over classes with true support in the slice.",
        "bacterial_spot_group_recall_definition": "Mean of Bacterial Spot image recall within each source/group, with equal group weights.",
        "bootstrap": "2000 paired draws of groups within source and group label set. Taiwan groups are approximate.",
        "benchmarks": benchmarks,
    }
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
