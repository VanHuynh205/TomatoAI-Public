"""Compare frozen DINOv3 v2 predictions on historical holdouts and the SLIF pilot."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from importlib import import_module
from pathlib import Path
from statistics import fmean
from typing import Any

from ai.compare_dinov2_field import paired_interval
from ai.dinov2_field import NEGATIVE
from ai.dinov3_field import CLASSES, metrics

v1 = import_module("infra.derived.dinov3-colab-v1.compare_results")
RUN = Path(__file__).resolve().parent
MODEL_VERSION = "dinov3-vits16-field-224-v2"
MODEL_FILES = {
    "LICENSE.md",
    "classifier.safetensors",
    "config.json",
    "leaf-classifier.json",
    "model.safetensors",
    "preprocessor_config.json",
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return data


def verify_model(
    directory: Path, frozen: dict[str, Any], version: str, lock_sha: str
) -> None:
    hashes = frozen.get("sha256")
    if (
        frozen.get("status") != "frozen_before_holdout_prediction"
        or frozen.get("model_version") != version
        or frozen.get("holdout_lock_sha256") != lock_sha
        or type(frozen.get("selected_epoch")) is not int
        or frozen["selected_epoch"] < 0
        or not isinstance(hashes, dict)
        or set(hashes) != MODEL_FILES
    ):
        raise ValueError(f"Invalid frozen model receipt for {version}: {directory}")
    for name, expected in hashes.items():
        if digest(directory / name) != expected:
            raise ValueError(f"Frozen model SHA-256 mismatch: {directory / name}")
    metadata = read_json(directory / "leaf-classifier.json")
    if (
        metadata.get("model_version") != version
        or metadata.get("class_ids") != list(CLASSES)
        or metadata.get("size") != 224
        or metadata.get("test_time_horizontal_flip") is not False
    ):
        raise ValueError(f"Unexpected inference metadata: {directory}")


def group_indices(
    rows: list[dict[str, str]], indices: list[int], *keys: str
) -> dict[tuple[str, ...], list[int]]:
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for i in indices:
        groups[tuple(rows[i][key] for key in keys)].append(i)
    return dict(sorted(groups.items()))


def historical_slice(
    rows: list[dict[str, str]], ids: list[int], labels: list[str], accepted: list[bool]
) -> dict[str, Any]:
    targets = [row["class_id"] for row in rows]

    def acceptance(members: list[int]) -> dict[str, Any]:
        chosen = [i for i in members if accepted[i]]
        return {
            "acceptance": v1.rate(len(chosen), len(members)),
            "accepted_accuracy": v1.rate(
                sum(labels[i] == targets[i] for i in chosen), len(chosen)
            ),
        }

    def summary(members: list[int]) -> dict[str, Any]:
        healthy = [i for i in members if targets[i] == v1.HEALTHY]
        negative = [i for i in members if targets[i] == NEGATIVE]
        positive = [i for i in members if targets[i] != NEGATIVE]
        return {
            "top1": metrics(
                [targets[i] for i in members], [labels[i] for i in members]
            ),
            **acceptance(members),
            "positive_acceptance": v1.rate(
                sum(accepted[i] for i in positive), len(positive)
            ),
            "healthy_accepted_as_disease": v1.rate(
                sum(accepted[i] and labels[i] in v1.DISEASES for i in healthy),
                len(healthy),
            ),
            "negative_rejection": v1.rate(
                sum(not accepted[i] for i in negative), len(negative)
            ),
            "acceptance_by_class": {
                label: acceptance(class_ids)
                for (label,), class_ids in group_indices(
                    rows, members, "class_id"
                ).items()
            },
        }

    return {
        **summary(ids),
        "by_source": {
            source: summary(members)
            for (source,), members in group_indices(rows, ids, "source").items()
        },
        "by_group": [
            {
                "source": source,
                "group": group,
                "top1_accuracy": v1.rate(
                    sum(labels[i] == targets[i] for i in members), len(members)
                ),
                **acceptance(members),
            }
            for (source, group), members in group_indices(
                rows, ids, "source", "group"
            ).items()
        ],
    }


def historical_comparison(
    rows: list[dict[str, str]], paths: dict[str, Path]
) -> dict[str, Any]:
    all_ids = list(range(len(rows)))
    field = [
        i
        for i, row in enumerate(rows)
        if row["class_id"] != NEGATIVE and row["domain"] == "field"
    ]
    field_rows = [rows[i] for i in field]
    field_predictions = {}
    models: dict[str, Any] = {}
    for model, path in paths.items():
        labels, accepted = v1.read_predictions(path, rows, model)
        field_predictions[model] = [labels[i] for i in field]
        models[model] = {
            "all_images": historical_slice(rows, all_ids, labels, accepted),
            "positive_field": historical_slice(rows, field, labels, accepted),
        }
        models[model]["positive_field"]["errors"] = v1.field_errors(
            field_rows, field_predictions[model]
        )

    candidate = models["dinov3_v2"]["positive_field"]["top1"]
    deltas = {
        baseline: {
            key: candidate[key] - models[baseline]["positive_field"]["top1"][key]
            for key in ("accuracy", "macro_f1")
        }
        for baseline in ("dinov3_v1", "p07", "dinov2")
    }
    deltas["dinov3_v1"]["macro_f1_ci95_group_bootstrap"] = paired_interval(
        field_rows,
        field_predictions["dinov3_v2"],
        field_predictions["dinov3_v1"],
        stratify_labels=True,
    )
    return {"models": models, "dinov3_v2_minus_baseline_positive_field": deltas}


def pilot_comparison(
    rows: list[dict[str, str]], sealed: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    all_ids = list(range(len(rows)))
    clear = [i for i, row in enumerate(rows) if row["quality"] == "clear"]
    groups = Counter(row["group"] for row in rows)
    blurry = [
        (row["path"], row["quality"]) for row in rows if row["quality"] != "clear"
    ]
    if (
        {row["class_id"] for row in rows} != {v1.BACTERIAL_SPOT}
        or {row["source"] for row in rows} != {sealed["source"]}
        or {row["domain"] for row in rows} != {sealed["domain"]}
        or len(clear) != sealed["clear_rows"]
        or groups != sealed["provisional_scene_groups"]
        or blurry != [("images/t0618.jpg", "blurry")]
    ):
        raise ValueError(
            "SLIF rows differ from the locked Bacterial Spot pilot definition"
        )

    models: dict[str, Any] = {}
    for model, path in paths.items():
        labels, accepted = v1.read_predictions(path, rows, model)
        models[model] = {}
        for scope, ids in (("all_images", all_ids), ("clear_images", clear)):
            by_group = {
                group: v1.rate(
                    sum(labels[i] == v1.BACTERIAL_SPOT for i in members), len(members)
                )
                for (group,), members in group_indices(rows, ids, "group").items()
            }
            models[model][scope] = {
                "bacterial_spot_recall": v1.rate(
                    sum(labels[i] == v1.BACTERIAL_SPOT for i in ids), len(ids)
                ),
                "accepted_bacterial_spot_recall": v1.rate(
                    sum(accepted[i] and labels[i] == v1.BACTERIAL_SPOT for i in ids),
                    len(ids),
                ),
                "bacterial_spot_recall_by_group": by_group,
                "bacterial_spot_mean_group_recall": fmean(
                    item["rate"] for item in by_group.values()
                ),
            }
    return {
        "scope": "Single-class Bacterial Spot pilot from a source absent from fine-tuning. Pretraining overlap is unknown. The scene groups are provisional and may share plants. The pilot measures Bacterial Spot recall only.",
        "images_by_provisional_scene_group": dict(sorted(groups.items())),
        "blurry_image": blurry[0][0],
        "models": models,
    }


def build_report(export_dir: Path, train_export_dir: Path) -> dict[str, Any]:
    lock_path = v1.RUN / "holdout-lock.json"
    lock_sha = digest(lock_path)
    lock = read_json(lock_path)
    receipt = read_json(RUN / "addition-receipt.json")
    old_receipt = read_json(v1.RUN / "input-receipt.json")
    pilot = receipt["pilot"]
    if (
        lock["status"] != "historical_benchmarks_locked_before_training"
        or lock_sha != old_receipt["holdout_lock_sha256"]
        or lock_sha != receipt["holdout_lock_sha256"]
        or digest(RUN / "train-val.csv") != receipt["train_val_sha256"]
        or receipt != read_json(train_export_dir / "addition-receipt.json")
    ):
        raise ValueError("Input receipts do not match the locked v2 experiment")
    train_frozen = train_export_dir / "frozen-model.json"
    frozen_sha = digest(train_frozen)
    frozen = read_json(train_frozen)
    if (
        frozen_sha != digest(export_dir / "frozen-model.json")
        or frozen.get("pilot_manifest_sha256") != pilot["manifest_sha256"]
        or frozen.get("train_val_sha256") != receipt["train_val_sha256"]
    ):
        raise ValueError(
            "Training and prediction exports do not describe the same frozen v2 model"
        )
    verify_model(train_export_dir / "model", frozen, MODEL_VERSION, lock_sha)
    old_frozen_path = v1.RUN / "frozen-model.json"
    old_frozen = read_json(old_frozen_path)
    verify_model(v1.RUN / "model", old_frozen, v1.MODEL_VERSION, lock_sha)

    previous = read_json(v1.RUN / "comparison.json")
    if (
        previous["holdout_lock_sha256"] != lock_sha
        or previous["dinov3_model_version"] != v1.MODEL_VERSION
        or previous["dinov3_selected_epoch"] != old_frozen["selected_epoch"]
    ):
        raise ValueError(
            "Historical baseline report does not match the frozen v1 experiment"
        )

    benchmarks = {}
    for name, (manifest, dinov2, p07) in v1.BENCHMARKS.items():
        sealed = lock["manifests"][name]
        rows = v1.locked_rows(manifest, sealed)
        paths = {
            "dinov3_v2": export_dir / f"dinov3-v2-{name}-predictions.csv",
            "dinov3_v1": v1.RUN / f"dinov3-{name}-predictions.csv",
            "dinov2": dinov2,
            "p07": p07,
        }
        hashes = {model: digest(path) for model, path in paths.items()}
        prior = previous["benchmarks"][name]
        if (
            prior["manifest_sha256"] != sealed["sha256"]
            or hashes["dinov3_v1"] != prior["predictions_sha256"]["dinov3"]
            or any(
                hashes[model] != prior["predictions_sha256"][model]
                for model in ("dinov2", "p07")
            )
        ):
            raise ValueError(f"Historical baseline SHA-256 mismatch: {name}")
        benchmarks[name] = {
            "manifest_sha256": sealed["sha256"],
            "predictions_sha256": hashes,
            **historical_comparison(rows, paths),
        }

    pilot_paths = {
        model: export_dir / f"{model.replace('_', '-')}-slif-predictions.csv"
        for model in ("dinov3_v2", "dinov3_v1")
    }
    pilot_rows = v1.locked_rows(
        RUN / "slif-pilot.csv", {**pilot, "sha256": pilot["manifest_sha256"]}
    )
    return {
        "historical_scope": "The legacy and external benchmarks have been scored before. Both were held out from DINOv3 v1 and v2 fine-tuning and checkpoint selection. Taiwan groups are approximate.",
        "holdout_lock_sha256": lock_sha,
        "frozen_models": {
            "dinov3_v2": {**frozen, "receipt_sha256": frozen_sha},
            "dinov3_v1": {**old_frozen, "receipt_sha256": digest(old_frozen_path)},
        },
        "inference": {
            "dinov3_v2": previous["inference"]["dinov3"],
            "dinov3_v1": previous["inference"]["dinov3"],
            **{model: previous["inference"][model] for model in ("dinov2", "p07")},
        },
        "acceptance_threshold": v1.THRESHOLD,
        "threshold_policy_definition": previous["threshold_policy_definition"],
        "positive_field_definition": previous["positive_field_definition"],
        "macro_f1_definition": previous["macro_f1_definition"],
        "bootstrap": previous["bootstrap"],
        "historical_benchmarks": benchmarks,
        "slif_pilot": {
            "manifest_sha256": pilot["manifest_sha256"],
            "predictions_sha256": {
                model: digest(path) for model, path in pilot_paths.items()
            },
            **pilot_comparison(pilot_rows, pilot, pilot_paths),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-dir",
        required=True,
        type=Path,
        help="Extracted Colab v2 test archive.",
    )
    parser.add_argument(
        "--train-export-dir",
        required=True,
        type=Path,
        help="Extracted Colab v2 train archive.",
    )
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(args.export_dir, args.train_export_dir)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
