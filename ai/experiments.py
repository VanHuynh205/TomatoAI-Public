"""Validation artifacts for completed full training runs; never evaluates TEST."""

import argparse
import json
import math
import os
import platform
from pathlib import Path
from typing import Any

import torch

from ai import train
from ai.data import ManifestDataset, _sha256
from ai.models import build_model


def _completed_run(run_dir: Path) -> tuple[Path, dict[str, Any], dict[str, Any], str]:
    """Read the same completed best/metadata contract for export and comparison."""
    train._local_path(str(run_dir), "run_dir")
    run_dir = Path(os.path.abspath(run_dir))
    if run_dir.resolve() != run_dir:
        raise ValueError("run_dir must not traverse symlinks or junctions")
    best_path, metadata_path = run_dir / "best.pth", run_dir / "metadata.json"
    for path in (best_path, metadata_path):
        if path.resolve() != path:
            raise ValueError("run artifacts must not traverse symlinks or junctions")
    if not 0 < metadata_path.stat().st_size <= 1_000_000:
        raise ValueError("metadata exceeds size limit")
    metadata_hash = _sha256(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    best = train._load_checkpoint(best_path, "best")
    best_hash = _sha256(best_path)
    config = best["config"]
    train._validate_training_config(config)
    if run_dir.is_relative_to(Path(config["dataset_dir"]).resolve()):
        raise ValueError("run_dir must not be inside dataset_dir")
    if best["identity"] != train._training_identity(config):
        raise ValueError("best data/parent identity mismatch or diagnostic run")
    try:
        train._finite_checkpoint(metadata)
        _, bad_epochs, selected = train._history_selection(metadata["history"], config)
        if (
            type(metadata["schema_version"]) is not int
            or metadata["schema_version"] != 1
            or not isinstance(metadata["environment"], dict)
            or not metadata["environment"]
            or not isinstance(metadata["source_hashes"], dict)
            or not metadata["source_hashes"]
            or metadata["config"] != config
            or metadata["identity"] != best["identity"]
            or metadata["checkpoint_hashes"]["best.pth"] != best_hash
            or metadata["epoch"] != len(metadata["history"])
            or not metadata["history"]
            or (
                metadata["epoch"] != config["epochs"]
                and bad_epochs < config["patience"]
            )
            or best["epoch"] != selected
            or metadata["best_epoch"] != selected
            or best["val"] != metadata["history"][selected - 1]["val"]
        ):
            raise ValueError("best/metadata mismatch or incomplete training run")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("invalid run metadata") from error
    return run_dir, best, metadata, metadata_hash


def export_validation_predictions(run_dir: Path) -> Path:
    """Reload best, verify full validation, then atomically publish sample predictions."""
    run_dir, best, metadata, metadata_hash = _completed_run(run_dir)
    output = run_dir / "validation-predictions.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"validation predictions already exist: {output}")
    best_path, metadata_path = run_dir / "best.pth", run_dir / "metadata.json"
    best_hash = metadata["checkpoint_hashes"]["best.pth"]
    config = best["config"]
    _, loader = train._training_loaders(config)
    dataset = loader.dataset
    if not isinstance(dataset, ManifestDataset):
        raise TypeError("validation export requires a sealed ManifestDataset")
    if run_dir.is_relative_to(dataset.source_root):
        raise ValueError("run_dir must not be inside source_root")
    model = build_model(config["architecture"], len(config["class_ids"]))
    train._check_model_state(model, best["model"])
    model.load_state_dict(best["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    predictions: list[dict[str, Any]] = []
    metrics = train._validate_epoch(
        model,
        loader,
        device=device,
        class_count=len(config["class_ids"]),
        predictions=predictions,
    )
    expected = [
        (row["sample_id"], config["class_ids"].index(row["class_id"]))
        for row in dataset._rows
    ]
    actual = [(row["sample_id"], row["target_index"]) for row in predictions]
    if actual != expected:
        raise ValueError("validation sample coverage/order/labels mismatch")
    # FP32 kernels may differ across CPU/CUDA; class decisions must still agree.
    for key, value in metrics.items():
        tolerance = 1e-5 if key == "loss" else 1e-12
        if not math.isclose(value, best["val"][key], rel_tol=0, abs_tol=tolerance):
            raise ValueError(f"reloaded validation metrics mismatch: {key}")
    if _sha256(best_path) != best_hash or _sha256(metadata_path) != metadata_hash:
        raise ValueError("run artifacts changed during validation export")
    train._atomic_save(
        output,
        {
            "schema_version": 1,
            "split": "val",
            "epoch": best["epoch"],
            "checkpoint_sha256": best_hash,
            "metadata_sha256": metadata_hash,
            "config": config,
            "identity": best["identity"],
            "class_ids": config["class_ids"],
            "sample_count": len(predictions),
            "metrics": metrics,
            "predictions": predictions,
            "training_environment": metadata["environment"],
            "training_source_hashes": metadata["source_hashes"],
            "export_environment": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "device": device.type,
                "precision": "fp32",
            },
            "export_source_hashes": {
                p.name: _sha256(p) for p in Path(__file__).parent.glob("*.py")
            },
        },
    )
    return output


def validate_experiment_set(configs: list[dict[str, Any]]) -> None:
    """Compare frozen/fine-tune pairs sharing data, classes and training protocol."""
    if not isinstance(configs, list) or len(configs) not in (2, 4):
        raise ValueError("comparison requires one or two frozen/fine_tune pairs")
    for config in configs:
        train._validate_training_config(config)
    first = configs[0]
    if len({c["run_id"] for c in configs}) != len(configs):
        raise ValueError("comparison run_ids must be unique")
    for config in configs:
        if config.get("sample_prefix_weights") != first.get("sample_prefix_weights"):
            raise ValueError("experiments must use same sample_prefix_weights")
        for field in (
            "split_hash",
            "seal_sha256",
            "dataset_dir",
            "seed",
            "class_ids",
            "precision",
            "patience",
        ):
            if config[field] != first[field]:
                raise ValueError(f"experiments must use same split/protocol: {field}")
        if (
            config["preprocessing"]["profile_version"]
            != first["preprocessing"]["profile_version"]
            or config["batch"] * config["accumulation"]
            != first["batch"] * first["accumulation"]
            or config["optimizer"]["weight_decay"] != first["optimizer"]["weight_decay"]
        ):
            raise ValueError("experiments must use same preprocessing/protocol")
        if config["optimizer"]["lr"] != (1e-3 if config["stage"] == "frozen" else 1e-4):
            raise ValueError("experiment learning rate must follow stage policy")
    for architecture in {c["architecture"] for c in configs}:
        pair = [c for c in configs if c["architecture"] == architecture]
        if len(pair) != 2 or {c["stage"] for c in pair} != {"frozen", "fine_tune"}:
            raise ValueError("each architecture requires frozen and fine_tune runs")
        for field in ("pretrained_path", "pretrained_sha256", "preprocessing"):
            if pair[0][field] != pair[1][field]:
                raise ValueError(f"backbone pair mismatch: {field}")
    for stage in ("frozen", "fine_tune"):
        if len({c["epochs"] for c in configs if c["stage"] == stage}) != 1:
            raise ValueError("same stage requires same epoch budget")


def select_threshold(scores: list[float], correct: list[bool]) -> float | None:
    """First feasible grid point maximizes coverage, breaking ties by smaller threshold."""
    if (
        not isinstance(scores, list)
        or not isinstance(correct, list)
        or not scores
        or len(scores) != len(correct)
        or any(not train._finite_number(s) or not 0 <= s <= 1 for s in scores)
        or any(type(c) is not bool for c in correct)
    ):
        raise ValueError(
            "scores/correct must be nonempty matching finite probabilities/booleans"
        )
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
        accepted = [c for s, c in zip(scores, correct, strict=True) if s >= threshold]
        if len(accepted) / len(scores) >= 0.5 and sum(accepted) / len(accepted) >= 0.9:
            return threshold
    return None


def _prediction_metrics(
    predictions: Any,
    rows: list[dict[str, str]],
    class_ids: list[str],
) -> tuple[dict[str, float], list[dict[str, Any]], list[float], list[bool]]:
    if not isinstance(predictions, list) or len(predictions) != len(rows) or not rows:
        raise ValueError("validation predictions must cover the full nonempty split")
    count = len(class_ids)
    confusion = [[0] * count for _ in class_ids]
    total_loss = 0.0
    scores: list[float] = []
    correct: list[bool] = []
    for prediction, row in zip(predictions, rows, strict=True):
        target = class_ids.index(row["class_id"])
        if (
            prediction["sample_id"] != row["sample_id"]
            or type(prediction["target_index"]) is not int
            or prediction["target_index"] != target
            or type(prediction["prediction_index"]) is not int
        ):
            raise ValueError("validation sample order/identity/label mismatch")
        logits, probabilities = prediction["logits"], prediction["probabilities"]
        if (
            not isinstance(logits, list)
            or not isinstance(probabilities, list)
            or len(logits) != count
            or len(probabilities) != count
            or any(not train._finite_number(x) for x in logits + probabilities)
        ):
            raise ValueError("invalid validation logits/probabilities")
        top = max(range(count), key=logits.__getitem__)
        shift = max(logits)
        denominator = sum(math.exp(x - shift) for x in logits)
        if (
            prediction["prediction_index"] != top
            or any(
                not 0 <= p <= 1 or abs(math.exp(x - shift) / denominator - p) > 1e-6
                for x, p in zip(logits, probabilities, strict=True)
            )
            or abs(sum(probabilities) - 1) > 1e-6
        ):
            raise ValueError("validation prediction/softmax mismatch")
        total_loss += shift + math.log(denominator) - logits[target]
        confusion[target][top] += 1
        scores.append(probabilities[top])
        correct.append(top == target)
    supports = [sum(row) for row in confusion]
    predicted = [sum(row[i] for row in confusion) for i in range(count)]
    classes: list[dict[str, Any]] = [
        {
            "class_id": label,
            "support": supports[i],
            "precision": confusion[i][i] / max(predicted[i], 1),
            "recall": confusion[i][i] / max(supports[i], 1),
            "f1": 2 * confusion[i][i] / max(supports[i] + predicted[i], 1),
        }
        for i, label in enumerate(class_ids)
    ]
    metrics = {
        "loss": total_loss / len(rows),
        "accuracy": sum(correct) / len(rows),
        "macro_f1": sum(row["f1"] for row in classes) / count,
    }
    return metrics, classes, scores, correct


def summarize_validation(run_dirs: list[Path]) -> dict[str, Any]:
    """Read completed exported runs; select checkpoints/thresholds using validation only."""
    records: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    data: dict[str, str] = {}
    for directory in run_dirs:
        directory, best, metadata, metadata_hash = _completed_run(directory)
        config = best["config"]
        path = directory / "validation-predictions.json"
        if path.resolve() != path or not 0 < path.stat().st_size <= 64_000_000:
            raise ValueError("invalid validation predictions path/size")
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = ManifestDataset(
            Path(config["dataset_dir"]), "val", lambda image: image
        )
        if tuple(config["class_ids"]) != dataset.class_ids or directory.is_relative_to(
            dataset.source_root
        ):
            raise ValueError("invalid class order or run inside source_root")
        try:
            if (
                type(payload["schema_version"]) is not int
                or payload["schema_version"] != 1
                or payload["split"] != "val"
                or payload["config"] != config
                or payload["identity"] != best["identity"]
                or payload["epoch"] != best["epoch"]
                or payload["class_ids"] != config["class_ids"]
                or payload["sample_count"] != len(dataset)
                or payload["training_source_hashes"] != metadata["source_hashes"]
                or payload["training_environment"] != metadata["environment"]
                or payload["checkpoint_sha256"]
                != metadata["checkpoint_hashes"]["best.pth"]
                or payload["metadata_sha256"] != metadata_hash
            ):
                raise ValueError("validation artifact provenance mismatch")
            metrics, per_class, scores, correct = _prediction_metrics(
                payload["predictions"], dataset._rows, config["class_ids"]
            )
            for key, value in metrics.items():
                for source in (payload["metrics"], best["val"]):
                    if not train._finite_number(source[key]) or not math.isclose(
                        value,
                        source[key],
                        rel_tol=0,
                        abs_tol=1e-5 if key == "loss" else 1e-12,
                    ):
                        raise ValueError("validation metrics mismatch")
        except (KeyError, TypeError, IndexError, OverflowError) as error:
            raise ValueError("invalid validation artifact") from error
        threshold = select_threshold(scores, correct)
        accepted = [
            c
            for s, c in zip(scores, correct, strict=True)
            if threshold is not None and s >= threshold
        ]
        records.append(
            {
                "run_id": config["run_id"],
                "run_dir": str(directory),
                "architecture": config["architecture"],
                "stage": config["stage"],
                "best_epoch": best["epoch"],
                "metrics": best["val"],
                "per_class": per_class,
                "threshold": threshold,
                "coverage": len(accepted) / len(scores),
                "accepted_accuracy": sum(accepted) / len(accepted)
                if accepted
                else None,
                "best_sha256": metadata["checkpoint_hashes"]["best.pth"],
                "validation_predictions_sha256": _sha256(path),
                "metadata_sha256": metadata_hash,
                "parent_best_sha256": best["identity"]["parent_best_sha256"],
            }
        )
        configs.append(config)
        data = {
            "dataset_dir": str(Path(config["dataset_dir"]).resolve()),
            "source_root": str(dataset.source_root),
        }
    validate_experiment_set(configs)
    selected: dict[str, str] = {}
    decisions: dict[str, Any] = {}
    records.sort(key=lambda row: (row["stage"] != "frozen", row["run_id"]))
    for architecture in sorted({row["architecture"] for row in records}):
        pair = [row for row in records if row["architecture"] == architecture]
        frozen = next(row for row in pair if row["stage"] == "frozen")
        tuned = next(row for row in pair if row["stage"] == "fine_tune")
        tuned_config = next(c for c in configs if c["run_id"] == tuned["run_id"])
        if (
            Path(tuned_config["parent_best"]).resolve()
            != Path(frozen["run_dir"]) / "best.pth"
            or tuned["parent_best_sha256"] != frozen["best_sha256"]
        ):
            raise ValueError("fine_tune must descend from compared frozen best")
        winner = max(
            pair, key=lambda row: (row["metrics"]["macro_f1"], -row["metrics"]["loss"])
        )
        selected[architecture] = winner["run_id"]
        decisions[architecture] = {
            "schema_version": 1,
            "architecture": architecture,
            "run_id": winner["run_id"],
            "best_epoch": winner["best_epoch"],
            "best_sha256": winner["best_sha256"],
            "validation_predictions_sha256": winner["validation_predictions_sha256"],
            "metadata_sha256": winner["metadata_sha256"],
            "threshold": winner["threshold"],
            "threshold_policy": {
                "grid": [0.5, 0.6, 0.7, 0.8, 0.9],
                "min_coverage": 0.5,
                "min_accepted_accuracy": 0.9,
                "comparison": ">=",
                "selection": "max coverage, then smaller threshold",
            },
            "metrics": winner["metrics"],
            "coverage": winner["coverage"],
            "accepted_accuracy": winner["accepted_accuracy"],
            "validation_quality": {
                "macro_f1_at_least_0_85": winner["metrics"]["macro_f1"] >= 0.85,
                "recall_each_at_least_0_70": all(
                    row["support"] > 0 and row["recall"] >= 0.7
                    for row in winner["per_class"]
                ),
                "threshold_found": winner["threshold"] is not None,
            },
            "selection_rule": "max val macro_f1, then min val loss; exact tie prefers frozen, then run_id",
            "compared_runs": [
                {
                    key: row[key]
                    for key in (
                        "run_id",
                        "best_sha256",
                        "validation_predictions_sha256",
                        "metadata_sha256",
                        "metrics",
                    )
                }
                for row in pair
            ],
            "identity": {
                key: configs[0][key]
                for key in ("seal_sha256", "split_hash", "seed", "class_ids")
            },
        }
    training = ManifestDataset(
        Path(configs[0]["dataset_dir"]), "train", lambda image: image
    )
    class_ids = configs[0]["class_ids"]
    counts = [
        sum(row["class_id"] == label for row in training._rows) for label in class_ids
    ]
    majority = max(range(len(class_ids)), key=counts.__getitem__)
    val_count = sum(row["support"] for row in records[0]["per_class"])
    support = records[0]["per_class"][majority]["support"]
    baseline = {
        "class_id": class_ids[majority],
        "train_support": counts[majority],
        "validation": {
            "samples": val_count,
            "accuracy": support / val_count,
            "macro_f1": 2 * support / (val_count + support) / len(class_ids),
        },
    }
    winner = max(
        records, key=lambda row: (row["metrics"]["macro_f1"], -row["metrics"]["loss"])
    )
    return {
        "schema_version": 1,
        "split": "val",
        "data": data,
        "runs": records,
        "selected_run": winner["run_id"],
        "selected_by_architecture": selected,
        "app_default_architecture": "efficientnet_b0",
        "baseline": baseline,
        "decisions": decisions,
    }


def _publish_summary(summary: dict[str, Any], output: Path) -> None:
    train._local_path(str(output), "out")
    output = Path(os.path.abspath(output))
    targets = [
        (
            Path(
                next(
                    row["run_dir"]
                    for row in summary["runs"]
                    if row["run_id"] == decision["run_id"]
                )
            )
            / "decision.json",
            decision,
        )
        for decision in summary["decisions"].values()
    ]
    targets.append((output, summary))
    if len({path for path, _ in targets}) != len(targets):
        raise ValueError("summary and decision output paths must differ")
    for path, value in targets:
        if (
            path.suffix != ".json"
            or path.resolve() != path
            or any(path.is_relative_to(Path(root)) for root in summary["data"].values())
        ):
            raise ValueError(
                "output must be local JSON outside dataset/raw and symlinks"
            )
        if path.exists() and (
            path.stat().st_size > 64_000_000
            or json.loads(path.read_text(encoding="utf-8")) != value
        ):
            raise FileExistsError(f"refusing to overwrite different artifact: {path}")
    for path, value in targets:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            train._atomic_save(path, value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    export = subcommands.add_parser("export-validation")
    export.add_argument("--run", required=True, type=Path)
    summarize = subcommands.add_parser("summarize")
    summarize.add_argument("--runs", nargs="+", required=True, type=Path)
    summarize.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "export-validation":
            output = export_validation_predictions(args.run)
            event = "validation_exported"
        else:
            summary = summarize_validation(args.runs)
            _publish_summary(summary, args.out)
            output, event = args.out, "validation_summarized"
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"event": event, "output": str(output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
