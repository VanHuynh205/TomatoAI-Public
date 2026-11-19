"""Evaluate a sealed split with the shipped CPU predictor and the M7 TEST gate."""

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from ai import evaluation_protocol
from ai.artifacts import _METADATA_LIMIT, _json, _local, _read
from ai.benchmark import benchmark, read_benchmark_images
from ai.data import ManifestDataset, _sha256, _write_json
from ai.evaluation_protocol import _LEDGER, reserve_test_open, verify_protocol
from ai.inference import Predictor
from ai.split import SPLITS, _csv
from ai.split import _json as _dataset_json

__all__ = ["benchmark", "compute_metrics", "evaluate", "main"]

_COLUMNS = (
    "sample_id",
    "group_id",
    "component_id",
    "true_class_id",
    "predicted_class_id",
    "score",
    "status",
)


def compute_metrics(
    y_true: list[int], y_pred: list[int], accepted: list[bool], class_ids: list[str]
) -> dict[str, Any]:
    """Rows are targets, columns are raw top-1 predictions, in bundle class order."""
    k = len(class_ids)
    if not k or any(not isinstance(c, str) or not c.strip() for c in class_ids):
        raise ValueError("class_ids must contain nonempty names")
    if len(set(class_ids)) != k:
        raise ValueError("duplicate class_ids")
    if not y_true or not len(y_true) == len(y_pred) == len(accepted):
        raise ValueError(
            "nonempty, equally sized targets, predictions and accepted required"
        )
    confusion = [[0] * k for _ in class_ids]
    accepted_count = accepted_correct = 0
    for target, pred, keep in zip(y_true, y_pred, accepted, strict=True):
        if type(target) is not int or not 0 <= target < k:
            raise ValueError("invalid target index")
        if type(pred) is not int or not 0 <= pred < k:
            raise ValueError("invalid prediction index")
        if type(keep) is not bool:
            raise ValueError("accepted values must be boolean")
        confusion[target][pred] += 1
        if keep:
            accepted_count += 1
            accepted_correct += target == pred

    classes: dict[str, Any] = {}
    for index, class_id in enumerate(class_ids):
        support = sum(confusion[index])
        predicted = sum(row[index] for row in confusion)
        correct = confusion[index][index]
        classes[class_id] = {
            "support": support,
            "evaluable": support > 0,
            "precision": correct / max(1, predicted),
            "recall": correct / max(1, support),
            "f1": 2 * correct / max(1, support + predicted),
        }
    return {
        "class_ids": list(class_ids),
        "sample_count": len(y_true),
        "accuracy": sum(row[i] for i, row in enumerate(confusion)) / len(y_true),
        "macro_f1": sum(c["f1"] for c in classes.values()) / k,
        "per_class": classes,
        "complete_class_support": all(c["evaluable"] for c in classes.values()),
        "accepted_count": accepted_count,
        "coverage": accepted_count / len(y_true),
        "accepted_accuracy": accepted_correct / accepted_count
        if accepted_count
        else None,
        "accepted_error": (accepted_count - accepted_correct) / accepted_count
        if accepted_count
        else None,
        "confusion_matrix": confusion,
    }


def _csv_metrics(directory: Path, class_ids: list[str]) -> dict[str, Any]:
    rows = _csv(directory, "predictions.csv", _COLUMNS)
    indices = {name: i for i, name in enumerate(class_ids)}
    metrics = compute_metrics(
        [indices[r["true_class_id"]] for r in rows],
        [indices[r["predicted_class_id"]] for r in rows],
        [r["status"] == "ok" for r in rows],
        class_ids,
    )
    component_support = {
        c: len({r["component_id"] for r in rows if r["true_class_id"] == c})
        for c in class_ids
    }
    metrics["component_support"] = component_support
    metrics["low_support_class_ids"] = [
        c for c in class_ids if component_support[c] < 30
    ]
    return metrics


def _check_output(
    output: Path, bundle: Path, dataset: Path, protocol: Path | None = None
) -> None:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    raw = (dataset / _dataset_json(dataset, "provenance.json")["source_root"]).resolve()
    protected = [bundle, dataset, raw]
    if protocol is not None:
        protocol = _local(protocol)
        locked = verify_protocol(protocol, bundle, dataset)
        protected.extend([protocol, protocol.parent / _LEDGER])
        for release in locked["bundles"]:
            protected.append(_local(protocol.parent / release["bundle_path"]))
            protected.append(_local(protocol.parent / release["decision_path"]).parent)
    if any(output.is_relative_to(root) for root in protected):
        raise ValueError("output conflicts with data, model, run or protocol inputs")
    if any(
        path.name.casefold() in {_LEDGER, ".test-protocols", "evaluation-protocol.json"}
        or (path.parent == evaluation_protocol._SOURCE_DIR and path.match("*.py"))
        for path in (output, *output.parents)
    ):
        raise ValueError("output conflicts with a reserved protocol path")
    # Validation and timing have no protocol argument. Identify other on-disk inputs.
    if protocol is None and any(
        ((parent / "metadata.json").is_file() and (parent / "model.pth").is_file())
        or ((parent / "decision.json").is_file() and (parent / "best.pth").is_file())
        for parent in output.parents
    ):
        raise ValueError("output must be outside model bundles and training runs")


def evaluate(
    bundle_dir: Path,
    dataset_dir: Path,
    split: str,
    output_dir: Path,
    protocol: Path | None = None,
) -> dict[str, Any]:
    if split not in SPLITS:
        raise ValueError("split must be train, val or test")
    if split == "test" and protocol is None:
        raise ValueError("TEST requires a locked evaluation protocol")
    if split != "test" and protocol is not None:
        raise ValueError("protocol applies only to the test split")
    bundle, dataset, output = map(_local, (bundle_dir, dataset_dir, output_dir))
    _check_output(output, bundle, dataset, protocol)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as temp:
        stage = Path(temp)
        receipt = (
            reserve_test_open(protocol, bundle, dataset)
            if protocol is not None
            else None
        )
        meta_bytes = _read(bundle / "metadata.json", _METADATA_LIMIT)
        predictor = Predictor(bundle)
        meta = _json(meta_bytes)
        if (
            meta["dataset_version"] != dataset.name
            or meta["dataset_sha256"] != _sha256(dataset / "seal.json")
            or meta["split_hash"] != _sha256(dataset / "split.json")
        ):
            raise ValueError("bundle and dataset seal/split hashes differ")
        samples = ManifestDataset(dataset, split, predictor.predict)
        if samples.class_ids != predictor.class_ids:
            raise ValueError("bundle and dataset class order differ")
        class_ids = list(predictor.class_ids)
        assignments = _dataset_json(dataset, "split.json")["assignments"]
        components = {
            r["sample_id"]: r["component_id"]
            for r in assignments
            if r["split"] == split
        }
        with (stage / "predictions.csv").open(
            "x", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(_COLUMNS)
            for index, row in enumerate(samples._rows):
                prediction, target, sample_id = samples[index]
                top = prediction.top_k[0]
                writer.writerow(
                    (
                        sample_id,
                        row["group_id"],
                        components[sample_id],
                        class_ids[target],
                        top.class_id,
                        top.score,
                        prediction.status,
                    )
                )
        report = _csv_metrics(stage, class_ids)
        report.update(
            schema_version=1,
            split=split,
            model_version=predictor.model_version,
            artifact_kind=meta["artifact_kind"],
            metadata_sha256=hashlib.sha256(meta_bytes).hexdigest(),
            weights_sha256=meta["weights_sha256"],
            threshold=meta["threshold"],
            dataset_sha256=meta["dataset_sha256"],
            split_hash=meta["split_hash"],
            protocol_sha256=_json(_read(receipt, _METADATA_LIMIT))["protocol_sha256"]
            if receipt
            else None,
            predictions_sha256=_sha256(stage / "predictions.csv"),
        )
        if _read(bundle / "metadata.json", _METADATA_LIMIT) != meta_bytes:
            raise ValueError("bundle metadata changed during evaluation")
        if protocol is not None:
            verify_protocol(protocol, bundle, dataset)
        _write_json(stage / "metrics.json", report)
        if output.exists():
            raise FileExistsError(f"evaluation output already exists: {output}")
        stage.rename(output)
    return report


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    timing = argv[:1] == ["benchmark"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    if timing:
        parser.add_argument("--images", required=True, type=Path)
        parser.add_argument("--repeats", type=int, default=100)
        parser.add_argument("--out", type=Path)
    else:
        parser.add_argument("--dataset", required=True, type=Path)
        parser.add_argument("--split", required=True, choices=SPLITS)
        parser.add_argument("--protocol", type=Path)
        parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv[1:] if timing else argv)
    try:
        if timing:
            manifest = _local(args.images)
            manifest_bytes = _read(manifest, _METADATA_LIMIT)
            images = read_benchmark_images(manifest)
            output = _local(args.out) if args.out is not None else None
            if output:
                dataset = _local(manifest.parent / _json(manifest_bytes)["dataset_dir"])
                _check_output(output, _local(args.bundle), dataset)
            result = benchmark(args.bundle, images, args.repeats)
            if _read(manifest, _METADATA_LIMIT) != manifest_bytes:
                raise ValueError("image manifest changed during benchmark")
            result["images_manifest_sha256"] = hashlib.sha256(
                manifest_bytes
            ).hexdigest()
            if output:
                output.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=output.parent) as temp:
                    staged = Path(temp) / "benchmark.json"
                    _write_json(staged, result)
                    if os.name == "nt":
                        staged.rename(output)  # Windows rename is exclusive.
                    else:
                        os.link(staged, output)
        else:
            result = evaluate(
                args.bundle, args.dataset, args.split, args.out, args.protocol
            )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
