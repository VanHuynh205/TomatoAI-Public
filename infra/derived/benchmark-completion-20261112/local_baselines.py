"""Score the frozen eight-class exports using the existing CPU predictor."""

import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ai.compare_dinov2_field import aligned, baseline
from ai.dinov2_field import NEGATIVE, summarize
from ai.dinov3_field import load_manifest


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    started = datetime.now(timezone.utc).isoformat()
    field = ROOT / "infra/derived/dinov2-field-v2"
    external = ROOT / "infra/derived/dinov2-external-v1"
    datasets = {
        "historic": (field, load_manifest(field, "test.csv", verify_hashes=True)),
        "taiwan": (external, load_manifest(external, "test.csv", verify_hashes=True)),
        "validation": (field, [row for row in load_manifest(field, "train-val.csv") if row["split"] == "val"]),
    }
    expected = {"historic": (2647, 1730), "taiwan": (460, 139), "validation": (1642, 791)}
    for key, (folder, rows) in datasets.items():
        assert (len(rows), sum(row["domain"] == "field" and row["class_id"] != NEGATIVE for row in rows)) == expected[key]
        assert len({row["path"] for row in rows}) == len(rows)
        assert all(digest(folder / row["path"]) == row["sha256"] for row in rows)
    models = {
        "b0-e06": ("b0-adapt-v1", 17, ["historic", "taiwan", "validation"]),
        "resnet-e08": ("resnet50-adapt-v1", 19, ["taiwan", "validation"]),
        "yolo-p08": ("yolo26s-cls-p08-v1", None, ["historic", "taiwan", "validation"]),
    }
    inputs = [Path(__file__).resolve(), field / "test.csv", field / "train-val.csv", external / "test.csv"]
    inputs += list((ROOT / "ai").glob("*.py"))
    for bundle_name, _, _ in models.values():
        bundle = ROOT / "ai/artifacts" / bundle_name
        metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
        assert digest(bundle / "model.pth") == metadata["weights_sha256"]
        inputs.extend([bundle / "model.pth", bundle / "metadata.json"])
    pins = {str(path.relative_to(ROOT)): digest(path) for path in inputs}
    registration = {"started_utc": started, "purpose": "Retrospective inference only; no training or model selection", "inputs": pins, "expected": expected}
    registration_path = OUT / "local-registration.json"
    assert not registration_path.exists(), "Use a fresh output directory for a new execution"
    registration_path.write_text(json.dumps(registration, indent=2), encoding="utf-8")
    results = []
    for model_id, (bundle_name, epoch, names) in models.items():
        bundle = ROOT / "ai/artifacts" / bundle_name
        for name in names:
            folder, rows = datasets[name]
            output = OUT / f"{model_id}-{name}-predictions.csv"
            assert not output.exists(), f"Refusing to overwrite predictions: {output}"
            t0 = time.monotonic()
            print(f"START {model_id} {name} n={len(rows)}", flush=True)
            baseline(folder, bundle, output, rows)
            predictions = aligned(output, rows)
            report = summarize(rows, [row["prediction"] for row in predictions])
            for section in [report["all"], report["field_tomato"]]:
                assert sum(sum(line) for line in section["confusion"]) == section["n"]
            record = {"model_id": model_id, "bundle": bundle_name, "epoch": epoch, "dataset": name, "prediction_file": output.name, "prediction_sha256": digest(output), "elapsed_seconds": time.monotonic() - t0, "metrics": report}
            results.append(record)
            (OUT / f"{model_id}-{name}-metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"DONE {model_id} {name} n={report['all']['n']} field_f1={report['field_tomato']['macro_f1']:.8f} elapsed={record['elapsed_seconds']:.1f}s", flush=True)
    assert {str(path.relative_to(ROOT)): digest(path) for path in inputs} == pins
    receipt = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "platform": platform.platform(), "device": "cpu", "versions": {key: importlib.metadata.version(key) for key in ["torch", "torchvision", "ultralytics", "Pillow"]}, "input_hashes_unchanged": True, "results": results}
    (OUT / "local-results.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"COMPLETE {len(results)} datasets", flush=True)


if __name__ == "__main__":
    main()
