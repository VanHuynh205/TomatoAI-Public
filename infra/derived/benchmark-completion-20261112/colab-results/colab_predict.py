"""Run frozen DINOv2 exports on GPU, without optimizer or checkpoint selection."""

import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ai.dinov2_field import predict, summarize
from ai.dinov2_improve import load_export, validation_loader
from ai.dinov3_field import CLASSES, load_manifest


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    assert torch.cuda.is_available(), "Run DINO inference on the authorized Colab GPU"
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    assert not (out / "results.json").exists(), "Completed results must not be overwritten"
    pins = json.loads((ROOT / "inputs.json").read_text())
    for name, sha in pins["files"].items():
        assert digest(ROOT / name) == sha, f"Input changed: {name}"
    data = ROOT / "data"
    datasets = {name: load_manifest(data, name + ".csv", verify_hashes=True) for name in ["historic", "validation", "taiwan"]}
    for name, n in {"historic": 2647, "validation": 1642, "taiwan": 460}.items():
        assert len(datasets[name]) == n
        assert {row["split"] for row in datasets[name]} == ({"val"} if name == "validation" else {"test"})
    started = datetime.now(timezone.utc).isoformat()
    receipt = {"started_utc": started, "python": platform.python_version(), "versions": {key: importlib.metadata.version(key) for key in ["torch", "torchvision", "transformers", "safetensors", "Pillow"]}, "gpu": torch.cuda.get_device_name(0), "device": "cuda", "size": 280, "test_time_horizontal_flip": True, "input_registration_sha256": digest(ROOT / "inputs.json"), "results": []}
    device = torch.device("cuda")
    for model_id, export in [("dinov2-v1-280", "prior"), ("dinov2-improvement-280", "candidate")]:
        model, meta = load_export(ROOT / "models" / export, device)
        for name in ["taiwan", "historic", "validation"]:
            rows = datasets[name]
            t0 = time.monotonic()
            print(f"START {model_id} {name} n={len(rows)} size=280", flush=True)
            loader = validation_loader(data, rows, 280, meta["preprocessing"], 24, 2, device)
            with torch.inference_mode():
                scores = predict(model, loader, device)
            assert scores.shape == (len(rows), len(CLASSES)) and torch.isfinite(scores).all()
            confidence, top = scores.max(1)
            closed = scores[:, :-1].argmax(1)
            predictions = [CLASSES[index] for index in top.tolist()]
            target = out / f"{model_id}-{name}-predictions.csv"
            assert not target.exists(), f"Refusing to overwrite {target}"
            fields = ["path", "sha256", "class_id", "source", "domain", "group", "prediction", "closed_set", "confidence"]
            with target.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fields, extrasaction="ignore")
                writer.writeheader()
                for row, prediction, closed_index, score in zip(rows, predictions, closed.tolist(), confidence.tolist(), strict=True):
                    writer.writerow({**row, "prediction": prediction, "closed_set": CLASSES[closed_index], "confidence": score})
            metrics = summarize(rows, predictions)
            record = {"model_id": model_id, "dataset": name, "selected_epoch": meta["selected_epoch"], "checkpoint_native_size": meta["size"], "evaluation_size": 280, "prediction_file": target.name, "prediction_sha256": digest(target), "elapsed_seconds": time.monotonic() - t0, "metrics": metrics}
            if name == "taiwan":
                with (ROOT / "parity" / f"{export}.csv").open(newline="", encoding="utf-8") as stream:
                    archived = list(csv.DictReader(stream))
                assert len(archived) == len(rows)
                assert all(all(a[k] == b[k] for k in ["path", "sha256", "class_id", "source", "domain", "group"]) for a, b in zip(archived, rows, strict=True))
                record["archived_top1_mismatches"] = sum(a["prediction"] != b for a, b in zip(archived, predictions, strict=True))
                record["archived_confidence_max_abs_diff"] = max(abs(float(a["confidence"]) - b) for a, b in zip(archived, confidence.tolist(), strict=True))
                assert record["archived_top1_mismatches"] == 0, "Investigate mismatch before filling new benchmark scores"
            receipt["results"].append(record)
            (out / f"{model_id}-{name}-metrics.json").write_text(json.dumps(record, indent=2))
            print(f"DONE {model_id} {name} field_f1={metrics['field_tomato']['macro_f1']:.8f} seconds={record['elapsed_seconds']:.1f}", flush=True)
        del model
        torch.cuda.empty_cache()
    for name, sha in pins["files"].items():
        assert digest(ROOT / name) == sha, f"Input changed during execution: {name}"
    receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), input_hashes_unchanged=True)
    (out / "results.json").write_text(json.dumps(receipt, indent=2))
    print("BENCH_COMPLETE", len(receipt["results"]), flush=True)


if __name__ == "__main__":
    main()
