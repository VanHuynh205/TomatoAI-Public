"""Run the registered candidates on Colab, retaining identities and score evidence."""

import csv
import hashlib
import importlib.metadata
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ai.dinov2_field import summarize
from ai.dinov3_field import CLASSES, LeafImages, load_manifest, predict
from ai.dinov3_inference import DinoV3Predictor
from ai.models import build_model
from ai.preprocessing import build_transform
from ai.train import _check_model_state, _load_checkpoint


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    assert torch.cuda.is_available(), "This run requires the authorized Colab GPU"
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    registration = json.loads((ROOT / "registration.json").read_text())
    for name, expected in registration["files"].items():
        assert digest(ROOT / name) == expected, name
    output = ROOT / "results"
    output.mkdir(exist_ok=False)
    datasets = {name: load_manifest(ROOT / "data", name + ".csv", verify_hashes=True) for name in registration["datasets"]}
    assert {name: len(rows) for name, rows in datasets.items()} == registration["datasets"]
    receipt = {"started_utc": datetime.now(timezone.utc).isoformat(), "gpu": torch.cuda.get_device_name(0), "device": "cuda", "registration_sha256": digest(ROOT / "registration.json"), "versions": {name: importlib.metadata.version(name) for name in ["torch", "torchvision", "transformers", "ultralytics"]}, "results": []}
    for item in registration["models"]:
        source = ROOT / item["path"]
        if item["kind"] == "dinov3":
            predictor = DinoV3Predictor(source)
            model, transform = predictor._model, predictor._transform
            del predictor
        else:
            checkpoint = _load_checkpoint(source if item["kind"] == "run" else ROOT / "models/E10/best.pth", "best")
            config = checkpoint["config"]
            assert config["class_ids"] == list(CLASSES[:-1])
            state = checkpoint["model"] if item["kind"] == "run" else torch.load(source, map_location="cpu", weights_only=True)
            model = build_model(config["architecture"], len(CLASSES) - 1)
            _check_model_state(model, state)
            model.load_state_dict(state, strict=True)
            transform = build_transform(config["preprocessing"]["spec"], training=False)
            del state, checkpoint
        model.to("cuda").eval().requires_grad_(False)
        for name in item["datasets"]:
            rows = datasets[name]
            started = time.monotonic()
            print("START", item["id"], name, len(rows), flush=True)
            loader = DataLoader(LeafImages(ROOT / "data", rows, transform), batch_size=24, shuffle=False, num_workers=2, pin_memory=True)
            with torch.inference_mode():
                if item["kind"] == "dinov3":
                    labels, confidence = predict(model, loader, torch.device("cuda"))
                else:
                    labels, confidence = [], []
                    for pixels, _ in loader:
                        logits = model(pixels.to("cuda", non_blocking=True))
                        assert logits.shape == (len(pixels), len(CLASSES) - 1) and torch.isfinite(logits).all()
                        scores, indices = logits.float().softmax(1).max(1)
                        labels.extend(CLASSES[i] for i in indices.tolist())
                        confidence.extend(scores.tolist())
            assert len(labels) == len(confidence) == len(rows)
            records = [{**row, "prediction": label, "confidence": score} for row, label, score in zip(rows, labels, confidence, strict=True)]
            target = output / f"{item['id']}-{name}-predictions.csv"
            with target.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, ["path", "sha256", "class_id", "source", "domain", "group", "prediction", "confidence"], extrasaction="ignore")
                writer.writeheader()
                writer.writerows(records)
            report = summarize(rows, labels)
            result = {"model_id": item["id"], "label": item["label"], "family": item["family"], "epoch": item["epoch"], "dataset": name, "prediction_file": target.name, "prediction_sha256": digest(target), "metrics": report, "seconds": time.monotonic() - started}
            parity = ROOT / f"parity/{item['id']}.csv"
            if name == "taiwan" and parity.exists():
                with parity.open(newline="", encoding="utf-8") as stream:
                    archived = list(csv.DictReader(stream))
                identity = ["path", "sha256", "class_id", "source", "domain", "group"]
                assert len(archived) == len(records)
                assert all(all(a[k] == b[k] for k in identity) for a, b in zip(archived, records, strict=True))
                result["parity_mismatches"] = sum(a["prediction"] != b["prediction"] for a, b in zip(archived, records, strict=True))
                assert result["parity_mismatches"] == 0, "Investigate CPU/GPU parity before selecting models"
            (output / f"{item['id']}-{name}-metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
            receipt["results"].append(result)
            print("DONE", item["id"], name, "f1", report["all"]["macro_f1"], "field_f1", report["field_tomato"]["macro_f1"], flush=True)
        del model
        torch.cuda.empty_cache()
    for name, expected in registration["files"].items():
        assert digest(ROOT / name) == expected, name
    receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), input_hashes_unchanged=True)
    (output / "results.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    assert len(receipt["results"]) == sum(len(item["datasets"]) for item in registration["models"])
    print("SELECTION_EVALUATION_COMPLETE", len(receipt["results"]), flush=True)


if __name__ == "__main__":
    main()
