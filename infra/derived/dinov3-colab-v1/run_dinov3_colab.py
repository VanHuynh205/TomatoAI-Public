"""Train and score the sealed DINOv3 experiment in separate Colab phases."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

CONTENT = Path("/content")
ROOT = CONTENT / "tomato-dinov3-colab-v1"
RUN = ROOT / "run"
MODEL = RUN / "model"
INPUT = CONTENT / "dinov3-colab-input.zip"
HOLDOUT = CONTENT / "dinov3-colab-holdout.zip"
TRAIN_RESULT = CONTENT / "dinov3-colab-train-results.zip"
TEST_RESULT = CONTENT / "dinov3-colab-test-results.zip"
INPUT_SHA = "df6292e93d7d14a07b8e8946c0c4fe36959200d35c5b2d77685b075f6d2b64f9"
HOLDOUT_SHA = "48f2da68ac520097c21ab4064fc9700af4b19ab974c331138fc6c8ccf17254dd"
VERSION = "dinov3-vits16-field-224-v1"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def verify(path, expected):
    if digest(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def unzip(path, sha, destination):
    verify(path, sha)
    with ZipFile(path) as bundle:
        bundle.extractall(destination)


def zip_result(path, files):
    with ZipFile(path, "w", ZIP_STORED) as bundle:
        for file, entry in files:
            bundle.write(file, entry)
    print(f"RESULT_SHA256 {path.name} {digest(path)} {path.stat().st_size}", flush=True)


def check_images(root, rows):
    for row in rows:
        verify(root / row["path"], row["sha256"])


def train_phase(torch, field):
    inputs = read_json(ROOT / "input-receipt.json")
    data = ROOT / "data"
    pretrained = ROOT / "backbone"
    verify(data / "train-val.csv", inputs["train_val_sha256"])
    verify(ROOT / "holdout-lock.json", inputs["holdout_lock_sha256"])
    for name, sha in inputs["gated_files_sha256"].items():
        verify(pretrained / name, sha)
    rows = field.load_manifest(data, "train-val.csv")
    field.validate_splits(rows)
    if len(rows) != sum(inputs["split_count"].values()):
        raise ValueError("Training manifest count changed")
    check_images(data, rows)
    print(f"Verified {len(rows)} train/val images on {torch.cuda.get_device_name()}", flush=True)
    field.train(argparse.Namespace(data=data, output=RUN, backbone=str(pretrained), **inputs["parameters"]))
    meta_path = MODEL / "leaf-classifier.json"
    meta = read_json(meta_path)
    processor = read_json(MODEL / "preprocessor_config.json")
    meta.update({
        "backbone": inputs["backbone"], "revision": inputs["revision"], "model_version": VERSION,
        "preprocessing": {"geometry": "fit_pad", "interpolation": "bicubic",
                          "mean": processor["image_mean"], "std": processor["image_std"]},
        "test_time_horizontal_flip": False,
    })
    write_json(meta_path, meta)
    shutil.copyfile(pretrained / "LICENSE.md", MODEL / "LICENSE.md")
    history = read_json(RUN / "history.json")
    if len(history) != inputs["parameters"]["epochs"]:
        raise ValueError("Training did not complete all configured epochs")
    files = sorted(MODEL.iterdir())
    frozen = {
        "status": "frozen_before_holdout_prediction", "model_version": VERSION,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "selected_epoch": max(history, key=lambda epoch: epoch["field"]["macro_f1"])["epoch"],
        "sha256": {p.name: digest(p) for p in files}, "holdout_lock_sha256": inputs["holdout_lock_sha256"],
    }
    frozen_path = RUN / "frozen-model.json"
    write_json(frozen_path, frozen)
    zip_result(TRAIN_RESULT, [
        *((p, f"model/{p.name}") for p in files), (frozen_path, frozen_path.name),
        (RUN / "history.json", "history.json"), (ROOT / "input-receipt.json", "input-receipt.json"),
        (ROOT / "holdout-lock.json", "holdout-lock.json"),
    ])


def test_phase(torch, field, transformers):
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader

    frozen = read_json(RUN / "frozen-model.json")
    if frozen["status"] != "frozen_before_holdout_prediction" or frozen["model_version"] != VERSION:
        raise ValueError("A frozen training export is required")
    for name, sha in frozen["sha256"].items():
        verify(MODEL / name, sha)
    test = ROOT / "holdout"
    unzip(HOLDOUT, HOLDOUT_SHA, test)
    verify(test / "holdout-lock.json", frozen["holdout_lock_sha256"])
    lock = read_json(test / "holdout-lock.json")
    meta = read_json(MODEL / "leaf-classifier.json")
    device = torch.device("cuda")
    backbone, loading = transformers.AutoModel.from_pretrained(
        MODEL, local_files_only=True, use_safetensors=True, trust_remote_code=False,
        attn_implementation="sdpa", output_loading_info=True,
    )
    if any(loading.values()):
        raise ValueError("Backbone did not reload exactly")
    model = field.LeafClassifier(backbone).to(device)
    model.head.load_state_dict(load_file(MODEL / "classifier.safetensors"))
    prep = meta["preprocessing"]
    transform = field.leaf_transform(meta["size"], prep["mean"], prep["std"], False)
    predictions = []
    for name, expected in lock["manifests"].items():
        root = test / name
        verify(root / "test.csv", expected["sha256"])
        rows = field.load_manifest(root, "test.csv")
        if len(rows) != expected["rows"] or {row["split"] for row in rows} != {"test"}:
            raise ValueError("Expected sealed test-only manifest")
        check_images(root, rows)
        loader = DataLoader(field.LeafImages(root, rows, transform), batch_size=32, num_workers=2)
        labels, scores = field.predict(model, loader, device)
        path = RUN / f"dinov3-{name}-predictions.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, (
                "path", "sha256", "class_id", "source", "domain", "group", "prediction", "confidence", "status",
            ), extrasaction="ignore")
            writer.writeheader()
            for row, label, score in zip(rows, labels, scores, strict=True):
                if not 0 <= score <= 1:
                    raise ValueError("Non-finite prediction")
                status = "ok" if label != "not_tomato_leaf" and score >= 0.5 else "uncertain"
                writer.writerow({**row, "prediction": label, "confidence": score, "status": status})
        predictions.append((path, path.name))
        print(f"Scored {len(rows)} images on {name}", flush=True)
    zip_result(TEST_RESULT, [*predictions, (RUN / "frozen-model.json", "frozen-model.json")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "test"))
    phase = parser.parse_args().phase
    if not CONTENT.is_dir():
        raise RuntimeError("Use this runner on Colab")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if phase == "train":
        unzip(INPUT, INPUT_SHA, ROOT)
    sys.path.insert(0, str(ROOT))
    import torch
    import transformers
    from ai import dinov3_field

    if not torch.cuda.is_available() or transformers.__version__ != "4.57.6":
        raise RuntimeError("Use a GPU runtime with Transformers 4.57.6")
    if phase == "train":
        train_phase(torch, dinov3_field)
    else:
        test_phase(torch, dinov3_field, transformers)


if __name__ == "__main__":
    main()
