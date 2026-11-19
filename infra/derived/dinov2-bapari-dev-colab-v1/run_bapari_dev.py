import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torchvision
import transformers
from PIL import __version__ as pillow_version

from ai.dinov2_field import predict
from ai.dinov2_improve import digest, load_export, validation_loader
from ai.dinov3_field import CLASSES

root = Path(__file__).resolve().parent
manifest = root / "dev.csv"
model_zip = Path(sys.argv[1]).resolve()
manifest_sha = "97d94488113d38d120579793cf2f078c266d251f79d5ac3edf265f759c3600ca"
model_zip_sha = "554b914a539d36cf846d2cc7d66f1b2f5afd8cf8a9b6d5ecd66f378ef9572ca9"
assert digest(manifest) == manifest_sha
assert digest(model_zip) == model_zip_sha

frozen = json.loads((root / "frozen-models.json").read_text(encoding="utf-8"))
prior = frozen["weights"]["prior"]
with ZipFile(model_zip) as archive:
    for name in prior:
        archive.extract("model/" + name, root)
model_dir = root / "model"
for name, sha in prior.items():
    assert digest(model_dir / name) == sha, name

with manifest.open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == len({row["path"] for row in rows}) == len({row["sha256"] for row in rows}) == 436
assert Counter(row["class_id"] for row in rows) == {
    "bacterial_spot": 73, "septoria_leaf_spot": 89, "healthy": 274,
}
for row in rows:
    assert row["split"] == "dev" and row["reserved_role"] == "dev_only"
    image_path = (root / row["path"]).resolve()
    assert image_path.is_relative_to(root / "raw-bapari")
    assert digest(image_path) == row["sha256"], row["path"]
assert torch.cuda.is_available(), "Select a Colab GPU runtime"
assert transformers.__version__ == "4.57.6", transformers.__version__
device = torch.device("cuda")
print("Validated 436 dev images, 5 export files, GPU:", torch.cuda.get_device_name(device), flush=True)
model, meta = load_export(model_dir, device)
assert meta["size"] == 224 and meta["selected_epoch"] == 6

outputs = {}
batch = 24
for size in (224, 280):
    loader = validation_loader(root, rows, size, meta["preprocessing"], batch, 0, device)
    scores = predict(model, loader, device)
    assert scores.shape == (436, 9) and torch.isfinite(scores).all().item()
    assert torch.allclose(scores.sum(dim=1), torch.ones(436), atol=1e-6)
    values, indices = scores.max(dim=1)
    outputs[size] = list(zip([CLASSES[idx] for idx in indices.tolist()], values.tolist(), strict=True))
    print(f"Finished {size} px on {len(outputs[size])} dev images", flush=True)

for name, sha in prior.items():
    assert digest(model_dir / name) == sha, name
csv_path = root / "bapari-dev-paired-predictions.csv"
keys = ("path", "sha256", "class_id", "source", "split", "capture_group_candidate", "label_basis")
with csv_path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow([*keys, "prediction_224", "confidence_224", "prediction_280", "confidence_280"])
    for row, result_224, result_280 in zip(rows, outputs[224], outputs[280], strict=True):
        writer.writerow([*(row[key] for key in keys), *result_224, *result_280])
receipt = {
    "backend": "Google Colab",
    "gpu": torch.cuda.get_device_name(device),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "transformers": transformers.__version__,
    "pillow": pillow_version,
    "split": "dev",
    "images": len(rows),
    "sizes": [224, 280],
    "batch": batch,
    "test_time_horizontal_flip": True,
    "precision": "CUDA autocast fp16, float32 logits and probabilities",
    "image_input": "Publisher originals, EXIF transpose/RGB, direct bicubic fit-pad, no JPEG restaging",
    "model_zip_sha256": model_zip_sha,
    "model_file_sha256": prior,
    "input_manifest_sha256": manifest_sha,
    "runner_sha256": digest(Path(__file__)),
    "code_sha256": {path.name: digest(path) for path in sorted((root / "ai").glob("*.py"))},
    "predictions_sha256": digest(csv_path),
    "trained": False,
    "new_test_predictions": 0,
}
receipt_path = root / "bapari-dev-colab-receipt.json"
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
output = Path("/content/dinov2-bapari-dev-results.zip")
with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
    for path in (csv_path, receipt_path, Path(__file__)):
        archive.write(path, path.name)
print("BAPARI_REPLAY_COMPLETE", digest(output), output.stat().st_size, flush=True)
