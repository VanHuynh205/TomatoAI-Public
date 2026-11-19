"""Collect receipt-bound DINOv3 logits on Colab; CPU use is for fixtures only.

This collector does not qualify data or authorize opening a final test.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, default_collate

from ai.artifacts import _json, _local, _read
from ai.dinov3_field import CLASSES, _file_sha256, predict
from ai.dinov3_inference import DinoV3Predictor

MODEL_FILES = {
    "config.json", "preprocessor_config.json", "leaf-classifier.json",
    "model.safetensors", "classifier.safetensors",
}
CODE_FILES = {
    "ai/dinov3_evaluate.py", "ai/dinov3_inference.py", "ai/dinov3_field.py",
    "ai/artifacts.py", "ai/contracts.py",
}


def _input_path(root: Path, name: str) -> Path:
    if "\\" in name or ":" in name or any(part in {"", ".", ".."} for part in name.split("/")):
        raise ValueError("Input path must be canonical and relative to its root")
    path = _local(root / name)
    if not path.is_relative_to(root):
        raise ValueError("Input path leaves its root")
    return path


class EvaluationImages(Dataset[tuple[torch.Tensor, int, int]]):
    def __init__(self, root: Path, rows: list[dict[str, str]], transform: Any):
        self.root = _local(root)
        snapshot = [row.copy() for row in rows]
        self.transform = transform
        paths, digests = set(), set()
        if not snapshot:
            raise ValueError("Evaluation manifest must not be empty")
        for row in snapshot:
            if any(not isinstance(row.get(key), str) or not row[key].strip()
                   for key in ("path", "sha256", "class_id", "source", "group", "split")):
                raise ValueError("Evaluation identity metadata is missing")
            path = _input_path(self.root, row["path"])
            if (
                not path.is_file() or path in paths
                or row["sha256"] in digests or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                or row["class_id"] not in CLASSES or row["split"] not in {"val", "test"}
            ):
                raise ValueError("Invalid or duplicate evaluation identity")
            paths.add(path)
            digests.add(row["sha256"])
        self._rows = tuple(tuple(row.items()) for row in snapshot)

    @property
    def rows(self) -> list[dict[str, str]]:
        return [dict(row) for row in self._rows]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        row = dict(self._rows[index])
        content = _read(self.root / row["path"], 64_000_000)
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError(f"Consumed image digest differs from manifest: {row['path']}")
        # Decode the verified buffer, never a second read of the mutable pathname.
        with Image.open(io.BytesIO(content)) as image:
            pixels = self.transform(ImageOps.exif_transpose(image).convert("RGB"))
        return pixels, CLASSES.index(row["class_id"]), index


def collect_logits(
    model: torch.nn.Module, loader: DataLoader[Any], device: torch.device,
) -> list[dict[str, Any]]:
    if not isinstance(loader.dataset, EvaluationImages):
        raise TypeError("Evaluation loader must carry verified image identities")
    if loader.collate_fn is not default_collate:
        raise ValueError("Evaluation target/identity binding requires default collation")
    rows = loader.dataset.rows
    consumed: list[int] = []
    seen: set[int] = set()

    def batches() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for pixels, targets, indices in loader:
            for values in (targets, indices):
                if (
                    not isinstance(values, torch.Tensor) or values.shape != (len(pixels),)
                    or values.dtype == torch.bool or values.is_floating_point() or values.is_complex()
                ):
                    raise ValueError("Evaluation identity and target indices must be integer vectors")
            for index, target in zip(indices.tolist(), targets.tolist(), strict=True):
                if not 0 <= index < len(rows) or index in seen:
                    raise ValueError("Unknown or repeated consumed image identity")
                if target != CLASSES.index(rows[index]["class_id"]):
                    raise ValueError("Consumed target differs from image identity")
                seen.add(index)
                consumed.append(index)
            yield pixels, targets
        if len(seen) != len(rows):
            raise ValueError("Evaluation did not consume the complete manifest")

    predictions: list[dict[str, Any]] = []
    labels, confidence = predict(model, batches(), device, predictions=predictions)
    return [
        {
            **rows[index], **record, "manifest_index": index, "image_id": rows[index]["sha256"],
            "prediction": label, "confidence": score,
        }
        for index, record, label, score in zip(consumed, predictions, labels, confidence, strict=True)
    ]


def _verify_pins(root: Path, pins: Any, required: set[str]) -> None:
    if not isinstance(pins, dict) or not required <= pins.keys():
        raise ValueError("Input receipt omits required file pins")
    for name, digest in pins.items():
        if not isinstance(name, str) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Input receipt has invalid file pins")
        path = _input_path(root, name)
        if not path.is_file():
            raise ValueError("Input receipt has an invalid local path")
        if not 0 < path.stat().st_size <= 256_000_000 or _file_sha256(path) != digest:
            raise ValueError(f"Frozen input digest differs from receipt: {name}")


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate evaluation receipt field: {name}")
        result[name] = value
    return result


def evaluate(
    data: Path, model_dir: Path, receipt_path: Path, output: Path,
    *, device: torch.device, batch_size: int = 32, workers: int = 2,
) -> Path:
    """Publish one complete collection without overwriting any previous result.

    The caller supplies frozen input hashes. Data admission, exposure history,
    calibration and final-test authorization remain separate experiment gates.
    """
    data, model_dir, receipt_path, output = map(_local, (data, model_dir, receipt_path, output))
    if output.exists():
        raise FileExistsError(output)
    if type(batch_size) is not int or batch_size < 1 or type(workers) is not int or workers < 0:
        raise ValueError("Evaluation batch size must be positive and workers nonnegative")
    receipt_bytes = _read(receipt_path, 1_000_000)
    receipt = json.loads(receipt_bytes, object_pairs_hook=_unique_fields)
    if not isinstance(receipt, dict):
        raise TypeError("Evaluation input receipt must be an object")
    if (
        type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1
        or receipt.get("status") != "frozen_inputs"
        or receipt.get("artifact_kind") not in {"fixture", "experimental"}
    ):
        raise ValueError("Evaluation input receipt must be frozen and versioned")
    if receipt.get("class_ids") != list(CLASSES):
        raise ValueError("Input receipt class order differs from DINOv3")
    if receipt["artifact_kind"] == "experimental" and device.type != "cuda":
        raise ValueError("Experimental DINOv3 evaluation must run on Colab CUDA")
    manifest = receipt.get("manifest")
    if (
        not isinstance(manifest, dict) or not isinstance(manifest.get("path"), str)
        or type(manifest.get("rows")) is not int or manifest["rows"] <= 0
    ):
        raise ValueError("Input receipt must pin a nonempty evaluation manifest")
    source_root = Path(__file__).resolve().parents[1]

    def verify_inputs() -> None:
        if _read(receipt_path, 1_000_000) != receipt_bytes:
            raise ValueError("Input receipt changed during evaluation")
        _verify_pins(data, {manifest["path"]: manifest.get("sha256")}, {manifest["path"]})
        _verify_pins(model_dir, receipt.get("model_sha256"), MODEL_FILES)
        _verify_pins(source_root, receipt.get("code_sha256"), CODE_FILES)
        names = {path.name for path in model_dir.iterdir()}
        if not names <= MODEL_FILES | {"LICENSE.md"} or names != set(receipt["model_sha256"]):
            raise ValueError("Model export contains unsupported or unpinned loader files")
        if "configuration_files" in _json(_read(model_dir / "config.json", 1_000_000)):
            raise ValueError("Model export must not redirect its configuration")

    verify_inputs()
    manifest_bytes = _read(_input_path(data, manifest["path"]), 256_000_000)
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest["sha256"]:
        raise ValueError("Consumed manifest digest differs from receipt")
    with io.StringIO(manifest_bytes.decode("utf-8"), newline="") as manifest_stream:
        reader = csv.DictReader(manifest_stream, strict=True)
        header = reader.fieldnames
        if not header or len(set(header)) != len(header) or any(not name for name in header):
            raise ValueError("Evaluation manifest has invalid or duplicate header columns")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Evaluation manifest row has a different number of columns")
    if len(rows) != manifest["rows"] or len({row["split"] for row in rows}) != 1:
        raise ValueError("Evaluation manifest count or split differs from receipt")
    dataset = EvaluationImages(data, rows, None)
    # Reuse the export validation, weights and preprocessing used by the application.
    predictor = DinoV3Predictor(model_dir)
    dataset.transform = predictor._transform
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    records = collect_logits(predictor._model.to(device), loader, device)
    verify_inputs()
    result = {
        "schema_version": 1, "artifact_kind": receipt["artifact_kind"],
        "data_qualification": "not_assessed", "class_ids": list(CLASSES),
        "input_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(), "input_receipt": receipt,
        "model_version": predictor.model_version,
        "runtime": {
            "device": str(device), "autocast": "float16" if device.type == "cuda" else "disabled",
            "batch_size": batch_size, "workers": workers,
            "versions": {name: version(name) for name in ("torch", "torchvision", "transformers", "safetensors", "Pillow")},
        },
        "predictions": records,
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".pending", delete=False) as stream:
        pending = Path(stream.name)
        try:
            stream.write(serialized.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            pending.unlink()
            raise
    try:
        # Hard-link publication is atomic and fails if another run claimed this name.
        os.link(pending, output)
    finally:
        pending.unlink()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Run experimental evaluation on Colab with a CUDA runtime")
    evaluate(args.data, args.model, args.receipt, args.output,
             device=torch.device("cuda"), batch_size=args.batch, workers=args.workers)


if __name__ == "__main__":
    main()
