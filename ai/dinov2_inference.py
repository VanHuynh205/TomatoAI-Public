"""Serve the frozen DINOv2 v1 export with the benchmark's 280 px inference."""

import hashlib
import os
import stat
from pathlib import Path

import torch
from PIL import Image, ImageOps

from ai.artifacts import _json, _local, _name, _read
from ai.contracts import Prediction
from ai.dinov2_field import BACKBONE, REVISION
from ai.dinov2_improve import load_export
from ai.dinov3_field import CLASSES, leaf_transform
from ai.dinov3_inference import prediction_from_logits

DINO_V2_MANIFEST = "dinov2-release.json"
EXPORT_LIMITS = {
    "leaf-classifier.json": 1_000_000,
    "config.json": 1_000_000,
    "preprocessor_config.json": 1_000_000,
    "model.safetensors": 360_000_000,
    "classifier.safetensors": 1_000_000,
}
DINO_V2_FILES = (*EXPORT_LIMITS, DINO_V2_MANIFEST)
INPUT_SIZE = 280
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
_BASE_CONFIG = {
    "model_type": "dinov2",
    "hidden_size": 768,
    "image_size": 518,
    "mlp_ratio": 4,
    "num_attention_heads": 12,
    "num_channels": 3,
    "num_hidden_layers": 12,
    "patch_size": 14,
}


def validate_dinov2_export(model_dir: Path) -> str:
    directory = _local(model_dir)
    manifest = _json(_read(directory / DINO_V2_MANIFEST, 1_000_000))
    version, hashes = manifest.get("model_version"), manifest.get("sha256")
    if (
        not isinstance(version, str)
        or not _name(version)
        or not isinstance(hashes, dict)
    ):
        raise ValueError("invalid DINOv2 release manifest")
    if (
        manifest.get("evaluation_size") != INPUT_SIZE
        or hashes.keys() != EXPORT_LIMITS.keys()
    ):
        raise ValueError("incompatible DINOv2 release manifest")
    for name, limit in EXPORT_LIMITS.items():
        with _local(directory / name).open("rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
                raise ValueError(f"{name} size limit")
            if hashlib.file_digest(stream, "sha256").hexdigest() != hashes[name]:
                raise ValueError(f"DINOv2 digest mismatch: {name}")

    config = _json(_read(directory / "config.json", 1_000_000))
    metadata = _json(_read(directory / "leaf-classifier.json", 1_000_000))
    if config.get("auto_map") is not None or any(
        config.get(key) != value for key, value in _BASE_CONFIG.items()
    ):
        raise ValueError("expected a DINOv2 ViT-B/14 backbone")
    if (
        metadata.get("backbone") != BACKBONE
        or metadata.get("revision") != REVISION
        or metadata.get("class_ids") != list(CLASSES)
        or metadata.get("size") != 224
        or metadata.get("selected_epoch") != 6
        or metadata.get("test_time_horizontal_flip") is not True
        or metadata.get("preprocessing")
        != {
            "geometry": "fit_pad",
            "interpolation": "bicubic",
            "mean": _MEAN,
            "std": _STD,
        }
    ):
        raise ValueError("incompatible DINOv2 v1 leaf export")
    return version


class DinoV2Predictor:
    class_ids = CLASSES[:-1]

    def __init__(self, model_dir: Path) -> None:
        self.model_version = validate_dinov2_export(model_dir)
        self._model, _ = load_export(model_dir, torch.device("cpu"))
        self._model.eval()
        self._transform = leaf_transform(INPUT_SIZE, _MEAN, _STD, training=False)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Prediction:
        if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
            raise ValueError("image must be a nonempty PIL image")
        pixels = self._transform(
            ImageOps.exif_transpose(image).convert("RGB")
        ).unsqueeze(0)
        logits = (
            self._model(pixels).float() + self._model(pixels.flip(-1)).float()
        ) / 2
        return prediction_from_logits(logits, self.model_version)
