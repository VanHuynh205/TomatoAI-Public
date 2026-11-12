"""Offline CPU prediction from a local DINOv3 ViT-S/16 leaf export."""

import stat
from pathlib import Path

import torch
from PIL import Image, ImageOps

from ai.artifacts import _json, _local, _name, _read
from ai.contracts import Prediction, ScoredClass
from ai.dinov3_field import CLASSES, LeafClassifier, leaf_transform

_SMALL_VIT_CONFIG = {
    "model_type": "dinov3_vit",
    "hidden_size": 384,
    "intermediate_size": 1536,
    "num_attention_heads": 6,
    "num_channels": 3,
    "num_hidden_layers": 12,
    "num_register_tokens": 4,
    "patch_size": 16,
}


def _load_model(directory: Path) -> LeafClassifier:
    from safetensors.torch import load_file
    from transformers import AutoModel

    backbone, loading = AutoModel.from_pretrained(
        directory,
        local_files_only=True,
        use_safetensors=True,
        trust_remote_code=False,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    if any(loading.values()):
        raise ValueError("DINOv3 backbone weights did not load exactly")
    model = LeafClassifier(backbone)
    model.head.load_state_dict(
        load_file(directory / "classifier.safetensors", device="cpu")
    )
    model.cpu().eval()
    return model


class DinoV3Predictor:
    def __init__(self, model_dir: Path) -> None:
        directory = _local(model_dir)
        metadata = _json(_read(directory / "leaf-classifier.json", 1_000_000))
        size = metadata.get("size")
        version = metadata.get("model_version", "dinov3-field-v1")
        if metadata.get("class_ids") != list(CLASSES):
            raise ValueError("DINOv3 class order does not match the leaf export")
        if type(size) is not int or not 224 <= size <= 336 or size % 16:
            raise ValueError(
                "DINOv3 input size must be a multiple of 16 from 224 to 336"
            )
        if not _name(version):
            raise ValueError("invalid DINOv3 model_version")

        config = _json(_read(directory / "config.json", 1_000_000))
        if any(config.get(key) != value for key, value in _SMALL_VIT_CONFIG.items()):
            raise ValueError("expected a DINOv3 ViT-S/16 backbone")
        processor = _json(_read(directory / "preprocessor_config.json", 1_000_000))
        mean, std = processor.get("image_mean"), processor.get("image_std")
        if (
            not isinstance(mean, list)
            or not isinstance(std, list)
            or any(
                len(values) != 3
                or any(
                    type(value) not in (float, int) or not 0 <= value <= 1
                    for value in values
                )
                for values in (mean, std)
            )
            or min(std) <= 0
        ):
            raise ValueError("invalid DINOv3 normalization")
        for name in ("model.safetensors", "classifier.safetensors"):
            info = _local(directory / name).stat()
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 256_000_000:
                raise ValueError(f"{name} size limit")

        self._model_version: str = version
        self._model = _load_model(directory)
        self._transform = leaf_transform(size, mean, std, training=False)

    @property
    def class_ids(self) -> tuple[str, ...]:
        return CLASSES[:-1]

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(self, image: Image.Image) -> Prediction:
        if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
            raise ValueError("image must be a nonempty PIL image")
        with torch.inference_mode():
            tensor = self._transform(ImageOps.exif_transpose(image).convert("RGB"))
            return prediction_from_logits(
                self._model(tensor.unsqueeze(0)), self._model_version
            )


def prediction_from_logits(logits: torch.Tensor, model_version: str) -> Prediction:
    if (
        not isinstance(logits, torch.Tensor)
        or logits.shape != (1, len(CLASSES))
        or not bool(torch.isfinite(logits).all())
    ):
        raise ValueError("invalid leaf classifier logits")
    scores = logits.float().softmax(dim=1)[0].tolist()
    order = sorted(range(len(CLASSES)), key=lambda index: (-scores[index], index))
    non_leaf = len(CLASSES) - 1
    top = [
        ScoredClass(CLASSES[index], scores[index])
        for index in order
        if index != non_leaf
    ][:3]
    accepted = order[0] != non_leaf and top[0].score >= 0.5
    return Prediction(
        top_k=top,
        status="ok" if accepted else "uncertain",
        class_id=top[0].class_id if accepted else None,
        model_version=model_version,
    )
