"""Offline CPU inference and export validation for CNN-GRU tomato leaf models."""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from safetensors.torch import load_file

from ai.artifacts import _json, _local, _name, _read, _threshold
from ai.cnn_gru_model import (
    SUPPORTED_BACKBONES,
    build_cnn_gru_model,
    cnn_gru_eval_transform,
)
from ai.contracts import Prediction
from ai.dinov3_field import CLASSES
from ai.dinov3_inference import prediction_from_logits

CNN_GRU_MANIFEST = "cnn-gru-release.json"
MAX_WEIGHT_BYTES = 256_000_000
EXPORT_LIMITS: dict[str, int] = {
    "leaf-classifier.json": 1_000_000,
    "model.safetensors": MAX_WEIGHT_BYTES,
}
CNN_GRU_FILES: tuple[str, ...] = (*EXPORT_LIMITS, CNN_GRU_MANIFEST)


def _validate_norm_vector(values: Any, *, positive_only: bool) -> list[float]:
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("CNN-GRU normalization vector must have length 3")
    result: list[float] = []
    for item in values:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            or not (0.0 < item <= 1.0 if positive_only else 0.0 <= item <= 1.0)
        ):
            raise ValueError("invalid CNN-GRU normalization value")
        result.append(float(item))
    return result


def validate_cnn_gru_export(
    model_dir: Path, *, require_trained: bool = False
) -> dict[str, Any]:
    """Verify exact 3-file CNN-GRU bundle digests, bounds and metadata."""
    if type(require_trained) is not bool:
        raise ValueError("require_trained must be bool")
    directory = _local(model_dir)
    if not directory.is_dir():
        raise ValueError("CNN-GRU model directory missing")

    entries = list(directory.iterdir())
    if {entry.name for entry in entries} != set(CNN_GRU_FILES):
        raise ValueError(
            "CNN-GRU export must contain only the three registered bundle files"
        )
    for entry in entries:
        info = entry.lstat()
        if (
            entry.is_symlink()
            or entry.is_junction()
            or bool(getattr(info, "st_file_attributes", 0) & 0x400)
            or not stat.S_ISREG(info.st_mode)
        ):
            raise ValueError(
                f"CNN-GRU bundle entry is not a regular local file: {entry.name}"
            )

    manifest = _json(_read(directory / CNN_GRU_MANIFEST, 1_000_000))
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        raise ValueError("unsupported CNN-GRU manifest schema")
    version = manifest.get("model_version")
    architecture = manifest.get("architecture")
    eval_size = manifest.get("evaluation_size")
    temperature = manifest.get("temperature")
    threshold = manifest.get("threshold")
    artifact_kind = manifest.get("artifact_kind")
    hashes = manifest.get("sha256")
    if not isinstance(version, str) or not _name(version):
        raise ValueError("invalid CNN-GRU model_version")
    if architecture not in SUPPORTED_BACKBONES:
        raise ValueError("unsupported CNN-GRU architecture")
    if type(eval_size) is not int or not 64 <= eval_size <= 1024 or eval_size % 32:
        raise ValueError(
            "CNN-GRU evaluation_size must be a multiple of 32 in [64, 1024]"
        )
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("invalid CNN-GRU temperature")
    if threshold is None or not _threshold(threshold):
        raise ValueError("invalid CNN-GRU threshold")
    if artifact_kind not in ("trained", "fixture"):
        raise ValueError("invalid CNN-GRU artifact_kind")
    if require_trained and artifact_kind != "trained":
        raise ValueError("fixture CNN-GRU bundle forbidden in release")
    if not isinstance(hashes, dict) or hashes.keys() != EXPORT_LIMITS.keys():
        raise ValueError("invalid CNN-GRU release manifest sha256 inventory")

    for name, limit in EXPORT_LIMITS.items():
        expected_digest = hashes[name]
        if (
            not isinstance(expected_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
        ):
            raise ValueError(f"invalid CNN-GRU digest entry: {name}")
        with _local(directory / name).open("rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
                raise ValueError(f"{name} size limit")
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected_digest:
                raise ValueError(f"CNN-GRU digest mismatch: {name}")

    metadata = _json(_read(directory / "leaf-classifier.json", 1_000_000))
    use_gru_head = metadata.get("use_gru_head")
    hidden_dim = metadata.get("hidden_dim", 256)
    proj_dim = metadata.get("proj_dim", 512)
    if (
        type(metadata.get("schema_version")) is not int
        or metadata.get("schema_version") != 1
        or metadata.get("class_ids") != list(CLASSES)
        or metadata.get("architecture") != architecture
        or metadata.get("model_version") != version
        or metadata.get("size") != eval_size
        or metadata.get("artifact_kind") != artifact_kind
        or metadata.get("test_time_horizontal_flip") is not True
        or type(use_gru_head) is not bool
        or type(hidden_dim) is not int
        or hidden_dim <= 0
        or type(proj_dim) is not int
        or proj_dim <= 0
        or isinstance(metadata.get("temperature"), bool)
        or metadata.get("temperature") != temperature
        or isinstance(metadata.get("threshold"), bool)
        or metadata.get("threshold") != threshold
    ):
        raise ValueError("incompatible CNN-GRU leaf export metadata")

    mean = _validate_norm_vector(metadata.get("mean"), positive_only=False)
    std = _validate_norm_vector(metadata.get("std"), positive_only=True)
    # Verify normalization stays finite in float32 before predictor construction.
    cnn_gru_eval_transform(eval_size, mean, std)
    return {
        "model_version": version,
        "architecture": architecture,
        "use_gru_head": use_gru_head,
        "hidden_dim": hidden_dim,
        "proj_dim": proj_dim,
        "size": eval_size,
        "mean": mean,
        "std": std,
        "temperature": float(temperature),
        "threshold": float(threshold),
        "artifact_kind": artifact_kind,
        "guardrail_passed": metadata.get("guardrail_passed") is True,
    }


class CnnGruPredictor:
    """Offline CPU predictor for verified CNN-GRU exports."""

    def __init__(self, model_dir: Path, *, require_trained: bool = False) -> None:
        directory = _local(model_dir)
        meta = validate_cnn_gru_export(directory, require_trained=require_trained)
        model = build_cnn_gru_model(
            meta["architecture"],
            len(CLASSES),
            use_gru_head=meta["use_gru_head"],
            hidden_dim=meta["hidden_dim"],
            proj_dim=meta["proj_dim"],
        )
        state = load_file(directory / "model.safetensors", device="cpu")
        expected = model.state_dict()
        if state.keys() != expected.keys():
            raise ValueError("CNN-GRU state_dict keys mismatch")
        for key, ref in expected.items():
            saved = state[key]
            if (
                not isinstance(saved, torch.Tensor)
                or saved.layout != torch.strided
                or saved.shape != ref.shape
                or saved.dtype != ref.dtype
                or not bool(torch.isfinite(saved).all())
            ):
                raise ValueError(f"CNN-GRU state_dict tensor invalid: {key}")
        model.load_state_dict(state, strict=True)
        model.cpu().eval().requires_grad_(False)

        self._model = model
        self._model_version: str = meta["model_version"]
        self._temperature: float = meta["temperature"]
        self._threshold: float = meta["threshold"]
        self._transform = cnn_gru_eval_transform(
            meta["size"], meta["mean"], meta["std"]
        )

    @property
    def class_ids(self) -> tuple[str, ...]:
        return CLASSES[:-1]

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def threshold(self) -> float:
        return self._threshold

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Prediction:
        if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
            raise ValueError("image must be a nonempty PIL image")
        pixels = self._transform(image).unsqueeze(0)
        logits = (
            self._model(pixels).float() + self._model(pixels.flip(-1)).float()
        ) / 2.0
        return prediction_from_logits(
            logits,
            self._model_version,
            temperature=self._temperature,
            threshold=self._threshold,
        )
