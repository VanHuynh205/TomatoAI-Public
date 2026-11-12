"""CPU prediction using the validated bundle's ordered classes and transform."""

from pathlib import Path

import torch
from PIL import Image

from ai.artifacts import load_bundle
from ai.contracts import Prediction, ScoredClass
from ai.preprocessing import build_transform


class Predictor:
    def __init__(self, bundle_dir: Path, device: str = "cpu") -> None:
        if device != "cpu":
            raise ValueError("Predictor supports cpu only")
        bundle = load_bundle(bundle_dir)
        self._model = bundle.model
        self._class_ids = tuple(bundle.metadata["class_ids"])
        self._model_version: str = bundle.metadata["model_version"]
        self._threshold: float | None = bundle.metadata["threshold"]
        self._transform = build_transform(
            bundle.metadata["preprocessing"]["spec"], training=False
        )

    @property
    def class_ids(self) -> tuple[str, ...]:
        return self._class_ids

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(self, image: Image.Image) -> Prediction:
        if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
            raise ValueError("image must be a nonempty PIL image")
        with torch.inference_mode():
            tensor = self._transform(image).unsqueeze(0)
            logits = self._model(tensor)
            if (
                not isinstance(logits, torch.Tensor)
                or logits.shape != (1, len(self._class_ids))
                or not bool(torch.isfinite(logits).all())
            ):
                raise ValueError("invalid model logits")
            scores = logits.softmax(dim=1)[0].tolist()
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[
            :3
        ]
        top = [ScoredClass(self._class_ids[index], scores[index]) for index in order]
        accepted = self._threshold is not None and top[0].score >= self._threshold
        return Prediction(
            top_k=top,
            status="ok" if accepted else "uncertain",
            class_id=top[0].class_id if accepted else None,
            model_version=self._model_version,
        )
