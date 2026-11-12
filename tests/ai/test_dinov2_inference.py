import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from ai import dinov2_inference as inference


def test_export_rejects_wrong_digest(tmp_path: Path) -> None:
    (tmp_path / "leaf-classifier.json").write_text("{}", encoding="utf-8")
    (tmp_path / inference.DINO_V2_MANIFEST).write_text(
        json.dumps(
            {
                "model_version": "dinov2-v1-280",
                "evaluation_size": 280,
                "sha256": dict.fromkeys(inference.EXPORT_LIMITS, "0" * 64),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="digest mismatch: leaf-classifier.json"):
        inference.validate_dinov2_export(tmp_path)


def test_runtime_averages_original_and_flipped_logits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = torch.zeros(1, 9)
    original[0, -1] = 6
    flipped = torch.zeros(1, 9)
    flipped[0, 7] = 12
    model = Mock(side_effect=[original, flipped])
    monkeypatch.setattr(inference, "validate_dinov2_export", lambda _: "dinov2-v1-280")
    monkeypatch.setattr(inference, "load_export", lambda *_: (model, {}))

    image = Image.new("RGB", (16, 16), color="black")
    image.paste((255, 255, 255), (0, 0, 6, 16))
    predictor = inference.DinoV2Predictor(tmp_path)
    result = predictor.predict(image)

    first, second = [call.args[0] for call in model.call_args_list]
    assert first.shape == (1, 3, 280, 280)
    assert torch.equal(second, first.flip(-1))
    assert result.class_id == "healthy"
    assert result.model_version == "dinov2-v1-280"
    assert result.top_k[0].score == pytest.approx(
        ((original + flipped) / 2).softmax(dim=1)[0, 7].item()
    )
