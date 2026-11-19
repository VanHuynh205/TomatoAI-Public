"""Exercise DINOv3's nine-class rejection without pretrained weights."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from ai import dinov3_inference
from ai.dinov3_field import CLASSES


@pytest.fixture
def export(tmp_path: Path) -> Path:
    files = {
        "leaf-classifier.json": {
            "size": 224,
            "class_ids": CLASSES,
            "model_version": "dinov3-test",
        },
        "config.json": dinov3_inference._SMALL_VIT_CONFIG,
        "preprocessor_config.json": {
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        },
    }
    for name, data in files.items():
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    for name in ("model.safetensors", "classifier.safetensors"):
        (tmp_path / name).write_bytes(b"fixture")
    return tmp_path


@pytest.mark.parametrize(
    ("leaf_logit", "non_leaf_logit", "class_id"),
    [(4.0, 0.0, "healthy"), (4.0, 5.0, None), (2.0, 0.0, None)],
)
def test_predict_uses_nine_way_scores_with_eight_public_classes(
    export: Path,
    monkeypatch: pytest.MonkeyPatch,
    leaf_logit: float,
    non_leaf_logit: float,
    class_id: str | None,
) -> None:
    logits = torch.tensor([[0.0] * 7 + [leaf_logit, non_leaf_logit]])
    model = Mock(return_value=logits)
    load = Mock(return_value=model)
    monkeypatch.setattr(dinov3_inference, "_load_model", load)

    predictor = dinov3_inference.DinoV3Predictor(export)
    result = predictor.predict(Image.new("L", (320, 128)))

    load.assert_called_once_with(export.resolve())
    assert model.call_args.args[0].shape == (1, 3, 224, 224)
    assert result.model_version == "dinov3-test"
    assert predictor.class_ids == CLASSES[:-1]
    assert result.status == ("ok" if class_id else "uncertain")
    assert result.class_id == class_id
    assert [row.class_id for row in result.top_k] == ["healthy", *CLASSES[:2]]
    expected = logits.softmax(dim=1)[0, [7, 0, 1]].tolist()
    assert [row.score for row in result.top_k] == pytest.approx(expected)


def test_class_order_mismatch_fails_before_loading_weights(
    export: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = {"size": 224, "class_ids": list(reversed(CLASSES))}
    (export / "leaf-classifier.json").write_text(json.dumps(metadata), encoding="utf-8")
    load = Mock()
    monkeypatch.setattr(dinov3_inference, "_load_model", load)

    with pytest.raises(ValueError, match="class order"):
        dinov3_inference.DinoV3Predictor(export)
    load.assert_not_called()
