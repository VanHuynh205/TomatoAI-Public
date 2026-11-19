"""Tests for CNN-GRU inference, export validation, and temperature/threshold contracts."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from ai.cnn_gru_inference import (
    CNN_GRU_FILES,
    CNN_GRU_MANIFEST,
    EXPORT_LIMITS,
    MAX_WEIGHT_BYTES,
    CnnGruPredictor,
    validate_cnn_gru_export,
)
from ai.cnn_gru_model import IMAGENET_MEAN, IMAGENET_STD, build_cnn_gru_model
from ai.dinov3_field import CLASSES
from ai.dinov3_inference import prediction_from_logits


def _write_cnn_gru_bundle(
    target_dir: Path,
    *,
    architecture: str = "resnet50",
    use_gru_head: bool = True,
    artifact_kind: str = "trained",
    temperature: float = 1.25,
    threshold: float = 0.45,
    model_version: str = "cnn-gru-test-v1",
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    model = build_cnn_gru_model(
        architecture,
        len(CLASSES),
        use_gru_head=use_gru_head,
        hidden_dim=16,
        proj_dim=32,
    )
    weights_path = target_dir / "model.safetensors"
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        weights_path,
    )
    meta = {
        "schema_version": 1,
        "architecture": architecture,
        "use_gru_head": use_gru_head,
        "hidden_dim": 16,
        "proj_dim": 32,
        "model_version": model_version,
        "class_ids": list(CLASSES),
        "size": 288,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "test_time_horizontal_flip": True,
        "temperature": temperature,
        "threshold": threshold,
        "artifact_kind": artifact_kind,
    }
    meta_path = target_dir / "leaf-classifier.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    hashes = {
        name: hashlib.sha256((target_dir / name).read_bytes()).hexdigest()
        for name in EXPORT_LIMITS
    }
    manifest = {
        "schema_version": 1,
        "model_version": model_version,
        "architecture": architecture,
        "evaluation_size": 288,
        "temperature": temperature,
        "threshold": threshold,
        "artifact_kind": artifact_kind,
        "sha256": hashes,
    }
    (target_dir / CNN_GRU_MANIFEST).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return target_dir


def test_validate_export_rejects_bad_hash_oversize_symlink_and_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert MAX_WEIGHT_BYTES == 256_000_000
    assert CNN_GRU_FILES == (
        "leaf-classifier.json",
        "model.safetensors",
        "cnn-gru-release.json",
    )

    fixture_dir = _write_cnn_gru_bundle(tmp_path / "fixture", artifact_kind="fixture")
    validated = validate_cnn_gru_export(fixture_dir, require_trained=False)
    assert validated["artifact_kind"] == "fixture"
    with pytest.raises(ValueError, match="fixture"):
        validate_cnn_gru_export(fixture_dir, require_trained=True)

    trained_dir = _write_cnn_gru_bundle(tmp_path / "trained", artifact_kind="trained")
    assert validate_cnn_gru_export(trained_dir, require_trained=True)[
        "model_version"
    ] == ("cnn-gru-test-v1")

    # Extra file must be rejected.
    extra = trained_dir / "LICENSE.md"
    extra.write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="only"):
        validate_cnn_gru_export(trained_dir, require_trained=True)
    extra.unlink()

    # Tampered weights hash must fail before deserialization.
    (trained_dir / "model.safetensors").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="digest"):
        validate_cnn_gru_export(trained_dir, require_trained=True)

    # Oversize file rejected before reading.
    oversized_dir = _write_cnn_gru_bundle(
        tmp_path / "oversized", artifact_kind="trained"
    )
    with (oversized_dir / "model.safetensors").open("wb") as stream:
        stream.truncate(MAX_WEIGHT_BYTES + 1)
    with pytest.raises(ValueError, match="size"):
        validate_cnn_gru_export(oversized_dir, require_trained=True)

    # Boolean schema_version/temperature/threshold and float32-overflow std must fail.
    for field, bad_val in (
        ("schema_version", True),
        ("temperature", True),
        ("threshold", True),
        ("std", [1e-40, 0.2, 0.2]),
    ):
        bad_dir = _write_cnn_gru_bundle(
            tmp_path / f"bad_{field}",
            artifact_kind="trained",
            temperature=1.0,
            threshold=1.0,
        )
        meta_path = bad_dir / "leaf-classifier.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta[field] = bad_val
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        man_path = bad_dir / CNN_GRU_MANIFEST
        man = json.loads(man_path.read_text(encoding="utf-8"))
        man["sha256"]["leaf-classifier.json"] = hashlib.sha256(
            meta_path.read_bytes()
        ).hexdigest()
        man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
        with pytest.raises(ValueError):
            validate_cnn_gru_export(bad_dir, require_trained=True)

    # Symlink / junction rejected.
    link = tmp_path / "linked"
    try:
        link.symlink_to(trained_dir, target_is_directory=True)
    except OSError:
        monkeypatch.setattr(
            Path,
            "resolve",
            lambda self, strict=False: (
                tmp_path / "other" if self.name == "trained" else Path(str(self))
            ),
        )
        with pytest.raises(ValueError, match="symlink|junction"):
            validate_cnn_gru_export(trained_dir, require_trained=True)
    else:
        with pytest.raises(ValueError, match="symlink|junction"):
            validate_cnn_gru_export(link, require_trained=True)


def test_predictor_exposes_eight_care_classes_and_loads_safetensors_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("CnnGruPredictor attempted network download")

    monkeypatch.setattr(torch.hub, "download_url_to_file", forbidden)
    bundle = _write_cnn_gru_bundle(tmp_path / "bundle", artifact_kind="trained")
    predictor = CnnGruPredictor(bundle, require_trained=True)
    assert predictor.class_ids == CLASSES[:-1]
    assert predictor.model_version == "cnn-gru-test-v1"
    for field in ("class_ids", "model_version"):
        with pytest.raises(AttributeError):
            setattr(predictor, field, "tampered")
    result = predictor.predict(Image.new("RGB", (64, 48), (30, 140, 40)))
    assert result.model_version == "cnn-gru-test-v1"
    assert len(result.top_k) == 3
    assert all(item.class_id in CLASSES[:-1] for item in result.top_k)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "class Block:\n"
                "    def find_spec(self, fullname, *args):\n"
                "        if fullname.split('.')[0] in {'transformers', 'timm', 'ultralytics'}:\n"
                "            raise RuntimeError('forbidden import: ' + fullname)\n"
                "sys.meta_path.insert(0, Block())\n"
                "from pathlib import Path\n"
                "from PIL import Image\n"
                "from ai.cnn_gru_inference import CnnGruPredictor\n"
                f"p = CnnGruPredictor(Path({str(bundle)!r}), require_trained=True)\n"
                "assert p.predict(Image.new('RGB', (32, 32), (20, 120, 30))).model_version == 'cnn-gru-test-v1'\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert probe.returncode == 0, probe.stderr


def test_high_temperature_tie_preserves_raw_logit_order() -> None:
    logits = torch.tensor(
        [[0.0, 1e-15, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0]],
        dtype=torch.float64,
    )
    pred = prediction_from_logits(
        logits, "cnn-gru-test-v1", temperature=20.0, threshold=0.1
    )
    assert pred.top_k[0].class_id == "early_blight"
    assert pred.class_id == "early_blight"
    assert pred.status == "ok"

    # Backward-compatible 2-argument invocation still works.
    compat = prediction_from_logits(
        torch.tensor([[0.0] * 7 + [4.0, 0.0]]), "dinov3-compat"
    )
    assert compat.status == "ok"
    assert compat.class_id == "healthy"

    with pytest.raises(ValueError, match="temperature"):
        prediction_from_logits(logits, "v1", temperature=0.0)
    with pytest.raises(ValueError, match="threshold"):
        prediction_from_logits(logits, "v1", threshold=1.5)
    with pytest.raises(ValueError, match="logits"):
        prediction_from_logits(torch.ones((1, 9), dtype=torch.long), "v1")
    with pytest.raises(ValueError, match="logits"):
        prediction_from_logits(torch.ones((1, 9), dtype=torch.bool), "v1")


def test_predictor_rejects_not_tomato_leaf_even_at_high_score(tmp_path: Path) -> None:
    bundle = _write_cnn_gru_bundle(tmp_path / "bundle", temperature=1.0, threshold=0.05)
    predictor = CnnGruPredictor(bundle)
    high_non_leaf = torch.tensor([[-5.0] * 7 + [1.0, 8.0]])
    predictor._model = Mock(return_value=high_non_leaf)
    result = predictor.predict(Image.new("RGB", (32, 32), (120, 120, 120)))
    assert result.status == "uncertain"
    assert result.class_id is None
    assert all(item.class_id != "not_tomato_leaf" for item in result.top_k)


def test_predictor_tta_and_threshold_boundary(tmp_path: Path) -> None:
    bundle = _write_cnn_gru_bundle(tmp_path / "bundle", temperature=2.0, threshold=0.6)
    predictor = CnnGruPredictor(bundle)
    orig = torch.zeros(1, 9)
    orig[0, 8] = 6.0
    flip = torch.zeros(1, 9)
    flip[0, 7] = 12.0
    mock_model = Mock(side_effect=[orig, flip])
    predictor._model = mock_model

    image = Image.new("RGB", (20, 20), (0, 0, 0))
    image.paste((255, 255, 255), (0, 0, 8, 20))
    res = predictor.predict(image)

    first, second = [call.args[0] for call in mock_model.call_args_list]
    assert first.shape == (1, 3, 288, 288)
    assert torch.equal(second, first.flip(-1))
    expected_probs = (((orig + flip) / 2.0) / 2.0).double().softmax(dim=1)
    top_score = expected_probs[0, 7].item()
    assert res.top_k[0].class_id == "healthy"
    assert res.top_k[0].score == pytest.approx(top_score)
    # top_score here is exp(3) / (exp(3) + exp(1.5) + 7) ~= 0.635 >= 0.6 -> ok
    assert res.status == "ok"
    assert res.class_id == "healthy"

    # Raising threshold above top_score rejects cleanly.
    bundle_strict = _write_cnn_gru_bundle(
        tmp_path / "bundle_strict", temperature=2.0, threshold=0.75
    )
    predictor_strict = CnnGruPredictor(bundle_strict)
    predictor_strict._model = Mock(side_effect=[orig, flip])
    res_uncertain = predictor_strict.predict(image)
    assert res_uncertain.status == "uncertain"
    assert res_uncertain.class_id is None
