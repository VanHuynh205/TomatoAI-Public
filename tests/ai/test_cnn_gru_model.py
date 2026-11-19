"""Tests for CNN + 2D Axial-BiGRU + Gated Spatial Attention model architecture."""

import pickle
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision import models  # type: ignore[import-untyped]

from ai.cnn_gru_model import (
    SUPPORTED_BACKBONES,
    CascadedAxialBiGRUGatedAttnHead,
    SpatialBiGruClassifier,
    build_cnn_gru_model,
    cnn_gru_eval_transform,
    convert_convnext_to_torchvision,
)


def test_factory_never_downloads_when_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("model factory attempted network download")

    monkeypatch.setattr(torch.hub, "download_url_to_file", forbidden)
    assert SUPPORTED_BACKBONES == (
        "resnet50",
        "convnext_tiny",
        "convnext_small",
        "efficientnet_v2_s",
    )
    for arch in SUPPORTED_BACKBONES:
        model = build_cnn_gru_model(arch, 9)
        assert isinstance(model, SpatialBiGruClassifier)
    ablation = build_cnn_gru_model("resnet50", 9, use_gru_head=False)
    assert isinstance(ablation, SpatialBiGruClassifier)
    with pytest.raises(ValueError, match="architecture"):
        build_cnn_gru_model("yolo26s_cls", 9)
    with pytest.raises(ValueError, match="class_count"):
        build_cnn_gru_model("resnet50", 0)


@pytest.mark.parametrize(
    ("arch", "c4", "use_gru"),
    [
        ("resnet50", 2048, True),
        ("resnet50", 2048, False),
        ("convnext_tiny", 768, True),
        ("convnext_small", 768, True),
        ("efficientnet_v2_s", 1280, True),
    ],
)
def test_model_shapes_encode_image_and_weight_budget(
    arch: str, c4: int, use_gru: bool
) -> None:
    torch.set_num_threads(2)
    model = build_cnn_gru_model(arch, 9, use_gru_head=use_gru)
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            assert not module.training

    model.eval()
    pixels = torch.randn(2, 3, 288, 288)
    with torch.inference_mode():
        emb = model.encode_image(pixels)
        logits = model(pixels)
    expected_dim = c4 + 512 if use_gru else c4
    assert emb.shape == (2, expected_dim)
    assert logits.shape == (2, 9)
    assert torch.isfinite(emb).all()
    assert torch.isfinite(logits).all()
    total_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()
    )
    assert total_bytes <= 256_000_000


def test_eval_transform_rejects_oversized_and_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transform = cnn_gru_eval_transform(
        288, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )

    oversized = Image.Image()
    oversized._size = (5000, 4001)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("oversized input reached decode or copy")

    monkeypatch.setattr(oversized, "load", forbidden)
    monkeypatch.setattr(oversized, "copy", forbidden)
    with pytest.raises(ValueError, match="pixel_limit"):
        transform(oversized)

    image = Image.new("RGB", (321, 197), (30, 120, 45))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 80, 196), fill=(210, 40, 20))
    draw.rectangle((260, 0, 320, 60), fill=(20, 50, 230))
    image.getexif()[274] = 6

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        first = transform(image)
        second = transform(image)
        unpickled = pickle.loads(pickle.dumps(transform))(image)
    finally:
        torch.set_default_dtype(previous_dtype)

    assert first.shape == (3, 288, 288)
    assert first.dtype == torch.float32
    assert first.device.type == "cpu"
    assert torch.equal(first, second)
    assert torch.equal(first, unpickled)

    # Verify exact ImageOps.pad bicubic geometry and padding color on transposed image
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    padded = ImageOps.pad(
        rgb, (288, 288), method=Image.Resampling.BICUBIC, color=(124, 116, 104)
    )
    assert padded.size == (288, 288)
    # Corner (0, 0) lies in the letterbox padding because 197 < 321 (transposed: 197x321).
    corner = first[:, 0, 0]
    expected_corner = torch.tensor(
        [
            (124 / 255 - 0.485) / 0.229,
            (116 / 255 - 0.456) / 0.224,
            (104 / 255 - 0.406) / 0.225,
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(corner, expected_corner, atol=1e-6, rtol=0)

    for bad_mean, bad_std in (
        ((True, 0.5, 0.5), (0.2, 0.2, 0.2)),
        ("010", (0.2, 0.2, 0.2)),
        ((0.5, 0.5, 0.5), (1e-40, 0.2, 0.2)),
    ):
        with pytest.raises(ValueError, match="normalization|mean|std"):
            cnn_gru_eval_transform(288, bad_mean, bad_std)  # type: ignore[arg-type]


def test_head_forces_fp32_under_fp16_autocast() -> None:
    head = CascadedAxialBiGRUGatedAttnHead(64, proj_dim=32, hidden_dim=16)
    assert float(head.gamma.item()) == pytest.approx(0.1)

    features = torch.randn(2, 64, 9, 9, dtype=torch.float32)
    # Constant channels (var == 0) + extreme amplitude near float16 overflow threshold.
    features[:, :16, :, :] = 0.0
    features[:, 16:32, :, :] = 1e4
    features_bf16 = features.to(torch.bfloat16).detach().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = head(features_bf16)
    assert out.dtype == torch.float32
    assert out.shape == (2, 96)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert features_bf16.grad is not None and torch.isfinite(features_bf16.grad).all()
    for param in head.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()

    # Also test float16 input tensors directly (as produced by CUDA fp16 backbone).
    head.zero_grad(set_to_none=True)
    features_fp16 = features.to(torch.float16).detach().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out_fp16 = head(features_fp16)
    assert out_fp16.dtype == torch.float32
    assert torch.isfinite(out_fp16).all()
    out_fp16.sum().backward()
    assert features_fp16.grad is not None and torch.isfinite(features_fp16.grad).all()


def _torchvision_to_meta(tv_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    meta: dict[str, torch.Tensor] = {}
    for key, val in tv_state.items():
        if key.startswith("features.0."):
            meta[key.replace("features.0.", "downsample_layers.0.", 1)] = val.clone()
        elif key.startswith(("features.2.", "features.4.", "features.6.")):
            idx = int(key.split(".")[1]) // 2
            rest = key.split(".", 2)[2]
            meta[f"downsample_layers.{idx}.{rest}"] = val.clone()
        elif key.startswith(
            ("features.1.", "features.3.", "features.5.", "features.7.")
        ):
            parts = key.split(".")
            stage_idx = (int(parts[1]) - 1) // 2
            block_idx = parts[2]
            if parts[3] == "layer_scale":
                meta[f"stages.{stage_idx}.{block_idx}.gamma"] = val.reshape(-1).clone()
            else:
                sub = {"0": "dwconv", "2": "norm", "3": "pwconv1", "5": "pwconv2"}[
                    parts[4]
                ]
                meta[f"stages.{stage_idx}.{block_idx}.{sub}.{parts[5]}"] = val.clone()
        elif key.startswith("classifier.0."):
            meta[key.replace("classifier.0.", "norm.", 1)] = val.clone()
        elif key.startswith("classifier.2."):
            meta[key.replace("classifier.2.", "head.", 1)] = val.clone()
    return meta


def _torchvision_to_timm(tv_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    timm_sd: dict[str, torch.Tensor] = {}
    for key, val in tv_state.items():
        if key.startswith("features.0."):
            timm_sd[key.replace("features.0.", "stem.", 1)] = val.clone()
        elif key.startswith(("features.2.", "features.4.", "features.6.")):
            idx = int(key.split(".")[1]) // 2
            rest = key.split(".", 2)[2]
            timm_sd[f"stages.{idx}.downsample.{rest}"] = val.clone()
        elif key.startswith(
            ("features.1.", "features.3.", "features.5.", "features.7.")
        ):
            parts = key.split(".")
            stage_idx = (int(parts[1]) - 1) // 2
            block_idx = parts[2]
            if parts[3] == "layer_scale":
                timm_sd[f"stages.{stage_idx}.blocks.{block_idx}.gamma"] = val.reshape(
                    -1
                ).clone()
            else:
                sub = {
                    "0": "conv_dw",
                    "2": "norm",
                    "3": "mlp.fc1",
                    "5": "mlp.fc2",
                }[parts[4]]
                timm_sd[f"stages.{stage_idx}.blocks.{block_idx}.{sub}.{parts[5]}"] = (
                    val.clone()
                )
        elif key.startswith("classifier.0."):
            timm_sd[key.replace("classifier.0.", "head.norm.", 1)] = val.clone()
        elif key.startswith("classifier.2."):
            timm_sd[key.replace("classifier.2.", "head.fc.", 1)] = val.clone()
    return timm_sd


def test_convert_convnext_to_torchvision_maps_all_keys_and_copies_gap_norm(
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    tv_ref = models.convnext_tiny(weights=None).eval()
    ref_state = tv_ref.state_dict()
    assert len(ref_state) == 182

    meta_converted = convert_convnext_to_torchvision(_torchvision_to_meta(ref_state))
    timm_converted = convert_convnext_to_torchvision(_torchvision_to_timm(ref_state))
    assert set(meta_converted) == set(ref_state)
    assert set(timm_converted) == set(ref_state)
    assert len(meta_converted) == len(timm_converted) == 182

    tv_from_meta = models.convnext_tiny(weights=None).eval()
    tv_from_meta.load_state_dict(meta_converted, strict=True)
    tv_from_timm = models.convnext_tiny(weights=None).eval()
    tv_from_timm.load_state_dict(timm_converted, strict=True)

    probe = torch.randn(1, 3, 288, 288)
    with torch.inference_mode():
        y_ref = tv_ref(probe)
        assert (y_ref - tv_from_meta(probe)).abs().max().item() < 1e-6
        assert (y_ref - tv_from_timm(probe)).abs().max().item() < 1e-6

    ckpt_path = tmp_path / "convnext_tiny_meta.pth"
    torch.save({"model": _torchvision_to_meta(ref_state)}, ckpt_path)
    built = build_cnn_gru_model("convnext_tiny", 9, convnext_checkpoint=ckpt_path)
    assert isinstance(built.head, CascadedAxialBiGRUGatedAttnHead)
    assert built.head.gap_norm.eps == 1e-6
    torch.testing.assert_close(
        built.head.gap_norm.weight, ref_state["classifier.0.weight"], atol=0, rtol=0
    )
    torch.testing.assert_close(
        built.head.gap_norm.bias, ref_state["classifier.0.bias"], atol=0, rtol=0
    )

    with pytest.raises(ValueError, match="Unsupported ConvNeXt"):
        convert_convnext_to_torchvision({"unknown.weight": torch.zeros(1)})


def test_convnext_mixstyle_train_only_parameter_free_and_rejected_elsewhere() -> None:
    with pytest.raises(ValueError, match="ConvNeXt"):
        build_cnn_gru_model("resnet50", 9, mixstyle=True)

    torch.manual_seed(11)
    plain = build_cnn_gru_model(
        "convnext_tiny", 9, hidden_dim=16, proj_dim=32, mixstyle=False
    )
    mixed = build_cnn_gru_model(
        "convnext_tiny",
        9,
        hidden_dim=16,
        proj_dim=32,
        mixstyle=True,
        mixstyle_p=1.0,
        mixstyle_alpha=0.1,
    )
    mixed.load_state_dict(plain.state_dict(), strict=True)
    assert set(mixed.state_dict()) == set(plain.state_dict())
    assert dict(mixed.named_buffers()).keys() == dict(plain.named_buffers()).keys()

    x = torch.randn(3, 3, 64, 64)
    # MixStyle mutates training forward across batch >= 2, leaves eval & batch=1 unchanged.
    plain.eval()
    mixed.eval()
    with torch.inference_mode():
        torch.testing.assert_close(mixed(x), plain(x), atol=1e-6, rtol=1e-6)

    plain.train()
    mixed.train()
    with torch.no_grad():
        y_plain = plain.extract_features(x)
        y_mixed = mixed.extract_features(x)
        assert (y_plain - y_mixed).abs().max().item() > 1e-4
        y_plain_1 = plain.extract_features(x[:1])
        y_mixed_1 = mixed.extract_features(x[:1])
        torch.testing.assert_close(y_mixed_1, y_plain_1, atol=1e-6, rtol=1e-6)
