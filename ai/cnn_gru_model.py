"""CNN backbones with FP32 2D Axial-BiGRU + Gated Spatial Attention head."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Self

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from torchvision import models  # type: ignore[import-untyped]
from torchvision import transforms as T

from ai.data import MAX_PIXELS

SUPPORTED_BACKBONES: tuple[str, ...] = (
    "resnet50",
    "convnext_tiny",
    "convnext_small",
    "efficientnet_v2_s",
)
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
_BACKBONE_CHANNELS: dict[str, int] = {
    "resnet50": 2048,
    "convnext_tiny": 768,
    "convnext_small": 768,
    "efficientnet_v2_s": 1280,
}

_META_STAGE_RE = re.compile(
    r"^stages\.(\d+)\.(\d+)\.(gamma|dwconv|norm|pwconv1|pwconv2)(?:\.(weight|bias))?$"
)
_TIMM_STAGE_RE = re.compile(
    r"^stages\.(\d+)\.blocks\.(\d+)\.(gamma|conv_dw|norm|mlp\.fc1|mlp\.fc2)(?:\.(weight|bias))?$"
)
_META_DOWNSAMPLE_RE = re.compile(r"^downsample_layers\.(\d+)\.(\d+)\.(weight|bias)$")
_TIMM_DOWNSAMPLE_RE = re.compile(r"^stages\.(\d+)\.downsample\.(\d+)\.(weight|bias)$")


class _CnnGruEvalTransform:
    def __init__(
        self,
        size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        self.size = size
        self.mean = mean
        self.std = std
        self.fill = tuple(round(value * 255) for value in mean)
        self._to_tensor = T.Compose(
            [
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean, std),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
            raise ValueError("image must be a nonempty PIL image")
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("pixel_limit: input exceeds budget")
        with ImageOps.exif_transpose(image) as oriented:
            rgb = oriented.convert("RGB")
            try:
                with ImageOps.pad(
                    rgb,
                    (self.size, self.size),
                    method=Image.Resampling.BICUBIC,
                    color=self.fill,
                ) as padded:
                    result: torch.Tensor = self._to_tensor(padded)
            finally:
                if rgb is not oriented:
                    rgb.close()
        return result.cpu()


def cnn_gru_eval_transform(
    size: int = 288,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> Callable[[Image.Image], torch.Tensor]:
    """Deterministic EXIF-corrected RGB fit_pad bicubic transform."""
    if type(size) is not int or size < 32 or size**2 > MAX_PIXELS:
        raise ValueError("size must be an integer >= 32 within MAX_PIXELS")
    raw_mean = [] if isinstance(mean, (str, bytes)) else list(mean)
    raw_std = [] if isinstance(std, (str, bytes)) else list(std)
    if (
        len(raw_mean) != 3
        or len(raw_std) != 3
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not 0.0 <= v <= 1.0
            for v in raw_mean
        )
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not 0.0 < v <= 1.0
            for v in raw_std
        )
    ):
        raise ValueError("invalid mean or std normalization constants")
    mean_vals = (float(raw_mean[0]), float(raw_mean[1]), float(raw_mean[2]))
    std_vals = (float(raw_std[0]), float(raw_std[1]), float(raw_std[2]))
    mean_t = torch.tensor(mean_vals, dtype=torch.float32)
    std_t = torch.tensor(std_vals, dtype=torch.float32)
    bounds = (torch.tensor([[0.0], [1.0]], dtype=torch.float32) - mean_t) / std_t
    if not (
        torch.isfinite(std_t).all()
        and (std_t > 0).all()
        and torch.isfinite(bounds).all()
    ):
        raise ValueError("normalization must stay finite in float32")
    return _CnnGruEvalTransform(size, mean_vals, std_vals)


def convert_convnext_to_torchvision(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map Meta or timm ConvNeXt keys to torchvision.models.convnext_* keys."""
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("ConvNeXt checkpoint state_dict must be a nonempty mapping")
    converted: dict[str, torch.Tensor] = {}
    for raw_key, value in state_dict.items():
        if not isinstance(raw_key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("ConvNeXt state_dict must map string keys to tensors")  # noqa: TRY004
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite tensor in ConvNeXt checkpoint: {raw_key}")
        key = raw_key.removeprefix("module.")
        tensor = value.detach().clone()

        if key.startswith(("features.", "classifier.")):
            if key.endswith(".layer_scale"):
                tensor = tensor.reshape(-1, 1, 1)
            converted[key] = tensor
            continue

        if key.startswith("stem."):
            converted["features.0." + key.removeprefix("stem.")] = tensor
            continue

        if (match := _META_DOWNSAMPLE_RE.match(key)) is not None:
            stage_idx = int(match.group(1))
            layer_idx, param = match.group(2), match.group(3)
            tv_stage = 0 if stage_idx == 0 else stage_idx * 2
            converted[f"features.{tv_stage}.{layer_idx}.{param}"] = tensor
            continue

        if (match := _TIMM_DOWNSAMPLE_RE.match(key)) is not None:
            stage_idx = int(match.group(1))
            layer_idx, param = match.group(2), match.group(3)
            converted[f"features.{stage_idx * 2}.{layer_idx}.{param}"] = tensor
            continue

        if (match := _META_STAGE_RE.match(key)) is not None:
            stage_s, block_idx, sub, param = match.groups()
            tv_stage = int(stage_s) * 2 + 1
            if sub == "gamma":
                converted[f"features.{tv_stage}.{block_idx}.layer_scale"] = (
                    tensor.reshape(-1, 1, 1)
                )
            else:
                block_num = {
                    "dwconv": "0",
                    "norm": "2",
                    "pwconv1": "3",
                    "pwconv2": "5",
                }[sub]
                converted[
                    f"features.{tv_stage}.{block_idx}.block.{block_num}.{param}"
                ] = tensor
            continue

        if (match := _TIMM_STAGE_RE.match(key)) is not None:
            stage_s, block_idx, sub, param = match.groups()
            tv_stage = int(stage_s) * 2 + 1
            if sub == "gamma":
                converted[f"features.{tv_stage}.{block_idx}.layer_scale"] = (
                    tensor.reshape(-1, 1, 1)
                )
            else:
                block_num = {
                    "conv_dw": "0",
                    "norm": "2",
                    "mlp.fc1": "3",
                    "mlp.fc2": "5",
                }[sub]
                if (
                    sub in {"mlp.fc1", "mlp.fc2"}
                    and param == "weight"
                    and tensor.ndim == 4
                ):
                    if tensor.shape[2:] != (1, 1):
                        raise ValueError(
                            f"Unsupported ConvNeXt checkpoint key: {raw_key}"
                        )
                    tensor = tensor.squeeze(-1).squeeze(-1)
                converted[
                    f"features.{tv_stage}.{block_idx}.block.{block_num}.{param}"
                ] = tensor
            continue

        if key in {"norm.weight", "norm.bias", "head.norm.weight", "head.norm.bias"}:
            suffix = key.rsplit(".", 1)[1]
            converted[f"classifier.0.{suffix}"] = tensor
            continue

        if key in {"head.weight", "head.bias", "head.fc.weight", "head.fc.bias"}:
            suffix = key.rsplit(".", 1)[1]
            converted[f"classifier.2.{suffix}"] = tensor
            continue

        raise ValueError(f"Unsupported ConvNeXt checkpoint key: {raw_key}")
    return converted


class CascadedAxialBiGRUGatedAttnHead(nn.Module):
    """FP32 cascaded row->column BiGRU with Ilse et al. gated spatial attention."""

    def __init__(
        self,
        in_channels: int,
        *,
        proj_dim: int = 512,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if type(in_channels) is not int or in_channels <= 0:
            raise ValueError("in_channels must be a positive integer")
        if type(proj_dim) is not int or proj_dim <= 0:
            raise ValueError("proj_dim must be a positive integer")
        if type(hidden_dim) is not int or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        self.in_channels = in_channels
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim

        self.gap_norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.in_proj = nn.Conv2d(in_channels, proj_dim, kernel_size=1, bias=True)
        self.in_norm = nn.LayerNorm(proj_dim, eps=1e-6)
        self.row_gru = nn.GRU(
            proj_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.row_proj: nn.Module = (
            nn.Identity()
            if 2 * hidden_dim == proj_dim
            else nn.Linear(2 * hidden_dim, proj_dim)
        )
        self.row_norm = nn.LayerNorm(proj_dim, eps=1e-6)
        self.col_gru = nn.GRU(
            proj_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.col_proj: nn.Module = (
            nn.Identity()
            if 2 * hidden_dim == proj_dim
            else nn.Linear(2 * hidden_dim, proj_dim)
        )
        self.col_norm = nn.LayerNorm(proj_dim, eps=1e-6)
        self.attn_v = nn.Linear(proj_dim, hidden_dim)
        self.attn_g = nn.Linear(proj_dim, hidden_dim)
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.gru_norm = nn.LayerNorm(proj_dim, eps=1e-6)
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(features.device.type, enabled=False):
            f4 = features.float()
            gap = self.gap_norm(f4.mean(dim=(-2, -1)))
            x = self.in_proj(f4).permute(0, 2, 3, 1)
            x = F.gelu(self.in_norm(x))
            batch_size, height, width, dim = x.shape

            row_out, _ = self.row_gru(x.reshape(batch_size * height, width, dim))
            row_proj = self.row_proj(row_out).reshape(batch_size, height, width, dim)
            h_row = x + self.row_norm(row_proj)

            col_in = h_row.permute(0, 2, 1, 3).reshape(batch_size * width, height, dim)
            col_out, _ = self.col_gru(col_in)
            col_proj = (
                self.col_proj(col_out)
                .reshape(batch_size, width, height, dim)
                .permute(0, 2, 1, 3)
            )
            spatial = h_row + self.col_norm(col_proj)

            spatial_flat = spatial.reshape(batch_size, height * width, dim)
            attn_gate = torch.tanh(self.attn_v(spatial_flat)) * torch.sigmoid(
                self.attn_g(spatial_flat)
            )
            attn_weights = torch.softmax(self.attn_w(attn_gate), dim=1)
            pooled = (attn_weights * spatial_flat).sum(dim=1)
            v_gru = self.gamma * self.gru_norm(pooled)
            return torch.cat((gap, v_gru), dim=1)


class GapOnlyHead(nn.Module):
    """Ablation head matching gap_norm without spatial BiGRU or gated attention."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.gap_norm = nn.LayerNorm(in_channels, eps=1e-6)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(features.device.type, enabled=False):
            out: torch.Tensor = self.gap_norm(features.float().mean(dim=(-2, -1)))
            return out


class _MixStyle(nn.Module):
    """Parameter-free channel feature-statistics mixing active only during training."""

    def __init__(self, p: float = 0.35, alpha: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        if isinstance(p, bool) or not math.isfinite(p) or not (0.0 <= p <= 1.0):
            raise ValueError("mixstyle_p must be finite within [0, 1]")
        if isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("mixstyle_alpha must be positive and finite")
        if isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be positive and finite")
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0 or x.size(0) < 2:
            return x
        if float(torch.rand((), device="cpu").item()) >= self.p:
            return x
        with torch.amp.autocast(x.device.type, enabled=False):
            x_fp32 = x.float()
            batch = x_fp32.size(0)
            mu = x_fp32.mean(dim=(-2, -1), keepdim=True)
            var = x_fp32.var(dim=(-2, -1), unbiased=False, keepdim=True)
            sig = (var + self.eps).sqrt()
            x_normed = (x_fp32 - mu) / sig
            mu_det = mu.detach()
            sig_det = sig.detach()
            lmda = torch.distributions.Beta(self.alpha, self.alpha).sample(
                (batch, 1, 1, 1)
            )
            lmda = lmda.to(device=x_fp32.device, dtype=x_fp32.dtype)
            perm = torch.randperm(batch, device=x_fp32.device)
            mu_mix = lmda * mu_det + (1.0 - lmda) * mu_det[perm]
            sig_mix = lmda * sig_det + (1.0 - lmda) * sig_det[perm]
            mixed: torch.Tensor = (x_normed * sig_mix + mu_mix).to(dtype=x.dtype)
            return mixed


class SpatialBiGruClassifier(nn.Module):
    """Torchvision CNN backbone paired with CascadedAxialBiGRUGatedAttnHead or GAP."""

    def __init__(
        self,
        architecture: str,
        backbone: nn.Module,
        class_count: int = 9,
        *,
        use_gru_head: bool = True,
        hidden_dim: int = 256,
        proj_dim: int = 512,
        freeze_bn_eval: bool = True,
        mixstyle: bool = False,
        mixstyle_p: float = 0.35,
        mixstyle_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        if architecture not in SUPPORTED_BACKBONES:
            raise ValueError("unsupported architecture")
        if type(class_count) is not int or class_count <= 0:
            raise ValueError("class_count must be a positive integer")
        if type(use_gru_head) is not bool:
            raise ValueError("use_gru_head must be bool")
        if type(mixstyle) is not bool:
            raise ValueError("mixstyle must be bool")
        if mixstyle and not architecture.startswith("convnext"):
            raise ValueError("mixstyle is only supported for ConvNeXt backbones")
        self.architecture = architecture
        self.use_gru_head = use_gru_head
        self.freeze_bn_eval = freeze_bn_eval
        self.backbone = backbone
        self.mixstyle: _MixStyle | None = (
            _MixStyle(p=mixstyle_p, alpha=mixstyle_alpha) if mixstyle else None
        )

        in_channels = _BACKBONE_CHANNELS[architecture]
        self.head: nn.Module
        if use_gru_head:
            self.head = CascadedAxialBiGRUGatedAttnHead(
                in_channels, proj_dim=proj_dim, hidden_dim=hidden_dim
            )
            embed_dim = in_channels + proj_dim
        else:
            self.head = GapOnlyHead(in_channels)
            embed_dim = in_channels

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embed_dim, class_count),
        )

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if mode and self.freeze_bn_eval:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def extract_features(self, pixels: torch.Tensor) -> torch.Tensor:
        if self.architecture == "resnet50":
            b: Any = self.backbone
            x = b.conv1(pixels)
            x = b.bn1(x)
            x = b.relu(x)
            x = b.maxpool(x)
            x = b.layer1(x)
            x = b.layer2(x)
            x = b.layer3(x)
            out: torch.Tensor = b.layer4(x)
            return out
        if self.architecture.startswith("convnext") and self.mixstyle is not None:
            x = pixels
            for idx, layer in enumerate(self.backbone.features):  # type: ignore[arg-type]
                x = layer(x)
                if idx in (1, 3):
                    x = self.mixstyle(x)
            out_cx: torch.Tensor = x
            return out_cx
        features_fn: Any = self.backbone.features
        result: torch.Tensor = features_fn(pixels)
        return result

    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(pixels)
        emb: torch.Tensor = self.head(features)
        return emb

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        emb = self.encode_image(pixels)
        with torch.amp.autocast(emb.device.type, enabled=False):
            logits: torch.Tensor = self.classifier(emb.float())
            return logits

    def backbone_stage_modules(self) -> dict[str, list[nn.Module]]:
        """Return trainable stage modules for GĐ 2 differential learning rates."""
        b: Any = self.backbone
        if self.architecture == "resnet50":
            return {
                "stage2": [b.layer2],
                "stage3": [b.layer3],
                "stage4": [b.layer4],
            }
        if self.architecture in ("convnext_tiny", "convnext_small"):
            return {
                "stage2": [b.features[2], b.features[3]],
                "stage3": [b.features[4], b.features[5]],
                "stage4": [b.features[6], b.features[7]],
            }
        return {
            "stage2": [b.features[3]],
            "stage3": [b.features[4], b.features[5]],
            "stage4": [b.features[6], b.features[7]],
        }


def _load_state_file(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        loaded = load_file(path, device="cpu")
        return {k: v for k, v in loaded.items()}
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict):
        for container in ("model", "state_dict"):
            if container in raw and isinstance(raw[container], dict):
                raw = raw[container]
                break
    if (
        not isinstance(raw, dict)
        or not raw
        or any(
            not isinstance(k, str) or not isinstance(v, torch.Tensor)
            for k, v in raw.items()
        )
    ):
        raise ValueError("checkpoint must be a tensor state_dict")
    return raw


def _build_tv_backbone(
    architecture: str,
    pretrained_weights: str | None,
    stochastic_depth_prob: float | None,
) -> nn.Module:
    kwargs: dict[str, Any] = {}
    if stochastic_depth_prob is not None and architecture.startswith("convnext"):
        kwargs["stochastic_depth_prob"] = stochastic_depth_prob

    factory = getattr(models, architecture)
    if pretrained_weights is None:
        model: nn.Module = factory(weights=None, **kwargs)
        return model

    maybe_path = Path(pretrained_weights)
    if maybe_path.is_file():
        model = factory(weights=None, **kwargs)
        state = _load_state_file(maybe_path)
        if any(not bool(torch.isfinite(v).all()) for v in state.values()):
            raise ValueError("checkpoint weights must remain finite")
        model.load_state_dict(state, strict=True)
        return model

    valid_weights = {w.name for w in models.get_model_weights(architecture)} | {
        "DEFAULT"
    }
    if pretrained_weights not in valid_weights:
        raise ValueError(
            f"Unsupported pretrained_weights {pretrained_weights!r} for {architecture}"
        )
    weighted_model: nn.Module = factory(weights=pretrained_weights, **kwargs)
    return weighted_model


def build_cnn_gru_model(
    architecture: str,
    class_count: int = 9,
    *,
    use_gru_head: bool = True,
    hidden_dim: int = 256,
    proj_dim: int = 512,
    pretrained_weights: str | None = None,
    convnext_checkpoint: Path | None = None,
    stochastic_depth_prob: float | None = None,
    freeze_bn_eval: bool = True,
    mixstyle: bool = False,
    mixstyle_p: float = 0.35,
    mixstyle_alpha: float = 0.1,
) -> SpatialBiGruClassifier:
    """Build a torchvision CNN + SpatialBiGRU (or GAP ablation) classifier."""
    if architecture not in SUPPORTED_BACKBONES:
        raise ValueError("unsupported architecture")
    if type(class_count) is not int or class_count <= 0:
        raise ValueError("class_count must be a positive integer")
    if convnext_checkpoint is not None and not architecture.startswith("convnext"):
        raise ValueError("convnext_checkpoint is only supported for ConvNeXt backbones")
    if type(mixstyle) is not bool:
        raise ValueError("mixstyle must be bool")
    if mixstyle and not architecture.startswith("convnext"):
        raise ValueError("mixstyle is only supported for ConvNeXt backbones")

    tv_model = _build_tv_backbone(
        architecture, pretrained_weights, stochastic_depth_prob
    )
    if convnext_checkpoint is not None:
        raw_state = _load_state_file(convnext_checkpoint)
        converted = convert_convnext_to_torchvision(raw_state)
        expected = tv_model.state_dict()
        if (
            converted.keys() == expected.keys()
            and converted["classifier.2.weight"].shape
            == expected["classifier.2.weight"].shape
        ):
            tv_model.load_state_dict(converted, strict=True)
        else:
            body_expected = {
                k: v for k, v in expected.items() if not k.startswith("classifier.2.")
            }
            body_converted = {
                k: v for k, v in converted.items() if not k.startswith("classifier.2.")
            }
            if body_converted.keys() != body_expected.keys():
                raise ValueError(
                    "ConvNeXt checkpoint keys mismatch torchvision backbone"
                )
            for k, v in body_expected.items():
                if body_converted[k].shape != v.shape:
                    raise ValueError(f"ConvNeXt tensor shape mismatch for {k}")
            tv_model.load_state_dict(body_converted, strict=False)

    convnext_gap_norm_state: dict[str, torch.Tensor] | None = None
    if architecture.startswith("convnext"):
        classifier_seq = tv_model.classifier
        assert isinstance(classifier_seq, nn.Sequential)
        convnext_gap_norm_state = {
            k: v.detach().clone() for k, v in classifier_seq[0].state_dict().items()
        }

    # Drop unused 1000-way ImageNet classification linear weights from runtime bundle.
    if architecture == "resnet50":
        tv_model.avgpool = nn.Identity()
        tv_model.fc = nn.Identity()
    else:
        tv_model.avgpool = nn.Identity()
        tv_model.classifier = nn.Identity()

    model = SpatialBiGruClassifier(
        architecture,
        tv_model,
        class_count=class_count,
        use_gru_head=use_gru_head,
        hidden_dim=hidden_dim,
        proj_dim=proj_dim,
        freeze_bn_eval=freeze_bn_eval,
        mixstyle=mixstyle,
        mixstyle_p=mixstyle_p,
        mixstyle_alpha=mixstyle_alpha,
    )
    if convnext_gap_norm_state is not None:
        gap_norm: Any = model.head.gap_norm
        gap_norm.load_state_dict(convnext_gap_norm_state, strict=True)
    return model
