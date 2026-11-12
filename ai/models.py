"""Offline backbone construction; checkpoint heads must match ImageNet first."""

from pathlib import Path
from typing import cast

import torch
from torch import nn
from torchvision import models  # type: ignore[import-untyped]


def build_model(
    architecture: str, class_count: int, pretrained_path: Path | None = None
) -> nn.Module:
    if architecture not in ("efficientnet_b0", "resnet50"):
        raise ValueError("unsupported architecture")
    if type(class_count) is not int or class_count <= 0:
        raise ValueError("class_count must be a positive integer")
    model: nn.Module = (
        models.efficientnet_b0(weights=None)
        if architecture == "efficientnet_b0"
        else models.resnet50(weights=None)
    )
    if pretrained_path is not None:
        state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        if (
            not isinstance(state, dict)
            or not state
            or any(
                not isinstance(key, str) or not isinstance(value, torch.Tensor)
                for key, value in state.items()
            )
        ):
            raise ValueError("checkpoint must be a tensor state_dict")
        model.load_state_dict(state, strict=True)
        # Check after dtype conversion so finite source values cannot overflow silently.
        if any(
            not torch.isfinite(value).all() for value in model.state_dict().values()
        ):
            raise ValueError("checkpoint weights must stay finite in the model dtype")
    if architecture == "efficientnet_b0":
        classifier = cast(nn.Sequential, model.classifier)
        head = cast(nn.Linear, classifier[1])
        classifier[1] = nn.Linear(head.in_features, class_count)
    else:
        head = cast(nn.Linear, model.fc)
        model.fc = nn.Linear(head.in_features, class_count)
    return model
