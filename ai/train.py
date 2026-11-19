"""Training/validation primitives; artifact verification belongs to the runner."""

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import platform
import random
import re
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from itertools import chain, islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import models  # type: ignore[import-untyped]
from torchvision.models.efficientnet import MBConv  # type: ignore[import-untyped]
from torchvision.models.resnet import Bottleneck  # type: ignore[import-untyped]

from ai.contracts import CLASS_ID_PATTERN
from ai.data import ManifestDataset, _sha256
from ai.models import build_model
from ai.preprocessing import YOLO_PREPROCESSING, build_transform
from ai.split import verify_seal
from ai.write_lock import exclusive_writer

_FIELDS = [
    "schema_version",
    "run_id",
    "architecture",
    "stage",
    "parent_best",
    "dataset_dir",
    "seal_sha256",
    "split_hash",
    "pretrained_path",
    "pretrained_sha256",
    "train_split",
    "val_split",
    "class_ids",
    "preprocessing",
    "seed",
    "batch",
    "accumulation",
    "precision",
    "optimizer",
    "epochs",
    "patience",
]


def _local_path(value: Any, field: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or value.replace("\\", "/").startswith("//")
        or "://" in value
        or (
            re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value.strip())
            and not re.match(r"^[A-Za-z]:", value.strip())
        )
    ):
        raise ValueError(f"{field} must be a nonempty local path")


def _finite_number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _validate_training_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError("config must be an object")  # noqa: TRY004 - Config boundary uses ValueError.
    for field, expected in (("train_split", "train"), ("val_split", "val")):
        if config.get(field) != expected:
            raise ValueError(f"{field} must be {expected}")
    for field in _FIELDS:
        if field not in config:
            raise ValueError(f"missing {field}")
    unknown = set(config).difference([*_FIELDS, "sample_prefix_weights"])
    if unknown:
        raise ValueError(f"config has unknown fields: {unknown}")
    if "sample_prefix_weights" in config:
        weights = config["sample_prefix_weights"]
        if (
            not isinstance(weights, dict)
            or not 1 <= len(weights) <= 16
            or any(not isinstance(key, str) or not key for key in weights)
            or any(
                not _finite_number(value) or not 0 < value <= 1
                for value in weights.values()
            )
            or not math.isclose(sum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-12)
            or any(a != b and a.startswith(b) for a in weights for b in weights)
        ):
            raise ValueError(
                "sample_prefix_weights requires disjoint prefixes and positive shares summing to one"
            )
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    for field, choices in (
        ("architecture", ("efficientnet_b0", "resnet50", "yolo26s_cls")),
        ("stage", ("frozen", "fine_tune")),
        ("precision", ("fp32", "amp_fp16")),
    ):
        if config[field] not in choices:
            raise ValueError(f"unsupported {field}")
    if not isinstance(config["run_id"], str) or not config["run_id"].strip():
        raise ValueError("run_id must be nonempty")
    for field in ("dataset_dir", "pretrained_path"):
        _local_path(config[field], field)
    if config["stage"] == "frozen":
        if config["parent_best"] is not None:
            raise ValueError("parent_best must be null for frozen stage")
    else:
        _local_path(config["parent_best"], "parent_best")
    for field in ("seal_sha256", "split_hash", "pretrained_sha256"):
        if not isinstance(config[field], str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", config[field]
        ):
            raise ValueError(f"{field} must be a SHA-256 hex digest")
    for field in ("batch", "accumulation", "epochs", "patience", "seed"):
        minimum = 0 if field == "seed" else 1
        if type(config[field]) is not int or config[field] < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}")
    class_ids = config["class_ids"]
    if (
        not isinstance(class_ids, list)
        or not class_ids
        or any(
            not isinstance(item, str) or re.fullmatch(CLASS_ID_PATTERN, item) is None
            for item in class_ids
        )
        or len(set(class_ids)) != len(class_ids)
    ):
        raise ValueError(
            "class_ids must be unique lowercase identifiers (1-64 characters)"
        )
    optimizer = config["optimizer"]
    if (
        not isinstance(optimizer, dict)
        or set(optimizer) != {"name", "lr", "weight_decay"}
        or optimizer["name"] != "AdamW"
    ):
        raise ValueError("optimizer must define AdamW, lr and weight_decay")
    for field in ("lr", "weight_decay"):
        value = optimizer[field]
        if not _finite_number(value) or value < 0 or (field == "lr" and value == 0):
            raise ValueError(f"optimizer.{field} is invalid")
    preprocessing = config["preprocessing"]
    if (
        not isinstance(preprocessing, dict)
        or set(preprocessing) != {"profile_version", "spec"}
        or not isinstance(preprocessing["profile_version"], str)
        or not preprocessing["profile_version"].strip()
    ):
        raise ValueError("preprocessing requires profile_version and spec")
    try:
        build_transform(preprocessing["spec"], training=True)
    except ValueError as error:
        raise ValueError(f"preprocessing.spec: {error}") from error
    if config["architecture"] == "yolo26s_cls":
        if preprocessing != YOLO_PREPROCESSING:
            raise ValueError("YOLO preprocessing must match whole-frame-yolo-v1")
        expected_lr = 1e-3 if config["stage"] == "frozen" else 1e-4
        if optimizer["lr"] != expected_lr:
            raise ValueError("optimizer.lr does not match the stage policy")
        if optimizer["weight_decay"] != 1e-4:
            raise ValueError("optimizer.weight_decay does not match the YOLO policy")


def configure_stage(
    model: nn.Module, architecture: str, stage: str
) -> list[dict[str, Any]]:
    """Return fresh optimizer groups and apply modes; reapply after model.train()."""
    if architecture not in ("efficientnet_b0", "resnet50", "yolo26s_cls"):
        raise ValueError("unsupported architecture")
    if stage not in ("frozen", "fine_tune"):
        raise ValueError("unsupported stage")
    head_container: nn.Module
    tail: list[nn.Module]
    channel_pairs: list[tuple[nn.Module | None, nn.Module | None, int, int]] = []
    if architecture == "efficientnet_b0":
        if not isinstance(model, models.EfficientNet):
            raise ValueError("model must be a torchvision EfficientNet")
        classifier = getattr(model, "classifier", None)
        features = getattr(model, "features", None)
        if (
            not isinstance(classifier, nn.Sequential)
            or len(classifier) != 2
            or not isinstance(classifier[0], nn.Dropout)
            or not isinstance(classifier[1], nn.Linear)
            or classifier[1].in_features != 1280
            or classifier[1].out_features <= 0
            or not isinstance(features, nn.Sequential)
            or len(features) != 9
        ):
            raise ValueError("model has an invalid EfficientNet head or tail")
        head = classifier[1]
        head_container = classifier
        tail = list(features.children())[7:]
        if (
            not isinstance(features[7], nn.Sequential)
            or len(features[7]) != 1
            or not isinstance(features[7][0], MBConv)
        ):
            raise ValueError("model has an invalid EfficientNet tail block")
        block = getattr(features[7][0], "block", None)
        if not isinstance(block, nn.Sequential) or len(block) != 4:
            raise ValueError("model has an invalid EfficientNet tail block")
        for layer, inputs, outputs in (
            (block[0], 192, 1152),
            (block[1], 1152, 1152),
            (block[3], 1152, 320),
            (features[8], 320, 1280),
        ):
            if not isinstance(layer, nn.Sequential) or len(layer) < 2:
                raise ValueError("model has an invalid EfficientNet tail projection")
            channel_pairs.append((layer[0], layer[1], inputs, outputs))
    elif architecture == "resnet50":
        if not isinstance(model, models.ResNet):
            raise ValueError("model must be a torchvision ResNet")
        resnet_head = getattr(model, "fc", None)
        layer4 = getattr(model, "layer4", None)
        if (
            not isinstance(resnet_head, nn.Linear)
            or resnet_head.in_features != 2048
            or resnet_head.out_features <= 0
            or not isinstance(layer4, nn.Sequential)
            or len(layer4) != 3
            or any(not isinstance(residual, Bottleneck) for residual in layer4)
        ):
            raise ValueError("model has an invalid ResNet head or tail")
        head = resnet_head
        head_container = head
        tail = [layer4]
        for index, residual in enumerate(layer4):
            inputs = 1024 if index == 0 else 2048
            for number, incoming, outgoing in (
                (1, inputs, 512),
                (2, 512, 512),
                (3, 512, 2048),
            ):
                channel_pairs.append(
                    (
                        getattr(residual, f"conv{number}", None),
                        getattr(residual, f"bn{number}", None),
                        incoming,
                        outgoing,
                    )
                )
            downsample = getattr(residual, "downsample", None)
            if index == 0:
                if not isinstance(downsample, nn.Sequential) or len(downsample) != 2:
                    raise ValueError("model has an invalid ResNet tail downsample")
                channel_pairs.append((downsample[0], downsample[1], 1024, 2048))
            elif downsample is not None:
                raise ValueError("model has an invalid ResNet tail downsample")
    else:
        from ai.yolo import Yolo26Classifier

        if not isinstance(model, Yolo26Classifier):
            raise ValueError("model must be a Yolo26Classifier")
        from ultralytics.nn.modules import C2PSA, C3k2, Classify, Conv

        blocks = model.model.model
        if (
            model.model.yaml.get("scale") != "s"
            or not isinstance(blocks, nn.Sequential)
            or len(blocks) != 11
            or not isinstance(blocks[8], C3k2)
            or not isinstance(blocks[9], C2PSA)
            or not isinstance(blocks[10], Classify)
        ):
            raise ValueError("model has an invalid YOLO26s head or tail")
        classify = blocks[10]
        if (
            not isinstance(classify.linear, nn.Linear)
            or classify.linear.in_features != 1280
            or classify.linear.out_features != model.class_count
            or model.class_count <= 0
            or classify.export is not False
        ):
            raise ValueError("model has an invalid YOLO26s classifier")
        head = classify.linear
        head_container = classify
        tail = [blocks[8], blocks[9], classify.conv]
        for layer, inputs, outputs in (
            (blocks[8].cv1, 512, 512),
            (blocks[8].cv2, 768, 512),
            (blocks[9].cv1, 512, 512),
            (blocks[9].cv2, 512, 512),
            (classify.conv, 512, 1280),
        ):
            if not isinstance(layer, Conv):
                raise ValueError("model has an invalid YOLO26s tail projection")  # noqa: TRY004 - Preserve the stage validator's ValueError contract.
            channel_pairs.append((layer.conv, layer.bn, inputs, outputs))
    if any(
        not isinstance(conv, nn.Conv2d)
        or conv.in_channels != inputs
        or conv.out_channels != outputs
        or not isinstance(norm, nn.BatchNorm2d)
        or norm.num_features != outputs
        for conv, norm, inputs, outputs in channel_pairs
    ):
        raise ValueError("model has invalid backbone tail channels")
    if any(
        (
            architecture != "yolo26s_cls"
            and (not isinstance(module, nn.Sequential) or not len(module))
        )
        or not list(module.parameters())
        for module in tail
    ):
        raise ValueError("model has an invalid backbone tail")
    head_parameters = list(head.parameters())
    backbone_parameters = [p for module in tail for p in module.parameters()]
    selected = head_parameters + backbone_parameters
    if (
        not head_parameters
        or not backbone_parameters
        or len({id(p) for p in selected}) != len(selected)
    ):
        raise ValueError("model has empty or overlapping parameter groups")
    groups: list[dict[str, Any]] = [
        {
            "name": "head",
            "params": head_parameters,
            "lr": 1e-3 if stage == "frozen" else 1e-4,
        }
    ]
    if stage == "fine_tune":
        groups.append({"name": "backbone", "params": backbone_parameters, "lr": 1e-5})
    trainable = {id(p) for group in groups for p in group["params"]}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in trainable)
        parameter.grad = None
    model.eval()
    head_container.train()
    if architecture == "yolo26s_cls":
        # The adapter and Classify must agree on logits while the backbone stays frozen.
        model.training = True
    if stage == "fine_tune":
        for module in tail:
            module.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    return groups


def train_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    accumulation: int,
    amp: bool,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    """Train one streaming epoch, preserving caller-owned modes and optimizer state.

    Caller moves the model before constructing the optimizer and configures its
    stage before each epoch. Pass a persistent scaler for AMP across epochs.
    Errors clear gradients; already completed optimizer updates remain applied.
    """
    if type(accumulation) is not int or accumulation <= 0:
        raise ValueError("accumulation must be a positive integer")
    if type(amp) is not bool:
        raise ValueError("amp must be bool")
    if not isinstance(device, torch.device) or device.type not in ("cpu", "cuda"):
        raise ValueError("device must be a CPU or CUDA torch.device")
    if amp and device.type != "cuda":
        raise ValueError("amp requires a CUDA device")
    if scaler is not None and (
        not isinstance(scaler, torch.amp.GradScaler) or scaler.is_enabled() != amp
    ):
        raise ValueError("scaler enabled state must match amp")
    if amp and scaler is None:
        scaler = torch.amp.GradScaler("cuda")
        if not scaler.is_enabled():
            raise ValueError("amp requires an enabled CUDA scaler")

    def step_window(samples: int) -> bool:
        gradients = [
            parameter.grad
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        scaled_finite = all(bool(torch.isfinite(g).all()) for g in gradients)
        if amp:
            assert scaler is not None
            scaler.unscale_(optimizer)
        if scaled_finite:
            for gradient in gradients:
                gradient.div_(samples)
            if not all(bool(torch.isfinite(g).all()) for g in gradients):
                raise FloatingPointError("non-finite gradients after normalization")
        elif not amp:
            raise FloatingPointError("non-finite gradients")
        if amp:
            assert scaler is not None
            # Overflow is recorded by unscale_; step skips it and update backs off.
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return scaled_finite

    total_loss = 0.0
    total_samples = window_samples = window_batches = 0
    optimizer_steps = skipped_steps = 0
    optimizer.zero_grad(set_to_none=True)
    try:
        for inputs, targets, _sample_ids in loader:
            if (
                not isinstance(inputs, torch.Tensor)
                or not isinstance(targets, torch.Tensor)
                or inputs.ndim < 1
                or targets.ndim != 1
                or inputs.shape[0] <= 0
                or inputs.shape[0] != targets.shape[0]
            ):
                raise ValueError("inputs and targets must be nonempty matching batches")
            if (
                targets.dtype == torch.bool
                or targets.is_floating_point()
                or targets.is_complex()
            ):
                raise ValueError("targets must be integer labels")
            inputs = inputs.to(device)
            targets = targets.to(device=device, dtype=torch.long)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp
            ):
                logits = model(inputs)
                if (
                    not isinstance(logits, torch.Tensor)
                    or logits.ndim != 2
                    or logits.shape[0] != targets.shape[0]
                    or logits.shape[1] <= 0
                ):
                    raise ValueError("logits must have shape (batch, classes)")
                if bool((targets < 0).any()) or bool(
                    (targets >= logits.shape[1]).any()
                ):
                    raise ValueError("targets must be in [0, classes)")
                loss = nn.functional.cross_entropy(logits, targets, reduction="sum")
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite loss")
            total_loss += loss.item()
            total_samples += targets.shape[0]
            window_samples += targets.shape[0]
            window_batches += 1
            if amp:
                assert scaler is not None
                torch.autograd.backward(scaler.scale(loss))
            else:
                torch.autograd.backward(loss)
            if window_batches == accumulation:
                updated = step_window(window_samples)
                optimizer_steps += int(updated)
                skipped_steps += int(not updated)
                window_samples = window_batches = 0
        if total_samples == 0:
            raise ValueError("loader must not be empty")
        if window_batches:
            updated = step_window(window_samples)
            optimizer_steps += int(updated)
            skipped_steps += int(not updated)
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    return {
        "loss": total_loss / total_samples,
        "optimizer_steps": float(optimizer_steps),
        "skipped_steps": float(skipped_steps),
    }


def _validate_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor, Any]],
    *,
    device: torch.device,
    class_count: int,
    predictions: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Evaluate in FP32 or higher; leave eval mode for the caller to reconfigure."""
    if type(class_count) is not int or class_count <= 0:
        raise ValueError("class_count must be a positive integer")
    if not isinstance(device, torch.device) or device.type not in ("cpu", "cuda"):
        raise ValueError("device must be a CPU or CUDA torch.device")
    confusion = torch.zeros(class_count, class_count, dtype=torch.long)
    total_loss = 0.0
    total_samples = 0
    sample_ids: set[str] = set()
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for inputs, targets, _sample_ids in loader:
            if (
                not isinstance(inputs, torch.Tensor)
                or not isinstance(targets, torch.Tensor)
                or inputs.ndim < 1
                or targets.ndim != 1
                or inputs.shape[0] <= 0
                or inputs.shape[0] != targets.shape[0]
            ):
                raise ValueError("inputs and targets must be nonempty matching batches")
            if (
                targets.dtype == torch.bool
                or targets.is_floating_point()
                or targets.is_complex()
            ):
                raise ValueError("targets must be integer labels")
            targets = targets.to(device=device, dtype=torch.long)
            if bool((targets < 0).any()) or bool((targets >= class_count).any()):
                raise ValueError("targets must be in [0, classes)")
            logits = model(inputs.to(device))
            if (
                not isinstance(logits, torch.Tensor)
                or logits.ndim != 2
                or logits.shape != (targets.shape[0], class_count)
                or not logits.is_floating_point()
            ):
                raise ValueError(
                    "logits must have shape (batch, class_count) and be real"
                )
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("non-finite validation logits")
            if logits.dtype in (torch.float16, torch.bfloat16):
                logits = logits.float()
            if predictions is not None:
                if (
                    not isinstance(_sample_ids, (list, tuple))
                    or len(_sample_ids) != targets.shape[0]
                    or any(not isinstance(s, str) or not s.strip() for s in _sample_ids)
                    or len(set(_sample_ids)) != len(_sample_ids)
                    or sample_ids.intersection(_sample_ids)
                ):
                    raise ValueError("sample_ids must be unique nonempty strings")
                sample_ids.update(_sample_ids)
                probabilities = logits.softmax(dim=1).cpu().tolist()
                saved_logits = logits.cpu().tolist()
                for index, target, predicted, scores, values in zip(
                    _sample_ids,
                    targets.cpu().tolist(),
                    logits.argmax(dim=1).cpu().tolist(),
                    probabilities,
                    saved_logits,
                    strict=True,
                ):
                    predictions.append(
                        {
                            "sample_id": index,
                            "target_index": target,
                            "prediction_index": predicted,
                            "probabilities": scores,
                            "logits": values,
                        }
                    )
            loss = nn.functional.cross_entropy(logits, targets, reduction="sum")
            total_loss += loss.item()
            if not math.isfinite(total_loss):
                raise FloatingPointError("non-finite validation loss")
            total_samples += targets.shape[0]
            indices = targets.cpu() * class_count + logits.argmax(dim=1).cpu()
            confusion += torch.bincount(indices, minlength=class_count**2).reshape(
                class_count, class_count
            )
    if total_samples == 0:
        raise ValueError("loader must not be empty")
    correct = confusion.diag().double()
    denominators = (confusion.sum(dim=0) + confusion.sum(dim=1)).clamp_min(1)
    return {
        "loss": total_loss / total_samples,
        "accuracy": correct.sum().item() / total_samples,
        "macro_f1": (2 * correct / denominators).mean().item(),
    }


def _training_loaders(
    config: dict[str, Any],
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Build only train/val loaders; verify run artifact identities in the runner."""
    _validate_training_config(config)
    loaders: list[DataLoader[Any]] = []
    for split in ("train", "val"):
        dataset = ManifestDataset(
            Path(config["dataset_dir"]),
            split,
            build_transform(config["preprocessing"]["spec"], training=split == "train"),
        )
        if dataset.class_ids != tuple(config["class_ids"]):
            raise ValueError("class_ids must match the sealed dataset order")
        if len(dataset) == 0:
            raise ValueError(f"{split} loader must not be empty")
        generator = torch.Generator().manual_seed(config["seed"])
        sampler = None
        if split == "train" and "sample_prefix_weights" in config:
            shares = config["sample_prefix_weights"]
            sources = [
                next(
                    (
                        prefix
                        for prefix in shares
                        if row["sample_id"].startswith(prefix)
                    ),
                    "",
                )
                for row in dataset._rows
            ]
            counts = Counter(sources)
            if set(counts) != set(shares):
                raise ValueError(
                    "sample_prefix_weights must cover every training sample and source"
                )
            sampler = WeightedRandomSampler(
                [shares[source] / counts[source] for source in sources],
                num_samples=len(dataset),
                replacement=True,
                generator=generator,
            )
        loaders.append(
            DataLoader(
                dataset,  # type: ignore[arg-type]  # ManifestDataset is map-style.
                batch_size=config["batch"],
                shuffle=split == "train" and sampler is None,
                sampler=sampler,
                drop_last=False,
                num_workers=0,
                generator=generator,
            )
        )
    return loaders[0], loaders[1]


def _batch_stream(loader: Iterable[Any], count: int) -> Iterator[Any]:
    """Repeat reads, never cached tensors, and never fetch batch count+1."""
    if type(count) is not int or count <= 0:
        raise ValueError("profile batch count must be positive")
    remaining = count
    while remaining:
        emitted = 0
        for batch in islice(loader, remaining):
            emitted += 1
            yield batch
        if not emitted:
            raise ValueError("profile loader must not be empty")
        remaining -= emitted


def _timestamp(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _diagnostic_loaders(
    loaders: tuple[DataLoader[Any], DataLoader[Any]],
    config: dict[str, Any],
    smoke: bool,
    profile_steps: int | None,
) -> tuple[tuple[DataLoader[Any], DataLoader[Any]], dict[str, Any]]:
    execution: dict[str, Any] = {
        "mode": "smoke" if smoke else "profile",
        "profile_steps": profile_steps,
        "selection": "seed-id-sha256-round-robin-v1",
        "train_sample_ids": None,
    }
    selected = []
    for split, loader in zip(("train", "val"), loaders, strict=True):
        if split == "train" and not smoke:
            selected.append(loader)
            continue
        dataset = loader.dataset
        if not isinstance(dataset, ManifestDataset):
            raise TypeError("diagnostic subsets require ManifestDataset")
        # Metadata only: preserve the loader's row indices without decoding any image.
        groups: list[list[int]] = []
        for class_id in config["class_ids"]:
            indices = [
                index
                for index, row in enumerate(dataset._rows)
                if row["class_id"] == class_id
            ]
            indices.sort(
                key=lambda index: hashlib.sha256(
                    f"{config['seed']}:{dataset._rows[index]['sample_id']}".encode()
                ).digest()
            )
            groups.append(indices)
        indices = []
        for offset in range(max(map(len, groups))):
            for group in groups:
                if offset < len(group):
                    indices.append(group[offset])
                    if len(indices) == 32:
                        break
            if len(indices) == 32:
                break
        execution[f"{split}_sample_ids"] = [
            dataset._rows[index]["sample_id"] for index in indices
        ]
        selected.append(
            DataLoader(
                Subset(dataset, indices),
                batch_size=config["batch"],
                shuffle=split == "train",
                num_workers=0,
                drop_last=False,
                generator=loader.generator,
            )
        )
    return (selected[0], selected[1]), execution


def _training_epochs(
    model: nn.Module,
    train_loader: Iterable[tuple[torch.Tensor, torch.Tensor, Any]],
    val_loader: Iterable[tuple[torch.Tensor, torch.Tensor, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    config: dict[str, Any],
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    start_epoch: int = 0,
    best: tuple[float, float] | None = None,
    bad_epochs: int = 0,
    measure: bool = False,
    profile_steps: int | None = None,
    phase_timing: dict[str, float] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield JSON-ready epoch logs before the next update, allowing M4.5 to save.

    Caller owns model/device/optimizer and persists each yielded record plus any
    checkpoint before advancing. Epochs are one-based; exact ties keep earlier best.
    """
    _validate_training_config(config)
    if not isinstance(device, torch.device) or device.type not in ("cpu", "cuda"):
        raise ValueError("device must be a CPU or CUDA torch.device")
    amp = config["precision"] == "amp_fp16"
    if amp and device.type != "cuda":
        raise ValueError("amp requires a CUDA device")
    if amp and scaler is None:
        scaler = torch.amp.GradScaler("cuda")
    if bad_epochs >= config["patience"]:
        return
    end_epoch = 1 if profile_steps is not None else config["epochs"]
    timed = measure or phase_timing is not None
    for epoch in range(start_epoch + 1, end_epoch + 1):
        configure_stage(model, config["architecture"], config["stage"])
        work: dict[str, Any] = {
            "physical_mini_batches": 0,
            "samples": 0,
            "sample_ids": [],
        }

        def observed(work: dict[str, Any] = work) -> Iterator[Any]:
            batches = (
                _batch_stream(train_loader, profile_steps)
                if profile_steps is not None
                else iter(train_loader)
            )
            for batch in batches:
                work["physical_mini_batches"] += 1
                work["samples"] += batch[0].shape[0]
                work["sample_ids"].extend(batch[2])
                yield batch

        started = _timestamp(device) if timed else 0.0
        train = train_epoch(
            model,
            observed() if measure else train_loader,
            optimizer,
            device=device,
            accumulation=config["accumulation"],
            amp=amp,
            scaler=scaler,
        )
        if any(not math.isfinite(value) for value in train.values()):
            raise FloatingPointError("non-finite training metrics")
        trained_at = _timestamp(device) if timed else 0.0
        val = _validate_epoch(
            model, val_loader, device=device, class_count=len(config["class_ids"])
        )
        candidate = (val["macro_f1"], -val["loss"])
        is_best = best is None or candidate > best
        if is_best:
            best = candidate
            bad_epochs = 0
        else:
            bad_epochs += 1
        record = {
            "epoch": epoch,
            "train": train,
            "val": val,
            "is_best": is_best,
            "bad_epochs": bad_epochs,
        }
        if timed:
            phases = {
                "train_seconds": trained_at - started,
                "validation_seconds": _timestamp(device) - trained_at,
            }
            if measure:
                work.update(phases)
                record["work"] = work
            if phase_timing is not None:
                phase_timing.update(phases)
        yield record
        if bad_epochs >= config["patience"]:
            break


def _atomic_save(path: Path, value: Any) -> None:
    """Replace one file only after serialization, flush and fsync succeed."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            if path.suffix == ".json":
                stream.write(
                    json.dumps(
                        value, ensure_ascii=False, allow_nan=False, indent=2
                    ).encode("utf-8")
                )
            else:
                torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _training_identity(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["dataset_dir"])
    identity: dict[str, Any] = {}
    for field, path in (
        ("split_hash", root / "split.json"),
        ("seal_sha256", root / "seal.json"),
        ("pretrained_sha256", Path(config["pretrained_path"])),
    ):
        if _sha256(path) != config[field].lower():
            raise ValueError(f"{field} mismatch")
        identity[field] = config[field].lower()
    verify_seal(root)
    identity["parent_best_sha256"] = (
        _sha256(Path(config["parent_best"])) if config["parent_best"] else None
    )
    return identity


def _finite_checkpoint(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("checkpoint contains non-finite tensors")
    elif isinstance(value, dict):
        for item in value.values():
            _finite_checkpoint(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_checkpoint(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("checkpoint contains non-finite numbers")
    elif value is not None and type(value) not in (str, int, float, bool):
        raise ValueError("checkpoint must contain tensors/primitives only")


def _load_checkpoint(path: Path, kind: str) -> dict[str, Any]:
    # A fine-tuned R50 plus Adam moments and selected best fits this local-run budget.
    if not 0 < path.stat().st_size <= 512_000_000:
        raise ValueError("checkpoint exceeds size limit")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, EOFError) as error:
        raise ValueError("invalid safe checkpoint") from error
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("kind") != kind
    ):
        raise ValueError(f"checkpoint must be a schema 1 {kind} artifact")
    fields = {"schema_version", "kind", "config", "identity", "epoch", "model"}
    fields |= (
        {
            "optimizer",
            "scaler",
            "rng",
            "best",
            "history",
            "environment",
            "source_hashes",
        }
        if kind == "last"
        else {"val"}
    )
    if set(value) != fields:
        raise ValueError("checkpoint has missing or unknown fields")
    _finite_checkpoint(value)
    return value


def _check_model_state(model: nn.Module, state: Any) -> None:
    expected = model.state_dict()
    if not isinstance(state, dict) or state.keys() != expected.keys():
        raise ValueError("checkpoint model keys mismatch")
    for key, tensor in expected.items():
        saved = state[key]
        if (
            not isinstance(saved, torch.Tensor)
            or saved.shape != tensor.shape
            or saved.dtype != tensor.dtype
        ):
            raise ValueError(f"checkpoint model tensor mismatch: {key}")


def _check_optimizer_state(
    optimizer: torch.optim.Optimizer, saved: Any, history: list[dict[str, Any]]
) -> None:
    expected = optimizer.state_dict()
    if (
        not isinstance(saved, dict)
        or set(saved) != {"state", "param_groups"}
        or saved["param_groups"] != expected["param_groups"]
    ):
        raise ValueError("checkpoint optimizer groups mismatch")
    updates = sum(record["train"]["optimizer_steps"] for record in history)
    parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    ids = [index for group in expected["param_groups"] for index in group["params"]]
    if set(saved["state"]) != (set(ids) if updates else set()):
        raise ValueError("checkpoint optimizer state missing")
    for index, parameter in zip(ids, parameters, strict=True):
        if not updates:
            continue
        state = saved["state"][index]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("checkpoint optimizer moments missing")
        if (
            not isinstance(state["step"], torch.Tensor)
            or state["step"].numel() != 1
            or state["step"].item() != updates
        ):
            raise ValueError("checkpoint optimizer step mismatch")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = state[name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != parameter.shape
                or tensor.dtype != parameter.dtype
            ):
                raise ValueError("checkpoint optimizer moment shape/dtype mismatch")


def _check_scaler_state(state: Any) -> None:
    if not isinstance(state, dict) or set(state) != {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    }:
        raise ValueError("checkpoint scaler fields mismatch")
    numbers = [state[key] for key in ("scale", "growth_factor", "backoff_factor")]
    if not all(_finite_number(value) for value in numbers):
        raise ValueError("checkpoint scaler numbers must be finite")
    values = torch.tensor(numbers, dtype=torch.float32)
    if not bool(torch.isfinite(values).all()) or not (
        values[0] > 0 and values[1] > 1 and 0 < values[2] < 1
    ):
        raise ValueError("checkpoint scaler scale/growth/backoff invalid")
    interval, tracker = state["growth_interval"], state["_growth_tracker"]
    if (
        type(interval) is not int
        or not 1 <= interval <= 2**31 - 1
        or type(tracker) is not int
        or not 0 <= tracker < interval
    ):
        raise ValueError("checkpoint scaler interval/tracker invalid")


def _rng_state(loaders: tuple[DataLoader[Any], DataLoader[Any]]) -> dict[str, Any]:
    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loaders": [
            loader.generator.get_state()
            for loader in loaders
            if loader.generator is not None
        ],
    }


def _restore_rng(
    state: dict[str, Any], loaders: tuple[DataLoader[Any], DataLoader[Any]]
) -> None:
    if (
        set(state) != {"python", "numpy", "torch", "cuda", "loaders"}
        or len(state["loaders"]) != 2
    ):
        raise ValueError("invalid RNG state")
    if len(state["cuda"]) != (
        torch.cuda.device_count() if torch.cuda.is_available() else 0
    ):
        raise ValueError("RNG CUDA device count changed")
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    for loader, saved in zip(loaders, state["loaders"], strict=True):
        if loader.generator is None:
            raise ValueError("loader RNG generator is missing")
        loader.generator.set_state(saved)


def _history_selection(
    history: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[tuple[float, float] | None, int, int]:
    best: tuple[float, float] | None = None
    bad_epochs, best_epoch = 0, 0
    if len(history) > config["epochs"]:
        raise ValueError("invalid checkpoint history")
    for epoch, record in enumerate(history, 1):
        val = record["val"]
        if (
            record["epoch"] != epoch
            or bad_epochs >= config["patience"]
            or not (
                0 <= val["macro_f1"] <= 1
                and 0 <= val["accuracy"] <= 1
                and val["loss"] >= 0
            )
        ):
            raise ValueError("invalid checkpoint epoch/metrics")
        candidate = (val["macro_f1"], -val["loss"])
        improved = best is None or candidate > best
        bad_epochs = 0 if improved else bad_epochs + 1
        if improved:
            best, best_epoch = candidate, epoch
        if record["is_best"] is not improved or record["bad_epochs"] != bad_epochs:
            raise ValueError("invalid checkpoint selection")
    return best, bad_epochs, best_epoch


def _publish_training(run_dir: Path, checkpoint: dict[str, Any]) -> Path:
    """Derived files can be rebuilt from the last committed epoch after interruption."""
    best = run_dir / "best.pth"
    selected = checkpoint["best"]
    if selected:
        _atomic_save(
            best,
            {
                "schema_version": 1,
                "kind": "best",
                "config": checkpoint["config"],
                "identity": checkpoint["identity"],
                **selected,
            },
        )
    _atomic_save(
        run_dir / "metadata.json",
        {
            "schema_version": 1,
            "config": checkpoint["config"],
            "identity": checkpoint["identity"],
            "epoch": checkpoint["epoch"],
            "best_epoch": selected.get("epoch"),
            "history": checkpoint["history"],
            "environment": checkpoint["environment"],
            "source_hashes": checkpoint["source_hashes"],
            "checkpoint_hashes": {
                name: _sha256(run_dir / name)
                for name in (("last.pth", "best.pth") if selected else ("last.pth",))
            },
        },
    )
    return best


def run_training(
    config: dict[str, Any],
    resume: Path | None = None,
    *,
    out: Path | None = None,
    smoke: bool = False,
    profile_steps: int | None = None,
    epoch_timing: bool = False,
) -> Path:
    """Train into ai/runs/<run_id>; resume last.pth at a completed epoch boundary."""
    started = time.perf_counter()
    if (
        type(smoke) is not bool
        or type(epoch_timing) is not bool
        or (
            profile_steps is not None
            and (type(profile_steps) is not int or profile_steps <= 0)
        )
    ):
        raise ValueError("smoke/profile/timing options invalid")
    if profile_steps is not None and (smoke or resume is not None):
        raise ValueError("profile cannot combine with smoke or resume")
    _validate_training_config(config)
    for name, path in (("out", out), ("resume", resume)):
        if path is not None:
            _local_path(str(path), name)
    config = copy.deepcopy(config)
    run_id: str = config["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id) or re.fullmatch(
        r"(?i:CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])", run_id
    ):
        raise ValueError("run_id must be a safe directory name")
    expected_lr = 1e-3 if config["stage"] == "frozen" else 1e-4
    if config["optimizer"]["lr"] != expected_lr:
        raise ValueError("optimizer.lr does not match the stage policy")
    run_root = Path("ai/runs").absolute()
    run_dir = Path(os.path.abspath(out)) if out is not None else run_root / run_id
    if run_dir.resolve() != run_dir:
        raise ValueError("run_id output must not traverse symlinks or junctions")
    if run_dir.is_relative_to(Path(config["dataset_dir"]).resolve()):
        raise ValueError("output must not be inside dataset_dir")
    if resume is not None and resume.resolve() != run_dir / "last.pth":
        raise ValueError("resume must reference this run's last.pth")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_writer(run_dir.with_name(f".{run_dir.name}.train.lock")):
        if resume is None and run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"run output already exists: {run_dir}")
        return _train_in_run(
            config, resume, run_dir, started, smoke, profile_steps, epoch_timing
        )


def _train_in_run(
    config: dict[str, Any],
    resume: Path | None,
    run_dir: Path,
    started: float,
    smoke: bool,
    profile_steps: int | None,
    epoch_timing: bool,
) -> Path:
    measure = smoke or profile_steps is not None
    checkpoint = _load_checkpoint(resume, "last") if resume is not None else None
    if checkpoint is not None:
        saved_config = checkpoint.get("config")
        if not isinstance(saved_config, dict):
            raise ValueError("checkpoint config missing")
        for field, value in config.items():
            if saved_config.get(field) != value:
                raise ValueError(f"resume config changed: {field}")
        if set(saved_config) != set(config):
            raise ValueError("resume config fields changed")
    identity = _training_identity(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    environment = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": device.type,
    }
    source_hashes = {
        path.name: _sha256(path) for path in Path(__file__).parent.glob("*.py")
    }
    if checkpoint is not None and (
        checkpoint["environment"] != environment
        or checkpoint["source_hashes"] != source_hashes
    ):
        raise ValueError("resume execution provenance changed (source/environment)")
    if config["precision"] == "amp_fp16" and device.type != "cuda":
        raise ValueError("amp requires a CUDA device")
    random.seed(config["seed"])
    np.random.seed(config["seed"] % 2**32)
    torch.manual_seed(config["seed"])
    loaders = _training_loaders(config)
    if isinstance(loaders[0].dataset, ManifestDataset) and run_dir.is_relative_to(
        loaders[0].dataset.source_root
    ):
        raise ValueError("output must not be inside source_root")
    if measure:
        loaders, execution = _diagnostic_loaders(loaders, config, smoke, profile_steps)
        identity["execution"] = execution
    if checkpoint is not None and checkpoint.get("identity") != identity:
        raise ValueError("resume data/parent_best identity or execution changed")
    pretrained = (
        Path(config["pretrained_path"])
        if checkpoint is None and config["stage"] == "frozen"
        else None
    )
    model = build_model(config["architecture"], len(config["class_ids"]), pretrained)
    if checkpoint is None and config["stage"] == "fine_tune":
        parent = _load_checkpoint(Path(config["parent_best"]), "best")
        if parent["identity"].get("execution") and not measure:
            raise ValueError("diagnostic best cannot initialize a full training run")
        for field in (
            "architecture",
            "class_ids",
            "dataset_dir",
            "seal_sha256",
            "split_hash",
            "pretrained_sha256",
            "preprocessing",
            "seed",
        ):
            if parent["config"].get(field) != config[field]:
                raise ValueError(f"parent_best mismatch: {field}")
        if parent["config"].get("stage") != "frozen":
            raise ValueError("parent_best must be a frozen run")
        if parent["config"].get("sample_prefix_weights") != config.get(
            "sample_prefix_weights"
        ):
            raise ValueError("parent_best mismatch: sample_prefix_weights")
        _check_model_state(model, parent["model"])
        model.load_state_dict(parent["model"], strict=True)
    if checkpoint is not None:
        _check_model_state(model, checkpoint["model"])
        if checkpoint["best"]:
            _check_model_state(model, checkpoint["best"].get("model"))
        model.load_state_dict(checkpoint["model"], strict=True)
    _finite_checkpoint(model.state_dict())
    model.to(device)
    groups = configure_stage(model, config["architecture"], config["stage"])
    optimizer = torch.optim.AdamW(
        groups, weight_decay=config["optimizer"]["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda") if config["precision"] == "amp_fp16" else None
    history: list[dict[str, Any]] = []
    best_metrics: tuple[float, float] | None = None
    bad_epochs, start_epoch = 0, 0
    best_state: dict[str, Any] = {}
    if checkpoint is not None:
        try:
            history = checkpoint["history"]
            best_metrics, bad_epochs, best_epoch = _history_selection(history, config)
            start_epoch = checkpoint["epoch"]
            best_state = checkpoint["best"]
            if (
                start_epoch != len(history)
                or (not history and best_state != {})
                or (
                    history
                    and (
                        best_state["epoch"] != best_epoch
                        or best_state["val"] != history[best_epoch - 1]["val"]
                    )
                )
            ):
                raise ValueError("checkpoint epoch/best mismatch")
            _check_optimizer_state(optimizer, checkpoint["optimizer"], history)
            optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None:
                _check_scaler_state(checkpoint["scaler"])
                scaler.load_state_dict(checkpoint["scaler"])
            elif checkpoint["scaler"] is not None:
                raise ValueError("unexpected scaler state")
            _restore_rng(checkpoint["rng"], loaders)
        except (KeyError, TypeError, RuntimeError, IndexError) as error:
            raise ValueError(f"invalid resume state: {error}") from error
        _publish_training(run_dir, checkpoint)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
    initialization_seconds = _timestamp(device) - started if measure else 0.0
    if (measure or epoch_timing) and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    save_seconds = 0.0
    phase_timing: dict[str, float] | None = {} if epoch_timing else None
    records: Iterable[dict[str, Any] | None] = _training_epochs(
        model,
        *loaders,
        optimizer,
        config=config,
        device=device,
        scaler=scaler,
        start_epoch=start_epoch,
        best=best_metrics,
        bad_epochs=bad_epochs,
        measure=measure,
        profile_steps=profile_steps,
        phase_timing=phase_timing,
    )
    if checkpoint is None or (
        measure
        and (start_epoch >= config["epochs"] or bad_epochs >= config["patience"])
    ):
        # Epoch zero makes even the first interrupted epoch restartable.
        # Completed diagnostic resumes also publish zero work for this invocation.
        records = chain((None,), records)
    for record in records:
        saving_at = _timestamp(device) if measure or epoch_timing else 0.0
        if record is not None:
            history.append(record)
        state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        if record is not None and record["is_best"]:
            best_state = {
                "epoch": record["epoch"],
                "val": record["val"],
                "model": state,
            }
        checkpoint = {
            "schema_version": 1,
            "kind": "last",
            "config": config,
            "identity": identity,
            "epoch": len(history),
            "history": history,
            "model": state,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "rng": _rng_state(loaders),
            "best": best_state,
            "environment": environment,
            "source_hashes": source_hashes,
        }
        _finite_checkpoint(checkpoint)
        if scaler is not None:
            _check_scaler_state(checkpoint["scaler"])
        _atomic_save(run_dir / "last.pth", checkpoint)
        _publish_training(run_dir, checkpoint)
        if phase_timing is not None and record is not None:
            print(
                json.dumps(
                    {
                        "event": "epoch_committed",
                        "epoch": record["epoch"],
                        **phase_timing,
                        "checkpoint_seconds": _timestamp(device) - saving_at,
                        "train": record["train"],
                        "val": record["val"],
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
                        if device.type == "cuda"
                        else None,
                    }
                ),
                flush=True,
            )
        if measure:
            save_seconds += _timestamp(device) - saving_at
            invocation = history[start_epoch:]
            performance = {
                "schema_version": 1,
                "mode": identity["execution"]["mode"],
                "config": config,
                "identity": identity,
                "invocation_start_epoch": start_epoch,
                "completed_epoch": len(history),
                "requested_mini_batches": profile_steps,
                "warmup_mini_batches": 0,
                "physical_mini_batches": sum(
                    item["work"]["physical_mini_batches"] for item in invocation
                ),
                "samples": sum(item["work"]["samples"] for item in invocation),
                "optimizer_updates": sum(
                    item["train"]["optimizer_steps"] for item in invocation
                ),
                "skipped_updates": sum(
                    item["train"]["skipped_steps"] for item in invocation
                ),
                "initialization_seconds": initialization_seconds,
                "train_seconds": sum(
                    item["work"]["train_seconds"] for item in invocation
                ),
                "validation_seconds": sum(
                    item["work"]["validation_seconds"] for item in invocation
                ),
                "save_seconds": save_seconds,
                "wall_seconds": _timestamp(device) - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
                if device.type == "cuda"
                else None,
                "device": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "cpu",
                "timing_scope": "this invocation; cold batches; train includes loader; save includes checkpoint preparation; peaks cover train/validation/save after setup; wall excludes final performance JSON write",
            }
            _atomic_save(run_dir / "performance.json", performance)
            if record is not None:
                print(
                    json.dumps(
                        {
                            "event": "epoch_committed",
                            "epoch": record["epoch"],
                            "train": record["train"],
                            "val": record["val"],
                            "physical_mini_batches": record["work"][
                                "physical_mini_batches"
                            ],
                        }
                    ),
                    flush=True,
                )
    return run_dir / "best.pth"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline train/validation runner and diagnostic smoke/profile"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--epoch-timing",
        action="store_true",
        help="report timing after each committed epoch",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument(
        "--profile-steps",
        type=int,
        help="physical mini-batches; separate diagnostic run",
    )
    args = parser.parse_args(argv)
    try:
        _local_path(str(args.config), "config")
        if args.config.stat().st_size > 1_000_000:
            raise ValueError("config exceeds size limit")
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        best = run_training(
            config,
            args.resume,
            out=args.out,
            smoke=args.smoke,
            profile_steps=args.profile_steps,
            epoch_timing=args.epoch_timing,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"event": "completed", "best": str(best)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
