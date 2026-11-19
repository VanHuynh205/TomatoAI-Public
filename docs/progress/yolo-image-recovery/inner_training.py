"""P07 group-separated development and one gate check for a refitted YOLO."""

import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Iterator
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.transforms import (  # type: ignore[import-untyped]
    ColorJitter,
    Compose,
    InterpolationMode,
    RandomAffine,
    RandomGrayscale,
    RandomResizedCrop,
)

from ai.data import ManifestDataset, _sha256
from ai.models import build_model
from ai.preprocessing import build_transform
from ai.train import _atomic_save, _validate_epoch, configure_stage, train_epoch

from .head_recovery import (  # type: ignore[import-not-found]
    HERE,
    THREADS,
    VENDOR_CHECK,
    collect,
    sample_weights,
    select_candidate,
    summarize,
)

SEED = 42
BATCH = 4
ACCUMULATION = 4
EPOCH_SAMPLES = 4096
WARM = 2
EVALUATE_AT = (2, 4, 6, 8)
GPU_LIMIT = int(3.2 * 2**30)
DEVICE = torch.device("cuda")
CONFIG = Path("ai/configs/E10.json")
DATA = Path("infra/derived/pv8-plantdoc-v1")


def inner_split(rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    components: dict[str, list[dict[str, str]]] = defaultdict(list)
    sample_ids = [r["sample_id"] for r in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("duplicate sample ID")
    for row in rows:
        components[row["component_id"]].append(row)
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for component, members in components.items():
        splits = {r["split"] for r in members}
        if "train" not in splits:
            continue
        if splits != {"train"}:
            raise ValueError("component crosses locked splits")
        keys = {(r["sample_id"].partition(":")[0], r["class_id"]) for r in members}
        if len(keys) != 1:
            raise ValueError("component spans training strata")
        strata[next(iter(keys))].append(component)
    selected = set()
    for groups in strata.values():
        if len(groups) < 2:
            raise ValueError("inner development needs two groups per stratum")
        groups.sort(key=lambda g: hashlib.sha256(f"P07:{SEED}:{g}".encode()).digest())
        selected.update(groups[: max(1, round(len(groups) / 4))])
    dev = sorted(r["sample_id"] for r in rows if r["component_id"] in selected)
    train = sorted(
        r["sample_id"]
        for r in rows
        if r["split"] == "train" and r["component_id"] not in selected
    )
    return train, dev


def loaders(
    config: dict[str, Any], train_ids: list[str], dev_ids: list[str]
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    spec = config["preprocessing"]["spec"]
    augment = Compose(
        [
            build_transform(spec, training=True),
            RandomResizedCrop(
                224,
                scale=(0.70, 1),
                ratio=(0.85, 1.17),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            RandomAffine(
                12,
                translate=(0.07, 0.07),
                interpolation=InterpolationMode.BILINEAR,
                fill=124 / 255,
            ),
            ColorJitter(0.25, 0.25, 0.20, 0.04),
            RandomGrayscale(0.08),
        ]
    )
    train = ManifestDataset(DATA, "train", augment)
    dev = ManifestDataset(DATA, "train", build_transform(spec))
    if train.class_ids != tuple(config["class_ids"]) or len(train) != 8238:
        raise ValueError("P07 dataset or class order mismatch")
    indices = {r["sample_id"]: i for i, r in enumerate(train._rows)}
    if not set(train_ids + dev_ids) <= indices.keys() or set(train_ids).intersection(
        dev_ids
    ):
        raise ValueError("P07 training/development membership mismatch")
    train_indices = [indices[sid] for sid in train_ids]
    labels = torch.tensor(
        [config["class_ids"].index(train._rows[i]["class_id"]) for i in train_indices]
    )
    weights = sample_weights(train_ids, labels, 0.25)
    sampler = WeightedRandomSampler(
        weights, EPOCH_SAMPLES, generator=torch.Generator().manual_seed(SEED)
    )
    train_subset: Subset[Any] = Subset(train, train_indices)  # type: ignore[arg-type]
    dev_subset: Subset[Any] = Subset(dev, [indices[sid] for sid in dev_ids])  # type: ignore[arg-type]
    image, _, _ = train_subset[0]
    if (
        image.shape != (3, 224, 224)
        or not torch.isfinite(image).all()
        or image.min() < 0
        or image.max() > 1
    ):
        raise ValueError("P07 train transform must preserve shape and RGB range")
    return DataLoader(train_subset, batch_size=BATCH, sampler=sampler), DataLoader(
        dev_subset, batch_size=BATCH
    )


def evaluate_inner(
    model: nn.Module, loader: DataLoader[Any], ids: list[str], classes: list[str]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    _validate_epoch(
        model, loader, device=DEVICE, class_count=len(classes), predictions=rows
    )
    if [r["sample_id"] for r in rows] != ids:
        raise ValueError("P07 development sample order mismatch")
    labels = torch.tensor([r["target_index"] for r in rows])
    logits = torch.tensor([r["logits"] for r in rows])
    metrics: dict[str, Any] = summarize({"ids": ids, "labels": labels}, logits, classes)
    return metrics


def run_training(
    config: dict[str, Any],
    train_ids: list[str],
    dev_ids: list[str],
    backbone_epochs: int,
    output: Path,
    label: str,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(SEED)
    model = build_model(
        "yolo26s_cls", len(config["class_ids"]), Path(config["pretrained_path"])
    ).to(DEVICE)
    loader, development = loaders(config, train_ids, dev_ids)
    head = list(configure_stage(model, "yolo26s_cls", "frozen")[0]["params"])
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    warm_optimizer = torch.optim.AdamW(head, lr=1e-3, weight_decay=1e-4)
    full_optimizer = torch.optim.AdamW(
        [{"params": head, "lr": 3e-4}, {"params": backbone, "lr": 5e-5}],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        full_optimizer, T_max=max(EVALUATE_AT)
    )
    scaler = torch.amp.GradScaler("cuda")
    history = []
    variants: dict[str, Any] = {}

    def checked_batches() -> Iterator[tuple[torch.Tensor, torch.Tensor, Any]]:
        for batch in loader:
            if torch.cuda.max_memory_reserved() > GPU_LIMIT:
                raise ValueError("P07 GPU budget exceeded")
            yield batch

    for epoch in range(1, WARM + backbone_epochs + 1):
        configure_stage(model, "yolo26s_cls", "frozen")
        full = epoch > WARM
        if full:
            for parameter in backbone:
                parameter.requires_grad_(True)
        row = train_epoch(
            model,
            checked_batches(),
            full_optimizer if full else warm_optimizer,
            device=DEVICE,
            accumulation=ACCUMULATION,
            amp=True,
            scaler=scaler,
        )
        if (
            torch.cuda.max_memory_reserved() > GPU_LIMIT
            or row["optimizer_steps"] + row["skipped_steps"]
            != EPOCH_SAMPLES // (BATCH * ACCUMULATION)
            or row["optimizer_steps"] < 1
            or not math.isfinite(row["loss"])
        ):
            raise ValueError("P07 training budget or optimizer update check failed")
        if full:
            scheduler.step()
        tail_epoch = epoch - WARM
        if dev_ids and tail_epoch in EVALUATE_AT:
            variants[str(tail_epoch)] = evaluate_inner(
                model, development, dev_ids, config["class_ids"]
            )
            m = variants[str(tail_epoch)]
            print(
                f"{label} development epoch {tail_epoch}: PlantDoc={m['plantdoc_f1']:.6f} PV={m['pv_f1']:.6f}",
                flush=True,
            )
        history.append(dict(epoch=epoch, phase="full" if full else "warm", **row))
        _atomic_save(output / f"{label}-history.json", history)
        if variants:
            _atomic_save(output / f"{label}-development.json", variants)
        print(label, history[-1], flush=True)
    return model, variants


def choose_epoch(variants: dict[str, Any]) -> str | None:
    return max(
        (
            k
            for k, m in variants.items()
            if m["pv_f1"] >= 0.95
            and m["plantdoc_f1"] >= 0.40
            and m["threshold"] is not None
        ),
        key=lambda k: tuple(
            variants[k][field] for field in ("plantdoc_f1", "pv_f1", "pooled_f1")
        ),
        default=None,
    )


def main() -> None:
    check = import_module(VENDOR_CHECK)
    lock_path = HERE / "p07-lock.json"
    lock = check.read_json(lock_path)
    for name, digest in lock["files_sha256"].items():
        if _sha256(Path(name)) != digest:
            raise ValueError(f"P07 input hash mismatch: {name}")
    check.verify_protocol()
    gpu = check.check_gpu()
    torch.set_num_threads(THREADS)
    config = check.read_json(CONFIG)
    assignments = check.read_json(DATA / "split.json")["assignments"]
    train_ids, dev_ids = inner_split(assignments)
    if (train_ids, dev_ids) != (lock["inner_train_ids"], lock["inner_dev_ids"]):
        raise ValueError("P07 locked inner membership mismatch")
    output = HERE / "P07"
    output.mkdir(exist_ok=False)
    print(
        f"Locked inputs verified. GPU: {gpu}. Inner train/dev: {len(train_ids)}/{len(dev_ids)}.",
        flush=True,
    )
    model, variants = run_training(
        config, train_ids, dev_ids, max(EVALUATE_AT), output, "inner"
    )
    del model
    torch.cuda.empty_cache()
    selected = choose_epoch(variants)
    _atomic_save(
        output / "inner-decision.json",
        dict(
            backbone_epochs=selected,
            p02_validation_opened=False,
            lock_sha256=_sha256(lock_path),
            gpu=gpu | {"peak_reserved_bytes": torch.cuda.max_memory_reserved()},
        ),
    )
    if selected is None:
        print("P07 did not meet inner development floors.", flush=True)
        return

    model, _ = run_training(
        config, sorted(train_ids + dev_ids), [], int(selected), output, "refit"
    )
    state = {key: value.cpu() for key, value in model.state_dict().items()}
    weights = output / "refit.pth"
    _atomic_save(weights, state)
    digest = _sha256(weights)
    model.load_state_dict(
        torch.load(weights, map_location="cpu", weights_only=True), strict=True
    )
    reservation = output / "p02-validation-reservation.json"
    with reservation.open("x", encoding="utf-8") as stream:
        json.dump({"weights_sha256": digest, "lock_sha256": _sha256(lock_path)}, stream)
    val = collect(model.eval(), config, "val", None)
    if _sha256(weights) != digest or torch.cuda.max_memory_reserved() > GPU_LIMIT:
        raise ValueError("P07 weights or GPU budget changed during validation")
    if val["ids"] != [r["sample_id"] for r in assignments if r["split"] == "val"]:
        raise ValueError("P07 locked validation membership mismatch")
    scores = output / "p02-validation-logits.json"
    _atomic_save(
        scores,
        {
            "ids": val["ids"],
            "targets": val["labels"].tolist(),
            "logits": val["logits"].tolist(),
        },
    )
    metric = summarize(val, val["logits"], config["class_ids"])
    winner = select_candidate({"p07-refit": metric}, lock["gate"])
    _atomic_save(
        output / "decision.json",
        dict(
            winner=winner,
            metric=metric,
            weights_sha256=digest,
            logits_sha256=_sha256(scores),
            lock_sha256=_sha256(lock_path),
            gpu=gpu | {"peak_reserved_bytes": torch.cuda.max_memory_reserved()},
            test_opened=False,
        ),
    )
    print(
        "P07 official gate:", winner, metric["plantdoc_f1"], metric["pv_f1"], flush=True
    )


if __name__ == "__main__":
    with TemporaryDirectory(prefix="tomato-p07-") as config_dir:
        os.environ.update(YOLO_CONFIG_DIR=config_dir, MPLCONFIGDIR=config_dir)
        main()
