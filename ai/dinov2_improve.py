"""Adapt a DINOv2 export on development data and predict a sealed test manifest."""

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms as T

from ai.dinov2_field import BACKBONE, NEGATIVE, REVISION, Classifier, predict, summarize
from ai.dinov3_field import (
    CLASSES,
    LeafImages,
    leaf_transform,
    load_manifest,
    save_atomic,
)

DOMAIN_MASS = {"field": 0.75, "web": 0.10, "controlled": 0.15}
FIELD_SOURCES = ("rachin", "plantcity", "ghana_ccmt")
SELECTION_WEIGHTS = {"field_mean": 0.75, "field_min": 0.15, "negative_recall": 0.10}
HEAD_LR, BACKBONE_LR, LAYER_DECAY = 1.5e-4, 1.2e-5, 0.75


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_rows(root: Path, manifest: Path, splits: set[str]) -> list[dict[str, str]]:
    rows = load_manifest(root, str(manifest.resolve()))
    if {row["split"] for row in rows} != splits:
        raise ValueError(f"Expected only splits {sorted(splits)} in {manifest}")
    for key in ("group", "sha256"):
        membership: dict[str, str] = {}
        for row in rows:
            value, split = row[key], row["split"]
            if not value or membership.setdefault(value, split) != split:
                raise ValueError(f"Missing or cross-split {key}: {value}")
    for row in rows:
        if not row["source"] or not row["domain"]:
            raise ValueError(f"Missing source or domain: {row['path']}")
        if digest(root / row["path"]) != row["sha256"]:
            raise ValueError(f"Image digest differs from manifest: {row['path']}")
    return rows


def sampling_weights(rows: list[dict[str, str]]) -> list[float]:
    """Distribute equal class mass over available domains, sources and groups."""
    keys = [
        (
            row["class_id"],
            row["domain"] if row["class_id"] != NEGATIVE else "negative",
            row["source"],
            row["group"],
        )
        for row in rows
    ]
    group_sizes = Counter(keys)
    source_groups = Counter(
        (label, domain, source) for label, domain, source, _ in group_sizes
    )
    source_mass = {
        key: 1.5 if key[1:3] == ("field", "rachin") else 1.0 for key in source_groups
    }
    domain_totals: defaultdict[tuple[str, str], float] = defaultdict(float)
    for (label, domain, _), mass in source_mass.items():
        domain_totals[label, domain] += mass
    domain_mass = {
        (label, domain): 1.0 if label == NEGATIVE else DOMAIN_MASS[domain]
        for label, domain in domain_totals
    }
    class_totals: defaultdict[str, float] = defaultdict(float)
    for (label, _), mass in domain_mass.items():
        class_totals[label] += mass
    return [
        domain_mass[label, domain]
        / class_totals[label]
        * source_mass[label, domain, source]
        / domain_totals[label, domain]
        / source_groups[label, domain, source]
        / group_sizes[label, domain, source, group]
        for label, domain, source, group in keys
    ]


def selection_score(rows: list[dict[str, str]], scores: torch.Tensor) -> dict[str, Any]:
    labels = [CLASSES[index] for index in scores.argmax(1).tolist()]
    report = summarize(rows, labels)
    field_rows = [
        (row, label)
        for row, label in zip(rows, labels, strict=True)
        if row["domain"] == "field" and row["class_id"] != NEGATIVE
    ]
    by_source = summarize(
        [row for row, _ in field_rows], [label for _, label in field_rows]
    )["by_source"]
    source_f1 = [by_source[source]["macro_f1"] for source in FIELD_SOURCES]
    parts = {
        "field_mean": sum(source_f1) / len(source_f1),
        "field_min": min(source_f1),
        "negative_recall": report["not_tomato"]["accuracy"],
    }
    return {
        "validation": report,
        "field_by_source": by_source,
        **parts,
        "selection": sum(
            SELECTION_WEIGHTS[key] * value for key, value in parts.items()
        ),
    }


def load_export(path: Path, device: torch.device) -> tuple[Classifier, dict[str, Any]]:
    from safetensors.torch import load_file
    from transformers import AutoModel

    metadata = json.loads((path / "leaf-classifier.json").read_text(encoding="utf-8"))
    size = metadata["size"]
    preprocessing = metadata["preprocessing"]
    if (
        metadata["backbone"] != BACKBONE
        or metadata["revision"] != REVISION
        or tuple(metadata["class_ids"]) != CLASSES
        or type(size) is not int
        or size < 196
        or size % 14
        or preprocessing["geometry"] != "fit_pad"
        or preprocessing["interpolation"] != "bicubic"
        or metadata.get("test_time_horizontal_flip") is not True
    ):
        raise ValueError(f"Incompatible DINOv2 export: {path}")
    backbone, info = AutoModel.from_pretrained(
        path,
        use_safetensors=True,
        local_files_only=True,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    if any(info.values()):
        raise ValueError(f"DINOv2 backbone weights did not load exactly: {info}")
    model = Classifier(backbone)
    model.head.load_state_dict(load_file(path / "classifier.safetensors"))
    return model.to(device), metadata


def train_transform(size: int, mean: list[float], std: list[float]) -> T.Compose:
    return T.Compose(
        [
            T.RandomApply([T.ColorJitter(0.18, 0.16, 0.14, 0.025)], p=0.75),
            leaf_transform(size, mean, std, False),
            T.RandomHorizontalFlip(),
            T.RandomAffine(
                14,
                translate=(0.04, 0.04),
                scale=(0.94, 1.06),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.RandomApply(
                [
                    T.RandomResizedCrop(
                        size,
                        scale=(0.88, 1.0),
                        ratio=(0.95, 1.05),
                        interpolation=T.InterpolationMode.BICUBIC,
                    ),
                ],
                p=0.35,
            ),
            T.RandomErasing(p=0.12, scale=(0.01, 0.035), ratio=(0.5, 2.0)),
        ]
    )


def validation_loader(
    root: Path,
    rows: list[dict[str, str]],
    size: int,
    preprocessing: dict[str, Any],
    batch: int,
    workers: int,
    device: torch.device,
) -> DataLoader[Any]:
    transform = leaf_transform(size, preprocessing["mean"], preprocessing["std"], False)
    return DataLoader(
        LeafImages(root, rows, transform),
        batch_size=batch,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        shuffle=False,
    )


def write_json(path: Path, data: Any) -> None:
    pending = path.with_suffix(".pending")
    pending.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    pending.replace(path)


def train(args: argparse.Namespace) -> None:
    from safetensors.torch import save_file

    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab")
    if not 1 <= args.epochs <= 6 or args.size < 196 or args.size % 14:
        raise ValueError(
            "Use 1 to 6 epochs and an image size at least 196 divisible by 14"
        )
    if args.initial.resolve().is_relative_to(args.output.resolve()):
        raise ValueError("Use a run directory separate from the initial export")
    manifest_hash = digest(args.development)
    rows = read_rows(args.data, args.development, {"train", "val"})
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [
        row
        for row in rows
        if row["split"] == "val"
        and (row["domain"] == "field" or row["class_id"] == NEGATIVE)
    ]
    if {row["class_id"] for row in train_rows} != set(CLASSES):
        raise ValueError("Training data must cover all nine classes")
    if any(
        row["domain"] not in DOMAIN_MASS for row in rows if row["class_id"] != NEGATIVE
    ):
        raise ValueError("Unknown tomato image domain in development data")
    if not set(FIELD_SOURCES).issubset(
        row["source"]
        for row in val_rows
        if row["domain"] == "field" and row["class_id"] != NEGATIVE
    ) or not any(row["class_id"] == NEGATIVE for row in val_rows):
        raise ValueError(
            "Validation needs field tomatoes from each selection source and negatives"
        )

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    model, initial = load_export(args.initial, device)
    backbone = model.backbone
    preprocessing = initial["preprocessing"]
    run = {
        "backbone": BACKBONE,
        "revision": REVISION,
        "class_ids": list(CLASSES),
        "size": args.size,
        "epochs": args.epochs,
        "samples": args.samples,
        "batch": args.batch,
        "workers": args.workers,
        "seed": args.seed,
        "preprocessing": preprocessing,
        "test_time_horizontal_flip": True,
        "development_sha256": manifest_hash,
        "initial_sha256": {
            name: digest(args.initial / name)
            for name in (
                "model.safetensors",
                "classifier.safetensors",
                "config.json",
                "leaf-classifier.json",
                "preprocessor_config.json",
            )
        },
        "domain_mass": DOMAIN_MASS,
        "rachin_source_mass": 1.5,
        "selection_weights": SELECTION_WEIGHTS,
        "field_sources": list(FIELD_SOURCES),
        "optimizer": {
            "name": "AdamW",
            "head_lr": HEAD_LR,
            "last_block_lr": BACKBONE_LR,
            "layer_decay": LAYER_DECAY,
            "weight_decay": 0.05,
            "label_smoothing": 0.03,
            "head_only_epochs": 1,
            "last_blocks": 4,
            "cosine_floor": 0.2,
        },
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
    }
    blocks = list(backbone.encoder.layer[-4:])
    rates = [HEAD_LR, BACKBONE_LR] + [
        BACKBONE_LR * LAYER_DECAY**power for power in (3, 2, 1, 0)
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": module.parameters(), "lr": rate}
            for module, rate in zip(
                [model.head, backbone.layernorm, *blocks], rates, strict=True
            )
        ],
        weight_decay=0.05,
    )
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.03)
    train_data = LeafImages(
        args.data,
        train_rows,
        train_transform(args.size, preprocessing["mean"], preprocessing["std"]),
    )
    weights = sampling_weights(train_rows)
    loader_args = (args.batch, args.workers, device)
    val_loader = validation_loader(
        args.data, val_rows, args.size, preprocessing, *loader_args
    )
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    best_files = (out / "best.pt", out / "best-alternate.pt")
    last_file = out / "last.pt"
    first, best, best_epoch, best_slot = 0, -1.0, 0, 1
    history: list[dict[str, Any]] = []
    if last_file.is_file():
        state = torch.load(last_file, map_location="cpu", weights_only=True)
        best_slot = state["best_slot"]
        if state["run"] != run or not best_files[best_slot].is_file():
            raise ValueError(
                "Checkpoint configuration differs or its best checkpoint is missing"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        first, best, best_epoch = state["epoch"], state["best"], state["best_epoch"]
        baseline, history = state["baseline"], state["history"]
        del state
    else:
        initial_loader = validation_loader(
            args.data, val_rows, initial["size"], preprocessing, *loader_args
        )
        baseline = selection_score(val_rows, predict(model, initial_loader, device))
        print(f"V1 validation selection: {baseline['selection']:.4f}", flush=True)
    write_json(out / "history.json", {"baseline": baseline, "epochs": history})

    for epoch in range(first, args.epochs):
        torch.manual_seed(args.seed + epoch)
        backbone.requires_grad_(False)
        for module in [backbone.layernorm, *blocks]:
            module.requires_grad_(epoch > 0)
        decay = 0.2 + 0.4 * (1 + math.cos(math.pi * epoch / max(args.epochs - 1, 1)))
        for group, rate in zip(optimizer.param_groups, rates, strict=True):
            group["lr"] = rate * decay
        sampler = WeightedRandomSampler(
            weights,
            args.samples,
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        loader = DataLoader(
            train_data,
            sampler=sampler,
            batch_size=args.batch,
            num_workers=args.workers,
            pin_memory=True,
        )
        model.train()
        backbone.train(False)
        total = 0.0
        for step, (pixels, target) in enumerate(loader, 1):
            pixels = pixels.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(pixels), target)
            value = loss.detach().item()
            if not math.isfinite(value):
                raise ValueError(f"Non-finite loss at epoch {epoch + 1}, batch {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += value * len(pixels)
            if step % 32 == 0:
                print(
                    f"epoch {epoch + 1}/{args.epochs}, batch {step}/{len(loader)}",
                    flush=True,
                )

        report = selection_score(val_rows, predict(model, val_loader, device))
        history.append({"epoch": epoch + 1, "loss": total / args.samples, **report})
        if report["selection"] > best:
            best, best_epoch = report["selection"], epoch + 1
            # Preserve the slot referenced by last.pt until that checkpoint commits.
            best_slot = 1 - best_slot
            save_atomic(
                {"model": model.state_dict(), "epoch": best_epoch},
                best_files[best_slot],
            )
        save_atomic(
            {
                "run": run,
                "epoch": epoch + 1,
                "best": best,
                "best_epoch": best_epoch,
                "best_slot": best_slot,
                "history": history,
                "baseline": baseline,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            },
            last_file,
        )
        write_json(out / "history.json", {"baseline": baseline, "epochs": history})
        print(
            f"epoch {epoch + 1}/{args.epochs}: selection={report['selection']:.4f}, field source mean={report['field_mean']:.4f}, negative recall={report['negative_recall']:.4f}",
            flush=True,
        )

    selected = torch.load(best_files[best_slot], map_location="cpu", weights_only=True)
    if selected["epoch"] != best_epoch:
        raise ValueError("Selected checkpoint does not match the recorded best epoch")
    model.load_state_dict(selected["model"])
    del selected
    export = out / "model"
    backbone.save_pretrained(export, safe_serialization=True)
    shutil.copyfile(
        args.initial / "preprocessor_config.json", export / "preprocessor_config.json"
    )
    save_file(
        {name: tensor.cpu() for name, tensor in model.head.state_dict().items()},
        export / "classifier.safetensors",
    )
    write_json(
        export / "leaf-classifier.json",
        {**run, "selected_epoch": best_epoch, "validation_selection": best},
    )
    print(f"Selected epoch {best_epoch}, validation selection={best:.4f}", flush=True)


def predict_manifest(args: argparse.Namespace) -> None:
    rows = read_rows(args.data, args.manifest, {"test"})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_export(args.model, device)
    loader = validation_loader(
        args.data,
        rows,
        metadata["size"],
        metadata["preprocessing"],
        args.batch,
        args.workers,
        device,
    )
    scores = predict(model, loader, device)
    confidence, top = scores.max(1)
    closed = scores[:, :-1].argmax(1)
    path = args.predictions
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending")
    fields = (
        "path",
        "sha256",
        "class_id",
        "source",
        "domain",
        "group",
        "prediction",
        "closed_set",
        "confidence",
    )
    with pending.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        for row, index, closed_index, score in zip(
            rows, top.tolist(), closed.tolist(), confidence.tolist(), strict=True
        ):
            writer.writerow(
                {
                    **row,
                    "prediction": CLASSES[index],
                    "closed_set": CLASSES[closed_index],
                    "confidence": score,
                }
            )
    pending.replace(path)
    print(f"Wrote {len(rows)} predictions to {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--development", type=Path, required=True)
    training.add_argument("--initial", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--epochs", type=int, default=6)
    training.add_argument("--samples", type=int, default=3072)
    training.add_argument("--batch", type=int, default=24)
    training.add_argument("--size", type=int, default=280)
    training.add_argument("--workers", type=int, default=2)
    training.add_argument("--seed", type=int, default=42)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--data", type=Path, required=True)
    prediction.add_argument("--manifest", type=Path, required=True)
    prediction.add_argument("--model", type=Path, required=True)
    prediction.add_argument("--predictions", type=Path, required=True)
    prediction.add_argument("--batch", type=int, default=24)
    prediction.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.batch < 1 or args.workers < 0 or getattr(args, "samples", 1) < 1:
        parser.error("Batch and samples must be positive and workers nonnegative")
    if args.command == "train":
        train(args)
    else:
        predict_manifest(args)


if __name__ == "__main__":
    main()
