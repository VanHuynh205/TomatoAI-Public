"""Fine-tune DINOv2-base on a sealed, field-first tomato-leaf dataset."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from ai.dinov3_field import (
    CLASSES,
    LeafImages,
    leaf_transform,
    load_manifest,
    metrics,
    save_atomic,
)

if TYPE_CHECKING:
    from transformers import Dinov2Model

BACKBONE = "facebook/dinov2-base"
REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
NEGATIVE = "not_tomato_leaf"
WARMUP_DOMAINS = {"field": 0.50, "web": 0.15, "controlled": 0.35}
FINETUNE_DOMAINS = {"field": 0.65, "web": 0.15, "controlled": 0.20}


class Classifier(nn.Module):
    def __init__(self, backbone: "Dinov2Model"):
        super().__init__()
        self.backbone = backbone
        width = backbone.config.hidden_size * 2
        self.head = nn.Sequential(
            nn.LayerNorm(width), nn.Dropout(0.15), nn.Linear(width, len(CLASSES))
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(pixel_values=images).last_hidden_state
        return cast(
            torch.Tensor,
            self.head(torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1)),
        )


def load_splits(root: Path) -> dict[str, list[dict[str, str]]]:
    development = load_manifest(root, "train-val.csv")
    test = load_manifest(root, "test.csv")
    if {row["split"] for row in development} != {"train", "val"} or {
        row["split"] for row in test
    } != {"test"}:
        raise ValueError("Expected separate train/val and test manifests")
    rows = development + test
    splits = {
        name: [row for row in rows if row["split"] == name]
        for name in ("train", "val", "test")
    }
    if {row["class_id"] for row in splits["train"]} != set(CLASSES):
        raise ValueError("Training data must cover all nine classes")
    for split, subset in splits.items():
        if not any(
            row["domain"] == "field" and row["class_id"] != NEGATIVE for row in subset
        ):
            raise ValueError(f"{split} needs field tomato leaf images")
        if not any(row["class_id"] == NEGATIVE for row in subset):
            raise ValueError(f"{split} needs labeled negatives")
    for key in ("path", "group", "sha256"):
        membership: dict[str, str] = {}
        for row in rows:
            value, split = row[key], row["split"]
            if not value or membership.setdefault(value, split) != split:
                raise ValueError(f"Missing or cross-split {key}: {value}")
    for row in rows:
        if (
            hashlib.sha256((root / row["path"]).read_bytes()).hexdigest()
            != row["sha256"]
        ):
            raise ValueError(f"Image does not match its manifest digest: {row['path']}")
    return splits


def summarize(rows: list[dict[str, str]], predictions: list[str]) -> dict[str, Any]:
    selectors: dict[str, Callable[[dict[str, str]], bool]] = {
        "all": lambda row: True,
        "field_tomato": lambda row: (
            row["domain"] == "field" and row["class_id"] != NEGATIVE
        ),
        "not_tomato": lambda row: row["class_id"] == NEGATIVE,
    }
    result = {}
    for name, matches in selectors.items():
        ids = [i for i, row in enumerate(rows) if matches(row)]
        if ids:
            result[name] = metrics(
                [rows[i]["class_id"] for i in ids], [predictions[i] for i in ids]
            )
    result["by_source"] = {}
    for source in sorted({row["source"] for row in rows}):
        ids = [i for i, row in enumerate(rows) if row["source"] == source]
        result["by_source"][source] = metrics(
            [rows[i]["class_id"] for i in ids], [predictions[i] for i in ids]
        )
    return result


def sampling_weights(
    rows: list[dict[str, str]], domain_mass: dict[str, float]
) -> list[float]:
    """Balance classes, prioritize field images, and limit correlated capture groups."""

    def stratum(row: dict[str, str]) -> tuple[str, str]:
        label = row["class_id"]
        return label, row["source"] if label == NEGATIVE else row["domain"]

    groups = Counter((stratum(row), row["group"]) for row in rows)
    strata = Counter(key for key, _ in groups)
    masses = {key: 1.0 if key[0] == NEGATIVE else domain_mass[key[1]] for key in strata}
    totals: defaultdict[str, float] = defaultdict(float)
    for (label, _), mass in masses.items():
        totals[label] += mass
    weights = []
    for row in rows:
        label, _ = key = stratum(row)
        weights.append(
            masses[key] / (totals[label] * strata[key] * groups[key, row["group"]])
        )
    return weights


@torch.inference_mode()
def predict(
    model: nn.Module, loader: DataLoader[Any], device: torch.device
) -> torch.Tensor:
    model.eval()
    predictions = []
    for pixels, _ in loader:
        pixels = pixels.to(device, non_blocking=True)
        with torch.autocast(
            device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = model(pixels).float()
            logits = (logits + model(pixels.flip(-1)).float()) / 2
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Non-finite DINOv2 predictions")
        predictions.append(logits.softmax(1).cpu())
    return torch.cat(predictions)


def train(args: argparse.Namespace) -> None:
    from safetensors.torch import save_file
    from transformers import AutoImageProcessor, AutoModel

    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab")
    if (
        args.size < 196
        or args.size % 14
        or min(args.epochs, args.samples, args.batch) < 1
    ):
        raise ValueError(
            "Use a size of at least 196 divisible by 14 and positive epochs, samples and batch"
        )
    device = torch.device("cuda")
    splits = load_splits(args.data)
    print(
        {
            split: dict(Counter(row["source"] for row in rows))
            for split, rows in splits.items()
        },
        flush=True,
    )
    torch.manual_seed(args.seed)
    processor = AutoImageProcessor.from_pretrained(BACKBONE, revision=REVISION)
    backbone = AutoModel.from_pretrained(
        BACKBONE, revision=REVISION, use_safetensors=True, attn_implementation="sdpa"
    )
    model = Classifier(backbone).to(device)
    tuned = list(backbone.encoder.layer[-4:].parameters()) + list(
        backbone.layernorm.parameters()
    )
    rates = (4e-4, 3e-5)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.head.parameters(), "lr": rates[0]},
            {"params": tuned, "lr": rates[1]},
        ],
        weight_decay=0.05,
    )
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    mean, std = processor.image_mean, processor.image_std
    train_data = LeafImages(
        args.data, splits["train"], leaf_transform(args.size, mean, std, True)
    )
    eval_transform = leaf_transform(args.size, mean, std, False)
    val_data = LeafImages(args.data, splits["val"], eval_transform)
    options = {
        "batch_size": args.batch,
        "num_workers": args.workers,
        "pin_memory": True,
    }
    val_loader = DataLoader(val_data, shuffle=False, **options)
    warmup_weights = sampling_weights(splits["train"], WARMUP_DOMAINS)
    field_weights = sampling_weights(splits["train"], FINETUNE_DOMAINS)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    run = {
        "backbone": BACKBONE,
        "revision": REVISION,
        "class_ids": CLASSES,
        "size": args.size,
        "epochs": args.epochs,
        "samples": args.samples,
        "batch": args.batch,
        "seed": args.seed,
        "workers": args.workers,
        "warmup_domain_mass": WARMUP_DOMAINS,
        "finetune_domain_mass": FINETUNE_DOMAINS,
        "preprocessing": {
            "geometry": "fit_pad",
            "interpolation": "bicubic",
            "mean": mean,
            "std": std,
        },
        "manifests": {
            name: hashlib.sha256((args.data / name).read_bytes()).hexdigest()
            for name in ("train-val.csv", "test.csv")
        },
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
    }
    best_file, last_file = out / "best.pt", out / "last.pt"
    history = []
    first, best = 0, -1.0
    if last_file.exists():
        state = torch.load(last_file, map_location="cpu", weights_only=True)
        if state["run"] != run or not best_file.is_file():
            raise ValueError(
                "Checkpoint run configuration differs or best.pt is missing"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        first, best, history = state["epoch"] + 1, state["best"], state["history"]
        del state
    for epoch in range(first, args.epochs):
        torch.manual_seed(args.seed + epoch)
        backbone.requires_grad_(False)
        for parameter in tuned:
            parameter.requires_grad_(epoch > 0)
        decay = 0.2 + 0.8 * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
        for group, rate in zip(optimizer.param_groups, rates, strict=True):
            group["lr"] = rate * decay
        generator = torch.Generator().manual_seed(args.seed + epoch)
        weights = warmup_weights if epoch == 0 else field_weights
        sampler = WeightedRandomSampler(weights, args.samples, generator=generator)
        loader = DataLoader(train_data, sampler=sampler, **options)
        model.train()
        backbone.eval()
        total = 0.0
        for step, (pixels, target) in enumerate(loader, 1):
            pixels, target = (
                pixels.to(device, non_blocking=True),
                target.to(device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(pixels), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.detach().item() * len(pixels)
            if step % 32 == 0:
                print(
                    f"epoch {epoch + 1}/{args.epochs}, batch {step}/{len(loader)}",
                    flush=True,
                )
        scores = predict(model, val_loader, device)
        report = summarize(
            splits["val"], [CLASSES[index] for index in scores.argmax(1).tolist()]
        )
        field = report["field_tomato"]["macro_f1"]
        negative = report["not_tomato"]["accuracy"]
        selection = 0.9 * field + 0.1 * negative
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total / args.samples,
                "validation": report,
                "selection": selection,
            }
        )
        if selection > best:
            best = selection
            save_atomic({"model": model.state_dict(), "epoch": epoch + 1}, best_file)
        save_atomic(
            {
                "run": run,
                "epoch": epoch,
                "best": best,
                "history": history,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            },
            last_file,
        )
        print(
            f"epoch {epoch + 1}/{args.epochs}: field val macro-F1={field:.4f}, negative recall={negative:.4f}",
            flush=True,
        )

    state = torch.load(best_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    best_epoch = state["epoch"]
    del state
    export = out / "model"
    backbone.save_pretrained(export, safe_serialization=True)
    processor.save_pretrained(export)
    save_file(
        {name: tensor.cpu() for name, tensor in model.head.state_dict().items()},
        export / "classifier.safetensors",
    )
    run.update(
        {
            "selected_epoch": best_epoch,
            "validation_selection": best,
            "test_time_horizontal_flip": True,
        }
    )
    (export / "leaf-classifier.json").write_text(
        json.dumps(run, indent=2), encoding="utf-8"
    )
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    test_rows = splits["test"]
    test_loader = DataLoader(
        LeafImages(args.data, test_rows, eval_transform), shuffle=False, **options
    )
    scores = predict(model, test_loader, device)
    top = scores.argmax(1).tolist()
    closed = scores[:, :-1].argmax(1).tolist()
    predictions = [CLASSES[index] for index in top]
    report = summarize(test_rows, predictions)
    (out / "test-metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (out / "test-predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fields = [
            "path",
            "sha256",
            "class_id",
            "source",
            "domain",
            "group",
            "prediction",
            "closed_set",
            "confidence",
        ]
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        for row, prediction, closed_index, index, score in zip(
            test_rows, predictions, closed, top, scores.tolist(), strict=True
        ):
            writer.writerow(
                {
                    **row,
                    "prediction": prediction,
                    "closed_set": CLASSES[closed_index],
                    "confidence": score[index],
                }
            )
    print(
        f"Selected epoch {best_epoch}. Test field macro-F1={report['field_tomato']['macro_f1']:.4f}.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--samples", type=int, default=3072)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
