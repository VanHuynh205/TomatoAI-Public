"""Train and evaluate a tomato-leaf classifier with a Transformers DINOv3 backbone."""

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

CLASSES = (
    "bacterial_spot", "early_blight", "late_blight", "leaf_mold",
    "septoria_leaf_spot", "spider_mites", "yellow_leaf_curl", "healthy",
    "not_tomato_leaf",
)
BACKBONE = "facebook/dinov3-vits16-pretrain-lvd1689m"


def load_manifest(root: Path, name: str) -> list[dict[str, str]]:
    with (root / name).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    paths = set()
    for row in rows:
        path = (root / row["path"]).resolve()
        if (
            not path.is_relative_to(root.resolve()) or not path.is_file()
            or row["class_id"] not in CLASSES or not row["group"]
            or row["split"] not in {"train", "val", "test"} or path in paths
        ):
            raise ValueError(f"Invalid manifest record: {row}")
        paths.add(path)
    if not rows:
        raise ValueError(f"Empty manifest: {name}")
    return rows


def metrics(targets: list[str], predictions: list[str]) -> dict[str, Any]:
    if not targets or len(targets) != len(predictions):
        raise ValueError("Targets and predictions must have equal positive length")
    cells = Counter(zip(targets, predictions, strict=True))
    classes = sorted(set(CLASSES) | set(predictions))
    per_class = {}
    for label in classes:
        support = sum(cells[label, prediction] for prediction in classes)
        predicted = sum(cells[target, label] for target in classes)
        tp = cells[label, label]
        per_class[label] = {
            "support": support,
            "recall": tp / support if support else None,
            "precision": tp / predicted if predicted else None,
            "f1": 2 * tp / (support + predicted) if support else None,
        }
    f1 = [entry["f1"] for entry in per_class.values() if entry["f1"] is not None]
    return {
        "n": len(targets),
        "accuracy": sum(cells[label, label] for label in classes) / len(targets),
        "macro_f1": sum(f1) / len(f1),
        "per_class": per_class,
        "confusion_labels": classes,
        "confusion": [[cells[a, b] for b in classes] for a in classes],
    }


def sampler_weights(rows: list[dict[str, str]]) -> list[float]:
    """Equal class mass, then 75% field mass where both capture domains exist."""
    def stratum(row: dict[str, str]) -> tuple[str, bool]:
        return row["class_id"], row["domain"] != "controlled"

    groups = Counter((stratum(row), row["group"]) for row in rows)
    strata = Counter(key for key, _ in groups)
    weights = []
    for row in rows:
        label, field = key = stratum(row)
        mass = (0.75 if field else 0.25) if (label, not field) in strata else 1.0
        weights.append(mass / (strata[key] * groups[key, row["group"]]))
    return weights


def continuation_weights(rows: list[dict[str, str]], samples: int) -> list[float]:
    """Balance classes and share the verified field budget across sources with a repeat limit."""
    domain_mass = {"field": 0.75, "controlled": 0.15, "other": 0.10, "negative": 1.0}
    fallback_labels = {row["class_id"] for row in rows if row["domain"] != "field"}
    keys = []
    for row in rows:
        label, domain = row["class_id"], row["domain"]
        if label == "not_tomato_leaf":
            domain = "negative"
        elif domain not in {"field", "controlled"}:
            domain = "other"
        source = (row.get("source") or "") if domain == "field" and label in fallback_labels else ""
        keys.append((label, domain, source, row["group"]))
    group_sizes = Counter(keys)
    source_totals: defaultdict[tuple[str, str, str], float] = defaultdict(float)
    for (label, domain, source, _), count in group_sizes.items():
        source_totals[label, domain, source] += math.sqrt(count)
    source_counts = Counter((label, domain) for label, domain, _ in source_totals)
    class_totals: defaultdict[str, float] = defaultdict(float)
    for label, domain in source_counts:
        class_totals[label] += domain_mass[domain]

    # With unit mass per class, this caps each field image at one expected draw per epoch.
    field_limit = len(class_totals) / samples
    excess: defaultdict[str, float] = defaultdict(float)
    weights = []
    for label, domain, source, group in keys:
        weight = domain_mass[domain] / class_totals[label] / source_counts[label, domain]
        weight /= source_totals[label, domain, source] * math.sqrt(group_sizes[label, domain, source, group])
        if domain == "field" and label in fallback_labels and weight > field_limit:
            excess[label] += weight - field_limit
            weight = field_limit
        weights.append(weight)

    return [
        weight * (1 + excess[label] / (1 - domain_mass["field"] / class_totals[label]))
        if domain != "field" and excess[label] else weight
        for (label, domain, _, _), weight in zip(keys, weights, strict=True)
    ]


class LeafImages(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, root: Path, rows: list[dict[str, str]], transform: Any):
        self.root, self.rows, self.transform = root, rows, transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(self.root / row["path"]) as image:
            return self.transform(ImageOps.exif_transpose(image).convert("RGB")), CLASSES.index(row["class_id"])


def leaf_transform(size: int, mean: list[float], std: list[float], training: bool) -> Any:
    fill = tuple(round(value * 255) for value in mean)
    ops: list[Any] = [T.Lambda(lambda im: ImageOps.pad(im, (size, size), Image.Resampling.BICUBIC, fill))]
    if training:
        ops += [
            T.RandomHorizontalFlip(), T.RandomVerticalFlip(0.25),
            T.RandomAffine(24, translate=(0.10, 0.10), scale=(0.83, 1.16), shear=8,
                           interpolation=T.InterpolationMode.BILINEAR, fill=fill),
            T.RandomApply([T.ColorJitter(0.32, 0.28, 0.24, 0.035)], p=0.85),
            T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.23),
            T.RandomAutocontrast(p=0.14), T.RandomAdjustSharpness(1.8, p=0.12),
        ]
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    if training:
        ops.append(T.RandomErasing(p=0.18, scale=(0.015, 0.08), ratio=(0.5, 2.0), value=0))
    return T.Compose(ops)


class LeafClassifier(nn.Module):
    def __init__(self, backbone: Any):
        super().__init__()
        self.backbone = backbone
        width = backbone.config.hidden_size * 2
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Dropout(0.15), nn.Linear(width, len(CLASSES)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(pixel_values=images).last_hidden_state
        patches = tokens[:, 1 + self.backbone.config.num_register_tokens:]
        return self.head(torch.cat((tokens[:, 0], patches.mean(dim=1)), dim=1))


def save_atomic(state: dict[str, Any], path: Path) -> None:
    pending = path.with_suffix(".pending")
    torch.save(state, pending)
    os.replace(pending, path)


def _best_checkpoint(out: Path, epoch: int | None) -> Path:
    if epoch is None:
        return out / "best.pt"
    if type(epoch) is not int or epoch < 0:
        raise ValueError("Invalid best checkpoint epoch")
    return out / f"best-epoch-{epoch}.pt"


def _save_epoch_checkpoint(out: Path, state: dict[str, Any], previous_best: Path) -> Path:
    selected = _best_checkpoint(out, state["best_epoch"])
    if selected != previous_best:
        # Keep the previous selection available until last.pt points to the new one.
        save_atomic({"model": state["model"]}, selected)
    save_atomic(state, out / "last.pt")
    if selected != previous_best and previous_best != out / "best.pt":
        previous_best.unlink(missing_ok=True)
    return selected


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader[Any], device: torch.device) -> tuple[list[str], list[float]]:
    model.eval()
    labels, scores = [], []
    for pixels, _ in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(pixels.to(device, non_blocking=True))
        score, label = logits.float().softmax(1).max(1)
        labels.extend(CLASSES[index] for index in label.tolist())
        scores.extend(score.tolist())
    return labels, scores


def validate_splits(rows: list[dict[str, str]]) -> None:
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "val"]
    if {row["split"] for row in rows} != {"train", "val"}:
        raise ValueError("Training manifest must contain only train and validation")
    if {row["class_id"] for row in train} != set(CLASSES):
        raise ValueError("Training data must cover all nine classes")
    for key in ("group", "sha256"):
        if {row[key] for row in train} & {row[key] for row in val}:
            raise ValueError(f"Train/validation overlap on {key}")
    if not any(row["domain"] == "field" for row in val):
        raise ValueError("Validation needs field images")


def validation_report(
    rows: list[dict[str, str]], labels: list[str], confidence: list[float], *, diagnostics: bool = False,
) -> dict[str, Any]:
    field = [(row, label, score) for row, label, score in zip(rows, labels, confidence, strict=True)
             if row["domain"] == "field"]
    result = {
        "field": metrics([row["class_id"] for row, _, _ in field], [label for _, label, _ in field]),
        "all": metrics([row["class_id"] for row in rows], labels),
    }
    if not diagnostics:
        return result
    groups: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for row, label, _ in field:
        if row["class_id"] == "bacterial_spot":
            groups[row["group"]] += 1
            correct[row["group"]] += label == row["class_id"]
    accepted = [(row, label) for row, label, score in field if label != "not_tomato_leaf" and score >= 0.5]
    result["field_diagnostics"] = {
        "bacterial_spot_groups": len(groups),
        "bacterial_spot_group_recall": (
            sum(correct[group] / count for group, count in groups.items()) / len(groups)
            if groups else None
        ),
        "accepted": len(accepted),
        "accepted_correct": sum(row["class_id"] == label for row, label in accepted),
        "healthy_accepted_as_disease": sum(
            row["class_id"] == "healthy" and label != "healthy" for row, label in accepted
        ),
        "field_by_source": {
            source: metrics(
                [row["class_id"] for row, _, _ in field if row.get("source") == source],
                [label for row, label, _ in field if row.get("source") == source],
            )
            for source in sorted({row["source"] for row, _, _ in field if row.get("source")})
        },
    }
    return result


def train(args: argparse.Namespace) -> None:
    from safetensors.torch import load_file, save_file
    from transformers import AutoImageProcessor, AutoModel

    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab")
    if args.size < 224 or args.size % 16 or min(args.epochs, args.samples, args.batch) < 1:
        raise ValueError("Use a multiple of 16 at least 224 and positive epochs, samples, batch")
    device = torch.device("cuda")
    rows = load_manifest(args.data, "train-val.csv")
    validate_splits(rows)
    torch.manual_seed(args.seed)
    init_export: Path | None = getattr(args, "init_export", None)
    out = args.output
    token = os.getenv("HF_TOKEN")
    continuation = {}
    if init_export is not None:
        if init_export.resolve() == (out / "model").resolve():
            raise ValueError("Continuation must export to a separate directory")
        metadata = json.loads((init_export / "leaf-classifier.json").read_text(encoding="utf-8"))
        if (
            metadata.get("class_ids") != list(CLASSES) or metadata.get("size") != args.size
            or metadata.get("test_time_horizontal_flip", False)
        ):
            raise ValueError("Continuation requires the same classes, resolution and single-view inference")
        hashes = {}
        for name in ("model.safetensors", "classifier.safetensors", "config.json",
                     "preprocessor_config.json", "leaf-classifier.json"):
            with (init_export / name).open("rb") as stream:
                hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        continuation = {
            "init_export_sha256": hashes,
            "init_model_version": metadata.get("model_version"),
            "sampler": "field_75_source_balanced_one_expected_draw_controlled_15_other_10_sqrt_group",
            "selection": "field_macro_f1_with_epoch_zero",
        }
        backbone, loading = AutoModel.from_pretrained(
            init_export, local_files_only=True, use_safetensors=True,
            trust_remote_code=False, attn_implementation="sdpa", output_loading_info=True,
        )
        if any(loading.values()) or backbone.config.model_type != "dinov3_vit":
            raise ValueError("Continuation requires an exact DINOv3 backbone export")
        model = LeafClassifier(backbone)
        model.head.load_state_dict(load_file(init_export / "classifier.safetensors"))
        backbone_name, revision = metadata["backbone"], metadata["revision"]
        rates = (7.5e-5, 8e-6)
    else:
        backbone = AutoModel.from_pretrained(args.backbone, token=token, attn_implementation="sdpa")
        model = LeafClassifier(backbone)
        backbone_name, revision = args.backbone, backbone.config._commit_hash
        rates = (3e-4, 2e-5)
    processor = AutoImageProcessor.from_pretrained(
        init_export or args.backbone, token=token if init_export is None else None,
        local_files_only=init_export is not None,
    )
    model.to(device)
    tuned = list(backbone.layer[-4:].parameters()) + list(backbone.norm.parameters())
    optimizer = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr": rates[0]},
        {"params": tuned, "lr": rates[1]},
    ], weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    weights = continuation_weights(train_rows, args.samples) if continuation else sampler_weights(train_rows)
    mean, std = processor.image_mean, processor.image_std
    train_data = LeafImages(args.data, train_rows, leaf_transform(args.size, mean, std, True))
    val_data = LeafImages(args.data, val_rows, leaf_transform(args.size, mean, std, False))
    loader_options = {"batch_size": args.batch, "num_workers": args.workers, "pin_memory": True}
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    out.mkdir(parents=True, exist_ok=True)
    run = {
        "manifest_sha256": hashlib.sha256((args.data / "train-val.csv").read_bytes()).hexdigest(),
        "backbone": backbone_name, "revision": revision, "size": args.size, "epochs": args.epochs,
        "samples": args.samples, "batch": args.batch, "seed": args.seed, "workers": args.workers,
        **continuation,
    }
    if continuation:
        run.update({"head_lr": rates[0], "backbone_lr": rates[1], "tuned_blocks": 4})
    history: list[dict[str, Any]] = []
    first, best = 0, -1.0
    best_epoch: int | None = None
    best_file = _best_checkpoint(out, best_epoch)

    def save_epoch(epoch: int) -> Path:
        return _save_epoch_checkpoint(out, {
            "run": run, "epoch": epoch, "best": best, "best_epoch": best_epoch,
            "history": history, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
        }, best_file)

    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        best_epoch = state.get("best_epoch")
        best_file = _best_checkpoint(out, best_epoch)
        if state["run"] != run or not best_file.is_file():
            raise ValueError("Checkpoint belongs to a different run or its best checkpoint is missing")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        first, best, history = state["epoch"] + 1, state["best"], state["history"]
        del state
    elif continuation:
        labels, scores = predict(model, val_loader, device)
        report = validation_report(val_rows, labels, scores, diagnostics=True)
        history.append({"epoch": 0, "loss": None, **report})
        best, best_epoch = report["field"]["macro_f1"], 0
        best_file = save_epoch(-1)
        print(f"epoch 0 baseline, field val macro-F1={best:.4f}", flush=True)
    for epoch in range(first, args.epochs):
        torch.manual_seed(args.seed + epoch)
        backbone.requires_grad_(False)
        for param in tuned:
            param.requires_grad_(bool(continuation) or epoch > 0)
        decay = 0.15 + 0.85 * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
        for group, rate in zip(optimizer.param_groups, rates, strict=True):
            group["lr"] = rate * decay
        sampler = WeightedRandomSampler(weights, args.samples, generator=torch.Generator().manual_seed(args.seed + epoch))
        loader = DataLoader(train_data, sampler=sampler, **loader_options)
        model.train()
        backbone.eval()
        total = 0.0
        for pixels, target in loader:
            optimizer.zero_grad(set_to_none=True)
            pixels, target = pixels.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(pixels), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.detach().item() * len(pixels)
        labels, scores = predict(model, val_loader, device)
        report = validation_report(val_rows, labels, scores, diagnostics=bool(continuation))
        score = report["field"]["macro_f1"]
        history.append({"epoch": epoch + 1, "loss": total / args.samples, **report})
        if score > best:
            best, best_epoch = score, epoch + 1
        best_file = save_epoch(epoch)
        print(f"epoch {epoch + 1}/{args.epochs}, loss={total / args.samples:.4f}, field val macro-F1={score:.4f}", flush=True)
    model.load_state_dict(torch.load(best_file, map_location="cpu", weights_only=True)["model"])
    export = out / "model"
    backbone.save_pretrained(export, safe_serialization=True)
    processor.save_pretrained(export)
    save_file({name: tensor.cpu() for name, tensor in model.head.state_dict().items()}, export / "classifier.safetensors")
    (export / "leaf-classifier.json").write_text(json.dumps({**run, "class_ids": CLASSES, "best_field_val_macro_f1": best}, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", default=BACKBONE)
    parser.add_argument("--init-export", type=Path)
    parser.add_argument("--size", type=int, default=336)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--samples", type=int, default=4800)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
