"""P08 staged YOLO26s-cls training with pre-pad PIL augmentation and regularization."""

import hashlib
import io
import json
import math
import os
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import (  # type: ignore[import-untyped]
    ColorJitter,
    InterpolationMode,
    RandomAffine,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomVerticalFlip,
)

from ai.data import ManifestDataset, _decode_rgb, _sha256, _source_path
from ai.models import build_model
from ai.preprocessing import build_transform
from ai.train import _atomic_save, configure_stage

from .head_recovery import (  # type: ignore[import-not-found]
    HERE,
    SOURCES,
    THREADS,
    VENDOR_CHECK,
    collect,
    select_candidate,
    summarize,
)
from .inner_training import CONFIG, DATA, DEVICE, GPU_LIMIT, SEED, inner_split  # type: ignore[import-not-found]

BATCH = 8
ACCUMULATION = 2
EPOCH_SAMPLES = 4096
WARM = 2
EVALUATE_AT = (2, 4, 6, 8)
PLANTDOC_MASS = 0.30
CLASS_POWER = -0.75
DROPOUT = 0.25
LABEL_SMOOTHING = 0.06
INNER_PV_FLOOR = 0.978
INNER_PLANTDOC_FLOOR = 0.49


class PrePadAugment:
    """High-res PIL crop/flip and mild color jitter before a single bicubic fit_pad(224)."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.fit_pad = build_transform(
            spec | {"horizontal_flip_p": 0.0}, training=False
        )
        self.pv_crop = RandomResizedCrop(
            256,
            scale=(0.76, 1.0),
            ratio=(0.70, 1.42),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        self.hflip = RandomHorizontalFlip(0.5)
        self.vflip = RandomVerticalFlip(0.15)
        self.affine = RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            interpolation=InterpolationMode.BILINEAR,
            fill=(124, 116, 104),
        )
        self.jitter = ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.12, hue=0.02
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        w, h = image.size
        if w == h:
            img = self.pv_crop(image) if float(torch.rand(1).item()) < 0.75 else image
        else:
            scale = 0.78 + 0.22 * float(torch.rand(1).item())
            cw, ch = max(1, round(w * scale)), max(1, round(h * scale))
            x0 = int(torch.randint(0, max(1, w - cw + 1), (1,)).item())
            y0 = int(torch.randint(0, max(1, h - ch + 1), (1,)).item())
            img = image.crop((x0, y0, x0 + cw, y0 + ch))
        img = self.hflip(img)
        img = self.vflip(img)
        img = self.affine(img)
        img = self.jitter(img)
        return self.fit_pad(img)


class CachedPlantDocDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Hash-verified training subset caching only PlantDoc RGB images (~25 MiB RAM)."""

    def __init__(
        self,
        base_ds: ManifestDataset,
        indices: list[int],
        transform: Any,
    ) -> None:
        self.base_ds = base_ds
        self.indices = indices
        self.transform = transform
        self.pd_cache: dict[int, Image.Image] = {}
        for idx in indices:
            row = base_ds._rows[idx]
            if row["sample_id"].startswith("plantdoc:"):
                path = _source_path(base_ds.source_root, row["source_path"])
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != row["source_sha256"]:
                    raise ValueError(f"source_sha256 mismatch: {row['sample_id']}")
                with _decode_rgb(io.BytesIO(raw)) as rgb:
                    if max(rgb.size) > 448:
                        ratio = 448.0 / max(rgb.size)
                        rgb = rgb.resize(
                            (
                                max(1, round(rgb.width * ratio)),
                                max(1, round(rgb.height * ratio)),
                            ),
                            resample=Image.Resampling.BICUBIC,
                        )
                    else:
                        rgb = rgb.copy()
                    self.pd_cache[idx] = rgb

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, str]:
        idx = self.indices[i]
        row = self.base_ds._rows[idx]
        target = self.base_ds._indices[row["class_id"]]
        sid = row["sample_id"]
        if idx in self.pd_cache:
            tensor: torch.Tensor = self.transform(self.pd_cache[idx])
        else:
            path = _source_path(self.base_ds.source_root, row["source_path"])
            with path.open("rb") as stream:
                if (
                    hashlib.file_digest(stream, "sha256").hexdigest()
                    != row["source_sha256"]
                ):
                    raise ValueError(f"source_sha256 mismatch: {sid}")
                stream.seek(0)
                with _decode_rgb(stream) as rgb:
                    tensor = self.transform(rgb)
        return tensor, target, sid


def balanced_weights(
    ids: list[str],
    labels: torch.Tensor,
    plantdoc_mass: float = PLANTDOC_MASS,
    power: float = CLASS_POWER,
) -> torch.Tensor:
    if len(ids) != len(labels) or not 0 < plantdoc_mass < 1:
        raise ValueError("invalid P08 source balance inputs")
    if any(not sid.startswith(SOURCES) for sid in ids):
        raise ValueError("unknown P08 sample source")
    weights = torch.empty(len(labels), dtype=torch.float64)
    for prefix, mass in zip(SOURCES, (1 - plantdoc_mass, plantdoc_mass), strict=True):
        mask = torch.tensor([sid.startswith(prefix) for sid in ids])
        if not mask.any():
            raise ValueError(f"empty P08 sample source: {prefix}")
        counts = torch.bincount(labels[mask], minlength=8)
        factors = counts[labels[mask]].double().pow(power)
        weights[mask] = factors * (mass / factors.sum())
    return weights


def stage_parameter_groups(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    configure_stage(model, "yolo26s_cls", "frozen")
    raw_blocks = getattr(getattr(model, "model", None), "model", None)
    if not isinstance(raw_blocks, nn.Sequential) or len(raw_blocks) != 11:
        raise ValueError("invalid YOLO26s block structure for P08")
    blocks: Any = raw_blocks
    head = list(blocks[10].linear.parameters())
    mid = [p for b in (blocks[5], blocks[6]) for p in b.parameters()]
    tail = [
        p
        for b in (blocks[7], blocks[8], blocks[9], blocks[10].conv)
        for p in b.parameters()
    ]
    all_selected = head + mid + tail
    if (
        not head
        or not mid
        or not tail
        or len({id(p) for p in all_selected}) != len(all_selected)
    ):
        raise ValueError("empty or overlapping P08 parameter groups")
    return head, mid, tail


def apply_p08_mode(
    model: nn.Module,
    head: list[nn.Parameter],
    mid: list[nn.Parameter],
    tail: list[nn.Parameter],
    *,
    full: bool,
) -> None:
    configure_stage(model, "yolo26s_cls", "frozen")
    blocks: Any = getattr(getattr(model, "model", None), "model", None)
    blocks[10].drop.p = DROPOUT
    blocks[10].drop.train()
    active = {id(p) for p in (head + mid + tail if full else head)}
    for p in model.parameters():
        p.requires_grad_(id(p) in active)


def choose_epoch(variants: dict[str, Any]) -> str | None:
    return max(
        (
            k
            for k, m in variants.items()
            if m["pv_f1"] >= INNER_PV_FLOOR
            and m["plantdoc_f1"] >= INNER_PLANTDOC_FLOOR
            and m["threshold"] is not None
        ),
        key=lambda k: tuple(
            variants[k][field] for field in ("plantdoc_f1", "pv_f1", "pooled_f1")
        ),
        default=None,
    )


def evaluate_dev(
    model: nn.Module,
    dev_x: torch.Tensor,
    dev_y: torch.Tensor,
    dev_ids: list[str],
    classes: list[str],
) -> dict[str, Any]:
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()
    try:
        model.eval()
        logits_list = []
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            for i in range(0, len(dev_ids), 32):
                if torch.cuda.max_memory_reserved() > GPU_LIMIT:
                    raise ValueError("P08 GPU budget exceeded during evaluation")
                logits_list.append(model(dev_x[i : i + 32].to(DEVICE)).float().cpu())
        logits = torch.cat(logits_list, dim=0)
        metrics: dict[str, Any] = summarize(
            {"ids": dev_ids, "labels": dev_y}, logits, classes
        )
        return metrics
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)


def run_training(
    config: dict[str, Any],
    train_ids: list[str],
    dev_ids: list[str],
    backbone_epochs: int,
    output: Path,
    label: str,
) -> tuple[nn.Module, dict[str, Any]]:
    spec = config["preprocessing"]["spec"]
    classes: list[str] = config["class_ids"]
    base_ds = ManifestDataset(DATA, "train", build_transform(spec))
    if base_ds.class_ids != tuple(classes) or len(base_ds) != 8238:
        raise ValueError("P08 dataset or class order mismatch")
    idx_map = {r["sample_id"]: i for i, r in enumerate(base_ds._rows)}
    if not set(train_ids + dev_ids) <= idx_map.keys() or set(train_ids).intersection(
        dev_ids
    ):
        raise ValueError("P08 training/development membership mismatch")
    train_indices = [idx_map[sid] for sid in train_ids]
    dev_indices = [idx_map[sid] for sid in dev_ids]

    dev_x = torch.empty((len(dev_ids), 3, 224, 224), dtype=torch.float32)
    dev_y = torch.empty(len(dev_ids), dtype=torch.long)
    for k, idx in enumerate(dev_indices):
        x, y, sid = base_ds[idx]
        if sid != dev_ids[k]:
            raise ValueError("P08 development sample order mismatch")
        dev_x[k] = x
        dev_y[k] = y

    train_ds = CachedPlantDocDataset(base_ds, train_indices, PrePadAugment(spec))
    sample_img, _, _ = train_ds[0]
    if (
        sample_img.shape != (3, 224, 224)
        or not torch.isfinite(sample_img).all()
        or sample_img.min() < 0
        or sample_img.max() > 1
    ):
        raise ValueError("P08 train transform must preserve shape and RGB range")

    labels = torch.tensor(
        [classes.index(base_ds._rows[i]["class_id"]) for i in train_indices]
    )
    weights: Any = balanced_weights(train_ids, labels)

    torch.manual_seed(SEED)
    sampler = WeightedRandomSampler(
        weights, EPOCH_SAMPLES, generator=torch.Generator().manual_seed(SEED)
    )
    loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler)
    model = build_model(
        "yolo26s_cls", len(classes), Path(config["pretrained_path"])
    ).to(DEVICE)
    head, mid, tail = stage_parameter_groups(model)

    warm_optimizer = torch.optim.AdamW(head, lr=1e-3, weight_decay=1e-4)
    full_optimizer = torch.optim.AdamW(
        [
            {"params": head, "lr": 3e-4},
            {"params": tail, "lr": 6e-5},
            {"params": mid, "lr": 2.5e-5},
        ],
        weight_decay=2e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        full_optimizer, T_max=max(EVALUATE_AT), eta_min=5e-6
    )
    scaler = torch.amp.GradScaler("cuda")
    history: list[dict[str, Any]] = []
    variants: dict[str, Any] = {}

    for epoch in range(1, WARM + backbone_epochs + 1):
        full = epoch > WARM
        apply_p08_mode(model, head, mid, tail, full=full)
        optimizer = full_optimizer if full else warm_optimizer
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_samples = 0
        optimizer_steps = 0
        skipped_steps = 0
        for step, (bx, by, _) in enumerate(loader, 1):
            if torch.cuda.max_memory_reserved() > GPU_LIMIT:
                raise ValueError("P08 GPU budget exceeded")
            bx = bx.to(DEVICE)
            by = by.to(DEVICE, dtype=torch.long)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits = model(bx)
                loss = F.cross_entropy(
                    logits, by, reduction="sum", label_smoothing=LABEL_SMOOTHING
                )
            if not math.isfinite(loss.item()):
                raise FloatingPointError("non-finite P08 loss")
            total_loss += loss.item()
            total_samples += int(by.shape[0])
            torch.autograd.backward(scaler.scale(loss / (BATCH * ACCUMULATION)))
            if step % ACCUMULATION == 0:
                scaler.unscale_(optimizer)
                grads = [
                    p.grad
                    for g in optimizer.param_groups
                    for p in g["params"]
                    if p.grad is not None
                ]
                finite = all(bool(torch.isfinite(g).all()) for g in grads)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += int(finite)
                skipped_steps += int(not finite)
        row = {
            "loss": total_loss / total_samples,
            "optimizer_steps": float(optimizer_steps),
            "skipped_steps": float(skipped_steps),
        }
        if (
            torch.cuda.max_memory_reserved() > GPU_LIMIT
            or row["optimizer_steps"] + row["skipped_steps"]
            != EPOCH_SAMPLES // (BATCH * ACCUMULATION)
            or row["optimizer_steps"] < 1
            or not math.isfinite(row["loss"])
        ):
            raise ValueError("P08 training budget or optimizer update check failed")
        if full:
            scheduler.step()
        tail_epoch = epoch - WARM
        if dev_ids and tail_epoch in EVALUATE_AT:
            variants[str(tail_epoch)] = evaluate_dev(
                model, dev_x, dev_y, dev_ids, classes
            )
            m = variants[str(tail_epoch)]
            print(
                f"{label} development epoch {tail_epoch}: PlantDoc={m['plantdoc_f1']:.6f} PV={m['pv_f1']:.6f}",
                flush=True,
            )
        history.append({"epoch": epoch, "phase": "full" if full else "warm", **row})
        _atomic_save(output / f"{label}-history.json", history)
        if variants:
            _atomic_save(output / f"{label}-development.json", variants)
        print(label, history[-1], flush=True)
    return model, variants


def main() -> None:
    check = import_module(VENDOR_CHECK)
    lock_path = HERE / "p08-lock.json"
    lock = check.read_json(lock_path)
    for name, digest in lock["files_sha256"].items():
        if _sha256(Path(name)) != digest:
            raise ValueError(f"P08 input hash mismatch: {name}")
    check.verify_protocol()
    gpu = check.check_gpu()
    torch.set_num_threads(THREADS)
    config = check.read_json(CONFIG)
    assignments = check.read_json(DATA / "split.json")["assignments"]
    train_ids, dev_ids = inner_split(assignments)
    if (train_ids, dev_ids) != (lock["inner_train_ids"], lock["inner_dev_ids"]):
        raise ValueError("P08 locked inner membership mismatch")
    output = HERE / "P08"
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
        {
            "backbone_epochs": selected,
            "p02_validation_opened": False,
            "lock_sha256": _sha256(lock_path),
            "gpu": gpu | {"peak_reserved_bytes": torch.cuda.max_memory_reserved()},
        },
    )
    if selected is None:
        print("P08 did not meet inner development floors.", flush=True)
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
        raise ValueError("P08 weights or GPU budget changed during validation")
    if val["ids"] != [r["sample_id"] for r in assignments if r["split"] == "val"]:
        raise ValueError("P08 locked validation membership mismatch")
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
    winner = select_candidate({"p08-refit": metric}, lock["gate"])
    _atomic_save(
        output / "decision.json",
        {
            "winner": winner,
            "metric": metric,
            "weights_sha256": digest,
            "logits_sha256": _sha256(scores),
            "lock_sha256": _sha256(lock_path),
            "gpu": gpu | {"peak_reserved_bytes": torch.cuda.max_memory_reserved()},
            "test_opened": False,
        },
    )
    print(
        "P08 official gate:", winner, metric["plantdoc_f1"], metric["pv_f1"], flush=True
    )


if __name__ == "__main__":
    with TemporaryDirectory(prefix="tomato-p08-") as config_dir:
        os.environ.update(YOLO_CONFIG_DIR=config_dir, MPLCONFIGDIR=config_dir)
        main()
