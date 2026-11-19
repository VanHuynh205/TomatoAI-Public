"""P04 bounded tail adaptation for YOLO26s on existing sealed images."""

import json
import os
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import ColorJitter, Compose, InterpolationMode, RandomAffine

from ai.data import ManifestDataset, _sha256
from ai.preprocessing import build_transform
from ai.train import _atomic_save, configure_stage, train_epoch

from .head_recovery import (
    BATCH,
    HERE,
    THREADS,
    VENDOR_CHECK,
    collect,
    load_run,
    sample_weights,
    select_candidate,
    summarize,
)

SEED = 42
EPOCH_SAMPLES = 4096
CHECKPOINTS = (2, 4)
PLANTDOC_MASS = 0.40
DEVICE = torch.device("cuda")


def make_loader(config: dict[str, Any]) -> tuple[DataLoader[Any], torch.Tensor]:
    transform = Compose(
        [
            build_transform(config["preprocessing"]["spec"], training=True),
            RandomAffine(
                12,
                translate=(0.08, 0.08),
                scale=(0.90, 1.05),
                interpolation=InterpolationMode.BILINEAR,
                fill=124 / 255,
            ),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        ]
    )
    dataset = ManifestDataset(Path(config["dataset_dir"]), "train", transform)
    ids = [r["sample_id"] for r in dataset._rows]
    labels = torch.tensor(
        [config["class_ids"].index(r["class_id"]) for r in dataset._rows]
    )
    if len(ids) != 8238 or dataset.class_ids != tuple(config["class_ids"]):
        raise ValueError("P04 train membership or class order mismatch")
    weights = sample_weights(ids, labels, PLANTDOC_MASS)
    sampler = WeightedRandomSampler(
        weights, EPOCH_SAMPLES, generator=torch.Generator().manual_seed(SEED)
    )
    image, _, _ = dataset[0]
    if (
        image.shape != (3, 224, 224)
        or not torch.isfinite(image).all()
        or image.min() < 0
        or image.max() > 1
    ):
        raise ValueError("P04 train transform must preserve finite RGB range and shape")
    loader = DataLoader(
        dataset,  # type: ignore[arg-type]  # ManifestDataset is map-style.
        batch_size=BATCH,
        sampler=sampler,
    )
    return loader, weights


def main() -> None:
    check = import_module(VENDOR_CHECK)
    lock_path = HERE / "p04-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for name, digest in lock["files_sha256"].items():
        if _sha256(Path(name)) != digest:
            raise ValueError(f"P04 input hash mismatch: {name}")
    check.verify_protocol()
    gpu = check.check_gpu()
    print(f"Locked inputs verified. GPU: {gpu}", flush=True)
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    model, config = load_run("E10")
    loader, weights = make_loader(config)
    output = HERE / "P04"
    output.mkdir(exist_ok=False)
    # The repo stage policy freezes early blocks and all BatchNorm statistics.
    groups = configure_stage(model, "yolo26s_cls", "fine_tune")
    expected_lr = {"head": 3e-4, "backbone": 5e-5}
    if {g["name"] for g in groups} != set(expected_lr):
        raise ValueError("unexpected P04 optimizer groups")
    for group in groups:
        group["lr"] = expected_lr[group["name"]]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    history: list[dict[str, Any]] = []
    for epoch in range(1, max(CHECKPOINTS) + 1):
        configure_stage(model, "yolo26s_cls", "fine_tune")
        row = train_epoch(
            model,
            loader,
            optimizer,
            device=DEVICE,
            accumulation=2,
            amp=True,
            scaler=scaler,
        )
        if (
            row["optimizer_steps"] + row["skipped_steps"]
            != EPOCH_SAMPLES // (BATCH * 2)
            or row["optimizer_steps"] < 1
        ):
            raise ValueError("P04 optimizer window count mismatch or no updates")
        if torch.cuda.max_memory_reserved() > 3.2 * 2**30:
            raise ValueError("P04 GPU budget exceeded")
        history.append({"epoch": epoch, **row})
        _atomic_save(output / "history.json", history)
        print(history[-1], flush=True)
        if epoch in CHECKPOINTS:
            state = {key: value.cpu() for key, value in model.state_dict().items()}
            _atomic_save(output / f"epoch-{epoch}.pth", state)
    variants: dict[str, Any] = {}
    for epoch in CHECKPOINTS:
        name = f"epoch-{epoch}"
        path = output / f"{name}.pth"
        model.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True), strict=True
        )
        val = collect(model.eval(), config, "val", None)
        result = summarize(val, val["logits"], config["class_ids"])
        variants[name] = result | {"checkpoint_sha256": _sha256(path)}
        print(
            name,
            {
                k: result[k]
                for k in (
                    "plantdoc_f1",
                    "pv_f1",
                    "pooled_f1",
                    "min_recall",
                    "threshold",
                )
            },
            flush=True,
        )
    winner = select_candidate(variants, lock["gate"])
    _atomic_save(
        output / "decision.json",
        {
            "schema_version": 1,
            "lock_sha256": _sha256(lock_path),
            "winner": winner,
            "variants": variants,
            "eligible_train_count": len(weights),
            "training_draws": EPOCH_SAMPLES * max(CHECKPOINTS),
            "validation_count": len(val["ids"]),
            "validation_ids": val["ids"],
            "gpu": gpu | {"peak_reserved_bytes": torch.cuda.max_memory_reserved()},
            "test_opened": False,
        },
    )
    print(f"P04 winner: {winner}", flush=True)


if __name__ == "__main__":
    with TemporaryDirectory(prefix="tomato-p04-") as config_dir:
        os.environ.update(YOLO_CONFIG_DIR=config_dir, MPLCONFIGDIR=config_dir)
        main()
