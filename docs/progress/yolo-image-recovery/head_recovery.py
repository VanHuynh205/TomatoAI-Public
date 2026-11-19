"""P03 frozen YOLO head experiment on sealed train and validation splits."""

import math
import os
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ai.data import ManifestDataset, _sha256
from ai.evaluate import compute_metrics
from ai.experiments import _completed_run, select_threshold
from ai.models import build_model
from ai.preprocessing import build_transform
from ai.train import _atomic_save

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE.parent / "yolo-image"
VENDOR_CHECK = "docs.progress.2026-11-08-yolo-task5.run_task5"
SOURCES = ("pv:", "plantdoc:")
BATCH = 8
RIDGE = 0.01
ITERATIONS = 100
TEMPERATURE = 2
THREADS = 4
VARIANTS = {
    f"{method}-{int(mass * 100)}": (mass, method == "kd")
    for mass in (0.25, 0.40, 0.55)
    for method in ("ce", "kd")
}


def sample_weights(
    ids: list[str], labels: torch.Tensor, plantdoc_mass: float
) -> torch.Tensor:
    if len(ids) != len(labels) or not 0 < plantdoc_mass < 1:
        raise ValueError("invalid source balance inputs")
    if any(not sid.startswith(SOURCES) for sid in ids):
        raise ValueError("unknown sample source")
    weights = torch.empty(len(labels), dtype=torch.float64)
    for prefix, mass in zip(SOURCES, (1 - plantdoc_mass, plantdoc_mass), strict=True):
        mask = torch.tensor([sid.startswith(prefix) for sid in ids])
        if not mask.any():
            raise ValueError(f"empty sample source: {prefix}")
        counts = torch.bincount(labels[mask])
        factors = counts[labels[mask]].double().rsqrt()
        weights[mask] = factors * (mass / factors.sum())
    return weights


def fold_head(
    weight: torch.Tensor, bias: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    folded = weight / scale
    return folded, bias - folded @ mean


def select_candidate(variants: dict[str, Any], gate: dict[str, float]) -> str | None:
    return max(
        (
            name
            for name, m in variants.items()
            if m["plantdoc_f1"]
            > gate["plantdoc_supported_macro_f1_strictly_greater_than"]
            and m["pv_f1"] >= gate["pv_macro_f1_minimum"]
            and m["pooled_f1"] >= 0.85
            and m["min_recall"] >= 0.70
            and m["threshold"] is not None
        ),
        key=lambda name: tuple(
            variants[name][k] for k in ("plantdoc_f1", "pv_f1", "pooled_f1")
        ),
        default=None,
    )


def collect(
    model: nn.Module, config: dict[str, Any], split: str, head: nn.Linear | None
) -> dict[str, Any]:
    if split not in ("train", "val"):
        raise ValueError("P03 may only open train and val")
    dataset = ManifestDataset(
        Path(config["dataset_dir"]),
        split,
        build_transform(config["preprocessing"]["spec"]),
    )
    loader: Any = DataLoader(dataset, batch_size=BATCH)
    ids: list[str] = []
    targets: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    hook = (
        None
        if head is None
        else head.register_forward_pre_hook(
            lambda _layer, args: features.append(args[0].cpu())
        )
    )
    try:
        with torch.inference_mode():
            for images, labels, sample_ids in loader:
                logits.append(model(images.to("cuda")).cpu())
                ids.extend(sample_ids)
                targets.append(labels)
                if len(logits) % 250 == 0:
                    print(
                        f"{config['run_id']} {split}: {len(ids)}/{len(dataset)}",
                        flush=True,
                    )
    finally:
        if hook is not None:
            hook.remove()
    labels = torch.cat(targets)
    expected = [
        (r["sample_id"], config["class_ids"].index(r["class_id"]))
        for r in dataset._rows
    ]
    expected_count = {"train": 8238, "val": 1785}[split]
    if (
        len(ids) != expected_count
        or list(zip(ids, labels.tolist(), strict=True)) != expected
    ):
        raise ValueError("P03 membership/order mismatch")
    return {
        "ids": ids,
        "labels": labels,
        "logits": torch.cat(logits),
        "features": torch.cat(features) if features else None,
    }


def fit(
    z: torch.Tensor,
    labels: torch.Tensor,
    sample_weight: torch.Tensor,
    initial_w: torch.Tensor,
    initial_b: torch.Tensor,
    teacher: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    w, b = initial_w.clone().requires_grad_(), initial_b.clone().requires_grad_()
    optimizer = torch.optim.LBFGS(
        [w, b], max_iter=ITERATIONS, line_search_fn="strong_wolfe"
    )
    soft_targets = None if teacher is None else F.softmax(teacher / TEMPERATURE, dim=1)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = F.linear(z, w, b)
        loss = F.cross_entropy(logits, labels, reduction="none")
        if soft_targets is not None:
            kd = F.kl_div(
                F.log_softmax(logits / TEMPERATURE, dim=1),
                soft_targets,
                reduction="none",
            ).sum(1)
            loss = (loss + kd * TEMPERATURE**2) / 2
        ridge = (w - initial_w).square().sum() + (b - initial_b).square().sum()
        objective = sample_weight @ loss + RIDGE * ridge
        objective.backward()
        return objective

    optimizer.step(closure)
    return w.detach(), b.detach()


def summarize(
    val: dict[str, Any], logits: torch.Tensor, class_ids: list[str]
) -> dict[str, Any]:
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite P03 validation logits")
    scores, predicted = logits.softmax(1).max(1)
    targets = val["labels"].tolist()
    preds, confidence = predicted.tolist(), scores.tolist()
    threshold = select_threshold(
        confidence, [p == t for p, t in zip(preds, targets, strict=True)]
    )
    groups = {"pooled": list(range(len(targets)))}
    for source in SOURCES:
        groups[source[:-1]] = [
            i for i, sid in enumerate(val["ids"]) if sid.startswith(source)
        ]

    metrics = {
        name: compute_metrics(
            [targets[i] for i in indices],
            [preds[i] for i in indices],
            [
                confidence[i] >= (threshold if threshold is not None else 0.5)
                for i in indices
            ],
            class_ids,
        )
        for name, indices in groups.items()
    }
    supported = [
        c["f1"] for c in metrics["plantdoc"]["per_class"].values() if c["evaluable"]
    ]
    return {
        "plantdoc_f1": math.fsum(supported) / len(supported),
        "pv_f1": metrics["pv"]["macro_f1"],
        "pooled_f1": metrics["pooled"]["macro_f1"],
        "min_recall": min(c["recall"] for c in metrics["pooled"]["per_class"].values()),
        "threshold": threshold,
        "metrics_threshold": threshold if threshold is not None else 0.5,
        "metrics": metrics,
    }


def load_run(name: str) -> tuple[nn.Module, dict[str, Any]]:
    _, best, _, _ = _completed_run(Path("ai/runs") / name)
    config = best["config"]
    model = build_model(config["architecture"], len(config["class_ids"]))
    model.load_state_dict(best["model"], strict=True)
    return model.eval().to("cuda"), config


def main() -> None:
    check = import_module(VENDOR_CHECK)
    lock = check.read_json(HERE / "p03-lock.json")
    for name, digest in lock["files_sha256"].items():
        if _sha256(Path(name)) != digest:
            raise ValueError(f"P03 input hash mismatch: {name}")
    check.verify_protocol()
    gpu = check.check_gpu()
    print(f"Locked inputs verified. GPU: {gpu}", flush=True)
    torch.set_num_threads(THREADS)
    output = HERE / "P03"
    output.mkdir(exist_ok=False)
    e10, config = load_run("E10")
    head = e10.model.model[-1].linear
    if not isinstance(head, nn.Linear) or (head.in_features, head.out_features) != (
        1280,
        8,
    ):
        raise ValueError("unexpected E10 head")
    original_w = head.weight.detach().cpu().double()
    original_b = head.bias.detach().cpu().double()
    train = collect(e10, config, "train", head)
    val = collect(e10, config, "val", head)
    saved = check.read_json(Path("ai/runs/E10/validation-predictions.json"))[
        "predictions"
    ]
    if [r["sample_id"] for r in saved] != val["ids"] or [
        r["target_index"] for r in saved
    ] != val["labels"].tolist():
        raise ValueError("E10 archive membership mismatch")
    saved_logits = torch.tensor([r["logits"] for r in saved])
    torch.testing.assert_close(val["logits"], saved_logits, rtol=1e-5, atol=5e-4)
    if not torch.equal(val["logits"].argmax(1), saved_logits.argmax(1)):
        raise ValueError("E10 archive top-1 mismatch")
    print("E10 train/val extracted. Archived validation reproduced.", flush=True)
    del head, e10
    teacher, teacher_config = load_run("E08")
    teacher_train = collect(teacher, teacher_config, "train", None)
    del teacher
    if (
        teacher_train["ids"] != train["ids"]
        or not torch.equal(teacher_train["labels"], train["labels"])
        or teacher_config["class_ids"] != config["class_ids"]
        or not torch.isfinite(teacher_train["logits"]).all()
    ):
        raise ValueError("teacher train labels/order/values mismatch")
    peak_reserved = torch.cuda.max_memory_reserved()
    if peak_reserved > 3.2 * 2**30:
        raise ValueError("P03 GPU budget exceeded")
    torch.cuda.empty_cache()
    x = train["features"].double()
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-4)
    z = (x - mean) / scale
    initial_w, initial_b = original_w * scale, original_b + original_w @ mean
    val_z = (val["features"].double() - mean) / scale
    variants: dict[str, Any] = {}
    heads: dict[str, dict[str, torch.Tensor]] = {}
    for name, (mass, kd) in VARIANTS.items():
        weights = sample_weights(train["ids"], train["labels"], mass)
        w, b = fit(
            z,
            train["labels"],
            weights,
            initial_w,
            initial_b,
            teacher_train["logits"].double() if kd else None,
        )
        folded_w, folded_b = fold_head(w, b, mean, scale)
        logits = F.linear(val["features"], folded_w.float(), folded_b.float())
        torch.testing.assert_close(
            logits.double(), F.linear(val_z, w, b), rtol=1e-5, atol=5e-4
        )
        result = summarize(val, logits, config["class_ids"])
        variants[name] = result
        heads[name] = {"weight": folded_w.float(), "bias": folded_b.float()}
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
    if winner is not None:
        _atomic_save(output / "head.pth", heads[winner])
    _atomic_save(
        output / "decision.json",
        {
            "schema_version": 1,
            "lock_sha256": _sha256(HERE / "p03-lock.json"),
            "head_sha256": _sha256(output / "head.pth") if winner else None,
            "winner": winner,
            "variants": variants,
            "train_count": len(train["ids"]),
            "validation_count": len(val["ids"]),
            "validation_ids": val["ids"],
            "gpu": gpu | {"peak_reserved_bytes": peak_reserved},
            "archive_max_logit_delta": float(
                (val["logits"] - saved_logits).abs().max()
            ),
            "test_opened": False,
        },
    )
    print(f"P03 winner: {winner}", flush=True)


if __name__ == "__main__":
    with TemporaryDirectory(prefix="tomato-p03-") as config_dir:
        os.environ.update(YOLO_CONFIG_DIR=config_dir, MPLCONFIGDIR=config_dir)
        main()
