"""P05 weight interpolation between E10 and descendants of the same lineage."""

import os
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch

from ai.data import _sha256
from ai.experiments import _completed_run
from ai.train import _atomic_save

from .head_recovery import (
    HERE,
    THREADS,
    VENDOR_CHECK,
    collect,
    load_run,
    select_candidate,
    summarize,
)

MASSES = (0.25, 0.50)


def interpolate(
    base: dict[str, torch.Tensor], adapted: dict[str, torch.Tensor], mass: float
) -> dict[str, torch.Tensor]:
    if base.keys() != adapted.keys() or not 0 < mass < 1:
        raise ValueError("incompatible checkpoints or mass")
    result = {}
    for key, a in base.items():
        b = adapted[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"incompatible P05 tensor: {key}")
        if torch.equal(a, b):
            value = a
        elif (
            key.endswith(("running_mean", "running_var", "num_batches_tracked"))
            or not a.is_floating_point()
        ):
            raise ValueError(f"changed fixed P05 buffer: {key}")
        else:
            value = torch.lerp(a, b, mass)
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite P05 tensor: {key}")
        result[key] = value
    return result


def main() -> None:
    check = import_module(VENDOR_CHECK)
    lock_path = HERE / "p05-lock.json"
    lock = check.read_json(lock_path)
    for name, digest in lock["files_sha256"].items():
        if _sha256(Path(name)) != digest:
            raise ValueError(f"P05 input hash mismatch: {name}")
    check.verify_protocol()
    gpu = check.check_gpu()
    torch.set_num_threads(THREADS)
    output = HERE / "P05"
    output.mkdir(exist_ok=False)
    model, config = load_run("E10")
    base = _completed_run(Path("ai/runs/E10"))[1]["model"]
    archive = check.read_json(Path("ai/runs/E10/validation-predictions.json"))[
        "predictions"
    ]
    original_ids = [row["sample_id"] for row in archive]
    targets = [row["target_index"] for row in archive]
    variants: dict[str, Any] = {}
    winner: str | None = None
    for epoch in (2, 4):
        path = HERE / f"P04/epoch-{epoch}.pth"
        adapted = torch.load(path, map_location="cpu", weights_only=True)
        if (
            _sha256(path)
            != lock["files_sha256"][path.relative_to(Path.cwd()).as_posix()]
        ):
            raise ValueError("P04 checkpoint changed during load")
        for mass in MASSES:
            name = f"e{epoch}-{int(mass * 100)}"
            state = interpolate(base, adapted, mass)
            model.load_state_dict(state, strict=True)
            val = collect(model, config, "val", None)
            if val["ids"] != original_ids or val["labels"].tolist() != targets:
                raise ValueError("P05 validation membership mismatch")
            if torch.cuda.max_memory_reserved() > 3.2 * 2**30:
                raise ValueError("P05 GPU budget exceeded")
            logits_path = output / f"{name}-logits.json"
            _atomic_save(logits_path, val["logits"].tolist())
            result = summarize(val, val["logits"], config["class_ids"])
            variants[name] = result | {"logits_sha256": _sha256(logits_path)}
            selected = select_candidate(variants, lock["gate"])
            if selected != winner:
                if selected != name:
                    raise ValueError("unexpected selection order")
                _atomic_save(output / "model.pth", state)
                winner = selected
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
    _atomic_save(
        output / "decision.json",
        {
            "schema_version": 1,
            "lock_sha256": _sha256(lock_path),
            "winner": winner,
            "weights_sha256": _sha256(output / "model.pth") if winner else None,
            "variants": variants,
            "validation_ids": original_ids,
            "validation_targets": targets,
            "class_ids": config["class_ids"],
            "gpu": gpu | {"peak_reserved_bytes": torch.cuda.max_memory_reserved()},
            "test_opened": False,
        },
    )
    print(f"P05 winner: {winner}", flush=True)


if __name__ == "__main__":
    with TemporaryDirectory(prefix="tomato-p05-") as config_dir:
        os.environ.update(YOLO_CONFIG_DIR=config_dir, MPLCONFIGDIR=config_dir)
        main()
