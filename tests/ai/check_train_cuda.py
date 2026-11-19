"""Assert-based M4.3 CUDA gate; run with .venv-torch Python -I -B and --out."""

import argparse
import copy
import gc
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from ai.models import build_model
from ai.train import configure_stage, train_epoch

DEVICE = torch.device("cuda")


def check_amp_math() -> dict[str, Any]:
    class ObservedLinear(nn.Linear):
        def __init__(self) -> None:
            super().__init__(2, 2, device=DEVICE)
            self.output_dtypes: list[torch.dtype] = []

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            output = super().forward(inputs)
            self.output_dtypes.append(output.dtype)
            return output

    model = ObservedLinear()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2, -0.1], [-0.3, 0.4]], device=DEVICE))
        model.bias.copy_(torch.tensor([0.05, -0.05], device=DEVICE))
    reference = copy.deepcopy(model)
    inputs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5], [0.5, -1.0]]
    )
    targets = torch.tensor([0, 1, 0, 1, 0])
    batches = [(inputs[a:b], targets[a:b], None) for a, b in ((0, 2), (2, 4), (4, 5))]
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    reference_loss = 0.0
    for a, b in ((0, 4), (4, 5)):
        loss = nn.functional.cross_entropy(
            reference(inputs[a:b].to(DEVICE)), targets[a:b].to(DEVICE)
        )
        reference_loss += loss.item() * (b - a)
        torch.autograd.backward(loss)
        reference_optimizer.step()
        reference_optimizer.zero_grad()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
    metrics = train_epoch(
        model,
        batches,
        optimizer,
        device=DEVICE,
        accumulation=2,
        amp=True,
        scaler=scaler,
    )
    assert model.output_dtypes == [torch.float16] * 3
    assert all(
        p.dtype == torch.float32 and torch.isfinite(p).all() and p.grad is None
        for p in model.parameters()
    )
    assert metrics["optimizer_steps"] == 2.0 and metrics["skipped_steps"] == 0.0
    assert math.isfinite(metrics["loss"])
    # FP16 matmul rounding permits absolute 2e-4 and relative 2e-3 error.
    atol, rtol = 2e-4, 2e-3
    torch.testing.assert_close(
        model.state_dict(), reference.state_dict(), atol=atol, rtol=rtol
    )
    assert math.isclose(metrics["loss"], reference_loss / 5, abs_tol=atol, rel_tol=rtol)
    return {
        "metrics": metrics,
        "atol": atol,
        "rtol": rtol,
        "output_dtype": "float16",
        "parameter_dtype": "float32",
    }


def check_overflow_recovery() -> dict[str, Any]:
    model = nn.Linear(1, 2, device=DEVICE)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0**18)
    scaler_identity = id(scaler)
    before = copy.deepcopy(model.state_dict())
    batches = [(torch.ones(1, 1), torch.zeros(1, dtype=torch.long), None)]
    attempts: list[dict[str, Any]] = []
    for attempt in range(12):
        old_scale = scaler.get_scale()
        metrics = train_epoch(
            model,
            batches,
            optimizer,
            device=DEVICE,
            accumulation=1,
            amp=True,
            scaler=scaler,
        )
        attempts.append(
            {
                "attempt": attempt,
                "scale_before": old_scale,
                "scale_after": scaler.get_scale(),
                **metrics,
            }
        )
        assert id(scaler) == scaler_identity
        assert math.isfinite(metrics["loss"])
        if metrics["optimizer_steps"]:
            assert (
                attempt > 0
                and metrics["optimizer_steps"] == 1.0
                and metrics["skipped_steps"] == 0.0
            )
            assert optimizer.state
            assert any(
                not torch.equal(p, before[n]) for n, p in model.named_parameters()
            )
            assert all(
                torch.isfinite(p).all() and p.grad is None for p in model.parameters()
            )
            break
        assert metrics["optimizer_steps"] == 0.0 and metrics["skipped_steps"] == 1.0
        torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
        assert not optimizer.state
        assert all(p.grad is None for p in model.parameters())
        assert scaler.get_scale() < old_scale
    else:
        raise AssertionError("real AMP scaler never recovered within 12 calls")
    assert attempts[0]["skipped_steps"] == 1.0
    state = scaler.state_dict().copy()
    scale = scaler.get_scale()
    continued = train_epoch(
        model,
        batches,
        optimizer,
        device=DEVICE,
        accumulation=1,
        amp=True,
        scaler=scaler,
    )
    assert continued["optimizer_steps"] == 1.0 and continued["skipped_steps"] == 0.0
    assert scaler.get_scale() == scale and scaler.state_dict() != state
    # LayerNorm keeps the backward path FP32, safe at the default scale 65536.
    default_model = nn.LayerNorm(2, device=DEVICE)
    default_inputs = torch.tensor([[1.0, -1.0]], device=DEVICE)
    default_targets = torch.zeros(1, dtype=torch.long, device=DEVICE)
    default_reference = copy.deepcopy(default_model)
    default_optimizer = torch.optim.SGD(default_model.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(default_reference.parameters(), lr=0.1)
    reference_loss = nn.functional.cross_entropy(
        default_reference(default_inputs), default_targets
    )
    torch.autograd.backward(reference_loss)
    reference_optimizer.step()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        assert default_model(default_inputs).dtype == torch.float32
    default_scaler_metrics = train_epoch(
        default_model,
        [(default_inputs, default_targets, None)],
        default_optimizer,
        device=DEVICE,
        accumulation=1,
        amp=True,
    )
    assert default_scaler_metrics["optimizer_steps"] == 1.0
    assert default_scaler_metrics["skipped_steps"] == 0.0
    assert default_scaler_metrics["loss"] == reference_loss.item()
    torch.testing.assert_close(
        default_model.state_dict(), default_reference.state_dict(), rtol=0, atol=0
    )
    assert all(p.grad is None for p in default_model.parameters())
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    window_optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # The preceding recovery established scale as finite and 2*scale as overflow.
    window_scale = scale * 4
    window_scaler = torch.amp.GradScaler("cuda", init_scale=window_scale)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        window_reference_loss = nn.functional.cross_entropy(
            model(batches[0][0].to(DEVICE)), batches[0][1].to(DEVICE), reduction="sum"
        ).item()
    mixed_windows = train_epoch(
        model,
        batches * 3,
        window_optimizer,
        device=DEVICE,
        accumulation=1,
        amp=True,
        scaler=window_scaler,
    )
    assert mixed_windows["optimizer_steps"] == 1.0
    assert mixed_windows["skipped_steps"] == 2.0
    assert mixed_windows["loss"] == window_reference_loss
    assert window_scaler.get_scale() == scale
    return {
        "attempts": attempts,
        "continued": continued,
        "persistent_scaler_state": scaler.state_dict(),
        "default_scaler": default_scaler_metrics,
        "mixed_windows": mixed_windows,
        "mixed_window_initial_scale": window_scale,
        "mixed_window_reference_loss": window_reference_loss,
    }


def check_unscale_overflow() -> dict[str, Any]:
    model = nn.Linear(1, 2, device=DEVICE)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda", init_scale=0.25)
    before = copy.deepcopy(model.state_dict())
    observed: list[bool] = []

    def finite_large_gradient(gradient: torch.Tensor) -> torch.Tensor:
        result = torch.full_like(gradient, torch.finfo(torch.float32).max * 0.75)
        observed.append(bool(torch.isfinite(result).all()))
        return result

    handle = model.weight.register_hook(finite_large_gradient)  # type: ignore[no-untyped-call]
    try:
        train_epoch(
            model,
            [(torch.ones(1, 1), torch.zeros(1, dtype=torch.long), None)],
            optimizer,
            device=DEVICE,
            accumulation=1,
            amp=True,
            scaler=scaler,
        )
    except FloatingPointError as error:
        assert "normalization" in str(error)
    else:
        raise AssertionError("finite scaled gradient overflow after unscale must fail")
    finally:
        handle.remove()
    assert observed == [True]
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    assert not optimizer.state
    assert all(p.grad is None for p in model.parameters())
    return {"scale": 0.25, "scaled_gradient_finite": True, "raised_before_step": True}


def check_backbone_stage(architecture: str, stage: str) -> dict[str, Any]:
    model = build_model(architecture, 3, pretrained_path=None).to(DEVICE)
    groups = configure_stage(model, architecture, stage)
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
    modes = {n: m.training for n, m in model.named_modules()}
    identities = {n: id(p) for n, p in model.named_parameters()}
    before = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    buffers = {n: b.cpu().clone() for n, b in model.named_buffers()}
    inputs = torch.randn(5, 3, 32, 32)
    targets = torch.tensor([0, 1, 2, 0, 2])
    batches = [(inputs[a:b], targets[a:b], None) for a, b in ((0, 2), (2, 4), (4, 5))]
    metrics = train_epoch(
        model,
        batches,
        optimizer,
        device=DEVICE,
        accumulation=2,
        amp=True,
        scaler=scaler,
    )
    assert metrics["optimizer_steps"] == 2.0 and metrics["skipped_steps"] == 0.0
    assert math.isfinite(metrics["loss"])
    assert {n: m.training for n, m in model.named_modules()} == modes
    assert {n: id(p) for n, p in model.named_parameters()} == identities
    assert all(torch.equal(b.cpu(), buffers[n]) for n, b in model.named_buffers())
    assert all(
        not module.training
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )
    names = {id(p): n for n, p in model.named_parameters()}
    for n, p in model.named_parameters():
        assert torch.isfinite(p).all() and p.grad is None
        if not p.requires_grad:
            assert torch.equal(p.cpu(), before[n])
    for group in optimizer.param_groups:
        assert any(
            not torch.equal(p.cpu(), before[names[id(p)]]) for p in group["params"]
        )
        assert all(optimizer.state[p]["step"].item() == 2 for p in group["params"])
    return {
        "architecture": architecture,
        "stage": stage,
        "metrics": metrics,
        "bn_and_frozen_preserved": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="new JSON evidence path; refuses overwrites",
    )
    arguments = parser.parse_args()
    output: Path = arguments.out.resolve()
    log_path = output.with_suffix(".log")
    if output == log_path or output.exists() or log_path.exists():
        raise FileExistsError(
            f"evidence already exists or JSON/log paths collide: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "status": "running",
        "cases": {},
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    with (
        output.open("x", encoding="utf-8") as json_stream,
        log_path.open("x", encoding="utf-8") as log_stream,
    ):
        try:
            assert torch.cuda.is_available(), "CUDA is required; CPU is not a CUDA pass"
            torch.set_num_threads(2)
            torch.manual_seed(42)
            result["device"] = torch.cuda.get_device_name(DEVICE)
            cases = [
                ("amp_math", check_amp_math),
                ("overflow_recovery", check_overflow_recovery),
                ("unscale_overflow", check_unscale_overflow),
            ]
            for name, check in cases:
                result["cases"][name] = check()
                print(f"PASS {name}", file=log_stream, flush=True)
                print(f"PASS {name}", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
            for architecture in ("efficientnet_b0", "resnet50"):
                for stage in ("frozen", "fine_tune"):
                    name = f"{architecture}_{stage}"
                    result["cases"][name] = check_backbone_stage(architecture, stage)
                    print(f"PASS {name}", file=log_stream, flush=True)
                    print(f"PASS {name}", flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
            result["status"] = "passed"
        except BaseException:
            result["status"] = "failed"
            result["error"] = traceback.format_exc()
            print(result["error"], file=log_stream, flush=True)
            raise
        finally:
            json.dump(result, json_stream, indent=2, allow_nan=False)
            json_stream.write("\n")


if __name__ == "__main__":
    main()
