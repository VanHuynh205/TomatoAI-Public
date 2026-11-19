"""Leak-free 3-way splitting, water-filling sampler, C0 schedule, EMA, and export trainer."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from PIL import Image, ImageOps
from safetensors.torch import save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T  # type: ignore[import-untyped]

from ai.calibration import (
    acceptance_mask,
    fit_temperature,
    select_coverage_threshold,
)
from ai.cnn_gru_inference import (
    CNN_GRU_FILES,
    CNN_GRU_MANIFEST,
    EXPORT_LIMITS,
    MAX_WEIGHT_BYTES,
    validate_cnn_gru_export,
)
from ai.cnn_gru_model import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SUPPORTED_BACKBONES,
    SpatialBiGruClassifier,
    build_cnn_gru_model,
    cnn_gru_eval_transform,
)
from ai.data import MAX_PIXELS
from ai.dinov3_field import CLASSES, save_atomic

DEFAULT_CAPS: dict[str, float] = {
    "field": 1.40,
    "other": 1.50,
    "controlled": 0.85,
    "negative": 1.05,
}
_DOMAIN_SHARES: dict[str, float] = {"field": 0.75, "controlled": 0.15, "other": 0.10}
_NON_LEAF_INDEX = len(CLASSES) - 1


def _det_hash(seed: str, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _domain_bucket(row: dict[str, str]) -> str:
    if row["class_id"] == "not_tomato_leaf":
        return "negative"
    domain = row.get("domain", "")
    return domain if domain in ("field", "controlled") else "other"


def build_three_way_split(
    rows: list[dict[str, str]],
    seed: str = "tomato-cnn-bigru-v1",
    ratios: tuple[float, float, float] = (0.74, 0.13, 0.13),
) -> dict[str, str]:
    """Deterministically partition groups into train / val_select / val_calib with zero leakage."""
    if not rows:
        raise ValueError("rows must not be empty")
    if (
        len(ratios) != 3
        or any(r <= 0 or not math.isfinite(r) for r in ratios)
        or not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise ValueError("ratios must be three positive numbers summing to 1")

    groups = sorted({row["group"] for row in rows})
    if any(not g for g in groups):
        raise ValueError("every row must define a nonempty group")
    group_idx = {g: i for i, g in enumerate(groups)}
    parent = list(range(len(groups)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    seen_keys: dict[tuple[str, str], int] = {}
    for row in rows:
        idx = group_idx[row["group"]]
        for field in ("sha256", "original_sha256", "normalized_sha256", "path"):
            val = row.get(field, "")
            if val:
                prev = seen_keys.setdefault((field, val), idx)
                union(idx, prev)
        # Keep bapari_2025_v1 split between val_select/val_calib while preventing any non-bapari split_block leakage.
        if row.get("source") != "bapari_2025_v1":
            sb = row.get("split_block", "")
            if sb:
                prev = seen_keys.setdefault(("split_block", sb), idx)
                union(idx, prev)

    comp_groups: defaultdict[int, list[str]] = defaultdict(list)
    for g, i in group_idx.items():
        comp_groups[find(i)].append(g)
    comp_rep = {root: min(gs) for root, gs in comp_groups.items()}
    comp_rows: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rep = comp_rep[find(group_idx[row["group"]])]
        comp_rows[rep].append(row)

    rep_assignment: dict[str, str] = {}
    bap_reps: list[str] = []
    for rep, rep_list in sorted(comp_rows.items()):
        sources = {r.get("source", "") for r in rep_list}
        if "hossain_v1" in sources and "bapari_2025_v1" in sources:
            raise ValueError("Component mixes hossain_v1 and bapari_2025_v1")
        if "hossain_v1" in sources:
            rep_assignment[rep] = "train"
        elif "bapari_2025_v1" in sources:
            bap_reps.append(rep)

    bap_reps.sort(key=lambda rep: (-len(comp_rows[rep]), _det_hash(seed, rep)))
    bap_counts: dict[str, Counter[str]] = {
        "val_select": Counter(),
        "val_calib": Counter(),
    }
    bap_tot = Counter(r["class_id"] for rep in bap_reps for r in comp_rows[rep])
    bap_weights = (
        ("bacterial_spot", 4.0),
        ("septoria_leaf_spot", 3.0),
        ("healthy", 1.0),
    )
    for rep in bap_reps:
        gc = Counter(r["class_id"] for r in comp_rows[rep])

        def pair_imbalance(
            split_name: str, gc: Counter[str] = gc, rep: str = rep
        ) -> tuple[float, int, str]:
            other = "val_calib" if split_name == "val_select" else "val_select"
            c_sp = bap_counts[split_name] + gc
            c_ot = bap_counts[other]
            loss = sum(
                w * (((c_sp[c] - c_ot[c]) / max(bap_tot[c], 1)) ** 2)
                for c, w in bap_weights
            )
            return (
                round(loss, 12),
                sum(bap_counts[split_name].values()),
                _det_hash(seed, f"{split_name}:{rep}"),
            )

        chosen = min(("val_select", "val_calib"), key=pair_imbalance)
        rep_assignment[rep] = chosen
        bap_counts[chosen].update(gc)

    strata: defaultdict[tuple[str, str, str], list[str]] = defaultdict(list)
    for rep, rep_list in comp_rows.items():
        if rep not in rep_assignment:
            key = min(
                (r.get("source", ""), r.get("domain", ""), r["class_id"])
                for r in rep_list
            )
            strata[key].append(rep)

    r_train, r_sel, r_cal = ratios
    for _, reps in sorted(strata.items()):
        if len(reps) < 6:
            for rep in reps:
                rep_assignment[rep] = "train"
            continue
        min_vg = 2 if len(reps) >= 10 else 1
        tot = sum(len(comp_rows[rep]) for rep in reps)
        targets = {
            "train": r_train * tot,
            "val_select": r_sel * tot,
            "val_calib": r_cal * tot,
        }
        cur_i: Counter[str] = Counter()
        cur_g: Counter[str] = Counter()
        for idx, rep in enumerate(sorted(reps, key=lambda x: _det_hash(seed, x))):
            rem = len(reps) - idx
            defs: dict[str, int] = (
                {} if cur_g["train"] >= 2 else {"train": 2 - cur_g["train"]}
            )
            for sp in ("val_select", "val_calib"):
                if cur_g[sp] < min_vg:
                    defs[sp] = min_vg - cur_g[sp]
            cands = (
                list(defs)
                if sum(defs.values()) >= rem
                else ["train", "val_select", "val_calib"]
            )
            best = max(
                cands,
                key=lambda sp: (
                    (targets[sp] - cur_i[sp]) / max(targets[sp], 1e-9),
                    -cur_g[sp],
                    _det_hash(seed, f"{sp}:{rep}"),
                ),
            )
            rep_assignment[rep] = best
            cur_i[best] += len(comp_rows[rep])
            cur_g[best] += 1

    return {
        group: rep_assignment[comp_rep[find(idx)]] for group, idx in group_idx.items()
    }


def _water_fill(
    raw_weights: Sequence[float], caps: Sequence[float], target: float
) -> list[float]:
    weights = [0.0] * len(raw_weights)
    remaining = target
    active = list(range(len(raw_weights)))
    while active and remaining > 1e-12:
        total_raw = sum(raw_weights[i] for i in active)
        if total_raw <= 0:
            break
        scale = remaining / total_raw
        capped = [
            i for i in active if weights[i] + raw_weights[i] * scale >= caps[i] - 1e-12
        ]
        if not capped:
            for i in active:
                weights[i] += raw_weights[i] * scale
            remaining = 0.0
            break
        capped_set = set(capped)
        for i in capped:
            remaining -= caps[i] - weights[i]
            weights[i] = caps[i]
        active = [i for i in active if i not in capped_set]
    return weights


def cnn_bigru_weights_strict(
    rows: list[dict[str, str]],
    samples: int = 4800,
    *,
    caps: dict[str, float] | None = None,
) -> tuple[list[float], list[float]]:
    """Compute water-filling expected draw counts and exact log-prior offsets."""
    if not rows:
        raise ValueError("rows must not be empty")
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be a positive integer")
    active_caps = dict(DEFAULT_CAPS if caps is None else caps)
    if set(active_caps) != set(DEFAULT_CAPS) or any(
        isinstance(v, bool)
        or not isinstance(v, (int, float))
        or not math.isfinite(v)
        or v <= 0
        for v in active_caps.values()
    ):
        raise ValueError(
            "caps must define positive finite values for all domain buckets"
        )

    by_cd: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    total_cap = 0.0
    for idx, row in enumerate(rows):
        if row.get("class_id") not in CLASSES:
            raise ValueError(f"Unknown class_id: {row.get('class_id')}")
        dom = _domain_bucket(row)
        by_cd[(row["class_id"], dom)].append(idx)
        total_cap += float(active_caps[dom])

    if total_cap < samples - 1e-9:
        raise ValueError("Requested sample budget exceeds total domain capacity")

    cd_cap = {
        (cls_name, dom): len(idxs) * float(active_caps[dom])
        for (cls_name, dom), idxs in by_cd.items()
    }
    cd_budget: defaultdict[tuple[str, str], float] = defaultdict(float)

    present_classes = [
        cls_name
        for cls_name in CLASSES
        if any((cls_name, d) in by_cd for d in active_caps)
    ]
    base = samples / len(present_classes)
    for cls_name in present_classes:
        if cls_name == "not_tomato_leaf":
            cd_budget[(cls_name, "negative")] = min(
                base, cd_cap[(cls_name, "negative")]
            )
        else:
            doms = [
                d for d in ("field", "other", "controlled") if (cls_name, d) in by_cd
            ]
            d_caps = [cd_cap[(cls_name, d)] for d in doms]
            alloc = _water_fill(
                [_DOMAIN_SHARES[d] for d in doms], d_caps, min(base, sum(d_caps))
            )
            for d, val in zip(doms, alloc, strict=True):
                cd_budget[(cls_name, d)] = val

    for target_doms in (("field", "negative"), ("other", "controlled")):
        excess = samples - sum(cd_budget.values())
        if excess <= 1e-12:
            break
        eligible: list[tuple[str, str]] = []
        for cls_name in present_classes:
            candidates = (
                ("negative",)
                if cls_name == "not_tomato_leaf"
                else ("field", "other", "controlled")
            )
            for cand_d in candidates:
                if (
                    cand_d in target_doms
                    and (cls_name, cand_d) in by_cd
                    and cd_cap[(cls_name, cand_d)] - cd_budget[(cls_name, cand_d)]
                    > 1e-12
                ):
                    eligible.append((cls_name, cand_d))
        if eligible:
            headrooms = [cd_cap[k] - cd_budget[k] for k in eligible]
            add = _water_fill(
                [1.0] * len(eligible), headrooms, min(excess, sum(headrooms))
            )
            for k, v in zip(eligible, add, strict=True):
                cd_budget[k] += v

    weights = [0.0] * len(rows)
    for (cls_name, dom), idxs in by_cd.items():
        group_counts = Counter(
            (rows[i].get("source") or "", rows[i].get("group") or str(i)) for i in idxs
        )
        source_totals: defaultdict[str, float] = defaultdict(float)
        for (source, _), count in group_counts.items():
            source_totals[source] += math.sqrt(count)
        raw = [
            1.0
            / (
                source_totals[rows[i].get("source") or ""]
                * math.sqrt(
                    group_counts[
                        (rows[i].get("source") or "", rows[i].get("group") or str(i))
                    ]
                )
            )
            for i in idxs
        ]
        allocated = _water_fill(
            raw, [float(active_caps[dom])] * len(idxs), cd_budget[(cls_name, dom)]
        )
        for idx, val in zip(idxs, allocated, strict=True):
            weights[idx] = val

    class_sums: defaultdict[str, float] = defaultdict(float)
    for (cls_name, _), val in cd_budget.items():
        class_sums[cls_name] += val
    deltas = [
        math.log(max(len(CLASSES) * (class_sums[cls_name] / samples), 1e-12))
        for cls_name in CLASSES
    ]
    return weights, deltas


def prior_matched_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    log_prior_offset: torch.Tensor,
    label_smoothing: float = 0.02,
) -> torch.Tensor:
    """Prior-matched cross-entropy with prior-weighted smoothing for zero inference bias."""
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    compute_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    offset = log_prior_offset.to(device=logits.device, dtype=compute_dtype)
    shifted = logits.to(dtype=compute_dtype) + offset
    log_probs = F.log_softmax(shifted, dim=-1)
    nll = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    if label_smoothing == 0.0:
        return nll.mean()
    pi = F.softmax(offset, dim=-1)
    smooth = -(log_probs * pi).sum(dim=-1)
    return ((1.0 - label_smoothing) * nll + label_smoothing * smooth).mean()


def continuous_c0_lr(
    global_step: int,
    total_steps: int,
    head_peak_lr: float,
    head_main_lr: float,
    backbone_lrs: dict[str, float],
    *,
    stage1_steps: int = 450,
    warmup_steps: int = 150,
) -> dict[str, float]:
    """C0 continuous learning rate schedule without step jumps at stage boundaries."""
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError("total_steps must be a positive integer")
    if type(global_step) is not int or not 0 <= global_step <= total_steps:
        raise ValueError("global_step must be in [0, total_steps]")
    if type(stage1_steps) is not int or stage1_steps < 1:
        raise ValueError("stage1_steps must be a positive integer")
    if type(warmup_steps) is not int or warmup_steps < 1:
        raise ValueError("warmup_steps must be a positive integer")
    if (
        isinstance(head_peak_lr, bool)
        or not isinstance(head_peak_lr, (int, float))
        or not math.isfinite(head_peak_lr)
        or head_peak_lr <= 0
        or isinstance(head_main_lr, bool)
        or not isinstance(head_main_lr, (int, float))
        or not math.isfinite(head_main_lr)
        or head_main_lr <= 0
    ):
        raise ValueError("head learning rates must be positive and finite")
    for stage, lr_val in backbone_lrs.items():
        if (
            isinstance(lr_val, bool)
            or not isinstance(lr_val, (int, float))
            or not math.isfinite(lr_val)
            or lr_val < 0
        ):
            raise ValueError(f"invalid backbone learning rate for {stage}")

    if total_steps >= stage1_steps + warmup_steps:
        s1 = stage1_steps
        w = warmup_steps
        s2 = s1 + w
    else:
        s1 = max(1, min(stage1_steps, round(total_steps * 3 / 22)))
        w = max(1, min(warmup_steps, max(1, round(total_steps / 22))))
        s2 = min(total_steps, s1 + w)

    # Ensure Stage 1 finishes head warmup and decays to head_main_lr before Stage 2 begins.
    head_w = w if s1 > w else (max(1, s1 // 3) if s1 >= 2 else 1)
    stage2_head_start = head_main_lr if s1 > head_w else head_peak_lr

    if global_step <= head_w:
        ratio = global_step / max(head_w, 1)
        head_lr = 0.1 * head_peak_lr + 0.9 * head_peak_lr * ratio
    elif global_step <= s1:
        t = (global_step - head_w) / max(s1 - head_w, 1)
        head_lr = head_main_lr + 0.5 * (head_peak_lr - head_main_lr) * (
            1.0 + math.cos(math.pi * t)
        )
    else:
        t = (global_step - s1) / max(total_steps - s1, 1)
        head_lr = stage2_head_start * (
            0.05 + 0.5 * 0.95 * (1.0 + math.cos(math.pi * t))
        )

    result: dict[str, float] = {"head": float(head_lr)}
    for stage, target_lr in backbone_lrs.items():
        if global_step <= s1:
            lr = 0.0
        elif global_step <= s2:
            lr = target_lr * ((global_step - s1) / max(s2 - s1, 1))
        else:
            t = (global_step - s2) / max(total_steps - s2, 1)
            lr = target_lr * (0.05 + 0.5 * 0.95 * (1.0 + math.cos(math.pi * t)))
        result[stage] = float(lr)
    return result


def ema_decay(ema_updates: int, max_decay: float = 0.993) -> float:
    """Warmup EMA decay rate min(max_decay, (1 + t) / (10 + t))."""
    if type(ema_updates) is not int or ema_updates < 0:
        raise ValueError("ema_updates must be a non-negative integer")
    return min(float(max_decay), (1.0 + ema_updates) / (10.0 + ema_updates))


def _finite_metric(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        return 0.0
    return float(value)


def selection_key(metrics: dict[str, Any]) -> tuple[bool, float, float, float]:
    """Hierarchical 4-level checkpoint ranker; always returns a deterministic tuple."""
    s_select = _finite_metric(metrics.get("s_select"))
    r_healthy = _finite_metric(metrics.get("recall_nonlab_healthy"))
    p_bs = _finite_metric(metrics.get("precision_nonlab_bs"))
    r_negative = _finite_metric(metrics.get("recall_negative"))
    guardrail_passed = bool(r_healthy >= 0.85 and p_bs >= 0.70 and r_negative >= 0.90)
    return (
        guardrail_passed,
        round(s_select, 8),
        round(r_healthy, 8),
        round(p_bs, 8),
    )


def _compute_val_select_metrics(
    rows: list[dict[str, str]], predictions: list[str]
) -> dict[str, Any]:
    nonlab = [
        (r, p)
        for r, p in zip(rows, predictions, strict=True)
        if r.get("domain") != "controlled" and r["class_id"] != "not_tomato_leaf"
    ]
    healthy_gt = [p for r, p in nonlab if r["class_id"] == "healthy"]
    recall_healthy = (
        sum(1 for p in healthy_gt if p == "healthy") / len(healthy_gt)
        if healthy_gt
        else 0.0
    )

    bs_pred = [r for r, p in nonlab if p == "bacterial_spot"]
    precision_bs = (
        sum(1 for r in bs_pred if r["class_id"] == "bacterial_spot") / len(bs_pred)
        if bs_pred
        else 0.0
    )

    neg_pairs = [
        (r, p)
        for r, p in zip(rows, predictions, strict=True)
        if r["class_id"] == "not_tomato_leaf"
    ]
    recall_neg = (
        sum(1 for _, p in neg_pairs if p == "not_tomato_leaf") / len(neg_pairs)
        if neg_pairs
        else 0.0
    )

    # Source-balanced macro-F1 on non-lab tomato rows + overall accuracy / negative recall.
    sources = sorted({r.get("source") or "" for r, _ in nonlab})
    source_f1s: list[float] = []
    for source in sources:
        sub = [
            (r["class_id"], p) for r, p in nonlab if (r.get("source") or "") == source
        ]
        present = sorted({gt for gt, _ in sub})
        f1s: list[float] = []
        for cls_name in present:
            tp = sum(1 for gt, pr in sub if gt == cls_name and pr == cls_name)
            sup = sum(1 for gt, _ in sub if gt == cls_name)
            pred_cnt = sum(1 for _, pr in sub if pr == cls_name)
            f1s.append((2.0 * tp / (sup + pred_cnt)) if (sup + pred_cnt) > 0 else 0.0)
        if f1s:
            source_f1s.append(sum(f1s) / len(f1s))
    source_macro_f1 = sum(source_f1s) / len(source_f1s) if source_f1s else 0.0

    all_f1s: list[float] = []
    for cls_name in CLASSES[:-1]:
        sup = sum(1 for r in rows if r["class_id"] == cls_name)
        if sup == 0:
            continue
        tp = sum(
            1
            for r, p in zip(rows, predictions, strict=True)
            if r["class_id"] == cls_name and p == cls_name
        )
        pred_cnt = sum(1 for p in predictions if p == cls_name)
        all_f1s.append((2.0 * tp / (sup + pred_cnt)) if (sup + pred_cnt) > 0 else 0.0)
    tomato_macro_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0.0

    s_select = 0.70 * source_macro_f1 + 0.20 * tomato_macro_f1 + 0.10 * recall_neg
    return {
        "s_select": float(s_select),
        "source_macro_f1_nonlab": float(source_macro_f1),
        "tomato_macro_f1": float(tomato_macro_f1),
        "recall_nonlab_healthy": float(recall_healthy),
        "precision_nonlab_bs": float(precision_bs),
        "recall_negative": float(recall_neg),
    }


@torch.inference_mode()
def collect_tta_logits(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Collect 2-view horizontal-flip TTA logits strictly in FP32 (TF32 disabled)."""
    model.eval()
    prev_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    prev_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    all_logits: list[NDArray[np.float64]] = []
    all_targets: list[NDArray[np.int64]] = []
    try:
        with torch.autocast(device_type=device.type, enabled=False):
            for pixels, targets in loader:
                x = pixels.to(device=device, dtype=torch.float32, non_blocking=True)
                logits_orig = model(x).float()
                logits_flip = model(x.flip(-1)).float()
                logits = (logits_orig + logits_flip) * 0.5
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("Non-finite logits encountered during evaluation")
                all_logits.append(logits.cpu().numpy().astype(np.float64))
                all_targets.append(targets.cpu().numpy().astype(np.int64))
    finally:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
    if not all_logits:
        raise ValueError("Evaluation loader must not be empty")
    return np.concatenate(all_logits, axis=0), np.concatenate(all_targets, axis=0)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending")
    pending.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(pending, path)


def calibrate_and_export(
    model: SpatialBiGruClassifier,
    val_calib_logits: torch.Tensor | NDArray[Any] | Sequence[Sequence[float]],
    val_calib_targets: torch.Tensor | NDArray[Any] | Sequence[int],
    output_dir: Path,
    *,
    model_version: str,
    target_coverage: float = 0.85,
    eval_size: int = 288,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    selected_epoch: int = 1,
    selected_variant: str = "ema",
    guardrail_passed: bool | None = None,
    artifact_kind: str = "trained",
    frozen_models_path: Path | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit T & tau solely on in-domain rows (target != 8), verify negatives, and export 3 files."""
    if guardrail_passed is not None and type(guardrail_passed) is not bool:
        raise ValueError("guardrail_passed must be a boolean")
    if extra_metadata and "guardrail_passed" in extra_metadata:
        raise ValueError("Pass guardrail_passed explicitly, outside extra_metadata")
    logits_np = np.asarray(
        val_calib_logits.detach().cpu().numpy()
        if isinstance(val_calib_logits, torch.Tensor)
        else val_calib_logits,
        dtype=np.float64,
    )
    targets_np = np.asarray(
        val_calib_targets.detach().cpu().numpy()
        if isinstance(val_calib_targets, torch.Tensor)
        else val_calib_targets
    )
    if (
        logits_np.ndim != 2
        or logits_np.shape[1] != len(CLASSES)
        or targets_np.ndim != 1
        or len(targets_np) != len(logits_np)
    ):
        raise ValueError("Calibration logits/targets shape mismatch")
    if targets_np.dtype.kind not in "iu":
        raise ValueError("Calibration targets must be integer class indices")

    in_domain_mask = targets_np != _NON_LEAF_INDEX
    neg_mask = targets_np == _NON_LEAF_INDEX
    if not in_domain_mask.any():
        raise ValueError("Calibration requires at least one in-domain tomato sample")
    if not neg_mask.any():
        raise ValueError("Calibration requires at least one negative sample")

    in_logits = logits_np[in_domain_mask]
    in_targets = targets_np[in_domain_mask]
    temp_fit = fit_temperature(in_logits, in_targets)
    threshold = select_coverage_threshold(
        in_logits,
        temp_fit.temperature,
        target_coverage,
        negative_class_index=_NON_LEAF_INDEX,
    )
    in_accept = acceptance_mask(
        in_logits,
        temp_fit.temperature,
        threshold,
        negative_class_index=_NON_LEAF_INDEX,
    )
    in_coverage = float(in_accept.mean())

    neg_accept = acceptance_mask(
        logits_np[neg_mask],
        temp_fit.temperature,
        threshold,
        negative_class_index=_NON_LEAF_INDEX,
    )
    neg_false_accept = float(neg_accept.mean())
    neg_rejection = 1.0 - neg_false_accept

    output_dir.mkdir(parents=True, exist_ok=True)
    export_dir = output_dir / "model"
    stage_dir = Path(tempfile.mkdtemp(prefix=".model-export-", dir=output_dir))
    try:
        cpu_state = {
            k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
        }
        weights_path = stage_dir / "model.safetensors"
        save_file(cpu_state, weights_path)
        if weights_path.stat().st_size > MAX_WEIGHT_BYTES:
            raise ValueError("Exported model.safetensors exceeds 256_000_000 bytes")

        head_proj_dim = getattr(model.head, "proj_dim", 512)
        head_hidden_dim = getattr(model.head, "hidden_dim", 256)
        meta: dict[str, Any] = {
            **(extra_metadata or {}),
            "schema_version": 1,
            "architecture": model.architecture,
            "use_gru_head": model.use_gru_head,
            "hidden_dim": int(head_hidden_dim),
            "proj_dim": int(head_proj_dim),
            "model_version": model_version,
            "class_ids": list(CLASSES),
            "size": int(eval_size),
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
            "test_time_horizontal_flip": True,
            "selected_epoch": int(selected_epoch),
            "selected_variant": selected_variant,
            "temperature": float(temp_fit.temperature),
            "threshold": float(threshold),
            "target_coverage": float(target_coverage),
            "artifact_kind": artifact_kind,
            **(
                {"guardrail_passed": guardrail_passed}
                if guardrail_passed is not None
                else {}
            ),
        }
        (stage_dir / "leaf-classifier.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        hashes = {name: _file_sha256(stage_dir / name) for name in EXPORT_LIMITS}
        release_manifest = {
            "schema_version": 1,
            "model_version": model_version,
            "architecture": model.architecture,
            "evaluation_size": int(eval_size),
            "temperature": float(temp_fit.temperature),
            "threshold": float(threshold),
            "artifact_kind": artifact_kind,
            "sha256": hashes,
        }
        (stage_dir / CNN_GRU_MANIFEST).write_text(
            json.dumps(release_manifest, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        validate_cnn_gru_export(stage_dir, require_trained=(artifact_kind == "trained"))
        if export_dir.exists():
            shutil.rmtree(export_dir)
        stage_dir.rename(export_dir)
    finally:
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)

    calibration_summary = {
        "schema_version": 1,
        "model_version": model_version,
        "temperature_fit_scope": "in_domain_9logits",
        "temperature": float(temp_fit.temperature),
        "nll_before": float(temp_fit.nll_before),
        "nll_after": float(temp_fit.nll_after),
        "at_bound": bool(temp_fit.at_bound),
        "threshold": float(threshold),
        "target_coverage": float(target_coverage),
        **(
            {"guardrail_passed": guardrail_passed}
            if guardrail_passed is not None
            else {}
        ),
        "in_domain_count": int(in_domain_mask.sum()),
        "negative_count": int(neg_mask.sum()),
        "in_domain_coverage": in_coverage,
        "negative_false_accept_rate": neg_false_accept,
        "negative_rejection_rate": neg_rejection,
    }
    _write_json_atomic(output_dir / "calibration.json", calibration_summary)

    bundle_hashes = {name: _file_sha256(export_dir / name) for name in CNN_GRU_FILES}
    frozen_entry = {
        "schema_version": 1,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "model_version": model_version,
        "architecture": model.architecture,
        "selected_epoch": int(selected_epoch),
        "selected_variant": selected_variant,
        "temperature": float(temp_fit.temperature),
        "threshold": float(threshold),
        "bundle_dir": str(export_dir),
        "files_sha256": bundle_hashes,
    }
    _write_json_atomic(output_dir / "frozen-model.json", frozen_entry)

    models_lock_path = (
        frozen_models_path
        if frozen_models_path is not None
        else output_dir / "frozen-models.json"
    )
    existing: dict[str, Any] = {"schema_version": 1, "models": {}}
    if models_lock_path.is_file():
        loaded = json.loads(models_lock_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and isinstance(loaded.get("models"), dict):
            existing = loaded
    existing["models"][model_version] = frozen_entry
    _write_json_atomic(models_lock_path, existing)

    return calibration_summary


class _CnnGruTrainTransform:
    def __init__(
        self,
        size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        *,
        cooldown: bool = False,
    ) -> None:
        self.size = size
        self.fill = tuple(round(v * 255) for v in mean)
        jitter = (
            T.ColorJitter(0.14, 0.14, 0.12, 0.015)
            if cooldown
            else T.ColorJitter(0.28, 0.28, 0.24, 0.03)
        )
        ops: list[Any] = [
            T.RandomHorizontalFlip(0.5),
            T.RandomAffine(
                20,
                translate=(0.08, 0.08),
                scale=(0.88, 1.12),
                shear=6,
                interpolation=T.InterpolationMode.BILINEAR,
                fill=self.fill,
            ),
            T.RandomApply([jitter], p=0.80),
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean, std),
        ]
        if not cooldown:
            ops.append(
                T.RandomErasing(p=0.15, scale=(0.015, 0.07), ratio=(0.5, 2.0), value=0)
            )
        self.pipeline = T.Compose(ops)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
            raise ValueError("image must be a nonempty PIL image")
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("pixel_limit: input exceeds budget")
        with ImageOps.exif_transpose(image) as oriented:
            rgb = oriented.convert("RGB")
            try:
                with ImageOps.pad(
                    rgb,
                    (self.size, self.size),
                    method=Image.Resampling.BICUBIC,
                    color=self.fill,
                ) as padded:
                    tensor: torch.Tensor = self.pipeline(padded)
            finally:
                if rgb is not oriented:
                    rgb.close()
        return tensor.cpu()


class CnnGruImageDataset(Dataset[tuple[torch.Tensor, int]]):
    """Map-style dataset applying deterministic or seeded transforms on verified images."""

    def __init__(
        self,
        data_root: Path,
        rows: list[dict[str, str]],
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.data_root = data_root
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(self.data_root / row["path"]) as image:
            pixels = self.transform(image)
        return pixels, CLASSES.index(row["class_id"])


def _freeze_all_batchnorms(model: nn.Module) -> bool:
    all_eval = True
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            if module.training:
                all_eval = False
    return all_eval


def _update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_val, cur_val in zip(
            ema_model.state_dict().values(),
            model.state_dict().values(),
            strict=True,
        ):
            if ema_val.is_floating_point():
                ema_val.mul_(decay).add_(
                    cur_val.detach().to(ema_val.device), alpha=1.0 - decay
                )
            else:
                ema_val.copy_(cur_val.detach().to(ema_val.device))


def _save_two_phase_checkpoint(
    out_dir: Path,
    *,
    state: dict[str, Any],
    best_state_dict: dict[str, torch.Tensor],
    best_epoch: int,
    selected_variant: str,
    previous_best: Path | None,
) -> Path:
    best_path = out_dir / f"best-epoch-{best_epoch}-{selected_variant}.pt"
    artifacts = dict(state.get("artifacts", {}))
    if best_path != previous_best:
        save_atomic(
            {
                "model": best_state_dict,
                "epoch": best_epoch,
                "variant": selected_variant,
                "selection_key": state["best_key"],
            },
            best_path,
        )
        if previous_best is not None:
            artifacts.pop(previous_best.name, None)
    artifacts[best_path.name] = _file_sha256(best_path)
    save_atomic({**state, "artifacts": artifacts}, out_dir / "last.pt")
    state["artifacts"] = artifacts
    if previous_best is not None and previous_best != best_path:
        previous_best.unlink(missing_ok=True)
    return best_path


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    pending = dst.with_name(dst.name + ".pending")
    shutil.copy2(src, pending)
    os.replace(pending, dst)


def _mirror_checkpoint_to_drive(
    drive_dir: Path,
    output_dir: Path,
    best_file: Path,
    previous_best: Path | None,
) -> None:
    drive_dir.mkdir(parents=True, exist_ok=True)
    if best_file != previous_best or not (drive_dir / best_file.name).is_file():
        _atomic_copy(best_file, drive_dir / best_file.name)
    _atomic_copy(output_dir / "last.pt", drive_dir / "last.pt")
    history_path = output_dir / "history.json"
    if history_path.is_file():
        _atomic_copy(history_path, drive_dir / "history.json")
    if previous_best is not None and previous_best != best_file:
        (drive_dir / previous_best.name).unlink(missing_ok=True)


def _download_pretrained_url(url: str, cache_dir: Path) -> Path:
    parsed = urlparse(url)
    filename = Path(parsed.path).name
    if parsed.scheme != "https" or not filename:
        raise ValueError(f"Unsupported pretrained_url: {url!r}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / filename
    if target.is_file():
        return target
    pending = target.with_name(target.name + ".pending")
    torch.hub.download_url_to_file(url, str(pending), progress=True)
    os.replace(pending, target)
    return target


def _resolve_registration_experiment(
    registration_path: Path, experiment_id: str
) -> dict[str, Any]:
    reg = json.loads(registration_path.read_text(encoding="utf-8"))
    experiments = reg.get("experiments")
    if not isinstance(experiments, dict) or experiment_id not in experiments:
        raise ValueError(f"Unknown experiment {experiment_id!r} in {registration_path}")
    exp = experiments[experiment_id]
    pretrained_url = exp.get("pretrained_url")
    pretrained_weights: str | None = None
    source = str(exp.get("pretrained_source") or "")
    if pretrained_url is None and source.startswith("torchvision:"):
        pretrained_weights = source.split(":", 1)[1].split(".")[-1]
    elif pretrained_url is None:
        raise ValueError(f"Unsupported pretrained_source {source!r}")
    return {
        "architecture": str(exp["architecture"]),
        "use_gru_head": bool(exp["use_gru_head"]),
        "hidden_dim": int(exp.get("hidden_dim", 256)),
        "proj_dim": int(exp.get("proj_dim", 512)),
        "model_version": f"{reg['run_id']}-{experiment_id.lower()}-{exp['name']}",
        "pretrained_weights": pretrained_weights,
        "pretrained_url": str(pretrained_url) if pretrained_url is not None else None,
        "stochastic_depth_prob": (
            float(exp["stochastic_depth"])
            if exp.get("stochastic_depth") is not None
            else None
        ),
        "freeze_bn_eval": bool(exp.get("freeze_bn_eval", True)),
        "mixstyle": bool(exp.get("mixstyle", False)),
        "mixstyle_p": float(exp.get("mixstyle_p", 0.35)),
        "mixstyle_alpha": float(exp.get("mixstyle_alpha", 0.1)),
        "size": int(exp["input_size"]),
        "epochs": int(exp["epochs"]),
        "cooldown_epochs": int(exp["cooldown_epochs"]),
        "samples": int(exp["samples_per_epoch"]),
        "batch_size": int(exp["batch_size"]),
        "head_peak_lr": float(exp["head_peak_lr"]),
        "head_main_lr": float(exp["head_main_lr"]),
        "backbone_lrs": {k: float(v) for k, v in exp["backbone_lrs"].items()},
        "label_smoothing": float(exp["label_smoothing"]),
        "ema_start_epoch": int(exp["ema_start_epoch"]),
        "ema_max_decay": float(exp["ema_max_decay"]),
        "target_coverage": float(reg["target_coverage"]),
    }


def _load_manifest_rows(
    manifest_csv: Path, data_root: Path, *, verify_hashes: bool = False
) -> list[dict[str, str]]:
    with manifest_csv.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty manifest: {manifest_csv}")
    for row in rows:
        if row.get("class_id") not in CLASSES or row.get("split") not in {
            "train",
            "val_select",
            "val_calib",
        }:
            raise ValueError(f"Invalid row in manifest: {row}")
        img_path = (data_root / row["path"]).resolve()
        if not img_path.is_relative_to(data_root.resolve()) or not img_path.is_file():
            raise ValueError(f"Missing image in manifest: {row['path']}")
        if verify_hashes and _file_sha256(img_path) != row.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {row['path']}")
    return rows


def _cuda_transfer_and_amp(
    device: torch.device, architecture: str
) -> tuple[bool, bool]:
    """Use FP32 for EfficientNetV2 on CUDA because MBConv+SE+SiLU overflows float16."""
    is_cuda = device.type == "cuda"
    use_amp = is_cuda and not architecture.startswith("efficientnet")
    return is_cuda, use_amp


def train_cnn_gru(
    manifest_csv: Path,
    data_root: Path,
    output_dir: Path,
    *,
    architecture: str = "resnet50",
    use_gru_head: bool = True,
    hidden_dim: int = 256,
    proj_dim: int = 512,
    model_version: str = "cnn-gru-v1",
    pretrained_weights: str | None = None,
    pretrained_url: str | None = None,
    convnext_checkpoint: Path | None = None,
    stochastic_depth_prob: float | None = None,
    freeze_bn_eval: bool = True,
    mixstyle: bool = False,
    mixstyle_p: float = 0.35,
    mixstyle_alpha: float = 0.1,
    size: int = 288,
    epochs: int = 22,
    cooldown_epochs: int = 3,
    samples: int = 4800,
    batch_size: int = 32,
    grad_accum: int = 1,
    workers: int = 0,
    seed: int = 42,
    head_peak_lr: float = 8e-4,
    head_main_lr: float = 3e-4,
    backbone_lrs: dict[str, float] | None = None,
    weight_decay: float = 0.05,
    label_smoothing: float = 0.02,
    ema_start_epoch: int = 4,
    ema_max_decay: float = 0.993,
    target_coverage: float = 0.85,
    artifact_kind: str = "trained",
    device_name: str | None = None,
    verify_hashes: bool = False,
    stop_after_epoch: int | None = None,
    frozen_models_path: Path | None = None,
    drive_dir: Path | None = None,
) -> dict[str, Any]:
    """Train, validate, resume atomically, calibrate, and export 3-file CNN-GRU bundle."""
    if architecture not in SUPPORTED_BACKBONES:
        raise ValueError("unsupported architecture")
    if min(epochs, samples, batch_size, grad_accum) < 1:
        raise ValueError("epochs, samples, batch_size and grad_accum must be positive")

    device = torch.device(
        device_name
        if device_name is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    is_cuda, use_amp = _cuda_transfer_and_amp(device, architecture)
    rows = _load_manifest_rows(manifest_csv, data_root, verify_hashes=verify_hashes)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_select_rows = [r for r in rows if r["split"] == "val_select"]
    val_calib_rows = [r for r in rows if r["split"] == "val_calib"]
    if not train_rows or not val_select_rows or not val_calib_rows:
        raise ValueError(
            "Manifest must contain train, val_select, and val_calib splits"
        )

    active_backbone_lrs = {
        k: float(v)
        for k, v in (
            backbone_lrs
            if backbone_lrs is not None
            else (
                {"stage4": 5e-5, "stage3": 2e-5, "stage2": 8e-6}
                if architecture.startswith("convnext")
                else {"stage4": 8e-5, "stage3": 2.5e-5, "stage2": 1e-5}
            )
        ).items()
    }
    if set(active_backbone_lrs) != {"stage4", "stage3", "stage2"} or any(
        not math.isfinite(v) or v < 0 for v in active_backbone_lrs.values()
    ):
        raise ValueError(
            "backbone_lrs must provide finite non-negative rates for stage2, stage3, stage4"
        )
    steps_per_epoch = math.ceil(samples / (batch_size * grad_accum))
    total_steps = epochs * steps_per_epoch
    stage1_epochs = min(3, max(1, epochs // 6)) if epochs < 12 else 3
    stage1_steps = stage1_epochs * steps_per_epoch
    warmup_steps = steps_per_epoch

    run_config: dict[str, Any] = {
        "schema_version": 1,
        "manifest_sha256": _file_sha256(manifest_csv),
        "architecture": architecture,
        "use_gru_head": use_gru_head,
        "hidden_dim": hidden_dim,
        "proj_dim": proj_dim,
        "model_version": model_version,
        "stochastic_depth_prob": (
            float(stochastic_depth_prob) if stochastic_depth_prob is not None else None
        ),
        "freeze_bn_eval": freeze_bn_eval,
        "mixstyle": mixstyle,
        "mixstyle_p": float(mixstyle_p),
        "mixstyle_alpha": float(mixstyle_alpha),
        "size": size,
        "epochs": epochs,
        "cooldown_epochs": cooldown_epochs,
        "samples": samples,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "seed": seed,
        "head_peak_lr": float(head_peak_lr),
        "head_main_lr": float(head_main_lr),
        "backbone_lrs": active_backbone_lrs,
        "weight_decay": float(weight_decay),
        "label_smoothing": float(label_smoothing),
        "ema_start_epoch": ema_start_epoch,
        "ema_max_decay": float(ema_max_decay),
        "target_coverage": float(target_coverage),
        "artifact_kind": artifact_kind,
    }

    last_path = output_dir / "last.pt"
    resume_state: dict[str, Any] | None = None
    best_epoch = 0
    selected_variant = "raw"
    best_file: Path | None = None
    artifacts: dict[str, str] = {}
    best_state_dict: dict[str, torch.Tensor] = {}
    if last_path.is_file():
        resume_state = torch.load(last_path, map_location="cpu", weights_only=True)
        if resume_state.get("run") != run_config:
            raise ValueError("Checkpoint run configuration differs from current run")
        best_epoch = int(resume_state["best_epoch"])
        selected_variant = str(resume_state["selected_variant"])
        best_file = output_dir / f"best-epoch-{best_epoch}-{selected_variant}.pt"
        artifacts = dict(resume_state.get("artifacts", {}))
        if not best_file.is_file() or artifacts.get(best_file.name) != _file_sha256(
            best_file
        ):
            raise ValueError(
                "Best checkpoint referenced by last.pt is missing or altered"
            )
        best_ckpt = torch.load(best_file, map_location="cpu", weights_only=True)
        if int(best_ckpt.get("epoch", -1)) != best_epoch:
            raise ValueError("Best checkpoint epoch disagrees with last.pt")
        best_state_dict = best_ckpt["model"]

    resolved_convnext_ckpt = convnext_checkpoint
    if (
        resume_state is None
        and resolved_convnext_ckpt is None
        and pretrained_url is not None
    ):
        resolved_convnext_ckpt = _download_pretrained_url(
            pretrained_url, output_dir.resolve().parent / "_pretrained"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    model = build_cnn_gru_model(
        architecture,
        len(CLASSES),
        use_gru_head=use_gru_head,
        hidden_dim=hidden_dim,
        proj_dim=proj_dim,
        pretrained_weights=None if resume_state is not None else pretrained_weights,
        convnext_checkpoint=None
        if resume_state is not None
        else resolved_convnext_ckpt,
        stochastic_depth_prob=stochastic_depth_prob,
        freeze_bn_eval=freeze_bn_eval,
        mixstyle=mixstyle,
        mixstyle_p=mixstyle_p,
        mixstyle_alpha=mixstyle_alpha,
    ).to(device)
    ema_model = copy.deepcopy(model).cpu().eval().requires_grad_(False)

    stage_modules = model.backbone_stage_modules()
    stage_param_ids = {
        id(p)
        for mods in stage_modules.values()
        for mod in mods
        for p in mod.parameters()
    }
    head_params = [
        *model.head.parameters(),
        *model.classifier.parameters(),
    ]
    if architecture.startswith("convnext"):
        gap_norm = model.head.gap_norm
        assert isinstance(gap_norm, nn.Module)
        gap_norm_ids = {id(p) for p in gap_norm.parameters()}
        head_params = [p for p in head_params if id(p) not in gap_norm_ids]
        stage_modules["stage4"].append(gap_norm)
        stage_param_ids |= gap_norm_ids

    param_groups: list[dict[str, Any]] = [
        {"name": "head", "params": head_params, "lr": head_peak_lr * 0.1}
    ]
    for stage_name in ("stage4", "stage3", "stage2"):
        params = [p for mod in stage_modules[stage_name] for p in mod.parameters()]
        param_groups.append({"name": stage_name, "params": params, "lr": 0.0})

    trainable_ids = {id(p) for g in param_groups for p in g["params"]}
    for p in model.parameters():
        if id(p) not in trainable_ids:
            p.requires_grad_(False)

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    scaler: torch.amp.GradScaler | None = (
        torch.amp.GradScaler("cuda") if use_amp else None
    )

    weights, delta_list = cnn_bigru_weights_strict(train_rows, samples=samples)
    log_prior_offset = torch.tensor(delta_list, dtype=torch.float32, device=device)

    eval_tf = cnn_gru_eval_transform(size, IMAGENET_MEAN, IMAGENET_STD)
    val_select_loader = DataLoader(
        CnnGruImageDataset(data_root, val_select_rows, eval_tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=is_cuda,
    )
    val_calib_loader = DataLoader(
        CnnGruImageDataset(data_root, val_calib_rows, eval_tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=is_cuda,
    )

    start_epoch = 1
    global_step = 0
    ema_updates = 0
    best_key: tuple[bool, float, float, float] | None = None
    history: list[dict[str, Any]] = []

    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        ema_model.load_state_dict(resume_state["ema_model"])
        ema_updates = int(resume_state["ema_updates"])
        optimizer.load_state_dict(resume_state["optimizer"])
        if scaler is not None and resume_state.get("scaler") is not None:
            scaler.load_state_dict(resume_state["scaler"])
        start_epoch = int(resume_state["epoch"]) + 1
        global_step = int(resume_state["global_step"])
        best_key_raw = resume_state["best_key"]
        best_key = (
            bool(best_key_raw[0]),
            float(best_key_raw[1]),
            float(best_key_raw[2]),
            float(best_key_raw[3]),
        )
        history = list(resume_state["history"])
        del resume_state

    bn_all_eval = True
    target_end_epoch = (
        min(epochs, stop_after_epoch) if stop_after_epoch is not None else epochs
    )
    for epoch in range(start_epoch, target_end_epoch + 1):
        torch.manual_seed(seed + epoch)
        in_stage1 = epoch <= stage1_epochs
        for p in model.parameters():
            if id(p) in stage_param_ids:
                p.requires_grad_(not in_stage1)

        if epoch == ema_start_epoch and ema_updates == 0:
            ema_model.load_state_dict(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            )

        cooldown = epoch > epochs - cooldown_epochs and epochs > cooldown_epochs
        train_ds = CnnGruImageDataset(
            data_root,
            train_rows,
            _CnnGruTrainTransform(size, IMAGENET_MEAN, IMAGENET_STD, cooldown=cooldown),
        )
        sampler = WeightedRandomSampler(
            weights,
            samples,
            replacement=True,
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=workers,
            pin_memory=is_cuda,
        )

        model.train()
        bn_all_eval = bn_all_eval and (
            _freeze_all_batchnorms(model) if freeze_bn_eval else True
        )
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        consumed = 0
        micro_batches = 0
        total_batches = len(train_loader)

        for batch_idx, (pixels, targets) in enumerate(train_loader, start=1):
            x = pixels.to(device=device, non_blocking=True)
            y = targets.to(device=device, dtype=torch.long, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                loss = prior_matched_ce_loss(
                    model(x),
                    y,
                    log_prior_offset,
                    label_smoothing=label_smoothing,
                )
            loss_val = float(loss.detach().item())
            if not math.isfinite(loss_val):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")

            # Scale microbatch loss by remaining steps in current accumulation window.
            remaining_batches = total_batches - (batch_idx - 1)
            accum_denom = (
                grad_accum
                if batch_idx <= (total_batches // grad_accum) * grad_accum
                else max(
                    1,
                    remaining_batches
                    if micro_batches == 0
                    else micro_batches + remaining_batches,
                )
            )
            scaled_loss = loss / accum_denom
            if scaler is not None:
                torch.autograd.backward(scaler.scale(scaled_loss))
            else:
                torch.autograd.backward(scaled_loss)

            total_loss += loss_val * len(x)
            consumed += len(x)
            micro_batches += 1

            if micro_batches == grad_accum or batch_idx == total_batches:
                step_lrs = continuous_c0_lr(
                    min(global_step + 1, total_steps),
                    total_steps,
                    head_peak_lr,
                    head_main_lr,
                    active_backbone_lrs,
                    stage1_steps=stage1_steps,
                    warmup_steps=warmup_steps,
                )
                for group in optimizer.param_groups:
                    group["lr"] = step_lrs[group["name"]]

                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    prev_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    updated = scaler.get_scale() >= prev_scale
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    updated = True
                optimizer.zero_grad(set_to_none=True)
                micro_batches = 0

                if updated:
                    global_step += 1
                    if epoch >= ema_start_epoch:
                        decay = ema_decay(ema_updates, max_decay=ema_max_decay)
                        _update_ema(ema_model, model, decay)
                        ema_updates += 1

        # Evaluate raw and (when active) ema on val_select using 2-view TTA FP32.
        raw_logits, _ = collect_tta_logits(model, val_select_loader, device)
        raw_preds = [CLASSES[idx] for idx in raw_logits.argmax(axis=1).tolist()]
        raw_metrics = _compute_val_select_metrics(val_select_rows, raw_preds)
        raw_key = selection_key(raw_metrics)

        variants: dict[
            str, tuple[tuple[bool, float, float, float], dict[str, Any], nn.Module]
        ] = {"raw": (raw_key, raw_metrics, model)}
        if ema_updates > 0:
            ema_eval_model = ema_model.to(device)
            try:
                ema_logits, _ = collect_tta_logits(
                    ema_eval_model, val_select_loader, device
                )
            finally:
                ema_model.cpu()
            ema_preds = [CLASSES[idx] for idx in ema_logits.argmax(axis=1).tolist()]
            ema_metrics = _compute_val_select_metrics(val_select_rows, ema_preds)
            ema_key = selection_key(ema_metrics)
            variants["ema"] = (ema_key, ema_metrics, ema_model)

        ep_best_variant = max(variants, key=lambda name: variants[name][0])
        ep_best_key, _ep_best_metrics, ep_best_model = variants[ep_best_variant]

        if best_key is None or ep_best_key > best_key:
            best_key = ep_best_key
            best_epoch = epoch
            selected_variant = ep_best_variant
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in ep_best_model.state_dict().items()
            }

        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(consumed, 1),
                "global_step": global_step,
                "ema_updates": ema_updates,
                "epoch_selected_variant": ep_best_variant,
                "raw": raw_metrics,
                **({"ema": variants["ema"][1]} if "ema" in variants else {}),
            }
        )

        checkpoint_state: dict[str, Any] = {
            "run": run_config,
            "epoch": epoch,
            "global_step": global_step,
            "best_epoch": best_epoch,
            "selected_variant": selected_variant,
            "best_key": list(best_key),
            "history": history,
            "model": {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            },
            "ema_model": {
                k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()
            },
            "ema_updates": ema_updates,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "artifacts": artifacts,
        }
        prev_best_file = best_file
        best_file = _save_two_phase_checkpoint(
            output_dir,
            state=checkpoint_state,
            best_state_dict=best_state_dict,
            best_epoch=best_epoch,
            selected_variant=selected_variant,
            previous_best=prev_best_file,
        )
        artifacts = checkpoint_state["artifacts"]
        _write_json_atomic(output_dir / "history.json", history)
        if drive_dir is not None:
            _mirror_checkpoint_to_drive(
                drive_dir, output_dir, best_file, prev_best_file
            )

        ep_metrics = variants[ep_best_variant][1]
        print(
            "CNN_GRU_EPOCH",
            json.dumps(
                {
                    "model_version": model_version,
                    "epoch": epoch,
                    "epochs": epochs,
                    "train_loss": round(total_loss / max(consumed, 1), 6),
                    "global_step": global_step,
                    "ema_updates": ema_updates,
                    "ep_variant": ep_best_variant,
                    "ep_s_select": round(float(ep_metrics["s_select"]), 6),
                    "ep_guardrail": bool(ep_best_key[0]),
                    "best_epoch": best_epoch,
                    "selected_variant": selected_variant,
                    "best_guardrail": bool(best_key[0]),
                    "best_s_select": round(float(best_key[1]), 6),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    _write_json_atomic(output_dir / "history.json", history)
    if stop_after_epoch is not None and target_end_epoch < epochs:
        return {
            "status": "paused",
            "epoch": target_end_epoch,
            "best_epoch": best_epoch,
            "selected_variant": selected_variant,
            "bn_all_eval": bn_all_eval,
            "history": history,
        }

    # Load selected best checkpoint, calibrate on val_calib, and export exact 3 files.
    assert best_file is not None
    selected_ckpt = torch.load(best_file, map_location="cpu", weights_only=True)
    if int(selected_ckpt.get("epoch", -1)) != best_epoch:
        raise ValueError("Selected checkpoint epoch disagrees with history")
    model.load_state_dict(selected_ckpt["model"])
    calib_logits, calib_targets = collect_tta_logits(model, val_calib_loader, device)
    guardrail_passed = bool(best_key[0]) if best_key is not None else False
    val_select_best = (
        history[best_epoch - 1].get(selected_variant)
        if 1 <= best_epoch <= len(history)
        else None
    )
    calibration = calibrate_and_export(
        model,
        calib_logits,
        calib_targets,
        output_dir,
        model_version=model_version,
        target_coverage=target_coverage,
        eval_size=size,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        selected_epoch=best_epoch,
        selected_variant=selected_variant,
        guardrail_passed=guardrail_passed,
        artifact_kind=artifact_kind,
        frozen_models_path=frozen_models_path,
        extra_metadata=(
            {"val_select_best": val_select_best}
            if val_select_best is not None
            else None
        ),
    )
    print(
        "CNN_GRU_CALIBRATED",
        json.dumps(
            {
                "model_version": model_version,
                "best_epoch": best_epoch,
                "selected_variant": selected_variant,
                "guardrail_passed": guardrail_passed,
                "temperature": round(float(calibration["temperature"]), 6),
                "threshold": round(float(calibration["threshold"]), 6),
                "negative_rejection_rate": round(
                    float(calibration["negative_rejection_rate"]), 6
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return {
        "status": "completed",
        "best_epoch": best_epoch,
        "selected_variant": selected_variant,
        "best_key": best_key,
        "guardrail_passed": guardrail_passed,
        "bn_all_eval": bn_all_eval,
        "calibration": calibration,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, default=None)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drive-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--architecture", choices=SUPPORTED_BACKBONES, default="resnet50"
    )
    parser.add_argument("--no-gru-head", action="store_true")
    parser.add_argument("--model-version", default="cnn-gru-v1")
    parser.add_argument("--pretrained-weights", default=None)
    parser.add_argument("--pretrained-url", default=None)
    parser.add_argument("--convnext-checkpoint", type=Path, default=None)
    parser.add_argument("--stochastic-depth", type=float, default=None)
    parser.add_argument("--no-freeze-bn-eval", action="store_true")
    parser.add_argument("--mixstyle", action="store_true")
    parser.add_argument("--size", type=int, default=288)
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--cooldown-epochs", type=int, default=3)
    parser.add_argument("--samples", type=int, default=4800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema-max-decay", type=float, default=0.992)
    parser.add_argument("--target-coverage", type=float, default=0.85)
    args = parser.parse_args()

    if (args.registration is None) != (args.experiment is None):
        parser.error("--registration and --experiment must be passed together")
    if args.resume and not (args.output / "last.pt").is_file():
        raise ValueError(f"--resume requested but {args.output / 'last.pt'} is missing")

    if args.registration is not None:
        reg_kwargs = _resolve_registration_experiment(
            args.registration, args.experiment
        )
        train_cnn_gru(
            args.manifest,
            args.data,
            args.output,
            grad_accum=args.grad_accum,
            workers=args.workers,
            seed=args.seed,
            drive_dir=args.drive_dir,
            **reg_kwargs,
        )
        return

    train_cnn_gru(
        args.manifest,
        args.data,
        args.output,
        architecture=args.architecture,
        use_gru_head=not args.no_gru_head,
        model_version=args.model_version,
        pretrained_weights=args.pretrained_weights,
        pretrained_url=args.pretrained_url,
        convnext_checkpoint=args.convnext_checkpoint,
        stochastic_depth_prob=args.stochastic_depth,
        freeze_bn_eval=not args.no_freeze_bn_eval,
        mixstyle=args.mixstyle,
        size=args.size,
        epochs=args.epochs,
        cooldown_epochs=args.cooldown_epochs,
        samples=args.samples,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        workers=args.workers,
        seed=args.seed,
        ema_max_decay=args.ema_max_decay,
        target_coverage=args.target_coverage,
        drive_dir=args.drive_dir,
    )


if __name__ == "__main__":
    main()
