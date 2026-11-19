"""Tests for CNN-GRU 3-way splitting, water-filling sampler, C0 schedule, calibration & atomic export."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from ai.cnn_gru_inference import (
    CNN_GRU_FILES,
    MAX_WEIGHT_BYTES,
    CnnGruPredictor,
    validate_cnn_gru_export,
)
from ai.cnn_gru_model import build_cnn_gru_model
from ai.cnn_gru_train import (
    _cuda_transfer_and_amp,
    _resolve_registration_experiment,
    build_three_way_split,
    calibrate_and_export,
    cnn_bigru_weights_strict,
    continuous_c0_lr,
    ema_decay,
    prior_matched_ce_loss,
    selection_key,
    train_cnn_gru,
)
from ai.cnn_gru_train import (
    main as train_main,
)
from ai.dinov3_field import CLASSES


def _make_split_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    idx = 0

    def add_row(
        *,
        source: str,
        domain: str,
        class_id: str,
        group: str,
        sha: str | None = None,
        orig_sha: str | None = None,
        norm_sha: str | None = None,
        split_block: str = "",
    ) -> None:
        nonlocal idx
        idx += 1
        digest = sha or hashlib.sha256(f"img-{idx}".encode()).hexdigest()
        rows.append(
            {
                "path": f"images/{idx:05d}.jpg",
                "class_id": class_id,
                "group": group,
                "sha256": digest,
                "original_sha256": orig_sha
                or hashlib.sha256(f"orig-{idx}".encode()).hexdigest(),
                "normalized_sha256": norm_sha
                or hashlib.sha256(f"norm-{idx}".encode()).hexdigest(),
                "source": source,
                "domain": domain,
                "split_block": split_block,
            }
        )

    # hossain_v1 must go 100% to train
    for g in range(4):
        add_row(
            source="hossain_v1",
            domain="controlled",
            class_id="early_blight",
            group=f"hossain:{g}",
            split_block="hossain:single_session",
        )

    # bapari_2025_v1 must be 100% excluded from train and split between val_select & val_calib
    for g in range(10):
        add_row(
            source="bapari_2025_v1",
            domain="capture_unverified",
            class_id=("healthy", "bacterial_spot", "septoria_leaf_spot")[g % 3],
            group=f"bapari:{g}",
            split_block="bapari:nazirpur",
        )

    # Small stratum (< 6 groups) stays in train
    for g in range(3):
        add_row(
            source="imtiaz",
            domain="field",
            class_id="leaf_mold",
            group=f"imtiaz:{g}",
        )

    # Large stratum (>= 6 groups) splits 74/13/13; include two distinct groups sharing split_block & sha
    for g in range(30):
        sb = "shared_block_A" if g in (0, 1) else f"block_{g}"
        shared_sha = "f" * 64 if g in (2, 3) else None
        add_row(
            source="plantcity",
            domain="field",
            class_id="healthy",
            group=f"pc:{g}",
            sha=shared_sha,
            split_block=sb,
        )
    return rows


def test_three_way_split_blocks_leakage_and_separates_bapari_hossain() -> None:
    rows = _make_split_rows()
    assignment = build_three_way_split(rows, seed="tomato-cnn-bigru-v1")
    assert set(assignment.values()) == {"train", "val_select", "val_calib"}

    splits: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        splits[assignment[row["group"]]].append(row)

    assert all(
        assignment[r["group"]] == "train" for r in rows if r["source"] == "hossain_v1"
    )
    assert all(
        assignment[r["group"]] in {"val_select", "val_calib"}
        for r in rows
        if r["source"] == "bapari_2025_v1"
    )
    assert {r["split_block"] for r in splits["train"] if r["split_block"]}.isdisjoint(
        {
            r["split_block"]
            for r in splits["val_select"] + splits["val_calib"]
            if r["split_block"]
        }
    )

    # Verify zero overlap across all 3 sets on group, sha256, original_sha256, normalized_sha256, path
    names = ("train", "val_select", "val_calib")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for key in (
                "group",
                "sha256",
                "original_sha256",
                "normalized_sha256",
                "path",
            ):
                a = {r[key] for r in splits[names[i]] if r.get(key)}
                b = {r[key] for r in splits[names[j]] if r.get(key)}
                assert not (a & b), (
                    f"Overlap on {key} between {names[i]} and {names[j]}"
                )

    # Shared non-bapari split_block and shared sha256 stay in the same split
    assert assignment["pc:0"] == assignment["pc:1"]
    assert assignment["pc:2"] == assignment["pc:3"]

    # Order-invariant under reversed rows and blocks cross-group path collisions even without hashes
    assert (
        build_three_way_split(list(reversed(rows)), seed="tomato-cnn-bigru-v1")
        == assignment
    )
    path_rows = [
        {
            "path": "shared.jpg",
            "class_id": "healthy",
            "group": f"g{i}",
            "source": "plantcity",
            "domain": "field",
        }
        for i in range(8)
    ]
    for i in range(8, 20):
        path_rows.append(
            {
                "path": f"uniq_{i}.jpg",
                "class_id": "healthy",
                "group": f"g{i}",
                "source": "plantcity",
                "domain": "field",
            }
        )
    path_assign = build_three_way_split(path_rows, seed="tomato-cnn-bigru-v1")
    assert len({path_assign[f"g{i}"] for i in range(8)}) == 1


def test_water_filling_weights_and_prior_matched_loss_zero_bias() -> None:
    rows: list[dict[str, str]] = []
    # Imbalanced 9-class synthetic dataset: spider_mites & yellow_leaf_curl hit caps
    for cls_idx, cls_name in enumerate(CLASSES):
        if cls_name == "not_tomato_leaf":
            for i in range(40):
                rows.append(
                    {
                        "class_id": cls_name,
                        "domain": "negative",
                        "source": "soil",
                        "group": f"neg:{i}",
                    }
                )
        elif cls_name == "spider_mites":
            for i in range(10):
                rows.append(
                    {
                        "class_id": cls_name,
                        "domain": "controlled",
                        "source": "pv",
                        "group": f"sm:{i}",
                    }
                )
        else:
            for d, n in (("field", 25), ("other", 6), ("controlled", 20)):
                for i in range(n):
                    rows.append(
                        {
                            "class_id": cls_name,
                            "domain": d,
                            "source": f"src_{d}_{cls_idx}",
                            "group": f"{cls_name}:{d}:{i}",
                        }
                    )

    caps = {"field": 1.40, "other": 1.50, "controlled": 0.85, "negative": 1.05}
    samples = 270
    weights, delta = cnn_bigru_weights_strict(rows, samples=samples, caps=caps)
    assert len(weights) == len(rows)
    assert len(delta) == len(CLASSES)
    assert sum(weights) == pytest.approx(samples, rel=1e-9, abs=1e-9)

    for row, w in zip(rows, weights, strict=True):
        dom = (
            "negative"
            if row["class_id"] == "not_tomato_leaf"
            else row["domain"]
            if row["domain"] in ("field", "controlled")
            else "other"
        )
        assert 0.0 < w <= caps[dom] + 1e-12

    # Check exact log prior match: delta_k == ln(K * pi_k)
    pi = torch.zeros(len(CLASSES), dtype=torch.float64)
    for row, w in zip(rows, weights, strict=True):
        pi[CLASSES.index(row["class_id"])] += w / samples
    expected_delta = torch.log(len(CLASSES) * pi)
    np.testing.assert_allclose(delta, expected_delta.numpy(), atol=1e-10)

    # Optimize raw logits z against class-prior-distributed targets -> wrong classes uniform (<1e-5)
    delta_t = torch.tensor(delta, dtype=torch.float64)
    z = torch.zeros(1, len(CLASSES), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([z], lr=1.0, max_iter=50)
    probs_pi = torch.softmax(delta_t, dim=0)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        total = torch.tensor(0.0, dtype=torch.float64)
        for k in range(len(CLASSES)):
            target = torch.tensor([k], dtype=torch.long)
            total = total + probs_pi[k] * prior_matched_ce_loss(
                z, target, delta_t, label_smoothing=0.02
            )
        torch.autograd.backward(total)
        return total

    opt.step(closure)  # type: ignore[no-untyped-call]
    z_centered = z.detach()[0] - z.detach()[0].mean()
    assert float(z_centered.abs().max().item()) < 1e-5

    # Also verify single fixed target y=0 leaves all 8 wrong raw logits uniform (< 1e-5)
    z_single = torch.zeros(1, len(CLASSES), dtype=torch.float64, requires_grad=True)
    opt2 = torch.optim.LBFGS(
        [z_single],
        lr=1.0,
        max_iter=80,
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def closure_single() -> torch.Tensor:
        opt2.zero_grad()
        loss = prior_matched_ce_loss(
            z_single, torch.tensor([0]), delta_t, label_smoothing=0.02
        )
        torch.autograd.backward(loss)
        return loss

    opt2.step(closure_single)  # type: ignore[no-untyped-call]
    wrong = z_single.detach()[0, 1:]
    assert float((wrong.max() - wrong.min()).item()) < 1e-5


def test_continuous_c0_lr_has_no_jump_at_stage_boundaries() -> None:
    backbone_lrs = {"stage4": 8e-5, "stage3": 2.5e-5, "stage2": 1e-5}
    for total_steps, g3_boundary in ((2700, 2250), (3300, 2850)):
        at_450 = continuous_c0_lr(450, total_steps, 8e-4, 3e-4, backbone_lrs)
        at_451 = continuous_c0_lr(451, total_steps, 8e-4, 3e-4, backbone_lrs)
        assert at_450["head"] == pytest.approx(3e-4, rel=1e-9)
        assert abs(at_451["head"] - at_450["head"]) < 1e-6
        assert at_450["stage4"] == 0.0
        assert 0.0 < at_451["stage4"] <= backbone_lrs["stage4"] / 150 + 1e-12

        at_600 = continuous_c0_lr(600, total_steps, 8e-4, 3e-4, backbone_lrs)
        at_601 = continuous_c0_lr(601, total_steps, 8e-4, 3e-4, backbone_lrs)
        assert at_600["stage4"] == pytest.approx(backbone_lrs["stage4"], rel=1e-9)
        assert abs(at_601["stage4"] - at_600["stage4"]) < 1e-7

        before_g3 = continuous_c0_lr(g3_boundary, total_steps, 8e-4, 3e-4, backbone_lrs)
        after_g3 = continuous_c0_lr(
            g3_boundary + 1, total_steps, 8e-4, 3e-4, backbone_lrs
        )
        for key in ("head", "stage4", "stage3", "stage2"):
            assert abs(after_g3[key] - before_g3[key]) < 1e-6

        final_lr = continuous_c0_lr(total_steps, total_steps, 8e-4, 3e-4, backbone_lrs)
        assert final_lr["head"] == pytest.approx(0.05 * 3e-4, rel=1e-6)
        assert final_lr["stage4"] == pytest.approx(
            0.05 * backbone_lrs["stage4"], rel=1e-6
        )

    assert ema_decay(0) == pytest.approx(0.1)
    assert ema_decay(10000) == pytest.approx(0.993)
    assert ema_decay(10000, max_decay=0.992) == pytest.approx(0.992)

    # Short/mid schedules where stage1_steps <= warmup_steps or total_steps == stage1_steps + warmup_steps
    import math

    for total_steps, s1, w in ((1500, 150, 150), (4, 2, 2), (600, 450, 150)):
        at_s1 = continuous_c0_lr(
            s1, total_steps, 8e-4, 3e-4, backbone_lrs, stage1_steps=s1, warmup_steps=w
        )
        after_s1 = continuous_c0_lr(
            s1 + 1,
            total_steps,
            8e-4,
            3e-4,
            backbone_lrs,
            stage1_steps=s1,
            warmup_steps=w,
        )
        assert at_s1["head"] == pytest.approx(3e-4, rel=1e-9)
        expected_after_head = 3e-4 * (
            0.05 + 0.5 * 0.95 * (1.0 + math.cos(math.pi / (total_steps - s1)))
        )
        assert after_s1["head"] == pytest.approx(expected_after_head, rel=1e-9)
        if total_steps >= 600:
            assert abs(after_s1["head"] - at_s1["head"]) < 1e-6
        assert at_s1["stage4"] == 0.0
        assert after_s1["stage4"] == pytest.approx(backbone_lrs["stage4"] / w, rel=1e-9)


def test_selection_key_never_returns_none_even_when_guardrail_fails() -> None:
    failing_a = {
        "s_select": 0.91,
        "recall_nonlab_healthy": 0.84,  # fails >= 0.85
        "precision_nonlab_bs": 0.80,
        "recall_negative": 0.95,
    }
    failing_b = {
        "s_select": 0.72,
        "recall_nonlab_healthy": 0.90,
        "precision_nonlab_bs": None,  # no BS predictions -> 0.0
        "recall_negative": 0.80,  # fails >= 0.90
    }
    passing = {
        "s_select": 0.65,
        "recall_nonlab_healthy": 0.85,
        "precision_nonlab_bs": 0.70,
        "recall_negative": 0.90,
    }
    k_fa = selection_key(failing_a)
    k_fb = selection_key(failing_b)
    k_pass = selection_key(passing)
    assert k_fa is not None and k_fb is not None and k_pass is not None
    assert k_fa[0] is False
    assert k_fb[0] is False and k_fb[3] == 0.0
    assert k_pass[0] is True
    assert k_pass > k_fa > k_fb


def test_calibration_splits_indomain_and_negative_masks(tmp_path: Path) -> None:
    model = build_cnn_gru_model("resnet50", 9, use_gru_head=False)
    rng = np.random.default_rng(123)
    # 70 in-domain rows (0..7) + 30 negative rows (8)
    logits = rng.normal(scale=0.5, size=(100, 9))
    targets = np.array([i % 8 for i in range(70)] + [8] * 30, dtype=np.int64)
    for i in range(70):
        logits[i, targets[i]] += 2.8
    for i in range(70, 100):
        logits[i, 8] += 4.0

    result = calibrate_and_export(
        model,
        logits,
        targets,
        tmp_path / "run",
        model_version="calib-test-v1",
        target_coverage=0.85,
        artifact_kind="trained",
    )
    assert 0.05 <= result["temperature"] <= 20.0
    assert 0.0 <= result["threshold"] <= 1.0
    assert result["in_domain_count"] == 70
    assert result["negative_count"] == 30
    assert result["in_domain_coverage"] >= 0.85 - 1e-12
    assert result["negative_rejection_rate"] > 0.9
    export_dir = tmp_path / "run/model"
    assert {p.name for p in export_dir.iterdir()} == set(CNN_GRU_FILES)

    # extra_metadata must not clobber locked schema/calibration fields
    import json

    calibrate_and_export(
        model,
        logits,
        targets,
        tmp_path / "run_extra",
        model_version="calib-test-v2",
        target_coverage=0.85,
        artifact_kind="trained",
        extra_metadata={
            "model_version": "tampered",
            "schema_version": 99,
            "note": "ok",
        },
    )
    exported_meta = json.loads(
        (tmp_path / "run_extra/model/leaf-classifier.json").read_text(encoding="utf-8")
    )
    assert exported_meta["model_version"] == "calib-test-v2"
    assert exported_meta["schema_version"] == 1
    assert exported_meta["note"] == "ok"


@pytest.mark.parametrize(
    ("targets", "message"),
    [([0, 0], "negative sample"), ([0.0, 8.9], "integer class indices")],
)
def test_calibration_rejects_unqualified_targets_before_export(
    tmp_path: Path, targets: list[float], message: str
) -> None:
    model = build_cnn_gru_model("resnet50", 9, use_gru_head=False)
    logits = np.zeros((2, 9))
    logits[:, 0] = 10.0
    output = tmp_path / "run"

    with pytest.raises(ValueError, match=message):
        calibrate_and_export(
            model, logits, np.asarray(targets), output, model_version="calib-v1"
        )

    assert not output.exists()


@pytest.mark.parametrize(
    "options",
    [{"guardrail_passed": "false"}, {"extra_metadata": {"guardrail_passed": True}}],
)
def test_export_requires_explicit_boolean_selection(
    tmp_path: Path, options: dict[str, Any]
) -> None:
    model = build_cnn_gru_model("resnet50", 9, use_gru_head=False)
    logits = np.eye(9)[[0, 8]]
    with pytest.raises(ValueError, match="guardrail_passed"):
        calibrate_and_export(
            model, logits, [0, 8], tmp_path / "run", model_version="guard-v1", **options
        )
    assert not (tmp_path / "run").exists()


def test_atomic_resume_and_export_bundle_contains_exactly_three_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows: list[dict[str, str]] = []
    for split, n_per_class in (("train", 2), ("val_select", 1), ("val_calib", 2)):
        for c_idx, c_name in enumerate(CLASSES):
            for k in range(n_per_class):
                rel = f"imgs/{split}_{c_idx}_{k}.png"
                img_path = data_dir / rel
                img_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), ((c_idx * 25) % 255, 110, 40)).save(img_path)
                rows.append(
                    {
                        "path": rel,
                        "class_id": c_name,
                        "group": f"{split}:{c_idx}:{k}",
                        "sha256": hashlib.sha256(img_path.read_bytes()).hexdigest(),
                        "source": "plantcity" if c_idx != 8 else "soil",
                        "domain": "field" if c_idx != 8 else "negative",
                        "split": split,
                    }
                )
    manifest_csv = data_dir / "train-val-calib.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    out_dir = tmp_path / "out"
    config: dict[str, Any] = {
        "architecture": "resnet50",
        "use_gru_head": True,
        "hidden_dim": 16,
        "proj_dim": 32,
        "model_version": "smoke-cnn-gru-v1",
        "size": 64,
        "epochs": 2,
        "samples": 16,
        "batch_size": 4,
        "grad_accum": 2,
        "workers": 0,
        "seed": 7,
        "ema_start_epoch": 1,
        "ema_max_decay": 0.993,
        "artifact_kind": "trained",
    }

    # Phase A: run epoch 1 and verify last.pt has ema_model & ema_updates, and BN stayed eval.
    res1 = train_cnn_gru(
        manifest_csv, data_dir, out_dir, **{**config, "stop_after_epoch": 1}
    )
    last_state = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=True)
    assert last_state["epoch"] == 1
    assert "ema_model" in last_state
    assert last_state["ema_updates"] > 0
    assert res1["bn_all_eval"] is True

    # Verify 2-phase atomic failure on epoch 2 leaves last.pt pointing at epoch 1 best
    replace_orig = os.replace
    with monkeypatch.context() as patch:

        def interrupt_last(src: Path, dst: Path) -> None:
            if dst == out_dir / "last.pt":
                raise OSError("simulated power loss before last.pt commit")
            replace_orig(src, dst)

        patch.setattr(os, "replace", interrupt_last)
        with pytest.raises(OSError, match="simulated power loss"):
            train_cnn_gru(manifest_csv, data_dir, out_dir, **config)

    still_ep1 = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=True)
    assert still_ep1["epoch"] == 1

    # Phase B: resume cleanly to epoch 2 & export exact 3 files
    res2 = train_cnn_gru(manifest_csv, data_dir, out_dir, **config)
    assert res2["bn_all_eval"] is True
    assert len(res2["history"]) == 2
    export_dir = out_dir / "model"
    assert {p.name for p in export_dir.iterdir()} == set(CNN_GRU_FILES)
    assert not (export_dir / "LICENSE.md").exists()
    assert (export_dir / "model.safetensors").stat().st_size <= MAX_WEIGHT_BYTES
    assert (out_dir / "frozen-models.json").is_file()
    validate_cnn_gru_export(export_dir, require_trained=True)
    predictor = CnnGruPredictor(export_dir, require_trained=True)
    pred = predictor.predict(Image.new("RGB", (40, 30), (20, 120, 40)))
    assert pred.model_version == "smoke-cnn-gru-v1"

    # Resuming with altered hyperparameters or invalid backbone_lrs must fail closed
    with pytest.raises(ValueError, match="configuration differs"):
        train_cnn_gru(
            manifest_csv, data_dir, out_dir, **{**config, "head_peak_lr": 1e-3}
        )
    with pytest.raises(ValueError, match="backbone_lrs"):
        train_cnn_gru(
            manifest_csv,
            data_dir,
            tmp_path / "bad_lrs",
            **{**config, "backbone_lrs": {"stage4": 1e-4}},
        )


def test_registration_cli_resolves_g0_g3_downloads_g2_and_syncs_drive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_reg = (
        Path(__file__).resolve().parents[2]
        / "docs/progress/2026-11-18-cnn-bigru/registration.json"
    )
    g0 = _resolve_registration_experiment(real_reg, "G0")
    g1 = _resolve_registration_experiment(real_reg, "G1")
    g2 = _resolve_registration_experiment(real_reg, "G2")
    g3 = _resolve_registration_experiment(real_reg, "G3")
    assert g0["architecture"] == "resnet50" and g0["use_gru_head"] is False
    assert g0["pretrained_weights"] == "IMAGENET1K_V2" and g0["pretrained_url"] is None
    assert g1["use_gru_head"] is True and g1["ema_max_decay"] == pytest.approx(0.992)
    assert g2["pretrained_weights"] is None and str(g2["pretrained_url"]).endswith(
        "convnext_tiny_22k_1k_384.pth"
    )
    assert g2["mixstyle"] is True and g2["mixstyle_p"] == pytest.approx(0.35)
    assert g2["freeze_bn_eval"] is False and g2["stochastic_depth_prob"] == 0.15
    assert g3["architecture"] == "efficientnet_v2_s"
    assert g3["pretrained_weights"] == "IMAGENET1K_V1"

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows: list[dict[str, str]] = []
    for split, n_per_class in (("train", 1), ("val_select", 1), ("val_calib", 1)):
        for c_idx, c_name in enumerate(CLASSES):
            rel = f"imgs/{split}_{c_idx}.png"
            img_path = data_dir / rel
            img_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), ((c_idx * 25) % 255, 110, 40)).save(img_path)
            rows.append(
                {
                    "path": rel,
                    "class_id": c_name,
                    "group": f"{split}:{c_idx}",
                    "sha256": hashlib.sha256(img_path.read_bytes()).hexdigest(),
                    "source": "plantcity" if c_idx != 8 else "soil",
                    "domain": "field" if c_idx != 8 else "negative",
                    "split": split,
                }
            )
    manifest_csv = data_dir / "train-val-calib.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tiny_reg = tmp_path / "registration.json"
    reg_payload = json.loads(real_reg.read_text(encoding="utf-8"))
    reg_payload["experiments"]["G2"].update(
        {
            "input_size": 64,
            "epochs": 2,
            "cooldown_epochs": 1,
            "samples_per_epoch": 8,
            "batch_size": 4,
            "proj_dim": 32,
            "hidden_dim": 16,
            "ema_start_epoch": 1,
        }
    )
    tiny_reg.write_text(json.dumps(reg_payload), encoding="utf-8")

    # Stage a tiny valid ConvNeXt state_dict without touching remote URLs.
    from torchvision import models as tv_models  # type: ignore[import-untyped]

    ref_sd = tv_models.convnext_tiny(weights=None).state_dict()
    downloaded_urls: list[str] = []

    def fake_download(url: str, dst: str, progress: bool = True) -> None:
        downloaded_urls.append(url)
        torch.save({"model": ref_sd}, dst)

    monkeypatch.setattr(torch.hub, "download_url_to_file", fake_download)

    out_dir = tmp_path / "runs/G2"
    drive_dir = tmp_path / "drive/G2"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ai.cnn_gru_train",
            "--registration",
            str(tiny_reg),
            "--experiment",
            "G2",
            "--manifest",
            str(manifest_csv),
            "--data",
            str(data_dir),
            "--output",
            str(out_dir),
            "--drive-dir",
            str(drive_dir),
            "--workers",
            "0",
        ],
    )
    train_main()
    assert len(downloaded_urls) == 1
    assert (out_dir.parent / "_pretrained/convnext_tiny_22k_1k_384.pth").is_file()
    leaf_meta = json.loads((out_dir / "model/leaf-classifier.json").read_text("utf-8"))
    assert type(leaf_meta["guardrail_passed"]) is bool
    assert (
        leaf_meta["model_version"] == "cnn-bigru-20261118-g2-convnext_tiny_axial_bigru"
    )
    assert (drive_dir / "last.pt").is_file()
    assert (drive_dir / "history.json").is_file()

    # --resume uses existing cached weights & completed epochs cleanly without re-downloading.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ai.cnn_gru_train",
            "--registration",
            str(tiny_reg),
            "--experiment",
            "G2",
            "--manifest",
            str(manifest_csv),
            "--data",
            str(data_dir),
            "--output",
            str(out_dir),
            "--drive-dir",
            str(drive_dir),
            "--workers",
            "0",
            "--resume",
        ],
    )
    train_main()
    assert len(downloaded_urls) == 1


def test_efficientnet_disables_cuda_fp16_amp_while_preserving_pin_memory() -> None:
    cuda_dev = torch.device("cuda")
    cpu_dev = torch.device("cpu")
    assert _cuda_transfer_and_amp(cuda_dev, "efficientnet_v2_s") == (True, False)
    assert _cuda_transfer_and_amp(cuda_dev, "resnet50") == (True, True)
    assert _cuda_transfer_and_amp(cuda_dev, "convnext_tiny") == (True, True)
    assert _cuda_transfer_and_amp(cpu_dev, "efficientnet_v2_s") == (False, False)
    assert _cuda_transfer_and_amp(cpu_dev, "resnet50") == (False, False)
