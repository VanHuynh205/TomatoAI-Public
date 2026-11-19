"""Prepare leak-free 3-way split, registration, <=50 MiB Colab data chunks, and code bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import shutil
import stat
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = Path(__file__).resolve().parent
SCRATCH_ROOT = ROOT / ".scratch"
RUN_ID = "cnn-bigru-20261118"
RUN_SCRATCH = SCRATCH_ROOT / RUN_ID
SEED = "tomato-cnn-bigru-v1"
MAX_PART_BYTES = 52_428_800  # 50 MiB

SOURCE_V2_CSV = ROOT / "infra/derived/dinov3-colab-v2/train-val.csv"
DATA_ARCHIVES: tuple[tuple[str, Path], ...] = (
    (
        "dinov3-colab-input.zip",
        ROOT / "infra/derived/dinov3-colab-v1/dinov3-colab-input.zip",
    ),
    (
        "dinov3-additions.zip",
        ROOT / "infra/derived/dinov3-colab-v2/dinov3-additions.zip",
    ),
    (
        "dinov3-colab-holdout.zip",
        ROOT / "infra/derived/dinov3-colab-v1/dinov3-colab-holdout.zip",
    ),
    (
        "dinov3-slif-pilot.zip",
        ROOT / "infra/derived/dinov3-colab-v2/dinov3-slif-pilot.zip",
    ),
)
BASE_AI_FILES = (
    "ai/__init__.py",
    "ai/artifacts.py",
    "ai/calibration.py",
    "ai/contracts.py",
    "ai/data.py",
    "ai/dinov2_field.py",
    "ai/dinov3_field.py",
    "ai/dinov3_inference.py",
    "ai/models.py",
    "ai/preprocessing.py",
)
CNN_GRU_AI_FILES = (
    "ai/cnn_gru_model.py",
    "ai/cnn_gru_train.py",
    "ai/cnn_gru_inference.py",
)
CLASSES = (
    "bacterial_spot",
    "early_blight",
    "late_blight",
    "leaf_mold",
    "septoria_leaf_spot",
    "spider_mites",
    "yellow_leaf_curl",
    "healthy",
    "not_tomato_leaf",
)


def sha256_file(path: Path) -> str:
    """Return SHA-256 hex digest of a regular file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _det_hash(seed: str, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def _ensure_scratch_prerequisites() -> None:
    for required in ("scratch-before.json", "scratch-rules.json"):
        if not (REPORT_DIR / required).is_file():
            raise RuntimeError(
                f"Missing {required} in {REPORT_DIR} before writing .scratch/"
            )


def build_three_way_split(
    rows: list[dict[str, str]],
    seed: str = SEED,
    ratios: tuple[float, float, float] = (0.74, 0.13, 0.13),
) -> dict[str, str]:
    """Deterministically partition groups into train / val_select / val_calib."""
    r_train, r_sel, r_cal = ratios
    grp_to_rows: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grp_to_rows[row["group"]].append(row)

    assignment = {
        grp: "train"
        for grp, grp_rows in grp_to_rows.items()
        if grp_rows[0]["source"] == "hossain_v1"
    }

    bap_groups = sorted(
        (
            grp
            for grp, grp_rows in grp_to_rows.items()
            if grp_rows[0]["source"] == "bapari_2025_v1"
        ),
        key=lambda grp: (-len(grp_to_rows[grp]), _det_hash(seed, grp)),
    )
    bap_counts: dict[str, Counter[str]] = {
        "val_select": Counter(),
        "val_calib": Counter(),
    }
    bap_tot = Counter(r["class_id"] for grp in bap_groups for r in grp_to_rows[grp])
    bap_weights = (
        ("bacterial_spot", 4.0),
        ("septoria_leaf_spot", 3.0),
        ("healthy", 1.0),
    )
    for grp in bap_groups:
        gc = Counter(r["class_id"] for r in grp_to_rows[grp])

        def pair_imbalance(
            split_name: str, gc: Counter[str] = gc, grp: str = grp
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
                _det_hash(seed, f"{split_name}:{grp}"),
            )

        chosen = min(("val_select", "val_calib"), key=pair_imbalance)
        assignment[grp] = chosen
        bap_counts[chosen].update(gc)

    strata: defaultdict[tuple[str, str, str], list[str]] = defaultdict(list)
    for grp, grp_rows in grp_to_rows.items():
        if grp not in assignment:
            strata[
                (grp_rows[0]["source"], grp_rows[0]["domain"], grp_rows[0]["class_id"])
            ].append(grp)

    for _, groups in sorted(strata.items()):
        if len(groups) < 6:
            for grp in groups:
                assignment[grp] = "train"
            continue
        min_vg = 2 if len(groups) >= 10 else 1
        tot = sum(len(grp_to_rows[grp]) for grp in groups)
        targets = {
            "train": r_train * tot,
            "val_select": r_sel * tot,
            "val_calib": r_cal * tot,
        }
        cur_i: Counter[str] = Counter()
        cur_g: Counter[str] = Counter()
        for idx, grp in enumerate(sorted(groups, key=lambda x: _det_hash(seed, x))):
            rem = len(groups) - idx
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
                    _det_hash(seed, f"{sp}:{grp}"),
                ),
            )
            assignment[grp] = best
            cur_i[best] += len(grp_to_rows[grp])
            cur_g[best] += 1
    return assignment


def _load_holdouts() -> tuple[dict[str, list[dict[str, str]]], dict[str, str]]:
    holdout_zip = ROOT / "infra/derived/dinov3-colab-v1/dinov3-colab-holdout.zip"
    slif_zip = ROOT / "infra/derived/dinov3-colab-v2/dinov3-slif-pilot.zip"
    holdouts: dict[str, list[dict[str, str]]] = {}
    csv_hashes: dict[str, str] = {}
    with ZipFile(holdout_zip) as zf:
        for name, member in (
            ("legacy", "legacy/test.csv"),
            ("external", "external/test.csv"),
        ):
            raw = zf.read(member)
            csv_hashes[name] = hashlib.sha256(raw).hexdigest()
            holdouts[name] = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    with ZipFile(slif_zip) as zf:
        raw = zf.read("slif/test.csv")
        csv_hashes["slif"] = hashlib.sha256(raw).hexdigest()
        holdouts["slif"] = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    return holdouts, csv_hashes


def verify_no_leakage(
    splits: dict[str, list[dict[str, str]]],
    holdouts: dict[str, list[dict[str, str]]],
) -> tuple[int, int]:
    """Assert 0 intersection across all 15 set pairs on 5 identity keys and 0 split_block overlap."""
    sets = {**splits, **holdouts}
    names = list(sets)
    pair_count = 0
    leakage_pairs = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair_count += 1
            for field in (
                "group",
                "sha256",
                "original_sha256",
                "normalized_sha256",
                "path",
            ):
                a = {r[field] for r in sets[names[i]] if r.get(field)}
                b = {r[field] for r in sets[names[j]] if r.get(field)}
                if a & b:
                    leakage_pairs += 1
                    raise AssertionError(
                        f"Leakage between {names[i]} and {names[j]} on {field}"
                    )
    assert pair_count == 15, pair_count
    sb_train = {r["split_block"] for r in splits["train"] if r.get("split_block")}
    sb_eval = {
        r["split_block"]
        for r in splits["val_select"] + splits["val_calib"]
        if r.get("split_block")
    }
    sb_overlap = len(sb_train & sb_eval)
    assert sb_overlap == 0, f"split_block overlap: {sb_overlap}"
    return leakage_pairs, sb_overlap


def generate_split_manifest() -> dict[str, Any]:
    """Generate and verify .scratch/cnn-bigru-20261118/manifests/train-val-calib.csv."""
    _ensure_scratch_prerequisites()
    with SOURCE_V2_CSV.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    assert len(rows) == 9711, len(rows)
    assignment = build_three_way_split(rows, seed=SEED)
    out_rows = [{**row, "split": assignment[row["group"]]} for row in rows]
    splits = {
        sp: [r for r in out_rows if r["split"] == sp]
        for sp in ("train", "val_select", "val_calib")
    }
    counts = {sp: len(sp_rows) for sp, sp_rows in splits.items()}
    assert counts == {"train": 6762, "val_select": 1422, "val_calib": 1527}, counts
    assert (
        sum(1 for r in splits["val_select"] if r["source"] == "bapari_2025_v1") == 149
    )
    assert sum(1 for r in splits["val_calib"] if r["source"] == "bapari_2025_v1") == 287
    assert sum(1 for r in splits["train"] if r["source"] == "bapari_2025_v1") == 0
    assert sum(1 for r in splits["train"] if r["source"] == "hossain_v1") == 381

    holdouts, holdout_csv_hashes = _load_holdouts()
    assert {k: len(v) for k, v in holdouts.items()} == {
        "legacy": 2647,
        "external": 460,
        "slif": 21,
    }
    leakage_pairs, sb_overlap = verify_no_leakage(splits, holdouts)

    manifest_path = RUN_SCRATCH / "manifests/train-val-calib.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".pending")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    os.replace(tmp, manifest_path)
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "counts": counts,
        "leakage_pairs": leakage_pairs,
        "split_block_overlap": sb_overlap,
        "holdout_csv_hashes": holdout_csv_hashes,
    }


def _check_zip_safe(archive: Path) -> None:
    with ZipFile(archive) as zf:
        entries = zf.infolist()
        assert len(entries) == len({e.filename for e in entries}), archive.name
        for entry in entries:
            assert not stat.S_ISLNK(entry.external_attr >> 16), entry.filename
            parts = Path(entry.filename).parts
            assert not entry.filename.startswith("/") and ".." not in parts, (
                entry.filename
            )


def build_data_parts(max_part_bytes: int = MAX_PART_BYTES) -> dict[str, Any]:
    """Split the 4 source image ZIPs into <=50 MiB chunks without modifying originals."""
    _ensure_scratch_prerequisites()
    parts_dir = RUN_SCRATCH / "colab-bundle/parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    archives_meta: list[dict[str, Any]] = []

    for archive_name, src_path in DATA_ARCHIVES:
        _check_zip_safe(src_path)
        src_size = src_path.stat().st_size
        src_sha = sha256_file(src_path)
        stem = src_path.stem
        parts: list[dict[str, Any]] = []
        reasm_hasher = hashlib.sha256()
        reasm_bytes = 0

        with src_path.open("rb") as src:
            idx = 0
            while True:
                chunk = src.read(max_part_bytes)
                if not chunk:
                    break
                part_name = f"{stem}.part-{idx:02d}.bin"
                part_path = parts_dir / part_name
                tmp = part_path.with_suffix(".pending")
                tmp.write_bytes(chunk)
                os.replace(tmp, part_path)
                p_size = part_path.stat().st_size
                assert 0 < p_size <= max_part_bytes, (part_name, p_size)
                p_sha = sha256_file(part_path)
                parts.append(
                    {
                        "part_name": part_name,
                        "index": idx,
                        "bytes": p_size,
                        "sha256": p_sha,
                    }
                )
                idx += 1

        for part in parts:
            data = (parts_dir / str(part["part_name"])).read_bytes()
            reasm_hasher.update(data)
            reasm_bytes += len(data)
        assert reasm_bytes == src_size and reasm_hasher.hexdigest() == src_sha, (
            archive_name
        )

        archives_meta.append(
            {
                "archive_name": archive_name,
                "source_relpath": src_path.relative_to(ROOT).as_posix(),
                "total_bytes": src_size,
                "sha256": src_sha,
                "part_count": len(parts),
                "parts": parts,
            }
        )

    manifest_parts = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "max_part_bytes": max_part_bytes,
        "archives": archives_meta,
    }
    mp_path = parts_dir / "manifest-parts.json"
    tmp_mp = mp_path.with_suffix(".pending")
    tmp_mp.write_text(json.dumps(manifest_parts, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_mp, mp_path)
    return {
        "manifest_parts_path": mp_path,
        "manifest_parts_sha256": sha256_file(mp_path),
        "manifest_parts": manifest_parts,
    }


def build_registration(
    manifest_info: dict[str, Any], parts_info: dict[str, Any]
) -> dict[str, Any]:
    """Write docs/progress/2026-11-18-cnn-bigru/registration.json locking G0..G3."""
    archive_hashes = {
        item["archive_name"]: item["sha256"]
        for item in parts_info["manifest_parts"]["archives"]
    }
    h_csv = manifest_info["holdout_csv_hashes"]
    inputs_sha256 = {
        "source_train_val_csv": sha256_file(SOURCE_V2_CSV),
        "three_way_manifest_csv": manifest_info["manifest_sha256"],
        "dinov3_colab_input_zip": archive_hashes["dinov3-colab-input.zip"],
        "dinov3_additions_zip": archive_hashes["dinov3-additions.zip"],
        "dinov3_colab_holdout_zip": archive_hashes["dinov3-colab-holdout.zip"],
        "dinov3_slif_pilot_zip": archive_hashes["dinov3-slif-pilot.zip"],
        "legacy_test_csv": h_csv["legacy"],
        "external_test_csv": h_csv["external"],
        "slif_test_csv": h_csv["slif"],
        "manifest_parts_json": parts_info["manifest_parts_sha256"],
    }
    shared_schedule = {
        "input_size": 288,
        "epochs": 22,
        "samples_per_epoch": 4800,
        "batch_size": 32,
        "steps_per_epoch": 150,
        "total_steps": 3300,
        "cooldown_epochs": 3,
        "head_peak_lr": 0.0008,
        "head_main_lr": 0.0003,
        "label_smoothing": 0.02,
        "ema_max_decay": 0.992,
        "ema_start_epoch": 4,
    }
    resnet_lrs = {"stage4": 8e-05, "stage3": 2.5e-05, "stage2": 1e-05}
    experiments = {
        "G0": {
            "name": "resnet50_gap_baseline",
            "architecture": "resnet50",
            "use_gru_head": False,
            "pretrained_source": "torchvision:ResNet50_Weights.IMAGENET1K_V2",
            "license": "BSD-3-Clause",
            "backbone_lrs": resnet_lrs,
            "freeze_bn_eval": True,
            "mixstyle": False,
            **shared_schedule,
        },
        "G1": {
            "name": "resnet50_axial_bigru",
            "architecture": "resnet50",
            "use_gru_head": True,
            "pretrained_source": "torchvision:ResNet50_Weights.IMAGENET1K_V2",
            "license": "BSD-3-Clause",
            "proj_dim": 512,
            "hidden_dim": 256,
            "backbone_lrs": resnet_lrs,
            "freeze_bn_eval": True,
            "mixstyle": False,
            **shared_schedule,
        },
        "G2": {
            "name": "convnext_tiny_axial_bigru",
            "architecture": "convnext_tiny",
            "use_gru_head": True,
            "pretrained_source": "convnext_tiny_22k_1k_384",
            "pretrained_url": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_384.pth",
            "license": "MIT",
            "proj_dim": 512,
            "hidden_dim": 256,
            "backbone_lrs": {"stage4": 5e-05, "stage3": 2e-05, "stage2": 8e-06},
            "freeze_bn_eval": False,
            "mixstyle": True,
            "mixstyle_p": 0.35,
            "mixstyle_alpha": 0.1,
            "stochastic_depth": 0.15,
            **shared_schedule,
        },
        "G3": {
            "name": "efficientnet_v2_s_axial_bigru",
            "architecture": "efficientnet_v2_s",
            "use_gru_head": True,
            "pretrained_source": "torchvision:EfficientNet_V2_S_Weights.IMAGENET1K_V1",
            "license": "BSD-3-Clause",
            "proj_dim": 512,
            "hidden_dim": 256,
            "backbone_lrs": resnet_lrs,
            "freeze_bn_eval": True,
            "mixstyle": False,
            **shared_schedule,
        },
    }
    registration = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "track": "Track A (v2-3way-9c)",
        "seed": SEED,
        "ratios": [0.74, 0.13, 0.13],
        "classes": list(CLASSES),
        "input_size": 288,
        "samples_per_epoch": 4800,
        "batch_size": 32,
        "steps_per_epoch": 150,
        "epochs": 22,
        "total_steps": 3300,
        "sampler": "cnn_bigru_weights_strict",
        "sampler_caps": {
            "field": 1.40,
            "other": 1.50,
            "controlled": 0.85,
            "negative": 1.05,
        },
        "temperature_fit_scope": "in_domain_9logits",
        "target_coverage": 0.85,
        "guardrail_thresholds": {
            "min_healthy_nonlab_recall": 0.85,
            "min_bs_nonlab_precision": 0.70,
            "min_negative_recall": 0.90,
        },
        "counts": manifest_info["counts"],
        "total_rows": 9711,
        "leakage_pairs": manifest_info["leakage_pairs"],
        "split_block_overlap": manifest_info["split_block_overlap"],
        "holdout_counts": {"legacy": 2647, "external": 460, "slif": 21},
        "experiments": experiments,
        "inputs_sha256": inputs_sha256,
    }
    reg_path = REPORT_DIR / "registration.json"
    tmp = reg_path.with_suffix(".pending")
    tmp.write_text(json.dumps(registration, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, reg_path)
    return {"registration_path": reg_path, "registration_sha256": sha256_file(reg_path)}


def build_code_bundle() -> dict[str, Any]:
    """Package ai/cnn_gru_*.py, shared ai/ modules, manifest, and registration into cnn-gru-code.zip."""
    _ensure_scratch_prerequisites()
    missing = [
        rel for rel in (*BASE_AI_FILES, *CNN_GRU_AI_FILES) if not (ROOT / rel).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Cannot build code bundle; missing files: {missing}")
    manifest_csv = RUN_SCRATCH / "manifests/train-val-calib.csv"
    reg_json = REPORT_DIR / "registration.json"
    assert manifest_csv.is_file() and reg_json.is_file()

    entries: list[tuple[Path, str]] = [
        *((ROOT / rel, rel) for rel in (*BASE_AI_FILES, *CNN_GRU_AI_FILES)),
        (manifest_csv, "manifests/train-val-calib.csv"),
        (reg_json, "registration.json"),
        (REPORT_DIR / "colab_cells.py", "colab_cells.py"),
    ]
    files_sha256 = {arcname: sha256_file(src) for src, arcname in entries}
    bundle_dir = RUN_SCRATCH / "colab-bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    hashes_path = bundle_dir / "bundle-hashes.json"
    bundle_hashes = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "files_sha256": files_sha256,
    }
    tmp_h = hashes_path.with_suffix(".pending")
    tmp_h.write_text(json.dumps(bundle_hashes, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_h, hashes_path)

    zip_path = bundle_dir / "cnn-gru-code.zip"
    tmp_z = zip_path.with_suffix(".pending")
    with ZipFile(tmp_z, "w", compression=ZIP_DEFLATED) as zf:
        for src, arcname in entries:
            assert not src.is_symlink(), src
            zf.write(src, arcname)
        zf.write(hashes_path, "bundle-hashes.json")
    os.replace(tmp_z, zip_path)
    _check_zip_safe(zip_path)
    zip_sha = sha256_file(zip_path)
    bundle_hashes["code_zip"] = {
        "name": zip_path.name,
        "bytes": zip_path.stat().st_size,
        "sha256": zip_sha,
    }
    hashes_path.write_text(json.dumps(bundle_hashes, indent=2) + "\n", encoding="utf-8")
    return {
        "zip_path": zip_path,
        "zip_sha256": zip_sha,
        "bytes": zip_path.stat().st_size,
    }


def run_smoke(steps: int) -> dict[str, Any]:
    """Run a bounded CPU/GPU 2-step smoke verification when ai/cnn_gru_*.py are available."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import torch

    inf_mod = importlib.import_module("ai.cnn_gru_inference")
    model_mod = importlib.import_module("ai.cnn_gru_model")
    train_mod = importlib.import_module("ai.cnn_gru_train")
    CNN_GRU_FILES = inf_mod.CNN_GRU_FILES
    validate_cnn_gru_export = inf_mod.validate_cnn_gru_export
    build_cnn_gru_model = model_mod.build_cnn_gru_model
    prior_matched_ce_loss = train_mod.prior_matched_ce_loss

    smoke_root = RUN_SCRATCH / "smoke-temp"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    smoke_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for arch in ("resnet50", "convnext_tiny"):
        model = build_cnn_gru_model(arch, len(CLASSES), use_gru_head=True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        offset = torch.zeros(len(CLASSES))
        x = torch.randn(2, 3, 288, 288)
        y = torch.tensor([0, 7], dtype=torch.long)
        losses = []
        for _ in range(steps):
            opt.zero_grad()
            loss = prior_matched_ce_loss(model(x), y, offset)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        assert all(math.isfinite(v) for v in losses), losses
        results[arch] = {"losses": losses}
    results["cnn_gru_files"] = list(CNN_GRU_FILES)
    results["validator_present"] = callable(validate_cnn_gru_export)
    return results


def _classify_scratch_path(relative: Path) -> tuple[str, str]:
    posix = relative.as_posix()
    parts = relative.parts
    if "__pycache__" in parts or ".pytest_cache" in parts or relative.suffix == ".pyc":
        return "DISPOSABLE", "Python/pytest cache; regenerable."
    if len(parts) >= 2 and parts[1] == RUN_ID:
        if relative.suffix == ".pending":
            return "DISPOSABLE", "Incomplete atomic-write staging file."
        if posix.endswith("/colab-bundle/parts/manifest-parts.json"):
            return (
                "KEEP_EVIDENCE",
                "Chunk verification manifest with byte sizes and SHA-256 digests.",
            )
        if posix.startswith(f".scratch/{RUN_ID}/colab-bundle/parts/"):
            return (
                "DISPOSABLE",
                "Split <=50 MiB upload chunk; safe to remove after verification.",
            )
        if posix.startswith(f".scratch/{RUN_ID}/colab-bundle/bundle-hashes.json"):
            return (
                "KEEP_EVIDENCE",
                "SHA-256 verification receipt for Colab code/data bundles.",
            )
        if (
            posix.startswith(f".scratch/{RUN_ID}/colab-bundle/")
            and relative.suffix == ".zip"
        ):
            return "KEEP_REUSE", "Primary compressed Colab bundle."
        if posix.startswith(f".scratch/{RUN_ID}/manifests/"):
            return "KEEP_REUSE", "Deterministic leak-free 3-way Track A split manifest."
        if posix.startswith(
            f".scratch/{RUN_ID}/results/"
        ) and relative.suffix.startswith(".part"):
            return (
                "DISPOSABLE",
                "Temporary <=45 MiB download chunk; safe to remove after reassembly verification.",
            )
        if posix.startswith(f".scratch/{RUN_ID}/results/") or relative.suffix == ".log":
            return (
                "KEEP_EVIDENCE",
                "Run logs, calibration, holdout metrics, or parity evidence.",
            )
        if (
            posix.startswith(
                (f".scratch/{RUN_ID}/tmp-extract/", f".scratch/{RUN_ID}/smoke")
            )
            or "trial-checkpoints" in parts
            or any(p.endswith("-temp") for p in parts)
        ):
            return "DISPOSABLE", "Temporary extraction or smoke directory."
        return "REVIEW_REQUIRED", "Unclassified artifact inside run scratch dir."

    # Preexisting files outside cnn-bigru-20261118 (aligned with 2026-11-07-retrain/assess_scratch.py)
    if len(parts) >= 2 and parts[1].startswith("plantdoc-"):
        if relative.suffix in {".py", ".csv"} or relative.name.endswith(".sha256"):
            return "KEEP_REUSE", "Preexisting PlantDoc audit/membership input."
        return (
            "KEEP_EVIDENCE",
            "Preexisting PlantDoc contact sheet or diagnosis report.",
        )
    if relative.suffix in {".py", ".ps1", ".spec", ".mjs"}:
        return "KEEP_REUSE", "Preexisting verification script; preserve untouched."
    if relative.suffix in {".json", ".jsonl", ".xml", ".log", ".md", ".txt", ".csv"}:
        return "KEEP_EVIDENCE", "Preexisting verification artifact; preserve untouched."
    return "REVIEW_REQUIRED", "Preexisting scratch file outside run scope."


def assess_scratch() -> dict[str, Any]:
    """Inventory .scratch/ safely via lstat() without following reparse points."""
    before_path = REPORT_DIR / "scratch-before.json"
    before_raw = json.loads(before_path.read_text(encoding="utf-8"))
    before = {k.replace("\\", "/"): v for k, v in before_raw.items()}

    records: list[dict[str, Any]] = []
    reparse_points: list[str] = []

    def walk(directory: Path) -> None:
        with os.scandir(directory) as it:
            for entry in sorted(it, key=lambda x: x.name):
                path = Path(entry.path)
                st = os.lstat(path)
                is_rp = stat.S_ISLNK(st.st_mode) or bool(
                    getattr(st, "st_file_attributes", 0) & 0x400
                )
                rel = path.relative_to(ROOT)
                if is_rp:
                    reparse_points.append(rel.as_posix())
                    continue
                if stat.S_ISDIR(st.st_mode):
                    walk(path)
                elif stat.S_ISREG(st.st_mode):
                    prior = before.get(rel.as_posix())
                    change = (
                        "new"
                        if prior is None
                        else "changed"
                        if prior != {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
                        else "preexisting_unchanged"
                    )
                    label, reason = _classify_scratch_path(rel)
                    digest = (
                        sha256_file(path)
                        if label in {"KEEP_REUSE", "KEEP_EVIDENCE"}
                        or change != "preexisting_unchanged"
                        else ""
                    )
                    records.append(
                        {
                            "path": rel.as_posix(),
                            "change_since_snapshot": change,
                            "label": label,
                            "reason": reason,
                            "size_bytes": st.st_size,
                            "mtime_ns": st.st_mtime_ns,
                            "sha256": digest,
                        }
                    )

    walk(SCRATCH_ROOT)
    missing = sorted(set(before) - {r["path"] for r in records})
    inv_path = REPORT_DIR / "scratch-inventory.csv"
    with inv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    changes = [
        r for r in records if r["change_since_snapshot"] != "preexisting_unchanged"
    ]
    summary = {
        "assessed_at_utc": datetime.now(UTC).isoformat(),
        "total_files": len(records),
        "labels": dict(Counter(r["label"] for r in records)),
        "changed_or_new_files": len(changes),
        "changed_or_new_labels": dict(Counter(r["label"] for r in changes)),
        "changed_or_new_bytes": sum(int(r["size_bytes"]) for r in changes),
        "missing_since_snapshot": missing,
        "review_required": [
            r["path"] for r in records if r["label"] == "REVIEW_REQUIRED"
        ],
        "reparse_points_preserved_and_skipped": reparse_points,
        "inventory_sha256": sha256_file(inv_path),
    }
    (REPORT_DIR / "scratch-assessment.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Only build 3-way split, data parts, and registration.",
    )
    parser.add_argument(
        "--code-bundle",
        action="store_true",
        help="Require ai/cnn_gru_*.py and build cnn-gru-code.zip.",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="Run bounded N-step smoke check after bundling code.",
    )
    args = parser.parse_args()

    manifest_info = generate_split_manifest()
    parts_info = build_data_parts()
    reg_info = build_registration(manifest_info, parts_info)
    output: dict[str, Any] = {
        "manifest_sha256": manifest_info["manifest_sha256"],
        "counts": manifest_info["counts"],
        "leakage_pairs": manifest_info["leakage_pairs"],
        "manifest_parts_sha256": parts_info["manifest_parts_sha256"],
        "registration_sha256": reg_info["registration_sha256"],
    }

    has_cnn_gru = all((ROOT / rel).is_file() for rel in CNN_GRU_AI_FILES)
    if (
        args.code_bundle
        or args.smoke_steps > 0
        or (not args.prepare_data and has_cnn_gru)
    ):
        output["code_bundle"] = build_code_bundle()
    elif not args.prepare_data:
        output["code_bundle"] = "skipped_waiting_for_ai_cnn_gru_modules"

    if args.smoke_steps > 0:
        output["smoke"] = run_smoke(args.smoke_steps)

    output["scratch_assessment"] = assess_scratch()
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
