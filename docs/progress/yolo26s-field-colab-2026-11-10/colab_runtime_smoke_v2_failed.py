"""Colab runtime fixture: prepare/inspect augmentation first, then call run_training()."""

import hashlib
import io
import json
import random
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import albumentations as A
import numpy as np
import torch
import ultralytics
from PIL import Image, ImageDraw
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset

if not __debug__:
    raise RuntimeError("Runtime checks require assertions enabled")
if ultralytics.__version__ != "8.4.204":
    raise RuntimeError("Unexpected Ultralytics version")

WORK = Path("/content/yolo26s-field/runtime-smoke-v2")
DRIVE = Path("/content/drive/MyDrive/TomatoAI/yolo26s-field-2026-11-10/runtime-smoke-v2")
PACKAGE_SHA256 = "fe1f69bee9206df310269f3601ac4fff82fcc59e550776675f2630c2755b20f6"
PRETRAINED = Path("/content/yolo26s-field/pretrained/yolo26s.pt")
PRETRAINED_SHA256 = "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
AUGMENT = {"mosaic": 0.5, "degrees": 10.0, "translate": 0.05, "scale": 0.3, "fliplr": 0.5, "flipud": 0.0, "hsv_h": 0.005, "hsv_s": 0.20, "hsv_v": 0.20, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0}


def extra_augmentations():
    return [A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.35), A.ImageCompression(quality_range=(65, 100), p=0.15), A.GaussNoise(std_range=(0.01, 0.03), p=0.10), A.MotionBlur(blur_limit=3, p=0.05), A.RandomShadow(p=0.15)]


def prepare():
    from google.colab import files
    from IPython.display import display

    cached = DRIVE.parent / "runtime-smoke-v1/runtime-smoke-data.zip"
    if cached.is_file():
        payload = cached.read_bytes()
    else:
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise ValueError("Upload exactly runtime-smoke-data.zip")
        payload = next(iter(uploaded.values()))
    if hashlib.sha256(payload).hexdigest() != PACKAGE_SHA256:
        raise ValueError("Runtime fixture hash mismatch")
    if not DRIVE.parent.is_dir():
        raise RuntimeError("Task Drive directory is not mounted")
    WORK.mkdir(parents=True, exist_ok=True)
    DRIVE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if sum(m.file_size for m in archive.infolist()) > 32 * 1024 * 1024:
            raise ValueError("Fixture exceeds size limit")
        manifest = json.loads(archive.read("upload-manifest.json"))
        if len(archive.namelist()) != len(manifest) + 1 or set(archive.namelist()) != set(manifest) | {"upload-manifest.json"}:
            raise ValueError("Unexpected fixture members")
        for name, expected in manifest.items():
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or "\\" in name or hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError("Unsafe or changed fixture member")
        archive.extractall(WORK / "data")
    fixture = json.loads((WORK / "data/fixture.json").read_bytes())
    if fixture["candidate_eligible"] or fixture["final_holdout_eligible"]:
        raise ValueError("Runtime fixture must never be a final benchmark")
    names = dict(enumerate(fixture["names"]))
    data = {"path": str(WORK / "data"), "train": "images/train", "val": "images/val", "names": names, "nc": len(names), "channels": 3}
    (WORK / "data.yaml").write_text(json.dumps(data), encoding="utf-8")
    (DRIVE / "fixture.json").write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    (DRIVE / "runtime-smoke-data.zip").write_bytes(payload)
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 640, "seed": 20261110, "augmentations": extra_augmentations(), **AUGMENT})
    random.seed(20261110)
    np.random.seed(20261110)
    torch.manual_seed(20261110)
    dataset = build_yolo_dataset(cfg, str(WORK / "data/images/train"), 2, data, mode="train", stride=32)
    config = {"purpose": "runtime_only_not_candidate_selection", "augmentation": AUGMENT, "extra_augmentations": [str(t) for t in cfg.augmentations], "seed": 20261110, "imgsz": 640, "package_sha256": PACKAGE_SHA256}
    (DRIVE / "augmentation-config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for page in range((len(dataset) + 3) // 4):
        sheet = Image.new("RGB", (1280, 1344), "white")
        for slot in range(4):
            index = page * 4 + slot
            if index >= len(dataset):
                break
            sample = dataset[index]
            rgb = sample["img"].permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(rgb)
            draw = ImageDraw.Draw(image)
            boxes = sample["bboxes"].cpu().numpy()
            classes = sample["cls"].flatten().cpu().numpy()
            if len(boxes) != len(classes) or not np.isfinite(boxes).all():
                raise ValueError("Invalid augmented boxes")
            for box, class_id in zip(boxes, classes):
                cx, cy, w, h = box
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1 and int(class_id) in names):
                    raise ValueError("Augmentation broke coordinates or class")
                xyxy = [(cx - w / 2) * image.width, (cy - h / 2) * image.height, (cx + w / 2) * image.width, (cy + h / 2) * image.height]
                draw.rectangle(xyxy, outline="red", width=2)
                draw.text((xyxy[0] + 2, xyxy[1] + 2), names[int(class_id)], fill="red", stroke_fill="white", stroke_width=1)
            x, y = (slot % 2) * 640, (slot // 2) * 672
            sheet.paste(image, (x, y))
            ImageDraw.Draw(sheet).text((x + 4, y + 644), f"augmented sample {index}; boxes={len(boxes)}", fill="black")
        destination = DRIVE / f"augmentation-{page + 1}.png"
        sheet.save(destination)
        display(sheet)
    print("RUNTIME_FIXTURE_PREPARED; inspect augmentation before run_training(); no training started")


class PlannedCheckpointStop(RuntimeError):
    pass


def run_training():
    if not (WORK / "data.yaml").is_file():
        raise RuntimeError("Prepare and inspect the runtime fixture first")
    if hashlib.sha256(PRETRAINED.read_bytes()).hexdigest() != PRETRAINED_SHA256:
        raise ValueError("Official checkpoint changed")
    if (DRIVE / "train").exists():
        raise FileExistsError("Smoke attempt exists; inspect it before any rerun")
    report = {"purpose": "runtime_smoke_only_not_an_accuracy_experiment", "candidate_eligible": False, "status": "running", "package_sha256": PACKAGE_SHA256}
    report_path = DRIVE / "runtime-report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def stop_after_save(trainer):
        if trainer.epoch == 0:
            raise PlannedCheckpointStop("Intentional stop after first complete checkpoint")

    try:
        model = YOLO(str(PRETRAINED))
        model.add_callback("on_model_save", stop_after_save)
        try:
            model.train(data=str(WORK / "data.yaml"), epochs=3, imgsz=640, batch=2, workers=0, device=0, optimizer="AdamW", lr0=0.001, warmup_epochs=0.0, seed=20261110, deterministic=True, project=str(DRIVE), name="train", save_period=1, close_mosaic=1, plots=True, amp=True, augmentations=extra_augmentations(), **AUGMENT)
        except PlannedCheckpointStop:
            pass
        checkpoint = DRIVE / "train/weights/last.pt"
        # This checkpoint was generated by this smoke run, not supplied by a third party.
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["epoch"] != 0 or not saved["optimizer"]["state"] or not saved["scaler"]:
            raise RuntimeError("First checkpoint lacks epoch, optimizer, or AMP scaler state")
        shutil.copy2(checkpoint, DRIVE / "interrupted-last.pt")
        report["interrupted_epoch"] = saved["epoch"]
        report["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

        def verify_resume(trainer):
            if trainer.start_epoch != 1 or trainer.scaler.state_dict() != saved["scaler"] or trainer.ema.updates != saved["updates"]:
                raise RuntimeError("Epoch/scaler/EMA resume mismatch")
            actual = trainer.optimizer.state_dict()["state"]
            expected = saved["optimizer"]["state"]
            if actual.keys() != expected.keys():
                raise RuntimeError("Optimizer state IDs mismatch")
            tensors = 0
            for parameter, state in expected.items():
                if actual[parameter].keys() != state.keys():
                    raise RuntimeError("Optimizer state fields mismatch")
                for key, value in state.items():
                    other = actual[parameter][key]
                    if isinstance(value, torch.Tensor):
                        if not torch.equal(other.cpu(), value.to(dtype=other.dtype)):
                            raise RuntimeError("Optimizer tensor failed restoration")
                        tensors += 1
                    elif other != value:
                        raise RuntimeError("Optimizer scalar failed restoration")
            report.update(resume_start_epoch=trainer.start_epoch, optimizer_state_tensors_verified=tensors, scaler_restored=True, ema_updates_restored=True)

        resumed = YOLO(str(checkpoint))
        resumed.add_callback("on_train_start", verify_resume)
        resumed.train(resume=True, augmentations=extra_augmentations())
        if resumed.trainer.epoch != 2:
            raise RuntimeError("Resumed run did not reach expected final epoch")
        checks = []
        for nms in (True, False):
            metrics = resumed.val(data=str(WORK / "data.yaml"), imgsz=640, batch=2, device=0, workers=0, nms=nms, plots=False, project=str(DRIVE), name=f"validator-nms-{nms}")
            if not all(np.isfinite(v) for v in metrics.results_dict.values()):
                raise RuntimeError("Non-finite validation metrics")
            checks.append({"nms": nms, "executed": True, "metrics_are_not_acceptance_evidence": True})
        report.update(status="success", resumed_last_epoch=2, validator_checks=checks, full_checkpoint_resume_verified=True)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("RUNTIME_SMOKE_OK", json.dumps(report))


if __name__ == "__main__":
    prepare()
