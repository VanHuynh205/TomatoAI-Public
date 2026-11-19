"""Run in the authorized Colab notebook before data preparation or training."""

import hashlib
import inspect
import json
from importlib import metadata
from pathlib import Path

if __debug__:
    import torch
    import ultralytics
    from ultralytics.cfg import get_cfg
    from ultralytics.data.augment import Albumentations
    from ultralytics.engine.trainer import BaseTrainer
else:
    raise RuntimeError("API checks require assertions enabled")

assert ultralytics.__version__ == "8.4.204"
assert torch.cuda.is_available(), "GPU runtime is required"
cfg = get_cfg(overrides={"task": "detect", "optimizer": "MuSGD", "augmentations": [], "nms": False})
assert cfg.augmentations == [] and cfg.nms is False

versions = {}
for name in ("ultralytics", "torch", "torchvision", "Pillow", "albumentations", "opencv-python"):
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None

sources = {}
for obj in (get_cfg, Albumentations, BaseTrainer):
    path = Path(inspect.getfile(obj))
    sources[path.relative_to(Path(ultralytics.__file__).parent).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()

record = {
    "versions": versions,
    "gpu": torch.cuda.get_device_name(0),
    "config_accepts_augmentations_key": True,
    "bbox_augmentation_verified": False,
    "inference_head_verified": False,
    "correction_to_initial_preflight": "Absence from DEFAULT_CFG_DICT does not imply unsupported: get_cfg explicitly permits augmentations.",
    "inference_choice": "nms=None is default one-to-many NMS; nms=False selects one-to-one head. Freeze before final evaluation.",
    "source_sha256": sources,
    "training_started": False,
}
destination = Path("/content/drive/MyDrive/TomatoAI/yolo26s-field-2026-11-10/api-check.json")
assert destination.parent.is_dir(), "Expected mounted Drive task directory"
destination.write_text(json.dumps(record, indent=2), encoding="utf-8")
print("API_CHECK_OK", json.dumps(record, indent=2))
