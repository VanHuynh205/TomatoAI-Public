"""Check PV-trained backbone compatibility in Colab without optimization or test images."""

import hashlib
import inspect
import io
import json
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import albumentations as A
import numpy as np
import torch
from ultralytics.data.augment import Albumentations
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel, _SafeLoad
from ultralytics.utils.instance import Instances
from ultralytics.utils.torch_utils import ModelEMA

WORK = Path("/content/yolo26s-field")
DRIVE = Path("/content/drive/MyDrive/TomatoAI/yolo26s-field-2026-11-10")
BASELINE = WORK / "baseline/ai/artifacts/yolo26s-cls-v1/model.pth"
PRETRAINED = WORK / "pretrained/yolo26s.pt"
if not __debug__ or version("ultralytics") != "8.4.204":
    raise RuntimeError("Structural checks require assertions enabled and ultralytics==8.4.204")
baseline_bytes, pretrained_bytes = BASELINE.read_bytes(), PRETRAINED.read_bytes()
if hashlib.sha256(baseline_bytes).hexdigest() != "322ec3442a5fa54e913da155e8964205ea964117a63fdeec0780820c30f6e56f":
    raise ValueError("Baseline checksum mismatch")
if hashlib.sha256(pretrained_bytes).hexdigest() != "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b":
    raise ValueError("Official detector checksum mismatch")
torch.set_num_threads(2)
torch.manual_seed(20261110)
baseline_state = torch.load(io.BytesIO(baseline_bytes), map_location="cpu", weights_only=True)
prefixes = tuple(f"model.{index}." for index in range(9))
selected = {key.removeprefix("model."): tensor for key, tensor in baseline_state.items() if key.removeprefix("model.").startswith(prefixes)}
assert len(selected) == 186
if not _SafeLoad.SUPPORTED:
    raise RuntimeError("Restricted checkpoint loading is required")
_SafeLoad.register(io.BytesIO(pretrained_bytes))
checkpoint = torch.load(io.BytesIO(pretrained_bytes), map_location="cpu", weights_only=True)
official = (checkpoint.get("ema") or checkpoint["model"]).float().eval()
if not isinstance(official, DetectionModel) or official.yaml["scale"] != "s":
    raise ValueError("Unexpected detector architecture")


def transfer_backbone(model):
    target = model.state_dict()
    expected = {key for key in target if key.startswith(prefixes)}
    if set(selected) != expected:
        raise ValueError("classifier and detector backbone keys differ")
    for key, tensor in selected.items():
        if tensor.shape != target[key].shape or tensor.dtype != target[key].dtype or not torch.isfinite(tensor).all():
            raise ValueError(f"incompatible backbone tensor: {key}")
    model.load_state_dict({**target, **selected}, strict=True)
    return model


class PVBackboneTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        return model if self.resume else transfer_backbone(model)


# Exercise get_model after detector reconstruction, at the seam used by training.
trainer = object.__new__(PVBackboneTrainer)
trainer.data = {"nc": 8, "channels": 3}
trainer.args = SimpleNamespace(nms=False)
trainer.resume = False
torch.manual_seed(20261110)
control = DetectionTrainer.get_model(trainer, cfg=deepcopy(official.yaml), weights=official, verbose=False)
torch.manual_seed(20261110)
transferred = trainer.get_model(cfg=deepcopy(official.yaml), weights=official, verbose=False)
control_state = control.state_dict()
transferred_state = transferred.state_dict()
assert all(torch.equal(transferred_state[key], value) for key, value in selected.items())
unchanged = [key for key in transferred_state if key not in selected]
assert all(torch.equal(transferred_state[key], control_state[key]) for key in unchanged)
changed = [key for key in selected if not torch.equal(control_state[key], transferred_state[key])]
assert changed
ema = ModelEMA(transferred)
assert all(torch.equal(ema.ema.state_dict()[key], value) for key, value in selected.items())
transferred.eval()
with torch.inference_mode():
    result = transferred(torch.zeros(1, 3, 320, 320))
    prediction = result[0] if isinstance(result, tuple) else result
assert isinstance(prediction, torch.Tensor) and torch.isfinite(prediction).all()
with torch.no_grad():
    next(transferred.parameters()).add_(0.01)
trainer.resume = True
resumed = trainer.get_model(cfg=deepcopy(transferred.yaml), weights=transferred, verbose=False)
assert all(torch.equal(resumed.state_dict()[key], value) for key, value in transferred.state_dict().items())


class ShapeCheckedPredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_imgs, **kwargs):
        raw = preds[0] if isinstance(preds, (list, tuple)) else preds
        self.observed_shape = list(raw.shape)
        assert torch.isfinite(raw).all()
        if self.args.nms is False:
            assert self.model.end2end and raw.ndim == 3 and raw.shape[-1] == 6
        else:
            assert not self.model.end2end and raw.ndim == 3 and raw.shape[1] == 84
        return super().postprocess(preds, img, orig_imgs, **kwargs)


head_checks = []
for use_nms in (True, False):
    predictor = ShapeCheckedPredictor(overrides={
        "nms": use_nms, "imgsz": 320, "device": "cpu", "verbose": False,
        "conf": 0.001, "max_det": 10, "save": False, "project": str(WORK / "checks"),
    })
    predictor.setup_model(model=deepcopy(official), verbose=False)
    outputs = predictor(source=np.zeros((320, 320, 3), dtype=np.uint8))
    assert len(outputs) == 1 and outputs[0].boxes.data.shape[1] == 6
    head_checks.append({"nms": use_nms, "end2end": predictor.model.end2end, "raw_shape": predictor.observed_shape})

# Fixed geometry and color changes give independent expected outputs for the installed wrapper.
box_xywh = np.array([[0.25, 0.4, 0.2, 0.3], [0.8, 0.6, 0.1, 0.2]], dtype=np.float32)
class_ids = np.array([[1], [3]], dtype=np.float32)
synthetic_image = np.full((100, 200, 3), 80, dtype=np.uint8)
synthetic_image[:, :60] = (25, 60, 110)
for spatial, transforms in (
    (True, [A.HorizontalFlip(p=1)]),
    (False, [A.RandomBrightnessContrast(brightness_limit=(0.2, 0.2), contrast_limit=(0, 0), p=1)]),
):
    augmentation = Albumentations(p=1, transforms=transforms)
    assert augmentation.transform is not None and augmentation.contains_spatial is spatial
    labels = {
        "img": synthetic_image.copy(), "cls": class_ids.copy(),
        "instances": Instances(box_xywh.copy(), segments=np.zeros((0, 1000, 2), dtype=np.float32), bbox_format="xywh", normalized=True),
    }
    augmented = augmentation(labels)
    expected_boxes = box_xywh.copy()
    if spatial:
        expected_boxes[:, 0] = 1 - expected_boxes[:, 0]
        np.testing.assert_array_equal(augmented["img"], synthetic_image[:, ::-1])
    else:
        assert not np.array_equal(augmented["img"], synthetic_image)
    np.testing.assert_allclose(augmented["instances"].bboxes, expected_boxes, atol=1e-6)
    np.testing.assert_array_equal(augmented["cls"], class_ids)

record = {
    "status": "structural_transfer_verified_not_trained_or_accuracy_evaluated",
    "ultralytics": version("ultralytics"), "torch": version("torch"),
    "baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
    "detector_sha256": hashlib.sha256(pretrained_bytes).hexdigest(),
    "checkpoint_load": "same_verified_bytes_weights_only_true",
    "backbone_layers": list(range(9)), "transferred_state_entries": len(selected),
    "changed_from_control_entries": len(changed), "unchanged_other_entries": len(unchanged),
    "ema_equal": True, "resume_get_model_preserves_in_memory_weights": True,
    "full_checkpoint_resume_verified": False, "synthetic_prediction_shape": list(prediction.shape),
    "predictor_head_checks": head_checks, "validator_head_verified": False,
    "synthetic_bbox_flip_and_photometric_checks": True, "training_augmentation_overlays_reviewed": False,
    "test_nc": 8, "taxonomy_locked": False, "transfer_selected_for_training": False,
    "get_model_source": inspect.getsource(DetectionTrainer.get_model),
    "setup_model_source": inspect.getsource(DetectionTrainer.setup_model),
    "selected_keys": sorted(selected),
}
(DRIVE / "backbone-transfer-check.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print("BACKBONE_TRANSFER_CHECK_OK", json.dumps({key: value for key, value in record.items() if not key.endswith("source") and key != "selected_keys"}, indent=2))
