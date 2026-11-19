import json
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from torch import nn

from ai.models import build_model

MODULE = "docs.progress.yolo-image-recovery.stage_p08"


def test_configure_p08_stage_freezes_early_stem_and_sets_dropout() -> None:
    module = import_module(MODULE)
    model: Any = build_model("yolo26s_cls", 8, None)
    head, mid, tail = module.stage_parameter_groups(model)
    blocks = model.model.model
    stem_ids = {id(p) for b in blocks[:5] for p in b.parameters()}
    group_ids = [{id(p) for p in group} for group in (head, mid, tail)]
    assert stem_ids.isdisjoint(group_ids[0] | group_ids[1] | group_ids[2])
    assert group_ids[0].isdisjoint(group_ids[1])
    assert group_ids[0].isdisjoint(group_ids[2])
    assert group_ids[1].isdisjoint(group_ids[2])

    module.apply_p08_mode(model, head, mid, tail, full=False)
    assert all(not p.requires_grad for b in blocks[:10] for p in b.parameters())
    assert all(p.requires_grad for p in head)
    assert blocks[10].drop.p == pytest.approx(0.25)
    assert blocks[10].drop.training is True
    assert all(
        not m.training
        for m in model.modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm)
    )

    module.apply_p08_mode(model, head, mid, tail, full=True)
    assert all(not p.requires_grad for b in blocks[:5] for p in b.parameters())
    assert all(p.requires_grad for p in head + mid + tail)
    assert all(
        not m.training
        for m in model.modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm)
    )


def test_pre_pad_augment_preserves_shape_and_rgb_bounds() -> None:
    module = import_module(MODULE)
    spec = json.loads(Path("ai/configs/E10.json").read_text(encoding="utf-8-sig"))[
        "preprocessing"
    ]["spec"]
    augment = module.PrePadAugment(spec)
    for size in ((256, 256), (640, 420)):
        image = Image.new("RGB", size, (60, 150, 70))
        tensor = augment(image)
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32
        assert bool(torch.isfinite(tensor).all())
        assert float(tensor.min()) >= 0.0
        assert float(tensor.max()) <= 1.0


def test_choose_epoch_enforces_strict_p08_inner_floors() -> None:
    choose = import_module(MODULE).choose_epoch
    variants = {
        "2": {
            "plantdoc_f1": 0.4595,
            "pv_f1": 0.9643,
            "pooled_f1": 0.9450,
            "threshold": 0.5,
        },
        "6": {
            "plantdoc_f1": 0.5137,
            "pv_f1": 0.9796,
            "pooled_f1": 0.9580,
            "threshold": 0.5,
        },
        "8": {
            "plantdoc_f1": 0.5217,
            "pv_f1": 0.9795,
            "pooled_f1": 0.9590,
            "threshold": 0.5,
        },
    }
    assert choose(variants) == "8"
    assert (
        choose(
            {
                "8": {
                    "plantdoc_f1": 0.4890,
                    "pv_f1": 0.9820,
                    "pooled_f1": 0.9600,
                    "threshold": 0.5,
                }
            }
        )
        is None
    )
