import hashlib
import json
from importlib import import_module
from pathlib import Path

import pytest
import torch

from ai.artifacts import load_bundle
from ai.models import build_model

MODULE = "docs.progress.yolo-image-recovery.task6_bundle"


def test_v3_gate_selects_p07_refit_and_rejects_e09_e10() -> None:
    module = import_module(MODULE)
    gate = module.V3_GATE
    variants = {
        "E09": {
            "plantdoc_f1": 0.351478,
            "pv_f1": 0.955012,
            "pooled_f1": 0.927739,
            "min_recall": 0.85,
            "threshold": 0.5,
        },
        "E10": {
            "plantdoc_f1": 0.345515,
            "pv_f1": 0.982152,
            "pooled_f1": 0.953841,
            "min_recall": 0.88,
            "threshold": 0.5,
        },
        "p07-refit": {
            "plantdoc_f1": 0.434886,
            "pv_f1": 0.981618,
            "pooled_f1": 0.955751,
            "min_recall": 0.883436,
            "threshold": 0.5,
        },
        "p08-refit": {
            "plantdoc_f1": 0.422915,
            "pv_f1": 0.985739,
            "pooled_f1": 0.958486,
            "min_recall": 0.871166,
            "threshold": 0.5,
        },
    }
    assert module.eligible_candidates(variants, gate) == ["p07-refit", "p08-refit"]
    assert module.select_v3_winner(variants, gate) == "p07-refit"


def test_write_yolo_bundle_creates_verified_trained_artifact(tmp_path: Path) -> None:
    module = import_module(MODULE)
    config = json.loads(Path("ai/configs/E10.json").read_text(encoding="utf-8-sig"))
    model = build_model("yolo26s_cls", len(config["class_ids"]), None).eval()
    weights_path = tmp_path / "candidate.pth"
    torch.save(model.state_dict(), weights_path)
    decision_path = tmp_path / "decision.json"
    decision_path.write_text('{"winner":"p07-refit"}\n', encoding="utf-8")
    bundle_dir = tmp_path / "yolo26s-cls-test"

    out = module.write_yolo_bundle(
        weights_path=weights_path,
        decision_path=decision_path,
        config=config,
        output_dir=bundle_dir,
        model_version="yolo26s-cls-test",
        threshold=0.5,
    )
    assert out == bundle_dir
    loaded = load_bundle(bundle_dir, require_trained=True)
    assert loaded.metadata["model_version"] == "yolo26s-cls-test"
    assert loaded.metadata["architecture"] == "yolo26s_cls"
    assert loaded.metadata["artifact_kind"] == "trained"
    assert (
        loaded.metadata["validation_decision_sha256"]
        == hashlib.sha256(decision_path.read_bytes()).hexdigest()
    )
    for key, tensor in loaded.model.state_dict().items():
        torch.testing.assert_close(tensor, model.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        module.write_yolo_bundle(
            weights_path=weights_path,
            decision_path=decision_path,
            config=config,
            output_dir=bundle_dir,
            model_version="yolo26s-cls-test",
            threshold=0.5,
        )
