"""Sampler, split, selection and checkpoint checks without training weights."""

from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

import torch

from ai import dinov2_improve as improve
from ai.dinov2_field import NEGATIVE
from ai.dinov3_field import CLASSES


def row(label: str, source: str, domain: str, group: str) -> dict[str, str]:
    return {"class_id": label, "source": source, "domain": domain, "group": group}


def test_sampler_balances_available_domains_sources_and_capture_groups() -> None:
    positive = "healthy"
    negative = NEGATIVE
    rows = [
        row(positive, "rachin", "field", "a"),
        row(positive, "rachin", "field", "a"),
        row(positive, "rachin", "field", "b"),
        row(positive, "plantcity", "field", "c"),
        row(positive, "plantdoc", "web", "d"),
        row(positive, "plantvillage", "controlled", "e"),
        row("spider_mites", "plantvillage", "controlled", "f"),
        row(negative, "soilclass", "negative", "g"),
        row(negative, "soilclass", "negative", "h"),
        row(negative, "openimages_hands", "negative", "i"),
        row(negative, "plantdoc_other_leaves", "negative", "j"),
    ]
    weights = improve.sampling_weights(rows)
    classes: defaultdict[str, float] = defaultdict(float)
    sources: defaultdict[tuple[str, str], float] = defaultdict(float)
    for entry, weight in zip(rows, weights, strict=True):
        label, source = entry["class_id"], entry["source"]
        classes[label] += weight
        sources[label, source] += weight
    assert classes == pytest.approx({positive: 1.0, "spider_mites": 1.0, negative: 1.0})
    assert sources[positive, "rachin"] == pytest.approx(0.45)
    assert sources[positive, "plantcity"] == pytest.approx(0.30)
    assert sources[positive, "plantdoc"] == pytest.approx(0.10)
    assert sources[positive, "plantvillage"] == pytest.approx(0.15)
    assert weights[0] + weights[1] == pytest.approx(weights[2])
    for source in ("soilclass", "openimages_hands", "plantdoc_other_leaves"):
        assert sources[negative, source] == pytest.approx(1 / 3)


def test_development_rejects_test_and_cross_split_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("data")
    manifest = Path("development.csv")
    first = {"path": "a.jpg", "split": "train", "group": "capture", "sha256": "a"}
    second = {"path": "b.jpg", "split": "val", "group": "capture", "sha256": "b"}
    monkeypatch.setattr(improve, "load_manifest", lambda *_: [first, second])
    with pytest.raises(ValueError, match="cross-split group"):
        improve.read_rows(root, manifest, {"train", "val"})
    second["split"] = "test"
    with pytest.raises(ValueError, match="Expected only splits"):
        improve.read_rows(root, manifest, {"train", "val"})


def test_selection_and_checkpoint_keep_nine_class_predictions() -> None:
    rows = [row("healthy", source, "field", source) for source in improve.FIELD_SOURCES]
    rows += [
        row("healthy", "imtiaz", "field", "i"),
        row(NEGATIVE, "soilclass", "negative", "n"),
    ]
    scores = torch.eye(len(CLASSES))[[7, 7, 8, 8, 8]]
    report = improve.selection_score(rows, scores)
    assert report["field_mean"] == pytest.approx(2 / 3)
    assert report["field_min"] == 0.0
    assert report["negative_recall"] == 1.0
    assert report["selection"] == pytest.approx(0.60)
    assert report["field_by_source"]["imtiaz"]["macro_f1"] == 0.0

    def round_trip(state: dict[str, Any]) -> dict[str, Any]:
        checkpoint = BytesIO()
        torch.save(state, checkpoint)
        checkpoint.seek(0)
        result: dict[str, Any] = torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )
        return result

    state = round_trip(
        {"baseline": report, "best_epoch": 1, "best_slot": 0, "scores": scores}
    )
    assert state["baseline"] == report
    assert (state["best_epoch"], state["best_slot"]) == (1, 0)
    assert torch.equal(state["scores"], scores)
