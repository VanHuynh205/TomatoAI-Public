"""Offline contracts for the original image trainer; no model download or training."""

import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import ai
from ai import dinov2_field as field
from ai import dinov2_improve as improve
from ai import dinov3_field as shared


def test_training_imports_resolve_to_this_public_checkout() -> None:
    root = Path(__file__).resolve().parents[2]
    for module in (ai, field, improve, shared):
        assert Path(module.__file__).resolve().is_relative_to(root / "ai")


def test_classifier_concatenates_cls_and_mean_patch_tokens() -> None:
    tokens = torch.tensor([[[1.0, 2.0], [3.0, 6.0], [5.0, 8.0]]])
    backbone = Mock(config=SimpleNamespace(hidden_size=2))
    backbone.return_value = SimpleNamespace(last_hidden_state=tokens)
    model = field.Classifier(backbone)
    assert model.head[-1].out_features == 9
    model.head = nn.Identity()
    pixels = torch.zeros(1, 3, 224, 224)
    assert torch.equal(model(pixels), torch.tensor([[1.0, 2.0, 4.0, 7.0]]))
    assert backbone.call_args.kwargs["pixel_values"] is pixels


def test_v1_sampler_balances_classes_domains_and_groups() -> None:
    rows = (
        [
            {"class_id": "healthy", "domain": "field", "source": "garden", "group": g}
            for g in ("a", "a", "b")
        ]
        + [
            {"class_id": "healthy", "domain": d, "source": d, "group": d}
            for d in ("web", "controlled")
        ]
        + [
            {"class_id": field.NEGATIVE, "domain": "negative", "source": s, "group": s}
            for s in ("soil", "hands")
        ]
    )
    weights = field.sampling_weights(rows, field.FINETUNE_DOMAINS)
    mass: defaultdict[tuple[str, str], float] = defaultdict(float)
    for row, weight in zip(rows, weights, strict=True):
        mass[row["class_id"], row["domain"]] += weight
    assert mass == pytest.approx(
        {
            ("healthy", "field"): 0.65,
            ("healthy", "web"): 0.15,
            ("healthy", "controlled"): 0.20,
            (field.NEGATIVE, "negative"): 1.0,
        }
    )
    assert weights[0] + weights[1] == pytest.approx(weights[2])
    assert weights[-2:] == pytest.approx([0.5, 0.5])


def test_predict_averages_logits_before_softmax_and_flips_width() -> None:
    original = torch.zeros(1, 9)
    original[0, 0] = 6
    flipped = torch.zeros(1, 9)
    flipped[0, 7] = 12
    model = Mock(side_effect=[original, flipped])
    pixels = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    loader = DataLoader(TensorDataset(pixels, torch.zeros(1, dtype=torch.long)))
    scores = field.predict(model, loader, torch.device("cpu"))
    assert torch.equal(model.call_args_list[1].args[0], pixels.flip(-1))
    assert torch.allclose(scores, ((original + flipped) / 2).softmax(1))
    assert scores.argmax(1).item() == 7
    model.eval.assert_called_once()


def test_predict_rejects_nonfinite_logits() -> None:
    model = Mock(return_value=torch.full((1, 9), float("nan")))
    loader = DataLoader(TensorDataset(torch.zeros(1, 3, 2, 2), torch.zeros(1)))
    with pytest.raises(ValueError, match="Non-finite"):
        field.predict(model, loader, torch.device("cpu"))


@pytest.fixture
def manifest_rows(tmp_path: Path) -> list[dict[str, str]]:
    rows = []
    for split, labels in (
        ("train", shared.CLASSES),
        ("val", ("healthy", field.NEGATIVE)),
        ("test", ("healthy", field.NEGATIVE)),
    ):
        for label in labels:
            name = f"{split}-{label}.png"
            image = Image.new("RGB", (4, 4), (len(rows) * 13, 15, 30))
            image.save(tmp_path / name)
            rows.append(
                {
                    "path": name,
                    "split": split,
                    "class_id": label,
                    "domain": "negative" if label == field.NEGATIVE else "field",
                    "source": "synthetic",
                    "group": name,
                    "sha256": hashlib.sha256(
                        (tmp_path / name).read_bytes()
                    ).hexdigest(),
                }
            )
    return rows


def write_manifests(root: Path, rows: list[dict[str, str]]) -> None:
    for name, splits in (("train-val.csv", {"train", "val"}), ("test.csv", {"test"})):
        with (root / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(row for row in rows if row["split"] in splits)


def test_v1_reads_locked_splits_without_moving_test_rows(
    tmp_path: Path,
    manifest_rows: list[dict[str, str]],
) -> None:
    write_manifests(tmp_path, manifest_rows)
    splits = field.load_splits(tmp_path)
    assert [len(splits[name]) for name in ("train", "val", "test")] == [9, 2, 2]
    assert {row["class_id"] for row in splits["train"]} == set(shared.CLASSES)
    assert all(row["split"] == "test" for row in splits["test"])


@pytest.mark.parametrize("key", ["path", "group", "sha256"])
def test_v1_rejects_cross_split_identity(
    tmp_path: Path,
    manifest_rows: list[dict[str, str]],
    key: str,
) -> None:
    manifest_rows[-2][key] = manifest_rows[0][key]
    write_manifests(tmp_path, manifest_rows)
    with pytest.raises(ValueError, match=f"cross-split {key}"):
        field.load_splits(tmp_path)


def test_v1_rejects_modified_image_bytes(
    tmp_path: Path,
    manifest_rows: list[dict[str, str]],
) -> None:
    write_manifests(tmp_path, manifest_rows)
    (tmp_path / manifest_rows[0]["path"]).write_bytes(b"changed after sealing")
    with pytest.raises(ValueError, match="manifest digest"):
        field.load_splits(tmp_path)
