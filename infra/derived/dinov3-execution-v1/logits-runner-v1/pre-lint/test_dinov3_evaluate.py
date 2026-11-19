"""Check image/logit provenance with tiny CPU fixtures, without DINO weights."""

import csv
import hashlib
import importlib
import importlib.util
import io
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ai import dinov3_inference
from ai.dinov3_field import CLASSES, leaf_transform


def evaluator() -> ModuleType:
    assert importlib.util.find_spec("ai.dinov3_evaluate") is not None, (
        "The receipt-bound DINOv3 evaluation runner is missing"
    )
    return importlib.import_module("ai.dinov3_evaluate")


@pytest.fixture
def images(tmp_path: Path) -> list[dict[str, str]]:
    rows = []
    for index, (color, label) in enumerate((("black", "healthy"), ("red", "bacterial_spot"))):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (16, 16), color).save(path)
        rows.append({
            "path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "class_id": label, "source": f"source-{index}", "group": f"plant-{index}", "split": "val",
        })
    return rows


class PixelLogits(torch.nn.Module):
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return torch.cat((pixels[:, :1, 0, 0], pixels.new_zeros((len(pixels), 8))), dim=1)


def image_loader(
    root: Path, rows: list[dict[str, str]], order: list[int],
) -> DataLoader[Any]:
    module = evaluator()
    data = module.EvaluationImages(root, rows, leaf_transform(224, [0.0] * 3, [1.0] * 3, False))
    return DataLoader(data, batch_size=1, sampler=order, num_workers=0)


def test_logits_follow_actual_loaded_images_when_sampler_reorders(
    tmp_path: Path, images: list[dict[str, str]],
) -> None:
    module = evaluator()
    records = module.collect_logits(PixelLogits(), image_loader(tmp_path, images, [1, 0]), torch.device("cpu"))

    assert [row["path"] for row in records] == ["image-1.png", "image-0.png"]
    assert [row["manifest_index"] for row in records] == [1, 0]
    assert [row["source"] for row in records] == ["source-1", "source-0"]
    assert [row["group"] for row in records] == ["plant-1", "plant-0"]
    assert [row["target_index"] for row in records] == [0, 7]
    assert records[0]["image_id"] == images[1]["sha256"]
    assert records[0]["sha256"] == images[1]["sha256"]
    assert records[0]["logits"] == [1.0] + [0.0] * 8
    assert records[0]["probabilities"][0] == pytest.approx(math.e / (math.e + 8))
    assert records[1]["logits"] == [0.0] * 9
    assert records[1]["probabilities"] == pytest.approx([1 / 9] * 9)


@pytest.mark.parametrize("order", [[0], [0, 0], []])
def test_incomplete_or_repeated_image_consumption_is_rejected(
    tmp_path: Path, images: list[dict[str, str]], order: list[int],
) -> None:
    module = evaluator()
    with pytest.raises(ValueError, match="identity|complete|empty"):
        module.collect_logits(PixelLogits(), image_loader(tmp_path, images, order), torch.device("cpu"))


def test_dataset_hashes_the_same_bytes_it_decodes(
    tmp_path: Path, images: list[dict[str, str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = image_loader(tmp_path, images, [0, 1])
    original_open = Image.open

    def replace_path_after_read(source: Any, *args: Any, **kwargs: Any) -> Any:
        Image.new("RGB", (16, 16), "white").save(tmp_path / "image-0.png")
        return original_open(source, *args, **kwargs)

    monkeypatch.setattr(Image, "open", replace_path_after_read)
    records = evaluator().collect_logits(PixelLogits(), loader, torch.device("cpu"))
    assert records[0]["logits"] == [0.0] * 9
    assert records[0]["sha256"] == images[0]["sha256"]


def test_replaced_image_is_rejected_before_model_receives_it(
    tmp_path: Path, images: list[dict[str, str]],
) -> None:
    loader = image_loader(tmp_path, images, [0, 1])
    Image.new("RGB", (16, 16), "white").save(tmp_path / "image-0.png")
    with pytest.raises(ValueError, match="digest"):
        evaluator().collect_logits(PixelLogits(), loader, torch.device("cpu"))


def test_target_index_must_match_consumed_identity(
    tmp_path: Path, images: list[dict[str, str]],
) -> None:
    loader = image_loader(tmp_path, images, [0, 1])

    def wrong_target(batch: list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pixels, target, index = batch[0]
        return pixels.unsqueeze(0), torch.tensor([(target + 1) % len(CLASSES)]), torch.tensor([index])

    loader.collate_fn = wrong_target
    with pytest.raises(ValueError, match="target"):
        evaluator().collect_logits(PixelLogits(), loader, torch.device("cpu"))


def test_duplicate_content_cannot_supply_two_image_identities(
    tmp_path: Path, images: list[dict[str, str]],
) -> None:
    (tmp_path / "duplicate.png").write_bytes((tmp_path / images[0]["path"]).read_bytes())
    images[1].update(path="duplicate.png", sha256=images[0]["sha256"])
    with pytest.raises(ValueError, match="identity"):
        image_loader(tmp_path, images, [0, 1])


@pytest.fixture
def frozen_inputs(
    tmp_path: Path, images: list[dict[str, str]], monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    export = tmp_path / "model"
    export.mkdir()
    for name, value in {
        "leaf-classifier.json": {"size": 224, "class_ids": CLASSES, "model_version": "dinov3-fixture"},
        "config.json": dinov3_inference._SMALL_VIT_CONFIG,
        "preprocessor_config.json": {"image_mean": [0.0] * 3, "image_std": [1.0] * 3},
    }.items():
        (export / name).write_text(json.dumps(value), encoding="utf-8")
    for name in ("model.safetensors", "classifier.safetensors"):
        (export / name).write_bytes(b"fixture, never load as real weights")
    manifest = tmp_path / "evaluation.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(images[0]))
        writer.writeheader()
        writer.writerows(images)
    project = Path(__file__).resolve().parents[2]
    receipt = tmp_path / "inputs.json"
    receipt.write_text(json.dumps({
        "schema_version": 1, "status": "frozen_inputs", "artifact_kind": "fixture",
        "class_ids": CLASSES,
        "manifest": {"path": "evaluation.csv", "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "rows": 2},
        "model_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in export.iterdir()},
        "code_sha256": {
            name: hashlib.sha256((project / name).read_bytes()).hexdigest()
            for name in ("ai/dinov3_evaluate.py", "ai/dinov3_inference.py", "ai/dinov3_field.py", "ai/artifacts.py", "ai/contracts.py")
        },
    }), encoding="utf-8")
    monkeypatch.setattr(dinov3_inference, "_load_model", lambda _: PixelLogits())
    return export, receipt


def run_evaluation(tmp_path: Path, inputs: tuple[Path, Path]) -> Path:
    module = evaluator()
    run = getattr(module, "evaluate", None)
    assert callable(run), "The frozen-input evaluation entry point is missing"
    output = tmp_path / "logits.json"
    run(tmp_path, inputs[0], inputs[1], output, device=torch.device("cpu"), batch_size=2, workers=0)
    return output


def test_runner_writes_complete_logits_with_frozen_receipt(
    tmp_path: Path, frozen_inputs: tuple[Path, Path],
) -> None:
    output = run_evaluation(tmp_path, frozen_inputs)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["class_ids"] == list(CLASSES)
    assert result["artifact_kind"] == "fixture"
    assert result["data_qualification"] == "not_assessed"
    assert result["input_receipt_sha256"] == hashlib.sha256(frozen_inputs[1].read_bytes()).hexdigest()
    assert result["input_receipt"] == json.loads(frozen_inputs[1].read_text())
    assert result["model_version"] == "dinov3-fixture"
    assert result["runtime"]["device"] == "cpu"
    assert result["runtime"]["autocast"] == "disabled"
    assert result["runtime"]["versions"]["torch"]
    assert [r["path"] for r in result["predictions"]] == ["image-0.png", "image-1.png"]
    assert [r["target_index"] for r in result["predictions"]] == [7, 0]
    assert result["predictions"][1]["logits"] == [1.0] + [0.0] * 8
    assert not list(tmp_path.glob("*.pending"))


@pytest.mark.parametrize("damage", ["manifest", "model", "preprocessing", "code", "class_order", "draft", "missing_pin", "rows"])
def test_runner_rejects_unfrozen_or_mismatched_inputs(
    tmp_path: Path, frozen_inputs: tuple[Path, Path], damage: str,
) -> None:
    export, receipt = frozen_inputs
    contract = json.loads(receipt.read_text())
    if damage == "manifest":
        (tmp_path / "evaluation.csv").write_text("changed", encoding="utf-8")
    elif damage in {"model", "preprocessing"}:
        (export / ("model.safetensors" if damage == "model" else "preprocessor_config.json")).write_bytes(b"changed")
    elif damage == "code":
        contract["code_sha256"]["ai/dinov3_field.py"] = "0" * 64
    elif damage == "class_order":
        contract["class_ids"].reverse()
    elif damage == "draft":
        contract["status"] = "draft"
    elif damage == "missing_pin":
        del contract["model_sha256"]["classifier.safetensors"]
    else:
        contract["manifest"]["rows"] = 3
    receipt.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt|digest|class order|manifest"):
        run_evaluation(tmp_path, frozen_inputs)
    assert not (tmp_path / "logits.json").exists()


@pytest.mark.parametrize("damage", ["weights", "receipt", "manifest", "nan", "shape"])
def test_failed_run_never_publishes_partial_logits(
    tmp_path: Path, frozen_inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, damage: str,
) -> None:
    class FailingModel(PixelLogits):
        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            if damage == "weights":
                (frozen_inputs[0] / "classifier.safetensors").write_bytes(b"changed during evaluation")
            elif damage == "receipt":
                frozen_inputs[1].write_text("{}", encoding="utf-8")
            elif damage == "manifest":
                (tmp_path / "evaluation.csv").write_text("changed", encoding="utf-8")
            logits = super().forward(pixels)
            if damage == "nan":
                logits[0, 0] = float("nan")
            return logits[:, :8] if damage == "shape" else logits

    monkeypatch.setattr(dinov3_inference, "_load_model", lambda _: FailingModel())
    with pytest.raises(ValueError):
        run_evaluation(tmp_path, frozen_inputs)
    assert not (tmp_path / "logits.json").exists()
    assert not list(tmp_path.glob("*.pending"))


def test_runner_preserves_an_existing_result(
    tmp_path: Path, frozen_inputs: tuple[Path, Path],
) -> None:
    output = tmp_path / "logits.json"
    output.write_bytes(b"previous sealed result")
    with pytest.raises(FileExistsError):
        run_evaluation(tmp_path, frozen_inputs)
    assert output.read_bytes() == b"previous sealed result"


def repin_manifest(tmp_path: Path, receipt: Path) -> None:
    contract = json.loads(receipt.read_text())
    contract["manifest"]["sha256"] = hashlib.sha256((tmp_path / "evaluation.csv").read_bytes()).hexdigest()
    receipt.write_text(json.dumps(contract), encoding="utf-8")


def test_manifest_parent_components_are_rejected_before_inference(
    tmp_path: Path, frozen_inputs: tuple[Path, Path],
) -> None:
    (tmp_path / "nested").mkdir()
    contract = json.loads(frozen_inputs[1].read_text())
    contract["manifest"]["path"] = "nested/../evaluation.csv"
    frozen_inputs[1].write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="path|manifest"):
        run_evaluation(tmp_path, frozen_inputs)


def test_manifest_parser_consumes_the_digest_checked_buffer(
    tmp_path: Path, frozen_inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "evaluation.csv"
    replacement = manifest.read_text().replace("healthy", "early_blight")
    original_open = Path.open

    def changed_text_read(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == manifest and kwargs.get("encoding") == "utf-8":
            return io.StringIO(replacement)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", changed_text_read)
    output = run_evaluation(tmp_path, frozen_inputs)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert [r["target_index"] for r in result["predictions"]] == [7, 0]


@pytest.mark.parametrize("name", ["adapter_config.json", "config.4.0.0.json"])
def test_export_rejects_additional_loader_inputs(
    tmp_path: Path, frozen_inputs: tuple[Path, Path], name: str,
) -> None:
    (frozen_inputs[0] / name).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="export|file|receipt"):
        run_evaluation(tmp_path, frozen_inputs)


def test_export_rejects_a_pinned_configuration_redirect(
    tmp_path: Path, frozen_inputs: tuple[Path, Path],
) -> None:
    config = frozen_inputs[0] / "config.json"
    value = json.loads(config.read_text())
    value["configuration_files"] = ["../config.4.0.0.json"]
    config.write_text(json.dumps(value), encoding="utf-8")
    contract = json.loads(frozen_inputs[1].read_text())
    contract["model_sha256"]["config.json"] = hashlib.sha256(config.read_bytes()).hexdigest()
    frozen_inputs[1].write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="config|export"):
        run_evaluation(tmp_path, frozen_inputs)


def test_custom_collation_cannot_swap_identities_of_same_class_images(
    tmp_path: Path, images: list[dict[str, str]],
) -> None:
    images[1]["class_id"] = "healthy"
    loader = image_loader(tmp_path, images, [0, 1])

    def swap_identity(batch: list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pixels, target, index = batch[0]
        return pixels.unsqueeze(0), torch.tensor([target]), torch.tensor([1 - index])

    loader.collate_fn = swap_identity
    with pytest.raises(ValueError, match="collat|identity"):
        evaluator().collect_logits(PixelLogits(), loader, torch.device("cpu"))


def test_collector_metadata_cannot_change_after_image_verification(
    tmp_path: Path, images: list[dict[str, str]],
) -> None:
    loader = image_loader(tmp_path, images, [0, 1])

    class MutatingModel(PixelLogits):
        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            metadata = getattr(loader.dataset, "rows")
            metadata[0]["sha256"] = "0" * 64
            metadata[0]["source"] = "changed-source"
            return super().forward(pixels)

    records = evaluator().collect_logits(MutatingModel(), loader, torch.device("cpu"))
    assert records[0]["image_id"] == images[0]["sha256"]
    assert records[0]["source"] == "source-0"


@pytest.mark.parametrize("damage", ["duplicate_header", "extra_column", "missing_column"])
def test_ambiguous_manifest_columns_are_rejected(
    tmp_path: Path, frozen_inputs: tuple[Path, Path], images: list[dict[str, str]], damage: str,
) -> None:
    manifest = tmp_path / "evaluation.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        header = list(images[0])
        writer.writerow(header + (["class_id"] if damage == "duplicate_header" else ["domain"] if damage == "missing_column" else []))
        for row in images:
            writer.writerow([row[key] for key in header] + (["early_blight"] if damage == "duplicate_header" else ["unexpected"] if damage == "extra_column" else []))
    repin_manifest(tmp_path, frozen_inputs[1])
    with pytest.raises(ValueError, match="header|column|manifest"):
        run_evaluation(tmp_path, frozen_inputs)


def test_duplicate_receipt_keys_are_rejected(
    tmp_path: Path, frozen_inputs: tuple[Path, Path],
) -> None:
    receipt = frozen_inputs[1]
    content = receipt.read_text()
    receipt.write_text('{"status":"draft",' + content[1:], encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate|receipt"):
        run_evaluation(tmp_path, frozen_inputs)
