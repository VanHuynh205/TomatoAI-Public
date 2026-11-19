import csv
import hashlib
import io
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms as T

from ai import dinov3_field as field
from ai.dinov3_field import (
    _best_checkpoint,
    _save_epoch_checkpoint,
    continuation_weights,
    save_atomic,
    validation_report,
)


@pytest.mark.parametrize("legacy", [False, True])
def test_interrupted_epoch_save_preserves_committed_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy: bool,
) -> None:
    last = tmp_path / "last.pt"
    committed = {
        "epoch": 0 if legacy else -1, "best": 0.4, "history": [{"epoch": 1 if legacy else 0}],
        "model": {"weight": torch.tensor([10])},
    }
    if legacy:
        selected = tmp_path / "best.pt"
        save_atomic({"model": committed["model"]}, selected)
        save_atomic(committed, last)
    else:
        committed["best_epoch"] = 0
        selected = _save_epoch_checkpoint(tmp_path, committed, tmp_path / "best.pt")

    next_epoch = 2 if legacy else 1
    candidate = {
        "epoch": next_epoch - 1, "best": 0.8, "best_epoch": next_epoch,
        "history": [{"epoch": next_epoch - 1}, {"epoch": next_epoch}],
        "model": {"weight": torch.tensor([20])},
    }
    replace = os.replace

    def interrupt_last_commit(source: Path, destination: Path) -> None:
        if destination == last:
            raise OSError("interrupted before last.pt replacement")
        replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupt_last_commit)
        with pytest.raises(OSError, match="interrupted"):
            _save_epoch_checkpoint(tmp_path, candidate, selected)

    assert (tmp_path / f"best-epoch-{next_epoch}.pt").is_file()
    resumed = torch.load(last, weights_only=True)
    assert resumed["epoch"] + 1 == candidate["epoch"]
    assert resumed["history"] == committed["history"]
    assert resumed["best"] == 0.4
    resumed_best = _best_checkpoint(tmp_path, resumed.get("best_epoch"))
    assert resumed_best == selected
    assert torch.load(resumed_best, weights_only=True)["model"]["weight"].item() == 10

    new_best = _save_epoch_checkpoint(tmp_path, candidate, resumed_best)
    resumed = torch.load(last, weights_only=True)
    assert resumed["best"] == 0.8
    assert resumed["history"] == candidate["history"]
    assert _best_checkpoint(tmp_path, resumed["best_epoch"]) == new_best
    assert torch.load(new_best, weights_only=True)["model"]["weight"].item() == 20
    assert selected.exists() == legacy


def test_continuation_sampler_balances_classes_and_softens_capture_group_mass() -> None:
    rows = [
        {"class_id": "bacterial_spot", "domain": domain, "group": group}
        for domain, group, count in (
            ("field", "small", 1), ("field", "large", 4),
            ("controlled", "paper", 4), ("capture_unverified", "unverified", 4), ("web", "web", 1),
        )
        for _ in range(count)
    ]
    rows.extend([
        {"class_id": "healthy", "domain": "field", "group": "healthy"},
        {"class_id": "not_tomato_leaf", "domain": "soil_capture", "group": "soil"},
    ])
    classes: defaultdict[str, float] = defaultdict(float)
    groups: defaultdict[str, float] = defaultdict(float)
    for row, weight in zip(rows, continuation_weights(rows, samples=3), strict=True):
        classes[row["class_id"]] += weight
        groups[row["group"]] += weight

    assert classes == pytest.approx({"bacterial_spot": 1, "healthy": 1, "not_tomato_leaf": 1})
    assert groups == pytest.approx({
        "small": 0.25, "large": 0.50, "paper": 0.15, "unverified": 0.10 * 2 / 3, "web": 0.10 / 3,
        "healthy": 1, "soil": 1,
    })


@pytest.mark.parametrize("independent_source", [False, True])
def test_continuation_field_sources_share_mass_without_repeating_tiny_sources(independent_source: bool) -> None:
    batches = [("field", "farm_a", 50), ("field", "farm_b", 2),
               ("controlled", "farm_a", 5), ("web", "web", 5)]
    if independent_source:
        batches.append(("field", "farm_c", 50))
    rows = [
        {"class_id": "bacterial_spot", "domain": domain, "source": source, "group": f"{source}-{i}"}
        for domain, source, count in batches
        for i in range(count)
    ]
    totals: defaultdict[tuple[str, str], float] = defaultdict(float)
    for row, weight in zip(rows, continuation_weights(rows, samples=100), strict=True):
        totals[row["domain"], row["source"]] += weight
        if row["domain"] == "field":
            assert weight <= 0.01

    assert totals["field", "farm_a"] == pytest.approx(0.25 if independent_source else 0.375)
    assert totals["field", "farm_b"] == pytest.approx(0.02)
    assert totals["field", "farm_c"] == pytest.approx(0.25 if independent_source else 0)
    assert totals["controlled", "farm_a"] / totals["web", "web"] == pytest.approx(1.5)
    assert sum(totals.values()) == pytest.approx(1)


def test_validation_diagnostics_weight_groups_equally_and_use_runtime_acceptance() -> None:
    rows = [
        {"class_id": label, "domain": "field", "group": group, "source": source}
        for label, group, source in (
            ("bacterial_spot", "a", "farm1"), ("bacterial_spot", "b", "farm2"),
            ("bacterial_spot", "b", "farm2"), ("bacterial_spot", "b", "farm2"),
            ("healthy", "healthy", "farm1"),
        )
    ]
    labels = ["bacterial_spot", "septoria_leaf_spot", "septoria_leaf_spot", "not_tomato_leaf", "bacterial_spot"]
    scores = [0.5, 0.49, 0.49, 0.9, 0.55]

    report = validation_report(rows, labels, scores, diagnostics=True)

    assert report["field"]["per_class"]["bacterial_spot"]["recall"] == 0.25
    diagnostics = report["field_diagnostics"]
    assert diagnostics["bacterial_spot_groups"] == 2
    assert diagnostics["bacterial_spot_group_recall"] == 0.5
    assert diagnostics["accepted"] == 2
    assert diagnostics["accepted_correct"] == 1
    assert diagnostics["healthy_accepted_as_disease"] == 1
    assert diagnostics["field_by_source"]["farm2"]["n"] == 3


@pytest.mark.parametrize("version, expected", [
    ("v1", [0.375, 0.375, 0.25]),
    ("v2", [0.01, 0.01, 0.98]),
])
def test_explicit_sampler_has_no_initialization_dependency(version: str, expected: list[float]) -> None:
    rows = [
        {"class_id": "bacterial_spot", "domain": domain, "group": group, "source": "farm"}
        for domain, group in (("field", "a"), ("field", "b"), ("controlled", "c"))
    ]
    assert field.experiment_weights(rows, samples=100, sampler=version) == pytest.approx(expected)


@pytest.mark.parametrize("sampler, samples", [("auto", 100), ("v1", 0), ("v2", -1)])
def test_experiment_sampler_rejects_unregistered_or_empty_budget(sampler: str, samples: int) -> None:
    with pytest.raises(ValueError):
        field.experiment_weights([], samples=samples, sampler=sampler)


def test_source_selection_weights_sources_equally_and_counts_rejection_as_error() -> None:
    rows = [
        {"class_id": "healthy", "source": "large", "group": str(i)} for i in range(8)
    ] + [
        {"class_id": "bacterial_spot", "source": "small", "group": "b"},
        {"class_id": "healthy", "source": "small", "group": "h"},
        {"class_id": "not_tomato_leaf", "source": "negative", "group": "n"},
    ]
    labels = ["healthy"] * 8 + ["not_tomato_leaf", "healthy", "bacterial_spot"]
    report = field.source_selection_report(rows, labels)
    assert report["macro_f1"] == pytest.approx(0.75)
    assert report["sources"] == {"large": ["healthy"], "small": ["bacterial_spot", "healthy"]}
    assert report["by_source"]["small"]["per_class"]["bacterial_spot"] == {
        "support": 1, "recall": 0.0, "precision": None, "f1": 0.0,
    }
    assert report["by_source"]["small"]["per_class"]["early_blight"]["recall"] is None
    assert report["groups_by_source"] == {"large": 8, "small": 2}


@pytest.mark.parametrize("change", ["manifest", "source", "classes", "empty_source"])
def test_selection_contract_refuses_changed_support_or_manifest(tmp_path: Path, change: str) -> None:
    rows = [{"class_id": "healthy", "source": "farm", "group": "plant", "split": "val"}]
    contract = {"manifest_sha256": "a" * 64, "sources": {"farm": ["healthy"]}}
    contract_path = tmp_path / "selection.json"
    if change == "manifest":
        contract["manifest_sha256"] = "b" * 64
    elif change == "source":
        contract["sources"] = {"different": ["healthy"]}
    elif change == "classes":
        contract["sources"] = {"farm": ["healthy", "early_blight"]}
    else:
        rows[0]["source"] = ""
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError):
        field.load_selection_contract(contract_path, rows, "a" * 64)


def test_selection_contract_keeps_training_and_negatives_out_of_support(tmp_path: Path) -> None:
    rows = [
        {"class_id": "healthy", "source": "farm", "group": "plant", "split": "val"},
        {"class_id": "early_blight", "source": "other", "group": "train", "split": "train"},
        {"class_id": "not_tomato_leaf", "source": "negative", "group": "soil", "split": "val"},
    ]
    contract = {"manifest_sha256": "a" * 64, "sources": {"farm": ["healthy"]}}
    contract_path = tmp_path / "selection.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    assert field.load_selection_contract(contract_path, rows, "a" * 64) == contract


def test_epoch_draws_replay_and_keep_view_seeds_independent_of_sample_identity() -> None:
    first = field.epoch_draws([1.0, 0.0], samples=4, seed=1729, epoch=0)
    second = field.epoch_draws([0.0, 1.0], samples=4, seed=1729, epoch=0)
    torch.manual_seed(55)
    torch.rand(200)
    assert field.epoch_draws([1.0, 0.0], samples=4, seed=1729, epoch=0) == first
    assert [index for index, _ in first] == [0, 0, 0, 0]
    assert [index for index, _ in second] == [1, 1, 1, 1]
    assert [seed for _, seed in first] == [seed for _, seed in second]
    assert len({seed for _, seed in first}) == 4
    assert first != field.epoch_draws([1.0, 0.0], samples=4, seed=1729, epoch=1)


def test_seeded_views_replay_without_consuming_model_randomness(tmp_path: Path) -> None:
    Image.new("RGB", (12, 12), (110, 180, 50)).save(tmp_path / "leaf.png")
    rows = [{"path": "leaf.png", "class_id": "healthy"}]
    dataset = field.LeafImages(tmp_path, rows, T.Compose([T.ColorJitter(brightness=0.8), T.ToTensor()]))
    torch.manual_seed(123)
    rng_before = torch.get_rng_state().clone()
    pixels, label = dataset[(0, 100)]
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert label == 7
    torch.rand(500)
    assert torch.equal(dataset[(0, 100)][0], pixels)
    assert not torch.equal(dataset[(0, 200)][0], pixels)
    assert dataset[0][1] == 7
    loader = DataLoader(dataset, sampler=[(0, 100), (0, 200)], batch_size=2, num_workers=1)
    loaded, labels = next(iter(loader))
    assert torch.equal(loaded[0], pixels)
    assert labels.tolist() == [7, 7]


def test_fixed_step_checkpoint_keeps_nonselected_state_and_optimizer_for_resume(tmp_path: Path) -> None:
    initial = {
        "epoch": 0, "best_epoch": 1, "optimizer_steps": 2,
        "model": {"weight": torch.tensor([10])}, "optimizer": {"step": 2}, "scaler": {"scale": 1024},
    }
    best = field._save_epoch_checkpoint(tmp_path, initial, tmp_path / "best.pt", retain_fixed=True)
    final = {**initial, "epoch": 1, "optimizer_steps": 4,
             "model": {"weight": torch.tensor([20])}, "optimizer": {"step": 4}}
    field._save_epoch_checkpoint(tmp_path, final, best, retain_fixed=True)
    assert torch.load(best, weights_only=True)["model"]["weight"].item() == 10
    state = torch.load(tmp_path / "step-4.pt", weights_only=True)
    assert state["model"]["weight"].item() == 20
    assert state["optimizer"] == {"step": 4}
    assert state["scaler"] == {"scale": 1024}
    assert torch.load(tmp_path / "step-2.pt", weights_only=True)["model"]["weight"].item() == 10


def test_trainable_parameters_isolate_head_only_and_last_four_blocks() -> None:
    backbone = torch.nn.Module()
    backbone.layer = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(6)])
    backbone.norm = torch.nn.LayerNorm(2)
    assert field.tunable_parameters(backbone, "head-only") == []
    assert not any(param.requires_grad for param in backbone.parameters())
    tuned = field.tunable_parameters(backbone, "last-four")
    assert {id(param) for param in tuned} == {
        id(param) for part in [*backbone.layer[2:], backbone.norm] for param in part.parameters()
    }
    with pytest.raises(ValueError):
        field.tunable_parameters(backbone, "all")


def test_draw_log_records_consumed_views_and_skipped_updates() -> None:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=field.DRAW_FIELDS)
    writer.writeheader()
    rows = [{"path": "leaf.png", "sha256": "a" * 64, "class_id": "healthy", "source": "farm", "group": "plant"}]
    pixels = torch.tensor([[[[0.25, 0.5]]]])
    field.write_draw_batch(writer, rows, [(0, 321)], pixels, start=12, step=3, updated=False)
    record = next(csv.DictReader(io.StringIO(output.getvalue())))
    assert record == {
        **rows[0], "draw": "12", "train_index": "0", "view_seed": "321", "step": "3", "updated": "False",
        "tensor_sha256": hashlib.sha256(b"\x00\x00\x80\x3e\x00\x00\x00\x3f").hexdigest(),
    }
    with pytest.raises(ValueError):
        field.write_draw_batch(writer, rows, [], pixels, start=0, step=1, updated=True)


def test_verified_manifest_rejects_replaced_image_under_same_record(tmp_path: Path) -> None:
    image = tmp_path / "leaf.png"
    Image.new("RGB", (2, 2), "green").save(image)
    row = {
        "path": "leaf.png", "class_id": "healthy", "group": "plant", "split": "val",
        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
    }
    with (tmp_path / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    assert field.load_manifest(tmp_path, "manifest.csv", verify_hashes=True) == [row]
    Image.new("RGB", (2, 2), "red").save(image)
    with pytest.raises(ValueError, match="digest"):
        field.load_manifest(tmp_path, "manifest.csv", verify_hashes=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_predict_refuses_nonfinite_logits_before_argmax(value: float) -> None:
    class InvalidLogits(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return torch.full((len(images), len(field.CLASSES)), value)

    loader = DataLoader([(torch.zeros(3, 2, 2), 0)])
    with pytest.raises(ValueError, match="finite"):
        field.predict(InvalidLogits(), loader, torch.device("cpu"))


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -0.1, 1.1])
def test_validation_rejects_invalid_probability(confidence: float) -> None:
    row = {"class_id": "bacterial_spot", "domain": "field"}
    with pytest.raises(ValueError, match="confidence"):
        field.validation_report([row], ["bacterial_spot"], [confidence])


def test_predict_collects_all_logits_and_targets_without_changing_top_one() -> None:
    values = torch.tensor([
        [0.0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.0, 0, 0, 0, 0, 0, 0, 0, 2],
        [0.0, 0, 0, 0, 0, 0, 0, 1, 0],
    ])
    loader = DataLoader(TensorDataset(values, torch.tensor([7, 8, 1])), batch_size=2)
    predictions: list[dict[str, Any]] = []
    labels, confidence = field.predict(torch.nn.Identity(), loader, torch.device("cpu"), predictions=predictions)

    assert labels == ["bacterial_spot", "not_tomato_leaf", "healthy"]
    assert confidence == pytest.approx([1 / 9, 0.48015005, 0.25361171])
    assert [row["target_index"] for row in predictions] == [7, 8, 1]
    assert [row["prediction_index"] for row in predictions] == [0, 8, 7]
    assert [row["logits"] for row in predictions] == values.tolist()
    assert predictions[0]["probabilities"] == pytest.approx([1 / 9] * 9)
    assert predictions[1]["probabilities"] == pytest.approx([0.06498124] * 8 + [0.48015005])
    assert json.loads(json.dumps(predictions, allow_nan=False)) == predictions
    assert field.predict(torch.nn.Identity(), loader, torch.device("cpu")) == (labels, confidence)


@pytest.mark.parametrize("invalid", [torch.ones(1, 8), torch.ones(2, 9), torch.ones(1, 9, dtype=torch.int64)])
def test_predict_rejects_wrong_logit_shape_or_type(invalid: torch.Tensor) -> None:
    class WrongLogits(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return invalid

    with pytest.raises(ValueError, match="shape.*real"):
        field.predict(
            WrongLogits(), DataLoader(TensorDataset(torch.zeros(1, 3, 2, 2), torch.tensor([0]))), torch.device("cpu"),
        )


@pytest.mark.parametrize("target", [-1, 9, 0.5, True])
def test_logit_collection_rejects_invalid_target(target: int | float | bool) -> None:
    predictions: list[dict[str, Any]] = []
    with pytest.raises(ValueError, match="target"):
        field.predict(
            torch.nn.Identity(), DataLoader(TensorDataset(torch.zeros(1, 9), torch.tensor([target]))),
            torch.device("cpu"), predictions=predictions,
        )
    assert predictions == []


def test_failed_prediction_does_not_leave_partial_logit_collection() -> None:
    loader = DataLoader(TensorDataset(torch.stack([torch.zeros(9), torch.full((9,), float("nan"))]), torch.tensor([0, 1])))
    predictions: list[dict[str, Any]] = []
    with pytest.raises(ValueError, match="finite"):
        field.predict(torch.nn.Identity(), loader, torch.device("cpu"), predictions=predictions)
    assert predictions == []


def test_logit_collection_refuses_to_mix_prediction_runs() -> None:
    previous = [{"logits": [0.0] * 9}]
    with pytest.raises(ValueError, match="empty"):
        field.predict(
            torch.nn.Identity(), DataLoader(TensorDataset(torch.zeros(1, 9), torch.tensor([0]))),
            torch.device("cpu"), predictions=previous,
        )
    assert previous == [{"logits": [0.0] * 9}]


def test_logit_collection_refuses_empty_evaluation() -> None:
    with pytest.raises(ValueError, match="empty"):
        field.predict(
            torch.nn.Identity(), DataLoader(TensorDataset(torch.empty(0, 9), torch.empty(0, dtype=torch.long))),
            torch.device("cpu"), predictions=[],
        )


def _committed_run(tmp_path: Path) -> dict[str, Any]:
    trace = tmp_path / "draws-epoch-1.csv"
    trace.write_text("test consumed draw trace", encoding="utf-8")
    return {
        "run": {"audit_draws": True, "checkpoint_epochs": [1, 2], "samples": 2, "batch": 1},
        "epoch": 0, "best_epoch": 1, "optimizer_steps": 2,
        "history": [{"epoch": 1, "optimizer_steps": 2,
                     "draw_log_sha256": hashlib.sha256(trace.read_bytes()).hexdigest()}],
        "model": {"weight": torch.tensor([10])}, "optimizer": {"step": 2}, "scaler": {"scale": 1024},
    }


@pytest.mark.parametrize("damage", ["draw", "fixed", "best", "receipt"])
def test_resume_checks_every_committed_artifact(tmp_path: Path, damage: str) -> None:
    state = _committed_run(tmp_path)
    best = field._save_epoch_checkpoint(tmp_path, state, tmp_path / "best.pt", retain_fixed=True)
    committed = torch.load(tmp_path / "last.pt", weights_only=True)
    field.validate_checkpoint_artifacts(tmp_path, committed)
    if damage == "draw":
        (tmp_path / "draws-epoch-1.csv").unlink()
    elif damage == "fixed":
        (tmp_path / "step-2.pt").unlink()
    elif damage == "best":
        field.save_atomic({"model": {"weight": torch.tensor([999])}}, best)
    else:
        committed["artifacts"].pop("step-2.pt")
    with pytest.raises(ValueError, match="artifact"):
        field.validate_checkpoint_artifacts(tmp_path, committed)


def test_interrupted_fixed_save_preserves_old_artifact_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    initial = _committed_run(tmp_path)
    best = field._save_epoch_checkpoint(tmp_path, initial, tmp_path / "best.pt", retain_fixed=True)
    committed = torch.load(tmp_path / "last.pt", weights_only=True)
    second_trace = tmp_path / "draws-epoch-2.csv"
    second_trace.write_text("second epoch draws", encoding="utf-8")
    candidate = {
        **committed, "epoch": 1, "optimizer_steps": 4, "model": {"weight": torch.tensor([20])},
        "history": [*committed["history"], {"epoch": 2, "optimizer_steps": 4,
                    "draw_log_sha256": hashlib.sha256(second_trace.read_bytes()).hexdigest()}],
    }
    replace = os.replace

    def fail_last(source: Path, destination: Path) -> None:
        if destination == tmp_path / "last.pt":
            raise OSError("interrupt commit")
        replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_last)
    with pytest.raises(OSError, match="interrupt"):
        field._save_epoch_checkpoint(tmp_path, candidate, best, retain_fixed=True)
    assert (tmp_path / "step-4.pt").is_file()
    resumed = torch.load(tmp_path / "last.pt", weights_only=True)
    assert resumed["epoch"] == 0
    assert resumed["artifacts"] == committed["artifacts"]
    field.validate_checkpoint_artifacts(tmp_path, resumed)
