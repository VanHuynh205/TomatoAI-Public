"""E01 validation exports use best weights and only the sealed validation rows."""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from ai import train
from ai.data import ManifestDataset
from tests.ai.test_loader import sealed_images as sealed_images  # noqa: PLC0414
from tests.ai.test_train import config as config  # noqa: PLC0414

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(run: Path) -> Path:
    from ai.experiments import export_validation_predictions

    return export_validation_predictions(run)


def test_e01_matches_approved_full_frozen_protocol() -> None:
    expected = json.loads((ROOT / "ai/configs/smoke.json").read_text())
    expected["run_id"] = "E01"
    actual = json.loads((ROOT / "ai/configs/E01.json").read_text())
    assert actual == expected
    train._validate_training_config(actual)


@pytest.mark.parametrize(
    "baseline,run_id,parent",
    [("E05", "E09", None), ("E06", "E10", "ai/runs/E09/best.pth")],
)
def test_yolo_configs_preserve_baseline_protocol(
    baseline: str, run_id: str, parent: str | None
) -> None:
    from ai.preprocessing import YOLO_PREPROCESSING

    expected = json.loads((ROOT / f"ai/configs/{baseline}.json").read_text())
    weights = "ai/pretrained/yolo26/yolo26s-cls-state-dict.pth"
    expected.update(
        run_id=run_id,
        architecture="yolo26s_cls",
        parent_best=parent,
        pretrained_path=weights,
        pretrained_sha256="bcc5048db549f5ca71ff4c65b88e043d13c26d4f86863a6a4bbc7c983dabb7ed",
        preprocessing=YOLO_PREPROCESSING,
    )
    actual = json.loads((ROOT / f"ai/configs/{run_id}.json").read_text())
    assert actual == expected
    train._validate_training_config(actual)


def yolo_configs() -> list[dict[str, Any]]:
    return [
        json.loads((ROOT / f"ai/configs/{name}.json").read_text())
        for name in ("E09", "E10")
    ]


def test_yolo_experiment_pair_is_valid() -> None:
    from ai.experiments import validate_experiment_set

    validate_experiment_set(yolo_configs())


@pytest.mark.parametrize(
    "field,value",
    [
        ("optimizer", {"name": "AdamW", "lr": 1e-3, "weight_decay": 1e-4}),
        ("optimizer", {"name": "SGD", "lr": 1e-4, "weight_decay": 1e-4}),
        ("optimizer", {"name": "AdamW", "lr": 1e-4, "weight_decay": 0.0}),
        ("sample_prefix_weights", {"pv:": 0.5, "plantdoc:": 0.5}),
        ("class_ids", ["healthy"]),
        ("batch", 4),
        ("seed", 43),
        ("pretrained_path", "other.pth"),
        ("pretrained_sha256", "a" * 64),
    ],
)
def test_yolo_experiment_pair_rejects_protocol_drift(field: str, value: Any) -> None:
    from ai.experiments import validate_experiment_set

    configs = yolo_configs()
    configs[1][field] = value
    with pytest.raises(ValueError):
        validate_experiment_set(configs)


def test_yolo_experiment_pair_rejects_baseline_preprocessing() -> None:
    from ai.experiments import validate_experiment_set

    configs = yolo_configs()
    configs[1]["preprocessing"] = json.loads(
        (ROOT / "ai/configs/E06.json").read_text()
    )["preprocessing"]
    with pytest.raises(ValueError, match="YOLO preprocessing"):
        validate_experiment_set(configs)


def test_validation_collector_preserves_order_tail_and_numeric_oracle() -> None:
    logits = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 2.0, 0.0],
        ]
    )
    targets = torch.tensor([0, 1, 1, 2, 2])
    rows: list[dict[str, Any]] = []
    result = train._validate_epoch(
        nn.Identity(),
        [
            (logits[:3], targets[:3], ["a", "b", "c"]),
            (logits[3:], targets[3:], ["d", "e"]),
        ],
        device=torch.device("cpu"),
        class_count=3,
        predictions=rows,
    )
    assert [r["sample_id"] for r in rows] == list("abcde")
    assert [r["target_index"] for r in rows] == targets.tolist()
    assert [r["prediction_index"] for r in rows] == [0, 1, 0, 2, 1]
    torch.testing.assert_close(torch.tensor([r["logits"] for r in rows]), logits)
    torch.testing.assert_close(
        torch.tensor([r["probabilities"] for r in rows]), logits.softmax(1)
    )
    assert result["accuracy"] == 3 / 5
    assert result["macro_f1"] == pytest.approx((2 / 3 + 1 / 2 + 2 / 3) / 3)
    assert result["loss"] == pytest.approx(
        nn.functional.cross_entropy(logits, targets).item()
    )


@pytest.mark.parametrize("ids", [["a", "a"], ["a"], ["a", ""], ["a", 1], "ab"])
def test_collector_rejects_invalid_sample_identity(ids: Any) -> None:
    with pytest.raises(ValueError, match="sample_ids"):
        train._validate_epoch(
            nn.Identity(),
            [(torch.ones(2, 2), torch.tensor([0, 1]), ids)],
            device=torch.device("cpu"),
            class_count=2,
            predictions=[],
        )


@pytest.fixture
def trained_run(
    config: dict[str, Any],
    sealed_images: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    dataset, _ = sealed_images

    def model_factory(
        architecture: str, count: int, pretrained_path: Path | None = None
    ) -> nn.Module:
        model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, count)
        )
        if pretrained_path is not None:
            model.load_state_dict(torch.load(pretrained_path, weights_only=True))
        return model

    weights = tmp_path / "fixture.pth"
    torch.manual_seed(42)
    torch.save(model_factory("efficientnet_b0", 8).state_dict(), weights)
    config.update(
        dataset_dir=str(dataset),
        seal_sha256=digest(dataset / "seal.json"),
        split_hash=digest(dataset / "split.json"),
        pretrained_path=str(weights),
        pretrained_sha256=digest(weights),
        precision="fp32",
        epochs=2,
        batch=3,
        class_ids=[
            r["class_id"]
            for r in json.loads((dataset / "label_map.json").read_text())["classes"]
        ],
    )
    monkeypatch.setattr(train, "build_model", model_factory)
    monkeypatch.setattr("ai.experiments.build_model", model_factory)
    monkeypatch.setattr(
        train,
        "configure_stage",
        lambda model, *_: [{"params": list(model.parameters()), "lr": 0.001}],
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.chdir(tmp_path)
    return train.run_training(config).parent


def test_export_best_full_val_provenance_and_no_train_or_test_pixels(
    trained_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    best = train._load_checkpoint(trained_run / "best.pth", "best")
    last = train._load_checkpoint(trained_run / "last.pth", "last")
    for value in last["model"].values():
        value.add_(100)
    train._atomic_save(trained_run / "last.pth", last)
    before = {
        name: digest(trained_run / name)
        for name in ("best.pth", "last.pth", "metadata.json")
    }
    getitem = ManifestDataset.__getitem__
    visited: list[str] = []

    def val_only(self: ManifestDataset, index: int) -> Any:
        assert self._rows[index]["split"] == "val"
        visited.append(self._rows[index]["sample_id"])
        return getitem(self, index)

    monkeypatch.setattr(ManifestDataset, "__getitem__", val_only)
    output = export(trained_run)
    value = json.loads(output.read_text())
    assert output == trained_run.absolute() / "validation-predictions.json"
    assert value["split"] == "val" and value["schema_version"] == 1
    assert value["checkpoint_sha256"] == before["best.pth"]
    assert value["metadata_sha256"] == before["metadata.json"]
    assert value["config"] == best["config"] and value["identity"] == best["identity"]
    assert value["epoch"] == best["epoch"]
    assert value["class_ids"] == best["config"]["class_ids"]
    assert value["metrics"] == pytest.approx(best["val"])
    assert value["sample_count"] == len(value["predictions"]) == 8
    assert [r["sample_id"] for r in value["predictions"]] == visited
    assert sorted(r["target_index"] for r in value["predictions"]) == list(range(8))
    for row in value["predictions"]:
        assert sum(row["probabilities"]) == pytest.approx(1)
        assert row["prediction_index"] == max(range(8), key=row["logits"].__getitem__)
    assert before == {name: digest(trained_run / name) for name in before}
    with pytest.raises(FileExistsError):
        export(trained_run)


@pytest.mark.parametrize(
    "mutation",
    ["hash", "config", "identity", "diagnostic", "epoch", "metrics", "incomplete"],
)
def test_export_rejects_mismatched_or_incomplete_run(
    trained_run: Path, mutation: str
) -> None:
    path = trained_run / "metadata.json"
    meta = json.loads(path.read_text())
    if mutation == "hash":
        meta["checkpoint_hashes"]["best.pth"] = "0" * 64
    elif mutation == "config":
        meta["config"]["class_ids"].reverse()
    elif mutation == "identity":
        meta["identity"]["split_hash"] = "0" * 64
    elif mutation == "diagnostic":
        cp = train._load_checkpoint(trained_run / "best.pth", "best")
        cp["identity"]["execution"] = {"mode": "smoke"}
        meta["identity"] = copy.deepcopy(cp["identity"])
        train._atomic_save(trained_run / "best.pth", cp)
        meta["checkpoint_hashes"]["best.pth"] = digest(trained_run / "best.pth")
    elif mutation == "epoch":
        meta["best_epoch"] += 1
    elif mutation == "metrics":
        meta["history"][meta["best_epoch"] - 1]["val"]["loss"] += 1
    else:
        meta["epoch"] = 1
        meta["history"] = meta["history"][:1]
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        export(trained_run)
    assert not (trained_run / "validation-predictions.json").exists()


def test_export_rejects_recomputed_metric_drift(
    trained_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = train._validate_epoch

    def changed(*args: Any, **kwargs: Any) -> dict[str, float]:
        result = original(*args, **kwargs)
        result["loss"] += 0.1
        return result

    monkeypatch.setattr(train, "_validate_epoch", changed)
    with pytest.raises(ValueError, match="metrics"):
        export(trained_run)
    assert not (trained_run / "validation-predictions.json").exists()


def test_export_atomic_failure_keeps_checkpoints(
    trained_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = digest(trained_run / "best.pth")

    def fail(*_: Any) -> None:
        raise OSError("disk failure")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="disk failure"):
        export(trained_run)
    assert digest(trained_run / "best.pth") == before
    assert not (trained_run / "validation-predictions.json").exists()
    assert not list(trained_run.glob("*.tmp"))


@pytest.mark.parametrize(
    "path", ["https://example.com/run", "//server/share/run", "\\\\server\\share\\run"]
)
def test_export_rejects_network_paths_before_io(path: str) -> None:
    with pytest.raises(ValueError, match="local"):
        export(Path(path))


@pytest.mark.parametrize("field", ["environment", "source_hashes"])
def test_export_rejects_missing_provenance(trained_run: Path, field: str) -> None:
    path = trained_run / "metadata.json"
    value = json.loads(path.read_text())
    del value[field]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="metadata"):
        export(trained_run)
    assert not (trained_run / "validation-predictions.json").exists()


def test_collector_rejects_duplicate_id_across_batches() -> None:
    with pytest.raises(ValueError, match="sample_ids"):
        train._validate_epoch(
            nn.Identity(),
            [(torch.ones(1, 2), torch.tensor([0]), ["same"])] * 2,
            device=torch.device("cpu"),
            class_count=2,
            predictions=[],
        )


def test_export_rejects_changed_prediction_order(
    trained_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = train._validate_epoch

    def changed(*args: Any, **kwargs: Any) -> dict[str, float]:
        result = original(*args, **kwargs)
        kwargs["predictions"].reverse()
        return result

    monkeypatch.setattr(train, "_validate_epoch", changed)
    with pytest.raises(ValueError, match="coverage/order/labels"):
        export(trained_run)


def test_export_cli_real_b0_reload(
    config: dict[str, Any], sealed_images: tuple[Path, Path], tmp_path: Path
) -> None:
    from torchvision import models  # type: ignore[import-untyped]

    dataset, _ = sealed_images
    # Build a real full-run artifact through the existing CLI on synthetic pixels only.
    pretrained = tmp_path / "efficientnet_b0.pth"
    torch.save(models.efficientnet_b0(weights=None).state_dict(), pretrained)
    config.update(
        dataset_dir=str(dataset),
        seal_sha256=digest(dataset / "seal.json"),
        split_hash=digest(dataset / "split.json"),
        precision="fp32",
        epochs=1,
        pretrained_path=str(pretrained),
        pretrained_sha256=digest(pretrained),
        class_ids=[
            r["class_id"]
            for r in json.loads((dataset / "label_map.json").read_text())["classes"]
        ],
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    run = tmp_path / "run"
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "CUDA_VISIBLE_DEVICES": "",
    }
    for args in (
        ["ai.train", "--config", str(path), "--out", str(run)],
        ["ai.experiments", "export-validation", "--run", str(run)],
    ):
        result = subprocess.run(
            [sys.executable, "-m", *args],
            check=False,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    output = json.loads((run / "validation-predictions.json").read_text())
    assert output["sample_count"] == 8
    assert output["export_environment"]["precision"] == "fp32"
    assert output["config"] == config


def test_e02_config_parent_and_fine_tune_protocol() -> None:
    parent = json.loads((ROOT / "ai/configs/E01.json").read_text())
    expected = copy.deepcopy(parent)
    expected.update(
        run_id="E02", stage="fine_tune", parent_best="ai/runs/E01/best.pth", epochs=20
    )
    expected["optimizer"]["lr"] = 0.0001
    actual = json.loads((ROOT / "ai/configs/E02.json").read_text())
    assert actual == expected
    train._validate_training_config(actual)


@pytest.mark.parametrize(
    "scores,correct,expected",
    [
        ([0.95, 0.85, 0.65, 0.55], [True, True, False, False], 0.7),
        ([0.9, 0.8], [False, False], None),
        ([0.5, 0.5], [True, True], 0.5),
        ([0.9, 0.9, 0.4], [True, True, False], 0.5),
        ([0.9, 0.4, 0.4], [True, False, False], None),
        ([0.6] * 10, [True] * 9 + [False], 0.5),
    ],
)
def test_threshold_validation_grid(
    scores: list[float], correct: list[bool], expected: float | None
) -> None:
    from ai.experiments import select_threshold

    assert select_threshold(scores, correct) == expected


@pytest.mark.parametrize(
    "scores,correct",
    [
        ([], []),
        ([0.9], []),
        ([float("nan")], [True]),
        ([float("inf")], [True]),
        ([-0.1], [True]),
        ([1.1], [True]),
        ([True], [True]),
        ([0.9], [1]),
    ],
)
def test_threshold_rejects_invalid_inputs(scores: Any, correct: Any) -> None:
    from ai.experiments import select_threshold

    with pytest.raises(ValueError):
        select_threshold(scores, correct)


@pytest.mark.parametrize(
    "run_id,stage,epochs,parent,lr",
    [
        ("E03", "frozen", 5, None, 0.001),
        ("E04", "fine_tune", 20, "ai/runs/E03/best.pth", 0.0001),
    ],
)
def test_resnet_configs_match_approved_protocol(
    run_id: str, stage: str, epochs: int, parent: str | None, lr: float
) -> None:
    expected = json.loads((ROOT / "ai/configs/smoke-resnet50.json").read_text())
    expected.update(run_id=run_id, stage=stage, epochs=epochs, parent_best=parent)
    expected["optimizer"]["lr"] = lr
    actual = json.loads((ROOT / f"ai/configs/{run_id}.json").read_text())
    assert actual == expected
    train._validate_training_config(actual)


def experiment_configs() -> list[dict[str, Any]]:
    configs = []
    for architecture, frozen_id, tuned_id, template in [
        ("efficientnet_b0", "E01", "E02", "E01.json"),
        ("resnet50", "E03", "E04", "smoke-resnet50.json"),
    ]:
        frozen = json.loads((ROOT / "ai/configs" / template).read_text())
        frozen.update(run_id=frozen_id, architecture=architecture)
        tuned = copy.deepcopy(frozen)
        tuned.update(
            run_id=tuned_id,
            stage="fine_tune",
            parent_best=f"ai/runs/{frozen_id}/best.pth",
            epochs=20,
        )
        tuned["optimizer"]["lr"] = 0.0001
        configs.extend([frozen, tuned])
    return configs


def test_experiment_set_supports_b0_pair_and_four_configs() -> None:
    from ai.experiments import validate_experiment_set

    configs = experiment_configs()
    validate_experiment_set(configs[:2])
    validate_experiment_set(configs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("split_hash", "f" * 64),
        ("seal_sha256", "f" * 64),
        ("seed", 43),
        ("class_ids", ["reordered", "class"]),
        ("precision", "fp32"),
        ("sample_prefix_weights", {"pv:": 0.75, "plantdoc:": 0.25}),
    ],
)
def test_experiment_set_rejects_comparison_mismatch(field: str, value: Any) -> None:
    from ai.experiments import validate_experiment_set

    configs = experiment_configs()
    configs[-1][field] = value
    with pytest.raises(ValueError):
        validate_experiment_set(configs)


@pytest.fixture
def summary_pair(trained_run: Path) -> list[Path]:
    parent = train._load_checkpoint(trained_run / "best.pth", "best")["config"]
    child = copy.deepcopy(parent)
    child.update(
        run_id="synthetic-fine",
        stage="fine_tune",
        parent_best=str(trained_run / "best.pth"),
    )
    child["optimizer"]["lr"] = 0.0001
    child_run = train.run_training(child).parent
    export(trained_run)
    export(child_run)
    return [trained_run.absolute(), child_run.absolute()]


def test_four_run_summary_preserves_published_b0_decision(
    summary_pair: list[Path], tmp_path: Path
) -> None:
    from ai.experiments import main, summarize_validation

    first = tmp_path / "m5a.json"
    assert (
        main(["summarize", "--runs", *map(str, summary_pair), "--out", str(first)]) == 0
    )
    original = summarize_validation(summary_pair)
    b0 = next(p for p in summary_pair if (p / "decision.json").exists())
    protected = {p: p.read_bytes() for p in (first, b0 / "decision.json")}
    frozen = copy.deepcopy(
        train._load_checkpoint(summary_pair[0] / "best.pth", "best")["config"]
    )
    frozen.update(run_id="resnet-frozen", architecture="resnet50")
    frozen["preprocessing"]["spec"]["interpolation"] = "bilinear"
    resnet_frozen = train.run_training(frozen).parent
    tuned = copy.deepcopy(frozen)
    tuned.update(
        run_id="resnet-fine",
        stage="fine_tune",
        parent_best=str(resnet_frozen / "best.pth"),
    )
    tuned["optimizer"]["lr"] = 0.0001
    resnet_tuned = train.run_training(tuned).parent
    export(resnet_frozen)
    export(resnet_tuned)
    runs = [*summary_pair, resnet_frozen, resnet_tuned]
    output = tmp_path / "m5b.json"
    args = ["summarize", "--runs", *map(str, runs), "--out", str(output)]
    assert main(args) == 0
    value = json.loads(output.read_text())
    assert len(value["runs"]) == 4
    assert set(value["decisions"]) == {"efficientnet_b0", "resnet50"}
    assert (
        value["decisions"]["efficientnet_b0"]
        == original["decisions"]["efficientnet_b0"]
    )
    assert value["app_default_architecture"] == "efficientnet_b0"
    winner = value["selected_by_architecture"]["resnet50"]
    selected = next(p for p in (resnet_frozen, resnet_tuned) if p.name == winner)
    assert (
        json.loads((selected / "decision.json").read_text())
        == value["decisions"]["resnet50"]
    )
    assert all(p.read_bytes() == data for p, data in protected.items())
    before = output.read_bytes()
    assert main(args) == 0 and output.read_bytes() == before


def test_summary_full_rows_baseline_parent_and_hashes_without_pixels(
    summary_pair: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from ai.experiments import summarize_validation

    def forbidden(*_: Any) -> Any:
        raise AssertionError("summary must not decode any pixels")

    monkeypatch.setattr(ManifestDataset, "__getitem__", forbidden)
    value = summarize_validation(summary_pair)
    assert len(value["runs"]) == 2
    assert value["baseline"]["validation"]["accuracy"] == 0.125
    assert value["baseline"]["validation"]["macro_f1"] == pytest.approx(1 / 36)
    assert value["selected_by_architecture"]["efficientnet_b0"] == value["selected_run"]
    decision = value["decisions"]["efficientnet_b0"]
    assert decision["run_id"] == value["selected_run"]
    chosen = next(
        p
        for p in summary_pair
        if json.loads((p / "metadata.json").read_text())["config"]["run_id"]
        == value["selected_run"]
    )
    assert decision["best_sha256"] == digest(chosen / "best.pth")
    assert decision["validation_predictions_sha256"] == digest(
        chosen / "validation-predictions.json"
    )
    assert decision["threshold"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "test_split",
        "missing_row",
        "duplicate_id",
        "wrong_label",
        "nan",
        "probability",
        "metric",
        "hash",
        "parent",
    ],
)
def test_summary_rejects_tampered_validation(
    summary_pair: list[Path], mutation: str
) -> None:
    from ai.experiments import summarize_validation

    path = summary_pair[-1] / "validation-predictions.json"
    pred = json.loads(path.read_text())
    if mutation == "test_split":
        pred["split"] = "test"
    elif mutation == "missing_row":
        pred["predictions"].pop()
    elif mutation == "duplicate_id":
        pred["predictions"][0]["sample_id"] = pred["predictions"][1]["sample_id"]
    elif mutation == "wrong_label":
        pred["predictions"][0]["target_index"] = 7
    elif mutation == "nan":
        pred["predictions"][0]["logits"][0] = float("nan")
    elif mutation == "probability":
        pred["predictions"][0]["probabilities"] = [1.0] * 8
    elif mutation == "metric":
        pred["metrics"]["macro_f1"] = 0.99
    elif mutation == "hash":
        pred["checkpoint_sha256"] = "0" * 64
    else:
        pred["identity"]["parent_best_sha256"] = "0" * 64
    path.write_text(json.dumps(pred))
    with pytest.raises(ValueError):
        summarize_validation(summary_pair)


def test_summary_cli_publishes_decision_idempotently_and_blocks_data_output(
    summary_pair: list[Path], tmp_path: Path
) -> None:
    from ai.experiments import _publish_summary, main

    output = tmp_path / "summary.json"
    args = ["summarize", "--runs", *map(str, summary_pair), "--out", str(output)]
    assert main(args) == 0
    value = json.loads(output.read_text())
    chosen = next(
        row for row in value["runs"] if row["run_id"] == value["selected_run"]
    )
    decision = Path(chosen["run_dir"]) / "decision.json"
    assert json.loads(decision.read_text()) == value["decisions"]["efficientnet_b0"]
    before = digest(decision)
    decision_bytes = decision.read_bytes()
    assert main(args) == 0 and digest(decision) == before
    decision.write_text('{"do_not_overwrite": true}')
    with pytest.raises(SystemExit):
        main(args)
    assert json.loads(decision.read_text()) == {"do_not_overwrite": True}
    decision.write_bytes(decision_bytes)
    for root in value["data"].values():
        unsafe = Path(root) / "unsafe-summary.json"
        with pytest.raises(ValueError, match="outside dataset/raw"):
            _publish_summary(value, unsafe)
        assert not unsafe.exists()
    with pytest.raises(ValueError, match="paths must differ"):
        _publish_summary(value, decision)
    assert digest(decision) == before


def controlled_validation(
    run: Path, wrong: int, strength: float, test_score: float
) -> None:
    """Synthetic records for selection rules; no claim these logits came from training."""
    best = train._load_checkpoint(run / "best.pth", "best")
    meta = json.loads((run / "metadata.json").read_text())
    pred = json.loads((run / "validation-predictions.json").read_text())
    targets = torch.tensor([row["target_index"] for row in pred["predictions"]])
    logits = torch.zeros(len(targets), len(best["config"]["class_ids"]))
    for i, target in enumerate(targets.tolist()):
        logits[i, (target + 1) % logits.shape[1] if i < wrong else target] = strength
    predictions: list[dict[str, Any]] = []
    metrics = train._validate_epoch(
        nn.Identity(),
        [(logits, targets, [row["sample_id"] for row in pred["predictions"]])],
        device=torch.device("cpu"),
        class_count=logits.shape[1],
        predictions=predictions,
    )
    best.update(epoch=1, val=metrics)
    best["identity"] = train._training_identity(best["config"])
    train._atomic_save(run / "best.pth", best)
    meta.update(best_epoch=1, identity=best["identity"], test={"macro_f1": test_score})
    for i, row in enumerate(meta["history"]):
        row.update(val=metrics, is_best=i == 0, bad_epochs=i)
    meta["checkpoint_hashes"]["best.pth"] = digest(run / "best.pth")
    train._atomic_save(run / "metadata.json", meta)
    pred.update(
        epoch=1,
        predictions=predictions,
        metrics=metrics,
        identity=best["identity"],
        checkpoint_sha256=digest(run / "best.pth"),
        metadata_sha256=digest(run / "metadata.json"),
    )
    train._atomic_save(run / "validation-predictions.json", pred)


@pytest.mark.parametrize(
    "mode,expected",
    [("f1", "synthetic-cpu"), ("loss", "synthetic-fine"), ("tie", "synthetic-cpu")],
)
def test_summary_ignores_test_metrics_and_resolves_ties(
    summary_pair: list[Path], mode: str, expected: str
) -> None:
    from ai.experiments import summarize_validation

    controlled_validation(summary_pair[0], wrong=0, strength=2, test_score=0)
    controlled_validation(
        summary_pair[1],
        wrong=1 if mode == "f1" else 0,
        strength=3 if mode == "loss" else 2,
        test_score=1,
    )
    result = summarize_validation(list(reversed(summary_pair)))
    assert result["selected_run"] == expected
    decision = result["decisions"]["efficientnet_b0"]
    assert decision["threshold"] == 0.5
    assert decision["coverage"] == 1
    assert decision["accepted_accuracy"] == 1


def test_summary_rejects_unrelated_parent_pair(
    summary_pair: list[Path], tmp_path: Path
) -> None:
    from ai.experiments import summarize_validation

    other = tmp_path / "unrelated-parent"
    shutil.copytree(summary_pair[0], other)
    best_path = summary_pair[1] / "best.pth"
    best = train._load_checkpoint(best_path, "best")
    best["config"]["parent_best"] = str(other / "best.pth")
    train._atomic_save(best_path, best)
    meta_path = summary_pair[1] / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["config"] = best["config"]
    meta["checkpoint_hashes"]["best.pth"] = digest(best_path)
    train._atomic_save(meta_path, meta)
    pred_path = summary_pair[1] / "validation-predictions.json"
    pred = json.loads(pred_path.read_text())
    pred.update(
        config=best["config"],
        checkpoint_sha256=digest(best_path),
        metadata_sha256=digest(meta_path),
    )
    train._atomic_save(pred_path, pred)
    with pytest.raises(ValueError, match="descend"):
        summarize_validation(summary_pair)


def test_export_completed_fine_tune_after_early_stop(
    trained_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = train._load_checkpoint(trained_run / "best.pth", "best")["config"]
    config.update(
        run_id="synthetic-early-stop",
        stage="fine_tune",
        parent_best=str(trained_run / "best.pth"),
        epochs=20,
        patience=1,
    )
    config["optimizer"]["lr"] = 0.0001
    monkeypatch.setattr(
        train,
        "train_epoch",
        lambda *_, **__: {"loss": 1.0, "optimizer_steps": 0.0, "skipped_steps": 0.0},
    )
    run = train.run_training(config).parent
    meta = json.loads((run / "metadata.json").read_text())
    assert meta["epoch"] == 2 and meta["history"][-1]["bad_epochs"] == 1
    predictions = json.loads(export(run).read_text())
    assert predictions["epoch"] == 1 and predictions["sample_count"] == 8
