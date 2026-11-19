"""M4 config guards, synthetic CPU stage steps and a five-sample oracle."""

import copy
import hashlib
import json
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ai.models import build_model
from ai.train import _training_loaders as real_training_loaders
from tests.ai.test_loader import sealed_images as sealed_images  # noqa: PLC0414 - Re-export registers the shared pytest fixture.

ROOT = Path(__file__).resolve().parents[2]
FIELDS = [
    "schema_version",
    "run_id",
    "architecture",
    "stage",
    "parent_best",
    "dataset_dir",
    "seal_sha256",
    "split_hash",
    "pretrained_path",
    "pretrained_sha256",
    "train_split",
    "val_split",
    "class_ids",
    "preprocessing",
    "seed",
    "batch",
    "accumulation",
    "precision",
    "optimizer",
    "epochs",
    "patience",
]


def validate(config: dict[str, Any]) -> None:
    from ai.train import _validate_training_config

    _validate_training_config(config)


def read_json(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_source_sampling_balances_train_only_and_restores_rng(
    config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from torch.utils.data import SequentialSampler, WeightedRandomSampler

    import ai.train as training

    class Samples(Dataset[Any]):
        class_ids = tuple(config["class_ids"])

        def __init__(self, *args: Any) -> None:
            self._rows = [{"sample_id": f"pv:{i}"} for i in range(6)] + [
                {"sample_id": f"plantdoc:{i}"} for i in range(2)
            ]

        def __len__(self) -> int:
            return len(self._rows)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
            return torch.zeros(3, 2, 2), 0, self._rows[index]["sample_id"]

    monkeypatch.setattr(training, "ManifestDataset", Samples)
    config["sample_prefix_weights"] = {"pv:": 0.5, "plantdoc:": 0.5}
    loaders = real_training_loaders(config)
    sampler = loaders[0].sampler
    assert isinstance(sampler, WeightedRandomSampler)
    assert sampler.num_samples == 8 and sampler.replacement
    assert float(sampler.weights[:6].sum()) == pytest.approx(0.5)
    assert float(sampler.weights[6:].sum()) == pytest.approx(0.5)
    assert isinstance(loaders[1].sampler, SequentialSampler)
    state = training._rng_state(loaders)
    first = [sample for _, _, ids in loaders[0] for sample in ids]
    training._restore_rng(state, loaders)
    assert first == [sample for _, _, ids in loaders[0] for sample in ids]
    config["sample_prefix_weights"] = {"pv:": 0.5, "missing:": 0.5}
    with pytest.raises(ValueError, match="sample_prefix_weights"):
        real_training_loaders(config)


@pytest.mark.parametrize(
    "value",
    [
        {},
        None,
        {"pv:": 0},
        {"pv:": True},
        {"pv:": float("nan")},
        {"pv:": 0.7},
        {"pv:": 0.5, "pv:a": 0.5},
    ],
)
def test_source_sampling_rejects_invalid_weights(
    config: dict[str, Any], value: Any
) -> None:
    config["sample_prefix_weights"] = value
    with pytest.raises(ValueError, match="sample_prefix_weights"):
        validate(config)


@pytest.fixture
def yolo_stage_model() -> Iterator[nn.Module]:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(71)
            yield build_model("yolo26s_cls", 8)
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("stage", ["frozen", "fine_tune"])
def test_yolo_stage_gradients_modes_and_real_optimizer_update(
    yolo_stage_model: nn.Module, stage: str
) -> None:
    from ai.train import configure_stage, train_epoch

    model = yolo_stage_model
    prefixes: tuple[str, ...] = ("model.model.10.linear.",)
    if stage == "fine_tune":
        prefixes += ("model.model.8.", "model.model.9.", "model.model.10.conv.")
    before = {name: value.clone() for name, value in model.state_dict().items()}
    groups = configure_stage(model, "yolo26s_cls", stage)
    expected_names = {
        name for name, _ in model.named_parameters() if name.startswith(prefixes)
    }
    assert expected_names
    assert {
        name for name, p in model.named_parameters() if p.requires_grad
    } == expected_names
    assert [g["name"] for g in groups] == (
        ["head"] if stage == "frozen" else ["head", "backbone"]
    )
    assert [g["lr"] for g in groups] == ([1e-3] if stage == "frozen" else [1e-4, 1e-5])
    grouped = [p for group in groups for p in group["params"]]
    assert len(grouped) == len({id(p) for p in grouped}) == len(expected_names)
    assert all(not m.training for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    inputs = torch.randn(2, 3, 64, 64)
    for _ in range(2):
        model.train()
        configure_stage(model, "yolo26s_cls", stage)
        output = model(inputs)
        assert output.shape == (2, 8) and torch.isfinite(output).all()
        result = train_epoch(
            model,
            [(inputs, torch.tensor([0, 2]), None)],
            optimizer,
            device=torch.device("cpu"),
            accumulation=1,
            amp=False,
        )
        assert result["optimizer_steps"] == 1 and result["skipped_steps"] == 0
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    after = model.state_dict()
    assert all(
        torch.equal(after[name], value)
        for name, value in before.items()
        if name not in expected_names
    )
    assert not torch.equal(
        after["model.model.10.linear.weight"], before["model.model.10.linear.weight"]
    )
    if stage == "fine_tune":
        for prefix in prefixes[1:]:
            assert any(
                not torch.equal(after[name], before[name])
                for name in expected_names
                if name.startswith(prefix)
            )
    assert {int(state["step"]) for state in optimizer.state.values()} == {2}
    model.eval()
    with torch.inference_mode():
        assert model(inputs).shape == (2, 8)


@pytest.mark.parametrize(
    "damage", ["scale", "head", "tail", "channels", "export", "overlap"]
)
def test_yolo_stage_rejects_changed_structure_without_mutation(
    yolo_stage_model: nn.Module, damage: str
) -> None:
    from ai.train import configure_stage
    from ai.yolo import Yolo26Classifier

    model = yolo_stage_model
    assert isinstance(model, Yolo26Classifier)
    blocks = model.model.model
    if damage == "scale":
        model.model.yaml["scale"] = "n"
    elif damage == "head":
        blocks[-1].linear = nn.Linear(640, 8)
    elif damage == "tail":
        blocks[8] = nn.Sequential(nn.Linear(2, 2))
    elif damage == "channels":
        blocks[9].cv1.conv = nn.Conv2d(512, 256, 1)
    elif damage == "export":
        blocks[-1].export = True
    else:
        blocks[8].register_parameter("shared", blocks[-1].linear.weight)
    modes = {name: m.training for name, m in model.named_modules()}
    flags = {name: p.requires_grad for name, p in model.named_parameters()}
    with pytest.raises(ValueError, match="model|backbone"):
        configure_stage(model, "yolo26s_cls", "fine_tune")
    assert {name: m.training for name, m in model.named_modules()} == modes
    assert {name: p.requires_grad for name, p in model.named_parameters()} == flags


def test_yolo_config_preserves_pinned_profile(config: dict[str, Any]) -> None:
    from ai.preprocessing import YOLO_PREPROCESSING

    config.update(
        architecture="yolo26s_cls", preprocessing=copy.deepcopy(YOLO_PREPROCESSING)
    )
    validate(config)
    config["preprocessing"]["spec"]["mean"] = [0.485, 0.456, 0.406]
    with pytest.raises(ValueError, match="preprocessing"):
        validate(config)


@pytest.mark.parametrize("field,value", [("lr", 0.01), ("weight_decay", 0.01)])
def test_yolo_runner_rejects_changed_optimizer_policy_before_io(
    config: dict[str, Any], field: str, value: float
) -> None:
    from ai.preprocessing import YOLO_PREPROCESSING
    from ai.train import run_training

    config.update(
        architecture="yolo26s_cls", preprocessing=copy.deepcopy(YOLO_PREPROCESSING)
    )
    config["optimizer"][field] = value
    with pytest.raises(ValueError, match=f"optimizer.{field}"):
        run_training(config)


@pytest.fixture
def config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": "synthetic-cpu",
        "architecture": "efficientnet_b0",
        "stage": "frozen",
        "parent_best": None,
        "dataset_dir": "nonexistent/synthetic-dataset",
        "seal_sha256": "a" * 64,
        "split_hash": "b" * 64,
        "pretrained_path": "nonexistent/synthetic-weights.pth",
        "pretrained_sha256": "c" * 64,
        "train_split": "train",
        "val_split": "val",
        "class_ids": ["synthetic_a", "synthetic_b"],
        "preprocessing": {
            "profile_version": "whole-frame-v1",
            "spec": read_json("docs/experiments/preprocessing-v1.json")["profiles"][
                "efficientnet_b0"
            ],
        },
        "seed": 42,
        "batch": 8,
        "accumulation": 2,
        "precision": "amp_fp16",
        "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 0.0001},
        "epochs": 5,
        "patience": 5,
    }


def test_smoke_matches_real_metadata_and_approved_profile() -> None:
    smoke = read_json("ai/configs/smoke.json")
    profiles = read_json("docs/experiments/preprocessing-v1.json")
    labels = read_json("infra/derived/pv8-v1/label_map.json")["classes"]
    registry = read_json("docs/experiments/pretrained-weights.json")["weights"]
    weights = next(w for w in registry if w["architecture"] == "efficientnet_b0")
    assert set(smoke) == set(FIELDS)
    assert smoke == {
        "schema_version": 1,
        "run_id": "smoke",
        "architecture": "efficientnet_b0",
        "stage": "frozen",
        "parent_best": None,
        "dataset_dir": "infra/derived/pv8-v1",
        "seal_sha256": hashlib.sha256(
            (ROOT / "infra/derived/pv8-v1/seal.json").read_bytes()
        ).hexdigest(),
        "split_hash": hashlib.sha256(
            (ROOT / "infra/derived/pv8-v1/split.json").read_bytes()
        ).hexdigest(),
        "pretrained_path": weights["path"],
        "pretrained_sha256": weights["sha256"],
        "train_split": "train",
        "val_split": "val",
        "class_ids": [
            item["class_id"] for item in sorted(labels, key=lambda x: x["index"])
        ],
        "preprocessing": {
            "profile_version": profiles["profile_version"],
            "spec": profiles["profiles"]["efficientnet_b0"],
        },
        "seed": 42,
        "batch": 8,
        "accumulation": 2,
        "precision": "amp_fp16",
        "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 0.0001},
        "epochs": 5,
        "patience": 5,
    }
    assert smoke["preprocessing"]["profile_version"] == "whole-frame-v1"
    assert smoke["preprocessing"]["spec"]["geometry"] == "fit_pad"
    validate(smoke)


@pytest.mark.parametrize("field", ["train_split", "val_split"])
@pytest.mark.parametrize("value", ["test", "", None, "TRAIN", "VAL", [], True])
def test_split_rejected_before_bad_transform(
    config: dict[str, Any], field: str, value: Any
) -> None:
    config[field] = value
    config["preprocessing"] = {"spec": "invalid"}
    with pytest.raises(ValueError, match=field):
        validate(config)


@pytest.mark.parametrize("field", FIELDS)
def test_required_fields(config: dict[str, Any], field: str) -> None:
    del config[field]
    with pytest.raises(ValueError, match=field):
        validate(config)


@pytest.mark.parametrize("value", [True, False, 0, 2, 1.0, "1", None])
def test_schema_version(config: dict[str, Any], value: Any) -> None:
    config["schema_version"] = value
    with pytest.raises(ValueError, match="schema_version"):
        validate(config)


@pytest.mark.parametrize(
    "field", ["batch", "accumulation", "epochs", "patience", "seed"]
)
@pytest.mark.parametrize(
    "value", [True, False, -1, 1.5, "2", None, float("inf"), float("nan")]
)
def test_integer_fields(config: dict[str, Any], field: str, value: Any) -> None:
    config[field] = value
    with pytest.raises(ValueError, match=field):
        validate(config)


@pytest.mark.parametrize("field", ["batch", "accumulation", "epochs", "patience"])
def test_positive_integers(config: dict[str, Any], field: str) -> None:
    config[field] = 0
    with pytest.raises(ValueError, match=field):
        validate(config)


@pytest.mark.parametrize("field", ["architecture", "stage", "precision"])
@pytest.mark.parametrize("value", ["", "unknown", None, [], True])
def test_choices(config: dict[str, Any], field: str, value: Any) -> None:
    config[field] = value
    with pytest.raises(ValueError, match=field):
        validate(config)


@pytest.mark.parametrize("field", ["seal_sha256", "split_hash", "pretrained_sha256"])
@pytest.mark.parametrize("value", ["", "a" * 63, "g" * 64, None, 123, "a" * 64 + "\n"])
def test_hash_shape(config: dict[str, Any], field: str, value: Any) -> None:
    config[field] = value
    with pytest.raises(ValueError, match=field):
        validate(config)


@pytest.mark.parametrize("field", ["run_id", "dataset_dir", "pretrained_path"])
@pytest.mark.parametrize("value", ["", "  ", None, 123])
def test_nonblank_strings(config: dict[str, Any], field: str, value: Any) -> None:
    config[field] = value
    with pytest.raises(ValueError, match=field):
        validate(config)


@pytest.mark.parametrize("field", ["dataset_dir", "pretrained_path", "parent_best"])
@pytest.mark.parametrize(
    "value",
    [
        "https://example.com/file",
        "file:///tmp/file",
        "s3://bucket/file",
        "https:/file",
        "x://bucket/file",
        "a\x00b",
    ],
)
def test_paths_are_local(config: dict[str, Any], field: str, value: str) -> None:
    if field == "parent_best":
        config["stage"] = "fine_tune"
    config[field] = value
    with pytest.raises(ValueError, match=field):
        validate(config)


@pytest.mark.parametrize(
    "value", [[], ["x", "x"], [""], [" "], [None], "ab", None, [["x"]]]
)
def test_class_ids(config: dict[str, Any], value: Any) -> None:
    config["class_ids"] = value
    with pytest.raises(ValueError, match="class_ids"):
        validate(config)


@pytest.mark.parametrize(
    "class_id",
    [
        "Healthy",
        "1healthy",
        "leaf.spot",
        "leaf-spot",
        " healthy",
        "healthy ",
        "healthy\n",
        "healthy\t",
        "a\nb",
        "a" * 65,
        "lá",
    ],
)
def test_class_id_syntax_rejected_before_training_io(
    config: dict[str, Any],
    class_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai import train

    config["class_ids"] = [class_id]
    monkeypatch.setattr(
        train,
        "_training_identity",
        lambda *_: pytest.fail("invalid class_ids reached artifact IO"),
    )
    with pytest.raises(ValueError, match="class_ids"):
        validate(config)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="class_ids"):
        train.run_training(config, out=output)
    assert not output.exists()


def test_valid_class_ids_preserve_order_in_training_config(
    config: dict[str, Any],
) -> None:
    config["class_ids"] = ["z", "a0_b", "a" * 64]
    validate(config)
    assert config["class_ids"] == ["z", "a0_b", "a" * 64]


@pytest.mark.parametrize("field", ["lr", "weight_decay"])
@pytest.mark.parametrize(
    "value", [True, False, -1, "0.1", None, float("inf"), float("nan"), 10**400]
)
def test_optimizer_numbers(config: dict[str, Any], field: str, value: Any) -> None:
    config["optimizer"][field] = value
    with pytest.raises(ValueError, match=field):
        validate(config)


def test_optimizer_lr_positive(config: dict[str, Any]) -> None:
    config["optimizer"]["lr"] = 0
    with pytest.raises(ValueError, match="lr"):
        validate(config)


@pytest.mark.parametrize(
    "value", [None, [], {}, {"name": "SGD", "lr": 0.1, "weight_decay": 0}]
)
def test_optimizer_shape(config: dict[str, Any], value: Any) -> None:
    config["optimizer"] = value
    with pytest.raises(ValueError, match="optimizer"):
        validate(config)


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"profile_version": "", "spec": {}},
        {"profile_version": "whole-frame-v1", "spec": {}},
    ],
)
def test_preprocessing_shape(config: dict[str, Any], value: Any) -> None:
    config["preprocessing"] = value
    with pytest.raises(ValueError, match="preprocessing"):
        validate(config)


def test_invalid_transform_is_rejected(config: dict[str, Any]) -> None:
    config["preprocessing"]["spec"]["std"] = [0, 1, 1]
    with pytest.raises(ValueError, match="std"):
        validate(config)


@pytest.mark.parametrize(
    "stage,parent",
    [
        ("frozen", "parent.pt"),
        ("fine_tune", None),
        ("fine_tune", ""),
        ("fine_tune", " "),
        ("fine_tune", 123),
    ],
)
def test_stage_parent_relationship(
    config: dict[str, Any], stage: str, parent: Any
) -> None:
    config.update(stage=stage, parent_best=parent)
    with pytest.raises(ValueError, match="parent_best"):
        validate(config)


def test_valid_synthetic_config_has_no_io_or_mutation(
    config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    before = copy.deepcopy(config)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("config validation attempted artifact/network/model/CUDA access")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr(torch.hub, "download_url_to_file", forbidden)
    validate(config)
    assert config == before
    config.update(
        architecture="resnet50",
        stage="fine_tune",
        parent_best="C:\\missing\\best.pt",
        precision="fp32",
        seed=0,
    )
    config["optimizer"]["weight_decay"] = 0
    config["pretrained_sha256"] = "A" * 64
    validate(config)


@pytest.mark.parametrize("value", [None, [], "config"])
def test_config_is_object(value: Any) -> None:
    with pytest.raises(ValueError, match="config"):
        validate(value)


def test_unknown_config_field(config: dict[str, Any]) -> None:
    config["train_splt"] = "test"
    with pytest.raises(ValueError, match="train_splt"):
        validate(config)


@dataclass
class TrainingCase:
    model: nn.Linear
    optimizer: torch.optim.SGD
    loader: DataLoader[Any]
    expected_weight: torch.Tensor


@pytest.fixture
def training_case() -> TrainingCase:
    with torch.random.fork_rng(devices=[]):
        model = nn.Linear(2, 2, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2, -0.1], [-0.3, 0.4]]))
        model.bias.copy_(torch.tensor([0.05, -0.05]))
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5], [0.5, -1.0]],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 0, 1, 0], dtype=torch.int64)
    sample_ids = [f"synthetic:{i}" for i in range(5)]
    loader: DataLoader[Any] = DataLoader(
        list(zip(features, targets, sample_ids, strict=True)),  # type: ignore[arg-type]
        batch_size=2,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(42),
    )
    reference = copy.deepcopy(model)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    loss = nn.CrossEntropyLoss()
    for section in (slice(0, 4), slice(4, 5)):
        reference_optimizer.zero_grad()
        loss(reference(features[section]), targets[section]).backward()
        reference_optimizer.step()
    return TrainingCase(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        loader,
        reference.weight.detach().clone(),
    )


def test_training_case_contract_and_independent_oracle(
    training_case: TrainingCase,
) -> None:
    case = training_case
    batches = list(case.loader)
    assert [len(y) for _, y, _ in batches] == [2, 2, 1]
    assert [item for _, _, ids in batches for item in ids] == [
        f"synthetic:{i}" for i in range(5)
    ]
    assert case.loader.batch_size == 2 and case.loader.num_workers == 0
    assert not case.loader.drop_last
    assert case.optimizer.param_groups[0]["lr"] == 0.1
    assert not case.optimizer.state
    assert tuple(case.model.weight.shape) == (2, 2)
    assert tuple(case.model.bias.shape) == (2,)
    for x, y, _ in batches:
        assert x.dtype == torch.float32 and y.dtype == torch.int64
        assert tuple(x.shape) == (len(y), 2)
        assert y.min() >= 0 and y.max() < 2
        assert x.device.type == y.device.type == "cpu"
        assert torch.isfinite(x).all()
    for parameter in case.model.parameters():
        assert parameter.device.type == "cpu" and torch.isfinite(parameter).all()
        assert parameter.grad is None
    assert case.expected_weight.device.type == "cpu"
    assert case.expected_weight.dtype == torch.float32
    assert (
        torch.isfinite(case.expected_weight).all()
        and not case.expected_weight.requires_grad
    )
    assert not torch.equal(case.expected_weight, case.model.weight)
    x = torch.cat([batch[0] for batch in batches])
    y = torch.cat([batch[1] for batch in batches])
    weight = torch.tensor([[0.2, -0.1], [-0.3, 0.4]])
    bias = torch.tensor([0.05, -0.05])
    torch.testing.assert_close(case.model.weight, weight, rtol=0, atol=0)
    torch.testing.assert_close(case.model.bias, bias, rtol=0, atol=0)
    for section in (slice(0, 4), slice(4, 5)):
        features, targets = x[section], y[section]
        error = (features @ weight.T + bias).softmax(dim=1) - nn.functional.one_hot(
            targets, 2
        )
        weight -= 0.1 * error.T @ features / len(targets)
        bias -= 0.1 * error.mean(dim=0)
    torch.testing.assert_close(case.expected_weight, weight)


def test_training_case_preserves_global_rng(request: pytest.FixtureRequest) -> None:
    before = torch.random.get_rng_state().clone()
    case = request.getfixturevalue("training_case")
    list(case.loader)
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.fixture(params=["efficientnet_b0", "resnet50"])
def stage_model(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[str, nn.Module]]:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("stage configuration attempted weights/network/pixels/CUDA access")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.hub, "download_url_to_file", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr(nn.Module, "cuda", forbidden)
    monkeypatch.setattr(torch.Tensor, "cuda", forbidden)
    monkeypatch.setattr(Image, "open", forbidden)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(23)
            architecture = str(request.param)
            yield architecture, build_model(architecture, 3, pretrained_path=None)
    finally:
        torch.set_num_threads(previous_threads)


def test_cli_help_and_invalid_config(tmp_path: Path) -> None:
    import subprocess
    import sys

    help_result = subprocess.run(
        [sys.executable, "-m", "ai.train", "--help"],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0 and "--profile-steps" in help_result.stdout
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}")
    rejected = subprocess.run(
        [sys.executable, "-m", "ai.train", "--config", str(invalid)],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0 and "train_split" in rejected.stderr


def test_profile_stream_exact_budget_reloads_without_overread() -> None:
    import ai.train as training

    assert hasattr(training, "_batch_stream"), "bounded profiling stream missing"
    visits = 0

    class Batches:
        def __iter__(self) -> Iterator[Any]:
            nonlocal visits
            for start in (0, 2, 4):
                visits += 1
                size = min(2, 5 - start)
                yield (
                    torch.ones(size, 2),
                    torch.zeros(size, dtype=torch.long),
                    list(range(start, start + size)),
                )

    stream = list(training._batch_stream(Batches(), 100))
    assert visits == len(stream) == 100
    assert sum(batch[0].shape[0] for batch in stream) == 167
    with pytest.raises(ValueError, match="empty"):
        list(training._batch_stream([], 100))


def test_profile_stream_tail_matches_numeric_reference(
    training_case: TrainingCase,
) -> None:
    import ai.train as training

    assert hasattr(training, "_batch_stream"), "bounded profiling stream missing"
    inputs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 2.0], [2.0, -1.0]]
    )
    targets = torch.tensor([0, 1, 0, 1, 0])
    batches = [
        (
            inputs[i : i + 2],
            targets[i : i + 2],
            [str(j) for j in range(i, min(i + 2, 5))],
        )
        for i in (0, 2, 4)
    ]
    model = nn.Linear(2, 2)
    reference = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    expected = torch.optim.SGD(reference.parameters(), lr=0.1)
    for x, y in [(inputs, targets), (inputs, targets), (inputs[:2], targets[:2])]:
        expected.zero_grad()
        torch.autograd.backward(nn.functional.cross_entropy(reference(x), y))
        expected.step()
    metrics = training.train_epoch(
        model,
        training._batch_stream(batches, 7),
        optimizer,
        device=torch.device("cpu"),
        accumulation=3,
        amp=False,
    )
    assert metrics["optimizer_steps"] == 3
    torch.testing.assert_close(
        model.state_dict(), reference.state_dict(), rtol=0, atol=1e-6
    )


@pytest.fixture
def diagnostic_case(
    resume_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    import ai.train as training

    def factory(
        architecture: str, class_count: int, pretrained_path: Path | None = None
    ) -> nn.Module:
        model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(3, class_count),
        )
        if pretrained_path is not None:
            model.load_state_dict(torch.load(pretrained_path, weights_only=True))
        return model

    path = Path(resume_case["pretrained_path"])
    torch.save(factory("efficientnet_b0", 8).state_dict(), path)
    resume_case["pretrained_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    resume_case.update(epochs=1, patience=1)
    monkeypatch.setattr(training, "build_model", factory)
    monkeypatch.setattr(training, "_training_loaders", real_training_loaders)
    return resume_case


def test_profile_budget_independent_of_epochs_and_patience(
    diagnostic_case: dict[str, Any], tmp_path: Path
) -> None:
    import ai.train as training

    out = tmp_path / "profile-output"
    best = training.run_training(diagnostic_case, out=out, profile_steps=7)
    assert best == out / "best.pth"
    checkpoint = torch.load(out / "last.pth", weights_only=True)
    work = checkpoint["history"][0]["work"]
    assert work["physical_mini_batches"] == 7 and work["samples"] == 14
    assert len(work["sample_ids"]) == 14
    assert checkpoint["history"][0]["train"]["optimizer_steps"] == 4
    report = json.loads((out / "performance.json").read_text())
    assert report["mode"] == "profile" and report["requested_mini_batches"] == 7
    assert report["physical_mini_batches"] == 7 and report["warmup_mini_batches"] == 0
    assert report["train_seconds"] > 0 and report["validation_seconds"] > 0
    assert (
        report["save_seconds"] > 0 and report["wall_seconds"] >= report["train_seconds"]
    )


def test_full_epoch_timing_preserves_checkpoint_and_validation(
    diagnostic_case: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import ai.train as training

    regular = tmp_path / "regular"
    timed = tmp_path / "timed"
    training.run_training(diagnostic_case, out=regular)
    assert capsys.readouterr().out == ""
    training.run_training(diagnostic_case, out=timed, epoch_timing=True)
    event = json.loads(capsys.readouterr().out)
    assert event["event"] == "epoch_committed" and event["epoch"] == 1
    for phase in ("train_seconds", "validation_seconds", "checkpoint_seconds"):
        assert math.isfinite(event[phase]) and event[phase] > 0
    assert event["peak_reserved_bytes"] is None
    actual = torch.load(timed / "last.pth", weights_only=True)
    assert "execution" not in actual["identity"]
    assert "work" not in actual["history"][0]
    assert event["val"] == actual["history"][0]["val"]
    expected = torch.load(regular / "last.pth", weights_only=True)
    assert_checkpoint_equal(actual, expected)


def test_diagnostic_output_cannot_write_into_raw(
    diagnostic_case: dict[str, Any],
) -> None:
    import ai.train as training
    from ai.data import ManifestDataset

    loaders = real_training_loaders(diagnostic_case)
    dataset = loaders[0].dataset
    assert isinstance(dataset, ManifestDataset)
    output = dataset.source_root / "new-training-output"
    with pytest.raises(ValueError, match="source_root"):
        training.run_training(diagnostic_case, out=output, smoke=True)
    assert not output.exists()


@pytest.mark.parametrize("field", ["source_hashes", "environment"])
def test_resume_rejects_changed_execution_provenance(
    resume_case: dict[str, Any], field: str
) -> None:
    best = run_case(resume_case)
    last = best.with_name("last.pth")
    checkpoint = torch.load(last, weights_only=True)
    key = "train.py" if field == "source_hashes" else "torch"
    checkpoint[field][key] = "different-revision"
    torch.save(checkpoint, last)
    before = last.read_bytes()
    with pytest.raises(ValueError, match="provenance"):
        run_case(resume_case, last)
    assert last.read_bytes() == before


def test_completed_smoke_resume_reports_zero_work(
    diagnostic_case: dict[str, Any], tmp_path: Path
) -> None:
    import ai.train as training

    out = tmp_path / "completed-smoke"
    best = training.run_training(diagnostic_case, out=out, smoke=True)
    last = best.with_name("last.pth")
    before = torch.load(last, weights_only=True)
    training.run_training(diagnostic_case, last, out=out, smoke=True)
    report = json.loads((out / "performance.json").read_text())
    assert report["invocation_start_epoch"] == report["completed_epoch"] == 1
    for field in (
        "physical_mini_batches",
        "samples",
        "optimizer_updates",
        "skipped_updates",
        "train_seconds",
        "validation_seconds",
    ):
        assert report[field] == 0
    assert_checkpoint_equal(before, torch.load(last, weights_only=True), atol=0)


@pytest.mark.parametrize(
    "path",
    [
        r"\\server\share\weights.pth",
        r"\\?\UNC\server\share\weights.pth",
        "//server/share/weights.pth",
    ],
)
def test_unc_paths_rejected_before_io(config: dict[str, Any], path: str) -> None:
    import ai.train as training

    for field in ("dataset_dir", "pretrained_path"):
        changed = copy.deepcopy(config)
        changed[field] = path
        with pytest.raises(ValueError, match="local"):
            training._validate_training_config(changed)
    for field in ("out", "resume"):
        options: dict[str, Any] = {field: Path(path)}
        with pytest.raises(ValueError, match="local"):
            training.run_training(config, **options)


def test_smoke_identity_and_diagnostic_parent_guard(
    diagnostic_case: dict[str, Any], tmp_path: Path
) -> None:
    import ai.train as training

    out = tmp_path / "smoke-output"
    best = training.run_training(diagnostic_case, out=out, smoke=True)
    last = out / "last.pth"
    checkpoint = torch.load(last, weights_only=True)
    execution = checkpoint["identity"]["execution"]
    assert execution["mode"] == "smoke"
    assert len(execution["train_sample_ids"]) == 8
    assert set(execution["train_sample_ids"]).isdisjoint(execution["val_sample_ids"])
    with pytest.raises(ValueError, match="execution|identity"):
        training.run_training(diagnostic_case, last, out=out)
    assert training.run_training(diagnostic_case, last, out=out, smoke=True) == best
    child = copy.deepcopy(diagnostic_case)
    child.update(run_id="not-an-experiment", stage="fine_tune", parent_best=str(best))
    child["optimizer"]["lr"] = 1e-4
    with pytest.raises(ValueError, match="diagnostic"):
        training.run_training(child)


@pytest.mark.parametrize(
    "options",
    [
        {"smoke": True, "profile_steps": 100},
        {"profile_steps": 0},
        {"profile_steps": True},
    ],
)
def test_diagnostic_options_rejected_before_io(
    config: dict[str, Any], options: dict[str, Any]
) -> None:
    import ai.train as training

    with pytest.raises(ValueError, match="profile|smoke"):
        training.run_training(config, **options)


def assert_stage_policy(
    model: nn.Module, architecture: str, stage: str, groups: list[dict[str, Any]]
) -> None:
    head_prefix = "classifier.1." if architecture == "efficientnet_b0" else "fc."
    tail_prefixes = (
        ("features.7.", "features.8.")
        if architecture == "efficientnet_b0"
        else ("layer4.",)
    )
    head = {id(p) for n, p in model.named_parameters() if n.startswith(head_prefix)}
    tail = {id(p) for n, p in model.named_parameters() if n.startswith(tail_prefixes)}
    expected = head | (tail if stage == "fine_tune" else set())
    assert {id(p) for p in model.parameters() if p.requires_grad} == expected
    assert [g["name"] for g in groups] == (
        ["head", "backbone"] if stage == "fine_tune" else ["head"]
    )
    assert [g["lr"] for g in groups] == (
        [1e-4, 1e-5] if stage == "fine_tune" else [1e-3]
    )
    assert {id(p) for p in groups[0]["params"]} == head
    if stage == "fine_tune":
        assert {id(p) for p in groups[1]["params"]} == tail
    grouped = [p for group in groups for p in group["params"]]
    assert len(grouped) == len({id(p) for p in grouped}) == len(expected)
    assert {id(p) for p in grouped} == expected
    assert all(set(g) == {"name", "params", "lr"} and g["params"] for g in groups)
    assert all(isinstance(g["params"], list) for g in groups)
    assert all(isinstance(p, nn.Parameter) for p in grouped)
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert not model.training
    for name, module in model.named_modules():
        expected_training = name == head_prefix[:-1] or name.startswith(head_prefix)
        if architecture == "efficientnet_b0" and name in ("classifier", "classifier.0"):
            expected_training = True
        if stage == "fine_tune":
            expected_training |= any(
                name == prefix[:-1] or name.startswith(prefix)
                for prefix in tail_prefixes
            )
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            expected_training = False
        assert module.training == expected_training, name


@pytest.mark.parametrize("stage", ["frozen", "fine_tune"])
def test_configure_stage_groups_modes_and_cpu_update(
    stage_model: tuple[str, nn.Module], stage: str
) -> None:
    from ai.train import configure_stage

    architecture, model = stage_model
    groups = configure_stage(model, architecture, stage)
    assert_stage_policy(model, architecture, stage, groups)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    bn_before = {
        n: b.clone()
        for n, b in model.named_buffers()
        if n.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    inputs = torch.randn(2, 3, 32, 32, device="cpu")
    loss = nn.CrossEntropyLoss()(model(inputs), torch.tensor([0, 2]))
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in model.parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None
    for group in groups:
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in group["params"]
        )
        assert any(torch.count_nonzero(p.grad).item() for p in group["params"])
    optimizer.step()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert torch.equal(parameter, before[name]), name
    names = {id(p): n for n, p in model.named_parameters()}
    for group in groups:
        assert any(not torch.equal(p, before[names[id(p)]]) for p in group["params"])
    buffers = dict(model.named_buffers())
    assert bn_before
    assert all(torch.equal(buffers[n], value) for n, value in bn_before.items())


def test_configure_stage_transitions_clear_gradients_without_changing_state(
    stage_model: tuple[str, nn.Module],
) -> None:
    from ai.train import configure_stage

    architecture, model = stage_model
    identities = {n: id(p) for n, p in model.named_parameters()}
    before = {n: value.clone() for n, value in model.state_dict().items()}
    for stage in ("frozen", "frozen", "fine_tune", "fine_tune", "frozen"):
        # M4.3 must reapply this policy after its own model.train() call.
        model.train()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
            parameter.grad = torch.ones_like(parameter)
        groups = configure_stage(model, architecture, stage)
        assert_stage_policy(model, architecture, stage, groups)
        assert all(p.grad is None for p in model.parameters())
        assert {n: id(p) for n, p in model.named_parameters()} == identities
        assert all(
            torch.equal(value, before[n]) for n, value in model.state_dict().items()
        )


@pytest.mark.parametrize(
    "damage,field",
    [
        ("architecture", "architecture"),
        ("stage", "stage"),
        ("family", "model"),
        ("head", "model"),
        ("head_width", "model"),
        ("head_empty", "model"),
        ("tail", "model"),
        ("empty_tail", "model"),
        ("parameterless_tail", "model"),
        ("tail_required_layer", "model"),
        ("tail_channels", "model"),
        ("overlap", "model"),
    ],
)
def test_configure_stage_rejects_invalid_without_mutation(
    stage_model: tuple[str, nn.Module], damage: str, field: str
) -> None:
    from ai.train import configure_stage

    architecture, model = stage_model
    stage = "fine_tune"
    if damage == "architecture":
        architecture = "unknown"
    elif damage == "stage":
        stage = "unknown"
    elif damage == "family":
        architecture = (
            "resnet50" if architecture == "efficientnet_b0" else "efficientnet_b0"
        )
    elif architecture == "efficientnet_b0":
        classifier = cast(nn.Sequential, model.classifier)
        features = cast(nn.Sequential, model.features)
        if damage == "head":
            classifier[1] = nn.Identity()
        elif damage == "head_width":
            classifier[1] = nn.Linear(1, 3)
        elif damage == "head_empty":
            cast(nn.Linear, classifier[1]).out_features = 0
        elif damage == "tail":
            model.features = nn.Identity()
        elif damage == "empty_tail":
            features[7] = nn.Sequential()
        elif damage == "parameterless_tail":
            features[7] = nn.Sequential(nn.Identity())
        elif damage == "tail_required_layer":
            cast(nn.Sequential, features[8])[0] = nn.Identity()
        elif damage == "tail_channels":
            cast(nn.Sequential, features[8])[0] = nn.Conv2d(320, 640, 1)
        else:
            features[7].register_parameter(
                "shared", classifier[1].get_parameter("weight")
            )
    elif damage == "head":
        model.fc = nn.Identity()
    elif damage == "head_width":
        model.fc = nn.Linear(1, 3)
    elif damage == "head_empty":
        cast(nn.Linear, model.fc).out_features = 0
    elif damage == "tail":
        model.layer4 = nn.Identity()
    elif damage == "empty_tail":
        model.layer4 = nn.Sequential()
    elif damage == "parameterless_tail":
        model.layer4 = nn.Sequential(nn.Identity(), nn.Identity(), nn.Identity())
    elif damage == "tail_required_layer":
        cast(nn.Sequential, model.layer4)[0] = nn.Identity()
    elif damage == "tail_channels":
        cast(nn.Sequential, model.layer4)[0].conv1 = nn.Conv2d(512, 512, 1)
    else:
        cast(nn.Sequential, model.layer4).register_parameter(
            "shared", cast(nn.Linear, model.fc).get_parameter("weight")
        )
    for index, parameter in enumerate(model.parameters()):
        parameter.requires_grad_(index % 2 == 0)
        if index < 2:
            parameter.grad = torch.ones_like(parameter)
    for index, module in enumerate(model.modules()):
        module.training = index % 2 == 0
    before = {n: value.clone() for n, value in model.state_dict().items()}
    modes = {n: m.training for n, m in model.named_modules()}
    parameters = {
        n: (id(p), p.requires_grad, p.grad) for n, p in model.named_parameters()
    }
    gradients = {
        n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
    }
    with pytest.raises(ValueError, match=field):
        configure_stage(model, architecture, stage)
    assert {n: m.training for n, m in model.named_modules()} == modes
    for name, parameter in model.named_parameters():
        identity, requires_grad, gradient = parameters[name]
        assert id(parameter) == identity
        assert parameter.requires_grad == requires_grad
        assert parameter.grad is gradient
        if gradient is not None:
            assert torch.equal(gradient, gradients[name])
    assert all(torch.equal(value, before[n]) for n, value in model.state_dict().items())


@pytest.mark.parametrize("kind", ["none", "object", "linear"])
def test_configure_stage_requires_torchvision_model(kind: str) -> None:
    from ai.train import configure_stage

    with torch.random.fork_rng(devices=[]):
        model: Any = {"none": None, "object": object(), "linear": nn.Linear(2, 3)}[kind]
    with pytest.raises(ValueError, match="model"):
        configure_stage(model, "efficientnet_b0", "frozen")


def test_accumulation_tail_matches_reference(training_case: TrainingCase) -> None:
    import ai.train as training

    assert hasattr(training, "train_epoch"), "M4.3 train_epoch is missing"
    reference = copy.deepcopy(training_case.model)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    batches = list(training_case.loader)
    features = torch.cat([x for x, _, _ in batches])
    targets = torch.cat([y for _, y, _ in batches])
    total_loss = 0.0
    for section in (slice(0, 4), slice(4, 5)):
        reference_optimizer.zero_grad()
        loss = nn.functional.cross_entropy(
            reference(features[section]), targets[section]
        )
        total_loss += loss.item() * len(targets[section])
        torch.autograd.backward(loss)
        reference_optimizer.step()
    metrics = training.train_epoch(
        training_case.model,
        training_case.loader,
        training_case.optimizer,
        device=torch.device("cpu"),
        accumulation=2,
        amp=False,
    )
    assert set(metrics) == {"loss", "optimizer_steps", "skipped_steps"}
    assert all(type(value) is float for value in metrics.values())
    assert metrics["optimizer_steps"] == 2.0
    assert metrics["skipped_steps"] == 0.0
    assert metrics["loss"] == pytest.approx(total_loss / 5, abs=1e-7)
    assert torch.allclose(
        training_case.model.weight, training_case.expected_weight, atol=1e-6
    )
    torch.testing.assert_close(
        training_case.model.weight, reference.weight, atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        training_case.model.bias, reference.bias, atol=1e-6, rtol=0
    )
    assert all(
        torch.isfinite(p).all() and p.grad is None
        for p in training_case.model.parameters()
    )


class OneShotBatches:
    def __init__(self, batches: list[tuple[torch.Tensor, torch.Tensor, Any]]) -> None:
        self.batches = iter(batches)
        self.iterations = 0

    def __iter__(self) -> "OneShotBatches":
        self.iterations += 1
        assert self.iterations == 1, "loader was rewound"
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor, Any]:
        return next(self.batches)

    def __len__(self) -> int:
        raise AssertionError("streaming epoch must not ask len(loader)")


@pytest.mark.parametrize(
    "sizes,accumulation", [([3, 1, 1], 2), ([2, 2, 1], 1), ([2, 2, 1], 10)]
)
def test_epoch_variable_batches_and_weighted_loss(
    training_case: TrainingCase, sizes: list[int], accumulation: int
) -> None:
    from ai.train import train_epoch

    initial_batches = list(training_case.loader)
    features = torch.cat([x for x, _, _ in initial_batches])
    targets = torch.cat([y for _, y, _ in initial_batches])
    batches = list(
        zip(
            torch.split(features, sizes),
            torch.split(targets, sizes),
            [None] * len(sizes),
            strict=True,
        )
    )
    reference = copy.deepcopy(training_case.model)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    loss_sum = 0.0
    for start in range(0, len(batches), accumulation):
        window = batches[start : start + accumulation]
        inputs = torch.cat([x for x, _, _ in window])
        labels = torch.cat([y for _, y, _ in window])
        loss = nn.functional.cross_entropy(reference(inputs), labels)
        loss_sum += loss.item() * len(labels)
        torch.autograd.backward(loss)
        reference_optimizer.step()
        reference_optimizer.zero_grad()
    loader = OneShotBatches(batches)
    metrics = train_epoch(
        training_case.model,
        loader,
        training_case.optimizer,
        device=torch.device("cpu"),
        accumulation=accumulation,
        amp=False,
    )
    torch.testing.assert_close(
        training_case.model.state_dict(), reference.state_dict(), rtol=0, atol=1e-6
    )
    assert metrics["loss"] == pytest.approx(loss_sum / 5, abs=1e-7)
    assert metrics["optimizer_steps"] == float(
        (len(batches) + accumulation - 1) // accumulation
    )
    assert metrics["skipped_steps"] == 0.0
    assert loader.iterations == 1


@pytest.mark.parametrize(
    "options,field",
    [
        ({"accumulation": 0}, "accumulation"),
        ({"accumulation": -1}, "accumulation"),
        ({"accumulation": True}, "accumulation"),
        ({"accumulation": 1.5}, "accumulation"),
        ({"amp": 1}, "amp"),
        ({"amp": True}, "CUDA"),
        ({"device": "cpu"}, "device"),
        ({"device": torch.device("meta")}, "device"),
        ({"scaler": object()}, "scaler"),
    ],
)
def test_epoch_invalid_arguments_do_not_mutate(
    training_case: TrainingCase, options: dict[str, Any], field: str
) -> None:
    from ai.train import train_epoch

    model = training_case.model
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = {n: (p.detach().clone(), p.grad) for n, p in model.named_parameters()}
    arguments: dict[str, Any] = {
        "device": torch.device("cpu"),
        "accumulation": 2,
        "amp": False,
        **options,
    }
    with pytest.raises(ValueError, match=field):
        train_epoch(model, training_case.loader, training_case.optimizer, **arguments)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name][0])
        assert parameter.grad is before[name][1]
    assert not training_case.optimizer.state


def test_epoch_scaler_enabled_state_validation(training_case: TrainingCase) -> None:
    from ai.train import train_epoch

    enabled = torch.amp.GradScaler("cpu")
    disabled = torch.amp.GradScaler("cuda", enabled=False)
    for amp, device, scaler in (
        (False, torch.device("cpu"), enabled),
        (True, torch.device("cuda"), disabled),
    ):
        with pytest.raises(ValueError, match="scaler"):
            train_epoch(
                training_case.model,
                training_case.loader,
                training_case.optimizer,
                device=device,
                accumulation=2,
                amp=amp,
                scaler=scaler,
            )
    metrics = train_epoch(
        training_case.model,
        training_case.loader,
        training_case.optimizer,
        device=torch.device("cpu"),
        accumulation=2,
        amp=False,
        scaler=disabled,
    )
    assert metrics["optimizer_steps"] == 2.0


@pytest.mark.parametrize(
    "kind",
    [
        "loader",
        "batch",
        "mismatch",
        "target_shape",
        "negative",
        "range",
        "float",
        "bool",
        "input_object",
        "target_object",
    ],
)
def test_epoch_invalid_batch_clears_gradients(
    training_case: TrainingCase, kind: str
) -> None:
    from ai.train import train_epoch

    inputs, targets, ids = next(iter(training_case.loader))
    if kind == "batch":
        inputs, targets = inputs[:0], targets[:0]
    elif kind == "mismatch":
        targets = targets[:1]
    elif kind == "target_shape":
        targets = targets[:, None]
    elif kind == "negative":
        targets[0] = -100
    elif kind == "range":
        targets[0] = 2
    elif kind == "float":
        targets = targets.float()
    elif kind == "bool":
        targets = targets.bool()
    elif kind == "input_object":
        inputs = cast(Any, [1, 2])
    elif kind == "target_object":
        targets = cast(Any, [0, 1])
    for parameter in training_case.model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = copy.deepcopy(training_case.model.state_dict())
    loader = [] if kind == "loader" else [(inputs, targets, ids)]
    with pytest.raises(ValueError):
        train_epoch(
            training_case.model,
            loader,
            training_case.optimizer,
            device=torch.device("cpu"),
            accumulation=2,
            amp=False,
        )
    torch.testing.assert_close(training_case.model.state_dict(), before, rtol=0, atol=0)
    assert not training_case.optimizer.state
    assert all(p.grad is None for p in training_case.model.parameters())


@pytest.mark.parametrize("shape", [(2,), (1, 2), (2, 0), (2, 2, 1)])
def test_epoch_rejects_nonclassification_logits(
    training_case: TrainingCase, shape: tuple[int, ...]
) -> None:
    from ai.train import train_epoch

    class BadLogits(nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.zeros(shape)

    with pytest.raises(ValueError, match="logits"):
        train_epoch(
            BadLogits(),
            training_case.loader,
            training_case.optimizer,
            device=torch.device("cpu"),
            accumulation=2,
            amp=False,
        )
    assert all(p.grad is None for p in training_case.model.parameters())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_epoch_nonfinite_loss_fails_before_backward(
    training_case: TrainingCase, value: float
) -> None:
    from ai.train import train_epoch

    inputs, targets, ids = next(iter(training_case.loader))
    inputs[0, 0] = value
    before = copy.deepcopy(training_case.model.state_dict())
    with pytest.raises(FloatingPointError, match="loss"):
        train_epoch(
            training_case.model,
            [(inputs, targets, ids)],
            training_case.optimizer,
            device=torch.device("cpu"),
            accumulation=1,
            amp=False,
        )
    torch.testing.assert_close(training_case.model.state_dict(), before, rtol=0, atol=0)
    assert not training_case.optimizer.state
    assert all(p.grad is None for p in training_case.model.parameters())


def test_epoch_finite_loss_infinite_backward_aborts_before_step() -> None:
    from ai.train import train_epoch

    class InfiniteGradient(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            value = self.weight.sqrt()
            return torch.stack((value, -value)).expand(len(inputs), 2)

    model = InfiniteGradient()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    inputs, targets = torch.ones(2, 1), torch.zeros(2, dtype=torch.long)
    assert torch.isfinite(nn.functional.cross_entropy(model(inputs), targets))
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(FloatingPointError, match="gradients"):
        train_epoch(
            model,
            [(inputs, targets, None)],
            optimizer,
            device=torch.device("cpu"),
            accumulation=2,
            amp=False,
        )
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    assert not optimizer.state
    assert model.weight.grad is None


@pytest.mark.parametrize("failure", ["fetch", "forward", "backward", "backward_error"])
def test_epoch_error_preserves_completed_window_without_flushing_tail(
    training_case: TrainingCase, failure: str
) -> None:
    from ai.train import train_epoch

    batches = list(training_case.loader)
    reference = copy.deepcopy(training_case.model)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1, momentum=0.9)
    torch.autograd.backward(
        nn.functional.cross_entropy(
            reference(torch.cat([x for x, _, _ in batches[:2]])),
            torch.cat([y for _, y, _ in batches[:2]]),
        )
    )
    reference_optimizer.step()
    reference_optimizer.zero_grad()
    optimizer = torch.optim.SGD(training_case.model.parameters(), lr=0.1, momentum=0.9)

    def broken_loader() -> Iterator[tuple[torch.Tensor, torch.Tensor, Any]]:
        yield from batches
        if failure == "fetch":
            raise RuntimeError("fetch failure")
        if failure == "forward":
            yield torch.ones(1, 3), torch.zeros(1, dtype=torch.long), None
        else:

            def fail_backward(gradient: torch.Tensor) -> torch.Tensor:
                raise RuntimeError("backward failure")

            training_case.model.weight.register_hook(  # type: ignore[no-untyped-call]
                fail_backward
                if failure == "backward_error"
                else lambda gradient: torch.full_like(gradient, float("inf"))
            )
            yield batches[1]

    with pytest.raises((RuntimeError, FloatingPointError)):
        train_epoch(
            training_case.model,
            broken_loader(),
            optimizer,
            device=torch.device("cpu"),
            accumulation=2,
            amp=False,
        )
    torch.testing.assert_close(
        training_case.model.state_dict(), reference.state_dict(), rtol=0, atol=1e-7
    )
    torch.testing.assert_close(
        optimizer.state_dict(), reference_optimizer.state_dict(), rtol=0, atol=1e-7
    )
    assert all(p.grad is None for p in training_case.model.parameters())


def test_epoch_fetch_error_drops_pending_window(training_case: TrainingCase) -> None:
    from ai.train import train_epoch

    batches = list(training_case.loader)
    reference = copy.deepcopy(training_case.model)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    torch.autograd.backward(
        nn.functional.cross_entropy(
            reference(torch.cat([x for x, _, _ in batches[:2]])),
            torch.cat([y for _, y, _ in batches[:2]]),
        )
    )
    reference_optimizer.step()

    def broken_loader() -> Iterator[tuple[torch.Tensor, torch.Tensor, Any]]:
        yield from batches
        raise RuntimeError("fetch tail failure")

    with pytest.raises(RuntimeError, match="fetch tail"):
        train_epoch(
            training_case.model,
            broken_loader(),
            training_case.optimizer,
            device=torch.device("cpu"),
            accumulation=2,
            amp=False,
        )
    torch.testing.assert_close(
        training_case.model.state_dict(), reference.state_dict(), rtol=0, atol=1e-7
    )
    assert all(p.grad is None for p in training_case.model.parameters())


@pytest.mark.parametrize("stage", ["frozen", "fine_tune"])
def test_epoch_preserves_stage_policy_and_optimizer_across_epochs(
    stage_model: tuple[str, nn.Module], stage: str
) -> None:
    from ai.train import configure_stage, train_epoch

    architecture, model = stage_model
    groups = configure_stage(model, architecture, stage)
    optimizer = torch.optim.AdamW([dict(group) for group in groups], weight_decay=1e-4)
    group_identities = [id(group) for group in optimizer.param_groups]
    parameter_identities = {n: id(p) for n, p in model.named_parameters()}
    modes = {n: m.training for n, m in model.named_modules()}
    frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    generator = torch.Generator().manual_seed(113)
    inputs = torch.randn(5, 3, 32, 32, generator=generator)
    targets = torch.tensor([0, 1, 2, 0, 2])
    batches = [
        (inputs[start:stop], targets[start:stop], None)
        for start, stop in ((0, 2), (2, 4), (4, 5))
    ]
    original_step = optimizer.step
    observed_steps = 0

    def checked_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal observed_steps
        assert_stage_policy(model, architecture, stage, groups)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        for group in optimizer.param_groups:
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in group["params"]
            )
            assert any(torch.count_nonzero(p.grad).item() for p in group["params"])
        observed_steps += 1
        return original_step(*args, **kwargs)

    optimizer.step = checked_step  # type: ignore[method-assign]
    for epoch in range(2):
        if epoch:
            model.eval()
            configure_stage(model, architecture, stage)
        before = {
            id(p): p.clone()
            for group in optimizer.param_groups
            for p in group["params"]
        }
        metrics = train_epoch(
            model,
            OneShotBatches(batches),
            optimizer,
            device=torch.device("cpu"),
            accumulation=2,
            amp=False,
        )
        assert metrics["optimizer_steps"] == 2.0 and metrics["skipped_steps"] == 0.0
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert {n: m.training for n, m in model.named_modules()} == modes
        assert {n: id(p) for n, p in model.named_parameters()} == parameter_identities
        assert [id(group) for group in optimizer.param_groups] == group_identities
        assert_stage_policy(model, architecture, stage, groups)
        assert all(
            torch.equal(dict(model.named_parameters())[n], value)
            for n, value in frozen.items()
        )
        assert all(
            torch.equal(dict(model.named_buffers())[n], value)
            for n, value in buffers.items()
        )
        assert all(p.grad is None for p in model.parameters())
        for group in optimizer.param_groups:
            assert any(not torch.equal(p, before[id(p)]) for p in group["params"])
            assert all(
                optimizer.state[p]["step"].item() == (epoch + 1) * 2
                for p in group["params"]
            )
    assert observed_steps == 4


def validation_metrics(
    model: nn.Module,
    batches: Any,
    *,
    class_count: int = 3,
    device: torch.device | None = None,
) -> dict[str, float]:
    import ai.train as training

    assert hasattr(training, "_validate_epoch"), "M4.4 validation is missing"
    return training._validate_epoch(
        model, batches, device=device or torch.device("cpu"), class_count=class_count
    )


def test_validation_weighted_loss_and_fixed_class_macro_f1() -> None:
    probabilities = torch.tensor(
        [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1]],
        dtype=torch.float64,
    )
    logits = probabilities.log()
    loader = OneShotBatches(
        [
            (logits[:2], torch.tensor([0, 0]), None),
            (logits[2:], torch.tensor([1]), None),
        ]
    )
    result = validation_metrics(nn.Identity(), loader)
    assert result["loss"] == pytest.approx(-(2 * math.log(0.8) + math.log(0.2)) / 3)
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(4 / 9)
    assert loader.iterations == 1
    assert all(type(value) is float for value in result.values())


def test_validation_uses_eval_and_inference_without_changing_weights_or_gradients() -> (
    None
):
    class ObservedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.BatchNorm1d(3)
            self.dropout = nn.Dropout(0.8)
            self.weight = nn.Parameter(torch.ones(3))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            assert torch.is_inference_mode_enabled()
            assert not torch.is_grad_enabled()
            assert all(not module.training for module in self.modules())
            return cast(torch.Tensor, self.dropout(self.norm(inputs)) * self.weight)

    model = ObservedModel()
    model.weight.grad = torch.full((3,), 0.25)
    before = copy.deepcopy(model.state_dict())
    gradients = model.weight.grad.clone()
    batches = [(torch.tensor([[2.0, 0.0, -1.0]]), torch.tensor([0]), None)]
    assert validation_metrics(model, batches)["macro_f1"] == pytest.approx(1 / 3)
    assert validation_metrics(model, batches)["accuracy"] == 1.0
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    torch.testing.assert_close(model.weight.grad, gradients, rtol=0, atol=0)


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_validation_independent_four_class_oracle(batch_size: int) -> None:
    probabilities = torch.tensor(
        [
            [1 / 2, 1 / 4, 1 / 8, 1 / 8],
            [1 / 4, 1 / 2, 1 / 8, 1 / 8],
            [1 / 8, 1 / 2, 1 / 4, 1 / 8],
            [1 / 8, 1 / 4, 1 / 2, 1 / 8],
            [1 / 16, 1 / 16, 3 / 4, 1 / 8],
        ],
        dtype=torch.float64,
    )
    targets = torch.tensor([0, 0, 1, 1, 2])
    batches = [
        (
            probabilities[start : start + batch_size].log(),
            targets[start : start + batch_size],
            None,
        )
        for start in range(0, 5, batch_size)
    ]
    metrics = validation_metrics(nn.Identity(), OneShotBatches(batches), class_count=4)
    assert metrics["macro_f1"] == pytest.approx(11 / 24, abs=1e-12)
    assert metrics["accuracy"] == pytest.approx(3 / 5, abs=1e-12)
    assert metrics["loss"] == pytest.approx(
        (8 * math.log(2) - math.log(3)) / 5, abs=1e-12
    )


def test_validation_half_precision_reduction_does_not_overflow() -> None:
    logits = torch.tensor([[0.0, -100.0]], dtype=torch.float16).repeat(800, 1)
    metrics = validation_metrics(
        nn.Identity(),
        [(logits, torch.ones(800, dtype=torch.long), None)],
        class_count=2,
    )
    assert metrics == {"loss": 100.0, "accuracy": 0.0, "macro_f1": 0.0}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_validation_rejects_nonfinite_logits(value: float) -> None:
    logits = torch.tensor([[1.0, value, 0.0]])
    with pytest.raises(FloatingPointError, match="non-finite"):
        validation_metrics(nn.Identity(), [(logits, torch.tensor([0]), None)])


@pytest.mark.parametrize(
    "inputs,targets,match",
    [
        (torch.ones(1, 3), torch.tensor([-1]), "targets"),
        (torch.ones(1, 3), torch.tensor([3]), "targets"),
        (torch.ones(1, 3), torch.tensor([0.0]), "integer"),
        (torch.ones(1, 3), torch.tensor([False]), "integer"),
        (torch.ones(1, 3), torch.tensor([[0]]), "batch"),
        (torch.ones(2, 3), torch.tensor([0]), "batch"),
        (torch.ones(0, 3), torch.tensor([], dtype=torch.long), "batch"),
        (torch.ones(1, 2), torch.tensor([0]), "logits"),
        (torch.ones(1, 3, 1), torch.tensor([0]), "logits"),
        (torch.ones(1, 3, dtype=torch.long), torch.tensor([0]), "logits"),
    ],
)
def test_validation_rejects_invalid_batches(
    inputs: torch.Tensor, targets: torch.Tensor, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        validation_metrics(nn.Identity(), [(inputs, targets, None)])


@pytest.mark.parametrize("class_count", [0, -1, True, 1.5])
def test_validation_rejects_invalid_class_count(class_count: Any) -> None:
    with pytest.raises(ValueError, match="class_count"):
        validation_metrics(nn.Identity(), [], class_count=class_count)


def test_validation_rejects_empty_loader_and_unsupported_device() -> None:
    with pytest.raises(ValueError, match="empty"):
        validation_metrics(nn.Identity(), [])
    with pytest.raises(ValueError, match="device"):
        validation_metrics(nn.Identity(), [], device=torch.device("meta"))


class ValidationSequence:
    """Known probabilities per epoch, independent of optimizer updates."""

    def __init__(self, probabilities: list[list[list[float]]]) -> None:
        self.probabilities = probabilities
        self.epochs = 0

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, Any]]:
        values = self.probabilities[self.epochs]
        self.epochs += 1
        yield (
            torch.tensor(values, dtype=torch.float64).log(),
            torch.tensor([0, 0, 1]),
            None,
        )


def epoch_records(
    config: dict[str, Any],
    sequence: ValidationSequence,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[dict[str, Any]]:
    import ai.train as training

    assert hasattr(training, "_training_epochs"), "M4.4 orchestration is missing"
    config["precision"] = "fp32"
    config["class_ids"] = ["a", "b", "c"]
    with torch.random.fork_rng(devices=[]):
        model = nn.Linear(3, 3, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.eye(3))
        model.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    def stage(*args: Any) -> list[dict[str, Any]]:
        model.train()
        return []

    def train(*args: Any, **kwargs: Any) -> dict[str, float]:
        assert args[0] is model and args[2] is optimizer
        assert kwargs["scaler"] is scaler
        assert model.training
        return {
            "loss": float(100 - sequence.epochs),
            "optimizer_steps": 1.0,
            "skipped_steps": 0.0,
        }

    monkeypatch.setattr(training, "configure_stage", stage)
    monkeypatch.setattr(training, "train_epoch", train)
    return training._training_epochs(
        model,
        [],
        sequence,
        optimizer,
        config=config,
        device=torch.device("cpu"),
        scaler=scaler,
    )


def test_selection_prioritizes_f1_then_loss_and_preserves_exact_tie(
    config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    a = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1]]
    b = [[0.4, 0.3, 0.3], [0.4, 0.3, 0.3], [0.3, 0.4, 0.3]]
    c = [[0.7, 0.2, 0.1], [0.7, 0.2, 0.1], [0.1, 0.7, 0.2]]
    d = [[0.999, 0.0005, 0.0005], [0.49, 0.50, 0.01], [0.0005, 0.999, 0.0005]]
    sequence = ValidationSequence([a, b, c, c, d])
    records = list(epoch_records(config, sequence, monkeypatch))
    assert [r["epoch"] for r in records] == [1, 2, 3, 4, 5]
    assert [r["is_best"] for r in records] == [True, True, True, False, False]
    assert [r["bad_epochs"] for r in records] == [0, 0, 0, 1, 2]
    assert records[1]["val"]["loss"] > records[0]["val"]["loss"]
    assert records[-1]["val"]["loss"] < records[2]["val"]["loss"]
    assert records[0]["val"]["macro_f1"] == pytest.approx(4 / 9)
    assert records[1]["val"]["macro_f1"] == pytest.approx(2 / 3)
    assert json.loads(json.dumps(records, allow_nan=False)) == records


def test_patience_five_resets_on_improvement_and_stops_at_fifth_bad_epoch(
    config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    a = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1]]
    b = [[0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]
    sequence = ValidationSequence([a] * 5 + [b] * 9)
    config["epochs"] = 14
    iterator = epoch_records(config, sequence, monkeypatch)
    first = next(iterator)
    assert first["epoch"] == 1 and sequence.epochs == 1
    records = [first, *iterator]
    assert len(records) == sequence.epochs == 11
    assert [r["epoch"] for r in records if r["is_best"]] == [1, 6]
    assert [r["bad_epochs"] for r in records] == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5]


def test_first_zero_f1_is_best_and_failed_validation_does_not_yield_epoch(
    config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    wrong = [[0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1]]
    invalid = [[float("nan"), 0.2, 0.8]] * 3
    sequence = ValidationSequence([wrong, invalid])
    iterator = epoch_records(config, sequence, monkeypatch)
    record = next(iterator)
    assert record["is_best"] and record["bad_epochs"] == 0
    assert record["val"]["macro_f1"] == 0.0
    with pytest.raises(FloatingPointError, match="non-finite"):
        next(iterator)
    assert sequence.epochs == 2
    with pytest.raises(StopIteration):
        next(iterator)


@pytest.mark.parametrize("stage", ["frozen", "fine_tune"])
def test_training_validation_reapplies_real_stage_without_resetting_optimizer(
    stage_model: tuple[str, nn.Module], stage: str, config: dict[str, Any]
) -> None:
    import ai.train as training

    assert hasattr(training, "_training_epochs"), "M4.4 orchestration is missing"
    architecture, model = stage_model
    config.update(architecture=architecture, stage=stage, precision="fp32", epochs=2)
    config["class_ids"] = ["a", "b", "c"]
    config["parent_best"] = "synthetic-parent.pth" if stage == "fine_tune" else None
    groups = training.configure_stage(model, architecture, stage)
    optimizer = torch.optim.AdamW([dict(group) for group in groups], weight_decay=1e-4)
    identities = [id(g) for g in optimizer.param_groups]
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    generator = torch.Generator().manual_seed(445)
    batches = [
        (torch.randn(2, 3, 32, 32, generator=generator), torch.tensor([0, 1]), None)
    ]
    observed_modes: list[bool] = []

    def observe(module: nn.Module, args: Any) -> None:
        observed_modes.append(torch.is_grad_enabled())
        if torch.is_grad_enabled():
            assert_stage_policy(model, architecture, stage, groups)
        else:
            assert torch.is_inference_mode_enabled()
            assert all(not m.training for m in model.modules())

    handle = model.register_forward_pre_hook(observe)
    try:
        records = list(
            training._training_epochs(
                model,
                batches,
                batches,
                optimizer,
                config=config,
                device=torch.device("cpu"),
            )
        )
    finally:
        handle.remove()
    assert len(records) == 2
    assert observed_modes == [True, False, True, False]
    assert [id(g) for g in optimizer.param_groups] == identities
    assert all(
        optimizer.state[p]["step"].item() == 2 for g in groups for p in g["params"]
    )
    assert all(
        torch.equal(dict(model.named_buffers())[n], b) for n, b in buffers.items()
    )
    assert all(r["train"]["optimizer_steps"] == 1.0 for r in records)


def test_training_loaders_use_only_train_val_and_locked_class_order(
    sealed_images: tuple[Path, Path],
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ai.train as training
    from ai.data import ManifestDataset

    assert hasattr(training, "_training_loaders"), "M4.4 loaders are missing"
    root, _ = sealed_images
    config["dataset_dir"] = str(root)
    config["class_ids"] = [
        row["class_id"]
        for row in json.loads((root / "label_map.json").read_text())["classes"]
    ]
    splits: list[str] = []

    def dataset(*args: Any, **kwargs: Any) -> ManifestDataset:
        splits.append(args[1])
        assert args[1] in ("train", "val")
        return ManifestDataset(*args, **kwargs)

    monkeypatch.setattr(training, "ManifestDataset", dataset)
    before_rng = torch.get_rng_state().clone()
    train, val = training._training_loaders(config)
    assert splits == ["train", "val"]
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert train.num_workers == val.num_workers == 0
    assert not train.drop_last and not val.drop_last
    assert len(list(train)) == len(list(val)) == 1
    assert train.generator is not val.generator
    config["class_ids"].reverse()
    with pytest.raises(ValueError, match="class_ids"):
        training._training_loaders(config)
    splits.clear()
    config["train_split"] = "test"
    with pytest.raises(ValueError, match="train_split"):
        training._training_loaders(config)
    assert splits == []


class ResumeSamples(Dataset[tuple[torch.Tensor, int, str]]):
    """Non-symmetric samples exercise shuffle and three independent RNG streams."""

    def __init__(self, training: bool) -> None:
        self.training = training

    def __len__(self) -> int:
        return 7

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        value = torch.tensor([index / 3 - 1, (index % 3) - 0.4])
        if self.training:
            value += torch.rand(2) + random.random() + float(np.random.random())
        return value, index % 8, str(index)


@pytest.fixture
def resume_case(
    config: dict[str, Any],
    sealed_images: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    import ai.train as training

    root, _ = sealed_images
    weights = tmp_path / "tiny-pretrained.pth"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        torch.save(
            nn.Sequential(nn.Dropout(0.4), nn.Linear(2, 8)).state_dict(), weights
        )
    config.update(
        dataset_dir=str(root),
        seal_sha256=hashlib.sha256((root / "seal.json").read_bytes()).hexdigest(),
        split_hash=hashlib.sha256((root / "split.json").read_bytes()).hexdigest(),
        pretrained_path=str(weights),
        pretrained_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        class_ids=[
            row["class_id"]
            for row in json.loads((root / "label_map.json").read_text())["classes"]
        ],
        precision="fp32",
        batch=2,
        epochs=4,
        patience=5,
    )

    def model_factory(
        architecture: str, class_count: int, pretrained_path: Path | None = None
    ) -> nn.Module:
        assert architecture == "efficientnet_b0" and class_count == 8
        model = nn.Sequential(nn.Dropout(0.4), nn.Linear(2, 8))
        if pretrained_path is not None:
            model.load_state_dict(torch.load(pretrained_path, weights_only=True))
        return model

    def stage(model: nn.Module, architecture: str, stage: str) -> list[dict[str, Any]]:
        model.train()
        return [{"params": list(model.parameters()), "name": "head", "lr": 1e-3}]

    def loaders(config: dict[str, Any]) -> tuple[DataLoader[Any], DataLoader[Any]]:
        return tuple(  # type: ignore[return-value]
            DataLoader(
                ResumeSamples(is_train),
                batch_size=config["batch"],
                shuffle=is_train,
                generator=torch.Generator().manual_seed(config["seed"]),
                num_workers=0,
            )
            for is_train in (True, False)
        )

    monkeypatch.setattr(training, "build_model", model_factory, raising=False)
    monkeypatch.setattr(training, "configure_stage", stage)
    monkeypatch.setattr(training, "_training_loaders", loaders)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.chdir(tmp_path)
    return config


def run_case(config: dict[str, Any], resume: Path | None = None) -> Path:
    import ai.train as training

    assert hasattr(training, "run_training"), "M4.5 public runner is missing"
    return training.run_training(config, resume=resume)


def assert_checkpoint_equal(actual: Any, expected: Any, atol: float = 1e-6) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=atol)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_checkpoint_equal(actual[key], expected[key], atol)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_checkpoint_equal(left, right, atol)
    else:
        assert actual == expected


def test_resume_matches_epoch_boundary(
    resume_case: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ai.train as training

    continuous = tmp_path / "continuous"
    interrupted = tmp_path / "interrupted"
    continuous.mkdir()
    interrupted.mkdir()
    monkeypatch.chdir(continuous)
    best = run_case(resume_case)
    reference = torch.load(best.with_name("last.pth"), weights_only=True)
    monkeypatch.chdir(interrupted)
    real_train = training.train_epoch
    calls = 0

    def interrupt(*args: Any, **kwargs: Any) -> dict[str, float]:
        nonlocal calls
        calls += 1
        if calls == 3:
            model, loader, optimizer = args

            def partial() -> Iterator[Any]:
                for index, batch in enumerate(loader):
                    if index == 2:
                        raise RuntimeError("power loss after an update")
                    yield batch

            return real_train(model, partial(), optimizer, **kwargs)
        return real_train(*args, **kwargs)

    monkeypatch.setattr(training, "train_epoch", interrupt)
    with pytest.raises(RuntimeError, match="power loss"):
        run_case(resume_case)
    last = Path("ai/runs/synthetic-cpu/last.pth")
    boundary = torch.load(last, weights_only=True)
    assert boundary["epoch"] == 2
    monkeypatch.setattr(training, "train_epoch", real_train)
    torch.manual_seed(123456)
    torch.rand(93)
    random.seed(983)
    np.random.seed(921)
    run_case(resume_case, last)
    actual = torch.load(last, weights_only=True)
    assert actual["epoch"] == 4
    assert actual["history"] == reference["history"]
    assert_checkpoint_equal(actual, reference)
    for state in actual["optimizer"]["state"].values():
        assert state["step"].item() == 8
    assert (
        json.loads(last.with_name("metadata.json").read_text())["history"]
        == actual["history"]
    )


def test_resume_restarts_interrupted_first_epoch(
    resume_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import ai.train as training

    baseline = tmp_path / "first-baseline"
    interrupted = tmp_path / "first-interrupted"
    baseline.mkdir()
    interrupted.mkdir()
    monkeypatch.chdir(baseline)
    best = run_case(resume_case)
    expected = torch.load(best.with_name("last.pth"), weights_only=True)
    monkeypatch.chdir(interrupted)
    real_train = training.train_epoch

    def fail(
        model: nn.Module,
        loader: DataLoader[Any],
        optimizer: torch.optim.Optimizer,
        **kwargs: Any,
    ) -> dict[str, float]:
        def partial() -> Iterator[Any]:
            for index, batch in enumerate(loader):
                if index == 2:
                    raise RuntimeError("first epoch interrupted after update")
                yield batch

        return real_train(model, partial(), optimizer, **kwargs)

    monkeypatch.setattr(training, "train_epoch", fail)
    with pytest.raises(RuntimeError, match="first epoch interrupted"):
        run_case(resume_case)
    last = Path("ai/runs/synthetic-cpu/last.pth")
    assert last.is_file(), "first epoch has no recovery checkpoint"
    initial = torch.load(last, weights_only=True)
    assert initial["epoch"] == 0 and initial["history"] == []
    assert initial["optimizer"]["state"] == {}
    monkeypatch.setattr(training, "train_epoch", real_train)
    run_case(resume_case, last)
    assert_checkpoint_equal(torch.load(last, weights_only=True), expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("scale", 0.0),
        ("scale", -1.0),
        ("scale", 1e100),
        ("growth_factor", 1.0),
        ("backoff_factor", 0.0),
        ("backoff_factor", 1.0),
        ("growth_interval", 0),
        ("growth_interval", True),
        ("_growth_tracker", -1),
        ("_growth_tracker", 3),
    ],
)
def test_checkpoint_rejects_invalid_scaler(field: str, value: Any) -> None:
    import ai.train as training

    assert hasattr(training, "_check_scaler_state"), (
        "scaler semantic validation missing"
    )
    state = torch.amp.GradScaler("cpu", init_scale=128, growth_interval=3).state_dict()
    state[field] = value
    with pytest.raises(ValueError, match="scaler"):
        training._check_scaler_state(state)


def test_new_run_retries_initial_checkpoint_write_failure(
    resume_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import ai.train as training

    real_save = training._atomic_save

    def fail(path: Path, value: Any) -> None:
        raise OSError("initial disk failure")

    monkeypatch.setattr(training, "_atomic_save", fail)
    with pytest.raises(OSError, match="initial disk failure"):
        run_case(resume_case)
    monkeypatch.setattr(training, "_atomic_save", real_save)
    assert run_case(resume_case).is_file()


def test_resume_rejects_changed_split(resume_case: dict[str, Any]) -> None:
    best = run_case(resume_case)
    resume_case["split_hash"] = "d" * 64
    with pytest.raises(ValueError, match="split_hash"):
        run_case(resume_case, best.with_name("last.pth"))


@pytest.mark.parametrize("field,value", [("seed", 9), ("epochs", 9), ("patience", 9)])
def test_resume_rejects_changed_config(
    resume_case: dict[str, Any], field: str, value: Any
) -> None:
    best = run_case(resume_case)
    last = best.with_name("last.pth")
    before = last.read_bytes()
    resume_case[field] = value
    with pytest.raises(ValueError, match=field):
        run_case(resume_case, last)
    assert last.read_bytes() == before


@pytest.mark.parametrize("field", ["pretrained_path", "dataset_dir"])
def test_resume_rechecks_actual_artifacts(
    resume_case: dict[str, Any], field: str
) -> None:
    best = run_case(resume_case)
    path = Path(resume_case[field])
    if field == "dataset_dir":
        path /= "split.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="split_hash|pretrained_sha256"):
        run_case(resume_case, best.with_name("last.pth"))


@pytest.mark.parametrize("change", ["weights", "prefix", "add", "remove"])
def test_resume_rejects_changed_sampling_policy(
    resume_case: dict[str, Any], change: str
) -> None:
    if change != "add":
        resume_case["sample_prefix_weights"] = {"pv:": 0.75, "plantdoc:": 0.25}
    best = run_case(resume_case)
    last = best.with_name("last.pth")
    before = last.read_bytes()
    if change == "remove":
        del resume_case["sample_prefix_weights"]
    elif change == "prefix":
        resume_case["sample_prefix_weights"] = {"pv:": 0.75, "other:": 0.25}
    else:
        resume_case["sample_prefix_weights"] = {"pv:": 0.5, "plantdoc:": 0.5}
    with pytest.raises(ValueError, match="resume config"):
        run_case(resume_case, last)
    assert last.read_bytes() == before


def test_resume_accepts_unchanged_sampling_policy(resume_case: dict[str, Any]) -> None:
    resume_case["sample_prefix_weights"] = {"pv:": 0.75, "plantdoc:": 0.25}
    best = run_case(resume_case)
    last = best.with_name("last.pth")
    before = last.read_bytes()
    resume_case["sample_prefix_weights"] = {"plantdoc:": 0.25, "pv:": 0.75}
    assert run_case(resume_case, last) == best
    assert last.read_bytes() == before


@pytest.mark.parametrize("failure", ["serialize", "replace"])
def test_atomic_checkpoint_failure_preserves_previous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    import os

    import ai.train as training

    assert hasattr(training, "_atomic_save"), "M4.5 atomic checkpoint writer is missing"
    target = tmp_path / "last.pth"
    training._atomic_save(target, {"epoch": 1})
    before = target.read_bytes()

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk failure")

    monkeypatch.setattr(
        torch if failure == "serialize" else os,
        "save" if failure == "serialize" else "replace",
        fail,
    )
    with pytest.raises(OSError, match="disk failure"):
        training._atomic_save(target, {"epoch": 2})
    assert target.read_bytes() == before
    assert list(tmp_path.iterdir()) == [target]


def test_resume_repairs_best_and_metadata_without_extra_updates(
    resume_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import ai.train as training

    best = run_case(resume_case)
    expected = torch.load(best, weights_only=True)
    best.unlink()
    best.with_name("metadata.json").write_text("incomplete")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("completed resume must not train")

    monkeypatch.setattr(training, "train_epoch", forbidden)
    assert run_case(resume_case, best.with_name("last.pth")) == best
    assert_checkpoint_equal(torch.load(best, weights_only=True), expected, atol=0)
    metadata = json.loads(best.with_name("metadata.json").read_text())
    assert metadata["epoch"] == 4
    assert metadata["config"] == resume_case


@pytest.mark.parametrize("run_id", ["../escape", "C:/escape", "CON", "a/b", ".."])
def test_runner_rejects_unsafe_output_name(
    resume_case: dict[str, Any], run_id: str
) -> None:
    resume_case["run_id"] = run_id
    with pytest.raises(ValueError, match="run_id"):
        run_case(resume_case)


def test_new_run_does_not_overwrite_existing_output(
    resume_case: dict[str, Any],
) -> None:
    best = run_case(resume_case)
    before = best.read_bytes()
    with pytest.raises(FileExistsError):
        run_case(resume_case)
    assert best.read_bytes() == before


@pytest.mark.parametrize(
    "field",
    ["model", "optimizer", "scaler", "rng", "epoch", "best", "history", "identity"],
)
def test_resume_rejects_incomplete_checkpoint(
    resume_case: dict[str, Any], field: str
) -> None:
    best = run_case(resume_case)
    last = best.with_name("last.pth")
    checkpoint = torch.load(last, weights_only=True)
    del checkpoint[field]
    torch.save(checkpoint, last)
    before = last.read_bytes()
    with pytest.raises(ValueError, match="checkpoint|resume|identity"):
        run_case(resume_case, last)
    assert last.read_bytes() == before


@pytest.mark.parametrize(
    "damage",
    [
        "optimizer_shape",
        "optimizer_missing",
        "optimizer_lr",
        "best_shape",
        "selection",
        "nonfinite",
    ],
)
def test_resume_rejects_inconsistent_state(
    resume_case: dict[str, Any], damage: str
) -> None:
    best = run_case(resume_case)
    last = best.with_name("last.pth")
    checkpoint = torch.load(last, weights_only=True)
    if damage == "optimizer_shape":
        next(iter(checkpoint["optimizer"]["state"].values()))["exp_avg"] = torch.ones(
            29
        )
    elif damage == "optimizer_missing":
        checkpoint["optimizer"]["state"].clear()
    elif damage == "optimizer_lr":
        checkpoint["optimizer"]["param_groups"][0]["lr"] = 1.0
    elif damage == "best_shape":
        checkpoint["best"]["model"]["1.weight"] = torch.ones(3, 4)
    elif damage == "selection":
        checkpoint["history"][-1]["bad_epochs"] = 100
    else:
        checkpoint["model"]["1.weight"][0, 0] = float("nan")
    torch.save(checkpoint, last)
    with pytest.raises(ValueError, match="checkpoint|optimizer|selection"):
        run_case(resume_case, last)


@pytest.mark.parametrize("improve_after_resume", [False, True])
def test_resume_preserves_best_and_patience(
    resume_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    improve_after_resume: bool,
) -> None:
    import ai.train as training

    resume_case.update(epochs=8, patience=5)
    values = [(0.5, 1.0), (0.4, 1.1), (0.4, 1.1)] + [
        (0.5, 0.9 if improve_after_resume else 1.0)
    ] * 5
    index = 0

    def metrics(*args: Any, **kwargs: Any) -> dict[str, float]:
        nonlocal index
        f1, loss = values[index]
        index += 1
        return {"macro_f1": f1, "loss": loss, "accuracy": 0.5}

    real_train = training.train_epoch

    def interrupt(*args: Any, **kwargs: Any) -> dict[str, float]:
        if index == 3:
            raise RuntimeError("interrupt epoch four")
        return real_train(*args, **kwargs)

    monkeypatch.setattr(training, "_validate_epoch", metrics)
    monkeypatch.setattr(training, "train_epoch", interrupt)
    with pytest.raises(RuntimeError, match="interrupt"):
        run_case(resume_case)
    last = Path("ai/runs/synthetic-cpu/last.pth")
    saved_best = torch.load(last.with_name("best.pth"), weights_only=True)
    monkeypatch.setattr(training, "train_epoch", real_train)
    best = run_case(resume_case, last)
    checkpoint = torch.load(last, weights_only=True)
    assert checkpoint["epoch"] == (8 if improve_after_resume else 6)
    assert checkpoint["best"]["epoch"] == (4 if improve_after_resume else 1)
    assert [record["bad_epochs"] for record in checkpoint["history"]] == (
        [0, 1, 2, 0, 1, 2, 3, 4] if improve_after_resume else [0, 1, 2, 3, 4, 5]
    )
    if not improve_after_resume:
        assert_checkpoint_equal(torch.load(best, weights_only=True), saved_best, atol=0)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("stopped resume must not update weights")

    monkeypatch.setattr(training, "train_epoch", forbidden)
    run_case(resume_case, last)


@pytest.mark.parametrize("sealed_images", [True], indirect=True)
@pytest.mark.parametrize("architecture", ["efficientnet_b0", "resnet50", "yolo26s_cls"])
def test_runner_real_backbone_frozen_then_fine_tune(
    resume_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    architecture: str,
) -> None:
    import ai.train as training

    # Undo only the tiny model/loader seams; use real sealed images and torchvision.
    monkeypatch.undo()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        weights = tmp_path / "imagenet-fixture.pth"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(71)
            base = build_model(architecture, 1000)
        if architecture == "yolo26s_cls":
            from ai.yolo import Yolo26Classifier

            assert isinstance(base, Yolo26Classifier)
            torch.save(base.model.state_dict(), weights)
        else:
            torch.save(base.state_dict(), weights)
        original_bn = {
            name: value.clone()
            for name, value in base.state_dict().items()
            if "running_" in name or "num_batches_tracked" in name
        }
        del base
        resume_case.update(
            architecture=architecture,
            pretrained_path=str(weights),
            pretrained_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
            epochs=2,
            batch=8,
            accumulation=1,
        )
        if architecture == "yolo26s_cls":
            from ai.preprocessing import YOLO_PREPROCESSING

            resume_case.update(
                preprocessing=copy.deepcopy(YOLO_PREPROCESSING),
                sample_prefix_weights={"pv:": 0.75, "plantdoc:": 0.25},
            )
        frozen_best = run_case(resume_case)
        frozen = torch.load(frozen_best.with_name("last.pth"), weights_only=True)
        assert frozen["epoch"] == 2
        assert {
            int(state["step"]) for state in frozen["optimizer"]["state"].values()
        } == {2}
        assert frozen["optimizer"]["param_groups"][0]["lr"] == 1e-3
        for name, value in original_bn.items():
            assert torch.equal(frozen["model"][name], value)
        parent = torch.load(frozen_best, weights_only=True)
        child = copy.deepcopy(resume_case)
        child.update(
            run_id="fine-tune",
            stage="fine_tune",
            parent_best=str(frozen_best),
            epochs=1,
        )
        child["optimizer"]["lr"] = 1e-4
        child_best = training.run_training(child)
        fine = torch.load(child_best.with_name("last.pth"), weights_only=True)
        assert (
            fine["identity"]["parent_best_sha256"]
            == hashlib.sha256(frozen_best.read_bytes()).hexdigest()
        )
        assert [group["lr"] for group in fine["optimizer"]["param_groups"]] == [
            1e-4,
            1e-5,
        ]
        assert {
            int(state["step"]) for state in fine["optimizer"]["state"].values()
        } == {1}
        assert fine["history"][0]["is_best"] is True
        for name, value in original_bn.items():
            assert torch.equal(fine["model"][name], value)
        head = {
            "efficientnet_b0": "classifier.1.weight",
            "resnet50": "fc.weight",
            "yolo26s_cls": "model.model.10.linear.weight",
        }[architecture]
        assert not torch.equal(fine["model"][head], parent["model"][head])
        frozen_best.write_bytes(frozen_best.read_bytes() + b" ")
        with pytest.raises(ValueError, match="parent_best identity"):
            training.run_training(child, child_best.with_name("last.pth"))
    finally:
        torch.set_num_threads(previous_threads)


@pytest.fixture
def yolo_resume_case(
    resume_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[dict[str, Any]]:
    from ai.preprocessing import YOLO_PREPROCESSING
    from ai.yolo import Yolo26Classifier

    monkeypatch.undo()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        weights = tmp_path / "yolo-imagenet-fixture.pth"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(71)
            model = Yolo26Classifier(1000)
        torch.save(model.model.state_dict(), weights)
        del model
        resume_case.update(
            architecture="yolo26s_cls",
            preprocessing=copy.deepcopy(YOLO_PREPROCESSING),
            pretrained_path=str(weights),
            pretrained_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
            sample_prefix_weights={"pv:": 0.75, "plantdoc:": 0.25},
            epochs=3,
        )
        yield resume_case
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("sealed_images", [True], indirect=True)
def test_yolo_weighted_resume_restores_real_loader_model_optimizer_and_rng(
    yolo_resume_case: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ai.train as training

    config = yolo_resume_case
    continuous, interrupted = tmp_path / "continuous", tmp_path / "interrupted"
    continuous.mkdir()
    interrupted.mkdir()
    real_train = training.train_epoch
    visits: list[str] = []

    def recording(*args: Any, **kwargs: Any) -> dict[str, float]:
        model, loader, optimizer = args
        assert isinstance(loader.sampler, WeightedRandomSampler)
        assert loader.sampler.generator is loader.generator

        def batches() -> Iterator[Any]:
            for batch in loader:
                visits.extend(batch[2])
                yield batch

        return real_train(model, batches(), optimizer, **kwargs)

    monkeypatch.setattr(training, "train_epoch", recording)
    monkeypatch.chdir(continuous)
    best = run_case(config)
    reference = torch.load(best.with_name("last.pth"), weights_only=True)
    expected_visits = visits.copy()
    assert len(expected_visits) == 24
    assert {sample.split(":")[0] for sample in expected_visits} == {"pv", "plantdoc"}
    visits.clear()
    monkeypatch.chdir(interrupted)
    calls = 0

    def interrupt(*args: Any, **kwargs: Any) -> dict[str, float]:
        nonlocal calls
        calls += 1
        if calls == 2:
            model, loader, optimizer = args

            def partial() -> Iterator[Any]:
                for index, batch in enumerate(loader):
                    if index == 2:
                        raise RuntimeError("power loss after weighted optimizer update")
                    yield batch

            return real_train(model, partial(), optimizer, **kwargs)
        return recording(*args, **kwargs)

    monkeypatch.setattr(training, "train_epoch", interrupt)
    with pytest.raises(RuntimeError, match="power loss after weighted"):
        run_case(config)
    last = interrupted / "ai/runs/synthetic-cpu/last.pth"
    assert torch.load(last, weights_only=True)["epoch"] == 1
    monkeypatch.setattr(training, "train_epoch", recording)
    torch.manual_seed(123456)
    torch.rand(93)
    random.seed(983)
    np.random.seed(921)
    run_case(config, last)
    actual = torch.load(last, weights_only=True)
    assert visits == expected_visits
    assert actual["history"] == reference["history"]
    assert_checkpoint_equal(actual, reference)
    assert_checkpoint_equal(actual["rng"], reference["rng"], atol=0)
    assert {int(s["step"]) for s in actual["optimizer"]["state"].values()} == {6}


@pytest.mark.parametrize("sealed_images", [True], indirect=True)
@pytest.mark.parametrize("change", ["weights", "prefix", "add", "remove"])
def test_yolo_real_loader_rejects_sampling_drift_without_checkpoint_writes(
    yolo_resume_case: dict[str, Any], change: str
) -> None:
    config = yolo_resume_case
    config["epochs"] = 1
    if change == "add":
        del config["sample_prefix_weights"]
    best = run_case(config)
    paths = [best, best.with_name("last.pth"), best.with_name("metadata.json")]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    if change == "remove":
        del config["sample_prefix_weights"]
    elif change == "prefix":
        config["sample_prefix_weights"] = {"pv:": 0.75, "other:": 0.25}
    else:
        config["sample_prefix_weights"] = {"pv:": 0.5, "plantdoc:": 0.5}
    with pytest.raises(ValueError, match="resume config"):
        run_case(config, best.with_name("last.pth"))
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths] == before
