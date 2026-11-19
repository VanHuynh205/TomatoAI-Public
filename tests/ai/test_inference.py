"""M6 bundles and predictions: synthetic fixtures, offline CPU and fail-closed IO."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from torch import nn

from ai.artifacts import load_bundle
from ai.inference import Predictor
from ai.models import build_model
from ai.preprocessing import build_transform
from tests.ai.test_experiments import summary_pair as summary_pair  # noqa: PLC0414
from tests.ai.test_experiments import trained_run as trained_run  # noqa: PLC0414
from tests.ai.test_loader import sealed_images as sealed_images  # noqa: PLC0414
from tests.ai.test_train import config as config  # noqa: PLC0414
from tests.ai.test_yolo import yolo_bundle as yolo_bundle  # noqa: PLC0414

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "defect",
    ["old-profile", "imagenet", "wrong-architecture", "crop", "fill", "unknown"],
)
def test_yolo_profile_is_bound_to_architecture(
    yolo_bundle: Path, tmp_path: Path, defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "yolo-metadata"
    directory.mkdir()
    meta = read_meta(yolo_bundle)
    if defect == "old-profile":
        meta["preprocessing"]["profile_version"] = "whole-frame-v1"
    elif defect == "imagenet":
        meta["preprocessing"]["spec"]["mean"] = [0.485, 0.456, 0.406]
    elif defect == "wrong-architecture":
        meta["architecture"] = "efficientnet_b0"
    elif defect == "crop":
        meta["preprocessing"]["spec"]["geometry"] = "center_crop"
    elif defect == "fill":
        meta["preprocessing"]["spec"]["fill"] = [0, 0, 0]
    else:
        meta["preprocessing"]["profile_version"] = "yolo-unknown"
    write_meta(directory, meta)
    monkeypatch.setattr(
        torch, "load", lambda *a, **kw: pytest.fail("bad metadata reached weights")
    )
    with pytest.raises(ValueError, match="preprocessing"):
        load_bundle(directory)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def class_hash(ids: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def read_meta(path: Path) -> dict[str, Any]:
    return json.loads((path / "metadata.json").read_text())  # type: ignore[no-any-return]


def write_meta(path: Path, meta: dict[str, Any]) -> None:
    (path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture(scope="module")
def bundle_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.set_num_threads(2)
    torch.manual_seed(42)
    out = tmp_path_factory.mktemp("bundle-template")
    model = build_model("efficientnet_b0", 8)
    torch.save(model.state_dict(), out / "model.pth")
    config = json.loads((ROOT / "ai/configs/E01.json").read_text())
    write_meta(
        out,
        {
            "schema_version": 1,
            "architecture": "efficientnet_b0",
            "model_version": "fixture-v1",
            "class_ids": config["class_ids"],
            "preprocessing": config["preprocessing"],
            "dataset_version": "synthetic-v1",
            "dataset_sha256": "a" * 64,
            "split_hash": "b" * 64,
            "weights_sha256": sha(out / "model.pth"),
            "threshold": 0.5,
            "class_map_sha256": class_hash(config["class_ids"]),
            "validation_decision_sha256": "c" * 64,
            "artifact_kind": "fixture",
        },
    )
    return out


@pytest.fixture
def valid_bundle(bundle_template: Path, tmp_path: Path) -> Path:
    return Path(shutil.copytree(bundle_template, tmp_path / "bundle"))


def test_cpu_bundle_prediction_and_readonly_contract(valid_bundle: Path) -> None:
    predictor = Predictor(valid_bundle)
    result = predictor.predict(Image.new("RGB", (256, 256)))
    assert len(result.top_k) == 3
    assert all(0 <= row.score <= 1 for row in result.top_k)
    assert [r.score for r in result.top_k] == sorted(
        (r.score for r in result.top_k), reverse=True
    )
    assert result.class_id == (
        result.top_k[0].class_id if result.status == "ok" else None
    )
    assert result.model_version == predictor.model_version == "fixture-v1"
    assert predictor.class_ids == tuple(read_meta(valid_bundle)["class_ids"])
    for field in ("class_ids", "model_version"):
        with pytest.raises(AttributeError):
            setattr(predictor, field, "changed")


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
def test_class_id_syntax_rejected_before_loading_weights(
    valid_bundle: Path, class_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta = read_meta(valid_bundle)
    meta["class_ids"][0] = class_id
    meta["class_map_sha256"] = class_hash(meta["class_ids"])
    write_meta(valid_bundle, meta)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *a, **kw: pytest.fail("invalid class_ids reached torch.load"),
    )
    with pytest.raises(ValueError, match="class_ids"):
        load_bundle(valid_bundle)


def test_bundle_class_ids_match_api_and_preserve_order(valid_bundle: Path) -> None:
    from backend.schemas import ScoredClassSchema

    meta = read_meta(valid_bundle)
    meta["class_ids"][:3] = ["z", "a0_b", "a" * 64]
    meta["class_map_sha256"] = class_hash(meta["class_ids"])
    write_meta(valid_bundle, meta)
    bundle = load_bundle(valid_bundle)
    assert bundle.metadata["class_ids"] == meta["class_ids"]
    for class_id in bundle.metadata["class_ids"]:
        assert ScoredClassSchema(class_id=class_id, score=0.5).class_id == class_id


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("schema_version", 2, "schema"),
        ("schema_version", True, "schema"),
        ("architecture", "os.system", "architecture"),
        ("architecture", [], "architecture"),
        ("model_version", "../bad", "model_version"),
        ("artifact_kind", "other", "artifact_kind"),
        ("class_ids", ["a", "a"], "class_ids"),
        ("class_ids", [], "class_ids"),
        ("class_ids", [str(i) for i in range(1001)], "class_ids"),
        ("threshold", float("nan"), "threshold"),
        ("threshold", -0.1, "threshold"),
        ("threshold", 1.1, "threshold"),
        ("threshold", True, "threshold"),
        ("weights_sha256", "a" * 64, "weights_sha256"),
        ("class_map_sha256", "d" * 64, "class_map_sha256"),
        ("validation_decision_sha256", "bad", "validation_decision_sha256"),
        ("dataset_sha256", None, "dataset_sha256"),
        ("split_hash", "bad", "split_hash"),
    ],
)
def test_metadata_tampering_rejected(
    valid_bundle: Path, field: str, value: Any, error: str
) -> None:
    meta = read_meta(valid_bundle)
    meta[field] = value
    write_meta(valid_bundle, meta)
    with pytest.raises(ValueError, match=error):
        load_bundle(valid_bundle)


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("order", "class_map_sha256"),
        ("count", "state_dict"),
        ("profile", "preprocessing"),
        ("transform", "preprocessing"),
        ("unknown", "fields"),
        ("missing", "fields"),
    ],
)
def test_map_profile_and_schema_rejected(
    valid_bundle: Path, mutation: str, error: str
) -> None:
    meta = read_meta(valid_bundle)
    if mutation == "order":
        meta["class_ids"].reverse()
    elif mutation == "count":
        meta["class_ids"].pop()
        meta["class_map_sha256"] = class_hash(meta["class_ids"])
    elif mutation == "profile":
        meta["preprocessing"]["profile_version"] = "unknown-v99"
    elif mutation == "transform":
        meta["preprocessing"]["spec"]["std"] = [0, 0, 0]
    elif mutation == "unknown":
        meta["code_module"] = "external"
    else:
        del meta["threshold"]
    write_meta(valid_bundle, meta)
    with pytest.raises(ValueError, match=error):
        load_bundle(valid_bundle)


@pytest.mark.parametrize(
    "mutation", ["bytes", "nan", "dtype", "missing", "wrapper", "garbage"]
)
def test_weights_tampering_rejected(valid_bundle: Path, mutation: str) -> None:
    path = valid_bundle / "model.pth"
    if mutation in ("bytes", "garbage"):
        path.write_bytes(b"not a checkpoint")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
        key = "classifier.1.weight"
        if mutation == "nan":
            state[key].fill_(float("nan"))
        elif mutation == "dtype":
            state[key] = state[key].double()
        elif mutation == "missing":
            del state[key]
        else:
            state = {"model": state}
        torch.save(state, path)
    if mutation != "bytes":
        meta = read_meta(valid_bundle)
        meta["weights_sha256"] = sha(path)
        write_meta(valid_bundle, meta)
    with pytest.raises(ValueError, match="weights_sha256|state_dict|weights"):
        load_bundle(valid_bundle)


@pytest.mark.parametrize(
    "name,limit", [("metadata.json", 1_000_000), ("model.pth", 256_000_000)]
)
def test_oversized_files_rejected_before_deserialization(
    valid_bundle: Path, name: str, limit: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    with (valid_bundle / name).open("wb") as stream:
        stream.truncate(limit + 1)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *a, **kw: pytest.fail("oversize must fail before torch.load"),
    )
    with pytest.raises(ValueError, match="size"):
        load_bundle(valid_bundle)


def test_fixture_rejected_by_release_guard(valid_bundle: Path) -> None:
    assert load_bundle(valid_bundle).metadata["artifact_kind"] == "fixture"
    with pytest.raises(ValueError, match="fixture"):
        load_bundle(valid_bundle, require_trained=True)


def test_train_eval_inference_parity(valid_bundle: Path) -> None:
    meta = read_meta(valid_bundle)
    image = Image.new("RGBA", (331, 177), (32, 67, 99, 128))
    reference = build_model(meta["architecture"], len(meta["class_ids"])).eval()
    reference.load_state_dict(
        torch.load(valid_bundle / "model.pth", weights_only=True), strict=True
    )
    transform = build_transform(meta["preprocessing"]["spec"], training=False)
    with torch.inference_mode():
        tensor = transform(image).unsqueeze(0)
        logits = reference(tensor)
        scores = logits.softmax(1)[0]
    predictor = Predictor(valid_bundle)
    captured: list[torch.Tensor] = []
    handle = predictor._model.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].clone())
    )
    result = predictor.predict(image)
    handle.remove()
    torch.testing.assert_close(captured[0], tensor, rtol=0, atol=1e-6)
    with torch.inference_mode():
        torch.testing.assert_close(predictor._model(tensor), logits, rtol=0, atol=1e-6)
    for row in result.top_k:
        assert row.score == pytest.approx(
            scores[meta["class_ids"].index(row.class_id)].item(), abs=1e-6
        )
    assert not predictor._model.training


@pytest.mark.parametrize(
    "count,threshold,expected",
    [(1, 1.0, "ok"), (2, 0.5, "ok"), (2, 0.50001, "uncertain"), (2, None, "uncertain")],
)
def test_topk_class_order_and_threshold_boundary(
    valid_bundle: Path, count: int, threshold: float | None, expected: str
) -> None:
    model = build_model("efficientnet_b0", count)
    with torch.no_grad():
        head = model.get_submodule("classifier.1")
        assert isinstance(head, nn.Linear)
        head.weight.zero_()
        head.bias.zero_()
    torch.save(model.state_dict(), valid_bundle / "model.pth")
    meta = read_meta(valid_bundle)
    meta.update(
        class_ids=["z", "a"][:count],
        threshold=threshold,
        weights_sha256=sha(valid_bundle / "model.pth"),
    )
    meta["class_map_sha256"] = class_hash(meta["class_ids"])
    write_meta(valid_bundle, meta)
    result = Predictor(valid_bundle).predict(Image.new("RGB", (20, 40)))
    assert [row.class_id for row in result.top_k] == meta["class_ids"]
    assert len(result.top_k) == count and result.status == expected
    assert result.class_id == ("z" if expected == "ok" else None)


@pytest.mark.parametrize(
    "bad",
    [
        torch.full((1, 8), float("nan")),
        torch.full((1, 8), float("inf")),
        torch.zeros(1, 7),
        torch.zeros(8),
    ],
)
def test_bad_logits_fail_closed(valid_bundle: Path, bad: torch.Tensor) -> None:
    class Bad(nn.Module):
        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            assert torch.is_inference_mode_enabled()
            return bad

    predictor = Predictor(valid_bundle)
    predictor._model = Bad()
    with pytest.raises(ValueError, match="logits"):
        predictor.predict(Image.new("RGB", (32, 32)))


def test_device_and_image_validation(valid_bundle: Path) -> None:
    with pytest.raises(ValueError, match="cpu"):
        Predictor(valid_bundle, device="cuda")
    predictor = Predictor(valid_bundle)
    for image in (None, Image.new("RGB", (0, 1))):
        with pytest.raises(ValueError, match="image"):
            predictor.predict(image)  # type: ignore[arg-type]


@pytest.fixture
def selected_run(summary_pair: list[Path], monkeypatch: pytest.MonkeyPatch) -> Path:
    from ai import train
    from ai.experiments import _publish_summary, summarize_validation

    summary = summarize_validation(summary_pair)
    _publish_summary(summary, summary_pair[0].parent / "summary.json")
    monkeypatch.setattr("ai.artifacts.build_model", train.build_model)  # type: ignore[attr-defined]
    return next(p for p in summary_pair if (p / "decision.json").exists())


def test_export_selected_best_and_cpu_reload(
    selected_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ai import train
    from ai.artifacts import export_bundle
    from ai.data import ManifestDataset

    decision = json.loads((selected_run / "decision.json").read_text())
    before = {p: sha(p) for p in selected_run.iterdir() if p.is_file()}
    monkeypatch.setattr(
        ManifestDataset, "__getitem__", lambda *_: pytest.fail("export decoded pixels")
    )
    output = tmp_path / "candidate"
    assert (
        export_bundle(selected_run, output, "candidate-v1", decision["threshold"])
        == output
    )
    assert sorted(p.name for p in output.iterdir()) == ["metadata.json", "model.pth"]
    bundle = load_bundle(output, require_trained=True)
    best = train._load_checkpoint(selected_run / "best.pth", "best")
    assert bundle.metadata["validation_decision_sha256"] == sha(
        selected_run / "decision.json"
    )
    assert bundle.metadata["threshold"] == decision["threshold"]
    assert bundle.metadata["dataset_sha256"] == best["config"]["seal_sha256"]
    assert bundle.metadata["split_hash"] == best["config"]["split_hash"]
    for key, tensor in bundle.model.state_dict().items():
        torch.testing.assert_close(tensor, best["model"][key], rtol=0, atol=0)
    assert all(sha(p) == value for p, value in before.items())
    assert Predictor(output).class_ids == tuple(best["config"]["class_ids"])
    with pytest.raises(FileExistsError):
        export_bundle(selected_run, output, "candidate-v1", decision["threshold"])


@pytest.mark.parametrize(
    "mutation",
    ["threshold", "selection", "hash", "path", "policy", "missing", "architecture"],
)
def test_export_rejects_changed_decision(
    selected_run: Path, tmp_path: Path, mutation: str
) -> None:
    from ai.artifacts import export_bundle

    path = selected_run / "decision.json"
    decision = json.loads(path.read_text())
    if mutation == "threshold":
        decision["threshold"] = 0.6
    elif mutation == "selection":
        decision["run_id"] = "other"
    elif mutation == "hash":
        decision["best_sha256"] = "f" * 64
    elif mutation == "path":
        decision["compared_runs"][0]["run_id"] = "../escape"
    elif mutation == "policy":
        decision["threshold_policy"]["min_accepted_accuracy"] = 0
    elif mutation == "architecture":
        decision["architecture"] = "resnet50"
    else:
        del decision["compared_runs"]
    path.write_text(json.dumps(decision))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="decision"):
        export_bundle(selected_run, output, "candidate-v1", decision["threshold"])
    assert not output.exists()


def test_export_rejects_threshold_override_and_protected_destinations(
    selected_run: Path, tmp_path: Path
) -> None:
    from ai import train
    from ai.artifacts import export_bundle
    from ai.data import ManifestDataset

    decision = json.loads((selected_run / "decision.json").read_text())
    with pytest.raises(ValueError, match="threshold"):
        export_bundle(selected_run, tmp_path / "bad", "candidate-v1", 0.123)
    cfg = train._load_checkpoint(selected_run / "best.pth", "best")["config"]
    dataset = ManifestDataset(Path(cfg["dataset_dir"]), "val", lambda image: image)
    for root in (selected_run, Path(cfg["dataset_dir"]), dataset.source_root):
        with pytest.raises(ValueError, match="output"):
            export_bundle(
                selected_run, root / "bundle", "candidate-v1", decision["threshold"]
            )
        assert not (root / "bundle").exists()


def test_export_cleans_owned_stage_on_failed_reload(
    selected_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ai.artifacts import export_bundle

    decision = json.loads((selected_run / "decision.json").read_text())
    before = set(tmp_path.iterdir())

    def reject(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("reload failed")

    monkeypatch.setattr("ai.artifacts.load_bundle", reject)
    with pytest.raises(ValueError, match="reload failed"):
        export_bundle(
            selected_run, tmp_path / "candidate", "candidate-v1", decision["threshold"]
        )
    assert set(tmp_path.iterdir()) == before


def test_cli_uses_saved_decision_and_release_guard(
    selected_run: Path, valid_bundle: Path, tmp_path: Path
) -> None:
    from ai.artifacts import main

    output = tmp_path / "cli-bundle"
    args = [
        "export",
        "--run",
        str(selected_run),
        "--out",
        str(output),
        "--model-version",
        "cli-v1",
        "--decision",
        str(selected_run / "decision.json"),
    ]
    assert main(args) == 0
    assert main(["verify", "--bundle", str(output), "--release"]) == 0
    with pytest.raises(SystemExit) as error:
        main(["verify", "--bundle", str(valid_bundle), "--release"])
    assert error.value.code == 2
    with pytest.raises(SystemExit):
        main([*args, "--threshold", "0.9"])
    with pytest.raises(SystemExit):
        main([*args[:-1], str(tmp_path / "another.json")])


@pytest.mark.parametrize(
    "path", ["//server/share/bundle", "https://example.org/bundle"]
)
def test_network_paths_rejected(path: str) -> None:
    with pytest.raises(ValueError, match="local"):
        load_bundle(Path(path))


def test_symlink_bundle_rejected(valid_bundle: Path, tmp_path: Path) -> None:
    link = tmp_path / "linked"
    try:
        link.symlink_to(valid_bundle, target_is_directory=True)
    except OSError:
        pytest.skip("Windows symlink privilege unavailable")
    with pytest.raises(ValueError, match="symlinks"):
        load_bundle(link)


def test_compressed_weights_budget_before_deserialization(
    valid_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = valid_bundle / "model.pth"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("archive/data/0", b"0" * 100_001)
    meta = read_meta(valid_bundle)
    meta["weights_sha256"] = sha(path)
    write_meta(valid_bundle, meta)
    monkeypatch.setattr("ai.artifacts._WEIGHTS_LIMIT", 100_000)
    monkeypatch.setattr(
        torch, "load", lambda *a, **kw: pytest.fail("expanded budget must precede load")
    )
    with pytest.raises(ValueError, match="expanded"):
        load_bundle(valid_bundle)


def test_pickle_code_is_never_executed(valid_bundle: Path, tmp_path: Path) -> None:
    marker = tmp_path / "executed"

    class Unsafe:
        def __reduce__(self) -> tuple[Any, ...]:
            return eval, (f"__import__('pathlib').Path({str(marker)!r}).touch()",)

    torch.save(Unsafe(), valid_bundle / "model.pth")
    meta = read_meta(valid_bundle)
    meta["weights_sha256"] = sha(valid_bundle / "model.pth")
    write_meta(valid_bundle, meta)
    with pytest.raises(ValueError, match="safe weights"):
        load_bundle(valid_bundle)
    assert not marker.exists()
