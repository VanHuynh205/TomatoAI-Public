"""YOLO tracer contracts on synthetic inputs; no disease accuracy claim."""

import hashlib
import io
import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from ai.artifacts import load_bundle
from ai.inference import Predictor
from ai.models import build_model
from ai.preprocessing import build_transform
from ai.yolo import Yolo26Classifier

ROOT = Path(__file__).resolve().parents[2]


def yolo_profile() -> dict[str, Any]:
    profile: dict[str, Any] = json.loads(
        (ROOT / "ai/configs/E08.json").read_text(encoding="utf-8")
    )["preprocessing"]
    profile["profile_version"] = "whole-frame-yolo-v1"
    profile["spec"].update(interpolation="bicubic", mean=[0, 0, 0], std=[1, 1, 1])
    return profile


@pytest.fixture(scope="module")
def yolo_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_yolo_fixture(tmp_path_factory.mktemp("yolo-fixture-bundle"))


def write_yolo_fixture(out: Path) -> Path:
    torch.set_num_threads(2)
    torch.manual_seed(42)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metadata.json").exists() or (out / "model.pth").exists():
        raise FileExistsError("fixture already exists")
    model = build_model("yolo26s_cls", 8)
    assert isinstance(model, Yolo26Classifier)
    head = model.model.model[-1]
    with torch.no_grad():
        head.linear.weight.zero_()
        head.linear.bias.copy_(
            torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
        )
    torch.save(model.state_dict(), out / "model.pth")
    ids = json.loads((ROOT / "ai/configs/E08.json").read_text())["class_ids"]
    metadata = {
        "schema_version": 1,
        "architecture": "yolo26s_cls",
        "model_version": "yolo-fixture-v1",
        "class_ids": ids,
        "preprocessing": yolo_profile(),
        "dataset_version": "synthetic-v1",
        "dataset_sha256": "a" * 64,
        "split_hash": "b" * 64,
        "weights_sha256": hashlib.sha256((out / "model.pth").read_bytes()).hexdigest(),
        "threshold": 0.5,
        "class_map_sha256": hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "validation_decision_sha256": "c" * 64,
        "artifact_kind": "fixture",
    }
    (out / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return out


def test_yolo_forward_returns_logits_in_train_and_eval() -> None:
    torch.set_num_threads(2)
    model = build_model("yolo26s_cls", 8)
    assert isinstance(model, Yolo26Classifier)
    images = torch.rand(2, 3, 224, 224)
    train_logits = model.train()(images)
    assert train_logits.shape == (2, 8) and torch.isfinite(train_logits).all()
    train_logits.square().mean().backward()
    assert model.model.model[-1].linear.weight.grad is not None
    model.eval()
    with torch.inference_mode():
        logits = model(images)
        probabilities, native_logits = model.model(images)
    assert logits.shape == (2, 8) and torch.isfinite(logits).all()
    torch.testing.assert_close(logits, native_logits, atol=0, rtol=0)
    torch.testing.assert_close(logits.softmax(1), probabilities, atol=1e-5, rtol=0)
    assert model.model.model[-1].export is False
    assert model.model.yaml["scale"] == "s"


def test_yolo_class_order_is_manifest_order(yolo_bundle: Path) -> None:
    bundle = load_bundle(yolo_bundle)
    ids = json.loads((ROOT / "ai/configs/E08.json").read_text())["class_ids"]
    assert bundle.metadata["class_ids"] == ids
    predictor = Predictor(yolo_bundle)
    result = predictor.predict(Image.new("RGB", (713, 91)))
    assert predictor.class_ids == tuple(ids)
    assert [row.class_id for row in result.top_k] == ids[:3]
    assert result.class_id == ids[0] and result.status == "ok"
    expected = torch.arange(3.0, -5.0, -1.0).softmax(0)
    assert [row.score for row in result.top_k] == pytest.approx(expected[:3].tolist())


def test_yolo_pretrained_rejects_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ai.yolo import prepare_pretrained

    source = tmp_path / "yolo26s-cls.pt"
    source.write_bytes(b"untrusted pickle must not execute")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "ultralytics_version": "8.4.204",
                "assets": [
                    {
                        "name": source.name,
                        "path": str(source),
                        "bytes": source.stat().st_size,
                        "sha256": "0" * 64,
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda *a, **kw: pytest.fail("hash mismatch reached deserialization"),
    )
    out = tmp_path / "safe.pth"
    with pytest.raises(ValueError, match="sha256"):
        prepare_pretrained(source, manifest, out)
    assert not out.exists()


def test_yolo_bundle_round_trip_is_offline(
    yolo_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def deny(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("offline bundle attempted network access")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    original = torch.load

    def safe_load(*args: Any, **kwargs: Any) -> Any:
        assert kwargs.get("weights_only") is True
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", safe_load)
    bundle = load_bundle(yolo_bundle)
    predictor = Predictor(yolo_bundle)
    image = Image.new("RGBA", (517, 93), (100, 150, 200, 127))
    tensor = build_transform(bundle.metadata["preprocessing"]["spec"])(image).unsqueeze(
        0
    )
    with torch.inference_mode():
        logits = bundle.model(tensor)
        predictor_logits = predictor._model(predictor._transform(image).unsqueeze(0))
    torch.testing.assert_close(logits, predictor_logits, rtol=0, atol=0)
    assert predictor.predict(image).model_version == "yolo-fixture-v1"
    with pytest.raises(ValueError, match="fixture"):
        load_bundle(yolo_bundle, require_trained=True)


@pytest.mark.parametrize(
    "mode,size", [("RGB", (997, 31)), ("L", (80, 511)), ("RGBA", (457, 52))]
)
def test_yolo_transform_preserves_whole_frame(mode: str, size: tuple[int, int]) -> None:
    from PIL import ImageOps

    image = Image.new(mode, size)
    image.putpixel((0, 0), 255 if mode == "L" else (255, 40, 30, 180)[: len(mode)])
    image.getexif()[274] = 6
    spec = yolo_profile()["spec"]
    oriented = ImageOps.exif_transpose(image).convert("RGB")
    long = max(oriented.size)
    scaled = oriented.resize(
        (
            max(1, round(oriented.width * 224 / long)),
            max(1, round(oriented.height * 224 / long)),
        ),
        Image.Resampling.BICUBIC,
    )
    expected = Image.new("RGB", (224, 224), (124, 116, 104))
    expected.paste(scaled, ((224 - scaled.width) // 2, (224 - scaled.height) // 2))
    pixels = (
        torch.frombuffer(bytearray(expected.tobytes()), dtype=torch.uint8)
        .reshape(224, 224, 3)
        .permute(2, 0, 1)
        .float()
        / 255
    )
    torch.testing.assert_close(build_transform(spec)(image), pixels, rtol=0, atol=0)


def test_yolo_runtime_rejects_native_pickle(tmp_path: Path) -> None:
    output = io.BytesIO()
    torch.save({"model": torch.nn.Linear(3, 8)}, output)
    path = tmp_path / "native.pt"
    path.write_bytes(output.getvalue())
    with pytest.raises(ValueError, match="weights|tensor"):
        build_model("yolo26s_cls", 8, path)


def test_yolo_fresh_runtime_import_is_offline(tmp_path: Path) -> None:
    script = """
import os, socket, sys
os.environ.pop('YOLO_OFFLINE', None)
os.environ.pop('YOLO_AUTOINSTALL', None)
os.environ['YOLO_CONFIG_DIR'] = sys.argv[1]
def deny(*a, **kw):
    raise AssertionError('runtime attempted network')
socket.socket.connect = deny
socket.create_connection = deny
socket.getaddrinfo = deny
from ai.models import build_model
model = build_model('yolo26s_cls', 8).eval()
assert model.class_count == 8
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_yolo_import_preserves_decoder_and_serialization(tmp_path: Path) -> None:
    script = """
import io, os, socket, sys
import torch
from PIL import Image
from backend.images import decode_image
from backend.security import ApiError
os.environ['YOLO_CONFIG_DIR'] = sys.argv[1]
def deny(*a, **kw):
    raise AssertionError('decoder attempted network')
socket.socket.connect = socket.create_connection = socket.getaddrinfo = deny
original_open, original_save = Image.open, torch.save
from ai.models import build_model
build_model('yolo26s_cls', 8)
for payload in (b'', b'not an image'):
    try:
        decode_image(payload)
    except ApiError as error:
        assert (error.status, error.code) == (422, 'invalid_input')
    else:
        raise AssertionError('invalid image accepted')
assert Image.open is original_open
assert torch.save is original_save
buffer = io.BytesIO()
Image.new('RGB', (10, 20)).save(buffer, format='PNG')
with decode_image(buffer.getvalue()) as image:
    assert image.size == (10, 20)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def verified_native_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    from ai import yolo

    source = tmp_path / "yolo26s-cls.pt"
    torch.save({"model": yolo._native()}, source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(yolo, "PRETRAINED_SHA256", digest)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "ultralytics_version": yolo.ULTRALYTICS_VERSION,
                "assets": [
                    {
                        "name": source.name,
                        "sha256": digest,
                        "bytes": source.stat().st_size,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return source, manifest


@pytest.mark.parametrize("phase", ["weights", "provenance"])
def test_preparation_rolls_back_failed_write(
    verified_native_source: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    from ai.yolo import prepare_pretrained

    source, manifest = verified_native_source
    out = tmp_path / "safe.pth"
    sidecar = out.with_suffix(".pth.json")

    def fail_save(state: Any, stream: Any) -> None:
        stream.write(b"partial weights")
        raise OSError("simulated disk failure")

    with monkeypatch.context() as patch:
        if phase == "weights":
            patch.setattr(torch, "save", fail_save)
        else:

            def fail_json(*args: Any, **kwargs: Any) -> str:
                raise OSError("simulated disk failure")

            patch.setattr(json, "dumps", fail_json)
        with pytest.raises(OSError, match="simulated disk failure"):
            prepare_pretrained(source, manifest, out)
    assert not out.exists() and not sidecar.exists()
    assert prepare_pretrained(source, manifest, out) == out
    data = json.loads(sidecar.read_text())
    assert data["weights_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()


def test_preparation_keeps_verified_manifest_provenance(
    verified_native_source: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai.yolo import prepare_pretrained

    source, manifest = verified_native_source
    verified_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    original_load = torch.load

    def load_and_change_manifest(*args: Any, **kwargs: Any) -> Any:
        manifest.write_bytes(b"changed after verification")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load_and_change_manifest)
    out = prepare_pretrained(source, manifest, tmp_path / "safe.pth")
    data = json.loads(out.with_suffix(".pth.json").read_text())
    assert data["manifest_sha256"] == verified_hash


@pytest.mark.parametrize("existing", ["weights", "provenance"])
def test_preparation_never_overwrites_existing_output(
    verified_native_source: tuple[Path, Path],
    tmp_path: Path,
    existing: str,
) -> None:
    from ai.yolo import prepare_pretrained

    out = tmp_path / "safe.pth"
    protected = out if existing == "weights" else out.with_suffix(".pth.json")
    protected.write_bytes(b"must survive")
    with pytest.raises(FileExistsError):
        prepare_pretrained(*verified_native_source, out)
    assert protected.read_bytes() == b"must survive"


def test_yolo_safe_pretrained_before_head(tmp_path: Path) -> None:
    model = build_model("yolo26s_cls", 1000)
    assert isinstance(model, Yolo26Classifier)
    state = model.model.state_dict()
    key = "model.0.conv.weight"
    state[key].fill_(0.125)
    path = tmp_path / "pretrained.pth"
    torch.save(state, path)
    loaded = build_model("yolo26s_cls", 8, path)
    assert isinstance(loaded, Yolo26Classifier)
    assert torch.equal(loaded.model.state_dict()[key], state[key])
    assert loaded.model.model[-1].linear.weight.shape == (8, 1280)
    state["model.10.linear.weight"] = torch.zeros(8, 1280)
    torch.save(state, path)
    with pytest.raises(ValueError, match="tensor mismatch"):
        build_model("yolo26s_cls", 8, path)


@pytest.mark.parametrize("defect", ["probabilities", "tuple-size", "shape", "nan"])
def test_yolo_rejects_invalid_eval_output(
    defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = build_model("yolo26s_cls", 8).eval()
    tensor = torch.zeros(2, 8)
    output = {
        "probabilities": tensor.softmax(1),
        "tuple-size": (tensor,),
        "shape": (tensor, torch.zeros(2, 7)),
        "nan": (tensor, torch.full((2, 8), float("nan"))),
    }[defect]
    monkeypatch.setattr(model.model, "forward", lambda _: output)
    with pytest.raises(ValueError, match="YOLO"):
        model(torch.zeros(2, 3, 224, 224))
