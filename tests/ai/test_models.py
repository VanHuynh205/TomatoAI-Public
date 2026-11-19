"""M3 CPU contracts: independent pixel expectations and synthetic checkpoints."""

import copy
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image, ImageDraw, ImageOps

from ai.contracts import Prediction, ScoredClass
from ai.models import build_model
from ai.preprocessing import build_transform


def profile(arch: str, reference: bool = False) -> dict[str, Any]:
    data = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "docs/experiments/preprocessing-v1.json"
        ).read_text(encoding="utf-8")
    )
    return dict(data["reference_profiles" if reference else "profiles"][arch])


def test_prediction_keeps_typed_values() -> None:
    prediction = Prediction([ScoredClass("healthy", 0.75)], "ok", "healthy", "v1")
    assert prediction.top_k[0].class_id == prediction.class_id == "healthy"
    assert prediction.top_k[0].score == 0.75
    assert prediction.status == "ok"
    assert prediction.model_version == "v1"


def test_legacy_models_do_not_import_optional_yolo() -> None:
    script = """
import builtins, sys, torch
torch.set_num_threads(2)
original = builtins.__import__
def without_yolo(name, *args, **kwargs):
    if name.startswith('ultralytics') or name == 'ai.yolo':
        raise AssertionError('legacy model imported optional YOLO')
    return original(name, *args, **kwargs)
builtins.__import__ = without_yolo
from ai.models import build_model
for architecture in ('efficientnet_b0', 'resnet50'):
    model = build_model(architecture, 8).eval()
    with torch.inference_mode():
        assert model(torch.zeros(1,3,224,224)).shape == (1,8)
assert 'ai.yolo' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("arch", ["efficientnet_b0", "resnet50"])
def test_model_and_transform_shape(arch: str, transform_spec: dict[str, Any]) -> None:
    torch.set_num_threads(2)
    spec = profile(arch, reference=True)
    assert transform_spec == profile("efficientnet_b0", reference=True)
    x = build_transform(spec)(Image.new("RGB", (300, 280)))
    model = build_model(arch, 8).eval()
    with torch.inference_mode():
        y = model(x.unsqueeze(0))
    assert tuple(x.shape) == (3, 224, 224)
    assert tuple(y.shape) == (1, 8)
    assert x.dtype == torch.float32 and x.device.type == "cpu"
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("arch", ["unknown", "ResNet50", "", None])
def test_unknown_architecture_rejected(arch: Any) -> None:
    with pytest.raises(ValueError, match="architecture"):
        build_model(arch, 8, Path("does-not-exist.pt"))


@pytest.mark.parametrize("count", [True, False, 0, -1, 8.0, "8", None])
def test_invalid_class_count(count: Any) -> None:
    with pytest.raises(ValueError, match="class_count"):
        build_model("efficientnet_b0", count)


@pytest.mark.parametrize("arch", ["efficientnet_b0", "resnet50"])
def test_safe_strict_pretrained_before_head(
    arch: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = build_model(arch, 1000)
    state = model.state_dict()
    key = "features.0.0.weight" if arch == "efficientnet_b0" else "conv1.weight"
    state[key].fill_(0.125)
    path = tmp_path / "synthetic.pt"
    torch.save(state, path)
    original_load = torch.load

    def safe_load(*args: Any, **kwargs: Any) -> Any:
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", safe_load)
    loaded = build_model(arch, 8, path)
    assert torch.equal(loaded.state_dict()[key], torch.full_like(state[key], 0.125))
    head = "classifier.1.weight" if arch == "efficientnet_b0" else "fc.weight"
    assert loaded.state_dict()[head].shape[0] == 8
    for defect in ("missing", "unexpected", "head_shape", "not_state_dict"):
        bad: Any = dict(state)
        if defect == "missing":
            del bad[key]
        elif defect == "unexpected":
            bad["unexpected.weight"] = torch.zeros(1)
        elif defect == "head_shape":
            bad[head] = torch.zeros(8, state[head].shape[1])
        else:
            bad = [torch.zeros(1)]
        torch.save(bad, path)
        with pytest.raises((ValueError, RuntimeError)):
            build_model(arch, 8, path)


def test_factory_never_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("model factory attempted network access")

    monkeypatch.setattr(torch.hub, "download_url_to_file", forbidden)
    for arch in ("efficientnet_b0", "resnet50"):
        assert build_model(arch, 8) is not None


@pytest.mark.parametrize("arch", ["efficientnet_b0", "resnet50"])
@pytest.mark.parametrize("damage", ["nan", "inf", "float32_overflow"])
def test_pretrained_rejects_nonfinite_loaded_weights(
    arch: str, damage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = build_model(arch, 1000).state_dict()
    key = "features.0.0.weight" if arch == "efficientnet_b0" else "conv1.weight"
    if damage == "float32_overflow":
        state[key] = state[key].to(torch.float64)
        state[key].view(-1)[0] = 2 * torch.finfo(torch.float32).max
    else:
        state[key].view(-1)[0] = float(damage)

    def load_checkpoint(*args: Any, **kwargs: Any) -> Any:
        return state

    monkeypatch.setattr(torch, "load", load_checkpoint)
    with pytest.raises(ValueError, match="finite"):
        build_model(arch, 8, Path("synthetic.pt"))


def pattern() -> Image.Image:
    image = Image.new("RGB", (320, 240))
    draw = ImageDraw.Draw(image)
    for x in range(0, 320, 16):
        draw.rectangle((x, 0, x + 7, 239), fill=(x % 256, 80, 200))
    draw.rectangle((0, 0, 319, 8), fill="red")
    return image


def test_eval_transform_is_deterministic(transform_spec: dict[str, Any]) -> None:
    original = copy.deepcopy(transform_spec)
    transform = build_transform(transform_spec)
    image = pattern()
    first = transform(image)
    assert torch.equal(first, transform(image))
    assert torch.equal(first, pickle.loads(pickle.dumps(transform))(image))
    assert transform_spec == original


@pytest.mark.parametrize("reference", [False, True])
def test_train_uses_caller_seed(reference: bool) -> None:
    transform = build_transform(profile("efficientnet_b0", reference), training=True)
    image = pattern()
    torch.manual_seed(17)
    first = transform(image)
    torch.manual_seed(17)
    assert torch.equal(first, transform(image))
    outcomes = []
    for seed in range(5):
        torch.manual_seed(seed)
        outcomes.append(transform(image))
    assert any(not torch.equal(first, other) for other in outcomes)


@pytest.mark.parametrize(
    "mode,color", [("RGB", (255, 0, 128)), ("L", 128), ("RGBA", (255, 0, 128, 0))]
)
def test_literal_normalization(mode: str, color: Any) -> None:
    x = build_transform(profile("efficientnet_b0"))(Image.new(mode, (224, 224), color))
    rgb = [128 / 255] * 3 if mode == "L" else [1.0, 0.0, 128 / 255]
    expected = torch.tensor(
        [(rgb[0] - 0.485) / 0.229, (rgb[1] - 0.456) / 0.224, (rgb[2] - 0.406) / 0.225]
    )
    assert torch.allclose(x[:, 0, 0], expected, atol=1e-6)
    assert torch.isfinite(x).all()


def test_exif_applied_before_geometry() -> None:
    image = pattern()
    image.getexif()[274] = 6
    transform = build_transform(profile("efficientnet_b0"))
    assert torch.equal(transform(image), transform(ImageOps.exif_transpose(image)))


def test_transform_float32_independent_of_default_dtype() -> None:
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        x = build_transform(profile("efficientnet_b0"))(
            Image.new("RGB", (224, 224), "red")
        )
        assert x.dtype == torch.float32 and x.device.type == "cpu"
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize(
    "arch,resample",
    [
        ("efficientnet_b0", Image.Resampling.BICUBIC),
        ("resnet50", Image.Resampling.BILINEAR),
    ],
)
def test_reference_resize_center_crop_pixels(
    arch: str, resample: Image.Resampling
) -> None:
    image = pattern().resize((300, 280))
    spec = profile(arch, reference=True)
    spec.update(mean=[0, 0, 0], std=[1, 1, 1])
    result = build_transform(spec)(image)
    # Short edge 280→256, long edge floor(300*256/280)=274; crop margins 25/16.
    expected = image.resize((274, 256), resample=resample).crop((25, 16, 249, 240))
    center, corner = expected.getpixel((90, 70)), expected.getpixel((0, 0))
    assert isinstance(center, tuple) and isinstance(corner, tuple)
    assert result[:, 70, 90].tolist() == pytest.approx(
        [v / 255 for v in center], abs=1e-7
    )
    assert result[:, 0, 0].tolist() == pytest.approx(
        [v / 255 for v in corner], abs=1e-7
    )


def test_reference_training_applies_rotation_and_random_crop() -> None:
    spec = profile("efficientnet_b0", reference=True)
    spec.update(horizontal_flip_p=0, rotation_degrees=0, random_crop_scale=[1, 1])
    # A square image admits both crop scales; a wide image can hit torchvision's fallback.
    image = pattern().resize((320, 320))
    torch.manual_seed(31)
    baseline = build_transform(spec, training=True)(image)
    for field, value in [("rotation_degrees", 10), ("random_crop_scale", [0.9, 0.9])]:
        changed = {**spec, field: value}
        torch.manual_seed(31)
        result = build_transform(changed, training=True)(image)
        assert result.shape == baseline.shape == (3, 224, 224)
        assert torch.isfinite(result).all()
        assert not torch.equal(result, baseline)


@pytest.mark.parametrize("arch", ["efficientnet_b0", "resnet50"])
@pytest.mark.parametrize("size", [(400, 200), (200, 400), (224, 224)])
def test_project_preserves_border_markers(arch: str, size: tuple[int, int]) -> None:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline="red", width=8)
    spec = profile(arch)
    spec.update(mean=[0, 0, 0], std=[1, 1, 1])
    tensor = build_transform(spec)(image)
    red = (tensor[0] > 0.9) & (tensor[1] < 0.1) & (tensor[2] < 0.1)
    ys, xs = torch.where(red)
    assert (xs.min().item(), xs.max().item()) == (
        (0, 223) if size[0] >= size[1] else (56, 167)
    )
    assert (ys.min().item(), ys.max().item()) == (
        (0, 223) if size[1] >= size[0] else (56, 167)
    )


@pytest.mark.parametrize("arch", ["efficientnet_b0", "resnet50"])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("size", [(1000, 1), (1, 1000)])
def test_project_skinny_images_keep_full_frame(
    arch: str, training: bool, size: tuple[int, int]
) -> None:
    image = Image.new("RGB", size, "green")
    draw = ImageDraw.Draw(image)
    if size[0] == 1000:
        draw.rectangle((0, 0, 249, 0), fill="red")
        draw.rectangle((750, 0, 999, 0), fill="blue")
    else:
        draw.rectangle((0, 0, 0, 249), fill="red")
        draw.rectangle((0, 750, 0, 999), fill="blue")
    spec = profile(arch)
    spec.update(mean=[0, 0, 0], std=[1, 1, 1], horizontal_flip_p=0)
    tensor = build_transform(spec, training=training)(image)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32 and tensor.device.type == "cpu"
    assert torch.isfinite(tensor).all()
    strip = tensor[:, 111, :] if size[0] == 1000 else tensor[:, :, 111]
    assert strip[:, 0].tolist() == [1, 0, 0]
    assert strip[:, 111].tolist() == pytest.approx([0, 128 / 255, 0])
    assert strip[:, 223].tolist() == [0, 0, 1]
    assert tensor[:, 0, 0].tolist() == pytest.approx([124 / 255, 116 / 255, 104 / 255])


@pytest.mark.parametrize("arch", ["efficientnet_b0", "resnet50"])
@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize(
    "ratio", [[0.9, 1.1], [2, 2], [1e300, 1e300], [1e-300, 1e-300]]
)
def test_schema_one_rejects_non_square_crop_ratio(
    arch: str, reference: bool, ratio: list[float]
) -> None:
    spec = profile(arch, reference)
    spec["random_crop_ratio"] = ratio
    with pytest.raises(ValueError):
        build_transform(spec, training=True)


@pytest.mark.parametrize(
    "scale",
    [[1e-300, 1], [1e-300, 1e-300], [1e300, 1e300], [10**1000, 10**1000]],
)
def test_crop_scale_rejects_float32_underflow_or_overflow(scale: list[float]) -> None:
    spec = profile("efficientnet_b0", reference=True)
    spec["random_crop_scale"] = scale
    with pytest.raises(ValueError, match="random_crop_scale"):
        build_transform(spec, training=True)


@pytest.mark.parametrize("arch", ["efficientnet_b0", "resnet50"])
@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_transform_input_pixel_limit_before_decode_or_copy(
    arch: str, reference: bool, training: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Header proxy avoids allocating an oversized image; load/copy must never run.
    image = Image.Image()
    image._size = (5000, 4001)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("oversized input reached decode/copy")

    monkeypatch.setattr(image, "load", forbidden)
    monkeypatch.setattr(image, "copy", forbidden)
    transform = build_transform(profile(arch, reference), training=training)
    with pytest.raises(ValueError, match="pixel_limit"):
        transform(image)


@pytest.mark.parametrize("reference", [False, True])
def test_transform_allows_input_at_pixel_budget(
    reference: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image.Image()
    image._size = (5000, 4000)

    def stop_before_decode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("decode_boundary")

    monkeypatch.setattr(image, "load", stop_before_decode)
    with pytest.raises(RuntimeError, match="decode_boundary"):
        build_transform(profile("efficientnet_b0", reference))(image)


@pytest.mark.parametrize("arch", ["efficientnet_b0", "resnet50"])
@pytest.mark.parametrize("size", [(1000, 1), (1, 1000)])
def test_reference_intermediate_pixel_limit_before_resize(
    arch: str, size: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image.new("RGB", size)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("reference attempted oversized intermediate resize")

    monkeypatch.setattr(Image.Image, "resize", forbidden)
    with pytest.raises(ValueError, match="pixel_limit"):
        build_transform(profile(arch, reference=True))(image)


@pytest.mark.parametrize("arch", ["efficientnet_b0", "resnet50"])
@pytest.mark.parametrize("size", [(1000, 1), (1, 1000)])
def test_reference_training_skinny_images_use_crop_budget(
    arch: str, size: tuple[int, int]
) -> None:
    tensor = build_transform(profile(arch, reference=True), training=True)(
        Image.new("RGB", size)
    )
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()


@pytest.mark.parametrize("resize_size", [4473, 10**1000])
def test_reference_resize_size_cannot_exceed_pixel_budget(resize_size: int) -> None:
    spec = profile("efficientnet_b0", reference=True)
    spec["resize_size"] = resize_size
    with pytest.raises(ValueError, match="pixel_limit"):
        build_transform(spec)


def test_reference_allows_intermediate_at_pixel_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = profile("efficientnet_b0", reference=True)
    spec["resize_size"] = 4000

    def stop_before_allocation(
        image: Image.Image, size: tuple[int, int], *args: Any, **kwargs: Any
    ) -> Any:
        assert size == (5000, 4000)
        raise RuntimeError("resize_boundary")

    monkeypatch.setattr(Image.Image, "resize", stop_before_allocation)
    with pytest.raises(RuntimeError, match="resize_boundary"):
        build_transform(spec)(Image.new("RGB", (5, 4)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("color_mode", "L"),
        ("geometry", "stretch"),
        ("resize_size", True),
        ("resize_size", 0),
        ("crop_size", 225),
        ("crop_size", 224.0),
        ("interpolation", "nearest"),
        ("antialias", False),
        ("antialias", 1),
        ("mean", [0, 1]),
        ("mean", [0, float("nan"), 0]),
        ("std", [1, 0, 1]),
        ("std", [1, float("inf"), 1]),
        ("horizontal_flip_p", -0.1),
        ("horizontal_flip_p", True),
        ("horizontal_flip_p", 1.1),
        ("rotation_degrees", -1),
        ("rotation_degrees", float("inf")),
        ("random_crop_scale", [1, 0.9]),
        ("random_crop_scale", [0, 1]),
        ("random_crop_scale", [0.9, 1.1]),
        ("random_crop_ratio", [0, 1]),
        ("random_crop_ratio", [2, 1]),
        ("fill", [0, 0, 256]),
        ("fill", [0, False, 0]),
        ("fill", [0, 0]),
        ("unknown", 1),
    ],
)
def test_invalid_specs_rejected(
    transform_spec: dict[str, Any], field: str, value: Any
) -> None:
    transform_spec[field] = value
    with pytest.raises(ValueError):
        build_transform(transform_spec)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mean", [1e300, 0, 0]),
        ("std", [1e-300, 1, 1]),
        ("std", [1e-40, 1, 1]),
        ("mean", [10**1000, 0, 0]),
    ],
)
def test_normalization_must_be_finite_float32(
    transform_spec: dict[str, Any], field: str, value: Any
) -> None:
    transform_spec[field] = value
    with pytest.raises(ValueError, match="normalization|mean|std"):
        build_transform(transform_spec)


def test_missing_fields_rejected(transform_spec: dict[str, Any]) -> None:
    for field in transform_spec:
        spec = dict(transform_spec)
        del spec[field]
        with pytest.raises(ValueError):
            build_transform(spec)


@pytest.mark.parametrize(
    "field,value",
    [
        ("resize_size", 256),
        ("rotation_degrees", 10),
        ("random_crop_scale", [0.9, 1]),
        ("random_crop_ratio", [0.9, 1.1]),
    ],
)
def test_fit_pad_rejects_destructive_recipe(field: str, value: Any) -> None:
    spec = profile("efficientnet_b0")
    spec[field] = value
    with pytest.raises(ValueError):
        build_transform(spec)


@pytest.mark.parametrize(
    "order",
    [
        "import ai.contracts; import ai.data; import ai.split",
        "import ai.split; import ai.data; import ai.contracts",
    ],
)
def test_audit_contract_imports_torch_free(order: str) -> None:
    script = (
        """import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'torchvision'}:
            raise RuntimeError('heavy import: ' + fullname)
sys.meta_path.insert(0, Block())
"""
        + order
        + """
from ai.contracts import Prediction, ScoredClass
p = Prediction([ScoredClass('healthy', .75)], 'ok', 'healthy', 'v1')
assert p.top_k[0].score == .75 and p.class_id == 'healthy'
"""
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
