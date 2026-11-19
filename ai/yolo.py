"""Pinned YOLO26s classifier; only verified preparation reads native pickle."""

import argparse
import hashlib
import io
import json
import os
import pickle
import zipfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn

ULTRALYTICS_VERSION = "8.4.204"
PRETRAINED_SHA256 = "816790029d5df3fef358f03c8144b96339d8824ee25577aeda8be0963e5c5f09"


def _native() -> Any:
    if version("ultralytics") != ULTRALYTICS_VERSION:
        raise ValueError(f"YOLO requires ultralytics=={ULTRALYTICS_VERSION}")
    # Ultralytics checks DNS at import; this adapter's runtime is always offline.
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    original_open, original_save = Image.open, torch.save
    try:
        from ultralytics.nn.tasks import ClassificationModel
        from ultralytics.utils import ROOT, YAML
    finally:
        # Keep app decoding and checkpoint writes independent of vendor patches.
        Image.open, torch.save = original_open, original_save

    cfg = YAML.load(ROOT / "cfg/models/26/yolo26-cls.yaml")  # type: ignore[no-untyped-call]
    cfg["scale"] = "s"
    model = ClassificationModel(cfg=cfg, ch=3, nc=1000, verbose=False)  # type: ignore[no-untyped-call]
    if model.yaml["scale"] != "s" or model.model[-1].linear.in_features != 1280:
        raise ValueError("unexpected YOLO26s architecture")
    model.model[-1].export = False
    return model


def _load_state(model: nn.Module, state: Any) -> None:
    expected = model.state_dict()
    if not isinstance(state, dict) or state.keys() != expected.keys():
        raise ValueError("pretrained tensor state_dict keys mismatch")
    for key, target in expected.items():
        value = state[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.shape != target.shape
            or value.dtype != target.dtype
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"pretrained tensor mismatch: {key}")
    model.load_state_dict(state, strict=True)


def _safe_state(path: Path) -> Any:
    # Local import reuses bounded IO without an artifacts/models import cycle.
    from ai.artifacts import _WEIGHTS_LIMIT, _read

    content = _read(path, _WEIGHTS_LIMIT)
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if sum(entry.file_size for entry in archive.infolist()) > _WEIGHTS_LIMIT:
                raise ValueError("pretrained weights expanded size limit")
        return torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    except (
        zipfile.BadZipFile,
        pickle.UnpicklingError,
        RuntimeError,
        EOFError,
    ) as error:
        raise ValueError("invalid safe pretrained weights") from error


class Yolo26Classifier(nn.Module):
    def __init__(self, class_count: int, pretrained_path: Path | None = None) -> None:
        super().__init__()
        if type(class_count) is not int or class_count <= 0:
            raise ValueError("class_count must be a positive integer")
        self.model = _native()
        if pretrained_path is not None:
            _load_state(self.model, _safe_state(pretrained_path))
        head = self.model.model[-1]
        head.linear = nn.Linear(head.linear.in_features, class_count)
        self.model.yaml["nc"] = class_count
        self.model.names = {index: str(index) for index in range(class_count)}
        self.class_count = class_count

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(images, torch.Tensor)
            or images.ndim != 4
            or images.shape[1] != 3
        ):
            raise ValueError("YOLO input must be an NCHW RGB tensor")
        output = self.model(images)
        if self.training:
            logits = output
        else:
            if not isinstance(output, tuple) or len(output) != 2:
                raise ValueError("YOLO eval must return probabilities and logits")
            logits = output[1]
        if (
            not isinstance(logits, torch.Tensor)
            or logits.shape != (images.shape[0], self.class_count)
            or not bool(torch.isfinite(logits).all())
        ):
            raise ValueError("invalid YOLO logits")
        return logits


def build_yolo26_classifier(
    class_count: int, pretrained_path: Path | None = None
) -> nn.Module:
    return Yolo26Classifier(class_count, pretrained_path)


def prepare_pretrained(source: Path, manifest: Path, out: Path) -> Path:
    """Convert the pinned official asset once; a supplied manifest is not trust."""
    from ai.artifacts import _METADATA_LIMIT, _WEIGHTS_LIMIT, _json, _local, _read

    source, out = _local(source), _local(out)
    sidecar = out.with_suffix(out.suffix + ".json")
    if out.exists() or sidecar.exists():
        raise FileExistsError("prepared weights or provenance already exists")
    manifest_bytes = _read(manifest, _METADATA_LIMIT)
    data = _json(manifest_bytes)
    assets = data.get("assets")
    if data.get("ultralytics_version") != ULTRALYTICS_VERSION or not isinstance(
        assets, list
    ):
        raise ValueError("invalid pretrained manifest")
    matches = [
        asset
        for asset in assets
        if isinstance(asset, dict) and asset.get("name") == "yolo26s-cls.pt"
    ]
    if len(matches) != 1:
        raise ValueError("manifest must identify one yolo26s-cls.pt")
    content = _read(source, _WEIGHTS_LIMIT)
    digest = hashlib.sha256(content).hexdigest()
    asset = matches[0]
    if digest != asset.get("sha256") or digest != PRETRAINED_SHA256:
        raise ValueError("pretrained sha256 mismatch with pinned official asset")
    if len(content) != asset.get("bytes"):
        raise ValueError("pretrained size mismatch")
    clean = _native()
    # Exact official bytes are pinned above; never use this path for runtime bundles.
    checkpoint = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
    native = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(native, type(clean)):
        raise ValueError("pretrained must be a classification model")  # noqa: TRY004
    state = {
        key: value.detach().cpu()
        for key, value in native.float().cpu().state_dict().items()
    }
    _load_state(clean, state)
    out.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        with out.open("xb") as stream:
            created.append(out)
            torch.save(state, stream)
        provenance = {
            "schema_version": 1,
            "architecture": "yolo26s_cls",
            "class_count": 1000,
            "ultralytics_version": ULTRALYTICS_VERSION,
            "source": str(source),
            "source_sha256": digest,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "weights_sha256": hashlib.sha256(_read(out, _WEIGHTS_LIMIT)).hexdigest(),
            "tensor_only": True,
            "fused": False,
            "tomato_disease_trained": False,
        }
        with sidecar.open("x", encoding="utf-8") as stream:
            created.append(sidecar)
            stream.write(json.dumps(provenance, indent=2) + "\n")
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-pretrained")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_pretrained(args.source, args.manifest, args.out))


if __name__ == "__main__":
    main()
