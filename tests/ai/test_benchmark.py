"""CPU timing protocol using random fixture weights and synthetic images."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from ai.inference import Predictor
from tests.ai.test_inference import bundle_template as bundle_template  # noqa: PLC0414
from tests.ai.test_inference import valid_bundle as valid_bundle  # noqa: PLC0414
from tests.ai.test_loader import manifest_rows
from tests.ai.test_loader import sealed_images as sealed_images  # noqa: PLC0414


def test_benchmark_clocks_each_decode_and_cpu_prediction(
    tmp_path: Path, bundle_template: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ai import benchmark as module

    files = [tmp_path / "a.png", tmp_path / "b.png"]
    for value, path in enumerate(files):
        Image.new("RGB", (32, 32), (value, 0, 0)).save(path)
    now = 0.0
    observed = []
    original_open = Image.open

    def open_image(*args: Any, **kwargs: Any) -> Image.Image:
        nonlocal now
        now += 0.125
        return original_open(*args, **kwargs)

    class TimedPredictor(Predictor):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            nonlocal now
            super().__init__(*args, **kwargs)
            now += 8.0

        def predict(self, image: Image.Image) -> Any:
            nonlocal now
            value = image.getpixel((0, 0))
            observed.append(value)
            # Ten slow warm-ups catch accidental inclusion in the reported percentiles.
            now += (
                10.0 if len(observed) <= 10 else (0.25 if value == (0, 0, 0) else 0.5)
            )
            return super().predict(image)

    monkeypatch.setattr(module, "Predictor", TimedPredictor)
    monkeypatch.setattr(module, "perf_counter", lambda: now)
    monkeypatch.setattr(Image, "open", open_image)
    original_threads = torch.get_num_threads()
    monkeypatch.setattr(
        torch, "set_num_threads", lambda _: pytest.fail("changed runtime thread count")
    )
    report = module.benchmark(bundle_template, files)
    assert observed == [(0, 0, 0), (1, 0, 0)] * 55
    assert report["cold_load_ms"] == 8000
    assert report["warm_p50_ms"] == 500
    assert report["warm_p95_ms"] == 625
    assert report["warmup"] == 10 and report["repeats"] == 100
    assert report["batch_size"] == 1 and report["device"] == "cpu"
    assert (
        report["weights_sha256"]
        == hashlib.sha256((bundle_template / "model.pth").read_bytes()).hexdigest()
    )
    assert report["model_version"] == "fixture-v1"
    assert (
        report["metadata_sha256"]
        == hashlib.sha256((bundle_template / "metadata.json").read_bytes()).hexdigest()
    )
    assert torch.get_num_threads() == original_threads
    env = report["environment"]
    assert env["torch_threads"] == original_threads and env["torch_interop_threads"] > 0
    assert env["logical_cpus"] == os.cpu_count() and env["ram_bytes"] > 0
    assert env["cpu"] and env["torch"] and env["python"] and env["os"]


@pytest.mark.parametrize("filename", ["metadata.json", "model.pth"])
def test_benchmark_rejects_bundle_swap_during_measurement(
    filename: str, valid_bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ai import benchmark as module

    image = tmp_path / "val.png"
    Image.new("RGB", (8, 8)).save(image)
    changed = False

    def swap(self: Predictor, image: Image.Image) -> None:
        nonlocal changed
        if not changed:
            path = valid_bundle / filename
            if filename == "metadata.json":
                metadata = json.loads(path.read_bytes())
                metadata["weights_sha256"] = "0" * 64
                path.write_text(json.dumps(metadata))
            else:
                path.write_bytes(b"replacement weights")
            changed = True

    monkeypatch.setattr(Predictor, "predict", swap)
    monkeypatch.setattr(module, "perf_counter", lambda: 1.0)
    with pytest.raises(ValueError, match="bundle"):
        module.benchmark(valid_bundle, [image])
    assert changed


def test_benchmark_rejects_metadata_replaced_only_while_loading(
    valid_bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ai import benchmark as module

    path = valid_bundle / "metadata.json"
    original = path.read_bytes()
    replacement = json.loads(original)
    replacement["threshold"] = None

    class SwappedPredictor(Predictor):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            path.write_text(json.dumps(replacement))
            try:
                super().__init__(*args, **kwargs)
            finally:
                path.write_bytes(original)

    monkeypatch.setattr(module, "Predictor", SwappedPredictor)
    monkeypatch.setattr(module, "perf_counter", lambda: 1.0)
    with pytest.raises(ValueError, match="bundle"):
        module.benchmark(valid_bundle, [tmp_path / "must-not-open.png"])


def test_reject_short_run_or_empty_images_before_loading(tmp_path: Path) -> None:
    from ai.benchmark import benchmark

    with pytest.raises(ValueError, match="100"):
        benchmark(tmp_path, [tmp_path / "absent"], 99)
    with pytest.raises(ValueError, match="image"):
        benchmark(tmp_path, [])


def test_manifest_allows_only_pinned_validation_images(
    tmp_path: Path, sealed_images: tuple[Path, Path]
) -> None:
    from ai.benchmark import read_benchmark_images

    dataset, source = sealed_images
    manifest = tmp_path / "benchmark.json"
    selected = manifest_rows(dataset, "val")[:2]
    spec = {
        "schema_version": 1,
        "split": "val",
        "dataset_dir": dataset.name,
        "seal_sha256": hashlib.sha256((dataset / "seal.json").read_bytes()).hexdigest(),
        "sample_ids": [r["sample_id"] for r in selected],
    }
    manifest.write_text(json.dumps(spec))
    expected = [source / r["source_path"] for r in selected]
    assert read_benchmark_images(manifest) == expected
    expected[0].write_bytes(b"changed after seal")
    with pytest.raises(ValueError, match="sha256"):
        read_benchmark_images(manifest)
    spec["sample_ids"] = [manifest_rows(dataset, "test")[0]["sample_id"]]
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="validation"):
        read_benchmark_images(manifest)
    spec["seal_sha256"] = "0" * 64
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="seal_sha256"):
        read_benchmark_images(manifest)
