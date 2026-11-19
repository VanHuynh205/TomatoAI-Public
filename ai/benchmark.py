"""Measure a fresh CPU predictor and repeated single-image predictions offline."""

import ctypes
import hashlib
import json
import os
import platform
import statistics
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from ai.artifacts import _METADATA_LIMIT, _WEIGHTS_LIMIT, _json, _local, _read
from ai.data import ManifestDataset, _decode_rgb, _sha256, _source_path
from ai.evaluation_protocol import _digest
from ai.inference import Predictor


def read_benchmark_images(manifest: Path) -> list[Path]:
    """Resolve ordered validation IDs against the pinned seal and raw byte hashes."""
    manifest = _local(manifest)
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(spec, dict)
        or spec.get("schema_version") != 1
        or spec.get("split") != "val"
    ):
        raise ValueError("expected a v1 validation image manifest")
    dataset_dir = spec.get("dataset_dir")
    ids = spec.get("sample_ids")
    if (
        not isinstance(dataset_dir, str)
        or not dataset_dir
        or not isinstance(ids, list)
        or not ids
        or not all(isinstance(item, str) and item for item in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("invalid dataset_dir or benchmark sample_ids")
    root = _local(manifest.parent / dataset_dir)
    if _sha256(root / "seal.json") != spec.get("seal_sha256"):
        raise ValueError("benchmark seal_sha256 mismatch")
    validation = ManifestDataset(root, "val", lambda image: image)
    rows = {row["sample_id"]: row for row in validation._rows}
    if any(item not in rows for item in ids):
        raise ValueError("benchmark sample outside sealed validation split")
    paths = []
    for item in ids:
        row = rows[item]
        path = _source_path(validation.source_root, row["source_path"])
        if _sha256(path) != row["source_sha256"]:
            raise ValueError(f"benchmark image sha256 mismatch: {item}")
        paths.append(path)
    return paths


def _environment() -> dict[str, Any]:
    cpu = platform.processor() or platform.machine()
    if sys.platform == "win32":
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_uint32),
                ("load", ctypes.c_uint32),
                ("total_physical", ctypes.c_uint64),
                ("other_counters", ctypes.c_uint64 * 6),
            ]

        memory = MemoryStatus()
        memory.length = ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise ctypes.WinError()
        ram_bytes = memory.total_physical
    else:
        ram_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "ram_bytes": ram_bytes,
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": version("torchvision"),
        "pillow": version("Pillow"),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }


def benchmark(
    bundle_dir: Path, image_paths: list[Path], repeats: int = 100
) -> dict[str, Any]:
    if type(repeats) is not int or repeats < 100:
        raise ValueError("benchmark requires at least 100 measured repeats")
    if not image_paths:
        raise ValueError("benchmark requires at least one image")
    paths = [_local(path) for path in image_paths]
    bundle = _local(bundle_dir)
    meta_path = bundle / "metadata.json"

    def load_stamp() -> tuple[int, ...]:
        # Catch a file or release swapped and restored within Predictor's own read.
        return tuple(p.stat().st_mtime_ns for p in (meta_path, bundle, bundle.parent))

    stamp = load_stamp()
    meta_bytes = _read(meta_path, _METADATA_LIMIT)
    metadata = _json(meta_bytes)
    environment = _environment()
    started = perf_counter()
    predictor = Predictor(bundle, device="cpu")
    cold_ms = (perf_counter() - started) * 1000
    if (
        load_stamp() != stamp
        or predictor.model_version != metadata["model_version"]
        or predictor.class_ids != tuple(metadata["class_ids"])
        or _read(meta_path, _METADATA_LIMIT) != meta_bytes
    ):
        raise ValueError("bundle changed during predictor construction")
    warmup = 10
    samples = []
    for index in range(warmup + repeats):
        path = paths[index % len(paths)]
        started = perf_counter()
        with path.open("rb") as stream, _decode_rgb(stream) as image:
            predictor.predict(image)
        elapsed_ms = (perf_counter() - started) * 1000
        if index >= warmup:
            samples.append(elapsed_ms)
    if (
        _read(meta_path, _METADATA_LIMIT) != meta_bytes
        or _digest(bundle / "model.pth", _WEIGHTS_LIMIT) != metadata["weights_sha256"]
    ):
        raise ValueError("bundle changed during benchmark")
    return {
        "model_version": predictor.model_version,
        "metadata_sha256": hashlib.sha256(meta_bytes).hexdigest(),
        "weights_sha256": metadata["weights_sha256"],
        "artifact_kind": metadata["artifact_kind"],
        "batch_size": 1,
        "device": "cpu",
        "image_count": len(paths),
        "warmup": warmup,
        "repeats": repeats,
        "cold_scope": "fresh Predictor construction and bundle validation/load; process startup and OS cache reset excluded",
        "warm_scope": "local file read, decode, preprocess, forward, softmax and top-k; API excluded",
        "cold_load_ms": cold_ms,
        "warm_p50_ms": statistics.median(samples),
        "warm_p95_ms": statistics.quantiles(samples, n=100, method="inclusive")[94],
        "quantile_method": "linear interpolation at (n-1)*p",
        "warm_samples_ms": samples,
        "environment": environment,
    }
